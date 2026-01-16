//! # Logical to Physical Plan Conversion
//!
//! This module handles the conversion from logical operators to physical operators,
//! including optimization decisions like index selection and join algorithm choice.
//!
//! ## Conversion Process
//!
//! 1. Optimize logical plan using rule-based optimizer
//! 2. Convert each logical operator to a physical operator
//! 3. Compute output schema for the physical plan
//!
//! ## Physical Operator Selection
//!
//! - **Scans**: Table scan by default, index scan when beneficial
//! - **Joins**: Grace hash join for equi-joins, nested loop otherwise
//! - **Aggregates**: Hash aggregate (streaming in future)
//! - **Sorts**: TopK for LIMIT + ORDER BY, regular sort otherwise

use crate::sql::ast::{Expr, JoinType};
use crate::sql::optimizer::{IndexSelector, Optimizer};
use super::encoding::encode_literal_to_bytes;
use super::logical::{LogicalOperator, LogicalPlan};
use super::physical::{
    AggregateExpr, AggregateFunction, PhysicalFilterExec, PhysicalGraceHashJoin,
    PhysicalHashAggregate, PhysicalHashAntiJoin, PhysicalHashSemiJoin,
    PhysicalIndexNestedLoopJoin, PhysicalLimitExec, PhysicalNestedLoopJoin, PhysicalOperator,
    PhysicalPlan, PhysicalProjectExec, PhysicalSetOpExec, PhysicalSortExec,
    PhysicalStreamingHashJoin, PhysicalSubqueryExec, PhysicalTableScan, PhysicalTopKExec,
    PhysicalWindowExec,
};
use super::logical::{LogicalProject, SortKey};
use super::Planner;
use crate::sql::ast::FunctionArgs;
use eyre::{bail, Result};

fn collect_column_refs<'a>(expr: &'a Expr<'a>, cols: &mut Vec<&'a str>) {
    match expr {
        Expr::Column(col_ref) => cols.push(col_ref.column),
        Expr::BinaryOp { left, right, .. } => {
            collect_column_refs(left, cols);
            collect_column_refs(right, cols);
        }
        Expr::UnaryOp { expr: e, .. } => collect_column_refs(e, cols),
        Expr::Function(func) => {
            if let FunctionArgs::Args(args) = func.args {
                for arg in args.iter() {
                    collect_column_refs(arg.value, cols);
                }
            }
        }
        Expr::Array(elems) => {
            for elem in elems.iter() {
                collect_column_refs(elem, cols);
            }
        }
        Expr::Case { operand, conditions, else_result } => {
            if let Some(op) = operand {
                collect_column_refs(op, cols);
            }
            for clause in conditions.iter() {
                collect_column_refs(clause.condition, cols);
                collect_column_refs(clause.result, cols);
            }
            if let Some(e) = else_result {
                collect_column_refs(e, cols);
            }
        }
        Expr::InList { expr: e, list, .. } => {
            collect_column_refs(e, cols);
            for item in list.iter() {
                collect_column_refs(item, cols);
            }
        }
        Expr::Between { expr: e, low, high, .. } => {
            collect_column_refs(e, cols);
            collect_column_refs(low, cols);
            collect_column_refs(high, cols);
        }
        Expr::Subquery(_) | Expr::Exists { .. } | Expr::InSubquery { .. } => {}
        Expr::IsNull { expr: e, .. } | Expr::Cast { expr: e, .. } => {
            collect_column_refs(e, cols);
        }
        Expr::Like { expr: e, pattern, .. } => {
            collect_column_refs(e, cols);
            collect_column_refs(pattern, cols);
        }
        Expr::IsDistinctFrom { left, right, .. } => {
            collect_column_refs(left, cols);
            collect_column_refs(right, cols);
        }
        Expr::ArraySubscript { array, index } => {
            collect_column_refs(array, cols);
            collect_column_refs(index, cols);
        }
        Expr::ArraySlice { array, lower, upper } => {
            collect_column_refs(array, cols);
            if let Some(l) = lower { collect_column_refs(l, cols); }
            if let Some(u) = upper { collect_column_refs(u, cols); }
        }
        Expr::Row(elems) => {
            for elem in elems.iter() {
                collect_column_refs(elem, cols);
            }
        }
        Expr::Literal(_) | Expr::Parameter(_) => {}
    }
}

fn order_by_uses_only_projected_columns<'a>(
    order_by: &[SortKey<'a>],
    project: &LogicalProject<'a>,
) -> bool {
    for key in order_by {
        if project.expressions.contains(&key.expr) {
            continue;
        }

        let mut cols_needed: Vec<&str> = Vec::new();
        collect_column_refs(key.expr, &mut cols_needed);

        for col_name in cols_needed {
            let found = project.expressions.iter().any(|e| {
                if let Expr::Column(col_ref) = e {
                    col_ref.column.eq_ignore_ascii_case(col_name)
                } else {
                    false
                }
            }) || project.aliases.iter().any(|a| {
                a.map(|alias| alias.eq_ignore_ascii_case(col_name)).unwrap_or(false)
            });

            if !found {
                return false;
            }
        }
    }
    true
}

impl<'a> Planner<'a> {
    pub(crate) fn optimize_to_physical(&self, logical: &LogicalPlan<'a>) -> Result<PhysicalPlan<'a>> {
        let optimizer = Optimizer::with_catalog(self.catalog);
        let optimized_root = optimizer.optimize(logical.root, self.arena)?;

        let physical_root = self.logical_to_physical(optimized_root)?;
        let output_schema = self.compute_output_schema(physical_root)?;
        Ok(PhysicalPlan {
            root: physical_root,
            output_schema,
        })
    }

    pub(crate) fn logical_to_physical(
        &self,
        op: &'a LogicalOperator<'a>,
    ) -> Result<&'a PhysicalOperator<'a>> {
        let index_selector = IndexSelector::new(self.catalog, self.arena);

        match op {
            LogicalOperator::Scan(scan) => {
                let table_name = if let Some(schema) = scan.schema {
                    self.arena.alloc_str(&format!("{}.{}", schema, scan.table))
                } else {
                    scan.table
                };
                let table_def = self.catalog.resolve_table(table_name).ok();

                let physical = self
                    .arena
                    .alloc(PhysicalOperator::TableScan(PhysicalTableScan {
                        schema: scan.schema,
                        table: scan.table,
                        alias: scan.alias,
                        post_scan_filter: None,
                        table_def,
                        reverse: false,
                    }));
                Ok(physical)
            }
            LogicalOperator::DualScan => Ok(self.arena.alloc(PhysicalOperator::DualScan)),
            LogicalOperator::Filter(filter) => {
                let encode_fn = |expr: &Expr<'a>| encode_literal_to_bytes(self.arena, expr);
                if let Some(index_scan) = index_selector.try_optimize_filter_to_index_scan(filter, encode_fn) {
                    return Ok(index_scan);
                }

                let input = self.logical_to_physical(filter.input)?;
                let physical = self
                    .arena
                    .alloc(PhysicalOperator::FilterExec(PhysicalFilterExec {
                        input,
                        predicate: filter.predicate,
                    }));
                Ok(physical)
            }
            LogicalOperator::Project(project) => {
                let input = self.logical_to_physical(project.input)?;
                let physical =
                    self.arena
                        .alloc(PhysicalOperator::ProjectExec(PhysicalProjectExec {
                            input,
                            expressions: project.expressions,
                            aliases: project.aliases,
                        }));
                Ok(physical)
            }
            LogicalOperator::Join(join) => {
                if let Some(index_join) = self.try_index_nested_loop_join(join)? {
                    return Ok(index_join);
                }

                let left = self.logical_to_physical(join.left)?;
                let right = self.logical_to_physical(join.right)?;

                match join.join_type {
                    JoinType::Semi => {
                        if self.has_equi_join_keys(join.condition) {
                            let equi_keys = self.extract_equi_join_keys_for_join(
                                join.condition,
                                join.left,
                                join.right,
                            );
                            let join_keys = self.convert_equi_keys_to_join_keys(equi_keys);
                            let physical = self.arena.alloc(PhysicalOperator::HashSemiJoin(
                                PhysicalHashSemiJoin {
                                    left,
                                    right,
                                    join_keys,
                                },
                            ));
                            Ok(physical)
                        } else {
                            let physical = self.arena.alloc(PhysicalOperator::NestedLoopJoin(
                                PhysicalNestedLoopJoin {
                                    left,
                                    right,
                                    join_type: join.join_type,
                                    condition: join.condition,
                                },
                            ));
                            Ok(physical)
                        }
                    }
                    JoinType::Anti => {
                        if self.has_equi_join_keys(join.condition) {
                            let equi_keys = self.extract_equi_join_keys_for_join(
                                join.condition,
                                join.left,
                                join.right,
                            );
                            let join_keys = self.convert_equi_keys_to_join_keys(equi_keys);
                            let physical = self.arena.alloc(PhysicalOperator::HashAntiJoin(
                                PhysicalHashAntiJoin {
                                    left,
                                    right,
                                    join_keys,
                                },
                            ));
                            Ok(physical)
                        } else {
                            let physical = self.arena.alloc(PhysicalOperator::NestedLoopJoin(
                                PhysicalNestedLoopJoin {
                                    left,
                                    right,
                                    join_type: join.join_type,
                                    condition: join.condition,
                                },
                            ));
                            Ok(physical)
                        }
                    }
                    _ => {
                        if self.has_equi_join_keys(join.condition) {
                            let equi_keys = self.extract_equi_join_keys_for_join(
                                join.condition,
                                join.left,
                                join.right,
                            );
                            let join_keys = self.convert_equi_keys_to_join_keys(equi_keys);

                            let left_card = self.estimate_cardinality(join.left);
                            let right_card = self.estimate_cardinality(join.right);

                            let use_streaming = matches!(
                                join.join_type,
                                JoinType::Inner | JoinType::Cross
                            );

                            if use_streaming {
                                let (build, probe, final_keys, swapped) = if left_card <= right_card {
                                    (left, right, join_keys, false)
                                } else {
                                    let swapped_keys: &[(&Expr<'a>, &Expr<'a>)] = self
                                        .arena
                                        .alloc_slice_fill_iter(join_keys.iter().map(|(l, r)| (*r, *l)));
                                    (right, left, swapped_keys, true)
                                };

                                let physical = self.arena.alloc(PhysicalOperator::StreamingHashJoin(
                                    PhysicalStreamingHashJoin {
                                        build,
                                        probe,
                                        join_type: join.join_type,
                                        join_keys: final_keys,
                                        swapped,
                                    },
                                ));
                                Ok(physical)
                            } else {
                                let physical = self.arena.alloc(PhysicalOperator::GraceHashJoin(
                                    PhysicalGraceHashJoin {
                                        left,
                                        right,
                                        join_type: join.join_type,
                                        join_keys,
                                        num_partitions: 16,
                                    },
                                ));
                                Ok(physical)
                            }
                        } else {
                            let physical = self.arena.alloc(PhysicalOperator::NestedLoopJoin(
                                PhysicalNestedLoopJoin {
                                    left,
                                    right,
                                    join_type: join.join_type,
                                    condition: join.condition,
                                },
                            ));
                            Ok(physical)
                        }
                    }
                }
            }
            LogicalOperator::Sort(sort) => {
                if let Some(optimized) = index_selector.try_optimize_sort_to_index_scan(sort) {
                    return Ok(optimized);
                }
                let input = self.logical_to_physical(sort.input)?;
                let physical = self
                    .arena
                    .alloc(PhysicalOperator::SortExec(PhysicalSortExec {
                        input,
                        order_by: sort.order_by,
                    }));
                Ok(physical)
            }
            LogicalOperator::Limit(limit) => {
                if let Some(limit_val) = limit.limit {
                    if let LogicalOperator::Sort(sort) = limit.input {
                        let effective_limit = (limit_val + limit.offset.unwrap_or(0)) as usize;
                        if let Some(optimized) =
                            index_selector.try_optimize_sort_to_index_scan_with_limit(sort, Some(effective_limit))
                        {
                            let physical = self
                                .arena
                                .alloc(PhysicalOperator::LimitExec(PhysicalLimitExec {
                                    input: optimized,
                                    limit: Some(limit_val),
                                    offset: limit.offset,
                                }));
                            return Ok(physical);
                        }

                        if let LogicalOperator::Project(project) = sort.input {
                            if !order_by_uses_only_projected_columns(sort.order_by, project) {
                                let inner_input = self.logical_to_physical(project.input)?;
                                let topk = self
                                    .arena
                                    .alloc(PhysicalOperator::TopKExec(PhysicalTopKExec {
                                        input: inner_input,
                                        order_by: sort.order_by,
                                        limit: limit_val,
                                        offset: limit.offset,
                                    }));
                                let physical = self
                                    .arena
                                    .alloc(PhysicalOperator::ProjectExec(PhysicalProjectExec {
                                        input: topk,
                                        expressions: project.expressions,
                                        aliases: project.aliases,
                                    }));
                                return Ok(physical);
                            }
                        }

                        let sort_input = self.logical_to_physical(sort.input)?;
                        let physical = self
                            .arena
                            .alloc(PhysicalOperator::TopKExec(PhysicalTopKExec {
                                input: sort_input,
                                order_by: sort.order_by,
                                limit: limit_val,
                                offset: limit.offset,
                            }));
                        return Ok(physical);
                    }
                }

                let input = self.logical_to_physical(limit.input)?;
                let physical = self
                    .arena
                    .alloc(PhysicalOperator::LimitExec(PhysicalLimitExec {
                        input,
                        limit: limit.limit,
                        offset: limit.offset,
                    }));
                Ok(physical)
            }
            LogicalOperator::Aggregate(agg) => {
                let input = self.logical_to_physical(agg.input)?;
                let aggregates = self.convert_aggregates_to_physical(agg.aggregates);
                let physical =
                    self.arena
                        .alloc(PhysicalOperator::HashAggregate(PhysicalHashAggregate {
                            input,
                            group_by: agg.group_by,
                            aggregates,
                        }));
                Ok(physical)
            }
            LogicalOperator::Values(_) => {
                bail!("Values operator cannot be directly converted to physical - only valid as INSERT source")
            }
            LogicalOperator::Insert(_) => {
                bail!(
                    "Insert operator cannot be converted to physical plan - DML handled separately"
                )
            }
            LogicalOperator::Update(_) => {
                bail!(
                    "Update operator cannot be converted to physical plan - DML handled separately"
                )
            }
            LogicalOperator::Delete(_) => {
                bail!(
                    "Delete operator cannot be converted to physical plan - DML handled separately"
                )
            }
            LogicalOperator::Subquery(subq) => {
                let child_plan = self.logical_to_physical(subq.plan)?;
                let physical =
                    self.arena
                        .alloc(PhysicalOperator::SubqueryExec(PhysicalSubqueryExec {
                            child_plan,
                            alias: subq.alias,
                            output_schema: subq.output_schema.clone(),
                        }));
                Ok(physical)
            }
            LogicalOperator::SetOp(set_op) => {
                let left = self.logical_to_physical(set_op.left)?;
                let right = self.logical_to_physical(set_op.right)?;
                let physical = self
                    .arena
                    .alloc(PhysicalOperator::SetOpExec(PhysicalSetOpExec {
                        left,
                        right,
                        kind: set_op.kind,
                        all: set_op.all,
                    }));
                Ok(physical)
            }
            LogicalOperator::Window(window) => {
                let input = self.logical_to_physical(window.input)?;
                let physical = self
                    .arena
                    .alloc(PhysicalOperator::WindowExec(PhysicalWindowExec {
                        input,
                        window_functions: window.window_functions,
                    }));
                Ok(physical)
            }
        }
    }

    fn estimate_cardinality(&self, op: &LogicalOperator<'a>) -> u64 {
        const DEFAULT_CARDINALITY: u64 = 1000;
        const FILTER_SELECTIVITY: f64 = 0.1;
        const JOIN_SELECTIVITY: f64 = 0.1;

        match op {
            LogicalOperator::Scan(scan) => {
                if let Ok(table_def) =
                    self.catalog.resolve_table_in_schema(scan.schema, scan.table)
                {
                    let row_count = table_def.row_count();
                    if row_count > 0 {
                        return row_count;
                    }
                }
                DEFAULT_CARDINALITY
            }
            LogicalOperator::DualScan => 1,
            LogicalOperator::Filter(filter) => {
                let input_card = self.estimate_cardinality(filter.input);
                ((input_card as f64) * FILTER_SELECTIVITY).max(1.0) as u64
            }
            LogicalOperator::Project(project) => self.estimate_cardinality(project.input),
            LogicalOperator::Aggregate(agg) => {
                let input_card = self.estimate_cardinality(agg.input);
                if agg.group_by.is_empty() {
                    1
                } else {
                    (input_card / 10).max(1)
                }
            }
            LogicalOperator::Join(join) => {
                let left_card = self.estimate_cardinality(join.left);
                let right_card = self.estimate_cardinality(join.right);
                if join.condition.is_some() {
                    ((left_card as f64 * right_card as f64 * JOIN_SELECTIVITY) as u64).max(1)
                } else {
                    left_card * right_card
                }
            }
            LogicalOperator::Sort(sort) => self.estimate_cardinality(sort.input),
            LogicalOperator::Limit(limit) => {
                let input_card = self.estimate_cardinality(limit.input);
                match limit.limit {
                    Some(l) => input_card.min(l),
                    None => input_card,
                }
            }
            LogicalOperator::Subquery(subq) => self.estimate_cardinality(subq.plan),
            _ => DEFAULT_CARDINALITY,
        }
    }

    pub(crate) fn convert_aggregates_to_physical(
        &self,
        aggregates: &'a [&'a Expr<'a>],
    ) -> &'a [AggregateExpr<'a>] {
        use crate::sql::ast::{FunctionArgs, FunctionCall};

        let mut result = bumpalo::collections::Vec::new_in(self.arena);

        for expr in aggregates {
            if let Expr::Function(FunctionCall {
                name,
                args,
                distinct,
                ..
            }) = expr
            {
                let func_name = name.name.to_ascii_lowercase();
                let function = match func_name.as_str() {
                    "count" => AggregateFunction::Count,
                    "sum" => AggregateFunction::Sum,
                    "avg" => AggregateFunction::Avg,
                    "min" => AggregateFunction::Min,
                    "max" => AggregateFunction::Max,
                    _ => continue,
                };

                let argument = match args {
                    FunctionArgs::Star => None,
                    FunctionArgs::None => None,
                    FunctionArgs::Args(func_args) => {
                        if func_args.is_empty() {
                            None
                        } else {
                            Some(func_args[0].value)
                        }
                    }
                };

                result.push(AggregateExpr {
                    function,
                    argument,
                    distinct: *distinct,
                });
            }
        }

        result.into_bump_slice()
    }

    fn try_index_nested_loop_join(
        &self,
        join: &super::logical::LogicalJoin<'a>,
    ) -> Result<Option<&'a PhysicalOperator<'a>>> {
        use crate::sql::ast::{BinaryOperator, ColumnRef};

        if !matches!(
            join.join_type,
            JoinType::Inner | JoinType::Left | JoinType::Right
        ) {
            return Ok(None);
        }

        let condition = match join.condition {
            Some(c) => c,
            None => return Ok(None),
        };

        let (left_col, right_col) = match condition {
            Expr::BinaryOp {
                left,
                op: BinaryOperator::Eq,
                right,
            } => {
                let left_col = match left {
                    Expr::Column(col) => col,
                    _ => return Ok(None),
                };
                let right_col = match right {
                    Expr::Column(col) => col,
                    _ => return Ok(None),
                };
                (left_col, right_col)
            }
            _ => return Ok(None),
        };

        let (outer_key, inner_col, inner_op, outer_op) =
            if let LogicalOperator::Scan(right_scan) = join.right {
                let table_name = if let Some(schema) = right_scan.schema {
                    format!("{}.{}", schema, right_scan.table)
                } else {
                    right_scan.table.to_string()
                };

                let table_def = match self.catalog.resolve_table(&table_name) {
                    Ok(t) => t,
                    Err(_) => return Ok(None),
                };

                let right_col_name = if let Some(table_ref) = right_col.table {
                    if table_ref.eq_ignore_ascii_case(right_scan.table)
                        || right_scan
                            .alias
                            .map(|a| a.eq_ignore_ascii_case(table_ref))
                            .unwrap_or(false)
                    {
                        right_col.column
                    } else {
                        left_col.column
                    }
                } else {
                    let col_exists_in_right = table_def
                        .columns()
                        .iter()
                        .any(|c| c.name().eq_ignore_ascii_case(right_col.column));
                    if col_exists_in_right {
                        right_col.column
                    } else {
                        left_col.column
                    }
                };

                let matching_index = table_def.indexes().iter().find(|idx| {
                    idx.columns()
                        .next()
                        .map(|first_col| first_col.eq_ignore_ascii_case(right_col_name))
                        .unwrap_or(false)
                });

                if matching_index.is_none() {
                    return Ok(None);
                }

                let outer_key_expr = if right_col_name == right_col.column {
                    self.arena.alloc(Expr::Column(ColumnRef {
                        schema: None,
                        table: left_col.table,
                        column: left_col.column,
                    }))
                } else {
                    self.arena.alloc(Expr::Column(ColumnRef {
                        schema: None,
                        table: right_col.table,
                        column: right_col.column,
                    }))
                };

                (
                    outer_key_expr,
                    right_col_name,
                    join.right,
                    join.left,
                )
            } else {
                return Ok(None);
            };

        let outer_card = self.estimate_cardinality(outer_op);
        let inner_card = self.estimate_cardinality(inner_op);

        let use_index_join = outer_card < inner_card / 5 || outer_card < 10000;

        if !use_index_join {
            return Ok(None);
        }

        let outer_physical = self.logical_to_physical(outer_op)?;

        let LogicalOperator::Scan(inner_scan) = inner_op else {
            return Ok(None);
        };

        let inner_table_name = if let Some(schema) = inner_scan.schema {
            format!("{}.{}", schema, inner_scan.table)
        } else {
            inner_scan.table.to_string()
        };
        let inner_table_def = self.catalog.resolve_table(&inner_table_name).ok();
        let inner_table_def_alloc = inner_table_def.map(|t| &*self.arena.alloc(t.clone()));

        let matching_index = inner_table_def
            .as_ref()
            .and_then(|td| {
                td.indexes().iter().find(|idx| {
                    idx.columns()
                        .next()
                        .map(|first_col| first_col.eq_ignore_ascii_case(inner_col))
                        .unwrap_or(false)
                })
            })
            .map(|idx| self.arena.alloc_str(idx.name()));

        let index_name = match matching_index {
            Some(name) => name,
            None => return Ok(None),
        };

        let physical = self.arena.alloc(PhysicalOperator::IndexNestedLoopJoin(
            PhysicalIndexNestedLoopJoin {
                outer: outer_physical,
                inner_table: inner_scan.table,
                inner_schema: inner_scan.schema,
                inner_alias: inner_scan.alias,
                inner_index_name: index_name,
                inner_table_def: inner_table_def_alloc,
                join_type: join.join_type,
                outer_key,
                inner_key_column: self.arena.alloc_str(inner_col),
            },
        ));

        Ok(Some(physical))
    }
}
