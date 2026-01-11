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
use crate::sql::optimizer::Optimizer;
use super::encoding::encode_literal_to_bytes;
use super::logical::{
    LogicalFilter, LogicalOperator, LogicalPlan, LogicalProject, LogicalScan, LogicalSort,
};
use super::physical::{
    AggregateExpr, AggregateFunction, PhysicalFilterExec, PhysicalGraceHashJoin,
    PhysicalHashAggregate, PhysicalHashAntiJoin, PhysicalHashSemiJoin, PhysicalLimitExec,
    PhysicalNestedLoopJoin, PhysicalOperator, PhysicalPlan, PhysicalProjectExec,
    PhysicalSecondaryIndexScan, PhysicalSetOpExec, PhysicalSortExec, PhysicalSubqueryExec,
    PhysicalTableScan, PhysicalTopKExec, PhysicalWindowExec,
};
use super::types::ScanRange;
use super::Planner;
use eyre::{bail, Result};

impl<'a> Planner<'a> {
    pub(crate) fn optimize_to_physical(&self, logical: &LogicalPlan<'a>) -> Result<PhysicalPlan<'a>> {
        let optimizer = Optimizer::new();
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
                if let Some(index_scan) = self.try_optimize_filter_to_index_scan(filter) {
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
                let left = self.logical_to_physical(join.left)?;
                let right = self.logical_to_physical(join.right)?;

                match join.join_type {
                    JoinType::Semi => {
                        if self.has_equi_join_keys(join.condition) {
                            let equi_keys = self.extract_equi_join_keys(join.condition);
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
                            let equi_keys = self.extract_equi_join_keys(join.condition);
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
                            let equi_keys = self.extract_equi_join_keys(join.condition);
                            let join_keys = self.convert_equi_keys_to_join_keys(equi_keys);
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
                if let Some(optimized) = self.try_optimize_sort_to_index_scan(sort) {
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
                            self.try_optimize_sort_to_index_scan_with_limit(sort, Some(effective_limit))
                        {
                            if limit.offset.unwrap_or(0) > 0 {
                                let physical = self
                                    .arena
                                    .alloc(PhysicalOperator::LimitExec(PhysicalLimitExec {
                                        input: optimized,
                                        limit: Some(limit_val),
                                        offset: limit.offset,
                                    }));
                                return Ok(physical);
                            }
                            return Ok(optimized);
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

    pub(crate) fn try_optimize_filter_to_index_scan(
        &self,
        filter: &LogicalFilter<'a>,
    ) -> Option<&'a PhysicalOperator<'a>> {
        fn find_scan<'b>(op: &'b LogicalOperator<'b>) -> Option<&'b LogicalScan<'b>> {
            match op {
                LogicalOperator::Scan(scan) => Some(scan),
                LogicalOperator::Project(proj) => find_scan(proj.input),
                _ => None,
            }
        }

        fn find_project<'b>(op: &'b LogicalOperator<'b>) -> Option<&'b LogicalProject<'b>> {
            match op {
                LogicalOperator::Project(proj) => Some(proj),
                _ => None,
            }
        }

        let scan = find_scan(filter.input)?;
        let table_def = self.catalog.resolve_table(scan.table).ok()?;

        let (col_name, literal_expr) = self.extract_equality_predicate(filter.predicate)?;

        let matching_index = table_def.indexes().iter().find(|idx| {
            if idx.has_expressions() || idx.is_partial() {
                return false;
            }
            if idx.index_type() != crate::schema::IndexType::BTree {
                return false;
            }
            idx.columns()
                .next()
                .map(|first_col| first_col.eq_ignore_ascii_case(col_name))
                .unwrap_or(false)
        })?;

        let key_bytes = encode_literal_to_bytes(self.arena, literal_expr)?;

        let index_name = self.arena.alloc_str(matching_index.name());
        let table_def_alloc = self.arena.alloc(table_def.clone());
        let _index_columns: Vec<String> = matching_index.columns().map(|s| s.to_string()).collect();

        let covered_columns = vec![col_name.to_string()];
        let residual = self.compute_residual_filter(filter.predicate, &covered_columns);

        let index_scan = self.arena.alloc(PhysicalOperator::SecondaryIndexScan(
            PhysicalSecondaryIndexScan {
                schema: scan.schema,
                table: scan.table,
                index_name,
                table_def: Some(table_def_alloc),
                reverse: false,
                is_unique_index: matching_index.is_unique(),
                key_range: Some(ScanRange::PrefixScan { prefix: key_bytes }),
                limit: None,
            },
        ));

        let with_residual = if let Some(residual_predicate) = residual {
            self.arena
                .alloc(PhysicalOperator::FilterExec(PhysicalFilterExec {
                    input: index_scan,
                    predicate: residual_predicate,
                }))
        } else {
            index_scan
        };

        let project = find_project(filter.input);
        if let Some(proj) = project {
            let physical_proj =
                self.arena
                    .alloc(PhysicalOperator::ProjectExec(PhysicalProjectExec {
                        input: with_residual,
                        expressions: proj.expressions,
                        aliases: proj.aliases,
                    }));
            return Some(physical_proj);
        }

        Some(with_residual)
    }

    pub(crate) fn extract_equality_predicate(
        &self,
        expr: &'a Expr<'a>,
    ) -> Option<(&'a str, &'a Expr<'a>)> {
        use crate::sql::ast::BinaryOperator;

        match expr {
            Expr::BinaryOp {
                left,
                op: BinaryOperator::Eq,
                right,
            } => match (left, right) {
                (Expr::Column(col_ref), lit @ Expr::Literal(_)) => Some((col_ref.column, lit)),
                (lit @ Expr::Literal(_), Expr::Column(col_ref)) => Some((col_ref.column, lit)),
                _ => None,
            },
            Expr::BinaryOp {
                left,
                op: BinaryOperator::And,
                right,
            } => self
                .extract_equality_predicate(left)
                .or_else(|| self.extract_equality_predicate(right)),
            _ => None,
        }
    }

    pub(crate) fn try_optimize_sort_to_index_scan(
        &self,
        sort: &LogicalSort<'a>,
    ) -> Option<&'a PhysicalOperator<'a>> {
        self.try_optimize_sort_to_index_scan_with_limit(sort, None)
    }

    pub(crate) fn try_optimize_sort_to_index_scan_with_limit(
        &self,
        sort: &LogicalSort<'a>,
        limit: Option<usize>,
    ) -> Option<&'a PhysicalOperator<'a>> {
        if sort.order_by.len() != 1 {
            return None;
        }

        let sort_key = &sort.order_by[0];

        let col_name = match sort_key.expr {
            Expr::Column(col_ref) => col_ref.column,
            _ => return None,
        };

        fn find_scan<'b>(op: &'b LogicalOperator<'b>) -> Option<&'b LogicalScan<'b>> {
            match op {
                LogicalOperator::Scan(scan) => Some(scan),
                LogicalOperator::Project(proj) => find_scan(proj.input),
                LogicalOperator::Filter(filter) => find_scan(filter.input),
                _ => None,
            }
        }

        fn has_filter(op: &LogicalOperator<'_>) -> bool {
            match op {
                LogicalOperator::Filter(_) => true,
                LogicalOperator::Project(proj) => has_filter(proj.input),
                _ => false,
            }
        }

        if has_filter(sort.input) {
            return None;
        }

        let scan = find_scan(sort.input)?;
        let table_def = self.catalog.resolve_table(scan.table).ok()?;

        let reverse = !sort_key.ascending;

        let pk_col = table_def
            .columns()
            .iter()
            .find(|c| c.has_constraint(&crate::schema::table::Constraint::PrimaryKey));

        if let Some(pk) = pk_col {
            if pk.name().eq_ignore_ascii_case(col_name) {
                let table_scan = self
                    .arena
                    .alloc(PhysicalOperator::TableScan(PhysicalTableScan {
                        schema: scan.schema,
                        table: scan.table,
                        alias: scan.alias,
                        post_scan_filter: None,
                        table_def: Some(table_def),
                        reverse,
                    }));

                return match sort.input {
                    LogicalOperator::Scan(_) => Some(table_scan),
                    LogicalOperator::Project(proj) => {
                        let physical_proj =
                            self.arena
                                .alloc(PhysicalOperator::ProjectExec(PhysicalProjectExec {
                                    input: table_scan,
                                    expressions: proj.expressions,
                                    aliases: proj.aliases,
                                }));
                        Some(physical_proj)
                    }
                    _ => None,
                };
            }
        }

        let matching_index = table_def.indexes().iter().find(|idx| {
            if idx.has_expressions() || idx.is_partial() {
                return false;
            }
            idx.columns()
                .next()
                .map(|first_col| first_col.eq_ignore_ascii_case(col_name))
                .unwrap_or(false)
        });

        if let Some(idx) = matching_index {
            let index_name = self.arena.alloc_str(idx.name());
            let table_def_alloc = self.arena.alloc(table_def.clone());

            let index_scan = self.arena.alloc(PhysicalOperator::SecondaryIndexScan(
                PhysicalSecondaryIndexScan {
                    schema: scan.schema,
                    table: scan.table,
                    index_name,
                    table_def: Some(table_def_alloc),
                    reverse,
                    is_unique_index: idx.is_unique(),
                    key_range: None,
                    limit,
                },
            ));

            return match sort.input {
                LogicalOperator::Scan(_) => Some(index_scan),
                LogicalOperator::Project(proj) => {
                    let physical_proj =
                        self.arena
                            .alloc(PhysicalOperator::ProjectExec(PhysicalProjectExec {
                                input: index_scan,
                                expressions: proj.expressions,
                                aliases: proj.aliases,
                            }));
                    Some(physical_proj)
                }
                _ => None,
            };
        }

        None
    }

    pub(crate) fn compute_residual_filter(
        &self,
        predicate: &'a Expr<'a>,
        index_columns: &[String],
    ) -> Option<&'a Expr<'a>> {
        use crate::sql::ast::BinaryOperator;

        match predicate {
            Expr::BinaryOp { left, op, right } => match op {
                BinaryOperator::And => {
                    let left_residual = self.compute_residual_filter(left, index_columns);
                    let right_residual = self.compute_residual_filter(right, index_columns);

                    match (left_residual, right_residual) {
                        (Some(l), Some(r)) => Some(self.arena.alloc(Expr::BinaryOp {
                            left: l,
                            op: BinaryOperator::And,
                            right: r,
                        })),
                        (Some(l), None) => Some(l),
                        (None, Some(r)) => Some(r),
                        (None, None) => None,
                    }
                }
                BinaryOperator::Eq
                | BinaryOperator::Lt
                | BinaryOperator::LtEq
                | BinaryOperator::Gt
                | BinaryOperator::GtEq => {
                    if self.predicate_uses_index_column(predicate, index_columns) {
                        None
                    } else {
                        Some(predicate)
                    }
                }
                _ => Some(predicate),
            },
            Expr::Between { expr, .. } => {
                if self.predicate_uses_index_column(expr, index_columns) {
                    None
                } else {
                    Some(predicate)
                }
            }
            _ => Some(predicate),
        }
    }

    pub(crate) fn predicate_uses_index_column(
        &self,
        expr: &Expr<'a>,
        index_columns: &[String],
    ) -> bool {
        match expr {
            Expr::Column(col_ref) => index_columns
                .iter()
                .any(|c| c.eq_ignore_ascii_case(col_ref.column)),
            Expr::BinaryOp { left, right, .. } => {
                self.predicate_uses_index_column(left, index_columns)
                    || self.predicate_uses_index_column(right, index_columns)
            }
            Expr::Between { expr, .. } => self.predicate_uses_index_column(expr, index_columns),
            _ => false,
        }
    }
}
