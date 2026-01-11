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
    PhysicalHashAggregate, PhysicalHashAntiJoin, PhysicalHashSemiJoin, PhysicalLimitExec,
    PhysicalNestedLoopJoin, PhysicalOperator, PhysicalPlan, PhysicalProjectExec,
    PhysicalSetOpExec, PhysicalSortExec, PhysicalSubqueryExec,
    PhysicalTableScan, PhysicalTopKExec, PhysicalWindowExec,
};
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
}
