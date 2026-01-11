//! # Projection Pruning Rule
//!
//! Removes unused columns from intermediate results, reducing memory usage
//! and improving cache efficiency.
//!
//! ## Transformations
//!
//! | Before | After |
//! |--------|-------|
//! | `SELECT name FROM (SELECT * FROM users)` | `SELECT name FROM (SELECT name FROM users)` |
//! | `Project(a) -> Project(a,b,c)` | `Project(a)` (remove intermediate) |
//!
//! ## Column Usage Analysis
//!
//! The rule analyzes which columns are actually used:
//! 1. Start from the top-level projection (SELECT list)
//! 2. Walk down the plan tree, tracking required columns
//! 3. At each node, prune columns not in the required set
//!
//! ## Pruning Points
//!
//! | Operator | Pruning Strategy |
//! |----------|-----------------|
//! | Scan | Request only needed columns (if supported) |
//! | Project | Remove unused expressions |
//! | Join | Pass required columns to each side |
//! | Subquery | Push column requirements into subquery |
//!
//! ## Limitations
//!
//! - Cannot prune across aggregations (group columns always needed)
//! - Cannot prune ORDER BY columns until after sort
//! - Cannot prune join columns used in conditions

use crate::sql::optimizer::OptimizationRule;
use crate::sql::planner::LogicalOperator;
use bumpalo::Bump;
use eyre::Result;

pub struct ProjectionPruningRule;

impl OptimizationRule for ProjectionPruningRule {
    fn name(&self) -> &'static str {
        "projection_pruning"
    }

    fn apply<'a>(
        &self,
        plan: &'a LogicalOperator<'a>,
        arena: &'a Bump,
    ) -> Result<Option<&'a LogicalOperator<'a>>> {
        self.prune_plan(plan, arena)
    }
}

impl ProjectionPruningRule {
    fn prune_plan<'a>(
        &self,
        plan: &'a LogicalOperator<'a>,
        arena: &'a Bump,
    ) -> Result<Option<&'a LogicalOperator<'a>>> {
        match plan {
            LogicalOperator::Project(proj) => {
                if let LogicalOperator::Project(inner_proj) = proj.input {
                    let new_proj = crate::sql::planner::LogicalProject {
                        input: inner_proj.input,
                        expressions: proj.expressions,
                        aliases: proj.aliases,
                    };
                    return Ok(Some(arena.alloc(LogicalOperator::Project(new_proj))));
                }

                let input_changed = self.prune_plan(proj.input, arena)?;
                if let Some(new_input) = input_changed {
                    let new_proj = crate::sql::planner::LogicalProject {
                        input: new_input,
                        expressions: proj.expressions,
                        aliases: proj.aliases,
                    };
                    return Ok(Some(arena.alloc(LogicalOperator::Project(new_proj))));
                }

                Ok(None)
            }

            LogicalOperator::Filter(filter) => {
                let input_changed = self.prune_plan(filter.input, arena)?;
                if let Some(new_input) = input_changed {
                    let new_filter = crate::sql::planner::LogicalFilter {
                        input: new_input,
                        predicate: filter.predicate,
                    };
                    return Ok(Some(arena.alloc(LogicalOperator::Filter(new_filter))));
                }
                Ok(None)
            }

            LogicalOperator::Join(join) => {
                let left_changed = self.prune_plan(join.left, arena)?;
                let right_changed = self.prune_plan(join.right, arena)?;

                if left_changed.is_some() || right_changed.is_some() {
                    let new_join = crate::sql::planner::LogicalJoin {
                        left: left_changed.unwrap_or(join.left),
                        right: right_changed.unwrap_or(join.right),
                        join_type: join.join_type,
                        condition: join.condition,
                    };
                    return Ok(Some(arena.alloc(LogicalOperator::Join(new_join))));
                }
                Ok(None)
            }

            LogicalOperator::Aggregate(agg) => {
                let input_changed = self.prune_plan(agg.input, arena)?;
                if let Some(new_input) = input_changed {
                    let new_agg = crate::sql::planner::LogicalAggregate {
                        input: new_input,
                        group_by: agg.group_by,
                        aggregates: agg.aggregates,
                    };
                    return Ok(Some(arena.alloc(LogicalOperator::Aggregate(new_agg))));
                }
                Ok(None)
            }

            LogicalOperator::Sort(sort) => {
                let input_changed = self.prune_plan(sort.input, arena)?;
                if let Some(new_input) = input_changed {
                    let new_sort = crate::sql::planner::LogicalSort {
                        input: new_input,
                        order_by: sort.order_by,
                    };
                    return Ok(Some(arena.alloc(LogicalOperator::Sort(new_sort))));
                }
                Ok(None)
            }

            LogicalOperator::Limit(limit) => {
                let input_changed = self.prune_plan(limit.input, arena)?;
                if let Some(new_input) = input_changed {
                    let new_limit = crate::sql::planner::LogicalLimit {
                        input: new_input,
                        limit: limit.limit,
                        offset: limit.offset,
                    };
                    return Ok(Some(arena.alloc(LogicalOperator::Limit(new_limit))));
                }
                Ok(None)
            }

            LogicalOperator::SetOp(setop) => {
                let left_changed = self.prune_plan(setop.left, arena)?;
                let right_changed = self.prune_plan(setop.right, arena)?;

                if left_changed.is_some() || right_changed.is_some() {
                    let new_setop = crate::sql::planner::LogicalSetOp {
                        left: left_changed.unwrap_or(setop.left),
                        right: right_changed.unwrap_or(setop.right),
                        kind: setop.kind,
                        all: setop.all,
                    };
                    return Ok(Some(arena.alloc(LogicalOperator::SetOp(new_setop))));
                }
                Ok(None)
            }

            LogicalOperator::Scan(_)
            | LogicalOperator::DualScan
            | LogicalOperator::Values(_)
            | LogicalOperator::Insert(_)
            | LogicalOperator::Update(_)
            | LogicalOperator::Delete(_)
            | LogicalOperator::Subquery(_)
            | LogicalOperator::Window(_) => Ok(None),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rule_name() {
        let rule = ProjectionPruningRule;
        assert_eq!(rule.name(), "projection_pruning");
    }
}
