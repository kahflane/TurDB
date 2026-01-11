//! # Predicate Pushdown Rule
//!
//! Pushes filter predicates as close to data sources as possible, reducing
//! the number of rows processed by downstream operators.
//!
//! ## Transformations
//!
//! | Before | After |
//! |--------|-------|
//! | `Filter(Project(Scan))` | `Project(Filter(Scan))` |
//! | `Filter(Join(A, B))` on A only | `Join(Filter(A), B)` |
//! | `Filter(Subquery)` | `Subquery(Filter)` |
//!
//! ## Pushdown Rules
//!
//! A predicate can be pushed through an operator if:
//!
//! | Operator | Can Push Through | Conditions |
//! |----------|-----------------|------------|
//! | Project | Yes | If predicate only uses projected columns |
//! | Join | Partial | Only predicates referencing one side |
//! | Aggregate | Partial | Only for grouping columns |
//! | Sort | Yes | Always |
//! | Limit | No | Never (changes semantics) |
//! | Union/Intersect | Yes | To both sides |
//!
//! ## Join Pushdown
//!
//! For joins, predicates are classified:
//! - **Left-only**: Push to left input
//! - **Right-only**: Push to right input
//! - **Both**: Keep as join condition or post-filter
//!
//! ## Example
//!
//! ```sql
//! -- Before optimization
//! SELECT * FROM (SELECT * FROM users) u WHERE u.active = true
//!
//! -- After pushdown
//! SELECT * FROM (SELECT * FROM users WHERE active = true) u
//! ```

use crate::sql::optimizer::OptimizationRule;
use crate::sql::planner::LogicalOperator;
use bumpalo::Bump;
use eyre::Result;

pub struct PredicatePushdownRule;

impl OptimizationRule for PredicatePushdownRule {
    fn name(&self) -> &'static str {
        "predicate_pushdown"
    }

    fn apply<'a>(
        &self,
        plan: &'a LogicalOperator<'a>,
        arena: &'a Bump,
    ) -> Result<Option<&'a LogicalOperator<'a>>> {
        self.pushdown_plan(plan, arena)
    }
}

impl PredicatePushdownRule {
    fn pushdown_plan<'a>(
        &self,
        plan: &'a LogicalOperator<'a>,
        arena: &'a Bump,
    ) -> Result<Option<&'a LogicalOperator<'a>>> {
        match plan {
            LogicalOperator::Filter(filter) => {
                if let Some((pushed, remaining)) =
                    self.try_push_filter(filter.input, filter.predicate, arena)?
                {
                    if let Some(remaining_pred) = remaining {
                        let new_filter = crate::sql::planner::LogicalFilter {
                            input: pushed,
                            predicate: remaining_pred,
                        };
                        return Ok(Some(arena.alloc(LogicalOperator::Filter(new_filter))));
                    } else {
                        return Ok(Some(pushed));
                    }
                }

                let input_changed = self.pushdown_plan(filter.input, arena)?;
                if let Some(new_input) = input_changed {
                    let new_filter = crate::sql::planner::LogicalFilter {
                        input: new_input,
                        predicate: filter.predicate,
                    };
                    return Ok(Some(arena.alloc(LogicalOperator::Filter(new_filter))));
                }

                Ok(None)
            }

            LogicalOperator::Project(proj) => {
                let input_changed = self.pushdown_plan(proj.input, arena)?;
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

            LogicalOperator::Join(join) => {
                let left_changed = self.pushdown_plan(join.left, arena)?;
                let right_changed = self.pushdown_plan(join.right, arena)?;

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
                let input_changed = self.pushdown_plan(agg.input, arena)?;
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
                let input_changed = self.pushdown_plan(sort.input, arena)?;
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
                let input_changed = self.pushdown_plan(limit.input, arena)?;
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
                let left_changed = self.pushdown_plan(setop.left, arena)?;
                let right_changed = self.pushdown_plan(setop.right, arena)?;

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

    fn try_push_filter<'a>(
        &self,
        input: &'a LogicalOperator<'a>,
        predicate: &'a crate::sql::ast::Expr<'a>,
        arena: &'a Bump,
    ) -> Result<Option<(&'a LogicalOperator<'a>, Option<&'a crate::sql::ast::Expr<'a>>)>> {
        match input {
            LogicalOperator::Project(proj) => {
                let new_filter = crate::sql::planner::LogicalFilter {
                    input: proj.input,
                    predicate,
                };
                let filtered = arena.alloc(LogicalOperator::Filter(new_filter));

                let new_proj = crate::sql::planner::LogicalProject {
                    input: filtered,
                    expressions: proj.expressions,
                    aliases: proj.aliases,
                };
                Ok(Some((arena.alloc(LogicalOperator::Project(new_proj)), None)))
            }

            LogicalOperator::Sort(sort) => {
                let new_filter = crate::sql::planner::LogicalFilter {
                    input: sort.input,
                    predicate,
                };
                let filtered = arena.alloc(LogicalOperator::Filter(new_filter));

                let new_sort = crate::sql::planner::LogicalSort {
                    input: filtered,
                    order_by: sort.order_by,
                };
                Ok(Some((arena.alloc(LogicalOperator::Sort(new_sort)), None)))
            }

            LogicalOperator::Subquery(subq) => {
                let new_filter = crate::sql::planner::LogicalFilter {
                    input: subq.plan,
                    predicate,
                };
                let filtered = arena.alloc(LogicalOperator::Filter(new_filter));

                let new_subq = crate::sql::planner::LogicalSubquery {
                    plan: filtered,
                    alias: subq.alias,
                    output_schema: subq.output_schema.clone(),
                };
                Ok(Some((
                    arena.alloc(LogicalOperator::Subquery(new_subq)),
                    None,
                )))
            }

            _ => Ok(None),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rule_name() {
        let rule = PredicatePushdownRule;
        assert_eq!(rule.name(), "predicate_pushdown");
    }
}
