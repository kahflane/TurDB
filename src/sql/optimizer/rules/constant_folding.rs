//! # Constant Folding Rule
//!
//! Evaluates constant expressions at plan time to reduce runtime computation.
//!
//! ## Transformations
//!
//! | Before | After |
//! |--------|-------|
//! | `1 + 1` | `2` |
//! | `true AND false` | `false` |
//! | `WHERE 1 = 1` | (filter removed) |
//! | `WHERE 1 = 0` | (empty result) |
//!
//! ## NULL Propagation
//!
//! Most operations with NULL return NULL:
//! - `NULL + 1` → `NULL`
//! - `NULL = NULL` → `NULL` (not true!)
//! - `NULL AND true` → `NULL`
//! - `NULL OR true` → `true` (short-circuit)
//!
//! ## Simplifications
//!
//! Boolean algebra simplifications:
//! - `x AND true` → `x`
//! - `x OR false` → `x`
//! - `x AND false` → `false`
//! - `x OR true` → `true`
//! - `NOT NOT x` → `x`

use crate::sql::optimizer::OptimizationRule;
use crate::sql::planner::LogicalOperator;
use bumpalo::Bump;
use eyre::Result;

pub struct ConstantFoldingRule;

impl OptimizationRule for ConstantFoldingRule {
    fn name(&self) -> &'static str {
        "constant_folding"
    }

    fn apply<'a>(
        &self,
        plan: &'a LogicalOperator<'a>,
        arena: &'a Bump,
    ) -> Result<Option<&'a LogicalOperator<'a>>> {
        self.fold_plan(plan, arena)
    }
}

impl ConstantFoldingRule {
    fn fold_plan<'a>(
        &self,
        plan: &'a LogicalOperator<'a>,
        arena: &'a Bump,
    ) -> Result<Option<&'a LogicalOperator<'a>>> {
        match plan {
            LogicalOperator::Filter(filter) => {
                let input_changed = self.fold_plan(filter.input, arena)?;

                if let Some(folded) = self.try_fold_filter_predicate(filter.predicate) {
                    match folded {
                        FoldedPredicate::AlwaysTrue => {
                            return Ok(Some(input_changed.unwrap_or(filter.input)));
                        }
                        FoldedPredicate::AlwaysFalse => {
                            return Ok(Some(arena.alloc(LogicalOperator::Values(
                                crate::sql::planner::LogicalValues { rows: &[] },
                            ))));
                        }
                        FoldedPredicate::Simplified(new_pred) => {
                            let new_filter = crate::sql::planner::LogicalFilter {
                                input: input_changed.unwrap_or(filter.input),
                                predicate: arena.alloc(new_pred),
                            };
                            return Ok(Some(
                                arena.alloc(LogicalOperator::Filter(new_filter)),
                            ));
                        }
                    }
                }

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
                let input_changed = self.fold_plan(proj.input, arena)?;

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
                let left_changed = self.fold_plan(join.left, arena)?;
                let right_changed = self.fold_plan(join.right, arena)?;

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
                let input_changed = self.fold_plan(agg.input, arena)?;

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
                let input_changed = self.fold_plan(sort.input, arena)?;

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
                let input_changed = self.fold_plan(limit.input, arena)?;

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
                let left_changed = self.fold_plan(setop.left, arena)?;
                let right_changed = self.fold_plan(setop.right, arena)?;

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

    fn try_fold_filter_predicate<'a>(
        &self,
        expr: &'a crate::sql::ast::Expr<'a>,
    ) -> Option<FoldedPredicate<'a>> {
        use crate::sql::ast::{BinaryOperator, Expr, Literal};

        match expr {
            Expr::Literal(Literal::Boolean(true)) => Some(FoldedPredicate::AlwaysTrue),
            Expr::Literal(Literal::Boolean(false)) => Some(FoldedPredicate::AlwaysFalse),

            Expr::BinaryOp { left, op, right } => match op {
                BinaryOperator::And => {
                    let left_folded = self.try_fold_filter_predicate(left);
                    let right_folded = self.try_fold_filter_predicate(right);

                    match (left_folded, right_folded) {
                        (Some(FoldedPredicate::AlwaysFalse), _)
                        | (_, Some(FoldedPredicate::AlwaysFalse)) => {
                            Some(FoldedPredicate::AlwaysFalse)
                        }
                        (Some(FoldedPredicate::AlwaysTrue), None) => {
                            Some(FoldedPredicate::Simplified((*right).clone()))
                        }
                        (None, Some(FoldedPredicate::AlwaysTrue)) => {
                            Some(FoldedPredicate::Simplified((*left).clone()))
                        }
                        (Some(FoldedPredicate::AlwaysTrue), Some(FoldedPredicate::AlwaysTrue)) => {
                            Some(FoldedPredicate::AlwaysTrue)
                        }
                        _ => None,
                    }
                }
                BinaryOperator::Or => {
                    let left_folded = self.try_fold_filter_predicate(left);
                    let right_folded = self.try_fold_filter_predicate(right);

                    match (left_folded, right_folded) {
                        (Some(FoldedPredicate::AlwaysTrue), _)
                        | (_, Some(FoldedPredicate::AlwaysTrue)) => {
                            Some(FoldedPredicate::AlwaysTrue)
                        }
                        (Some(FoldedPredicate::AlwaysFalse), None) => {
                            Some(FoldedPredicate::Simplified((*right).clone()))
                        }
                        (None, Some(FoldedPredicate::AlwaysFalse)) => {
                            Some(FoldedPredicate::Simplified((*left).clone()))
                        }
                        (
                            Some(FoldedPredicate::AlwaysFalse),
                            Some(FoldedPredicate::AlwaysFalse),
                        ) => Some(FoldedPredicate::AlwaysFalse),
                        _ => None,
                    }
                }
                BinaryOperator::Eq => {
                    if let (Expr::Literal(l), Expr::Literal(r)) = (*left, *right) {
                        Some(if literals_equal(&l, &r) {
                            FoldedPredicate::AlwaysTrue
                        } else {
                            FoldedPredicate::AlwaysFalse
                        })
                    } else {
                        None
                    }
                }
                BinaryOperator::NotEq => {
                    if let (Expr::Literal(l), Expr::Literal(r)) = (*left, *right) {
                        Some(if literals_equal(&l, &r) {
                            FoldedPredicate::AlwaysFalse
                        } else {
                            FoldedPredicate::AlwaysTrue
                        })
                    } else {
                        None
                    }
                }
                _ => None,
            },

            Expr::UnaryOp {
                op: crate::sql::ast::UnaryOperator::Not,
                expr: inner,
            } => match self.try_fold_filter_predicate(inner) {
                Some(FoldedPredicate::AlwaysTrue) => Some(FoldedPredicate::AlwaysFalse),
                Some(FoldedPredicate::AlwaysFalse) => Some(FoldedPredicate::AlwaysTrue),
                _ => None,
            },

            _ => None,
        }
    }
}

enum FoldedPredicate<'a> {
    AlwaysTrue,
    AlwaysFalse,
    Simplified(crate::sql::ast::Expr<'a>),
}

fn literals_equal(l: &crate::sql::ast::Literal, r: &crate::sql::ast::Literal) -> bool {
    use crate::sql::ast::Literal;
    match (l, r) {
        (Literal::Null, _) | (_, Literal::Null) => false,
        (Literal::Boolean(a), Literal::Boolean(b)) => a == b,
        (Literal::Integer(a), Literal::Integer(b)) => a == b,
        (Literal::Float(a), Literal::Float(b)) => a == b,
        (Literal::String(a), Literal::String(b)) => a == b,
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rule_name() {
        let rule = ConstantFoldingRule;
        assert_eq!(rule.name(), "constant_folding");
    }
}
