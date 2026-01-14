//! # Join Condition Extraction Rule
//!
//! Extracts equi-join conditions from WHERE clauses and moves them into
//! the join operator, enabling hash joins for comma joins.
//!
//! ## Problem
//!
//! When parsing `FROM a, b WHERE a.id = b.id`, the parser creates:
//! - A Cross join with no condition
//! - A separate Filter with the `a.id = b.id` predicate
//!
//! This prevents hash join optimization because the join has no equi-keys.
//!
//! ## Transformation
//!
//! ```text
//! Before:
//! Filter(a.id = b.id AND a.x > 10)
//!   └─ Join(Cross, condition=None)
//!        ├─ Scan(a)
//!        └─ Scan(b)
//!
//! After:
//! Filter(a.x > 10)
//!   └─ Join(Inner, condition=a.id = b.id)
//!        ├─ Scan(a)
//!        └─ Scan(b)
//! ```
//!
//! ## Equi-Join Detection
//!
//! A predicate is an equi-join condition if:
//! 1. It's an equality comparison (`=`)
//! 2. Both sides are column references
//! 3. The columns reference different tables (one from left, one from right)

use crate::sql::ast::{BinaryOperator, Expr, JoinType};
use crate::sql::optimizer::OptimizationRule;
use crate::sql::planner::{LogicalFilter, LogicalJoin, LogicalOperator};
use bumpalo::Bump;
use eyre::Result;
use smallvec::SmallVec;
use std::collections::HashSet;

pub struct JoinConditionExtractionRule;

impl OptimizationRule for JoinConditionExtractionRule {
    fn name(&self) -> &'static str {
        "join_condition_extraction"
    }

    fn apply<'a>(
        &self,
        plan: &'a LogicalOperator<'a>,
        arena: &'a Bump,
    ) -> Result<Option<&'a LogicalOperator<'a>>> {
        self.transform(plan, arena)
    }
}

impl JoinConditionExtractionRule {
    fn transform<'a>(
        &self,
        plan: &'a LogicalOperator<'a>,
        arena: &'a Bump,
    ) -> Result<Option<&'a LogicalOperator<'a>>> {
        match plan {
            LogicalOperator::Filter(filter) => {
                if let LogicalOperator::Join(join) = filter.input {
                    if join.join_type == JoinType::Cross && join.condition.is_none() {
                        let left_tables = self.collect_table_names(join.left);
                        let right_tables = self.collect_table_names(join.right);

                        let (join_conditions, remaining_predicates) =
                            self.split_predicates(filter.predicate, &left_tables, &right_tables);

                        if !join_conditions.is_empty() {
                            let join_condition = self.combine_predicates(&join_conditions, arena);

                            let new_join = arena.alloc(LogicalOperator::Join(LogicalJoin {
                                left: join.left,
                                right: join.right,
                                join_type: JoinType::Inner,
                                condition: Some(join_condition),
                            }));

                            if remaining_predicates.is_empty() {
                                return Ok(Some(new_join));
                            } else {
                                let remaining =
                                    self.combine_predicates(&remaining_predicates, arena);
                                let new_filter =
                                    arena.alloc(LogicalOperator::Filter(LogicalFilter {
                                        input: new_join,
                                        predicate: remaining,
                                    }));
                                return Ok(Some(new_filter));
                            }
                        }
                    }
                }

                let transformed_input = self.transform(filter.input, arena)?;
                if let Some(new_input) = transformed_input {
                    let new_filter = arena.alloc(LogicalOperator::Filter(LogicalFilter {
                        input: new_input,
                        predicate: filter.predicate,
                    }));
                    return Ok(Some(new_filter));
                }
                Ok(None)
            }

            LogicalOperator::Join(join) => {
                let left_changed = self.transform(join.left, arena)?;
                let right_changed = self.transform(join.right, arena)?;

                if left_changed.is_some() || right_changed.is_some() {
                    let new_join = arena.alloc(LogicalOperator::Join(LogicalJoin {
                        left: left_changed.unwrap_or(join.left),
                        right: right_changed.unwrap_or(join.right),
                        join_type: join.join_type,
                        condition: join.condition,
                    }));
                    return Ok(Some(new_join));
                }
                Ok(None)
            }

            LogicalOperator::Project(project) => {
                let transformed = self.transform(project.input, arena)?;
                if let Some(new_input) = transformed {
                    let new_project =
                        arena.alloc(LogicalOperator::Project(crate::sql::planner::LogicalProject {
                            input: new_input,
                            expressions: project.expressions,
                            aliases: project.aliases,
                        }));
                    return Ok(Some(new_project));
                }
                Ok(None)
            }

            LogicalOperator::Aggregate(agg) => {
                let transformed = self.transform(agg.input, arena)?;
                if let Some(new_input) = transformed {
                    let new_agg = arena.alloc(LogicalOperator::Aggregate(
                        crate::sql::planner::LogicalAggregate {
                            input: new_input,
                            group_by: agg.group_by,
                            aggregates: agg.aggregates,
                        },
                    ));
                    return Ok(Some(new_agg));
                }
                Ok(None)
            }

            LogicalOperator::Sort(sort) => {
                let transformed = self.transform(sort.input, arena)?;
                if let Some(new_input) = transformed {
                    let new_sort =
                        arena.alloc(LogicalOperator::Sort(crate::sql::planner::LogicalSort {
                            input: new_input,
                            order_by: sort.order_by,
                        }));
                    return Ok(Some(new_sort));
                }
                Ok(None)
            }

            LogicalOperator::Limit(limit) => {
                let transformed = self.transform(limit.input, arena)?;
                if let Some(new_input) = transformed {
                    let new_limit =
                        arena.alloc(LogicalOperator::Limit(crate::sql::planner::LogicalLimit {
                            input: new_input,
                            limit: limit.limit,
                            offset: limit.offset,
                        }));
                    return Ok(Some(new_limit));
                }
                Ok(None)
            }

            LogicalOperator::Subquery(subq) => {
                let transformed = self.transform(subq.plan, arena)?;
                if let Some(new_plan) = transformed {
                    let new_subq = arena.alloc(LogicalOperator::Subquery(
                        crate::sql::planner::LogicalSubquery {
                            plan: new_plan,
                            alias: subq.alias,
                            output_schema: subq.output_schema.clone(),
                        },
                    ));
                    return Ok(Some(new_subq));
                }
                Ok(None)
            }

            _ => Ok(None),
        }
    }

    fn collect_table_names<'a>(&self, op: &'a LogicalOperator<'a>) -> HashSet<&'a str> {
        let mut tables = HashSet::new();
        self.collect_tables_recursive(op, &mut tables);
        tables
    }

    fn collect_tables_recursive<'a>(
        &self,
        op: &'a LogicalOperator<'a>,
        tables: &mut HashSet<&'a str>,
    ) {
        match op {
            LogicalOperator::Scan(scan) => {
                if let Some(alias) = scan.alias {
                    tables.insert(alias);
                } else {
                    tables.insert(scan.table);
                }
            }
            LogicalOperator::Join(join) => {
                self.collect_tables_recursive(join.left, tables);
                self.collect_tables_recursive(join.right, tables);
            }
            LogicalOperator::Filter(filter) => {
                self.collect_tables_recursive(filter.input, tables);
            }
            LogicalOperator::Project(project) => {
                self.collect_tables_recursive(project.input, tables);
            }
            LogicalOperator::Aggregate(agg) => {
                self.collect_tables_recursive(agg.input, tables);
            }
            LogicalOperator::Sort(sort) => {
                self.collect_tables_recursive(sort.input, tables);
            }
            LogicalOperator::Limit(limit) => {
                self.collect_tables_recursive(limit.input, tables);
            }
            LogicalOperator::Subquery(subq) => {
                tables.insert(subq.alias);
            }
            _ => {}
        }
    }

    fn split_predicates<'a>(
        &self,
        predicate: &'a Expr<'a>,
        left_tables: &HashSet<&'a str>,
        right_tables: &HashSet<&'a str>,
    ) -> (SmallVec<[&'a Expr<'a>; 8]>, SmallVec<[&'a Expr<'a>; 8]>) {
        let mut join_conditions: SmallVec<[&'a Expr<'a>; 8]> = SmallVec::new();
        let mut remaining: SmallVec<[&'a Expr<'a>; 8]> = SmallVec::new();

        let parts = self.flatten_and(predicate);
        for part in parts {
            if self.is_equi_join_condition(part, left_tables, right_tables) {
                join_conditions.push(part);
            } else {
                remaining.push(part);
            }
        }

        (join_conditions, remaining)
    }

    fn flatten_and<'a>(&self, expr: &'a Expr<'a>) -> SmallVec<[&'a Expr<'a>; 8]> {
        match expr {
            Expr::BinaryOp {
                left,
                op: BinaryOperator::And,
                right,
            } => {
                let mut parts = self.flatten_and(left);
                parts.extend(self.flatten_and(right));
                parts
            }
            _ => smallvec::smallvec![expr],
        }
    }

    fn is_equi_join_condition<'a>(
        &self,
        expr: &'a Expr<'a>,
        left_tables: &HashSet<&'a str>,
        right_tables: &HashSet<&'a str>,
    ) -> bool {
        if let Expr::BinaryOp {
            left,
            op: BinaryOperator::Eq,
            right,
        } = expr
        {
            let left_table = self.get_column_table(left);
            let right_table = self.get_column_table(right);

            if let (Some(lt), Some(rt)) = (left_table, right_table) {
                let left_in_left = left_tables.contains(lt);
                let left_in_right = right_tables.contains(lt);
                let right_in_left = left_tables.contains(rt);
                let right_in_right = right_tables.contains(rt);

                return (left_in_left && right_in_right) || (left_in_right && right_in_left);
            }
        }
        false
    }

    fn get_column_table<'a>(&self, expr: &'a Expr<'a>) -> Option<&'a str> {
        match expr {
            Expr::Column(col) => col.table,
            _ => None,
        }
    }

    fn combine_predicates<'a>(&self, predicates: &[&'a Expr<'a>], arena: &'a Bump) -> &'a Expr<'a> {
        if predicates.len() == 1 {
            return predicates[0];
        }

        let mut result = predicates[0];
        for pred in &predicates[1..] {
            result = arena.alloc(Expr::BinaryOp {
                left: result,
                op: BinaryOperator::And,
                right: pred,
            });
        }
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rule_name() {
        let rule = JoinConditionExtractionRule;
        assert_eq!(rule.name(), "join_condition_extraction");
    }
}
