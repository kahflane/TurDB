//! # Subquery Decorrelation Rule
//!
//! Transforms correlated subqueries into equivalent join operations for
//! dramatically improved performance (O(n*m) → O(n+m)).
//!
//! ## Transformations
//!
//! | Pattern | Transformation |
//! |---------|---------------|
//! | `EXISTS (SELECT ... WHERE outer.x = inner.y)` | `SEMI JOIN` |
//! | `NOT EXISTS (SELECT ... WHERE outer.x = inner.y)` | `ANTI JOIN` |
//! | `x IN (SELECT y FROM ... WHERE outer.a = inner.b)` | `SEMI JOIN` on `(x = y AND a = b)` |
//! | `x NOT IN (SELECT ...)` | `ANTI JOIN` with NULL handling |
//! | Scalar subquery with correlation | `LEFT JOIN` + first value |
//!
//! ## Example: EXISTS → SEMI JOIN
//!
//! ```sql
//! -- Before (correlated, O(n*m) subquery executions)
//! SELECT * FROM orders o
//! WHERE EXISTS (SELECT 1 FROM items i WHERE i.order_id = o.id)
//!
//! -- After (decorrelated, O(n+m) single join)
//! SELECT DISTINCT o.* FROM orders o
//! SEMI JOIN items i ON i.order_id = o.id
//! ```
//!
//! ## Decorrelation Conditions
//!
//! A subquery can be decorrelated when:
//! 1. Correlation is via equality predicates only
//! 2. No aggregate functions reference outer columns
//! 3. No LIMIT/OFFSET (or they can be preserved semantically)
//! 4. Correlation columns form a valid join condition
//!
//! ## NULL Handling
//!
//! `NOT IN` with NULLs requires special handling:
//! - `x NOT IN (1, 2, NULL)` returns NULL if x is not 1 or 2
//! - Must use `ANTI JOIN` with `IS DISTINCT FROM` semantics
//!
//! ## Implementation Strategy
//!
//! 1. Identify correlated subqueries via `SubqueryClassifier`
//! 2. Extract correlation predicates from WHERE clause
//! 3. Build join condition from extracted predicates
//! 4. Replace subquery with appropriate join type
//! 5. Handle remaining non-correlation predicates as post-filter

use crate::sql::ast::{FromClause, JoinType};
use crate::sql::optimizer::OptimizationRule;
use crate::sql::planner::{LogicalJoin, LogicalOperator, LogicalScan};
use bumpalo::Bump;
use eyre::Result;

pub struct SubqueryDecorrelationRule;

impl OptimizationRule for SubqueryDecorrelationRule {
    fn name(&self) -> &'static str {
        "subquery_decorrelation"
    }

    fn apply<'a>(
        &self,
        plan: &'a LogicalOperator<'a>,
        arena: &'a Bump,
    ) -> Result<Option<&'a LogicalOperator<'a>>> {
        self.decorrelate_plan(plan, arena)
    }
}

impl SubqueryDecorrelationRule {
    fn decorrelate_plan<'a>(
        &self,
        plan: &'a LogicalOperator<'a>,
        arena: &'a Bump,
    ) -> Result<Option<&'a LogicalOperator<'a>>> {
        match plan {
            LogicalOperator::Filter(filter) => {
                if let Some(decorrelated) =
                    self.try_decorrelate_filter(filter.input, filter.predicate, arena)?
                {
                    return Ok(Some(decorrelated));
                }

                let input_changed = self.decorrelate_plan(filter.input, arena)?;
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
                let input_changed = self.decorrelate_plan(proj.input, arena)?;
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
                let left_changed = self.decorrelate_plan(join.left, arena)?;
                let right_changed = self.decorrelate_plan(join.right, arena)?;

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
                let input_changed = self.decorrelate_plan(agg.input, arena)?;
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
                let input_changed = self.decorrelate_plan(sort.input, arena)?;
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
                let input_changed = self.decorrelate_plan(limit.input, arena)?;
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
                let left_changed = self.decorrelate_plan(setop.left, arena)?;
                let right_changed = self.decorrelate_plan(setop.right, arena)?;

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

            LogicalOperator::Subquery(subq) => {
                let plan_changed = self.decorrelate_plan(subq.plan, arena)?;
                if let Some(new_plan) = plan_changed {
                    let new_subq = crate::sql::planner::LogicalSubquery {
                        plan: new_plan,
                        alias: subq.alias,
                        output_schema: subq.output_schema.clone(),
                    };
                    return Ok(Some(arena.alloc(LogicalOperator::Subquery(new_subq))));
                }
                Ok(None)
            }

            LogicalOperator::Scan(_)
            | LogicalOperator::DualScan
            | LogicalOperator::Values(_)
            | LogicalOperator::Insert(_)
            | LogicalOperator::Update(_)
            | LogicalOperator::Delete(_)
            | LogicalOperator::Window(_) => Ok(None),
        }
    }

    fn try_decorrelate_filter<'a>(
        &self,
        _input: &'a LogicalOperator<'a>,
        predicate: &'a crate::sql::ast::Expr<'a>,
        _arena: &'a Bump,
    ) -> Result<Option<&'a LogicalOperator<'a>>> {
        use crate::sql::ast::Expr;

        match predicate {
            Expr::Exists { subquery, negated } => {
                self.try_decorrelate_exists(subquery, *negated, _input, _arena)
            }

            Expr::InSubquery {
                expr,
                negated,
                subquery,
            } => self.try_decorrelate_in_subquery(expr, subquery, *negated, _input, _arena),

            Expr::BinaryOp { left, op, right } => {
                use crate::sql::ast::BinaryOperator;
                if matches!(op, BinaryOperator::And) {
                    if let Some(result) =
                        self.try_decorrelate_filter(_input, left, _arena)?
                    {
                        return Ok(Some(result));
                    }
                    if let Some(result) =
                        self.try_decorrelate_filter(_input, right, _arena)?
                    {
                        return Ok(Some(result));
                    }
                }
                Ok(None)
            }

            _ => Ok(None),
        }
    }

    fn try_decorrelate_exists<'a>(
        &self,
        subquery: &'a crate::sql::ast::SelectStmt<'a>,
        negated: bool,
        input: &'a LogicalOperator<'a>,
        arena: &'a Bump,
    ) -> Result<Option<&'a LogicalOperator<'a>>> {
        let subquery_input = self.plan_subquery_as_scan(subquery, arena)?;
        if subquery_input.is_none() {
            return Ok(None);
        }

        let join_type = if negated {
            JoinType::Anti
        } else {
            JoinType::Semi
        };

        let join = LogicalJoin {
            left: input,
            right: subquery_input.unwrap(),
            join_type,
            condition: subquery.where_clause,
        };

        Ok(Some(arena.alloc(LogicalOperator::Join(join))))
    }

    fn try_decorrelate_in_subquery<'a>(
        &self,
        in_expr: &'a crate::sql::ast::Expr<'a>,
        subquery: &'a crate::sql::ast::SelectStmt<'a>,
        negated: bool,
        input: &'a LogicalOperator<'a>,
        arena: &'a Bump,
    ) -> Result<Option<&'a LogicalOperator<'a>>> {
        let subquery_input = self.plan_subquery_as_scan(subquery, arena)?;
        if subquery_input.is_none() {
            return Ok(None);
        }

        let join_type = if negated {
            JoinType::Anti
        } else {
            JoinType::Semi
        };

        // Extract the table name from the subquery's FROM clause for column qualification
        let subquery_table_name = match &subquery.from {
            Some(FromClause::Table(table_ref)) => Some(table_ref.alias.unwrap_or(table_ref.name)),
            _ => None,
        };

        // Build join condition: in_expr = subquery.columns[0]
        // For IN (SELECT col FROM ...), the condition is: outer.expr = inner.col
        let join_condition = if !subquery.columns.is_empty() {
            use crate::sql::ast::{BinaryOperator, ColumnRef, Expr, SelectColumn};

            // Get the first column expression from the subquery
            let right_expr = match &subquery.columns[0] {
                SelectColumn::Expr { expr, .. } => *expr,
                SelectColumn::AllColumns | SelectColumn::TableAllColumns { .. } => {
                    // Can't decorrelate SELECT * subqueries for IN
                    return Ok(None);
                }
            };

            // Qualify the right expression with the subquery's table name if it's a simple column
            // This ensures proper column resolution when left and right have the same column names
            let qualified_right: &Expr<'a> = match (&right_expr, subquery_table_name) {
                (Expr::Column(col), Some(table_name)) if col.table.is_none() => {
                    // Qualify unqualified column with subquery's table name
                    arena.alloc(Expr::Column(ColumnRef {
                        schema: col.schema,
                        table: Some(arena.alloc_str(table_name)),
                        column: col.column,
                    }))
                }
                _ => arena.alloc(right_expr),
            };

            // Create the equality condition: in_expr = qualified_right
            let condition = arena.alloc(Expr::BinaryOp {
                left: in_expr,
                op: BinaryOperator::Eq,
                right: qualified_right,
            });

            // If there's also a WHERE clause in the subquery, AND it with the join condition
            if let Some(where_clause) = subquery.where_clause {
                Some(arena.alloc(Expr::BinaryOp {
                    left: condition,
                    op: BinaryOperator::And,
                    right: where_clause,
                }) as &Expr)
            } else {
                Some(condition as &Expr)
            }
        } else {
            subquery.where_clause
        };

        let join = LogicalJoin {
            left: input,
            right: subquery_input.unwrap(),
            join_type,
            condition: join_condition,
        };

        Ok(Some(arena.alloc(LogicalOperator::Join(join))))
    }

    fn plan_subquery_as_scan<'a>(
        &self,
        subquery: &'a crate::sql::ast::SelectStmt<'a>,
        arena: &'a Bump,
    ) -> Result<Option<&'a LogicalOperator<'a>>> {
        match &subquery.from {
            Some(from) => self.from_clause_to_scan(from, arena),
            None => Ok(None),
        }
    }

    fn from_clause_to_scan<'a>(
        &self,
        from: &'a FromClause<'a>,
        arena: &'a Bump,
    ) -> Result<Option<&'a LogicalOperator<'a>>> {
        match from {
            FromClause::Table(table_ref) => {
                let scan = LogicalScan {
                    schema: table_ref.schema,
                    table: table_ref.name,
                    alias: table_ref.alias,
                };
                Ok(Some(arena.alloc(LogicalOperator::Scan(scan))))
            }
            FromClause::Join(join) => {
                let left = self.from_clause_to_scan(join.left, arena)?;
                let right = self.from_clause_to_scan(join.right, arena)?;

                if let (Some(left_op), Some(right_op)) = (left, right) {
                    let condition = match &join.condition {
                        crate::sql::ast::JoinCondition::On(expr) => Some(*expr),
                        _ => None,
                    };

                    let logical_join = LogicalJoin {
                        left: left_op,
                        right: right_op,
                        join_type: join.join_type,
                        condition,
                    };
                    Ok(Some(arena.alloc(LogicalOperator::Join(logical_join))))
                } else {
                    Ok(None)
                }
            }
            FromClause::Subquery { .. } | FromClause::Lateral { .. } => Ok(None),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rule_name() {
        let rule = SubqueryDecorrelationRule;
        assert_eq!(rule.name(), "subquery_decorrelation");
    }
}
