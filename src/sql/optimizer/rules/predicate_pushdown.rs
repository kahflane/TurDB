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
use smallvec::SmallVec;

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
                if self.predicate_uses_alias(predicate, proj.aliases) {
                    return Ok(None);
                }
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

            LogicalOperator::Join(join) => {
                let left_tables = self.collect_table_names(join.left);
                let right_tables = self.collect_table_names(join.right);
                let pred_tables = self.collect_predicate_tables(predicate);

                let refs_left = pred_tables.iter().any(|t| left_tables.contains(*t));
                let refs_right = pred_tables.iter().any(|t| right_tables.contains(*t));

                if refs_left && !refs_right {
                    let new_filter = crate::sql::planner::LogicalFilter {
                        input: join.left,
                        predicate,
                    };
                    let filtered = arena.alloc(LogicalOperator::Filter(new_filter));
                    let new_join = crate::sql::planner::LogicalJoin {
                        left: filtered,
                        right: join.right,
                        join_type: join.join_type,
                        condition: join.condition,
                    };
                    Ok(Some((arena.alloc(LogicalOperator::Join(new_join)), None)))
                } else if refs_right && !refs_left {
                    let new_filter = crate::sql::planner::LogicalFilter {
                        input: join.right,
                        predicate,
                    };
                    let filtered = arena.alloc(LogicalOperator::Filter(new_filter));
                    let new_join = crate::sql::planner::LogicalJoin {
                        left: join.left,
                        right: filtered,
                        join_type: join.join_type,
                        condition: join.condition,
                    };
                    Ok(Some((arena.alloc(LogicalOperator::Join(new_join)), None)))
                } else {
                    Ok(None)
                }
            }

            _ => Ok(None),
        }
    }

    fn collect_table_names<'a>(&self, op: &'a LogicalOperator<'a>) -> std::collections::HashSet<&'a str> {
        let mut tables = std::collections::HashSet::new();
        self.collect_tables_recursive(op, &mut tables);
        tables
    }

    fn collect_tables_recursive<'a>(&self, op: &'a LogicalOperator<'a>, tables: &mut std::collections::HashSet<&'a str>) {
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

    fn collect_predicate_tables<'a>(&self, expr: &'a crate::sql::ast::Expr<'a>) -> SmallVec<[&'a str; 4]> {
        let mut tables: SmallVec<[&'a str; 4]> = SmallVec::new();
        self.collect_expr_tables(expr, &mut tables);
        tables
    }

    fn collect_expr_tables<'a>(&self, expr: &'a crate::sql::ast::Expr<'a>, tables: &mut SmallVec<[&'a str; 4]>) {
        use crate::sql::ast::Expr;
        match expr {
            Expr::Column(col) => {
                if let Some(table) = col.table {
                    tables.push(table);
                }
            }
            Expr::BinaryOp { left, right, .. } => {
                self.collect_expr_tables(left, tables);
                self.collect_expr_tables(right, tables);
            }
            Expr::UnaryOp { expr, .. } => {
                self.collect_expr_tables(expr, tables);
            }
            Expr::Function(func) => {
                if let crate::sql::ast::FunctionArgs::Args(args) = &func.args {
                    for arg in args.iter() {
                        self.collect_expr_tables(arg.value, tables);
                    }
                }
            }
            _ => {}
        }
    }

    fn predicate_uses_alias(&self, predicate: &crate::sql::ast::Expr<'_>, aliases: &[Option<&str>]) -> bool {
        let alias_set: std::collections::HashSet<&str> = aliases
            .iter()
            .filter_map(|a| *a)
            .collect();

        if alias_set.is_empty() {
            return false;
        }

        self.expr_uses_alias(predicate, &alias_set)
    }

    fn expr_uses_alias(&self, expr: &crate::sql::ast::Expr<'_>, aliases: &std::collections::HashSet<&str>) -> bool {
        use crate::sql::ast::{Expr, FunctionArgs};

        match expr {
            Expr::Column(col) => {
                aliases.contains(col.column)
            }
            Expr::BinaryOp { left, right, .. } => {
                self.expr_uses_alias(left, aliases) || self.expr_uses_alias(right, aliases)
            }
            Expr::UnaryOp { expr, .. } => self.expr_uses_alias(expr, aliases),
            Expr::Function(func) => {
                match &func.args {
                    FunctionArgs::Args(args) => args.iter().any(|arg| self.expr_uses_alias(arg.value, aliases)),
                    _ => false,
                }
            }
            Expr::Between { expr, low, high, .. } => {
                self.expr_uses_alias(expr, aliases)
                    || self.expr_uses_alias(low, aliases)
                    || self.expr_uses_alias(high, aliases)
            }
            Expr::InList { expr, list, .. } => {
                self.expr_uses_alias(expr, aliases)
                    || list.iter().any(|e| self.expr_uses_alias(e, aliases))
            }
            Expr::Like { expr, pattern, .. } => {
                self.expr_uses_alias(expr, aliases) || self.expr_uses_alias(pattern, aliases)
            }
            Expr::IsNull { expr, .. } => self.expr_uses_alias(expr, aliases),
            Expr::Case { operand, conditions, else_result } => {
                if let Some(op) = operand {
                    if self.expr_uses_alias(op, aliases) {
                        return true;
                    }
                }
                for clause in conditions.iter() {
                    if self.expr_uses_alias(clause.condition, aliases) || self.expr_uses_alias(clause.result, aliases) {
                        return true;
                    }
                }
                if let Some(else_expr) = else_result {
                    if self.expr_uses_alias(else_expr, aliases) {
                        return true;
                    }
                }
                false
            }
            Expr::Cast { expr, .. } => self.expr_uses_alias(expr, aliases),
            Expr::Subquery(_) => false,
            Expr::Exists { .. } => false,
            _ => false,
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
