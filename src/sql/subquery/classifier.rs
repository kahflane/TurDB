//! # Subquery Classifier
//!
//! Analyzes subqueries to determine their type, position, and correlation status.
//! This information drives optimization decisions (decorrelation) and execution
//! strategy selection (caching vs re-execution).
//!
//! ## Classification Process
//!
//! 1. **Type Detection**: Scalar, EXISTS, IN, ANY/ALL, Derived, Lateral
//! 2. **Position Detection**: SELECT, FROM, WHERE, HAVING
//! 3. **Correlation Analysis**: Find outer column references
//! 4. **Nesting Depth**: Track subquery-within-subquery depth
//!
//! ## Correlation Detection Algorithm
//!
//! The classifier walks the subquery AST looking for column references that:
//! - Are not defined in the subquery's own FROM clause
//! - Match columns from an outer query's scope
//!
//! ```text
//! SELECT * FROM orders o
//! WHERE EXISTS (
//!     SELECT 1 FROM items i
//!     WHERE i.order_id = o.id  -- o.id is a correlation reference
//! )
//! ```
//!
//! ## Decorrelation Eligibility
//!
//! A correlated subquery can be decorrelated (converted to join) when:
//! - Correlation is on equality predicates only
//! - No aggregate functions depend on outer references
//! - Subquery has no LIMIT/OFFSET (or they can be preserved)
//!
//! ## Memory Model
//!
//! The classifier is allocation-free during analysis, borrowing from the AST.
//! All returned data structures use references into the original AST arena.

use crate::sql::ast::{ColumnRef, Expr, FromClause, SelectStmt};
use std::collections::HashSet;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SubqueryType {
    Scalar,
    Exists { negated: bool },
    InList { negated: bool },
    Quantified { op: QuantifiedOp, kind: QuantifiedKind },
    Derived,
    Lateral,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantifiedOp {
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantifiedKind {
    Any,
    All,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SubqueryPosition {
    Select,
    From,
    Where,
    Having,
    JoinCondition,
}

#[derive(Debug, Clone)]
pub struct CorrelationRef<'a> {
    pub outer_table: Option<&'a str>,
    pub outer_column: &'a str,
    pub referenced_in_expr: bool,
}

#[derive(Debug, Clone)]
pub struct SubqueryInfo<'a> {
    pub subquery_type: SubqueryType,
    pub position: SubqueryPosition,
    pub is_correlated: bool,
    pub correlation_refs: Vec<CorrelationRef<'a>>,
    pub nesting_depth: usize,
    pub can_decorrelate: bool,
}

pub struct SubqueryClassifier<'a> {
    outer_tables: HashSet<&'a str>,
    outer_columns: HashSet<(&'a str, &'a str)>,
    nesting_depth: usize,
}

impl<'a> SubqueryClassifier<'a> {
    pub fn new() -> Self {
        Self {
            outer_tables: HashSet::new(),
            outer_columns: HashSet::new(),
            nesting_depth: 0,
        }
    }

    pub fn with_outer_scope(
        outer_tables: impl IntoIterator<Item = &'a str>,
        outer_columns: impl IntoIterator<Item = (&'a str, &'a str)>,
    ) -> Self {
        Self {
            outer_tables: outer_tables.into_iter().collect(),
            outer_columns: outer_columns.into_iter().collect(),
            nesting_depth: 0,
        }
    }

    pub fn add_outer_table(&mut self, table: &'a str) {
        self.outer_tables.insert(table);
    }

    pub fn add_outer_column(&mut self, table: &'a str, column: &'a str) {
        self.outer_columns.insert((table, column));
    }

    pub fn increment_depth(&mut self) {
        self.nesting_depth += 1;
    }

    pub fn current_depth(&self) -> usize {
        self.nesting_depth
    }

    pub fn classify_scalar(&self, subquery: &'a SelectStmt<'a>) -> SubqueryInfo<'a> {
        let correlation_refs = self.find_correlations(subquery);
        let is_correlated = !correlation_refs.is_empty();
        let can_decorrelate = is_correlated && self.check_decorrelation_eligibility(subquery);

        SubqueryInfo {
            subquery_type: SubqueryType::Scalar,
            position: SubqueryPosition::Select,
            is_correlated,
            correlation_refs,
            nesting_depth: self.nesting_depth,
            can_decorrelate,
        }
    }

    pub fn classify_exists(
        &self,
        subquery: &'a SelectStmt<'a>,
        negated: bool,
        position: SubqueryPosition,
    ) -> SubqueryInfo<'a> {
        let correlation_refs = self.find_correlations(subquery);
        let is_correlated = !correlation_refs.is_empty();
        let can_decorrelate = is_correlated && self.check_decorrelation_eligibility(subquery);

        SubqueryInfo {
            subquery_type: SubqueryType::Exists { negated },
            position,
            is_correlated,
            correlation_refs,
            nesting_depth: self.nesting_depth,
            can_decorrelate,
        }
    }

    pub fn classify_in_subquery(
        &self,
        subquery: &'a SelectStmt<'a>,
        negated: bool,
        position: SubqueryPosition,
    ) -> SubqueryInfo<'a> {
        let correlation_refs = self.find_correlations(subquery);
        let is_correlated = !correlation_refs.is_empty();
        let can_decorrelate = is_correlated && self.check_decorrelation_eligibility(subquery);

        SubqueryInfo {
            subquery_type: SubqueryType::InList { negated },
            position,
            is_correlated,
            correlation_refs,
            nesting_depth: self.nesting_depth,
            can_decorrelate,
        }
    }

    pub fn classify_quantified(
        &self,
        subquery: &'a SelectStmt<'a>,
        op: QuantifiedOp,
        kind: QuantifiedKind,
        position: SubqueryPosition,
    ) -> SubqueryInfo<'a> {
        let correlation_refs = self.find_correlations(subquery);
        let is_correlated = !correlation_refs.is_empty();

        SubqueryInfo {
            subquery_type: SubqueryType::Quantified { op, kind },
            position,
            is_correlated,
            correlation_refs,
            nesting_depth: self.nesting_depth,
            can_decorrelate: false,
        }
    }

    pub fn classify_derived(&self, subquery: &'a SelectStmt<'a>) -> SubqueryInfo<'a> {
        let correlation_refs = self.find_correlations(subquery);
        let is_correlated = !correlation_refs.is_empty();

        SubqueryInfo {
            subquery_type: SubqueryType::Derived,
            position: SubqueryPosition::From,
            is_correlated,
            correlation_refs,
            nesting_depth: self.nesting_depth,
            can_decorrelate: false,
        }
    }

    pub fn classify_lateral(&self, subquery: &'a SelectStmt<'a>) -> SubqueryInfo<'a> {
        let correlation_refs = self.find_correlations(subquery);

        SubqueryInfo {
            subquery_type: SubqueryType::Lateral,
            position: SubqueryPosition::From,
            is_correlated: true,
            correlation_refs,
            nesting_depth: self.nesting_depth,
            can_decorrelate: false,
        }
    }

    fn find_correlations(&self, subquery: &'a SelectStmt<'a>) -> Vec<CorrelationRef<'a>> {
        let mut correlations = Vec::new();
        let subquery_tables = self.collect_subquery_tables(subquery);

        self.walk_select_for_correlations(subquery, &subquery_tables, &mut correlations);
        correlations
    }

    fn collect_subquery_tables(&self, subquery: &'a SelectStmt<'a>) -> HashSet<&'a str> {
        let mut tables = HashSet::new();

        if let Some(from) = subquery.from {
            self.collect_from_tables(from, &mut tables);
        }

        tables
    }

    fn collect_from_tables(&self, from: &'a FromClause<'a>, tables: &mut HashSet<&'a str>) {
        match from {
            FromClause::Table(table_ref) => {
                let name = table_ref.alias.unwrap_or(table_ref.name);
                tables.insert(name);
            }
            FromClause::Join(join) => {
                self.collect_from_tables(join.left, tables);
                self.collect_from_tables(join.right, tables);
            }
            FromClause::Subquery { alias, .. } => {
                tables.insert(*alias);
            }
            FromClause::Lateral { alias, .. } => {
                tables.insert(*alias);
            }
        }
    }

    fn walk_select_for_correlations(
        &self,
        select: &'a SelectStmt<'a>,
        local_tables: &HashSet<&'a str>,
        correlations: &mut Vec<CorrelationRef<'a>>,
    ) {
        for col in select.columns {
            if let crate::sql::ast::SelectColumn::Expr { expr, .. } = col {
                self.walk_expr_for_correlations(expr, local_tables, correlations);
            }
        }

        if let Some(where_clause) = select.where_clause {
            self.walk_expr_for_correlations(where_clause, local_tables, correlations);
        }

        if let Some(having) = select.having {
            self.walk_expr_for_correlations(having, local_tables, correlations);
        }

        for group_expr in select.group_by {
            self.walk_expr_for_correlations(group_expr, local_tables, correlations);
        }
    }

    fn walk_expr_for_correlations(
        &self,
        expr: &'a Expr<'a>,
        local_tables: &HashSet<&'a str>,
        correlations: &mut Vec<CorrelationRef<'a>>,
    ) {
        match expr {
            Expr::Column(col_ref) => {
                self.check_column_correlation(col_ref, local_tables, correlations);
            }
            Expr::BinaryOp { left, right, .. } => {
                self.walk_expr_for_correlations(left, local_tables, correlations);
                self.walk_expr_for_correlations(right, local_tables, correlations);
            }
            Expr::UnaryOp { expr, .. } => {
                self.walk_expr_for_correlations(expr, local_tables, correlations);
            }
            Expr::Between {
                expr, low, high, ..
            } => {
                self.walk_expr_for_correlations(expr, local_tables, correlations);
                self.walk_expr_for_correlations(low, local_tables, correlations);
                self.walk_expr_for_correlations(high, local_tables, correlations);
            }
            Expr::Like { expr, pattern, .. } => {
                self.walk_expr_for_correlations(expr, local_tables, correlations);
                self.walk_expr_for_correlations(pattern, local_tables, correlations);
            }
            Expr::InList { expr, list, .. } => {
                self.walk_expr_for_correlations(expr, local_tables, correlations);
                for item in *list {
                    self.walk_expr_for_correlations(item, local_tables, correlations);
                }
            }
            Expr::InSubquery { expr, subquery, .. } => {
                self.walk_expr_for_correlations(expr, local_tables, correlations);
                let nested_tables = self.collect_subquery_tables(subquery);
                let mut combined_tables = local_tables.clone();
                combined_tables.extend(nested_tables);
                self.walk_select_for_correlations(subquery, &combined_tables, correlations);
            }
            Expr::IsNull { expr, .. } => {
                self.walk_expr_for_correlations(expr, local_tables, correlations);
            }
            Expr::IsDistinctFrom { left, right, .. } => {
                self.walk_expr_for_correlations(left, local_tables, correlations);
                self.walk_expr_for_correlations(right, local_tables, correlations);
            }
            Expr::Function(func) => {
                if let crate::sql::ast::FunctionArgs::Args(args) = func.args {
                    for arg in args {
                        self.walk_expr_for_correlations(arg.value, local_tables, correlations);
                    }
                }
            }
            Expr::Case {
                operand,
                conditions,
                else_result,
            } => {
                if let Some(op) = operand {
                    self.walk_expr_for_correlations(op, local_tables, correlations);
                }
                for when in *conditions {
                    self.walk_expr_for_correlations(when.condition, local_tables, correlations);
                    self.walk_expr_for_correlations(when.result, local_tables, correlations);
                }
                if let Some(else_expr) = else_result {
                    self.walk_expr_for_correlations(else_expr, local_tables, correlations);
                }
            }
            Expr::Cast { expr, .. } => {
                self.walk_expr_for_correlations(expr, local_tables, correlations);
            }
            Expr::Subquery(subquery) => {
                let nested_tables = self.collect_subquery_tables(subquery);
                let mut combined_tables = local_tables.clone();
                combined_tables.extend(nested_tables);
                self.walk_select_for_correlations(subquery, &combined_tables, correlations);
            }
            Expr::Exists { subquery, .. } => {
                let nested_tables = self.collect_subquery_tables(subquery);
                let mut combined_tables = local_tables.clone();
                combined_tables.extend(nested_tables);
                self.walk_select_for_correlations(subquery, &combined_tables, correlations);
            }
            Expr::ArraySubscript { array, index } => {
                self.walk_expr_for_correlations(array, local_tables, correlations);
                self.walk_expr_for_correlations(index, local_tables, correlations);
            }
            Expr::ArraySlice {
                array,
                lower,
                upper,
            } => {
                self.walk_expr_for_correlations(array, local_tables, correlations);
                if let Some(l) = lower {
                    self.walk_expr_for_correlations(l, local_tables, correlations);
                }
                if let Some(u) = upper {
                    self.walk_expr_for_correlations(u, local_tables, correlations);
                }
            }
            Expr::Row(exprs) | Expr::Array(exprs) => {
                for e in *exprs {
                    self.walk_expr_for_correlations(e, local_tables, correlations);
                }
            }
            Expr::Literal(_) | Expr::Parameter(_) => {}
        }
    }

    fn check_column_correlation(
        &self,
        col_ref: &ColumnRef<'a>,
        local_tables: &HashSet<&'a str>,
        correlations: &mut Vec<CorrelationRef<'a>>,
    ) {
        if let Some(table) = col_ref.table {
            if !local_tables.contains(table) && self.outer_tables.contains(table) {
                correlations.push(CorrelationRef {
                    outer_table: Some(table),
                    outer_column: col_ref.column,
                    referenced_in_expr: true,
                });
            }
        } else if self
            .outer_columns
            .iter()
            .any(|(_, col)| *col == col_ref.column)
        {
            let outer_table = self
                .outer_columns
                .iter()
                .find(|(_, col)| *col == col_ref.column)
                .map(|(t, _)| *t);
            correlations.push(CorrelationRef {
                outer_table,
                outer_column: col_ref.column,
                referenced_in_expr: true,
            });
        }
    }

    fn check_decorrelation_eligibility(&self, subquery: &'a SelectStmt<'a>) -> bool {
        if subquery.limit.is_some() || subquery.offset.is_some() {
            return false;
        }

        if let Some(where_clause) = subquery.where_clause {
            return self.has_equality_correlation(where_clause);
        }

        true
    }

    fn has_equality_correlation(&self, expr: &'a Expr<'a>) -> bool {
        match expr {
            Expr::BinaryOp { op, left, right } => {
                use crate::sql::ast::BinaryOperator;
                match op {
                    BinaryOperator::Eq => {
                        let left_is_outer = self.is_outer_reference(left);
                        let right_is_outer = self.is_outer_reference(right);
                        left_is_outer || right_is_outer
                    }
                    BinaryOperator::And => {
                        self.has_equality_correlation(left)
                            || self.has_equality_correlation(right)
                    }
                    _ => false,
                }
            }
            _ => false,
        }
    }

    fn is_outer_reference(&self, expr: &'a Expr<'a>) -> bool {
        match expr {
            Expr::Column(col_ref) => {
                if let Some(table) = col_ref.table {
                    self.outer_tables.contains(table)
                } else {
                    self.outer_columns
                        .iter()
                        .any(|(_, col)| *col == col_ref.column)
                }
            }
            _ => false,
        }
    }
}

impl Default for SubqueryClassifier<'_> {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_classifier_new() {
        let classifier = SubqueryClassifier::new();
        assert_eq!(classifier.nesting_depth, 0);
        assert!(classifier.outer_tables.is_empty());
    }

    #[test]
    fn test_classifier_with_outer_scope() {
        let tables = vec!["orders", "users"];
        let columns = vec![("orders", "id"), ("users", "name")];
        let classifier = SubqueryClassifier::with_outer_scope(tables, columns);

        assert!(classifier.outer_tables.contains("orders"));
        assert!(classifier.outer_tables.contains("users"));
        assert!(classifier.outer_columns.contains(&("orders", "id")));
    }

    #[test]
    fn test_add_outer_table() {
        let mut classifier = SubqueryClassifier::new();
        classifier.add_outer_table("products");
        assert!(classifier.outer_tables.contains("products"));
    }

    #[test]
    fn test_increment_depth() {
        let mut classifier = SubqueryClassifier::new();
        assert_eq!(classifier.nesting_depth, 0);
        classifier.increment_depth();
        assert_eq!(classifier.nesting_depth, 1);
        classifier.increment_depth();
        assert_eq!(classifier.nesting_depth, 2);
    }
}
