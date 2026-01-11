//! # Join Analysis Module
//!
//! This module provides utilities for analyzing join conditions and extracting
//! information needed for join algorithm selection and optimization.
//!
//! ## Key Functions
//!
//! - **Equi-join key extraction**: Identifies column equality conditions
//! - **Non-equi condition extraction**: Separates other join conditions
//! - **Table extraction**: Collects tables involved in joins
//! - **Cardinality-based ordering**: Orders tables for optimal join order
//!
//! ## Join Algorithm Selection
//!
//! - Equi-joins with column equality → Grace Hash Join
//! - Non-equi joins → Nested Loop Join
//! - Semi-joins with equality → Hash Semi Join
//! - Anti-joins with equality → Hash Anti Join
//!
//! ## Usage
//!
//! ```ignore
//! let analyzer = JoinAnalyzer::new(arena);
//! let keys = analyzer.extract_equi_join_keys(condition);
//! if !keys.is_empty() {
//!     // Use hash join
//! }
//! ```

use crate::sql::ast::{BinaryOperator, Expr};
use crate::sql::planner::LogicalOperator;
use bumpalo::Bump;

pub struct JoinAnalyzer<'a> {
    arena: &'a Bump,
}

impl<'a> JoinAnalyzer<'a> {
    pub fn new(arena: &'a Bump) -> Self {
        Self { arena }
    }

    pub fn extract_equi_join_keys(&self, condition: Option<&'a Expr<'a>>) -> &'a [EquiJoinKey<'a>] {
        let condition = match condition {
            Some(c) => c,
            None => return &[],
        };

        let mut keys = bumpalo::collections::Vec::new_in(self.arena);
        self.collect_equi_join_keys(condition, &mut keys);
        keys.into_bump_slice()
    }

    fn collect_equi_join_keys(
        &self,
        expr: &'a Expr<'a>,
        keys: &mut bumpalo::collections::Vec<'a, EquiJoinKey<'a>>,
    ) {
        if let Expr::BinaryOp { left, op, right } = expr {
            match op {
                BinaryOperator::And => {
                    self.collect_equi_join_keys(left, keys);
                    self.collect_equi_join_keys(right, keys);
                }
                BinaryOperator::Eq => {
                    if let (Expr::Column(left_col), Expr::Column(right_col)) = (*left, *right) {
                        keys.push(EquiJoinKey {
                            left_expr: left,
                            right_expr: right,
                            left_column: left_col.column,
                            right_column: right_col.column,
                            left_table: left_col.table,
                            right_table: right_col.table,
                        });
                    }
                }
                _ => {}
            }
        }
    }

    pub fn has_equi_join_keys(&self, condition: Option<&'a Expr<'a>>) -> bool {
        !self.extract_equi_join_keys(condition).is_empty()
    }

    pub fn convert_equi_keys_to_join_keys(
        &self,
        equi_keys: &'a [EquiJoinKey<'a>],
    ) -> &'a [(&'a Expr<'a>, &'a Expr<'a>)] {
        let mut join_keys = bumpalo::collections::Vec::new_in(self.arena);
        for key in equi_keys {
            join_keys.push((key.left_expr, key.right_expr));
        }
        join_keys.into_bump_slice()
    }

    pub fn extract_non_equi_conditions(
        &self,
        condition: Option<&'a Expr<'a>>,
    ) -> Option<&'a Expr<'a>> {
        let condition = condition?;
        self.collect_non_equi_conditions(condition)
    }

    fn collect_non_equi_conditions(&self, expr: &'a Expr<'a>) -> Option<&'a Expr<'a>> {
        match expr {
            Expr::BinaryOp { left, op, right } => match op {
                BinaryOperator::And => {
                    let left_non_equi = self.collect_non_equi_conditions(left);
                    let right_non_equi = self.collect_non_equi_conditions(right);

                    match (left_non_equi, right_non_equi) {
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
                BinaryOperator::Eq => {
                    if matches!((*left, *right), (Expr::Column(_), Expr::Column(_))) {
                        None
                    } else {
                        Some(expr)
                    }
                }
                _ => Some(expr),
            },
            _ => Some(expr),
        }
    }

    pub fn extract_join_tables(&self, op: &'a LogicalOperator<'a>) -> &'a [&'a LogicalOperator<'a>] {
        let mut tables = bumpalo::collections::Vec::new_in(self.arena);
        self.collect_join_tables(op, &mut tables);
        tables.into_bump_slice()
    }

    fn collect_join_tables(
        &self,
        op: &'a LogicalOperator<'a>,
        tables: &mut bumpalo::collections::Vec<'a, &'a LogicalOperator<'a>>,
    ) {
        match op {
            LogicalOperator::Join(join) => {
                self.collect_join_tables(join.left, tables);
                self.collect_join_tables(join.right, tables);
            }
            LogicalOperator::Scan(_) => {
                tables.push(op);
            }
            LogicalOperator::Filter(filter) => {
                self.collect_join_tables(filter.input, tables);
            }
            _ => {
                tables.push(op);
            }
        }
    }

    pub fn order_tables_by_cardinality(
        &self,
        tables_with_card: &[(&'a LogicalOperator<'a>, u64)],
    ) -> &'a [&'a LogicalOperator<'a>] {
        let mut sorted: bumpalo::collections::Vec<(&'a LogicalOperator<'a>, u64)> =
            bumpalo::collections::Vec::from_iter_in(tables_with_card.iter().copied(), self.arena);
        sorted.sort_by_key(|(_, card)| *card);

        let mut result = bumpalo::collections::Vec::new_in(self.arena);
        for (op, _) in sorted {
            result.push(op);
        }
        result.into_bump_slice()
    }

    pub fn extract_filter_columns(&self, expr: &Expr<'a>) -> &'a [&'a str] {
        let mut columns = bumpalo::collections::Vec::new_in(self.arena);
        self.collect_columns_from_expr(expr, &mut columns);
        columns.into_bump_slice()
    }

    fn collect_columns_from_expr(
        &self,
        expr: &Expr<'a>,
        columns: &mut bumpalo::collections::Vec<'a, &'a str>,
    ) {
        match expr {
            Expr::Column(col_ref) => {
                columns.push(col_ref.column);
            }
            Expr::BinaryOp { left, op, right } => match op {
                BinaryOperator::And | BinaryOperator::Or => {
                    self.collect_columns_from_expr(left, columns);
                    self.collect_columns_from_expr(right, columns);
                }
                BinaryOperator::Eq
                | BinaryOperator::NotEq
                | BinaryOperator::Lt
                | BinaryOperator::LtEq
                | BinaryOperator::Gt
                | BinaryOperator::GtEq => {
                    self.collect_columns_from_expr(left, columns);
                    self.collect_columns_from_expr(right, columns);
                }
                _ => {}
            },
            Expr::IsNull { expr, .. } => {
                self.collect_columns_from_expr(expr, columns);
            }
            Expr::Between {
                expr, low, high, ..
            } => {
                self.collect_columns_from_expr(expr, columns);
                self.collect_columns_from_expr(low, columns);
                self.collect_columns_from_expr(high, columns);
            }
            Expr::InList { expr, list, .. } => {
                self.collect_columns_from_expr(expr, columns);
                for item in *list {
                    self.collect_columns_from_expr(item, columns);
                }
            }
            _ => {}
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EquiJoinKey<'a> {
    pub left_expr: &'a Expr<'a>,
    pub right_expr: &'a Expr<'a>,
    pub left_column: &'a str,
    pub right_column: &'a str,
    pub left_table: Option<&'a str>,
    pub right_table: Option<&'a str>,
}
