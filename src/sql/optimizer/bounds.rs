//! # Column Scan Bounds Extraction
//!
//! This module provides utilities for extracting scan bounds from predicates.
//! When a query contains conditions like `WHERE id > 10 AND id < 100`, this
//! module extracts the bounds that can be used for efficient index scans.
//!
//! ## Bound Types
//!
//! - `ColumnBound`: A single bound value with inclusivity flag
//! - `ColumnScanBounds`: Collection of lower, upper, and point bounds
//! - `ScanBoundType`: Classification of scan type (Point, Range, Full)
//!
//! ## Extraction Process
//!
//! 1. Traverse the predicate expression tree
//! 2. Identify comparison operators (=, <, <=, >, >=)
//! 3. Match operators to specific columns
//! 4. Combine into usable scan bounds
//!
//! ## Usage
//!
//! ```ignore
//! let bounds = extract_scan_bounds_for_column(predicate, "id", arena);
//! let scan_type = bounds_to_scan_type(&bounds);
//! ```

use crate::sql::ast::{BinaryOperator, Expr};
use bumpalo::Bump;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ColumnBound<'a> {
    pub value: &'a Expr<'a>,
    pub inclusive: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct ColumnScanBounds<'a> {
    pub lower: Option<ColumnBound<'a>>,
    pub upper: Option<ColumnBound<'a>>,
    pub point_value: Option<ColumnBound<'a>>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScanBoundType {
    Point,
    Range,
    Full,
}

pub fn extract_scan_bounds_for_column<'a>(
    predicate: &'a Expr<'a>,
    target_column: &str,
    _arena: &'a Bump,
) -> ColumnScanBounds<'a> {
    let mut bounds = ColumnScanBounds {
        lower: None,
        upper: None,
        point_value: None,
    };

    collect_bounds_from_expr(predicate, target_column, &mut bounds);
    bounds
}

pub fn collect_bounds_from_expr<'a>(
    expr: &'a Expr<'a>,
    target_column: &str,
    bounds: &mut ColumnScanBounds<'a>,
) {
    match expr {
        Expr::BinaryOp { left, op, right } => match op {
            BinaryOperator::And => {
                collect_bounds_from_expr(left, target_column, bounds);
                collect_bounds_from_expr(right, target_column, bounds);
            }
            BinaryOperator::Eq => {
                if expr_references_column(left, target_column) {
                    bounds.point_value = Some(ColumnBound {
                        value: right,
                        inclusive: true,
                    });
                } else if expr_references_column(right, target_column) {
                    bounds.point_value = Some(ColumnBound {
                        value: left,
                        inclusive: true,
                    });
                }
            }
            BinaryOperator::Lt => {
                if expr_references_column(left, target_column) {
                    bounds.upper = Some(ColumnBound {
                        value: right,
                        inclusive: false,
                    });
                } else if expr_references_column(right, target_column) {
                    bounds.lower = Some(ColumnBound {
                        value: left,
                        inclusive: false,
                    });
                }
            }
            BinaryOperator::LtEq => {
                if expr_references_column(left, target_column) {
                    bounds.upper = Some(ColumnBound {
                        value: right,
                        inclusive: true,
                    });
                } else if expr_references_column(right, target_column) {
                    bounds.lower = Some(ColumnBound {
                        value: left,
                        inclusive: true,
                    });
                }
            }
            BinaryOperator::Gt => {
                if expr_references_column(left, target_column) {
                    bounds.lower = Some(ColumnBound {
                        value: right,
                        inclusive: false,
                    });
                } else if expr_references_column(right, target_column) {
                    bounds.upper = Some(ColumnBound {
                        value: left,
                        inclusive: false,
                    });
                }
            }
            BinaryOperator::GtEq => {
                if expr_references_column(left, target_column) {
                    bounds.lower = Some(ColumnBound {
                        value: right,
                        inclusive: true,
                    });
                } else if expr_references_column(right, target_column) {
                    bounds.upper = Some(ColumnBound {
                        value: left,
                        inclusive: true,
                    });
                }
            }
            _ => {}
        },
        Expr::Between {
            expr: between_expr,
            negated,
            low,
            high,
        } => {
            if !negated && expr_references_column(between_expr, target_column) {
                bounds.lower = Some(ColumnBound {
                    value: low,
                    inclusive: true,
                });
                bounds.upper = Some(ColumnBound {
                    value: high,
                    inclusive: true,
                });
            }
        }
        _ => {}
    }
}

pub fn expr_references_column(expr: &Expr<'_>, target_column: &str) -> bool {
    match expr {
        Expr::Column(col_ref) => col_ref.column.eq_ignore_ascii_case(target_column),
        _ => false,
    }
}

pub fn bounds_to_scan_type(bounds: &ColumnScanBounds<'_>) -> ScanBoundType {
    if bounds.point_value.is_some() {
        ScanBoundType::Point
    } else if bounds.lower.is_some() || bounds.upper.is_some() {
        ScanBoundType::Range
    } else {
        ScanBoundType::Full
    }
}
