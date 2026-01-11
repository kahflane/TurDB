//! # Query Plan Traversal Helpers
//!
//! This module provides helper functions for traversing and analyzing physical query plans.
//! These functions are used during query execution to find specific operators in the plan
//! tree and determine query characteristics.
//!
//! ## Functions
//!
//! - Plan source finding: `find_plan_source`, `find_table_scan`, `find_nested_subquery`
//! - Plan analysis: `has_filter`, `has_aggregate`, `has_window`, `has_order_by_expression`
//! - Operator finding: `find_limit`, `find_sort_exec`, `find_projections`
//! - Optimization: `is_simple_count_star`
//! - Value comparison: `compare_owned_values`

use crate::schema::TableDef;
use crate::sql::ast::Expr;
use crate::sql::planner::{
    AggregateFunction, PhysicalGraceHashJoin, PhysicalHashAntiJoin, PhysicalHashSemiJoin,
    PhysicalIndexScan, PhysicalNestedLoopJoin, PhysicalOperator, PhysicalSecondaryIndexScan,
    PhysicalSetOpExec, PhysicalSortExec, PhysicalSubqueryExec, PhysicalTableScan,
};
use crate::types::OwnedValue;
use std::cmp::Ordering;

/// Represents the source of data in a physical query plan.
///
/// This enum is used to identify what type of data source is at the root
/// of a query plan, which determines how the query executor will retrieve data.
pub enum PlanSource<'a> {
    TableScan(&'a PhysicalTableScan<'a>),
    IndexScan(&'a PhysicalIndexScan<'a>),
    SecondaryIndexScan(&'a PhysicalSecondaryIndexScan<'a>),
    Subquery(&'a PhysicalSubqueryExec<'a>),
    NestedLoopJoin(&'a PhysicalNestedLoopJoin<'a>),
    GraceHashJoin(&'a PhysicalGraceHashJoin<'a>),
    HashSemiJoin(&'a PhysicalHashSemiJoin<'a>),
    HashAntiJoin(&'a PhysicalHashAntiJoin<'a>),
    SetOp(&'a PhysicalSetOpExec<'a>),
    DualScan,
}

/// Finds the data source at the bottom of a physical plan tree.
///
/// Recursively traverses through intermediate operators (filter, project, etc.)
/// to find the actual data source (table scan, index scan, join, etc.).
pub fn find_plan_source<'a>(op: &'a PhysicalOperator<'a>) -> Option<PlanSource<'a>> {
    match op {
        PhysicalOperator::TableScan(scan) => Some(PlanSource::TableScan(scan)),
        PhysicalOperator::DualScan => Some(PlanSource::DualScan),
        PhysicalOperator::IndexScan(scan) => Some(PlanSource::IndexScan(scan)),
        PhysicalOperator::SecondaryIndexScan(scan) => Some(PlanSource::SecondaryIndexScan(scan)),
        PhysicalOperator::SubqueryExec(subq) => Some(PlanSource::Subquery(subq)),
        PhysicalOperator::NestedLoopJoin(join) => Some(PlanSource::NestedLoopJoin(join)),
        PhysicalOperator::GraceHashJoin(join) => Some(PlanSource::GraceHashJoin(join)),
        PhysicalOperator::SetOpExec(set_op) => Some(PlanSource::SetOp(set_op)),
        PhysicalOperator::FilterExec(filter) => find_plan_source(filter.input),
        PhysicalOperator::ProjectExec(project) => find_plan_source(project.input),
        PhysicalOperator::LimitExec(limit) => find_plan_source(limit.input),
        PhysicalOperator::SortExec(sort) => find_plan_source(sort.input),
        PhysicalOperator::TopKExec(topk) => find_plan_source(topk.input),
        PhysicalOperator::HashAggregate(agg) => find_plan_source(agg.input),
        PhysicalOperator::SortedAggregate(agg) => find_plan_source(agg.input),
        PhysicalOperator::WindowExec(window) => find_plan_source(window.input),
        PhysicalOperator::HashSemiJoin(join) => Some(PlanSource::HashSemiJoin(join)),
        PhysicalOperator::HashAntiJoin(join) => Some(PlanSource::HashAntiJoin(join)),
        PhysicalOperator::ScalarSubqueryExec(subq) => find_plan_source(subq.subquery),
        PhysicalOperator::ExistsSubqueryExec(subq) => find_plan_source(subq.subquery),
        PhysicalOperator::InListSubqueryExec(subq) => find_plan_source(subq.subquery),
    }
}

/// Finds a TableScan operator in the plan tree.
pub fn find_table_scan<'a>(op: &'a PhysicalOperator<'a>) -> Option<&'a PhysicalTableScan<'a>> {
    match op {
        PhysicalOperator::TableScan(scan) => Some(scan),
        PhysicalOperator::FilterExec(filter) => find_table_scan(filter.input),
        PhysicalOperator::ProjectExec(project) => find_table_scan(project.input),
        PhysicalOperator::LimitExec(limit) => find_table_scan(limit.input),
        PhysicalOperator::SortExec(sort) => find_table_scan(sort.input),
        PhysicalOperator::TopKExec(topk) => find_table_scan(topk.input),
        PhysicalOperator::HashAggregate(agg) => find_table_scan(agg.input),
        PhysicalOperator::SortedAggregate(agg) => find_table_scan(agg.input),
        PhysicalOperator::SubqueryExec(subq) => find_table_scan(subq.child_plan),
        PhysicalOperator::WindowExec(window) => find_table_scan(window.input),
        _ => None,
    }
}

/// Finds a nested subquery operator in the plan tree.
pub fn find_nested_subquery<'a>(
    op: &'a PhysicalOperator<'a>,
) -> Option<&'a PhysicalSubqueryExec<'a>> {
    match op {
        PhysicalOperator::SubqueryExec(subq) => Some(subq),
        PhysicalOperator::FilterExec(filter) => find_nested_subquery(filter.input),
        PhysicalOperator::ProjectExec(project) => find_nested_subquery(project.input),
        PhysicalOperator::LimitExec(limit) => find_nested_subquery(limit.input),
        PhysicalOperator::SortExec(sort) => find_nested_subquery(sort.input),
        PhysicalOperator::TopKExec(topk) => find_nested_subquery(topk.input),
        PhysicalOperator::HashAggregate(agg) => find_nested_subquery(agg.input),
        PhysicalOperator::SortedAggregate(agg) => find_nested_subquery(agg.input),
        PhysicalOperator::WindowExec(window) => find_nested_subquery(window.input),
        _ => None,
    }
}

/// Checks if the plan contains a filter operator.
pub fn has_filter(op: &PhysicalOperator<'_>) -> bool {
    match op {
        PhysicalOperator::FilterExec(_) => true,
        PhysicalOperator::ProjectExec(project) => has_filter(project.input),
        PhysicalOperator::LimitExec(limit) => has_filter(limit.input),
        PhysicalOperator::SortExec(sort) => has_filter(sort.input),
        PhysicalOperator::TopKExec(topk) => has_filter(topk.input),
        PhysicalOperator::WindowExec(window) => has_filter(window.input),
        _ => false,
    }
}

/// Checks if the plan contains an aggregate operator.
pub fn has_aggregate(op: &PhysicalOperator<'_>) -> bool {
    match op {
        PhysicalOperator::HashAggregate(_) | PhysicalOperator::SortedAggregate(_) => true,
        PhysicalOperator::ProjectExec(project) => has_aggregate(project.input),
        PhysicalOperator::LimitExec(limit) => has_aggregate(limit.input),
        PhysicalOperator::SortExec(sort) => has_aggregate(sort.input),
        PhysicalOperator::TopKExec(topk) => has_aggregate(topk.input),
        PhysicalOperator::FilterExec(filter) => has_aggregate(filter.input),
        PhysicalOperator::WindowExec(window) => has_aggregate(window.input),
        _ => false,
    }
}

/// Checks if the plan contains a window function operator.
pub fn has_window(op: &PhysicalOperator<'_>) -> bool {
    match op {
        PhysicalOperator::WindowExec(_) => true,
        PhysicalOperator::ProjectExec(project) => has_window(project.input),
        PhysicalOperator::LimitExec(limit) => has_window(limit.input),
        PhysicalOperator::SortExec(sort) => has_window(sort.input),
        PhysicalOperator::TopKExec(topk) => has_window(topk.input),
        PhysicalOperator::FilterExec(filter) => has_window(filter.input),
        _ => false,
    }
}

/// Checks if the plan has an ORDER BY with non-column expressions.
pub fn has_order_by_expression(op: &PhysicalOperator<'_>) -> bool {
    match op {
        PhysicalOperator::SortExec(sort) => sort
            .order_by
            .iter()
            .any(|key| !matches!(key.expr, Expr::Column(_))),
        PhysicalOperator::TopKExec(topk) => topk
            .order_by
            .iter()
            .any(|key| !matches!(key.expr, Expr::Column(_))),
        PhysicalOperator::ProjectExec(project) => has_order_by_expression(project.input),
        PhysicalOperator::LimitExec(limit) => has_order_by_expression(limit.input),
        PhysicalOperator::FilterExec(filter) => has_order_by_expression(filter.input),
        PhysicalOperator::WindowExec(window) => has_order_by_expression(window.input),
        _ => false,
    }
}

/// Finds a LIMIT clause in the plan, returning (limit, offset).
pub fn find_limit(op: &PhysicalOperator<'_>) -> Option<(Option<u64>, Option<u64>)> {
    match op {
        PhysicalOperator::LimitExec(limit) => Some((limit.limit, limit.offset)),
        PhysicalOperator::TopKExec(topk) => Some((Some(topk.limit), topk.offset)),
        PhysicalOperator::ProjectExec(project) => find_limit(project.input),
        PhysicalOperator::FilterExec(filter) => find_limit(filter.input),
        PhysicalOperator::SortExec(sort) => find_limit(sort.input),
        _ => None,
    }
}

/// Finds a sort operator in the plan.
pub fn find_sort_exec<'a>(op: &'a PhysicalOperator<'a>) -> Option<&'a PhysicalSortExec<'a>> {
    match op {
        PhysicalOperator::SortExec(sort) => Some(sort),
        PhysicalOperator::ProjectExec(project) => find_sort_exec(project.input),
        PhysicalOperator::LimitExec(limit) => find_sort_exec(limit.input),
        PhysicalOperator::TopKExec(topk) => find_sort_exec(topk.input),
        PhysicalOperator::FilterExec(filter) => find_sort_exec(filter.input),
        PhysicalOperator::WindowExec(window) => find_sort_exec(window.input),
        _ => None,
    }
}

/// Finds column projections in the plan and returns their indices in the table.
pub fn find_projections(op: &PhysicalOperator<'_>, table_def: &TableDef) -> Option<Vec<usize>> {
    match op {
        PhysicalOperator::ProjectExec(project) => {
            let mut indices = Vec::new();
            for expr in project.expressions.iter() {
                if let Expr::Column(col_ref) = expr {
                    for (idx, col) in table_def.columns().iter().enumerate() {
                        if col.name().eq_ignore_ascii_case(col_ref.column) {
                            indices.push(idx);
                            break;
                        }
                    }
                }
            }
            if indices.is_empty() || indices.len() != project.expressions.len() {
                None
            } else {
                Some(indices)
            }
        }
        PhysicalOperator::FilterExec(filter) => find_projections(filter.input, table_def),
        PhysicalOperator::LimitExec(limit) => find_projections(limit.input, table_def),
        PhysicalOperator::SortExec(sort) => find_projections(sort.input, table_def),
        PhysicalOperator::TopKExec(topk) => find_projections(topk.input, table_def),
        _ => None,
    }
}

/// Checks if a plan is a simple COUNT(*) query without filters or grouping.
///
/// Returns the table scan if this is a simple count that can be optimized
/// by reading the row count directly from the table header.
pub fn is_simple_count_star<'a>(
    op: &'a PhysicalOperator<'a>,
) -> Option<&'a PhysicalTableScan<'a>> {
    use crate::sql::ast::FunctionArgs;

    match op {
        PhysicalOperator::HashAggregate(agg) => {
            if !agg.group_by.is_empty() {
                return None;
            }
            if agg.aggregates.len() != 1 {
                return None;
            }
            let agg_expr = &agg.aggregates[0];
            if agg_expr.function != AggregateFunction::Count || agg_expr.distinct {
                return None;
            }
            match agg.input {
                PhysicalOperator::TableScan(scan) => {
                    if scan.post_scan_filter.is_none() {
                        Some(scan)
                    } else {
                        None
                    }
                }
                _ => None,
            }
        }
        PhysicalOperator::ProjectExec(proj) => {
            fn is_simple_aggregate_projection(expressions: &[&Expr<'_>]) -> bool {
                if expressions.len() != 1 {
                    return false;
                }
                match expressions[0] {
                    Expr::Function(func) => {
                        let name = func.name.name.to_uppercase();
                        if !matches!(name.as_str(), "COUNT" | "SUM" | "AVG" | "MIN" | "MAX") {
                            return false;
                        }
                        matches!(func.args, FunctionArgs::Star | FunctionArgs::None)
                            || matches!(&func.args, FunctionArgs::Args(args) if args.len() <= 1)
                    }
                    _ => false,
                }
            }
            if is_simple_aggregate_projection(proj.expressions) {
                is_simple_count_star(proj.input)
            } else {
                None
            }
        }
        _ => None,
    }
}

/// Compares two OwnedValues for sorting purposes.
///
/// Handles NULL values (NULL < any other value) and type coercion
/// between integers and floats.
pub fn compare_owned_values(a: &OwnedValue, b: &OwnedValue) -> Ordering {
    match (a, b) {
        (OwnedValue::Null, OwnedValue::Null) => Ordering::Equal,
        (OwnedValue::Null, _) => Ordering::Less,
        (_, OwnedValue::Null) => Ordering::Greater,
        (OwnedValue::Int(a), OwnedValue::Int(b)) => a.cmp(b),
        (OwnedValue::Float(a), OwnedValue::Float(b)) => {
            a.partial_cmp(b).unwrap_or(Ordering::Equal)
        }
        (OwnedValue::Int(a), OwnedValue::Float(b)) => {
            (*a as f64).partial_cmp(b).unwrap_or(Ordering::Equal)
        }
        (OwnedValue::Float(a), OwnedValue::Int(b)) => {
            a.partial_cmp(&(*b as f64)).unwrap_or(Ordering::Equal)
        }
        (OwnedValue::Text(a), OwnedValue::Text(b)) => a.cmp(b),
        (OwnedValue::Bool(a), OwnedValue::Bool(b)) => a.cmp(b),
        (OwnedValue::Blob(a), OwnedValue::Blob(b)) => a.cmp(b),
        (OwnedValue::Date(a), OwnedValue::Date(b)) => a.cmp(b),
        (OwnedValue::Time(a), OwnedValue::Time(b)) => a.cmp(b),
        (OwnedValue::Timestamp(a), OwnedValue::Timestamp(b)) => a.cmp(b),
        _ => Ordering::Equal,
    }
}
