//! # Query Plan Traversal Helpers
//!
//! This module provides helper functions for traversing and analyzing physical query plans.
//! These functions are used during query execution to find specific operators in the plan
//! tree and determine query characteristics.
//!
//! ## Functions
//!
//! - Plan source finding: `find_plan_source`, `find_table_scan`, `find_nested_subquery`
//! - Plan analysis: `has_filter`, `has_aggregate`, `has_window`, `has_ordering`, `has_non_simple_root`
//! - Operator finding: `find_limit`, `find_sort_exec`, `find_projections`
//! - Optimization: `is_simple_count_star`
//! - Value comparison: `compare_owned_values`

use crate::memory::{PeriodicBudgetTracker, ROW_SIZE_ESTIMATE};
use crate::schema::{Catalog, TableDef};
use crate::sql::ast::Expr;
use crate::sql::executor::{RowSource, StreamingBTreeSource};
use crate::sql::planner::{
    AggregateFunction, PhysicalGraceHashJoin, PhysicalHashAntiJoin, PhysicalHashSemiJoin,
    PhysicalIndexNestedLoopJoin, PhysicalIndexScan, PhysicalNestedLoopJoin, PhysicalOperator,
    PhysicalSecondaryIndexScan, PhysicalSetOpExec, PhysicalSortExec, PhysicalStreamingHashJoin,
    PhysicalSubqueryExec, PhysicalTableScan,
};
use crate::storage::{FileManager, TableFileHeader, DEFAULT_SCHEMA};
use crate::types::{DataType, OwnedValue};
use eyre::Result;
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
    IndexNestedLoopJoin(&'a PhysicalIndexNestedLoopJoin<'a>),
    GraceHashJoin(&'a PhysicalGraceHashJoin<'a>),
    StreamingHashJoin(&'a PhysicalStreamingHashJoin<'a>),
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
        PhysicalOperator::IndexNestedLoopJoin(join) => Some(PlanSource::IndexNestedLoopJoin(join)),
        PhysicalOperator::GraceHashJoin(join) => Some(PlanSource::GraceHashJoin(join)),
        PhysicalOperator::StreamingHashJoin(join) => Some(PlanSource::StreamingHashJoin(join)),
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

/// Checks if the plan contains any ordering operation (SortExec or TopKExec).
pub fn has_ordering(op: &PhysicalOperator<'_>) -> bool {
    match op {
        PhysicalOperator::SortExec(_) => true,
        PhysicalOperator::TopKExec(_) => true,
        PhysicalOperator::ProjectExec(project) => has_ordering(project.input),
        PhysicalOperator::LimitExec(limit) => has_ordering(limit.input),
        PhysicalOperator::FilterExec(filter) => has_ordering(filter.input),
        PhysicalOperator::WindowExec(window) => has_ordering(window.input),
        PhysicalOperator::HashAggregate(agg) => has_ordering(agg.input),
        PhysicalOperator::SortedAggregate(agg) => has_ordering(agg.input),
        _ => false,
    }
}

/// Checks if the plan root is NOT a simple ProjectExec or TableScan.
/// When the root is LimitExec, SortExec, etc., source-level projection
/// optimization cannot be safely applied because executors expect full schema indices.
pub fn has_non_simple_root(op: &PhysicalOperator<'_>) -> bool {
    !matches!(
        op,
        PhysicalOperator::ProjectExec(_)
            | PhysicalOperator::TableScan(_)
            | PhysicalOperator::IndexScan(_)
    )
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

/// Materializes all rows from a table into owned values.
///
/// This helper extracts the common table scan pattern used across join execution:
/// 1. Resolves table definition from catalog
/// 2. Gets storage handle from file manager
/// 3. Reads root page from table header
/// 4. Scans all rows via StreamingBTreeSource
/// 5. Converts to owned values
///
/// Returns the materialized rows along with column types for further processing.
pub fn materialize_table_rows(
    catalog: &Catalog,
    file_manager: &mut FileManager,
    schema: Option<&str>,
    table_name: &str,
) -> Result<(Vec<Vec<OwnedValue>>, Vec<DataType>)> {
    let (rows, column_types, _) = materialize_table_rows_with_def(catalog, file_manager, schema, table_name)?;
    Ok((rows, column_types))
}

/// Materializes all rows from a table using a shared budget tracker.
///
/// The tracker handles both allocation and automatic release on drop.
pub fn materialize_table_rows_with_tracker(
    catalog: &Catalog,
    file_manager: &mut FileManager,
    schema: Option<&str>,
    table_name: &str,
    tracker: &mut PeriodicBudgetTracker<'_>,
) -> Result<(Vec<Vec<OwnedValue>>, Vec<DataType>)> {
    let schema_name = schema.unwrap_or(crate::storage::DEFAULT_SCHEMA);
    let table_def = catalog.resolve_table_in_schema(schema, table_name)?;
    let column_types: Vec<DataType> = table_def.columns().iter().map(|c| c.data_type()).collect();

    let storage_arc = file_manager.table_data(schema_name, table_name)?;
    let storage = storage_arc.read();
    let root_page = {
        let page = storage.page(0)?;
        TableFileHeader::from_bytes(page)?.root_page()
    };

    let mut source = StreamingBTreeSource::from_btree_scan_with_projections(
        &storage,
        root_page,
        column_types.clone(),
        None,
    )?;

    let mut rows = Vec::new();
    while let Some(row) = source.next_row()? {
        rows.push(row.iter().map(OwnedValue::from).collect());
        tracker.track(ROW_SIZE_ESTIMATE)?;
    }

    Ok((rows, column_types))
}

/// Materializes all rows from a table, also returning the table definition.
///
/// Use this variant when you need access to the TableDef for building column maps
/// or other metadata operations after materializing rows.
///
/// Note: This function does NOT track memory allocations. For memory-tracked
/// materialization, use `materialize_table_rows_with_tracker` instead, which
/// uses `PeriodicBudgetTracker` for automatic allocation and release.
#[allow(clippy::type_complexity)]
pub fn materialize_table_rows_with_def<'a>(
    catalog: &'a Catalog,
    file_manager: &mut FileManager,
    schema: Option<&str>,
    table_name: &str,
) -> Result<(Vec<Vec<OwnedValue>>, Vec<DataType>, &'a TableDef)> {
    let schema_name = schema.unwrap_or(DEFAULT_SCHEMA);
    let table_def = catalog.resolve_table_in_schema(schema, table_name)?;
    let column_types: Vec<DataType> = table_def.columns().iter().map(|c| c.data_type()).collect();

    let storage_arc = file_manager.table_data(schema_name, table_name)?;
    let storage = storage_arc.read();
    let root_page = {
        let page = storage.page(0)?;
        TableFileHeader::from_bytes(page)?.root_page()
    };

    let mut source = StreamingBTreeSource::from_btree_scan_with_projections(
        &storage,
        root_page,
        column_types.clone(),
        None,
    )?;

    let mut rows = Vec::new();
    while let Some(row) = source.next_row()? {
        rows.push(row.iter().map(OwnedValue::from).collect());
    }

    Ok((rows, column_types, table_def))
}

/// Builds a column map from a TableDef for predicate evaluation.
///
/// This helper creates a mapping from lowercase column names to their indices,
/// used throughout query execution for column lookups. Centralizing this logic
/// ensures consistent case-insensitive matching across all query operators.
pub fn build_simple_column_map(table_def: &TableDef) -> Vec<(String, usize)> {
    table_def
        .columns()
        .iter()
        .enumerate()
        .map(|(idx, col)| (col.name().to_lowercase(), idx))
        .collect()
}

/// Compares two OwnedValues for equality with type coercion.
///
/// This function handles the case where Int and Float types need to be
/// compared as equal when they represent the same numeric value.
/// This is essential for hash joins where one side may have BIGINT
/// and the other DOUBLE PRECISION for the join column.
pub fn owned_values_equal_with_coercion(a: &OwnedValue, b: &OwnedValue) -> bool {
    match (a, b) {
        (OwnedValue::Null, _) | (_, OwnedValue::Null) => false,
        (OwnedValue::Int(a), OwnedValue::Int(b)) => a == b,
        (OwnedValue::Float(a), OwnedValue::Float(b)) => a == b,
        (OwnedValue::Int(a), OwnedValue::Float(b)) => (*a as f64) == *b,
        (OwnedValue::Float(a), OwnedValue::Int(b)) => *a == (*b as f64),
        (OwnedValue::Text(a), OwnedValue::Text(b)) => a == b,
        (OwnedValue::Bool(a), OwnedValue::Bool(b)) => a == b,
        (OwnedValue::Blob(a), OwnedValue::Blob(b)) => a == b,
        (OwnedValue::Date(a), OwnedValue::Date(b)) => a == b,
        (OwnedValue::Time(a), OwnedValue::Time(b)) => a == b,
        (OwnedValue::Timestamp(a), OwnedValue::Timestamp(b)) => a == b,
        (OwnedValue::Uuid(a), OwnedValue::Uuid(b)) => a == b,
        _ => a == b,
    }
}

/// Hashes an OwnedValue with type normalization for hash joins.
///
/// This function normalizes Int to Float before hashing so that
/// Int(18) and Float(18.0) produce the same hash value. This is
/// essential for hash joins between columns of different numeric types.
pub fn hash_owned_value_normalized(val: &OwnedValue, hasher: &mut impl std::hash::Hasher) {
    use std::hash::Hash;
    match val {
        OwnedValue::Null => 0u8.hash(hasher),
        OwnedValue::Int(i) => (*i as f64).to_bits().hash(hasher),
        OwnedValue::Float(f) => f.to_bits().hash(hasher),
        OwnedValue::Text(s) => s.hash(hasher),
        OwnedValue::Bool(b) => b.hash(hasher),
        OwnedValue::Blob(b) => b.hash(hasher),
        OwnedValue::Date(d) => d.hash(hasher),
        OwnedValue::Time(t) => t.hash(hasher),
        OwnedValue::Timestamp(ts) => ts.hash(hasher),
        OwnedValue::Uuid(u) => u.hash(hasher),
        OwnedValue::Vector(v) => {
            for f in v {
                f.to_bits().hash(hasher);
            }
        }
        OwnedValue::TimestampTz(ts, tz) => {
            ts.hash(hasher);
            tz.hash(hasher);
        }
        OwnedValue::Interval(a, b, c) => {
            a.hash(hasher);
            b.hash(hasher);
            c.hash(hasher);
        }
        OwnedValue::Inet4(addr) => addr.hash(hasher),
        OwnedValue::Inet6(addr) => addr.hash(hasher),
        OwnedValue::MacAddr(m) => m.hash(hasher),
        OwnedValue::Jsonb(j) => j.hash(hasher),
        OwnedValue::Decimal(d, scale) => {
            d.hash(hasher);
            scale.hash(hasher);
        }
        OwnedValue::Point(x, y) => {
            x.to_bits().hash(hasher);
            y.to_bits().hash(hasher);
        }
        OwnedValue::Box(p1, p2) => {
            p1.0.to_bits().hash(hasher);
            p1.1.to_bits().hash(hasher);
            p2.0.to_bits().hash(hasher);
            p2.1.to_bits().hash(hasher);
        }
        OwnedValue::Circle(center, radius) => {
            center.0.to_bits().hash(hasher);
            center.1.to_bits().hash(hasher);
            radius.to_bits().hash(hasher);
        }
        OwnedValue::Enum(a, b) => {
            a.hash(hasher);
            b.hash(hasher);
        }
        OwnedValue::ToastPointer(p) => p.hash(hasher),
    }
}

/// Builds a column map with alias/table name qualified entries.
/// More efficient than allocating per-column - reuses a format buffer.
pub fn build_column_map_with_alias(
    table_def: &TableDef,
    alias: &str,
    table_name: Option<&str>,
    start_idx: usize,
    output: &mut Vec<(String, usize)>,
) {
    let mut buf = String::with_capacity(64);
    let alias_lower = alias.to_lowercase();
    let table_lower = table_name.map(|t| t.to_lowercase());

    for (i, col) in table_def.columns().iter().enumerate() {
        let idx = start_idx + i;
        let col_lower = col.name().to_lowercase();

        output.push((col_lower.clone(), idx));

        buf.clear();
        buf.push_str(&alias_lower);
        buf.push('.');
        buf.push_str(&col_lower);
        output.push((buf.clone(), idx));

        if let Some(ref tbl) = table_lower {
            if tbl != &alias_lower {
                buf.clear();
                buf.push_str(tbl);
                buf.push('.');
                buf.push_str(&col_lower);
                output.push((buf.clone(), idx));
            }
        }
    }
}
