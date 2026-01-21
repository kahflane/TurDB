//! # UPDATE Operation Module
//!
//! This module implements UPDATE operations for TurDB, handling both simple
//! UPDATE statements and UPDATE...FROM syntax for joins.
//!
//! ## Purpose
//!
//! UPDATE operations modify existing rows while:
//! - Validating constraints on new values
//! - Processing TOAST for large updated values
//! - Supporting UPDATE...FROM for join-based updates
//! - Recording undo data for transaction rollback
//!
//! ## Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────────────┐
//! │                        UPDATE Operation Flow                            │
//! ├─────────────────────────────────────────────────────────────────────────┤
//! │                                                                         │
//! │   UPDATE table SET col=val WHERE condition                              │
//! │       │                                                                 │
//! │       ▼                                                                 │
//! │   ┌─────────────────────────────────────────────────────────────────┐   │
//! │   │ 1. Scan table for matching rows                                 │   │
//! │   │    - Apply WHERE predicate                                      │   │
//! │   │    - Collect rows to update                                     │   │
//! │   └─────────────────────────────────────────────────────────────────┘   │
//! │       │                                                                 │
//! │       ▼                                                                 │
//! │   ┌─────────────────────────────────────────────────────────────────┐   │
//! │   │ 2. For each matching row:                                       │   │
//! │   │    a. Store old value for undo                                  │   │
//! │   │    b. Apply SET assignments                                     │   │
//! │   │    c. Validate constraints on new values                        │   │
//! │   │    d. Check for UNIQUE violations                               │   │
//! │   └─────────────────────────────────────────────────────────────────┘   │
//! │       │                                                                 │
//! │       ▼                                                                 │
//! │   ┌─────────────────────────────────────────────────────────────────┐   │
//! │   │ 3. Handle TOAST                                                 │   │
//! │   │    - Delete old TOAST chunks if column was toasted              │   │
//! │   │    - Toast new value if exceeds threshold                       │   │
//! │   └─────────────────────────────────────────────────────────────────┘   │
//! │       │                                                                 │
//! │       ▼                                                                 │
//! │   ┌─────────────────────────────────────────────────────────────────┐   │
//! │   │ 4. Apply updates to BTree                                       │   │
//! │   │    - Delete old record                                          │   │
//! │   │    - Insert updated record                                      │   │
//! │   │    - WAL tracking if enabled                                    │   │
//! │   └─────────────────────────────────────────────────────────────────┘   │
//! │       │                                                                 │
//! │       ▼                                                                 │
//! │   ┌─────────────────────────────────────────────────────────────────┐   │
//! │   │ 5. Record transaction write entries with undo data              │   │
//! │   └─────────────────────────────────────────────────────────────────┘   │
//! │                                                                         │
//! └─────────────────────────────────────────────────────────────────────────┘
//! ```
//!
//! ## UPDATE...FROM Syntax
//!
//! Supports PostgreSQL-style UPDATE with FROM clause:
//! ```sql
//! UPDATE target SET col = source.val
//! FROM source
//! WHERE target.id = source.id
//! ```
//!
//! This performs a join between target and source tables, updating target
//! rows based on matching source rows.
//!
//! ## Performance Characteristics
//!
//! - Simple UPDATE: O(n) scan + O(m * log n) for m matching rows
//! - UPDATE...FROM: O(n * k) for cartesian product, filtered by predicate
//! - UNIQUE check: O(n) scan of existing rows (could be optimized)
//! - TOAST cleanup: O(chunks) per toasted column
//!
//! ## Thread Safety
//!
//! UPDATE acquires write lock on file_manager. Transaction write entries
//! include undo data (old row value) for rollback support.
//!
//! ## Scalar Subquery Support
//!
//! UPDATE SET clauses support scalar subqueries in value expressions:
//! ```sql
//! UPDATE orders SET total = (SELECT SUM(amount) FROM order_items WHERE order_id = 1)
//! ```
//!
//! ### Strategy
//!
//! 1. **Collection**: All scalar subqueries are collected from SET expressions before
//!    the UPDATE loop begins
//! 2. **Pre-computation**: Subquery results are computed once and stored in a HashMap
//!    keyed by AST node address (stable within the function scope)
//! 3. **Substitution**: During expression evaluation, subquery nodes are replaced with
//!    their pre-computed results
//!
//! ### Streaming Aggregate Optimization
//!
//! For simple aggregate subqueries (SUM, AVG, MIN, MAX, COUNT over a single expression),
//! we bypass the full executor and stream directly through BTree cursors:
//! - Creates `StreamingAggregateState` to accumulate results
//! - Iterates through table/index cursors evaluating expressions per-row
//! - Avoids materializing all rows into memory
//!
//! This optimization applies when the subquery plan has:
//! - A single aggregate function
//! - An expression to aggregate (e.g., `SUM(quantity * price)`)
//! - No complex operators like joins or sorts
//!
//! ### Limitations
//!
//! - Subqueries are not correlated (cannot reference outer UPDATE row)
//! - Streaming optimization only applies when no WHERE filters are present
//!   (filtered queries fall back to the full executor)
//! - Multiple rows from non-aggregate subqueries return only the first row

use crate::btree::BTree;
use crate::database::dml::mvcc_helpers::{get_user_data, wrap_record_for_update};
use crate::database::macros::with_btree_storage;
use crate::database::row::Row;
use crate::database::{Database, ExecuteResult};
use crate::mvcc::WriteEntry;
use crate::records::RecordView;
use crate::schema::table::{Constraint, IndexType};
use crate::sql::context::ScalarSubqueryResults;
use crate::sql::decoder::RecordDecoder;
use crate::sql::executor::ExecutorRow;
use crate::sql::predicate::CompiledPredicate;
use crate::storage::{IndexFileHeader, TableFileHeader, DEFAULT_SCHEMA};
use crate::types::{create_record_schema, ArithmeticOp, OwnedValue, Value};
use bumpalo::Bump;
use eyre::{bail, Result, WrapErr};
use hashbrown::{HashMap, HashSet};
use smallvec::SmallVec;
use std::borrow::Cow;
use std::sync::atomic::Ordering;

fn count_params_in_expr(expr: &crate::sql::ast::Expr) -> usize {
    use crate::sql::ast::Expr;
    match expr {
        Expr::Parameter(_) => 1,
        Expr::BinaryOp { left, right, .. } => {
            count_params_in_expr(left) + count_params_in_expr(right)
        }
        Expr::UnaryOp { expr, .. } => count_params_in_expr(expr),
        Expr::Cast { expr, .. } => count_params_in_expr(expr),
        _ => 0,
    }
}

fn expr_contains_column_ref(expr: &crate::sql::ast::Expr) -> bool {
    use crate::sql::ast::Expr;
    match expr {
        Expr::Column(_) => true,
        Expr::Literal(_) | Expr::Parameter(_) => false,
        Expr::BinaryOp { left, right, .. } => {
            expr_contains_column_ref(left) || expr_contains_column_ref(right)
        }
        Expr::UnaryOp { expr, .. } => expr_contains_column_ref(expr),
        Expr::Cast { expr, .. } => expr_contains_column_ref(expr),
        Expr::IsNull { expr, .. } => expr_contains_column_ref(expr),
        Expr::Between {
            expr, low, high, ..
        } => {
            expr_contains_column_ref(expr)
                || expr_contains_column_ref(low)
                || expr_contains_column_ref(high)
        }
        Expr::Like {
            expr,
            pattern,
            escape,
            ..
        } => {
            expr_contains_column_ref(expr)
                || expr_contains_column_ref(pattern)
                || escape.is_some_and(expr_contains_column_ref)
        }
        Expr::InList { expr, list, .. } => {
            expr_contains_column_ref(expr) || list.iter().any(|e| expr_contains_column_ref(e))
        }
        Expr::InSubquery { .. } | Expr::Exists { .. } => true,
        Expr::Subquery(_) => false,
        Expr::IsDistinctFrom { left, right, .. } => {
            expr_contains_column_ref(left) || expr_contains_column_ref(right)
        }
        Expr::Case {
            operand,
            conditions,
            else_result,
        } => {
            operand.is_some_and(expr_contains_column_ref)
                || conditions.iter().any(|wc| {
                    expr_contains_column_ref(wc.condition) || expr_contains_column_ref(wc.result)
                })
                || else_result.is_some_and(expr_contains_column_ref)
        }
        Expr::Function(func) => {
            if let crate::sql::ast::FunctionArgs::Args(args) = &func.args {
                args.iter().any(|arg| expr_contains_column_ref(arg.value))
            } else {
                false
            }
        }
        _ => false,
    }
}

/// Recursively collects scalar subqueries from an expression tree.
fn collect_scalar_subqueries_from_expr<'a>(
    expr: &'a crate::sql::ast::Expr<'a>,
    subqueries: &mut SmallVec<[&'a crate::sql::ast::SelectStmt<'a>; 4]>,
) {
    use crate::sql::ast::Expr;
    match expr {
        Expr::Subquery(subq) => {
            subqueries.push(subq);
        }
        Expr::BinaryOp { left, right, .. } => {
            collect_scalar_subqueries_from_expr(left, subqueries);
            collect_scalar_subqueries_from_expr(right, subqueries);
        }
        Expr::UnaryOp { expr, .. } => {
            collect_scalar_subqueries_from_expr(expr, subqueries);
        }
        Expr::IsNull { expr, .. } => {
            collect_scalar_subqueries_from_expr(expr, subqueries);
        }
        Expr::InList { expr, list, .. } => {
            collect_scalar_subqueries_from_expr(expr, subqueries);
            for item in list.iter() {
                collect_scalar_subqueries_from_expr(item, subqueries);
            }
        }
        Expr::Between {
            expr, low, high, ..
        } => {
            collect_scalar_subqueries_from_expr(expr, subqueries);
            collect_scalar_subqueries_from_expr(low, subqueries);
            collect_scalar_subqueries_from_expr(high, subqueries);
        }
        Expr::Case {
            operand,
            conditions,
            else_result,
        } => {
            if let Some(op) = operand {
                collect_scalar_subqueries_from_expr(op, subqueries);
            }
            for cond in conditions.iter() {
                collect_scalar_subqueries_from_expr(cond.condition, subqueries);
                collect_scalar_subqueries_from_expr(cond.result, subqueries);
            }
            if let Some(else_e) = else_result {
                collect_scalar_subqueries_from_expr(else_e, subqueries);
            }
        }
        Expr::Cast { expr, .. } => {
            collect_scalar_subqueries_from_expr(expr, subqueries);
        }
        Expr::Function(func) => {
            if let crate::sql::ast::FunctionArgs::Args(args) = &func.args {
                for arg in args.iter() {
                    collect_scalar_subqueries_from_expr(arg.value, subqueries);
                }
            }
        }
        _ => {}
    }
}

/// Finds a single aggregate function with expression in a query plan.
fn find_expression_aggregate<'a>(
    op: &'a crate::sql::planner::PhysicalOperator<'a>,
) -> Option<(
    &'a crate::sql::planner::AggregateFunction,
    &'a crate::sql::ast::Expr<'a>,
)> {
    use crate::sql::planner::PhysicalOperator;

    match op {
        PhysicalOperator::HashAggregate(agg) => {
            for agg_expr in agg.aggregates.iter() {
                if let Some(arg) = agg_expr.argument {
                    if !matches!(arg, crate::sql::ast::Expr::Column(_)) {
                        return Some((&agg_expr.function, arg));
                    }
                }
            }
            find_expression_aggregate(agg.input)
        }
        PhysicalOperator::SortedAggregate(agg) => {
            for agg_expr in agg.aggregates.iter() {
                if let Some(arg) = agg_expr.argument {
                    if !matches!(arg, crate::sql::ast::Expr::Column(_)) {
                        return Some((&agg_expr.function, arg));
                    }
                }
            }
            find_expression_aggregate(agg.input)
        }
        PhysicalOperator::FilterExec(f) => find_expression_aggregate(f.input),
        PhysicalOperator::ProjectExec(p) => find_expression_aggregate(p.input),
        PhysicalOperator::SortExec(s) => find_expression_aggregate(s.input),
        PhysicalOperator::LimitExec(l) => find_expression_aggregate(l.input),
        _ => None,
    }
}

/// Finds ANY single aggregate function in a query plan (including simple column aggregates).
fn find_any_aggregate<'a>(
    op: &'a crate::sql::planner::PhysicalOperator<'a>,
) -> Option<(
    &'a crate::sql::planner::AggregateFunction,
    &'a crate::sql::ast::Expr<'a>,
)> {
    use crate::sql::planner::PhysicalOperator;

    match op {
        PhysicalOperator::HashAggregate(agg) => {
            for agg_expr in agg.aggregates.iter() {
                if let Some(arg) = agg_expr.argument {
                    return Some((&agg_expr.function, arg));
                }
            }
            find_any_aggregate(agg.input)
        }
        PhysicalOperator::SortedAggregate(agg) => {
            for agg_expr in agg.aggregates.iter() {
                if let Some(arg) = agg_expr.argument {
                    return Some((&agg_expr.function, arg));
                }
            }
            find_any_aggregate(agg.input)
        }
        PhysicalOperator::FilterExec(f) => find_any_aggregate(f.input),
        PhysicalOperator::ProjectExec(p) => find_any_aggregate(p.input),
        PhysicalOperator::SortExec(s) => find_any_aggregate(s.input),
        PhysicalOperator::LimitExec(l) => find_any_aggregate(l.input),
        _ => None,
    }
}

/// Evaluates a binary operation on two OwnedValues.
fn eval_binary_op(left: &OwnedValue, op: &crate::sql::ast::BinaryOperator, right: &OwnedValue) -> OwnedValue {
    use crate::sql::ast::BinaryOperator;
    let arith_op = match op {
        BinaryOperator::Plus => Some(ArithmeticOp::Plus),
        BinaryOperator::Minus => Some(ArithmeticOp::Minus),
        BinaryOperator::Multiply => Some(ArithmeticOp::Multiply),
        BinaryOperator::Divide => Some(ArithmeticOp::Divide),
        _ => None,
    };
    arith_op
        .and_then(|aop| OwnedValue::eval_arithmetic(left, aop, right))
        .unwrap_or(OwnedValue::Null)
}

/// Evaluates an expression against a record for streaming aggregation.
fn eval_expr_for_record_streaming(
    expr: &crate::sql::ast::Expr<'_>,
    record: &RecordView<'_>,
    column_info: &HashMap<String, (usize, crate::records::types::DataType)>,
) -> OwnedValue {
    use crate::sql::ast::{Expr, Literal};

    match expr {
        Expr::Column(col_ref) => {
            let col_lower = col_ref.column.to_lowercase();
            if let Some(&(idx, data_type)) = column_info.get(&col_lower) {
                OwnedValue::from_record_column(record, idx, data_type).unwrap_or(OwnedValue::Null)
            } else {
                OwnedValue::Null
            }
        }
        Expr::Literal(lit) => match lit {
            Literal::Integer(s) => s.parse::<i64>().map(OwnedValue::Int).unwrap_or(OwnedValue::Null),
            Literal::Float(s) => s.parse::<f64>().map(OwnedValue::Float).unwrap_or(OwnedValue::Null),
            Literal::String(s) => OwnedValue::Text((*s).to_string()),
            Literal::Boolean(b) => OwnedValue::Int(if *b { 1 } else { 0 }),
            Literal::Null => OwnedValue::Null,
            _ => OwnedValue::Null,
        },
        Expr::BinaryOp { left, op, right } => {
            let left_val = eval_expr_for_record_streaming(left, record, column_info);
            let right_val = eval_expr_for_record_streaming(right, record, column_info);
            eval_binary_op(&left_val, op, &right_val)
        }
        Expr::UnaryOp { op, expr: inner } => {
            let val = eval_expr_for_record_streaming(inner, record, column_info);
            match op {
                crate::sql::ast::UnaryOperator::Minus => match val {
                    OwnedValue::Int(i) => OwnedValue::Int(-i),
                    OwnedValue::Float(f) => OwnedValue::Float(-f),
                    _ => OwnedValue::Null,
                },
                crate::sql::ast::UnaryOperator::Plus => val,
                _ => OwnedValue::Null,
            }
        }
        _ => OwnedValue::Null,
    }
}

/// Compares i64 and f64 without precision loss for large integers.
fn compare_int_float(i: i64, f: f64) -> std::cmp::Ordering {
    use std::cmp::Ordering;

    if f.is_nan() {
        return Ordering::Less;
    }
    if f == f64::INFINITY {
        return Ordering::Less;
    }
    if f == f64::NEG_INFINITY {
        return Ordering::Greater;
    }

    const MAX_EXACT: i64 = 1i64 << 53;
    const MIN_EXACT: i64 = -(1i64 << 53);

    if (MIN_EXACT..=MAX_EXACT).contains(&i) {
        let i_as_f64 = i as f64;
        i_as_f64.partial_cmp(&f).unwrap_or(Ordering::Equal)
    } else if f >= (i64::MAX as f64) {
        Ordering::Less
    } else if f <= (i64::MIN as f64) {
        Ordering::Greater
    } else {
        let f_truncated = f as i64;
        match i.cmp(&f_truncated) {
            Ordering::Equal => {
                let f_frac = f - (f_truncated as f64);
                if f_frac > 0.0 {
                    Ordering::Less
                } else if f_frac < 0.0 {
                    Ordering::Greater
                } else {
                    Ordering::Equal
                }
            }
            other => other,
        }
    }
}

struct StreamingAggregateState {
    sum_int: i64,
    sum_float: f64,
    has_float: bool,
    count: usize,
    min_int: Option<i64>,
    min_float: Option<f64>,
    max_int: Option<i64>,
    max_float: Option<f64>,
}

impl StreamingAggregateState {
    fn new() -> Self {
        Self {
            sum_int: 0,
            sum_float: 0.0,
            has_float: false,
            count: 0,
            min_int: None,
            min_float: None,
            max_int: None,
            max_float: None,
        }
    }

    fn update(&mut self, value: &OwnedValue) {
        match value {
            OwnedValue::Int(i) => {
                self.sum_int = self.sum_int.saturating_add(*i);
                self.count = self.count.saturating_add(1);
                self.min_int = Some(self.min_int.map_or(*i, |m| m.min(*i)));
                self.max_int = Some(self.max_int.map_or(*i, |m| m.max(*i)));
            }
            OwnedValue::Float(f) => {
                self.sum_float += f;
                self.has_float = true;
                self.count = self.count.saturating_add(1);
                self.min_float = Some(self.min_float.map_or(*f, |m| m.min(*f)));
                self.max_float = Some(self.max_float.map_or(*f, |m| m.max(*f)));
            }
            _ => {}
        }
    }

    fn finalize(&self, agg_func: &crate::sql::planner::AggregateFunction) -> OwnedValue {
        use crate::sql::planner::AggregateFunction;

        match agg_func {
            AggregateFunction::Sum => {
                if self.has_float {
                    OwnedValue::Float(self.sum_float + self.sum_int as f64)
                } else if self.count > 0 {
                    OwnedValue::Int(self.sum_int)
                } else {
                    OwnedValue::Null
                }
            }
            AggregateFunction::Avg => {
                if self.count > 0 {
                    let total = self.sum_float + self.sum_int as f64;
                    OwnedValue::Float(total / self.count as f64)
                } else {
                    OwnedValue::Null
                }
            }
            AggregateFunction::Count => OwnedValue::Int(self.count as i64),
            AggregateFunction::Min => {
                if self.has_float {
                    match (self.min_int, self.min_float) {
                        (Some(i), Some(f)) => {
                            if compare_int_float(i, f).is_lt() {
                                OwnedValue::Int(i)
                            } else {
                                OwnedValue::Float(f)
                            }
                        }
                        (None, Some(f)) => OwnedValue::Float(f),
                        (Some(i), None) => OwnedValue::Int(i),
                        (None, None) => OwnedValue::Null,
                    }
                } else {
                    self.min_int.map(OwnedValue::Int).unwrap_or(OwnedValue::Null)
                }
            }
            AggregateFunction::Max => {
                if self.has_float {
                    match (self.max_int, self.max_float) {
                        (Some(i), Some(f)) => {
                            if compare_int_float(i, f).is_gt() {
                                OwnedValue::Int(i)
                            } else {
                                OwnedValue::Float(f)
                            }
                        }
                        (None, Some(f)) => OwnedValue::Float(f),
                        (Some(i), None) => OwnedValue::Int(i),
                        (None, None) => OwnedValue::Null,
                    }
                } else {
                    self.max_int.map(OwnedValue::Int).unwrap_or(OwnedValue::Null)
                }
            }
        }
    }
}

fn plan_has_filter(op: &crate::sql::planner::physical::PhysicalOperator<'_>) -> bool {
    use crate::sql::planner::physical::PhysicalOperator;
    match op {
        PhysicalOperator::FilterExec(_) => true,
        PhysicalOperator::ProjectExec(p) => plan_has_filter(p.input),
        PhysicalOperator::LimitExec(l) => plan_has_filter(l.input),
        PhysicalOperator::SortExec(s) => plan_has_filter(s.input),
        PhysicalOperator::TopKExec(t) => plan_has_filter(t.input),
        PhysicalOperator::HashAggregate(a) => plan_has_filter(a.input),
        PhysicalOperator::SortedAggregate(a) => plan_has_filter(a.input),
        PhysicalOperator::WindowExec(w) => plan_has_filter(w.input),
        PhysicalOperator::ScalarSubqueryExec(s) => plan_has_filter(s.subquery),
        PhysicalOperator::ExistsSubqueryExec(s) => plan_has_filter(s.subquery),
        PhysicalOperator::InListSubqueryExec(s) => plan_has_filter(s.subquery),
        _ => false,
    }
}

fn read_root_page(storage: &crate::storage::MmapStorage) -> Result<u32> {
    let page = storage.page(0)?;
    TableFileHeader::from_bytes(page)
        .map(|h| h.root_page())
        .wrap_err("failed to read table file header for root page")
}

fn read_index_root_page(storage: &crate::storage::MmapStorage) -> Result<u32> {
    let page = storage.page(0)?;
    IndexFileHeader::from_bytes(page)
        .map(|h| h.root_page())
        .wrap_err("failed to read index file header for root page")
}

impl Database {
    /// Executes a scalar subquery and returns its single result value.
    fn execute_scalar_subquery_for_update<'a>(
        subq: &'a crate::sql::ast::SelectStmt<'a>,
        catalog: &crate::schema::catalog::Catalog,
        file_manager: &mut crate::storage::FileManager,
        arena: &'a Bump,
    ) -> Result<OwnedValue> {
        use crate::btree::BTreeReader;
        use crate::database::query::{find_plan_source, PlanSource};
        use crate::records::RecordView;
        use crate::sql::builder::ExecutorBuilder;
        use crate::sql::context::ExecutionContext;
        use crate::sql::executor::{Executor, StreamingBTreeSource};
        use crate::sql::planner::{Planner, ScanRange};
        use crate::types::create_record_schema;

        let planner = Planner::new(catalog, arena);
        let stmt = crate::sql::ast::Statement::Select(subq);
        let subq_plan = planner.create_physical_plan(&stmt)
            .wrap_err("failed to create physical plan for scalar subquery")?;

        let plan_source = find_plan_source(subq_plan.root);

        match plan_source {
            Some(PlanSource::TableScan(scan)) => {
                let schema_name = scan.schema.unwrap_or(DEFAULT_SCHEMA);
                let table_name = scan.table;

                let table_def = catalog.resolve_table_in_schema(scan.schema, table_name)
                    .wrap_err_with(|| format!("failed to resolve table '{}' in scalar subquery", table_name))?;
                let column_types: Vec<_> =
                    table_def.columns().iter().map(|c| c.data_type()).collect();
                let columns = table_def.columns();

                let storage_arc = file_manager.table_data(schema_name, table_name)
                    .wrap_err_with(|| format!("failed to open table data for '{}' in scalar subquery", table_name))?;
                let storage = storage_arc.read();

                let root_page = read_root_page(&storage)
                    .wrap_err_with(|| format!("failed to read root page for table '{}'", table_name))?;

                let column_info: HashMap<String, (usize, crate::records::types::DataType)> = table_def
                    .columns()
                    .iter()
                    .enumerate()
                    .map(|(i, c)| (c.name().to_lowercase(), (i, c.data_type())))
                    .collect();

                let column_map: Vec<(String, usize)> = table_def
                    .columns()
                    .iter()
                    .enumerate()
                    .map(|(i, c)| (c.name().to_lowercase(), i))
                    .collect();

                let has_filter = plan_has_filter(subq_plan.root) || scan.post_scan_filter.is_some();
                if !has_filter {
                    if let Some((agg_func, expr)) = find_expression_aggregate(subq_plan.root) {
                        let schema = create_record_schema(columns);
                        let table_reader = crate::btree::BTreeReader::new(&storage, root_page)
                            .wrap_err_with(|| format!("failed to create BTreeReader for table '{}'", table_name))?;
                        let mut cursor = table_reader.cursor_first()?;
                        let mut agg_state = StreamingAggregateState::new();

                        while cursor.valid() {
                            let row_data = cursor.value()?;
                            let user_data = crate::database::dml::mvcc_helpers::get_user_data(row_data);
                            let record = RecordView::new(user_data, &schema)?;
                            let expr_value = eval_expr_for_record_streaming(expr, &record, &column_info);
                            agg_state.update(&expr_value);
                            if !cursor.advance()? {
                                break;
                            }
                        }

                        return Ok(agg_state.finalize(agg_func));
                    }
                }

                let source = StreamingBTreeSource::from_btree_scan_with_projections(
                    &storage,
                    root_page,
                    column_types,
                    None,
                )?;

                let ctx = ExecutionContext::new(arena);
                let builder = ExecutorBuilder::new(&ctx);
                let mut executor =
                    builder.build_with_source_and_column_map(&subq_plan, source, &column_map)?;

                executor.open()?;
                let result = if let Some(row) = executor.next()? {
                    if row.values.is_empty() {
                        bail!("scalar subquery returned row with no columns for table '{}'", table_name);
                    }
                    OwnedValue::from(row.values.first().unwrap())
                } else {
                    OwnedValue::Null
                };
                executor.close()?;
                Ok(result)
            }
            Some(PlanSource::SecondaryIndexScan(scan)) => {
                let schema_name = scan.schema.unwrap_or(DEFAULT_SCHEMA);
                let table_name = scan.table;
                let index_name = scan.index_name;

                let table_def = scan.table_def.ok_or_else(|| {
                    eyre::eyre!("SecondaryIndexScan missing table_def for table '{}' in scalar subquery", table_name)
                })?;

                let columns = table_def.columns();
                let schema = create_record_schema(columns);

                let column_info: HashMap<String, (usize, crate::records::types::DataType)> = table_def
                    .columns()
                    .iter()
                    .enumerate()
                    .map(|(i, c)| (c.name().to_lowercase(), (i, c.data_type())))
                    .collect();

                let row_id_suffix_len = if scan.is_unique_index { 0 } else { 8 };

                let has_filter = plan_has_filter(subq_plan.root);
                if !has_filter {
                    if let Some((agg_func, expr)) = find_any_aggregate(subq_plan.root) {
                    let mut agg_state = StreamingAggregateState::new();

                    let index_storage_arc = file_manager.index_data(schema_name, table_name, index_name)
                        .wrap_err_with(|| format!("failed to open index '{}' for table '{}'", index_name, table_name))?;
                    let index_storage = index_storage_arc.read();
                    let index_root = read_index_root_page(&index_storage)
                        .wrap_err_with(|| format!("failed to read root page for index '{}'", index_name))?;
                    let index_reader = BTreeReader::new(&index_storage, index_root)?;

                    let table_storage_arc = file_manager.table_data(schema_name, table_name)
                        .wrap_err_with(|| format!("failed to open table data for '{}' in scalar subquery", table_name))?;
                    let table_storage = table_storage_arc.read();
                    let table_root = read_root_page(&table_storage)
                        .wrap_err_with(|| format!("failed to read root page for table '{}'", table_name))?;
                    let table_reader = BTreeReader::new(&table_storage, table_root)?;

                    let process_index_cursor = |cursor: &mut crate::btree::Cursor<'_, crate::storage::MmapStorage>| -> Result<Option<[u8; 8]>> {
                        let index_key = cursor.key()?;
                        let row_id_bytes = if scan.is_unique_index {
                            cursor.value()?
                        } else {
                            &index_key[index_key.len().saturating_sub(row_id_suffix_len)..]
                        };
                        if row_id_bytes.len() == 8 {
                            let row_key: [u8; 8] = row_id_bytes.try_into()
                                .map_err(|_| eyre::eyre!("invalid row key in index '{}': expected 8 bytes", index_name))?;
                            Ok(Some(row_key))
                        } else {
                            Ok(None)
                        }
                    };

                    match &scan.key_range {
                        Some(ScanRange::PrefixScan { prefix }) => {
                            let mut cursor = index_reader.cursor_seek(prefix)?;
                            while cursor.valid() {
                                let index_key = cursor.key()?;
                                if !index_key.starts_with(prefix) {
                                    break;
                                }
                                if let Some(row_key) = process_index_cursor(&mut cursor)? {
                                    if let Some(row_data) = table_reader.get(&row_key)? {
                                        let user_data = crate::database::dml::mvcc_helpers::get_user_data(row_data);
                                        let record = RecordView::new(user_data, &schema)?;
                                        let expr_value = eval_expr_for_record_streaming(expr, &record, &column_info);
                                        agg_state.update(&expr_value);
                                    }
                                }
                                if !cursor.advance()? {
                                    break;
                                }
                            }
                        }
                        Some(ScanRange::RangeScan { start, end }) => {
                            let mut cursor = if let Some(start_key) = start {
                                index_reader.cursor_seek(start_key)?
                            } else {
                                index_reader.cursor_first()?
                            };
                            while cursor.valid() {
                                let index_key = cursor.key()?;
                                if let Some(end_key) = end {
                                    if index_key >= *end_key {
                                        break;
                                    }
                                }
                                if let Some(row_key) = process_index_cursor(&mut cursor)? {
                                    if let Some(row_data) = table_reader.get(&row_key)? {
                                        let user_data = crate::database::dml::mvcc_helpers::get_user_data(row_data);
                                        let record = RecordView::new(user_data, &schema)?;
                                        let expr_value = eval_expr_for_record_streaming(expr, &record, &column_info);
                                        agg_state.update(&expr_value);
                                    }
                                }
                                if !cursor.advance()? {
                                    break;
                                }
                            }
                        }
                        Some(ScanRange::FullScan) | None => {
                            let mut cursor = index_reader.cursor_first()?;
                            while cursor.valid() {
                                if let Some(row_key) = process_index_cursor(&mut cursor)? {
                                    if let Some(row_data) = table_reader.get(&row_key)? {
                                        let user_data = crate::database::dml::mvcc_helpers::get_user_data(row_data);
                                        let record = RecordView::new(user_data, &schema)?;
                                        let expr_value = eval_expr_for_record_streaming(expr, &record, &column_info);
                                        agg_state.update(&expr_value);
                                    }
                                }
                                if !cursor.advance()? {
                                    break;
                                }
                            }
                        }
                    }

                        return Ok(agg_state.finalize(agg_func));
                    }
                }

                let index_storage_arc = file_manager.index_data(schema_name, table_name, index_name)
                    .wrap_err_with(|| format!("failed to open index '{}' for table '{}'", index_name, table_name))?;
                let index_storage = index_storage_arc.read();
                let index_root = read_index_root_page(&index_storage)
                    .wrap_err_with(|| format!("failed to read root page for index '{}'", index_name))?;
                let index_reader = BTreeReader::new(&index_storage, index_root)?;

                let table_storage_arc = file_manager.table_data(schema_name, table_name)
                    .wrap_err_with(|| format!("failed to open table data for '{}' in scalar subquery", table_name))?;
                let table_storage = table_storage_arc.read();
                let table_root = read_root_page(&table_storage)
                    .wrap_err_with(|| format!("failed to read root page for table '{}'", table_name))?;
                let table_reader = BTreeReader::new(&table_storage, table_root)?;

                let extract_and_lookup_first_row =
                    |cursor: &mut crate::btree::Cursor<'_, crate::storage::MmapStorage>| -> Result<Option<OwnedValue>> {
                        while cursor.valid() {
                            let index_key = cursor.key()?;
                            let row_id_bytes = if scan.is_unique_index {
                                cursor.value()?
                            } else {
                                &index_key[index_key.len().saturating_sub(row_id_suffix_len)..]
                            };
                            if row_id_bytes.len() == 8 {
                                let row_key: [u8; 8] = row_id_bytes.try_into()
                                    .map_err(|_| eyre::eyre!("invalid row key in index '{}': expected 8 bytes", index_name))?;
                                if let Some(row_data) = table_reader.get(&row_key)? {
                                    let user_data = crate::database::dml::mvcc_helpers::get_user_data(row_data);
                                    let record = RecordView::new(user_data, &schema)?;
                                    let first_col_type = columns.first().map(|c| c.data_type())
                                        .ok_or_else(|| eyre::eyre!("table '{}' has no columns", table_name))?;
                                    let col_value = OwnedValue::from_record_column(&record, 0, first_col_type)
                                        .wrap_err_with(|| eyre::eyre!(
                                            "failed to read column 0 from record in table '{}' via index '{}'",
                                            table_name, index_name
                                        ))?;
                                    return Ok(Some(col_value));
                                }
                            }
                            if !cursor.advance()? {
                                break;
                            }
                        }
                        Ok(None)
                    };

                let result = match &scan.key_range {
                    Some(ScanRange::PrefixScan { prefix }) => {
                        let mut cursor = index_reader.cursor_seek(prefix)?;
                        if cursor.valid() {
                            let index_key = cursor.key()?;
                            if index_key.starts_with(prefix) {
                                extract_and_lookup_first_row(&mut cursor)?
                            } else {
                                None
                            }
                        } else {
                            None
                        }
                    }
                    Some(ScanRange::RangeScan { start, end }) => {
                        let mut cursor = if let Some(start_key) = start {
                            index_reader.cursor_seek(start_key)?
                        } else {
                            index_reader.cursor_first()?
                        };
                        if cursor.valid() {
                            if let Some(end_key) = end {
                                let index_key = cursor.key()?;
                                if index_key < *end_key {
                                    extract_and_lookup_first_row(&mut cursor)?
                                } else {
                                    None
                                }
                            } else {
                                extract_and_lookup_first_row(&mut cursor)?
                            }
                        } else {
                            None
                        }
                    }
                    Some(ScanRange::FullScan) | None => {
                        let mut cursor = index_reader.cursor_first()?;
                        extract_and_lookup_first_row(&mut cursor)?
                    }
                };

                Ok(result.unwrap_or(OwnedValue::Null))
            }
            _ => Ok(OwnedValue::Null),
        }
    }

    pub(crate) fn execute_update(
        &self,
        update: &crate::sql::ast::UpdateStmt<'_>,
        params: &[OwnedValue],
        arena: &Bump,
    ) -> Result<ExecuteResult> {
        self.ensure_catalog()?;
        self.ensure_file_manager()?;

        let catalog_guard = self.shared.catalog.read();
        let catalog = catalog_guard.as_ref().unwrap();

        let schema_name = update.table.schema.unwrap_or(DEFAULT_SCHEMA);
        let table_name = update.table.name;
        let table_alias = update.table.alias;

        let table_def = catalog.resolve_table_in_schema(update.table.schema, table_name)?.clone();
        let table_id = table_def.id();
        let columns = table_def.columns().to_vec();
        let has_toast = table_def.has_toast();

        let secondary_indexes: Vec<(String, Vec<usize>)> = table_def
            .indexes()
            .iter()
            .filter(|idx| idx.index_type() == IndexType::BTree)
            .map(|idx| {
                let col_indices: Vec<usize> = idx
                    .columns()
                    .filter_map(|col_name| columns.iter().position(|c| c.name() == col_name))
                    .collect();
                (idx.name().to_string(), col_indices)
            })
            .collect();

        let hnsw_indexes: Vec<(String, usize)> = table_def
            .indexes()
            .iter()
            .filter(|idx| idx.index_type() == IndexType::Hnsw)
            .filter_map(|idx| {
                let col_name = idx.columns().next()?;
                let col_idx = columns
                    .iter()
                    .position(|c| c.name().eq_ignore_ascii_case(col_name))?;
                Some((idx.name().to_string(), col_idx))
            })
            .collect();

        let unique_columns: Vec<(usize, String, bool)> = columns
            .iter()
            .enumerate()
            .filter_map(|(idx, col)| {
                let is_pk = col.has_constraint(&Constraint::PrimaryKey);
                let is_unique = col.has_constraint(&Constraint::Unique);
                if is_pk || is_unique {
                    let index_name = if is_pk {
                        format!("{}_pkey", col.name())
                    } else {
                        format!("{}_key", col.name())
                    };
                    Some((idx, index_name, is_pk))
                } else {
                    None
                }
            })
            .collect();

        #[allow(clippy::type_complexity)]
        let from_tables_data: Option<
            Vec<(
                String,
                String,
                Option<&str>,
                Vec<crate::schema::table::ColumnDef>,
            )>,
        > = if let Some(from_clause) = update.from {
            let mut tables = Vec::new();
            Self::extract_tables_from_clause(*from_clause, catalog, &mut tables)?;
            Some(tables)
        } else {
            None
        };

        let mut fk_references: Vec<(
            String,
            String,
            String,
            usize,
            Option<crate::schema::ReferentialAction>,
        )> = Vec::new();
        for (schema_key, schema_val) in catalog.schemas() {
            for (child_table_name, child_table_def) in schema_val.tables() {
                for col in child_table_def.columns().iter() {
                    for constraint in col.constraints() {
                        if let Constraint::ForeignKey {
                            table,
                            column,
                            on_update,
                            ..
                        } = constraint
                        {
                            if table == table_name {
                                let ref_col_idx =
                                    columns.iter().position(|c| c.name() == column).unwrap_or(0);
                                fk_references.push((
                                    schema_key.clone(),
                                    child_table_name.clone(),
                                    col.name().to_string(),
                                    ref_col_idx,
                                    *on_update,
                                ));
                            }
                        }
                    }
                }
            }
        }

        #[allow(clippy::type_complexity)]
        let child_table_schemas: Vec<(
            String,
            String,
            Vec<crate::schema::table::ColumnDef>,
            usize,
            usize,
            Option<crate::schema::ReferentialAction>,
        )> = fk_references
            .iter()
            .map(|(schema_key, child_name, fk_col_name, ref_col_idx, on_update)| {
                let child_def = catalog
                    .schemas()
                    .get(schema_key)
                    .unwrap()
                    .tables()
                    .get(child_name)
                    .unwrap();
                let fk_col_idx = child_def
                    .columns()
                    .iter()
                    .position(|c| c.name() == fk_col_name)
                    .unwrap_or(0);
                (
                    schema_key.clone(),
                    child_name.clone(),
                    child_def.columns().to_vec(),
                    fk_col_idx,
                    *ref_col_idx,
                    *on_update,
                )
            })
            .collect();

        let schema = create_record_schema(&columns);

        if let Some(from_tables) = from_tables_data {
            drop(catalog_guard);
            return self.execute_update_with_from(
                update,
                arena,
                schema_name,
                table_name,
                table_alias,
                &table_def,
                table_id as usize,
                &columns,
                &schema,
                from_tables,
            );
        }

        let assignment_indices: Vec<(usize, &crate::sql::ast::Expr<'_>)> = update
            .assignments
            .iter()
            .filter_map(|a| {
                columns
                    .iter()
                    .position(|c| c.name().eq_ignore_ascii_case(a.column.column))
                    .map(|idx| (idx, a.value))
            })
            .collect();

        let mut scalar_subquery_results: ScalarSubqueryResults = ScalarSubqueryResults::new();
        {
            let mut subqueries: SmallVec<[&crate::sql::ast::SelectStmt<'_>; 4]> = SmallVec::new();
            for (_, value_expr) in &assignment_indices {
                collect_scalar_subqueries_from_expr(value_expr, &mut subqueries);
            }

            if !subqueries.is_empty() {
                let mut fm_guard = self.shared.file_manager.write();
                let fm = fm_guard.as_mut().unwrap();
                for subq in subqueries {
                    // SAFETY: Using AST node address as HashMap key is safe because:
                    // 1. The AST is arena-allocated and lives for the duration of this function
                    // 2. We only read from scalar_subquery_results within this same scope
                    // 3. The same subquery AST node will have the same address when looked up
                    let key = std::ptr::from_ref(subq) as usize;
                    if !scalar_subquery_results.contains_key(&key) {
                        let result =
                            Self::execute_scalar_subquery_for_update(subq, catalog, fm, arena)?;
                        scalar_subquery_results.insert(key, result);
                    }
                }
            }
        }

        drop(catalog_guard);

        let mut file_manager_guard = self.shared.file_manager.write();
        let file_manager = file_manager_guard.as_mut().unwrap();
        let storage_arc = file_manager.table_data_mut(schema_name, table_name)?;
        let mut storage = storage_arc.write();

        let column_types: Vec<crate::records::types::DataType> =
            columns.iter().map(|c| c.data_type()).collect();
        let decoder = crate::sql::decoder::SimpleDecoder::new(column_types.clone());

        let root_page = {
            use crate::storage::TableFileHeader;
            let page = storage.page(0)?;
            TableFileHeader::from_bytes(page)?.root_page()
        };
        let btree = BTree::new(&mut *storage, root_page)?;

        let mut pk_lookup_info: Option<(SmallVec<[u8; 16]>, OwnedValue)> = 'pk_analysis: {
            if let Some(crate::sql::ast::Expr::BinaryOp {
                left,
                op: crate::sql::ast::BinaryOperator::Eq,
                right,
            }) = update.where_clause.as_ref()
            {
                if let Some(pk_idx) = columns
                    .iter()
                    .position(|c| c.has_constraint(&Constraint::PrimaryKey))
                {
                    let pk_col_name = columns[pk_idx].name();

                    let val_expr = match (&**left, &**right) {
                        (crate::sql::ast::Expr::Column(c), val)
                            if c.column.eq_ignore_ascii_case(pk_col_name) =>
                        {
                            Some(val)
                        }
                        (val, crate::sql::ast::Expr::Column(c))
                            if c.column.eq_ignore_ascii_case(pk_col_name) =>
                        {
                            Some(val)
                        }
                        _ => None,
                    };

                    if let Some(expr) = val_expr {
                        let val_opt = if let crate::sql::ast::Expr::Parameter(param_ref) = expr {
                            match param_ref {
                                crate::sql::ast::ParameterRef::Anonymous => {
                                    let mut param_offset = 0;
                                    for assign in update.assignments {
                                        param_offset += count_params_in_expr(assign.value);
                                    }
                                    params.get(param_offset).cloned()
                                }
                                crate::sql::ast::ParameterRef::Positional(idx) => {
                                    if *idx > 0 {
                                        params.get((*idx - 1) as usize).cloned()
                                    } else {
                                        None
                                    }
                                }
                                _ => None,
                            }
                        } else {
                            Self::eval_literal(expr).ok()
                        };

                        if let Some(val) = val_opt {
                            let pk_index_name = format!("{}_pkey", pk_col_name);

                            if file_manager.index_exists(schema_name, table_name, &pk_index_name) {
                                if let Ok(index_storage_arc) = file_manager.index_data_mut(
                                    schema_name,
                                    table_name,
                                    &pk_index_name,
                                ) {
                                    let mut index_storage = index_storage_arc.write();

                                    let index_root_page = {
                                        use crate::storage::IndexFileHeader;
                                        let page0 = index_storage.page(0)?;
                                        let header = IndexFileHeader::from_bytes(page0)?;
                                        header.root_page()
                                    };

                                    let index_btree =
                                        BTree::new(&mut *index_storage, index_root_page)?;

                                    let mut index_key: SmallVec<[u8; 16]> = SmallVec::new();
                                    Self::encode_value_as_key(&val, &mut index_key);

                                    if let Some(handle) = index_btree.search(&index_key)? {
                                        let row_key_slice = index_btree.get_value(&handle)?;
                                        break 'pk_analysis Some((SmallVec::from_slice(row_key_slice), val));
                                    }
                                }
                            }
                            break 'pk_analysis None;
                        }
                    }
                }
            }
            None
        };

        #[allow(clippy::type_complexity)]
        let mut rows_to_update: Vec<(
            Vec<u8>,
            Vec<u8>,
            Vec<OwnedValue>,
            Vec<OwnedValue>,
            Vec<(usize, OwnedValue)>,
        )> = Vec::new();

        let mut precomputed_assignments: SmallVec<[(usize, OwnedValue); 8]> = SmallVec::new();
        let mut deferred_assignments: SmallVec<[(usize, usize); 4]> = SmallVec::new();
        let set_param_count: usize;
        {
            let mut param_idx = 0;
            for (assign_idx, (col_idx, value_expr)) in assignment_indices.iter().enumerate() {
                if expr_contains_column_ref(value_expr) {
                    deferred_assignments.push((*col_idx, assign_idx));
                    param_idx += count_params_in_expr(value_expr);
                } else {
                    let val = Self::eval_expr_with_params_and_subqueries(
                        value_expr,
                        column_types.get(*col_idx),
                        Some(params),
                        &mut param_idx,
                        &scalar_subquery_results,
                    )?
                    .into_owned();
                    precomputed_assignments.push((*col_idx, val));
                }
            }
            set_param_count = param_idx;
        }

        let has_where = update.where_clause.is_some();
        let has_deferred = !deferred_assignments.is_empty();

        let needs_column_map = has_where || has_deferred;
        let base_column_map: Option<Vec<(String, usize)>> = if needs_column_map {
            Some(
                columns
                    .iter()
                    .enumerate()
                    .map(|(i, c)| (c.name().to_string(), i))
                    .collect(),
            )
        } else {
            None
        };

        let (predicate, column_map): (
            Option<crate::sql::predicate::CompiledPredicate>,
            Option<Vec<(String, usize)>>,
        ) = match (has_where, has_deferred, base_column_map) {
            (true, true, Some(col_map)) => {
                let pred = crate::sql::predicate::CompiledPredicate::with_params_from_slice(
                    update.where_clause.unwrap(),
                    &col_map,
                    params,
                    set_param_count,
                );
                (Some(pred), Some(col_map))
            }
            (true, false, Some(col_map)) => {
                let pred = crate::sql::predicate::CompiledPredicate::with_params(
                    update.where_clause.unwrap(),
                    col_map,
                    params,
                    set_param_count,
                );
                (Some(pred), None)
            }
            (false, true, Some(col_map)) => (None, Some(col_map)),
            _ => (None, None),
        };

        let modified_col_indices: HashSet<usize> =
            assignment_indices.iter().map(|(idx, _)| *idx).collect();

        let needs_old_row_for_secondary_index = secondary_indexes
            .iter()
            .any(|(_, col_indices)| col_indices.iter().any(|idx| modified_col_indices.contains(idx)));

        let unique_col_indices: Vec<usize> = columns
            .iter()
            .enumerate()
            .filter(|(idx, col)| {
                (col.has_constraint(&Constraint::Unique)
                    || col.has_constraint(&Constraint::PrimaryKey))
                    && modified_col_indices.contains(idx)
            })
            .map(|(idx, _)| idx)
            .collect();

        let can_onepass = pk_lookup_info.is_some()
            && unique_col_indices.is_empty()
            && !has_toast
            && deferred_assignments.is_empty();

        if can_onepass {
            if let Some((ref target_key, ref target_val)) = pk_lookup_info {
                let cursor = btree.cursor_seek(target_key)?;

                if cursor.valid() && cursor.key()? == target_key.as_slice() {
                    let key = cursor.key()?;
                    let value = cursor.value()?;
                    let user_data = get_user_data(value);
                    let values = decoder.decode(key, user_data)?;
                    let mut row_values: Vec<OwnedValue> =
                        values.into_iter().map(OwnedValue::from).collect();
                    let pk_idx = columns
                        .iter()
                        .position(|c| c.has_constraint(&Constraint::PrimaryKey))
                        .unwrap();

                    if &row_values[pk_idx] == target_val {
                        let old_value = value.to_vec();

                        for (col_idx, val) in &precomputed_assignments {
                            row_values[*col_idx] = val.clone();
                        }

                        let validator = crate::constraints::ConstraintValidator::new(&table_def);
                        validator.validate_update(&row_values)?;

                        for (col_idx, col) in columns.iter().enumerate() {
                            for constraint in col.constraints() {
                                if let Constraint::Check(expr_str) = constraint {
                                    let col_value = row_values.get(col_idx);
                                    if !Self::evaluate_check_expression(
                                        expr_str,
                                        col.name(),
                                        col_value,
                                    )? {
                                        bail!(
                                            "CHECK constraint violated on column '{}' in table '{}': {}",
                                            col.name(),
                                            table_name,
                                            expr_str
                                        );
                                    }
                                }
                            }
                        }

                        let user_record =
                            OwnedValue::build_record_from_values(&row_values, &schema)?;

                        let (txn_id, in_transaction) = {
                            let active_txn = self.active_txn.lock();
                            if let Some(ref txn) = *active_txn {
                                (txn.txn_id, true)
                            } else {
                                (
                                    self.shared
                                        .txn_manager
                                        .global_ts
                                        .fetch_add(1, Ordering::SeqCst),
                                    false,
                                )
                            }
                        };
                        let record_data =
                            wrap_record_for_update(txn_id, &user_record, 0, 0, in_transaction);

                        drop(storage);

                        let wal_enabled = self.shared.wal_enabled.load(Ordering::Acquire);
                        if wal_enabled {
                            self.ensure_wal()?;
                        }

                        let storage_arc = file_manager.table_data_mut(schema_name, table_name)?;
                        let mut storage_inner = storage_arc.write();

                        with_btree_storage!(
                            wal_enabled,
                            &mut *storage_inner,
                            &self.shared.dirty_tracker,
                            table_id as u32,
                            root_page,
                            |btree_mut: &mut crate::btree::BTree<_>| {
                                if !btree_mut.update(target_key, &record_data)? {
                                    btree_mut.delete(target_key)?;
                                    btree_mut.insert(target_key, &record_data)?;
                                }
                                Ok::<_, eyre::Report>(())
                            }
                        );
                        drop(storage_inner);

                        self.flush_wal_if_autocommit(
                            file_manager,
                            schema_name,
                            table_name,
                            table_id as u32,
                        )?;

                        {
                            let mut active_txn = self.active_txn.lock();
                            if let Some(ref mut txn) = *active_txn {
                                txn.add_write_entry_with_undo(
                                    WriteEntry {
                                        table_id: table_id as u32,
                                        key: target_key.to_vec(),
                                        page_id: 0,
                                        offset: 0,
                                        undo_page_id: None,
                                        undo_offset: None,
                                        is_insert: false,
                                    },
                                    old_value,
                                );
                            }
                        }

                        return Ok(ExecuteResult::Update {
                            rows_affected: 1,
                            returned: None,
                        });
                    }
                }

                // Key not found or didn't match - return 0 rows affected
                return Ok(ExecuteResult::Update {
                    rows_affected: 0,
                    returned: None,
                });
            }
        }

        // MULTIPASS: Fallback for complex queries
        let mut deferred_values_buf: Vec<(usize, OwnedValue)> =
            Vec::with_capacity(deferred_assignments.len());

        loop {
            let mut cursor = if let Some((ref key, _)) = pk_lookup_info {
                btree.cursor_seek(key)?
            } else {
                btree.cursor_first()?
            };

            while cursor.valid() {
                let key = cursor.key()?;

                if let Some((ref target_key, _)) = pk_lookup_info {
                    if key != target_key.as_slice() {
                        break;
                    }
                }

                let value = cursor.value()?;
                let user_data = get_user_data(value);
                let values = decoder.decode(key, user_data)?;
                let mut row_values: Vec<OwnedValue> =
                    values.into_iter().map(OwnedValue::from).collect();

                let should_update = if let Some((_, ref target_val)) = pk_lookup_info {
                    if let Some(pk_idx) = columns
                        .iter()
                        .position(|c| c.has_constraint(&Constraint::PrimaryKey))
                    {
                        &row_values[pk_idx] == target_val
                    } else {
                        false
                    }
                } else if let Some(ref pred) = predicate {
                    let values_iter = row_values.iter().map(|ov| ov.to_value());
                    let values_slice = arena.alloc_slice_fill_iter(values_iter);
                    let exec_row = ExecutorRow::new(values_slice);
                    pred.evaluate(&exec_row)
                } else {
                    true
                };

                if should_update {
                    let old_value = value.to_vec();
                    let old_row_values = if needs_old_row_for_secondary_index {
                        row_values.clone()
                    } else {
                        Vec::new()
                    };

                    let mut old_toast_values: Vec<(usize, OwnedValue)> = Vec::new();

                    for (col_idx, val) in &precomputed_assignments {
                        let old = std::mem::replace(&mut row_values[*col_idx], val.clone());
                        if let OwnedValue::ToastPointer(_) = old {
                            old_toast_values.push((*col_idx, old));
                        }
                    }

                    if !deferred_assignments.is_empty() {
                        let col_map = column_map.as_ref().unwrap();
                        deferred_values_buf.clear();
                        {
                            let values_iter = row_values.iter().map(|ov| ov.to_value());
                            let values_slice = arena.alloc_slice_fill_iter(values_iter);
                            let exec_row = ExecutorRow::new(values_slice);

                            for (col_idx, assign_idx) in &deferred_assignments {
                                let (_, value_expr) = &assignment_indices[*assign_idx];
                                let new_val =
                                    self.eval_expr_with_row(value_expr, &exec_row, col_map)?;
                                deferred_values_buf.push((*col_idx, new_val));
                            }
                        }

                        for (col_idx, new_val) in deferred_values_buf.drain(..) {
                            let old = std::mem::replace(&mut row_values[col_idx], new_val);
                            if let OwnedValue::ToastPointer(_) = old {
                                old_toast_values.push((col_idx, old));
                            }
                        }
                    }

                    let validator = crate::constraints::ConstraintValidator::new(&table_def);
                    validator.validate_update(&row_values)?;

                    for (col_idx, col) in columns.iter().enumerate() {
                        for constraint in col.constraints() {
                            if let Constraint::Check(expr_str) = constraint {
                                let col_value = row_values.get(col_idx);
                                if !Self::evaluate_check_expression(expr_str, col.name(), col_value)?
                                {
                                    bail!(
                                        "CHECK constraint violated on column '{}' in table '{}': {}",
                                        col.name(),
                                        table_name,
                                        expr_str
                                    );
                                }
                            }
                        }
                    }

                    rows_to_update.push((
                        key.to_vec(),
                        old_value,
                        row_values,
                        old_row_values,
                        old_toast_values,
                    ));
                }

                cursor.advance()?;
            }

            if pk_lookup_info.is_some() && rows_to_update.is_empty() {
                pk_lookup_info = None;
                continue;
            }

            break;
        }

        drop(storage);

        if !unique_col_indices.is_empty() {
            for (update_key, _old_value, updated_values, _old_row_values, _old_toast) in
                &rows_to_update
            {
                for &col_idx in &unique_col_indices {
                    let new_val = &updated_values[col_idx];
                    if new_val.is_null() {
                        continue;
                    }

                    let col_name = columns[col_idx].name();
                    let index_name = if columns[col_idx].has_constraint(&Constraint::PrimaryKey) {
                        format!("{}_pkey", col_name)
                    } else {
                        format!("{}_key", col_name)
                    };

                    if let Ok(index_storage_arc) =
                        file_manager.index_data_mut(schema_name, table_name, &index_name)
                    {
                        let mut index_storage = index_storage_arc.write();
                        let index_root_page = {
                            use crate::storage::IndexFileHeader;
                            let page0 = index_storage.page(0)?;
                            let header = IndexFileHeader::from_bytes(page0)?;
                            header.root_page()
                        };
                        let index_btree = BTree::new(&mut *index_storage, index_root_page)?;

                        let mut key_buf = Vec::new();
                        Self::encode_value_as_key(new_val, &mut key_buf);

                        // O(log n) index lookup
                        if let Some(handle) = index_btree.search(&key_buf)? {
                            let existing_row_key = index_btree.get_value(&handle)?;
                            if existing_row_key != update_key.as_slice() {
                                // Different row has this value - UNIQUE violation
                                bail!(
                                    "UNIQUE constraint violated on column '{}' in table '{}': value already exists",
                                    col_name,
                                    table_name
                                );
                            }
                        }
                    }
                }
            }
        }

        let rows_affected = rows_to_update.len();

        let returned_rows: Option<Vec<Row>> = update.returning.map(|returning_cols| {
            rows_to_update
                .iter()
                .map(
                    |(_key, _old_value, updated_values, _old_row_values, _old_toast)| {
                        let row_values: Vec<OwnedValue> = returning_cols
                            .iter()
                            .flat_map(|col| match col {
                                crate::sql::ast::SelectColumn::AllColumns => updated_values.clone(),
                                crate::sql::ast::SelectColumn::TableAllColumns(_) => {
                                    updated_values.clone()
                                }
                                crate::sql::ast::SelectColumn::Expr { expr, .. } => {
                                    if let crate::sql::ast::Expr::Column(col_ref) = expr {
                                        columns
                                            .iter()
                                            .position(|c| {
                                                c.name().eq_ignore_ascii_case(col_ref.column)
                                            })
                                            .and_then(|idx| updated_values.get(idx).cloned())
                                            .map(|v| vec![v])
                                            .unwrap_or_default()
                                    } else {
                                        vec![]
                                    }
                                }
                            })
                            .collect();
                        Row::new(row_values)
                    },
                )
                .collect()
        });

        let mut processed_rows: Vec<(Vec<u8>, Vec<OwnedValue>)> =
            Vec::with_capacity(rows_to_update.len());

        let wal_enabled = self
            .shared
            .wal_enabled
            .load(std::sync::atomic::Ordering::Acquire);
        if wal_enabled {
            self.ensure_wal()?;
        }

        if has_toast {
            use crate::storage::toast::ToastPointer;
            for row_tuple in &mut rows_to_update {
                let (_key, _old_value, updated_values, _old_row_values, old_toast_values) =
                    row_tuple;

                for (_col_idx, old_val) in old_toast_values.iter() {
                    if let OwnedValue::ToastPointer(ptr) = old_val {
                        if let Ok(pointer) = ToastPointer::decode(ptr) {
                            let _ = self.delete_toast_chunks(
                                file_manager,
                                schema_name,
                                table_name,
                                pointer.row_id(),
                                pointer.column_index(),
                                pointer.total_size,
                            );
                        }
                    }
                }

                let pk_value = if let Some(pk_idx) = columns
                    .iter()
                    .position(|c| c.has_constraint(&Constraint::PrimaryKey))
                {
                    if let OwnedValue::Int(id) = &updated_values[pk_idx] {
                        *id as u64
                    } else {
                        0
                    }
                } else {
                    0
                };

                for (col_idx, val) in updated_values.iter_mut().enumerate() {
                    if columns[col_idx].data_type().is_toastable() {
                        let needs_toast = match val {
                            OwnedValue::Text(s) => crate::storage::toast::needs_toast(s.as_bytes()),
                            OwnedValue::Blob(b) => crate::storage::toast::needs_toast(b),
                            _ => false,
                        };
                        if needs_toast {
                            let data = match val {
                                OwnedValue::Text(s) => s.as_bytes().to_vec(),
                                OwnedValue::Blob(b) => b.clone(),
                                _ => continue,
                            };
                            let (pointer, _) = self.toast_value(
                                file_manager,
                                schema_name,
                                table_name,
                                pk_value,
                                col_idx as u16,
                                &data,
                                wal_enabled,
                                None,
                            )?;
                            *val = OwnedValue::ToastPointer(pointer);
                        }
                    }
                }
            }
        }

        let mut key_buf: SmallVec<[u8; 64]> = SmallVec::new();

        for (col_idx, index_name, _is_pk) in &unique_columns {
            if !modified_col_indices.contains(col_idx) {
                continue;
            }
            if file_manager.index_exists(schema_name, table_name, index_name) {
                let index_storage_arc =
                    file_manager.index_data_mut(schema_name, table_name, index_name)?;
                let mut index_storage = index_storage_arc.write();

                let index_root_page = {
                    let page0 = index_storage.page(0)?;
                    let header = IndexFileHeader::from_bytes(page0)?;
                    header.root_page()
                };

                let mut index_btree = BTree::new(&mut *index_storage, index_root_page)?;

                for (_row_key, _old_value, new_row_values, old_row_values, _old_toast) in
                    &rows_to_update
                {
                    if let Some(old_value) = old_row_values.get(*col_idx) {
                        if !old_value.is_null() {
                            key_buf.clear();
                            Self::encode_value_as_key(old_value, &mut key_buf);
                            let _ = index_btree.delete(&key_buf);
                        }
                    }

                    if let Some(new_value) = new_row_values.get(*col_idx) {
                        if !new_value.is_null() {
                            key_buf.clear();
                            Self::encode_value_as_key(new_value, &mut key_buf);
                            if let Some(pk_idx) = columns
                                .iter()
                                .position(|c| c.has_constraint(&Constraint::PrimaryKey))
                            {
                                if let Some(OwnedValue::Int(pk_val)) = new_row_values.get(pk_idx) {
                                    let row_id_bytes = (*pk_val as u64).to_be_bytes();
                                    let _ = index_btree.insert(&key_buf, &row_id_bytes);
                                }
                            }
                        }
                    }
                }
            }
        }

        for (index_name, col_indices) in &secondary_indexes {
            if col_indices.is_empty() {
                continue;
            }
            let any_modified = col_indices
                .iter()
                .any(|idx| modified_col_indices.contains(idx));
            if !any_modified {
                continue;
            }
            if file_manager.index_exists(schema_name, table_name, index_name) {
                let index_storage_arc =
                    file_manager.index_data_mut(schema_name, table_name, index_name)?;
                let mut index_storage = index_storage_arc.write();

                let index_root_page = {
                    let page0 = index_storage.page(0)?;
                    let header = IndexFileHeader::from_bytes(page0)?;
                    header.root_page()
                };

                let mut index_btree = BTree::new(&mut *index_storage, index_root_page)?;

                for (_row_key, _old_value, new_row_values, old_row_values, _old_toast) in
                    &rows_to_update
                {
                    let old_all_non_null = col_indices
                        .iter()
                        .all(|&idx| old_row_values.get(idx).is_some_and(|v| !v.is_null()));

                    if old_all_non_null {
                        key_buf.clear();
                        for &col_idx in col_indices {
                            if let Some(value) = old_row_values.get(col_idx) {
                                Self::encode_value_as_key(value, &mut key_buf);
                            }
                        }
                        let _ = index_btree.delete(&key_buf);
                    }

                    let new_all_non_null = col_indices
                        .iter()
                        .all(|&idx| new_row_values.get(idx).is_some_and(|v| !v.is_null()));

                    if new_all_non_null {
                        key_buf.clear();
                        for &col_idx in col_indices {
                            if let Some(value) = new_row_values.get(col_idx) {
                                Self::encode_value_as_key(value, &mut key_buf);
                            }
                        }
                        if let Some(pk_idx) = columns
                            .iter()
                            .position(|c| c.has_constraint(&Constraint::PrimaryKey))
                        {
                            if let Some(OwnedValue::Int(pk_val)) = new_row_values.get(pk_idx) {
                                let row_id_bytes = (*pk_val as u64).to_be_bytes();
                                let _ = index_btree.insert(&key_buf, &row_id_bytes);
                            }
                        }
                    }
                }
            }
        }

        for (index_name, col_idx) in &hnsw_indexes {
            if !modified_col_indices.contains(col_idx) {
                continue;
            }
            if file_manager.hnsw_exists(schema_name, table_name, index_name) {
                let hnsw = self.get_or_create_hnsw_index(schema_name, table_name, index_name)?;
                let mut hnsw_guard = hnsw.write();

                for (row_key, _old_value, new_row_values, _old_row_values, _old_toast) in
                    &rows_to_update
                {
                    if row_key.len() == 8 {
                        let row_id = u64::from_be_bytes(row_key[..8].try_into().unwrap());
                        let _ = hnsw_guard.delete_by_row_id(row_id);

                        if let Some(OwnedValue::Vector(vec)) = new_row_values.get(*col_idx) {
                            let random = Self::generate_random_for_hnsw(row_id);
                            let _ = hnsw_guard.insert(row_id, vec, random);
                        }
                    }
                }
            }
        }

        for (key, _old_value, updated_values, _old_row_values, _old_toast) in &rows_to_update {
            processed_rows.push((key.clone(), updated_values.clone()));
        }

        let (txn_id, in_transaction) = {
            let active_txn = self.active_txn.lock();
            if let Some(ref txn) = *active_txn {
                (txn.txn_id, true)
            } else {
                (
                    self.shared
                        .txn_manager
                        .global_ts
                        .fetch_add(1, Ordering::SeqCst),
                    false,
                )
            }
        };

        let storage_arc = file_manager.table_data_mut(schema_name, table_name)?;
        let mut storage = storage_arc.write();

        with_btree_storage!(
            wal_enabled,
            &mut *storage,
            &self.shared.dirty_tracker,
            table_id as u32,
            root_page,
            |btree_mut: &mut crate::btree::BTree<_>| {
                for (key, updated_values) in &processed_rows {
                    let user_record =
                        OwnedValue::build_record_from_values(updated_values, &schema)?;
                    let record_data =
                        wrap_record_for_update(txn_id, &user_record, 0, 0, in_transaction);

                    if !btree_mut.update(key, &record_data)? {
                        btree_mut.delete(key)?;
                        btree_mut.insert(key, &record_data)?;
                    }
                }
                Ok::<_, eyre::Report>(())
            }
        );
        drop(storage);

        let relevant_fk_refs: Vec<_> = child_table_schemas
            .iter()
            .filter(|(_, _, _, _, _, on_update)| on_update.is_some())
            .filter(|(_, _, _, _, parent_ref_col_idx, _)| {
                modified_col_indices.contains(parent_ref_col_idx)
            })
            .collect();

        if !relevant_fk_refs.is_empty() {
            let mut value_changes: Vec<(usize, OwnedValue, OwnedValue)> = Vec::new();
            for (_key, _old_value, new_values, old_values, _toast) in &rows_to_update {
                for &parent_col_idx in &modified_col_indices {
                    let is_fk_ref = relevant_fk_refs
                        .iter()
                        .any(|(_, _, _, _, ref_idx, _)| *ref_idx == parent_col_idx);
                    if is_fk_ref {
                        if let (Some(old_v), Some(new_v)) =
                            (old_values.get(parent_col_idx), new_values.get(parent_col_idx))
                        {
                            if old_v != new_v {
                                value_changes
                                    .push((parent_col_idx, old_v.clone(), new_v.clone()));
                            }
                        }
                    }
                }
            }

            if !value_changes.is_empty() {
                for (
                    child_schema,
                    child_name,
                    child_columns,
                    fk_col_idx,
                    parent_ref_col_idx,
                    on_update,
                ) in &relevant_fk_refs
                {
                    let child_storage_arc =
                        file_manager.table_data_mut(child_schema, child_name)?;
                    let mut child_storage = child_storage_arc.write();
                    let child_btree = BTree::new(&mut *child_storage, 1u32)?;
                    let mut child_cursor = child_btree.cursor_first()?;
                    let child_record_schema = create_record_schema(child_columns);

                    let mut cascade_updates: Vec<(Vec<u8>, Vec<OwnedValue>, OwnedValue)> =
                        Vec::new();

                    while child_cursor.valid() {
                        let child_key = child_cursor.key()?.to_vec();
                        let child_value = child_cursor.value()?;
                        let child_user_data = get_user_data(child_value);
                        let child_record =
                            RecordView::new(child_user_data, &child_record_schema)?;
                        let child_row =
                            OwnedValue::extract_row_from_record(&child_record, child_columns)?;

                        if let Some(child_fk_val) = child_row.get(*fk_col_idx) {
                            for (changed_parent_col_idx, old_val, new_val) in &value_changes {
                                if *changed_parent_col_idx == *parent_ref_col_idx
                                    && child_fk_val == old_val
                                {
                                    match on_update {
                                        Some(crate::schema::ReferentialAction::Cascade) => {
                                            cascade_updates.push((
                                                child_key.clone(),
                                                child_row.clone(),
                                                new_val.clone(),
                                            ));
                                        }
                                        Some(crate::schema::ReferentialAction::SetNull) => {
                                            cascade_updates.push((
                                                child_key.clone(),
                                                child_row.clone(),
                                                OwnedValue::Null,
                                            ));
                                        }
                                        Some(crate::schema::ReferentialAction::SetDefault) => {
                                            let default_val = child_columns
                                                .get(*fk_col_idx)
                                                .and_then(
                                                    |c: &crate::schema::table::ColumnDef| {
                                                        c.default_value()
                                                    },
                                                )
                                                .map(|d: &str| OwnedValue::Text(d.to_string()))
                                                .unwrap_or(OwnedValue::Null);
                                            cascade_updates.push((
                                                child_key.clone(),
                                                child_row.clone(),
                                                default_val,
                                            ));
                                        }
                                        Some(crate::schema::ReferentialAction::Restrict)
                                        | Some(crate::schema::ReferentialAction::NoAction)
                                        | None => {
                                            bail!(
                                                "FOREIGN KEY constraint violated: cannot update '{}' because row is referenced by '{}'",
                                                table_name,
                                                child_name
                                            );
                                        }
                                    }
                                }
                            }
                        }

                        child_cursor.advance()?;
                    }

                    drop(child_storage);

                    if !cascade_updates.is_empty() {
                        let child_storage_arc =
                            file_manager.table_data_mut(child_schema, child_name)?;
                        let mut child_storage = child_storage_arc.write();
                        let mut child_btree = BTree::new(&mut *child_storage, 1u32)?;

                        for (child_key, mut child_row, new_fk_val) in cascade_updates {
                            child_row[*fk_col_idx] = new_fk_val;
                            let child_record = OwnedValue::build_record_from_values(
                                &child_row,
                                &child_record_schema,
                            )?;
                            let child_record_data = wrap_record_for_update(
                                txn_id,
                                &child_record,
                                0,
                                0,
                                in_transaction,
                            );
                            if !child_btree.update(&child_key, &child_record_data)? {
                                child_btree.delete(&child_key)?;
                                child_btree.insert(&child_key, &child_record_data)?;
                            }
                        }
                    }
                }
            }
        }

        self.flush_wal_if_autocommit(file_manager, schema_name, table_name, table_id as u32)?;

        drop(file_manager_guard);

        {
            let mut active_txn = self.active_txn.lock();
            if let Some(ref mut txn) = *active_txn {
                for (key, old_value, _updated_values, _old_row_values, _old_toast) in rows_to_update
                {
                    txn.add_write_entry_with_undo(
                        WriteEntry {
                            table_id: table_id as u32,
                            key,
                            page_id: 0,
                            offset: 0,
                            undo_page_id: None,
                            undo_offset: None,
                            is_insert: false,
                        },
                        old_value,
                    );
                }
            }
        }

        Ok(ExecuteResult::Update {
            rows_affected,
            returned: returned_rows,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn execute_update_with_from(
        &self,
        update: &crate::sql::ast::UpdateStmt<'_>,
        arena: &Bump,
        schema_name: &str,
        table_name: &str,
        table_alias: Option<&str>,
        table_def: &crate::schema::table::TableDef,
        table_id: usize,
        columns: &[crate::schema::table::ColumnDef],
        schema: &crate::records::Schema,
        from_tables: Vec<(
            String,
            String,
            Option<&str>,
            Vec<crate::schema::table::ColumnDef>,
        )>,
    ) -> Result<ExecuteResult> {
        let mut combined_column_map: Vec<(String, usize)> = Vec::new();
        for (idx, col) in columns.iter().enumerate() {
            combined_column_map.push((col.name().to_string(), idx));
            combined_column_map.push((format!("{}.{}", table_name, col.name()), idx));
            if let Some(alias) = table_alias {
                combined_column_map.push((format!("{}.{}", alias, col.name()), idx));
            }
        }

        let mut current_col_offset = columns.len();
        let mut from_schemas: Vec<crate::records::Schema> = Vec::new();
        for (_, from_table_name, from_alias, from_columns) in &from_tables {
            for (idx, col) in from_columns.iter().enumerate() {
                combined_column_map.push((col.name().to_string(), current_col_offset + idx));
                combined_column_map.push((
                    format!("{}.{}", from_table_name, col.name()),
                    current_col_offset + idx,
                ));
                if let Some(alias) = from_alias {
                    combined_column_map.push((
                        format!("{}.{}", alias, col.name()),
                        current_col_offset + idx,
                    ));
                }
            }
            current_col_offset += from_columns.len();
            from_schemas.push(create_record_schema(from_columns));
        }

        let predicate = update
            .where_clause
            .map(|expr| CompiledPredicate::with_column_map_ref(expr, &combined_column_map));

        let assignment_indices: Vec<(usize, &crate::sql::ast::Expr<'_>)> = update
            .assignments
            .iter()
            .filter_map(|a| {
                columns
                    .iter()
                    .position(|c| c.name().eq_ignore_ascii_case(a.column.column))
                    .map(|idx| (idx, a.value))
            })
            .collect();

        let mut file_manager_guard = self.shared.file_manager.write();
        let file_manager = file_manager_guard.as_mut().unwrap();

        let mut all_from_rows: Vec<Vec<Vec<OwnedValue>>> = Vec::new();
        for (i, (from_schema_name, from_table_name, _, from_columns)) in
            from_tables.iter().enumerate()
        {
            let from_storage_arc =
                file_manager.table_data_mut(from_schema_name, from_table_name)?;
            let mut from_storage = from_storage_arc.write();
            let from_btree = BTree::new(&mut *from_storage, 1u32)?;
            let mut from_cursor = from_btree.cursor_first()?;

            let mut table_rows: Vec<Vec<OwnedValue>> = Vec::new();
            while from_cursor.valid() {
                let value = from_cursor.value()?;
                let user_data = get_user_data(value);
                let record = RecordView::new(user_data, &from_schemas[i])?;
                let row_values = OwnedValue::extract_row_from_record(&record, from_columns)?;
                table_rows.push(row_values);
                from_cursor.advance()?;
            }
            all_from_rows.push(table_rows);
        }

        let combined_from_rows = Self::cartesian_product(&all_from_rows);

        let storage_arc = file_manager.table_data_mut(schema_name, table_name)?;
        let mut storage = storage_arc.write();
        let root_page = {
            use crate::storage::TableFileHeader;
            let page = storage.page(0)?;
            TableFileHeader::from_bytes(page)?.root_page()
        };
        let btree = BTree::new(&mut *storage, root_page)?;
        let mut cursor = btree.cursor_first()?;

        let column_types: Vec<crate::records::types::DataType> =
            columns.iter().map(|c| c.data_type()).collect();
        let decoder = crate::sql::decoder::SimpleDecoder::new(column_types);

        let mut rows_to_update: Vec<(Vec<u8>, Vec<u8>, Vec<OwnedValue>)> = Vec::new();
        let mut updated_keys: HashSet<Vec<u8>> = HashSet::new();

        while cursor.valid() {
            let key = cursor.key()?;
            let value = cursor.value()?;

            let user_data = get_user_data(value);
            let values = decoder.decode(key, user_data)?;
            let target_row_values: Vec<OwnedValue> =
                values.into_iter().map(OwnedValue::from).collect();

            for from_row in &combined_from_rows {
                let mut combined_values: Vec<Value<'_>> =
                    Vec::with_capacity(target_row_values.len() + from_row.len());

                for val in &target_row_values {
                    combined_values.push(match val {
                        OwnedValue::Null => Value::Null,
                        OwnedValue::Bool(b) => Value::Int(if *b { 1 } else { 0 }),
                        OwnedValue::Int(i) => Value::Int(*i),
                        OwnedValue::Float(f) => Value::Float(*f),
                        OwnedValue::Text(s) => Value::Text(Cow::Borrowed(s.as_str())),
                        OwnedValue::Blob(b) => Value::Blob(Cow::Borrowed(b.as_slice())),
                        _ => Value::Null,
                    });
                }

                for val in from_row {
                    combined_values.push(match val {
                        OwnedValue::Null => Value::Null,
                        OwnedValue::Bool(b) => Value::Int(if *b { 1 } else { 0 }),
                        OwnedValue::Int(i) => Value::Int(*i),
                        OwnedValue::Float(f) => Value::Float(*f),
                        OwnedValue::Text(s) => Value::Text(Cow::Borrowed(s.as_str())),
                        OwnedValue::Blob(b) => Value::Blob(Cow::Borrowed(b.as_slice())),
                        _ => Value::Null,
                    });
                }

                let values_slice = arena.alloc_slice_fill_iter(combined_values.into_iter());
                let exec_row = ExecutorRow::new(values_slice);

                let should_update = if let Some(ref pred) = predicate {
                    pred.evaluate(&exec_row)
                } else {
                    true
                };

                if should_update && !updated_keys.contains(&key.to_vec()) {
                    let old_value = value.to_vec();
                    let mut row_values = target_row_values.clone();

                    for (col_idx, value_expr) in &assignment_indices {
                        let new_value =
                            self.eval_expr_with_row(value_expr, &exec_row, &combined_column_map)?;
                        row_values[*col_idx] = new_value;
                    }

                    let validator = crate::constraints::ConstraintValidator::new(table_def);
                    validator.validate_update(&row_values)?;

                    for (col_idx, col) in columns.iter().enumerate() {
                        for constraint in col.constraints() {
                            if let Constraint::Check(expr_str) = constraint {
                                let col_value = row_values.get(col_idx);
                                if !Self::evaluate_check_expression(expr_str, col.name(), col_value)?
                                {
                                    bail!(
                                        "CHECK constraint violated on column '{}' in table '{}': {}",
                                        col.name(),
                                        table_name,
                                        expr_str
                                    );
                                }
                            }
                        }
                    }

                    updated_keys.insert(key.to_vec());
                    rows_to_update.push((key.to_vec(), old_value, row_values));
                }
            }

            cursor.advance()?;
        }

        drop(storage);

        let unique_col_indices: Vec<usize> = columns
            .iter()
            .enumerate()
            .filter(|(_, col)| {
                col.has_constraint(&Constraint::Unique)
                    || col.has_constraint(&Constraint::PrimaryKey)
            })
            .map(|(idx, _)| idx)
            .collect();

        if !unique_col_indices.is_empty() {
            let storage_for_check_arc = file_manager.table_data_mut(schema_name, table_name)?;
            let mut storage_for_check = storage_for_check_arc.write();
            let btree_for_check = BTree::new(&mut *storage_for_check, root_page)?;
            let mut check_cursor = btree_for_check.cursor_first()?;

            for (update_key, _old_value, updated_values) in &rows_to_update {
                while check_cursor.valid() {
                    let existing_key = check_cursor.key()?;

                    if existing_key != update_key.as_slice() {
                        let existing_value = check_cursor.value()?;
                        let existing_user_data = get_user_data(existing_value);
                        let existing_record = RecordView::new(existing_user_data, schema)?;
                        let existing_values =
                            OwnedValue::extract_row_from_record(&existing_record, columns)?;

                        for &col_idx in &unique_col_indices {
                            let new_val = updated_values.get(col_idx);
                            let existing_val = existing_values.get(col_idx);

                            if let (Some(new_v), Some(existing_v)) = (new_val, existing_val) {
                                if !new_v.is_null() && !existing_v.is_null() && new_v == existing_v
                                {
                                    let col_name = &columns[col_idx].name();
                                    bail!(
                                        "UNIQUE constraint violated on column '{}' in table '{}': value already exists",
                                        col_name,
                                        table_name
                                    );
                                }
                            }
                        }
                    }
                    check_cursor.advance()?;
                }
                check_cursor = btree_for_check.cursor_first()?;
            }
        }

        let rows_affected = rows_to_update.len();

        let returned_rows: Option<Vec<Row>> = update.returning.map(|returning_cols| {
            rows_to_update
                .iter()
                .map(|(_key, _old_value, updated_values)| {
                    let row_values: Vec<OwnedValue> = returning_cols
                        .iter()
                        .flat_map(|col| match col {
                            crate::sql::ast::SelectColumn::AllColumns => updated_values.clone(),
                            crate::sql::ast::SelectColumn::TableAllColumns(_) => {
                                updated_values.clone()
                            }
                            crate::sql::ast::SelectColumn::Expr { expr, .. } => {
                                if let crate::sql::ast::Expr::Column(col_ref) = expr {
                                    columns
                                        .iter()
                                        .position(|c| c.name().eq_ignore_ascii_case(col_ref.column))
                                        .and_then(|idx| updated_values.get(idx).cloned())
                                        .map(|v| vec![v])
                                        .unwrap_or_default()
                                } else {
                                    vec![]
                                }
                            }
                        })
                        .collect();
                    Row::new(row_values)
                })
                .collect()
        });

        let wal_enabled = self.shared.wal_enabled.load(Ordering::Acquire);
        if wal_enabled {
            self.ensure_wal()?;
        }

        let (txn_id, in_transaction) = {
            let active_txn = self.active_txn.lock();
            if let Some(ref txn) = *active_txn {
                (txn.txn_id, true)
            } else {
                (
                    self.shared
                        .txn_manager
                        .global_ts
                        .fetch_add(1, Ordering::SeqCst),
                    false,
                )
            }
        };

        let storage_arc = file_manager.table_data_mut(schema_name, table_name)?;
        let mut storage = storage_arc.write();

        with_btree_storage!(
            wal_enabled,
            &mut *storage,
            &self.shared.dirty_tracker,
            table_id as u32,
            root_page,
            |btree_mut: &mut crate::btree::BTree<_>| {
                for (key, _old_value, updated_values) in &rows_to_update {
                    let user_record = OwnedValue::build_record_from_values(updated_values, schema)?;
                    let record_data =
                        wrap_record_for_update(txn_id, &user_record, 0, 0, in_transaction);

                    if !btree_mut.update(key, &record_data)? {
                        btree_mut.delete(key)?;
                        btree_mut.insert(key, &record_data)?;
                    }
                }
                Ok::<_, eyre::Report>(())
            }
        );

        self.flush_wal_if_autocommit(file_manager, schema_name, table_name, table_id as u32)?;

        drop(file_manager_guard);
        {
            let mut active_txn = self.active_txn.lock();
            if let Some(ref mut txn) = *active_txn {
                for (key, old_value, _updated_values) in rows_to_update {
                    txn.add_write_entry_with_undo(
                        WriteEntry {
                            table_id: table_id as u32,
                            key,
                            page_id: 0,
                            offset: 0,
                            undo_page_id: None,
                            undo_offset: None,
                            is_insert: false,
                        },
                        old_value,
                    );
                }
            }
        }

        Ok(ExecuteResult::Update {
            rows_affected,
            returned: returned_rows,
        })
    }

    pub(crate) fn cartesian_product(tables: &[Vec<Vec<OwnedValue>>]) -> Vec<Vec<OwnedValue>> {
        if tables.is_empty() {
            return vec![vec![]];
        }

        let mut result: Vec<Vec<OwnedValue>> = vec![vec![]];

        for table_rows in tables {
            let mut new_result: Vec<Vec<OwnedValue>> = Vec::new();
            for existing in &result {
                for row in table_rows {
                    let mut combined = existing.clone();
                    combined.extend(row.clone());
                    new_result.push(combined);
                }
            }
            result = new_result;
        }

        result
    }

    pub(crate) fn eval_expr_with_row(
        &self,
        expr: &crate::sql::ast::Expr<'_>,
        row: &ExecutorRow<'_>,
        column_map: &[(String, usize)],
    ) -> Result<OwnedValue> {
        use crate::sql::ast::{BinaryOperator, Expr, UnaryOperator};

        match expr {
            Expr::Literal(_) => Self::eval_literal(expr),
            Expr::Column(col_ref) => {
                let col_idx = column_map
                    .iter()
                    .find(|(name, _)| name.eq_ignore_ascii_case(col_ref.column))
                    .map(|(_, idx)| *idx);

                if let Some(idx) = col_idx {
                    if let Some(val) = row.get(idx) {
                        Ok(OwnedValue::from(val))
                    } else {
                        Ok(OwnedValue::Null)
                    }
                } else {
                    bail!(
                        "column '{}' not found in UPDATE context",
                        col_ref.column
                    )
                }
            }
            Expr::BinaryOp { left, op, right } => {
                let left_val = self.eval_expr_with_row(left, row, column_map)?;
                let right_val = self.eval_expr_with_row(right, row, column_map)?;

                let arith_op = match op {
                    BinaryOperator::Plus => Some(ArithmeticOp::Plus),
                    BinaryOperator::Minus => Some(ArithmeticOp::Minus),
                    BinaryOperator::Multiply => Some(ArithmeticOp::Multiply),
                    BinaryOperator::Divide => Some(ArithmeticOp::Divide),
                    _ => None,
                };
                if let Some(aop) = arith_op {
                    OwnedValue::eval_arithmetic(&left_val, aop, &right_val).ok_or_else(|| {
                        eyre::eyre!("unsupported types or division by zero for {:?}", aop)
                    })
                } else {
                    match op {
                        BinaryOperator::Concat => match (&left_val, &right_val) {
                            (OwnedValue::Text(a), OwnedValue::Text(b)) => {
                                Ok(OwnedValue::Text(format!("{}{}", a, b)))
                            }
                            _ => bail!("unsupported types for concatenation"),
                        },
                        _ => bail!("unsupported binary operator in UPDATE...FROM SET expression"),
                    }
                }
            }
            Expr::UnaryOp { op, expr: inner } => {
                let inner_val = self.eval_expr_with_row(inner, row, column_map)?;
                match (op, inner_val) {
                    (UnaryOperator::Minus, OwnedValue::Int(i)) => Ok(OwnedValue::Int(-i)),
                    (UnaryOperator::Minus, OwnedValue::Float(f)) => Ok(OwnedValue::Float(-f)),
                    (UnaryOperator::Plus, val) => Ok(val),
                    (UnaryOperator::Not, OwnedValue::Bool(b)) => Ok(OwnedValue::Bool(!b)),
                    _ => bail!("unsupported unary operation"),
                }
            }
            _ => Self::eval_literal(expr),
        }
    }

    pub(crate) fn execute_update_cached(
        &self,
        cached: &crate::database::prepared::CachedUpdatePlan,
        params: &[OwnedValue],
    ) -> Result<ExecuteResult> {
        self.ensure_file_manager()?;

        let wal_enabled = self
            .shared
            .wal_enabled
            .load(std::sync::atomic::Ordering::Acquire);
        if wal_enabled {
            self.ensure_wal()?;
        }

        if cached.is_simple_pk_update && cached.assignment_indices.len() + 1 == params.len() {
            return self.execute_update_param_only(cached, params, wal_enabled);
        }

        let arena = bumpalo::Bump::new();
        let mut parser = crate::sql::parser::Parser::new(&cached.original_sql, &arena);

        let stmt = parser
            .parse_statement()
            .wrap_err("failed to re-parse cached UPDATE statement")?;
        parser.expect_end_of_statement()?;

        if let crate::sql::ast::Statement::Update(update) = stmt {
            self.execute_update(update, params, &arena)
        } else {
            bail!("cached plan produced non-UPDATE statement")
        }
    }

    fn execute_update_param_only(
        &self,
        cached: &crate::database::prepared::CachedUpdatePlan,
        params: &[OwnedValue],
        wal_enabled: bool,
    ) -> Result<ExecuteResult> {
        let storage_arc = {
            let storage_weak = cached.storage.borrow();
            storage_weak
                .as_ref()
                .and_then(|weak| weak.upgrade())
                .ok_or_else(|| eyre::eyre!("cached plan storage no longer valid"))?
        };

        let root_page = cached.root_page.get();
        if root_page == 0 {
            let storage = storage_arc.write();
            let new_root = {
                use crate::storage::TableFileHeader;
                let page0 = storage.page(0)?;
                let header = TableFileHeader::from_bytes(page0)?;
                header.root_page()
            };
            drop(storage);
            cached.root_page.set(new_root);
        }
        let root_page = cached.root_page.get();

        let where_param_value = params
            .last()
            .ok_or_else(|| eyre::eyre!("missing WHERE parameter"))?;

        let pk_col_idx = cached
            .pk_column_index
            .ok_or_else(|| eyre::eyre!("PK column index not cached"))?;

        self.ensure_catalog()?;
        let catalog_guard = self.shared.catalog.read();
        let catalog = catalog_guard.as_ref().unwrap();
        let table_def = catalog.resolve_table(&cached.table_name)?;
        let pk_col_name = table_def.columns()[pk_col_idx].name();
        let pk_index_name = format!("{}_pkey", pk_col_name);
        drop(catalog_guard);

        let mut file_manager_guard = self.shared.file_manager.write();
        let file_manager = file_manager_guard.as_mut().unwrap();

        let target_key =
            if file_manager.index_exists(&cached.schema_name, &cached.table_name, &pk_index_name) {
                if let Ok(index_storage_arc) = file_manager.index_data_mut(
                    &cached.schema_name,
                    &cached.table_name,
                    &pk_index_name,
                ) {
                    let mut index_storage = index_storage_arc.write();
                    let index_root_page = {
                        use crate::storage::IndexFileHeader;
                        let page0 = index_storage.page(0)?;
                        let header = IndexFileHeader::from_bytes(page0)?;
                        header.root_page()
                    };
                    let index_btree = BTree::new(&mut *index_storage, index_root_page)?;
                    let mut index_key = Vec::new();
                    Self::encode_value_as_key(where_param_value, &mut index_key);
                    if let Some(handle) = index_btree.search(&index_key)? {
                        index_btree.get_value(&handle)?.to_vec()
                    } else {
                        return Ok(ExecuteResult::Update {
                            rows_affected: 0,
                            returned: None,
                        });
                    }
                } else {
                    bail!("failed to get index storage")
                }
            } else {
                bail!("PK index not found for fast path")
            };

        drop(file_manager_guard);

        let mut storage = storage_arc.write();
        let decoder = crate::sql::decoder::SimpleDecoder::new(cached.column_types.clone());
        let btree = BTree::new(&mut *storage, root_page)?;

        let cursor = btree.cursor_seek(&target_key)?;

        if !cursor.valid() || cursor.key()? != target_key.as_slice() {
            return Ok(ExecuteResult::Update {
                rows_affected: 0,
                returned: None,
            });
        }

        let key = cursor.key()?;
        let value = cursor.value()?;
        let values = decoder.decode(key, value)?;
        let mut row_values: Vec<OwnedValue> = values.iter().map(OwnedValue::from).collect();

        for (i, (col_idx, _)) in cached.assignment_indices.iter().enumerate() {
            row_values[*col_idx] = params[i].clone();
        }

        drop(storage);

        let mut storage = storage_arc.write();

        with_btree_storage!(
            wal_enabled,
            &mut *storage,
            &self.shared.dirty_tracker,
            cached.table_id as u32,
            root_page,
            |btree_mut: &mut crate::btree::BTree<_>| {
                let record_data =
                    OwnedValue::build_record_from_values(&row_values, &cached.record_schema)?;

                if !btree_mut.update(&target_key, &record_data)? {
                    btree_mut.delete(&target_key)?;
                    btree_mut.insert(&target_key, &record_data)?;
                }
                Ok::<_, eyre::Report>(())
            }
        );

        Ok(ExecuteResult::Update {
            rows_affected: 1,
            returned: None,
        })
    }

    pub(crate) fn extract_tables_from_clause<'a>(
        from_clause: crate::sql::ast::FromClause<'a>,
        catalog: &crate::schema::Catalog,
        tables: &mut Vec<(
            String,
            String,
            Option<&'a str>,
            Vec<crate::schema::table::ColumnDef>,
        )>,
    ) -> Result<()> {
        use crate::sql::ast::FromClause;

        match from_clause {
            FromClause::Table(table_ref) => {
                let schema = table_ref.schema.unwrap_or(DEFAULT_SCHEMA);
                let table_name = table_ref.name;
                let alias = table_ref.alias;
                let table_def = catalog.resolve_table_in_schema(table_ref.schema, table_name)?;
                let columns = table_def.columns().to_vec();
                tables.push((schema.to_string(), table_name.to_string(), alias, columns));
            }
            FromClause::Join(join_clause) => {
                Self::extract_tables_from_clause(*join_clause.left, catalog, tables)?;
                Self::extract_tables_from_clause(*join_clause.right, catalog, tables)?;
            }
            FromClause::Subquery { .. } => {
                bail!("UPDATE...FROM does not support subqueries in FROM clause")
            }
            FromClause::Lateral { .. } => {
                bail!("UPDATE...FROM does not support LATERAL in FROM clause")
            }
        }
        Ok(())
    }
}
