//! # Query Execution Module
//!
//! This module contains the query execution engine components for TurDB.
//! It provides functions for executing physical query plans, including:
//!
//! - Plan traversal and analysis helpers
//! - Table and index scan execution
//! - Join execution (nested loop, hash, semi, anti)
//! - Subquery execution
//! - Set operations (UNION, INTERSECT, EXCEPT)
//!
//! ## Module Structure
//!
//! - `helpers`: Plan traversal functions for finding operators and analyzing plans
//!
//! ## Usage
//!
//! The query module is used internally by the Database struct to execute
//! SELECT queries. The `query_with_columns` method in database.rs orchestrates
//! the execution by calling into these modules.

mod helpers;
mod set_ops;

pub use helpers::{
    build_column_map_with_alias, build_simple_column_map, compare_owned_values, find_limit,
    find_nested_subquery, find_plan_source, find_projections, find_sort_exec, find_table_scan,
    has_aggregate, has_filter, has_order_by_expression, has_window, hash_owned_value_normalized,
    is_simple_count_star, materialize_table_rows, materialize_table_rows_with_tracker,
    owned_values_equal_with_coercion, PlanSource,
};
