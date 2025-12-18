//! # SQL Functions Module
//!
//! This module provides built-in SQL functions organized by category.
//!
//! ## Module Structure
//!
//! - `string`: String manipulation (UPPER, LOWER, CONCAT, SUBSTR, etc.)
//! - `numeric`: Math functions (ABS, ROUND, SIN, COS, LOG, etc.)
//! - `datetime`: Date/time functions (NOW, DATE_ADD, YEAR, MONTH, etc.)
//! - `system`: System info and control flow (VERSION, IF, COALESCE, etc.)
//!
//! ## Design Philosophy
//!
//! Functions are dispatched by name through `eval_function()` which routes
//! to the appropriate category module. Each module handles its own function
//! implementations and helper utilities.
//!
//! ## Adding New Functions
//!
//! 1. Add the function implementation in the appropriate category module
//! 2. Register the function name in that module's dispatch function
//! 3. The main dispatcher will automatically try all modules

pub mod datetime;
pub mod numeric;
pub mod string;
pub mod system;

use crate::types::Value;

/// Evaluates a SQL function by name with the given arguments.
/// Returns None if the function is unknown or arguments are invalid.
pub fn eval_function<'a>(name: &str, args: &[Option<Value<'a>>]) -> Option<Value<'a>> {
    let upper_name = name.to_uppercase();

    string::eval_string_function(&upper_name, args)
        .or_else(|| numeric::eval_numeric_function(&upper_name, args))
        .or_else(|| datetime::eval_datetime_function(&upper_name, args))
        .or_else(|| system::eval_system_function(&upper_name, args))
}
