//! # System Functions Module
//!
//! This module provides system and database information functions:
//!
//! ## Database Info
//! - `VERSION()` - Returns the database version string
//! - `DATABASE()` - Returns the current database name (placeholder)
//!
//! ## Type Conversion
//! - `TYPEOF(expr)` - Returns the type name of the expression
//!
//! ## Utility
//! - `COALESCE(a, b, ...)` - Returns first non-NULL argument

use crate::types::Value;
use std::borrow::Cow;

const TURDB_VERSION: &str = "TurDB 0.1.0";

/// Evaluates system functions by name.
pub fn eval_system_function<'a>(name: &str, args: &[Option<Value<'a>>]) -> Option<Value<'a>> {
    match name {
        "VERSION" => Some(Value::Text(Cow::Borrowed(TURDB_VERSION))),
        "DATABASE" | "CURRENT_DATABASE" => Some(Value::Text(Cow::Borrowed("turdb"))),
        "TYPEOF" => eval_typeof(args),
        _ => None,
    }
}

fn eval_typeof<'a>(args: &[Option<Value<'a>>]) -> Option<Value<'a>> {
    let val = args.first()?.as_ref()?;
    let type_name = match val {
        Value::Null => "null",
        Value::Int(_) => "integer",
        Value::Float(_) => "real",
        Value::Text(_) => "text",
        Value::Blob(_) => "blob",
        Value::Vector(_) => "vector",
        Value::Uuid(_) => "uuid",
        Value::MacAddr(_) => "macaddr",
        Value::Inet4(_) => "inet4",
        Value::Inet6(_) => "inet6",
        Value::Jsonb(_) => "jsonb",
        Value::TimestampTz { .. } => "timestamptz",
        Value::Interval { .. } => "interval",
        Value::Point { .. } => "point",
        Value::GeoBox { .. } => "box",
        Value::Circle { .. } => "circle",
        Value::Enum { .. } => "enum",
        Value::Decimal { .. } => "decimal",
    };
    Some(Value::Text(Cow::Borrowed(type_name)))
}
