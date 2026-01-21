//! # Type Conversion Module
//!
//! This module provides type conversion utilities for the database, handling
//! the mapping between SQL AST types, internal storage types, and runtime values.
//!
//! ## Purpose
//!
//! Type conversion is central to database operations. When parsing SQL statements,
//! we need to convert AST type representations to internal storage types. When
//! evaluating expressions, we need to convert literal values to typed OwnedValues.
//! This module centralizes all such conversions.
//!
//! ## Architecture
//!
//! ```text
//! ┌──────────────────────────────────────────────────────────────────────────┐
//! │                        Type Conversion Flow                              │
//! ├──────────────────────────────────────────────────────────────────────────┤
//! │                                                                          │
//! │   SQL AST Types                    Internal Types                        │
//! │   ──────────────                   ──────────────                        │
//! │   crate::sql::ast::DataType  ───►  crate::records::types::DataType       │
//! │        INTEGER                          Int4                             │
//! │        BIGINT                           Int8                             │
//! │        VARCHAR(n)                       Varchar                          │
//! │        JSONB                            Jsonb                            │
//! │        ...                              ...                              │
//! │                                                                          │
//! │   SQL Literals                     Runtime Values                        │
//! │   ────────────                     ──────────────                        │
//! │   crate::sql::ast::Literal   ───►  OwnedValue                            │
//! │        Integer("42")                    Int(42)                          │
//! │        String("hello")                  Text("hello")                    │
//! │        Boolean(true)                    Bool(true)                       │
//! │        ...                              ...                              │
//! │                                                                          │
//! │   JSON Strings                     JSONB Binary                          │
//! │   ────────────                     ────────────                          │
//! │   '{"key": "value"}'         ───►  Vec<u8> (JSONB format)                │
//! │                                                                          │
//! └──────────────────────────────────────────────────────────────────────────┘
//! ```
//!
//! ## Key Functions
//!
//! - `convert_data_type`: SQL AST type → Internal storage type
//! - `extract_type_length`: Extract length parameter from VARCHAR/CHAR
//! - `eval_literal`: Evaluate AST literal expression → OwnedValue
//! - `eval_literal_with_type`: Evaluate literal with target type hint
//! - `parse_json_string`: Parse JSON string → JSONB binary format
//! - `generate_row_key`: Row ID → Big-endian 8-byte key
//!
//! ## Usage Patterns
//!
//! ### Creating a table column from SQL type
//!
//! ```ignore
//! let sql_type = SqlType::Varchar(Some(255));
//! let internal_type = Database::convert_data_type(&sql_type);
//! let length = Database::extract_type_length(&sql_type);
//! ```
//!
//! ### Evaluating a literal for INSERT
//!
//! ```ignore
//! let expr = Expr::Literal(Literal::Integer("42"));
//! let value = Database::eval_literal(&expr)?;
//! // value = OwnedValue::Int(42)
//! ```
//!
//! ### Parsing JSON for JSONB column
//!
//! ```ignore
//! let json_str = r#"{"name": "Alice", "age": 30}"#;
//! let jsonb_value = Database::parse_json_string(json_str)?;
//! // jsonb_value = OwnedValue::Jsonb(<binary data>)
//! ```
//!
//! ## Performance Characteristics
//!
//! - Type mapping: O(1) - simple pattern matching
//! - Literal evaluation: O(1) for most types, O(n) for strings
//! - JSON parsing: O(n) where n is JSON size, with nested object overhead
//!
//! ## Thread Safety
//!
//! All functions in this module are pure (no shared mutable state) and can
//! be called safely from any thread.
//!
//! ## JSON Parsing Details
//!
//! The JSON parser is a simple recursive descent parser that handles:
//! - Null, boolean, number, string primitives
//! - Nested objects and arrays
//! - Unicode escape sequences (\uXXXX)
//! - Standard escape characters (\n, \r, \t, \\, \", \/)
//!
//! The parser converts JSON text to the internal JSONB binary format for
//! efficient storage and querying.

use crate::parsing::{
    parse_binary_blob, parse_date, parse_hex_blob, parse_interval, parse_time, parse_timestamp,
    parse_uuid, parse_vector,
};
use crate::types::{ArithmeticOp, DataType, OwnedValue, Value};
use eyre::{bail, Result, WrapErr};
use std::borrow::Cow;

use super::Database;

pub(crate) fn convert_value_with_type(val: &Value<'_>, col_type: DataType) -> OwnedValue {
    match (val, col_type) {
        (Value::Int(i), DataType::Bool) => OwnedValue::Bool(*i != 0),
        (Value::Int(i), DataType::Date) => OwnedValue::Date(*i as i32),
        (Value::Int(i), DataType::Time) => OwnedValue::Time(*i),
        (Value::Int(i), DataType::Timestamp) => OwnedValue::Timestamp(*i),
        _ => OwnedValue::from(val),
    }
}

impl Database {
    pub(crate) fn convert_data_type(
        sql_type: &crate::sql::ast::DataType,
    ) -> crate::records::types::DataType {
        use crate::records::types::DataType;
        use crate::sql::ast::DataType as SqlType;

        match sql_type {
            SqlType::Integer => DataType::Int4,
            SqlType::BigInt => DataType::Int8,
            SqlType::SmallInt => DataType::Int2,
            SqlType::TinyInt => DataType::Int2,
            SqlType::Serial => DataType::Int4,
            SqlType::BigSerial => DataType::Int8,
            SqlType::SmallSerial => DataType::Int2,
            SqlType::Real | SqlType::DoublePrecision => DataType::Float8,
            SqlType::Decimal(_, _) | SqlType::Numeric(_, _) => DataType::Float8,
            SqlType::Varchar(_) => DataType::Varchar,
            SqlType::Text => DataType::Text,
            SqlType::Char(_) => DataType::Char,
            SqlType::Blob => DataType::Blob,
            SqlType::Boolean => DataType::Bool,
            SqlType::Date => DataType::Date,
            SqlType::Time => DataType::Time,
            SqlType::Timestamp => DataType::Timestamp,
            SqlType::TimestampTz => DataType::TimestampTz,
            SqlType::Uuid => DataType::Uuid,
            SqlType::Json | SqlType::Jsonb => DataType::Jsonb,
            SqlType::Vector(_) => DataType::Vector,
            SqlType::Array(_) => DataType::Array,
            SqlType::Interval => DataType::Interval,
            SqlType::Point => DataType::Point,
            SqlType::Box => DataType::Box,
            SqlType::Circle => DataType::Circle,
            SqlType::MacAddr => DataType::MacAddr,
            SqlType::Inet => DataType::Inet6,
            SqlType::Int4Range => DataType::Int4Range,
            SqlType::Int8Range => DataType::Int8Range,
            SqlType::DateRange => DataType::DateRange,
            SqlType::TsRange => DataType::TimestampRange,
            _ => DataType::Text,
        }
    }

    pub(crate) fn extract_type_length(sql_type: &crate::sql::ast::DataType) -> Option<u32> {
        use crate::sql::ast::DataType as SqlType;

        match sql_type {
            SqlType::Varchar(len) => *len,
            SqlType::Char(len) => *len,
            _ => None,
        }
    }

    pub(crate) fn expr_to_default_string(expr: &crate::sql::ast::Expr<'_>) -> Option<String> {
        use crate::sql::ast::Expr;

        match expr {
            Expr::Literal(lit) => match lit {
                crate::sql::ast::Literal::Integer(n) => Some(n.to_string()),
                crate::sql::ast::Literal::Float(f) => Some(f.to_string()),
                crate::sql::ast::Literal::String(s) => Some(s.to_string()),
                crate::sql::ast::Literal::Boolean(b) => Some(b.to_string()),
                crate::sql::ast::Literal::Null => None,
                _ => None,
            },
            Expr::Function(func) => {
                let name = func.name.name.to_uppercase();
                match name.as_str() {
                    "CURRENT_TIMESTAMP" | "NOW" | "CURRENT_DATE" | "CURRENT_TIME" | "LOCALTIME"
                    | "LOCALTIMESTAMP" => Some(name),
                    _ => None,
                }
            }
            Expr::Column(col) => {
                let name = col.column.to_uppercase();
                match name.as_str() {
                    "CURRENT_TIMESTAMP" | "NOW" | "CURRENT_DATE" | "CURRENT_TIME" | "LOCALTIME"
                    | "LOCALTIMESTAMP" => Some(name),
                    _ => None,
                }
            }
            _ => None,
        }
    }

    pub(crate) fn expr_to_string(expr: &crate::sql::ast::Expr<'_>) -> Option<String> {
        use crate::sql::ast::{BinaryOperator, Expr, UnaryOperator};

        match expr {
            Expr::BinaryOp { left, op, right } => {
                let left_str = Self::expr_to_string(left)?;
                let right_str = Self::expr_to_string(right)?;
                let op_str = match op {
                    BinaryOperator::Plus => "+",
                    BinaryOperator::Minus => "-",
                    BinaryOperator::Multiply => "*",
                    BinaryOperator::Divide => "/",
                    BinaryOperator::Modulo => "%",
                    BinaryOperator::Eq => "=",
                    BinaryOperator::NotEq => "!=",
                    BinaryOperator::Lt => "<",
                    BinaryOperator::LtEq => "<=",
                    BinaryOperator::Gt => ">",
                    BinaryOperator::GtEq => ">=",
                    BinaryOperator::And => "AND",
                    BinaryOperator::Or => "OR",
                    _ => "?",
                };
                Some(format!("{} {} {}", left_str, op_str, right_str))
            }
            Expr::UnaryOp { op, expr: inner } => {
                let inner_str = Self::expr_to_string(inner)?;
                let op_str = match op {
                    UnaryOperator::Minus => "-",
                    UnaryOperator::Plus => "+",
                    UnaryOperator::Not => "NOT ",
                    UnaryOperator::BitwiseNot => "~",
                };
                Some(format!("{}{}", op_str, inner_str))
            }
            Expr::Column(col_ref) => Some(col_ref.column.to_string()),
            Expr::Literal(lit) => match lit {
                crate::sql::ast::Literal::Integer(n) => Some(n.to_string()),
                crate::sql::ast::Literal::Float(f) => Some(f.to_string()),
                crate::sql::ast::Literal::String(s) => Some(format!("'{}'", s)),
                crate::sql::ast::Literal::Boolean(b) => Some(b.to_string()),
                _ => None,
            },
            _ => None,
        }
    }

    pub(crate) fn convert_referential_action(
        action: Option<crate::sql::ast::ReferentialAction>,
    ) -> Option<crate::schema::ReferentialAction> {
        action.map(|a| match a {
            crate::sql::ast::ReferentialAction::Cascade => {
                crate::schema::ReferentialAction::Cascade
            }
            crate::sql::ast::ReferentialAction::Restrict => {
                crate::schema::ReferentialAction::Restrict
            }
            crate::sql::ast::ReferentialAction::NoAction => {
                crate::schema::ReferentialAction::NoAction
            }
            crate::sql::ast::ReferentialAction::SetNull => {
                crate::schema::ReferentialAction::SetNull
            }
            crate::sql::ast::ReferentialAction::SetDefault => {
                crate::schema::ReferentialAction::SetDefault
            }
        })
    }

    pub(crate) fn eval_literal(expr: &crate::sql::ast::Expr<'_>) -> Result<OwnedValue> {
        Self::eval_literal_with_type(expr, None)
    }

    pub(crate) fn eval_literal_with_type(
        expr: &crate::sql::ast::Expr<'_>,
        target_type: Option<&crate::records::types::DataType>,
    ) -> Result<OwnedValue> {
        use crate::records::types::DataType;
        use crate::sql::ast::{Expr, FunctionArgs, Literal, UnaryOperator};

        match expr {
            Expr::Literal(lit) => match lit {
                Literal::Null => Ok(OwnedValue::Null),
                Literal::Integer(s) => {
                    let i: i64 = s
                        .parse()
                        .wrap_err_with(|| format!("failed to parse integer: {}", s))?;
                    Ok(OwnedValue::Int(i))
                }
                Literal::Float(s) => {
                    let f: f64 = s
                        .parse()
                        .wrap_err_with(|| format!("failed to parse float: {}", s))?;
                    Ok(OwnedValue::Float(f))
                }
                Literal::String(s) => match target_type {
                    Some(DataType::Uuid) => parse_uuid(s),
                    Some(DataType::Jsonb) => Self::parse_json_string(s),
                    Some(DataType::Vector) => parse_vector(s),
                    Some(DataType::Interval) => parse_interval(s),
                    Some(DataType::Timestamp) => parse_timestamp(s),
                    Some(DataType::TimestampTz) => parse_timestamp(s),
                    Some(DataType::Date) => parse_date(s),
                    Some(DataType::Time) => parse_time(s),
                    _ => Ok(OwnedValue::Text(s.to_string())),
                },
                Literal::Boolean(b) => Ok(OwnedValue::Bool(*b)),
                Literal::HexNumber(s) => parse_hex_blob(s),
                Literal::BinaryNumber(s) => parse_binary_blob(s),
            },
            Expr::UnaryOp { op, expr: inner } => {
                let inner_val = Self::eval_literal_with_type(inner, target_type)?;
                match (op, inner_val) {
                    (UnaryOperator::Minus, OwnedValue::Int(i)) => Ok(OwnedValue::Int(-i)),
                    (UnaryOperator::Minus, OwnedValue::Float(f)) => Ok(OwnedValue::Float(-f)),
                    (UnaryOperator::Plus, val) => Ok(val),
                    (UnaryOperator::Not, OwnedValue::Bool(b)) => Ok(OwnedValue::Bool(!b)),
                    _ => bail!("unsupported unary operation"),
                }
            }
            Expr::Function(func) => {
                let name = func.name.name.to_uppercase();
                let is_nullary = match &func.args {
                    FunctionArgs::None => true,
                    FunctionArgs::Args(args) => args.is_empty(),
                    FunctionArgs::Star => false,
                };

                if is_nullary {
                    if let Some(result) = crate::sql::functions::eval_function(&name, &[]) {
                        if let crate::types::Value::Text(s) = &result {
                            return match target_type {
                                Some(DataType::Timestamp) | Some(DataType::TimestampTz) => {
                                    parse_timestamp(s)
                                }
                                Some(DataType::Date) => parse_date(s),
                                Some(DataType::Time) => parse_time(s),
                                _ => Ok(OwnedValue::from(&result)),
                            };
                        }
                        return Ok(OwnedValue::from(&result));
                    }
                }

                bail!("unsupported function in literal context: {}", name)
            }
            _ => bail!("expected literal expression, got {:?}", expr),
        }
    }

    /// Evaluates an expression to an OwnedValue, supporting parameter placeholders.
    ///
    /// When `params` is provided, `Expr::Parameter` nodes are resolved from it.
    /// This enables true prepared statement execution without SQL string reconstruction.
    ///
    /// # Arguments
    /// * `expr` - The expression to evaluate
    /// * `target_type` - Optional target column type for type-aware parsing
    /// * `params` - Optional slice of bound parameter values
    /// * `param_idx` - Mutable index for tracking anonymous (`?`) parameters
    pub(crate) fn eval_expr_with_params(
        expr: &crate::sql::ast::Expr<'_>,
        target_type: Option<&crate::records::types::DataType>,
        params: Option<&[OwnedValue]>,
        param_idx: &mut usize,
    ) -> Result<OwnedValue> {
        use crate::sql::ast::{Expr, ParameterRef};

        if let Expr::Parameter(param_ref) = expr {
            if let Some(params) = params {
                let idx = match param_ref {
                    ParameterRef::Anonymous => {
                        let i = *param_idx;
                        *param_idx += 1;
                        i
                    }
                    ParameterRef::Positional(n) => (*n as usize).saturating_sub(1),
                    ParameterRef::Named(_) => {
                        let i = *param_idx;
                        *param_idx += 1;
                        i
                    }
                };

                if idx >= params.len() {
                    bail!(
                        "parameter index {} out of range (only {} parameters bound)",
                        idx + 1,
                        params.len()
                    );
                }

                return Ok(params[idx].clone());
            } else {
                bail!("parameter placeholder found but no parameters were bound");
            }
        }

        Self::eval_literal_with_type(expr, target_type)
    }

    pub(crate) fn eval_expr_with_params_and_subqueries<'a>(
        expr: &crate::sql::ast::Expr<'_>,
        target_type: Option<&crate::records::types::DataType>,
        params: Option<&'a [OwnedValue]>,
        param_idx: &mut usize,
        scalar_subquery_results: &'a crate::sql::context::ScalarSubqueryResults,
    ) -> Result<Cow<'a, OwnedValue>> {
        use crate::sql::ast::{Expr, ParameterRef};

        match expr {
            Expr::Parameter(param_ref) => {
                if let Some(params) = params {
                    let idx = match param_ref {
                        ParameterRef::Anonymous => {
                            let i = *param_idx;
                            *param_idx += 1;
                            i
                        }
                        ParameterRef::Positional(n) => (*n as usize).saturating_sub(1),
                        ParameterRef::Named(_) => {
                            let i = *param_idx;
                            *param_idx += 1;
                            i
                        }
                    };

                    if idx >= params.len() {
                        bail!(
                            "parameter index {} out of range (only {} parameters bound)",
                            idx + 1,
                            params.len()
                        );
                    }

                    Ok(Cow::Borrowed(&params[idx]))
                } else {
                    bail!("parameter placeholder found but no parameters were bound")
                }
            }
            Expr::Subquery(subq) => {
                let key = std::ptr::from_ref(*subq) as usize;
                scalar_subquery_results
                    .get(&key)
                    .map(Cow::Borrowed)
                    .ok_or_else(|| eyre::eyre!("scalar subquery result not found for key 0x{:x}", key))
            }
            Expr::BinaryOp { left, op, right } => {
                let left_val = Self::eval_expr_with_params_and_subqueries(
                    left,
                    target_type,
                    params,
                    param_idx,
                    scalar_subquery_results,
                )?;
                let right_val = Self::eval_expr_with_params_and_subqueries(
                    right,
                    target_type,
                    params,
                    param_idx,
                    scalar_subquery_results,
                )?;

                use crate::sql::ast::BinaryOperator;
                let arith_op = match op {
                    BinaryOperator::Plus => Some(ArithmeticOp::Plus),
                    BinaryOperator::Minus => Some(ArithmeticOp::Minus),
                    BinaryOperator::Multiply => Some(ArithmeticOp::Multiply),
                    BinaryOperator::Divide => Some(ArithmeticOp::Divide),
                    _ => None,
                };
                if let Some(aop) = arith_op {
                    OwnedValue::eval_arithmetic(left_val.as_ref(), aop, right_val.as_ref())
                        .map(Cow::Owned)
                        .ok_or_else(|| {
                            eyre::eyre!(
                                "unsupported types or division by zero for {:?} in UPDATE SET",
                                aop
                            )
                        })
                } else {
                    Self::eval_literal_with_type(expr, target_type).map(Cow::Owned)
                }
            }
            _ => Self::eval_literal_with_type(expr, target_type).map(Cow::Owned),
        }
    }

    pub(crate) fn parse_json_string(s: &str) -> Result<OwnedValue> {
        let value = Self::parse_json_to_value(s.trim())?;
        let bytes = Self::jsonb_value_to_bytes(&value);
        Ok(OwnedValue::Jsonb(bytes))
    }

    pub(crate) fn parse_json_to_value(s: &str) -> Result<crate::records::jsonb::JsonbBuilderValue> {
        use crate::records::jsonb::JsonbBuilderValue;
        let s = s.trim();

        if s == "null" {
            Ok(JsonbBuilderValue::Null)
        } else if s == "true" {
            Ok(JsonbBuilderValue::Bool(true))
        } else if s == "false" {
            Ok(JsonbBuilderValue::Bool(false))
        } else if s.starts_with('"') && s.ends_with('"') {
            let inner = &s[1..s.len() - 1];
            let unescaped = Self::unescape_json_string(inner)?;
            Ok(JsonbBuilderValue::String(unescaped))
        } else if s.starts_with('{') && s.ends_with('}') {
            Self::parse_json_object_to_value(&s[1..s.len() - 1])
        } else if s.starts_with('[') && s.ends_with(']') {
            Self::parse_json_array_to_value(&s[1..s.len() - 1])
        } else if let Ok(n) = s.parse::<f64>() {
            Ok(JsonbBuilderValue::Number(n))
        } else {
            bail!("invalid JSON value: '{}'", s)
        }
    }

    pub(crate) fn jsonb_value_to_bytes(
        value: &crate::records::jsonb::JsonbBuilderValue,
    ) -> Vec<u8> {
        use crate::records::jsonb::{JsonbBuilder, JsonbBuilderValue};

        fn build_from_value(value: &JsonbBuilderValue) -> JsonbBuilder {
            match value {
                JsonbBuilderValue::Null => JsonbBuilder::new_null(),
                JsonbBuilderValue::Bool(b) => JsonbBuilder::new_bool(*b),
                JsonbBuilderValue::Number(n) => JsonbBuilder::new_number(*n),
                JsonbBuilderValue::String(s) => JsonbBuilder::new_string(s.clone()),
                JsonbBuilderValue::Array(elements) => {
                    let mut builder = JsonbBuilder::new_array();
                    for elem in elements {
                        builder.push(elem.clone());
                    }
                    builder
                }
                JsonbBuilderValue::Object(entries) => {
                    let mut builder = JsonbBuilder::new_object();
                    for (key, val) in entries {
                        builder.set(key.clone(), val.clone());
                    }
                    builder
                }
            }
        }

        build_from_value(value).build()
    }

    pub(crate) fn unescape_json_string(s: &str) -> Result<String> {
        let mut result = String::with_capacity(s.len());
        let mut chars = s.chars().peekable();

        while let Some(c) = chars.next() {
            if c == '\\' {
                match chars.next() {
                    Some('n') => result.push('\n'),
                    Some('r') => result.push('\r'),
                    Some('t') => result.push('\t'),
                    Some('\\') => result.push('\\'),
                    Some('"') => result.push('"'),
                    Some('/') => result.push('/'),
                    Some('u') => {
                        let hex: String = chars.by_ref().take(4).collect();
                        if hex.len() != 4 {
                            bail!("invalid unicode escape in JSON string");
                        }
                        let cp =
                            u32::from_str_radix(&hex, 16).wrap_err("invalid unicode escape")?;
                        if let Some(ch) = char::from_u32(cp) {
                            result.push(ch);
                        } else {
                            bail!("invalid unicode codepoint: {}", cp);
                        }
                    }
                    Some(other) => bail!("invalid escape sequence: \\{}", other),
                    None => bail!("unexpected end of string after escape"),
                }
            } else {
                result.push(c);
            }
        }

        Ok(result)
    }

    pub(crate) fn parse_json_object_to_value(
        s: &str,
    ) -> Result<crate::records::jsonb::JsonbBuilderValue> {
        use crate::records::jsonb::JsonbBuilderValue;
        let s = s.trim();

        if s.is_empty() {
            return Ok(JsonbBuilderValue::Object(Vec::new()));
        }

        let mut entries = Vec::new();
        let mut depth = 0;
        let mut in_string = false;
        let mut escape_next = false;
        let mut current_start = 0;

        for (i, c) in s.char_indices() {
            if escape_next {
                escape_next = false;
                continue;
            }

            match c {
                '\\' if in_string => escape_next = true,
                '"' => in_string = !in_string,
                '{' | '[' if !in_string => depth += 1,
                '}' | ']' if !in_string => depth -= 1,
                ',' if !in_string && depth == 0 => {
                    let (key, value) = Self::parse_json_kv_pair_to_value(&s[current_start..i])?;
                    entries.push((key, value));
                    current_start = i + 1;
                }
                _ => {}
            }
        }

        if current_start < s.len() {
            let (key, value) = Self::parse_json_kv_pair_to_value(&s[current_start..])?;
            entries.push((key, value));
        }

        Ok(JsonbBuilderValue::Object(entries))
    }

    pub(crate) fn parse_json_kv_pair_to_value(
        s: &str,
    ) -> Result<(String, crate::records::jsonb::JsonbBuilderValue)> {
        let s = s.trim();
        let colon_pos = Self::find_json_colon(s)?;

        let key_part = s[..colon_pos].trim();
        let value_part = s[colon_pos + 1..].trim();

        if !key_part.starts_with('"') || !key_part.ends_with('"') {
            bail!("JSON object key must be a string: '{}'", key_part);
        }

        let key = Self::unescape_json_string(&key_part[1..key_part.len() - 1])?;
        let value = Self::parse_json_to_value(value_part)?;

        Ok((key, value))
    }

    pub(crate) fn find_json_colon(s: &str) -> Result<usize> {
        let mut in_string = false;
        let mut escape_next = false;

        for (i, c) in s.char_indices() {
            if escape_next {
                escape_next = false;
                continue;
            }

            match c {
                '\\' if in_string => escape_next = true,
                '"' => in_string = !in_string,
                ':' if !in_string => return Ok(i),
                _ => {}
            }
        }

        bail!("no colon found in JSON key-value pair: '{}'", s)
    }

    pub(crate) fn parse_json_array_to_value(
        s: &str,
    ) -> Result<crate::records::jsonb::JsonbBuilderValue> {
        use crate::records::jsonb::JsonbBuilderValue;
        let s = s.trim();

        if s.is_empty() {
            return Ok(JsonbBuilderValue::Array(Vec::new()));
        }

        let mut elements = Vec::new();
        let mut depth = 0;
        let mut in_string = false;
        let mut escape_next = false;
        let mut current_start = 0;

        for (i, c) in s.char_indices() {
            if escape_next {
                escape_next = false;
                continue;
            }

            match c {
                '\\' if in_string => escape_next = true,
                '"' => in_string = !in_string,
                '{' | '[' if !in_string => depth += 1,
                '}' | ']' if !in_string => depth -= 1,
                ',' if !in_string && depth == 0 => {
                    elements.push(Self::parse_json_to_value(&s[current_start..i])?);
                    current_start = i + 1;
                }
                _ => {}
            }
        }

        if current_start < s.len() {
            elements.push(Self::parse_json_to_value(&s[current_start..])?);
        }

        Ok(JsonbBuilderValue::Array(elements))
    }

    pub(crate) fn generate_row_key(row_id: u64) -> [u8; 8] {
        row_id.to_be_bytes()
    }
}
