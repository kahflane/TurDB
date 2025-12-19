//! # System/Advanced Functions Module
//!
//! This module provides system information and control flow functions:
//!
//! ## Database Info
//! - `VERSION()` - Database version string
//! - `DATABASE()` / `CURRENT_DATABASE()` - Current database name
//! - `USER()` / `CURRENT_USER()` / `SESSION_USER()` / `SYSTEM_USER()` - Current user
//! - `CONNECTION_ID()` - Connection identifier
//! - `LAST_INSERT_ID()` - Last auto-increment ID
//!
//! ## Type Info
//! - `TYPEOF(expr)` - Type name of expression
//!
//! ## Control Flow
//! - `IF(cond, then, else)` / `IIF(cond, then, else)` - Conditional
//! - `IFNULL(expr, alt)` / `NVL(expr, alt)` - NULL replacement
//! - `NULLIF(a, b)` - Return NULL if equal
//! - `COALESCE(a, b, ...)` - First non-NULL value
//! - `ISNULL(expr)` - Check if NULL (returns 1 or 0)
//!
//! ## Type Conversion
//! - `CAST(expr AS type)` - Type conversion (handled in parser)
//! - `CONVERT(expr, type)` - Type conversion
//! - `BIN(n)` - Convert to binary string
//! - `CONV(n, from_base, to_base)` - Convert between bases

use crate::types::Value;
use std::borrow::Cow;

const TURDB_VERSION: &str = "TurDB 0.1.0";

pub fn eval_system_function<'a>(name: &str, args: &[Option<Value<'a>>]) -> Option<Value<'a>> {
    match name {
        "VERSION" => Some(Value::Text(Cow::Borrowed(TURDB_VERSION))),
        "DATABASE" | "CURRENT_DATABASE" => Some(Value::Text(Cow::Borrowed("turdb"))),
        "USER" | "CURRENT_USER" | "SESSION_USER" | "SYSTEM_USER" => {
            Some(Value::Text(Cow::Borrowed("root@localhost")))
        }
        "CONNECTION_ID" => Some(Value::Int(1)),
        "LAST_INSERT_ID" => Some(Value::Int(0)),
        "TYPEOF" => eval_typeof(args),
        "IF" | "IIF" => eval_if(args),
        "IFNULL" | "NVL" => eval_ifnull(args),
        "NULLIF" => eval_nullif(args),
        "COALESCE" => eval_coalesce(args),
        "ISNULL" => eval_isnull(args),
        "BIN" => eval_bin(args),
        "CONV" => eval_conv(args),
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
        Value::ToastPointer(_) => "toast_pointer",
    };
    Some(Value::Text(Cow::Borrowed(type_name)))
}

fn eval_if<'a>(args: &[Option<Value<'a>>]) -> Option<Value<'a>> {
    if args.len() < 3 {
        return None;
    }
    let condition = match args.first()?.as_ref()? {
        Value::Int(n) => *n != 0,
        Value::Float(f) => *f != 0.0,
        Value::Text(s) => !s.is_empty(),
        Value::Null => false,
        _ => false,
    };

    if condition {
        args.get(1)?.clone()
    } else {
        args.get(2)?.clone()
    }
}

fn eval_ifnull<'a>(args: &[Option<Value<'a>>]) -> Option<Value<'a>> {
    if args.len() < 2 {
        return None;
    }
    match args.first()? {
        Some(Value::Null) | None => args.get(1)?.clone(),
        Some(v) => Some(v.clone()),
    }
}

fn eval_nullif<'a>(args: &[Option<Value<'a>>]) -> Option<Value<'a>> {
    if args.len() < 2 {
        return None;
    }
    let a = args.first()?.as_ref()?;
    let b = args.get(1)?.as_ref()?;

    if values_equal(a, b) {
        Some(Value::Null)
    } else {
        Some(a.clone())
    }
}

fn eval_coalesce<'a>(args: &[Option<Value<'a>>]) -> Option<Value<'a>> {
    for arg in args {
        match arg {
            Some(Value::Null) | None => continue,
            Some(v) => return Some(v.clone()),
        }
    }
    Some(Value::Null)
}

fn eval_isnull<'a>(args: &[Option<Value<'a>>]) -> Option<Value<'a>> {
    match args.first()? {
        Some(Value::Null) | None => Some(Value::Int(1)),
        Some(_) => Some(Value::Int(0)),
    }
}

fn eval_bin<'a>(args: &[Option<Value<'a>>]) -> Option<Value<'a>> {
    let n = match args.first()?.as_ref()? {
        Value::Int(n) => *n,
        Value::Float(f) => *f as i64,
        Value::Null => return Some(Value::Null),
        _ => return None,
    };
    Some(Value::Text(Cow::Owned(format!("{:b}", n))))
}

fn eval_conv<'a>(args: &[Option<Value<'a>>]) -> Option<Value<'a>> {
    if args.len() < 3 {
        return None;
    }
    
    let num_str = match args.first()?.as_ref()? {
        Value::Text(s) => s.to_string(),
        Value::Int(n) => n.to_string(),
        Value::Null => return Some(Value::Null),
        _ => return None,
    };
    
    let from_base = match args.get(1)?.as_ref()? {
        Value::Int(n) => *n as u32,
        _ => return None,
    };
    
    let to_base = match args.get(2)?.as_ref()? {
        Value::Int(n) => *n as u32,
        _ => return None,
    };
    
    if !(2..=36).contains(&from_base) || !(2..=36).contains(&to_base) {
        return Some(Value::Null);
    }
    
    let num = i64::from_str_radix(&num_str, from_base).ok()?;
    
    let result = if to_base == 10 {
        num.to_string()
    } else {
        let mut n = num.unsigned_abs();
        let mut digits = Vec::new();
        if n == 0 {
            digits.push('0');
        }
        while n > 0 {
            let d = (n % to_base as u64) as u32;
            let c = if d < 10 {
                (b'0' + d as u8) as char
            } else {
                (b'A' + (d - 10) as u8) as char
            };
            digits.push(c);
            n /= to_base as u64;
        }
        if num < 0 {
            digits.push('-');
        }
        digits.reverse();
        digits.into_iter().collect()
    };
    
    Some(Value::Text(Cow::Owned(result)))
}

fn values_equal(a: &Value, b: &Value) -> bool {
    match (a, b) {
        (Value::Int(x), Value::Int(y)) => x == y,
        (Value::Float(x), Value::Float(y)) => (x - y).abs() < f64::EPSILON,
        (Value::Int(x), Value::Float(y)) | (Value::Float(y), Value::Int(x)) => {
            (*x as f64 - y).abs() < f64::EPSILON
        }
        (Value::Text(x), Value::Text(y)) => x == y,
        (Value::Null, Value::Null) => true,
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_version() {
        let result = eval_system_function("VERSION", &[]);
        assert!(matches!(result, Some(Value::Text(_))));
    }

    #[test]
    fn test_if() {
        let args = vec![
            Some(Value::Int(1)),
            Some(Value::Text(Cow::Borrowed("yes"))),
            Some(Value::Text(Cow::Borrowed("no"))),
        ];
        assert_eq!(eval_if(&args), Some(Value::Text(Cow::Borrowed("yes"))));

        let args = vec![
            Some(Value::Int(0)),
            Some(Value::Text(Cow::Borrowed("yes"))),
            Some(Value::Text(Cow::Borrowed("no"))),
        ];
        assert_eq!(eval_if(&args), Some(Value::Text(Cow::Borrowed("no"))));
    }

    #[test]
    fn test_coalesce() {
        let args = vec![
            Some(Value::Null),
            Some(Value::Null),
            Some(Value::Int(5)),
        ];
        assert_eq!(eval_coalesce(&args), Some(Value::Int(5)));
    }

    #[test]
    fn test_bin() {
        let args = vec![Some(Value::Int(12))];
        assert_eq!(eval_bin(&args), Some(Value::Text(Cow::Owned("1100".to_string()))));
    }

    #[test]
    fn test_conv() {
        let args = vec![
            Some(Value::Text(Cow::Borrowed("a"))),
            Some(Value::Int(16)),
            Some(Value::Int(10)),
        ];
        assert_eq!(eval_conv(&args), Some(Value::Text(Cow::Owned("10".to_string()))));
    }
}
