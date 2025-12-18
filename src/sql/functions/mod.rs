//! # SQL Functions Module
//!
//! This module provides built-in SQL functions organized by category.
//! Each category has its own submodule for maintainability.
//!
//! ## Module Structure
//!
//! - `datetime`: Date and time functions (NOW, DATE_ADD, DATEDIFF, etc.)
//! - Future: `string`, `numeric`, `aggregate`, `json`, etc.
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
//! 3. The main dispatcher in `eval_function()` will route to it
//!
//! ## Value Handling
//!
//! All functions work with `Value<'a>` from the types module and return
//! `Option<Value<'a>>` to handle NULL propagation and error cases.

pub mod datetime;
pub mod system;

use crate::types::Value;
use std::borrow::Cow;

/// Evaluates a SQL function by name with the given arguments.
/// Returns None if the function is unknown or arguments are invalid.
pub fn eval_function<'a>(name: &str, args: &[Option<Value<'a>>]) -> Option<Value<'a>> {
    let upper_name = name.to_uppercase();
    
    match upper_name.as_str() {
        "NOW" | "CURRENT_TIMESTAMP" | "CURRENT_DATE" | "CURRENT_TIME" 
        | "DATE_FORMAT" | "STRFTIME" | "DATE_ADD" | "DATE_SUB" 
        | "DATEDIFF" | "FROM_DAYS" | "TO_DAYS" | "LAST_DAY" => {
            datetime::eval_datetime_function(&upper_name, args)
        }
        "VERSION" | "DATABASE" | "CURRENT_DATABASE" | "TYPEOF" => {
            system::eval_system_function(&upper_name, args)
        }
        "FORMAT" => eval_format(args),
        "COALESCE" => eval_coalesce(args),
        "ABS" => eval_abs(args),
        "ROUND" => eval_round(args),
        "FLOOR" => eval_floor(args),
        "CEIL" | "CEILING" => eval_ceil(args),
        "SQRT" => eval_sqrt(args),
        "POWER" | "POW" => eval_power(args),
        "UPPER" => eval_upper(args),
        "LOWER" => eval_lower(args),
        "LENGTH" | "LEN" => eval_length(args),
        "TRIM" => eval_trim(args),
        "LTRIM" => eval_ltrim(args),
        "RTRIM" => eval_rtrim(args),
        "SUBSTR" | "SUBSTRING" => eval_substr(args),
        "CONCAT" => eval_concat(args),
        "REPLACE" => eval_replace(args),
        "NULLIF" => eval_nullif(args),
        "IFNULL" | "NVL" => eval_ifnull(args),
        "IIF" | "IF" => eval_iif(args),
        "INSTR" => eval_instr(args),
        "LEFT" => eval_left(args),
        "RIGHT" => eval_right(args),
        _ => None,
    }
}

fn eval_format<'a>(args: &[Option<Value<'a>>]) -> Option<Value<'a>> {
    if args.len() < 2 {
        return None;
    }
    let number = match args.first()?.as_ref()? {
        Value::Float(f) => *f,
        Value::Int(n) => *n as f64,
        Value::Null => return Some(Value::Null),
        _ => return None,
    };
    let decimals = match args.get(1)?.as_ref()? {
        Value::Int(n) => *n as usize,
        Value::Float(f) => *f as usize,
        _ => return None,
    };
    let formatted = format_number_with_decimals(number, decimals);
    Some(Value::Text(Cow::Owned(formatted)))
}

fn format_number_with_decimals(number: f64, decimals: usize) -> String {
    let formatted = format!("{:.prec$}", number, prec = decimals);
    
    let parts: Vec<&str> = formatted.split('.').collect();
    let integer_part = parts.first().unwrap_or(&"0");
    let decimal_part = parts.get(1);

    let is_negative = integer_part.starts_with('-');
    let abs_integer: String = integer_part.chars().filter(|c| c.is_ascii_digit()).collect();
    
    let mut with_commas = String::new();
    for (i, c) in abs_integer.chars().rev().enumerate() {
        if i > 0 && i % 3 == 0 {
            with_commas.push(',');
        }
        with_commas.push(c);
    }
    let with_commas: String = with_commas.chars().rev().collect();

    let result = match decimal_part {
        Some(dec) => format!("{}.{}", with_commas, dec),
        None => with_commas,
    };

    if is_negative {
        format!("-{}", result)
    } else {
        result
    }
}

fn eval_coalesce<'a>(args: &[Option<Value<'a>>]) -> Option<Value<'a>> {
    for arg in args {
        if let Some(val) = arg {
            if !matches!(val, Value::Null) {
                return Some(val.clone());
            }
        }
    }
    Some(Value::Null)
}

fn eval_abs<'a>(args: &[Option<Value<'a>>]) -> Option<Value<'a>> {
    match args.first()?.as_ref()? {
        Value::Int(n) => Some(Value::Int(n.abs())),
        Value::Float(f) => Some(Value::Float(f.abs())),
        Value::Null => Some(Value::Null),
        _ => None,
    }
}

fn eval_round<'a>(args: &[Option<Value<'a>>]) -> Option<Value<'a>> {
    let val = args.first()?.as_ref()?;
    let decimals = args
        .get(1)
        .and_then(|v| v.as_ref())
        .map(|v| match v {
            Value::Int(n) => *n as i32,
            _ => 0,
        })
        .unwrap_or(0);

    match val {
        Value::Float(f) => {
            let multiplier = 10_f64.powi(decimals);
            Some(Value::Float((f * multiplier).round() / multiplier))
        }
        Value::Int(n) => Some(Value::Int(*n)),
        Value::Null => Some(Value::Null),
        _ => None,
    }
}

fn eval_floor<'a>(args: &[Option<Value<'a>>]) -> Option<Value<'a>> {
    match args.first()?.as_ref()? {
        Value::Float(f) => Some(Value::Float(f.floor())),
        Value::Int(n) => Some(Value::Int(*n)),
        Value::Null => Some(Value::Null),
        _ => None,
    }
}

fn eval_ceil<'a>(args: &[Option<Value<'a>>]) -> Option<Value<'a>> {
    match args.first()?.as_ref()? {
        Value::Float(f) => Some(Value::Float(f.ceil())),
        Value::Int(n) => Some(Value::Int(*n)),
        Value::Null => Some(Value::Null),
        _ => None,
    }
}

fn eval_sqrt<'a>(args: &[Option<Value<'a>>]) -> Option<Value<'a>> {
    match args.first()?.as_ref()? {
        Value::Float(f) if *f >= 0.0 => Some(Value::Float(f.sqrt())),
        Value::Int(n) if *n >= 0 => Some(Value::Float((*n as f64).sqrt())),
        Value::Null => Some(Value::Null),
        _ => None,
    }
}

fn eval_power<'a>(args: &[Option<Value<'a>>]) -> Option<Value<'a>> {
    if args.len() < 2 {
        return None;
    }
    let base = args.first()?.as_ref()?;
    let exp = args.get(1)?.as_ref()?;

    match (base, exp) {
        (Value::Float(b), Value::Float(e)) => Some(Value::Float(b.powf(*e))),
        (Value::Int(b), Value::Int(e)) if *e >= 0 => Some(Value::Int(b.pow(*e as u32))),
        (Value::Int(b), Value::Float(e)) => Some(Value::Float((*b as f64).powf(*e))),
        (Value::Float(b), Value::Int(e)) => Some(Value::Float(b.powi(*e as i32))),
        (Value::Null, _) | (_, Value::Null) => Some(Value::Null),
        _ => None,
    }
}

fn eval_upper<'a>(args: &[Option<Value<'a>>]) -> Option<Value<'a>> {
    match args.first()?.as_ref()? {
        Value::Text(s) => Some(Value::Text(Cow::Owned(s.to_uppercase()))),
        Value::Null => Some(Value::Null),
        _ => None,
    }
}

fn eval_lower<'a>(args: &[Option<Value<'a>>]) -> Option<Value<'a>> {
    match args.first()?.as_ref()? {
        Value::Text(s) => Some(Value::Text(Cow::Owned(s.to_lowercase()))),
        Value::Null => Some(Value::Null),
        _ => None,
    }
}

fn eval_length<'a>(args: &[Option<Value<'a>>]) -> Option<Value<'a>> {
    match args.first()?.as_ref()? {
        Value::Text(s) => Some(Value::Int(s.len() as i64)),
        Value::Null => Some(Value::Null),
        _ => None,
    }
}

fn eval_trim<'a>(args: &[Option<Value<'a>>]) -> Option<Value<'a>> {
    match args.first()?.as_ref()? {
        Value::Text(s) => Some(Value::Text(Cow::Owned(s.trim().to_string()))),
        Value::Null => Some(Value::Null),
        _ => None,
    }
}

fn eval_ltrim<'a>(args: &[Option<Value<'a>>]) -> Option<Value<'a>> {
    match args.first()?.as_ref()? {
        Value::Text(s) => Some(Value::Text(Cow::Owned(s.trim_start().to_string()))),
        Value::Null => Some(Value::Null),
        _ => None,
    }
}

fn eval_rtrim<'a>(args: &[Option<Value<'a>>]) -> Option<Value<'a>> {
    match args.first()?.as_ref()? {
        Value::Text(s) => Some(Value::Text(Cow::Owned(s.trim_end().to_string()))),
        Value::Null => Some(Value::Null),
        _ => None,
    }
}

fn eval_substr<'a>(args: &[Option<Value<'a>>]) -> Option<Value<'a>> {
    if args.is_empty() {
        return None;
    }
    let text = match args.first()?.as_ref()? {
        Value::Text(s) => s,
        Value::Null => return Some(Value::Null),
        _ => return None,
    };

    let start = match args.get(1)?.as_ref()? {
        Value::Int(n) => (*n as usize).saturating_sub(1),
        _ => return None,
    };

    let len = args.get(2).and_then(|v| v.as_ref()).map(|v| match v {
        Value::Int(n) => *n as usize,
        _ => text.len(),
    });

    let result: String = match len {
        Some(l) => text.chars().skip(start).take(l).collect(),
        None => text.chars().skip(start).collect(),
    };

    Some(Value::Text(Cow::Owned(result)))
}

fn eval_concat<'a>(args: &[Option<Value<'a>>]) -> Option<Value<'a>> {
    let mut result = String::new();
    for arg in args {
        match arg.as_ref()? {
            Value::Text(s) => result.push_str(s),
            Value::Int(n) => result.push_str(&n.to_string()),
            Value::Float(f) => result.push_str(&f.to_string()),

            Value::Null => return Some(Value::Null),
            _ => {}
        }
    }
    Some(Value::Text(Cow::Owned(result)))
}

fn eval_replace<'a>(args: &[Option<Value<'a>>]) -> Option<Value<'a>> {
    if args.len() < 3 {
        return None;
    }
    let text = match args.first()?.as_ref()? {
        Value::Text(s) => s,
        Value::Null => return Some(Value::Null),
        _ => return None,
    };
    let from = match args.get(1)?.as_ref()? {
        Value::Text(s) => s,
        Value::Null => return Some(Value::Null),
        _ => return None,
    };
    let to = match args.get(2)?.as_ref()? {
        Value::Text(s) => s,
        Value::Null => return Some(Value::Null),
        _ => return None,
    };

    Some(Value::Text(Cow::Owned(text.replace(from.as_ref(), to.as_ref()))))
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

fn eval_ifnull<'a>(args: &[Option<Value<'a>>]) -> Option<Value<'a>> {
    if args.len() < 2 {
        return None;
    }
    match args.first()?.as_ref()? {
        Value::Null => args.get(1)?.clone(),
        v => Some(v.clone()),
    }
}

fn eval_iif<'a>(args: &[Option<Value<'a>>]) -> Option<Value<'a>> {
    if args.len() < 3 {
        return None;
    }
    let condition = match args.first()?.as_ref()? {

        Value::Int(n) => *n != 0,
        Value::Null => false,
        _ => return None,
    };

    if condition {
        args.get(1)?.clone()
    } else {
        args.get(2)?.clone()
    }
}

fn eval_instr<'a>(args: &[Option<Value<'a>>]) -> Option<Value<'a>> {
    if args.len() < 2 {
        return None;
    }
    let haystack = match args.first()?.as_ref()? {
        Value::Text(s) => s,
        Value::Null => return Some(Value::Null),
        _ => return None,
    };
    let needle = match args.get(1)?.as_ref()? {
        Value::Text(s) => s,
        Value::Null => return Some(Value::Null),
        _ => return None,
    };

    match haystack.find(needle.as_ref()) {
        Some(pos) => Some(Value::Int((pos + 1) as i64)),
        None => Some(Value::Int(0)),
    }
}

fn eval_left<'a>(args: &[Option<Value<'a>>]) -> Option<Value<'a>> {
    if args.len() < 2 {
        return None;
    }
    let text = match args.first()?.as_ref()? {
        Value::Text(s) => s,
        Value::Null => return Some(Value::Null),
        _ => return None,
    };
    let len = match args.get(1)?.as_ref()? {
        Value::Int(n) if *n >= 0 => *n as usize,
        Value::Null => return Some(Value::Null),
        _ => return None,
    };

    let result: String = text.chars().take(len).collect();
    Some(Value::Text(Cow::Owned(result)))
}

fn eval_right<'a>(args: &[Option<Value<'a>>]) -> Option<Value<'a>> {
    if args.len() < 2 {
        return None;
    }
    let text = match args.first()?.as_ref()? {
        Value::Text(s) => s,
        Value::Null => return Some(Value::Null),
        _ => return None,
    };
    let len = match args.get(1)?.as_ref()? {
        Value::Int(n) if *n >= 0 => *n as usize,
        Value::Null => return Some(Value::Null),
        _ => return None,
    };

    let char_count = text.chars().count();
    let skip = char_count.saturating_sub(len);
    let result: String = text.chars().skip(skip).collect();
    Some(Value::Text(Cow::Owned(result)))
}

fn values_equal(a: &Value, b: &Value) -> bool {
    match (a, b) {
        (Value::Int(x), Value::Int(y)) => x == y,
        (Value::Float(x), Value::Float(y)) => (x - y).abs() < f64::EPSILON,
        (Value::Text(x), Value::Text(y)) => x == y,

        (Value::Null, Value::Null) => true,
        _ => false,
    }
}
