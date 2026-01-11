//! # Numeric Functions Module
//!
//! This module provides numeric/mathematical SQL functions:
//!
//! ## Basic Math
//! - `ABS(n)` - Absolute value
//! - `SIGN(n)` - Sign (-1, 0, 1)
//! - `MOD(n, m)` - Modulo (remainder)
//! - `DIV(n, m)` - Integer division
//!
//! ## Rounding
//! - `CEIL(n)` / `CEILING(n)` - Round up
//! - `FLOOR(n)` - Round down
//! - `ROUND(n, d)` - Round to d decimals
//! - `TRUNCATE(n, d)` - Truncate to d decimals
//!
//! ## Powers & Roots
//! - `SQRT(n)` - Square root
//! - `POW(x, y)` / `POWER(x, y)` - x raised to y
//! - `EXP(n)` - e raised to n
//!
//! ## Logarithms
//! - `LN(n)` - Natural log
//! - `LOG(n)` / `LOG(b, n)` - Log base b (default e)
//! - `LOG2(n)` - Log base 2
//! - `LOG10(n)` - Log base 10
//!
//! ## Trigonometry
//! - `SIN(n)`, `COS(n)`, `TAN(n)` - Trig functions
//! - `ASIN(n)`, `ACOS(n)`, `ATAN(n)` - Inverse trig
//! - `ATAN2(y, x)` - Two-argument arctangent
//! - `COT(n)` - Cotangent
//!
//! ## Angle Conversion
//! - `DEGREES(n)` - Radians to degrees
//! - `RADIANS(n)` - Degrees to radians
//!
//! ## Constants & Random
//! - `PI()` - Value of π
//! - `RAND()` / `RAND(seed)` - Random number [0, 1)
//!
//! ## Comparison
//! - `GREATEST(a, b, ...)` - Maximum value
//! - `LEAST(a, b, ...)` - Minimum value

use crate::types::Value;
use std::borrow::Cow;
use std::f64::consts::PI;

pub fn eval_numeric_function<'a>(name: &str, args: &[Option<Value<'a>>]) -> Option<Value<'a>> {
    match name {
        "ABS" => eval_abs(args),
        "SIGN" => eval_sign(args),
        "MOD" => eval_mod(args),
        "DIV" => eval_div(args),
        "CEIL" | "CEILING" => eval_ceil(args),
        "FLOOR" => eval_floor(args),
        "ROUND" => eval_round(args),
        "TRUNCATE" | "TRUNC" => eval_truncate(args),
        "SQRT" => eval_sqrt(args),
        "POW" | "POWER" => eval_power(args),
        "EXP" => eval_exp(args),
        "LN" => eval_ln(args),
        "LOG" => eval_log(args),
        "LOG2" => eval_log2(args),
        "LOG10" => eval_log10(args),
        "SIN" => eval_sin(args),
        "COS" => eval_cos(args),
        "TAN" => eval_tan(args),
        "ASIN" => eval_asin(args),
        "ACOS" => eval_acos(args),
        "ATAN" => eval_atan(args),
        "ATAN2" => eval_atan2(args),
        "COT" => eval_cot(args),
        "DEGREES" => eval_degrees(args),
        "RADIANS" => eval_radians(args),
        "PI" => Some(Value::Float(PI)),
        "RAND" | "RANDOM" => eval_rand(args),
        "GREATEST" => eval_greatest(args),
        "LEAST" => eval_least(args),
        _ => None,
    }
}

fn get_float(val: &Option<Value>) -> Option<f64> {
    match val.as_ref()? {
        Value::Float(f) => Some(*f),
        Value::Int(n) => Some(*n as f64),
        Value::Null => None,
        _ => None,
    }
}

fn get_int(val: &Option<Value>) -> Option<i64> {
    match val.as_ref()? {
        Value::Int(n) => Some(*n),
        Value::Float(f) => Some(*f as i64),
        Value::Null => None,
        _ => None,
    }
}

fn eval_abs<'a>(args: &[Option<Value<'a>>]) -> Option<Value<'a>> {
    match args.first()?.as_ref()? {
        Value::Int(n) => Some(Value::Int(n.abs())),
        Value::Float(f) => Some(Value::Float(f.abs())),
        Value::Null => Some(Value::Null),
        _ => None,
    }
}

fn eval_sign<'a>(args: &[Option<Value<'a>>]) -> Option<Value<'a>> {
    match args.first()?.as_ref()? {
        Value::Int(n) => Some(Value::Int(n.signum())),
        Value::Float(f) => {
            let sign = if *f > 0.0 {
                1
            } else if *f < 0.0 {
                -1
            } else {
                0
            };
            Some(Value::Int(sign))
        }
        Value::Null => Some(Value::Null),
        _ => None,
    }
}

fn eval_mod<'a>(args: &[Option<Value<'a>>]) -> Option<Value<'a>> {
    let a = get_float(args.first()?)?;
    let b = get_float(args.get(1)?)?;

    if b == 0.0 {
        return Some(Value::Null);
    }

    Some(Value::Float(a % b))
}

fn eval_div<'a>(args: &[Option<Value<'a>>]) -> Option<Value<'a>> {
    let a = get_int(args.first()?)?;
    let b = get_int(args.get(1)?)?;

    if b == 0 {
        return Some(Value::Null);
    }

    Some(Value::Int(a / b))
}

fn eval_ceil<'a>(args: &[Option<Value<'a>>]) -> Option<Value<'a>> {
    match args.first()?.as_ref()? {
        Value::Float(f) => Some(Value::Int(f.ceil() as i64)),
        Value::Int(n) => Some(Value::Int(*n)),
        Value::Null => Some(Value::Null),
        _ => None,
    }
}

fn eval_floor<'a>(args: &[Option<Value<'a>>]) -> Option<Value<'a>> {
    match args.first()?.as_ref()? {
        Value::Float(f) => Some(Value::Int(f.floor() as i64)),
        Value::Int(n) => Some(Value::Int(*n)),
        Value::Null => Some(Value::Null),
        _ => None,
    }
}

fn eval_round<'a>(args: &[Option<Value<'a>>]) -> Option<Value<'a>> {
    let val = get_float(args.first()?)?;
    let decimals = args.get(1).and_then(get_int).unwrap_or(0);

    let multiplier = 10_f64.powi(decimals as i32);
    let rounded = (val * multiplier).round() / multiplier;

    if decimals <= 0 {
        Some(Value::Int(rounded as i64))
    } else {
        Some(Value::Float(rounded))
    }
}

fn eval_truncate<'a>(args: &[Option<Value<'a>>]) -> Option<Value<'a>> {
    let val = get_float(args.first()?)?;
    let decimals = args.get(1).and_then(get_int).unwrap_or(0);

    let multiplier = 10_f64.powi(decimals as i32);
    let truncated = (val * multiplier).trunc() / multiplier;

    if decimals <= 0 {
        Some(Value::Int(truncated as i64))
    } else {
        Some(Value::Float(truncated))
    }
}

fn eval_sqrt<'a>(args: &[Option<Value<'a>>]) -> Option<Value<'a>> {
    let val = get_float(args.first()?)?;
    if val < 0.0 {
        return Some(Value::Null);
    }
    Some(Value::Float(val.sqrt()))
}

fn eval_power<'a>(args: &[Option<Value<'a>>]) -> Option<Value<'a>> {
    let base = get_float(args.first()?)?;
    let exp = get_float(args.get(1)?)?;
    Some(Value::Float(base.powf(exp)))
}

fn eval_exp<'a>(args: &[Option<Value<'a>>]) -> Option<Value<'a>> {
    let val = get_float(args.first()?)?;
    Some(Value::Float(val.exp()))
}

fn eval_ln<'a>(args: &[Option<Value<'a>>]) -> Option<Value<'a>> {
    let val = get_float(args.first()?)?;
    if val <= 0.0 {
        return Some(Value::Null);
    }
    Some(Value::Float(val.ln()))
}

fn eval_log<'a>(args: &[Option<Value<'a>>]) -> Option<Value<'a>> {
    if args.len() == 1 {
        let val = get_float(args.first()?)?;
        if val <= 0.0 {
            return Some(Value::Null);
        }
        return Some(Value::Float(val.ln()));
    }

    let base = get_float(args.first()?)?;
    let val = get_float(args.get(1)?)?;

    if base <= 0.0 || base == 1.0 || val <= 0.0 {
        return Some(Value::Null);
    }

    Some(Value::Float(val.log(base)))
}

fn eval_log2<'a>(args: &[Option<Value<'a>>]) -> Option<Value<'a>> {
    let val = get_float(args.first()?)?;
    if val <= 0.0 {
        return Some(Value::Null);
    }
    Some(Value::Float(val.log2()))
}

fn eval_log10<'a>(args: &[Option<Value<'a>>]) -> Option<Value<'a>> {
    let val = get_float(args.first()?)?;
    if val <= 0.0 {
        return Some(Value::Null);
    }
    Some(Value::Float(val.log10()))
}

fn eval_sin<'a>(args: &[Option<Value<'a>>]) -> Option<Value<'a>> {
    let val = get_float(args.first()?)?;
    Some(Value::Float(val.sin()))
}

fn eval_cos<'a>(args: &[Option<Value<'a>>]) -> Option<Value<'a>> {
    let val = get_float(args.first()?)?;
    Some(Value::Float(val.cos()))
}

fn eval_tan<'a>(args: &[Option<Value<'a>>]) -> Option<Value<'a>> {
    let val = get_float(args.first()?)?;
    Some(Value::Float(val.tan()))
}

fn eval_asin<'a>(args: &[Option<Value<'a>>]) -> Option<Value<'a>> {
    let val = get_float(args.first()?)?;
    if !(-1.0..=1.0).contains(&val) {
        return Some(Value::Null);
    }
    Some(Value::Float(val.asin()))
}

fn eval_acos<'a>(args: &[Option<Value<'a>>]) -> Option<Value<'a>> {
    let val = get_float(args.first()?)?;
    if !(-1.0..=1.0).contains(&val) {
        return Some(Value::Null);
    }
    Some(Value::Float(val.acos()))
}

fn eval_atan<'a>(args: &[Option<Value<'a>>]) -> Option<Value<'a>> {
    let val = get_float(args.first()?)?;
    Some(Value::Float(val.atan()))
}

fn eval_atan2<'a>(args: &[Option<Value<'a>>]) -> Option<Value<'a>> {
    let y = get_float(args.first()?)?;
    let x = get_float(args.get(1)?)?;
    Some(Value::Float(y.atan2(x)))
}

fn eval_cot<'a>(args: &[Option<Value<'a>>]) -> Option<Value<'a>> {
    let val = get_float(args.first()?)?;
    let tan_val = val.tan();
    if tan_val == 0.0 {
        return Some(Value::Null);
    }
    Some(Value::Float(1.0 / tan_val))
}

fn eval_degrees<'a>(args: &[Option<Value<'a>>]) -> Option<Value<'a>> {
    let val = get_float(args.first()?)?;
    Some(Value::Float(val.to_degrees()))
}

fn eval_radians<'a>(args: &[Option<Value<'a>>]) -> Option<Value<'a>> {
    let val = get_float(args.first()?)?;
    Some(Value::Float(val.to_radians()))
}

fn eval_rand<'a>(args: &[Option<Value<'a>>]) -> Option<Value<'a>> {
    let seed = args.first().and_then(get_int);

    let random = match seed {
        Some(s) => {
            let mut state = s as u64;
            state ^= state >> 12;
            state ^= state << 25;
            state ^= state >> 27;
            (state.wrapping_mul(0x2545F4914F6CDD1D) as f64) / (u64::MAX as f64)
        }
        None => {
            use std::time::SystemTime;
            let nanos = SystemTime::now()
                .duration_since(SystemTime::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0) as u64;
            let mut state = nanos;
            state ^= state >> 12;
            state ^= state << 25;
            state ^= state >> 27;
            (state.wrapping_mul(0x2545F4914F6CDD1D) as f64) / (u64::MAX as f64)
        }
    };

    Some(Value::Float(random))
}

fn eval_greatest<'a>(args: &[Option<Value<'a>>]) -> Option<Value<'a>> {
    if args.is_empty() {
        return Some(Value::Null);
    }

    let mut max_int: Option<i64> = None;
    let mut max_float: Option<f64> = None;
    let mut max_text: Option<Cow<'a, str>> = None;
    let mut has_null = false;

    for arg in args {
        match arg.as_ref() {
            Some(Value::Int(n)) => {
                max_int = Some(max_int.map_or(*n, |m| m.max(*n)));
            }
            Some(Value::Float(f)) => {
                max_float = Some(max_float.map_or(*f, |m| m.max(*f)));
            }
            Some(Value::Text(s)) => {
                max_text = Some(max_text.map_or(s.clone(), |m| {
                    if s.as_ref() > m.as_ref() {
                        s.clone()
                    } else {
                        m
                    }
                }));
            }
            Some(Value::Null) | None => {
                has_null = true;
            }
            _ => {}
        }
    }

    if has_null && max_int.is_none() && max_float.is_none() && max_text.is_none() {
        return Some(Value::Null);
    }

    if let Some(t) = max_text {
        return Some(Value::Text(t));
    }

    match (max_int, max_float) {
        (Some(i), Some(f)) => {
            if (i as f64) > f {
                Some(Value::Int(i))
            } else {
                Some(Value::Float(f))
            }
        }
        (Some(i), None) => Some(Value::Int(i)),
        (None, Some(f)) => Some(Value::Float(f)),
        (None, None) => Some(Value::Null),
    }
}

fn eval_least<'a>(args: &[Option<Value<'a>>]) -> Option<Value<'a>> {
    if args.is_empty() {
        return Some(Value::Null);
    }

    let mut min_int: Option<i64> = None;
    let mut min_float: Option<f64> = None;
    let mut min_text: Option<Cow<'a, str>> = None;
    let mut has_null = false;

    for arg in args {
        match arg.as_ref() {
            Some(Value::Int(n)) => {
                min_int = Some(min_int.map_or(*n, |m| m.min(*n)));
            }
            Some(Value::Float(f)) => {
                min_float = Some(min_float.map_or(*f, |m| m.min(*f)));
            }
            Some(Value::Text(s)) => {
                min_text = Some(min_text.map_or(s.clone(), |m| {
                    if s.as_ref() < m.as_ref() {
                        s.clone()
                    } else {
                        m
                    }
                }));
            }
            Some(Value::Null) | None => {
                has_null = true;
            }
            _ => {}
        }
    }

    if has_null && min_int.is_none() && min_float.is_none() && min_text.is_none() {
        return Some(Value::Null);
    }

    if let Some(t) = min_text {
        return Some(Value::Text(t));
    }

    match (min_int, min_float) {
        (Some(i), Some(f)) => {
            if (i as f64) < f {
                Some(Value::Int(i))
            } else {
                Some(Value::Float(f))
            }
        }
        (Some(i), None) => Some(Value::Int(i)),
        (None, Some(f)) => Some(Value::Float(f)),
        (None, None) => Some(Value::Null),
    }
}

#[cfg(test)]
#[allow(clippy::useless_vec)]
mod tests {
    use super::*;

    #[test]
    fn test_abs() {
        let args = vec![Some(Value::Int(-5))];
        assert_eq!(eval_abs(&args), Some(Value::Int(5)));

        let args = vec![Some(Value::Float(-3.125))];
        if let Some(Value::Float(f)) = eval_abs(&args) {
            assert!((f - 3.125).abs() < 0.001);
        } else {
            panic!("Expected float");
        }
    }

    #[test]
    fn test_sign() {
        assert_eq!(eval_sign(&vec![Some(Value::Int(5))]), Some(Value::Int(1)));
        assert_eq!(eval_sign(&vec![Some(Value::Int(-5))]), Some(Value::Int(-1)));
        assert_eq!(eval_sign(&vec![Some(Value::Int(0))]), Some(Value::Int(0)));
    }

    #[test]
    fn test_round() {
        let args = vec![Some(Value::Float(3.567)), Some(Value::Int(2))];
        if let Some(Value::Float(f)) = eval_round(&args) {
            assert!((f - 3.57).abs() < 0.001);
        } else {
            panic!("Expected float");
        }
    }

    #[test]
    fn test_truncate() {
        let args = vec![Some(Value::Float(3.567)), Some(Value::Int(2))];
        if let Some(Value::Float(f)) = eval_truncate(&args) {
            assert!((f - 3.56).abs() < 0.001);
        } else {
            panic!("Expected float");
        }
    }

    #[test]
    fn test_greatest_least() {
        let args = vec![
            Some(Value::Int(3)),
            Some(Value::Int(7)),
            Some(Value::Int(1)),
        ];
        assert_eq!(eval_greatest(&args), Some(Value::Int(7)));
        assert_eq!(eval_least(&args), Some(Value::Int(1)));
    }

    #[test]
    fn test_trig() {
        let args = vec![Some(Value::Float(0.0))];
        if let Some(Value::Float(f)) = eval_sin(&args) {
            assert!(f.abs() < 0.001);
        }
        if let Some(Value::Float(f)) = eval_cos(&args) {
            assert!((f - 1.0).abs() < 0.001);
        }
    }
}
