//! # SQL Literal Parsing
//!
//! This module provides parsing for SQL literal values into `OwnedValue`.
//! Consolidates parsing logic from database.rs for typed literals.
//!
//! ## Supported Literal Types
//!
//! | Type | Format | Example |
//! |------|--------|---------|
//! | UUID | Standard/compact | `550e8400-e29b-41d4-a716-446655440000` |
//! | Vector | Bracketed floats | `[1.0, 2.0, 3.0]` |
//! | Blob (hex) | `\x` prefix | `\x48454C4C4F` |
//! | Blob (binary) | `0b` or raw | `01001000` |
//! | Date | ISO 8601 | `2024-01-15` |
//! | Time | ISO 8601 | `13:45:30` |
//! | Timestamp | ISO 8601 | `2024-01-15T13:45:30` |
//!
//! ## Architecture
//!
//! ```text
//! ┌─────────────┐    ┌───────────────┐    ┌────────────┐
//! │ SQL Literal │───>│ LiteralParser │───>│ OwnedValue │
//! └─────────────┘    └───────────────┘    └────────────┘
//! ```
//!
//! ## Usage
//!
//! ```ignore
//! use turdb::parsing::literal::{parse_uuid, parse_vector, parse_hex_blob};
//!
//! let uuid = parse_uuid("550e8400-e29b-41d4-a716-446655440000")?;
//! let vec = parse_vector("[1.0, 2.0, 3.0]")?;
//! let blob = parse_hex_blob("48454C4C4F")?;  // "HELLO"
//! ```
//!
//! ## LiteralParser
//!
//! For comprehensive literal parsing with type inference:
//!
//! ```ignore
//! use turdb::parsing::literal::LiteralParser;
//!
//! let parser = LiteralParser::new();
//! let value = parser.parse("'hello'")?;  // Text
//! let value = parser.parse("42")?;       // Int
//! let value = parser.parse("3.14")?;     // Float
//! ```
//!
//! ## Error Handling
//!
//! All functions return `eyre::Result` with detailed context:
//!
//! ```text
//! "invalid UUID format '...' : expected 32 hex chars, got 28"
//! "failed to parse vector element: 'abc' is not a valid float"
//! ```

use crate::types::OwnedValue;
use eyre::{bail, Result, WrapErr};

#[derive(Debug, Clone, PartialEq)]
pub enum ParsedLiteral {
    Null,
    Bool(bool),
    Int(i64),
    Float(f64),
    Text(String),
    Blob(Vec<u8>),
    Vector(Vec<f32>),
    Uuid([u8; 16]),
}

impl From<ParsedLiteral> for OwnedValue {
    fn from(lit: ParsedLiteral) -> Self {
        match lit {
            ParsedLiteral::Null => OwnedValue::Null,
            ParsedLiteral::Bool(b) => OwnedValue::Bool(b),
            ParsedLiteral::Int(i) => OwnedValue::Int(i),
            ParsedLiteral::Float(f) => OwnedValue::Float(f),
            ParsedLiteral::Text(s) => OwnedValue::Text(s),
            ParsedLiteral::Blob(b) => OwnedValue::Blob(b),
            ParsedLiteral::Vector(v) => OwnedValue::Vector(v),
            ParsedLiteral::Uuid(u) => OwnedValue::Uuid(u),
        }
    }
}

pub fn parse_uuid(s: &str) -> Result<OwnedValue> {
    let s = s.trim();
    let hex_only: String = s.chars().filter(|c| *c != '-').collect();

    if hex_only.len() != 32 {
        bail!(
            "invalid UUID format '{}': expected 32 hex chars, got {}",
            s,
            hex_only.len()
        );
    }

    let mut bytes = [0u8; 16];
    for (i, chunk) in hex_only.as_bytes().chunks(2).enumerate() {
        let hex_pair = std::str::from_utf8(chunk)
            .wrap_err_with(|| format!("invalid UTF-8 in UUID hex: {:?}", chunk))?;
        bytes[i] = u8::from_str_radix(hex_pair, 16)
            .wrap_err_with(|| format!("invalid hex in UUID: '{}'", hex_pair))?;
    }

    Ok(OwnedValue::Uuid(bytes))
}

pub fn parse_vector(s: &str) -> Result<OwnedValue> {
    let s = s.trim();

    let inner = if s.starts_with('[') && s.ends_with(']') {
        &s[1..s.len() - 1]
    } else {
        s
    };

    if inner.trim().is_empty() {
        return Ok(OwnedValue::Vector(vec![]));
    }

    let values: Vec<f32> = inner
        .split(',')
        .map(|part| {
            part.trim()
                .parse::<f32>()
                .wrap_err_with(|| format!("failed to parse vector element: '{}'", part.trim()))
        })
        .collect::<Result<Vec<_>>>()?;

    Ok(OwnedValue::Vector(values))
}

pub fn parse_hex_blob(s: &str) -> Result<OwnedValue> {
    if !s.len().is_multiple_of(2) {
        bail!("hex string must have even length, got {}", s.len());
    }

    let bytes: Vec<u8> = (0..s.len())
        .step_by(2)
        .map(|i| {
            u8::from_str_radix(&s[i..i + 2], 16)
                .wrap_err_with(|| format!("invalid hex byte: '{}'", &s[i..i + 2]))
        })
        .collect::<Result<Vec<_>>>()?;

    Ok(OwnedValue::Blob(bytes))
}

pub fn parse_binary_blob(s: &str) -> Result<OwnedValue> {
    if s.is_empty() {
        return Ok(OwnedValue::Blob(vec![]));
    }

    let bytes: Vec<u8> = (0..s.len())
        .step_by(8)
        .map(|i| {
            let end = std::cmp::min(i + 8, s.len());
            u8::from_str_radix(&s[i..end], 2)
                .wrap_err_with(|| format!("invalid binary byte: '{}'", &s[i..end]))
        })
        .collect::<Result<Vec<_>>>()?;

    Ok(OwnedValue::Blob(bytes))
}

pub fn parse_date(s: &str) -> Result<OwnedValue> {
    let s = s.trim();
    let parts: Vec<&str> = s.split('-').collect();

    if parts.len() != 3 {
        bail!("invalid date format '{}': expected YYYY-MM-DD", s);
    }

    let year: i32 = parts[0]
        .parse()
        .wrap_err_with(|| format!("invalid year in date: '{}'", parts[0]))?;
    let month: u32 = parts[1]
        .parse()
        .wrap_err_with(|| format!("invalid month in date: '{}'", parts[1]))?;
    let day: u32 = parts[2]
        .parse()
        .wrap_err_with(|| format!("invalid day in date: '{}'", parts[2]))?;

    if !(1..=12).contains(&month) {
        bail!("invalid month {} in date '{}': must be 1-12", month, s);
    }

    let days_in_month = days_in_month(year, month);
    if day < 1 || day > days_in_month {
        bail!(
            "invalid day {} in date '{}': {} has {} days",
            day,
            s,
            month,
            days_in_month
        );
    }

    let days = date_to_days_since_epoch(year, month, day);
    Ok(OwnedValue::Date(days))
}

pub fn parse_time(s: &str) -> Result<OwnedValue> {
    let s = s.trim();

    let (time_part, micros_part) = if let Some(idx) = s.find('.') {
        (&s[..idx], Some(&s[idx + 1..]))
    } else {
        (s, None)
    };

    let parts: Vec<&str> = time_part.split(':').collect();
    if parts.len() != 3 {
        bail!("invalid time format '{}': expected HH:MM:SS", s);
    }

    let hour: u32 = parts[0]
        .parse()
        .wrap_err_with(|| format!("invalid hour in time: '{}'", parts[0]))?;
    let minute: u32 = parts[1]
        .parse()
        .wrap_err_with(|| format!("invalid minute in time: '{}'", parts[1]))?;
    let second: u32 = parts[2]
        .parse()
        .wrap_err_with(|| format!("invalid second in time: '{}'", parts[2]))?;

    if hour > 23 {
        bail!("invalid hour {} in time '{}': must be 0-23", hour, s);
    }
    if minute > 59 {
        bail!("invalid minute {} in time '{}': must be 0-59", minute, s);
    }
    if second > 59 {
        bail!("invalid second {} in time '{}': must be 0-59", second, s);
    }

    let base_micros = (hour as i64 * 3600 + minute as i64 * 60 + second as i64) * 1_000_000;

    let fractional_micros: i64 = if let Some(frac) = micros_part {
        let padded = format!("{:0<6}", frac);
        let truncated = &padded[..6.min(padded.len())];
        truncated
            .parse()
            .wrap_err_with(|| format!("invalid fractional seconds: '{}'", frac))?
    } else {
        0
    };

    Ok(OwnedValue::Time(base_micros + fractional_micros))
}

pub fn parse_timestamp(s: &str) -> Result<OwnedValue> {
    let s = s.trim();

    let (date_str, time_str) = if let Some(idx) = s.find('T') {
        (&s[..idx], &s[idx + 1..])
    } else if let Some(idx) = s.find(' ') {
        (&s[..idx], &s[idx + 1..])
    } else {
        bail!(
            "invalid timestamp format '{}': expected YYYY-MM-DDTHH:MM:SS or YYYY-MM-DD HH:MM:SS",
            s
        );
    };

    let date_val = parse_date(date_str)?;
    let time_val = parse_time(time_str)?;

    let days = match date_val {
        OwnedValue::Date(d) => d,
        _ => unreachable!(),
    };

    let time_micros = match time_val {
        OwnedValue::Time(t) => t,
        _ => unreachable!(),
    };

    let micros_per_day = 86400i64 * 1_000_000;
    let total_micros = days as i64 * micros_per_day + time_micros;

    Ok(OwnedValue::Timestamp(total_micros))
}

fn is_leap_year(year: i32) -> bool {
    (year % 4 == 0 && year % 100 != 0) || (year % 400 == 0)
}

fn days_in_month(year: i32, month: u32) -> u32 {
    match month {
        1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
        4 | 6 | 9 | 11 => 30,
        2 => {
            if is_leap_year(year) {
                29
            } else {
                28
            }
        }
        _ => 0,
    }
}

fn date_to_days_since_epoch(year: i32, month: u32, day: u32) -> i32 {
    let mut days: i32 = 0;

    if year >= 1970 {
        for y in 1970..year {
            days += if is_leap_year(y) { 366 } else { 365 };
        }
    } else {
        for y in year..1970 {
            days -= if is_leap_year(y) { 366 } else { 365 };
        }
    }

    for m in 1..month {
        days += days_in_month(year, m) as i32;
    }

    days += day as i32 - 1;

    days
}

pub struct LiteralParser;

impl LiteralParser {
    pub fn new() -> Self {
        Self
    }

    pub fn parse(&self, s: &str) -> Result<ParsedLiteral> {
        let s = s.trim();

        if s.eq_ignore_ascii_case("null") {
            return Ok(ParsedLiteral::Null);
        }

        if s.eq_ignore_ascii_case("true") {
            return Ok(ParsedLiteral::Bool(true));
        }

        if s.eq_ignore_ascii_case("false") {
            return Ok(ParsedLiteral::Bool(false));
        }

        if (s.starts_with('\'') && s.ends_with('\'')) || (s.starts_with('"') && s.ends_with('"')) {
            let inner = &s[1..s.len() - 1];
            return Ok(ParsedLiteral::Text(inner.to_string()));
        }

        if let Ok(i) = s.parse::<i64>() {
            return Ok(ParsedLiteral::Int(i));
        }

        if let Ok(f) = s.parse::<f64>() {
            return Ok(ParsedLiteral::Float(f));
        }

        Ok(ParsedLiteral::Text(s.to_string()))
    }

    pub fn parse_typed(&self, s: &str, type_hint: &str) -> Result<OwnedValue> {
        let s = s.trim();
        let type_hint = type_hint.to_lowercase();

        match type_hint.as_str() {
            "uuid" => parse_uuid(s),
            "vector" => parse_vector(s),
            "bytea" | "blob" => {
                if let Some(hex) = s.strip_prefix("\\x") {
                    parse_hex_blob(hex)
                } else if let Some(hex) = s.strip_prefix("0x") {
                    parse_hex_blob(hex)
                } else if s.chars().all(|c| c == '0' || c == '1') {
                    parse_binary_blob(s)
                } else {
                    parse_hex_blob(s)
                }
            }
            "int" | "integer" | "int4" | "int8" | "bigint" | "smallint" | "int2" => {
                let i: i64 = s
                    .parse()
                    .wrap_err_with(|| format!("invalid integer: '{}'", s))?;
                Ok(OwnedValue::Int(i))
            }
            "float" | "float4" | "float8" | "double" | "real" => {
                let f: f64 = s
                    .parse()
                    .wrap_err_with(|| format!("invalid float: '{}'", s))?;
                Ok(OwnedValue::Float(f))
            }
            "bool" | "boolean" => {
                if s.eq_ignore_ascii_case("true") || s == "1" {
                    Ok(OwnedValue::Bool(true))
                } else if s.eq_ignore_ascii_case("false") || s == "0" {
                    Ok(OwnedValue::Bool(false))
                } else {
                    bail!("invalid boolean: '{}'", s)
                }
            }
            "text" | "varchar" | "char" => {
                let inner = if (s.starts_with('\'') && s.ends_with('\''))
                    || (s.starts_with('"') && s.ends_with('"'))
                {
                    &s[1..s.len() - 1]
                } else {
                    s
                };
                Ok(OwnedValue::Text(inner.to_string()))
            }
            _ => {
                let parsed = self.parse(s)?;
                Ok(parsed.into())
            }
        }
    }
}

impl Default for LiteralParser {
    fn default() -> Self {
        Self::new()
    }
}

pub fn parse_interval(s: &str) -> Result<OwnedValue> {
    let s = s.trim();

    if s.starts_with('P') || s.starts_with('p') {
        return parse_iso8601_interval(s);
    }

    parse_postgres_interval(s)
}

fn parse_iso8601_interval(s: &str) -> Result<OwnedValue> {
    let mut months: i32 = 0;
    let mut days: i32 = 0;
    let mut micros: i64 = 0;

    let s = &s[1..];

    let (date_part, time_part) = if let Some(idx) = s.find('T') {
        (&s[..idx], Some(&s[idx + 1..]))
    } else {
        (s, None)
    };

    let mut current_num = String::new();
    for c in date_part.chars() {
        if c.is_ascii_digit() {
            current_num.push(c);
        } else if !current_num.is_empty() {
            let num: i32 = current_num
                .parse()
                .wrap_err_with(|| format!("invalid number in ISO interval: '{}'", current_num))?;
            current_num.clear();

            match c.to_ascii_uppercase() {
                'Y' => months += num * 12,
                'M' => months += num,
                'W' => days += num * 7,
                'D' => days += num,
                _ => bail!("unknown ISO 8601 interval designator: '{}'", c),
            }
        }
    }

    if let Some(time_str) = time_part {
        current_num.clear();
        for c in time_str.chars() {
            if c.is_ascii_digit() || c == '.' {
                current_num.push(c);
            } else if !current_num.is_empty() {
                let num: f64 = current_num.parse().wrap_err_with(|| {
                    format!("invalid number in ISO interval time: '{}'", current_num)
                })?;
                current_num.clear();

                match c.to_ascii_uppercase() {
                    'H' => micros += (num * 3600.0 * 1_000_000.0) as i64,
                    'M' => micros += (num * 60.0 * 1_000_000.0) as i64,
                    'S' => micros += (num * 1_000_000.0) as i64,
                    _ => bail!("unknown ISO 8601 time designator: '{}'", c),
                }
            }
        }
    }

    Ok(OwnedValue::Interval(micros, days, months))
}

fn parse_postgres_interval(s: &str) -> Result<OwnedValue> {
    let mut months: i32 = 0;
    let mut days: i32 = 0;
    let mut micros: i64 = 0;

    let s_lower = s.to_lowercase();
    let parts: Vec<&str> = s_lower.split_whitespace().collect();

    let mut i = 0;
    while i < parts.len() {
        let part = parts[i];

        if let Ok(num) = part.parse::<f64>() {
            if i + 1 < parts.len() {
                let unit = parts[i + 1];
                match unit {
                    "year" | "years" | "yr" | "yrs" => months += (num * 12.0) as i32,
                    "month" | "months" | "mon" | "mons" => months += num as i32,
                    "week" | "weeks" => days += (num * 7.0) as i32,
                    "day" | "days" => days += num as i32,
                    "hour" | "hours" | "hr" | "hrs" => {
                        micros += (num * 3600.0 * 1_000_000.0) as i64
                    }
                    "minute" | "minutes" | "min" | "mins" => {
                        micros += (num * 60.0 * 1_000_000.0) as i64
                    }
                    "second" | "seconds" | "sec" | "secs" => micros += (num * 1_000_000.0) as i64,
                    "millisecond" | "milliseconds" | "ms" => micros += (num * 1_000.0) as i64,
                    "microsecond" | "microseconds" | "us" => micros += num as i64,
                    _ => bail!("unknown interval unit: '{}'", unit),
                }
                i += 2;
                continue;
            }
        }

        i += 1;
    }

    Ok(OwnedValue::Interval(micros, days, months))
}
