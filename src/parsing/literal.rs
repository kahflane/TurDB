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

        if (s.starts_with('\'') && s.ends_with('\''))
            || (s.starts_with('"') && s.ends_with('"'))
        {
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
                let i: i64 = s.parse().wrap_err_with(|| format!("invalid integer: '{}'", s))?;
                Ok(OwnedValue::Int(i))
            }
            "float" | "float4" | "float8" | "double" | "real" => {
                let f: f64 = s.parse().wrap_err_with(|| format!("invalid float: '{}'", s))?;
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_uuid_standard() {
        let result = parse_uuid("550e8400-e29b-41d4-a716-446655440000").unwrap();
        match result {
            OwnedValue::Uuid(bytes) => {
                assert_eq!(bytes[0], 0x55);
                assert_eq!(bytes[1], 0x0e);
            }
            _ => panic!("expected UUID"),
        }
    }

    #[test]
    fn parse_uuid_compact() {
        let result = parse_uuid("550e8400e29b41d4a716446655440000").unwrap();
        match result {
            OwnedValue::Uuid(bytes) => {
                assert_eq!(bytes[0], 0x55);
            }
            _ => panic!("expected UUID"),
        }
    }

    #[test]
    fn parse_uuid_invalid_length() {
        let result = parse_uuid("550e8400");
        assert!(result.is_err());
    }

    #[test]
    fn parse_vector_bracketed() {
        let result = parse_vector("[1.0, 2.0, 3.0]").unwrap();
        match result {
            OwnedValue::Vector(v) => {
                assert_eq!(v, vec![1.0, 2.0, 3.0]);
            }
            _ => panic!("expected Vector"),
        }
    }

    #[test]
    fn parse_vector_unbracketed() {
        let result = parse_vector("1.0, 2.0, 3.0").unwrap();
        match result {
            OwnedValue::Vector(v) => {
                assert_eq!(v, vec![1.0, 2.0, 3.0]);
            }
            _ => panic!("expected Vector"),
        }
    }

    #[test]
    fn parse_vector_empty() {
        let result = parse_vector("[]").unwrap();
        match result {
            OwnedValue::Vector(v) => {
                assert!(v.is_empty());
            }
            _ => panic!("expected Vector"),
        }
    }

    #[test]
    fn parse_hex_blob_valid() {
        let result = parse_hex_blob("48454C4C4F").unwrap();
        match result {
            OwnedValue::Blob(b) => {
                assert_eq!(b, b"HELLO");
            }
            _ => panic!("expected Blob"),
        }
    }

    #[test]
    fn parse_hex_blob_odd_length() {
        let result = parse_hex_blob("48454C4C4");
        assert!(result.is_err());
    }

    #[test]
    fn parse_binary_blob_valid() {
        let result = parse_binary_blob("01001000").unwrap();
        match result {
            OwnedValue::Blob(b) => {
                assert_eq!(b, vec![0x48]);
            }
            _ => panic!("expected Blob"),
        }
    }

    #[test]
    fn literal_parser_null() {
        let parser = LiteralParser::new();
        assert_eq!(parser.parse("null").unwrap(), ParsedLiteral::Null);
        assert_eq!(parser.parse("NULL").unwrap(), ParsedLiteral::Null);
    }

    #[test]
    fn literal_parser_bool() {
        let parser = LiteralParser::new();
        assert_eq!(parser.parse("true").unwrap(), ParsedLiteral::Bool(true));
        assert_eq!(parser.parse("FALSE").unwrap(), ParsedLiteral::Bool(false));
    }

    #[test]
    fn literal_parser_int() {
        let parser = LiteralParser::new();
        assert_eq!(parser.parse("42").unwrap(), ParsedLiteral::Int(42));
        assert_eq!(parser.parse("-100").unwrap(), ParsedLiteral::Int(-100));
    }

    #[test]
    fn literal_parser_float() {
        let parser = LiteralParser::new();
        assert_eq!(parser.parse("3.25").unwrap(), ParsedLiteral::Float(3.25));
    }

    #[test]
    fn literal_parser_text() {
        let parser = LiteralParser::new();
        assert_eq!(
            parser.parse("'hello'").unwrap(),
            ParsedLiteral::Text("hello".to_string())
        );
    }

    #[test]
    fn literal_parser_typed_uuid() {
        let parser = LiteralParser::new();
        let result = parser
            .parse_typed("550e8400-e29b-41d4-a716-446655440000", "uuid")
            .unwrap();
        assert!(matches!(result, OwnedValue::Uuid(_)));
    }

    #[test]
    fn literal_parser_typed_vector() {
        let parser = LiteralParser::new();
        let result = parser.parse_typed("[1.0, 2.0]", "vector").unwrap();
        assert!(matches!(result, OwnedValue::Vector(_)));
    }

    #[test]
    fn literal_parser_typed_blob_hex() {
        let parser = LiteralParser::new();
        let result = parser.parse_typed("\\x48454C4C4F", "bytea").unwrap();
        match result {
            OwnedValue::Blob(b) => assert_eq!(b, b"HELLO"),
            _ => panic!("expected Blob"),
        }
    }

    #[test]
    fn parse_date_iso8601() {
        let result = parse_date("2024-01-15").unwrap();
        match result {
            OwnedValue::Date(days) => {
                assert_eq!(days, 19737);
            }
            _ => panic!("Expected Date"),
        }
    }

    #[test]
    fn parse_date_epoch() {
        let result = parse_date("1970-01-01").unwrap();
        match result {
            OwnedValue::Date(days) => {
                assert_eq!(days, 0);
            }
            _ => panic!("Expected Date"),
        }
    }

    #[test]
    fn parse_date_invalid() {
        assert!(parse_date("not-a-date").is_err());
        assert!(parse_date("2024-13-01").is_err());
        assert!(parse_date("2024-01-32").is_err());
    }

    #[test]
    fn parse_time_iso8601() {
        let result = parse_time("13:45:30").unwrap();
        match result {
            OwnedValue::Time(micros) => {
                let expected = (13 * 3600 + 45 * 60 + 30) * 1_000_000i64;
                assert_eq!(micros, expected);
            }
            _ => panic!("Expected Time"),
        }
    }

    #[test]
    fn parse_time_with_micros() {
        let result = parse_time("13:45:30.123456").unwrap();
        match result {
            OwnedValue::Time(micros) => {
                let expected = (13 * 3600 + 45 * 60 + 30) * 1_000_000i64 + 123456;
                assert_eq!(micros, expected);
            }
            _ => panic!("Expected Time"),
        }
    }

    #[test]
    fn parse_time_midnight() {
        let result = parse_time("00:00:00").unwrap();
        match result {
            OwnedValue::Time(micros) => {
                assert_eq!(micros, 0);
            }
            _ => panic!("Expected Time"),
        }
    }

    #[test]
    fn parse_timestamp_iso8601_t() {
        let result = parse_timestamp("2024-01-15T13:45:30").unwrap();
        match result {
            OwnedValue::Timestamp(micros) => {
                let expected_days = 19737i64;
                let expected_time = (13 * 3600 + 45 * 60 + 30) * 1_000_000i64;
                let expected = expected_days * 86400 * 1_000_000 + expected_time;
                assert_eq!(micros, expected);
            }
            _ => panic!("Expected Timestamp"),
        }
    }

    #[test]
    fn parse_timestamp_iso8601_space() {
        let result = parse_timestamp("2024-01-15 13:45:30").unwrap();
        match result {
            OwnedValue::Timestamp(micros) => {
                let expected_days = 19737i64;
                let expected_time = (13 * 3600 + 45 * 60 + 30) * 1_000_000i64;
                let expected = expected_days * 86400 * 1_000_000 + expected_time;
                assert_eq!(micros, expected);
            }
            _ => panic!("Expected Timestamp"),
        }
    }

    #[test]
    fn parse_timestamp_epoch() {
        let result = parse_timestamp("1970-01-01T00:00:00").unwrap();
        match result {
            OwnedValue::Timestamp(micros) => {
                assert_eq!(micros, 0);
            }
            _ => panic!("Expected Timestamp"),
        }
    }
}
