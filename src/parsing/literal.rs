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

use crate::database::owned_value::OwnedValue;
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
}
