//! # SQL Literal and JSON Parsing
//!
//! This module provides unified parsing for SQL literals and JSON data,
//! consolidating duplicate parsing logic from database.rs and predicate.rs.
//!
//! ## Module Structure
//!
//! - `json`: JSON tokenizer and parser with zero-copy support
//! - `literal`: SQL literal parsing (UUID, blob, vector, dates)
//!
//! ## Design Goals
//!
//! 1. **Single Implementation**: One parser for JSON, one for each literal type
//! 2. **Zero-Copy Where Possible**: Return references when input allows
//! 3. **Error Context**: Rich error messages with position information
//! 4. **Streaming Support**: Parse incrementally for large documents
//!
//! ## JSON Parsing
//!
//! The JSON parser supports two modes:
//!
//! - **Build Mode**: Constructs `JsonbBuilderValue` for INSERT/UPDATE
//! - **Navigate Mode**: Traverses JSON for path extraction (->>, ->, @>)
//!
//! ```ignore
//! use turdb::parsing::json::{parse_json, JsonValue};
//!
//! let value = parse_json(r#"{"name": "test"}"#)?;
//! ```
//!
//! ## SQL Literal Parsing
//!
//! Parses typed SQL literals to `OwnedValue`:
//!
//! ```ignore
//! use turdb::parsing::literal::{parse_uuid, parse_vector, parse_blob};
//!
//! let uuid = parse_uuid("550e8400-e29b-41d4-a716-446655440000")?;
//! let vec = parse_vector("[1.0, 2.0, 3.0]")?;
//! let blob = parse_blob("\\x48454C4C4F")?;
//! ```
//!
//! ## Shared Utilities
//!
//! Common parsing primitives used by both JSON and literal parsers:
//!
//! - `skip_whitespace`: Advance past whitespace
//! - `unescape_string`: Handle JSON escape sequences
//! - `parse_number`: Parse int/float with type inference
//!
//! ## Error Handling
//!
//! All parsing functions return `eyre::Result` with context:
//!
//! ```ignore
//! // Error includes position and context
//! // "invalid JSON at position 15: expected ':' after object key"
//! ```

mod json;
mod literal;

pub use json::{
    parse_json, parse_json_path, JsonNavigator, JsonParseResult, JsonToken, JsonTokenizer,
    JsonValue,
};
pub use literal::{
    parse_binary_blob, parse_date, parse_hex_blob, parse_interval, parse_time, parse_timestamp,
    parse_uuid, parse_vector, LiteralParser, ParsedLiteral,
};
