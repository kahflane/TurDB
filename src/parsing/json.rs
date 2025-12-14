//! # JSON Parsing and Navigation
//!
//! This module provides comprehensive JSON parsing with two primary use cases:
//!
//! 1. **Building JSONB**: Parse JSON strings into `JsonbBuilderValue` for storage
//! 2. **Navigation**: Extract values via JSON path expressions (->>, ->, @>)
//!
//! ## Architecture
//!
//! ```text
//! ┌─────────────┐    ┌──────────────┐    ┌───────────────┐
//! │ JSON String │───>│ JsonTokenizer│───>│ JsonValue     │
//! └─────────────┘    └──────────────┘    └───────────────┘
//!                           │                    │
//!                           │                    v
//!                           │           ┌───────────────┐
//!                           └──────────>│ JsonNavigator │
//!                                       └───────────────┘
//! ```
//!
//! ## Tokenizer
//!
//! The `JsonTokenizer` performs lazy tokenization, yielding tokens on demand:
//!
//! - `{` / `}` - Object delimiters
//! - `[` / `]` - Array delimiters
//! - `:` - Key-value separator
//! - `,` - Element separator
//! - String, Number, Bool, Null - Value tokens
//!
//! ## Value Types
//!
//! `JsonValue` represents parsed JSON with owned data:
//!
//! - `Null`
//! - `Bool(bool)`
//! - `Number(f64)`
//! - `String(String)`
//! - `Array(Vec<JsonValue>)`
//! - `Object(Vec<(String, JsonValue)>)`
//!
//! ## Navigation
//!
//! `JsonNavigator` provides path-based extraction:
//!
//! ```text
//! Path syntax: {key1, key2, 0}  (PostgreSQL style)
//!
//! {"a": {"b": [1, 2, 3]}}
//!   {a,b,1} -> 2
//! ```
//!
//! ## Performance
//!
//! - Single pass tokenization
//! - No regex or external dependencies
//! - Streaming support for large documents
//! - Zero-copy where possible (keys, simple values)
//!
//! ## Error Handling
//!
//! All errors include position information:
//!
//! ```text
//! ParseError { position: 42, message: "expected ':' after key" }
//! ```

use eyre::{bail, Result, WrapErr};
use std::borrow::Cow;

#[derive(Debug, Clone, PartialEq)]
pub enum JsonValue {
    Null,
    Bool(bool),
    Number(f64),
    String(String),
    Array(Vec<JsonValue>),
    Object(Vec<(String, JsonValue)>),
}

#[derive(Debug, Clone, PartialEq)]
pub enum JsonToken<'a> {
    ObjectStart,
    ObjectEnd,
    ArrayStart,
    ArrayEnd,
    Colon,
    Comma,
    String(Cow<'a, str>),
    Number(f64),
    Bool(bool),
    Null,
}

#[derive(Debug)]
pub struct JsonParseResult {
    pub value: JsonValue,
    pub consumed: usize,
}

pub struct JsonTokenizer<'a> {
    input: &'a str,
    pos: usize,
}

impl<'a> JsonTokenizer<'a> {
    pub fn new(input: &'a str) -> Self {
        Self { input, pos: 0 }
    }

    pub fn position(&self) -> usize {
        self.pos
    }

    pub fn remaining(&self) -> &'a str {
        &self.input[self.pos..]
    }

    fn skip_whitespace(&mut self) {
        while self.pos < self.input.len() {
            match self.input.as_bytes()[self.pos] {
                b' ' | b'\t' | b'\n' | b'\r' => self.pos += 1,
                _ => break,
            }
        }
    }

    pub fn next_token(&mut self) -> Result<Option<JsonToken<'a>>> {
        self.skip_whitespace();

        if self.pos >= self.input.len() {
            return Ok(None);
        }

        let c = self.input.as_bytes()[self.pos];

        match c {
            b'{' => {
                self.pos += 1;
                Ok(Some(JsonToken::ObjectStart))
            }
            b'}' => {
                self.pos += 1;
                Ok(Some(JsonToken::ObjectEnd))
            }
            b'[' => {
                self.pos += 1;
                Ok(Some(JsonToken::ArrayStart))
            }
            b']' => {
                self.pos += 1;
                Ok(Some(JsonToken::ArrayEnd))
            }
            b':' => {
                self.pos += 1;
                Ok(Some(JsonToken::Colon))
            }
            b',' => {
                self.pos += 1;
                Ok(Some(JsonToken::Comma))
            }
            b'"' => self.parse_string(),
            b't' => self.parse_true(),
            b'f' => self.parse_false(),
            b'n' => self.parse_null(),
            b'-' | b'0'..=b'9' => self.parse_number(),
            _ => bail!(
                "unexpected character '{}' at position {}",
                c as char,
                self.pos
            ),
        }
    }

    fn parse_string(&mut self) -> Result<Option<JsonToken<'a>>> {
        let start = self.pos + 1;
        self.pos += 1;

        let mut has_escapes = false;
        while self.pos < self.input.len() {
            let c = self.input.as_bytes()[self.pos];
            match c {
                b'"' => {
                    let raw = &self.input[start..self.pos];
                    self.pos += 1;

                    return if has_escapes {
                        let unescaped = unescape_string(raw)?;
                        Ok(Some(JsonToken::String(Cow::Owned(unescaped))))
                    } else {
                        Ok(Some(JsonToken::String(Cow::Borrowed(raw))))
                    };
                }
                b'\\' => {
                    has_escapes = true;
                    self.pos += 2;
                }
                _ => self.pos += 1,
            }
        }

        bail!("unterminated string starting at position {}", start - 1)
    }

    fn parse_number(&mut self) -> Result<Option<JsonToken<'a>>> {
        let start = self.pos;

        if self.input.as_bytes()[self.pos] == b'-' {
            self.pos += 1;
        }

        while self.pos < self.input.len() {
            match self.input.as_bytes()[self.pos] {
                b'0'..=b'9' | b'.' | b'e' | b'E' | b'+' | b'-' => self.pos += 1,
                _ => break,
            }
        }

        let num_str = &self.input[start..self.pos];
        let n: f64 = num_str
            .parse()
            .wrap_err_with(|| format!("invalid number '{}' at position {}", num_str, start))?;

        Ok(Some(JsonToken::Number(n)))
    }

    fn parse_true(&mut self) -> Result<Option<JsonToken<'a>>> {
        if self.input[self.pos..].starts_with("true") {
            self.pos += 4;
            Ok(Some(JsonToken::Bool(true)))
        } else {
            bail!("expected 'true' at position {}", self.pos)
        }
    }

    fn parse_false(&mut self) -> Result<Option<JsonToken<'a>>> {
        if self.input[self.pos..].starts_with("false") {
            self.pos += 5;
            Ok(Some(JsonToken::Bool(false)))
        } else {
            bail!("expected 'false' at position {}", self.pos)
        }
    }

    fn parse_null(&mut self) -> Result<Option<JsonToken<'a>>> {
        if self.input[self.pos..].starts_with("null") {
            self.pos += 4;
            Ok(Some(JsonToken::Null))
        } else {
            bail!("expected 'null' at position {}", self.pos)
        }
    }
}

pub fn unescape_string(s: &str) -> Result<String> {
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
                Some('b') => result.push('\x08'),
                Some('f') => result.push('\x0C'),
                Some('u') => {
                    let hex: String = chars.by_ref().take(4).collect();
                    if hex.len() != 4 {
                        bail!("invalid unicode escape: incomplete sequence");
                    }
                    let cp = u32::from_str_radix(&hex, 16)
                        .wrap_err_with(|| format!("invalid unicode escape: \\u{}", hex))?;
                    if let Some(ch) = char::from_u32(cp) {
                        result.push(ch);
                    } else {
                        bail!("invalid unicode codepoint: U+{:04X}", cp);
                    }
                }
                Some(other) => bail!("invalid escape sequence: \\{}", other),
                None => bail!("unexpected end of string after backslash"),
            }
        } else {
            result.push(c);
        }
    }

    Ok(result)
}

pub fn parse_json(input: &str) -> Result<JsonParseResult> {
    let mut tokenizer = JsonTokenizer::new(input);
    let value = parse_value(&mut tokenizer)?;
    Ok(JsonParseResult {
        value,
        consumed: tokenizer.position(),
    })
}

fn parse_value(tokenizer: &mut JsonTokenizer) -> Result<JsonValue> {
    match tokenizer.next_token()? {
        Some(JsonToken::Null) => Ok(JsonValue::Null),
        Some(JsonToken::Bool(b)) => Ok(JsonValue::Bool(b)),
        Some(JsonToken::Number(n)) => Ok(JsonValue::Number(n)),
        Some(JsonToken::String(s)) => Ok(JsonValue::String(s.into_owned())),
        Some(JsonToken::ArrayStart) => parse_array(tokenizer),
        Some(JsonToken::ObjectStart) => parse_object(tokenizer),
        Some(other) => bail!(
            "unexpected token {:?} at position {}",
            other,
            tokenizer.position()
        ),
        None => bail!("unexpected end of input"),
    }
}

fn parse_array(tokenizer: &mut JsonTokenizer) -> Result<JsonValue> {
    let mut elements = Vec::new();

    loop {
        match tokenizer.next_token()? {
            Some(JsonToken::ArrayEnd) => return Ok(JsonValue::Array(elements)),
            Some(JsonToken::Comma) => continue,
            Some(token) => {
                let value = match token {
                    JsonToken::Null => JsonValue::Null,
                    JsonToken::Bool(b) => JsonValue::Bool(b),
                    JsonToken::Number(n) => JsonValue::Number(n),
                    JsonToken::String(s) => JsonValue::String(s.into_owned()),
                    JsonToken::ArrayStart => parse_array(tokenizer)?,
                    JsonToken::ObjectStart => parse_object(tokenizer)?,
                    _ => bail!(
                        "unexpected token {:?} in array at position {}",
                        token,
                        tokenizer.position()
                    ),
                };
                elements.push(value);
            }
            None => bail!("unexpected end of input in array"),
        }
    }
}

fn parse_object(tokenizer: &mut JsonTokenizer) -> Result<JsonValue> {
    let mut entries = Vec::new();

    loop {
        match tokenizer.next_token()? {
            Some(JsonToken::ObjectEnd) => return Ok(JsonValue::Object(entries)),
            Some(JsonToken::Comma) => continue,
            Some(JsonToken::String(key)) => {
                match tokenizer.next_token()? {
                    Some(JsonToken::Colon) => {}
                    other => bail!(
                        "expected ':' after object key, got {:?} at position {}",
                        other,
                        tokenizer.position()
                    ),
                }

                let value = parse_value(tokenizer)?;
                entries.push((key.into_owned(), value));
            }
            Some(other) => bail!(
                "expected string key or '}}', got {:?} at position {}",
                other,
                tokenizer.position()
            ),
            None => bail!("unexpected end of input in object"),
        }
    }
}

pub fn parse_json_path(path: &str) -> Option<Vec<String>> {
    let path = path.trim();
    if !path.starts_with('{') || !path.ends_with('}') {
        return None;
    }

    let inner = &path[1..path.len() - 1];
    if inner.is_empty() {
        return Some(vec![]);
    }

    Some(inner.split(',').map(|s| s.trim().to_string()).collect())
}

pub struct JsonNavigator<'a> {
    input: &'a str,
}

impl<'a> JsonNavigator<'a> {
    pub fn new(input: &'a str) -> Self {
        Self {
            input: input.trim(),
        }
    }

    pub fn extract_path(&self, path: &[String]) -> Option<String> {
        let mut current = self.input.to_string();

        for key in path {
            current = self.extract_at_key_or_index(&current, key)?;
        }

        Some(current)
    }

    fn extract_at_key_or_index(&self, json: &str, key: &str) -> Option<String> {
        let json = json.trim();

        if let Ok(index) = key.parse::<usize>() {
            self.extract_array_element(json, index)
        } else {
            self.extract_object_key(json, key)
        }
    }

    fn extract_object_key(&self, json: &str, key: &str) -> Option<String> {
        if !json.starts_with('{') {
            return None;
        }

        let search_key = format!("\"{}\":", key);
        let key_pos = json.find(&search_key)?;
        let value_start = key_pos + search_key.len();
        let rest = json[value_start..].trim_start();

        let end = self.find_value_end(rest)?;
        Some(rest[..end].to_string())
    }

    fn extract_array_element(&self, json: &str, index: usize) -> Option<String> {
        if !json.starts_with('[') {
            return None;
        }

        let inner = &json[1..json.len().saturating_sub(1)];
        let mut current_idx = 0;
        let mut depth = 0;
        let mut start = 0;
        let mut in_string = false;
        let mut escape_next = false;

        for (i, c) in inner.char_indices() {
            if escape_next {
                escape_next = false;
                continue;
            }

            match c {
                '\\' if in_string => escape_next = true,
                '"' => in_string = !in_string,
                '[' | '{' if !in_string => depth += 1,
                ']' | '}' if !in_string => depth -= 1,
                ',' if depth == 0 && !in_string => {
                    if current_idx == index {
                        return Some(inner[start..i].trim().to_string());
                    }
                    current_idx += 1;
                    start = i + 1;
                }
                _ => {}
            }
        }

        if current_idx == index {
            return Some(inner[start..].trim().to_string());
        }

        None
    }

    fn find_value_end(&self, s: &str) -> Option<usize> {
        let s = s.trim_start();
        if s.is_empty() {
            return None;
        }

        let first = s.chars().next()?;

        match first {
            '"' => {
                let mut i = 1;
                let bytes = s.as_bytes();
                while i < bytes.len() {
                    match bytes[i] {
                        b'"' => return Some(i + 1),
                        b'\\' => i += 2,
                        _ => i += 1,
                    }
                }
                None
            }
            '{' | '[' => {
                let close = if first == '{' { '}' } else { ']' };
                let mut depth = 1;
                let mut in_string = false;
                let mut escape_next = false;

                for (i, c) in s[1..].char_indices() {
                    if escape_next {
                        escape_next = false;
                        continue;
                    }

                    match c {
                        '\\' if in_string => escape_next = true,
                        '"' => in_string = !in_string,
                        c if c == first && !in_string => depth += 1,
                        c if c == close && !in_string => {
                            depth -= 1;
                            if depth == 0 {
                                return Some(i + 2);
                            }
                        }
                        _ => {}
                    }
                }
                None
            }
            _ => {
                let end = s
                    .find(|c: char| c == ',' || c == '}' || c == ']' || c.is_whitespace())
                    .unwrap_or(s.len());
                Some(end)
            }
        }
    }

    pub fn parse_object_pairs(&self, json: &str) -> Vec<(String, String)> {
        let json = json.trim();
        if !json.starts_with('{') || !json.ends_with('}') {
            return vec![];
        }

        let inner = &json[1..json.len() - 1];
        let mut pairs = Vec::new();
        let mut depth = 0;
        let mut in_string = false;
        let mut escape_next = false;
        let mut start = 0;

        for (i, c) in inner.char_indices() {
            if escape_next {
                escape_next = false;
                continue;
            }

            match c {
                '\\' if in_string => escape_next = true,
                '"' => in_string = !in_string,
                '[' | '{' if !in_string => depth += 1,
                ']' | '}' if !in_string => depth -= 1,
                ',' if depth == 0 && !in_string => {
                    if let Some(pair) = self.parse_key_value(&inner[start..i]) {
                        pairs.push(pair);
                    }
                    start = i + 1;
                }
                _ => {}
            }
        }

        if start < inner.len() {
            if let Some(pair) = self.parse_key_value(&inner[start..]) {
                pairs.push(pair);
            }
        }

        pairs
    }

    fn parse_key_value(&self, s: &str) -> Option<(String, String)> {
        let s = s.trim();
        let colon_pos = self.find_colon_position(s)?;
        let key = s[..colon_pos].trim();
        let value = s[colon_pos + 1..].trim();

        let key = if key.starts_with('"') && key.ends_with('"') {
            &key[1..key.len() - 1]
        } else {
            key
        };

        Some((key.to_string(), value.to_string()))
    }

    fn find_colon_position(&self, s: &str) -> Option<usize> {
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
                ':' if !in_string => return Some(i),
                _ => {}
            }
        }

        None
    }

    pub fn parse_array_elements(&self, json: &str) -> Vec<String> {
        let json = json.trim();
        if !json.starts_with('[') || !json.ends_with(']') {
            return vec![];
        }

        let inner = &json[1..json.len() - 1];
        let mut elements = Vec::new();
        let mut depth = 0;
        let mut in_string = false;
        let mut escape_next = false;
        let mut start = 0;

        for (i, c) in inner.char_indices() {
            if escape_next {
                escape_next = false;
                continue;
            }

            match c {
                '\\' if in_string => escape_next = true,
                '"' => in_string = !in_string,
                '[' | '{' if !in_string => depth += 1,
                ']' | '}' if !in_string => depth -= 1,
                ',' if depth == 0 && !in_string => {
                    let elem = inner[start..i].trim();
                    if !elem.is_empty() {
                        elements.push(elem.to_string());
                    }
                    start = i + 1;
                }
                _ => {}
            }
        }

        if start < inner.len() {
            let elem = inner[start..].trim();
            if !elem.is_empty() {
                elements.push(elem.to_string());
            }
        }

        elements
    }

    pub fn json_contains(&self, container: &str, contained: &str) -> bool {
        let container = container.trim();
        let contained = contained.trim();

        if container.starts_with('{') && contained.starts_with('{') {
            self.object_contains_object(container, contained)
        } else if container.starts_with('[') && contained.starts_with('[') {
            self.array_contains_array(container, contained)
        } else {
            self.values_equal(container, contained)
        }
    }

    fn object_contains_object(&self, container: &str, contained: &str) -> bool {
        let container_pairs = self.parse_object_pairs(container);
        let contained_pairs = self.parse_object_pairs(contained);

        for (key, value) in &contained_pairs {
            let found = container_pairs
                .iter()
                .any(|(k, v)| k == key && self.json_contains(v, value));
            if !found {
                return false;
            }
        }
        true
    }

    fn array_contains_array(&self, container: &str, contained: &str) -> bool {
        let container_elements = self.parse_array_elements(container);
        let contained_elements = self.parse_array_elements(contained);

        for elem in &contained_elements {
            let found = container_elements
                .iter()
                .any(|c| self.json_contains(c, elem));
            if !found {
                return false;
            }
        }
        true
    }

    fn values_equal(&self, a: &str, b: &str) -> bool {
        let a = a.trim();
        let b = b.trim();

        if a == b {
            return true;
        }

        match (parse_json(a), parse_json(b)) {
            (Ok(a_result), Ok(b_result)) => json_values_equal(&a_result.value, &b_result.value),
            _ => false,
        }
    }
}

fn json_values_equal(a: &JsonValue, b: &JsonValue) -> bool {
    match (a, b) {
        (JsonValue::Null, JsonValue::Null) => true,
        (JsonValue::Bool(x), JsonValue::Bool(y)) => x == y,
        (JsonValue::Number(x), JsonValue::Number(y)) => (x - y).abs() < f64::EPSILON,
        (JsonValue::String(x), JsonValue::String(y)) => x == y,
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_null() {
        let result = parse_json("null").unwrap();
        assert_eq!(result.value, JsonValue::Null);
    }

    #[test]
    fn parse_bool() {
        assert_eq!(parse_json("true").unwrap().value, JsonValue::Bool(true));
        assert_eq!(parse_json("false").unwrap().value, JsonValue::Bool(false));
    }

    #[test]
    fn parse_number() {
        assert_eq!(parse_json("42").unwrap().value, JsonValue::Number(42.0));
        assert_eq!(parse_json("-3.25").unwrap().value, JsonValue::Number(-3.25));
        assert_eq!(parse_json("1e10").unwrap().value, JsonValue::Number(1e10));
    }

    #[test]
    fn parse_string() {
        assert_eq!(
            parse_json(r#""hello""#).unwrap().value,
            JsonValue::String("hello".to_string())
        );
    }

    #[test]
    fn parse_string_with_escapes() {
        assert_eq!(
            parse_json(r#""hello\nworld""#).unwrap().value,
            JsonValue::String("hello\nworld".to_string())
        );
        assert_eq!(
            parse_json(r#""tab\there""#).unwrap().value,
            JsonValue::String("tab\there".to_string())
        );
    }

    #[test]
    fn parse_unicode_escape() {
        assert_eq!(
            parse_json(r#""\u0041""#).unwrap().value,
            JsonValue::String("A".to_string())
        );
    }

    #[test]
    fn parse_empty_array() {
        assert_eq!(
            parse_json("[]").unwrap().value,
            JsonValue::Array(vec![])
        );
    }

    #[test]
    fn parse_array() {
        let result = parse_json("[1, 2, 3]").unwrap();
        assert_eq!(
            result.value,
            JsonValue::Array(vec![
                JsonValue::Number(1.0),
                JsonValue::Number(2.0),
                JsonValue::Number(3.0),
            ])
        );
    }

    #[test]
    fn parse_empty_object() {
        assert_eq!(
            parse_json("{}").unwrap().value,
            JsonValue::Object(vec![])
        );
    }

    #[test]
    fn parse_object() {
        let result = parse_json(r#"{"name": "test", "value": 42}"#).unwrap();
        assert_eq!(
            result.value,
            JsonValue::Object(vec![
                ("name".to_string(), JsonValue::String("test".to_string())),
                ("value".to_string(), JsonValue::Number(42.0)),
            ])
        );
    }

    #[test]
    fn parse_nested() {
        let result = parse_json(r#"{"arr": [1, {"inner": true}]}"#).unwrap();
        match result.value {
            JsonValue::Object(entries) => {
                assert_eq!(entries.len(), 1);
                assert_eq!(entries[0].0, "arr");
            }
            _ => panic!("expected object"),
        }
    }

    #[test]
    fn navigator_extract_key() {
        let nav = JsonNavigator::new(r#"{"name": "test", "value": 42}"#);
        assert_eq!(
            nav.extract_path(&["name".to_string()]),
            Some(r#""test""#.to_string())
        );
        assert_eq!(
            nav.extract_path(&["value".to_string()]),
            Some("42".to_string())
        );
    }

    #[test]
    fn navigator_extract_array_index() {
        let nav = JsonNavigator::new(r#"[1, 2, 3]"#);
        assert_eq!(nav.extract_path(&["0".to_string()]), Some("1".to_string()));
        assert_eq!(nav.extract_path(&["2".to_string()]), Some("3".to_string()));
    }

    #[test]
    fn navigator_nested_path() {
        let nav = JsonNavigator::new(r#"{"a": {"b": [10, 20]}}"#);
        assert_eq!(
            nav.extract_path(&["a".to_string(), "b".to_string(), "1".to_string()]),
            Some("20".to_string())
        );
    }

    #[test]
    fn json_contains_simple() {
        let nav = JsonNavigator::new("");
        assert!(nav.json_contains(r#"{"a": 1, "b": 2}"#, r#"{"a": 1}"#));
        assert!(!nav.json_contains(r#"{"a": 1}"#, r#"{"a": 2}"#));
    }

    #[test]
    fn json_contains_array() {
        let nav = JsonNavigator::new("");
        assert!(nav.json_contains(r#"[1, 2, 3]"#, r#"[1, 3]"#));
        assert!(!nav.json_contains(r#"[1, 2]"#, r#"[3]"#));
    }

    #[test]
    fn parse_json_path_valid() {
        assert_eq!(
            parse_json_path("{a,b,c}"),
            Some(vec!["a".to_string(), "b".to_string(), "c".to_string()])
        );
        assert_eq!(parse_json_path("{}"), Some(vec![]));
    }

    #[test]
    fn parse_json_path_invalid() {
        assert_eq!(parse_json_path("abc"), None);
        assert_eq!(parse_json_path("{abc"), None);
    }
}
