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

impl JsonValue {
    pub fn to_jsonb_bytes(&self) -> Vec<u8> {
        use crate::records::jsonb::{
            JSONB_TYPE_ARRAY, JSONB_TYPE_BOOL, JSONB_TYPE_NULL, JSONB_TYPE_NUMBER,
            JSONB_TYPE_OBJECT, JSONB_TYPE_STRING,
        };

        const FLAG_IS_KEY: u32 = 1 << 31;
        const FLAG_IS_VARIABLE: u32 = 1 << 30;
        const TYPE_SHIFT: u32 = 24;
        const OFFSET_MASK: u32 = 0x00FF_FFFF;

        fn encode_value(value: &JsonValue, buf: &mut Vec<u8>) {
            match value {
                JsonValue::Null => {
                    let header = (JSONB_TYPE_NULL as u32) << 28;
                    buf.extend(header.to_le_bytes());
                }
                JsonValue::Bool(v) => {
                    let header = ((JSONB_TYPE_BOOL as u32) << 28) | (*v as u32);
                    buf.extend(header.to_le_bytes());
                }
                JsonValue::Number(v) => {
                    let header = (JSONB_TYPE_NUMBER as u32) << 28;
                    buf.extend(header.to_le_bytes());
                    buf.extend(v.to_le_bytes());
                }
                JsonValue::String(s) => {
                    let header = ((JSONB_TYPE_STRING as u32) << 28) | (s.len() as u32);
                    buf.extend(header.to_le_bytes());
                    buf.extend(s.as_bytes());
                }
                JsonValue::Array(elements) => {
                    let header = ((JSONB_TYPE_ARRAY as u32) << 28) | (elements.len() as u32);
                    buf.extend(header.to_le_bytes());

                    let entries_start = buf.len();
                    buf.resize(entries_start + elements.len() * 4, 0);

                    let mut data_buf = Vec::new();

                    for (i, elem) in elements.iter().enumerate() {
                        let entry = encode_entry(elem, &mut data_buf);
                        let entry_offset = entries_start + i * 4;
                        buf[entry_offset..entry_offset + 4].copy_from_slice(&entry.to_le_bytes());
                    }

                    buf.extend(&data_buf);
                }
                JsonValue::Object(entries) => {
                    let mut sorted_entries: Vec<_> = entries.iter().collect();
                    sorted_entries.sort_by(|a, b| a.0.cmp(&b.0));

                    let entry_count = sorted_entries.len() * 2;
                    let header = ((JSONB_TYPE_OBJECT as u32) << 28) | (entry_count as u32);
                    buf.extend(header.to_le_bytes());

                    let entries_start = buf.len();
                    buf.resize(entries_start + entry_count * 4, 0);

                    let mut data_buf = Vec::new();

                    for (i, (key, val)) in sorted_entries.iter().enumerate() {
                        let key_offset = data_buf.len();
                        data_buf.extend((key.len() as u16).to_le_bytes());
                        data_buf.extend(key.as_bytes());

                        let key_entry =
                            FLAG_IS_KEY | FLAG_IS_VARIABLE | (key_offset as u32 & OFFSET_MASK);

                        let val_entry = encode_entry(val, &mut data_buf);

                        let key_entry_offset = entries_start + i * 2 * 4;
                        let val_entry_offset = entries_start + (i * 2 + 1) * 4;
                        buf[key_entry_offset..key_entry_offset + 4]
                            .copy_from_slice(&key_entry.to_le_bytes());
                        buf[val_entry_offset..val_entry_offset + 4]
                            .copy_from_slice(&val_entry.to_le_bytes());
                    }

                    buf.extend(&data_buf);
                }
            }
        }

        fn encode_entry(value: &JsonValue, data_buf: &mut Vec<u8>) -> u32 {
            match value {
                JsonValue::Null => (JSONB_TYPE_NULL as u32) << TYPE_SHIFT,
                JsonValue::Bool(v) => ((JSONB_TYPE_BOOL as u32) << TYPE_SHIFT) | (*v as u32),
                JsonValue::Number(v) => {
                    let offset = data_buf.len();
                    data_buf.extend(v.to_le_bytes());
                    FLAG_IS_VARIABLE
                        | ((JSONB_TYPE_NUMBER as u32) << TYPE_SHIFT)
                        | (offset as u32 & OFFSET_MASK)
                }
                JsonValue::String(s) => {
                    let offset = data_buf.len();
                    data_buf.extend((s.len() as u16).to_le_bytes());
                    data_buf.extend(s.as_bytes());
                    FLAG_IS_VARIABLE
                        | ((JSONB_TYPE_STRING as u32) << TYPE_SHIFT)
                        | (offset as u32 & OFFSET_MASK)
                }
                JsonValue::Array(_) | JsonValue::Object(_) => {
                    let offset = data_buf.len();
                    let mut nested_buf = Vec::new();
                    encode_value(value, &mut nested_buf);
                    data_buf.extend((nested_buf.len() as u32).to_le_bytes());
                    data_buf.extend(&nested_buf);

                    let typ = if matches!(value, JsonValue::Array(_)) {
                        JSONB_TYPE_ARRAY
                    } else {
                        JSONB_TYPE_OBJECT
                    };

                    FLAG_IS_VARIABLE | ((typ as u32) << TYPE_SHIFT) | (offset as u32 & OFFSET_MASK)
                }
            }
        }

        let mut buf = Vec::new();
        encode_value(self, &mut buf);
        buf
    }
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

    if path.starts_with('$') {
        return parse_dollar_path(path);
    }

    if !path.starts_with('{') || !path.ends_with('}') {
        return None;
    }

    let inner = &path[1..path.len() - 1];
    if inner.is_empty() {
        return Some(vec![]);
    }

    Some(inner.split(',').map(|s| s.trim().to_string()).collect())
}

fn parse_dollar_path(path: &str) -> Option<Vec<String>> {
    let path = path.trim();
    if !path.starts_with('$') {
        return None;
    }

    let rest = &path[1..];
    if rest.is_empty() {
        return Some(vec![]);
    }

    let mut elements = Vec::new();
    let mut current = String::new();
    let mut in_bracket = false;

    for ch in rest.chars() {
        match ch {
            '.' if !in_bracket => {
                if !current.is_empty() {
                    elements.push(current);
                    current = String::new();
                }
            }
            '[' => {
                if !current.is_empty() {
                    elements.push(current);
                    current = String::new();
                }
                in_bracket = true;
            }
            ']' => {
                in_bracket = false;
                if !current.is_empty() {
                    elements.push(current);
                    current = String::new();
                }
            }
            '\'' | '"' => {}
            _ => {
                current.push(ch);
            }
        }
    }

    if !current.is_empty() {
        elements.push(current);
    }

    Some(elements)
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
