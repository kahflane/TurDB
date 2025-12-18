//! # JSONB Binary Format with O(log n) Key Lookup
//!
//! This module provides JSONB storage with efficient key access using binary search.
//!
//! ## Binary Format
//!
//! ```text
//! +----------+------------------+------------------+
//! | Header   | Entries Table    | Data             |
//! | (u32)    | [Entry; N]       | [u8; ...]        |
//! +----------+------------------+------------------+
//!
//! Header (4 bytes):
//!   Bits 31-28: Type tag (0=object, 1=array, 2=null, 3=bool, 4=number, 5=string)
//!   Bits 27-0: Entry count (for object: key-value pairs * 2, for array: elements)
//!
//! Entry (4 bytes):
//!   Bit 31: Is_Key (1 = key string, 0 = value)
//!   Bit 30: Is_Variable (1 = variable length)
//!   Bits 29-24: Type tag
//!   Bits 23-0: Offset or inline value
//! ```
//!
//! ## Key Features
//!
//! - **O(log n) key lookup**: Binary search on sorted keys
//! - **Zero-copy access**: JsonbView returns references into original buffer
//! - **Nested structures**: Recursive encoding for arrays and objects
//! - **Inline scalars**: Booleans stored inline in entry, no data section needed

use eyre::{bail, ensure, Result};
use std::cmp::Ordering;

pub const JSONB_TYPE_OBJECT: u8 = 0;
pub const JSONB_TYPE_ARRAY: u8 = 1;
pub const JSONB_TYPE_NULL: u8 = 2;
pub const JSONB_TYPE_BOOL: u8 = 3;
pub const JSONB_TYPE_NUMBER: u8 = 4;
pub const JSONB_TYPE_STRING: u8 = 5;

const FLAG_IS_KEY: u32 = 1 << 31;
const FLAG_IS_VARIABLE: u32 = 1 << 30;
const TYPE_SHIFT: u32 = 24;
const TYPE_MASK: u32 = 0x3F << TYPE_SHIFT;
const OFFSET_MASK: u32 = 0x00FF_FFFF;

#[derive(Debug, Clone, PartialEq)]
pub enum JsonbValue<'a> {
    Null,
    Bool(bool),
    Number(f64),
    String(&'a str),
    Array(JsonbView<'a>),
    Object(JsonbView<'a>),
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct JsonbView<'a>(pub(crate) &'a [u8]);

impl<'a> JsonbView<'a> {
    pub fn new(data: &'a [u8]) -> Result<Self> {
        ensure!(
            data.len() >= 4,
            "jsonb data too short: {} bytes",
            data.len()
        );
        Ok(Self(data))
    }

    pub fn data(&self) -> &'a [u8] {
        self.0
    }

    fn header(&self) -> u32 {
        u32::from_le_bytes([self.0[0], self.0[1], self.0[2], self.0[3]])
    }

    pub fn root_type(&self) -> u8 {
        ((self.header() >> 28) & 0x0F) as u8
    }

    pub fn entry_count(&self) -> usize {
        (self.header() & 0x0FFF_FFFF) as usize
    }

    fn entries_start(&self) -> usize {
        4
    }

    fn data_start(&self) -> usize {
        4 + self.entry_count() * 4
    }

    fn read_entry(&self, idx: usize) -> u32 {
        let offset = self.entries_start() + idx * 4;
        u32::from_le_bytes([
            self.0[offset],
            self.0[offset + 1],
            self.0[offset + 2],
            self.0[offset + 3],
        ])
    }

    fn entry_is_key(entry: u32) -> bool {
        (entry & FLAG_IS_KEY) != 0
    }

    #[allow(dead_code)]
    fn entry_is_variable(entry: u32) -> bool {
        (entry & FLAG_IS_VARIABLE) != 0
    }

    fn entry_type(entry: u32) -> u8 {
        ((entry & TYPE_MASK) >> TYPE_SHIFT) as u8
    }

    fn entry_offset(entry: u32) -> usize {
        (entry & OFFSET_MASK) as usize
    }

    fn read_key_at(&self, pair_idx: usize) -> Result<&'a str> {
        let key_entry = self.read_entry(pair_idx * 2);
        ensure!(
            Self::entry_is_key(key_entry),
            "expected key entry at index {}",
            pair_idx * 2
        );

        let offset = Self::entry_offset(key_entry);
        let data_section = &self.0[self.data_start()..];

        let len_bytes: [u8; 2] = data_section[offset..offset + 2]
            .try_into()
            .map_err(|_| eyre::eyre!("key length read failed"))?;
        let len = u16::from_le_bytes(len_bytes) as usize;

        let key_bytes = &data_section[offset + 2..offset + 2 + len];
        std::str::from_utf8(key_bytes).map_err(|e| eyre::eyre!("invalid UTF-8 in jsonb key: {}", e))
    }

    fn read_value_at(&self, pair_idx: usize) -> Result<JsonbValue<'a>> {
        let value_entry = self.read_entry(pair_idx * 2 + 1);
        self.decode_entry(value_entry)
    }

    fn decode_entry(&self, entry: u32) -> Result<JsonbValue<'a>> {
        let typ = Self::entry_type(entry);
        let offset = Self::entry_offset(entry);

        match typ {
            JSONB_TYPE_NULL => Ok(JsonbValue::Null),
            JSONB_TYPE_BOOL => Ok(JsonbValue::Bool(offset != 0)),
            JSONB_TYPE_NUMBER => {
                let data_section = &self.0[self.data_start()..];
                let bytes: [u8; 8] = data_section[offset..offset + 8]
                    .try_into()
                    .map_err(|_| eyre::eyre!("number read failed"))?;
                Ok(JsonbValue::Number(f64::from_le_bytes(bytes)))
            }
            JSONB_TYPE_STRING => {
                let data_section = &self.0[self.data_start()..];
                let len_bytes: [u8; 2] = data_section[offset..offset + 2]
                    .try_into()
                    .map_err(|_| eyre::eyre!("string length read failed"))?;
                let len = u16::from_le_bytes(len_bytes) as usize;
                let str_bytes = &data_section[offset + 2..offset + 2 + len];
                let s = std::str::from_utf8(str_bytes)
                    .map_err(|e| eyre::eyre!("invalid UTF-8 in jsonb string: {}", e))?;
                Ok(JsonbValue::String(s))
            }
            JSONB_TYPE_ARRAY | JSONB_TYPE_OBJECT => {
                let data_section = &self.0[self.data_start()..];
                let len_bytes: [u8; 4] = data_section[offset..offset + 4]
                    .try_into()
                    .map_err(|_| eyre::eyre!("nested length read failed"))?;
                let len = u32::from_le_bytes(len_bytes) as usize;
                let nested_data = &data_section[offset + 4..offset + 4 + len];
                let nested_view = JsonbView::new(nested_data)?;
                if typ == JSONB_TYPE_ARRAY {
                    Ok(JsonbValue::Array(nested_view))
                } else {
                    Ok(JsonbValue::Object(nested_view))
                }
            }
            _ => bail!("unknown jsonb type tag: {}", typ),
        }
    }

    pub fn get(&self, key: &str) -> Result<Option<JsonbValue<'a>>> {
        if self.root_type() != JSONB_TYPE_OBJECT {
            bail!("cannot get key from non-object jsonb");
        }

        let pair_count = self.entry_count() / 2;
        if pair_count == 0 {
            return Ok(None);
        }

        let mut low: isize = 0;
        let mut high: isize = pair_count as isize - 1;

        while low <= high {
            let mid = ((low + high) / 2) as usize;
            let current_key = self.read_key_at(mid)?;

            match current_key.cmp(key) {
                Ordering::Equal => return self.read_value_at(mid).map(Some),
                Ordering::Less => low = mid as isize + 1,
                Ordering::Greater => high = mid as isize - 1,
            }
        }
        Ok(None)
    }

    pub fn get_path(&self, path: &[&str]) -> Result<Option<JsonbValue<'a>>> {
        if path.is_empty() {
            return Ok(Some(self.as_value()?));
        }

        let mut current = self.get(path[0])?;
        for key in &path[1..] {
            match current {
                Some(JsonbValue::Object(view)) => {
                    current = view.get(key)?;
                }
                _ => return Ok(None),
            }
        }
        Ok(current)
    }

    pub fn as_value(&self) -> Result<JsonbValue<'a>> {
        match self.root_type() {
            JSONB_TYPE_OBJECT => Ok(JsonbValue::Object(*self)),
            JSONB_TYPE_ARRAY => Ok(JsonbValue::Array(*self)),
            JSONB_TYPE_NULL => Ok(JsonbValue::Null),
            JSONB_TYPE_BOOL => {
                let val = self.entry_count() != 0;
                Ok(JsonbValue::Bool(val))
            }
            JSONB_TYPE_NUMBER => {
                let bytes: [u8; 8] = self.0[4..12]
                    .try_into()
                    .map_err(|_| eyre::eyre!("scalar number read failed"))?;
                Ok(JsonbValue::Number(f64::from_le_bytes(bytes)))
            }
            JSONB_TYPE_STRING => {
                let len = self.entry_count();
                let str_bytes = &self.0[4..4 + len];
                let s = std::str::from_utf8(str_bytes)
                    .map_err(|e| eyre::eyre!("invalid UTF-8 in jsonb scalar string: {}", e))?;
                Ok(JsonbValue::String(s))
            }
            t => bail!("unknown jsonb root type: {}", t),
        }
    }

    pub fn array_len(&self) -> Result<usize> {
        if self.root_type() != JSONB_TYPE_ARRAY {
            bail!("cannot get array length from non-array jsonb");
        }
        Ok(self.entry_count())
    }

    pub fn array_get(&self, idx: usize) -> Result<Option<JsonbValue<'a>>> {
        if self.root_type() != JSONB_TYPE_ARRAY {
            bail!("cannot index non-array jsonb");
        }
        if idx >= self.entry_count() {
            return Ok(None);
        }
        let entry = self.read_entry(idx);
        self.decode_entry(entry).map(Some)
    }

    pub fn object_len(&self) -> Result<usize> {
        if self.root_type() != JSONB_TYPE_OBJECT {
            bail!("cannot get object length from non-object jsonb");
        }
        Ok(self.entry_count() / 2)
    }

    pub fn iter_object(&self) -> Result<ObjectIter<'a>> {
        if self.root_type() != JSONB_TYPE_OBJECT {
            bail!("cannot iterate over non-object jsonb");
        }
        Ok(ObjectIter {
            view: *self,
            pair_idx: 0,
            pair_count: self.entry_count() / 2,
        })
    }

    pub fn iter_array(&self) -> Result<ArrayIter<'a>> {
        if self.root_type() != JSONB_TYPE_ARRAY {
            bail!("cannot iterate over non-array jsonb");
        }
        Ok(ArrayIter {
            view: *self,
            idx: 0,
            len: self.entry_count(),
        })
    }

    pub fn to_json_string(&self) -> Result<String> {
        match self.root_type() {
            JSONB_TYPE_OBJECT => {
                let mut result = String::from("{");
                let mut first = true;
                for item in self.iter_object()? {
                    let (key, value) = item?;
                    if !first {
                        result.push(',');
                    }
                    first = false;
                    result.push_str(&escape_json_string(key));
                    result.push(':');
                    result.push_str(&value.to_json_string()?);
                }
                result.push('}');
                Ok(result)
            }
            JSONB_TYPE_ARRAY => {
                let mut result = String::from("[");
                let mut first = true;
                for item in self.iter_array()? {
                    let value = item?;
                    if !first {
                        result.push(',');
                    }
                    first = false;
                    result.push_str(&value.to_json_string()?);
                }
                result.push(']');
                Ok(result)
            }
            _ => self.as_value()?.to_json_string(),
        }
    }
}

pub struct ObjectIter<'a> {
    view: JsonbView<'a>,
    pair_idx: usize,
    pair_count: usize,
}

impl<'a> Iterator for ObjectIter<'a> {
    type Item = Result<(&'a str, JsonbValue<'a>)>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.pair_idx >= self.pair_count {
            return None;
        }
        let key = match self.view.read_key_at(self.pair_idx) {
            Ok(k) => k,
            Err(e) => return Some(Err(e)),
        };
        let value = match self.view.read_value_at(self.pair_idx) {
            Ok(v) => v,
            Err(e) => return Some(Err(e)),
        };
        self.pair_idx += 1;
        Some(Ok((key, value)))
    }
}

pub struct ArrayIter<'a> {
    view: JsonbView<'a>,
    idx: usize,
    len: usize,
}

impl<'a> Iterator for ArrayIter<'a> {
    type Item = Result<JsonbValue<'a>>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.idx >= self.len {
            return None;
        }
        let entry = self.view.read_entry(self.idx);
        self.idx += 1;
        Some(self.view.decode_entry(entry))
    }
}

impl<'a> JsonbValue<'a> {
    pub fn to_json_string(&self) -> Result<String> {
        match self {
            JsonbValue::Null => Ok("null".to_string()),
            JsonbValue::Bool(b) => Ok(if *b { "true" } else { "false" }.to_string()),
            JsonbValue::Number(n) => Ok(format_json_number(*n)),
            JsonbValue::String(s) => Ok(escape_json_string(s)),
            JsonbValue::Array(view) => view.to_json_string(),
            JsonbValue::Object(view) => view.to_json_string(),
        }
    }
}

fn escape_json_string(s: &str) -> String {
    let mut result = String::with_capacity(s.len() + 2);
    result.push('"');
    for c in s.chars() {
        match c {
            '"' => result.push_str("\\\""),
            '\\' => result.push_str("\\\\"),
            '\n' => result.push_str("\\n"),
            '\r' => result.push_str("\\r"),
            '\t' => result.push_str("\\t"),
            c if c.is_control() => {
                result.push_str(&format!("\\u{:04x}", c as u32));
            }
            c => result.push(c),
        }
    }
    result.push('"');
    result
}

fn format_json_number(n: f64) -> String {
    if n.is_nan() || n.is_infinite() {
        "null".to_string()
    } else if n.fract() == 0.0 && n.abs() < (i64::MAX as f64) {
        format!("{}", n as i64)
    } else {
        format!("{}", n)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum JsonbBuilderValue {
    Null,
    Bool(bool),
    Number(f64),
    String(String),
    Array(Vec<JsonbBuilderValue>),
    Object(Vec<(String, JsonbBuilderValue)>),
}

impl From<bool> for JsonbBuilderValue {
    fn from(v: bool) -> Self {
        JsonbBuilderValue::Bool(v)
    }
}

impl From<f64> for JsonbBuilderValue {
    fn from(v: f64) -> Self {
        JsonbBuilderValue::Number(v)
    }
}

impl From<i64> for JsonbBuilderValue {
    fn from(v: i64) -> Self {
        JsonbBuilderValue::Number(v as f64)
    }
}

impl From<i32> for JsonbBuilderValue {
    fn from(v: i32) -> Self {
        JsonbBuilderValue::Number(v as f64)
    }
}

impl From<&str> for JsonbBuilderValue {
    fn from(v: &str) -> Self {
        JsonbBuilderValue::String(v.to_string())
    }
}

impl From<String> for JsonbBuilderValue {
    fn from(v: String) -> Self {
        JsonbBuilderValue::String(v)
    }
}

pub struct JsonbBuilder {
    root: JsonbBuilderValue,
}

impl JsonbBuilder {
    pub fn new_object() -> Self {
        Self {
            root: JsonbBuilderValue::Object(Vec::new()),
        }
    }

    pub fn new_array() -> Self {
        Self {
            root: JsonbBuilderValue::Array(Vec::new()),
        }
    }

    pub fn new_null() -> Self {
        Self {
            root: JsonbBuilderValue::Null,
        }
    }

    pub fn new_bool(v: bool) -> Self {
        Self {
            root: JsonbBuilderValue::Bool(v),
        }
    }

    pub fn new_number(v: f64) -> Self {
        Self {
            root: JsonbBuilderValue::Number(v),
        }
    }

    pub fn new_string(v: impl Into<String>) -> Self {
        Self {
            root: JsonbBuilderValue::String(v.into()),
        }
    }

    pub fn set(&mut self, key: impl Into<String>, value: impl Into<JsonbBuilderValue>) {
        if let JsonbBuilderValue::Object(ref mut entries) = self.root {
            entries.push((key.into(), value.into()));
        }
    }

    pub fn push(&mut self, value: impl Into<JsonbBuilderValue>) {
        if let JsonbBuilderValue::Array(ref mut elements) = self.root {
            elements.push(value.into());
        }
    }

    pub fn build(&self) -> Vec<u8> {
        let mut buf = Vec::new();
        self.encode_value(&self.root, &mut buf);
        buf
    }

    fn encode_value(&self, value: &JsonbBuilderValue, buf: &mut Vec<u8>) {
        match value {
            JsonbBuilderValue::Null => {
                let header = (JSONB_TYPE_NULL as u32) << 28;
                buf.extend(header.to_le_bytes());
            }
            JsonbBuilderValue::Bool(v) => {
                let header = ((JSONB_TYPE_BOOL as u32) << 28) | (*v as u32);
                buf.extend(header.to_le_bytes());
            }
            JsonbBuilderValue::Number(v) => {
                let header = (JSONB_TYPE_NUMBER as u32) << 28;
                buf.extend(header.to_le_bytes());
                buf.extend(v.to_le_bytes());
            }
            JsonbBuilderValue::String(s) => {
                let header = ((JSONB_TYPE_STRING as u32) << 28) | (s.len() as u32);
                buf.extend(header.to_le_bytes());
                buf.extend(s.as_bytes());
            }
            JsonbBuilderValue::Array(elements) => {
                let header = ((JSONB_TYPE_ARRAY as u32) << 28) | (elements.len() as u32);
                buf.extend(header.to_le_bytes());

                let entries_start = buf.len();
                buf.resize(entries_start + elements.len() * 4, 0);

                let mut data_buf = Vec::new();

                for (i, elem) in elements.iter().enumerate() {
                    let entry = self.encode_entry(elem, &mut data_buf);
                    let entry_offset = entries_start + i * 4;
                    buf[entry_offset..entry_offset + 4].copy_from_slice(&entry.to_le_bytes());
                }

                buf.extend(&data_buf);
            }
            JsonbBuilderValue::Object(entries) => {
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

                    let val_entry = self.encode_entry(val, &mut data_buf);

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

    fn encode_entry(&self, value: &JsonbBuilderValue, data_buf: &mut Vec<u8>) -> u32 {
        match value {
            JsonbBuilderValue::Null => (JSONB_TYPE_NULL as u32) << TYPE_SHIFT,
            JsonbBuilderValue::Bool(v) => ((JSONB_TYPE_BOOL as u32) << TYPE_SHIFT) | (*v as u32),
            JsonbBuilderValue::Number(v) => {
                let offset = data_buf.len();
                data_buf.extend(v.to_le_bytes());
                FLAG_IS_VARIABLE
                    | ((JSONB_TYPE_NUMBER as u32) << TYPE_SHIFT)
                    | (offset as u32 & OFFSET_MASK)
            }
            JsonbBuilderValue::String(s) => {
                let offset = data_buf.len();
                data_buf.extend((s.len() as u16).to_le_bytes());
                data_buf.extend(s.as_bytes());
                FLAG_IS_VARIABLE
                    | ((JSONB_TYPE_STRING as u32) << TYPE_SHIFT)
                    | (offset as u32 & OFFSET_MASK)
            }
            JsonbBuilderValue::Array(_) | JsonbBuilderValue::Object(_) => {
                let offset = data_buf.len();
                let mut nested_buf = Vec::new();
                self.encode_value(value, &mut nested_buf);
                data_buf.extend((nested_buf.len() as u32).to_le_bytes());
                data_buf.extend(&nested_buf);

                let typ = if matches!(value, JsonbBuilderValue::Array(_)) {
                    JSONB_TYPE_ARRAY
                } else {
                    JSONB_TYPE_OBJECT
                };

                FLAG_IS_VARIABLE | ((typ as u32) << TYPE_SHIFT) | (offset as u32 & OFFSET_MASK)
            }
        }
    }
}
