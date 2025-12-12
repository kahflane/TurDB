//! # Record Serialization with O(1) Column Access
//!
//! This module provides zero-copy record access for TurDB's row storage. Unlike
//! SQLite's sequential record parsing (O(N) to find column N), TurDB records use
//! a header with null bitmap and offset table for O(1) column lookup.
//!
//! ## Record Binary Layout
//!
//! ```text
//! +------------------+------------------+------------------+------------------+
//! | Header Length    | Null Bitmap      | Offset Table     | Data Payload     |
//! | (u16)            | [u8; (N+7)/8]    | [u16; M]         | [u8; ...]        |
//! +------------------+------------------+------------------+------------------+
//! ```
//!
//! | Component | Type | Description |
//! |-----------|------|-------------|
//! | **Header Length** | `u16` | Total header size (allows skipping to data) |
//! | **Null Bitmap** | `[u8; (N+7)/8]` | 1 bit per column. `1` = NULL, `0` = has data |
//! | **Offset Table** | `[u16; M]` | End offsets for variable-length columns only |
//! | **Data Payload** | `[u8; ...]` | Concatenated fixed-width and variable-width values |
//!
//! ## Design Goals
//!
//! 1. **O(1) column access**: Direct offset calculation, no sequential parsing
//! 2. **Zero-copy reads**: All getters return references into the underlying buffer
//! 3. **Schema-dependent**: Types come from schema, not stored per-row
//! 4. **Compact headers**: u16 offsets (16KB pages), bitmap for NULLs
//!
//! ## Storage Classes
//!
//! Columns are categorized by storage class:
//!
//! | Class | Examples | Storage |
//! |-------|----------|---------|
//! | **Fixed** | int4, float8, uuid, timestamp | Direct bytes, no offset needed |
//! | **Variable** | text, blob, jsonb, vector | Offset in table → data in payload |
//!
//! Fixed-width columns are stored contiguously after the header, followed by
//! variable-width columns. The offset table only contains entries for variable
//! columns, saving space.
//!
//! ## Fixed-Width Type Sizes
//!
//! | Type | Size (bytes) |
//! |------|--------------|
//! | bool | 1 |
//! | int2 | 2 |
//! | int4 | 4 |
//! | int8 | 8 |
//! | float4 | 4 |
//! | float8 | 8 |
//! | date | 4 (days since epoch) |
//! | time | 8 (microseconds) |
//! | timestamp | 8 (microseconds since epoch) |
//! | uuid | 16 |
//! | macaddr | 6 |
//!
//! ## Zero-Copy Access Pattern
//!
//! ```ignore
//! // Record data comes from mmap'd page
//! let record = RecordView::new(page_data, &schema)?;
//!
//! // All accessors return references into the original buffer
//! let name: &str = record.get_text(1)?;  // Points into mmap
//! let age: i32 = record.get_int4(2)?;    // Reads from mmap
//! ```
//!
//! ## Schema Evolution
//!
//! Records support schema evolution for `ALTER TABLE ADD COLUMN`:
//!
//! - New columns are added at the end of schema
//! - Old records have fewer columns than current schema
//! - Accessing columns beyond record's column count returns NULL or default
//!
//! ## Thread Safety
//!
//! `RecordView` borrows immutably from a byte slice. Multiple `RecordView`
//! instances can read the same data concurrently. Write access requires
//! exclusive access to the underlying buffer via `RecordBuilder`.
//!
//! ## Performance
//!
//! - Column access: O(1) offset calculation
//! - NULL check: O(1) bit test
//! - Memory: ~1 byte per 8 columns for bitmap + 2 bytes per variable column

use eyre::{ensure, Result};

#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DataType {
    Bool = 0,
    Int2 = 1,
    Int4 = 2,
    Int8 = 3,
    Float4 = 4,
    Float8 = 5,
    Date = 6,
    Time = 7,
    Timestamp = 8,
    TimestampTz = 9,
    Uuid = 10,
    MacAddr = 11,
    Inet4 = 12,
    Inet6 = 13,
    Text = 20,
    Blob = 21,
    Vector = 22,
    Jsonb = 23,
}

impl DataType {
    pub fn fixed_size(&self) -> Option<usize> {
        match self {
            DataType::Bool => Some(1),
            DataType::Int2 => Some(2),
            DataType::Int4 => Some(4),
            DataType::Int8 => Some(8),
            DataType::Float4 => Some(4),
            DataType::Float8 => Some(8),
            DataType::Date => Some(4),
            DataType::Time => Some(8),
            DataType::Timestamp => Some(8),
            DataType::TimestampTz => Some(12),
            DataType::Uuid => Some(16),
            DataType::MacAddr => Some(6),
            DataType::Inet4 => Some(4),
            DataType::Inet6 => Some(16),
            DataType::Text => None,
            DataType::Blob => None,
            DataType::Vector => None,
            DataType::Jsonb => None,
        }
    }

    pub fn is_variable(&self) -> bool {
        self.fixed_size().is_none()
    }
}

#[derive(Debug, Clone)]
pub struct ColumnDef {
    pub name: String,
    pub data_type: DataType,
}

impl ColumnDef {
    pub fn new(name: impl Into<String>, data_type: DataType) -> Self {
        Self {
            name: name.into(),
            data_type,
        }
    }
}

pub mod jsonb {
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
            std::str::from_utf8(key_bytes)
                .map_err(|e| eyre::eyre!("invalid UTF-8 in jsonb key: {}", e))
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
                JsonbBuilderValue::Bool(v) => {
                    ((JSONB_TYPE_BOOL as u32) << TYPE_SHIFT) | (*v as u32)
                }
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
}

pub use jsonb::{JsonbBuilder, JsonbBuilderValue, JsonbValue, JsonbView};

#[derive(Debug, Clone)]
pub struct Schema {
    columns: Vec<ColumnDef>,
    var_column_indices: Vec<usize>,
    fixed_offsets: Vec<usize>,
    total_fixed_size: usize,
}

impl Schema {
    pub fn new(columns: Vec<ColumnDef>) -> Self {
        let mut var_column_indices = Vec::new();
        let mut fixed_offsets = Vec::new();
        let mut offset = 0;

        for (idx, col) in columns.iter().enumerate() {
            fixed_offsets.push(offset);
            if let Some(size) = col.data_type.fixed_size() {
                offset += size;
            } else {
                var_column_indices.push(idx);
            }
        }

        Self {
            columns,
            var_column_indices,
            fixed_offsets,
            total_fixed_size: offset,
        }
    }

    pub fn column_count(&self) -> usize {
        self.columns.len()
    }

    pub fn var_column_count(&self) -> usize {
        self.var_column_indices.len()
    }

    pub fn column(&self, idx: usize) -> Option<&ColumnDef> {
        self.columns.get(idx)
    }

    pub fn var_column_index(&self, col_idx: usize) -> Option<usize> {
        self.var_column_indices
            .iter()
            .position(|&idx| idx == col_idx)
    }

    pub fn fixed_offset(&self, col_idx: usize) -> usize {
        self.fixed_offsets[col_idx]
    }

    pub fn total_fixed_size(&self) -> usize {
        self.total_fixed_size
    }

    pub fn null_bitmap_size(column_count: usize) -> usize {
        column_count.div_ceil(8)
    }
}

#[derive(Debug)]
pub struct RecordView<'a> {
    data: &'a [u8],
    schema: &'a Schema,
}

impl<'a> RecordView<'a> {
    pub fn new(data: &'a [u8], schema: &'a Schema) -> Result<Self> {
        ensure!(!data.is_empty(), "record data cannot be empty");
        ensure!(data.len() >= 2, "record too small for header length");
        Ok(Self { data, schema })
    }

    pub fn data(&self) -> &'a [u8] {
        self.data
    }

    pub fn schema(&self) -> &'a Schema {
        self.schema
    }

    pub fn header_len(&self) -> u16 {
        u16::from_le_bytes([self.data[0], self.data[1]])
    }

    pub fn null_bitmap(&self) -> &'a [u8] {
        let bitmap_size = Schema::null_bitmap_size(self.schema.column_count());
        &self.data[2..2 + bitmap_size]
    }

    pub fn offset_table(&self) -> &'a [u8] {
        let bitmap_size = Schema::null_bitmap_size(self.schema.column_count());
        let offset_table_start = 2 + bitmap_size;
        let offset_table_bytes = self.schema.var_column_count() * 2;
        &self.data[offset_table_start..offset_table_start + offset_table_bytes]
    }

    pub fn data_offset(&self) -> usize {
        self.header_len() as usize
    }

    pub fn is_null(&self, col_idx: usize) -> bool {
        let byte_idx = col_idx / 8;
        let bit_idx = col_idx % 8;
        let bitmap = self.null_bitmap();
        (bitmap[byte_idx] & (1 << bit_idx)) != 0
    }

    pub fn get_fixed_col_offset(&self, col_idx: usize) -> usize {
        self.data_offset() + self.schema.fixed_offset(col_idx)
    }

    pub fn get_var_bounds(&self, col_idx: usize) -> Result<(usize, usize)> {
        let var_idx = self
            .schema
            .var_column_index(col_idx)
            .ok_or_else(|| eyre::eyre!("column {} is not a variable column", col_idx))?;

        let offset_table = self.offset_table();
        let var_data_start = self.data_offset() + self.schema.total_fixed_size();

        let end_offset =
            u16::from_le_bytes([offset_table[var_idx * 2], offset_table[var_idx * 2 + 1]]) as usize;

        let start_offset = if var_idx == 0 {
            0
        } else {
            u16::from_le_bytes([
                offset_table[(var_idx - 1) * 2],
                offset_table[(var_idx - 1) * 2 + 1],
            ]) as usize
        };

        Ok((var_data_start + start_offset, var_data_start + end_offset))
    }

    pub fn get_bool(&self, col_idx: usize) -> Result<bool> {
        let offset = self.get_fixed_col_offset(col_idx);
        Ok(self.data[offset] != 0)
    }

    pub fn get_int2(&self, col_idx: usize) -> Result<i16> {
        let offset = self.get_fixed_col_offset(col_idx);
        let bytes: [u8; 2] = self.data[offset..offset + 2]
            .try_into()
            .map_err(|_| eyre::eyre!("insufficient data for int2 at col {}", col_idx))?;
        Ok(i16::from_le_bytes(bytes))
    }

    pub fn get_int4(&self, col_idx: usize) -> Result<i32> {
        let offset = self.get_fixed_col_offset(col_idx);
        let bytes: [u8; 4] = self.data[offset..offset + 4]
            .try_into()
            .map_err(|_| eyre::eyre!("insufficient data for int4 at col {}", col_idx))?;
        Ok(i32::from_le_bytes(bytes))
    }

    pub fn get_int8(&self, col_idx: usize) -> Result<i64> {
        let offset = self.get_fixed_col_offset(col_idx);
        let bytes: [u8; 8] = self.data[offset..offset + 8]
            .try_into()
            .map_err(|_| eyre::eyre!("insufficient data for int8 at col {}", col_idx))?;
        Ok(i64::from_le_bytes(bytes))
    }

    pub fn get_float4(&self, col_idx: usize) -> Result<f32> {
        let offset = self.get_fixed_col_offset(col_idx);
        let bytes: [u8; 4] = self.data[offset..offset + 4]
            .try_into()
            .map_err(|_| eyre::eyre!("insufficient data for float4 at col {}", col_idx))?;
        Ok(f32::from_le_bytes(bytes))
    }

    pub fn get_float8(&self, col_idx: usize) -> Result<f64> {
        let offset = self.get_fixed_col_offset(col_idx);
        let bytes: [u8; 8] = self.data[offset..offset + 8]
            .try_into()
            .map_err(|_| eyre::eyre!("insufficient data for float8 at col {}", col_idx))?;
        Ok(f64::from_le_bytes(bytes))
    }

    pub fn get_date(&self, col_idx: usize) -> Result<i32> {
        let offset = self.get_fixed_col_offset(col_idx);
        let bytes: [u8; 4] = self.data[offset..offset + 4]
            .try_into()
            .map_err(|_| eyre::eyre!("insufficient data for date at col {}", col_idx))?;
        Ok(i32::from_le_bytes(bytes))
    }

    pub fn get_time(&self, col_idx: usize) -> Result<i64> {
        let offset = self.get_fixed_col_offset(col_idx);
        let bytes: [u8; 8] = self.data[offset..offset + 8]
            .try_into()
            .map_err(|_| eyre::eyre!("insufficient data for time at col {}", col_idx))?;
        Ok(i64::from_le_bytes(bytes))
    }

    pub fn get_timestamp(&self, col_idx: usize) -> Result<i64> {
        let offset = self.get_fixed_col_offset(col_idx);
        let bytes: [u8; 8] = self.data[offset..offset + 8]
            .try_into()
            .map_err(|_| eyre::eyre!("insufficient data for timestamp at col {}", col_idx))?;
        Ok(i64::from_le_bytes(bytes))
    }

    pub fn get_uuid(&self, col_idx: usize) -> Result<&'a [u8; 16]> {
        let offset = self.get_fixed_col_offset(col_idx);
        self.data[offset..offset + 16]
            .try_into()
            .map_err(|_| eyre::eyre!("insufficient data for uuid at col {}", col_idx))
    }

    pub fn get_macaddr(&self, col_idx: usize) -> Result<&'a [u8; 6]> {
        let offset = self.get_fixed_col_offset(col_idx);
        self.data[offset..offset + 6]
            .try_into()
            .map_err(|_| eyre::eyre!("insufficient data for macaddr at col {}", col_idx))
    }

    pub fn get_text(&self, col_idx: usize) -> Result<&'a str> {
        let (start, end) = self.get_var_bounds(col_idx)?;
        let bytes = &self.data[start..end];
        std::str::from_utf8(bytes)
            .map_err(|e| eyre::eyre!("invalid UTF-8 in text column {}: {}", col_idx, e))
    }

    pub fn get_blob(&self, col_idx: usize) -> Result<&'a [u8]> {
        let (start, end) = self.get_var_bounds(col_idx)?;
        Ok(&self.data[start..end])
    }

    pub fn get_vector(&self, col_idx: usize) -> Result<&'a [f32]> {
        let (start, end) = self.get_var_bounds(col_idx)?;
        let bytes = &self.data[start..end];
        if bytes.len() < 4 {
            return Err(eyre::eyre!(
                "vector data too short at col {}: {} bytes",
                col_idx,
                bytes.len()
            ));
        }
        let len = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]) as usize;
        let expected_size = 4 + len * 4;
        if bytes.len() != expected_size {
            return Err(eyre::eyre!(
                "vector size mismatch at col {}: expected {} bytes, got {}",
                col_idx,
                expected_size,
                bytes.len()
            ));
        }
        let float_bytes = &bytes[4..];
        if !float_bytes.len().is_multiple_of(4)
            || !(float_bytes.as_ptr() as usize).is_multiple_of(4)
        {
            let mut floats = Vec::with_capacity(len);
            for i in 0..len {
                let offset = i * 4;
                let f = f32::from_le_bytes([
                    float_bytes[offset],
                    float_bytes[offset + 1],
                    float_bytes[offset + 2],
                    float_bytes[offset + 3],
                ]);
                floats.push(f);
            }
            return Err(eyre::eyre!(
                "vector data not aligned for zero-copy at col {}",
                col_idx
            ));
        }
        let floats = unsafe { std::slice::from_raw_parts(float_bytes.as_ptr() as *const f32, len) };
        Ok(floats)
    }

    pub fn get_vector_copy(&self, col_idx: usize) -> Result<Vec<f32>> {
        let (start, end) = self.get_var_bounds(col_idx)?;
        let bytes = &self.data[start..end];
        if bytes.len() < 4 {
            return Err(eyre::eyre!(
                "vector data too short at col {}: {} bytes",
                col_idx,
                bytes.len()
            ));
        }
        let len = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]) as usize;
        let expected_size = 4 + len * 4;
        if bytes.len() != expected_size {
            return Err(eyre::eyre!(
                "vector size mismatch at col {}: expected {} bytes, got {}",
                col_idx,
                expected_size,
                bytes.len()
            ));
        }
        let float_bytes = &bytes[4..];
        let mut floats = Vec::with_capacity(len);
        for i in 0..len {
            let offset = i * 4;
            let f = f32::from_le_bytes([
                float_bytes[offset],
                float_bytes[offset + 1],
                float_bytes[offset + 2],
                float_bytes[offset + 3],
            ]);
            floats.push(f);
        }
        Ok(floats)
    }

    pub fn get_vector_opt(&self, col_idx: usize) -> Result<Option<Vec<f32>>> {
        if self.is_null_or_missing(col_idx) {
            return Ok(None);
        }
        self.get_vector_copy(col_idx).map(Some)
    }

    pub fn get_jsonb(&self, col_idx: usize) -> Result<JsonbView<'a>> {
        let (start, end) = self.get_var_bounds(col_idx)?;
        let bytes = &self.data[start..end];
        JsonbView::new(bytes)
    }

    pub fn get_jsonb_opt(&self, col_idx: usize) -> Result<Option<JsonbView<'a>>> {
        if self.is_null_or_missing(col_idx) {
            return Ok(None);
        }
        self.get_jsonb(col_idx).map(Some)
    }

    pub fn record_column_count(&self) -> usize {
        let data_len = self.data.len();
        let header_len = self.header_len() as usize;

        if data_len <= header_len {
            return 0;
        }

        let available_fixed_data = data_len - header_len;
        let mut col_count = 0;
        let mut consumed = 0;

        for col in &self.schema.columns {
            if let Some(size) = col.data_type.fixed_size() {
                if consumed + size > available_fixed_data {
                    break;
                }
                consumed += size;
            }
            col_count += 1;
        }

        col_count
    }

    pub fn is_null_or_missing(&self, col_idx: usize) -> bool {
        if col_idx >= self.record_column_count() {
            return true;
        }
        self.is_null(col_idx)
    }

    pub fn get_int4_opt(&self, col_idx: usize) -> Result<Option<i32>> {
        if self.is_null_or_missing(col_idx) {
            return Ok(None);
        }
        self.get_int4(col_idx).map(Some)
    }

    pub fn get_int2_opt(&self, col_idx: usize) -> Result<Option<i16>> {
        if self.is_null_or_missing(col_idx) {
            return Ok(None);
        }
        self.get_int2(col_idx).map(Some)
    }

    pub fn get_int8_opt(&self, col_idx: usize) -> Result<Option<i64>> {
        if self.is_null_or_missing(col_idx) {
            return Ok(None);
        }
        self.get_int8(col_idx).map(Some)
    }

    pub fn get_float4_opt(&self, col_idx: usize) -> Result<Option<f32>> {
        if self.is_null_or_missing(col_idx) {
            return Ok(None);
        }
        self.get_float4(col_idx).map(Some)
    }

    pub fn get_float8_opt(&self, col_idx: usize) -> Result<Option<f64>> {
        if self.is_null_or_missing(col_idx) {
            return Ok(None);
        }
        self.get_float8(col_idx).map(Some)
    }

    pub fn get_text_opt(&self, col_idx: usize) -> Result<Option<&'a str>> {
        if self.is_null_or_missing(col_idx) {
            return Ok(None);
        }
        self.get_text(col_idx).map(Some)
    }

    pub fn get_blob_opt(&self, col_idx: usize) -> Result<Option<&'a [u8]>> {
        if self.is_null_or_missing(col_idx) {
            return Ok(None);
        }
        self.get_blob(col_idx).map(Some)
    }

    pub fn get_bool_opt(&self, col_idx: usize) -> Result<Option<bool>> {
        if self.is_null_or_missing(col_idx) {
            return Ok(None);
        }
        self.get_bool(col_idx).map(Some)
    }

    pub fn get_date_opt(&self, col_idx: usize) -> Result<Option<i32>> {
        if self.is_null_or_missing(col_idx) {
            return Ok(None);
        }
        self.get_date(col_idx).map(Some)
    }

    pub fn get_time_opt(&self, col_idx: usize) -> Result<Option<i64>> {
        if self.is_null_or_missing(col_idx) {
            return Ok(None);
        }
        self.get_time(col_idx).map(Some)
    }

    pub fn get_timestamp_opt(&self, col_idx: usize) -> Result<Option<i64>> {
        if self.is_null_or_missing(col_idx) {
            return Ok(None);
        }
        self.get_timestamp(col_idx).map(Some)
    }

    pub fn get_uuid_opt(&self, col_idx: usize) -> Result<Option<&'a [u8; 16]>> {
        if self.is_null_or_missing(col_idx) {
            return Ok(None);
        }
        self.get_uuid(col_idx).map(Some)
    }

    pub fn get_macaddr_opt(&self, col_idx: usize) -> Result<Option<&'a [u8; 6]>> {
        if self.is_null_or_missing(col_idx) {
            return Ok(None);
        }
        self.get_macaddr(col_idx).map(Some)
    }

    pub fn get_timestamptz(&self, col_idx: usize) -> Result<(i64, i32)> {
        let offset = self.get_fixed_col_offset(col_idx);
        let micros_bytes: [u8; 8] = self.data[offset..offset + 8].try_into().map_err(|_| {
            eyre::eyre!(
                "insufficient data for timestamptz micros at col {}",
                col_idx
            )
        })?;
        let offset_bytes: [u8; 4] =
            self.data[offset + 8..offset + 12].try_into().map_err(|_| {
                eyre::eyre!(
                    "insufficient data for timestamptz offset at col {}",
                    col_idx
                )
            })?;
        Ok((
            i64::from_le_bytes(micros_bytes),
            i32::from_le_bytes(offset_bytes),
        ))
    }

    pub fn get_timestamptz_opt(&self, col_idx: usize) -> Result<Option<(i64, i32)>> {
        if self.is_null_or_missing(col_idx) {
            return Ok(None);
        }
        self.get_timestamptz(col_idx).map(Some)
    }

    pub fn get_inet4(&self, col_idx: usize) -> Result<&'a [u8; 4]> {
        let offset = self.get_fixed_col_offset(col_idx);
        self.data[offset..offset + 4]
            .try_into()
            .map_err(|_| eyre::eyre!("insufficient data for inet4 at col {}", col_idx))
    }

    pub fn get_inet4_opt(&self, col_idx: usize) -> Result<Option<&'a [u8; 4]>> {
        if self.is_null_or_missing(col_idx) {
            return Ok(None);
        }
        self.get_inet4(col_idx).map(Some)
    }

    pub fn get_inet6(&self, col_idx: usize) -> Result<&'a [u8; 16]> {
        let offset = self.get_fixed_col_offset(col_idx);
        self.data[offset..offset + 16]
            .try_into()
            .map_err(|_| eyre::eyre!("insufficient data for inet6 at col {}", col_idx))
    }

    pub fn get_inet6_opt(&self, col_idx: usize) -> Result<Option<&'a [u8; 16]>> {
        if self.is_null_or_missing(col_idx) {
            return Ok(None);
        }
        self.get_inet6(col_idx).map(Some)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ColumnValue {
    Null,
    Fixed { offset: usize, len: usize },
    Variable { idx: usize },
}

pub struct RecordBuilder<'a> {
    schema: &'a Schema,
    null_bitmap: Vec<u8>,
    fixed_data: Vec<u8>,
    var_data: Vec<Vec<u8>>,
    column_values: Vec<ColumnValue>,
}

impl<'a> RecordBuilder<'a> {
    pub fn new(schema: &'a Schema) -> Self {
        let bitmap_size = Schema::null_bitmap_size(schema.column_count());
        let mut null_bitmap = vec![0u8; bitmap_size];
        for i in 0..schema.column_count() {
            let byte_idx = i / 8;
            let bit_idx = i % 8;
            null_bitmap[byte_idx] |= 1 << bit_idx;
        }

        let fixed_data = vec![0u8; schema.total_fixed_size()];
        let var_data = vec![Vec::new(); schema.var_column_count()];
        let column_values = vec![ColumnValue::Null; schema.column_count()];

        Self {
            schema,
            null_bitmap,
            fixed_data,
            var_data,
            column_values,
        }
    }

    pub fn reset(&mut self) {
        for i in 0..self.schema.column_count() {
            let byte_idx = i / 8;
            let bit_idx = i % 8;
            self.null_bitmap[byte_idx] |= 1 << bit_idx;
        }
        self.fixed_data.fill(0);
        for var in &mut self.var_data {
            var.clear();
        }
        for val in &mut self.column_values {
            *val = ColumnValue::Null;
        }
    }

    pub fn set_null(&mut self, col_idx: usize) {
        let byte_idx = col_idx / 8;
        let bit_idx = col_idx % 8;
        self.null_bitmap[byte_idx] |= 1 << bit_idx;
        self.column_values[col_idx] = ColumnValue::Null;
    }

    fn clear_null(&mut self, col_idx: usize) {
        let byte_idx = col_idx / 8;
        let bit_idx = col_idx % 8;
        self.null_bitmap[byte_idx] &= !(1 << bit_idx);
    }

    fn set_fixed_bytes(&mut self, col_idx: usize, bytes: &[u8]) {
        self.clear_null(col_idx);
        let offset = self.schema.fixed_offset(col_idx);
        self.fixed_data[offset..offset + bytes.len()].copy_from_slice(bytes);
        self.column_values[col_idx] = ColumnValue::Fixed {
            offset,
            len: bytes.len(),
        };
    }

    pub fn set_bool(&mut self, col_idx: usize, value: bool) -> Result<()> {
        self.set_fixed_bytes(col_idx, &[if value { 1 } else { 0 }]);
        Ok(())
    }

    pub fn set_int2(&mut self, col_idx: usize, value: i16) -> Result<()> {
        self.set_fixed_bytes(col_idx, &value.to_le_bytes());
        Ok(())
    }

    pub fn set_int4(&mut self, col_idx: usize, value: i32) -> Result<()> {
        self.set_fixed_bytes(col_idx, &value.to_le_bytes());
        Ok(())
    }

    pub fn set_int8(&mut self, col_idx: usize, value: i64) -> Result<()> {
        self.set_fixed_bytes(col_idx, &value.to_le_bytes());
        Ok(())
    }

    pub fn set_float4(&mut self, col_idx: usize, value: f32) -> Result<()> {
        self.set_fixed_bytes(col_idx, &value.to_le_bytes());
        Ok(())
    }

    pub fn set_float8(&mut self, col_idx: usize, value: f64) -> Result<()> {
        self.set_fixed_bytes(col_idx, &value.to_le_bytes());
        Ok(())
    }

    pub fn set_date(&mut self, col_idx: usize, days: i32) -> Result<()> {
        self.set_fixed_bytes(col_idx, &days.to_le_bytes());
        Ok(())
    }

    pub fn set_time(&mut self, col_idx: usize, micros: i64) -> Result<()> {
        self.set_fixed_bytes(col_idx, &micros.to_le_bytes());
        Ok(())
    }

    pub fn set_timestamp(&mut self, col_idx: usize, micros: i64) -> Result<()> {
        self.set_fixed_bytes(col_idx, &micros.to_le_bytes());
        Ok(())
    }

    pub fn set_uuid(&mut self, col_idx: usize, uuid: &[u8; 16]) -> Result<()> {
        self.set_fixed_bytes(col_idx, uuid);
        Ok(())
    }

    pub fn set_macaddr(&mut self, col_idx: usize, mac: &[u8; 6]) -> Result<()> {
        self.set_fixed_bytes(col_idx, mac);
        Ok(())
    }

    pub fn set_timestamptz(&mut self, col_idx: usize, micros: i64, offset_secs: i32) -> Result<()> {
        self.clear_null(col_idx);
        let offset = self.schema.fixed_offset(col_idx);
        self.fixed_data[offset..offset + 8].copy_from_slice(&micros.to_le_bytes());
        self.fixed_data[offset + 8..offset + 12].copy_from_slice(&offset_secs.to_le_bytes());
        self.column_values[col_idx] = ColumnValue::Fixed { offset, len: 12 };
        Ok(())
    }

    pub fn set_inet4(&mut self, col_idx: usize, ip: &[u8; 4]) -> Result<()> {
        self.set_fixed_bytes(col_idx, ip);
        Ok(())
    }

    pub fn set_inet6(&mut self, col_idx: usize, ip: &[u8; 16]) -> Result<()> {
        self.set_fixed_bytes(col_idx, ip);
        Ok(())
    }

    pub fn set_vector(&mut self, col_idx: usize, vec: &[f32]) -> Result<()> {
        self.clear_null(col_idx);
        let var_idx = self
            .schema
            .var_column_index(col_idx)
            .ok_or_else(|| eyre::eyre!("column {} is not a variable column", col_idx))?;
        let mut bytes = Vec::with_capacity(4 + vec.len() * 4);
        bytes.extend((vec.len() as u32).to_le_bytes());
        for &f in vec {
            bytes.extend(f.to_le_bytes());
        }
        self.var_data[var_idx] = bytes;
        self.column_values[col_idx] = ColumnValue::Variable { idx: var_idx };
        Ok(())
    }

    pub fn set_text(&mut self, col_idx: usize, text: &str) -> Result<()> {
        self.set_blob(col_idx, text.as_bytes())
    }

    pub fn set_blob(&mut self, col_idx: usize, data: &[u8]) -> Result<()> {
        self.clear_null(col_idx);
        let var_idx = self
            .schema
            .var_column_index(col_idx)
            .ok_or_else(|| eyre::eyre!("column {} is not a variable column", col_idx))?;
        self.var_data[var_idx] = data.to_vec();
        self.column_values[col_idx] = ColumnValue::Variable { idx: var_idx };
        Ok(())
    }

    pub fn set_jsonb(&mut self, col_idx: usize, jsonb: &JsonbBuilder) -> Result<()> {
        self.clear_null(col_idx);
        let var_idx = self
            .schema
            .var_column_index(col_idx)
            .ok_or_else(|| eyre::eyre!("column {} is not a variable column", col_idx))?;
        self.var_data[var_idx] = jsonb.build();
        self.column_values[col_idx] = ColumnValue::Variable { idx: var_idx };
        Ok(())
    }

    pub fn set_jsonb_bytes(&mut self, col_idx: usize, data: &[u8]) -> Result<()> {
        self.clear_null(col_idx);
        let var_idx = self
            .schema
            .var_column_index(col_idx)
            .ok_or_else(|| eyre::eyre!("column {} is not a variable column", col_idx))?;
        self.var_data[var_idx] = data.to_vec();
        self.column_values[col_idx] = ColumnValue::Variable { idx: var_idx };
        Ok(())
    }

    pub fn build(&self) -> Result<Vec<u8>> {
        let bitmap_size = self.null_bitmap.len();
        let offset_table_size = self.schema.var_column_count() * 2;
        let header_len = 2 + bitmap_size + offset_table_size;

        let mut result = Vec::with_capacity(
            header_len
                + self.fixed_data.len()
                + self.var_data.iter().map(|v| v.len()).sum::<usize>(),
        );

        result.extend((header_len as u16).to_le_bytes());
        result.extend(&self.null_bitmap);

        let mut var_offset: u16 = 0;
        for var_data in &self.var_data {
            var_offset += var_data.len() as u16;
            result.extend(var_offset.to_le_bytes());
        }

        result.extend(&self.fixed_data);

        for var_data in &self.var_data {
            result.extend(var_data);
        }

        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn record_view_can_be_created_with_data_and_schema() {
        let schema = Schema::new(vec![
            ColumnDef::new("id", DataType::Int4),
            ColumnDef::new("name", DataType::Text),
        ]);

        let data = vec![0x00, 0x00];

        let view = RecordView::new(&data, &schema);
        assert!(view.is_ok());

        let view = view.unwrap();
        assert_eq!(view.data().len(), 2);
        assert_eq!(view.schema().column_count(), 2);
    }

    #[test]
    fn record_view_rejects_empty_data() {
        let schema = Schema::new(vec![ColumnDef::new("id", DataType::Int4)]);
        let data: Vec<u8> = vec![];

        let result = RecordView::new(&data, &schema);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("empty"));
    }

    #[test]
    fn record_view_borrows_data_zero_copy() {
        let schema = Schema::new(vec![ColumnDef::new("id", DataType::Int4)]);
        let data = vec![0x10, 0x00, 0x01, 0x02, 0x03, 0x04];

        let view = RecordView::new(&data, &schema).unwrap();

        assert!(std::ptr::eq(view.data().as_ptr(), data.as_ptr()));
    }

    #[test]
    fn schema_tracks_fixed_and_variable_columns() {
        let schema = Schema::new(vec![
            ColumnDef::new("id", DataType::Int4),
            ColumnDef::new("name", DataType::Text),
            ColumnDef::new("age", DataType::Int2),
            ColumnDef::new("bio", DataType::Blob),
        ]);

        assert_eq!(schema.column_count(), 4);
        assert_eq!(schema.var_column_count(), 2);

        assert_eq!(schema.var_column_index(1), Some(0));
        assert_eq!(schema.var_column_index(3), Some(1));
        assert_eq!(schema.var_column_index(0), None);
        assert_eq!(schema.var_column_index(2), None);
    }

    #[test]
    fn schema_calculates_fixed_offsets() {
        let schema = Schema::new(vec![
            ColumnDef::new("a", DataType::Int4),
            ColumnDef::new("b", DataType::Int8),
            ColumnDef::new("c", DataType::Text),
            ColumnDef::new("d", DataType::Int2),
        ]);

        assert_eq!(schema.fixed_offset(0), 0);
        assert_eq!(schema.fixed_offset(1), 4);
        assert_eq!(schema.fixed_offset(2), 12);
        assert_eq!(schema.fixed_offset(3), 12);

        assert_eq!(schema.total_fixed_size(), 14);
    }

    #[test]
    fn data_type_fixed_sizes() {
        assert_eq!(DataType::Bool.fixed_size(), Some(1));
        assert_eq!(DataType::Int2.fixed_size(), Some(2));
        assert_eq!(DataType::Int4.fixed_size(), Some(4));
        assert_eq!(DataType::Int8.fixed_size(), Some(8));
        assert_eq!(DataType::Float4.fixed_size(), Some(4));
        assert_eq!(DataType::Float8.fixed_size(), Some(8));
        assert_eq!(DataType::Date.fixed_size(), Some(4));
        assert_eq!(DataType::Time.fixed_size(), Some(8));
        assert_eq!(DataType::Timestamp.fixed_size(), Some(8));
        assert_eq!(DataType::Uuid.fixed_size(), Some(16));
        assert_eq!(DataType::MacAddr.fixed_size(), Some(6));
        assert_eq!(DataType::Text.fixed_size(), None);
        assert_eq!(DataType::Blob.fixed_size(), None);
    }

    #[test]
    fn data_type_is_variable() {
        assert!(!DataType::Int4.is_variable());
        assert!(DataType::Text.is_variable());
        assert!(DataType::Blob.is_variable());
    }

    #[test]
    fn record_view_header_len_parses_little_endian() {
        let schema = Schema::new(vec![ColumnDef::new("id", DataType::Int4)]);
        let data = vec![0x05, 0x00, 0x00, 0x00, 0x00];

        let view = RecordView::new(&data, &schema).unwrap();
        assert_eq!(view.header_len(), 5);
    }

    #[test]
    fn record_view_header_len_larger_value() {
        let schema = Schema::new(vec![ColumnDef::new("id", DataType::Int4)]);
        let mut data = vec![0x00, 0x01];
        data.resize(256 + 10, 0);

        let view = RecordView::new(&data, &schema).unwrap();
        assert_eq!(view.header_len(), 256);
    }

    #[test]
    fn record_view_null_bitmap_size_calculation() {
        assert_eq!(Schema::null_bitmap_size(1), 1);
        assert_eq!(Schema::null_bitmap_size(8), 1);
        assert_eq!(Schema::null_bitmap_size(9), 2);
        assert_eq!(Schema::null_bitmap_size(16), 2);
        assert_eq!(Schema::null_bitmap_size(17), 3);
    }

    #[test]
    fn record_view_null_bitmap_slice() {
        let schema = Schema::new(vec![
            ColumnDef::new("a", DataType::Int4),
            ColumnDef::new("b", DataType::Int4),
            ColumnDef::new("c", DataType::Int4),
        ]);

        let data = vec![0x05, 0x00, 0b0000_0101, 0x00, 0x00];

        let view = RecordView::new(&data, &schema).unwrap();
        let bitmap = view.null_bitmap();
        assert_eq!(bitmap.len(), 1);
        assert_eq!(bitmap[0], 0b0000_0101);
    }

    #[test]
    fn record_view_offset_table_slice() {
        let schema = Schema::new(vec![
            ColumnDef::new("id", DataType::Int4),
            ColumnDef::new("name", DataType::Text),
            ColumnDef::new("bio", DataType::Blob),
        ]);

        let data = vec![0x09, 0x00, 0x00, 0x10, 0x00, 0x20, 0x00, 0x00, 0x00];

        let view = RecordView::new(&data, &schema).unwrap();
        let offsets = view.offset_table();
        assert_eq!(offsets.len(), 4);
        assert_eq!(offsets[0], 0x10);
        assert_eq!(offsets[1], 0x00);
        assert_eq!(offsets[2], 0x20);
        assert_eq!(offsets[3], 0x00);
    }

    #[test]
    fn record_view_data_payload_offset() {
        let schema = Schema::new(vec![ColumnDef::new("id", DataType::Int4)]);
        let data = vec![0x04, 0x00, 0x00, 0x01, 0x02, 0x03, 0x04];

        let view = RecordView::new(&data, &schema).unwrap();
        assert_eq!(view.data_offset(), 4);
    }

    #[test]
    fn is_null_checks_bitmap_bit_correctly() {
        let schema = Schema::new(vec![
            ColumnDef::new("a", DataType::Int4),
            ColumnDef::new("b", DataType::Int4),
            ColumnDef::new("c", DataType::Int4),
            ColumnDef::new("d", DataType::Int4),
        ]);

        let data = vec![0x03, 0x00, 0b0000_0101];

        let view = RecordView::new(&data, &schema).unwrap();

        assert!(view.is_null(0));
        assert!(!view.is_null(1));
        assert!(view.is_null(2));
        assert!(!view.is_null(3));
    }

    #[test]
    fn is_null_handles_multi_byte_bitmap() {
        let schema = Schema::new(vec![
            ColumnDef::new("c0", DataType::Int4),
            ColumnDef::new("c1", DataType::Int4),
            ColumnDef::new("c2", DataType::Int4),
            ColumnDef::new("c3", DataType::Int4),
            ColumnDef::new("c4", DataType::Int4),
            ColumnDef::new("c5", DataType::Int4),
            ColumnDef::new("c6", DataType::Int4),
            ColumnDef::new("c7", DataType::Int4),
            ColumnDef::new("c8", DataType::Int4),
            ColumnDef::new("c9", DataType::Int4),
        ]);

        let data = vec![0x04, 0x00, 0b1000_0001, 0b0000_0010];

        let view = RecordView::new(&data, &schema).unwrap();

        assert!(view.is_null(0));
        assert!(!view.is_null(1));
        assert!(!view.is_null(2));
        assert!(!view.is_null(3));
        assert!(!view.is_null(4));
        assert!(!view.is_null(5));
        assert!(!view.is_null(6));
        assert!(view.is_null(7));
        assert!(!view.is_null(8));
        assert!(view.is_null(9));
    }

    #[test]
    fn is_null_all_null_columns() {
        let schema = Schema::new(vec![
            ColumnDef::new("a", DataType::Int4),
            ColumnDef::new("b", DataType::Int4),
        ]);

        let data = vec![0x03, 0x00, 0b0000_0011];

        let view = RecordView::new(&data, &schema).unwrap();

        assert!(view.is_null(0));
        assert!(view.is_null(1));
    }

    #[test]
    fn is_null_no_null_columns() {
        let schema = Schema::new(vec![
            ColumnDef::new("a", DataType::Int4),
            ColumnDef::new("b", DataType::Int4),
        ]);

        let data = vec![0x03, 0x00, 0b0000_0000];

        let view = RecordView::new(&data, &schema).unwrap();

        assert!(!view.is_null(0));
        assert!(!view.is_null(1));
    }

    #[test]
    fn get_fixed_col_offset_calculates_correctly() {
        let schema = Schema::new(vec![
            ColumnDef::new("a", DataType::Int4),
            ColumnDef::new("b", DataType::Int8),
            ColumnDef::new("c", DataType::Int2),
        ]);

        let data = vec![0x03, 0x00, 0x00];

        let view = RecordView::new(&data, &schema).unwrap();

        assert_eq!(view.get_fixed_col_offset(0), 3);
        assert_eq!(view.get_fixed_col_offset(1), 7);
        assert_eq!(view.get_fixed_col_offset(2), 15);
    }

    #[test]
    fn get_var_bounds_reads_offset_table() {
        let schema = Schema::new(vec![
            ColumnDef::new("id", DataType::Int4),
            ColumnDef::new("name", DataType::Text),
            ColumnDef::new("bio", DataType::Blob),
        ]);

        let data = vec![
            0x0B, 0x00, 0x00, 0x05, 0x00, 0x0B, 0x00, 0x01, 0x02, 0x03, 0x04, b'h', b'e', b'l',
            b'l', b'o', b'b', b'i', b'o', b'!', b'!', b'!',
        ];

        let view = RecordView::new(&data, &schema).unwrap();

        let (start, end) = view.get_var_bounds(1).unwrap();
        assert_eq!(start, 15);
        assert_eq!(end, 20);

        let (start, end) = view.get_var_bounds(2).unwrap();
        assert_eq!(start, 20);
        assert_eq!(end, 26);
    }

    #[test]
    fn get_int4_reads_little_endian() {
        let schema = Schema::new(vec![ColumnDef::new("id", DataType::Int4)]);

        let data = vec![0x03, 0x00, 0x00, 0x2A, 0x00, 0x00, 0x00];

        let view = RecordView::new(&data, &schema).unwrap();

        assert_eq!(view.get_int4(0).unwrap(), 42);
    }

    #[test]
    fn get_int4_negative_value() {
        let schema = Schema::new(vec![ColumnDef::new("id", DataType::Int4)]);

        let mut data = vec![0x03, 0x00, 0x00];
        data.extend((-100i32).to_le_bytes());

        let view = RecordView::new(&data, &schema).unwrap();

        assert_eq!(view.get_int4(0).unwrap(), -100);
    }

    #[test]
    fn get_int2_reads_correctly() {
        let schema = Schema::new(vec![ColumnDef::new("val", DataType::Int2)]);

        let data = vec![0x03, 0x00, 0x00, 0xD2, 0x04];

        let view = RecordView::new(&data, &schema).unwrap();

        assert_eq!(view.get_int2(0).unwrap(), 1234);
    }

    #[test]
    fn get_int8_reads_correctly() {
        let schema = Schema::new(vec![ColumnDef::new("val", DataType::Int8)]);

        let mut data = vec![0x03, 0x00, 0x00];
        data.extend(123456789012345_i64.to_le_bytes());

        let view = RecordView::new(&data, &schema).unwrap();

        assert_eq!(view.get_int8(0).unwrap(), 123456789012345);
    }

    #[test]
    fn get_float4_reads_correctly() {
        let schema = Schema::new(vec![ColumnDef::new("val", DataType::Float4)]);

        let mut data = vec![0x03, 0x00, 0x00];
        data.extend(1.25_f32.to_le_bytes());

        let view = RecordView::new(&data, &schema).unwrap();

        let val = view.get_float4(0).unwrap();
        assert!((val - 1.25).abs() < 0.001);
    }

    #[test]
    fn get_float8_reads_correctly() {
        let schema = Schema::new(vec![ColumnDef::new("val", DataType::Float8)]);

        let mut data = vec![0x03, 0x00, 0x00];
        data.extend(1.23456789012345_f64.to_le_bytes());

        let view = RecordView::new(&data, &schema).unwrap();

        let val = view.get_float8(0).unwrap();
        assert!((val - 1.23456789012345).abs() < 1e-10);
    }

    #[test]
    fn get_text_returns_zero_copy_str() {
        let schema = Schema::new(vec![ColumnDef::new("name", DataType::Text)]);

        let data = vec![0x05, 0x00, 0x00, 0x05, 0x00, b'h', b'e', b'l', b'l', b'o'];

        let view = RecordView::new(&data, &schema).unwrap();

        let text = view.get_text(0).unwrap();
        assert_eq!(text, "hello");

        let text_ptr = text.as_ptr();
        let data_ptr = data.as_ptr();
        assert!(text_ptr >= data_ptr && text_ptr < unsafe { data_ptr.add(data.len()) });
    }

    #[test]
    fn get_blob_returns_zero_copy_bytes() {
        let schema = Schema::new(vec![ColumnDef::new("data", DataType::Blob)]);

        let data = vec![0x05, 0x00, 0x00, 0x03, 0x00, 0xDE, 0xAD, 0xBE];

        let view = RecordView::new(&data, &schema).unwrap();

        let blob = view.get_blob(0).unwrap();
        assert_eq!(blob, &[0xDE, 0xAD, 0xBE]);

        let blob_ptr = blob.as_ptr();
        let data_ptr = data.as_ptr();
        assert!(blob_ptr >= data_ptr && blob_ptr < unsafe { data_ptr.add(data.len()) });
    }

    #[test]
    fn get_date_reads_days_since_epoch() {
        let schema = Schema::new(vec![ColumnDef::new("d", DataType::Date)]);

        let mut data = vec![0x03, 0x00, 0x00];
        data.extend(19000_i32.to_le_bytes());

        let view = RecordView::new(&data, &schema).unwrap();

        assert_eq!(view.get_date(0).unwrap(), 19000);
    }

    #[test]
    fn get_time_reads_microseconds() {
        let schema = Schema::new(vec![ColumnDef::new("t", DataType::Time)]);

        let mut data = vec![0x03, 0x00, 0x00];
        data.extend(43200000000_i64.to_le_bytes());

        let view = RecordView::new(&data, &schema).unwrap();

        assert_eq!(view.get_time(0).unwrap(), 43200000000);
    }

    #[test]
    fn get_timestamp_reads_microseconds_since_epoch() {
        let schema = Schema::new(vec![ColumnDef::new("ts", DataType::Timestamp)]);

        let mut data = vec![0x03, 0x00, 0x00];
        data.extend(1702300000000000_i64.to_le_bytes());

        let view = RecordView::new(&data, &schema).unwrap();

        assert_eq!(view.get_timestamp(0).unwrap(), 1702300000000000);
    }

    #[test]
    fn get_uuid_returns_reference() {
        let schema = Schema::new(vec![ColumnDef::new("id", DataType::Uuid)]);

        let uuid_bytes: [u8; 16] = [
            0x12, 0x34, 0x56, 0x78, 0x9a, 0xbc, 0xde, 0xf0, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66,
            0x77, 0x88,
        ];

        let mut data = vec![0x03, 0x00, 0x00];
        data.extend(uuid_bytes);

        let view = RecordView::new(&data, &schema).unwrap();

        let uuid = view.get_uuid(0).unwrap();
        assert_eq!(uuid, &uuid_bytes);

        let uuid_ptr = uuid.as_ptr();
        let data_ptr = data.as_ptr();
        assert!(uuid_ptr >= data_ptr && uuid_ptr < unsafe { data_ptr.add(data.len()) });
    }

    #[test]
    fn get_bool_reads_correctly() {
        let schema = Schema::new(vec![ColumnDef::new("flag", DataType::Bool)]);

        let data_true = vec![0x03, 0x00, 0x00, 0x01];

        let data_false = vec![0x03, 0x00, 0x00, 0x00];

        let view_true = RecordView::new(&data_true, &schema).unwrap();
        let view_false = RecordView::new(&data_false, &schema).unwrap();

        assert!(view_true.get_bool(0).unwrap());
        assert!(!view_false.get_bool(0).unwrap());
    }

    #[test]
    fn get_macaddr_returns_reference() {
        let schema = Schema::new(vec![ColumnDef::new("mac", DataType::MacAddr)]);

        let mac_bytes: [u8; 6] = [0xAA, 0xBB, 0xCC, 0xDD, 0xEE, 0xFF];

        let mut data = vec![0x03, 0x00, 0x00];
        data.extend(mac_bytes);

        let view = RecordView::new(&data, &schema).unwrap();

        let mac = view.get_macaddr(0).unwrap();
        assert_eq!(mac, &mac_bytes);
    }

    #[test]
    fn schema_evolution_record_column_count() {
        let schema = Schema::new(vec![
            ColumnDef::new("a", DataType::Int4),
            ColumnDef::new("b", DataType::Int4),
        ]);

        let data = vec![0x03, 0x00, 0b0000_0000, 0x01, 0x00, 0x00, 0x00];

        let view = RecordView::new(&data, &schema).unwrap();
        assert_eq!(view.record_column_count(), 1);
    }

    #[test]
    fn schema_evolution_column_beyond_record_is_null() {
        let schema = Schema::new(vec![
            ColumnDef::new("id", DataType::Int4),
            ColumnDef::new("age", DataType::Int4),
            ColumnDef::new("score", DataType::Int4),
        ]);

        let data = vec![0x03, 0x00, 0x00, 0x2A, 0x00, 0x00, 0x00];

        let view = RecordView::new(&data, &schema).unwrap();

        assert!(!view.is_null_or_missing(0));
        assert!(view.is_null_or_missing(1));
        assert!(view.is_null_or_missing(2));
    }

    #[test]
    fn schema_evolution_get_optional_returns_none_for_missing() {
        let schema = Schema::new(vec![
            ColumnDef::new("id", DataType::Int4),
            ColumnDef::new("age", DataType::Int4),
        ]);

        let data = vec![0x03, 0x00, 0x00, 0x2A, 0x00, 0x00, 0x00];

        let view = RecordView::new(&data, &schema).unwrap();

        assert_eq!(view.get_int4_opt(0).unwrap(), Some(42));
        assert_eq!(view.get_int4_opt(1).unwrap(), None);
    }

    #[test]
    fn schema_evolution_mixed_null_and_missing() {
        let schema = Schema::new(vec![
            ColumnDef::new("a", DataType::Int4),
            ColumnDef::new("b", DataType::Int4),
            ColumnDef::new("c", DataType::Int4),
        ]);

        let data = vec![0x03, 0x00, 0b0000_0001, 0x2A, 0x00, 0x00, 0x00];

        let view = RecordView::new(&data, &schema).unwrap();

        assert!(view.is_null_or_missing(0));
        assert!(view.is_null_or_missing(2));

        assert_eq!(view.get_int4_opt(0).unwrap(), None);
        assert_eq!(view.get_int4_opt(2).unwrap(), None);
    }

    #[test]
    fn get_bool_opt_returns_none_for_null() {
        let schema = Schema::new(vec![ColumnDef::new("flag", DataType::Bool)]);
        let data = vec![0x03, 0x00, 0b0000_0001, 0x01];

        let view = RecordView::new(&data, &schema).unwrap();
        assert_eq!(view.get_bool_opt(0).unwrap(), None);
    }

    #[test]
    fn get_date_opt_returns_none_for_null() {
        let schema = Schema::new(vec![ColumnDef::new("d", DataType::Date)]);
        let data = vec![0x03, 0x00, 0b0000_0001];

        let view = RecordView::new(&data, &schema).unwrap();
        assert_eq!(view.get_date_opt(0).unwrap(), None);
    }

    #[test]
    fn get_time_opt_returns_none_for_null() {
        let schema = Schema::new(vec![ColumnDef::new("t", DataType::Time)]);
        let data = vec![0x03, 0x00, 0b0000_0001];

        let view = RecordView::new(&data, &schema).unwrap();
        assert_eq!(view.get_time_opt(0).unwrap(), None);
    }

    #[test]
    fn get_timestamp_opt_returns_none_for_null() {
        let schema = Schema::new(vec![ColumnDef::new("ts", DataType::Timestamp)]);
        let data = vec![0x03, 0x00, 0b0000_0001];

        let view = RecordView::new(&data, &schema).unwrap();
        assert_eq!(view.get_timestamp_opt(0).unwrap(), None);
    }

    #[test]
    fn get_uuid_opt_returns_none_for_null() {
        let schema = Schema::new(vec![ColumnDef::new("id", DataType::Uuid)]);
        let data = vec![0x03, 0x00, 0b0000_0001];

        let view = RecordView::new(&data, &schema).unwrap();
        assert_eq!(view.get_uuid_opt(0).unwrap(), None);
    }

    #[test]
    fn get_macaddr_opt_returns_none_for_null() {
        let schema = Schema::new(vec![ColumnDef::new("mac", DataType::MacAddr)]);
        let data = vec![0x03, 0x00, 0b0000_0001];

        let view = RecordView::new(&data, &schema).unwrap();
        assert_eq!(view.get_macaddr_opt(0).unwrap(), None);
    }

    #[test]
    fn data_type_timestamptz_fixed_size() {
        assert_eq!(DataType::TimestampTz.fixed_size(), Some(12));
    }

    #[test]
    fn data_type_inet4_fixed_size() {
        assert_eq!(DataType::Inet4.fixed_size(), Some(4));
    }

    #[test]
    fn data_type_inet6_fixed_size() {
        assert_eq!(DataType::Inet6.fixed_size(), Some(16));
    }

    #[test]
    fn data_type_vector_is_variable() {
        assert!(DataType::Vector.is_variable());
    }

    #[test]
    fn get_timestamptz_reads_correctly() {
        let schema = Schema::new(vec![ColumnDef::new("ts", DataType::TimestampTz)]);

        let mut data = vec![0x03, 0x00, 0x00];
        data.extend(1702300000000000_i64.to_le_bytes());
        data.extend((-300_i32).to_le_bytes());

        let view = RecordView::new(&data, &schema).unwrap();

        let (micros, offset_secs) = view.get_timestamptz(0).unwrap();
        assert_eq!(micros, 1702300000000000);
        assert_eq!(offset_secs, -300);
    }

    #[test]
    fn get_inet4_reads_correctly() {
        let schema = Schema::new(vec![ColumnDef::new("ip", DataType::Inet4)]);

        let data = vec![0x03, 0x00, 0x00, 192, 168, 1, 1];

        let view = RecordView::new(&data, &schema).unwrap();

        let ip = view.get_inet4(0).unwrap();
        assert_eq!(ip, &[192, 168, 1, 1]);
    }

    #[test]
    fn get_inet6_reads_correctly() {
        let schema = Schema::new(vec![ColumnDef::new("ip", DataType::Inet6)]);

        let ipv6: [u8; 16] = [
            0x20, 0x01, 0x0d, 0xb8, 0x85, 0xa3, 0x00, 0x00, 0x00, 0x00, 0x8a, 0x2e, 0x03, 0x70,
            0x73, 0x34,
        ];

        let mut data = vec![0x03, 0x00, 0x00];
        data.extend(ipv6);

        let view = RecordView::new(&data, &schema).unwrap();

        let ip = view.get_inet6(0).unwrap();
        assert_eq!(ip, &ipv6);
    }

    #[test]
    fn record_builder_creates_simple_record() {
        let schema = Schema::new(vec![
            ColumnDef::new("id", DataType::Int4),
            ColumnDef::new("age", DataType::Int2),
        ]);

        let mut builder = RecordBuilder::new(&schema);
        builder.set_int4(0, 42).unwrap();
        builder.set_int2(1, 25).unwrap();

        let data = builder.build().unwrap();

        let view = RecordView::new(&data, &schema).unwrap();
        assert_eq!(view.get_int4(0).unwrap(), 42);
        assert_eq!(view.get_int2(1).unwrap(), 25);
    }

    #[test]
    fn record_builder_handles_null_values() {
        let schema = Schema::new(vec![
            ColumnDef::new("id", DataType::Int4),
            ColumnDef::new("age", DataType::Int4),
        ]);

        let mut builder = RecordBuilder::new(&schema);
        builder.set_int4(0, 42).unwrap();
        builder.set_null(1);

        let data = builder.build().unwrap();

        let view = RecordView::new(&data, &schema).unwrap();
        assert_eq!(view.get_int4(0).unwrap(), 42);
        assert!(view.is_null(1));
    }

    #[test]
    fn record_builder_with_variable_columns() {
        let schema = Schema::new(vec![
            ColumnDef::new("id", DataType::Int4),
            ColumnDef::new("name", DataType::Text),
        ]);

        let mut builder = RecordBuilder::new(&schema);
        builder.set_int4(0, 1).unwrap();
        builder.set_text(1, "hello").unwrap();

        let data = builder.build().unwrap();

        let view = RecordView::new(&data, &schema).unwrap();
        assert_eq!(view.get_int4(0).unwrap(), 1);
        assert_eq!(view.get_text(1).unwrap(), "hello");
    }

    #[test]
    fn record_builder_with_multiple_variable_columns() {
        let schema = Schema::new(vec![
            ColumnDef::new("name", DataType::Text),
            ColumnDef::new("bio", DataType::Blob),
        ]);

        let mut builder = RecordBuilder::new(&schema);
        builder.set_text(0, "alice").unwrap();
        builder.set_blob(1, &[0xDE, 0xAD, 0xBE, 0xEF]).unwrap();

        let data = builder.build().unwrap();

        let view = RecordView::new(&data, &schema).unwrap();
        assert_eq!(view.get_text(0).unwrap(), "alice");
        assert_eq!(view.get_blob(1).unwrap(), &[0xDE, 0xAD, 0xBE, 0xEF]);
    }

    #[test]
    fn record_builder_roundtrip_all_fixed_types() {
        let schema = Schema::new(vec![
            ColumnDef::new("b", DataType::Bool),
            ColumnDef::new("i2", DataType::Int2),
            ColumnDef::new("i4", DataType::Int4),
            ColumnDef::new("i8", DataType::Int8),
            ColumnDef::new("f4", DataType::Float4),
            ColumnDef::new("f8", DataType::Float8),
            ColumnDef::new("d", DataType::Date),
            ColumnDef::new("t", DataType::Time),
            ColumnDef::new("ts", DataType::Timestamp),
            ColumnDef::new("u", DataType::Uuid),
            ColumnDef::new("mac", DataType::MacAddr),
        ]);

        let uuid: [u8; 16] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];
        let mac: [u8; 6] = [0xAA, 0xBB, 0xCC, 0xDD, 0xEE, 0xFF];

        let mut builder = RecordBuilder::new(&schema);
        builder.set_bool(0, true).unwrap();
        builder.set_int2(1, 1234).unwrap();
        builder.set_int4(2, 567890).unwrap();
        builder.set_int8(3, 123456789012345).unwrap();
        builder.set_float4(4, 1.5).unwrap();
        builder.set_float8(5, 2.5).unwrap();
        builder.set_date(6, 19000).unwrap();
        builder.set_time(7, 43200000000).unwrap();
        builder.set_timestamp(8, 1702300000000000).unwrap();
        builder.set_uuid(9, &uuid).unwrap();
        builder.set_macaddr(10, &mac).unwrap();

        let data = builder.build().unwrap();

        let view = RecordView::new(&data, &schema).unwrap();
        assert!(view.get_bool(0).unwrap());
        assert_eq!(view.get_int2(1).unwrap(), 1234);
        assert_eq!(view.get_int4(2).unwrap(), 567890);
        assert_eq!(view.get_int8(3).unwrap(), 123456789012345);
        assert!((view.get_float4(4).unwrap() - 1.5).abs() < 0.001);
        assert!((view.get_float8(5).unwrap() - 2.5).abs() < 0.001);
        assert_eq!(view.get_date(6).unwrap(), 19000);
        assert_eq!(view.get_time(7).unwrap(), 43200000000);
        assert_eq!(view.get_timestamp(8).unwrap(), 1702300000000000);
        assert_eq!(view.get_uuid(9).unwrap(), &uuid);
        assert_eq!(view.get_macaddr(10).unwrap(), &mac);
    }

    #[test]
    fn record_builder_reset_allows_reuse() {
        let schema = Schema::new(vec![
            ColumnDef::new("id", DataType::Int4),
            ColumnDef::new("name", DataType::Text),
        ]);

        let mut builder = RecordBuilder::new(&schema);
        builder.set_int4(0, 100).unwrap();
        builder.set_text(1, "first").unwrap();
        let data1 = builder.build().unwrap();

        let view1 = RecordView::new(&data1, &schema).unwrap();
        assert_eq!(view1.get_int4(0).unwrap(), 100);
        assert_eq!(view1.get_text(1).unwrap(), "first");

        builder.reset();
        builder.set_int4(0, 200).unwrap();
        builder.set_text(1, "second").unwrap();
        let data2 = builder.build().unwrap();

        let view2 = RecordView::new(&data2, &schema).unwrap();
        assert_eq!(view2.get_int4(0).unwrap(), 200);
        assert_eq!(view2.get_text(1).unwrap(), "second");
    }

    #[test]
    fn record_builder_set_timestamptz_roundtrip() {
        let schema = Schema::new(vec![ColumnDef::new("ts", DataType::TimestampTz)]);

        let mut builder = RecordBuilder::new(&schema);
        builder
            .set_timestamptz(0, 1702300000000000, -18000)
            .unwrap();

        let data = builder.build().unwrap();

        let view = RecordView::new(&data, &schema).unwrap();
        let (micros, offset_secs) = view.get_timestamptz(0).unwrap();
        assert_eq!(micros, 1702300000000000);
        assert_eq!(offset_secs, -18000);
    }

    #[test]
    fn record_builder_set_inet4_roundtrip() {
        let schema = Schema::new(vec![ColumnDef::new("ip", DataType::Inet4)]);

        let mut builder = RecordBuilder::new(&schema);
        builder.set_inet4(0, &[192, 168, 1, 1]).unwrap();

        let data = builder.build().unwrap();

        let view = RecordView::new(&data, &schema).unwrap();
        assert_eq!(view.get_inet4(0).unwrap(), &[192, 168, 1, 1]);
    }

    #[test]
    fn record_builder_set_inet6_roundtrip() {
        let schema = Schema::new(vec![ColumnDef::new("ip", DataType::Inet6)]);

        let ipv6: [u8; 16] = [
            0x20, 0x01, 0x0d, 0xb8, 0x85, 0xa3, 0x00, 0x00, 0x00, 0x00, 0x8a, 0x2e, 0x03, 0x70,
            0x73, 0x34,
        ];

        let mut builder = RecordBuilder::new(&schema);
        builder.set_inet6(0, &ipv6).unwrap();

        let data = builder.build().unwrap();

        let view = RecordView::new(&data, &schema).unwrap();
        assert_eq!(view.get_inet6(0).unwrap(), &ipv6);
    }

    #[test]
    fn record_builder_set_vector_roundtrip() {
        let schema = Schema::new(vec![ColumnDef::new("embedding", DataType::Vector)]);

        let vector = vec![1.0_f32, 2.5, -3.0, 0.0, 4.25];

        let mut builder = RecordBuilder::new(&schema);
        builder.set_vector(0, &vector).unwrap();

        let data = builder.build().unwrap();

        let view = RecordView::new(&data, &schema).unwrap();
        let result = view.get_vector_copy(0).unwrap();
        assert_eq!(result.len(), 5);
        for (a, b) in result.iter().zip(vector.iter()) {
            assert!((a - b).abs() < 0.0001);
        }
    }

    #[test]
    fn record_builder_vector_with_fixed_column() {
        let schema = Schema::new(vec![
            ColumnDef::new("id", DataType::Int4),
            ColumnDef::new("embedding", DataType::Vector),
        ]);

        let vector = vec![0.5_f32, 1.5, 2.5];

        let mut builder = RecordBuilder::new(&schema);
        builder.set_int4(0, 42).unwrap();
        builder.set_vector(1, &vector).unwrap();

        let data = builder.build().unwrap();

        let view = RecordView::new(&data, &schema).unwrap();
        assert_eq!(view.get_int4(0).unwrap(), 42);
        let result = view.get_vector_copy(1).unwrap();
        assert_eq!(result.len(), 3);
        assert!((result[0] - 0.5).abs() < 0.0001);
        assert!((result[1] - 1.5).abs() < 0.0001);
        assert!((result[2] - 2.5).abs() < 0.0001);
    }

    #[test]
    fn get_vector_opt_returns_none_for_null() {
        let schema = Schema::new(vec![ColumnDef::new("embedding", DataType::Vector)]);

        let mut builder = RecordBuilder::new(&schema);
        builder.set_null(0);

        let data = builder.build().unwrap();

        let view = RecordView::new(&data, &schema).unwrap();
        assert_eq!(view.get_vector_opt(0).unwrap(), None);
    }

    #[test]
    fn record_builder_empty_vector() {
        let schema = Schema::new(vec![ColumnDef::new("embedding", DataType::Vector)]);

        let vector: Vec<f32> = vec![];

        let mut builder = RecordBuilder::new(&schema);
        builder.set_vector(0, &vector).unwrap();

        let data = builder.build().unwrap();

        let view = RecordView::new(&data, &schema).unwrap();
        let result = view.get_vector_copy(0).unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn record_builder_large_vector() {
        let schema = Schema::new(vec![ColumnDef::new("embedding", DataType::Vector)]);

        let vector: Vec<f32> = (0..1024).map(|i| i as f32 / 100.0).collect();

        let mut builder = RecordBuilder::new(&schema);
        builder.set_vector(0, &vector).unwrap();

        let data = builder.build().unwrap();

        let view = RecordView::new(&data, &schema).unwrap();
        let result = view.get_vector_copy(0).unwrap();
        assert_eq!(result.len(), 1024);
        for (i, (&a, &b)) in result.iter().zip(vector.iter()).enumerate() {
            assert!(
                (a - b).abs() < 0.0001,
                "mismatch at index {}: {} vs {}",
                i,
                a,
                b
            );
        }
    }

    #[test]
    fn data_type_jsonb_is_variable() {
        assert!(DataType::Jsonb.is_variable());
        assert_eq!(DataType::Jsonb.fixed_size(), None);
    }

    #[test]
    fn jsonb_builder_simple_object() {
        let mut builder = JsonbBuilder::new_object();
        builder.set("name", "Alice");
        builder.set("age", 30);

        let data = builder.build();
        let view = JsonbView::new(&data).unwrap();

        assert_eq!(view.root_type(), jsonb::JSONB_TYPE_OBJECT);
        assert_eq!(view.object_len().unwrap(), 2);
    }

    #[test]
    fn jsonb_builder_object_get_key() {
        let mut builder = JsonbBuilder::new_object();
        builder.set("name", "Bob");
        builder.set("score", 95.5);
        builder.set("active", true);

        let data = builder.build();
        let view = JsonbView::new(&data).unwrap();

        let name = view.get("name").unwrap().unwrap();
        assert_eq!(name, JsonbValue::String("Bob"));

        let score = view.get("score").unwrap().unwrap();
        if let JsonbValue::Number(n) = score {
            assert!((n - 95.5).abs() < 0.0001);
        } else {
            panic!("expected number");
        }

        let active = view.get("active").unwrap().unwrap();
        assert_eq!(active, JsonbValue::Bool(true));

        assert!(view.get("missing").unwrap().is_none());
    }

    #[test]
    fn jsonb_builder_array() {
        let mut builder = JsonbBuilder::new_array();
        builder.push(1);
        builder.push(2);
        builder.push(3);

        let data = builder.build();
        let view = JsonbView::new(&data).unwrap();

        assert_eq!(view.root_type(), jsonb::JSONB_TYPE_ARRAY);
        assert_eq!(view.array_len().unwrap(), 3);

        let first = view.array_get(0).unwrap().unwrap();
        if let JsonbValue::Number(n) = first {
            assert!((n - 1.0).abs() < 0.0001);
        } else {
            panic!("expected number");
        }
    }

    #[test]
    fn jsonb_builder_nested_object() {
        let mut inner = JsonbBuilder::new_object();
        inner.set("city", "NYC");
        inner.set("zip", "10001");

        let mut outer = JsonbBuilder::new_object();
        outer.set("name", "Charlie");
        outer.set(
            "address",
            JsonbBuilderValue::Object(vec![
                (
                    "city".to_string(),
                    JsonbBuilderValue::String("NYC".to_string()),
                ),
                (
                    "zip".to_string(),
                    JsonbBuilderValue::String("10001".to_string()),
                ),
            ]),
        );

        let data = outer.build();
        let view = JsonbView::new(&data).unwrap();

        let addr = view.get("address").unwrap().unwrap();
        if let JsonbValue::Object(addr_view) = addr {
            let city = addr_view.get("city").unwrap().unwrap();
            assert_eq!(city, JsonbValue::String("NYC"));
        } else {
            panic!("expected object");
        }
    }

    #[test]
    fn jsonb_get_path() {
        let mut outer = JsonbBuilder::new_object();
        outer.set(
            "user",
            JsonbBuilderValue::Object(vec![
                (
                    "name".to_string(),
                    JsonbBuilderValue::String("Dave".to_string()),
                ),
                (
                    "profile".to_string(),
                    JsonbBuilderValue::Object(vec![(
                        "email".to_string(),
                        JsonbBuilderValue::String("dave@example.com".to_string()),
                    )]),
                ),
            ]),
        );

        let data = outer.build();
        let view = JsonbView::new(&data).unwrap();

        let email = view
            .get_path(&["user", "profile", "email"])
            .unwrap()
            .unwrap();
        assert_eq!(email, JsonbValue::String("dave@example.com"));

        assert!(view.get_path(&["user", "missing"]).unwrap().is_none());
    }

    #[test]
    fn jsonb_null_value() {
        let mut builder = JsonbBuilder::new_object();
        builder.set("value", JsonbBuilderValue::Null);

        let data = builder.build();
        let view = JsonbView::new(&data).unwrap();

        let value = view.get("value").unwrap().unwrap();
        assert_eq!(value, JsonbValue::Null);
    }

    #[test]
    fn jsonb_scalar_types() {
        let null_data = JsonbBuilder::new_null().build();
        let null_view = JsonbView::new(&null_data).unwrap();
        assert_eq!(null_view.as_value().unwrap(), JsonbValue::Null);

        let bool_data = JsonbBuilder::new_bool(true).build();
        let bool_view = JsonbView::new(&bool_data).unwrap();
        assert_eq!(bool_view.as_value().unwrap(), JsonbValue::Bool(true));

        let num_data = JsonbBuilder::new_number(42.5).build();
        let num_view = JsonbView::new(&num_data).unwrap();
        if let JsonbValue::Number(n) = num_view.as_value().unwrap() {
            assert!((n - 42.5).abs() < 0.0001);
        } else {
            panic!("expected number");
        }

        let str_data = JsonbBuilder::new_string("hello").build();
        let str_view = JsonbView::new(&str_data).unwrap();
        assert_eq!(str_view.as_value().unwrap(), JsonbValue::String("hello"));
    }

    #[test]
    fn jsonb_record_roundtrip() {
        let schema = Schema::new(vec![
            ColumnDef::new("id", DataType::Int4),
            ColumnDef::new("data", DataType::Jsonb),
        ]);

        let mut jsonb = JsonbBuilder::new_object();
        jsonb.set("key", "value");
        jsonb.set("count", 42);

        let mut builder = RecordBuilder::new(&schema);
        builder.set_int4(0, 1).unwrap();
        builder.set_jsonb(1, &jsonb).unwrap();

        let record_data = builder.build().unwrap();
        let view = RecordView::new(&record_data, &schema).unwrap();

        assert_eq!(view.get_int4(0).unwrap(), 1);

        let jsonb_view = view.get_jsonb(1).unwrap();
        let key_val = jsonb_view.get("key").unwrap().unwrap();
        assert_eq!(key_val, JsonbValue::String("value"));

        let count_val = jsonb_view.get("count").unwrap().unwrap();
        if let JsonbValue::Number(n) = count_val {
            assert!((n - 42.0).abs() < 0.0001);
        } else {
            panic!("expected number");
        }
    }

    #[test]
    fn jsonb_opt_returns_none_for_null() {
        let schema = Schema::new(vec![ColumnDef::new("data", DataType::Jsonb)]);

        let mut builder = RecordBuilder::new(&schema);
        builder.set_null(0);

        let data = builder.build().unwrap();
        let view = RecordView::new(&data, &schema).unwrap();

        assert!(view.get_jsonb_opt(0).unwrap().is_none());
    }

    #[test]
    fn jsonb_empty_object() {
        let builder = JsonbBuilder::new_object();
        let data = builder.build();
        let view = JsonbView::new(&data).unwrap();

        assert_eq!(view.object_len().unwrap(), 0);
        assert!(view.get("any").unwrap().is_none());
    }

    #[test]
    fn jsonb_empty_array() {
        let builder = JsonbBuilder::new_array();
        let data = builder.build();
        let view = JsonbView::new(&data).unwrap();

        assert_eq!(view.array_len().unwrap(), 0);
        assert!(view.array_get(0).unwrap().is_none());
    }

    #[test]
    fn jsonb_sorted_keys_binary_search() {
        let mut builder = JsonbBuilder::new_object();
        builder.set("zebra", "last");
        builder.set("apple", "first");
        builder.set("middle", "center");

        let data = builder.build();
        let view = JsonbView::new(&data).unwrap();

        assert_eq!(
            view.get("apple").unwrap().unwrap(),
            JsonbValue::String("first")
        );
        assert_eq!(
            view.get("middle").unwrap().unwrap(),
            JsonbValue::String("center")
        );
        assert_eq!(
            view.get("zebra").unwrap().unwrap(),
            JsonbValue::String("last")
        );
    }

    #[test]
    fn jsonb_array_with_mixed_types() {
        let mut builder = JsonbBuilder::new_array();
        builder.push(JsonbBuilderValue::Null);
        builder.push(true);
        builder.push(42);
        builder.push("text");

        let data = builder.build();
        let view = JsonbView::new(&data).unwrap();

        assert_eq!(view.array_len().unwrap(), 4);
        assert_eq!(view.array_get(0).unwrap().unwrap(), JsonbValue::Null);
        assert_eq!(view.array_get(1).unwrap().unwrap(), JsonbValue::Bool(true));
    }
}
