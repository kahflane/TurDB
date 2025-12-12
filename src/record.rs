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
}
