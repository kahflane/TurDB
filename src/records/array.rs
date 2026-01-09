//! # SQL Array Binary Format with O(1) Element Access
//!
//! This module provides PostgreSQL-style SQL arrays with efficient random access.
//! Arrays can contain any scalar type supported by TurDB, with O(1) access for
//! fixed-width element types and O(1) access via offset table for variable-width types.
//!
//! ## Binary Format
//!
//! ```text
//! +------------------+------------------+------------------+------------------+
//! | Header (8 bytes) | Null Bitmap      | Offset Table     | Data Payload     |
//! |                  | [u8; (N+7)/8]    | [u32; N] (if var)| [u8; ...]        |
//! +------------------+------------------+------------------+------------------+
//!
//! Header Layout:
//!   Bytes 0-3: Total size in bytes (u32 LE)
//!   Byte 4:    Element type (DataType as u8)
//!   Byte 5:    Number of dimensions (ndims, always 1 for now)
//!   Bytes 6-7: Element count (u16 LE, max 65535 elements)
//! ```
//!
//! ## Fixed-Width Element Layout
//!
//! For fixed-width types (Int4, Float8, etc.), elements are stored contiguously
//! without an offset table, enabling true O(1) access by computing:
//!   `offset = header_size + null_bitmap_size + (index * element_size)`
//!
//! ```text
//! +----------+-------------+----------------+
//! | Header   | Null Bitmap | Elements       |
//! | (8 bytes)| ((N+7)/8)   | [T; N]         |
//! +----------+-------------+----------------+
//! ```
//!
//! ## Variable-Width Element Layout
//!
//! For variable-width types (Text, Blob), an offset table provides O(1) lookup:
//!
//! ```text
//! +----------+-------------+------------------+------------------+
//! | Header   | Null Bitmap | Offset Table     | Data Payload     |
//! | (8 bytes)| ((N+7)/8)   | [u32; N]         | [u8; ...]        |
//! +----------+-------------+------------------+------------------+
//!
//! Offset Table Entry:
//!   Each entry is a u32 offset from the start of Data Payload section.
//!   The length of element[i] = offset[i+1] - offset[i], or for the last
//!   element, total_size - header_size - bitmap_size - offset_table_size - offset[N-1].
//! ```
//!
//! ## Null Bitmap
//!
//! Null elements are tracked in a bitmap following the header:
//!   - Bit 0 of byte 0 = element 0 null status
//!   - Bit 7 of byte 0 = element 7 null status
//!   - Bit 0 of byte 1 = element 8 null status
//!
//! When an element is null, its data slot contains undefined bytes (for fixed)
//! or the offset table entry points to the same offset as the next element (for var).
//!
//! ## Zero-Copy Design
//!
//! ArrayView borrows the underlying byte slice and provides zero-copy access:
//! - `get_int4(idx)` reads directly from the data section
//! - `get_text(idx)` returns `&str` pointing into the original buffer
//!
//! ## Performance Characteristics
//!
//! | Operation        | Fixed-Width | Variable-Width |
//! |------------------|-------------|----------------|
//! | len()            | O(1)        | O(1)           |
//! | is_null(idx)     | O(1)        | O(1)           |
//! | get_element(idx) | O(1)        | O(1)           |
//! | iteration        | O(n)        | O(n)           |
//!
//! ## Usage Example
//!
//! ```rust,ignore
//! // Building an array
//! let mut builder = ArrayBuilder::new(DataType::Int4);
//! builder.push_int4(10);
//! builder.push_null();
//! builder.push_int4(30);
//! let data = builder.build();
//!
//! // Reading an array
//! let view = ArrayView::new(&data)?;
//! assert_eq!(view.len(), 3);
//! assert_eq!(view.get_int4(0)?, 10);
//! assert!(view.is_null(1));
//! ```

use crate::records::types::DataType;
use eyre::{bail, ensure, Result};

const HEADER_SIZE: usize = 8;

#[derive(Debug, Clone, Copy)]
pub struct ArrayView<'a> {
    data: &'a [u8],
}

impl<'a> ArrayView<'a> {
    pub fn new(data: &'a [u8]) -> Result<Self> {
        ensure!(
            data.len() >= HEADER_SIZE,
            "array data too short: {} bytes, need at least {}",
            data.len(),
            HEADER_SIZE
        );
        Ok(Self { data })
    }

    fn total_size(&self) -> u32 {
        u32::from_le_bytes([self.data[0], self.data[1], self.data[2], self.data[3]])
    }

    pub fn elem_type(&self) -> DataType {
        DataType::try_from(self.data[4]).expect("corrupted array: invalid type byte")
    }

    #[allow(dead_code)]
    pub fn ndims(&self) -> u8 {
        self.data[5]
    }

    pub fn len(&self) -> usize {
        u16::from_le_bytes([self.data[6], self.data[7]]) as usize
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    fn null_bitmap_size(&self) -> usize {
        self.len().div_ceil(8)
    }

    fn null_bitmap_start(&self) -> usize {
        HEADER_SIZE
    }

    pub fn is_null(&self, idx: usize) -> bool {
        if idx >= self.len() {
            return true;
        }
        let byte_idx = self.null_bitmap_start() + idx / 8;
        let bit_idx = idx % 8;
        (self.data[byte_idx] & (1 << bit_idx)) != 0
    }

    fn data_start_fixed(&self) -> usize {
        HEADER_SIZE + self.null_bitmap_size()
    }

    fn offset_table_start(&self) -> usize {
        HEADER_SIZE + self.null_bitmap_size()
    }

    fn data_start_variable(&self) -> usize {
        self.offset_table_start() + self.len() * 4
    }

    fn read_offset(&self, idx: usize) -> u32 {
        let pos = self.offset_table_start() + idx * 4;
        u32::from_le_bytes([
            self.data[pos],
            self.data[pos + 1],
            self.data[pos + 2],
            self.data[pos + 3],
        ])
    }

    fn get_var_bounds(&self, idx: usize) -> Result<(usize, usize)> {
        ensure!(
            idx < self.len(),
            "array index {} out of bounds (len={})",
            idx,
            self.len()
        );

        let data_start = self.data_start_variable();
        let start = self.read_offset(idx) as usize;

        let end = if idx + 1 < self.len() {
            self.read_offset(idx + 1) as usize
        } else {
            self.total_size() as usize - data_start
        };

        Ok((data_start + start, data_start + end))
    }

    pub fn get_int2(&self, idx: usize) -> Result<i16> {
        ensure!(
            idx < self.len(),
            "array index {} out of bounds (len={})",
            idx,
            self.len()
        );
        let offset = self.data_start_fixed() + idx * 2;
        Ok(i16::from_le_bytes([
            self.data[offset],
            self.data[offset + 1],
        ]))
    }

    pub fn get_int4(&self, idx: usize) -> Result<i32> {
        ensure!(
            idx < self.len(),
            "array index {} out of bounds (len={})",
            idx,
            self.len()
        );
        let offset = self.data_start_fixed() + idx * 4;
        Ok(i32::from_le_bytes([
            self.data[offset],
            self.data[offset + 1],
            self.data[offset + 2],
            self.data[offset + 3],
        ]))
    }

    pub fn get_int8(&self, idx: usize) -> Result<i64> {
        ensure!(
            idx < self.len(),
            "array index {} out of bounds (len={})",
            idx,
            self.len()
        );
        let offset = self.data_start_fixed() + idx * 8;
        Ok(i64::from_le_bytes([
            self.data[offset],
            self.data[offset + 1],
            self.data[offset + 2],
            self.data[offset + 3],
            self.data[offset + 4],
            self.data[offset + 5],
            self.data[offset + 6],
            self.data[offset + 7],
        ]))
    }

    pub fn get_float4(&self, idx: usize) -> Result<f32> {
        ensure!(
            idx < self.len(),
            "array index {} out of bounds (len={})",
            idx,
            self.len()
        );
        let offset = self.data_start_fixed() + idx * 4;
        Ok(f32::from_le_bytes([
            self.data[offset],
            self.data[offset + 1],
            self.data[offset + 2],
            self.data[offset + 3],
        ]))
    }

    pub fn get_float8(&self, idx: usize) -> Result<f64> {
        ensure!(
            idx < self.len(),
            "array index {} out of bounds (len={})",
            idx,
            self.len()
        );
        let offset = self.data_start_fixed() + idx * 8;
        Ok(f64::from_le_bytes([
            self.data[offset],
            self.data[offset + 1],
            self.data[offset + 2],
            self.data[offset + 3],
            self.data[offset + 4],
            self.data[offset + 5],
            self.data[offset + 6],
            self.data[offset + 7],
        ]))
    }

    pub fn get_bool(&self, idx: usize) -> Result<bool> {
        ensure!(
            idx < self.len(),
            "array index {} out of bounds (len={})",
            idx,
            self.len()
        );
        let offset = self.data_start_fixed() + idx;
        Ok(self.data[offset] != 0)
    }

    pub fn get_text(&self, idx: usize) -> Result<&'a str> {
        if self.is_null(idx) {
            bail!("array element {} is null", idx);
        }
        let (start, end) = self.get_var_bounds(idx)?;
        let bytes = &self.data[start..end];
        std::str::from_utf8(bytes)
            .map_err(|e| eyre::eyre!("invalid UTF-8 in array text element {}: {}", idx, e))
    }

    pub fn get_blob(&self, idx: usize) -> Result<&'a [u8]> {
        if self.is_null(idx) {
            bail!("array element {} is null", idx);
        }
        let (start, end) = self.get_var_bounds(idx)?;
        Ok(&self.data[start..end])
    }
}

pub struct ArrayBuilder {
    elem_type: DataType,
    null_bitmap: Vec<u8>,
    offsets: Vec<u32>,
    data: Vec<u8>,
    count: u16,
}

impl ArrayBuilder {
    pub fn new(elem_type: DataType) -> Self {
        Self {
            elem_type,
            null_bitmap: Vec::new(),
            offsets: Vec::new(),
            data: Vec::new(),
            count: 0,
        }
    }

    fn set_null_bit(&mut self, idx: usize) {
        let byte_idx = idx / 8;
        let bit_idx = idx % 8;
        while self.null_bitmap.len() <= byte_idx {
            self.null_bitmap.push(0);
        }
        self.null_bitmap[byte_idx] |= 1 << bit_idx;
    }

    fn ensure_bitmap_size(&mut self, count: usize) {
        let needed = count.div_ceil(8);
        while self.null_bitmap.len() < needed {
            self.null_bitmap.push(0);
        }
    }

    pub fn push_null(&mut self) {
        let idx = self.count as usize;
        self.set_null_bit(idx);

        if self.elem_type.is_variable() {
            self.offsets.push(self.data.len() as u32);
        } else if let Some(size) = self.elem_type.fixed_size() {
            self.data.extend(std::iter::repeat_n(0u8, size));
        }

        self.count += 1;
        self.ensure_bitmap_size(self.count as usize);
    }

    pub fn push_int2(&mut self, value: i16) {
        self.ensure_bitmap_size((self.count + 1) as usize);
        self.data.extend(value.to_le_bytes());
        self.count += 1;
    }

    pub fn push_int4(&mut self, value: i32) {
        self.ensure_bitmap_size((self.count + 1) as usize);
        self.data.extend(value.to_le_bytes());
        self.count += 1;
    }

    pub fn push_int8(&mut self, value: i64) {
        self.ensure_bitmap_size((self.count + 1) as usize);
        self.data.extend(value.to_le_bytes());
        self.count += 1;
    }

    pub fn push_float4(&mut self, value: f32) {
        self.ensure_bitmap_size((self.count + 1) as usize);
        self.data.extend(value.to_le_bytes());
        self.count += 1;
    }

    pub fn push_float8(&mut self, value: f64) {
        self.ensure_bitmap_size((self.count + 1) as usize);
        self.data.extend(value.to_le_bytes());
        self.count += 1;
    }

    pub fn push_bool(&mut self, value: bool) {
        self.ensure_bitmap_size((self.count + 1) as usize);
        self.data.push(value as u8);
        self.count += 1;
    }

    pub fn push_text(&mut self, value: &str) {
        self.ensure_bitmap_size((self.count + 1) as usize);
        self.offsets.push(self.data.len() as u32);
        self.data.extend(value.as_bytes());
        self.count += 1;
    }

    pub fn push_blob(&mut self, value: &[u8]) {
        self.ensure_bitmap_size((self.count + 1) as usize);
        self.offsets.push(self.data.len() as u32);
        self.data.extend(value);
        self.count += 1;
    }

    pub fn build(&self) -> Vec<u8> {
        let is_variable = self.elem_type.is_variable();
        let bitmap_size = (self.count as usize).div_ceil(8);
        let offset_table_size = if is_variable {
            self.count as usize * 4
        } else {
            0
        };
        let total_size = HEADER_SIZE + bitmap_size + offset_table_size + self.data.len();

        let mut buf = Vec::with_capacity(total_size);

        buf.extend((total_size as u32).to_le_bytes());
        buf.push(self.elem_type as u8);
        buf.push(1);
        buf.extend(self.count.to_le_bytes());

        let mut bitmap = self.null_bitmap.clone();
        bitmap.resize(bitmap_size, 0);
        buf.extend(&bitmap);

        if is_variable {
            for offset in &self.offsets {
                buf.extend(offset.to_le_bytes());
            }
        }

        buf.extend(&self.data);

        buf
    }

    pub fn reset(&mut self) {
        self.null_bitmap.clear();
        self.offsets.clear();
        self.data.clear();
        self.count = 0;
    }
}
