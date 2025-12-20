//! # RecordView - Zero-Copy Record Access
//!
//! This module provides `RecordView` for reading records with O(1) column access.
//! All getters return references into the underlying buffer for zero-copy operation.
//!
//! ## Usage
//!
//! ```ignore
//! let record = RecordView::new(page_data, &schema)?;
//! let name: &str = record.get_text(1)?;  // Zero-copy reference
//! let age: i32 = record.get_int4(2)?;    // Direct read from buffer
//! ```
//!
//! ## Thread Safety
//!
//! `RecordView` borrows immutably from a byte slice. Multiple `RecordView`
//! instances can read the same data concurrently.

use eyre::Result;

use crate::records::array::ArrayView;
use crate::records::composite::CompositeView;
use crate::records::jsonb::JsonbView;
use crate::records::schema::Schema;
use crate::records::types::{range_flags, DecimalView, Range};

#[derive(Debug)]
pub struct RecordView<'a> {
    data: &'a [u8],
    schema: &'a Schema,
}

impl<'a> RecordView<'a> {
    pub fn new(data: &'a [u8], schema: &'a Schema) -> Result<Self> {
        eyre::ensure!(!data.is_empty(), "record data cannot be empty");
        eyre::ensure!(data.len() >= 2, "record too small for header length");
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

    pub fn get_char(&self, col_idx: usize) -> Result<&'a str> {
        self.get_text(col_idx)
    }

    pub fn get_varchar(&self, col_idx: usize) -> Result<&'a str> {
        self.get_text(col_idx)
    }

    pub fn get_blob(&self, col_idx: usize) -> Result<&'a [u8]> {
        let (start, end) = self.get_var_bounds(col_idx)?;
        Ok(&self.data[start..end])
    }

    pub fn get_var_raw(&self, col_idx: usize) -> Result<&'a [u8]> {
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
        // SAFETY: This is safe because:
        // 1. We verified float_bytes.as_ptr() is 4-byte aligned above (returns Err if not)
        // 2. float_bytes.len() >= len * 4, so we have enough bytes for `len` f32 values
        // 3. float_bytes is a valid slice into self.data, which is valid for 'a lifetime
        // 4. f32 has no invalid bit patterns - all 32-bit patterns are valid floats
        // 5. The returned slice lifetime is tied to self, which borrows the underlying data
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

        for col in self.schema.columns() {
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

    pub fn get_interval(&self, col_idx: usize) -> Result<(i64, i32, i32)> {
        let offset = self.get_fixed_col_offset(col_idx);
        let micros_bytes: [u8; 8] = self.data[offset..offset + 8]
            .try_into()
            .map_err(|_| eyre::eyre!("insufficient data for interval micros at col {}", col_idx))?;
        let days_bytes: [u8; 4] = self.data[offset + 8..offset + 12]
            .try_into()
            .map_err(|_| eyre::eyre!("insufficient data for interval days at col {}", col_idx))?;
        let months_bytes: [u8; 4] = self.data[offset + 12..offset + 16]
            .try_into()
            .map_err(|_| eyre::eyre!("insufficient data for interval months at col {}", col_idx))?;
        Ok((
            i64::from_le_bytes(micros_bytes),
            i32::from_le_bytes(days_bytes),
            i32::from_le_bytes(months_bytes),
        ))
    }

    pub fn get_interval_opt(&self, col_idx: usize) -> Result<Option<(i64, i32, i32)>> {
        if self.is_null_or_missing(col_idx) {
            return Ok(None);
        }
        self.get_interval(col_idx).map(Some)
    }

    pub fn get_enum(&self, col_idx: usize) -> Result<(u16, u16)> {
        let offset = self.get_fixed_col_offset(col_idx);
        let type_id_bytes: [u8; 2] = self.data[offset..offset + 2]
            .try_into()
            .map_err(|_| eyre::eyre!("insufficient data for enum type_id at col {}", col_idx))?;
        let ordinal_bytes: [u8; 2] = self.data[offset + 2..offset + 4]
            .try_into()
            .map_err(|_| eyre::eyre!("insufficient data for enum ordinal at col {}", col_idx))?;
        Ok((
            u16::from_le_bytes(type_id_bytes),
            u16::from_le_bytes(ordinal_bytes),
        ))
    }

    pub fn get_enum_opt(&self, col_idx: usize) -> Result<Option<(u16, u16)>> {
        if self.is_null_or_missing(col_idx) {
            return Ok(None);
        }
        self.get_enum(col_idx).map(Some)
    }

    pub fn get_point(&self, col_idx: usize) -> Result<(f64, f64)> {
        let offset = self.get_fixed_col_offset(col_idx);
        let x_bytes: [u8; 8] = self.data[offset..offset + 8]
            .try_into()
            .map_err(|_| eyre::eyre!("insufficient data for point x at col {}", col_idx))?;
        let y_bytes: [u8; 8] = self.data[offset + 8..offset + 16]
            .try_into()
            .map_err(|_| eyre::eyre!("insufficient data for point y at col {}", col_idx))?;
        Ok((f64::from_le_bytes(x_bytes), f64::from_le_bytes(y_bytes)))
    }

    pub fn get_point_opt(&self, col_idx: usize) -> Result<Option<(f64, f64)>> {
        if self.is_null_or_missing(col_idx) {
            return Ok(None);
        }
        self.get_point(col_idx).map(Some)
    }

    pub fn get_box(&self, col_idx: usize) -> Result<((f64, f64), (f64, f64))> {
        let offset = self.get_fixed_col_offset(col_idx);
        let lx_bytes: [u8; 8] = self.data[offset..offset + 8]
            .try_into()
            .map_err(|_| eyre::eyre!("insufficient data for box lx at col {}", col_idx))?;
        let ly_bytes: [u8; 8] = self.data[offset + 8..offset + 16]
            .try_into()
            .map_err(|_| eyre::eyre!("insufficient data for box ly at col {}", col_idx))?;
        let hx_bytes: [u8; 8] = self.data[offset + 16..offset + 24]
            .try_into()
            .map_err(|_| eyre::eyre!("insufficient data for box hx at col {}", col_idx))?;
        let hy_bytes: [u8; 8] = self.data[offset + 24..offset + 32]
            .try_into()
            .map_err(|_| eyre::eyre!("insufficient data for box hy at col {}", col_idx))?;
        Ok((
            (f64::from_le_bytes(lx_bytes), f64::from_le_bytes(ly_bytes)),
            (f64::from_le_bytes(hx_bytes), f64::from_le_bytes(hy_bytes)),
        ))
    }

    #[allow(clippy::type_complexity)]
    pub fn get_box_opt(&self, col_idx: usize) -> Result<Option<((f64, f64), (f64, f64))>> {
        if self.is_null_or_missing(col_idx) {
            return Ok(None);
        }
        self.get_box(col_idx).map(Some)
    }

    pub fn get_circle(&self, col_idx: usize) -> Result<((f64, f64), f64)> {
        let offset = self.get_fixed_col_offset(col_idx);
        let cx_bytes: [u8; 8] = self.data[offset..offset + 8]
            .try_into()
            .map_err(|_| eyre::eyre!("insufficient data for circle cx at col {}", col_idx))?;
        let cy_bytes: [u8; 8] = self.data[offset + 8..offset + 16]
            .try_into()
            .map_err(|_| eyre::eyre!("insufficient data for circle cy at col {}", col_idx))?;
        let r_bytes: [u8; 8] = self.data[offset + 16..offset + 24]
            .try_into()
            .map_err(|_| eyre::eyre!("insufficient data for circle radius at col {}", col_idx))?;
        Ok((
            (f64::from_le_bytes(cx_bytes), f64::from_le_bytes(cy_bytes)),
            f64::from_le_bytes(r_bytes),
        ))
    }

    pub fn get_circle_opt(&self, col_idx: usize) -> Result<Option<((f64, f64), f64)>> {
        if self.is_null_or_missing(col_idx) {
            return Ok(None);
        }
        self.get_circle(col_idx).map(Some)
    }

    pub fn get_int4_range(&self, col_idx: usize) -> Result<Range<i32>> {
        let offset = self.get_fixed_col_offset(col_idx);
        let flags = self.data[offset];

        if flags & range_flags::EMPTY != 0 {
            return Ok(Range::empty());
        }

        let lower = if flags & range_flags::LOWER_INFINITE != 0 {
            None
        } else {
            let bytes: [u8; 4] = self.data[offset + 1..offset + 5].try_into().map_err(|_| {
                eyre::eyre!("insufficient data for int4_range lower at col {}", col_idx)
            })?;
            Some(i32::from_le_bytes(bytes))
        };

        let upper = if flags & range_flags::UPPER_INFINITE != 0 {
            None
        } else {
            let bytes: [u8; 4] = self.data[offset + 5..offset + 9].try_into().map_err(|_| {
                eyre::eyre!("insufficient data for int4_range upper at col {}", col_idx)
            })?;
            Some(i32::from_le_bytes(bytes))
        };

        Ok(Range::new(
            lower,
            upper,
            flags & range_flags::LOWER_INCLUSIVE != 0,
            flags & range_flags::UPPER_INCLUSIVE != 0,
        ))
    }

    pub fn get_int4_range_opt(&self, col_idx: usize) -> Result<Option<Range<i32>>> {
        if self.is_null_or_missing(col_idx) {
            return Ok(None);
        }
        self.get_int4_range(col_idx).map(Some)
    }

    pub fn get_int8_range(&self, col_idx: usize) -> Result<Range<i64>> {
        let offset = self.get_fixed_col_offset(col_idx);
        let flags = self.data[offset];

        if flags & range_flags::EMPTY != 0 {
            return Ok(Range::empty());
        }

        let lower = if flags & range_flags::LOWER_INFINITE != 0 {
            None
        } else {
            let bytes: [u8; 8] = self.data[offset + 1..offset + 9].try_into().map_err(|_| {
                eyre::eyre!("insufficient data for int8_range lower at col {}", col_idx)
            })?;
            Some(i64::from_le_bytes(bytes))
        };

        let upper = if flags & range_flags::UPPER_INFINITE != 0 {
            None
        } else {
            let bytes: [u8; 8] = self.data[offset + 9..offset + 17].try_into().map_err(|_| {
                eyre::eyre!("insufficient data for int8_range upper at col {}", col_idx)
            })?;
            Some(i64::from_le_bytes(bytes))
        };

        Ok(Range::new(
            lower,
            upper,
            flags & range_flags::LOWER_INCLUSIVE != 0,
            flags & range_flags::UPPER_INCLUSIVE != 0,
        ))
    }

    pub fn get_int8_range_opt(&self, col_idx: usize) -> Result<Option<Range<i64>>> {
        if self.is_null_or_missing(col_idx) {
            return Ok(None);
        }
        self.get_int8_range(col_idx).map(Some)
    }

    pub fn get_date_range(&self, col_idx: usize) -> Result<Range<i32>> {
        self.get_int4_range(col_idx)
    }

    pub fn get_date_range_opt(&self, col_idx: usize) -> Result<Option<Range<i32>>> {
        self.get_int4_range_opt(col_idx)
    }

    pub fn get_timestamp_range(&self, col_idx: usize) -> Result<Range<i64>> {
        self.get_int8_range(col_idx)
    }

    pub fn get_timestamp_range_opt(&self, col_idx: usize) -> Result<Option<Range<i64>>> {
        self.get_int8_range_opt(col_idx)
    }

    pub fn get_decimal(&self, col_idx: usize) -> Result<DecimalView<'a>> {
        let (start, end) = self.get_var_bounds(col_idx)?;
        Ok(DecimalView::new(&self.data[start..end]))
    }

    pub fn get_decimal_opt(&self, col_idx: usize) -> Result<Option<DecimalView<'a>>> {
        if self.is_null_or_missing(col_idx) {
            return Ok(None);
        }
        self.get_decimal(col_idx).map(Some)
    }

    pub fn get_composite(&self, col_idx: usize, field_count: usize) -> Result<CompositeView<'a>> {
        let (start, end) = self.get_var_bounds(col_idx)?;
        let bytes = &self.data[start..end];
        CompositeView::new(bytes, field_count)
    }

    pub fn get_composite_opt(
        &self,
        col_idx: usize,
        field_count: usize,
    ) -> Result<Option<CompositeView<'a>>> {
        if self.is_null_or_missing(col_idx) {
            return Ok(None);
        }
        self.get_composite(col_idx, field_count).map(Some)
    }

    pub fn get_array(&self, col_idx: usize) -> Result<ArrayView<'a>> {
        let (start, end) = self.get_var_bounds(col_idx)?;
        let bytes = &self.data[start..end];
        ArrayView::new(bytes)
    }

    pub fn get_array_opt(&self, col_idx: usize) -> Result<Option<ArrayView<'a>>> {
        if self.is_null_or_missing(col_idx) {
            return Ok(None);
        }
        self.get_array(col_idx).map(Some)
    }
}
