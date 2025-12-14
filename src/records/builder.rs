//! # RecordBuilder - Record Construction
//!
//! This module provides `RecordBuilder` for constructing records with type-safe setters.
//! The builder pre-allocates space based on schema and supports reset for zero-alloc reuse.
//!
//! ## Usage
//!
//! ```ignore
//! let mut builder = RecordBuilder::new(&schema);
//! builder.set_int4(0, 42)?;
//! builder.set_text(1, "hello")?;
//! let data = builder.build()?;
//!
//! // Reuse builder for next record
//! builder.reset();
//! builder.set_int4(0, 100)?;
//! ```

use eyre::Result;

use crate::records::jsonb::JsonbBuilder;
use crate::records::schema::Schema;
use crate::records::types::range_flags;

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

    pub fn set_char(&mut self, col_idx: usize, text: &str) -> Result<()> {
        let col = self
            .schema
            .column(col_idx)
            .ok_or_else(|| eyre::eyre!("column {} not found", col_idx))?;
        let max_len = col
            .char_length()
            .ok_or_else(|| eyre::eyre!("CHAR column {} has no length constraint", col_idx))?
            as usize;

        let char_count = text.chars().count();
        if char_count > max_len {
            eyre::bail!(
                "value length {} exceeds CHAR({}) limit for column {}",
                char_count,
                max_len,
                col_idx
            );
        }

        let padded: String = if char_count < max_len {
            let padding = max_len - char_count;
            let mut s = text.to_string();
            s.extend(std::iter::repeat_n(' ', padding));
            s
        } else {
            text.to_string()
        };

        self.set_blob(col_idx, padded.as_bytes())
    }

    pub fn set_varchar(&mut self, col_idx: usize, text: &str) -> Result<()> {
        let col = self
            .schema
            .column(col_idx)
            .ok_or_else(|| eyre::eyre!("column {} not found", col_idx))?;

        if let Some(max_len) = col.char_length() {
            let char_count = text.chars().count();
            if char_count > max_len as usize {
                eyre::bail!(
                    "value length {} exceeds VARCHAR({}) limit for column {}",
                    char_count,
                    max_len,
                    col_idx
                );
            }
        }

        self.set_blob(col_idx, text.as_bytes())
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

    pub fn set_interval(
        &mut self,
        col_idx: usize,
        micros: i64,
        days: i32,
        months: i32,
    ) -> Result<()> {
        self.clear_null(col_idx);
        let offset = self.schema.fixed_offset(col_idx);
        self.fixed_data[offset..offset + 8].copy_from_slice(&micros.to_le_bytes());
        self.fixed_data[offset + 8..offset + 12].copy_from_slice(&days.to_le_bytes());
        self.fixed_data[offset + 12..offset + 16].copy_from_slice(&months.to_le_bytes());
        self.column_values[col_idx] = ColumnValue::Fixed { offset, len: 16 };
        Ok(())
    }

    pub fn set_enum(&mut self, col_idx: usize, type_id: u16, ordinal: u16) -> Result<()> {
        self.clear_null(col_idx);
        let offset = self.schema.fixed_offset(col_idx);
        self.fixed_data[offset..offset + 2].copy_from_slice(&type_id.to_le_bytes());
        self.fixed_data[offset + 2..offset + 4].copy_from_slice(&ordinal.to_le_bytes());
        self.column_values[col_idx] = ColumnValue::Fixed { offset, len: 4 };
        Ok(())
    }

    pub fn set_point(&mut self, col_idx: usize, x: f64, y: f64) -> Result<()> {
        self.clear_null(col_idx);
        let offset = self.schema.fixed_offset(col_idx);
        self.fixed_data[offset..offset + 8].copy_from_slice(&x.to_le_bytes());
        self.fixed_data[offset + 8..offset + 16].copy_from_slice(&y.to_le_bytes());
        self.column_values[col_idx] = ColumnValue::Fixed { offset, len: 16 };
        Ok(())
    }

    pub fn set_box(&mut self, col_idx: usize, low: (f64, f64), high: (f64, f64)) -> Result<()> {
        self.clear_null(col_idx);
        let offset = self.schema.fixed_offset(col_idx);
        self.fixed_data[offset..offset + 8].copy_from_slice(&low.0.to_le_bytes());
        self.fixed_data[offset + 8..offset + 16].copy_from_slice(&low.1.to_le_bytes());
        self.fixed_data[offset + 16..offset + 24].copy_from_slice(&high.0.to_le_bytes());
        self.fixed_data[offset + 24..offset + 32].copy_from_slice(&high.1.to_le_bytes());
        self.column_values[col_idx] = ColumnValue::Fixed { offset, len: 32 };
        Ok(())
    }

    pub fn set_circle(&mut self, col_idx: usize, center: (f64, f64), radius: f64) -> Result<()> {
        self.clear_null(col_idx);
        let offset = self.schema.fixed_offset(col_idx);
        self.fixed_data[offset..offset + 8].copy_from_slice(&center.0.to_le_bytes());
        self.fixed_data[offset + 8..offset + 16].copy_from_slice(&center.1.to_le_bytes());
        self.fixed_data[offset + 16..offset + 24].copy_from_slice(&radius.to_le_bytes());
        self.column_values[col_idx] = ColumnValue::Fixed { offset, len: 24 };
        Ok(())
    }

    pub fn set_int4_range(
        &mut self,
        col_idx: usize,
        lower: Option<i32>,
        upper: Option<i32>,
        lower_inclusive: bool,
        upper_inclusive: bool,
    ) -> Result<()> {
        self.clear_null(col_idx);
        let offset = self.schema.fixed_offset(col_idx);

        let mut flags: u8 = 0;
        if lower_inclusive {
            flags |= range_flags::LOWER_INCLUSIVE;
        }
        if upper_inclusive {
            flags |= range_flags::UPPER_INCLUSIVE;
        }
        if lower.is_none() {
            flags |= range_flags::LOWER_INFINITE;
        }
        if upper.is_none() {
            flags |= range_flags::UPPER_INFINITE;
        }

        self.fixed_data[offset] = flags;
        self.fixed_data[offset + 1..offset + 5].copy_from_slice(&lower.unwrap_or(0).to_le_bytes());
        self.fixed_data[offset + 5..offset + 9].copy_from_slice(&upper.unwrap_or(0).to_le_bytes());
        self.column_values[col_idx] = ColumnValue::Fixed { offset, len: 9 };
        Ok(())
    }

    pub fn set_int4_range_empty(&mut self, col_idx: usize) -> Result<()> {
        self.clear_null(col_idx);
        let offset = self.schema.fixed_offset(col_idx);
        self.fixed_data[offset] = range_flags::EMPTY;
        self.fixed_data[offset + 1..offset + 9].fill(0);
        self.column_values[col_idx] = ColumnValue::Fixed { offset, len: 9 };
        Ok(())
    }

    pub fn set_int8_range(
        &mut self,
        col_idx: usize,
        lower: Option<i64>,
        upper: Option<i64>,
        lower_inclusive: bool,
        upper_inclusive: bool,
    ) -> Result<()> {
        self.clear_null(col_idx);
        let offset = self.schema.fixed_offset(col_idx);

        let mut flags: u8 = 0;
        if lower_inclusive {
            flags |= range_flags::LOWER_INCLUSIVE;
        }
        if upper_inclusive {
            flags |= range_flags::UPPER_INCLUSIVE;
        }
        if lower.is_none() {
            flags |= range_flags::LOWER_INFINITE;
        }
        if upper.is_none() {
            flags |= range_flags::UPPER_INFINITE;
        }

        self.fixed_data[offset] = flags;
        self.fixed_data[offset + 1..offset + 9].copy_from_slice(&lower.unwrap_or(0).to_le_bytes());
        self.fixed_data[offset + 9..offset + 17].copy_from_slice(&upper.unwrap_or(0).to_le_bytes());
        self.column_values[col_idx] = ColumnValue::Fixed { offset, len: 17 };
        Ok(())
    }

    pub fn set_int8_range_empty(&mut self, col_idx: usize) -> Result<()> {
        self.clear_null(col_idx);
        let offset = self.schema.fixed_offset(col_idx);
        self.fixed_data[offset] = range_flags::EMPTY;
        self.fixed_data[offset + 1..offset + 17].fill(0);
        self.column_values[col_idx] = ColumnValue::Fixed { offset, len: 17 };
        Ok(())
    }

    pub fn set_date_range(
        &mut self,
        col_idx: usize,
        lower: Option<i32>,
        upper: Option<i32>,
        lower_inclusive: bool,
        upper_inclusive: bool,
    ) -> Result<()> {
        self.set_int4_range(col_idx, lower, upper, lower_inclusive, upper_inclusive)
    }

    pub fn set_date_range_empty(&mut self, col_idx: usize) -> Result<()> {
        self.set_int4_range_empty(col_idx)
    }

    pub fn set_timestamp_range(
        &mut self,
        col_idx: usize,
        lower: Option<i64>,
        upper: Option<i64>,
        lower_inclusive: bool,
        upper_inclusive: bool,
    ) -> Result<()> {
        self.set_int8_range(col_idx, lower, upper, lower_inclusive, upper_inclusive)
    }

    pub fn set_timestamp_range_empty(&mut self, col_idx: usize) -> Result<()> {
        self.set_int8_range_empty(col_idx)
    }

    pub fn set_decimal(
        &mut self,
        col_idx: usize,
        digits: i128,
        scale: i16,
        is_negative: bool,
    ) -> Result<()> {
        self.clear_null(col_idx);
        let var_idx = self
            .schema
            .var_column_index(col_idx)
            .ok_or_else(|| eyre::eyre!("column {} is not a variable column", col_idx))?;

        let mut bytes = Vec::with_capacity(19);
        bytes.push(if is_negative { 0x80 } else { 0x00 });
        bytes.extend(scale.to_le_bytes());
        bytes.extend(digits.to_le_bytes());

        self.var_data[var_idx] = bytes;
        self.column_values[col_idx] = ColumnValue::Variable { idx: var_idx };
        Ok(())
    }

    pub fn set_composite(&mut self, col_idx: usize, data: &[u8]) -> Result<()> {
        self.clear_null(col_idx);
        let var_idx = self
            .schema
            .var_column_index(col_idx)
            .ok_or_else(|| eyre::eyre!("column {} is not a variable column", col_idx))?;
        self.var_data[var_idx] = data.to_vec();
        self.column_values[col_idx] = ColumnValue::Variable { idx: var_idx };
        Ok(())
    }

    pub fn set_array(&mut self, col_idx: usize, data: &[u8]) -> Result<()> {
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
