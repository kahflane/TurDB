//! # Data Types and Column Definitions for Records
//!
//! This module re-exports `DataType` from the unified types module and provides
//! records-specific types like `ColumnDef` for record building.
//!
//! ## Type Categories
//!
//! | Category | Types | Storage |
//! |----------|-------|---------|
//! | **Fixed** | bool, int2, int4, int8, float4, float8, date, time, timestamp, uuid, macaddr | Direct bytes |
//! | **Variable** | text, blob, vector, jsonb | Offset table + data section |
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
//! | timestamptz | 12 (8 micros + 4 offset) |
//! | uuid | 16 |
//! | macaddr | 6 |
//! | inet4 | 4 |
//! | inet6 | 16 |

pub use crate::types::DataType;

#[derive(Debug, Clone)]
pub struct ColumnDef {
    pub name: String,
    pub data_type: DataType,
    char_length: Option<u32>,
}

impl ColumnDef {
    pub fn new(name: impl Into<String>, data_type: DataType) -> Self {
        Self {
            name: name.into(),
            data_type,
            char_length: None,
        }
    }

    pub fn new_char(name: impl Into<String>, length: u32) -> Self {
        Self {
            name: name.into(),
            data_type: DataType::Char,
            char_length: Some(length),
        }
    }

    pub fn new_varchar(name: impl Into<String>, length: Option<u32>) -> Self {
        Self {
            name: name.into(),
            data_type: DataType::Varchar,
            char_length: length,
        }
    }

    pub fn char_length(&self) -> Option<u32> {
        self.char_length
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Range<T> {
    pub lower: Option<T>,
    pub upper: Option<T>,
    pub lower_inclusive: bool,
    pub upper_inclusive: bool,
    pub is_empty: bool,
}

impl<T> Range<T> {
    pub fn empty() -> Self {
        Self {
            lower: None,
            upper: None,
            lower_inclusive: false,
            upper_inclusive: false,
            is_empty: true,
        }
    }

    pub fn new(
        lower: Option<T>,
        upper: Option<T>,
        lower_inclusive: bool,
        upper_inclusive: bool,
    ) -> Self {
        Self {
            lower,
            upper,
            lower_inclusive,
            upper_inclusive,
            is_empty: false,
        }
    }
}

pub mod range_flags {
    pub const EMPTY: u8 = 0x01;
    pub const LOWER_INCLUSIVE: u8 = 0x02;
    pub const UPPER_INCLUSIVE: u8 = 0x04;
    pub const LOWER_INFINITE: u8 = 0x08;
    pub const UPPER_INFINITE: u8 = 0x10;
}

#[derive(Debug, Clone, Copy)]
pub struct DecimalView<'a> {
    data: &'a [u8],
}

impl<'a> DecimalView<'a> {
    pub fn new(data: &'a [u8]) -> Self {
        Self { data }
    }

    pub fn is_negative(&self) -> bool {
        self.data.first().map(|b| b & 0x80 != 0).unwrap_or(false)
    }

    pub fn scale(&self) -> i16 {
        if self.data.len() < 3 {
            return 0;
        }
        i16::from_le_bytes([self.data[1], self.data[2]])
    }

    pub fn digits(&self) -> i128 {
        if self.data.len() < 19 {
            return 0;
        }
        let bytes: [u8; 16] = self.data[3..19].try_into().unwrap_or([0; 16]);
        i128::from_le_bytes(bytes)
    }
}
