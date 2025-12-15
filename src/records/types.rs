//! # Data Types and Column Definitions for Records
//!
//! This module re-exports types from the unified types module for use in
//! record building and processing.
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

pub use crate::types::{range_flags, DataType, DecimalView, Range};

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
