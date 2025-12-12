//! # Data Types and Column Definitions
//!
//! This module defines the core type system for TurDB records:
//! - `DataType`: Enumeration of all supported column types
//! - `ColumnDef`: Column definition with name and type
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
