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
    Varchar = 24,
    Char = 25,
    Decimal = 30,
    Interval = 31,
    Int4Range = 40,
    Int8Range = 41,
    DateRange = 42,
    TimestampRange = 43,
    Enum = 50,
    Point = 60,
    Box = 61,
    Circle = 62,
    Composite = 70,
    Array = 71,
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
            DataType::Varchar => None,
            DataType::Char => None,
            DataType::Decimal => None,
            DataType::Interval => Some(16),
            DataType::Int4Range => Some(9),
            DataType::Int8Range => Some(17),
            DataType::DateRange => Some(9),
            DataType::TimestampRange => Some(17),
            DataType::Enum => Some(4),
            DataType::Point => Some(16),
            DataType::Box => Some(32),
            DataType::Circle => Some(24),
            DataType::Composite => None,
            DataType::Array => None,
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
