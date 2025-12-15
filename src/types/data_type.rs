//! # Unified Data Type System
//!
//! This module provides the canonical `DataType` enum for TurDB, used across
//! schema definitions, record storage, and query evaluation.
//!
//! ## Design Principles
//!
//! 1. **Single source of truth**: One DataType enum used everywhere
//! 2. **Storage-efficient**: `#[repr(u8)]` for single-byte discriminant
//! 3. **Metadata-free**: Length/dimension stored in ColumnDef, not enum
//! 4. **Complete coverage**: All PostgreSQL-compatible types supported
//!
//! ## Type Categories
//!
//! | Category | Types | Fixed Size |
//! |----------|-------|------------|
//! | **Boolean** | Bool | 1 byte |
//! | **Integer** | Int2, Int4, Int8 | 2, 4, 8 bytes |
//! | **Float** | Float4, Float8 | 4, 8 bytes |
//! | **Date/Time** | Date, Time, Timestamp, TimestampTz, Interval | 4-16 bytes |
//! | **Network** | Uuid, MacAddr, Inet4, Inet6 | 4-16 bytes |
//! | **Geometry** | Point, Box, Circle | 16-32 bytes |
//! | **Range** | Int4Range, Int8Range, DateRange, TimestampRange | 9-17 bytes |
//! | **Text** | Text, Varchar, Char | Variable |
//! | **Binary** | Blob | Variable |
//! | **Structured** | Jsonb, Array, Composite, Enum | Variable |
//! | **Vector** | Vector | Variable |
//! | **Numeric** | Decimal | Variable |
//!
//! ## Discriminant Values
//!
//! Discriminants are grouped by category for cache-friendly comparison:
//! - 0-13: Fixed-width primitives (bool, int, float, datetime, network)
//! - 20-25: Variable-length text/binary
//! - 30-31: Numeric types
//! - 40-43: Range types
//! - 50: Enum
//! - 60-62: Geometry types
//! - 70-71: Composite types
//!
//! ## Storage Encoding
//!
//! The `#[repr(u8)]` ensures the discriminant fits in a single byte,
//! enabling efficient storage in record headers and indexes.
//!
//! ## Type Affinity
//!
//! Following SQLite's type affinity system for flexible typing:
//! - **Integer**: Int2, Int4, Int8, Bool
//! - **Real**: Float4, Float8
//! - **Text**: Text, Varchar, Char, Json, Jsonb
//! - **Blob**: Blob, Vector, Array, Composite
//! - **Numeric**: Decimal, Date, Time, Timestamp, Uuid, etc.
//!
//! ## Usage
//!
//! ```ignore
//! use turdb::types::DataType;
//!
//! let dt = DataType::Int8;
//! assert_eq!(dt.fixed_size(), Some(8));
//! assert!(!dt.is_variable());
//! ```

/// Canonical data type enum for all TurDB operations.
///
/// Uses `#[repr(u8)]` for efficient single-byte storage encoding.
/// Type metadata (VARCHAR length, VECTOR dimension) is stored in `ColumnDef`.
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
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

/// Type affinity for SQLite-compatible type system.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TypeAffinity {
    Integer,
    Real,
    Text,
    Blob,
    Numeric,
}

impl DataType {
    /// Returns the fixed byte size for this type, or None for variable-length types.
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
            DataType::Interval => Some(16),
            DataType::Int4Range => Some(9),
            DataType::Int8Range => Some(17),
            DataType::DateRange => Some(9),
            DataType::TimestampRange => Some(17),
            DataType::Enum => Some(4),
            DataType::Point => Some(16),
            DataType::Box => Some(32),
            DataType::Circle => Some(24),
            DataType::Text
            | DataType::Blob
            | DataType::Vector
            | DataType::Jsonb
            | DataType::Varchar
            | DataType::Char
            | DataType::Decimal
            | DataType::Composite
            | DataType::Array => None,
        }
    }

    /// Returns true if this type requires variable-length encoding.
    pub fn is_variable(&self) -> bool {
        self.fixed_size().is_none()
    }

    /// Returns the SQLite-compatible type affinity for this type.
    pub fn affinity(&self) -> TypeAffinity {
        match self {
            DataType::Int2 | DataType::Int4 | DataType::Int8 | DataType::Bool => {
                TypeAffinity::Integer
            }
            DataType::Float4 | DataType::Float8 => TypeAffinity::Real,
            DataType::Text | DataType::Varchar | DataType::Char | DataType::Jsonb => {
                TypeAffinity::Text
            }
            DataType::Blob | DataType::Vector | DataType::Array | DataType::Composite => {
                TypeAffinity::Blob
            }
            DataType::Date
            | DataType::Time
            | DataType::Timestamp
            | DataType::TimestampTz
            | DataType::Interval
            | DataType::Uuid
            | DataType::MacAddr
            | DataType::Inet4
            | DataType::Inet6
            | DataType::Decimal
            | DataType::Int4Range
            | DataType::Int8Range
            | DataType::DateRange
            | DataType::TimestampRange
            | DataType::Enum
            | DataType::Point
            | DataType::Box
            | DataType::Circle => TypeAffinity::Numeric,
        }
    }

    /// Returns true if this is a numeric type (integer or float).
    pub fn is_numeric(&self) -> bool {
        matches!(
            self,
            DataType::Int2
                | DataType::Int4
                | DataType::Int8
                | DataType::Float4
                | DataType::Float8
                | DataType::Decimal
        )
    }

    /// Returns true if this is a text-like type.
    pub fn is_text(&self) -> bool {
        matches!(
            self,
            DataType::Text | DataType::Varchar | DataType::Char | DataType::Jsonb
        )
    }

    /// Returns true if this is a date/time type.
    pub fn is_datetime(&self) -> bool {
        matches!(
            self,
            DataType::Date
                | DataType::Time
                | DataType::Timestamp
                | DataType::TimestampTz
                | DataType::Interval
        )
    }

    /// Returns true if this is a range type.
    pub fn is_range(&self) -> bool {
        matches!(
            self,
            DataType::Int4Range
                | DataType::Int8Range
                | DataType::DateRange
                | DataType::TimestampRange
        )
    }

    /// Returns true if this is a geometry type.
    pub fn is_geometry(&self) -> bool {
        matches!(self, DataType::Point | DataType::Box | DataType::Circle)
    }

    /// Returns true if this is a network address type.
    pub fn is_network(&self) -> bool {
        matches!(self, DataType::Inet4 | DataType::Inet6 | DataType::MacAddr)
    }
}

impl TryFrom<u8> for DataType {
    type Error = eyre::Report;

    fn try_from(value: u8) -> Result<Self, Self::Error> {
        match value {
            0 => Ok(DataType::Bool),
            1 => Ok(DataType::Int2),
            2 => Ok(DataType::Int4),
            3 => Ok(DataType::Int8),
            4 => Ok(DataType::Float4),
            5 => Ok(DataType::Float8),
            6 => Ok(DataType::Date),
            7 => Ok(DataType::Time),
            8 => Ok(DataType::Timestamp),
            9 => Ok(DataType::TimestampTz),
            10 => Ok(DataType::Uuid),
            11 => Ok(DataType::MacAddr),
            12 => Ok(DataType::Inet4),
            13 => Ok(DataType::Inet6),
            20 => Ok(DataType::Text),
            21 => Ok(DataType::Blob),
            22 => Ok(DataType::Vector),
            23 => Ok(DataType::Jsonb),
            24 => Ok(DataType::Varchar),
            25 => Ok(DataType::Char),
            30 => Ok(DataType::Decimal),
            31 => Ok(DataType::Interval),
            40 => Ok(DataType::Int4Range),
            41 => Ok(DataType::Int8Range),
            42 => Ok(DataType::DateRange),
            43 => Ok(DataType::TimestampRange),
            50 => Ok(DataType::Enum),
            60 => Ok(DataType::Point),
            61 => Ok(DataType::Box),
            62 => Ok(DataType::Circle),
            70 => Ok(DataType::Composite),
            71 => Ok(DataType::Array),
            _ => eyre::bail!("invalid DataType discriminant: {}", value),
        }
    }
}
