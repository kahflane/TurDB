//! # Column Definitions and Type Metadata
//!
//! This module provides column definitions that pair a `DataType` with
//! additional metadata like VARCHAR length, VECTOR dimension, and ARRAY
//! element types.
//!
//! ## Design
//!
//! The `DataType` enum is metadata-free for storage efficiency. Type metadata
//! is stored separately in `ColumnDef`:
//!
//! - `char_length`: For VARCHAR(n), CHAR(n)
//! - `dimension`: For VECTOR(n)
//! - `element_type`: For ARRAY element type
//! - `precision/scale`: For DECIMAL(p,s)
//!
//! ## Usage
//!
//! ```ignore
//! use turdb::types::{DataType, ColumnDef};
//!
//! // Simple column
//! let id_col = ColumnDef::new("id", DataType::Int8);
//!
//! // VARCHAR with length
//! let name_col = ColumnDef::varchar("name", Some(255));
//!
//! // VECTOR with dimension
//! let embedding_col = ColumnDef::vector("embedding", 1536);
//!
//! // ARRAY of integers
//! let tags_col = ColumnDef::array("tags", DataType::Int4);
//! ```

use super::DataType;

/// Column definition with type and metadata.
#[derive(Debug, Clone)]
pub struct ColumnDef {
    name: String,
    data_type: DataType,
    char_length: Option<u32>,
    dimension: Option<u32>,
    element_type: Option<Box<ColumnDef>>,
    precision: Option<u8>,
    scale: Option<i8>,
}

impl ColumnDef {
    /// Creates a new column definition with the given name and type.
    pub fn new(name: impl Into<String>, data_type: DataType) -> Self {
        Self {
            name: name.into(),
            data_type,
            char_length: None,
            dimension: None,
            element_type: None,
            precision: None,
            scale: None,
        }
    }

    /// Creates a CHAR(n) column.
    pub fn char(name: impl Into<String>, length: u32) -> Self {
        Self {
            name: name.into(),
            data_type: DataType::Char,
            char_length: Some(length),
            dimension: None,
            element_type: None,
            precision: None,
            scale: None,
        }
    }

    /// Creates a VARCHAR(n) column. Pass None for unlimited length.
    pub fn varchar(name: impl Into<String>, length: Option<u32>) -> Self {
        Self {
            name: name.into(),
            data_type: DataType::Varchar,
            char_length: length,
            dimension: None,
            element_type: None,
            precision: None,
            scale: None,
        }
    }

    /// Creates a VECTOR(n) column with the specified dimension.
    pub fn vector(name: impl Into<String>, dimension: u32) -> Self {
        Self {
            name: name.into(),
            data_type: DataType::Vector,
            char_length: None,
            dimension: Some(dimension),
            element_type: None,
            precision: None,
            scale: None,
        }
    }

    /// Creates an ARRAY column with the specified element type.
    pub fn array(name: impl Into<String>, element_type: DataType) -> Self {
        Self {
            name: name.into(),
            data_type: DataType::Array,
            char_length: None,
            dimension: None,
            element_type: Some(Box::new(ColumnDef::new("", element_type))),
            precision: None,
            scale: None,
        }
    }

    /// Creates an ARRAY column with a full element ColumnDef (for nested arrays).
    pub fn array_of(name: impl Into<String>, element: ColumnDef) -> Self {
        Self {
            name: name.into(),
            data_type: DataType::Array,
            char_length: None,
            dimension: None,
            element_type: Some(Box::new(element)),
            precision: None,
            scale: None,
        }
    }

    /// Creates a DECIMAL(precision, scale) column.
    pub fn decimal(name: impl Into<String>, precision: u8, scale: i8) -> Self {
        Self {
            name: name.into(),
            data_type: DataType::Decimal,
            char_length: None,
            dimension: None,
            element_type: None,
            precision: Some(precision),
            scale: Some(scale),
        }
    }

    /// Returns the column name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Returns the data type.
    pub fn data_type(&self) -> DataType {
        self.data_type
    }

    /// Returns the character length for CHAR/VARCHAR columns.
    pub fn char_length(&self) -> Option<u32> {
        self.char_length
    }

    /// Returns the dimension for VECTOR columns.
    pub fn dimension(&self) -> Option<u32> {
        self.dimension
    }

    /// Returns the element type for ARRAY columns.
    pub fn element_type(&self) -> Option<&ColumnDef> {
        self.element_type.as_deref()
    }

    /// Returns the precision for DECIMAL columns.
    pub fn precision(&self) -> Option<u8> {
        self.precision
    }

    /// Returns the scale for DECIMAL columns.
    pub fn scale(&self) -> Option<i8> {
        self.scale
    }

    /// Sets the column name.
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = name.into();
        self
    }
}

/// Range type with bounds.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Range<T> {
    pub lower: Option<T>,
    pub upper: Option<T>,
    pub lower_inclusive: bool,
    pub upper_inclusive: bool,
    pub is_empty: bool,
}

impl<T> Range<T> {
    /// Creates an empty range.
    pub fn empty() -> Self {
        Self {
            lower: None,
            upper: None,
            lower_inclusive: false,
            upper_inclusive: false,
            is_empty: true,
        }
    }

    /// Creates a range with the specified bounds.
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

/// Range flag constants for binary encoding.
pub mod range_flags {
    pub const EMPTY: u8 = 0x01;
    pub const LOWER_INCLUSIVE: u8 = 0x02;
    pub const UPPER_INCLUSIVE: u8 = 0x04;
    pub const LOWER_INFINITE: u8 = 0x08;
    pub const UPPER_INFINITE: u8 = 0x10;
}

/// Zero-copy view into decimal data.
#[derive(Debug, Clone, Copy)]
pub struct DecimalView<'a> {
    data: &'a [u8],
}

impl<'a> DecimalView<'a> {
    /// Creates a view into decimal data.
    pub fn new(data: &'a [u8]) -> Self {
        Self { data }
    }

    /// Returns true if the decimal is negative.
    pub fn is_negative(&self) -> bool {
        self.data.first().map(|b| b & 0x80 != 0).unwrap_or(false)
    }

    /// Returns the scale (number of decimal places).
    pub fn scale(&self) -> i16 {
        if self.data.len() < 3 {
            return 0;
        }
        i16::from_le_bytes([self.data[1], self.data[2]])
    }

    /// Returns the unscaled digits.
    pub fn digits(&self) -> i128 {
        if self.data.len() < 19 {
            return 0;
        }
        let bytes: [u8; 16] = self.data[3..19].try_into().unwrap_or([0; 16]);
        i128::from_le_bytes(bytes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn column_def_simple() {
        let col = ColumnDef::new("id", DataType::Int8);
        assert_eq!(col.name(), "id");
        assert_eq!(col.data_type(), DataType::Int8);
        assert!(col.char_length().is_none());
    }

    #[test]
    fn column_def_varchar() {
        let col = ColumnDef::varchar("name", Some(255));
        assert_eq!(col.data_type(), DataType::Varchar);
        assert_eq!(col.char_length(), Some(255));
    }

    #[test]
    fn column_def_vector() {
        let col = ColumnDef::vector("embedding", 1536);
        assert_eq!(col.data_type(), DataType::Vector);
        assert_eq!(col.dimension(), Some(1536));
    }

    #[test]
    fn column_def_array() {
        let col = ColumnDef::array("tags", DataType::Int4);
        assert_eq!(col.data_type(), DataType::Array);
        let elem = col.element_type().unwrap();
        assert_eq!(elem.data_type(), DataType::Int4);
    }

    #[test]
    fn column_def_decimal() {
        let col = ColumnDef::decimal("price", 10, 2);
        assert_eq!(col.data_type(), DataType::Decimal);
        assert_eq!(col.precision(), Some(10));
        assert_eq!(col.scale(), Some(2));
    }

    #[test]
    fn range_empty() {
        let r: Range<i32> = Range::empty();
        assert!(r.is_empty);
    }

    #[test]
    fn range_bounds() {
        let r = Range::new(Some(1), Some(10), true, false);
        assert!(!r.is_empty);
        assert_eq!(r.lower, Some(1));
        assert_eq!(r.upper, Some(10));
        assert!(r.lower_inclusive);
        assert!(!r.upper_inclusive);
    }
}
