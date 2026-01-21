//! # Unified Type System for TurDB
//!
//! This module provides the canonical type system for TurDB, consolidating
//! type definitions previously scattered across multiple modules.
//!
//! ## Module Structure
//!
//! - `data_type`: Canonical `DataType` enum and `TypeAffinity`
//! - `value`: Runtime `Value<'a>` with zero-copy support
//! - `owned_value`: Heap-owned `OwnedValue` for persistence
//! - `column`: `ColumnDef` with type metadata
//!
//! ## Key Types
//!
//! | Type | Purpose |
//! |------|---------|
//! | `DataType` | Storage-level type discriminant |
//! | `TypeAffinity` | SQLite-compatible type affinity |
//! | `Value<'a>` | Runtime value (zero-copy from pages) |
//! | `OwnedValue` | Heap-owned value (for INSERT/UPDATE) |
//! | `ColumnDef` | Column definition with metadata |
//!
//! ## Usage
//!
//! ```ignore
//! use turdb::types::{DataType, Value, OwnedValue, ColumnDef};
//!
//! // Define a column
//! let col = ColumnDef::varchar("name", Some(255));
//!
//! // Work with values
//! let val = Value::Int(42);
//! let owned: OwnedValue = (&val).into();
//! ```

mod column;
mod data_type;
mod owned_value;
mod value;

pub use column::{range_flags, ColumnDef, DecimalView, Range};
pub use data_type::{DataType, TypeAffinity};
pub use owned_value::{
    create_column_map, create_record_schema, owned_values_to_values, ArithmeticOp, OwnedValue,
};
pub use value::Value;
