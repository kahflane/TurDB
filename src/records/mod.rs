//! # Record Serialization with O(1) Column Access
//!
//! This module provides zero-copy record access for TurDB's row storage. Unlike
//! SQLite's sequential record parsing (O(N) to find column N), TurDB records use
//! a header with null bitmap and offset table for O(1) column lookup.
//!
//! ## Record Binary Layout
//!
//! ```text
//! +------------------+------------------+------------------+------------------+
//! | Header Length    | Null Bitmap      | Offset Table     | Data Payload     |
//! | (u16)            | [u8; (N+7)/8]    | [u16; M]         | [u8; ...]        |
//! +------------------+------------------+------------------+------------------+
//! ```
//!
//! | Component | Type | Description |
//! |-----------|------|-------------|
//! | **Header Length** | `u16` | Total header size (allows skipping to data) |
//! | **Null Bitmap** | `[u8; (N+7)/8]` | 1 bit per column. `1` = NULL, `0` = has data |
//! | **Offset Table** | `[u16; M]` | End offsets for variable-length columns only |
//! | **Data Payload** | `[u8; ...]` | Concatenated fixed-width and variable-width values |
//!
//! ## Design Goals
//!
//! 1. **O(1) column access**: Direct offset calculation, no sequential parsing
//! 2. **Zero-copy reads**: All getters return references into the underlying buffer
//! 3. **Schema-dependent**: Types come from schema, not stored per-row
//! 4. **Compact headers**: u16 offsets (16KB pages), bitmap for NULLs
//!
//! ## Storage Classes
//!
//! Columns are categorized by storage class:
//!
//! | Class | Examples | Storage |
//! |-------|----------|---------|
//! | **Fixed** | int4, float8, uuid, timestamp | Direct bytes, no offset needed |
//! | **Variable** | text, blob, jsonb, vector | Offset in table -> data in payload |
//!
//! ## Module Structure
//!
//! - `types`: DataType enum and ColumnDef struct
//! - `schema`: Schema definition with pre-computed offsets
//! - `view`: RecordView for zero-copy reading
//! - `builder`: RecordBuilder for construction
//! - `jsonb`: JSONB binary format with O(log n) key lookup
//! - `array`: SQL array format with O(1) element access

pub mod array;
pub mod builder;
pub mod composite;
pub mod jsonb;
pub mod schema;
pub mod types;
pub mod view;

#[cfg(test)]
mod tests;

pub use array::{ArrayBuilder, ArrayView};
pub use builder::RecordBuilder;
pub use composite::{CompositeView, MAX_NESTING_DEPTH};
pub use jsonb::{JsonbBuilder, JsonbBuilderValue, JsonbValue, JsonbView};
pub use schema::Schema;
pub use types::{range_flags, ColumnDef, DataType, DecimalView, Range};
pub use view::RecordView;
