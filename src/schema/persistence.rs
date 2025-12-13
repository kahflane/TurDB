//! # Catalog Persistence
//!
//! This module implements serialization and deserialization of the catalog
//! to/from the turdb.meta file. It follows a compact binary format for
//! efficient storage and supports lazy loading of schema metadata.
//!
//! ## File Format
//!
//! The turdb.meta file has the following structure:
//!
//! ```text
//! +-------------------+ Offset 0
//! | File Header       |
//! | (128 bytes)       |
//! +-------------------+ Offset 128
//! | Catalog Data      |
//! | (variable length) |
//! +-------------------+
//! ```
//!
//! ### File Header Format
//!
//! ```text
//! Offset  Size  Description
//! 0       16    Magic: "TurDB Rust v1\x00\x00\x00"
//! 16      4     Version: 1 (u32 little-endian)
//! 20      4     Page size: 16384 (u32 little-endian)
//! 24      8     Schema count (u64 little-endian)
//! 32      8     Default schema ID (u64 little-endian)
//! 40      8     Next table ID (u64 little-endian)
//! 48      8     Next index ID (u64 little-endian)
//! 56      8     Flags (u64 little-endian)
//! 64      8     Catalog data offset (u64 little-endian)
//! 72      8     Catalog data length (u64 little-endian)
//! 80      48    Reserved for future use
//! ```
//!
//! ### Catalog Data Format
//!
//! The catalog data section contains a serialized representation of all
//! schemas and their tables. The format is:
//!
//! ```text
//! For each schema:
//!   - schema_id: u32 (4 bytes)
//!   - name_len: u16 (2 bytes)
//!   - name: UTF-8 bytes (name_len bytes)
//!   - table_count: u32 (4 bytes)
//!   - For each table:
//!       - table_id: u64 (8 bytes)
//!       - name_len: u16 (2 bytes)
//!       - name: UTF-8 bytes (name_len bytes)
//!       - column_count: u32 (4 bytes)
//!       - For each column:
//!           - name_len: u16 (2 bytes)
//!           - name: UTF-8 bytes (name_len bytes)
//!           - data_type: u8 (1 byte, matches DataType repr)
//!           - constraint_count: u16 (2 bytes)
//!           - For each constraint:
//!               - constraint_type: u8 (1 byte)
//!               - constraint_data: variable (depends on type)
//!           - has_default: u8 (1 byte, 0 or 1)
//!           - default_len: u16 (2 bytes, if has_default)
//!           - default_value: UTF-8 bytes (default_len bytes, if has_default)
//!       - has_primary_key: u8 (1 byte, 0 or 1)
//!       - primary_key_count: u16 (2 bytes, if has_primary_key)
//!       - For each primary key column:
//!           - name_len: u16 (2 bytes)
//!           - name: UTF-8 bytes (name_len bytes)
//!       - index_count: u32 (4 bytes)
//!       - For each index:
//!           - name_len: u16 (2 bytes)
//!           - name: UTF-8 bytes (name_len bytes)
//!           - column_count: u16 (2 bytes)
//!           - For each index column:
//!               - name_len: u16 (2 bytes)
//!               - name: UTF-8 bytes (name_len bytes)
//!           - is_unique: u8 (1 byte, 0 or 1)
//!           - index_type: u8 (1 byte, 0=BTree, 1=Hnsw)
//! ```
//!
//! ## Lazy Loading Strategy
//!
//! To support the 1MB memory budget requirement, the catalog uses a lazy
//! loading strategy:
//! 1. At startup, only read the file header and schema names
//! 2. Table definitions are loaded on first access to a schema
//! 3. This minimizes memory usage for databases with many tables
//!
//! ## Thread Safety
//!
//! Persistence operations are not thread-safe on their own. The Catalog
//! struct wraps these operations with appropriate locking (RwLock).
//!
//! ## Error Handling
//!
//! All I/O errors include rich context via eyre::WrapErr to aid debugging:
//! - File path being accessed
//! - Operation being performed (read/write)
//! - Position in file if applicable

use crate::records::types::DataType;
use crate::schema::{Catalog, ColumnDef, Constraint, IndexDef, IndexType, TableDef};
use eyre::{ensure, Result};

#[allow(dead_code)]
const MAGIC: &[u8; 16] = b"TurDB Rust v1\0\0\0";
#[allow(dead_code)]
const VERSION: u32 = 1;
#[allow(dead_code)]
const PAGE_SIZE: u32 = 16384;
#[allow(dead_code)]
const HEADER_SIZE: usize = 128;

pub struct CatalogPersistence;

impl CatalogPersistence {
    pub fn serialize(catalog: &Catalog) -> Result<Vec<u8>> {
        let mut buf = Vec::new();

        for (_name, schema) in catalog.schemas() {
            buf.extend(schema.id().to_le_bytes());

            let name_bytes = schema.name().as_bytes();
            ensure!(
                name_bytes.len() <= u16::MAX as usize,
                "schema name '{}' is too long (max {} bytes)",
                schema.name(),
                u16::MAX
            );
            buf.extend((name_bytes.len() as u16).to_le_bytes());
            buf.extend(name_bytes);

            let table_count = schema.tables().len() as u32;
            buf.extend(table_count.to_le_bytes());

            for (_table_name, table) in schema.tables() {
                Self::serialize_table(table, &mut buf)?;
            }
        }

        Ok(buf)
    }

    fn serialize_table(table: &TableDef, buf: &mut Vec<u8>) -> Result<()> {
        buf.extend(table.id().to_le_bytes());

        let name_bytes = table.name().as_bytes();
        ensure!(
            name_bytes.len() <= u16::MAX as usize,
            "table name '{}' is too long (max {} bytes)",
            table.name(),
            u16::MAX
        );
        buf.extend((name_bytes.len() as u16).to_le_bytes());
        buf.extend(name_bytes);

        let column_count = table.columns().len() as u32;
        buf.extend(column_count.to_le_bytes());

        for column in table.columns() {
            Self::serialize_column(column, buf)?;
        }

        if let Some(pk_columns) = table.primary_key() {
            buf.push(1);
            buf.extend((pk_columns.len() as u16).to_le_bytes());
            for pk_col in pk_columns {
                let name_bytes = pk_col.as_bytes();
                buf.extend((name_bytes.len() as u16).to_le_bytes());
                buf.extend(name_bytes);
            }
        } else {
            buf.push(0);
        }

        let index_count = table.indexes().len() as u32;
        buf.extend(index_count.to_le_bytes());

        for index in table.indexes() {
            Self::serialize_index(index, buf)?;
        }

        Ok(())
    }

    fn serialize_column(column: &ColumnDef, buf: &mut Vec<u8>) -> Result<()> {
        let name_bytes = column.name().as_bytes();
        ensure!(
            name_bytes.len() <= u16::MAX as usize,
            "column name '{}' is too long (max {} bytes)",
            column.name(),
            u16::MAX
        );
        buf.extend((name_bytes.len() as u16).to_le_bytes());
        buf.extend(name_bytes);

        buf.push(column.data_type() as u8);

        let constraint_count = column.constraints().len() as u16;
        buf.extend(constraint_count.to_le_bytes());

        for constraint in column.constraints() {
            Self::serialize_constraint(constraint, buf)?;
        }

        if let Some(default_val) = column.default_value() {
            buf.push(1);
            let default_bytes = default_val.as_bytes();
            buf.extend((default_bytes.len() as u16).to_le_bytes());
            buf.extend(default_bytes);
        } else {
            buf.push(0);
        }

        Ok(())
    }

    fn serialize_constraint(constraint: &Constraint, buf: &mut Vec<u8>) -> Result<()> {
        match constraint {
            Constraint::NotNull => {
                buf.push(0);
            }
            Constraint::PrimaryKey => {
                buf.push(1);
            }
            Constraint::Unique => {
                buf.push(2);
            }
            Constraint::ForeignKey { table, column } => {
                buf.push(3);
                let table_bytes = table.as_bytes();
                buf.extend((table_bytes.len() as u16).to_le_bytes());
                buf.extend(table_bytes);
                let column_bytes = column.as_bytes();
                buf.extend((column_bytes.len() as u16).to_le_bytes());
                buf.extend(column_bytes);
            }
            Constraint::Check(expr) => {
                buf.push(4);
                let expr_bytes = expr.as_bytes();
                buf.extend((expr_bytes.len() as u16).to_le_bytes());
                buf.extend(expr_bytes);
            }
        }
        Ok(())
    }

    fn serialize_index(index: &IndexDef, buf: &mut Vec<u8>) -> Result<()> {
        let name_bytes = index.name().as_bytes();
        ensure!(
            name_bytes.len() <= u16::MAX as usize,
            "index name '{}' is too long (max {} bytes)",
            index.name(),
            u16::MAX
        );
        buf.extend((name_bytes.len() as u16).to_le_bytes());
        buf.extend(name_bytes);

        let column_count = index.columns().len() as u16;
        buf.extend(column_count.to_le_bytes());

        for col_name in index.columns() {
            let col_bytes = col_name.as_bytes();
            buf.extend((col_bytes.len() as u16).to_le_bytes());
            buf.extend(col_bytes);
        }

        buf.push(if index.is_unique() { 1 } else { 0 });
        buf.push(match index.index_type() {
            IndexType::BTree => 0,
            IndexType::Hnsw => 1,
        });

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schema::Catalog;

    #[test]
    fn test_serialize_empty_catalog() {
        let catalog = Catalog::new();
        let result = CatalogPersistence::serialize(&catalog);
        assert!(result.is_ok());

        let bytes = result.unwrap();
        assert!(!bytes.is_empty());
    }

    #[test]
    fn test_serialize_catalog_with_schema() {
        let mut catalog = Catalog::new();
        catalog.create_schema("test_schema").unwrap();

        let result = CatalogPersistence::serialize(&catalog);
        assert!(result.is_ok());
    }

    #[test]
    fn test_serialize_catalog_with_table() {
        let mut catalog = Catalog::new();

        let columns = vec![
            ColumnDef::new("id", DataType::Int8),
            ColumnDef::new("name", DataType::Text),
        ];

        catalog.create_table("root", "users", columns).unwrap();

        let result = CatalogPersistence::serialize(&catalog);
        assert!(result.is_ok());

        let bytes = result.unwrap();
        assert!(!bytes.is_empty());
    }
}
