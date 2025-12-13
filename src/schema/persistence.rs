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
use crate::schema::{Catalog, ColumnDef, Constraint, IndexDef, IndexType, Schema, TableDef};
use eyre::{bail, ensure, Result};

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

    pub fn deserialize(bytes: &[u8], catalog: &mut Catalog) -> Result<()> {
        let mut pos = 0;

        while pos < bytes.len() {
            let (schema, new_pos) = Self::deserialize_schema(bytes, pos)?;
            pos = new_pos;

            if let Some(existing_schema) = catalog.get_schema_mut(schema.name()) {
                for (_table_name, table) in schema.tables() {
                    existing_schema.add_table(table.clone());
                }
            } else {
                bail!("schema '{}' not found in catalog during deserialization", schema.name());
            }
        }

        Ok(())
    }

    fn deserialize_schema(bytes: &[u8], mut pos: usize) -> Result<(Schema, usize)> {
        ensure!(pos + 4 <= bytes.len(), "unexpected end of data reading schema ID");
        let schema_id = u32::from_le_bytes([bytes[pos], bytes[pos + 1], bytes[pos + 2], bytes[pos + 3]]);
        pos += 4;

        ensure!(pos + 2 <= bytes.len(), "unexpected end of data reading schema name length");
        let name_len = u16::from_le_bytes([bytes[pos], bytes[pos + 1]]) as usize;
        pos += 2;

        ensure!(pos + name_len <= bytes.len(), "unexpected end of data reading schema name");
        let name = std::str::from_utf8(&bytes[pos..pos + name_len])
            .map_err(|e| eyre::eyre!("invalid UTF-8 in schema name: {}", e))?
            .to_string();
        pos += name_len;

        let mut schema = Schema::new(schema_id, name);

        ensure!(pos + 4 <= bytes.len(), "unexpected end of data reading table count");
        let table_count = u32::from_le_bytes([bytes[pos], bytes[pos + 1], bytes[pos + 2], bytes[pos + 3]]);
        pos += 4;

        for _ in 0..table_count {
            let (table, new_pos) = Self::deserialize_table(bytes, pos)?;
            pos = new_pos;
            schema.add_table(table);
        }

        Ok((schema, pos))
    }

    fn deserialize_table(bytes: &[u8], mut pos: usize) -> Result<(TableDef, usize)> {
        ensure!(pos + 8 <= bytes.len(), "unexpected end of data reading table ID");
        let table_id = u64::from_le_bytes([
            bytes[pos], bytes[pos + 1], bytes[pos + 2], bytes[pos + 3],
            bytes[pos + 4], bytes[pos + 5], bytes[pos + 6], bytes[pos + 7],
        ]);
        pos += 8;

        ensure!(pos + 2 <= bytes.len(), "unexpected end of data reading table name length");
        let name_len = u16::from_le_bytes([bytes[pos], bytes[pos + 1]]) as usize;
        pos += 2;

        ensure!(pos + name_len <= bytes.len(), "unexpected end of data reading table name");
        let name = std::str::from_utf8(&bytes[pos..pos + name_len])
            .map_err(|e| eyre::eyre!("invalid UTF-8 in table name: {}", e))?
            .to_string();
        pos += name_len;

        ensure!(pos + 4 <= bytes.len(), "unexpected end of data reading column count");
        let column_count = u32::from_le_bytes([bytes[pos], bytes[pos + 1], bytes[pos + 2], bytes[pos + 3]]);
        pos += 4;

        let mut columns = Vec::new();
        for _ in 0..column_count {
            let (column, new_pos) = Self::deserialize_column(bytes, pos)?;
            pos = new_pos;
            columns.push(column);
        }

        let mut table = TableDef::new(table_id, name, columns);

        ensure!(pos + 1 <= bytes.len(), "unexpected end of data reading has_primary_key");
        let has_primary_key = bytes[pos] != 0;
        pos += 1;

        if has_primary_key {
            ensure!(pos + 2 <= bytes.len(), "unexpected end of data reading primary key count");
            let pk_count = u16::from_le_bytes([bytes[pos], bytes[pos + 1]]) as usize;
            pos += 2;

            let mut pk_columns = Vec::new();
            for _ in 0..pk_count {
                ensure!(pos + 2 <= bytes.len(), "unexpected end of data reading PK column name length");
                let col_name_len = u16::from_le_bytes([bytes[pos], bytes[pos + 1]]) as usize;
                pos += 2;

                ensure!(pos + col_name_len <= bytes.len(), "unexpected end of data reading PK column name");
                let col_name = std::str::from_utf8(&bytes[pos..pos + col_name_len])
                    .map_err(|e| eyre::eyre!("invalid UTF-8 in PK column name: {}", e))?
                    .to_string();
                pos += col_name_len;

                pk_columns.push(col_name);
            }

            table = table.with_primary_key(pk_columns);
        }

        ensure!(pos + 4 <= bytes.len(), "unexpected end of data reading index count");
        let index_count = u32::from_le_bytes([bytes[pos], bytes[pos + 1], bytes[pos + 2], bytes[pos + 3]]);
        pos += 4;

        for _ in 0..index_count {
            let (index, new_pos) = Self::deserialize_index(bytes, pos)?;
            pos = new_pos;
            table = table.with_index(index);
        }

        Ok((table, pos))
    }

    fn deserialize_column(bytes: &[u8], mut pos: usize) -> Result<(ColumnDef, usize)> {
        ensure!(pos + 2 <= bytes.len(), "unexpected end of data reading column name length");
        let name_len = u16::from_le_bytes([bytes[pos], bytes[pos + 1]]) as usize;
        pos += 2;

        ensure!(pos + name_len <= bytes.len(), "unexpected end of data reading column name");
        let name = std::str::from_utf8(&bytes[pos..pos + name_len])
            .map_err(|e| eyre::eyre!("invalid UTF-8 in column name: {}", e))?
            .to_string();
        pos += name_len;

        ensure!(pos + 1 <= bytes.len(), "unexpected end of data reading data type");
        let data_type = Self::deserialize_data_type(bytes[pos])?;
        pos += 1;

        let mut column = ColumnDef::new(name, data_type);

        ensure!(pos + 2 <= bytes.len(), "unexpected end of data reading constraint count");
        let constraint_count = u16::from_le_bytes([bytes[pos], bytes[pos + 1]]) as usize;
        pos += 2;

        for _ in 0..constraint_count {
            let (constraint, new_pos) = Self::deserialize_constraint(bytes, pos)?;
            pos = new_pos;
            column = column.with_constraint(constraint);
        }

        ensure!(pos + 1 <= bytes.len(), "unexpected end of data reading has_default");
        let has_default = bytes[pos] != 0;
        pos += 1;

        if has_default {
            ensure!(pos + 2 <= bytes.len(), "unexpected end of data reading default value length");
            let default_len = u16::from_le_bytes([bytes[pos], bytes[pos + 1]]) as usize;
            pos += 2;

            ensure!(pos + default_len <= bytes.len(), "unexpected end of data reading default value");
            let default_val = std::str::from_utf8(&bytes[pos..pos + default_len])
                .map_err(|e| eyre::eyre!("invalid UTF-8 in default value: {}", e))?
                .to_string();
            pos += default_len;

            column = column.with_default(default_val);
        }

        Ok((column, pos))
    }

    fn deserialize_data_type(type_byte: u8) -> Result<DataType> {
        match type_byte {
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
            _ => bail!("unknown data type byte: {}", type_byte),
        }
    }

    fn deserialize_constraint(bytes: &[u8], mut pos: usize) -> Result<(Constraint, usize)> {
        ensure!(pos + 1 <= bytes.len(), "unexpected end of data reading constraint type");
        let constraint_type = bytes[pos];
        pos += 1;

        match constraint_type {
            0 => Ok((Constraint::NotNull, pos)),
            1 => Ok((Constraint::PrimaryKey, pos)),
            2 => Ok((Constraint::Unique, pos)),
            3 => {
                ensure!(pos + 2 <= bytes.len(), "unexpected end of data reading FK table name length");
                let table_len = u16::from_le_bytes([bytes[pos], bytes[pos + 1]]) as usize;
                pos += 2;

                ensure!(pos + table_len <= bytes.len(), "unexpected end of data reading FK table name");
                let table = std::str::from_utf8(&bytes[pos..pos + table_len])
                    .map_err(|e| eyre::eyre!("invalid UTF-8 in FK table name: {}", e))?
                    .to_string();
                pos += table_len;

                ensure!(pos + 2 <= bytes.len(), "unexpected end of data reading FK column name length");
                let column_len = u16::from_le_bytes([bytes[pos], bytes[pos + 1]]) as usize;
                pos += 2;

                ensure!(pos + column_len <= bytes.len(), "unexpected end of data reading FK column name");
                let column = std::str::from_utf8(&bytes[pos..pos + column_len])
                    .map_err(|e| eyre::eyre!("invalid UTF-8 in FK column name: {}", e))?
                    .to_string();
                pos += column_len;

                Ok((Constraint::ForeignKey { table, column }, pos))
            }
            4 => {
                ensure!(pos + 2 <= bytes.len(), "unexpected end of data reading CHECK expression length");
                let expr_len = u16::from_le_bytes([bytes[pos], bytes[pos + 1]]) as usize;
                pos += 2;

                ensure!(pos + expr_len <= bytes.len(), "unexpected end of data reading CHECK expression");
                let expr = std::str::from_utf8(&bytes[pos..pos + expr_len])
                    .map_err(|e| eyre::eyre!("invalid UTF-8 in CHECK expression: {}", e))?
                    .to_string();
                pos += expr_len;

                Ok((Constraint::Check(expr), pos))
            }
            _ => bail!("unknown constraint type: {}", constraint_type),
        }
    }

    fn deserialize_index(bytes: &[u8], mut pos: usize) -> Result<(IndexDef, usize)> {
        ensure!(pos + 2 <= bytes.len(), "unexpected end of data reading index name length");
        let name_len = u16::from_le_bytes([bytes[pos], bytes[pos + 1]]) as usize;
        pos += 2;

        ensure!(pos + name_len <= bytes.len(), "unexpected end of data reading index name");
        let name = std::str::from_utf8(&bytes[pos..pos + name_len])
            .map_err(|e| eyre::eyre!("invalid UTF-8 in index name: {}", e))?
            .to_string();
        pos += name_len;

        ensure!(pos + 2 <= bytes.len(), "unexpected end of data reading index column count");
        let column_count = u16::from_le_bytes([bytes[pos], bytes[pos + 1]]) as usize;
        pos += 2;

        let mut columns = Vec::new();
        for _ in 0..column_count {
            ensure!(pos + 2 <= bytes.len(), "unexpected end of data reading index column name length");
            let col_name_len = u16::from_le_bytes([bytes[pos], bytes[pos + 1]]) as usize;
            pos += 2;

            ensure!(pos + col_name_len <= bytes.len(), "unexpected end of data reading index column name");
            let col_name = std::str::from_utf8(&bytes[pos..pos + col_name_len])
                .map_err(|e| eyre::eyre!("invalid UTF-8 in index column name: {}", e))?
                .to_string();
            pos += col_name_len;

            columns.push(col_name);
        }

        ensure!(pos + 1 <= bytes.len(), "unexpected end of data reading is_unique");
        let is_unique = bytes[pos] != 0;
        pos += 1;

        ensure!(pos + 1 <= bytes.len(), "unexpected end of data reading index type");
        let index_type = match bytes[pos] {
            0 => IndexType::BTree,
            1 => IndexType::Hnsw,
            _ => bail!("unknown index type: {}", bytes[pos]),
        };
        pos += 1;

        Ok((IndexDef::new(name, columns, is_unique, index_type), pos))
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

    #[test]
    fn test_roundtrip_empty_catalog() {
        let original = Catalog::new();
        let bytes = CatalogPersistence::serialize(&original).unwrap();

        let mut deserialized = Catalog::new();
        CatalogPersistence::deserialize(&bytes, &mut deserialized).unwrap();

        assert_eq!(deserialized.default_schema(), "root");
        assert!(deserialized.schema_exists("root"));
        assert!(deserialized.schema_exists("turdb_catalog"));
    }

    #[test]
    fn test_roundtrip_catalog_with_table() {
        let mut original = Catalog::new();

        let columns = vec![
            ColumnDef::new("id", DataType::Int8)
                .with_constraint(Constraint::NotNull)
                .with_constraint(Constraint::PrimaryKey),
            ColumnDef::new("name", DataType::Text)
                .with_constraint(Constraint::NotNull),
            ColumnDef::new("email", DataType::Text)
                .with_constraint(Constraint::Unique),
        ];

        original.create_table("root", "users", columns).unwrap();

        let bytes = CatalogPersistence::serialize(&original).unwrap();

        let mut deserialized = Catalog::new();
        CatalogPersistence::deserialize(&bytes, &mut deserialized).unwrap();

        let table = deserialized.get_table("root", "users").unwrap();
        assert_eq!(table.name(), "users");
        assert_eq!(table.columns().len(), 3);
        assert_eq!(table.columns()[0].name(), "id");
        assert_eq!(table.columns()[1].name(), "name");
        assert_eq!(table.columns()[2].name(), "email");
    }

    #[test]
    fn test_roundtrip_catalog_with_indexes() {
        let mut original = Catalog::new();

        let columns = vec![
            ColumnDef::new("id", DataType::Int8),
            ColumnDef::new("email", DataType::Text),
        ];

        let idx = IndexDef::new("idx_users_email", vec!["email"], true, IndexType::BTree);

        original.create_table("root", "users", columns).unwrap();

        let schema = original.get_schema_mut("root").unwrap();
        let mut table = schema.remove_table("users").unwrap();
        table = table.with_index(idx);
        schema.add_table(table);

        let bytes = CatalogPersistence::serialize(&original).unwrap();

        let mut deserialized = Catalog::new();
        CatalogPersistence::deserialize(&bytes, &mut deserialized).unwrap();

        let table = deserialized.get_table("root", "users").unwrap();
        assert_eq!(table.indexes().len(), 1);
        assert_eq!(table.indexes()[0].name(), "idx_users_email");
        assert!(table.indexes()[0].is_unique());
    }
}
