//! # TOAST (The Oversized-Attribute Storage Technique) Implementation
//!
//! This module implements PostgreSQL-style TOAST for handling large values that
//! exceed the inline storage threshold. Large TEXT, BLOB, and JSONB values are
//! automatically chunked and stored in a separate TOAST table.
//!
//! ## Architecture
//!
//! Each table with potentially large columns (TEXT, BLOB, JSONB, VECTOR) has an
//! associated TOAST table named `<table_name>_toast`. When a column value exceeds
//! `TOAST_THRESHOLD` bytes, it is:
//!
//! 1. Split into chunks of `TOAST_CHUNK_SIZE` bytes
//! 2. Each chunk stored as a row in the TOAST table
//! 3. The main table stores a 17-byte TOAST pointer instead
//!
//! ## TOAST Table Schema
//!
//! ```sql
//! CREATE TABLE <table>_toast (
//!     chunk_id  BIGINT,   -- (row_id << 16) | column_index
//!     chunk_seq INT,      -- 0-based sequence number
//!     chunk_data BLOB,    -- actual data (up to TOAST_CHUNK_SIZE bytes)
//!     PRIMARY KEY (chunk_id, chunk_seq)
//! );
//! ```
//!
//! ## TOAST Pointer Format
//!
//! When a value is toasted, the main table stores:
//!
//! ```text
//! +--------+-------------+----------+
//! | Marker | Total Size  | Chunk ID |
//! | 1 byte | 8 bytes     | 8 bytes  |
//! | 0xFE   | u64 LE      | u64 LE   |
//! +--------+-------------+----------+
//! ```
//!
//! ## Chunk ID Encoding
//!
//! chunk_id = (row_id << 16) | column_index
//!
//! This allows up to 65536 columns per table (more than enough) and ensures
//! unique chunk_id for each (row, column) pair.
//!
//! ## Performance
//!
//! - Write: O(n/CHUNK_SIZE) chunks written sequentially
//! - Read: O(n/CHUNK_SIZE) chunks read and concatenated
//! - The TOAST table uses a B-tree with (chunk_id, chunk_seq) as key for
//!   efficient range scans during reconstruction
//!
//! ## Threshold Selection
//!
//! TOAST_THRESHOLD = 2000 bytes chosen because:
//! - Keeps most rows under 8KB (half a page, good for splits)
//! - Typical text columns rarely exceed this
//! - Matches PostgreSQL's default behavior
//!
//! ## Thread Safety
//!
//! TOAST operations use the same locking as regular table operations.
//! The FileManager handles concurrent access to TOAST files.

use eyre::{ensure, Result};

pub const TOAST_MARKER: u8 = 0xFE;
pub const TOAST_POINTER_SIZE: usize = 17;
pub const TOAST_THRESHOLD: usize = 1000;
pub const TOAST_CHUNK_SIZE: usize = 4000;

pub trait Detoaster {
    fn detoast(&self, toast_pointer: &[u8]) -> Result<Vec<u8>>;
}

#[derive(Debug, Clone, Copy)]
pub struct ToastPointer {
    pub total_size: u64,
    pub chunk_id: u64,
}

impl ToastPointer {
    pub fn new(row_id: u64, column_index: u16, total_size: u64) -> Self {
        let chunk_id = (row_id << 16) | (column_index as u64);
        Self {
            total_size,
            chunk_id,
        }
    }

    pub fn encode(&self) -> [u8; TOAST_POINTER_SIZE] {
        let mut buf = [0u8; TOAST_POINTER_SIZE];
        buf[0] = TOAST_MARKER;
        buf[1..9].copy_from_slice(&self.total_size.to_le_bytes());
        buf[9..17].copy_from_slice(&self.chunk_id.to_le_bytes());
        buf
    }

    pub fn decode(data: &[u8]) -> Result<Self> {
        ensure!(
            data.len() >= TOAST_POINTER_SIZE,
            "toast pointer too short: {} < {}",
            data.len(),
            TOAST_POINTER_SIZE
        );
        ensure!(
            data[0] == TOAST_MARKER,
            "invalid toast marker: {:02x}",
            data[0]
        );

        let total_size = u64::from_le_bytes(data[1..9].try_into().unwrap());
        let chunk_id = u64::from_le_bytes(data[9..17].try_into().unwrap());

        Ok(Self {
            total_size,
            chunk_id,
        })
    }

    pub fn row_id(&self) -> u64 {
        self.chunk_id >> 16
    }

    pub fn column_index(&self) -> u16 {
        (self.chunk_id & 0xFFFF) as u16
    }
}

pub fn is_toast_pointer(data: &[u8]) -> bool {
    data.len() == TOAST_POINTER_SIZE && data[0] == TOAST_MARKER
}

pub fn needs_toast(data: &[u8]) -> bool {
    data.len() > TOAST_THRESHOLD
}

pub fn toast_table_name(table_name: &str) -> String {
    format!("{}_toast", table_name)
}

pub fn chunk_count(total_size: usize) -> usize {
    total_size.div_ceil(TOAST_CHUNK_SIZE)
}

pub fn make_chunk_key(chunk_id: u64, chunk_seq: u32) -> [u8; 12] {
    let mut key = [0u8; 12];
    key[0..8].copy_from_slice(&chunk_id.to_be_bytes());
    key[8..12].copy_from_slice(&chunk_seq.to_be_bytes());
    key
}

pub fn parse_chunk_key(key: &[u8]) -> Result<(u64, u32)> {
    ensure!(key.len() >= 12, "chunk key too short");
    let chunk_id = u64::from_be_bytes(key[0..8].try_into().unwrap());
    let chunk_seq = u32::from_be_bytes(key[8..12].try_into().unwrap());
    Ok((chunk_id, chunk_seq))
}

use parking_lot::RwLock;
use std::sync::Arc;

pub struct TableDetoaster {
    file_manager: Arc<RwLock<Option<crate::storage::FileManager>>>,
    schema_name: String,
    table_name: String,
}

impl TableDetoaster {
    pub fn new(
        file_manager: Arc<RwLock<Option<crate::storage::FileManager>>>,
        schema_name: String,
        table_name: String,
    ) -> Self {
        Self {
            file_manager,
            schema_name,
            table_name,
        }
    }
}

impl Detoaster for TableDetoaster {
    fn detoast(&self, toast_pointer: &[u8]) -> Result<Vec<u8>> {
        use crate::btree::BTree;

        let pointer = ToastPointer::decode(toast_pointer)?;
        let chunk_id = pointer.chunk_id;
        let total_size = pointer.total_size as usize;
        let num_chunks = chunk_count(total_size);

        let toast_table_name = toast_table_name(&self.table_name);

        let mut file_manager_guard = self.file_manager.write();
        let file_manager = file_manager_guard
            .as_mut()
            .ok_or_else(|| eyre::eyre!("file manager not available"))?;

        let toast_storage = file_manager.table_data_mut(&self.schema_name, &toast_table_name)?;

        let btree = BTree::new(toast_storage, 1)?;

        let mut result = Vec::with_capacity(total_size);

        for seq in 0..num_chunks {
            let chunk_key = make_chunk_key(chunk_id, seq as u32);
            let handle = btree
                .search(&chunk_key)?
                .ok_or_else(|| eyre::eyre!("TOAST chunk not found: {:?}", chunk_key))?;
            let chunk_data = btree.get_value(&handle)?;
            result.extend_from_slice(chunk_data);
        }

        result.truncate(total_size);
        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_toast_pointer_roundtrip() {
        let ptr = ToastPointer::new(12345, 7, 50000);
        let encoded = ptr.encode();
        let decoded = ToastPointer::decode(&encoded).unwrap();

        assert_eq!(decoded.total_size, 50000);
        assert_eq!(decoded.row_id(), 12345);
        assert_eq!(decoded.column_index(), 7);
    }

    #[test]
    fn test_is_toast_pointer() {
        let ptr = ToastPointer::new(1, 0, 100);
        let encoded = ptr.encode();
        assert!(is_toast_pointer(&encoded));
        assert!(!is_toast_pointer(&[0u8; 10]));
        assert!(!is_toast_pointer(&[0xFF; 17]));
    }

    #[test]
    fn test_chunk_count() {
        assert_eq!(chunk_count(100), 1);
        assert_eq!(chunk_count(4000), 1);
        assert_eq!(chunk_count(4001), 2);
        assert_eq!(chunk_count(8000), 2);
        assert_eq!(chunk_count(8001), 3);
    }

    #[test]
    fn test_chunk_key_roundtrip() {
        let key = make_chunk_key(0x123456789ABCDEF0, 42);
        let (chunk_id, chunk_seq) = parse_chunk_key(&key).unwrap();
        assert_eq!(chunk_id, 0x123456789ABCDEF0);
        assert_eq!(chunk_seq, 42);
    }
}
