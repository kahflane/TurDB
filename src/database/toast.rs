//! # TOAST (The Oversized-Attribute Storage Technique) Module
//!
//! This module implements TOAST for TurDB, handling storage and retrieval of
//! large values that exceed the inline storage threshold. Large text and blob
//! values are split into chunks and stored in a separate TOAST table.
//!
//! ## Purpose
//!
//! Without TOAST, large values would bloat B-tree pages, degrading performance
//! for scans and point lookups. TOAST solves this by:
//!
//! 1. Storing large values out-of-line in a dedicated TOAST table
//! 2. Replacing inline values with small fixed-size pointers
//! 3. Transparently reassembling values when queried
//!
//! ## Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────────────┐
//! │                        TOAST Storage Flow                               │
//! ├─────────────────────────────────────────────────────────────────────────┤
//! │                                                                         │
//! │   Main Table                          TOAST Table                       │
//! │   ──────────                          ───────────                       │
//! │   ┌──────────────────┐                ┌──────────────────┐              │
//! │   │ Row: id=1        │                │ chunk_key → data │              │
//! │   │ ├── col1: INT    │                │ ├── (id,col,0)   │              │
//! │   │ ├── col2: TEXT ──┼────pointer────►│ │    → chunk_0   │              │
//! │   │ │   (toast_ptr)  │                │ ├── (id,col,1)   │              │
//! │   │ └── col3: BOOL   │                │ │    → chunk_1   │              │
//! │   └──────────────────┘                │ └── ...          │              │
//! │                                       └──────────────────┘              │
//! │                                                                         │
//! │   ToastPointer (17 bytes):                                              │
//! │   ┌────────────────────────────────────────────────────────┐            │
//! │   │ chunk_id (8B) │ total_size (8B) │ marker (1B)          │            │
//! │   └────────────────────────────────────────────────────────┘            │
//! │                                                                         │
//! │   Chunk Key (12 bytes):                                                 │
//! │   ┌────────────────────────────────────────────────────────┐            │
//! │   │ chunk_id (8B) │ sequence_no (4B)                       │            │
//! │   └────────────────────────────────────────────────────────┘            │
//! │                                                                         │
//! └─────────────────────────────────────────────────────────────────────────┘
//! ```
//!
//! ## Key Data Structures
//!
//! - **ToastPointer**: Fixed-size reference stored inline in the main table
//! - **Chunk Key**: Composite key for chunk lookup in TOAST table
//! - **TOAST Table**: Separate B-tree storing value chunks
//!
//! ## Operations
//!
//! - `toast_value`: Split large value into chunks and store in TOAST table
//! - `detoast_value`: Reassemble value from chunks
//! - `detoast_rows`: Transparently detoast all ToastPointer values in result rows
//! - `delete_toast_chunks`: Remove chunks when row is deleted/updated
//!
//! ## Usage Patterns
//!
//! TOAST is transparent to the user. The database automatically:
//! 1. Toasts values during INSERT/UPDATE if they exceed threshold
//! 2. Detoasts values during SELECT before returning to user
//! 3. Deletes toast chunks during DELETE or UPDATE of toasted columns
//!
//! ## Performance Characteristics
//!
//! - Toasting: O(n/chunk_size) B-tree insertions
//! - Detoasting: O(n/chunk_size) B-tree lookups (sequential keys)
//! - Chunk size: 8KB (optimized for page alignment)
//! - Pointer overhead: 17 bytes per toasted value
//!
//! ## Thread Safety
//!
//! TOAST operations use the same locking as regular table operations:
//! - FileManager provides storage access
//! - DirtyTracker coordinates WAL writes
//! - All operations require mutable FileManager access
//!
//! ## Threshold
//!
//! Values are toasted when they exceed the inline threshold (typically 2KB).
//! This balances between:
//! - Small values: inline storage (no overhead)
//! - Large values: out-of-line (pointer overhead, extra I/O)
//!
//! ## Integration with WAL
//!
//! When WAL is enabled, TOAST chunk writes go through WalStoragePerTable
//! to ensure crash consistency. TOAST chunks are recovered along with
//! the main table during WAL replay.

use crate::database::row::Row;
use crate::storage::WalStoragePerTable;
use crate::types::OwnedValue;
use eyre::Result;

use super::Database;

impl Database {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn toast_value(
        &self,
        file_manager: &mut crate::storage::FileManager,
        schema_name: &str,
        table_name: &str,
        row_id: u64,
        column_index: u16,
        data: &[u8],
        wal_enabled: bool,
        hint: Option<u32>,
    ) -> Result<(Vec<u8>, Option<u32>)> {
        use crate::btree::BTree;
        use crate::storage::toast::{make_chunk_key, ToastPointer, TOAST_CHUNK_SIZE};

        let toast_table_name = crate::storage::toast::toast_table_name(table_name);
        let toast_storage_arc = file_manager.table_data_mut(schema_name, &toast_table_name)?;
        let mut toast_storage = toast_storage_arc.write();

        let (toast_table_id, root_page) = {
            let page0 = toast_storage.page(0)?;
            let header = crate::storage::TableFileHeader::from_bytes(page0)?;
            (header.table_id() as u32, header.root_page())
        };

        let pointer = ToastPointer::new(row_id, column_index, data.len() as u64);
        let chunk_id = pointer.chunk_id;

        let (new_hint, new_root) = if wal_enabled {
            let mut wal_storage =
                WalStoragePerTable::new(&mut toast_storage, &self.shared.dirty_tracker, toast_table_id);
            let mut btree = BTree::with_rightmost_hint(&mut wal_storage, root_page, hint)?;
            for (seq, chunk) in data.chunks(TOAST_CHUNK_SIZE).enumerate() {
                let chunk_key = make_chunk_key(chunk_id, seq as u32);
                btree.insert(&chunk_key, chunk)?;
            }
            (btree.rightmost_hint(), btree.root_page())
        } else {
            let mut btree = BTree::with_rightmost_hint(&mut *toast_storage, root_page, hint)?;
            for (seq, chunk) in data.chunks(TOAST_CHUNK_SIZE).enumerate() {
                let chunk_key = make_chunk_key(chunk_id, seq as u32);
                btree.insert(&chunk_key, chunk)?;
            }
            (btree.rightmost_hint(), btree.root_page())
        };

        if new_root != root_page {
            // Already holding lock, reuse toast_storage
            let page0 = toast_storage.page_mut(0)?;
            let header = crate::storage::TableFileHeader::from_bytes_mut(page0)?;
            header.set_root_page(new_root);
        }

        Ok((pointer.encode().to_vec(), new_hint))
    }

    pub(crate) fn detoast_value(
        &self,
        file_manager: &mut crate::storage::FileManager,
        schema_name: &str,
        table_name: &str,
        toast_pointer: &[u8],
    ) -> Result<Vec<u8>> {
        use crate::btree::BTree;
        use crate::storage::toast::{chunk_count, make_chunk_key, ToastPointer};

        let pointer = ToastPointer::decode(toast_pointer)?;
        let chunk_id = pointer.chunk_id;
        let total_size = pointer.total_size as usize;
        let num_chunks = chunk_count(total_size);

        let toast_table_name = crate::storage::toast::toast_table_name(table_name);
        let toast_storage_arc = file_manager.table_data_mut(schema_name, &toast_table_name)?;
        let mut toast_storage = toast_storage_arc.write();

        let root_page = {
            let page0 = toast_storage.page(0)?;
            crate::storage::TableFileHeader::from_bytes(page0)?.root_page()
        };


        let btree = BTree::new(&mut *toast_storage, root_page)?;

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

    pub(crate) fn detoast_rows(
        &self,
        file_manager: &mut crate::storage::FileManager,
        schema_name: &str,
        table_name: &str,
        rows: Vec<Row>,
    ) -> Result<Vec<Row>> {
        let mut result = Vec::with_capacity(rows.len());
        for row in rows {
            let mut new_values = Vec::with_capacity(row.values.len());
            for val in row.values {
                let new_val = match val {
                    OwnedValue::ToastPointer(ptr) => {
                        let data = self.detoast_value(file_manager, schema_name, table_name, &ptr)?;
                        if let Ok(s) = String::from_utf8(data.clone()) {
                            OwnedValue::Text(s)
                        } else {
                            OwnedValue::Blob(data)
                        }
                    }
                    other => other,
                };
                new_values.push(new_val);
            }
            result.push(Row::new(new_values));
        }
        Ok(result)
    }

    pub(crate) fn delete_toast_chunks(
        &self,
        file_manager: &mut crate::storage::FileManager,
        schema_name: &str,
        table_name: &str,
        row_id: u64,
        column_index: u16,
        total_size: u64,
    ) -> Result<()> {
        use crate::btree::BTree;
        use crate::storage::toast::{chunk_count, make_chunk_key, ToastPointer};

        let pointer = ToastPointer::new(row_id, column_index, total_size);
        let chunk_id = pointer.chunk_id;
        let num_chunks = chunk_count(total_size as usize);

        let toast_table_name = crate::storage::toast::toast_table_name(table_name);
        let toast_storage_arc = file_manager.table_data_mut(schema_name, &toast_table_name)?;
        let mut toast_storage = toast_storage_arc.write();

        let root_page = {
            let page0 = toast_storage.page(0)?;
            crate::storage::TableFileHeader::from_bytes(page0)?.root_page()
        };

        let mut btree = BTree::new(&mut *toast_storage, root_page)?;

        for seq in 0..num_chunks {
            let chunk_key = make_chunk_key(chunk_id, seq as u32);
            let _ = btree.delete(&chunk_key);
        }

        Ok(())
    }
}
