//! # MVCC Table Layer
//!
//! This module provides the `MvccTable` struct that wraps a B-tree with MVCC
//! semantics. The B-tree remains a general-purpose key-value store, while
//! MvccTable handles:
//!
//! - Prepending RecordHeader to values on insert/update
//! - Parsing RecordHeader and checking visibility on read
//! - Tracking writes in the transaction's write set
//! - Storing old versions to undo pages before updates
//!
//! ## Value Format
//!
//! Every value stored in the B-tree has this format:
//!
//! ```text
//! +------------------+------------------+
//! | RecordHeader     | User Value       |
//! | (17 bytes)       | (variable)       |
//! +------------------+------------------+
//! ```
//!
//! This means the B-tree stores `RecordHeader || user_value` as the value.
//! MvccTable transparently handles this prefix on reads and writes.
//!
//! ## Read Path
//!
//! ```text
//! MvccTable::get(key, read_ts)
//!    │
//!    ▼
//! BTree::get(key) → Option<&[u8]>
//!    │
//!    ▼
//! Parse RecordHeader from value[0..17]
//!    │
//!    ▼
//! Check visibility with read_ts
//!    │
//!    ├─ Visible → Return &value[17..]
//!    │
//!    ├─ Invisible → Follow prev_version chain
//!    │
//!    └─ Deleted → Return None
//! ```
//!
//! ## Write Path (Insert)
//!
//! ```text
//! MvccTable::insert(key, value, txn)
//!    │
//!    ▼
//! Create RecordHeader with txn.id(), LOCK_BIT set
//!    │
//!    ▼
//! BTree::insert(key, header || value)
//!    │
//!    ▼
//! Add to txn.write_entries()
//! ```
//!
//! ## Write Path (Update)
//!
//! ```text
//! MvccTable::update(key, new_value, txn)
//!    │
//!    ▼
//! Read current value from BTree
//!    │
//!    ▼
//! Check can_write() for conflicts
//!    │
//!    ▼
//! Copy old version to undo page
//!    │
//!    ▼
//! Create new RecordHeader with prev_version pointer
//!    │
//!    ▼
//! BTree::insert(key, header || new_value) (overwrites)
//!    │
//!    ▼
//! Add to txn.write_entries()
//! ```
//!
//! ## Delete Path
//!
//! ```text
//! MvccTable::delete(key, txn)
//!    │
//!    ▼
//! Read current header, set DELETE_BIT and LOCK_BIT
//!    │
//!    ▼
//! Update in place or write tombstone
//! ```
//!
//! ## Thread Safety
//!
//! MvccTable is not thread-safe. It should be accessed through a single
//! transaction at a time. Concurrent transactions should use separate
//! MvccTable instances or external synchronization.

use super::record_header::RecordHeader;
use super::transaction::{PageId, TableId, WriteEntry};
use super::undo_page::UndoRecord;
use super::version::{VersionChainReader, VisibilityResult, WriteCheckResult};
use super::TxnId;
use eyre::{bail, Result};

pub struct MvccValue<'a> {
    pub header: RecordHeader,
    pub data: &'a [u8],
}

pub struct MvccTable {
    table_id: TableId,
}

impl MvccTable {
    pub fn new(table_id: TableId) -> Self {
        Self { table_id }
    }

    pub fn table_id(&self) -> TableId {
        self.table_id
    }

    pub fn parse_value(raw_value: &[u8]) -> Result<MvccValue<'_>> {
        if raw_value.len() < RecordHeader::SIZE {
            bail!(
                "value too small for RecordHeader: {} < {}",
                raw_value.len(),
                RecordHeader::SIZE
            );
        }

        let header = RecordHeader::from_bytes(raw_value);
        let data = &raw_value[RecordHeader::SIZE..];

        Ok(MvccValue { header, data })
    }

    pub fn is_visible(raw_value: &[u8], read_ts: TxnId) -> Result<VisibilityResult> {
        let mvcc_value = Self::parse_value(raw_value)?;
        Ok(mvcc_value.header.is_visible_to(read_ts))
    }

    pub fn get_visible(raw_value: &[u8], read_ts: TxnId) -> Result<Option<&[u8]>> {
        let reader = VersionChainReader::new(raw_value, read_ts);

        match reader.visibility() {
            VisibilityResult::Visible => Ok(Some(reader.data())),
            VisibilityResult::Deleted => Ok(None),
            VisibilityResult::Invisible => Ok(None),
        }
    }

    pub fn get_visible_with_undo<'a, F, E>(
        raw_value: &'a [u8],
        read_ts: TxnId,
        load_undo: F,
    ) -> Result<Option<&'a [u8]>, E>
    where
        F: FnMut(u64, u16) -> Result<Option<(RecordHeader, &'a [u8])>, E>,
        E: From<eyre::Report>,
    {
        let reader = VersionChainReader::new(raw_value, read_ts);

        match reader.find_visible_version(load_undo)? {
            Some(visible) => Ok(Some(visible.data)),
            None => Ok(None),
        }
    }

    pub fn can_write(
        raw_value: &[u8],
        writer_txn_id: TxnId,
        writer_read_ts: TxnId,
    ) -> Result<WriteCheckResult> {
        let mvcc_value = Self::parse_value(raw_value)?;
        Ok(mvcc_value.header.can_write(writer_txn_id, writer_read_ts))
    }

    pub fn prepare_insert_value(txn_id: TxnId, user_value: &[u8]) -> Vec<u8> {
        let mut header = RecordHeader::new(txn_id);
        header.set_locked(true);

        let mut result = vec![0u8; RecordHeader::SIZE + user_value.len()];
        header.write_to(&mut result[..RecordHeader::SIZE]);
        result[RecordHeader::SIZE..].copy_from_slice(user_value);
        result
    }

    pub fn prepare_update_value(
        txn_id: TxnId,
        user_value: &[u8],
        undo_page_id: PageId,
        undo_offset: u16,
    ) -> Vec<u8> {
        let mut header = RecordHeader::new(txn_id);
        header.set_locked(true);
        header.prev_version = RecordHeader::encode_ptr(undo_page_id, undo_offset);

        let mut result = vec![0u8; RecordHeader::SIZE + user_value.len()];
        header.write_to(&mut result[..RecordHeader::SIZE]);
        result[RecordHeader::SIZE..].copy_from_slice(user_value);
        result
    }

    pub fn prepare_delete_value(txn_id: TxnId, existing_value: &[u8]) -> Result<Vec<u8>> {
        if existing_value.len() < RecordHeader::SIZE {
            bail!("existing value too small for delete");
        }

        let mut header = RecordHeader::from_bytes(existing_value);
        header.set_locked(true);
        header.set_deleted(true);
        header.txn_id = txn_id;

        let mut result = existing_value.to_vec();
        header.write_to(&mut result[..RecordHeader::SIZE]);
        Ok(result)
    }

    pub fn create_undo_record(&self, key: &[u8], raw_value: &[u8]) -> Result<UndoRecord> {
        let mvcc_value = Self::parse_value(raw_value)?;
        Ok(UndoRecord::new(
            self.table_id,
            mvcc_value.header,
            key.to_vec(),
            mvcc_value.data.to_vec(),
        ))
    }

    pub fn create_write_entry(
        &self,
        key: &[u8],
        page_id: PageId,
        offset: u16,
        undo_page_id: Option<PageId>,
        undo_offset: Option<u16>,
        is_insert: bool,
    ) -> WriteEntry {
        WriteEntry {
            table_id: self.table_id,
            key: key.to_vec(),
            page_id,
            offset,
            undo_page_id,
            undo_offset,
            is_insert,
        }
    }
}
