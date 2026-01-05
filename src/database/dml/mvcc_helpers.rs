//! # MVCC Helper Functions for DML Operations
//!
//! This module provides helper functions for integrating MVCC into DML operations.
//! It handles the wrapping/unwrapping of record data with RecordHeaders and
//! coordinates undo log writes for updates and deletes.
//!
//! ## Record Format (with MVCC)
//!
//! ```text
//! +------------------+------------------+
//! | RecordHeader     | User Data        |
//! | (17 bytes)       | (variable)       |
//! +------------------+------------------+
//! ```
//!
//! ## Insert Flow
//!
//! 1. Build user record from values
//! 2. Wrap with RecordHeader (LOCK_BIT set for transactions, unset for auto-commit)
//! 3. Insert into BTree
//! 4. On commit: clear LOCK_BIT, set commit_ts
//!
//! ## Update Flow
//!
//! 1. Read existing record, check can_write
//! 2. Write old version to undo page
//! 3. Build new record with prev_version pointer
//! 4. Update BTree
//! 5. On commit: clear LOCK_BIT, set commit_ts
//!
//! ## Delete Flow
//!
//! 1. Read existing record, check can_write
//! 2. Write old version to undo page
//! 3. Set DELETE_BIT in header
//! 4. Update BTree (tombstone remains)
//! 5. On commit: clear LOCK_BIT, set commit_ts

#![allow(dead_code)]

use crate::mvcc::{
    MvccTable, PageId, RecordHeader, TableId, TxnId, UndoRecord, WriteCheckResult, WriteEntry,
};
use crate::storage::Storage;
use eyre::Result;

pub fn wrap_record_for_insert(txn_id: TxnId, user_record: &[u8], in_transaction: bool) -> Vec<u8> {
    let mut header = RecordHeader::new(txn_id);
    header.set_locked(in_transaction);

    let mut result = vec![0u8; RecordHeader::SIZE + user_record.len()];
    header.write_to(&mut result[..RecordHeader::SIZE]);
    result[RecordHeader::SIZE..].copy_from_slice(user_record);
    result
}

pub fn wrap_record_for_update(
    txn_id: TxnId,
    user_record: &[u8],
    undo_page_id: PageId,
    undo_offset: u16,
    in_transaction: bool,
) -> Vec<u8> {
    let mut header = RecordHeader::new(txn_id);
    header.set_locked(in_transaction);
    header.prev_version = RecordHeader::encode_ptr(undo_page_id, undo_offset);

    let mut result = vec![0u8; RecordHeader::SIZE + user_record.len()];
    header.write_to(&mut result[..RecordHeader::SIZE]);
    result[RecordHeader::SIZE..].copy_from_slice(user_record);
    result
}

pub fn wrap_record_for_delete(txn_id: TxnId, existing_record: &[u8], in_transaction: bool) -> Result<Vec<u8>> {
    if existing_record.len() < RecordHeader::SIZE {
        eyre::bail!("existing value too small for delete");
    }

    let mut header = RecordHeader::from_bytes(existing_record);
    header.set_locked(in_transaction);
    header.set_deleted(true);
    header.txn_id = txn_id;

    let mut result = existing_record.to_vec();
    header.write_to(&mut result[..RecordHeader::SIZE]);
    Ok(result)
}

pub fn check_can_write(
    existing_value: &[u8],
    writer_txn_id: TxnId,
    writer_read_ts: TxnId,
) -> Result<WriteCheckResult> {
    if existing_value.len() < RecordHeader::SIZE {
        return Ok(WriteCheckResult::CanWrite);
    }
    MvccTable::can_write(existing_value, writer_txn_id, writer_read_ts)
}

pub fn get_user_data(raw_value: &[u8]) -> &[u8] {
    if raw_value.len() > RecordHeader::SIZE {
        &raw_value[RecordHeader::SIZE..]
    } else {
        raw_value
    }
}

pub fn has_mvcc_header(raw_value: &[u8]) -> bool {
    raw_value.len() >= RecordHeader::SIZE
}

pub fn create_undo_record(
    table_id: TableId,
    key: &[u8],
    raw_value: &[u8],
) -> Result<UndoRecord> {
    let mvcc_table = MvccTable::new(table_id);
    mvcc_table.create_undo_record(key, raw_value)
}

pub fn create_write_entry(
    table_id: TableId,
    key: Vec<u8>,
    page_id: PageId,
    offset: u16,
    undo_page_id: Option<PageId>,
    undo_offset: Option<u16>,
    is_insert: bool,
) -> WriteEntry {
    WriteEntry {
        table_id,
        key,
        page_id,
        offset,
        undo_page_id,
        undo_offset,
        is_insert,
    }
}

pub fn finalize_commit<S: Storage>(
    storage: &mut S,
    entry: &WriteEntry,
    commit_ts: TxnId,
) -> Result<()> {
    let page = storage.page_mut(entry.page_id as u32)?;
    let offset = entry.offset as usize;
    
    if offset + RecordHeader::SIZE <= page.len() {
        MvccTable::finalize_commit_value(&mut page[offset..], commit_ts)?;
    }
    
    Ok(())
}

pub fn finalize_abort<S: Storage>(
    storage: &mut S,
    entry: &WriteEntry,
    original_txn_id: TxnId,
) -> Result<()> {
    let page = storage.page_mut(entry.page_id as u32)?;
    let offset = entry.offset as usize;
    
    if offset + RecordHeader::SIZE <= page.len() {
        MvccTable::finalize_abort_value(&mut page[offset..], original_txn_id)?;
    }
    
    Ok(())
}
