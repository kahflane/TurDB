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

    pub fn get_visible(
        raw_value: &[u8],
        read_ts: TxnId,
    ) -> Result<Option<&[u8]>> {
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

    pub fn can_write(raw_value: &[u8], writer_txn_id: TxnId, writer_read_ts: TxnId) -> Result<WriteCheckResult> {
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

    pub fn finalize_commit_value(raw_value: &mut [u8], commit_ts: TxnId) -> Result<()> {
        if raw_value.len() < RecordHeader::SIZE {
            bail!("value too small for commit finalization");
        }

        let mut header = RecordHeader::from_bytes(raw_value);
        header.set_locked(false);
        header.txn_id = commit_ts;
        header.write_to(&mut raw_value[..RecordHeader::SIZE]);
        Ok(())
    }

    pub fn finalize_abort_value(raw_value: &mut [u8], original_txn_id: TxnId) -> Result<()> {
        if raw_value.len() < RecordHeader::SIZE {
            bail!("value too small for abort finalization");
        }

        let mut header = RecordHeader::from_bytes(raw_value);
        header.set_locked(false);
        header.set_deleted(false);
        header.txn_id = original_txn_id;
        header.write_to(&mut raw_value[..RecordHeader::SIZE]);
        Ok(())
    }

    pub fn create_undo_record(
        &self,
        key: &[u8],
        raw_value: &[u8],
    ) -> Result<UndoRecord> {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_value_extracts_header_and_data() {
        let mut raw = vec![0u8; RecordHeader::SIZE + 10];
        let header = RecordHeader::new(42);
        header.write_to(&mut raw[..RecordHeader::SIZE]);
        raw[RecordHeader::SIZE..].copy_from_slice(b"0123456789");

        let parsed = MvccTable::parse_value(&raw).unwrap();
        assert_eq!(parsed.header.txn_id, 42);
        assert_eq!(parsed.data, b"0123456789");
    }

    #[test]
    fn parse_value_fails_if_too_small() {
        let raw = vec![0u8; 10];
        assert!(MvccTable::parse_value(&raw).is_err());
    }

    #[test]
    fn is_visible_checks_header() {
        let value = MvccTable::prepare_insert_value(50, b"data");
        let mut unlocked = value.clone();
        MvccTable::finalize_commit_value(&mut unlocked, 50).unwrap();

        assert_eq!(
            MvccTable::is_visible(&unlocked, 100).unwrap(),
            VisibilityResult::Visible
        );
        assert_eq!(
            MvccTable::is_visible(&unlocked, 25).unwrap(),
            VisibilityResult::Invisible
        );
    }

    #[test]
    fn get_visible_returns_data_when_visible() {
        let value = MvccTable::prepare_insert_value(50, b"hello");
        let mut committed = value.clone();
        MvccTable::finalize_commit_value(&mut committed, 50).unwrap();

        let result = MvccTable::get_visible(&committed, 100).unwrap();
        assert_eq!(result, Some(b"hello".as_slice()));
    }

    #[test]
    fn get_visible_returns_none_when_invisible() {
        let value = MvccTable::prepare_insert_value(150, b"hello");
        let mut committed = value.clone();
        MvccTable::finalize_commit_value(&mut committed, 150).unwrap();

        let result = MvccTable::get_visible(&committed, 100).unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn get_visible_returns_none_when_locked() {
        let value = MvccTable::prepare_insert_value(50, b"hello");
        let result = MvccTable::get_visible(&value, 100).unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn prepare_insert_value_creates_locked_header() {
        let value = MvccTable::prepare_insert_value(42, b"test");

        assert_eq!(value.len(), RecordHeader::SIZE + 4);

        let parsed = MvccTable::parse_value(&value).unwrap();
        assert_eq!(parsed.header.txn_id, 42);
        assert!(parsed.header.is_locked());
        assert!(!parsed.header.is_deleted());
        assert_eq!(parsed.data, b"test");
    }

    #[test]
    fn prepare_update_value_sets_prev_version() {
        let value = MvccTable::prepare_update_value(100, b"new", 5, 200);

        let parsed = MvccTable::parse_value(&value).unwrap();
        assert_eq!(parsed.header.txn_id, 100);
        assert!(parsed.header.is_locked());
        assert!(parsed.header.has_prev_version());

        let (page_id, offset) = RecordHeader::decode_ptr(parsed.header.prev_version);
        assert_eq!(page_id, 5);
        assert_eq!(offset, 200);
    }

    #[test]
    fn prepare_delete_value_sets_delete_bit() {
        let original = MvccTable::prepare_insert_value(50, b"data");
        let deleted = MvccTable::prepare_delete_value(100, &original).unwrap();

        let parsed = MvccTable::parse_value(&deleted).unwrap();
        assert_eq!(parsed.header.txn_id, 100);
        assert!(parsed.header.is_locked());
        assert!(parsed.header.is_deleted());
    }

    #[test]
    fn finalize_commit_clears_lock_updates_ts() {
        let mut value = MvccTable::prepare_insert_value(50, b"data");
        MvccTable::finalize_commit_value(&mut value, 100).unwrap();

        let parsed = MvccTable::parse_value(&value).unwrap();
        assert_eq!(parsed.header.txn_id, 100);
        assert!(!parsed.header.is_locked());
    }

    #[test]
    fn finalize_abort_clears_lock_and_delete() {
        let original = MvccTable::prepare_insert_value(50, b"data");
        let mut deleted = MvccTable::prepare_delete_value(100, &original).unwrap();
        MvccTable::finalize_abort_value(&mut deleted, 50).unwrap();

        let parsed = MvccTable::parse_value(&deleted).unwrap();
        assert_eq!(parsed.header.txn_id, 50);
        assert!(!parsed.header.is_locked());
        assert!(!parsed.header.is_deleted());
    }

    #[test]
    fn can_write_checks_conflicts() {
        let mut value = MvccTable::prepare_insert_value(50, b"data");
        MvccTable::finalize_commit_value(&mut value, 50).unwrap();

        assert_eq!(
            MvccTable::can_write(&value, 100, 60).unwrap(),
            WriteCheckResult::CanWrite
        );
        assert_eq!(
            MvccTable::can_write(&value, 100, 40).unwrap(),
            WriteCheckResult::ConcurrentModification
        );
    }

    #[test]
    fn create_undo_record_from_value() {
        let table = MvccTable::new(123);
        let value = MvccTable::prepare_insert_value(50, b"data");
        let mut committed = value.clone();
        MvccTable::finalize_commit_value(&mut committed, 50).unwrap();

        let undo = table.create_undo_record(b"key", &committed).unwrap();
        assert_eq!(undo.table_id, 123);
        assert_eq!(undo.record_header.txn_id, 50);
        assert_eq!(undo.key, b"key");
        assert_eq!(undo.value, b"data");
    }

    #[test]
    fn create_write_entry_for_insert() {
        let table = MvccTable::new(1);
        let entry = table.create_write_entry(b"key", 100, 50, None, None, true);

        assert_eq!(entry.table_id, 1);
        assert_eq!(entry.key, b"key");
        assert_eq!(entry.page_id, 100);
        assert_eq!(entry.offset, 50);
        assert!(entry.is_insert);
        assert!(entry.undo_page_id.is_none());
    }

    #[test]
    fn create_write_entry_for_update() {
        let table = MvccTable::new(1);
        let entry = table.create_write_entry(b"key", 100, 50, Some(200), Some(60), false);

        assert!(!entry.is_insert);
        assert_eq!(entry.undo_page_id, Some(200));
        assert_eq!(entry.undo_offset, Some(60));
    }

    #[test]
    fn full_mvcc_insert_commit_flow() {
        use crate::mvcc::TransactionManager;

        let manager = TransactionManager::new();
        let _table = MvccTable::new(1);

        let txn = manager.begin_txn().unwrap();
        let txn_id = txn.id();

        let value = MvccTable::prepare_insert_value(txn_id, b"test_data");
        assert!(MvccTable::get_visible(&value, txn_id + 10).unwrap().is_none());

        let commit_ts = txn.commit();

        let mut committed = value.clone();
        MvccTable::finalize_commit_value(&mut committed, commit_ts).unwrap();

        let visible = MvccTable::get_visible(&committed, commit_ts + 1).unwrap();
        assert_eq!(visible, Some(b"test_data".as_slice()));
    }

    #[test]
    fn visibility_snapshot_isolation() {
        use crate::mvcc::TransactionManager;

        let manager = TransactionManager::new();

        let txn1 = manager.begin_txn().unwrap();
        let txn1_id = txn1.id();
        let commit_ts1 = txn1.commit();

        let mut v1 = MvccTable::prepare_insert_value(txn1_id, b"v1");
        MvccTable::finalize_commit_value(&mut v1, commit_ts1).unwrap();

        let reader = manager.begin_txn().unwrap();
        let reader_ts = reader.id();

        let txn2 = manager.begin_txn().unwrap();
        let txn2_id = txn2.id();
        let commit_ts2 = txn2.commit();

        let mut v2 = MvccTable::prepare_insert_value(txn2_id, b"v2");
        MvccTable::finalize_commit_value(&mut v2, commit_ts2).unwrap();

        assert!(MvccTable::get_visible(&v1, reader_ts).unwrap().is_some());
        assert!(MvccTable::get_visible(&v2, reader_ts).unwrap().is_none());
        assert!(MvccTable::get_visible(&v2, commit_ts2 + 1).unwrap().is_some());
    }

    #[test]
    fn conflict_detection_between_writers() {
        use crate::mvcc::TransactionManager;

        let manager = TransactionManager::new();

        let writer1 = manager.begin_txn().unwrap();
        let writer1_ts = writer1.id();

        let mut existing = MvccTable::prepare_insert_value(writer1_ts, b"original");
        let commit_ts = writer1.commit();
        MvccTable::finalize_commit_value(&mut existing, commit_ts).unwrap();

        let writer2 = manager.begin_txn().unwrap();
        let writer2_id = writer2.id();
        let writer2_read_ts = writer2.id();

        let writer3 = manager.begin_txn().unwrap();
        let writer3_id = writer3.id();
        let writer3_read_ts = commit_ts - 1;

        assert_eq!(
            MvccTable::can_write(&existing, writer2_id, writer2_read_ts).unwrap(),
            WriteCheckResult::CanWrite
        );
        assert_eq!(
            MvccTable::can_write(&existing, writer3_id, writer3_read_ts).unwrap(),
            WriteCheckResult::ConcurrentModification
        );
    }

    #[test]
    fn delete_creates_tombstone_visible_to_old_readers() {
        use crate::mvcc::TransactionManager;

        let manager = TransactionManager::new();

        let inserter = manager.begin_txn().unwrap();
        let insert_ts = inserter.id();
        let insert_commit = inserter.commit();

        let mut value = MvccTable::prepare_insert_value(insert_ts, b"data");
        MvccTable::finalize_commit_value(&mut value, insert_commit).unwrap();

        let old_reader = manager.begin_txn().unwrap();
        let old_read_ts = old_reader.id();

        let deleter = manager.begin_txn().unwrap();
        let delete_ts = deleter.id();

        let mut deleted = MvccTable::prepare_delete_value(delete_ts, &value).unwrap();
        let delete_commit = deleter.commit();
        MvccTable::finalize_commit_value(&mut deleted, delete_commit).unwrap();

        assert!(MvccTable::get_visible(&value, old_read_ts).unwrap().is_some());

        let new_reader = manager.begin_txn().unwrap();
        let new_read_ts = new_reader.id();
        assert!(MvccTable::get_visible(&deleted, new_read_ts).unwrap().is_none());
    }

    #[test]
    fn update_creates_version_chain() {
        let v1 = MvccTable::prepare_insert_value(10, b"version1");
        let mut v1_committed = v1.clone();
        MvccTable::finalize_commit_value(&mut v1_committed, 10).unwrap();

        let v2 = MvccTable::prepare_update_value(20, b"version2", 100, 50);

        let parsed = MvccTable::parse_value(&v2).unwrap();
        assert!(parsed.header.has_prev_version());

        let (undo_page, undo_offset) = RecordHeader::decode_ptr(parsed.header.prev_version);
        assert_eq!(undo_page, 100);
        assert_eq!(undo_offset, 50);
    }

    #[test]
    fn version_chain_traversal_finds_visible_version() {
        let v2_data = MvccTable::prepare_update_value(200, b"v2", 1, 0);
        let mut v2_committed = v2_data.clone();
        MvccTable::finalize_commit_value(&mut v2_committed, 200).unwrap();

        let mut v1_header = RecordHeader::new(100);
        v1_header.set_locked(false);
        let v1_user_data = b"v1";

        let load_undo = |_page: u64, _offset: u16| -> Result<Option<(RecordHeader, &[u8])>, eyre::Report> {
            Ok(Some((v1_header, v1_user_data.as_slice())))
        };

        let result = MvccTable::get_visible_with_undo(&v2_committed, 150, load_undo).unwrap();
        assert_eq!(result, Some(b"v1".as_slice()));

        let result_new = MvccTable::get_visible_with_undo(&v2_committed, 250, |_, _| {
            Ok::<_, eyre::Report>(None)
        })
        .unwrap();
        assert_eq!(result_new, Some(b"v2".as_slice()));
    }

    #[test]
    fn undo_record_preserves_original_header() {
        let table = MvccTable::new(42);

        let mut original = MvccTable::prepare_insert_value(100, b"original_data");
        MvccTable::finalize_commit_value(&mut original, 100).unwrap();

        let undo = table.create_undo_record(b"mykey", &original).unwrap();

        assert_eq!(undo.table_id, 42);
        assert_eq!(undo.key, b"mykey");
        assert_eq!(undo.value, b"original_data");
        assert_eq!(undo.record_header.txn_id, 100);
        assert!(!undo.record_header.is_locked());
    }

    #[test]
    fn concurrent_inserts_get_unique_timestamps() {
        use crate::mvcc::TransactionManager;
        use std::sync::Arc;
        use std::thread;

        let manager = Arc::new(TransactionManager::new());
        let mut handles = vec![];

        for _ in 0..10 {
            let mgr: Arc<TransactionManager> = Arc::clone(&manager);
            handles.push(thread::spawn(move || {
                let txn = mgr.begin_txn().unwrap();
                let id = txn.id();
                let value = MvccTable::prepare_insert_value(id, b"data");
                let commit_ts = txn.commit();
                (id, commit_ts, value)
            }));
        }

        let results: Vec<(u64, u64, Vec<u8>)> = handles.into_iter().map(|h| h.join().unwrap()).collect();

        let txn_ids: Vec<_> = results.iter().map(|(id, _, _)| *id).collect();
        let mut unique_ids = txn_ids.clone();
        unique_ids.sort();
        unique_ids.dedup();
        assert_eq!(unique_ids.len(), 10);

        let commit_tss: Vec<_> = results.iter().map(|(_, ts, _)| *ts).collect();
        let mut unique_commits = commit_tss.clone();
        unique_commits.sort();
        unique_commits.dedup();
        assert_eq!(unique_commits.len(), 10);
    }

    #[test]
    fn write_entry_tracks_undo_location() {
        let table = MvccTable::new(1);

        let insert_entry = table.create_write_entry(b"key1", 10, 100, None, None, true);
        assert!(insert_entry.is_insert);
        assert!(insert_entry.undo_page_id.is_none());

        let update_entry = table.create_write_entry(b"key1", 10, 200, Some(5), Some(50), false);
        assert!(!update_entry.is_insert);
        assert_eq!(update_entry.undo_page_id, Some(5));
        assert_eq!(update_entry.undo_offset, Some(50));
    }

    #[test]
    fn locked_value_invisible_to_other_transactions() {
        use crate::mvcc::TransactionManager;

        let manager = TransactionManager::new();

        let writer = manager.begin_txn().unwrap();
        let writer_id = writer.id();

        let locked_value = MvccTable::prepare_insert_value(writer_id, b"uncommitted");

        let reader = manager.begin_txn().unwrap();
        let reader_ts = reader.id();

        assert!(MvccTable::get_visible(&locked_value, reader_ts).unwrap().is_none());

        let own_read = MvccTable::get_visible(&locked_value, writer_id).unwrap();
        assert!(own_read.is_none());
    }

    #[test]
    fn abort_restores_original_state() {
        let mut value = MvccTable::prepare_insert_value(50, b"original");
        MvccTable::finalize_commit_value(&mut value, 50).unwrap();

        let mut modified = MvccTable::prepare_delete_value(100, &value).unwrap();

        let parsed_before = MvccTable::parse_value(&modified).unwrap();
        assert!(parsed_before.header.is_locked());
        assert!(parsed_before.header.is_deleted());
        assert_eq!(parsed_before.header.txn_id, 100);

        MvccTable::finalize_abort_value(&mut modified, 50).unwrap();

        let parsed_after = MvccTable::parse_value(&modified).unwrap();
        assert!(!parsed_after.header.is_locked());
        assert!(!parsed_after.header.is_deleted());
        assert_eq!(parsed_after.header.txn_id, 50);

        let visible = MvccTable::get_visible(&modified, 100).unwrap();
        assert_eq!(visible, Some(b"original".as_slice()));
    }
}
