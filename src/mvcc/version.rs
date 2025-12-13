//! # Version Chain Management
//!
//! This module manages multi-version storage for MVCC. Each row can have
//! multiple versions forming a chain from newest to oldest.
//!
//! ## Version Chain Model
//!
//! ```text
//! B-Tree Leaf (newest version)
//!        │
//!        ▼
//! ┌──────────────────┐
//! │ RecordHeader     │
//! │ - txn_id: 105    │
//! │ - prev: Page4:64 ├────┐
//! │ [Row Data]       │    │
//! └──────────────────┘    │
//!                         ▼
//!                  ┌──────────────────┐
//!                  │ UndoRecord       │
//!                  │ - txn_id: 100    │
//!                  │ - prev: NULL     │
//!                  │ [Old Row Data]   │
//!                  └──────────────────┘
//! ```
//!
//! ## Visibility Check (Snapshot Isolation)
//!
//! A version V is visible to a reader at timestamp read_ts if:
//! 1. V.txn_id <= read_ts (created before or at snapshot)
//! 2. V is not locked (flags & LOCK_BIT == 0)
//! 3. V is not deleted, OR deleted after read_ts
//!
//! ## Version Storage Strategy
//!
//! - **Newest version in B-tree**: Optimizes current data reads
//! - **Old versions in undo pages**: Separate storage for version chains
//! - **Delta vs Full**: We store full copies for simplicity (deltas later)
//!
//! ## Memory Management
//!
//! VersionChain doesn't own the data - it works with references to
//! mmap'd page data. This maintains zero-copy semantics.
//!
//! ## Thread Safety
//!
//! Version chains are modified via the B-tree's locking protocol:
//! - Readers: Traverse chain without locks (immutable once committed)
//! - Writers: Hold row lock (LOCK_BIT) during modification
//!
//! ## Garbage Collection
//!
//! Old versions can be pruned when txn_id < global_watermark:
//! - No active transaction can see versions older than watermark
//! - GC runs in background, marking VACUUM_BIT then reclaiming

use super::record_header::RecordHeader;
use super::TxnId;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VisibilityResult {
    Visible,
    Invisible,
    Deleted,
}

impl RecordHeader {
    pub fn is_visible_to(&self, read_ts: TxnId) -> VisibilityResult {
        if self.is_locked() {
            return VisibilityResult::Invisible;
        }
        if self.txn_id > read_ts {
            return VisibilityResult::Invisible;
        }
        if self.is_deleted() {
            return VisibilityResult::Deleted;
        }
        VisibilityResult::Visible
    }

    pub fn can_write(&self, writer_txn_id: TxnId, writer_read_ts: TxnId) -> WriteCheckResult {
        if self.is_locked() {
            if self.txn_id == writer_txn_id {
                return WriteCheckResult::CanWrite;
            }
            return WriteCheckResult::LockedByOther;
        }
        if self.txn_id > writer_read_ts {
            return WriteCheckResult::ConcurrentModification;
        }
        WriteCheckResult::CanWrite
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WriteCheckResult {
    CanWrite,
    LockedByOther,
    ConcurrentModification,
}

pub struct VersionChainReader<'a> {
    current_header: RecordHeader,
    current_data: &'a [u8],
    read_ts: TxnId,
}

impl<'a> VersionChainReader<'a> {
    pub fn new(data: &'a [u8], read_ts: TxnId) -> Self {
        debug_assert!(data.len() >= RecordHeader::SIZE);
        let header = RecordHeader::from_bytes(data);
        let row_data = &data[RecordHeader::SIZE..];
        Self {
            current_header: header,
            current_data: row_data,
            read_ts,
        }
    }

    pub fn header(&self) -> &RecordHeader {
        &self.current_header
    }

    pub fn data(&self) -> &'a [u8] {
        self.current_data
    }

    pub fn visibility(&self) -> VisibilityResult {
        self.current_header.is_visible_to(self.read_ts)
    }

    pub fn is_visible(&self) -> bool {
        matches!(self.visibility(), VisibilityResult::Visible)
    }

    pub fn prev_version_ptr(&self) -> Option<(u64, u16)> {
        if self.current_header.has_prev_version() {
            Some(RecordHeader::decode_ptr(self.current_header.prev_version))
        } else {
            None
        }
    }

    pub fn has_newer_version(&self) -> bool {
        self.current_header.txn_id > self.read_ts
    }

    pub fn is_reclaimable(&self, global_watermark: TxnId) -> bool {
        !self.current_header.is_locked() && self.current_header.txn_id < global_watermark
    }
}

pub struct VersionChainWriter {
    header: RecordHeader,
}

impl VersionChainWriter {
    pub fn from_existing(data: &[u8]) -> Self {
        debug_assert!(data.len() >= RecordHeader::SIZE);
        let header = RecordHeader::from_bytes(data);
        Self { header }
    }

    pub fn new_version(txn_id: TxnId) -> Self {
        Self {
            header: RecordHeader::new(txn_id),
        }
    }

    pub fn check_write(&self, writer_txn_id: TxnId, writer_read_ts: TxnId) -> WriteCheckResult {
        self.header.can_write(writer_txn_id, writer_read_ts)
    }

    pub fn prepare_update(
        &mut self,
        writer_txn_id: TxnId,
        prev_page_id: u64,
        prev_offset: u16,
    ) -> &RecordHeader {
        self.header.set_locked(true);
        self.header.txn_id = writer_txn_id;
        self.header.prev_version = RecordHeader::encode_ptr(prev_page_id, prev_offset);
        &self.header
    }

    pub fn prepare_insert(&mut self, writer_txn_id: TxnId) -> &RecordHeader {
        self.header.set_locked(true);
        self.header.txn_id = writer_txn_id;
        self.header.prev_version = 0;
        &self.header
    }

    pub fn prepare_delete(&mut self, writer_txn_id: TxnId) -> &RecordHeader {
        self.header.set_locked(true);
        self.header.set_deleted(true);
        self.header.txn_id = writer_txn_id;
        &self.header
    }

    pub fn finalize_commit(&mut self, commit_ts: TxnId) -> &RecordHeader {
        self.header.set_locked(false);
        self.header.txn_id = commit_ts;
        &self.header
    }

    pub fn finalize_abort(&mut self, original_txn_id: TxnId) -> &RecordHeader {
        self.header.set_locked(false);
        self.header.set_deleted(false);
        self.header.txn_id = original_txn_id;
        &self.header
    }

    pub fn header(&self) -> &RecordHeader {
        &self.header
    }

    pub fn write_header_to(&self, buf: &mut [u8]) {
        self.header.write_to(buf);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_record(header: &RecordHeader, data: &[u8]) -> Vec<u8> {
        let mut buf = vec![0u8; RecordHeader::SIZE + data.len()];
        header.write_to(&mut buf[..RecordHeader::SIZE]);
        buf[RecordHeader::SIZE..].copy_from_slice(data);
        buf
    }

    #[test]
    fn visibility_committed_before_read_ts() {
        let mut hdr = RecordHeader::new(10);
        hdr.set_locked(false);
        assert_eq!(hdr.is_visible_to(20), VisibilityResult::Visible);
    }

    #[test]
    fn visibility_committed_at_read_ts() {
        let mut hdr = RecordHeader::new(10);
        hdr.set_locked(false);
        assert_eq!(hdr.is_visible_to(10), VisibilityResult::Visible);
    }

    #[test]
    fn visibility_committed_after_read_ts_is_invisible() {
        let mut hdr = RecordHeader::new(20);
        hdr.set_locked(false);
        assert_eq!(hdr.is_visible_to(10), VisibilityResult::Invisible);
    }

    #[test]
    fn visibility_locked_is_invisible() {
        let mut hdr = RecordHeader::new(5);
        hdr.set_locked(true);
        assert_eq!(hdr.is_visible_to(20), VisibilityResult::Invisible);
    }

    #[test]
    fn visibility_deleted_returns_deleted() {
        let mut hdr = RecordHeader::new(5);
        hdr.set_deleted(true);
        assert_eq!(hdr.is_visible_to(20), VisibilityResult::Deleted);
    }

    #[test]
    fn can_write_unlocked_old_version() {
        let hdr = RecordHeader::new(5);
        assert_eq!(hdr.can_write(100, 10), WriteCheckResult::CanWrite);
    }

    #[test]
    fn cannot_write_locked_by_other() {
        let mut hdr = RecordHeader::new(50);
        hdr.set_locked(true);
        assert_eq!(hdr.can_write(100, 60), WriteCheckResult::LockedByOther);
    }

    #[test]
    fn can_write_own_lock() {
        let mut hdr = RecordHeader::new(100);
        hdr.set_locked(true);
        assert_eq!(hdr.can_write(100, 99), WriteCheckResult::CanWrite);
    }

    #[test]
    fn cannot_write_concurrent_modification() {
        let hdr = RecordHeader::new(15);
        assert_eq!(
            hdr.can_write(100, 10),
            WriteCheckResult::ConcurrentModification
        );
    }

    #[test]
    fn version_chain_reader_parses_header_and_data() {
        let hdr = RecordHeader::new(42);
        let data = b"hello world";
        let record = make_record(&hdr, data);
        let reader = VersionChainReader::new(&record, 100);

        assert_eq!(reader.header().txn_id, 42);
        assert_eq!(reader.data(), b"hello world");
        assert!(reader.is_visible());
    }

    #[test]
    fn version_chain_reader_invisible_for_future_version() {
        let hdr = RecordHeader::new(100);
        let record = make_record(&hdr, b"data");
        let reader = VersionChainReader::new(&record, 50);

        assert!(!reader.is_visible());
    }

    #[test]
    fn version_chain_reader_prev_version_ptr() {
        let mut hdr = RecordHeader::new(42);
        hdr.prev_version = RecordHeader::encode_ptr(123, 456);
        let record = make_record(&hdr, b"data");
        let reader = VersionChainReader::new(&record, 100);

        let (page_id, offset) = reader.prev_version_ptr().unwrap();
        assert_eq!(page_id, 123);
        assert_eq!(offset, 456);
    }

    #[test]
    fn version_chain_reader_no_prev_version() {
        let hdr = RecordHeader::new(42);
        let record = make_record(&hdr, b"data");
        let reader = VersionChainReader::new(&record, 100);

        assert!(reader.prev_version_ptr().is_none());
    }

    #[test]
    fn version_chain_writer_prepare_insert() {
        let mut writer = VersionChainWriter::new_version(0);
        let hdr = writer.prepare_insert(50);

        assert!(hdr.is_locked());
        assert_eq!(hdr.txn_id, 50);
        assert_eq!(hdr.prev_version, 0);
    }

    #[test]
    fn version_chain_writer_prepare_update() {
        let mut writer = VersionChainWriter::new_version(0);
        let hdr = writer.prepare_update(60, 100, 200);

        assert!(hdr.is_locked());
        assert_eq!(hdr.txn_id, 60);
        let (page, off) = RecordHeader::decode_ptr(hdr.prev_version);
        assert_eq!(page, 100);
        assert_eq!(off, 200);
    }

    #[test]
    fn version_chain_writer_prepare_delete() {
        let mut writer = VersionChainWriter::new_version(0);
        let hdr = writer.prepare_delete(70);

        assert!(hdr.is_locked());
        assert!(hdr.is_deleted());
        assert_eq!(hdr.txn_id, 70);
    }

    #[test]
    fn version_chain_writer_finalize_commit() {
        let mut writer = VersionChainWriter::new_version(0);
        writer.prepare_insert(50);
        let hdr = writer.finalize_commit(100);

        assert!(!hdr.is_locked());
        assert_eq!(hdr.txn_id, 100);
    }

    #[test]
    fn version_chain_writer_finalize_abort() {
        let mut writer = VersionChainWriter::new_version(0);
        writer.prepare_delete(50);
        let hdr = writer.finalize_abort(25);

        assert!(!hdr.is_locked());
        assert!(!hdr.is_deleted());
        assert_eq!(hdr.txn_id, 25);
    }

    #[test]
    fn version_chain_writer_write_header_to() {
        let mut writer = VersionChainWriter::new_version(0);
        writer.prepare_insert(42);

        let mut buf = [0u8; RecordHeader::SIZE];
        writer.write_header_to(&mut buf);

        let parsed = RecordHeader::from_bytes(&buf);
        assert_eq!(parsed.txn_id, 42);
        assert!(parsed.is_locked());
    }

    #[test]
    fn has_newer_version_when_txn_id_greater_than_read_ts() {
        let hdr = RecordHeader::new(100);
        let record = make_record(&hdr, b"data");
        let reader = VersionChainReader::new(&record, 50);

        assert!(reader.has_newer_version());
    }

    #[test]
    fn no_newer_version_when_txn_id_at_read_ts() {
        let hdr = RecordHeader::new(50);
        let record = make_record(&hdr, b"data");
        let reader = VersionChainReader::new(&record, 50);

        assert!(!reader.has_newer_version());
    }

    #[test]
    fn no_newer_version_when_txn_id_before_read_ts() {
        let hdr = RecordHeader::new(30);
        let record = make_record(&hdr, b"data");
        let reader = VersionChainReader::new(&record, 50);

        assert!(!reader.has_newer_version());
    }

    #[test]
    fn is_reclaimable_when_older_than_watermark() {
        let hdr = RecordHeader::new(10);
        let record = make_record(&hdr, b"data");
        let reader = VersionChainReader::new(&record, 100);

        assert!(reader.is_reclaimable(20));
    }

    #[test]
    fn not_reclaimable_when_at_watermark() {
        let hdr = RecordHeader::new(20);
        let record = make_record(&hdr, b"data");
        let reader = VersionChainReader::new(&record, 100);

        assert!(!reader.is_reclaimable(20));
    }

    #[test]
    fn not_reclaimable_when_locked() {
        let mut hdr = RecordHeader::new(10);
        hdr.set_locked(true);
        let record = make_record(&hdr, b"data");
        let reader = VersionChainReader::new(&record, 100);

        assert!(!reader.is_reclaimable(20));
    }

    #[test]
    fn not_reclaimable_when_newer_than_watermark() {
        let hdr = RecordHeader::new(30);
        let record = make_record(&hdr, b"data");
        let reader = VersionChainReader::new(&record, 100);

        assert!(!reader.is_reclaimable(20));
    }
}
