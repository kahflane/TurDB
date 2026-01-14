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

    pub fn is_visible_with_clog<F>(&self, read_ts: TxnId, get_commit_ts: F) -> VisibilityResult
    where
        F: Fn(TxnId) -> Option<TxnId>,
    {
        let effective_ts = if self.is_locked() {
            match get_commit_ts(self.txn_id) {
                Some(commit_ts) => commit_ts,
                None => return VisibilityResult::Invisible,
            }
        } else {
            self.txn_id
        };

        if effective_ts > read_ts {
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

    pub fn find_visible_version<F, E>(
        &self,
        mut load_undo: F,
    ) -> Result<Option<VisibleVersion<'a>>, E>
    where
        F: FnMut(u64, u16) -> Result<Option<(RecordHeader, &'a [u8])>, E>,
    {
        if matches!(self.visibility(), VisibilityResult::Visible) {
            return Ok(Some(VisibleVersion {
                header: self.current_header,
                data: self.current_data,
            }));
        }

        if matches!(self.visibility(), VisibilityResult::Deleted) {
            return Ok(None);
        }

        let mut current_header = self.current_header;

        while current_header.has_prev_version() {
            let (page_id, offset) = RecordHeader::decode_ptr(current_header.prev_version);

            match load_undo(page_id, offset)? {
                Some((prev_header, prev_data)) => {
                    let vis = prev_header.is_visible_to(self.read_ts);
                    match vis {
                        VisibilityResult::Visible => {
                            return Ok(Some(VisibleVersion {
                                header: prev_header,
                                data: prev_data,
                            }));
                        }
                        VisibilityResult::Deleted => return Ok(None),
                        VisibilityResult::Invisible => {
                            current_header = prev_header;
                        }
                    }
                }
                None => break,
            }
        }

        Ok(None)
    }
}

#[derive(Debug, Clone, Copy)]
pub struct VisibleVersion<'a> {
    pub header: RecordHeader,
    pub data: &'a [u8],
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

    pub fn header(&self) -> &RecordHeader {
        &self.header
    }

    pub fn write_header_to(&self, buf: &mut [u8]) {
        self.header.write_to(buf);
    }
}
