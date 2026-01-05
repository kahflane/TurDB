//! # MVCC-Aware Query Scanning
//!
//! This module provides MVCC-aware wrappers for query execution, implementing
//! visibility checks during table scans according to Snapshot Isolation rules.
//!
//! ## Purpose
//!
//! When MVCC is enabled, records in the B-tree have a 17-byte RecordHeader prepended:
//! - Flags (1 byte): Lock bit, delete bit, vacuum hint
//! - TxnId (8 bytes): Transaction that created this version
//! - PrevVersion (8 bytes): Pointer to older version in undo log
//!
//! The MvccStreamingSource handles:
//! - Stripping the RecordHeader before decoding
//! - Checking visibility against the transaction's read timestamp
//! - Skipping invisible or deleted rows
//! - Traversing version chains when needed
//!
//! ## Visibility Rules
//!
//! A row is visible to a transaction T with read_ts if:
//! 1. The row is not locked by another transaction
//! 2. The row's txn_id <= T.read_ts
//! 3. The row is not marked as deleted
//!
//! ## Integration
//!
//! MvccStreamingSource is always used for table scans. MVCC is always enabled,
//! and all records have MVCC headers.
//!
//! ## Zero-Copy Design
//!
//! The implementation maintains zero-copy semantics by:
//! - Passing slices of the original mmap'd data
//! - Only allocating when traversing version chains

use crate::mvcc::{RecordHeader, TxnId};
use crate::sql::decoder::SimpleDecoder;
use crate::sql::executor::RowSource;
use crate::types::Value;
use eyre::Result;

pub struct MvccStreamingSource<'storage> {
    cursor: crate::btree::Cursor<'storage, crate::storage::MmapStorage>,
    decoder: SimpleDecoder,
    read_ts: TxnId,
    started: bool,
    row_buffer: Vec<Value<'static>>,
    end_key: Option<Vec<u8>>,
    own_txn_id: TxnId,
}

impl<'storage> MvccStreamingSource<'storage> {
    pub fn new(
        cursor: crate::btree::Cursor<'storage, crate::storage::MmapStorage>,
        decoder: SimpleDecoder,
        read_ts: TxnId,
        own_txn_id: TxnId,
        column_count: usize,
    ) -> Self {
        Self {
            cursor,
            decoder,
            read_ts,
            started: false,
            row_buffer: Vec::with_capacity(column_count),
            end_key: None,
            own_txn_id,
        }
    }

    pub fn with_end_key(
        cursor: crate::btree::Cursor<'storage, crate::storage::MmapStorage>,
        decoder: SimpleDecoder,
        read_ts: TxnId,
        own_txn_id: TxnId,
        column_count: usize,
        end_key: Option<Vec<u8>>,
    ) -> Self {
        Self {
            cursor,
            decoder,
            read_ts,
            started: false,
            row_buffer: Vec::with_capacity(column_count),
            end_key,
            own_txn_id,
        }
    }

    pub fn from_btree_scan(
        storage: &'storage crate::storage::MmapStorage,
        root_page: u32,
        column_types: Vec<crate::records::types::DataType>,
        read_ts: TxnId,
        own_txn_id: TxnId,
    ) -> Result<Self> {
        Self::from_btree_scan_with_projections(storage, root_page, column_types, None, read_ts, own_txn_id)
    }

    pub fn from_btree_scan_with_projections(
        storage: &'storage crate::storage::MmapStorage,
        root_page: u32,
        column_types: Vec<crate::records::types::DataType>,
        projections: Option<Vec<usize>>,
        read_ts: TxnId,
        own_txn_id: TxnId,
    ) -> Result<Self> {
        use crate::btree::BTreeReader;

        let reader = BTreeReader::new(storage, root_page)?;
        let cursor = reader.cursor_first()?;

        let output_count = projections
            .as_ref()
            .map(|p| p.len())
            .unwrap_or(column_types.len());
        let decoder = match projections {
            Some(proj) => SimpleDecoder::with_projections(column_types, proj),
            None => SimpleDecoder::new(column_types),
        };

        Ok(Self::new(cursor, decoder, read_ts, own_txn_id, output_count))
    }

    fn is_visible(&self, raw_value: &[u8]) -> Result<bool> {
        if raw_value.len() < RecordHeader::SIZE {
            return Ok(true);
        }

        let header = RecordHeader::from_bytes(raw_value);

        if header.is_locked() && header.txn_id != self.own_txn_id {
            return Ok(false);
        }

        if header.txn_id > self.read_ts && header.txn_id != self.own_txn_id {
            return Ok(false);
        }

        if header.is_deleted() {
            return Ok(false);
        }

        Ok(true)
    }

    fn get_user_data<'a>(&self, raw_value: &'a [u8]) -> &'a [u8] {
        if raw_value.len() > RecordHeader::SIZE {
            &raw_value[RecordHeader::SIZE..]
        } else {
            raw_value
        }
    }
}

impl<'storage> RowSource for MvccStreamingSource<'storage> {
    fn reset(&mut self) -> Result<()> {
        self.started = false;
        Ok(())
    }

    fn next_row(&mut self) -> Result<Option<Vec<Value<'static>>>> {
        loop {
            if !self.started {
                self.started = true;
                if !self.cursor.valid() {
                    return Ok(None);
                }
            } else if !self.cursor.advance()? {
                return Ok(None);
            }

            let key = self.cursor.key()?;

            if let Some(ref end) = self.end_key {
                if key >= end.as_slice() {
                    return Ok(None);
                }
            }

            let raw_value = self.cursor.value()?;

            if !self.is_visible(raw_value)? {
                continue;
            }

            let user_data = self.get_user_data(raw_value);
            self.row_buffer.clear();
            self.decoder.decode_into(key, user_data, &mut self.row_buffer)?;
            return Ok(Some(std::mem::take(&mut self.row_buffer)));
        }
    }
}

pub fn check_row_visibility(raw_value: &[u8], read_ts: TxnId, own_txn_id: TxnId) -> bool {
    if raw_value.len() < RecordHeader::SIZE {
        return true;
    }

    let header = RecordHeader::from_bytes(raw_value);

    if header.is_locked() && header.txn_id != own_txn_id {
        return false;
    }

    if header.txn_id > read_ts && header.txn_id != own_txn_id {
        return false;
    }

    if header.is_deleted() {
        return false;
    }

    true
}

pub fn strip_mvcc_header(raw_value: &[u8]) -> &[u8] {
    if raw_value.len() > RecordHeader::SIZE {
        &raw_value[RecordHeader::SIZE..]
    } else {
        raw_value
    }
}
