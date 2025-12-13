//! # Undo Page for MVCC Version Storage
//!
//! This module implements the append-only undo page format for storing old
//! versions of rows during MVCC transactions.
//!
//! ## Purpose
//!
//! When a row is updated, the previous version must be preserved so that:
//! - Concurrent readers with older read timestamps can still see the old data
//! - Rollback operations can restore the original data
//! - Garbage collection knows which versions are safe to reclaim
//!
//! ## Page Layout
//!
//! ```text
//! +------------------+------------------+
//! | Page Header      | Undo Header      |
//! | (16 bytes)       | (16 bytes)       |
//! +------------------+------------------+
//! | Undo Records (append-only)         |
//! | +--------------------------------+ |
//! | | UndoRecord 1                   | |
//! | | - entry_size: u16              | |
//! | | - table_id: u32                | |
//! | | - RecordHeader: 17 bytes       | |
//! | | - key_len: u16                 | |
//! | | - key: [u8; key_len]           | |
//! | | - value: [u8; remaining]       | |
//! | +--------------------------------+ |
//! | | UndoRecord 2                   | |
//! | | ...                            | |
//! | +--------------------------------+ |
//! +------------------------------------+
//! | Free Space                         |
//! +------------------------------------+
//! ```
//!
//! ## Undo Header (16 bytes)
//!
//! ```text
//! Offset  Size  Field        Description
//! ------  ----  -----------  ----------------------------------------
//! 0       4     next_page    Next undo page in chain (0 = none)
//! 4       2     entry_count  Number of undo records in this page
//! 6       2     free_offset  Offset where next record will be written
//! 8       8     min_txn_id   Minimum txn_id in this page (for GC)
//! ```
//!
//! ## Design Decisions
//!
//! - **Append-only**: New undo records are appended at `free_offset`. No deletion
//!   until GC determines the entire page is reclaimable.
//!
//! - **Linked pages**: When a page fills up, a new page is allocated and linked
//!   via `next_page`. This forms a chain of undo pages per transaction or global.
//!
//! - **min_txn_id tracking**: Each page tracks the minimum txn_id it contains.
//!   When `min_txn_id < global_watermark`, the entire page can be reclaimed.
//!
//! - **Variable-length records**: Each undo record stores its size to allow
//!   sequential traversal and to support variable-length keys and values.
//!
//! ## Memory Layout
//!
//! UndoPage works with raw byte slices from mmap'd pages. The header is read
//! using zerocopy for safe, unaligned access.
//!
//! ## Thread Safety
//!
//! Undo pages are only written by the transaction that owns them. Readers
//! traverse version chains but never modify undo pages. GC runs single-threaded
//! after determining watermark safety.

use super::record_header::RecordHeader;
use super::TxnId;
use crate::storage::{PAGE_HEADER_SIZE, PAGE_SIZE};
use eyre::{bail, ensure, Result};

pub const UNDO_HEADER_SIZE: usize = 16;
pub const UNDO_DATA_START: usize = PAGE_HEADER_SIZE + UNDO_HEADER_SIZE;
pub const UNDO_RECORD_HEADER_SIZE: usize = 2 + 4 + RecordHeader::SIZE + 2;

#[derive(Debug, Clone, Copy)]
pub struct UndoHeader {
    pub next_page: u32,
    pub entry_count: u16,
    pub free_offset: u16,
    pub min_txn_id: TxnId,
}

impl UndoHeader {
    pub fn new() -> Self {
        Self {
            next_page: 0,
            entry_count: 0,
            free_offset: UNDO_DATA_START as u16,
            min_txn_id: u64::MAX,
        }
    }

    pub fn from_bytes(data: &[u8]) -> Self {
        debug_assert!(data.len() >= UNDO_HEADER_SIZE);
        let next_page = u32::from_le_bytes(data[0..4].try_into().unwrap()); // INVARIANT: length asserted above
        let entry_count = u16::from_le_bytes(data[4..6].try_into().unwrap()); // INVARIANT: length asserted above
        let free_offset = u16::from_le_bytes(data[6..8].try_into().unwrap()); // INVARIANT: length asserted above
        let min_txn_id = u64::from_le_bytes(data[8..16].try_into().unwrap()); // INVARIANT: length asserted above
        Self {
            next_page,
            entry_count,
            free_offset,
            min_txn_id,
        }
    }

    pub fn write_to(&self, data: &mut [u8]) {
        debug_assert!(data.len() >= UNDO_HEADER_SIZE);
        data[0..4].copy_from_slice(&self.next_page.to_le_bytes());
        data[4..6].copy_from_slice(&self.entry_count.to_le_bytes());
        data[6..8].copy_from_slice(&self.free_offset.to_le_bytes());
        data[8..16].copy_from_slice(&self.min_txn_id.to_le_bytes());
    }

    pub fn free_space(&self) -> usize {
        PAGE_SIZE.saturating_sub(self.free_offset as usize)
    }

    pub fn can_fit(&self, record_size: usize) -> bool {
        self.free_space() >= record_size
    }
}

impl Default for UndoHeader {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone)]
pub struct UndoRecord {
    pub entry_size: u16,
    pub table_id: u32,
    pub record_header: RecordHeader,
    pub key: Vec<u8>,
    pub value: Vec<u8>,
}

impl UndoRecord {
    pub fn new(table_id: u32, record_header: RecordHeader, key: Vec<u8>, value: Vec<u8>) -> Self {
        let entry_size = (UNDO_RECORD_HEADER_SIZE + key.len() + value.len()) as u16;
        Self {
            entry_size,
            table_id,
            record_header,
            key,
            value,
        }
    }

    pub fn serialized_size(&self) -> usize {
        self.entry_size as usize
    }

    pub fn from_bytes(data: &[u8]) -> Result<Self> {
        ensure!(
            data.len() >= UNDO_RECORD_HEADER_SIZE,
            "undo record too small: {} < {}",
            data.len(),
            UNDO_RECORD_HEADER_SIZE
        );

        let entry_size = u16::from_le_bytes(data[0..2].try_into().unwrap()); // INVARIANT: length validated by ensure above
        ensure!(
            data.len() >= entry_size as usize,
            "undo record truncated: {} < {}",
            data.len(),
            entry_size
        );

        let table_id = u32::from_le_bytes(data[2..6].try_into().unwrap()); // INVARIANT: length validated by ensure above
        let record_header = RecordHeader::from_bytes(&data[6..6 + RecordHeader::SIZE]);
        let key_len = u16::from_le_bytes(
            data[6 + RecordHeader::SIZE..6 + RecordHeader::SIZE + 2]
                .try_into()
                .unwrap(), // INVARIANT: length validated by ensure above (UNDO_RECORD_HEADER_SIZE includes this)
        );

        let key_start = UNDO_RECORD_HEADER_SIZE;
        let key_end = key_start + key_len as usize;
        ensure!(
            entry_size as usize >= key_end,
            "undo record key extends past entry: {} > {}",
            key_end,
            entry_size
        );

        let key = data[key_start..key_end].to_vec();
        let value = data[key_end..entry_size as usize].to_vec();

        Ok(Self {
            entry_size,
            table_id,
            record_header,
            key,
            value,
        })
    }

    pub fn write_to(&self, data: &mut [u8]) -> Result<()> {
        let size = self.serialized_size();
        ensure!(
            data.len() >= size,
            "buffer too small for undo record: {} < {}",
            data.len(),
            size
        );

        data[0..2].copy_from_slice(&self.entry_size.to_le_bytes());
        data[2..6].copy_from_slice(&self.table_id.to_le_bytes());
        self.record_header
            .write_to(&mut data[6..6 + RecordHeader::SIZE]);
        data[6 + RecordHeader::SIZE..6 + RecordHeader::SIZE + 2]
            .copy_from_slice(&(self.key.len() as u16).to_le_bytes());

        let key_start = UNDO_RECORD_HEADER_SIZE;
        let key_end = key_start + self.key.len();
        data[key_start..key_end].copy_from_slice(&self.key);
        data[key_end..key_end + self.value.len()].copy_from_slice(&self.value);

        Ok(())
    }
}

pub struct UndoPageReader<'a> {
    data: &'a [u8],
    header: UndoHeader,
}

impl<'a> UndoPageReader<'a> {
    pub fn new(data: &'a [u8]) -> Result<Self> {
        ensure!(
            data.len() >= UNDO_DATA_START,
            "page too small for undo header"
        );
        let header = UndoHeader::from_bytes(&data[PAGE_HEADER_SIZE..]);
        Ok(Self { data, header })
    }

    pub fn header(&self) -> &UndoHeader {
        &self.header
    }

    pub fn next_page(&self) -> Option<u32> {
        if self.header.next_page == 0 {
            None
        } else {
            Some(self.header.next_page)
        }
    }

    pub fn entry_count(&self) -> u16 {
        self.header.entry_count
    }

    pub fn read_record_at(&self, offset: u16) -> Result<UndoRecord> {
        let offset = offset as usize;
        ensure!(
            offset >= UNDO_DATA_START && offset < self.header.free_offset as usize,
            "undo record offset out of bounds: {}",
            offset
        );
        UndoRecord::from_bytes(&self.data[offset..])
    }

    pub fn iter(&self) -> UndoRecordIter<'a> {
        UndoRecordIter {
            data: self.data,
            current_offset: UNDO_DATA_START,
            end_offset: self.header.free_offset as usize,
        }
    }

    pub fn is_reclaimable(&self, global_watermark: TxnId) -> bool {
        self.header.min_txn_id < global_watermark
    }
}

pub struct UndoRecordIter<'a> {
    data: &'a [u8],
    current_offset: usize,
    end_offset: usize,
}

impl<'a> Iterator for UndoRecordIter<'a> {
    type Item = Result<(u16, UndoRecord)>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.current_offset >= self.end_offset {
            return None;
        }

        let offset = self.current_offset as u16;
        match UndoRecord::from_bytes(&self.data[self.current_offset..]) {
            Ok(record) => {
                self.current_offset += record.serialized_size();
                Some(Ok((offset, record)))
            }
            Err(e) => Some(Err(e)),
        }
    }
}

pub struct UndoPageWriter<'a> {
    data: &'a mut [u8],
    header: UndoHeader,
}

impl<'a> UndoPageWriter<'a> {
    pub fn new(data: &'a mut [u8]) -> Result<Self> {
        ensure!(
            data.len() >= PAGE_SIZE,
            "page buffer too small: {} < {}",
            data.len(),
            PAGE_SIZE
        );
        let header = UndoHeader::from_bytes(&data[PAGE_HEADER_SIZE..]);
        Ok(Self { data, header })
    }

    pub fn init_empty(data: &'a mut [u8]) -> Result<Self> {
        ensure!(
            data.len() >= PAGE_SIZE,
            "page buffer too small: {} < {}",
            data.len(),
            PAGE_SIZE
        );

        use crate::storage::{PageHeader, PageType};
        let page_header = PageHeader::new(PageType::Undo);
        page_header.write_to(&mut data[..PAGE_HEADER_SIZE])?;

        let undo_header = UndoHeader::new();
        undo_header.write_to(&mut data[PAGE_HEADER_SIZE..]);

        Ok(Self {
            data,
            header: undo_header,
        })
    }

    pub fn header(&self) -> &UndoHeader {
        &self.header
    }

    pub fn append(&mut self, record: &UndoRecord) -> Result<u16> {
        let size = record.serialized_size();
        if !self.header.can_fit(size) {
            bail!(
                "undo page full: need {} bytes, have {} free",
                size,
                self.header.free_space()
            );
        }

        let offset = self.header.free_offset;
        record.write_to(&mut self.data[offset as usize..])?;

        self.header.free_offset += size as u16;
        self.header.entry_count += 1;

        if record.record_header.txn_id < self.header.min_txn_id {
            self.header.min_txn_id = record.record_header.txn_id;
        }

        self.flush_header();
        Ok(offset)
    }

    pub fn set_next_page(&mut self, page_no: u32) {
        self.header.next_page = page_no;
        self.flush_header();
    }

    fn flush_header(&mut self) {
        self.header
            .write_to(&mut self.data[PAGE_HEADER_SIZE..PAGE_HEADER_SIZE + UNDO_HEADER_SIZE]);
    }

    pub fn free_space(&self) -> usize {
        self.header.free_space()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_page() -> Vec<u8> {
        vec![0u8; PAGE_SIZE]
    }

    #[test]
    fn undo_header_new_defaults() {
        let header = UndoHeader::new();
        assert_eq!(header.next_page, 0);
        assert_eq!(header.entry_count, 0);
        assert_eq!(header.free_offset, UNDO_DATA_START as u16);
        assert_eq!(header.min_txn_id, u64::MAX);
    }

    #[test]
    fn undo_header_roundtrip() {
        let original = UndoHeader {
            next_page: 42,
            entry_count: 5,
            free_offset: 1000,
            min_txn_id: 12345,
        };
        let mut buf = [0u8; UNDO_HEADER_SIZE];
        original.write_to(&mut buf);
        let restored = UndoHeader::from_bytes(&buf);

        assert_eq!(restored.next_page, 42);
        assert_eq!(restored.entry_count, 5);
        assert_eq!(restored.free_offset, 1000);
        assert_eq!(restored.min_txn_id, 12345);
    }

    #[test]
    fn undo_header_free_space() {
        let mut header = UndoHeader::new();
        let expected = PAGE_SIZE - UNDO_DATA_START;
        assert_eq!(header.free_space(), expected);

        header.free_offset = 1000;
        assert_eq!(header.free_space(), PAGE_SIZE - 1000);
    }

    #[test]
    fn undo_record_new_calculates_size() {
        let hdr = RecordHeader::new(100);
        let record = UndoRecord::new(1, hdr, vec![1, 2, 3], vec![4, 5, 6, 7]);

        assert_eq!(record.entry_size as usize, UNDO_RECORD_HEADER_SIZE + 3 + 4);
    }

    #[test]
    fn undo_record_roundtrip() {
        let hdr = RecordHeader::new(42);
        let original = UndoRecord::new(123, hdr, b"key".to_vec(), b"value".to_vec());

        let mut buf = vec![0u8; original.serialized_size() + 10];
        original.write_to(&mut buf).unwrap();

        let restored = UndoRecord::from_bytes(&buf).unwrap();

        assert_eq!(restored.table_id, 123);
        assert_eq!(restored.record_header.txn_id, 42);
        assert_eq!(restored.key, b"key");
        assert_eq!(restored.value, b"value");
    }

    #[test]
    fn undo_page_writer_init_empty() {
        let mut page = make_page();
        let writer = UndoPageWriter::init_empty(&mut page).unwrap();

        assert_eq!(writer.header().entry_count, 0);
        assert_eq!(writer.header().free_offset, UNDO_DATA_START as u16);
    }

    #[test]
    fn undo_page_writer_append_record() {
        let mut page = make_page();
        let mut writer = UndoPageWriter::init_empty(&mut page).unwrap();

        let hdr = RecordHeader::new(100);
        let record = UndoRecord::new(1, hdr, b"key1".to_vec(), b"value1".to_vec());
        let offset = writer.append(&record).unwrap();

        assert_eq!(offset, UNDO_DATA_START as u16);
        assert_eq!(writer.header().entry_count, 1);
        assert_eq!(writer.header().min_txn_id, 100);
    }

    #[test]
    fn undo_page_writer_append_multiple() {
        let mut page = make_page();
        let mut writer = UndoPageWriter::init_empty(&mut page).unwrap();

        let hdr1 = RecordHeader::new(200);
        let record1 = UndoRecord::new(1, hdr1, b"k1".to_vec(), b"v1".to_vec());
        let offset1 = writer.append(&record1).unwrap();

        let hdr2 = RecordHeader::new(100);
        let record2 = UndoRecord::new(1, hdr2, b"k2".to_vec(), b"v2".to_vec());
        let offset2 = writer.append(&record2).unwrap();

        assert_eq!(offset1, UNDO_DATA_START as u16);
        assert!(offset2 > offset1);
        assert_eq!(writer.header().entry_count, 2);
        assert_eq!(writer.header().min_txn_id, 100);
    }

    #[test]
    fn undo_page_reader_iterate_records() {
        let mut page = make_page();
        {
            let mut writer = UndoPageWriter::init_empty(&mut page).unwrap();

            for i in 0..3 {
                let hdr = RecordHeader::new(i * 10);
                let record =
                    UndoRecord::new(1, hdr, format!("key{}", i).into_bytes(), b"val".to_vec());
                writer.append(&record).unwrap();
            }
        }

        let reader = UndoPageReader::new(&page).unwrap();
        assert_eq!(reader.entry_count(), 3);

        let records: Vec<_> = reader.iter().collect();
        assert_eq!(records.len(), 3);

        let (_, r0) = records[0].as_ref().unwrap();
        assert_eq!(r0.record_header.txn_id, 0);
        assert_eq!(r0.key, b"key0");

        let (_, r2) = records[2].as_ref().unwrap();
        assert_eq!(r2.record_header.txn_id, 20);
        assert_eq!(r2.key, b"key2");
    }

    #[test]
    fn undo_page_reader_read_at_offset() {
        let mut page = make_page();
        let offset2;
        {
            let mut writer = UndoPageWriter::init_empty(&mut page).unwrap();

            let hdr1 = RecordHeader::new(10);
            writer
                .append(&UndoRecord::new(1, hdr1, b"k1".to_vec(), b"v1".to_vec()))
                .unwrap();

            let hdr2 = RecordHeader::new(20);
            offset2 = writer
                .append(&UndoRecord::new(2, hdr2, b"k2".to_vec(), b"v2".to_vec()))
                .unwrap();
        }

        let reader = UndoPageReader::new(&page).unwrap();
        let record = reader.read_record_at(offset2).unwrap();

        assert_eq!(record.table_id, 2);
        assert_eq!(record.record_header.txn_id, 20);
        assert_eq!(record.key, b"k2");
    }

    #[test]
    fn undo_page_is_reclaimable() {
        let mut page = make_page();
        {
            let mut writer = UndoPageWriter::init_empty(&mut page).unwrap();
            let hdr = RecordHeader::new(50);
            writer
                .append(&UndoRecord::new(1, hdr, b"k".to_vec(), b"v".to_vec()))
                .unwrap();
        }

        let reader = UndoPageReader::new(&page).unwrap();
        assert!(reader.is_reclaimable(100));
        assert!(!reader.is_reclaimable(50));
        assert!(!reader.is_reclaimable(25));
    }

    #[test]
    fn undo_page_next_page_chain() {
        let mut page = make_page();
        {
            let mut writer = UndoPageWriter::init_empty(&mut page).unwrap();
            writer.set_next_page(42);
        }

        let reader = UndoPageReader::new(&page).unwrap();
        assert_eq!(reader.next_page(), Some(42));
    }

    #[test]
    fn undo_page_no_next_page() {
        let mut page = make_page();
        UndoPageWriter::init_empty(&mut page).unwrap();

        let reader = UndoPageReader::new(&page).unwrap();
        assert_eq!(reader.next_page(), None);
    }
}
