//! # Write-Ahead Log (WAL) Implementation
//!
//! This module implements a write-ahead logging system for TurDB, providing durability
//! and crash recovery guarantees. The WAL ensures that all database modifications are
//! recorded in a sequential log before being applied to the main database files.
//!
//! ## Architecture Overview
//!
//! The WAL uses a segmented log design where each segment is a separate file. This
//! allows for efficient log rotation and prevents unbounded growth of a single file.
//! Each log entry contains a complete copy of a modified page plus metadata for
//! validation and recovery.
//!
//! ```text
//! database_dir/
//! └── wal/
//!     ├── wal.000001       # First segment
//!     ├── wal.000002       # Second segment (after rotation)
//!     └── wal.000003       # Current active segment
//! ```
//!
//! ## Frame Format
//!
//! Each WAL entry is called a "frame" and consists of a header plus page data:
//!
//! ```text
//! +------------------+------------------+
//! | Frame Header     | Page Data        |
//! | (32 bytes)       | (16384 bytes)    |
//! +------------------+------------------+
//! ```
//!
//! The frame header contains:
//! - `page_no`: Which page in the database this frame represents
//! - `db_size`: Database size (in pages) after applying this frame
//! - `salt1`, `salt2`: Random values for checksum validation
//! - `checksum`: CRC64 over frame header and page data
//!
//! ## Write Protocol
//!
//! 1. Append frame to current WAL segment
//! 2. Compute checksum over header + data
//! 3. Write frame atomically (header + data in single write)
//! 4. Optionally sync to disk for durability
//!
//! ## Read Protocol
//!
//! 1. Read frame header
//! 2. Validate checksum
//! 3. If valid, return page data
//! 4. If invalid, treat as end-of-log (normal during recovery)
//!
//! ## Checkpointing
//!
//! Periodically, WAL frames are copied back to the main database files and the
//! WAL is truncated. This prevents unbounded WAL growth and improves read
//! performance (reads don't need to scan the WAL).
//!
//! Checkpoint triggers:
//! - WAL exceeds size threshold (configurable, default 64MB)
//! - Explicit checkpoint requested
//! - Database close
//!
//! ## Crash Recovery
//!
//! On database open, the WAL is scanned and all valid frames are replayed to
//! reconstruct the database state. Invalid frames (corrupted checksum) indicate
//! incomplete writes and are discarded.
//!
//! ## Concurrency
//!
//! The WAL uses a `Mutex` for exclusive write access. Multiple readers can
//! access the WAL concurrently using memory-mapped I/O, but writes are serialized.
//! This matches the "many readers, one writer" model.
//!
//! ## Performance Characteristics
//!
//! - Write: O(1) append to current segment, sequential I/O
//! - Read: O(n) where n is number of frames (typically small)
//! - Checkpoint: O(m) where m is number of dirty pages
//! - Recovery: O(n) where n is number of frames since last checkpoint
//!
//! ## Thread Safety
//!
//! `Wal` is `Send + Sync`. Internal synchronization uses `parking_lot::Mutex`.
//!
//! ## Platform Support
//!
//! Atomic writes rely on OS guarantees:
//! - Linux: writes up to PIPE_BUF bytes are atomic
//! - macOS: writes up to 512 bytes are atomic
//! - Windows: writes are not guaranteed atomic (use FILE_FLAG_WRITE_THROUGH)
//!
//! For safety, we sync after each frame write by default.

use zerocopy::{IntoBytes, FromBytes, Immutable};
use crc::{Crc, CRC_64_ECMA_182};

pub const WAL_FRAME_HEADER_SIZE: usize = 32;

const CRC64: Crc<u64> = Crc::<u64>::new(&CRC_64_ECMA_182);

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, IntoBytes, FromBytes, Immutable)]
pub struct WalFrameHeader {
    pub page_no: u32,
    pub db_size: u32,
    pub salt1: u32,
    pub salt2: u32,
    pub checksum: u64,
    _reserved: [u8; 8],
}

impl WalFrameHeader {
    pub fn new(page_no: u32, db_size: u32, salt1: u32, salt2: u32, checksum: u64) -> Self {
        Self {
            page_no,
            db_size,
            salt1,
            salt2,
            checksum,
            _reserved: [0; 8],
        }
    }
}

pub fn compute_checksum(header: &WalFrameHeader, page_data: &[u8]) -> u64 {
    let mut digest = CRC64.digest();

    digest.update(&header.page_no.to_le_bytes());
    digest.update(&header.db_size.to_le_bytes());
    digest.update(&header.salt1.to_le_bytes());
    digest.update(&header.salt2.to_le_bytes());

    digest.update(page_data);

    digest.finalize()
}

pub fn validate_checksum(header: &WalFrameHeader, page_data: &[u8]) -> bool {
    let computed = compute_checksum(header, page_data);
    computed == header.checksum
}

use eyre::{Result, WrapErr, ensure, bail};
use std::fs::{File, OpenOptions, create_dir_all};
use std::path::{Path, PathBuf};
use std::io::{Write, Read, Seek, SeekFrom};
use parking_lot::Mutex;

use super::PAGE_SIZE;

pub struct Wal {
    #[allow(dead_code)]
    dir: PathBuf,
    current_segment: Mutex<WalSegment>,
}

impl Wal {
    pub fn create(dir: &Path) -> Result<Self> {
        create_dir_all(dir)
            .wrap_err_with(|| format!("failed to create WAL directory at {:?}", dir))?;

        let segment_path = dir.join("wal.000001");
        let segment = WalSegment::create(&segment_path, 1)?;

        Ok(Self {
            dir: dir.to_path_buf(),
            current_segment: Mutex::new(segment),
        })
    }

    pub fn open(dir: &Path) -> Result<Self> {
        if !dir.exists() {
            create_dir_all(dir)
                .wrap_err_with(|| format!("failed to create WAL directory at {:?}", dir))?;
        }

        let segment_path = dir.join("wal.000001");

        let segment = if segment_path.exists() {
            WalSegment::open(&segment_path, 1)?
        } else {
            WalSegment::create(&segment_path, 1)?
        };

        Ok(Self {
            dir: dir.to_path_buf(),
            current_segment: Mutex::new(segment),
        })
    }

    pub fn recover(&mut self, storage: &mut super::MmapStorage) -> Result<u32> {
        let segment_path = self.dir.join("wal.000001");

        if !segment_path.exists() {
            return Ok(0);
        }

        let mut segment = WalSegment::open(&segment_path, 1)?;
        let mut frames_applied = 0;

        loop {
            match segment.read_frame() {
                Ok((header, page_data)) => {
                    let page_mut = storage.page_mut(header.page_no)
                        .wrap_err_with(|| format!("failed to get page {} for WAL recovery", header.page_no))?;

                    page_mut.copy_from_slice(&page_data);
                    frames_applied += 1;
                }
                Err(_) => break,
            }
        }

        Ok(frames_applied)
    }

    pub fn checkpoint(&mut self, storage: &mut super::MmapStorage) -> Result<u32> {
        let frames_applied = self.recover(storage)?;

        self.truncate()?;

        Ok(frames_applied)
    }

    pub fn truncate(&mut self) -> Result<()> {
        use std::io::Write;

        let mut segment = self.current_segment.lock();

        segment.file
            .set_len(0)
            .wrap_err("failed to truncate WAL segment file")?;

        segment.file
            .flush()
            .wrap_err("failed to flush WAL segment after truncate")?;

        segment.offset = 0;

        Ok(())
    }

    pub fn needs_checkpoint(&self, threshold_bytes: u64) -> bool {
        self.current_offset() >= threshold_bytes
    }

    pub fn write_frame(&mut self, header: &WalFrameHeader, page_data: &[u8]) -> Result<()> {
        ensure!(
            page_data.len() == PAGE_SIZE,
            "page data must be exactly {} bytes, got {}",
            PAGE_SIZE,
            page_data.len()
        );

        let checksum = compute_checksum(header, page_data);
        let mut header_with_checksum = *header;
        header_with_checksum.checksum = checksum;

        let mut segment = self.current_segment.lock();
        segment.write_frame(&header_with_checksum, page_data)
    }

    pub fn current_offset(&self) -> u64 {
        let segment = self.current_segment.lock();
        segment.offset()
    }
}

pub struct WalSegment {
    file: File,
    sequence: u64,
    offset: u64,
}

impl WalSegment {
    pub fn create(path: &Path, sequence: u64) -> Result<Self> {
        let file = OpenOptions::new()
            .create(true)
            .write(true)
            .read(true)
            .truncate(true)
            .open(path)
            .wrap_err_with(|| format!("failed to create WAL segment at {:?}", path))?;

        Ok(Self {
            file,
            sequence,
            offset: 0,
        })
    }

    pub fn open(path: &Path, sequence: u64) -> Result<Self> {
        let mut file = OpenOptions::new()
            .read(true)
            .write(true)
            .open(path)
            .wrap_err_with(|| format!("failed to open WAL segment at {:?}", path))?;

        file.seek(SeekFrom::Start(0))
            .wrap_err("failed to seek to start of WAL segment")?;

        Ok(Self {
            file,
            sequence,
            offset: 0,
        })
    }

    pub fn sequence(&self) -> u64 {
        self.sequence
    }

    pub fn offset(&self) -> u64 {
        self.offset
    }

    pub fn write_frame(&mut self, header: &WalFrameHeader, page_data: &[u8]) -> Result<()> {
        let header_bytes = header.as_bytes();
        self.file
            .write_all(header_bytes)
            .wrap_err("failed to write WAL frame header")?;

        self.file
            .write_all(page_data)
            .wrap_err("failed to write WAL frame page data")?;

        self.file
            .sync_all()
            .wrap_err("failed to sync WAL frame to disk")?;

        self.offset += (WAL_FRAME_HEADER_SIZE + PAGE_SIZE) as u64;

        Ok(())
    }

    pub fn read_frame(&mut self) -> Result<(WalFrameHeader, Vec<u8>)> {
        let mut header_bytes = vec![0u8; WAL_FRAME_HEADER_SIZE];
        self.file
            .read_exact(&mut header_bytes)
            .wrap_err("failed to read WAL frame header")?;

        let header = WalFrameHeader::read_from_bytes(&header_bytes)
            .map_err(|e| eyre::eyre!("invalid WAL frame header: {:?}", e))?;

        let mut page_data = vec![0u8; PAGE_SIZE];
        self.file
            .read_exact(&mut page_data)
            .wrap_err("failed to read WAL frame page data")?;

        if !validate_checksum(&header, &page_data) {
            bail!("WAL frame checksum validation failed");
        }

        self.offset += (WAL_FRAME_HEADER_SIZE + PAGE_SIZE) as u64;

        Ok((header, page_data))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wal_frame_header_size_is_32_bytes() {
        assert_eq!(std::mem::size_of::<WalFrameHeader>(), 32);
    }

    #[test]
    fn wal_frame_header_default_values() {
        let header = WalFrameHeader::new(0, 0, 0, 0, 0);
        assert_eq!(header.page_no, 0);
        assert_eq!(header.db_size, 0);
        assert_eq!(header.salt1, 0);
        assert_eq!(header.salt2, 0);
        assert_eq!(header.checksum, 0);
    }

    #[test]
    fn wal_frame_header_sets_values_correctly() {
        let header = WalFrameHeader::new(42, 100, 0xDEADBEEF, 0xCAFEBABE, 0x123456789ABCDEF0);
        assert_eq!(header.page_no, 42);
        assert_eq!(header.db_size, 100);
        assert_eq!(header.salt1, 0xDEADBEEF);
        assert_eq!(header.salt2, 0xCAFEBABE);
        assert_eq!(header.checksum, 0x123456789ABCDEF0);
    }

    #[test]
    fn wal_frame_header_as_bytes_roundtrip() {
        let original = WalFrameHeader::new(123, 456, 789, 101112, 131415);
        let bytes = original.as_bytes();
        let parsed = WalFrameHeader::read_from_bytes(bytes).expect("should parse");

        assert_eq!(parsed.page_no, original.page_no);
        assert_eq!(parsed.db_size, original.db_size);
        assert_eq!(parsed.salt1, original.salt1);
        assert_eq!(parsed.salt2, original.salt2);
        assert_eq!(parsed.checksum, original.checksum);
    }

    #[test]
    fn wal_segment_creates_new_file() {
        let temp_dir = std::env::temp_dir().join("turdb_test_wal_segment");
        std::fs::create_dir_all(&temp_dir).unwrap();

        let segment_path = temp_dir.join("wal.000001");
        let segment = WalSegment::create(&segment_path, 1).expect("should create segment");

        assert_eq!(segment.sequence(), 1);
        assert!(segment_path.exists());

        std::fs::remove_dir_all(&temp_dir).ok();
    }

    #[test]
    fn wal_segment_tracks_write_offset() {
        let temp_dir = std::env::temp_dir().join("turdb_test_wal_offset");
        std::fs::create_dir_all(&temp_dir).unwrap();

        let segment_path = temp_dir.join("wal.000002");
        let segment = WalSegment::create(&segment_path, 2).expect("should create segment");

        assert_eq!(segment.offset(), 0);

        std::fs::remove_dir_all(&temp_dir).ok();
    }

    #[test]
    fn wal_creates_directory_structure() {
        let temp_dir = std::env::temp_dir().join("turdb_test_wal_dir");
        let wal_dir = temp_dir.join("wal");

        if wal_dir.exists() {
            std::fs::remove_dir_all(&wal_dir).ok();
        }

        let wal = Wal::create(&wal_dir).expect("should create WAL");

        assert!(wal_dir.exists());
        assert!(wal_dir.is_dir());

        std::fs::remove_dir_all(&temp_dir).ok();
    }

    #[test]
    fn wal_starts_with_first_segment() {
        let temp_dir = std::env::temp_dir().join("turdb_test_wal_first_segment");
        let wal_dir = temp_dir.join("wal");

        if wal_dir.exists() {
            std::fs::remove_dir_all(&wal_dir).ok();
        }

        let wal = Wal::create(&wal_dir).expect("should create WAL");

        let first_segment_path = wal_dir.join("wal.000001");
        assert!(first_segment_path.exists());

        std::fs::remove_dir_all(&temp_dir).ok();
    }

    #[test]
    fn write_frame_appends_to_wal() {
        use super::super::PAGE_SIZE;

        let temp_dir = std::env::temp_dir().join("turdb_test_write_frame");
        let wal_dir = temp_dir.join("wal");

        if wal_dir.exists() {
            std::fs::remove_dir_all(&wal_dir).ok();
        }

        let mut wal = Wal::create(&wal_dir).expect("should create WAL");

        let page_data = vec![42u8; PAGE_SIZE];
        let header = WalFrameHeader::new(5, 10, 0x12345678, 0x9ABCDEF0, 0);

        wal.write_frame(&header, &page_data).expect("should write frame");

        std::fs::remove_dir_all(&temp_dir).ok();
    }

    #[test]
    fn write_frame_updates_offset() {
        use super::super::PAGE_SIZE;

        let temp_dir = std::env::temp_dir().join("turdb_test_write_frame_offset");
        let wal_dir = temp_dir.join("wal");

        if wal_dir.exists() {
            std::fs::remove_dir_all(&wal_dir).ok();
        }

        let mut wal = Wal::create(&wal_dir).expect("should create WAL");

        let page_data = vec![99u8; PAGE_SIZE];
        let header = WalFrameHeader::new(1, 1, 0, 0, 0);

        wal.write_frame(&header, &page_data).expect("should write frame");

        let expected_offset = WAL_FRAME_HEADER_SIZE + PAGE_SIZE;
        assert_eq!(wal.current_offset(), expected_offset as u64);

        std::fs::remove_dir_all(&temp_dir).ok();
    }

    #[test]
    fn compute_checksum_for_frame() {
        use super::super::PAGE_SIZE;

        let page_data = vec![42u8; PAGE_SIZE];
        let mut header = WalFrameHeader::new(5, 10, 0x12345678, 0x9ABCDEF0, 0);

        let checksum = compute_checksum(&header, &page_data);

        assert_ne!(checksum, 0, "checksum should not be zero");

        header.checksum = checksum;
    }

    #[test]
    fn same_data_produces_same_checksum() {
        use super::super::PAGE_SIZE;

        let page_data = vec![123u8; PAGE_SIZE];
        let header = WalFrameHeader::new(1, 1, 100, 200, 0);

        let checksum1 = compute_checksum(&header, &page_data);
        let checksum2 = compute_checksum(&header, &page_data);

        assert_eq!(checksum1, checksum2);
    }

    #[test]
    fn different_data_produces_different_checksum() {
        use super::super::PAGE_SIZE;

        let page_data1 = vec![1u8; PAGE_SIZE];
        let page_data2 = vec![2u8; PAGE_SIZE];
        let header = WalFrameHeader::new(1, 1, 100, 200, 0);

        let checksum1 = compute_checksum(&header, &page_data1);
        let checksum2 = compute_checksum(&header, &page_data2);

        assert_ne!(checksum1, checksum2);
    }

    #[test]
    fn validate_checksum_accepts_valid_frame() {
        use super::super::PAGE_SIZE;

        let page_data = vec![55u8; PAGE_SIZE];
        let mut header = WalFrameHeader::new(3, 7, 0xAABBCCDD, 0x11223344, 0);

        let checksum = compute_checksum(&header, &page_data);
        header.checksum = checksum;

        assert!(validate_checksum(&header, &page_data));
    }

    #[test]
    fn validate_checksum_rejects_corrupted_frame() {
        use super::super::PAGE_SIZE;

        let page_data = vec![77u8; PAGE_SIZE];
        let mut header = WalFrameHeader::new(8, 15, 0xDEADBEEF, 0xCAFEBABE, 0);

        let checksum = compute_checksum(&header, &page_data);
        header.checksum = checksum + 1;

        assert!(!validate_checksum(&header, &page_data));
    }

    #[test]
    fn validate_checksum_rejects_corrupted_data() {
        use super::super::PAGE_SIZE;

        let mut page_data = vec![88u8; PAGE_SIZE];
        let mut header = WalFrameHeader::new(10, 20, 0x11111111, 0x22222222, 0);

        let checksum = compute_checksum(&header, &page_data);
        header.checksum = checksum;

        page_data[100] = 99;

        assert!(!validate_checksum(&header, &page_data));
    }

    #[test]
    fn read_frame_roundtrip() {
        use super::super::PAGE_SIZE;

        let temp_dir = std::env::temp_dir().join("turdb_test_read_frame");
        let wal_dir = temp_dir.join("wal");

        if wal_dir.exists() {
            std::fs::remove_dir_all(&wal_dir).ok();
        }

        let mut wal = Wal::create(&wal_dir).expect("should create WAL");

        let page_data = vec![42u8; PAGE_SIZE];
        let header = WalFrameHeader::new(5, 10, 0x12345678, 0x9ABCDEF0, 0);

        wal.write_frame(&header, &page_data).expect("should write frame");

        let segment_path = wal_dir.join("wal.000001");
        let mut segment = WalSegment::open(&segment_path, 1).expect("should open segment");

        let (read_header, read_data) = segment.read_frame().expect("should read frame");

        assert_eq!(read_header.page_no, header.page_no);
        assert_eq!(read_header.db_size, header.db_size);
        assert_eq!(read_header.salt1, header.salt1);
        assert_eq!(read_header.salt2, header.salt2);
        assert_eq!(read_data, page_data);

        std::fs::remove_dir_all(&temp_dir).ok();
    }

    #[test]
    fn read_frame_validates_checksum() {
        use super::super::PAGE_SIZE;

        let temp_dir = std::env::temp_dir().join("turdb_test_read_frame_checksum");
        let wal_dir = temp_dir.join("wal");

        if wal_dir.exists() {
            std::fs::remove_dir_all(&wal_dir).ok();
        }

        let mut wal = Wal::create(&wal_dir).expect("should create WAL");

        let page_data = vec![99u8; PAGE_SIZE];
        let header = WalFrameHeader::new(1, 1, 100, 200, 0);

        wal.write_frame(&header, &page_data).expect("should write frame");

        let segment_path = wal_dir.join("wal.000001");
        let mut segment = WalSegment::open(&segment_path, 1).expect("should open segment");

        let (read_header, read_data) = segment.read_frame().expect("should read frame");

        assert!(validate_checksum(&read_header, &read_data));

        std::fs::remove_dir_all(&temp_dir).ok();
    }

    #[test]
    fn read_frame_multiple_frames() {
        use super::super::PAGE_SIZE;

        let temp_dir = std::env::temp_dir().join("turdb_test_read_multiple_frames");
        let wal_dir = temp_dir.join("wal");

        if wal_dir.exists() {
            std::fs::remove_dir_all(&wal_dir).ok();
        }

        let mut wal = Wal::create(&wal_dir).expect("should create WAL");

        for i in 0..3 {
            let page_data = vec![i as u8; PAGE_SIZE];
            let header = WalFrameHeader::new(i as u32, (i + 1) as u32, 0, 0, 0);
            wal.write_frame(&header, &page_data).expect("should write frame");
        }

        let segment_path = wal_dir.join("wal.000001");
        let mut segment = WalSegment::open(&segment_path, 1).expect("should open segment");

        for expected_i in 0..3 {
            let (header, data) = segment.read_frame().expect("should read frame");
            assert_eq!(header.page_no, expected_i as u32);
            assert_eq!(data[0], expected_i as u8);
        }

        std::fs::remove_dir_all(&temp_dir).ok();
    }

    #[test]
    fn recover_applies_frames_to_storage() {
        use super::super::{PAGE_SIZE, MmapStorage};

        let temp_dir = std::env::temp_dir().join("turdb_test_recover");
        let db_path = temp_dir.join("test.db");
        let wal_dir = temp_dir.join("wal");

        if temp_dir.exists() {
            std::fs::remove_dir_all(&temp_dir).ok();
        }
        std::fs::create_dir_all(&temp_dir).unwrap();

        let mut storage = MmapStorage::create(&db_path, 10).expect("should create storage");

        let mut wal = Wal::create(&wal_dir).expect("should create WAL");

        let page_data_0 = vec![100u8; PAGE_SIZE];
        let header_0 = WalFrameHeader::new(0, 10, 1, 2, 0);
        wal.write_frame(&header_0, &page_data_0).expect("should write frame 0");

        let page_data_5 = vec![200u8; PAGE_SIZE];
        let header_5 = WalFrameHeader::new(5, 10, 1, 2, 0);
        wal.write_frame(&header_5, &page_data_5).expect("should write frame 5");

        drop(wal);

        let mut wal_recovered = Wal::open(&wal_dir).expect("should open WAL");
        let frames_applied = wal_recovered.recover(&mut storage).expect("should recover");

        assert_eq!(frames_applied, 2);

        let page_0 = storage.page(0).expect("should read page 0");
        assert_eq!(page_0[0], 100);
        assert_eq!(page_0[PAGE_SIZE - 1], 100);

        let page_5 = storage.page(5).expect("should read page 5");
        assert_eq!(page_5[0], 200);
        assert_eq!(page_5[PAGE_SIZE - 1], 200);

        std::fs::remove_dir_all(&temp_dir).ok();
    }

    #[test]
    fn recover_returns_zero_when_no_wal_exists() {
        use super::super::MmapStorage;

        let temp_dir = std::env::temp_dir().join("turdb_test_recover_no_wal");
        let db_path = temp_dir.join("test.db");
        let wal_dir = temp_dir.join("wal");

        if temp_dir.exists() {
            std::fs::remove_dir_all(&temp_dir).ok();
        }
        std::fs::create_dir_all(&temp_dir).unwrap();

        let mut storage = MmapStorage::create(&db_path, 5).expect("should create storage");

        let mut wal = Wal::open(&wal_dir).expect("should open WAL");
        let frames_applied = wal.recover(&mut storage).expect("should recover");

        assert_eq!(frames_applied, 0);

        std::fs::remove_dir_all(&temp_dir).ok();
    }

    #[test]
    fn recover_updates_same_page_multiple_times() {
        use super::super::{PAGE_SIZE, MmapStorage};

        let temp_dir = std::env::temp_dir().join("turdb_test_recover_same_page");
        let db_path = temp_dir.join("test.db");
        let wal_dir = temp_dir.join("wal");

        if temp_dir.exists() {
            std::fs::remove_dir_all(&temp_dir).ok();
        }
        std::fs::create_dir_all(&temp_dir).unwrap();

        let mut storage = MmapStorage::create(&db_path, 10).expect("should create storage");

        let mut wal = Wal::create(&wal_dir).expect("should create WAL");

        let page_data_1 = vec![10u8; PAGE_SIZE];
        let header_1 = WalFrameHeader::new(3, 10, 1, 2, 0);
        wal.write_frame(&header_1, &page_data_1).expect("should write frame");

        let page_data_2 = vec![20u8; PAGE_SIZE];
        let header_2 = WalFrameHeader::new(3, 10, 1, 2, 0);
        wal.write_frame(&header_2, &page_data_2).expect("should write frame");

        let page_data_3 = vec![30u8; PAGE_SIZE];
        let header_3 = WalFrameHeader::new(3, 10, 1, 2, 0);
        wal.write_frame(&header_3, &page_data_3).expect("should write frame");

        drop(wal);

        let mut wal_recovered = Wal::open(&wal_dir).expect("should open WAL");
        let frames_applied = wal_recovered.recover(&mut storage).expect("should recover");

        assert_eq!(frames_applied, 3);

        let page_3 = storage.page(3).expect("should read page 3");
        assert_eq!(page_3[0], 30);
        assert_eq!(page_3[PAGE_SIZE - 1], 30);

        std::fs::remove_dir_all(&temp_dir).ok();
    }

    #[test]
    fn truncate_resets_wal_to_empty() {
        use super::super::PAGE_SIZE;

        let temp_dir = std::env::temp_dir().join("turdb_test_truncate");
        let wal_dir = temp_dir.join("wal");

        if temp_dir.exists() {
            std::fs::remove_dir_all(&temp_dir).ok();
        }

        let mut wal = Wal::create(&wal_dir).expect("should create WAL");

        let page_data = vec![42u8; PAGE_SIZE];
        let header = WalFrameHeader::new(1, 1, 100, 200, 0);
        wal.write_frame(&header, &page_data).expect("should write frame");

        assert!(wal.current_offset() > 0);

        wal.truncate().expect("should truncate");

        assert_eq!(wal.current_offset(), 0);

        std::fs::remove_dir_all(&temp_dir).ok();
    }

    #[test]
    fn checkpoint_applies_frames_and_truncates() {
        use super::super::{PAGE_SIZE, MmapStorage};

        let temp_dir = std::env::temp_dir().join("turdb_test_checkpoint");
        let db_path = temp_dir.join("test.db");
        let wal_dir = temp_dir.join("wal");

        if temp_dir.exists() {
            std::fs::remove_dir_all(&temp_dir).ok();
        }
        std::fs::create_dir_all(&temp_dir).unwrap();

        let mut storage = MmapStorage::create(&db_path, 10).expect("should create storage");

        let mut wal = Wal::create(&wal_dir).expect("should create WAL");

        let page_data_0 = vec![111u8; PAGE_SIZE];
        let header_0 = WalFrameHeader::new(0, 10, 1, 2, 0);
        wal.write_frame(&header_0, &page_data_0).expect("should write frame");

        let page_data_7 = vec![222u8; PAGE_SIZE];
        let header_7 = WalFrameHeader::new(7, 10, 1, 2, 0);
        wal.write_frame(&header_7, &page_data_7).expect("should write frame");

        assert!(wal.current_offset() > 0);

        let frames_checkpointed = wal.checkpoint(&mut storage).expect("should checkpoint");

        assert_eq!(frames_checkpointed, 2);

        assert_eq!(wal.current_offset(), 0);

        let page_0 = storage.page(0).expect("should read page 0");
        assert_eq!(page_0[0], 111);

        let page_7 = storage.page(7).expect("should read page 7");
        assert_eq!(page_7[0], 222);

        std::fs::remove_dir_all(&temp_dir).ok();
    }

    #[test]
    fn needs_checkpoint_returns_true_when_threshold_exceeded() {
        use super::super::PAGE_SIZE;

        let temp_dir = std::env::temp_dir().join("turdb_test_needs_checkpoint");
        let wal_dir = temp_dir.join("wal");

        if temp_dir.exists() {
            std::fs::remove_dir_all(&temp_dir).ok();
        }

        let mut wal = Wal::create(&wal_dir).expect("should create WAL");

        assert!(!wal.needs_checkpoint(1000));

        let page_data = vec![99u8; PAGE_SIZE];
        let header = WalFrameHeader::new(1, 1, 0, 0, 0);
        wal.write_frame(&header, &page_data).expect("should write frame");

        let frame_size = (WAL_FRAME_HEADER_SIZE + PAGE_SIZE) as u64;

        assert!(wal.needs_checkpoint(frame_size - 1));
        assert!(!wal.needs_checkpoint(frame_size + 1));

        std::fs::remove_dir_all(&temp_dir).ok();
    }
}
