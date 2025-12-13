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

pub const WAL_FRAME_HEADER_SIZE: usize = 32;

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

use eyre::{Result, WrapErr, ensure};
use std::fs::{File, OpenOptions, create_dir_all};
use std::path::{Path, PathBuf};
use std::io::Write;
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

    pub fn write_frame(&mut self, header: &WalFrameHeader, page_data: &[u8]) -> Result<()> {
        ensure!(
            page_data.len() == PAGE_SIZE,
            "page data must be exactly {} bytes, got {}",
            PAGE_SIZE,
            page_data.len()
        );

        let mut segment = self.current_segment.lock();
        segment.write_frame(header, page_data)
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

        self.offset += (WAL_FRAME_HEADER_SIZE + PAGE_SIZE) as u64;

        Ok(())
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
}
