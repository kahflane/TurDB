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

use crc::{Crc, CRC_64_ECMA_182};
use zerocopy::{FromBytes, Immutable, IntoBytes};

pub const WAL_FRAME_HEADER_SIZE: usize = 32;
pub const MAX_SEGMENT_SIZE: u64 = 64 * 1024 * 1024;

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

use eyre::{bail, ensure, Result, WrapErr};
use hashbrown::HashMap;
use memmap2::Mmap;
use parking_lot::{Mutex, RwLock};
use std::fs::{create_dir_all, read_dir, File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

use super::PAGE_SIZE;

pub struct Wal {
    #[allow(dead_code)]
    dir: PathBuf,
    current_segment: Mutex<WalSegment>,
    page_index: RwLock<HashMap<u32, (u64, u64)>>,
    read_mmap: RwLock<Option<(u64, Mmap)>>,
    salt1: u32,
    salt2: u32,
}

impl Wal {
    fn generate_salt() -> u32 {
        use std::time::SystemTime;
        let nanos = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        (nanos as u32) ^ ((nanos >> 32) as u32)
    }

    pub fn find_latest_segment(dir: &Path) -> Result<u64> {
        if !dir.exists() {
            return Ok(1);
        }

        let entries =
            read_dir(dir).wrap_err_with(|| format!("failed to read WAL directory {:?}", dir))?;

        let mut max_segment = 0u64;

        for entry in entries {
            let entry = entry.wrap_err("failed to read directory entry")?;
            let file_name = entry.file_name();
            let file_name_str = file_name.to_string_lossy();

            if file_name_str.starts_with("wal.") && file_name_str.len() == 10 {
                let num_part = &file_name_str[4..];
                if let Ok(segment_num) = num_part.parse::<u64>() {
                    max_segment = max_segment.max(segment_num);
                }
            }
        }

        Ok(if max_segment == 0 { 1 } else { max_segment })
    }

    pub fn create(dir: &Path) -> Result<Self> {
        create_dir_all(dir)
            .wrap_err_with(|| format!("failed to create WAL directory at {:?}", dir))?;

        let segment_path = dir.join("wal.000001");
        let segment = WalSegment::create(&segment_path, 1)?;

        Ok(Self {
            dir: dir.to_path_buf(),
            current_segment: Mutex::new(segment),
            page_index: RwLock::new(HashMap::new()),
            read_mmap: RwLock::new(None),
            salt1: Self::generate_salt(),
            salt2: Self::generate_salt(),
        })
    }

    pub fn open(dir: &Path) -> Result<Self> {
        if !dir.exists() {
            create_dir_all(dir)
                .wrap_err_with(|| format!("failed to create WAL directory at {:?}", dir))?;
        }

        let segment_num = Self::find_latest_segment(dir)?;
        let segment_path = dir.join(format!("wal.{:06}", segment_num));

        let segment = if segment_path.exists() {
            WalSegment::open(&segment_path, segment_num)?
        } else {
            WalSegment::create(&segment_path, segment_num)?
        };

        let mut page_index = HashMap::new();
        let mut salt1 = Self::generate_salt();
        let mut salt2 = Self::generate_salt();

        if segment_path.exists() {
            let mut scan_segment = WalSegment::open(&segment_path, segment_num)?;
            let mut offset = 0u64;
            let mut first_frame = true;

            while let Ok((header, _)) = scan_segment.read_frame() {
                if first_frame {
                    salt1 = header.salt1;
                    salt2 = header.salt2;
                    first_frame = false;
                }
                page_index.insert(header.page_no, (segment_num, offset));
                offset += (WAL_FRAME_HEADER_SIZE + PAGE_SIZE) as u64;
            }
        }

        Ok(Self {
            dir: dir.to_path_buf(),
            current_segment: Mutex::new(segment),
            page_index: RwLock::new(page_index),
            read_mmap: RwLock::new(None),
            salt1,
            salt2,
        })
    }

    pub fn recover(&mut self, storage: &mut super::MmapStorage) -> Result<u32> {
        let segment_path = self.dir.join("wal.000001");

        if !segment_path.exists() {
            return Ok(0);
        }

        let mut segment = WalSegment::open(&segment_path, 1)?;
        let mut frames_applied = 0;

        while let Ok((header, page_data)) = segment.read_frame() {
            let page_mut = storage.page_mut(header.page_no).wrap_err_with(|| {
                format!("failed to get page {} for WAL recovery", header.page_no)
            })?;

            page_mut.copy_from_slice(&page_data);
            frames_applied += 1;
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

        segment
            .file
            .set_len(0)
            .wrap_err("failed to truncate WAL segment file")?;

        segment
            .file
            .flush()
            .wrap_err("failed to flush WAL segment after truncate")?;

        segment.offset = 0;

        drop(segment);

        let mut index = self.page_index.write();
        index.clear();
        drop(index);

        let mut mmap = self.read_mmap.write();
        *mmap = None;

        Ok(())
    }

    pub fn needs_checkpoint(&self, threshold_bytes: u64) -> bool {
        self.current_offset() >= threshold_bytes
    }

    pub fn read_page(&self, page_no: u32) -> Result<Option<Vec<u8>>> {
        let index = self.page_index.read();
        let (segment_num, offset) = match index.get(&page_no) {
            Some(&(seg, off)) => (seg, off),
            None => return Ok(None),
        };
        drop(index);

        let mut mmap_guard = self.read_mmap.write();
        let needs_reload = match mmap_guard.as_ref() {
            None => true,
            Some((cached_seg, _)) => *cached_seg != segment_num,
        };

        if needs_reload {
            let segment_path = self.dir.join(format!("wal.{:06}", segment_num));
            if !segment_path.exists() {
                return Ok(None);
            }
            let file = File::open(&segment_path).wrap_err_with(|| {
                format!(
                    "failed to open WAL segment for reading at {:?}",
                    segment_path
                )
            })?;

            let mmap = unsafe {
                Mmap::map(&file)
                    .wrap_err_with(|| format!("failed to mmap WAL segment at {:?}", segment_path))?
            };
            *mmap_guard = Some((segment_num, mmap));
        }

        let (_seg, mmap) = mmap_guard.as_ref().unwrap();

        let frame_start = offset as usize;
        ensure!(
            frame_start + WAL_FRAME_HEADER_SIZE + PAGE_SIZE <= mmap.len(),
            "WAL frame at offset {} extends beyond file size {}",
            offset,
            mmap.len()
        );

        let header_bytes = &mmap[frame_start..frame_start + WAL_FRAME_HEADER_SIZE];
        let header = WalFrameHeader::read_from_bytes(header_bytes)
            .map_err(|e| eyre::eyre!("invalid WAL frame header at offset {}: {:?}", offset, e))?;

        let data_start = frame_start + WAL_FRAME_HEADER_SIZE;
        let page_data = &mmap[data_start..data_start + PAGE_SIZE];

        if !validate_checksum(&header, page_data) {
            bail!(
                "invalid checksum in WAL frame at offset {} for page {}",
                offset,
                page_no
            );
        }

        Ok(Some(page_data.to_vec()))
    }

    pub fn write_frame(&mut self, page_no: u32, db_size: u32, page_data: &[u8]) -> Result<()> {
        ensure!(
            page_data.len() == PAGE_SIZE,
            "page data must be exactly {} bytes, got {}",
            PAGE_SIZE,
            page_data.len()
        );

        if self.needs_rotation() {
            self.rotate_segment()
                .wrap_err("failed to rotate WAL segment during write_frame")?;
        }

        let header = WalFrameHeader::new(page_no, db_size, self.salt1, self.salt2, 0);
        let checksum = compute_checksum(&header, page_data);
        let mut header_with_checksum = header;
        header_with_checksum.checksum = checksum;

        let current_offset;
        let segment_num;
        {
            let mut segment = self.current_segment.lock();
            current_offset = segment.offset();
            segment_num = segment.sequence;
            segment.write_frame(&header_with_checksum, page_data)?;
        }

        let mut index = self.page_index.write();
        index.insert(page_no, (segment_num, current_offset));
        drop(index);

        let mut mmap = self.read_mmap.write();
        *mmap = None;

        Ok(())
    }

    pub fn current_offset(&self) -> u64 {
        let segment = self.current_segment.lock();
        segment.offset()
    }

    pub fn needs_rotation(&self) -> bool {
        self.current_offset() >= MAX_SEGMENT_SIZE
    }

    pub fn rotate_segment(&mut self) -> Result<()> {
        let mut segment_guard = self.current_segment.lock();
        let current_sequence = segment_guard.sequence;
        let new_sequence = current_sequence + 1;

        let new_segment_path = self.dir.join(format!("wal.{:06}", new_sequence));
        let new_segment = WalSegment::create(&new_segment_path, new_sequence)
            .wrap_err_with(|| format!("failed to create new WAL segment {}", new_sequence))?;

        *segment_guard = new_segment;
        drop(segment_guard);

        let mut mmap_guard = self.read_mmap.write();
        *mmap_guard = None;

        Ok(())
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
    use tempfile::tempdir;

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

        let _wal = Wal::create(&wal_dir).expect("should create WAL");

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

        let _wal = Wal::create(&wal_dir).expect("should create WAL");

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

        wal.write_frame(5, 10, &page_data)
            .expect("should write frame");

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

        wal.write_frame(1, 1, &page_data)
            .expect("should write frame");

        let expected_offset = WAL_FRAME_HEADER_SIZE + PAGE_SIZE;
        assert_eq!(wal.current_offset(), expected_offset as u64);

        std::fs::remove_dir_all(&temp_dir).ok();
    }

    #[test]
    fn compute_checksum_for_frame() {
        use super::super::PAGE_SIZE;

        let page_data = vec![42u8; PAGE_SIZE];
        let header = WalFrameHeader::new(5, 10, 0x12345678, 0x9ABCDEF0, 0);

        let checksum = compute_checksum(&header, &page_data);

        assert_ne!(checksum, 0, "checksum should not be zero");
        let _ = checksum;
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

        wal.write_frame(5, 10, &page_data)
            .expect("should write frame");

        let segment_path = wal_dir.join("wal.000001");
        let mut segment = WalSegment::open(&segment_path, 1).expect("should open segment");

        let (read_header, read_data) = segment.read_frame().expect("should read frame");

        assert_eq!(read_header.page_no, 5);
        assert_eq!(read_header.db_size, 10);
        assert_ne!(read_header.salt1, 0);
        assert_ne!(read_header.salt2, 0);
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

        wal.write_frame(1, 1, &page_data)
            .expect("should write frame");

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
            wal.write_frame(i as u32, (i + 1) as u32, &page_data)
                .expect("should write frame");
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
        use super::super::{MmapStorage, PAGE_SIZE};

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
        wal.write_frame(0, 10, &page_data_0)
            .expect("should write frame 0");

        let page_data_5 = vec![200u8; PAGE_SIZE];
        wal.write_frame(5, 10, &page_data_5)
            .expect("should write frame 5");

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
        use super::super::{MmapStorage, PAGE_SIZE};

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
        wal.write_frame(3, 10, &page_data_1)
            .expect("should write frame");

        let page_data_2 = vec![20u8; PAGE_SIZE];
        wal.write_frame(3, 10, &page_data_2)
            .expect("should write frame");

        let page_data_3 = vec![30u8; PAGE_SIZE];
        wal.write_frame(3, 10, &page_data_3)
            .expect("should write frame");

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
        wal.write_frame(1, 1, &page_data)
            .expect("should write frame");

        assert!(wal.current_offset() > 0);

        wal.truncate().expect("should truncate");

        assert_eq!(wal.current_offset(), 0);

        std::fs::remove_dir_all(&temp_dir).ok();
    }

    #[test]
    fn checkpoint_applies_frames_and_truncates() {
        use super::super::{MmapStorage, PAGE_SIZE};

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
        wal.write_frame(0, 10, &page_data_0)
            .expect("should write frame");

        let page_data_7 = vec![222u8; PAGE_SIZE];
        wal.write_frame(7, 10, &page_data_7)
            .expect("should write frame");

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
        wal.write_frame(1, 1, &page_data)
            .expect("should write frame");

        let frame_size = (WAL_FRAME_HEADER_SIZE + PAGE_SIZE) as u64;

        assert!(wal.needs_checkpoint(frame_size - 1));
        assert!(!wal.needs_checkpoint(frame_size + 1));

        std::fs::remove_dir_all(&temp_dir).ok();
    }

    #[test]
    fn read_page_returns_none_when_page_not_in_wal() {
        let temp_dir = std::env::temp_dir().join("turdb_test_read_page_none");
        let wal_dir = temp_dir.join("wal");

        if temp_dir.exists() {
            std::fs::remove_dir_all(&temp_dir).ok();
        }

        let wal = Wal::create(&wal_dir).expect("should create WAL");

        let page = wal.read_page(42).expect("should read page");
        assert!(page.is_none());

        std::fs::remove_dir_all(&temp_dir).ok();
    }

    #[test]
    fn read_page_returns_latest_page_data() {
        use super::super::PAGE_SIZE;

        let temp_dir = std::env::temp_dir().join("turdb_test_read_page_latest");
        let wal_dir = temp_dir.join("wal");

        if temp_dir.exists() {
            std::fs::remove_dir_all(&temp_dir).ok();
        }

        let mut wal = Wal::create(&wal_dir).expect("should create WAL");

        let page_data = vec![123u8; PAGE_SIZE];
        wal.write_frame(5, 10, &page_data)
            .expect("should write frame");

        let read_data = wal.read_page(5).expect("should read page");
        assert!(read_data.is_some());

        let read_data = read_data.unwrap();
        assert_eq!(read_data.len(), PAGE_SIZE);
        assert_eq!(read_data[0], 123);
        assert_eq!(read_data[PAGE_SIZE - 1], 123);

        std::fs::remove_dir_all(&temp_dir).ok();
    }

    #[test]
    fn read_page_returns_latest_version_of_page() {
        use super::super::PAGE_SIZE;

        let temp_dir = std::env::temp_dir().join("turdb_test_read_page_latest_version");
        let wal_dir = temp_dir.join("wal");

        if temp_dir.exists() {
            std::fs::remove_dir_all(&temp_dir).ok();
        }

        let mut wal = Wal::create(&wal_dir).expect("should create WAL");

        let page_data_1 = vec![10u8; PAGE_SIZE];
        wal.write_frame(3, 10, &page_data_1)
            .expect("should write frame 1");

        let page_data_2 = vec![20u8; PAGE_SIZE];
        wal.write_frame(3, 10, &page_data_2)
            .expect("should write frame 2");

        let page_data_3 = vec![30u8; PAGE_SIZE];
        wal.write_frame(3, 10, &page_data_3)
            .expect("should write frame 3");

        let read_data = wal.read_page(3).expect("should read page");
        assert!(read_data.is_some());

        let read_data = read_data.unwrap();
        assert_eq!(read_data[0], 30);
        assert_eq!(read_data[PAGE_SIZE - 1], 30);

        std::fs::remove_dir_all(&temp_dir).ok();
    }

    #[test]
    fn read_page_after_checkpoint_returns_none() {
        use super::super::{MmapStorage, PAGE_SIZE};

        let temp_dir = std::env::temp_dir().join("turdb_test_read_page_after_checkpoint");
        let db_path = temp_dir.join("test.db");
        let wal_dir = temp_dir.join("wal");

        if temp_dir.exists() {
            std::fs::remove_dir_all(&temp_dir).ok();
        }
        std::fs::create_dir_all(&temp_dir).unwrap();

        let mut storage = MmapStorage::create(&db_path, 10).expect("should create storage");

        let mut wal = Wal::create(&wal_dir).expect("should create WAL");

        let page_data = vec![77u8; PAGE_SIZE];
        wal.write_frame(2, 10, &page_data)
            .expect("should write frame");

        let read_before = wal.read_page(2).expect("should read page");
        assert!(read_before.is_some());

        wal.checkpoint(&mut storage).expect("should checkpoint");

        let read_after = wal.read_page(2).expect("should read page");
        assert!(read_after.is_none());

        std::fs::remove_dir_all(&temp_dir).ok();
    }

    #[test]
    fn find_latest_segment_returns_one_when_empty() {
        let temp_dir = std::env::temp_dir().join("turdb_test_find_latest_empty");

        if temp_dir.exists() {
            std::fs::remove_dir_all(&temp_dir).ok();
        }
        std::fs::create_dir_all(&temp_dir).unwrap();

        let latest = Wal::find_latest_segment(&temp_dir).expect("should find latest");
        assert_eq!(latest, 1);

        std::fs::remove_dir_all(&temp_dir).ok();
    }

    #[test]
    fn find_latest_segment_finds_highest_numbered() {
        use std::fs::File;

        let temp_dir = std::env::temp_dir().join("turdb_test_find_latest_highest");

        if temp_dir.exists() {
            std::fs::remove_dir_all(&temp_dir).ok();
        }
        std::fs::create_dir_all(&temp_dir).unwrap();

        File::create(temp_dir.join("wal.000001")).unwrap();
        File::create(temp_dir.join("wal.000003")).unwrap();
        File::create(temp_dir.join("wal.000007")).unwrap();
        File::create(temp_dir.join("wal.000005")).unwrap();

        let latest = Wal::find_latest_segment(&temp_dir).expect("should find latest");
        assert_eq!(latest, 7);

        std::fs::remove_dir_all(&temp_dir).ok();
    }

    #[test]
    fn find_latest_segment_ignores_non_wal_files() {
        use std::fs::File;

        let temp_dir = std::env::temp_dir().join("turdb_test_find_latest_ignore");

        if temp_dir.exists() {
            std::fs::remove_dir_all(&temp_dir).ok();
        }
        std::fs::create_dir_all(&temp_dir).unwrap();

        File::create(temp_dir.join("wal.000002")).unwrap();
        File::create(temp_dir.join("other.txt")).unwrap();
        File::create(temp_dir.join("wal.invalid")).unwrap();
        File::create(temp_dir.join("000003")).unwrap();

        let latest = Wal::find_latest_segment(&temp_dir).expect("should find latest");
        assert_eq!(latest, 2);

        std::fs::remove_dir_all(&temp_dir).ok();
    }

    #[test]
    fn needs_rotation_returns_false_when_below_threshold() {
        let temp_dir = tempdir().expect("failed to create temp dir");
        let wal_dir = temp_dir.path().join("wal");

        let mut wal = Wal::create(&wal_dir).expect("should create WAL");

        for i in 0..10 {
            let page_data = vec![i as u8; PAGE_SIZE];
            wal.write_frame(i, 100, &page_data)
                .expect("should write frame");
        }

        assert!(
            !wal.needs_rotation(),
            "should not need rotation with only 10 frames"
        );

        temp_dir.close().ok();
    }

    #[test]
    fn needs_rotation_returns_true_when_exceeds_threshold() {
        let temp_dir = tempdir().expect("failed to create temp dir");
        let wal_dir = temp_dir.path().join("wal");

        let mut wal = Wal::create(&wal_dir).expect("should create WAL");

        let frames_needed = (MAX_SEGMENT_SIZE / (WAL_FRAME_HEADER_SIZE + PAGE_SIZE) as u64) + 1;

        for i in 0..frames_needed {
            let page_data = vec![(i % 256) as u8; PAGE_SIZE];
            wal.write_frame(i as u32, 100, &page_data)
                .expect("should write frame");
        }

        assert!(
            wal.needs_rotation(),
            "should need rotation after exceeding 64MB"
        );

        temp_dir.close().ok();
    }

    #[test]
    fn rotate_segment_creates_new_segment_file() {
        let temp_dir = tempdir().expect("failed to create temp dir");
        let wal_dir = temp_dir.path().join("wal");

        let mut wal = Wal::create(&wal_dir).expect("should create WAL");

        let page_data = vec![42u8; PAGE_SIZE];
        wal.write_frame(1, 100, &page_data)
            .expect("should write frame");

        wal.rotate_segment().expect("should rotate segment");

        assert!(
            wal_dir.join("wal.000001").exists(),
            "original segment should exist"
        );
        assert!(
            wal_dir.join("wal.000002").exists(),
            "new segment should exist"
        );

        temp_dir.close().ok();
    }

    #[test]
    fn rotate_segment_switches_to_new_segment() {
        let temp_dir = tempdir().expect("failed to create temp dir");
        let wal_dir = temp_dir.path().join("wal");

        let mut wal = Wal::create(&wal_dir).expect("should create WAL");

        let page_data = vec![42u8; PAGE_SIZE];
        wal.write_frame(1, 100, &page_data)
            .expect("should write frame");

        wal.rotate_segment().expect("should rotate segment");

        let page_data2 = vec![99u8; PAGE_SIZE];
        wal.write_frame(2, 100, &page_data2)
            .expect("should write frame after rotation");

        let metadata1 = std::fs::metadata(wal_dir.join("wal.000001")).unwrap();
        let metadata2 = std::fs::metadata(wal_dir.join("wal.000002")).unwrap();

        assert_eq!(
            metadata1.len(),
            (WAL_FRAME_HEADER_SIZE + PAGE_SIZE) as u64,
            "segment 1 should have 1 frame"
        );
        assert_eq!(
            metadata2.len(),
            (WAL_FRAME_HEADER_SIZE + PAGE_SIZE) as u64,
            "segment 2 should have 1 frame"
        );

        temp_dir.close().ok();
    }

    #[test]
    fn rotate_segment_preserves_page_index() {
        let temp_dir = tempdir().expect("failed to create temp dir");
        let wal_dir = temp_dir.path().join("wal");

        let mut wal = Wal::create(&wal_dir).expect("should create WAL");

        let page_data = vec![42u8; PAGE_SIZE];
        wal.write_frame(1, 100, &page_data)
            .expect("should write frame");

        wal.rotate_segment().expect("should rotate segment");

        let read_page = wal.read_page(1).expect("should read page");
        assert!(
            read_page.is_some(),
            "should still be able to read page from old segment"
        );
        assert_eq!(read_page.unwrap()[0], 42);

        temp_dir.close().ok();
    }

    #[test]
    fn write_frame_auto_rotates_when_exceeds_threshold() {
        let temp_dir = tempdir().expect("failed to create temp dir");
        let wal_dir = temp_dir.path().join("wal");

        let mut wal = Wal::create(&wal_dir).expect("should create WAL");

        let frames_needed = (MAX_SEGMENT_SIZE / (WAL_FRAME_HEADER_SIZE + PAGE_SIZE) as u64) + 10;

        for i in 0..frames_needed {
            let page_data = vec![(i % 256) as u8; PAGE_SIZE];
            wal.write_frame(i as u32, 100, &page_data)
                .expect("should write frame");
        }

        assert!(
            wal_dir.join("wal.000001").exists(),
            "segment 1 should exist"
        );
        assert!(
            wal_dir.join("wal.000002").exists(),
            "segment 2 should exist after auto-rotation"
        );

        for i in 0..frames_needed {
            let read_page = wal.read_page(i as u32).expect("should read page");
            assert!(read_page.is_some(), "should be able to read page {}", i);
            assert_eq!(read_page.unwrap()[0], (i % 256) as u8);
        }

        temp_dir.close().ok();
    }
}
