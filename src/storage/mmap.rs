//! # Memory-Mapped File Storage
//!
//! This module implements `MmapStorage`, a low-level building block for memory-mapped
//! database file access. It provides zero-copy page access with compile-time
//! safety guarantees through Rust's borrow checker.
//!
//! ## Internal Component
//!
//! `MmapStorage` is an internal component used by `FileManager` to manage individual
//! database files (`.tbd`, `.idx`, `.hnsw`). Users should not create `MmapStorage`
//! instances directly; instead use the higher-level `Database` API which manages
//! the file-per-table directory structure automatically.
//!
//! ## Design Philosophy
//!
//! Traditional database systems copy page data between kernel buffers and
//! user-space page caches. Memory-mapped I/O eliminates this copy by mapping
//! the file directly into the process address space. The OS handles paging
//! transparently, leveraging its existing page cache infrastructure.
//!
//! ## Safety Considerations
//!
//! Memory-mapped regions become invalid when remapped (during `grow()`). The
//! typical solutions involve runtime overhead:
//!
//! - **Hazard pointers**: Defer unmapping until no readers
//! - **Epoch-based reclamation**: Track read epochs
//! - **Reference counting**: Arc-wrapped regions
//!
//! TurDB instead leverages Rust's borrow checker:
//!
//! ```text
//! page(&self) -> &[u8]      // Immutable borrow of self
//! page_mut(&mut self) -> &mut [u8]  // Mutable borrow of self
//! grow(&mut self)           // Mutable borrow (exclusive)
//! ```
//!
//! Since `grow()` requires `&mut self`, the compiler ensures no page references
//! exist when grow is called. This provides:
//!
//! - **Zero runtime overhead**: No locks, guards, or epoch tracking
//! - **Compile-time safety**: Dangling pointer bugs are caught by rustc
//! - **Idiomatic Rust**: Works with standard borrow semantics
//!
//! ## Page Layout
//!
//! Each page is exactly 16KB (16384 bytes):
//!
//! ```text
//! +---------------------------+
//! |    Header (128 bytes)     |
//! +---------------------------+
//! |                           |
//! |    Usable Space           |
//! |    (16256 bytes)          |
//! |                           |
//! +---------------------------+
//! ```
//!
//! The large header reservation (vs SQLite's ~100 bytes) provides room for:
//! - Extended page metadata
//! - Checksums for corruption detection
//! - Future format extensions
//!
//! ## File Format
//!
//! Database files are simply concatenated pages:
//!
//! ```text
//! Offset 0:        Page 0 (16KB)
//! Offset 16384:    Page 1 (16KB)
//! Offset 32768:    Page 2 (16KB)
//! ...
//! ```
//!
//! Page 0 typically contains file header metadata. The file size must always
//! be a multiple of PAGE_SIZE.
//!
//! ## Platform Behavior
//!
//! ### Linux/macOS
//! - Uses `mmap()` with `MAP_SHARED` for writes to persist
//! - `msync()` ensures durability before returning from `sync()`
//! - Page faults may block on disk I/O
//!
//! ### Windows
//! - Uses `CreateFileMapping` / `MapViewOfFile`
//! - `FlushViewOfFile` + `FlushFileBuffers` for durability
//!
//! ## Error Handling
//!
//! All fallible operations return `eyre::Result` with rich context:
//! - File path and operation being performed
//! - Page numbers for out-of-bounds access
//! - Underlying OS error details

use std::fs::{File, OpenOptions};
use std::path::Path;

use eyre::{ensure, Result, WrapErr};
use memmap2::MmapMut;

use super::PAGE_SIZE;

#[derive(Debug)]
pub struct MmapStorage {
    file: File,
    mmap: MmapMut,
    page_count: u32,
}

impl MmapStorage {
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self> {
        let path = path.as_ref();

        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .open(path)
            .wrap_err_with(|| format!("failed to open database file '{}'", path.display()))?;

        let metadata = file
            .metadata()
            .wrap_err_with(|| format!("failed to get metadata for '{}'", path.display()))?;

        let file_size = metadata.len();

        ensure!(
            file_size > 0,
            "cannot open empty database file '{}'",
            path.display()
        );

        ensure!(
            file_size % PAGE_SIZE as u64 == 0,
            "database file '{}' size {} is not a multiple of page size {}",
            path.display(),
            file_size,
            PAGE_SIZE
        );

        let page_count = (file_size / PAGE_SIZE as u64) as u32;

        // SAFETY: MmapMut::map_mut is unsafe because memory-mapped files can be
        // modified externally, leading to undefined behavior. This is safe because:
        // 1. The file is opened with exclusive write access (read+write mode)
        // 2. Database files are not meant to be modified by external processes
        // 3. The mmap lifetime is tied to MmapStorage, preventing use-after-unmap
        // 4. All access goes through page()/page_mut() which bounds-check page_no
        let mmap = unsafe {
            MmapMut::map_mut(&file)
                .wrap_err_with(|| format!("failed to memory-map '{}'", path.display()))?
        };

        Ok(Self {
            file,
            mmap,
            page_count,
        })
    }

    pub fn create<P: AsRef<Path>>(path: P, initial_page_count: u32) -> Result<Self> {
        let path = path.as_ref();

        ensure!(
            initial_page_count > 0,
            "initial page count must be at least 1"
        );

        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(true)
            .open(path)
            .wrap_err_with(|| format!("failed to create database file '{}'", path.display()))?;

        let file_size = initial_page_count as u64 * PAGE_SIZE as u64;

        file.set_len(file_size)
            .wrap_err_with(|| format!("failed to set file size to {} bytes", file_size))?;

        // SAFETY: MmapMut::map_mut is unsafe because memory-mapped files can be
        // modified externally. This is safe because:
        // 1. We just created this file with exclusive access (truncate=true)
        // 2. The file size is set to a valid multiple of PAGE_SIZE
        // 3. The mmap lifetime is tied to MmapStorage, preventing use-after-unmap
        // 4. All access goes through page()/page_mut() which bounds-check page_no
        let mmap = unsafe {
            MmapMut::map_mut(&file)
                .wrap_err_with(|| format!("failed to memory-map '{}'", path.display()))?
        };

        Ok(Self {
            file,
            mmap,
            page_count: initial_page_count,
        })
    }

    pub fn page(&self, page_no: u32) -> Result<&[u8]> {
        ensure!(
            page_no < self.page_count,
            "page {} out of bounds (page_count={})",
            page_no,
            self.page_count
        );

        let offset = page_no as usize * PAGE_SIZE;
        Ok(&self.mmap[offset..offset + PAGE_SIZE])
    }

    pub fn page_mut(&mut self, page_no: u32) -> Result<&mut [u8]> {
        ensure!(
            page_no < self.page_count,
            "page {} out of bounds (page_count={})",
            page_no,
            self.page_count
        );

        let offset = page_no as usize * PAGE_SIZE;
        Ok(&mut self.mmap[offset..offset + PAGE_SIZE])
    }

    pub fn grow(&mut self, new_page_count: u32) -> Result<()> {
        if new_page_count <= self.page_count {
            return Ok(());
        }

        self.mmap
            .flush()
            .wrap_err("failed to flush mmap before grow")?;

        let new_size = new_page_count as u64 * PAGE_SIZE as u64;

        self.file
            .set_len(new_size)
            .wrap_err_with(|| format!("failed to extend file to {} bytes", new_size))?;

        // SAFETY: MmapMut::map_mut is unsafe because the old mmap becomes invalid.
        // This is safe because:
        // 1. grow() requires &mut self, so no page references can exist (borrow checker)
        // 2. We flushed the old mmap above, ensuring data is written to disk
        // 3. The file was extended to new_size before remapping
        // 4. The old mmap is dropped when we assign the new one
        self.mmap =
            unsafe { MmapMut::map_mut(&self.file).wrap_err("failed to remap file after grow")? };

        self.page_count = new_page_count;

        Ok(())
    }

    pub fn sync(&self) -> Result<()> {
        self.mmap.flush().wrap_err("failed to sync mmap to disk")
    }

    pub fn page_count(&self) -> u32 {
        self.page_count
    }

    pub fn file_size(&self) -> u64 {
        self.page_count as u64 * PAGE_SIZE as u64
    }

    pub fn prefetch_pages(&self, start_page: u32, count: u32) {
        if start_page >= self.page_count {
            return;
        }

        let end_page = (start_page + count).min(self.page_count);
        let start_offset = start_page as usize * PAGE_SIZE;
        let len = (end_page - start_page) as usize * PAGE_SIZE;

        #[cfg(unix)]
        // SAFETY: madvise with MADV_WILLNEED is a hint to the kernel and does not
        // cause undefined behavior even if the memory region is invalid. However,
        // this is safe because:
        // 1. start_page was bounds-checked above (start_page >= self.page_count returns early)
        // 2. end_page is clamped to self.page_count, so we never exceed the mmap bounds
        // 3. start_offset + len is at most self.page_count * PAGE_SIZE = file_size
        // 4. The mmap is valid for the entire file size
        unsafe {
            libc::madvise(
                self.mmap.as_ptr().add(start_offset) as *mut libc::c_void,
                len,
                libc::MADV_WILLNEED,
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn create_new_database() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.db");

        let storage = MmapStorage::create(&path, 10).unwrap();

        assert_eq!(storage.page_count(), 10);
        assert_eq!(storage.file_size(), 10 * PAGE_SIZE as u64);
    }

    #[test]
    fn create_fails_with_zero_pages() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.db");

        let result = MmapStorage::create(&path, 0);

        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("initial page count must be at least 1"));
    }

    #[test]
    fn open_existing_database() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.db");

        {
            let mut storage = MmapStorage::create(&path, 5).unwrap();
            let page = storage.page_mut(0).unwrap();
            page[0] = 0xAB;
            storage.sync().unwrap();
        }

        let storage = MmapStorage::open(&path).unwrap();

        assert_eq!(storage.page_count(), 5);
        assert_eq!(storage.page(0).unwrap()[0], 0xAB);
    }

    #[test]
    fn open_fails_for_nonexistent_file() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("nonexistent.db");

        let result = MmapStorage::open(&path);

        assert!(result.is_err());
    }

    #[test]
    fn page_returns_correct_slice() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.db");

        let mut storage = MmapStorage::create(&path, 3).unwrap();

        {
            let page0 = storage.page_mut(0).unwrap();
            page0[0] = 0x01;
            page0[PAGE_SIZE - 1] = 0x11;
        }

        {
            let page1 = storage.page_mut(1).unwrap();
            page1[0] = 0x02;
            page1[PAGE_SIZE - 1] = 0x22;
        }

        {
            let page2 = storage.page_mut(2).unwrap();
            page2[0] = 0x03;
            page2[PAGE_SIZE - 1] = 0x33;
        }

        let page0 = storage.page(0).unwrap();
        let page1 = storage.page(1).unwrap();
        let page2 = storage.page(2).unwrap();

        assert_eq!(page0.len(), PAGE_SIZE);
        assert_eq!(page0[0], 0x01);
        assert_eq!(page0[PAGE_SIZE - 1], 0x11);

        assert_eq!(page1.len(), PAGE_SIZE);
        assert_eq!(page1[0], 0x02);
        assert_eq!(page1[PAGE_SIZE - 1], 0x22);

        assert_eq!(page2.len(), PAGE_SIZE);
        assert_eq!(page2[0], 0x03);
        assert_eq!(page2[PAGE_SIZE - 1], 0x33);
    }

    #[test]
    fn page_out_of_bounds() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.db");

        let storage = MmapStorage::create(&path, 5).unwrap();

        assert!(storage.page(4).is_ok());
        assert!(storage.page(5).is_err());
        assert!(storage.page(100).is_err());
    }

    #[test]
    fn page_mut_modifies_storage() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.db");

        let mut storage = MmapStorage::create(&path, 2).unwrap();

        {
            let page = storage.page_mut(1).unwrap();
            page[100] = 0xDE;
            page[101] = 0xAD;
        }

        let page = storage.page(1).unwrap();
        assert_eq!(page[100], 0xDE);
        assert_eq!(page[101], 0xAD);
    }

    #[test]
    fn grow_extends_file() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.db");

        let mut storage = MmapStorage::create(&path, 5).unwrap();

        assert_eq!(storage.page_count(), 5);

        storage.grow(10).unwrap();

        assert_eq!(storage.page_count(), 10);
        assert_eq!(storage.file_size(), 10 * PAGE_SIZE as u64);

        assert!(storage.page(9).is_ok());
    }

    #[test]
    fn grow_preserves_existing_data() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.db");

        let mut storage = MmapStorage::create(&path, 3).unwrap();

        {
            let page = storage.page_mut(2).unwrap();
            page[0] = 0xCA;
            page[1] = 0xFE;
        }

        storage.grow(10).unwrap();

        let page = storage.page(2).unwrap();
        assert_eq!(page[0], 0xCA);
        assert_eq!(page[1], 0xFE);
    }

    #[test]
    fn grow_with_same_size_is_noop() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.db");

        let mut storage = MmapStorage::create(&path, 5).unwrap();

        storage.grow(5).unwrap();
        storage.grow(3).unwrap();

        assert_eq!(storage.page_count(), 5);
    }

    #[test]
    fn sync_persists_changes() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.db");

        {
            let mut storage = MmapStorage::create(&path, 2).unwrap();
            let page = storage.page_mut(0).unwrap();
            page[50] = 0xBE;
            page[51] = 0xEF;
            storage.sync().unwrap();
        }

        let storage = MmapStorage::open(&path).unwrap();
        let page = storage.page(0).unwrap();

        assert_eq!(page[50], 0xBE);
        assert_eq!(page[51], 0xEF);
    }

    #[test]
    fn zero_copy_page_access() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.db");

        let storage = MmapStorage::create(&path, 1).unwrap();

        let page1 = storage.page(0).unwrap();
        let page2 = storage.page(0).unwrap();

        assert_eq!(page1.as_ptr(), page2.as_ptr());
    }
}
