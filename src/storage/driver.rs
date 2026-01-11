//! # Storage Driver Abstraction Layer
//!
//! This module provides the `StorageDriver` trait, a copy-based abstraction for storage
//! backends that enables TurDB to run on different platforms (native via mmap, browser via OPFS).
//!
//! ## Design Philosophy
//!
//! The storage abstraction follows Option D from the architecture design: PageCache as
//! the abstraction layer. This approach provides:
//!
//! - **Minimal changes to B-tree/cursor code**: They continue receiving `&[u8]` slices
//! - **Simple backend implementations**: Just read/write bytes, no complex lifetime management
//! - **Zero-copy preservation**: Via `supports_zero_copy()` + `page_direct()` bypass for mmap
//!
//! ## Copy-Based Interface
//!
//! The primary interface uses copy semantics for maximum portability:
//!
//! ```text
//! fn read_page(&self, page_no: u32, buf: &mut [u8; PAGE_SIZE]) -> Result<()>;
//! fn write_page(&mut self, page_no: u32, data: &[u8; PAGE_SIZE]) -> Result<()>;
//! ```
//!
//! This works efficiently with:
//! - OPFS FileSystemSyncAccessHandle (WASM)
//! - Traditional file I/O
//! - In-memory storage for testing
//!
//! ## Zero-Copy Bypass
//!
//! For mmap-based storage on native platforms, the `page_direct()` methods bypass the
//! copy interface entirely, returning slices directly into the mmap region:
//!
//! ```text
//! fn supports_zero_copy(&self) -> bool { true }
//! fn page_direct(&self, page_no: u32) -> Option<Result<&[u8]>> { Some(self.page(page_no)) }
//! ```
//!
//! PageCache checks `supports_zero_copy()` and uses direct access when available,
//! falling back to copy-based access otherwise.
//!
//! ## Storage Backends
//!
//! | Backend     | Platform   | Zero-Copy | Sync Required |
//! |-------------|------------|-----------|---------------|
//! | MmapStorage | Native     | Yes       | msync         |
//! | OpfsStorage | WASM/Worker| No        | flush         |
//!
//! ## StorageKind Enum
//!
//! `StorageKind` configures which backend to use at database open time:
//!
//! ```ignore
//! let db = Database::builder()
//!     .storage_kind(StorageKind::Mmap { path: "./mydb".into() })
//!     .open()?;
//!
//! // Or for WASM:
//! let db = Database::builder()
//!     .storage_kind(StorageKind::Opfs { name: "my-database".into() })
//!     .open()?;
//! ```
//!
//! ## AnyStorage Enum
//!
//! `AnyStorage` is a type-erased wrapper that implements `StorageDriver`, allowing
//! FileManager to work with any backend without generics:
//!
//! ```text
//! pub enum AnyStorage {
//!     Mmap(MmapStorage),
//!     #[cfg(target_arch = "wasm32")]
//!     Opfs(OpfsStorage),
//! }
//! ```
//!
//! ## Thread Safety
//!
//! `StorageDriver` requires `Send + Sync` bounds for thread-safe database operations.
//! Individual implementations may have additional safety considerations:
//!
//! - `MmapStorage`: Send but not Sync (wrapped in RwLock at FileManager level)
//! - `OpfsStorage`: Worker-only, single-threaded access
//!
//! ## Platform Considerations
//!
//! ### Native (Linux, macOS, Windows)
//! Uses `MmapStorage` with memory-mapped files for zero-copy page access.
//! WAL is fully supported for durability.
//!
//! ### WASM (Browser Worker)
//! Uses `OpfsStorage` with FileSystemSyncAccessHandle. Requires dedicated Web Worker
//! context for synchronous API access. WAL is disabled (OPFS provides its own
//! durability guarantees).

use eyre::Result;
use std::path::PathBuf;

use super::mmap::MmapStorage;
use super::{Storage, PAGE_SIZE};

/// Storage driver trait providing copy-based page access with optional zero-copy bypass.
///
/// This is the primary abstraction for storage backends, enabling TurDB to run on
/// different platforms while preserving zero-copy semantics where available.
pub trait StorageDriver: Send {
    /// Reads a page into the provided buffer.
    ///
    /// This is the primary read interface, suitable for all storage backends.
    /// For zero-copy access on mmap, use `page_direct()` instead.
    fn read_page(&self, page_no: u32, buf: &mut [u8; PAGE_SIZE]) -> Result<()>;

    /// Writes a page from the provided buffer.
    ///
    /// Changes may be buffered until `sync()` is called.
    fn write_page(&mut self, page_no: u32, data: &[u8; PAGE_SIZE]) -> Result<()>;

    /// Extends the storage to accommodate the specified number of pages.
    ///
    /// If `new_page_count` is less than or equal to current page count, this is a no-op.
    fn grow(&mut self, new_page_count: u32) -> Result<()>;

    /// Returns the current number of pages in the storage.
    fn page_count(&self) -> u32;

    /// Flushes all pending writes to durable storage.
    fn sync(&self) -> Result<()>;

    /// Returns true if this storage supports zero-copy page access.
    ///
    /// When true, `page_direct()` and `page_direct_mut()` return direct references
    /// to page data, bypassing the copy interface for maximum performance.
    fn supports_zero_copy(&self) -> bool {
        false
    }

    /// Returns a direct read-only reference to page data (zero-copy).
    ///
    /// Only available when `supports_zero_copy()` returns true. Returns `None` for
    /// storage backends that don't support zero-copy access.
    ///
    /// The returned slice is valid only while no `&mut self` operations are in progress.
    /// The borrow checker enforces this at compile time for mmap-based storage.
    fn page_direct(&self, _page_no: u32) -> Option<Result<&[u8]>> {
        None
    }

    /// Returns a direct mutable reference to page data (zero-copy).
    ///
    /// Only available when `supports_zero_copy()` returns true. Returns `None` for
    /// storage backends that don't support zero-copy access.
    ///
    /// The caller must ensure the page is within bounds. Changes are persisted on `sync()`.
    fn page_direct_mut(&mut self, _page_no: u32) -> Option<Result<&mut [u8]>> {
        None
    }

    /// Hints to the OS that the specified page range will be accessed soon.
    ///
    /// For mmap-based storage, this uses `madvise(MADV_WILLNEED)` to trigger
    /// prefetching. For other backends, this is typically a no-op.
    fn prefetch(&self, _start_page: u32, _count: u32) {}
}

/// Configuration for storage backend selection.
///
/// Used by `DatabaseBuilder` to configure which storage backend to use when
/// opening or creating a database.
#[derive(Debug, Clone)]
pub enum StorageKind {
    /// Memory-mapped storage (native platforms only).
    ///
    /// Uses `mmap()` for zero-copy page access. This is the default and
    /// recommended backend for native applications.
    Mmap {
        /// Path to the database directory.
        path: PathBuf,
    },

    /// Origin Private File System storage (WASM in Worker context).
    ///
    /// Uses `FileSystemSyncAccessHandle` for synchronous file access.
    /// Requires running in a dedicated Web Worker (not main thread).
    /// WAL is automatically disabled for this backend.
    #[cfg(target_arch = "wasm32")]
    Opfs {
        /// Name of the OPFS directory for this database.
        name: String,
    },
}

impl StorageKind {
    /// Creates a new Mmap storage kind with the given path.
    pub fn mmap<P: Into<PathBuf>>(path: P) -> Self {
        StorageKind::Mmap { path: path.into() }
    }

    /// Creates a new OPFS storage kind with the given name.
    #[cfg(target_arch = "wasm32")]
    pub fn opfs(name: impl Into<String>) -> Self {
        StorageKind::Opfs { name: name.into() }
    }

    /// Returns true if this is mmap storage.
    pub fn is_mmap(&self) -> bool {
        matches!(self, StorageKind::Mmap { .. })
    }

    /// Returns true if this is OPFS storage.
    #[cfg(target_arch = "wasm32")]
    pub fn is_opfs(&self) -> bool {
        matches!(self, StorageKind::Opfs { .. })
    }

    /// Returns the path for mmap storage, or None for other backends.
    pub fn path(&self) -> Option<&PathBuf> {
        match self {
            StorageKind::Mmap { path } => Some(path),
            #[cfg(target_arch = "wasm32")]
            StorageKind::Opfs { .. } => None,
        }
    }
}

/// Type-erased storage backend that implements `StorageDriver`.
///
/// This enum allows `FileManager` to work with different storage backends
/// without requiring generics throughout the codebase.
#[derive(Debug)]
pub enum AnyStorage {
    /// Memory-mapped storage for native platforms.
    Mmap(MmapStorage),

    /// OPFS storage for WASM (Worker context only).
    #[cfg(target_arch = "wasm32")]
    Opfs(super::opfs::OpfsStorage),
}

impl AnyStorage {
    /// Creates a new AnyStorage from an MmapStorage.
    pub fn from_mmap(storage: MmapStorage) -> Self {
        AnyStorage::Mmap(storage)
    }

    /// Returns true if this is mmap storage.
    pub fn is_mmap(&self) -> bool {
        matches!(self, AnyStorage::Mmap(_))
    }

    /// Returns a reference to the inner MmapStorage, if applicable.
    pub fn as_mmap(&self) -> Option<&MmapStorage> {
        match self {
            AnyStorage::Mmap(s) => Some(s),
            #[cfg(target_arch = "wasm32")]
            AnyStorage::Opfs(_) => None,
        }
    }

    /// Returns a mutable reference to the inner MmapStorage, if applicable.
    pub fn as_mmap_mut(&mut self) -> Option<&mut MmapStorage> {
        match self {
            AnyStorage::Mmap(s) => Some(s),
            #[cfg(target_arch = "wasm32")]
            AnyStorage::Opfs(_) => None,
        }
    }
}

impl StorageDriver for AnyStorage {
    fn read_page(&self, page_no: u32, buf: &mut [u8; PAGE_SIZE]) -> Result<()> {
        match self {
            AnyStorage::Mmap(s) => s.read_page(page_no, buf),
            #[cfg(target_arch = "wasm32")]
            AnyStorage::Opfs(s) => s.read_page(page_no, buf),
        }
    }

    fn write_page(&mut self, page_no: u32, data: &[u8; PAGE_SIZE]) -> Result<()> {
        match self {
            AnyStorage::Mmap(s) => s.write_page(page_no, data),
            #[cfg(target_arch = "wasm32")]
            AnyStorage::Opfs(s) => s.write_page(page_no, data),
        }
    }

    fn grow(&mut self, new_page_count: u32) -> Result<()> {
        match self {
            AnyStorage::Mmap(s) => s.grow(new_page_count),
            #[cfg(target_arch = "wasm32")]
            AnyStorage::Opfs(s) => s.grow(new_page_count),
        }
    }

    fn page_count(&self) -> u32 {
        match self {
            AnyStorage::Mmap(s) => s.page_count(),
            #[cfg(target_arch = "wasm32")]
            AnyStorage::Opfs(s) => s.page_count(),
        }
    }

    fn sync(&self) -> Result<()> {
        match self {
            AnyStorage::Mmap(s) => s.sync(),
            #[cfg(target_arch = "wasm32")]
            AnyStorage::Opfs(s) => s.sync(),
        }
    }

    fn supports_zero_copy(&self) -> bool {
        match self {
            AnyStorage::Mmap(s) => s.supports_zero_copy(),
            #[cfg(target_arch = "wasm32")]
            AnyStorage::Opfs(s) => s.supports_zero_copy(),
        }
    }

    fn page_direct(&self, page_no: u32) -> Option<Result<&[u8]>> {
        match self {
            AnyStorage::Mmap(s) => s.page_direct(page_no),
            #[cfg(target_arch = "wasm32")]
            AnyStorage::Opfs(s) => s.page_direct(page_no),
        }
    }

    fn page_direct_mut(&mut self, page_no: u32) -> Option<Result<&mut [u8]>> {
        match self {
            AnyStorage::Mmap(s) => s.page_direct_mut(page_no),
            #[cfg(target_arch = "wasm32")]
            AnyStorage::Opfs(s) => s.page_direct_mut(page_no),
        }
    }

    fn prefetch(&self, start_page: u32, count: u32) {
        match self {
            AnyStorage::Mmap(s) => s.prefetch(start_page, count),
            #[cfg(target_arch = "wasm32")]
            AnyStorage::Opfs(s) => s.prefetch(start_page, count),
        }
    }
}

impl Storage for AnyStorage {
    fn page(&self, page_no: u32) -> Result<&[u8]> {
        match self {
            AnyStorage::Mmap(s) => s.page(page_no),
            #[cfg(target_arch = "wasm32")]
            AnyStorage::Opfs(_) => {
                eyre::bail!("OPFS storage does not support zero-copy page access; use StorageDriver::read_page instead")
            }
        }
    }

    fn page_mut(&mut self, page_no: u32) -> Result<&mut [u8]> {
        match self {
            AnyStorage::Mmap(s) => s.page_mut(page_no),
            #[cfg(target_arch = "wasm32")]
            AnyStorage::Opfs(_) => {
                eyre::bail!("OPFS storage does not support zero-copy page access; use StorageDriver::write_page instead")
            }
        }
    }

    fn grow(&mut self, new_page_count: u32) -> Result<()> {
        StorageDriver::grow(self, new_page_count)
    }

    fn page_count(&self) -> u32 {
        StorageDriver::page_count(self)
    }

    fn sync(&self) -> Result<()> {
        StorageDriver::sync(self)
    }

    fn prefetch_pages(&self, start_page: u32, count: u32) {
        StorageDriver::prefetch(self, start_page, count)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn storage_kind_mmap_creation_returns_mmap_variant() {
        let kind = StorageKind::mmap("/tmp/test");

        assert!(kind.is_mmap());
        assert_eq!(kind.path(), Some(&PathBuf::from("/tmp/test")));
    }

    #[test]
    fn storage_kind_mmap_path_accessor_returns_correct_path() {
        let expected_path = PathBuf::from("/var/lib/turdb/mydb");
        let kind = StorageKind::Mmap {
            path: expected_path.clone(),
        };

        assert_eq!(kind.path(), Some(&expected_path));
    }

    #[cfg(target_arch = "wasm32")]
    #[test]
    fn storage_kind_opfs_creation_returns_opfs_variant() {
        let kind = StorageKind::opfs("my-database");

        assert!(kind.is_opfs());
        assert!(!kind.is_mmap());
        assert_eq!(kind.path(), None);
    }

    mod any_storage_tests {
        use super::*;
        use tempfile::tempdir;

        #[test]
        fn any_storage_from_mmap_creates_mmap_variant() {
            let dir = tempdir().unwrap();
            let path = dir.path().join("test.db");

            let mmap = MmapStorage::create(&path, 1).unwrap();
            let any = AnyStorage::from_mmap(mmap);

            assert!(any.is_mmap());
        }

        #[test]
        fn any_storage_as_mmap_returns_inner_storage() {
            let dir = tempdir().unwrap();
            let path = dir.path().join("test.db");

            let mmap = MmapStorage::create(&path, 1).unwrap();
            let any = AnyStorage::from_mmap(mmap);

            assert!(any.as_mmap().is_some());
        }

        #[test]
        fn any_storage_read_page_works() {
            let dir = tempdir().unwrap();
            let path = dir.path().join("test.db");

            let mut mmap = MmapStorage::create(&path, 1).unwrap();
            mmap.page_mut(0).unwrap()[0] = 42;

            let any = AnyStorage::from_mmap(mmap);

            let mut buf = [0u8; PAGE_SIZE];
            any.read_page(0, &mut buf).unwrap();
            assert_eq!(buf[0], 42);
        }

        #[test]
        fn any_storage_write_page_works() {
            let dir = tempdir().unwrap();
            let path = dir.path().join("test.db");

            let mmap = MmapStorage::create(&path, 1).unwrap();
            let mut any = AnyStorage::from_mmap(mmap);

            let mut data = [0u8; PAGE_SIZE];
            data[0] = 99;
            any.write_page(0, &data).unwrap();

            let page = any.as_mmap().unwrap().page(0).unwrap();
            assert_eq!(page[0], 99);
        }

        #[test]
        fn any_storage_page_count_returns_correct_value() {
            let dir = tempdir().unwrap();
            let path = dir.path().join("test.db");

            let mmap = MmapStorage::create(&path, 5).unwrap();
            let any = AnyStorage::from_mmap(mmap);

            assert_eq!(StorageDriver::page_count(&any), 5);
        }

        #[test]
        fn any_storage_grow_increases_page_count() {
            let dir = tempdir().unwrap();
            let path = dir.path().join("test.db");

            let mmap = MmapStorage::create(&path, 1).unwrap();
            let mut any = AnyStorage::from_mmap(mmap);

            assert_eq!(StorageDriver::page_count(&any), 1);

            StorageDriver::grow(&mut any, 10).unwrap();
            assert_eq!(StorageDriver::page_count(&any), 10);
        }

        #[test]
        fn any_storage_supports_zero_copy_returns_true_for_mmap() {
            let dir = tempdir().unwrap();
            let path = dir.path().join("test.db");

            let mmap = MmapStorage::create(&path, 1).unwrap();
            let any = AnyStorage::from_mmap(mmap);

            assert!(any.supports_zero_copy());
        }

        #[test]
        fn any_storage_page_direct_returns_valid_slice() {
            let dir = tempdir().unwrap();
            let path = dir.path().join("test.db");

            let mut mmap = MmapStorage::create(&path, 1).unwrap();
            mmap.page_mut(0).unwrap()[0] = 77;

            let any = AnyStorage::from_mmap(mmap);

            let result = any.page_direct(0);
            assert!(result.is_some());

            let page = result.unwrap().unwrap();
            assert_eq!(page[0], 77);
        }

        #[test]
        fn any_storage_implements_storage_trait() {
            let dir = tempdir().unwrap();
            let path = dir.path().join("test.db");

            let mmap = MmapStorage::create(&path, 2).unwrap();
            let mut any = AnyStorage::from_mmap(mmap);

            Storage::page_mut(&mut any, 0).unwrap()[0] = 123;

            let page = Storage::page(&any, 0).unwrap();
            assert_eq!(page[0], 123);
        }

        #[test]
        fn any_storage_sync_does_not_fail() {
            let dir = tempdir().unwrap();
            let path = dir.path().join("test.db");

            let mmap = MmapStorage::create(&path, 1).unwrap();
            let any = AnyStorage::from_mmap(mmap);

            let result = StorageDriver::sync(&any);
            assert!(result.is_ok());
        }
    }
}
