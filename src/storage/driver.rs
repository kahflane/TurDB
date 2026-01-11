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
//! `StorageDriver` requires `Send` bound. For `Sync` access:
//!
//! - `MmapStorage`: `Send` but not `Sync` (contains `MmapMut`). External `RwLock`
//!   synchronization is required for concurrent access.
//! - `AnyStorage`: Wraps storage in `RwLock` internally to provide `Sync` bound.
//! - `OpfsStorage`: Worker-only, single-threaded access (WASM)
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
use parking_lot::RwLock;
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
    #[inline]
    pub fn is_mmap(&self) -> bool {
        matches!(self, StorageKind::Mmap { .. })
    }

    /// Returns true if this is OPFS storage.
    #[inline]
    #[cfg(target_arch = "wasm32")]
    pub fn is_opfs(&self) -> bool {
        matches!(self, StorageKind::Opfs { .. })
    }

    /// Returns the path for mmap storage, or None for other backends.
    #[inline]
    pub fn path(&self) -> Option<&PathBuf> {
        match self {
            StorageKind::Mmap { path } => Some(path),
            #[cfg(target_arch = "wasm32")]
            StorageKind::Opfs { .. } => None,
        }
    }
}

/// Type-erased storage backend that implements `StorageDriver` with `Sync` support.
///
/// This enum allows `FileManager` to work with different storage backends
/// without requiring generics throughout the codebase. Internal `RwLock` wrappers
/// provide thread-safety (`Sync` bound) for concurrent access.
///
/// # Zero-Copy Support
///
/// `AnyStorage` does **not** support zero-copy page access because `RwLock` guards
/// cannot return references that outlive the lock. Use `MmapStorage` directly
/// (wrapped in external `Arc<RwLock<...>>`) for zero-copy access.
pub enum AnyStorage {
    /// Memory-mapped storage for native platforms (wrapped in RwLock for Sync).
    Mmap(RwLock<MmapStorage>),

    /// OPFS storage for WASM (Worker context only, wrapped in RwLock for Sync).
    #[cfg(target_arch = "wasm32")]
    Opfs(RwLock<super::opfs::OpfsStorage>),
}

// Debug impl since RwLock<MmapStorage> doesn't auto-derive Debug
impl std::fmt::Debug for AnyStorage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AnyStorage::Mmap(_) => f.debug_tuple("Mmap").field(&"<RwLock<MmapStorage>>").finish(),
            #[cfg(target_arch = "wasm32")]
            AnyStorage::Opfs(_) => f.debug_tuple("Opfs").field(&"<RwLock<OpfsStorage>>").finish(),
        }
    }
}

impl AnyStorage {
    /// Creates a new AnyStorage from an MmapStorage.
    #[inline]
    pub fn from_mmap(storage: MmapStorage) -> Self {
        AnyStorage::Mmap(RwLock::new(storage))
    }

    /// Returns true if this is mmap storage.
    #[inline]
    pub fn is_mmap(&self) -> bool {
        matches!(self, AnyStorage::Mmap(_))
    }

    /// Executes a function with read access to the inner MmapStorage.
    ///
    /// Returns `None` if this is not mmap storage.
    #[inline]
    pub fn with_mmap<R, F: FnOnce(&MmapStorage) -> R>(&self, f: F) -> Option<R> {
        match self {
            AnyStorage::Mmap(s) => Some(f(&s.read())),
            #[cfg(target_arch = "wasm32")]
            AnyStorage::Opfs(_) => None,
        }
    }

    /// Executes a function with write access to the inner MmapStorage.
    ///
    /// Returns `None` if this is not mmap storage.
    #[inline]
    pub fn with_mmap_mut<R, F: FnOnce(&mut MmapStorage) -> R>(&self, f: F) -> Option<R> {
        match self {
            AnyStorage::Mmap(s) => Some(f(&mut s.write())),
            #[cfg(target_arch = "wasm32")]
            AnyStorage::Opfs(_) => None,
        }
    }
}

impl StorageDriver for AnyStorage {
    #[inline]
    fn read_page(&self, page_no: u32, buf: &mut [u8; PAGE_SIZE]) -> Result<()> {
        match self {
            AnyStorage::Mmap(s) => s.read().read_page(page_no, buf),
            #[cfg(target_arch = "wasm32")]
            AnyStorage::Opfs(s) => s.read().read_page(page_no, buf),
        }
    }

    #[inline]
    fn write_page(&mut self, page_no: u32, data: &[u8; PAGE_SIZE]) -> Result<()> {
        match self {
            AnyStorage::Mmap(s) => s.write().write_page(page_no, data),
            #[cfg(target_arch = "wasm32")]
            AnyStorage::Opfs(s) => s.write().write_page(page_no, data),
        }
    }

    #[inline]
    fn grow(&mut self, new_page_count: u32) -> Result<()> {
        match self {
            AnyStorage::Mmap(s) => s.write().grow(new_page_count),
            #[cfg(target_arch = "wasm32")]
            AnyStorage::Opfs(s) => s.write().grow(new_page_count),
        }
    }

    #[inline]
    fn page_count(&self) -> u32 {
        match self {
            AnyStorage::Mmap(s) => s.read().page_count(),
            #[cfg(target_arch = "wasm32")]
            AnyStorage::Opfs(s) => s.read().page_count(),
        }
    }

    #[inline]
    fn sync(&self) -> Result<()> {
        match self {
            AnyStorage::Mmap(s) => s.read().sync(),
            #[cfg(target_arch = "wasm32")]
            AnyStorage::Opfs(s) => s.read().sync(),
        }
    }

    /// AnyStorage does not support zero-copy due to RwLock wrapper.
    #[inline]
    fn supports_zero_copy(&self) -> bool {
        false
    }

    /// AnyStorage cannot return direct references through RwLock.
    /// Use MmapStorage directly for zero-copy access.
    #[inline]
    fn page_direct(&self, _page_no: u32) -> Option<Result<&[u8]>> {
        None
    }

    /// AnyStorage cannot return direct references through RwLock.
    /// Use MmapStorage directly for zero-copy access.
    #[inline]
    fn page_direct_mut(&mut self, _page_no: u32) -> Option<Result<&mut [u8]>> {
        None
    }

    fn prefetch(&self, start_page: u32, count: u32) {
        match self {
            AnyStorage::Mmap(s) => s.read().prefetch(start_page, count),
            #[cfg(target_arch = "wasm32")]
            AnyStorage::Opfs(s) => s.read().prefetch(start_page, count),
        }
    }
}

// NOTE: AnyStorage does NOT implement Storage trait because RwLock guards
// cannot return references that outlive the lock. Use StorageDriver's
// copy-based interface (read_page/write_page) instead, or access MmapStorage
// directly via with_mmap/with_mmap_mut for zero-copy operations.

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
        fn any_storage_with_mmap_provides_read_access() {
            let dir = tempdir().unwrap();
            let path = dir.path().join("test.db");

            let mmap = MmapStorage::create(&path, 1).unwrap();
            let any = AnyStorage::from_mmap(mmap);

            let page_count = any.with_mmap(|s| s.page_count());
            assert_eq!(page_count, Some(1));
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

            // Read back via StorageDriver interface
            let mut buf = [0u8; PAGE_SIZE];
            any.read_page(0, &mut buf).unwrap();
            assert_eq!(buf[0], 99);
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
        fn any_storage_does_not_support_zero_copy() {
            // AnyStorage wraps in RwLock, so zero-copy is disabled
            let dir = tempdir().unwrap();
            let path = dir.path().join("test.db");

            let mmap = MmapStorage::create(&path, 1).unwrap();
            let any = AnyStorage::from_mmap(mmap);

            assert!(!any.supports_zero_copy());
        }

        #[test]
        fn any_storage_page_direct_returns_none() {
            // AnyStorage cannot return direct references through RwLock
            let dir = tempdir().unwrap();
            let path = dir.path().join("test.db");

            let mmap = MmapStorage::create(&path, 1).unwrap();
            let any = AnyStorage::from_mmap(mmap);

            assert!(any.page_direct(0).is_none());
        }

        #[test]
        fn any_storage_with_mmap_mut_provides_write_access() {
            let dir = tempdir().unwrap();
            let path = dir.path().join("test.db");

            let mmap = MmapStorage::create(&path, 1).unwrap();
            let any = AnyStorage::from_mmap(mmap);

            // Write directly to inner storage
            any.with_mmap_mut(|s| {
                s.page_mut(0).unwrap()[0] = 55;
            });

            // Read back via StorageDriver interface
            let mut buf = [0u8; PAGE_SIZE];
            any.read_page(0, &mut buf).unwrap();
            assert_eq!(buf[0], 55);
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

        #[test]
        fn any_storage_is_sync() {
            // Verify AnyStorage can be shared across threads
            fn assert_sync<T: Sync>() {}
            assert_sync::<AnyStorage>();
        }

        #[test]
        fn any_storage_is_send() {
            fn assert_send<T: Send>() {}
            assert_send::<AnyStorage>();
        }
    }
}
