//! # OPFS Storage Backend
//!
//! This module provides `OpfsStorage`, a storage backend for browser WASM environments
//! using the Origin Private File System (OPFS) API with synchronous FileSystemSyncAccessHandle.
//!
//! ## OPFS Overview
//!
//! The Origin Private File System is a Web API that provides fast, sandboxed file access
//! for web applications. Unlike the traditional File System Access API, OPFS files are:
//!
//! - **Private**: Only accessible by the origin that created them
//! - **Persistent**: Survive browser restarts
//! - **Fast**: Optimized for performance, especially with sync handles
//!
//! ## FileSystemSyncAccessHandle
//!
//! The synchronous access handle (`FileSystemSyncAccessHandle`) provides:
//!
//! - `read()` / `write()`: Synchronous byte I/O at specific offsets
//! - `truncate()`: Resize the file
//! - `flush()`: Ensure data is persisted to disk
//! - `close()`: Release the handle
//!
//! This API is only available in Web Worker contexts (not the main thread).
//!
//! ## Worker Requirement
//!
//! CRITICAL: `OpfsStorage` MUST run in a dedicated Web Worker. The synchronous
//! FileSystemSyncAccessHandle API blocks the calling thread, which would freeze
//! the browser UI if used on the main thread.
//!
//! ## File Layout
//!
//! OpfsStorage creates files in the OPFS with this structure:
//!
//! ```text
//! /{database_name}/
//! ├── turdb.meta           # Global metadata
//! ├── root/                # Default schema
//! │   ├── users.tbd        # Table data
//! │   └── users_pk.idx     # Index data
//! └── ...
//! ```
//!
//! ## Limitations vs Mmap
//!
//! - **No zero-copy**: All page access involves memory copies
//! - **No memory mapping**: Cannot return &[u8] slices into storage
//! - **Worker-only**: Cannot be used from main thread
//! - **No prefetch**: `madvise` not available
//!
//! ## WAL Disabled
//!
//! WAL is automatically disabled when using OPFS storage because:
//!
//! - OPFS provides its own durability guarantees via `flush()`
//! - WAL complexity isn't warranted for browser storage
//! - Simplifies the WASM build
//!
//! ## Usage
//!
//! ```ignore
//! // In a Web Worker:
//! let storage = OpfsStorage::open("my-database", "turdb.meta").await?;
//! let mut buf = [0u8; PAGE_SIZE];
//! storage.read_page(0, &mut buf)?;
//! ```
//!
//! ## Platform Support
//!
//! OPFS with synchronous access is supported in:
//! - Chrome 102+ (desktop and Android)
//! - Edge 102+
//! - Safari 15.4+ (partial support)
//! - Firefox 111+

#![cfg(target_arch = "wasm32")]

use eyre::{bail, Result, WrapErr};

use super::driver::StorageDriver;
use super::PAGE_SIZE;

/// Storage backend using OPFS FileSystemSyncAccessHandle.
///
/// This struct wraps the OPFS synchronous access handle for page-based storage.
/// It requires running in a Web Worker context.
#[derive(Debug)]
pub struct OpfsStorage {
    page_count: u32,
}

impl OpfsStorage {
    /// Opens an existing OPFS file or creates it if it doesn't exist.
    ///
    /// # Arguments
    ///
    /// * `_database_name` - Name of the database directory in OPFS
    /// * `_file_name` - Name of the file within the database directory
    ///
    /// # Returns
    ///
    /// An OpfsStorage instance ready for page I/O.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Called from the main thread (not a Worker)
    /// - OPFS API is not available
    /// - File access fails
    pub fn open(_database_name: &str, _file_name: &str) -> Result<Self> {
        bail!("OpfsStorage::open is not yet implemented - OPFS support requires web-sys bindings")
    }

    /// Creates a new OPFS file with the specified initial page count.
    ///
    /// # Arguments
    ///
    /// * `_database_name` - Name of the database directory in OPFS
    /// * `_file_name` - Name of the file to create
    /// * `_initial_page_count` - Number of pages to allocate initially
    ///
    /// # Returns
    ///
    /// An OpfsStorage instance ready for page I/O.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Called from the main thread (not a Worker)
    /// - OPFS API is not available
    /// - File creation fails
    pub fn create(_database_name: &str, _file_name: &str, _initial_page_count: u32) -> Result<Self> {
        bail!("OpfsStorage::create is not yet implemented - OPFS support requires web-sys bindings")
    }
}

impl StorageDriver for OpfsStorage {
    fn read_page(&self, _page_no: u32, _buf: &mut [u8; PAGE_SIZE]) -> Result<()> {
        bail!("OpfsStorage::read_page is not yet implemented")
    }

    fn write_page(&mut self, _page_no: u32, _data: &[u8; PAGE_SIZE]) -> Result<()> {
        bail!("OpfsStorage::write_page is not yet implemented")
    }

    fn grow(&mut self, _new_page_count: u32) -> Result<()> {
        bail!("OpfsStorage::grow is not yet implemented")
    }

    fn page_count(&self) -> u32 {
        self.page_count
    }

    fn sync(&self) -> Result<()> {
        bail!("OpfsStorage::sync is not yet implemented")
    }

    fn supports_zero_copy(&self) -> bool {
        false
    }

    fn page_direct(&self, _page_no: u32) -> Option<Result<&[u8]>> {
        None
    }

    fn page_direct_mut(&mut self, _page_no: u32) -> Option<Result<&mut [u8]>> {
        None
    }

    fn prefetch(&self, _start_page: u32, _count: u32) {
        // No-op: OPFS doesn't support prefetching hints
    }
}
