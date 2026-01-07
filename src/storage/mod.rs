//! # Storage Module
//!
//! This module provides the foundational storage layer for TurDB, implementing
//! memory-mapped file access with zero-copy semantics and compile-time safety
//! guarantees through Rust's borrow checker.
//!
//! ## Architecture Overview
//!
//! The storage layer is built around memory-mapped I/O for maximum performance.
//! Instead of copying data between kernel and user space, we map database files
//! directly into the process address space. This enables:
//!
//! - **Zero-copy reads**: Return `&[u8]` slices pointing directly into mmap region
//! - **Minimal syscall overhead**: Page faults handled transparently by the OS
//! - **Efficient caching**: Leverage the OS page cache instead of duplicating
//!
//! ## File-Per-Table Architecture
//!
//! TurDB uses a MySQL-style file layout where each table has dedicated files:
//!
//! ```text
//! database_dir/
//! ├── turdb.meta           # Global metadata and catalog
//! ├── root/                # Default schema
//! │   ├── users.tbd        # Table data file
//! │   ├── users.idx        # B-tree indexes
//! │   └── users.hnsw       # HNSW vector indexes (if any)
//! ├── analytics/           # Custom schema
//! │   └── events.tbd
//! └── wal/
//!     └── wal.000001       # Write-ahead log segments
//! ```
//!
//! The `FileManager` component manages this directory structure and creates
//! `MmapStorage` instances for each file. Users should not create `MmapStorage`
//! directly; instead use the higher-level `Database` API.
//!
//! ## Safety Model
//!
//! Memory-mapped files present a unique safety challenge: the underlying memory
//! can become invalid when the file is grown and remapped. Traditional approaches
//! use runtime checks (guards, epochs, reference counting) which add overhead.
//!
//! TurDB uses Rust's borrow checker for **compile-time enforcement**:
//!
//! ```text
//! MmapStorage::page(&self) -> &[u8]     // Borrows &self immutably
//! MmapStorage::grow(&mut self)          // Requires &mut self exclusively
//! ```
//!
//! The borrow checker prevents holding page references across grow() calls
//! at compile time, with zero runtime cost.
//!
//! ## Page Size
//!
//! All storage uses 16KB (16384 byte) pages:
//! - Larger than SQLite's 4KB default for better sequential throughput
//! - Aligned to common OS page sizes (4KB multiple)
//! - Reduces B-tree depth for large datasets
//!
//! ## Module Organization
//!
//! - `mmap`: Low-level memory-mapped storage (`MmapStorage`)
//! - `page`: Page type definitions and header layouts
//! - `cache`: SIEVE-based page cache with lock sharding
//! - `freelist`: Free page tracking and allocation
//! - `file_manager`: File-per-table directory management
//!
//! ## Performance Characteristics
//!
//! - Page access: O(1) pointer arithmetic, no syscalls for cached pages
//! - Grow: O(1) amortized (remap), may trigger page faults on first access
//! - Sync: O(n) where n is number of dirty pages
//!
//! ## Thread Safety
//!
//! `MmapStorage` is `Send` but not `Sync`. For concurrent access, wrap in
//! appropriate synchronization (e.g., `RwLock` or the `PageCache` layer).
//!
//! ## Platform Support
//!
//! Uses `memmap2` crate which supports:
//! - Linux (mmap/munmap/msync)
//! - macOS (mmap/munmap/msync)
//! - Windows (CreateFileMapping/MapViewOfFile)

mod cache;
mod file_manager;
mod freelist;
mod headers;
mod mmap;
mod page;
pub mod toast;
mod wal;
mod wal_storage;

pub use cache::{PageCache, PageKey, PageRef};
pub use file_manager::{
    FileKey, FileManager, LruFileCache, TableFiles, CATALOG_FILE_NAME, DEFAULT_MAX_OPEN_FILES,
    DEFAULT_SCHEMA, HNSW_FILE_EXTENSION, HNSW_MAGIC, INDEX_FILE_EXTENSION, MIN_MAX_OPEN_FILES,
    TABLE_FILE_EXTENSION,
};
pub use freelist::{Freelist, TrunkHeader, TRUNK_HEADER_SIZE, TRUNK_MAX_ENTRIES};
pub use headers::{
    IndexFileHeader, MetaFileHeader, TableFileHeader, CURRENT_VERSION, DEFAULT_PAGE_SIZE,
    INDEX_MAGIC, INDEX_TYPE_BTREE, INDEX_TYPE_HASH, META_MAGIC, TABLE_MAGIC,
};
pub use mmap::MmapStorage;
pub use page::{validate_page, PageHeader, PageType};
pub use wal::{SyncMode, Wal, WalFrameHeader, WalSegment};
pub use wal_storage::{WalStorage, WalStoragePerTable};

use eyre::{ensure, Result};
use zerocopy::{FromBytes, Immutable, KnownLayout};

/// Parses a zerocopy struct from a byte slice with size validation.
#[inline]
pub fn parse_zerocopy<'a, T: FromBytes + KnownLayout + Immutable>(
    bytes: &'a [u8],
    type_name: &str,
) -> Result<&'a T> {
    let size = std::mem::size_of::<T>();
    ensure!(
        bytes.len() >= size,
        "buffer too small for {}: {} < {}",
        type_name,
        bytes.len(),
        size
    );
    T::ref_from_bytes(&bytes[..size])
        .map_err(|e| eyre::eyre!("failed to parse {}: {:?}", type_name, e))
}

/// Parses a mutable zerocopy struct from a byte slice with size validation.
#[inline]
pub fn parse_zerocopy_mut<'a, T: FromBytes + KnownLayout + zerocopy::IntoBytes>(
    bytes: &'a mut [u8],
    type_name: &str,
) -> Result<&'a mut T> {
    let size = std::mem::size_of::<T>();
    ensure!(
        bytes.len() >= size,
        "buffer too small for {}: {} < {}",
        type_name,
        bytes.len(),
        size
    );
    T::mut_from_bytes(&mut bytes[..size])
        .map_err(|e| eyre::eyre!("failed to parse {}: {:?}", type_name, e))
}

pub trait Storage {
    fn page(&self, page_no: u32) -> Result<&[u8]>;
    fn page_mut(&mut self, page_no: u32) -> Result<&mut [u8]>;
    fn grow(&mut self, new_page_count: u32) -> Result<()>;
    fn page_count(&self) -> u32;
    fn sync(&self) -> Result<()>;

    fn prefetch_pages(&self, _start_page: u32, _count: u32) {}
}

impl Storage for MmapStorage {
    fn page(&self, page_no: u32) -> Result<&[u8]> {
        MmapStorage::page(self, page_no)
    }

    fn page_mut(&mut self, page_no: u32) -> Result<&mut [u8]> {
        MmapStorage::page_mut(self, page_no)
    }

    fn grow(&mut self, new_page_count: u32) -> Result<()> {
        MmapStorage::grow(self, new_page_count)
    }

    fn page_count(&self) -> u32 {
        MmapStorage::page_count(self)
    }

    fn sync(&self) -> Result<()> {
        MmapStorage::sync(self)
    }

    fn prefetch_pages(&self, start_page: u32, count: u32) {
        MmapStorage::prefetch_pages(self, start_page, count)
    }
}

impl<S: Storage> Storage for parking_lot::RwLockWriteGuard<'_, S> {
    fn page(&self, page_no: u32) -> Result<&[u8]> {
        (**self).page(page_no)
    }

    fn page_mut(&mut self, page_no: u32) -> Result<&mut [u8]> {
        (**self).page_mut(page_no)
    }

    fn grow(&mut self, new_page_count: u32) -> Result<()> {
        (**self).grow(new_page_count)
    }

    fn page_count(&self) -> u32 {
        (**self).page_count()
    }

    fn sync(&self) -> Result<()> {
        (**self).sync()
    }

    fn prefetch_pages(&self, start_page: u32, count: u32) {
        (**self).prefetch_pages(start_page, count)
    }
}

pub const PAGE_SIZE: usize = 16384;
pub const PAGE_HEADER_SIZE: usize = 16;
pub const PAGE_USABLE_SIZE: usize = PAGE_SIZE - PAGE_HEADER_SIZE;
pub const FILE_HEADER_SIZE: usize = 128;
pub const PAGE0_USABLE_SIZE: usize = PAGE_SIZE - FILE_HEADER_SIZE;
