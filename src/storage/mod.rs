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
mod mmap;
mod page;

pub use cache::{PageCache, PageKey, PageRef};
pub use mmap::MmapStorage;
pub use page::{validate_page, PageHeader, PageType};

pub const PAGE_SIZE: usize = 16384;
pub const PAGE_HEADER_SIZE: usize = 16;
pub const PAGE_USABLE_SIZE: usize = PAGE_SIZE - PAGE_HEADER_SIZE;
pub const FILE_HEADER_SIZE: usize = 128;
pub const PAGE0_USABLE_SIZE: usize = PAGE_SIZE - FILE_HEADER_SIZE;
