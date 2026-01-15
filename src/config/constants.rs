//! # TurDB Configuration Constants
//!
//! This module centralizes all configuration constants, grouping interdependent
//! values together and documenting their relationships. Constants that depend
//! on each other are co-located to prevent mismatch bugs.
//!
//! ## Dependency Graph
//!
//! The following diagram shows how constants relate to each other. When changing
//! any constant, check if dependent constants need adjustment.
//!
//! ```text
//! DEFAULT_BUFFER_POOL_SIZE (16)
//!       │
//!       ├─> COMMIT_BATCH_SIZE (must be <=)
//!       │     The execute_small_commit path acquires buffer pool buffers for
//!       │     all dirty pages at once. If COMMIT_BATCH_SIZE > pool size,
//!       │     acquire_blocking() will deadlock waiting for buffers.
//!       │
//!       └─> BUFFER_POOL_SHARD_COUNT (typically equal for even distribution)
//!
//! PAGE_SIZE (16384 bytes)
//!       │
//!       ├─> PAGE_HEADER_SIZE (16 bytes, fixed)
//!       │
//!       ├─> PAGE_USABLE_SIZE (derived: PAGE_SIZE - PAGE_HEADER_SIZE)
//!       │
//!       ├─> FILE_HEADER_SIZE (128 bytes, page 0 only)
//!       │
//!       ├─> PAGE0_USABLE_SIZE (derived: PAGE_SIZE - FILE_HEADER_SIZE)
//!       │
//!       └─> WAL_FRAME_HEADER_SIZE (32 bytes per frame)
//!             Each WAL frame = header + full page data
//!
//! CACHE_SHARD_COUNT (64)
//!       │
//!       └─> TABLE_SHARD_COUNT (64, for page locks by table)
//!             PAGE_SHARD_COUNT (256, for page locks by page)
//!
//! DEFAULT_CHECKPOINT_THRESHOLD (100,000 frames)
//!       │
//!       └─> At this many WAL frames, checkpoint is triggered
//!           WAL_BUFFER_SIZE should be large enough to batch writes
//! ```
//!
//! ## Critical Invariants
//!
//! These invariants are enforced by compile-time assertions:
//!
//! 1. `COMMIT_BATCH_SIZE <= DEFAULT_BUFFER_POOL_SIZE` (prevents deadlock)
//! 2. `PAGE_USABLE_SIZE == PAGE_SIZE - PAGE_HEADER_SIZE` (derived correctly)
//! 3. `PAGE0_USABLE_SIZE == PAGE_SIZE - FILE_HEADER_SIZE` (derived correctly)
//!
//! ## Usage
//!
//! Import constants from this module rather than defining them locally:
//!
//! ```ignore
//! use crate::config::{PAGE_SIZE, COMMIT_BATCH_SIZE};
//! ```
//!
//! ## Modifying Constants
//!
//! Before changing any constant:
//! 1. Check the dependency graph above
//! 2. Run `cargo build` to verify compile-time assertions
//! 3. Run full test suite
//! 4. Benchmark affected operations
//!
//! ## Performance Implications
//!
//! - `DEFAULT_BUFFER_POOL_SIZE`: Larger = more memory, fewer disk round-trips
//! - `COMMIT_BATCH_SIZE`: Larger = fewer WAL writes, more latency per commit
//! - `CACHE_SHARD_COUNT`: More shards = less lock contention, more memory
//! - `DEFAULT_CHECKPOINT_THRESHOLD`: Higher = larger WAL, faster writes
//!
//! ## Memory Budget Relationships
//!
//! The memory budget constants define reservation pools:
//! - `CACHE_RESERVED`: Minimum guaranteed for page cache
//! - `QUERY_RESERVED`: For sort buffers and hash tables
//! - `RECOVERY_RESERVED`: For WAL frame processing
//! - `SCHEMA_RESERVED`: For catalog metadata
//!
//! Total reserved = sum of all pools. Remaining budget is shared dynamically.

// ============================================================================
// BUFFER POOL CONFIGURATION
// These constants are tightly coupled - changing one may require changing others
// ============================================================================

/// Default number of page buffers in the pool.
/// This determines the maximum number of pages that can be held concurrently
/// during operations like commit.
pub const DEFAULT_BUFFER_POOL_SIZE: usize = 16;

/// Maximum number of dirty pages to commit in the "small commit" path.
/// MUST be <= DEFAULT_BUFFER_POOL_SIZE to prevent deadlock.
///
/// The execute_small_commit function acquires buffers from the pool for all
/// dirty pages simultaneously. If this exceeds the pool size, acquire_blocking()
/// will wait forever for buffers that will never be released.
pub const COMMIT_BATCH_SIZE: usize = DEFAULT_BUFFER_POOL_SIZE;

/// Number of shards for the buffer pool.
/// Equal to pool size for even distribution of buffers across shards.
pub const BUFFER_POOL_SHARD_COUNT: usize = 16;

/// Number of shards for dirty page tracking.
/// Matches buffer pool sharding for consistent concurrency characteristics.
pub const DIRTY_SHARD_COUNT: usize = 16;

const _: () = assert!(
    COMMIT_BATCH_SIZE <= DEFAULT_BUFFER_POOL_SIZE,
    "COMMIT_BATCH_SIZE must be <= DEFAULT_BUFFER_POOL_SIZE to prevent deadlock in execute_small_commit"
);

// ============================================================================
// PAGE LAYOUT CONSTANTS
// These define the fundamental page structure used throughout the database
// ============================================================================

/// Size of each database page in bytes (16KB).
/// This is the fundamental unit of I/O and caching.
pub const PAGE_SIZE: usize = 16384;

/// Size of the page header in bytes.
/// Every page begins with this header containing type, flags, and metadata.
pub const PAGE_HEADER_SIZE: usize = 16;

/// Size of the file header in bytes (page 0 only).
/// The first page has an extended header with database metadata.
pub const FILE_HEADER_SIZE: usize = 128;

/// Usable space in a regular page after the header.
pub const PAGE_USABLE_SIZE: usize = PAGE_SIZE - PAGE_HEADER_SIZE;

/// Usable space in page 0 after the file header.
pub const PAGE0_USABLE_SIZE: usize = PAGE_SIZE - FILE_HEADER_SIZE;

const _: () = assert!(
    PAGE_USABLE_SIZE == PAGE_SIZE - PAGE_HEADER_SIZE,
    "PAGE_USABLE_SIZE derivation mismatch"
);

const _: () = assert!(
    PAGE0_USABLE_SIZE == PAGE_SIZE - FILE_HEADER_SIZE,
    "PAGE0_USABLE_SIZE derivation mismatch"
);

// ============================================================================
// SHARDING CONSTANTS
// These control lock contention characteristics
// ============================================================================

/// Number of shards for the page cache.
/// Higher values reduce contention but increase memory overhead.
pub const CACHE_SHARD_COUNT: usize = 64;

/// Number of shards for table-level locks.
pub const TABLE_SHARD_COUNT: usize = 64;

/// Number of shards for page-level locks.
/// Higher than table shards since page access is more frequent.
pub const PAGE_SHARD_COUNT: usize = 256;

// ============================================================================
// WAL CONFIGURATION
// Write-ahead log settings for durability and recovery
// ============================================================================

/// Size of the WAL frame header in bytes.
/// Each frame contains: page_no, db_size, salt, checksums.
pub const WAL_FRAME_HEADER_SIZE: usize = 32;

/// Maximum size of a single WAL segment file.
/// New segment created when this is exceeded.
pub const MAX_WAL_SEGMENT_SIZE: u64 = 64 * 1024 * 1024;

/// Number of WAL frames before triggering automatic checkpoint.
/// Higher values improve write throughput but increase recovery time.
pub const DEFAULT_CHECKPOINT_THRESHOLD: u32 = 100_000;

/// Size of the in-memory WAL write buffer.
/// Larger buffer = fewer fsync calls, more memory usage.
pub const WAL_BUFFER_SIZE: usize = 8 * 1024 * 1024;

// ============================================================================
// MEMORY BUDGET CONFIGURATION
// These define the memory allocation strategy
// ============================================================================

/// Default memory budget as percentage of system RAM.
pub const DEFAULT_BUDGET_PERCENT: usize = 25;

/// Minimum memory budget floor in bytes (4MB).
/// Even on low-memory systems, we need this much to function.
pub const MIN_BUDGET_FLOOR: usize = 4 * 1024 * 1024;

/// Memory reserved for page cache in bytes (512KB).
pub const CACHE_RESERVED: usize = 512 * 1024;

/// Memory reserved for query execution (sort buffers, hash tables).
pub const QUERY_RESERVED: usize = 256 * 1024;

/// Memory reserved for WAL recovery operations.
pub const RECOVERY_RESERVED: usize = 256 * 1024;

/// Memory reserved for schema/catalog metadata.
pub const SCHEMA_RESERVED: usize = 128 * 1024;

/// Total reserved memory across all pools.
pub const TOTAL_RESERVED: usize =
    CACHE_RESERVED + QUERY_RESERVED + RECOVERY_RESERVED + SCHEMA_RESERVED;

// ============================================================================
// RECOVERY CONFIGURATION
// ============================================================================

/// Default batch size for recovery operations.
pub const DEFAULT_RECOVERY_BATCH_SIZE: usize = 1000;

/// Size of a complete WAL frame (header + page data).
pub const WAL_FRAME_SIZE: usize = WAL_FRAME_HEADER_SIZE + PAGE_SIZE;

// ============================================================================
// CONCURRENCY LIMITS
// ============================================================================

/// Maximum number of concurrent transactions.
pub const MAX_CONCURRENT_TXNS: usize = 64;

/// Maximum number of open file handles.
pub const DEFAULT_MAX_OPEN_FILES: usize = 64;

/// Minimum allowed value for max open files setting.
pub const MIN_MAX_OPEN_FILES: usize = 8;
