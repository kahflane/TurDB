//! # Timing Statistics Module
//!
//! This module provides global timing statistics for performance monitoring
//! and benchmarking of database operations. It tracks cumulative time spent
//! in various hot paths during query execution and data manipulation.
//!
//! ## Purpose
//!
//! Performance profiling is critical for an embedded database. This module
//! provides lightweight timing instrumentation that can remain enabled even
//! in production builds with minimal overhead. The statistics help identify
//! bottlenecks and validate optimization efforts.
//!
//! ## Architecture
//!
//! The module uses global atomic counters to accumulate timing data:
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────┐
//! │                    Timing Statistics                         │
//! ├─────────────────────────────────────────────────────────────┤
//! │  PARSE_TIME_NS     │  SQL parsing duration                  │
//! │  INSERT_TIME_NS    │  Single-row insert duration            │
//! │  RECORD_BUILD_NS   │  Record serialization in batches       │
//! │  BTREE_INSERT_NS   │  BTree operations in batches           │
//! └─────────────────────────────────────────────────────────────┘
//! ```
//!
//! Each counter is an AtomicU64 using Relaxed ordering. This provides:
//! - Lock-free updates from any thread
//! - Minimal CPU overhead (no memory barriers)
//! - Approximate measurements suitable for profiling
//!
//! ## Key Data Structures
//!
//! - `PARSE_TIME_NS`: Cumulative nanoseconds in SQL parser
//! - `INSERT_TIME_NS`: Cumulative nanoseconds in insert path
//! - `RECORD_BUILD_NS`: Cumulative nanoseconds building records for batches
//! - `BTREE_INSERT_NS`: Cumulative nanoseconds in BTree insert during batches
//!
//! ## Usage Patterns
//!
//! ### Benchmarking a Workload
//!
//! ```ignore
//! use turdb::database::{reset_timing_stats, get_timing_stats};
//!
//! reset_timing_stats();
//!
//! for _ in 0..1000 {
//!     db.execute("INSERT INTO users VALUES (1, 'Alice')")?;
//! }
//!
//! let (parse_ns, insert_ns) = get_timing_stats();
//! println!("Parse: {}ms, Insert: {}ms", parse_ns / 1_000_000, insert_ns / 1_000_000);
//! ```
//!
//! ### Profiling Batch Operations
//!
//! ```ignore
//! use turdb::database::{reset_timing_stats, get_batch_timing_stats};
//!
//! reset_timing_stats();
//!
//! db.insert_batch("users", &rows)?;
//!
//! let (record_ns, btree_ns) = get_batch_timing_stats();
//! println!("Record build: {}ms, BTree: {}ms",
//!          record_ns / 1_000_000, btree_ns / 1_000_000);
//! ```
//!
//! ## Performance Characteristics
//!
//! - Counter increment: ~1-2ns (single atomic add)
//! - Counter read: ~1ns (single atomic load)
//! - Reset: ~4ns (four atomic stores)
//! - No contention overhead with Relaxed ordering
//!
//! The overhead is negligible compared to actual database operations which
//! typically take microseconds to milliseconds.
//!
//! ## Thread Safety
//!
//! All counters use atomic operations, making this module fully thread-safe.
//! Multiple threads can update counters concurrently without synchronization.
//! The Relaxed ordering means updates may not be immediately visible across
//! threads, but this is acceptable for cumulative timing measurements.
//!
//! ## Limitations
//!
//! - Counters can overflow after ~584 years of continuous nanosecond counting
//! - Relaxed ordering means cross-thread visibility is not guaranteed
//! - Does not track per-operation latency distributions (only cumulative)
//!
//! For detailed latency analysis, consider external profiling tools like
//! `perf`, `flamegraph`, or `criterion` benchmarks.

use std::sync::atomic::{AtomicU64, Ordering};

pub(crate) static PARSE_TIME_NS: AtomicU64 = AtomicU64::new(0);
pub(crate) static INSERT_TIME_NS: AtomicU64 = AtomicU64::new(0);
pub(crate) static RECORD_BUILD_NS: AtomicU64 = AtomicU64::new(0);
pub(crate) static BTREE_INSERT_NS: AtomicU64 = AtomicU64::new(0);

/// Resets all timing statistics to zero.
pub fn reset_timing_stats() {
    PARSE_TIME_NS.store(0, Ordering::Relaxed);
    INSERT_TIME_NS.store(0, Ordering::Relaxed);
    RECORD_BUILD_NS.store(0, Ordering::Relaxed);
    BTREE_INSERT_NS.store(0, Ordering::Relaxed);
}

/// Returns (parse_time_ns, insert_time_ns) tuple.
pub fn get_timing_stats() -> (u64, u64) {
    (
        PARSE_TIME_NS.load(Ordering::Relaxed),
        INSERT_TIME_NS.load(Ordering::Relaxed),
    )
}

/// Returns (record_build_ns, btree_insert_ns) tuple.
pub fn get_batch_timing_stats() -> (u64, u64) {
    (
        RECORD_BUILD_NS.load(Ordering::Relaxed),
        BTREE_INSERT_NS.load(Ordering::Relaxed),
    )
}
