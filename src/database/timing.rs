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

pub(crate) static STORAGE_LOCK_NS: AtomicU64 = AtomicU64::new(0);
pub(crate) static PAGE0_READ_NS: AtomicU64 = AtomicU64::new(0);
pub(crate) static MVCC_WRAP_NS: AtomicU64 = AtomicU64::new(0);
pub(crate) static TXN_LOOKUP_NS: AtomicU64 = AtomicU64::new(0);
pub(crate) static INDEX_UPDATE_NS: AtomicU64 = AtomicU64::new(0);
pub(crate) static PAGE0_UPDATE_NS: AtomicU64 = AtomicU64::new(0);
pub(crate) static WAL_FLUSH_NS: AtomicU64 = AtomicU64::new(0);

pub(crate) static INSERT_COUNT: AtomicU64 = AtomicU64::new(0);

/// Resets all timing statistics to zero.
pub fn reset_timing_stats() {
    PARSE_TIME_NS.store(0, Ordering::Relaxed);
    INSERT_TIME_NS.store(0, Ordering::Relaxed);
    RECORD_BUILD_NS.store(0, Ordering::Relaxed);
    BTREE_INSERT_NS.store(0, Ordering::Relaxed);
    STORAGE_LOCK_NS.store(0, Ordering::Relaxed);
    PAGE0_READ_NS.store(0, Ordering::Relaxed);
    MVCC_WRAP_NS.store(0, Ordering::Relaxed);
    TXN_LOOKUP_NS.store(0, Ordering::Relaxed);
    INDEX_UPDATE_NS.store(0, Ordering::Relaxed);
    PAGE0_UPDATE_NS.store(0, Ordering::Relaxed);
    WAL_FLUSH_NS.store(0, Ordering::Relaxed);
    INSERT_COUNT.store(0, Ordering::Relaxed);
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

#[derive(Debug, Clone, Copy)]
pub struct InsertTimingBreakdown {
    pub total_ns: u64,
    pub storage_lock_ns: u64,
    pub page0_read_ns: u64,
    pub record_build_ns: u64,
    pub txn_lookup_ns: u64,
    pub mvcc_wrap_ns: u64,
    pub btree_insert_ns: u64,
    pub index_update_ns: u64,
    pub page0_update_ns: u64,
    pub wal_flush_ns: u64,
    pub insert_count: u64,
}

impl InsertTimingBreakdown {
    pub fn avg_per_insert_ns(&self) -> InsertTimingBreakdown {
        let count = self.insert_count.max(1);
        InsertTimingBreakdown {
            total_ns: self.total_ns / count,
            storage_lock_ns: self.storage_lock_ns / count,
            page0_read_ns: self.page0_read_ns / count,
            record_build_ns: self.record_build_ns / count,
            txn_lookup_ns: self.txn_lookup_ns / count,
            mvcc_wrap_ns: self.mvcc_wrap_ns / count,
            btree_insert_ns: self.btree_insert_ns / count,
            index_update_ns: self.index_update_ns / count,
            page0_update_ns: self.page0_update_ns / count,
            wal_flush_ns: self.wal_flush_ns / count,
            insert_count: 1,
        }
    }

    pub fn print_breakdown(&self) {
        let total = self.total_ns.max(1) as f64;
        println!("\n=== Insert Timing Breakdown ({} inserts) ===", self.insert_count);
        println!("Total:         {:>10} ns ({:>6.2} ms)", self.total_ns, self.total_ns as f64 / 1_000_000.0);
        println!("─────────────────────────────────────────────");
        println!("Storage lock:  {:>10} ns ({:>5.1}%)", self.storage_lock_ns, self.storage_lock_ns as f64 / total * 100.0);
        println!("Page 0 read:   {:>10} ns ({:>5.1}%)", self.page0_read_ns, self.page0_read_ns as f64 / total * 100.0);
        println!("Record build:  {:>10} ns ({:>5.1}%)", self.record_build_ns, self.record_build_ns as f64 / total * 100.0);
        println!("Txn lookup:    {:>10} ns ({:>5.1}%)", self.txn_lookup_ns, self.txn_lookup_ns as f64 / total * 100.0);
        println!("MVCC wrap:     {:>10} ns ({:>5.1}%)", self.mvcc_wrap_ns, self.mvcc_wrap_ns as f64 / total * 100.0);
        println!("B-tree insert: {:>10} ns ({:>5.1}%)", self.btree_insert_ns, self.btree_insert_ns as f64 / total * 100.0);
        println!("Index update:  {:>10} ns ({:>5.1}%)", self.index_update_ns, self.index_update_ns as f64 / total * 100.0);
        println!("Page 0 update: {:>10} ns ({:>5.1}%)", self.page0_update_ns, self.page0_update_ns as f64 / total * 100.0);
        println!("WAL flush:     {:>10} ns ({:>5.1}%)", self.wal_flush_ns, self.wal_flush_ns as f64 / total * 100.0);

        let accounted = self.storage_lock_ns + self.page0_read_ns + self.record_build_ns
            + self.txn_lookup_ns + self.mvcc_wrap_ns + self.btree_insert_ns
            + self.index_update_ns + self.page0_update_ns + self.wal_flush_ns;
        let unaccounted = self.total_ns.saturating_sub(accounted);
        println!("Unaccounted:   {:>10} ns ({:>5.1}%)", unaccounted, unaccounted as f64 / total * 100.0);
        println!("─────────────────────────────────────────────");

        if self.insert_count > 0 {
            let avg = self.avg_per_insert_ns();
            println!("\nPer-insert average: {} ns ({:.2} µs)", avg.total_ns, avg.total_ns as f64 / 1000.0);
        }
    }
}

pub fn get_insert_timing_breakdown() -> InsertTimingBreakdown {
    InsertTimingBreakdown {
        total_ns: INSERT_TIME_NS.load(Ordering::Relaxed),
        storage_lock_ns: STORAGE_LOCK_NS.load(Ordering::Relaxed),
        page0_read_ns: PAGE0_READ_NS.load(Ordering::Relaxed),
        record_build_ns: RECORD_BUILD_NS.load(Ordering::Relaxed),
        txn_lookup_ns: TXN_LOOKUP_NS.load(Ordering::Relaxed),
        mvcc_wrap_ns: MVCC_WRAP_NS.load(Ordering::Relaxed),
        btree_insert_ns: BTREE_INSERT_NS.load(Ordering::Relaxed),
        index_update_ns: INDEX_UPDATE_NS.load(Ordering::Relaxed),
        page0_update_ns: PAGE0_UPDATE_NS.load(Ordering::Relaxed),
        wal_flush_ns: WAL_FLUSH_NS.load(Ordering::Relaxed),
        insert_count: INSERT_COUNT.load(Ordering::Relaxed),
    }
}
