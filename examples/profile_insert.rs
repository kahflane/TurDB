//! # Insert Path Profiler
//!
//! This example profiles the prepared statement insert path to identify
//! bottlenecks. It uses the timing infrastructure in `database::timing`
//! to measure time spent in each component of the insert operation.
//!
//! ## Usage
//!
//! ```bash
//! cargo run --example profile_insert --release
//! ```
//!
//! ## Output
//!
//! The profiler outputs a breakdown of time spent in:
//! - Storage lock acquisition
//! - Page 0 header reads
//! - Record building/serialization
//! - Transaction ID lookup
//! - MVCC record wrapping
//! - B-tree insert operation
//! - Secondary index updates
//! - Page 0 header updates
//!
//! The "unaccounted" time represents overhead not captured by the
//! individual measurements (function call overhead, loop iterations, etc).

use std::time::Instant;
use turdb::database::{get_insert_timing_breakdown, reset_timing_stats, Database};
use turdb::types::OwnedValue;

fn main() {
    let temp_dir = tempfile::tempdir().expect("failed to create temp dir");
    let db_path = temp_dir.path().join("profile_test.db");

    let db = Database::open_or_create(&db_path).expect("failed to open database");

    db.execute("CREATE TABLE bench (id INTEGER PRIMARY KEY, name TEXT, value REAL)")
        .expect("failed to create table");

    let wal_mode = std::env::var("WAL").unwrap_or_else(|_| "OFF".to_string());
    db.execute(&format!("PRAGMA WAL = {}", wal_mode))
        .expect("failed to set WAL mode");
    println!("WAL mode: {}", wal_mode);

    let stmt = db
        .prepare("INSERT INTO bench (id, name, value) VALUES (?, ?, ?)")
        .expect("failed to prepare");

    let insert_count = 10_000;

    println!("Warming up with 1000 inserts...");
    for i in 0..1000 {
        stmt.bind(OwnedValue::Int(i))
            .bind(OwnedValue::Text(format!("warmup_{}", i)))
            .bind(OwnedValue::Float(i as f64 * 1.5))
            .execute(&db)
            .expect("insert failed");
    }

    reset_timing_stats();

    println!("\nRunning {} profiled inserts...", insert_count);
    let total_start = Instant::now();

    for i in 1000..(1000 + insert_count) {
        stmt.bind(OwnedValue::Int(i))
            .bind(OwnedValue::Text(format!("name_{}", i)))
            .bind(OwnedValue::Float(i as f64 * 1.5))
            .execute(&db)
            .expect("insert failed");
    }

    let total_elapsed = total_start.elapsed();

    let mut breakdown = get_insert_timing_breakdown();
    breakdown.total_ns = total_elapsed.as_nanos() as u64;

    breakdown.print_breakdown();

    println!("\n=== Throughput ===");
    let throughput = insert_count as f64 / total_elapsed.as_secs_f64();
    println!(
        "Total time: {:.2} ms for {} inserts",
        total_elapsed.as_secs_f64() * 1000.0,
        insert_count
    );
    println!("Throughput: {:.0} inserts/sec", throughput);
    println!(
        "Per-insert: {:.2} µs",
        total_elapsed.as_secs_f64() * 1_000_000.0 / insert_count as f64
    );

    println!("\n=== Analysis ===");
    let avg = breakdown.avg_per_insert_ns();
    let total = breakdown.total_ns.max(1) as f64;

    let mut components = vec![
        ("Storage lock", avg.storage_lock_ns),
        ("Page 0 read", avg.page0_read_ns),
        ("Record build", avg.record_build_ns),
        ("Txn lookup", avg.txn_lookup_ns),
        ("MVCC wrap", avg.mvcc_wrap_ns),
        ("B-tree insert", avg.btree_insert_ns),
        ("Index update", avg.index_update_ns),
        ("Page 0 update", avg.page0_update_ns),
        ("WAL flush", avg.wal_flush_ns),
    ];

    components.sort_by(|a, b| b.1.cmp(&a.1));

    println!("\nTop bottlenecks (sorted by time):");
    for (name, ns) in &components {
        let pct = (*ns as f64 * insert_count as f64) / total * 100.0;
        println!("  {:<15} {:>6} ns/insert ({:>5.1}%)", name, ns, pct);
    }

    let accounted: u64 = components.iter().map(|(_, ns)| ns).sum();
    let unaccounted = avg.total_ns.saturating_sub(accounted);
    let unaccounted_pct = (unaccounted as f64 * insert_count as f64) / total * 100.0;
    println!(
        "  {:<15} {:>6} ns/insert ({:>5.1}%)",
        "Unaccounted", unaccounted, unaccounted_pct
    );
}
