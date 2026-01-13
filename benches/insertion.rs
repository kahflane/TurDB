//! # Insertion Benchmarks: TurDB vs SQLite
//!
//! This module provides apple-to-apple comparisons between TurDB and SQLite
//! for insertion operations under various configurations using SQL statements only.
//!
//! ## Test Matrix
//!
//! | Database | WAL Mode | Insert Type      | Configuration |
//! |----------|----------|------------------|---------------|
//! | SQLite   | ON       | Single (raw SQL) | Best practices|
//! | SQLite   | OFF      | Single (raw SQL) | Best practices|
//! | SQLite   | ON       | Prepared stmt    | Best practices|
//! | SQLite   | OFF      | Prepared stmt    | Best practices|
//! | SQLite   | ON       | Batch (multi-row)| Best practices|
//! | SQLite   | OFF      | Batch (multi-row)| Best practices|
//! | TurDB    | ON       | Single (raw SQL) | Best practices|
//! | TurDB    | OFF      | Single (raw SQL) | Best practices|
//! | TurDB    | ON       | Prepared stmt    | Best practices|
//! | TurDB    | OFF      | Prepared stmt    | Best practices|
//! | TurDB    | ON       | Batch (multi-row)| Best practices|
//! | TurDB    | OFF      | Batch (multi-row)| Best practices|
//!
//! ## SQLite Best Practices Configuration
//!
//! - `journal_mode = WAL` or `DELETE` (for WAL off)
//! - `synchronous = OFF` (benchmarks only, not production)
//! - `mmap_size = 268435456` (256MB for optimal I/O)
//! - `cache_size = -64000` (64MB page cache)
//! - `temp_store = MEMORY` (temp tables in memory)
//! - `page_size = 16384` (match TurDB page size)
//!
//! ## TurDB Best Practices Configuration
//!
//! - `PRAGMA WAL = ON/OFF`
//! - `PRAGMA SYNCHRONOUS = OFF` (benchmarks only)
//!
//! ## Benchmark Parameters
//!
//! - Single insert (raw SQL): 1,000 rows (full SQL parsing each time)
//! - Single insert (prepared): 10,000 rows (parse once, execute many)
//! - Batch insert: 100,000 rows (in batches of 1,000)
//!
//! ## Running Benchmarks
//!
//! ```bash
//! cargo bench --bench insertion
//! cargo bench --bench insertion -- single    # Only single insert benchmarks
//! cargo bench --bench insertion -- prepared  # Only prepared statement benchmarks
//! cargo bench --bench insertion -- batch     # Only batch insert benchmarks
//! ```
//!
//! ## Notes on Fair Comparison
//!
//! All benchmarks use pure SQL statements - no native Rust APIs.
//! Both databases use:
//! - Same page size (16KB)
//! - Same synchronous mode (OFF)
//! - Equivalent WAL configurations
//! - Transaction wrapping
//! - Pre-created tables with identical schemas

use criterion::{
    black_box, criterion_group, criterion_main, BatchSize, BenchmarkId, Criterion, Throughput,
};
use rusqlite::Connection;
use std::path::Path;
use tempfile::TempDir;
use turdb::{Database, OwnedValue};

const SINGLE_RAW_SQL_ROWS: u64 = 1_000;
const SINGLE_PREPARED_ROWS: u64 = 10_000;
const BATCH_INSERT_ROWS: u64 = 100_000;
const BATCH_SIZE: usize = 1_000;

fn setup_sqlite_wal_on(path: &Path) -> Connection {
    let conn = Connection::open(path).expect("Failed to open SQLite");

    conn.execute_batch(
        "PRAGMA page_size = 16384;
         PRAGMA journal_mode = WAL;
         PRAGMA synchronous = OFF;
         PRAGMA mmap_size = 268435456;
         PRAGMA cache_size = -64000;
         PRAGMA temp_store = MEMORY;
         PRAGMA wal_autocheckpoint = 10000;",
    )
    .expect("Failed to set SQLite pragmas");

    conn.execute(
        "CREATE TABLE IF NOT EXISTS test_data (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            value REAL NOT NULL,
            data BLOB
        )",
        [],
    )
    .expect("Failed to create SQLite table");

    conn
}

fn setup_sqlite_wal_off(path: &Path) -> Connection {
    let conn = Connection::open(path).expect("Failed to open SQLite");

    conn.execute_batch(
        "PRAGMA page_size = 16384;
         PRAGMA journal_mode = DELETE;
         PRAGMA synchronous = OFF;
         PRAGMA mmap_size = 268435456;
         PRAGMA cache_size = -64000;
         PRAGMA temp_store = MEMORY;",
    )
    .expect("Failed to set SQLite pragmas");

    conn.execute(
        "CREATE TABLE IF NOT EXISTS test_data (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            value REAL NOT NULL,
            data BLOB
        )",
        [],
    )
    .expect("Failed to create SQLite table");

    conn
}

fn setup_turdb_wal_on(path: &Path) -> Database {
    let db = Database::create(path).expect("Failed to create TurDB");

    db.execute("PRAGMA WAL = ON").expect("Failed to enable WAL");
    db.execute("PRAGMA SYNCHRONOUS = OFF")
        .expect("Failed to set synchronous mode");
    db.execute("PRAGMA WAL_AUTOFLUSH = OFF")
        .expect("Failed to disable autoflush");

    db.execute(
        "CREATE TABLE test_data (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            value REAL NOT NULL,
            data BLOB
        )",
    )
    .expect("Failed to create TurDB table");

    db
}

fn setup_turdb_wal_off(path: &Path) -> Database {
    let db = Database::create(path).expect("Failed to create TurDB");

    db.execute("PRAGMA WAL = OFF")
        .expect("Failed to disable WAL");
    db.execute("PRAGMA SYNCHRONOUS = OFF")
        .expect("Failed to set synchronous mode");

    db.execute(
        "CREATE TABLE test_data (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            value REAL NOT NULL,
            data BLOB
        )",
    )
    .expect("Failed to create TurDB table");

    db
}

fn generate_test_name(i: u64) -> String {
    format!("user_{:08}", i)
}

fn generate_hex_blob() -> String {
    "X'ABABABABABABABABABABABABABABABABABABABABABABABABABABABABABABABABABABABABABABABABABABABABABABABABABABABABABABABABABABABABABABABAB'".to_string()
}

fn bench_single_raw_sql(c: &mut Criterion) {
    let mut group = c.benchmark_group("single_raw_sql");
    group.throughput(Throughput::Elements(SINGLE_RAW_SQL_ROWS));
    group.sample_size(10);

    group.bench_function(BenchmarkId::new("sqlite", "wal_on"), |b| {
        b.iter_batched(
            || {
                let dir = TempDir::new().unwrap();
                let db_path = dir.path().join("sqlite_bench.db");
                let conn = setup_sqlite_wal_on(&db_path);
                (dir, conn)
            },
            |(dir, conn)| {
                conn.execute("BEGIN TRANSACTION", []).unwrap();
                for i in 0..SINGLE_RAW_SQL_ROWS {
                    let sql = format!(
                        "INSERT INTO test_data (id, name, value, data) VALUES ({}, '{}', {}, {})",
                        i,
                        generate_test_name(i),
                        i as f64 * 1.5,
                        generate_hex_blob()
                    );
                    conn.execute(&sql, []).unwrap();
                }
                conn.execute("COMMIT", []).unwrap();
                black_box((dir, conn))
            },
            BatchSize::PerIteration,
        );
    });

    group.bench_function(BenchmarkId::new("sqlite", "wal_off"), |b| {
        b.iter_batched(
            || {
                let dir = TempDir::new().unwrap();
                let db_path = dir.path().join("sqlite_bench.db");
                let conn = setup_sqlite_wal_off(&db_path);
                (dir, conn)
            },
            |(dir, conn)| {
                conn.execute("BEGIN TRANSACTION", []).unwrap();
                for i in 0..SINGLE_RAW_SQL_ROWS {
                    let sql = format!(
                        "INSERT INTO test_data (id, name, value, data) VALUES ({}, '{}', {}, {})",
                        i,
                        generate_test_name(i),
                        i as f64 * 1.5,
                        generate_hex_blob()
                    );
                    conn.execute(&sql, []).unwrap();
                }
                conn.execute("COMMIT", []).unwrap();
                black_box((dir, conn))
            },
            BatchSize::PerIteration,
        );
    });

    group.bench_function(BenchmarkId::new("turdb", "wal_on"), |b| {
        b.iter_batched(
            || {
                let dir = TempDir::new().unwrap();
                let db_path = dir.path().join("turdb_bench");
                let db = setup_turdb_wal_on(&db_path);
                (dir, db)
            },
            |(dir, db)| {
                db.execute("BEGIN").unwrap();
                for i in 0..SINGLE_RAW_SQL_ROWS {
                    let sql = format!(
                        "INSERT INTO test_data (id, name, value, data) VALUES ({}, '{}', {}, {})",
                        i,
                        generate_test_name(i),
                        i as f64 * 1.5,
                        generate_hex_blob()
                    );
                    db.execute(&sql).unwrap();
                }
                db.execute("COMMIT").unwrap();
                black_box((dir, db))
            },
            BatchSize::PerIteration,
        );
    });

    group.bench_function(BenchmarkId::new("turdb", "wal_off"), |b| {
        b.iter_batched(
            || {
                let dir = TempDir::new().unwrap();
                let db_path = dir.path().join("turdb_bench");
                let db = setup_turdb_wal_off(&db_path);
                (dir, db)
            },
            |(dir, db)| {
                db.execute("BEGIN").unwrap();
                for i in 0..SINGLE_RAW_SQL_ROWS {
                    let sql = format!(
                        "INSERT INTO test_data (id, name, value, data) VALUES ({}, '{}', {}, {})",
                        i,
                        generate_test_name(i),
                        i as f64 * 1.5,
                        generate_hex_blob()
                    );
                    db.execute(&sql).unwrap();
                }
                db.execute("COMMIT").unwrap();
                black_box((dir, db))
            },
            BatchSize::PerIteration,
        );
    });

    group.finish();
}

fn bench_single_prepared(c: &mut Criterion) {
    let mut group = c.benchmark_group("single_prepared");
    group.throughput(Throughput::Elements(SINGLE_PREPARED_ROWS));
    group.sample_size(10);

    group.bench_function(BenchmarkId::new("sqlite", "wal_on"), |b| {
        b.iter_batched(
            || {
                let dir = TempDir::new().unwrap();
                let db_path = dir.path().join("sqlite_bench.db");
                let conn = setup_sqlite_wal_on(&db_path);
                (dir, conn)
            },
            |(dir, conn)| {
                conn.execute("BEGIN TRANSACTION", []).unwrap();
                {
                    let mut stmt = conn
                        .prepare_cached(
                            "INSERT INTO test_data (id, name, value, data) VALUES (?1, ?2, ?3, ?4)",
                        )
                        .unwrap();

                    for i in 0..SINGLE_PREPARED_ROWS {
                        let name = generate_test_name(i);
                        let blob = vec![0xABu8; 64];
                        stmt.execute(rusqlite::params![i as i64, name, i as f64 * 1.5, blob])
                            .unwrap();
                    }
                }
                conn.execute("COMMIT", []).unwrap();
                black_box((dir, conn))
            },
            BatchSize::PerIteration,
        );
    });

    group.bench_function(BenchmarkId::new("sqlite", "wal_off"), |b| {
        b.iter_batched(
            || {
                let dir = TempDir::new().unwrap();
                let db_path = dir.path().join("sqlite_bench.db");
                let conn = setup_sqlite_wal_off(&db_path);
                (dir, conn)
            },
            |(dir, conn)| {
                conn.execute("BEGIN TRANSACTION", []).unwrap();
                {
                    let mut stmt = conn
                        .prepare_cached(
                            "INSERT INTO test_data (id, name, value, data) VALUES (?1, ?2, ?3, ?4)",
                        )
                        .unwrap();

                    for i in 0..SINGLE_PREPARED_ROWS {
                        let name = generate_test_name(i);
                        let blob = vec![0xABu8; 64];
                        stmt.execute(rusqlite::params![i as i64, name, i as f64 * 1.5, blob])
                            .unwrap();
                    }
                }
                conn.execute("COMMIT", []).unwrap();
                black_box((dir, conn))
            },
            BatchSize::PerIteration,
        );
    });

    group.bench_function(BenchmarkId::new("turdb", "wal_on"), |b| {
        b.iter_batched(
            || {
                let dir = TempDir::new().unwrap();
                let db_path = dir.path().join("turdb_bench");
                let db = setup_turdb_wal_on(&db_path);
                (dir, db)
            },
            |(dir, db)| {
                db.execute("BEGIN").unwrap();
                let stmt = db
                    .prepare("INSERT INTO test_data (id, name, value, data) VALUES (?, ?, ?, ?)")
                    .unwrap();

                for i in 0..SINGLE_PREPARED_ROWS {
                    let name = generate_test_name(i);
                    let blob = vec![0xABu8; 64];
                    stmt.bind(OwnedValue::Int(i as i64))
                        .bind(OwnedValue::Text(name))
                        .bind(OwnedValue::Float(i as f64 * 1.5))
                        .bind(OwnedValue::Blob(blob))
                        .execute(&db)
                        .unwrap();
                }
                db.execute("COMMIT").unwrap();
                black_box((dir, db))
            },
            BatchSize::PerIteration,
        );
    });

    group.bench_function(BenchmarkId::new("turdb", "wal_off"), |b| {
        b.iter_batched(
            || {
                let dir = TempDir::new().unwrap();
                let db_path = dir.path().join("turdb_bench");
                let db = setup_turdb_wal_off(&db_path);
                (dir, db)
            },
            |(dir, db)| {
                db.execute("BEGIN").unwrap();
                let stmt = db
                    .prepare("INSERT INTO test_data (id, name, value, data) VALUES (?, ?, ?, ?)")
                    .unwrap();

                for i in 0..SINGLE_PREPARED_ROWS {
                    let name = generate_test_name(i);
                    let blob = vec![0xABu8; 64];
                    stmt.bind(OwnedValue::Int(i as i64))
                        .bind(OwnedValue::Text(name))
                        .bind(OwnedValue::Float(i as f64 * 1.5))
                        .bind(OwnedValue::Blob(blob))
                        .execute(&db)
                        .unwrap();
                }
                db.execute("COMMIT").unwrap();
                black_box((dir, db))
            },
            BatchSize::PerIteration,
        );
    });

    group.finish();
}

fn bench_batch_insert(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch_insert");
    group.throughput(Throughput::Elements(BATCH_INSERT_ROWS));
    group.sample_size(10);

    group.bench_function(BenchmarkId::new("sqlite", "wal_on"), |b| {
        b.iter_batched(
            || {
                let dir = TempDir::new().unwrap();
                let db_path = dir.path().join("sqlite_bench.db");
                let conn = setup_sqlite_wal_on(&db_path);
                (dir, conn)
            },
            |(dir, conn)| {
                let batch_count = (BATCH_INSERT_ROWS as usize) / BATCH_SIZE;

                for batch_idx in 0..batch_count {
                    conn.execute("BEGIN TRANSACTION", []).unwrap();

                    let start = batch_idx * BATCH_SIZE;
                    let mut sql = String::from(
                        "INSERT INTO test_data (id, name, value, data) VALUES ",
                    );

                    for (j, i) in (start..(start + BATCH_SIZE)).enumerate() {
                        if j > 0 {
                            sql.push_str(", ");
                        }
                        sql.push_str(&format!(
                            "({}, '{}', {}, {})",
                            i,
                            generate_test_name(i as u64),
                            i as f64 * 1.5,
                            generate_hex_blob()
                        ));
                    }

                    conn.execute(&sql, []).unwrap();
                    conn.execute("COMMIT", []).unwrap();
                }
                black_box((dir, conn))
            },
            BatchSize::PerIteration,
        );
    });

    group.bench_function(BenchmarkId::new("sqlite", "wal_off"), |b| {
        b.iter_batched(
            || {
                let dir = TempDir::new().unwrap();
                let db_path = dir.path().join("sqlite_bench.db");
                let conn = setup_sqlite_wal_off(&db_path);
                (dir, conn)
            },
            |(dir, conn)| {
                let batch_count = (BATCH_INSERT_ROWS as usize) / BATCH_SIZE;

                for batch_idx in 0..batch_count {
                    conn.execute("BEGIN TRANSACTION", []).unwrap();

                    let start = batch_idx * BATCH_SIZE;
                    let mut sql = String::from(
                        "INSERT INTO test_data (id, name, value, data) VALUES ",
                    );

                    for (j, i) in (start..(start + BATCH_SIZE)).enumerate() {
                        if j > 0 {
                            sql.push_str(", ");
                        }
                        sql.push_str(&format!(
                            "({}, '{}', {}, {})",
                            i,
                            generate_test_name(i as u64),
                            i as f64 * 1.5,
                            generate_hex_blob()
                        ));
                    }

                    conn.execute(&sql, []).unwrap();
                    conn.execute("COMMIT", []).unwrap();
                }
                black_box((dir, conn))
            },
            BatchSize::PerIteration,
        );
    });

    group.bench_function(BenchmarkId::new("turdb", "wal_on"), |b| {
        b.iter_batched(
            || {
                let dir = TempDir::new().unwrap();
                let db_path = dir.path().join("turdb_bench");
                let db = setup_turdb_wal_on(&db_path);
                (dir, db)
            },
            |(dir, db)| {
                let batch_count = (BATCH_INSERT_ROWS as usize) / BATCH_SIZE;

                for batch_idx in 0..batch_count {
                    db.execute("BEGIN").unwrap();

                    let start = batch_idx * BATCH_SIZE;
                    let mut sql = String::from(
                        "INSERT INTO test_data (id, name, value, data) VALUES ",
                    );

                    for (j, i) in (start..(start + BATCH_SIZE)).enumerate() {
                        if j > 0 {
                            sql.push_str(", ");
                        }
                        sql.push_str(&format!(
                            "({}, '{}', {}, {})",
                            i,
                            generate_test_name(i as u64),
                            i as f64 * 1.5,
                            generate_hex_blob()
                        ));
                    }

                    db.execute(&sql).unwrap();
                    db.execute("COMMIT").unwrap();
                }
                black_box((dir, db))
            },
            BatchSize::PerIteration,
        );
    });

    group.bench_function(BenchmarkId::new("turdb", "wal_off"), |b| {
        b.iter_batched(
            || {
                let dir = TempDir::new().unwrap();
                let db_path = dir.path().join("turdb_bench");
                let db = setup_turdb_wal_off(&db_path);
                (dir, db)
            },
            |(dir, db)| {
                let batch_count = (BATCH_INSERT_ROWS as usize) / BATCH_SIZE;

                for batch_idx in 0..batch_count {
                    db.execute("BEGIN").unwrap();

                    let start = batch_idx * BATCH_SIZE;
                    let mut sql = String::from(
                        "INSERT INTO test_data (id, name, value, data) VALUES ",
                    );

                    for (j, i) in (start..(start + BATCH_SIZE)).enumerate() {
                        if j > 0 {
                            sql.push_str(", ");
                        }
                        sql.push_str(&format!(
                            "({}, '{}', {}, {})",
                            i,
                            generate_test_name(i as u64),
                            i as f64 * 1.5,
                            generate_hex_blob()
                        ));
                    }

                    db.execute(&sql).unwrap();
                    db.execute("COMMIT").unwrap();
                }
                black_box((dir, db))
            },
            BatchSize::PerIteration,
        );
    });

    group.finish();
}

fn bench_batch_prepared(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch_prepared");
    group.throughput(Throughput::Elements(BATCH_INSERT_ROWS));
    group.sample_size(10);

    group.bench_function(BenchmarkId::new("sqlite", "wal_on"), |b| {
        b.iter_batched(
            || {
                let dir = TempDir::new().unwrap();
                let db_path = dir.path().join("sqlite_bench.db");
                let conn = setup_sqlite_wal_on(&db_path);
                (dir, conn)
            },
            |(dir, conn)| {
                let batch_count = (BATCH_INSERT_ROWS as usize) / BATCH_SIZE;

                for batch_idx in 0..batch_count {
                    conn.execute("BEGIN TRANSACTION", []).unwrap();
                    {
                        let mut stmt = conn
                            .prepare_cached(
                                "INSERT INTO test_data (id, name, value, data) VALUES (?1, ?2, ?3, ?4)",
                            )
                            .unwrap();

                        let start = batch_idx * BATCH_SIZE;
                        for i in start..(start + BATCH_SIZE) {
                            let name = generate_test_name(i as u64);
                            let blob = vec![0xABu8; 64];
                            stmt.execute(rusqlite::params![i as i64, name, i as f64 * 1.5, blob])
                                .unwrap();
                        }
                    }
                    conn.execute("COMMIT", []).unwrap();
                }
                black_box((dir, conn))
            },
            BatchSize::PerIteration,
        );
    });

    group.bench_function(BenchmarkId::new("sqlite", "wal_off"), |b| {
        b.iter_batched(
            || {
                let dir = TempDir::new().unwrap();
                let db_path = dir.path().join("sqlite_bench.db");
                let conn = setup_sqlite_wal_off(&db_path);
                (dir, conn)
            },
            |(dir, conn)| {
                let batch_count = (BATCH_INSERT_ROWS as usize) / BATCH_SIZE;

                for batch_idx in 0..batch_count {
                    conn.execute("BEGIN TRANSACTION", []).unwrap();
                    {
                        let mut stmt = conn
                            .prepare_cached(
                                "INSERT INTO test_data (id, name, value, data) VALUES (?1, ?2, ?3, ?4)",
                            )
                            .unwrap();

                        let start = batch_idx * BATCH_SIZE;
                        for i in start..(start + BATCH_SIZE) {
                            let name = generate_test_name(i as u64);
                            let blob = vec![0xABu8; 64];
                            stmt.execute(rusqlite::params![i as i64, name, i as f64 * 1.5, blob])
                                .unwrap();
                        }
                    }
                    conn.execute("COMMIT", []).unwrap();
                }
                black_box((dir, conn))
            },
            BatchSize::PerIteration,
        );
    });

    group.bench_function(BenchmarkId::new("turdb", "wal_on"), |b| {
        b.iter_batched(
            || {
                let dir = TempDir::new().unwrap();
                let db_path = dir.path().join("turdb_bench");
                let db = setup_turdb_wal_on(&db_path);
                (dir, db)
            },
            |(dir, db)| {
                let batch_count = (BATCH_INSERT_ROWS as usize) / BATCH_SIZE;

                for batch_idx in 0..batch_count {
                    db.execute("BEGIN").unwrap();
                    let stmt = db
                        .prepare("INSERT INTO test_data (id, name, value, data) VALUES (?, ?, ?, ?)")
                        .unwrap();

                    let start = batch_idx * BATCH_SIZE;
                    for i in start..(start + BATCH_SIZE) {
                        let name = generate_test_name(i as u64);
                        let blob = vec![0xABu8; 64];
                        stmt.bind(OwnedValue::Int(i as i64))
                            .bind(OwnedValue::Text(name))
                            .bind(OwnedValue::Float(i as f64 * 1.5))
                            .bind(OwnedValue::Blob(blob))
                            .execute(&db)
                            .unwrap();
                    }
                    db.execute("COMMIT").unwrap();
                }
                black_box((dir, db))
            },
            BatchSize::PerIteration,
        );
    });

    group.bench_function(BenchmarkId::new("turdb", "wal_off"), |b| {
        b.iter_batched(
            || {
                let dir = TempDir::new().unwrap();
                let db_path = dir.path().join("turdb_bench");
                let db = setup_turdb_wal_off(&db_path);
                (dir, db)
            },
            |(dir, db)| {
                let batch_count = (BATCH_INSERT_ROWS as usize) / BATCH_SIZE;

                for batch_idx in 0..batch_count {
                    db.execute("BEGIN").unwrap();
                    let stmt = db
                        .prepare("INSERT INTO test_data (id, name, value, data) VALUES (?, ?, ?, ?)")
                        .unwrap();

                    let start = batch_idx * BATCH_SIZE;
                    for i in start..(start + BATCH_SIZE) {
                        let name = generate_test_name(i as u64);
                        let blob = vec![0xABu8; 64];
                        stmt.bind(OwnedValue::Int(i as i64))
                            .bind(OwnedValue::Text(name))
                            .bind(OwnedValue::Float(i as f64 * 1.5))
                            .bind(OwnedValue::Blob(blob))
                            .execute(&db)
                            .unwrap();
                    }
                    db.execute("COMMIT").unwrap();
                }
                black_box((dir, db))
            },
            BatchSize::PerIteration,
        );
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_single_raw_sql,
    bench_single_prepared,
    bench_batch_insert,
    bench_batch_prepared
);
criterion_main!(benches);
