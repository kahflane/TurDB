//! SQLite vs TurDB Benchmark - Self-contained version
//!
//! Tests insert performance comparison without requiring external data files.

use rusqlite::Connection;
use std::time::Instant;
use turdb::{Database, OwnedValue, reset_timing_stats, get_batch_timing_stats, reset_fastpath_stats, get_fastpath_stats};

fn rusqlite_to_owned(val: &rusqlite::types::Value) -> OwnedValue {
    match val {
        rusqlite::types::Value::Null => OwnedValue::Null,
        rusqlite::types::Value::Integer(i) => OwnedValue::Int(*i),
        rusqlite::types::Value::Real(f) => {
            if f.is_nan() || f.is_infinite() {
                OwnedValue::Null
            } else {
                OwnedValue::Float(*f)
            }
        }
        rusqlite::types::Value::Text(s) => OwnedValue::Text(s.clone()),
        rusqlite::types::Value::Blob(b) => OwnedValue::Blob(b.clone()),
    }
}

#[test]
fn benchmark_sqlite_vs_turdb_synthetic() {
    const ROW_COUNT: usize = 1_000_000;
    const BATCH_SIZE: usize = 5000;

    let sqlite_target = tempfile::NamedTempFile::new().unwrap();
    let turdb_dir = tempfile::tempdir().unwrap();

    println!("\n============================================================");
    println!("  SQLite vs TurDB Benchmark - {} rows", ROW_COUNT);
    println!("============================================================\n");

    // Generate synthetic test data
    println!("Generating {} rows of synthetic data...", ROW_COUNT);
    let gen_start = Instant::now();

    let rows: Vec<Vec<rusqlite::types::Value>> = (0..ROW_COUNT)
        .map(|i| {
            vec![
                rusqlite::types::Value::Integer(i as i64),                          // id
                rusqlite::types::Value::Integer((i % 10000) as i64),                // dataset_id
                rusqlite::types::Value::Integer((i % 5000) as i64),                 // datasource_version_id
                rusqlite::types::Value::Integer((i % 1000) as i64),                 // creator_user_id
                rusqlite::types::Value::Text(format!("license_{}", i % 20)),        // license_name
                rusqlite::types::Value::Text(format!("2024-{:02}-{:02}", (i % 12) + 1, (i % 28) + 1)), // creation_date
                rusqlite::types::Value::Real((i as f64) / 100.0),                   // version_number
                rusqlite::types::Value::Text(format!("Dataset Title {} with some extra text for realism", i)), // title
                rusqlite::types::Value::Text(format!("dataset-slug-{}", i)),        // slug
                rusqlite::types::Value::Text(format!("Subtitle for dataset {}", i)), // subtitle
                rusqlite::types::Value::Real((i * 1024) as f64),                    // total_compressed_bytes
                rusqlite::types::Value::Real((i * 2048) as f64),                    // total_uncompressed_bytes
            ]
        })
        .collect();

    println!("Generated {} rows in {:.2}s\n", rows.len(), gen_start.elapsed().as_secs_f64());

    // ========== SQLITE BENCHMARK (Optimized for bulk insert) ==========
    println!("--- SQLite Insert Benchmark (Optimized) ---");
    let sqlite_conn = Connection::open(sqlite_target.path()).expect("Failed to create SQLite DB");

    sqlite_conn.execute_batch("
        PRAGMA journal_mode = OFF;
        PRAGMA synchronous = OFF;
        PRAGMA cache_size = -64000;
        PRAGMA temp_store = MEMORY;
        PRAGMA locking_mode = EXCLUSIVE;
        PRAGMA mmap_size = 268435456;
        CREATE TABLE dataset_versions (
            id INTEGER PRIMARY KEY,
            dataset_id INTEGER,
            datasource_version_id INTEGER,
            creator_user_id INTEGER,
            license_name TEXT,
            creation_date TEXT,
            version_number REAL,
            title TEXT,
            slug TEXT,
            subtitle TEXT,
            total_compressed_bytes REAL,
            total_uncompressed_bytes REAL
        );
    ").expect("Failed to create SQLite table");

    let sqlite_start = Instant::now();

    {
        let mut insert_stmt = sqlite_conn.prepare_cached(
            "INSERT INTO dataset_versions VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
        ).expect("Failed to prepare insert");

        sqlite_conn.execute("BEGIN IMMEDIATE", []).expect("BEGIN failed");

        for row in &rows {
            insert_stmt.execute(rusqlite::params_from_iter(row.iter())).expect("Insert failed");
        }

        sqlite_conn.execute("COMMIT", []).expect("COMMIT failed");
    }

    let sqlite_elapsed = sqlite_start.elapsed();
    let sqlite_rows_per_sec = rows.len() as f64 / sqlite_elapsed.as_secs_f64();
    println!("SQLite: {} rows in {:.3}s = {:.0} rows/sec\n",
        rows.len(), sqlite_elapsed.as_secs_f64(), sqlite_rows_per_sec);

    // ========== TURDB BENCHMARK ==========
    println!("--- TurDB Insert Benchmark (batch API) ---");
    let turdb = Database::create(turdb_dir.path().join("benchmark")).expect("Failed to create TurDB");

    turdb.execute("
        CREATE TABLE dataset_versions (
          id BIGINT,
          dataset_id BIGINT,
          datasource_version_id BIGINT,
          creator_user_id BIGINT,
          license_name VARCHAR(100),
          creation_date VARCHAR(20),
          version_number FLOAT,
          title VARCHAR(300),
          slug VARCHAR(100),
          subtitle VARCHAR(100),
          total_compressed_bytes FLOAT,
          total_uncompressed_bytes FLOAT
        )
    ").expect("Failed to create TurDB table");

    reset_timing_stats();
    reset_fastpath_stats();
    let turdb_start = Instant::now();
    let mut convert_time = std::time::Duration::ZERO;
    let mut insert_time = std::time::Duration::ZERO;

    for batch in rows.chunks(BATCH_SIZE) {
        let convert_start = Instant::now();
        let owned_rows: Vec<Vec<turdb::OwnedValue>> = batch.iter().map(|row| {
            row.iter().map(rusqlite_to_owned).collect()
        }).collect();
        convert_time += convert_start.elapsed();

        let insert_start = Instant::now();
        turdb.insert_batch("dataset_versions", &owned_rows).expect("TurDB batch insert failed");
        insert_time += insert_start.elapsed();
    }

    let (record_ns, btree_ns) = get_batch_timing_stats();
    let record_time = std::time::Duration::from_nanos(record_ns);
    let btree_time = std::time::Duration::from_nanos(btree_ns);
    let (fastpath_hits, fastpath_misses) = get_fastpath_stats();

    let turdb_elapsed = turdb_start.elapsed();
    let turdb_rows_per_sec = rows.len() as f64 / turdb_elapsed.as_secs_f64();

    println!("  Convert: {:.3}s, Insert: {:.3}s",
        convert_time.as_secs_f64(), insert_time.as_secs_f64());
    println!("  -> Record build: {:.3}s, BTree insert: {:.3}s",
        record_time.as_secs_f64(), btree_time.as_secs_f64());
    if fastpath_hits + fastpath_misses > 0 {
        println!("  -> Fastpath: {} hits, {} misses ({:.2}% hit rate)",
            fastpath_hits, fastpath_misses,
            100.0 * fastpath_hits as f64 / (fastpath_hits + fastpath_misses) as f64);
    }

    println!("TurDB:  {} rows in {:.3}s = {:.0} rows/sec\n",
        rows.len(), turdb_elapsed.as_secs_f64(), turdb_rows_per_sec);

    // ========== COMPARISON ==========
    println!("============================================================");
    println!("  RESULTS");
    println!("============================================================");
    println!("SQLite: {:.3}s ({:.0} rows/sec)",
        sqlite_elapsed.as_secs_f64(), sqlite_rows_per_sec);
    println!("TurDB:  {:.3}s ({:.0} rows/sec)",
        turdb_elapsed.as_secs_f64(), turdb_rows_per_sec);

    let ratio = sqlite_elapsed.as_secs_f64() / turdb_elapsed.as_secs_f64();
    if ratio > 1.0 {
        println!("\n✓ TurDB is {:.2}x FASTER than SQLite!", ratio);
    } else {
        println!("\n✗ TurDB is {:.2}x SLOWER than SQLite (target: 2x faster)", 1.0/ratio);
    }
    println!("============================================================\n");

    // Note: SQLite is using aggressive optimizations (journal_mode=OFF, synchronous=OFF)
    // that trade durability for speed. A fair comparison would need similar settings.
    // The primary goal of this test is to verify functional correctness.
    println!("\nNote: SQLite benchmark uses aggressive optimizations (journal_mode=OFF, synchronous=OFF)");
    println!("TurDB is using default settings. Performance comparison may not be apples-to-apples.");
}
