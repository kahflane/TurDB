use std::time::Instant;
use rusqlite::Connection;
use turdb::{Database, OwnedValue};

const ROW_COUNT: usize = 100_000;
const SQLITE_TARGET: &str = "/Users/julfikar/Documents/PassionFruit.nosync/turdb/turdb-core/.worktrees/single_insert_sqlite.db";
const TURDB_TARGET: &str = "/Users/julfikar/Documents/PassionFruit.nosync/turdb/turdb-core/.worktrees/single_insert_turdb";

fn generate_test_data(count: usize) -> Vec<Vec<OwnedValue>> {
    (0..count)
        .map(|i| {
            vec![
                OwnedValue::Int(i as i64),
                OwnedValue::Int((i % 1000) as i64),
                OwnedValue::Int((i % 500) as i64),
                OwnedValue::Int((i % 100) as i64),
                OwnedValue::Text(format!("license_{}", i % 10)),
                OwnedValue::Text(format!("2024-01-{:02}", (i % 28) + 1)),
                OwnedValue::Float((i % 100) as f64 / 10.0),
                OwnedValue::Text(format!("Title for row {}", i)),
                OwnedValue::Text(format!("slug-{}", i)),
                OwnedValue::Text(format!("Subtitle text for row number {}", i)),
                OwnedValue::Float((i * 1024) as f64),
                OwnedValue::Float((i * 2048) as f64),
            ]
        })
        .collect()
}

fn owned_to_rusqlite(val: &OwnedValue) -> rusqlite::types::Value {
    match val {
        OwnedValue::Null => rusqlite::types::Value::Null,
        OwnedValue::Int(i) => rusqlite::types::Value::Integer(*i),
        OwnedValue::Float(f) => rusqlite::types::Value::Real(*f),
        OwnedValue::Text(s) => rusqlite::types::Value::Text(s.clone()),
        OwnedValue::Blob(b) => rusqlite::types::Value::Blob(b.clone()),
        _ => rusqlite::types::Value::Null,
    }
}

#[test]
fn benchmark_single_inserts() {
    use turdb::{reset_fastpath_stats, get_fastpath_stats, get_fastpath_fail_stats, get_slowpath_stats};
    use turdb::{reset_timing_stats, get_batch_timing_stats};

    let _ = std::fs::remove_file(SQLITE_TARGET);
    let _ = std::fs::remove_dir_all(TURDB_TARGET);

    println!("\n============================================================");
    println!("  Single Insert Benchmark (Apple-to-Apple) - {} rows", ROW_COUNT);
    println!("============================================================\n");

    println!("Generating {} test rows...", ROW_COUNT);
    let gen_start = Instant::now();
    let test_data = generate_test_data(ROW_COUNT);
    println!("Generated in {:.2}s\n", gen_start.elapsed().as_secs_f64());

    // ========== SQLITE BENCHMARK ==========
    println!("--- SQLite Single Insert Benchmark ---");
    let sqlite_conn = Connection::open(SQLITE_TARGET).expect("Failed to create SQLite DB");

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

        for row in &test_data {
            let sqlite_row: Vec<rusqlite::types::Value> = row.iter().map(owned_to_rusqlite).collect();
            insert_stmt.execute(rusqlite::params_from_iter(sqlite_row.iter())).expect("Insert failed");
        }

        sqlite_conn.execute("COMMIT", []).expect("COMMIT failed");
    }

    let sqlite_elapsed = sqlite_start.elapsed();
    println!("SQLite: {} rows in {:.3}s = {:.0} rows/sec\n",
             test_data.len(), sqlite_elapsed.as_secs_f64(),
             test_data.len() as f64 / sqlite_elapsed.as_secs_f64());

    // ========== TURDB BENCHMARK ==========
    println!("--- TurDB Single Insert Benchmark ---");
    let turdb = Database::create(TURDB_TARGET).expect("Failed to create TurDB");

    turdb.execute("PRAGMA WAL = OFF;").expect("Failed to SET WAL");
    turdb.execute("PRAGMA synchronous = OFF;").expect("Failed to SET Synchronous");
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
          subtitle TEXT,
          total_compressed_bytes FLOAT,
          total_uncompressed_bytes FLOAT
        )
    ").expect("Failed to create TurDB table");

    let insert_stmt = turdb.prepare(
        "INSERT INTO dataset_versions VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
    ).expect("Failed to prepare TurDB insert");

    reset_timing_stats();
    reset_fastpath_stats();
    let turdb_start = Instant::now();

    turdb.execute("BEGIN").expect("BEGIN failed");

    for row in &test_data {
        insert_stmt.bind(row[0].clone())
            .bind(row[1].clone())
            .bind(row[2].clone())
            .bind(row[3].clone())
            .bind(row[4].clone())
            .bind(row[5].clone())
            .bind(row[6].clone())
            .bind(row[7].clone())
            .bind(row[8].clone())
            .bind(row[9].clone())
            .bind(row[10].clone())
            .bind(row[11].clone())
            .execute(&turdb)
            .expect("TurDB insert failed");
    }

    turdb.execute("COMMIT").expect("COMMIT failed");

    let turdb_elapsed = turdb_start.elapsed();

    let (record_ns, btree_ns) = get_batch_timing_stats();
    let record_time = std::time::Duration::from_nanos(record_ns);
    let btree_time = std::time::Duration::from_nanos(btree_ns);
    let (fastpath_hits, fastpath_misses) = get_fastpath_stats();
    let (fail_next_leaf, fail_space) = get_fastpath_fail_stats();
    let (slowpath_splits, slowpath_no_split) = get_slowpath_stats();

    println!("  -> Record build: {:.3}s, BTree insert: {:.3}s",
             record_time.as_secs_f64(), btree_time.as_secs_f64());
    println!("  -> Fastpath: {} hits, {} misses ({:.2}% hit rate)",
             fastpath_hits, fastpath_misses,
             100.0 * fastpath_hits as f64 / (fastpath_hits + fastpath_misses).max(1) as f64);
    println!("  -> Fail reasons: next_leaf={}, space={}", fail_next_leaf, fail_space);
    println!("  -> Slowpath: {} splits, {} no-split", slowpath_splits, slowpath_no_split);

    println!("TurDB:  {} rows in {:.3}s = {:.0} rows/sec\n",
             test_data.len(), turdb_elapsed.as_secs_f64(),
             test_data.len() as f64 / turdb_elapsed.as_secs_f64());

    // ========== COMPARISON ==========
    println!("============================================================");
    println!("  RESULTS (Single Inserts - Apple-to-Apple)");
    println!("============================================================");
    println!("SQLite (prepare_cached): {:.3}s ({:.0} rows/sec)",
             sqlite_elapsed.as_secs_f64(),
             test_data.len() as f64 / sqlite_elapsed.as_secs_f64());
    println!("TurDB (PreparedStatement): {:.3}s ({:.0} rows/sec)",
             turdb_elapsed.as_secs_f64(),
             test_data.len() as f64 / turdb_elapsed.as_secs_f64());

    let ratio = sqlite_elapsed.as_secs_f64() / turdb_elapsed.as_secs_f64();
    if ratio > 1.0 {
        println!("\nTurDB is {:.2}x FASTER than SQLite!", ratio);
    } else {
        println!("\nTurDB is {:.2}x SLOWER than SQLite", 1.0/ratio);
    }

    let _ = std::fs::remove_file(SQLITE_TARGET);
    let _ = std::fs::remove_dir_all(TURDB_TARGET);
}
