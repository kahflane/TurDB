use std::path::Path;
use std::time::Instant;
use rusqlite::Connection;
use turdb::{Database, OwnedValue};

const SQLITE_DB_PATH: &str = "/Users/julfikar/Downloads/_meta-kaggle.db";

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

fn sqlite_db_exists() -> bool {
    Path::new(SQLITE_DB_PATH).exists()
}

#[test]
fn benchmark_prepared_statements() {
    use turdb::{reset_fastpath_stats, get_fastpath_stats, get_fastpath_fail_stats, get_slowpath_stats};
    use turdb::{reset_timing_stats, get_batch_timing_stats};

    if !sqlite_db_exists() {
        eprintln!("Skipping test: SQLite database not found at {}", SQLITE_DB_PATH);
        return;
    }

    const SQLITE_TARGET: &str = "/Users/julfikar/Documents/PassionFruit.nosync/turdb/turdb-core/.worktrees/turdb_prep_sqlite.db";
    const TURDB_TARGET: &str = "/Users/julfikar/Documents/PassionFruit.nosync/turdb/turdb-core/.worktrees/turdb_prep_turdb";
    const ROW_COUNT: usize = 1_000_000;

    let _ = std::fs::remove_file(SQLITE_TARGET);
    let _ = std::fs::remove_dir_all(TURDB_TARGET);

    println!("\n============================================================");
    println!("  Prepared Statement Benchmark (Apple-to-Apple) - {} rows", ROW_COUNT);
    println!("============================================================\n");

    let source_conn = Connection::open(SQLITE_DB_PATH).expect("Failed to open source SQLite");

    let query = format!(
        "SELECT Id, DatasetId, DatasourceVersionId, CreatorUserId, LicenseName, CreationDate, \
         VersionNumber, Title, Slug, Subtitle, \
         TotalCompressedBytes, TotalUncompressedBytes FROM DatasetVersions LIMIT {}",
        ROW_COUNT
    );

    println!("Reading {} rows from source...", ROW_COUNT);
    let read_start = Instant::now();
    let mut stmt = source_conn.prepare(&query).expect("Failed to prepare");
    let rows: Vec<Vec<rusqlite::types::Value>> = stmt
        .query_map([], |row| {
            Ok((0..12).map(|i| row.get_ref(i).unwrap().into()).collect())
        })
        .expect("Failed to query")
        .collect::<Result<Vec<_>, _>>()
        .expect("Failed to collect");
    println!("Read {} rows in {:.2}s\n", rows.len(), read_start.elapsed().as_secs_f64());

    // ========== SQLITE BENCHMARK (prepare_cached) ==========
    println!("--- SQLite Prepared Statement Benchmark ---");
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

        for row in &rows {
            insert_stmt.execute(rusqlite::params_from_iter(row.iter())).expect("Insert failed");
        }

        sqlite_conn.execute("COMMIT", []).expect("COMMIT failed");
    }

    let sqlite_elapsed = sqlite_start.elapsed();
    println!("SQLite: {} rows in {:.3}s = {:.0} rows/sec\n",
             rows.len(), sqlite_elapsed.as_secs_f64(),
             rows.len() as f64 / sqlite_elapsed.as_secs_f64());

    // ========== TURDB BENCHMARK (Prepared Statement with CachedInsertPlan) ==========
    println!("--- TurDB Prepared Statement Benchmark (Cached) ---");
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

    println!("Converting {} rows to TurDB format...", rows.len());
    let convert_start = Instant::now();
    let turdb_rows: Vec<Vec<OwnedValue>> = rows.iter()
        .map(|row| row.iter().map(rusqlite_to_owned).collect())
        .collect();
    println!("Conversion done in {:.2}s\n", convert_start.elapsed().as_secs_f64());

    reset_timing_stats();
    reset_fastpath_stats();
    let turdb_start = Instant::now();

    turdb.execute("BEGIN").expect("BEGIN failed");

    for row in &turdb_rows {
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
             turdb_rows.len(), turdb_elapsed.as_secs_f64(),
             turdb_rows.len() as f64 / turdb_elapsed.as_secs_f64());

    // ========== COMPARISON ==========
    println!("============================================================");
    println!("  RESULTS (Prepared Statements - Apple-to-Apple)");
    println!("============================================================");
    println!("SQLite (prepare_cached): {:.3}s ({:.0} rows/sec)",
             sqlite_elapsed.as_secs_f64(),
             rows.len() as f64 / sqlite_elapsed.as_secs_f64());
    println!("TurDB (CachedInsertPlan): {:.3}s ({:.0} rows/sec)",
             turdb_elapsed.as_secs_f64(),
             turdb_rows.len() as f64 / turdb_elapsed.as_secs_f64());

    let ratio = sqlite_elapsed.as_secs_f64() / turdb_elapsed.as_secs_f64();
    if ratio > 1.0 {
        println!("\nTurDB is {:.2}x FASTER than SQLite!", ratio);
    } else {
        println!("\nTurDB is {:.2}x SLOWER than SQLite (target: 2-3x faster)", 1.0/ratio);
    }

    let _ = std::fs::remove_file(SQLITE_TARGET);
    let _ = std::fs::remove_dir_all(TURDB_TARGET);
}