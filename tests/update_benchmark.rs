use std::time::Instant;
use rusqlite::Connection;
use std::path::Path;
use turdb::{Database, OwnedValue};

const ROW_COUNT: usize = 1000;
const SQLITE_TARGET: &str = "/Users/julfikar/Documents/PassionFruit.nosync/turdb/turdb-core/.worktrees/update_bench_sqlite.db";
const TURDB_TARGET: &str = "/Users/julfikar/Documents/PassionFruit.nosync/turdb/turdb-core/.worktrees/update_bench_turdb";

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

fn setup_sqlite_with_data(path: &str, data: &[Vec<OwnedValue>]) -> Connection {
    let conn = Connection::open(path).expect("Failed to create SQLite DB");

    conn.execute_batch("
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

    {
        let mut insert_stmt = conn.prepare_cached(
            "INSERT INTO dataset_versions VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
        ).expect("Failed to prepare insert");

        conn.execute("BEGIN IMMEDIATE", []).expect("BEGIN failed");

        for row in data {
            let sqlite_row: Vec<rusqlite::types::Value> = row.iter().map(owned_to_rusqlite).collect();
            insert_stmt.execute(rusqlite::params_from_iter(sqlite_row.iter())).expect("Insert failed");
        }

        conn.execute("COMMIT", []).expect("COMMIT failed");
    }

    conn
}

fn setup_turdb_with_data(path: &Path, data: &[Vec<OwnedValue>]) -> Database {
    if path.exists() {
        std::fs::remove_dir_all(path).unwrap();
    }
    let db = Database::create(path).unwrap();

    // Aggressive performance settings for TurDB
    db.execute("PRAGMA synchronous = OFF;").expect("Failed to SET Synchronous");
    db.execute("
        CREATE TABLE dataset_versions (
          id BIGINT PRIMARY KEY,
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

    let insert_stmt = db.prepare(
        "INSERT INTO dataset_versions VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
    ).expect("Failed to prepare TurDB insert");

    db.execute("BEGIN").expect("BEGIN failed");

    for row in data {
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
            .execute(&db)
            .expect("TurDB insert failed");
    }

    db.execute("COMMIT").expect("COMMIT failed");

    db
}

#[test]
fn benchmark_single_updates() {
    let _ = std::fs::remove_file(SQLITE_TARGET);
    let _ = std::fs::remove_dir_all(TURDB_TARGET);

    println!("\n============================================================");
    println!("  Single Update Benchmark (Apple-to-Apple) - {} rows", ROW_COUNT);
    println!("============================================================\n");

    println!("Generating {} test rows...", ROW_COUNT);
    let gen_start = Instant::now();
    let test_data = generate_test_data(ROW_COUNT);
    println!("Generated in {:.2}s\n", gen_start.elapsed().as_secs_f64());

    // ========== SQLITE BENCHMARK ==========
    println!("--- SQLite Single Update Benchmark ---");
    println!("Setting up SQLite with {} rows...", ROW_COUNT);
    let sqlite_conn = setup_sqlite_with_data(SQLITE_TARGET, &test_data);

    let sqlite_start = Instant::now();

    {
        let mut update_stmt = sqlite_conn.prepare_cached(
            "UPDATE dataset_versions SET title = ?, version_number = ? WHERE dataset_id = ?"
        ).expect("Failed to prepare update");

        sqlite_conn.execute("BEGIN IMMEDIATE", []).expect("BEGIN failed");

        for i in 0..ROW_COUNT {
            let new_title = format!("Updated title {}", i);
            let new_version = (i as f64) * 1.5;
            update_stmt.execute(rusqlite::params![new_title, new_version, (i % 1000) as i64])
                .expect("Update failed");
        }

        sqlite_conn.execute("COMMIT", []).expect("COMMIT failed");
    }

    let sqlite_elapsed = sqlite_start.elapsed();
    println!("SQLite: {} updates in {:.3}s = {:.0} updates/sec\n",
             ROW_COUNT, sqlite_elapsed.as_secs_f64(),
             ROW_COUNT as f64 / sqlite_elapsed.as_secs_f64());

    // ========== TURDB BENCHMARK ==========
    println!("--- TurDB Single Update Benchmark ---");
    println!("Setting up TurDB with {} rows...", ROW_COUNT);
    let turdb = setup_turdb_with_data(Path::new(TURDB_TARGET), &test_data);

    turdb.execute("PRAGMA WAL=OFF").expect("Failed to set PRAGMA WAL");
    turdb.execute("PRAGMA synchronous = OFF;").expect("Failed to set PRAGMA synchronous");
    
    let update_stmt = turdb.prepare(
        "UPDATE dataset_versions SET title = ?, version_number = ? WHERE dataset_id = ?"
    ).expect("Failed to prepare TurDB update");

    let turdb_start = Instant::now();

    turdb.execute("BEGIN").expect("BEGIN failed");

    for i in 0..ROW_COUNT {
        let new_title = format!("Updated title {}", i);
        let new_version = (i as f64) * 1.5;
        update_stmt.bind(OwnedValue::Text(new_title))
            .bind(OwnedValue::Float(new_version))
            .bind(OwnedValue::Int((i % 1000) as i64))
            .execute(&turdb)
            .expect("TurDB update failed");
    }

    turdb.execute("COMMIT").expect("COMMIT failed");

    let turdb_elapsed = turdb_start.elapsed();
    println!("TurDB:  {} updates in {:.3}s = {:.0} updates/sec\n",
             ROW_COUNT, turdb_elapsed.as_secs_f64(),
             ROW_COUNT as f64 / turdb_elapsed.as_secs_f64());

    println!("--- Verifying TurDB Updates ---");
    let verification_indices = vec![0, ROW_COUNT / 4, ROW_COUNT / 2, ROW_COUNT - 1];
    for &i in &verification_indices {
        let rows = turdb.query(&format!(
            "SELECT title, version_number FROM dataset_versions WHERE dataset_id = {}",
            (i % 1000)
        )).expect("Verification query failed");

        assert!(!rows.is_empty(), "Row with dataset_id {} not found", i % 1000);

        let expected_title = format!("Updated title {}", i);
        let expected_version = (i as f64) * 1.5;

        let actual_title = rows[0].get_text(0).expect("Failed to get title");
        let actual_version = rows[0].get_float(1).expect("Failed to get version");

        assert_eq!(actual_title, expected_title,
            "Title mismatch for dataset_id {}: expected '{}', got '{}'",
            i % 1000, expected_title, actual_title);
        assert!((actual_version - expected_version).abs() < 0.001,
            "Version mismatch for dataset_id {}: expected {}, got {}",
            i % 1000, expected_version, actual_version);
    }
    println!("✓ Verified {} sample rows - all updates correct\n", verification_indices.len());

    // ========== COMPARISON ==========
    println!("============================================================");
    println!("  RESULTS (Single Updates - Apple-to-Apple)");
    println!("============================================================");
    println!("SQLite (prepare_cached): {:.3}s ({:.0} updates/sec)",
             sqlite_elapsed.as_secs_f64(),
             ROW_COUNT as f64 / sqlite_elapsed.as_secs_f64());
    println!("TurDB (PreparedStatement): {:.3}s ({:.0} updates/sec)",
             turdb_elapsed.as_secs_f64(),
             ROW_COUNT as f64 / turdb_elapsed.as_secs_f64());

    let ratio = sqlite_elapsed.as_secs_f64() / turdb_elapsed.as_secs_f64();
    if ratio > 1.0 {
        println!("\nTurDB is {:.2}x FASTER than SQLite!", ratio);
    } else {
        println!("\nTurDB is {:.2}x SLOWER than SQLite", 1.0/ratio);
    }

    let _ = std::fs::remove_file(SQLITE_TARGET);
    let _ = std::fs::remove_dir_all(TURDB_TARGET);
}

#[test]
fn benchmark_bulk_update() {
    let sqlite_path = "/Users/julfikar/Documents/PassionFruit.nosync/turdb/turdb-core/.worktrees/bulk_update_sqlite.db";
    let turdb_path = "/Users/julfikar/Documents/PassionFruit.nosync/turdb/turdb-core/.worktrees/bulk_update_turdb";

    let _ = std::fs::remove_file(sqlite_path);
    let _ = std::fs::remove_dir_all(turdb_path);

    println!("\n============================================================");
    println!("  Bulk Update Benchmark (Apple-to-Apple) - {} rows", ROW_COUNT);
    println!("============================================================\n");

    println!("Generating {} test rows...", ROW_COUNT);
    let gen_start = Instant::now();
    let test_data = generate_test_data(ROW_COUNT);
    println!("Generated in {:.2}s\n", gen_start.elapsed().as_secs_f64());

    // ========== SQLITE BENCHMARK ==========
    println!("--- SQLite Bulk Update Benchmark ---");
    println!("Setting up SQLite with {} rows...", ROW_COUNT);
    let sqlite_conn = setup_sqlite_with_data(sqlite_path, &test_data);

    let sqlite_start = Instant::now();

    sqlite_conn.execute("BEGIN IMMEDIATE", []).expect("BEGIN failed");
    let sqlite_updated = sqlite_conn.execute(
        "UPDATE dataset_versions SET title = 'Bulk Updated', version_number = 999.0 WHERE dataset_id < 500",
        []
    ).expect("Bulk update failed");
    sqlite_conn.execute("COMMIT", []).expect("COMMIT failed");

    let sqlite_elapsed = sqlite_start.elapsed();
    println!("SQLite: {} rows updated in {:.3}s = {:.0} updates/sec\n",
             sqlite_updated, sqlite_elapsed.as_secs_f64(),
             sqlite_updated as f64 / sqlite_elapsed.as_secs_f64());

    // ========== TURDB BENCHMARK ==========
    println!("--- TurDB Bulk Update Benchmark ---");
    println!("Setting up TurDB with {} rows...", ROW_COUNT);
    let turdb = setup_turdb_with_data(Path::new(turdb_path), &test_data);

    let turdb_start = Instant::now();

    turdb.execute("BEGIN").expect("BEGIN failed");
    turdb.execute("UPDATE dataset_versions SET title = 'Bulk Updated', version_number = 999.0 WHERE dataset_id < 500")
        .expect("TurDB bulk update failed");
    turdb.execute("COMMIT").expect("COMMIT failed");

    let turdb_elapsed = turdb_start.elapsed();

    let rows = turdb.query("SELECT COUNT(*) FROM dataset_versions WHERE title = 'Bulk Updated'")
        .expect("Count failed");
    let turdb_updated: i64 = rows[0].get_int(0).unwrap();

    println!("TurDB:  {} rows updated in {:.3}s = {:.0} updates/sec\n",
             turdb_updated, turdb_elapsed.as_secs_f64(),
             turdb_updated as f64 / turdb_elapsed.as_secs_f64());

    // ========== COMPARISON ==========
    println!("============================================================");
    println!("  RESULTS (Bulk Update - Apple-to-Apple)");
    println!("============================================================");
    println!("SQLite: {:.3}s ({:.0} updates/sec)",
             sqlite_elapsed.as_secs_f64(),
             sqlite_updated as f64 / sqlite_elapsed.as_secs_f64());
    println!("TurDB:  {:.3}s ({:.0} updates/sec)",
             turdb_elapsed.as_secs_f64(),
             turdb_updated as f64 / turdb_elapsed.as_secs_f64());

    let ratio = sqlite_elapsed.as_secs_f64() / turdb_elapsed.as_secs_f64();
    if ratio > 1.0 {
        println!("\nTurDB is {:.2}x FASTER than SQLite!", ratio);
    } else {
        println!("\nTurDB is {:.2}x SLOWER than SQLite", 1.0/ratio);
    }

    let _ = std::fs::remove_file(sqlite_path);
    let _ = std::fs::remove_dir_all(turdb_path);
}
