//! # MVCC Concurrent Transaction Tests
//!
//! This test module verifies that TurDB's MVCC implementation works correctly
//! under concurrent workloads using the new connection-per-session architecture.
//!
//! ## Architecture
//!
//! TurDB now supports true concurrent transactions through a split architecture:
//! - `SharedDatabase`: Contains shared state (catalog, file_manager, WAL, etc.)
//! - `Database`: A lightweight connection handle with session-local transaction state
//!
//! Cloning a `Database` creates a new connection that:
//! - Shares the underlying storage via `Arc<SharedDatabase>`
//! - Has its own independent `active_txn` for transaction isolation
//! - Can hold a transaction simultaneously with other connections
//!
//! ## Test Goals
//!
//! 1. **Transaction Isolation**: Each cloned connection can hold its own transaction
//! 2. **Concurrent Access**: Multiple threads can INSERT into different tables simultaneously
//! 3. **No Deadlocks**: The connection model prevents transaction-level deadlocks
//! 4. **Data Integrity**: All committed data is visible after transaction completion
//!
//! ## Usage
//!
//! ```sh
//! cargo test --test mvcc_concurrent --release -- --nocapture
//! ```

use rusqlite::Connection;
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Barrier};
use std::thread;
use std::time::Instant;
use turdb::Database;

const SQLITE_DB_PATH: &str = "/Users/julfikar/Downloads/_meta-kaggle.db";
const TURDB_PATH: &str =
    "/Users/julfikar/Documents/PassionFruit.nosync/turdb/turdb-core/.worktrees/mvcc_test";
const BATCH_SIZE: i64 = 10000;
const INSERT_BATCH_SIZE: usize = 5000;

fn sqlite_db_exists() -> bool {
    Path::new(SQLITE_DB_PATH).exists()
}

struct TableSchema {
    name: &'static str,
    turdb_ddl: &'static str,
    columns: &'static str,
}

fn camel_to_snake(name: &str) -> String {
    name.chars()
        .enumerate()
        .flat_map(|(i, c)| {
            if c.is_uppercase() && i > 0 {
                vec!['_', c.to_ascii_lowercase()]
            } else {
                vec![c.to_ascii_lowercase()]
            }
        })
        .collect()
}

const SMALL_TABLES: &[TableSchema] = &[
    TableSchema {
        name: "Tags",
        turdb_ddl: "CREATE TABLE tags (
            id BIGINT primary key auto_increment, parent_tag_id FLOAT, name TEXT, slug TEXT, full_path TEXT,
            description TEXT, dataset_count BIGINT, competition_count BIGINT, kernel_count INT
        )",
        columns: "Id, ParentTagId, Name, Slug, FullPath, Description, DatasetCount, CompetitionCount, KernelCount",
    },
    TableSchema {
        name: "KernelLanguages",
        turdb_ddl: "CREATE TABLE kernel_languages (id BIGINT primary key auto_increment, name TEXT, display_name TEXT, is_notebook BIGINT)",
        columns: "Id, Name, DisplayName, IsNotebook",
    },
    TableSchema {
        name: "Organizations",
        turdb_ddl: "CREATE TABLE organizations (id BIGINT primary key auto_increment, name TEXT, slug TEXT, creation_date TEXT, description TEXT)",
        columns: "Id, Name, Slug, CreationDate, Description",
    },
    TableSchema {
        name: "CompetitionTags",
        turdb_ddl: "CREATE TABLE competition_tags (id BIGINT primary key auto_increment, competition_id BIGINT, tag_id BIGINT)",
        columns: "Id, CompetitionId, TagId",
    },
    TableSchema {
        name: "Competitions",
        turdb_ddl: "CREATE TABLE competitions (
            id BIGINT primary key auto_increment, slug TEXT, title TEXT, subtitle TEXT, host_segment_title TEXT,
            forum_id BIGINT, organization_id FLOAT, enabled_date TEXT, deadline_date TEXT,
            prohibit_new_entrants_deadline_date TEXT, team_merger_deadline_date TEXT,
            team_model_deadline_date TEXT, model_submission_deadline_date TEXT,
            final_leaderboard_has_been_verified BIGINT, has_kernels BIGINT,
            only_allow_kernel_submissions BIGINT, has_leaderboard BIGINT,
            leaderboard_percentage BIGINT, leaderboard_display_format FLOAT,
            evaluation_algorithm_abbreviation TEXT, evaluation_algorithm_name TEXT,
            evaluation_algorithm_description TEXT, evaluation_algorithm_is_max BIGINT,
            max_daily_submissions BIGINT, num_scored_submissions BIGINT, max_team_size BIGINT,
            ban_team_mergers BIGINT, enable_team_models BIGINT, reward_type TEXT,
            reward_quantity FLOAT, num_prizes BIGINT, user_rank_multiplier FLOAT,
            can_qualify_tiers BIGINT, total_teams BIGINT, total_competitors BIGINT,
            total_submissions BIGINT, validation_set_name FLOAT, validation_set_value FLOAT,
            enable_submission_model_hashes BIGINT, enable_submission_model_attachments BIGINT,
            host_name FLOAT, competition_type_id INT
        )",
        columns: "Id, Slug, Title, Subtitle, HostSegmentTitle, ForumId, OrganizationId, EnabledDate, DeadlineDate, ProhibitNewEntrantsDeadlineDate, TeamMergerDeadlineDate, TeamModelDeadlineDate, ModelSubmissionDeadlineDate, FinalLeaderboardHasBeenVerified, HasKernels, OnlyAllowKernelSubmissions, HasLeaderboard, LeaderboardPercentage, LeaderboardDisplayFormat, EvaluationAlgorithmAbbreviation, EvaluationAlgorithmName, EvaluationAlgorithmDescription, EvaluationAlgorithmIsMax, MaxDailySubmissions, NumScoredSubmissions, MaxTeamSize, BanTeamMergers, EnableTeamModels, RewardType, RewardQuantity, NumPrizes, UserRankMultiplier, CanQualifyTiers, TotalTeams, TotalCompetitors, TotalSubmissions, ValidationSetName, ValidationSetValue, EnableSubmissionModelHashes, EnableSubmissionModelAttachments, HostName, CompetitionTypeId",
    },
    TableSchema {
        name: "DatasetTasks",
        turdb_ddl: "CREATE TABLE dataset_tasks (
            id BIGINT primary key auto_increment, dataset_id BIGINT, owner_user_id BIGINT, creation_date TEXT,
            description TEXT, forum_id FLOAT, title TEXT, subtitle TEXT,
            deadline TEXT, total_votes INT
        )",
        columns: "Id, DatasetId, OwnerUserId, CreationDate, Description, ForumId, Title, Subtitle, Deadline, TotalVotes",
    },
    TableSchema {
        name: "DatasetTaskSubmissions",
        turdb_ddl: "CREATE TABLE dataset_task_submissions (
            id BIGINT primary key auto_increment, dataset_task_id BIGINT, submitted_user_id FLOAT, creation_date TEXT,
            kernel_id FLOAT, dataset_id FLOAT, accepted_date TEXT
        )",
        columns: "Id, DatasetTaskId, SubmittedUserId, CreationDate, KernelId, DatasetId, AcceptedDate",
    },
    TableSchema {
        name: "UserOrganizations",
        turdb_ddl: "CREATE TABLE user_organizations (id BIGINT primary key auto_increment, user_id BIGINT, organization_id BIGINT, join_date TEXT)",
        columns: "Id, UserId, OrganizationId, JoinDate",
    },
    TableSchema {
        name: "Forums",
        turdb_ddl: "CREATE TABLE forums (id BIGINT primary key auto_increment, parent_forum_id FLOAT, title TEXT)",
        columns: "Id, ParentForumId, Title",
    },
    TableSchema {
        name: "DatasetVersions",
        turdb_ddl: "CREATE TABLE dataset_versions (
            id BIGINT primary key auto_increment, dataset_id BIGINT, datasource_version_id BIGINT, creator_user_id BIGINT,
            license_name TEXT, creation_date TEXT, version_number FLOAT, title TEXT,
            slug TEXT, subtitle TEXT, description TEXT, version_notes TEXT,
            total_compressed_bytes FLOAT, total_uncompressed_bytes REAL
        )",
        columns: "Id, DatasetId, DatasourceVersionId, CreatorUserId, LicenseName, CreationDate, VersionNumber, Title, Slug, Subtitle, Description, VersionNotes, TotalCompressedBytes, TotalUncompressedBytes",
    },
];

fn escape_sql_value(value: rusqlite::types::Value) -> String {
    use rusqlite::types::Value;
    match value {
        Value::Null => "NULL".to_string(),
        Value::Integer(i) => i.to_string(),
        Value::Real(f) => {
            if f.is_nan() || f.is_infinite() {
                "NULL".to_string()
            } else {
                format!("{:.15e}", f)
            }
        }
        Value::Text(s) => format!("'{}'", s.replace('\'', "''")),
        Value::Blob(b) => {
            use std::fmt::Write;
            let mut s = String::with_capacity(b.len() * 2 + 3);
            s.push_str("X'");
            for byte in b {
                write!(s, "{:02X}", byte).unwrap();
            }
            s.push('\'');
            s
        }
    }
}

fn sqlite_to_owned_value(value: rusqlite::types::Value) -> turdb::OwnedValue {
    use rusqlite::types::Value;
    match value {
        Value::Null => turdb::OwnedValue::Null,
        Value::Integer(i) => turdb::OwnedValue::Int(i),
        Value::Real(f) => {
            if f.is_nan() || f.is_infinite() {
                turdb::OwnedValue::Null
            } else {
                turdb::OwnedValue::Float(f)
            }
        }
        Value::Text(s) => turdb::OwnedValue::Text(s),
        Value::Blob(b) => turdb::OwnedValue::Blob(b),
    }
}

#[derive(Debug)]
struct TableImportResult {
    table_name: String,
    rows_inserted: u64,
    duration_ms: u64,
    success: bool,
    error_message: Option<String>,
}

fn import_table_concurrent(
    db: Database,
    table: &'static TableSchema,
    thread_id: usize,
    total_rows_counter: Arc<AtomicU64>,
) -> TableImportResult {
    let start = Instant::now();
    let table_name = table.name.to_string();

    let result = (|| -> eyre::Result<u64> {
        println!(
            "[Thread {}] Starting import of table: {}",
            thread_id, table.name
        );

        db.execute(table.turdb_ddl)?;

        let sqlite_conn = Connection::open(SQLITE_DB_PATH)?;

        let count: i64 =
            sqlite_conn.query_row(&format!("SELECT COUNT(*) FROM {}", table.name), [], |row| {
                row.get(0)
            })?;
        println!("[Thread {}] {} has {} rows", thread_id, table.name, count);

        if count == 0 {
            return Ok(0);
        }

        let col_count = table.columns.split(',').count();
        let turdb_table = camel_to_snake(table.name);
        let mut total_inserted: u64 = 0;
        let mut offset: i64 = 0;

        db.execute("BEGIN")?;

        loop {
            let query = format!(
                "SELECT {} FROM {} LIMIT {} OFFSET {}",
                table.columns, table.name, BATCH_SIZE, offset
            );

            let mut stmt = sqlite_conn.prepare(&query)?;
            let mut rows = stmt.query([])?;
            let mut value_batches: Vec<String> = Vec::with_capacity(INSERT_BATCH_SIZE);
            let mut loop_batch_count = 0u64;

            while let Some(row) = rows.next()? {
                let mut values = Vec::with_capacity(col_count);
                for i in 0..col_count {
                    let val = row.get_ref(i)?.into();
                    values.push(escape_sql_value(val));
                }
                value_batches.push(format!("({})", values.join(", ")));
                loop_batch_count += 1;
                total_inserted += 1;
                if total_inserted.is_multiple_of(100) {
                    eprintln!("[Thread {}] Inserted in {} rows", thread_id, total_inserted);
                }
                if value_batches.len() >= INSERT_BATCH_SIZE {
                    let insert_sql = format!(
                        "INSERT INTO {} VALUES {}",
                        turdb_table,
                        value_batches.join(", ")
                    );
                    db.execute(&insert_sql)?;
                    total_rows_counter.fetch_add(value_batches.len() as u64, Ordering::Relaxed);
                    value_batches.clear();
                }
            }

            if !value_batches.is_empty() {
                let insert_sql = format!(
                    "INSERT INTO {} VALUES {}",
                    turdb_table,
                    value_batches.join(", ")
                );
                db.execute(&insert_sql)?;
                total_rows_counter.fetch_add(value_batches.len() as u64, Ordering::Relaxed);
            }

            if loop_batch_count == 0 {
                break;
            }

            offset += BATCH_SIZE;
            if offset >= count {
                break;
            }
        }

        db.execute("COMMIT")?;
        println!(
            "[Thread {}] {} committed successfully ({} rows)",
            thread_id, table.name, total_inserted
        );

        Ok(total_inserted)
    })();

    let duration_ms = start.elapsed().as_millis() as u64;

    match result {
        Ok(rows) => TableImportResult {
            table_name,
            rows_inserted: rows,
            duration_ms,
            success: true,
            error_message: None,
        },
        Err(e) => TableImportResult {
            table_name,
            rows_inserted: 0,
            duration_ms,
            success: false,
            error_message: Some(format!("{:?}", e)),
        },
    }
}

fn verify_table_data(db: &Database, table: &TableSchema) -> eyre::Result<(u64, bool)> {
    let turdb_table = camel_to_snake(table.name);
    let query = format!("SELECT COUNT(*) FROM {}", turdb_table);
    let rows = db.query(&query)?;

    if rows.is_empty() {
        return Ok((0, false));
    }

    let count = match rows[0].get(0) {
        Some(turdb::OwnedValue::Int(n)) => *n as u64,
        _ => 0,
    };

    let sqlite_conn = Connection::open(SQLITE_DB_PATH)?;
    let expected_count: i64 =
        sqlite_conn.query_row(&format!("SELECT COUNT(*) FROM {}", table.name), [], |row| {
            row.get(0)
        })?;

    Ok((count, count == expected_count as u64))
}

#[test]
fn concurrent_import_small_tables() {
    if !sqlite_db_exists() {
        eprintln!(
            "Skipping test: SQLite database not found at {}",
            SQLITE_DB_PATH
        );
        return;
    }

    if Path::new(TURDB_PATH).exists() {
        std::fs::remove_dir_all(TURDB_PATH).expect("Failed to remove existing TurDB directory");
        println!("Removed existing TurDB directory at {}", TURDB_PATH);
    }

    let db = Database::create(TURDB_PATH).expect("Failed to create database");
    db.execute("PRAGMA WAL=ON").expect("Failed to set WAL mode");
    db.execute("PRAGMA synchronous=NORMAL")
        .expect("Failed to set synchronous mode");
    db.execute("SET foreign_keys = ON")
        .expect("Failed to disable foreign keys");

    println!("\n=== MVCC Concurrent Import Test ===");
    println!("Tables to import: {}", SMALL_TABLES.len());
    println!("Using {} concurrent threads\n", SMALL_TABLES.len());

    let overall_start = Instant::now();
    let total_rows = Arc::new(AtomicU64::new(0));

    let handles: Vec<_> = SMALL_TABLES
        .iter()
        .enumerate()
        .map(|(idx, table)| {
            let db_conn = db.clone();
            let total_rows_clone = Arc::clone(&total_rows);
            thread::spawn(move || import_table_concurrent(db_conn, table, idx, total_rows_clone))
        })
        .collect();

    let mut results: Vec<TableImportResult> = Vec::new();
    for handle in handles {
        match handle.join() {
            Ok(result) => results.push(result),
            Err(e) => {
                eprintln!("Thread panicked: {:?}", e);
            }
        }
    }

    let overall_elapsed = overall_start.elapsed();

    println!("\n=== Import Results ===");
    let mut success_count = 0;
    let mut failure_count = 0;
    let mut total_imported_rows = 0u64;

    for result in &results {
        if result.success {
            success_count += 1;
            total_imported_rows += result.rows_inserted;
            println!(
                "✓ {} - {} rows in {}ms ({:.0} rows/sec)",
                result.table_name,
                result.rows_inserted,
                result.duration_ms,
                if result.duration_ms > 0 {
                    (result.rows_inserted as f64 / result.duration_ms as f64) * 1000.0
                } else {
                    0.0
                }
            );
        } else {
            failure_count += 1;
            println!(
                "✗ {} - FAILED: {}",
                result.table_name,
                result.error_message.as_deref().unwrap_or("unknown error")
            );
        }
    }

    println!("\n=== Verification Phase ===");
    let mut verification_passed = 0;
    let mut verification_failed = 0;

    for table in SMALL_TABLES {
        match verify_table_data(&db, table) {
            Ok((count, matches)) => {
                if matches {
                    verification_passed += 1;
                    println!("✓ {} - {} rows verified", camel_to_snake(table.name), count);
                } else {
                    verification_failed += 1;
                    println!(
                        "✗ {} - row count mismatch: {} rows",
                        camel_to_snake(table.name),
                        count
                    );
                }
            }
            Err(e) => {
                verification_failed += 1;
                println!(
                    "✗ {} - verification error: {}",
                    camel_to_snake(table.name),
                    e
                );
            }
        }
    }

    println!("\n=== Summary ===");
    println!(
        "Tables imported:     {}/{}",
        success_count,
        SMALL_TABLES.len()
    );
    println!("Import failures:     {}", failure_count);
    println!(
        "Verification passed: {}/{}",
        verification_passed,
        SMALL_TABLES.len()
    );
    println!("Verification failed: {}", verification_failed);
    println!("Total rows imported: {}", total_imported_rows);
    println!("Total wall time:     {:.2}s", overall_elapsed.as_secs_f64());
    println!(
        "Throughput:          {:.0} rows/sec",
        total_imported_rows as f64 / overall_elapsed.as_secs_f64()
    );

    assert_eq!(failure_count, 0, "Some imports failed");
    assert_eq!(verification_failed, 0, "Some verifications failed");
    assert!(total_imported_rows > 0, "No rows were imported");

    println!("\n=== MVCC Concurrent Import Test PASSED ===\n");

    let _ = db.close();
}

fn import_table_fast(
    db: Database,
    table: &'static TableSchema,
    thread_id: usize,
    total_rows_counter: Arc<AtomicU64>,
) -> TableImportResult {
    let start = Instant::now();
    let table_name = table.name.to_string();

    let result = (|| -> eyre::Result<u64> {
        println!(
            "[Thread {}] Starting FAST import of table: {}",
            thread_id, table.name
        );

        db.execute(table.turdb_ddl)?;

        let sqlite_conn = Connection::open(SQLITE_DB_PATH)?;

        let count: i64 =
            sqlite_conn.query_row(&format!("SELECT COUNT(*) FROM {}", table.name), [], |row| {
                row.get(0)
            })?;
        println!("[Thread {}] {} has {} rows", thread_id, table.name, count);

        if count == 0 {
            return Ok(0);
        }

        let col_count = table.columns.split(',').count();
        let turdb_table = camel_to_snake(table.name);
        let mut total_inserted: u64 = 0;
        let mut offset: i64 = 0;
        const FAST_BATCH_SIZE: usize = 10000;

        loop {
            let query = format!(
                "SELECT {} FROM {} LIMIT {} OFFSET {}",
                table.columns, table.name, BATCH_SIZE, offset
            );

            let mut stmt = sqlite_conn.prepare(&query)?;
            let mut rows = stmt.query([])?;
            let mut value_batches: Vec<Vec<turdb::OwnedValue>> = Vec::with_capacity(FAST_BATCH_SIZE);
            let mut loop_batch_count = 0u64;

            while let Some(row) = rows.next()? {
                let mut values: Vec<turdb::OwnedValue> = Vec::with_capacity(col_count);
                for i in 0..col_count {
                    let val = row.get_ref(i)?.into();
                    values.push(sqlite_to_owned_value(val));
                }
                value_batches.push(values);
                loop_batch_count += 1;
                total_inserted += 1;
                if value_batches.len() >= FAST_BATCH_SIZE {
                    db.bulk_insert(&turdb_table, std::mem::take(&mut value_batches))?;
                    total_rows_counter.fetch_add(FAST_BATCH_SIZE as u64, Ordering::Relaxed);
                    value_batches = Vec::with_capacity(FAST_BATCH_SIZE);
                }
            }

            if !value_batches.is_empty() {
                let batch_len = value_batches.len();
                db.bulk_insert(&turdb_table, value_batches)?;
                total_rows_counter.fetch_add(batch_len as u64, Ordering::Relaxed);
            }

            if loop_batch_count < BATCH_SIZE as u64 {
                break;
            }
            offset += BATCH_SIZE;
        }

        Ok(total_inserted)
    })();

    let duration = start.elapsed();
    match result {
        Ok(rows) => TableImportResult {
            table_name,
            rows_inserted: rows,
            duration_ms: duration.as_millis() as u64,
            success: true,
            error_message: None,
        },
        Err(e) => TableImportResult {
            table_name,
            rows_inserted: 0,
            duration_ms: duration.as_millis() as u64,
            success: false,
            error_message: Some(e.to_string()),
        },
    }
}

#[test]
fn fast_import_small_tables() {
    if !sqlite_db_exists() {
        eprintln!(
            "Skipping test: SQLite database not found at {}",
            SQLITE_DB_PATH
        );
        return;
    }

    let fast_turdb_path = format!("{}_fast", TURDB_PATH);
    if Path::new(&fast_turdb_path).exists() {
        std::fs::remove_dir_all(&fast_turdb_path).expect("Failed to remove existing TurDB directory");
        println!("Removed existing TurDB directory at {}", fast_turdb_path);
    }

    let db = Database::create(&fast_turdb_path).expect("Failed to create database");

    println!("\n=== FAST Import Test (using bulk_insert) ===");
    println!("Tables to import: {}", SMALL_TABLES.len());
    println!("Using {} concurrent threads\n", SMALL_TABLES.len());

    let overall_start = Instant::now();
    let total_rows = Arc::new(AtomicU64::new(0));

    let handles: Vec<_> = SMALL_TABLES
        .iter()
        .enumerate()
        .map(|(idx, table)| {
            let db_conn = db.clone();
            let total_rows_clone = Arc::clone(&total_rows);
            thread::spawn(move || import_table_fast(db_conn, table, idx, total_rows_clone))
        })
        .collect();

    let mut results: Vec<TableImportResult> = Vec::new();
    for handle in handles {
        match handle.join() {
            Ok(result) => results.push(result),
            Err(e) => {
                eprintln!("Thread panicked: {:?}", e);
            }
        }
    }

    let overall_elapsed = overall_start.elapsed();

    println!("\n=== Fast Import Results ===");
    let mut success_count = 0;
    let mut failure_count = 0;
    let mut total_imported_rows = 0u64;

    for result in &results {
        if result.success {
            success_count += 1;
            total_imported_rows += result.rows_inserted;
            println!(
                "✓ {} - {} rows in {}ms ({:.0} rows/sec)",
                result.table_name,
                result.rows_inserted,
                result.duration_ms,
                if result.duration_ms > 0 {
                    (result.rows_inserted as f64 / result.duration_ms as f64) * 1000.0
                } else {
                    0.0
                }
            );
        } else {
            failure_count += 1;
            println!(
                "✗ {} - FAILED: {}",
                result.table_name,
                result.error_message.as_deref().unwrap_or("unknown error")
            );
        }
    }

    println!("\n=== Summary ===");
    println!(
        "Tables imported:     {}/{}",
        success_count,
        SMALL_TABLES.len()
    );
    println!("Import failures:     {}", failure_count);
    println!("Total rows imported: {}", total_imported_rows);
    println!("Total wall time:     {:.2}s", overall_elapsed.as_secs_f64());
    println!(
        "Throughput:          {:.0} rows/sec",
        total_imported_rows as f64 / overall_elapsed.as_secs_f64()
    );

    assert_eq!(failure_count, 0, "Some imports failed");
    assert!(total_imported_rows > 0, "No rows were imported");

    println!("\n=== FAST Import Test PASSED ===\n");
}

#[test]
fn concurrent_read_write_isolation() {
    if !sqlite_db_exists() {
        eprintln!(
            "Skipping test: SQLite database not found at {}",
            SQLITE_DB_PATH
        );
        return;
    }

    let test_path = format!("{}_isolation", TURDB_PATH);
    if Path::new(&test_path).exists() {
        std::fs::remove_dir_all(&test_path).expect("Failed to remove existing directory");
    }

    let db = Database::create(&test_path).expect("Failed to create database");
    db.execute("PRAGMA WAL=ON").expect("Failed to enable WAL");

    db.execute("CREATE TABLE test_isolation (id BIGINT primary key auto_increment, value TEXT)")
        .expect("Failed to create table");

    println!("\n=== Read-Write Isolation Test ===");

    for i in 1..=100 {
        db.execute(&format!(
            "INSERT INTO test_isolation VALUES ({}, 'initial_{}')",
            i, i
        ))
        .expect("Failed to insert initial data");
    }

    let db_writer = db.clone();
    let db_reader = db.clone();

    let writer_handle = thread::spawn(move || {
        let mut inserted = 0;
        for batch in 0..10 {
            db_writer.execute("BEGIN").expect("Failed to begin");
            for i in 0..50 {
                let id = 1000 + batch * 50 + i;
                db_writer
                    .execute(&format!(
                        "INSERT INTO test_isolation VALUES ({}, 'writer_{}')",
                        id, id
                    ))
                    .expect("Failed to insert");
                inserted += 1;
            }
            db_writer.execute("COMMIT").expect("Failed to commit");
            thread::sleep(std::time::Duration::from_millis(10));
        }
        inserted
    });

    let reader_handle = thread::spawn(move || {
        let mut read_counts = Vec::new();
        for _ in 0..20 {
            let rows = db_reader
                .query("SELECT COUNT(*) FROM test_isolation")
                .expect("Failed to query");
            if let Some(row) = rows.first() {
                if let Some(turdb::OwnedValue::Int(count)) = row.get(0) {
                    read_counts.push(*count);
                }
            }
            thread::sleep(std::time::Duration::from_millis(25));
        }
        read_counts
    });

    let inserted = writer_handle.join().expect("Writer thread panicked");
    let read_counts = reader_handle.join().expect("Reader thread panicked");

    println!("Writer inserted: {} rows", inserted);
    println!("Reader saw counts: {:?}", read_counts);

    let final_rows = db
        .query("SELECT COUNT(*) FROM test_isolation")
        .expect("Failed to final query");
    let final_count = if let Some(turdb::OwnedValue::Int(n)) = final_rows[0].get(0) {
        *n
    } else {
        0
    };

    println!("Final count: {}", final_count);
    assert_eq!(final_count, 100 + inserted as i64, "Row count mismatch");

    for count in &read_counts {
        assert!(
            *count >= 100,
            "Read count should never be less than initial: {}",
            count
        );
    }

    let is_monotonic = read_counts.windows(2).all(|w| w[0] <= w[1]);
    println!("Read counts monotonically increasing: {}", is_monotonic);

    println!("=== Read-Write Isolation Test PASSED ===\n");
}

#[test]
fn concurrent_multi_table_transactions() {
    if !sqlite_db_exists() {
        eprintln!(
            "Skipping test: SQLite database not found at {}",
            SQLITE_DB_PATH
        );
        return;
    }

    let test_path = format!("{}_multi_txn", TURDB_PATH);
    if Path::new(&test_path).exists() {
        std::fs::remove_dir_all(&test_path).expect("Failed to remove existing directory");
    }

    let db = Database::create(&test_path).expect("Failed to create database");
    db.execute("PRAGMA WAL=ON").expect("Failed to enable WAL");

    println!("\n=== Multi-Table Transaction Test ===");

    for i in 1..=5 {
        db.execute(&format!(
            "CREATE TABLE test_table_{} (id BIGINT primary key auto_increment, data TEXT)",
            i
        ))
        .expect("Failed to create table");
    }

    let handles: Vec<_> = (1..=5)
        .map(|table_id| {
            let db_conn = db.clone();
            thread::spawn(move || {
                let mut success_count = 0;
                for batch in 0..10 {
                    if db_conn.execute("BEGIN").is_ok() {
                        let mut batch_ok = true;
                        for row in 0..20 {
                            let id = batch * 20 + row + 1;
                            let insert_sql = format!(
                                "INSERT INTO test_table_{} VALUES ({}, 'data_{}_{}')",
                                table_id, id, table_id, id
                            );
                            if db_conn.execute(&insert_sql).is_err() {
                                batch_ok = false;
                                break;
                            }
                        }
                        if batch_ok {
                            if db_conn.execute("COMMIT").is_ok() {
                                success_count += 20;
                            }
                        } else {
                            let _ = db_conn.execute("ROLLBACK");
                        }
                    }
                    thread::sleep(std::time::Duration::from_millis(5));
                }
                (table_id, success_count)
            })
        })
        .collect();

    let mut results = Vec::new();
    for handle in handles {
        match handle.join() {
            Ok(result) => results.push(result),
            Err(e) => eprintln!("Thread panicked: {:?}", e),
        }
    }

    println!("\n=== Results ===");
    let mut total_inserted = 0;
    for (table_id, count) in &results {
        println!("Table {} - {} rows inserted", table_id, count);
        total_inserted += count;

        let query = format!("SELECT COUNT(*) FROM test_table_{}", table_id);
        let rows = db.query(&query).expect("Failed to query");
        if let Some(turdb::OwnedValue::Int(n)) = rows[0].get(0) {
            assert_eq!(*n as i32, *count, "Mismatch for table {}", table_id);
        }
    }

    println!("Total rows across all tables: {}", total_inserted);
    assert!(total_inserted > 0, "No rows were inserted");

    println!("=== Multi-Table Transaction Test PASSED ===\n");
}

#[test]
fn stress_test_transaction_slots() {
    let test_path = format!("{}_stress", TURDB_PATH);
    if Path::new(&test_path).exists() {
        std::fs::remove_dir_all(&test_path).expect("Failed to remove existing directory");
    }

    let db = Database::create(&test_path).expect("Failed to create database");

    println!("\n=== Transaction Slot Stress Test ===");

    db.execute(
        "CREATE TABLE stress_test (id BIGINT primary key auto_increment, thread_id INT, seq INT)",
    )
    .expect("Failed to create table");

    let num_threads = 32;
    let ops_per_thread = 50;
    let success_counter = Arc::new(AtomicU64::new(0));
    let failure_counter = Arc::new(AtomicU64::new(0));

    let handles: Vec<_> = (0..num_threads)
        .map(|thread_id| {
            let db_conn = db.clone();
            let success = Arc::clone(&success_counter);
            let failure = Arc::clone(&failure_counter);
            thread::spawn(move || {
                for seq in 0..ops_per_thread {
                    match db_conn.execute("BEGIN") {
                        Ok(_) => {
                            let insert_sql = format!(
                                "INSERT INTO stress_test VALUES ({}, {}, {})",
                                thread_id * 1000 + seq,
                                thread_id,
                                seq
                            );
                            if db_conn.execute(&insert_sql).is_ok()
                                && db_conn.execute("COMMIT").is_ok()
                            {
                                success.fetch_add(1, Ordering::Relaxed);
                                continue;
                            }
                            let _ = db_conn.execute("ROLLBACK");
                            failure.fetch_add(1, Ordering::Relaxed);
                        }
                        Err(_) => {
                            failure.fetch_add(1, Ordering::Relaxed);
                            thread::sleep(std::time::Duration::from_millis(1));
                        }
                    }
                }
            })
        })
        .collect();

    for handle in handles {
        handle.join().expect("Thread panicked");
    }

    let total_success = success_counter.load(Ordering::Relaxed);
    let total_failure = failure_counter.load(Ordering::Relaxed);

    println!("Threads:           {}", num_threads);
    println!("Ops per thread:    {}", ops_per_thread);
    println!("Total attempted:   {}", num_threads * ops_per_thread);
    println!("Successful:        {}", total_success);
    println!("Failed:            {}", total_failure);

    let rows = db
        .query("SELECT COUNT(*) FROM stress_test")
        .expect("Failed to count");
    if let Some(turdb::OwnedValue::Int(n)) = rows[0].get(0) {
        println!("Rows in table:     {}", n);
        assert_eq!(
            *n as u64, total_success,
            "Row count doesn't match success count"
        );
    }

    println!("=== Transaction Slot Stress Test PASSED ===\n");
}

#[test]
fn two_connections_can_hold_simultaneous_transactions() {
    let temp_dir = tempfile::tempdir().expect("Failed to create temp dir");
    let db_path = temp_dir.path().join("concurrent_txn_test");

    let db = Database::create(&db_path).expect("Failed to create database");

    db.execute("CREATE TABLE t1 (id INT PRIMARY KEY, value TEXT)")
        .expect("Failed to create t1");
    db.execute("CREATE TABLE t2 (id INT PRIMARY KEY, value TEXT)")
        .expect("Failed to create t2");

    let db1 = db.clone();
    let db2 = db.clone();

    let barrier = Arc::new(Barrier::new(2));
    let barrier1 = Arc::clone(&barrier);
    let barrier2 = Arc::clone(&barrier);

    let handle1 = thread::spawn(move || {
        db1.execute("BEGIN").expect("Thread 1: Failed to BEGIN");

        barrier1.wait();

        db1.execute("INSERT INTO t1 VALUES (1, 'from thread 1')")
            .expect("Thread 1: Failed to INSERT");

        barrier1.wait();

        db1.execute("COMMIT").expect("Thread 1: Failed to COMMIT");
    });

    let handle2 = thread::spawn(move || {
        db2.execute("BEGIN").expect("Thread 2: Failed to BEGIN");

        barrier2.wait();

        db2.execute("INSERT INTO t2 VALUES (2, 'from thread 2')")
            .expect("Thread 2: Failed to INSERT");

        barrier2.wait();

        db2.execute("COMMIT").expect("Thread 2: Failed to COMMIT");
    });

    handle1.join().expect("Thread 1 panicked");
    handle2.join().expect("Thread 2 panicked");

    let rows1 = db.query("SELECT * FROM t1").expect("Failed to query t1");
    let rows2 = db.query("SELECT * FROM t2").expect("Failed to query t2");

    assert_eq!(rows1.len(), 1, "t1 should have 1 row");
    assert_eq!(rows2.len(), 1, "t2 should have 1 row");

    println!("=== Two Connections Simultaneous Transactions Test PASSED ===");
}

#[test]
fn cloned_connections_have_independent_transactions() {
    let temp_dir = tempfile::tempdir().expect("Failed to create temp dir");
    let db_path = temp_dir.path().join("independent_txn_test");

    let db = Database::create(&db_path).expect("Failed to create database");

    db.execute("CREATE TABLE test (id INT PRIMARY KEY, value INT)")
        .expect("Failed to create table");

    db.execute("INSERT INTO test VALUES (1, 100)")
        .expect("Failed to insert");

    let conn1 = db.clone();
    let conn2 = db.clone();

    conn1.execute("BEGIN").expect("conn1: Failed to BEGIN");
    conn1
        .execute("INSERT INTO test VALUES (2, 200)")
        .expect("conn1: Failed to INSERT");

    conn2.execute("BEGIN").expect("conn2: Failed to BEGIN");
    conn2
        .execute("INSERT INTO test VALUES (3, 300)")
        .expect("conn2: Failed to INSERT");

    conn1
        .execute("ROLLBACK")
        .expect("conn1: Failed to ROLLBACK");

    conn2.execute("COMMIT").expect("conn2: Failed to COMMIT");

    let rows = db
        .query("SELECT id FROM test ORDER BY id")
        .expect("Failed to query");
    assert_eq!(
        rows.len(),
        2,
        "Should have 2 rows (initial + conn2's insert)"
    );

    let ids: Vec<i64> = rows
        .iter()
        .filter_map(|r| match r.get(0) {
            Some(turdb::OwnedValue::Int(i)) => Some(*i),
            _ => None,
        })
        .collect();
    assert_eq!(
        ids,
        vec![1, 3],
        "Should have rows 1 and 3 (2 was rolled back)"
    );

    println!("=== Cloned Connections Independent Transactions Test PASSED ===");
}
