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
const TURDB_PATH: &str = "/Users/julfikar/Documents/PassionFruit.nosync/turdb/turdb-core/.worktrees/mvcc_test";
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
        println!("[Thread {}] Starting import of table: {}", thread_id, table.name);

        db.execute(table.turdb_ddl)?;

        let sqlite_conn = Connection::open(SQLITE_DB_PATH)?;

        let count: i64 = sqlite_conn.query_row(
            &format!("SELECT COUNT(*) FROM {}", table.name),
            [],
            |row| row.get(0),
        )?;
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
        println!("[Thread {}] {} committed successfully ({} rows)", thread_id, table.name, total_inserted);
        
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
    let expected_count: i64 = sqlite_conn.query_row(
        &format!("SELECT COUNT(*) FROM {}", table.name),
        [],
        |row| row.get(0),
    )?;

    Ok((count, count == expected_count as u64))
}

#[test]
fn concurrent_import_small_tables() {
    if !sqlite_db_exists() {
        eprintln!("Skipping test: SQLite database not found at {}", SQLITE_DB_PATH);
        return;
    }

    if Path::new(TURDB_PATH).exists() {
        std::fs::remove_dir_all(TURDB_PATH).expect("Failed to remove existing TurDB directory");
        println!("Removed existing TurDB directory at {}", TURDB_PATH);
    }

    let db = Database::create(TURDB_PATH).expect("Failed to create database");
    db.execute("PRAGMA WAL=ON").expect("Failed to enable WAL");
    db.execute("PRAGMA synchronous=NORMAL").expect("Failed to set synchronous mode");
    db.execute("SET foreign_keys = OFF").expect("Failed to disable foreign keys");

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
            thread::spawn(move || {
                import_table_concurrent(db_conn, table, idx, total_rows_clone)
            })
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
                    println!("✗ {} - row count mismatch: {} rows", camel_to_snake(table.name), count);
                }
            }
            Err(e) => {
                verification_failed += 1;
                println!("✗ {} - verification error: {}", camel_to_snake(table.name), e);
            }
        }
    }

    println!("\n=== Summary ===");
    println!("Tables imported:     {}/{}", success_count, SMALL_TABLES.len());
    println!("Import failures:     {}", failure_count);
    println!("Verification passed: {}/{}", verification_passed, SMALL_TABLES.len());
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
}

const CRUD_TEST_PATH: &str = "/Users/julfikar/Documents/PassionFruit.nosync/turdb/turdb-core/.worktrees/crud_test";

#[test]
fn concurrent_crud_operations() {
    if !sqlite_db_exists() {
        eprintln!("Skipping test: SQLite database not found at {}", SQLITE_DB_PATH);
        return;
    }

    if Path::new(CRUD_TEST_PATH).exists() {
        std::fs::remove_dir_all(CRUD_TEST_PATH).expect("Failed to remove existing test directory");
    }

    let db = Database::create(CRUD_TEST_PATH).expect("Failed to create database");
    db.execute("PRAGMA WAL=ON").expect("Failed to enable WAL");
    db.execute("PRAGMA synchronous=NORMAL").expect("Failed to set synchronous mode");

    println!("\n=== MVCC Concurrent CRUD Test (DatasetVersions) ===\n");

    let table = &SMALL_TABLES[9];
    assert_eq!(table.name, "DatasetVersions");

    println!("Phase 1: Import DatasetVersions from SQLite");
    let import_start = Instant::now();

    db.execute(table.turdb_ddl).expect("Failed to create table");

    let sqlite_conn = Connection::open(SQLITE_DB_PATH).expect("Failed to open SQLite");
    let total_count: i64 = sqlite_conn
        .query_row(&format!("SELECT COUNT(*) FROM {}", table.name), [], |row| row.get(0))
        .expect("Count failed");

    println!("  DatasetVersions has {} rows in SQLite", total_count);

    let col_count = table.columns.split(',').count();
    let turdb_table = camel_to_snake(table.name);

    db.execute("BEGIN").expect("Failed to begin");
    let mut offset: i64 = 0;
    let mut total_inserted: u64 = 0;

    loop {
        let query = format!(
            "SELECT {} FROM {} LIMIT {} OFFSET {}",
            table.columns, table.name, BATCH_SIZE, offset
        );
        let mut stmt = sqlite_conn.prepare(&query).expect("Prepare failed");
        let mut rows = stmt.query([]).expect("Query failed");
        let mut value_batches: Vec<String> = Vec::with_capacity(INSERT_BATCH_SIZE);
        let mut batch_count = 0u64;

        while let Some(row) = rows.next().expect("Row iteration failed") {
            let mut values = Vec::with_capacity(col_count);
            for i in 0..col_count {
                let val = row.get_ref(i).expect("Get ref failed").into();
                values.push(escape_sql_value(val));
            }
            value_batches.push(format!("({})", values.join(", ")));
            batch_count += 1;
            total_inserted += 1;

            if value_batches.len() >= INSERT_BATCH_SIZE {
                let insert_sql = format!("INSERT INTO {} VALUES {}", turdb_table, value_batches.join(", "));
                db.execute(&insert_sql).expect("Insert failed");
                value_batches.clear();
            }
        }

        if !value_batches.is_empty() {
            let insert_sql = format!("INSERT INTO {} VALUES {}", turdb_table, value_batches.join(", "));
            db.execute(&insert_sql).expect("Insert failed");
        }

        if batch_count == 0 {
            break;
        }
        offset += BATCH_SIZE;
        if offset >= total_count {
            break;
        }
    }
    db.execute("COMMIT").expect("Failed to commit");

    let import_elapsed = import_start.elapsed();
    println!("  Imported {} rows in {:.2}s", total_inserted, import_elapsed.as_secs_f64());

    let turdb_count = db.query("SELECT COUNT(*) FROM dataset_versions").expect("Count failed");
    let actual_count = match turdb_count[0].get(0) {
        Some(turdb::OwnedValue::Int(n)) => *n as u64,
        _ => 0,
    };
    assert_eq!(actual_count, total_inserted, "Import verification failed");
    println!("  Import verified: {} rows\n", actual_count);

    let all_ids: Vec<i64> = {
        let id_rows = db.query("SELECT id FROM dataset_versions").expect("ID query failed");
        id_rows
            .iter()
            .filter_map(|row| match row.get(0) {
                Some(turdb::OwnedValue::Int(n)) => Some(*n),
                _ => None,
            })
            .collect()
    };
    let total_rows = all_ids.len();
    println!("  Total IDs collected: {}\n", total_rows);

    println!("Phase 2: Concurrent SELECTs (sampling)");
    let num_threads = 8;
    let sample_per_thread = 500;

    let ids_arc = Arc::new(all_ids.clone());
    let barrier = Arc::new(Barrier::new(num_threads));
    let select_start = Instant::now();

    let select_handles: Vec<_> = (0..num_threads)
        .map(|thread_id| {
            let db_conn = db.clone();
            let barrier_clone = Arc::clone(&barrier);
            let ids = Arc::clone(&ids_arc);
            thread::spawn(move || {
                barrier_clone.wait();

                let step = ids.len() / (num_threads * sample_per_thread);
                let step = step.max(1);
                let start_offset = thread_id * step;

                let mut found_count = 0;
                for i in 0..sample_per_thread {
                    let idx = (start_offset + i * num_threads * step) % ids.len();
                    let id = ids[idx];
                    let sql = format!("SELECT id, dataset_id, title FROM dataset_versions WHERE id = {}", id);
                    let rows = db_conn.query(&sql).expect("Select failed");
                    if !rows.is_empty() {
                        found_count += 1;
                    }
                }

                println!("  [Thread {}] Selected {} rows", thread_id, found_count);
                found_count
            })
        })
        .collect();

    let select_counts: Vec<usize> = select_handles
        .into_iter()
        .map(|h| h.join().expect("Select thread panicked"))
        .collect();

    let select_elapsed = select_start.elapsed();
    let total_selected: usize = select_counts.iter().sum();
    println!("  Total selected: {} rows in {:.2}s", total_selected, select_elapsed.as_secs_f64());
    let expected_samples = num_threads * sample_per_thread;
    assert_eq!(total_selected, expected_samples, "SELECT verification failed");
    println!("  SELECT verification: {} sampled rows found\n", total_selected);

    println!("Phase 3: Concurrent UPDATEs (sampling version_number updates)");

    let update_per_thread = 500;
    let barrier = Arc::new(Barrier::new(num_threads));
    let update_start = Instant::now();

    let update_handles: Vec<_> = (0..num_threads)
        .map(|thread_id| {
            let db_conn = db.clone();
            let barrier_clone = Arc::clone(&barrier);
            let ids = Arc::clone(&ids_arc);
            thread::spawn(move || {
                barrier_clone.wait();

                db_conn.execute("BEGIN").expect("Failed to begin");

                let step = ids.len() / (num_threads * update_per_thread);
                let step = step.max(1);
                let start_offset = thread_id * step;

                let mut updated_ids: Vec<(usize, i64)> = Vec::with_capacity(update_per_thread);
                for i in 0..update_per_thread {
                    let idx = (start_offset + i * num_threads * step) % ids.len();
                    let id = ids[idx];
                    let new_version = (thread_id * 10000 + i) as f64;
                    let sql = format!(
                        "UPDATE dataset_versions SET version_number = {} WHERE id = {}",
                        new_version, id
                    );
                    db_conn.execute(&sql).expect("Update failed");
                    updated_ids.push((i, id));
                }

                db_conn.execute("COMMIT").expect("Failed to commit");
                println!("  [Thread {}] Updated {} rows", thread_id, updated_ids.len());
                (thread_id, updated_ids)
            })
        })
        .collect();

    let update_results: Vec<(usize, Vec<(usize, i64)>)> = update_handles
        .into_iter()
        .map(|h| h.join().expect("Update thread panicked"))
        .collect();

    let update_elapsed = update_start.elapsed();
    let total_updated: usize = update_results.iter().map(|(_, ids)| ids.len()).sum();
    println!("  Total updated: {} rows in {:.2}s", total_updated, update_elapsed.as_secs_f64());

    println!("\n  Verifying UPDATE results...");
    let mut update_verified = 0;
    let mut update_failed = 0;

    for (thread_id, updated_ids) in &update_results {
        for (i, id) in updated_ids {
            let sql = format!("SELECT version_number FROM dataset_versions WHERE id = {}", id);
            let rows = db.query(&sql).expect("Verify failed");

            if rows.is_empty() {
                update_failed += 1;
                continue;
            }

            let version = match rows[0].get(0) {
                Some(turdb::OwnedValue::Float(f)) => *f,
                Some(turdb::OwnedValue::Int(n)) => *n as f64,
                _ => -1.0,
            };

            let expected_version = (thread_id * 10000 + i) as f64;
            if (version - expected_version).abs() < 0.001 {
                update_verified += 1;
            } else {
                update_failed += 1;
                if update_failed <= 5 {
                    println!("    Mismatch at id {}: version={} (expected {})", id, version, expected_version);
                }
            }
        }
    }

    println!("  UPDATE verification: {}/{} rows correctly updated", update_verified, total_updated);
    assert_eq!(update_verified, total_updated, "UPDATE verification failed: {} rows not properly updated", update_failed);

    println!("\nPhase 4: Concurrent DELETEs (sampling deletions)");

    let delete_per_thread = 500;
    let total_to_delete = num_threads * delete_per_thread;

    let count_before_delete = db.query("SELECT COUNT(*) FROM dataset_versions").expect("Count failed");
    let count_before = match count_before_delete[0].get(0) {
        Some(turdb::OwnedValue::Int(n)) => *n as usize,
        _ => 0,
    };
    println!("  Rows before delete: {}", count_before);

    let ids_to_delete: Vec<i64> = all_ids.iter()
        .step_by(all_ids.len() / total_to_delete)
        .take(total_to_delete)
        .cloned()
        .collect();

    println!("  Will delete {} rows across {} threads", ids_to_delete.len(), num_threads);

    let ids_to_delete_arc = Arc::new(ids_to_delete.clone());

    let barrier = Arc::new(Barrier::new(num_threads));
    let delete_start = Instant::now();

    let delete_handles: Vec<_> = (0..num_threads)
        .map(|thread_id| {
            let db_conn = db.clone();
            let barrier_clone = Arc::clone(&barrier);
            let ids = Arc::clone(&ids_to_delete_arc);
            thread::spawn(move || {
                barrier_clone.wait();

                db_conn.execute("BEGIN").expect("Failed to begin");

                let start_idx = thread_id * delete_per_thread;
                let end_idx = start_idx + delete_per_thread;

                let mut deleted_count = 0;
                for idx in start_idx..end_idx.min(ids.len()) {
                    let id = ids[idx];
                    let sql = format!("DELETE FROM dataset_versions WHERE id = {}", id);
                    db_conn.execute(&sql).expect("Delete failed");
                    deleted_count += 1;
                }

                db_conn.execute("COMMIT").expect("Failed to commit");
                println!("  [Thread {}] Deleted {} rows", thread_id, deleted_count);
                deleted_count
            })
        })
        .collect();

    let delete_counts: Vec<usize> = delete_handles
        .into_iter()
        .map(|h| h.join().expect("Delete thread panicked"))
        .collect();

    let delete_elapsed = delete_start.elapsed();
    let total_deleted: usize = delete_counts.iter().sum();
    println!("  Total deleted: {} rows in {:.2}s", total_deleted, delete_elapsed.as_secs_f64());

    println!("\n  Verifying DELETE results...");

    let count_after_delete = db.query("SELECT COUNT(*) FROM dataset_versions").expect("Count failed");
    let remaining_count = match count_after_delete[0].get(0) {
        Some(turdb::OwnedValue::Int(n)) => *n as usize,
        _ => 0,
    };

    let expected_remaining = count_before - total_deleted;
    println!("  Rows remaining: {} (expected {})", remaining_count, expected_remaining);
    assert_eq!(remaining_count, expected_remaining, "DELETE count verification failed");

    let mut deleted_still_exist = 0;
    for id in ids_to_delete.iter().take(100) {
        let sql = format!("SELECT id FROM dataset_versions WHERE id = {}", id);
        let rows = db.query(&sql).expect("Verify failed");
        if !rows.is_empty() {
            deleted_still_exist += 1;
        }
    }
    println!("  Sampled 100 deleted IDs, {} still exist (should be 0)", deleted_still_exist);
    assert_eq!(deleted_still_exist, 0, "DELETE verification failed: deleted rows still exist");

    println!("\nPhase 5: Mixed concurrent operations on remaining data");

    let remaining_ids: Vec<i64> = {
        let id_rows = db.query("SELECT id FROM dataset_versions").expect("ID query failed");
        id_rows
            .iter()
            .filter_map(|row| match row.get(0) {
                Some(turdb::OwnedValue::Int(n)) => Some(*n),
                _ => None,
            })
            .collect()
    };
    let remaining_total = remaining_ids.len();
    println!("  Remaining rows for mixed ops: {}", remaining_total);

    let quarter = remaining_total / 4;
    let select_ids: Vec<i64> = remaining_ids[0..quarter].to_vec();
    let update_ids: Vec<i64> = remaining_ids[quarter..quarter * 2].to_vec();
    let delete_ids: Vec<i64> = remaining_ids[quarter * 2..quarter * 3].to_vec();

    let barrier = Arc::new(Barrier::new(4));
    let mixed_start = Instant::now();

    let insert_thread = {
        let db_conn = db.clone();
        let barrier_clone = Arc::clone(&barrier);
        thread::spawn(move || {
            barrier_clone.wait();
            db_conn.execute("BEGIN").expect("Begin failed");

            let max_id: i64 = db_conn
                .query("SELECT MAX(id) FROM dataset_versions")
                .expect("Max query failed")[0]
                .get(0)
                .map(|v| match v {
                    turdb::OwnedValue::Int(n) => *n,
                    _ => 0,
                })
                .unwrap_or(0);

            let mut inserted = 0;
            for i in 1..=50 {
                let new_id = max_id + i;
                let sql = format!(
                    "INSERT INTO dataset_versions (id, dataset_id, datasource_version_id, creator_user_id, license_name, creation_date, version_number, title, slug, subtitle, description, version_notes, total_compressed_bytes, total_uncompressed_bytes) VALUES ({}, 999, 999, 999, 'MIT', '2024-01-01', {}, 'Test Title {}', 'test-slug-{}', 'Subtitle', 'Desc', 'Notes', 1000, 2000)",
                    new_id, i as f64, i, i
                );
                db_conn.execute(&sql).expect("Insert failed");
                inserted += 1;
            }

            db_conn.execute("COMMIT").expect("Commit failed");
            println!("  [INSERT] Completed {} inserts", inserted);
            inserted
        })
    };

    let select_thread = {
        let db_conn = db.clone();
        let barrier_clone = Arc::clone(&barrier);
        let ids = select_ids.clone();
        thread::spawn(move || {
            barrier_clone.wait();
            let mut count = 0;
            for id in ids.iter().take(100) {
                let sql = format!("SELECT * FROM dataset_versions WHERE id = {}", id);
                let rows = db_conn.query(&sql).expect("Select failed");
                if !rows.is_empty() {
                    count += 1;
                }
            }
            println!("  [SELECT] Completed {} selects", count);
            count
        })
    };

    let update_thread = {
        let db_conn = db.clone();
        let barrier_clone = Arc::clone(&barrier);
        let ids = update_ids.clone();
        thread::spawn(move || {
            barrier_clone.wait();
            db_conn.execute("BEGIN").expect("Begin failed");
            let mut updated = 0;
            for (i, id) in ids.iter().take(50).enumerate() {
                let sql = format!(
                    "UPDATE dataset_versions SET version_number = {} WHERE id = {}",
                    9999.0 + i as f64, id
                );
                db_conn.execute(&sql).expect("Update failed");
                updated += 1;
            }
            db_conn.execute("COMMIT").expect("Commit failed");
            println!("  [UPDATE] Completed {} updates", updated);
            updated
        })
    };

    let delete_thread = {
        let db_conn = db.clone();
        let barrier_clone = Arc::clone(&barrier);
        let ids = delete_ids.clone();
        thread::spawn(move || {
            barrier_clone.wait();
            db_conn.execute("BEGIN").expect("Begin failed");
            let mut deleted = 0;
            for id in ids.iter().take(50) {
                let sql = format!("DELETE FROM dataset_versions WHERE id = {}", id);
                db_conn.execute(&sql).expect("Delete failed");
                deleted += 1;
            }
            db_conn.execute("COMMIT").expect("Commit failed");
            println!("  [DELETE] Completed {} deletes", deleted);
            deleted
        })
    };

    let insert_result = insert_thread.join().expect("Insert thread panicked");
    let select_result = select_thread.join().expect("Select thread panicked");
    let update_result = update_thread.join().expect("Update thread panicked");
    let delete_result = delete_thread.join().expect("Delete thread panicked");

    let mixed_elapsed = mixed_start.elapsed();
    println!("  Mixed operations completed in {:.2}s", mixed_elapsed.as_secs_f64());

    assert_eq!(insert_result, 50, "Mixed INSERT failed");
    assert!(select_result > 0, "Mixed SELECT failed");
    assert_eq!(update_result, 50, "Mixed UPDATE failed");
    assert_eq!(delete_result, 50, "Mixed DELETE failed");

    println!("\n  Verifying mixed operations...");

    let mut mixed_update_verified = 0;
    for (i, id) in update_ids.iter().take(50).enumerate() {
        let sql = format!("SELECT version_number FROM dataset_versions WHERE id = {}", id);
        let rows = db.query(&sql).expect("Verify failed");
        if !rows.is_empty() {
            let version = match rows[0].get(0) {
                Some(turdb::OwnedValue::Float(f)) => *f,
                Some(turdb::OwnedValue::Int(n)) => *n as f64,
                _ => -1.0,
            };
            let expected = 9999.0 + i as f64;
            if (version - expected).abs() < 0.001 {
                mixed_update_verified += 1;
            }
        }
    }
    println!("  Mixed UPDATE verification: {}/50 rows correctly updated", mixed_update_verified);
    assert_eq!(mixed_update_verified, 50, "Mixed UPDATE verification failed");

    let mut mixed_delete_verified = 0;
    for id in delete_ids.iter().take(50) {
        let sql = format!("SELECT id FROM dataset_versions WHERE id = {}", id);
        let rows = db.query(&sql).expect("Verify failed");
        if rows.is_empty() {
            mixed_delete_verified += 1;
        }
    }
    println!("  Mixed DELETE verification: {}/50 rows properly deleted", mixed_delete_verified);
    assert_eq!(mixed_delete_verified, 50, "Mixed DELETE verification failed");

    let final_count = db.query("SELECT COUNT(*) FROM dataset_versions").expect("Count failed");
    let final_rows = match final_count[0].get(0) {
        Some(turdb::OwnedValue::Int(n)) => *n as usize,
        _ => 0,
    };

    println!("\n=== Summary ===");
    println!("Initial import:    {} rows", total_inserted);
    println!("Phase 2 (SELECT):  {} rows queried", total_selected);
    println!("Phase 3 (UPDATE):  {} rows updated", total_updated);
    println!("Phase 4 (DELETE):  {} rows deleted", total_deleted);
    println!("Phase 5 (MIXED):   INSERT={}, SELECT={}, UPDATE={}, DELETE={}",
             insert_result, select_result, update_result, delete_result);
    println!("\nFinal row count:   {}", final_rows);

    println!("\n=== MVCC Concurrent CRUD Test PASSED ===\n");
}
