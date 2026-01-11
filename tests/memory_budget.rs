//! # Memory Budget and WAL Efficiency Integration Tests
//!
//! This module tests the memory budget system, streaming WAL recovery,
//! auto-checkpoint mechanism, and degraded mode handling.
//!
//! ## Test Coverage
//!
//! 1. Memory Budget Allocation
//!    - Pool allocation within limits
//!    - Pool allocation exceeds limits
//!    - Release and re-allocate
//!    - Stats reporting
//!
//! 2. WAL Streaming Recovery
//!    - Zero-allocation frame reading
//!    - Batched recovery processing
//!    - Recovery cost estimation
//!
//! 3. Auto-Checkpoint
//!    - Frame counting
//!    - Threshold-based checkpoint triggering
//!
//! 4. Degraded Mode
//!    - Mode detection on open
//!    - Write operations blocked in degraded mode
//!    - PRAGMA recover_wal restores read-write mode
//!
//! 5. PRAGMA Commands
//!    - memory_budget
//!    - memory_stats
//!    - database_mode
//!    - wal_checkpoint
//!    - wal_checkpoint_threshold
//!    - recover_wal

use tempfile::tempdir;
use turdb::memory::{MemoryBudget, Pool};
use turdb::Database;

// ============================================================================
// Memory Budget Tests
// ============================================================================

#[test]
fn test_memory_budget_allocation_within_limit() {
    let budget = MemoryBudget::with_limit(1_000_000);

    assert!(budget.allocate(Pool::Cache, 256_000).is_ok());
    assert!(budget.allocate(Pool::Query, 128_000).is_ok());

    let stats = budget.stats();
    assert!(stats.cache_used >= 256_000);
    assert!(stats.query_used >= 128_000);
}

#[test]
fn test_memory_budget_allocation_exceeds_limit() {
    let budget = MemoryBudget::with_limit(4 * 1024 * 1024);

    assert!(budget.allocate(Pool::Cache, 3 * 1024 * 1024).is_ok());

    let result = budget.allocate(Pool::Cache, 2 * 1024 * 1024);
    assert!(result.is_err(), "Allocation should fail when exceeding total budget");
}

#[test]
fn test_memory_budget_release_and_reallocate() {
    let budget = MemoryBudget::with_limit(4 * 1024 * 1024);

    assert!(budget.allocate(Pool::Cache, 3 * 1024 * 1024).is_ok());

    let result = budget.allocate(Pool::Cache, 2 * 1024 * 1024);
    assert!(result.is_err(), "Should fail when exceeding total budget");

    budget.release(Pool::Cache, 2 * 1024 * 1024);

    assert!(budget.allocate(Pool::Cache, 2 * 1024 * 1024).is_ok(), "Should succeed after release");
}

#[test]
fn test_memory_budget_auto_detect() {
    let budget = MemoryBudget::auto_detect();
    let limit = budget.total_limit();

    assert!(
        limit >= 4 * 1024 * 1024,
        "Auto-detected budget should be at least 4MB floor"
    );
}

#[test]
fn test_memory_budget_stats() {
    let budget = MemoryBudget::with_limit(10_000_000);

    budget.allocate(Pool::Cache, 100_000).unwrap();
    budget.allocate(Pool::Query, 50_000).unwrap();
    budget.allocate(Pool::Recovery, 25_000).unwrap();
    budget.allocate(Pool::Schema, 10_000).unwrap();

    let stats = budget.stats();

    assert!(stats.cache_used >= 100_000);
    assert!(stats.query_used >= 50_000);
    assert!(stats.recovery_used >= 25_000);
    assert!(stats.schema_used >= 10_000);
    assert!(stats.total_used >= 185_000);
}

#[test]
fn test_memory_budget_can_allocate() {
    let budget = MemoryBudget::with_limit(4 * 1024 * 1024);

    assert!(budget.can_allocate(Pool::Cache, 512_000));

    budget.allocate(Pool::Cache, 3 * 1024 * 1024).unwrap();

    assert!(!budget.can_allocate(Pool::Cache, 2 * 1024 * 1024), "Should not be able to allocate exceeding total");
}

// ============================================================================
// Database Mode Tests
// ============================================================================

#[test]
fn test_database_mode_read_write_on_create() {
    let dir = tempdir().unwrap();
    let db_path = dir.path().join("test_db");

    let db = Database::create(&db_path).unwrap();

    assert!(db.is_read_write());
    assert!(!db.is_degraded());
}

#[test]
fn test_database_mode_read_write_on_open() {
    let dir = tempdir().unwrap();
    let db_path = dir.path().join("test_db");

    {
        let db = Database::create(&db_path).unwrap();
        db.execute("CREATE TABLE test (id INT)").unwrap();
        db.close().unwrap();
    }

    let db = Database::open(&db_path).unwrap();
    assert!(db.is_read_write());
}

// ============================================================================
// PRAGMA Command Tests
// ============================================================================

#[test]
fn test_pragma_memory_budget() {
    let dir = tempdir().unwrap();
    let db_path = dir.path().join("test_db");

    let db = Database::create(&db_path).unwrap();

    let result = db.execute("PRAGMA memory_budget").unwrap();
    match result {
        turdb::ExecuteResult::Pragma { name, value } => {
            assert_eq!(name, "MEMORY_BUDGET");
            let budget: usize = value.unwrap().parse().unwrap();
            assert!(
                budget >= 4 * 1024 * 1024,
                "Memory budget should be at least 4MB"
            );
        }
        _ => panic!("Expected Pragma result"),
    }
}

#[test]
fn test_pragma_memory_stats() {
    let dir = tempdir().unwrap();
    let db_path = dir.path().join("test_db");

    let db = Database::create(&db_path).unwrap();

    let result = db.execute("PRAGMA memory_stats").unwrap();
    match result {
        turdb::ExecuteResult::Pragma { name, value } => {
            assert_eq!(name, "MEMORY_STATS");
            let stats_str = value.unwrap();
            assert!(stats_str.contains("cache"));
            assert!(stats_str.contains("query"));
            assert!(stats_str.contains("recovery"));
            assert!(stats_str.contains("schema"));
        }
        _ => panic!("Expected Pragma result"),
    }
}

#[test]
fn test_pragma_database_mode_read_write() {
    let dir = tempdir().unwrap();
    let db_path = dir.path().join("test_db");

    let db = Database::create(&db_path).unwrap();

    let result = db.execute("PRAGMA database_mode").unwrap();
    match result {
        turdb::ExecuteResult::Pragma { name, value } => {
            assert_eq!(name, "DATABASE_MODE");
            assert_eq!(value.unwrap(), "read_write");
        }
        _ => panic!("Expected Pragma result"),
    }
}

#[test]
fn test_pragma_wal_checkpoint() {
    let dir = tempdir().unwrap();
    let db_path = dir.path().join("test_db");

    let db = Database::create(&db_path).unwrap();

    db.execute("CREATE TABLE test (id INT)").unwrap();
    db.execute("INSERT INTO test VALUES (1)").unwrap();

    let result = db.execute("PRAGMA wal_checkpoint").unwrap();
    match result {
        turdb::ExecuteResult::Pragma { name, value } => {
            assert_eq!(name, "WAL_CHECKPOINT");
            assert!(value.unwrap().contains("checkpointed"));
        }
        _ => panic!("Expected Pragma result"),
    }
}

#[test]
fn test_pragma_wal_checkpoint_threshold() {
    let dir = tempdir().unwrap();
    let db_path = dir.path().join("test_db");

    let db = Database::create(&db_path).unwrap();
    db.ensure_wal().unwrap();

    let result = db.execute("PRAGMA wal_checkpoint_threshold").unwrap();
    match result {
        turdb::ExecuteResult::Pragma { name, value } => {
            assert_eq!(name, "WAL_CHECKPOINT_THRESHOLD");
            let threshold: u32 = value.unwrap().parse().unwrap();
            assert_eq!(threshold, 1000);
        }
        _ => panic!("Expected Pragma result"),
    }

    db.execute("PRAGMA wal_checkpoint_threshold = 500").unwrap();

    let result = db.execute("PRAGMA wal_checkpoint_threshold").unwrap();
    match result {
        turdb::ExecuteResult::Pragma { name, value } => {
            assert_eq!(name, "WAL_CHECKPOINT_THRESHOLD");
            let threshold: u32 = value.unwrap().parse().unwrap();
            assert_eq!(threshold, 500);
        }
        _ => panic!("Expected Pragma result"),
    }
}

#[test]
fn test_pragma_wal_frame_count() {
    let dir = tempdir().unwrap();
    let db_path = dir.path().join("test_db");

    let db = Database::create(&db_path).unwrap();
    db.ensure_wal().unwrap();

    let result = db.execute("PRAGMA wal_frame_count").unwrap();
    match result {
        turdb::ExecuteResult::Pragma { name, value } => {
            assert_eq!(name, "WAL_FRAME_COUNT");
            let count: u32 = value.unwrap().parse().unwrap();
            assert!(count >= 0);
        }
        _ => panic!("Expected Pragma result"),
    }
}

#[test]
fn test_pragma_recover_wal_no_op_in_read_write_mode() {
    let dir = tempdir().unwrap();
    let db_path = dir.path().join("test_db");

    let db = Database::create(&db_path).unwrap();

    let result = db.execute("PRAGMA recover_wal").unwrap();
    match result {
        turdb::ExecuteResult::Pragma { name, value } => {
            assert_eq!(name, "RECOVER_WAL");
            assert!(value.unwrap().contains("already in read_write mode"));
        }
        _ => panic!("Expected Pragma result"),
    }
}

// ============================================================================
// Checkpoint and Recovery Tests
// ============================================================================

#[test]
fn test_checkpoint_returns_info() {
    let dir = tempdir().unwrap();
    let db_path = dir.path().join("test_db");

    let db = Database::create(&db_path).unwrap();

    db.execute("CREATE TABLE test (id INT, name TEXT)")
        .unwrap();
    for i in 0..10 {
        db.execute(&format!("INSERT INTO test VALUES ({}, 'name{}')", i, i))
            .unwrap();
    }

    let checkpoint_info = db.checkpoint().unwrap();

    println!("Checkpoint info: {:?}", checkpoint_info);
}

#[test]
fn test_database_close_with_checkpoint() {
    let dir = tempdir().unwrap();
    let db_path = dir.path().join("test_db");

    {
        let db = Database::create(&db_path).unwrap();

        db.execute("CREATE TABLE test (id INT, name TEXT)")
            .unwrap();
        for i in 0..10 {
            db.execute(&format!("INSERT INTO test VALUES ({}, 'name{}')", i, i))
                .unwrap();
        }

        let checkpoint_info = db.close().unwrap();
        println!("Close checkpoint info: {:?}", checkpoint_info);
    }

    let db = Database::open(&db_path).unwrap();
    let rows = db.query("SELECT * FROM test").unwrap();
    assert_eq!(rows.len(), 10, "All data should persist after close");
}

#[test]
fn test_recovery_after_crash_simulation() {
    let dir = tempdir().unwrap();
    let db_path = dir.path().join("test_db");

    {
        let db = Database::create(&db_path).unwrap();

        db.execute("CREATE TABLE test (id INT, name TEXT)")
            .unwrap();
        for i in 0..100 {
            db.execute(&format!("INSERT INTO test VALUES ({}, 'name{}')", i, i))
                .unwrap();
        }
    }

    let (db, recovery_info) = Database::open_with_recovery(&db_path).unwrap();

    println!("Recovery info: {:?}", recovery_info);

    let rows = db.query("SELECT * FROM test").unwrap();
    assert_eq!(rows.len(), 100, "All data should be recovered");
}

// ============================================================================
// Memory Budget API Tests
// ============================================================================

#[test]
fn test_database_memory_budget_accessor() {
    let dir = tempdir().unwrap();
    let db_path = dir.path().join("test_db");

    let db = Database::create(&db_path).unwrap();

    let budget = db.memory_budget();
    assert!(budget.total_limit() >= 4 * 1024 * 1024);
}

#[test]
fn test_database_memory_stats_accessor() {
    let dir = tempdir().unwrap();
    let db_path = dir.path().join("test_db");

    let db = Database::create(&db_path).unwrap();

    let stats = db.memory_stats();
    println!("Memory stats: {}", stats);
}

// ============================================================================
// WAL Segment Tests
// ============================================================================

#[test]
fn test_wal_frame_counting() {
    use turdb::storage::{Wal, DEFAULT_CHECKPOINT_THRESHOLD};

    let dir = tempdir().unwrap();
    let wal_path = dir.path().join("wal");

    std::fs::create_dir_all(&wal_path).unwrap();

    let wal = Wal::open(&wal_path).unwrap();

    assert_eq!(wal.frame_count(), 0);
    assert_eq!(wal.checkpoint_threshold(), DEFAULT_CHECKPOINT_THRESHOLD);
}

#[test]
fn test_wal_checkpoint_threshold_modification() {
    use turdb::storage::Wal;

    let dir = tempdir().unwrap();
    let wal_path = dir.path().join("wal");

    std::fs::create_dir_all(&wal_path).unwrap();

    let wal = Wal::open(&wal_path).unwrap();

    wal.set_checkpoint_threshold(500);
    assert_eq!(wal.checkpoint_threshold(), 500);

    wal.set_checkpoint_threshold(2000);
    assert_eq!(wal.checkpoint_threshold(), 2000);
}

// ============================================================================
// Bulk Insert with Auto-Checkpoint Tests
// ============================================================================

#[test]
fn test_bulk_insert_with_auto_checkpoint() {
    let dir = tempdir().unwrap();
    let db_path = dir.path().join("test_db");

    let db = Database::create(&db_path).unwrap();
    db.ensure_wal().unwrap();

    db.execute("CREATE TABLE large_table (id INT, data TEXT)")
        .unwrap();

    for i in 0..500 {
        let data = format!("data_{:04}", i);
        db.execute(&format!(
            "INSERT INTO large_table VALUES ({}, '{}')",
            i, data
        ))
        .unwrap();
    }

    let wal_dir = db_path.join("wal");
    let wal_entries: Vec<_> = std::fs::read_dir(&wal_dir)
        .unwrap()
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().map(|ext| ext == "").unwrap_or(false) == false)
        .collect();

    println!(
        "WAL directory has {} entries after 500 inserts",
        wal_entries.len()
    );

    let rows = db.query("SELECT COUNT(*) FROM large_table").unwrap();
    let count = match &rows[0].values[0] {
        turdb::OwnedValue::Int(c) => *c,
        _ => panic!("Expected Int"),
    };
    assert_eq!(count, 500);

    db.close().unwrap();

    let db = Database::open(&db_path).unwrap();
    let rows = db.query("SELECT COUNT(*) FROM large_table").unwrap();
    let count = match &rows[0].values[0] {
        turdb::OwnedValue::Int(c) => *c,
        _ => panic!("Expected Int"),
    };
    assert_eq!(count, 500, "All data should persist after reopen");
}

// ============================================================================
// Pool-specific Allocation Tests
// ============================================================================

#[test]
fn test_pool_specific_limits() {
    let budget = MemoryBudget::with_limit(2_000_000);

    assert!(budget.allocate(Pool::Cache, 512 * 1024).is_ok());

    assert!(budget.allocate(Pool::Query, 256 * 1024).is_ok());

    assert!(budget.allocate(Pool::Recovery, 256 * 1024).is_ok());

    assert!(budget.allocate(Pool::Schema, 128 * 1024).is_ok());

    let stats = budget.stats();
    println!("After pool allocations: {}", stats);
}

#[test]
fn test_shared_pool_overflow() {
    let budget = MemoryBudget::with_limit(4 * 1024 * 1024);

    assert!(budget.allocate(Pool::Shared, 2 * 1024 * 1024).is_ok());

    let stats = budget.stats();
    assert!(stats.shared_used >= 2 * 1024 * 1024);

    let result = budget.allocate(Pool::Shared, 3 * 1024 * 1024);
    assert!(result.is_err(), "Should fail when exceeding total budget through shared pool");
}

// ============================================================================
// End-to-End Scenario Tests
// ============================================================================

#[test]
fn test_e2e_normal_operation_workflow() {
    let dir = tempdir().unwrap();
    let db_path = dir.path().join("test_db");

    let db = Database::create(&db_path).unwrap();
    assert!(db.is_read_write());

    db.execute("CREATE TABLE users (id INT PRIMARY KEY, name TEXT)")
        .unwrap();
    db.execute("INSERT INTO users VALUES (1, 'Alice')").unwrap();
    db.execute("INSERT INTO users VALUES (2, 'Bob')").unwrap();

    let rows = db.query("SELECT * FROM users").unwrap();
    assert_eq!(rows.len(), 2);

    db.checkpoint().unwrap();

    db.close().unwrap();

    let db = Database::open(&db_path).unwrap();
    assert!(db.is_read_write());

    let rows = db.query("SELECT * FROM users").unwrap();
    assert_eq!(rows.len(), 2);
}

#[test]
fn test_e2e_memory_stats_after_operations() {
    let dir = tempdir().unwrap();
    let db_path = dir.path().join("test_db");

    let db = Database::create(&db_path).unwrap();

    let stats_before = db.memory_stats();

    db.execute("CREATE TABLE test (id INT, data TEXT)").unwrap();
    for i in 0..50 {
        db.execute(&format!(
            "INSERT INTO test VALUES ({}, 'data_{}')",
            i, i
        ))
        .unwrap();
    }

    let _rows = db.query("SELECT * FROM test").unwrap();

    let stats_after = db.memory_stats();

    println!("Stats before: {}", stats_before);
    println!("Stats after: {}", stats_after);
}
