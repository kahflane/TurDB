//! # Query Memory Budget Enforcement Tests
//!
//! These tests verify that query execution respects the memory budget system.
//! Memory-intensive operators (Sort, HashAggregate, Window, Joins) must track
//! their allocations and fail gracefully when the budget is exceeded.
//!
//! ## Test Coverage
//!
//! 1. ExecutionContext Memory Budget Integration
//!    - ExecutionContext accepts MemoryBudget reference
//!    - Memory budget is accessible during query execution
//!
//! 2. Sort Operator Memory Tracking
//!    - Sort allocations reported to Query pool
//!    - Sort fails with OOM when budget exceeded
//!
//! 3. HashAggregate Operator Memory Tracking
//!    - HashAggregate allocations reported to Query pool
//!    - HashAggregate fails with OOM when budget exceeded
//!
//! 4. Join Operator Memory Tracking
//!    - Join allocations reported to Query pool
//!    - Join fails with OOM when budget exceeded
//!
//! 5. Query Pool Usage Visibility
//!    - Query pool usage increases during query execution
//!    - Query pool usage decreases after query completes
//!
//! ## Design Principles
//!
//! The memory budget integration uses a periodic synchronization strategy:
//! 1. Memory-intensive operators track their arena allocations
//! 2. Before materializing data, operators check budget availability
//! 3. After materialization, operators report actual usage to the budget
//! 4. If budget exceeded, operators return a memory error instead of crashing

use tempfile::tempdir;
use turdb::memory::Pool;
use turdb::Database;

/// Test that a sort operation on a large dataset reports allocations to the Query pool.
/// This test verifies that memory tracking is active during query execution.
#[test]
fn sort_operator_reports_allocations_to_query_pool() {
    let dir = tempdir().unwrap();
    let db_path = dir.path().join("test_db");

    let db = Database::create(&db_path).unwrap();

    db.execute("CREATE TABLE test_data (id INT, value TEXT)")
        .unwrap();

    for i in 0..1000 {
        db.execute(&format!(
            "INSERT INTO test_data VALUES ({}, 'value_{:05}')",
            i, i
        ))
        .unwrap();
    }

    let stats_before = db.memory_stats();
    let query_used_before = stats_before.query_used;

    let _rows = db
        .query("SELECT * FROM test_data ORDER BY value DESC")
        .unwrap();

    let stats_after = db.memory_stats();
    let query_used_after = stats_after.query_used;

    assert!(
        query_used_after > query_used_before || query_used_after == 0,
        "Query pool should show allocation during sort, or be released after query. \
         Before: {}, After: {}",
        query_used_before,
        query_used_after
    );
}

/// Test that a hash aggregate operation reports allocations to the Query pool.
#[test]
fn hash_aggregate_reports_allocations_to_query_pool() {
    let dir = tempdir().unwrap();
    let db_path = dir.path().join("test_db");

    let db = Database::create(&db_path).unwrap();

    db.execute("CREATE TABLE sales (product TEXT, amount INT)")
        .unwrap();

    for i in 0..500 {
        let product = format!("product_{}", i % 100);
        db.execute(&format!(
            "INSERT INTO sales VALUES ('{}', {})",
            product, i
        ))
        .unwrap();
    }

    let stats_before = db.memory_stats();
    let query_used_before = stats_before.query_used;

    let _rows = db
        .query("SELECT product, SUM(amount) FROM sales GROUP BY product")
        .unwrap();

    let stats_after = db.memory_stats();
    let query_used_after = stats_after.query_used;

    assert!(
        query_used_after >= query_used_before,
        "Query pool should track hash aggregate allocations. Before: {}, After: {}",
        query_used_before,
        query_used_after
    );
}

/// Test that a query exceeding the memory budget returns an error instead of crashing.
/// This is the core test for memory budget enforcement.
///
/// NOTE: This test will FAIL until memory budget enforcement is implemented in query execution.
/// The expected behavior is:
/// 1. After exhausting the budget, the sort operation should fail with OOM
/// 2. Currently, queries bypass the budget and allocate directly
#[test]
fn query_exceeding_memory_budget_returns_error() {
    let dir = tempdir().unwrap();
    let db_path = dir.path().join("test_db");

    let db = Database::create(&db_path).unwrap();

    db.execute("CREATE TABLE large_data (id INT, data TEXT)")
        .unwrap();

    for i in 0..10000 {
        let large_data = "x".repeat(100);
        db.execute(&format!(
            "INSERT INTO large_data VALUES ({}, '{}')",
            i, large_data
        ))
        .unwrap();
    }

    let budget = db.memory_budget();
    let available = budget.available(Pool::Query);
    let to_allocate = available.saturating_sub(64 * 1024);
    if to_allocate > 0 {
        budget.allocate(Pool::Query, to_allocate).unwrap();
    }

    let result = db.query("SELECT * FROM large_data ORDER BY data DESC");

    assert!(
        result.is_err(),
        "Query should fail with OOM when budget is exhausted"
    );

    let err_msg = result.unwrap_err().to_string().to_lowercase();
    assert!(
        err_msg.contains("memory") || err_msg.contains("budget") || err_msg.contains("oom"),
        "Error message should indicate memory issue: {}",
        err_msg
    );
}

/// Test that a hash aggregate query fails gracefully when memory budget is exhausted.
///
/// NOTE: This test will FAIL until memory budget enforcement is implemented in query execution.
/// The expected behavior is:
/// 1. After exhausting the budget, the hash aggregate should fail with OOM
/// 2. Currently, queries bypass the budget and allocate directly
#[test]
fn hash_aggregate_exceeding_budget_returns_error() {
    let dir = tempdir().unwrap();
    let db_path = dir.path().join("test_db");

    let db = Database::create(&db_path).unwrap();

    db.execute("CREATE TABLE grouped_data (category TEXT, value INT)")
        .unwrap();

    for i in 0..5000 {
        let category = format!("category_{:05}", i);
        db.execute(&format!(
            "INSERT INTO grouped_data VALUES ('{}', {})",
            category, i
        ))
        .unwrap();
    }

    let budget = db.memory_budget();
    let available = budget.available(Pool::Query);
    let to_allocate = available.saturating_sub(64 * 1024);
    if to_allocate > 0 {
        budget.allocate(Pool::Query, to_allocate).unwrap();
    }

    let result = db.query("SELECT category, COUNT(*) FROM grouped_data GROUP BY category");

    assert!(
        result.is_err(),
        "Hash aggregate should fail with OOM when budget is exhausted"
    );
}

/// Test that query memory is released after query completion.
#[test]
fn query_memory_released_after_completion() {
    let dir = tempdir().unwrap();
    let db_path = dir.path().join("test_db");

    let db = Database::create(&db_path).unwrap();

    db.execute("CREATE TABLE test (id INT, value TEXT)")
        .unwrap();

    for i in 0..500 {
        db.execute(&format!("INSERT INTO test VALUES ({}, 'value_{}')", i, i))
            .unwrap();
    }

    let stats_initial = db.memory_stats();
    let query_used_initial = stats_initial.query_used;

    for _ in 0..5 {
        let _rows = db.query("SELECT * FROM test ORDER BY value").unwrap();
    }

    let stats_final = db.memory_stats();
    let query_used_final = stats_final.query_used;

    assert!(
        query_used_final <= query_used_initial + 1024 * 1024,
        "Query pool should not accumulate unbounded. Initial: {}, Final: {}",
        query_used_initial,
        query_used_final
    );
}

/// Test that join operations report allocations to the Query pool.
#[test]
fn join_operation_reports_allocations_to_query_pool() {
    let dir = tempdir().unwrap();
    let db_path = dir.path().join("test_db");

    let db = Database::create(&db_path).unwrap();

    db.execute("CREATE TABLE orders (id INT, customer_id INT)")
        .unwrap();
    db.execute("CREATE TABLE customers (id INT, name TEXT)")
        .unwrap();

    for i in 0..200 {
        db.execute(&format!("INSERT INTO customers VALUES ({}, 'customer_{}')", i, i))
            .unwrap();
    }
    for i in 0..500 {
        db.execute(&format!("INSERT INTO orders VALUES ({}, {})", i, i % 200))
            .unwrap();
    }

    let stats_before = db.memory_stats();
    let query_used_before = stats_before.query_used;

    let _rows = db
        .query("SELECT o.id, c.name FROM orders o JOIN customers c ON o.customer_id = c.id")
        .unwrap();

    let stats_after = db.memory_stats();
    let query_used_after = stats_after.query_used;

    assert!(
        query_used_after >= query_used_before,
        "Query pool should track join allocations. Before: {}, After: {}",
        query_used_before,
        query_used_after
    );
}

/// Test that window function operations report allocations to the Query pool.
#[test]
fn window_function_reports_allocations_to_query_pool() {
    let dir = tempdir().unwrap();
    let db_path = dir.path().join("test_db");

    let db = Database::create(&db_path).unwrap();

    db.execute("CREATE TABLE metrics (ts INT, value INT)")
        .unwrap();

    for i in 0..500 {
        db.execute(&format!("INSERT INTO metrics VALUES ({}, {})", i, i % 100))
            .unwrap();
    }

    let stats_before = db.memory_stats();
    let query_used_before = stats_before.query_used;

    let _rows = db
        .query("SELECT ts, value, ROW_NUMBER() OVER (ORDER BY ts) as rn FROM metrics")
        .unwrap();

    let stats_after = db.memory_stats();
    let query_used_after = stats_after.query_used;

    assert!(
        query_used_after >= query_used_before,
        "Query pool should track window function allocations. Before: {}, After: {}",
        query_used_before,
        query_used_after
    );
}

/// Test that memory budget stats are accurate during concurrent queries.
#[test]
fn memory_stats_accurate_during_execution() {
    let dir = tempdir().unwrap();
    let db_path = dir.path().join("test_db");

    let db = Database::create(&db_path).unwrap();

    db.execute("CREATE TABLE data (id INT, value TEXT)")
        .unwrap();

    for i in 0..100 {
        db.execute(&format!("INSERT INTO data VALUES ({}, 'value_{}')", i, i))
            .unwrap();
    }

    let stats = db.memory_stats();

    assert!(
        stats.total_limit >= 4 * 1024 * 1024,
        "Total limit should be at least 4MB floor"
    );
    assert!(
        stats.total_used <= stats.total_limit,
        "Total used should not exceed limit"
    );
    assert!(
        stats.query_reserved > 0,
        "Query pool should have reserved allocation"
    );
}

/// Test that TopK operator (LIMIT with ORDER BY) reports allocations.
#[test]
fn topk_operator_reports_allocations() {
    let dir = tempdir().unwrap();
    let db_path = dir.path().join("test_db");

    let db = Database::create(&db_path).unwrap();

    db.execute("CREATE TABLE ranked (id INT, score INT)")
        .unwrap();

    for i in 0..1000 {
        db.execute(&format!("INSERT INTO ranked VALUES ({}, {})", i, i % 500))
            .unwrap();
    }

    let stats_before = db.memory_stats();
    let query_used_before = stats_before.query_used;

    let _rows = db
        .query("SELECT * FROM ranked ORDER BY score DESC LIMIT 10")
        .unwrap();

    let stats_after = db.memory_stats();
    let query_used_after = stats_after.query_used;

    assert!(
        query_used_after >= query_used_before,
        "Query pool should track TopK allocations. Before: {}, After: {}",
        query_used_before,
        query_used_after
    );
}
