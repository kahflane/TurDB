//! # MVCC Integration Tests
//!
//! Tests for Multi-Version Concurrency Control (MVCC) implementation.
//! These tests verify Snapshot Isolation semantics.

use tempfile::TempDir;
use turdb::Database;

fn create_test_db() -> (TempDir, Database) {
    let dir = tempfile::tempdir().unwrap();
    let db = Database::create(dir.path()).unwrap();
    (dir, db)
}

fn execute_sql(db: &Database, sql: &str) {
    let result = db.execute(sql);
    if let Err(e) = &result {
        panic!("SQL failed: {}\nError: {:?}", sql, e);
    }
}

fn query_count(db: &Database, table_name: &str) -> i64 {
    let sql = format!("SELECT COUNT(*) FROM {}", table_name);
    let rows = db.query(&sql).unwrap();
    if let Some(row) = rows.into_iter().next() {
        if let Some(turdb::OwnedValue::Int(n)) = row.values.into_iter().next() {
            return n;
        }
    }
    0
}

fn query_value(db: &Database, sql: &str) -> Option<turdb::OwnedValue> {
    let rows = db.query(sql).unwrap();
    rows.into_iter()
        .next()
        .and_then(|row| row.values.into_iter().next())
}

#[test]
fn test_basic_transaction_visibility() {
    let (_dir, db) = create_test_db();

    execute_sql(&db, "CREATE TABLE users (id INT PRIMARY KEY, name TEXT)");
    execute_sql(&db, "INSERT INTO users (id, name) VALUES (1, 'Alice')");

    execute_sql(&db, "BEGIN");
    execute_sql(&db, "INSERT INTO users (id, name) VALUES (2, 'Bob')");

    assert_eq!(query_count(&db, "users"), 2);

    execute_sql(&db, "COMMIT");

    assert_eq!(query_count(&db, "users"), 2);
}

#[test]
fn test_rollback_restores_state() {
    let (_dir, db) = create_test_db();

    execute_sql(&db, "CREATE TABLE items (id INT PRIMARY KEY, qty INT)");
    execute_sql(&db, "INSERT INTO items (id, qty) VALUES (1, 100)");

    execute_sql(&db, "BEGIN");
    execute_sql(&db, "UPDATE items SET qty = 50 WHERE id = 1");

    let qty = query_value(&db, "SELECT qty FROM items WHERE id = 1");
    assert_eq!(qty, Some(turdb::OwnedValue::Int(50)));

    execute_sql(&db, "ROLLBACK");

    let qty = query_value(&db, "SELECT qty FROM items WHERE id = 1");
    assert_eq!(qty, Some(turdb::OwnedValue::Int(100)));
}

#[test]
fn test_savepoint_basic() {
    let (_dir, db) = create_test_db();

    execute_sql(&db, "CREATE TABLE test (id INT PRIMARY KEY, val INT)");

    execute_sql(&db, "BEGIN");
    execute_sql(&db, "INSERT INTO test (id, val) VALUES (1, 10)");
    execute_sql(&db, "SAVEPOINT sp1");
    execute_sql(&db, "INSERT INTO test (id, val) VALUES (2, 20)");

    assert_eq!(query_count(&db, "test"), 2);

    execute_sql(&db, "ROLLBACK TO sp1");

    assert_eq!(query_count(&db, "test"), 1);

    execute_sql(&db, "COMMIT");
    assert_eq!(query_count(&db, "test"), 1);
}

#[test]
fn test_multiple_updates_in_transaction() {
    let (_dir, db) = create_test_db();

    execute_sql(&db, "CREATE TABLE counter (id INT PRIMARY KEY, val INT)");
    execute_sql(&db, "INSERT INTO counter (id, val) VALUES (1, 0)");

    execute_sql(&db, "BEGIN");

    for i in 1..=10 {
        execute_sql(&db, &format!("UPDATE counter SET val = {} WHERE id = 1", i));
    }

    let val = query_value(&db, "SELECT val FROM counter WHERE id = 1");
    assert_eq!(val, Some(turdb::OwnedValue::Int(10)));

    execute_sql(&db, "COMMIT");

    let val = query_value(&db, "SELECT val FROM counter WHERE id = 1");
    assert_eq!(val, Some(turdb::OwnedValue::Int(10)));
}

#[test]
fn test_transaction_with_simple_updates() {
    let (_dir, db) = create_test_db();

    execute_sql(
        &db,
        "CREATE TABLE accounts (id INT PRIMARY KEY, balance INT)",
    );

    execute_sql(&db, "INSERT INTO accounts (id, balance) VALUES (1, 1000)");
    execute_sql(&db, "INSERT INTO accounts (id, balance) VALUES (2, 500)");

    execute_sql(&db, "BEGIN");
    execute_sql(&db, "UPDATE accounts SET balance = 900 WHERE id = 1");
    execute_sql(&db, "UPDATE accounts SET balance = 600 WHERE id = 2");

    let balance1 = query_value(&db, "SELECT balance FROM accounts WHERE id = 1");
    let balance2 = query_value(&db, "SELECT balance FROM accounts WHERE id = 2");
    assert_eq!(balance1, Some(turdb::OwnedValue::Int(900)));
    assert_eq!(balance2, Some(turdb::OwnedValue::Int(600)));

    execute_sql(&db, "COMMIT");

    let balance1 = query_value(&db, "SELECT balance FROM accounts WHERE id = 1");
    let balance2 = query_value(&db, "SELECT balance FROM accounts WHERE id = 2");
    assert_eq!(balance1, Some(turdb::OwnedValue::Int(900)));
    assert_eq!(balance2, Some(turdb::OwnedValue::Int(600)));
}

#[test]
fn test_autocommit_mode() {
    let (_dir, db) = create_test_db();

    execute_sql(&db, "CREATE TABLE autotest (id INT PRIMARY KEY, val INT)");

    execute_sql(&db, "INSERT INTO autotest (id, val) VALUES (1, 100)");

    assert_eq!(query_count(&db, "autotest"), 1);

    execute_sql(&db, "UPDATE autotest SET val = 200 WHERE id = 1");
    let val = query_value(&db, "SELECT val FROM autotest WHERE id = 1");
    assert_eq!(val, Some(turdb::OwnedValue::Int(200)));

    execute_sql(&db, "DELETE FROM autotest WHERE id = 1");
    assert_eq!(query_count(&db, "autotest"), 0);
}

#[test]
fn test_insert_delete_rollback() {
    let (_dir, db) = create_test_db();

    execute_sql(&db, "CREATE TABLE products (id INT PRIMARY KEY, name TEXT)");

    execute_sql(&db, "BEGIN");
    execute_sql(&db, "INSERT INTO products (id, name) VALUES (1, 'Widget')");
    execute_sql(&db, "INSERT INTO products (id, name) VALUES (2, 'Gadget')");
    assert_eq!(query_count(&db, "products"), 2);

    execute_sql(&db, "ROLLBACK");

    assert_eq!(query_count(&db, "products"), 0);
}

#[test]
fn test_nested_savepoints() {
    let (_dir, db) = create_test_db();

    execute_sql(&db, "CREATE TABLE nested (id INT PRIMARY KEY, level INT)");

    execute_sql(&db, "BEGIN");
    execute_sql(&db, "INSERT INTO nested (id, level) VALUES (1, 0)");

    execute_sql(&db, "SAVEPOINT s1");
    execute_sql(&db, "INSERT INTO nested (id, level) VALUES (2, 1)");

    execute_sql(&db, "SAVEPOINT s2");
    execute_sql(&db, "INSERT INTO nested (id, level) VALUES (3, 2)");

    assert_eq!(query_count(&db, "nested"), 3);

    execute_sql(&db, "RELEASE s2");

    execute_sql(&db, "ROLLBACK TO s1");
    assert_eq!(query_count(&db, "nested"), 1);

    execute_sql(&db, "COMMIT");
    assert_eq!(query_count(&db, "nested"), 1);
}
