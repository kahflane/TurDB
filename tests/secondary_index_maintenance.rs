//! # Secondary Index Maintenance Tests
//!
//! Tests for verifying that secondary indexes are correctly maintained during
//! DELETE and UPDATE operations, and that transaction rollback properly restores
//! index entries.

use std::path::Path;
use turdb::Database;

fn create_test_db(name: &str) -> Database {
    let path = format!("/tmp/turdb_test_secondary_idx_{}", name);
    if Path::new(&path).exists() {
        std::fs::remove_dir_all(&path).unwrap();
    }
    Database::create(&path).unwrap()
}

/// Test that DELETE removes entries from secondary indexes
#[test]
fn delete_removes_from_secondary_index() {
    let db = create_test_db("delete_sec_idx");

    db.execute("CREATE TABLE users (id INT PRIMARY KEY, email TEXT UNIQUE, name TEXT)")
        .unwrap();
    db.execute("INSERT INTO users VALUES (1, 'alice@test.com', 'Alice')")
        .unwrap();
    db.execute("INSERT INTO users VALUES (2, 'bob@test.com', 'Bob')")
        .unwrap();

    let rows = db
        .query("SELECT * FROM users WHERE email = 'alice@test.com'")
        .unwrap();
    assert_eq!(rows.len(), 1, "Should find Alice by email before delete");

    db.execute("DELETE FROM users WHERE id = 1").unwrap();

    let rows_after = db
        .query("SELECT * FROM users WHERE email = 'alice@test.com'")
        .unwrap();
    assert_eq!(
        rows_after.len(),
        0,
        "Should NOT find Alice by email after delete"
    );

    let bob_rows = db
        .query("SELECT * FROM users WHERE email = 'bob@test.com'")
        .unwrap();
    assert_eq!(bob_rows.len(), 1, "Bob should still be findable");

    db.execute("INSERT INTO users VALUES (3, 'alice@test.com', 'Alice2')")
        .unwrap();

    let alice2_rows = db
        .query("SELECT * FROM users WHERE email = 'alice@test.com'")
        .unwrap();
    assert_eq!(alice2_rows.len(), 1, "Should find new Alice by email");
    assert_eq!(alice2_rows[0].get_int(0).unwrap(), 3);
}

/// Test that UPDATE removes old index entry and inserts new one
#[test]
fn update_maintains_secondary_index() {
    let db = create_test_db("update_sec_idx");

    db.execute("CREATE TABLE users (id INT PRIMARY KEY, email TEXT UNIQUE, name TEXT)")
        .unwrap();
    db.execute("INSERT INTO users VALUES (1, 'old@test.com', 'User')")
        .unwrap();

    let old_rows = db
        .query("SELECT * FROM users WHERE email = 'old@test.com'")
        .unwrap();
    assert_eq!(old_rows.len(), 1, "Should find by old email");

    db.execute("UPDATE users SET email = 'new@test.com' WHERE id = 1")
        .unwrap();

    let old_after = db
        .query("SELECT * FROM users WHERE email = 'old@test.com'")
        .unwrap();
    assert_eq!(
        old_after.len(),
        0,
        "Should NOT find by old email after update"
    );

    let new_rows = db
        .query("SELECT * FROM users WHERE email = 'new@test.com'")
        .unwrap();
    assert_eq!(new_rows.len(), 1, "Should find by new email");

    db.execute("INSERT INTO users VALUES (2, 'old@test.com', 'User2')")
        .unwrap();

    let reused_rows = db
        .query("SELECT * FROM users WHERE email = 'old@test.com'")
        .unwrap();
    assert_eq!(
        reused_rows.len(),
        1,
        "Should find new user with reused email"
    );
    assert_eq!(reused_rows[0].get_int(0).unwrap(), 2);
}

/// Test transaction rollback restores index entries for INSERT
#[test]
fn rollback_insert_removes_index_entry() {
    let db = create_test_db("rollback_insert_idx");

    db.execute("CREATE TABLE users (id INT PRIMARY KEY, email TEXT UNIQUE, name TEXT)")
        .unwrap();

    db.execute("BEGIN").unwrap();
    db.execute("INSERT INTO users VALUES (1, 'test@test.com', 'Test')")
        .unwrap();

    let during_txn = db
        .query("SELECT * FROM users WHERE email = 'test@test.com'")
        .unwrap();
    assert_eq!(during_txn.len(), 1, "Should find during transaction");

    db.execute("ROLLBACK").unwrap();

    let after_rollback = db
        .query("SELECT * FROM users WHERE email = 'test@test.com'")
        .unwrap();
    assert_eq!(after_rollback.len(), 0, "Should NOT find after rollback");

    db.execute("INSERT INTO users VALUES (2, 'test@test.com', 'Test2')")
        .unwrap();
    let reinserted = db
        .query("SELECT * FROM users WHERE email = 'test@test.com'")
        .unwrap();
    assert_eq!(
        reinserted.len(),
        1,
        "Should be able to insert same email after rollback"
    );
}

/// Test transaction rollback restores index entries for DELETE
#[test]
fn rollback_delete_restores_index_entry() {
    let db = create_test_db("rollback_delete_idx");

    db.execute("CREATE TABLE users (id INT PRIMARY KEY, email TEXT UNIQUE, name TEXT)")
        .unwrap();
    db.execute("INSERT INTO users VALUES (1, 'alice@test.com', 'Alice')")
        .unwrap();

    db.execute("BEGIN").unwrap();
    db.execute("DELETE FROM users WHERE id = 1").unwrap();

    let during_txn = db
        .query("SELECT * FROM users WHERE email = 'alice@test.com'")
        .unwrap();
    assert_eq!(
        during_txn.len(),
        0,
        "Should NOT find during transaction after delete"
    );

    db.execute("ROLLBACK").unwrap();

    let after_rollback = db
        .query("SELECT * FROM users WHERE email = 'alice@test.com'")
        .unwrap();
    assert_eq!(
        after_rollback.len(),
        1,
        "Should find after rollback - index entry restored"
    );
}

/// Test transaction rollback restores index entries for UPDATE
#[test]
fn rollback_update_restores_index_entry() {
    let db = create_test_db("rollback_update_idx");

    db.execute("CREATE TABLE users (id INT PRIMARY KEY, email TEXT UNIQUE, name TEXT)")
        .unwrap();
    db.execute("INSERT INTO users VALUES (1, 'original@test.com', 'User')")
        .unwrap();

    db.execute("BEGIN").unwrap();
    db.execute("UPDATE users SET email = 'changed@test.com' WHERE id = 1")
        .unwrap();

    let old_during = db
        .query("SELECT * FROM users WHERE email = 'original@test.com'")
        .unwrap();
    let new_during = db
        .query("SELECT * FROM users WHERE email = 'changed@test.com'")
        .unwrap();
    assert_eq!(old_during.len(), 0, "Old email not findable during txn");
    assert_eq!(new_during.len(), 1, "New email findable during txn");

    db.execute("ROLLBACK").unwrap();

    let old_after = db
        .query("SELECT * FROM users WHERE email = 'original@test.com'")
        .unwrap();
    let new_after = db
        .query("SELECT * FROM users WHERE email = 'changed@test.com'")
        .unwrap();
    assert_eq!(old_after.len(), 1, "Original email findable after rollback");
    assert_eq!(
        new_after.len(),
        0,
        "Changed email NOT findable after rollback"
    );
}

/// Test that multi-column indexes are properly maintained
#[test]
fn delete_removes_from_multi_column_index() {
    let db = create_test_db("multi_col_idx");

    db.execute(
        "CREATE TABLE orders (id INT PRIMARY KEY, customer_id INT, product_id INT, qty INT)",
    )
    .unwrap();
    db.execute("CREATE UNIQUE INDEX idx_cust_prod ON orders (customer_id, product_id)")
        .unwrap();

    db.execute("INSERT INTO orders VALUES (1, 100, 200, 5)")
        .unwrap();
    db.execute("INSERT INTO orders VALUES (2, 100, 201, 3)")
        .unwrap();

    db.execute("DELETE FROM orders WHERE id = 1").unwrap();

    db.execute("INSERT INTO orders VALUES (3, 100, 200, 10)")
        .unwrap();

    let rows = db
        .query("SELECT * FROM orders WHERE customer_id = 100 AND product_id = 200")
        .unwrap();
    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0].get_int(0).unwrap(), 3);
}

/// Test that index bloat doesn't occur after multiple delete/insert cycles
#[test]
fn no_index_bloat_after_delete_insert_cycles() {
    let db = create_test_db("index_bloat");

    db.execute("CREATE TABLE items (id INT PRIMARY KEY, code TEXT UNIQUE)")
        .unwrap();

    for i in 0..100 {
        db.execute(&format!("INSERT INTO items VALUES ({}, 'CODE{}')", i, i))
            .unwrap();
    }

    for i in 0..100 {
        db.execute(&format!("DELETE FROM items WHERE id = {}", i))
            .unwrap();
    }

    for i in 0..100 {
        db.execute(&format!(
            "INSERT INTO items VALUES ({}, 'CODE{}')",
            i + 100,
            i
        ))
        .unwrap();
    }

    for i in 0..100 {
        let rows = db
            .query(&format!("SELECT * FROM items WHERE code = 'CODE{}'", i))
            .unwrap();
        assert_eq!(rows.len(), 1, "Should find CODE{} once", i);
        assert_eq!(rows[0].get_int(0).unwrap(), (i + 100) as i64);
    }
}

/// Test savepoint rollback correctly handles index entries
#[test]
fn savepoint_rollback_restores_index() {
    let db = create_test_db("savepoint_idx");

    db.execute("CREATE TABLE users (id INT PRIMARY KEY, email TEXT UNIQUE)")
        .unwrap();
    db.execute("INSERT INTO users VALUES (1, 'alice@test.com')")
        .unwrap();

    db.execute("BEGIN").unwrap();
    db.execute("INSERT INTO users VALUES (2, 'bob@test.com')")
        .unwrap();

    db.execute("SAVEPOINT sp1").unwrap();
    db.execute("DELETE FROM users WHERE id = 1").unwrap();

    let alice_during = db
        .query("SELECT * FROM users WHERE email = 'alice@test.com'")
        .unwrap();
    assert_eq!(alice_during.len(), 0, "Alice deleted within savepoint");

    db.execute("ROLLBACK TO sp1").unwrap();

    let alice_after = db
        .query("SELECT * FROM users WHERE email = 'alice@test.com'")
        .unwrap();
    assert_eq!(
        alice_after.len(),
        1,
        "Alice restored after savepoint rollback"
    );

    let bob_still = db
        .query("SELECT * FROM users WHERE email = 'bob@test.com'")
        .unwrap();
    assert_eq!(
        bob_still.len(),
        1,
        "Bob still exists (inserted before savepoint)"
    );

    db.execute("COMMIT").unwrap();

    let alice_final = db
        .query("SELECT * FROM users WHERE email = 'alice@test.com'")
        .unwrap();
    let bob_final = db
        .query("SELECT * FROM users WHERE email = 'bob@test.com'")
        .unwrap();
    assert_eq!(alice_final.len(), 1);
    assert_eq!(bob_final.len(), 1);
}
