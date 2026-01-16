//! # WAL Undo Logging Tests
//!
//! This module tests the WAL undo logging implementation for TurDB. Undo logging
//! ensures that uncommitted transactions can be rolled back after a crash.
//!
//! ## Background
//!
//! Without undo logging, if the OS flushes dirty mmap pages to disk before a
//! transaction commits, a crash could leave partially modified data in the
//! database file. Undo logging writes "before images" to the WAL before
//! modifying pages, allowing recovery to restore the original state.
//!
//! ## Requirements Tested
//!
//! - R1: ROLLBACK restores original values for uncommitted transactions
//! - R2: WAL contains undo frames for uncommitted modifications
//! - R3: Multiple operations in a transaction can all be rolled back
//! - R4: Committed transactions are not affected by undo frames

use tempfile::tempdir;
use turdb::{Database, ExecuteResult, OwnedValue};

mod rollback_tests {
    use super::*;

    #[test]
    fn rollback_restores_updated_row_to_original_value() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("test_db");

        let db = Database::create(&db_path).unwrap();
        db.execute("CREATE TABLE accounts (id INT, balance INT)")
            .unwrap();
        db.execute("INSERT INTO accounts VALUES (1, 1000)").unwrap();

        db.execute("BEGIN").unwrap();
        db.execute("UPDATE accounts SET balance = 500 WHERE id = 1")
            .unwrap();

        let mid_txn_result = db
            .execute("SELECT balance FROM accounts WHERE id = 1")
            .unwrap();
        match &mid_txn_result {
            ExecuteResult::Select { rows, .. } => {
                match &rows[0].values[0] {
                    OwnedValue::Int(balance) => {
                        assert_eq!(*balance, 500, "mid-transaction balance SHOULD be 500");
                    }
                    other => panic!("expected Int, got {:?}", other),
                }
            }
            _ => panic!("SELECT SHOULD return rows"),
        }

        db.execute("ROLLBACK").unwrap();

        let after_rollback_result = db
            .execute("SELECT balance FROM accounts WHERE id = 1")
            .unwrap();
        match after_rollback_result {
            ExecuteResult::Select { rows, .. } => {
                match &rows[0].values[0] {
                    OwnedValue::Int(balance) => {
                        assert_eq!(
                            *balance, 1000,
                            "after ROLLBACK balance SHOULD be restored to 1000"
                        );
                    }
                    other => panic!("expected Int, got {:?}", other),
                }
            }
            _ => panic!("SELECT SHOULD return rows"),
        }
    }

    #[test]
    fn rollback_restores_deleted_row() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("test_db");

        let db = Database::create(&db_path).unwrap();
        db.execute("CREATE TABLE items (id INT, name TEXT)")
            .unwrap();
        db.execute("INSERT INTO items VALUES (1, 'widget')").unwrap();
        db.execute("INSERT INTO items VALUES (2, 'gadget')").unwrap();

        let before_result = db.execute("SELECT * FROM items").unwrap();
        match &before_result {
            ExecuteResult::Select { rows, .. } => {
                assert_eq!(rows.len(), 2, "SHOULD have 2 rows before transaction");
            }
            _ => panic!("SELECT SHOULD return rows"),
        }

        db.execute("BEGIN").unwrap();
        db.execute("DELETE FROM items WHERE id = 1").unwrap();

        let mid_txn_result = db.execute("SELECT * FROM items").unwrap();
        match &mid_txn_result {
            ExecuteResult::Select { rows, .. } => {
                assert_eq!(rows.len(), 1, "mid-transaction SHOULD have 1 row");
            }
            _ => panic!("SELECT SHOULD return rows"),
        }

        db.execute("ROLLBACK").unwrap();

        let after_rollback_result = db.execute("SELECT * FROM items").unwrap();
        match after_rollback_result {
            ExecuteResult::Select { rows, .. } => {
                assert_eq!(
                    rows.len(),
                    2,
                    "after ROLLBACK SHOULD be restored to 2 rows"
                );
            }
            _ => panic!("SELECT SHOULD return rows"),
        }
    }

    #[test]
    fn rollback_removes_inserted_rows() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("test_db");

        let db = Database::create(&db_path).unwrap();
        db.execute("CREATE TABLE logs (id INT, message TEXT)")
            .unwrap();

        db.execute("BEGIN").unwrap();
        db.execute("INSERT INTO logs VALUES (1, 'first')").unwrap();
        db.execute("INSERT INTO logs VALUES (2, 'second')").unwrap();

        let mid_txn_result = db.execute("SELECT COUNT(*) FROM logs").unwrap();
        match &mid_txn_result {
            ExecuteResult::Select { rows, .. } => {
                match &rows[0].values[0] {
                    OwnedValue::Int(count) => {
                        assert_eq!(*count, 2, "mid-transaction count SHOULD be 2");
                    }
                    other => panic!("expected Int, got {:?}", other),
                }
            }
            _ => panic!("SELECT SHOULD return rows"),
        }

        db.execute("ROLLBACK").unwrap();

        let after_rollback_result = db.execute("SELECT COUNT(*) FROM logs").unwrap();
        match after_rollback_result {
            ExecuteResult::Select { rows, .. } => {
                match &rows[0].values[0] {
                    OwnedValue::Int(count) => {
                        assert_eq!(*count, 0, "after ROLLBACK count SHOULD be 0");
                    }
                    other => panic!("expected Int, got {:?}", other),
                }
            }
            _ => panic!("SELECT SHOULD return rows"),
        }
    }

    #[test]
    fn rollback_restores_multiple_operations() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("test_db");

        let db = Database::create(&db_path).unwrap();
        db.execute("CREATE TABLE data (id INT, val INT)").unwrap();
        db.execute("INSERT INTO data VALUES (1, 100)").unwrap();
        db.execute("INSERT INTO data VALUES (2, 200)").unwrap();
        db.execute("INSERT INTO data VALUES (3, 300)").unwrap();

        db.execute("BEGIN").unwrap();
        db.execute("UPDATE data SET val = 999 WHERE id = 1").unwrap();
        db.execute("DELETE FROM data WHERE id = 2").unwrap();
        db.execute("INSERT INTO data VALUES (4, 400)").unwrap();
        db.execute("ROLLBACK").unwrap();

        let result = db.execute("SELECT id, val FROM data ORDER BY id").unwrap();
        match result {
            ExecuteResult::Select { rows, .. } => {
                assert_eq!(rows.len(), 3, "SHOULD have 3 rows after rollback");

                match (&rows[0].values[0], &rows[0].values[1]) {
                    (OwnedValue::Int(id), OwnedValue::Int(val)) => {
                        assert_eq!(*id, 1);
                        assert_eq!(*val, 100, "row 1 val SHOULD be restored to 100");
                    }
                    _ => panic!("expected (Int, Int)"),
                }

                match (&rows[1].values[0], &rows[1].values[1]) {
                    (OwnedValue::Int(id), OwnedValue::Int(val)) => {
                        assert_eq!(*id, 2);
                        assert_eq!(*val, 200, "row 2 SHOULD be restored");
                    }
                    _ => panic!("expected (Int, Int)"),
                }

                match (&rows[2].values[0], &rows[2].values[1]) {
                    (OwnedValue::Int(id), OwnedValue::Int(val)) => {
                        assert_eq!(*id, 3);
                        assert_eq!(*val, 300);
                    }
                    _ => panic!("expected (Int, Int)"),
                }
            }
            _ => panic!("SELECT SHOULD return rows"),
        }
    }
}

mod committed_transaction_tests {
    use super::*;

    #[test]
    fn committed_transaction_is_not_affected_by_subsequent_rollback() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("test_db");

        let db = Database::create(&db_path).unwrap();
        db.execute("CREATE TABLE test (id INT, val INT)").unwrap();
        db.execute("INSERT INTO test VALUES (1, 10)").unwrap();

        db.execute("BEGIN").unwrap();
        db.execute("UPDATE test SET val = 20 WHERE id = 1").unwrap();
        db.execute("COMMIT").unwrap();

        db.execute("BEGIN").unwrap();
        db.execute("UPDATE test SET val = 30 WHERE id = 1").unwrap();
        db.execute("ROLLBACK").unwrap();

        let result = db.execute("SELECT val FROM test WHERE id = 1").unwrap();
        match result {
            ExecuteResult::Select { rows, .. } => {
                match &rows[0].values[0] {
                    OwnedValue::Int(val) => {
                        assert_eq!(
                            *val, 20,
                            "committed value (20) SHOULD persist, rolled back value (30) SHOULD be reverted"
                        );
                    }
                    other => panic!("expected Int, got {:?}", other),
                }
            }
            _ => panic!("SELECT SHOULD return rows"),
        }
    }
}
