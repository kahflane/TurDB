//! # WAL Durability Tests
//!
//! This module tests the WAL durability guarantees, specifically:
//! 1. After commit with synchronous=FULL, both WAL and storage are synced
//! 2. Data persists correctly across database close/reopen cycles
//! 3. Uncommitted transactions don't persist after crash simulation
//!
//! ## Background
//!
//! With mmap, the OS can flush dirty pages to disk at any time. Without
//! explicit storage sync after WAL sync, a crash could leave uncommitted
//! data in the main database file while WAL is empty (violating atomicity).
//!
//! The fix ensures that when `PRAGMA synchronous = FULL`:
//! - WAL is synced after commit (existing behavior)
//! - Storage files are also synced after WAL sync (new behavior)
//!
//! ## Requirements Tested
//!
//! - R1: PRAGMA synchronous defaults to FULL
//! - R2: Data persists after commit with synchronous=FULL
//! - R3: Uncommitted transactions don't persist after crash simulation

use tempfile::tempdir;
use turdb::{Database, ExecuteResult, OwnedValue};

mod synchronous_pragma_tests {
    use super::*;

    #[test]
    fn pragma_synchronous_defaults_to_full() {
        let dir = tempdir().unwrap();
        let db = Database::create(dir.path().join("test_db")).unwrap();

        let result = db.execute("PRAGMA synchronous").unwrap();

        match result {
            ExecuteResult::Pragma { name, value } => {
                assert_eq!(name, "SYNCHRONOUS");
                assert_eq!(value, Some("FULL".to_string()), "default synchronous SHOULD be FULL");
            }
            _ => panic!("PRAGMA synchronous SHOULD return a Pragma result"),
        }
    }

    #[test]
    fn pragma_synchronous_can_be_set_to_normal() {
        let dir = tempdir().unwrap();
        let db = Database::create(dir.path().join("test_db")).unwrap();

        db.execute("PRAGMA synchronous = NORMAL").unwrap();
        let result = db.execute("PRAGMA synchronous").unwrap();

        match result {
            ExecuteResult::Pragma { name, value } => {
                assert_eq!(name, "SYNCHRONOUS");
                assert_eq!(value, Some("NORMAL".to_string()));
            }
            _ => panic!("PRAGMA synchronous SHOULD return a Pragma result"),
        }
    }

    #[test]
    fn pragma_synchronous_can_be_set_to_off() {
        let dir = tempdir().unwrap();
        let db = Database::create(dir.path().join("test_db")).unwrap();

        db.execute("PRAGMA synchronous = OFF").unwrap();
        let result = db.execute("PRAGMA synchronous").unwrap();

        match result {
            ExecuteResult::Pragma { name, value } => {
                assert_eq!(name, "SYNCHRONOUS");
                assert_eq!(value, Some("OFF".to_string()));
            }
            _ => panic!("PRAGMA synchronous SHOULD return a Pragma result"),
        }
    }
}

mod commit_durability_tests {
    use super::*;

    #[test]
    fn data_persists_after_commit_and_reopen_with_full_sync() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("test_db");

        {
            let db = Database::create(&db_path).unwrap();
            db.execute("CREATE TABLE test (id INT, name TEXT)")
                .unwrap();
            db.execute("INSERT INTO test VALUES (1, 'alice')").unwrap();
            db.execute("INSERT INTO test VALUES (2, 'bob')").unwrap();
        }

        {
            let db = Database::open(&db_path).unwrap();
            let result = db
                .execute("SELECT id, name FROM test ORDER BY id")
                .unwrap();

            match result {
                ExecuteResult::Select { rows, .. } => {
                    assert_eq!(rows.len(), 2, "both rows SHOULD persist after reopen");
                    match &rows[0].values[0] {
                        OwnedValue::Int(id) => assert_eq!(*id, 1),
                        other => panic!("expected Int, got {:?}", other),
                    }
                    match &rows[0].values[1] {
                        OwnedValue::Text(name) => assert_eq!(name, "alice"),
                        other => panic!("expected Text, got {:?}", other),
                    }
                    match &rows[1].values[0] {
                        OwnedValue::Int(id) => assert_eq!(*id, 2),
                        other => panic!("expected Int, got {:?}", other),
                    }
                    match &rows[1].values[1] {
                        OwnedValue::Text(name) => assert_eq!(name, "bob"),
                        other => panic!("expected Text, got {:?}", other),
                    }
                }
                _ => panic!("SELECT SHOULD return rows"),
            }
        }
    }

    #[test]
    fn explicit_transaction_commit_persists_data() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("test_db");

        {
            let db = Database::create(&db_path).unwrap();
            db.execute("CREATE TABLE accounts (id INT, balance INT)")
                .unwrap();

            db.execute("BEGIN").unwrap();
            db.execute("INSERT INTO accounts VALUES (1, 100)").unwrap();
            db.execute("INSERT INTO accounts VALUES (2, 200)").unwrap();
            db.execute("COMMIT").unwrap();
        }

        {
            let db = Database::open(&db_path).unwrap();
            let result = db
                .execute("SELECT SUM(balance) FROM accounts")
                .unwrap();

            match result {
                ExecuteResult::Select { rows, .. } => {
                    assert_eq!(rows.len(), 1);
                    match &rows[0].values[0] {
                        OwnedValue::Int(sum) => {
                            assert_eq!(*sum, 300, "committed transaction SHOULD persist with sum=300");
                        }
                        other => panic!("expected Int, got {:?}", other),
                    }
                }
                _ => panic!("SELECT SHOULD return aggregate result"),
            }
        }
    }

    #[test]
    fn uncommitted_transaction_does_not_persist_after_crash_simulation() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("test_db");

        {
            let db = Database::create(&db_path).unwrap();
            db.execute("CREATE TABLE data (id INT, val INT)").unwrap();
            db.execute("INSERT INTO data VALUES (1, 10)").unwrap();
        }

        {
            let db = Database::open(&db_path).unwrap();
            db.execute("BEGIN").unwrap();
            db.execute("UPDATE data SET val = 99 WHERE id = 1").unwrap();
        }

        {
            let db = Database::open(&db_path).unwrap();
            let result = db.execute("SELECT val FROM data WHERE id = 1").unwrap();

            match result {
                ExecuteResult::Select { rows, .. } => {
                    assert_eq!(rows.len(), 1);
                    match &rows[0].values[0] {
                        OwnedValue::Int(val) => {
                            assert_eq!(
                                *val, 10,
                                "uncommitted update SHOULD NOT persist after crash (val should be 10, not 99)"
                            );
                        }
                        other => panic!("expected Int, got {:?}", other),
                    }
                }
                _ => panic!("SELECT SHOULD return row"),
            }
        }
    }
}
