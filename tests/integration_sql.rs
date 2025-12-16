//! # Integration Tests for TurDB SQL Operations
//!
//! This module provides end-to-end integration tests for TurDB's SQL functionality.
//! Tests are organized by feature area and verify behavior from a user's perspective
//! through the public Database API.
//!
//! ## Test Philosophy
//!
//! All tests follow specification-first design:
//! - Tests derived from SQL standard requirements, not implementation details
//! - Expected values independently computed (not derived from running the code)
//! - Each test verifies observable behavior through the public API
//! - Edge cases and error conditions are explicitly tested
//!
//! ## Test Categories
//!
//! 1. **DDL Tests**: CREATE/DROP for tables, schemas, indexes
//! 2. **DML Tests**: INSERT, SELECT, UPDATE, DELETE operations
//! 3. **Transaction Tests**: BEGIN, COMMIT, ROLLBACK, SAVEPOINT
//! 4. **Persistence Tests**: Data survives database close/reopen
//! 5. **Constraint Tests**: UNIQUE, CHECK, FOREIGN KEY enforcement
//! 6. **Data Type Tests**: Various SQL types work correctly
//!
//! ## Requirements Tested
//!
//! These tests verify the following requirements:
//! - R1: Database can create and open databases at filesystem paths
//! - R2: SQL DDL statements create persistent schema objects
//! - R3: SQL DML statements modify data correctly
//! - R4: Transactions provide atomicity (all-or-nothing)
//! - R5: Constraints enforce data integrity
//! - R6: Data persists across database close/reopen cycles
//!
//! ## Running Tests
//!
//! ```sh
//! cargo test --test integration_sql
//! ```

use tempfile::tempdir;
use turdb::{Database, ExecuteResult, OwnedValue};

mod ddl_tests {
    use super::*;

    #[test]
    fn create_table_returns_created_true_for_new_table() {
        let dir = tempdir().unwrap();
        let db = Database::create(dir.path().join("test_db")).unwrap();

        let result = db
            .execute("CREATE TABLE users (id INT, name TEXT)")
            .unwrap();

        assert!(
            matches!(result, ExecuteResult::CreateTable { created: true }),
            "CREATE TABLE for new table SHOULD return created: true"
        );
    }

    #[test]
    fn create_table_if_not_exists_returns_created_false_for_existing() {
        let dir = tempdir().unwrap();
        let db = Database::create(dir.path().join("test_db")).unwrap();
        db.execute("CREATE TABLE users (id INT)").unwrap();

        let result = db
            .execute("CREATE TABLE IF NOT EXISTS users (id INT)")
            .unwrap();

        assert!(
            matches!(result, ExecuteResult::CreateTable { created: false }),
            "CREATE TABLE IF NOT EXISTS for existing table SHOULD return created: false"
        );
    }

    #[test]
    fn create_table_without_if_not_exists_fails_for_existing() {
        let dir = tempdir().unwrap();
        let db = Database::create(dir.path().join("test_db")).unwrap();
        db.execute("CREATE TABLE users (id INT)").unwrap();

        let result = db.execute("CREATE TABLE users (id INT)");

        assert!(
            result.is_err(),
            "CREATE TABLE without IF NOT EXISTS SHOULD fail for existing table"
        );
    }

    #[test]
    fn drop_table_removes_table_and_returns_dropped_true() {
        let dir = tempdir().unwrap();
        let db = Database::create(dir.path().join("test_db")).unwrap();
        db.execute("CREATE TABLE users (id INT)").unwrap();

        let result = db.execute("DROP TABLE users").unwrap();

        assert!(
            matches!(result, ExecuteResult::DropTable { dropped: true }),
            "DROP TABLE SHOULD return dropped: true"
        );

        let query_result = db.execute("CREATE TABLE users (id INT)");
        assert!(
            query_result.is_ok(),
            "SHOULD be able to recreate dropped table"
        );
    }

    #[test]
    fn drop_table_if_exists_returns_dropped_false_for_nonexistent() {
        let dir = tempdir().unwrap();
        let db = Database::create(dir.path().join("test_db")).unwrap();

        let result = db.execute("DROP TABLE IF EXISTS nonexistent").unwrap();

        assert!(
            matches!(result, ExecuteResult::DropTable { dropped: false }),
            "DROP TABLE IF EXISTS for nonexistent SHOULD return dropped: false"
        );
    }

    #[test]
    fn create_schema_creates_namespace_for_tables() {
        let dir = tempdir().unwrap();
        let db = Database::create(dir.path().join("test_db")).unwrap();

        let result = db.execute("CREATE SCHEMA sales").unwrap();

        assert!(
            matches!(result, ExecuteResult::CreateSchema { created: true }),
            "CREATE SCHEMA SHOULD return created: true"
        );
    }

    #[test]
    fn create_index_creates_btree_index_on_column() {
        let dir = tempdir().unwrap();
        let db = Database::create(dir.path().join("test_db")).unwrap();
        db.execute("CREATE TABLE users (id INT, email TEXT)")
            .unwrap();

        let result = db
            .execute("CREATE INDEX idx_email ON users (email)")
            .unwrap();

        assert!(
            matches!(result, ExecuteResult::CreateIndex { created: true }),
            "CREATE INDEX SHOULD return created: true"
        );
    }

    #[test]
    fn create_unique_index_enforces_uniqueness() {
        let dir = tempdir().unwrap();
        let db = Database::create(dir.path().join("test_db")).unwrap();
        db.execute("CREATE TABLE users (id INT, email TEXT)")
            .unwrap();
        db.execute("CREATE UNIQUE INDEX idx_email ON users (email)")
            .unwrap();
        db.execute("INSERT INTO users VALUES (1, 'alice@test.com')")
            .unwrap();

        let result = db.execute("INSERT INTO users VALUES (2, 'alice@test.com')");

        assert!(
            result.is_err(),
            "UNIQUE INDEX SHOULD prevent duplicate values"
        );
    }

    #[test]
    fn drop_index_removes_index() {
        let dir = tempdir().unwrap();
        let db = Database::create(dir.path().join("test_db")).unwrap();
        db.execute("CREATE TABLE users (id INT, name TEXT)")
            .unwrap();
        db.execute("CREATE INDEX idx_name ON users (name)").unwrap();

        let result = db.execute("DROP INDEX idx_name").unwrap();

        assert!(
            matches!(result, ExecuteResult::DropIndex { dropped: true }),
            "DROP INDEX SHOULD return dropped: true"
        );
    }
}

mod dml_tests {
    use super::*;

    #[test]
    fn insert_single_row_returns_rows_affected_one() {
        let dir = tempdir().unwrap();
        let db = Database::create(dir.path().join("test_db")).unwrap();
        db.execute("CREATE TABLE users (id INT, name TEXT)")
            .unwrap();

        let result = db.execute("INSERT INTO users VALUES (1, 'Alice')").unwrap();

        assert!(
            matches!(
                result,
                ExecuteResult::Insert {
                    rows_affected: 1,
                    ..
                }
            ),
            "INSERT single row SHOULD return rows_affected: 1"
        );
    }

    #[test]
    fn select_returns_inserted_data_with_correct_values() {
        let dir = tempdir().unwrap();
        let db = Database::create(dir.path().join("test_db")).unwrap();
        db.execute("CREATE TABLE users (id INT, name TEXT)")
            .unwrap();
        db.execute("INSERT INTO users VALUES (42, 'Bob')").unwrap();

        let rows = db.query("SELECT id, name FROM users").unwrap();

        assert_eq!(rows.len(), 1, "SHOULD return exactly 1 row");
        assert_eq!(rows[0].values.len(), 2, "Row SHOULD have exactly 2 columns");

        match &rows[0].values[0] {
            OwnedValue::Int(id) => assert_eq!(*id, 42, "id SHOULD be 42"),
            other => panic!("id SHOULD be Int, got {:?}", other),
        }
        match &rows[0].values[1] {
            OwnedValue::Text(name) => assert_eq!(name, "Bob", "name SHOULD be 'Bob'"),
            other => panic!("name SHOULD be Text, got {:?}", other),
        }
    }

    #[test]
    fn select_with_where_filters_rows_correctly() {
        let dir = tempdir().unwrap();
        let db = Database::create(dir.path().join("test_db")).unwrap();
        db.execute("CREATE TABLE users (id INT, active INT)")
            .unwrap();
        db.execute("INSERT INTO users VALUES (1, 1)").unwrap();
        db.execute("INSERT INTO users VALUES (2, 0)").unwrap();
        db.execute("INSERT INTO users VALUES (3, 1)").unwrap();

        let rows = db.query("SELECT id FROM users WHERE active = 1").unwrap();

        assert_eq!(rows.len(), 2, "WHERE active=1 SHOULD return 2 rows");
    }

    #[test]
    fn select_with_limit_restricts_row_count() {
        let dir = tempdir().unwrap();
        let db = Database::create(dir.path().join("test_db")).unwrap();
        db.execute("CREATE TABLE nums (n INT)").unwrap();
        for i in 1..=10 {
            db.execute(&format!("INSERT INTO nums VALUES ({})", i))
                .unwrap();
        }

        let rows = db.query("SELECT n FROM nums LIMIT 3").unwrap();

        assert_eq!(rows.len(), 3, "LIMIT 3 SHOULD return exactly 3 rows");
    }

    #[test]
    fn select_with_offset_skips_rows() {
        let dir = tempdir().unwrap();
        let db = Database::create(dir.path().join("test_db")).unwrap();
        db.execute("CREATE TABLE nums (n INT)").unwrap();
        for i in 1..=5 {
            db.execute(&format!("INSERT INTO nums VALUES ({})", i))
                .unwrap();
        }

        let rows = db.query("SELECT n FROM nums LIMIT 2 OFFSET 2").unwrap();

        assert_eq!(rows.len(), 2, "LIMIT 2 OFFSET 2 SHOULD return 2 rows");
    }

    #[test]
    fn select_distinct_removes_duplicates() {
        let dir = tempdir().unwrap();
        let db = Database::create(dir.path().join("test_db")).unwrap();
        db.execute("CREATE TABLE items (category TEXT)").unwrap();
        db.execute("INSERT INTO items VALUES ('A')").unwrap();
        db.execute("INSERT INTO items VALUES ('B')").unwrap();
        db.execute("INSERT INTO items VALUES ('A')").unwrap();
        db.execute("INSERT INTO items VALUES ('A')").unwrap();

        let rows = db.query("SELECT DISTINCT category FROM items").unwrap();

        assert_eq!(
            rows.len(),
            2,
            "DISTINCT SHOULD return 2 unique categories (A, B)"
        );
    }

    #[test]
    fn update_modifies_matching_rows_and_returns_count() {
        let dir = tempdir().unwrap();
        let db = Database::create(dir.path().join("test_db")).unwrap();
        db.execute("CREATE TABLE users (id INT, score INT)")
            .unwrap();
        db.execute("INSERT INTO users VALUES (1, 10)").unwrap();
        db.execute("INSERT INTO users VALUES (2, 20)").unwrap();
        db.execute("INSERT INTO users VALUES (3, 10)").unwrap();

        let result = db
            .execute("UPDATE users SET score = 99 WHERE score = 10")
            .unwrap();

        assert!(
            matches!(
                result,
                ExecuteResult::Update {
                    rows_affected: 2,
                    ..
                }
            ),
            "UPDATE SHOULD affect 2 rows where score=10"
        );

        let rows = db.query("SELECT id FROM users WHERE score = 99").unwrap();
        assert_eq!(rows.len(), 2, "2 rows SHOULD now have score=99");
    }

    #[test]
    fn update_with_no_match_returns_zero_affected() {
        let dir = tempdir().unwrap();
        let db = Database::create(dir.path().join("test_db")).unwrap();
        db.execute("CREATE TABLE users (id INT)").unwrap();
        db.execute("INSERT INTO users VALUES (1)").unwrap();

        let result = db
            .execute("UPDATE users SET id = 99 WHERE id = 999")
            .unwrap();

        assert!(
            matches!(
                result,
                ExecuteResult::Update {
                    rows_affected: 0,
                    ..
                }
            ),
            "UPDATE with no match SHOULD return rows_affected: 0"
        );
    }

    #[test]
    fn delete_removes_matching_rows_and_returns_count() {
        let dir = tempdir().unwrap();
        let db = Database::create(dir.path().join("test_db")).unwrap();
        db.execute("CREATE TABLE users (id INT, active INT)")
            .unwrap();
        db.execute("INSERT INTO users VALUES (1, 1)").unwrap();
        db.execute("INSERT INTO users VALUES (2, 0)").unwrap();
        db.execute("INSERT INTO users VALUES (3, 0)").unwrap();

        let result = db.execute("DELETE FROM users WHERE active = 0").unwrap();

        assert!(
            matches!(
                result,
                ExecuteResult::Delete {
                    rows_affected: 2,
                    ..
                }
            ),
            "DELETE SHOULD affect 2 rows where active=0"
        );

        let rows = db.query("SELECT * FROM users").unwrap();
        assert_eq!(rows.len(), 1, "Only 1 row SHOULD remain");
    }

    #[test]
    fn delete_all_removes_all_rows() {
        let dir = tempdir().unwrap();
        let db = Database::create(dir.path().join("test_db")).unwrap();
        db.execute("CREATE TABLE users (id INT)").unwrap();
        db.execute("INSERT INTO users VALUES (1)").unwrap();
        db.execute("INSERT INTO users VALUES (2)").unwrap();
        db.execute("INSERT INTO users VALUES (3)").unwrap();

        let result = db.execute("DELETE FROM users").unwrap();

        assert!(
            matches!(
                result,
                ExecuteResult::Delete {
                    rows_affected: 3,
                    ..
                }
            ),
            "DELETE without WHERE SHOULD affect all 3 rows"
        );

        let rows = db.query("SELECT * FROM users").unwrap();
        assert_eq!(rows.len(), 0, "Table SHOULD be empty");
    }

    #[test]
    fn select_from_empty_table_returns_empty_result() {
        let dir = tempdir().unwrap();
        let db = Database::create(dir.path().join("test_db")).unwrap();
        db.execute("CREATE TABLE empty (id INT)").unwrap();

        let rows = db.query("SELECT * FROM empty").unwrap();

        assert_eq!(rows.len(), 0, "Empty table SHOULD return 0 rows");
    }

    #[test]
    fn select_nonexistent_table_returns_error() {
        let dir = tempdir().unwrap();
        let db = Database::create(dir.path().join("test_db")).unwrap();

        let result = db.query("SELECT * FROM nonexistent");

        assert!(result.is_err(), "SELECT from nonexistent table SHOULD fail");
    }
}

mod transaction_tests {
    use super::*;

    #[test]
    fn begin_starts_transaction() {
        let dir = tempdir().unwrap();
        let db = Database::create(dir.path().join("test_db")).unwrap();

        let result = db.execute("BEGIN").unwrap();

        assert!(
            matches!(result, ExecuteResult::Begin),
            "BEGIN SHOULD return ExecuteResult::Begin"
        );
    }

    #[test]
    fn commit_finalizes_transaction() {
        let dir = tempdir().unwrap();
        let db = Database::create(dir.path().join("test_db")).unwrap();
        db.execute("BEGIN").unwrap();

        let result = db.execute("COMMIT").unwrap();

        assert!(
            matches!(result, ExecuteResult::Commit),
            "COMMIT SHOULD return ExecuteResult::Commit"
        );
    }

    #[test]
    fn commit_without_begin_fails() {
        let dir = tempdir().unwrap();
        let db = Database::create(dir.path().join("test_db")).unwrap();

        let result = db.execute("COMMIT");

        assert!(result.is_err(), "COMMIT without BEGIN SHOULD fail");
    }

    #[test]
    fn rollback_undoes_transaction_changes() {
        let dir = tempdir().unwrap();
        let db = Database::create(dir.path().join("test_db")).unwrap();
        db.execute("CREATE TABLE users (id INT)").unwrap();
        db.execute("INSERT INTO users VALUES (1)").unwrap();

        db.execute("BEGIN").unwrap();
        db.execute("INSERT INTO users VALUES (2)").unwrap();
        db.execute("ROLLBACK").unwrap();

        let rows = db.query("SELECT * FROM users").unwrap();
        assert_eq!(
            rows.len(),
            1,
            "ROLLBACK SHOULD undo INSERT, leaving only 1 row"
        );
    }

    #[test]
    fn rollback_without_begin_fails() {
        let dir = tempdir().unwrap();
        let db = Database::create(dir.path().join("test_db")).unwrap();

        let result = db.execute("ROLLBACK");

        assert!(result.is_err(), "ROLLBACK without BEGIN SHOULD fail");
    }

    #[test]
    fn commit_persists_transaction_changes() {
        let dir = tempdir().unwrap();
        let db = Database::create(dir.path().join("test_db")).unwrap();
        db.execute("CREATE TABLE users (id INT)").unwrap();

        db.execute("BEGIN").unwrap();
        db.execute("INSERT INTO users VALUES (1)").unwrap();
        db.execute("INSERT INTO users VALUES (2)").unwrap();
        db.execute("COMMIT").unwrap();

        let rows = db.query("SELECT * FROM users").unwrap();
        assert_eq!(rows.len(), 2, "COMMIT SHOULD persist both INSERTs");
    }

    #[test]
    fn savepoint_allows_partial_rollback() {
        let dir = tempdir().unwrap();
        let db = Database::create(dir.path().join("test_db")).unwrap();
        db.execute("CREATE TABLE users (id INT)").unwrap();

        db.execute("BEGIN").unwrap();
        db.execute("INSERT INTO users VALUES (1)").unwrap();
        db.execute("SAVEPOINT sp1").unwrap();
        db.execute("INSERT INTO users VALUES (2)").unwrap();
        db.execute("ROLLBACK TO SAVEPOINT sp1").unwrap();
        db.execute("COMMIT").unwrap();

        let rows = db.query("SELECT * FROM users").unwrap();
        assert_eq!(
            rows.len(),
            1,
            "ROLLBACK TO SAVEPOINT SHOULD undo INSERT after savepoint"
        );
    }

    #[test]
    fn nested_begin_fails() {
        let dir = tempdir().unwrap();
        let db = Database::create(dir.path().join("test_db")).unwrap();
        db.execute("BEGIN").unwrap();

        let result = db.execute("BEGIN");

        assert!(
            result.is_err(),
            "Nested BEGIN SHOULD fail (use SAVEPOINT instead)"
        );
    }

    #[test]
    fn rollback_undoes_delete() {
        let dir = tempdir().unwrap();
        let db = Database::create(dir.path().join("test_db")).unwrap();
        db.execute("CREATE TABLE users (id INT, name TEXT)")
            .unwrap();
        db.execute("INSERT INTO users VALUES (1, 'Alice')").unwrap();
        db.execute("INSERT INTO users VALUES (2, 'Bob')").unwrap();

        db.execute("BEGIN").unwrap();
        db.execute("DELETE FROM users WHERE id = 1").unwrap();
        let rows_during = db.query("SELECT * FROM users").unwrap();
        assert_eq!(rows_during.len(), 1, "DELETE SHOULD remove 1 row");

        db.execute("ROLLBACK").unwrap();

        let rows_after = db.query("SELECT * FROM users").unwrap();
        assert_eq!(rows_after.len(), 2, "ROLLBACK SHOULD restore deleted row");
    }

    #[test]
    fn rollback_undoes_update() {
        let dir = tempdir().unwrap();
        let db = Database::create(dir.path().join("test_db")).unwrap();
        db.execute("CREATE TABLE users (id INT, name TEXT)")
            .unwrap();
        db.execute("INSERT INTO users VALUES (1, 'Original')")
            .unwrap();

        db.execute("BEGIN").unwrap();
        db.execute("UPDATE users SET name = 'Modified' WHERE id = 1")
            .unwrap();
        db.execute("ROLLBACK").unwrap();

        let rows = db.query("SELECT name FROM users WHERE id = 1").unwrap();
        match &rows[0].values[0] {
            OwnedValue::Text(name) => {
                assert_eq!(name, "Original", "ROLLBACK SHOULD restore original value")
            }
            other => panic!("Expected Text, got {:?}", other),
        }
    }
}

mod persistence_tests {
    use super::*;

    #[test]
    fn data_persists_after_close_and_reopen() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("test_db");

        {
            let db = Database::create(&db_path).unwrap();
            db.execute("CREATE TABLE users (id INT, name TEXT)")
                .unwrap();
            db.execute("INSERT INTO users VALUES (1, 'Alice')").unwrap();
            db.execute("INSERT INTO users VALUES (2, 'Bob')").unwrap();
            db.close().unwrap();
        }

        let db = Database::open(&db_path).unwrap();
        let rows = db.query("SELECT * FROM users").unwrap();

        assert_eq!(rows.len(), 2, "Data SHOULD persist after close/reopen");
    }

    #[test]
    fn schema_persists_after_close_and_reopen() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("test_db");

        {
            let db = Database::create(&db_path).unwrap();
            db.execute("CREATE TABLE users (id INT, name TEXT, age INT)")
                .unwrap();
            db.close().unwrap();
        }

        let db = Database::open(&db_path).unwrap();
        let result = db.execute("INSERT INTO users VALUES (1, 'Test', 25)");

        assert!(
            result.is_ok(),
            "Schema SHOULD persist - INSERT SHOULD work after reopen"
        );
    }

    #[test]
    fn updates_persist_after_close_and_reopen() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("test_db");

        {
            let db = Database::create(&db_path).unwrap();
            db.execute("CREATE TABLE users (id INT, score INT)")
                .unwrap();
            db.execute("INSERT INTO users VALUES (1, 10)").unwrap();
            db.execute("UPDATE users SET score = 99 WHERE id = 1")
                .unwrap();
            db.close().unwrap();
        }

        let db = Database::open(&db_path).unwrap();
        let rows = db.query("SELECT score FROM users WHERE id = 1").unwrap();

        match &rows[0].values[0] {
            OwnedValue::Int(score) => {
                assert_eq!(*score, 99, "Updated score SHOULD persist as 99")
            }
            other => panic!("Expected Int, got {:?}", other),
        }
    }

    #[test]
    fn deletes_persist_after_close_and_reopen() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("test_db");

        {
            let db = Database::create(&db_path).unwrap();
            db.execute("CREATE TABLE users (id INT)").unwrap();
            db.execute("INSERT INTO users VALUES (1)").unwrap();
            db.execute("INSERT INTO users VALUES (2)").unwrap();
            db.execute("INSERT INTO users VALUES (3)").unwrap();
            db.execute("DELETE FROM users WHERE id = 2").unwrap();
            db.close().unwrap();
        }

        let db = Database::open(&db_path).unwrap();
        let rows = db.query("SELECT * FROM users").unwrap();

        assert_eq!(rows.len(), 2, "DELETE SHOULD persist - only 2 rows remain");
    }

    #[test]
    fn multiple_tables_persist_independently() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("test_db");

        {
            let db = Database::create(&db_path).unwrap();
            db.execute("CREATE TABLE users (id INT)").unwrap();
            db.execute("CREATE TABLE orders (id INT)").unwrap();
            db.execute("INSERT INTO users VALUES (1)").unwrap();
            db.execute("INSERT INTO users VALUES (2)").unwrap();
            db.execute("INSERT INTO orders VALUES (100)").unwrap();
            db.close().unwrap();
        }

        let db = Database::open(&db_path).unwrap();
        let users = db.query("SELECT * FROM users").unwrap();
        let orders = db.query("SELECT * FROM orders").unwrap();

        assert_eq!(users.len(), 2, "users table SHOULD have 2 rows");
        assert_eq!(orders.len(), 1, "orders table SHOULD have 1 row");
    }
}

mod constraint_tests {
    use super::*;

    #[test]
    fn primary_key_rejects_duplicate() {
        let dir = tempdir().unwrap();
        let db = Database::create(dir.path().join("test_db")).unwrap();
        db.execute("CREATE TABLE users (id INT PRIMARY KEY, name TEXT)")
            .unwrap();
        db.execute("INSERT INTO users VALUES (1, 'Alice')").unwrap();

        let result = db.execute("INSERT INTO users VALUES (1, 'Bob')");

        assert!(result.is_err(), "PRIMARY KEY SHOULD reject duplicate id=1");
    }

    #[test]
    fn unique_constraint_rejects_duplicate() {
        let dir = tempdir().unwrap();
        let db = Database::create(dir.path().join("test_db")).unwrap();
        db.execute("CREATE TABLE users (id INT, email TEXT UNIQUE)")
            .unwrap();
        db.execute("INSERT INTO users VALUES (1, 'alice@test.com')")
            .unwrap();

        let result = db.execute("INSERT INTO users VALUES (2, 'alice@test.com')");

        assert!(result.is_err(), "UNIQUE SHOULD reject duplicate email");
    }

    #[test]
    fn unique_constraint_allows_multiple_nulls() {
        let dir = tempdir().unwrap();
        let db = Database::create(dir.path().join("test_db")).unwrap();
        db.execute("CREATE TABLE users (id INT, email TEXT UNIQUE)")
            .unwrap();
        db.execute("INSERT INTO users VALUES (1, NULL)").unwrap();
        db.execute("INSERT INTO users VALUES (2, NULL)").unwrap();

        let rows = db.query("SELECT * FROM users").unwrap();

        assert_eq!(rows.len(), 2, "UNIQUE SHOULD allow multiple NULLs");
    }

    #[test]
    fn check_constraint_rejects_invalid_value() {
        let dir = tempdir().unwrap();
        let db = Database::create(dir.path().join("test_db")).unwrap();
        db.execute("CREATE TABLE users (id INT, age INT CHECK(age >= 0))")
            .unwrap();

        let result = db.execute("INSERT INTO users VALUES (1, -5)");

        assert!(result.is_err(), "CHECK(age >= 0) SHOULD reject age=-5");
    }

    #[test]
    fn check_constraint_accepts_valid_value() {
        let dir = tempdir().unwrap();
        let db = Database::create(dir.path().join("test_db")).unwrap();
        db.execute("CREATE TABLE users (id INT, age INT CHECK(age >= 0))")
            .unwrap();

        let result = db.execute("INSERT INTO users VALUES (1, 25)");

        assert!(result.is_ok(), "CHECK(age >= 0) SHOULD accept age=25");
    }

    #[test]
    fn foreign_key_rejects_missing_reference() {
        let dir = tempdir().unwrap();
        let db = Database::create(dir.path().join("test_db")).unwrap();
        db.execute("CREATE TABLE users (id INT PRIMARY KEY)")
            .unwrap();
        db.execute("CREATE TABLE orders (id INT, user_id INT REFERENCES users(id))")
            .unwrap();
        db.execute("INSERT INTO users VALUES (1)").unwrap();

        let result = db.execute("INSERT INTO orders VALUES (1, 999)");

        assert!(
            result.is_err(),
            "FOREIGN KEY SHOULD reject reference to nonexistent user_id=999"
        );
    }

    #[test]
    fn foreign_key_accepts_valid_reference() {
        let dir = tempdir().unwrap();
        let db = Database::create(dir.path().join("test_db")).unwrap();
        db.execute("CREATE TABLE users (id INT PRIMARY KEY)")
            .unwrap();
        db.execute("CREATE TABLE orders (id INT, user_id INT REFERENCES users(id))")
            .unwrap();
        db.execute("INSERT INTO users VALUES (1)").unwrap();

        let result = db.execute("INSERT INTO orders VALUES (1, 1)");

        assert!(
            result.is_ok(),
            "FOREIGN KEY SHOULD accept reference to existing user_id=1"
        );
    }

    #[test]
    fn foreign_key_blocks_delete_of_referenced_row() {
        let dir = tempdir().unwrap();
        let db = Database::create(dir.path().join("test_db")).unwrap();
        db.execute("CREATE TABLE users (id INT PRIMARY KEY)")
            .unwrap();
        db.execute("CREATE TABLE orders (id INT, user_id INT REFERENCES users(id))")
            .unwrap();
        db.execute("INSERT INTO users VALUES (1)").unwrap();
        db.execute("INSERT INTO orders VALUES (1, 1)").unwrap();

        let result = db.execute("DELETE FROM users WHERE id = 1");

        assert!(
            result.is_err(),
            "FOREIGN KEY SHOULD block DELETE of referenced row"
        );
    }

    #[test]
    fn unique_constraint_rejects_duplicate_on_update() {
        let dir = tempdir().unwrap();
        let db = Database::create(dir.path().join("test_db")).unwrap();
        db.execute("CREATE TABLE users (id INT, email TEXT UNIQUE)")
            .unwrap();
        db.execute("INSERT INTO users VALUES (1, 'alice@test.com')")
            .unwrap();
        db.execute("INSERT INTO users VALUES (2, 'bob@test.com')")
            .unwrap();

        let result = db.execute("UPDATE users SET email = 'alice@test.com' WHERE id = 2");

        assert!(
            result.is_err(),
            "UNIQUE SHOULD reject duplicate email on UPDATE"
        );
    }
}

mod data_type_tests {
    use super::*;

    #[test]
    fn integer_types_store_and_retrieve_correctly() {
        let dir = tempdir().unwrap();
        let db = Database::create(dir.path().join("test_db")).unwrap();
        db.execute("CREATE TABLE nums (tiny TINYINT, small SMALLINT, reg INT, big BIGINT)")
            .unwrap();
        db.execute("INSERT INTO nums VALUES (127, 32767, 2147483647, 9223372036854775807)")
            .unwrap();

        let rows = db.query("SELECT * FROM nums").unwrap();
        assert_eq!(rows.len(), 1);

        match &rows[0].values[0] {
            OwnedValue::Int(v) => assert_eq!(*v, 127, "TINYINT max SHOULD be 127"),
            other => panic!("Expected Int for TINYINT, got {:?}", other),
        }
        match &rows[0].values[3] {
            OwnedValue::Int(v) => {
                assert_eq!(*v, 9223372036854775807i64, "BIGINT max SHOULD be i64::MAX")
            }
            other => panic!("Expected Int for BIGINT, got {:?}", other),
        }
    }

    #[test]
    fn float_types_store_and_retrieve_correctly() {
        let dir = tempdir().unwrap();
        let db = Database::create(dir.path().join("test_db")).unwrap();
        db.execute("CREATE TABLE nums (r REAL, f FLOAT, d DOUBLE PRECISION)")
            .unwrap();
        db.execute("INSERT INTO nums VALUES (3.25, 2.5, 1.75)")
            .unwrap();

        let rows = db.query("SELECT * FROM nums").unwrap();
        assert_eq!(rows.len(), 1);

        match &rows[0].values[0] {
            OwnedValue::Float(v) => {
                assert!((*v - 3.25).abs() < 0.001, "REAL should store 3.25")
            }
            other => panic!("Expected Float for REAL, got {:?}", other),
        }
    }

    #[test]
    fn text_stores_and_retrieves_correctly() {
        let dir = tempdir().unwrap();
        let db = Database::create(dir.path().join("test_db")).unwrap();
        db.execute("CREATE TABLE docs (content TEXT)").unwrap();
        db.execute("INSERT INTO docs VALUES ('Hello, World!')")
            .unwrap();

        let rows = db.query("SELECT content FROM docs").unwrap();

        match &rows[0].values[0] {
            OwnedValue::Text(s) => assert_eq!(s, "Hello, World!"),
            other => panic!("Expected Text, got {:?}", other),
        }
    }

    #[test]
    fn boolean_stores_and_retrieves_correctly() {
        let dir = tempdir().unwrap();
        let db = Database::create(dir.path().join("test_db")).unwrap();
        db.execute("CREATE TABLE flags (active BOOLEAN)").unwrap();
        db.execute("INSERT INTO flags VALUES (TRUE)").unwrap();
        db.execute("INSERT INTO flags VALUES (FALSE)").unwrap();

        let rows = db.query("SELECT * FROM flags").unwrap();
        assert_eq!(rows.len(), 2);

        match &rows[0].values[0] {
            OwnedValue::Bool(b) => assert!(*b, "First row SHOULD be TRUE"),
            other => panic!("Expected Bool, got {:?}", other),
        }
        match &rows[1].values[0] {
            OwnedValue::Bool(b) => assert!(!*b, "Second row SHOULD be FALSE"),
            other => panic!("Expected Bool, got {:?}", other),
        }
    }

    #[test]
    fn null_stores_and_retrieves_correctly() {
        let dir = tempdir().unwrap();
        let db = Database::create(dir.path().join("test_db")).unwrap();
        db.execute("CREATE TABLE data (value INT)").unwrap();
        db.execute("INSERT INTO data VALUES (NULL)").unwrap();

        let rows = db.query("SELECT value FROM data").unwrap();

        assert!(
            matches!(&rows[0].values[0], OwnedValue::Null),
            "NULL SHOULD be stored and retrieved as Null"
        );
    }

    #[test]
    fn blob_stores_and_retrieves_correctly() {
        let dir = tempdir().unwrap();
        let db = Database::create(dir.path().join("test_db")).unwrap();
        db.execute("CREATE TABLE binaries (data BLOB)").unwrap();
        db.execute("INSERT INTO binaries VALUES (x'DEADBEEF')")
            .unwrap();

        let rows = db.query("SELECT data FROM binaries").unwrap();

        match &rows[0].values[0] {
            OwnedValue::Blob(b) => {
                assert_eq!(b.as_slice(), &[0xDE, 0xAD, 0xBE, 0xEF]);
            }
            other => panic!("Expected Blob, got {:?}", other),
        }
    }

    #[test]
    fn uuid_stores_and_retrieves_correctly() {
        let dir = tempdir().unwrap();
        let db = Database::create(dir.path().join("test_db")).unwrap();
        db.execute("CREATE TABLE items (id UUID)").unwrap();
        db.execute("INSERT INTO items VALUES ('550e8400-e29b-41d4-a716-446655440000')")
            .unwrap();

        let rows = db.query("SELECT id FROM items").unwrap();

        match &rows[0].values[0] {
            OwnedValue::Uuid(bytes) => {
                let expected: [u8; 16] = [
                    0x55, 0x0e, 0x84, 0x00, 0xe2, 0x9b, 0x41, 0xd4, 0xa7, 0x16, 0x44, 0x66, 0x55,
                    0x44, 0x00, 0x00,
                ];
                assert_eq!(bytes, &expected, "UUID bytes SHOULD match");
            }
            other => panic!("Expected Uuid, got {:?}", other),
        }
    }

    #[test]
    fn vector_stores_and_retrieves_correctly() {
        let dir = tempdir().unwrap();
        let db = Database::create(dir.path().join("test_db")).unwrap();
        db.execute("CREATE TABLE embeddings (vec VECTOR(3))")
            .unwrap();
        db.execute("INSERT INTO embeddings VALUES ('[1.0, 2.0, 3.0]')")
            .unwrap();

        let rows = db.query("SELECT vec FROM embeddings").unwrap();

        match &rows[0].values[0] {
            OwnedValue::Vector(v) => {
                assert_eq!(v.len(), 3, "VECTOR(3) SHOULD have 3 elements");
                assert!((v[0] - 1.0).abs() < 0.001);
                assert!((v[1] - 2.0).abs() < 0.001);
                assert!((v[2] - 3.0).abs() < 0.001);
            }
            other => panic!("Expected Vector, got {:?}", other),
        }
    }
}

mod aggregation_tests {
    use super::*;

    #[test]
    fn count_returns_row_count() {
        let dir = tempdir().unwrap();
        let db = Database::create(dir.path().join("test_db")).unwrap();
        db.execute("CREATE TABLE users (id INT)").unwrap();
        db.execute("INSERT INTO users VALUES (1)").unwrap();
        db.execute("INSERT INTO users VALUES (2)").unwrap();
        db.execute("INSERT INTO users VALUES (3)").unwrap();

        let rows = db.query("SELECT COUNT(*) FROM users").unwrap();

        match &rows[0].values[0] {
            OwnedValue::Int(count) => assert_eq!(*count, 3, "COUNT(*) SHOULD return 3"),
            other => panic!("Expected Int for COUNT, got {:?}", other),
        }
    }

    #[test]
    fn count_with_where_filters_before_counting() {
        let dir = tempdir().unwrap();
        let db = Database::create(dir.path().join("test_db")).unwrap();
        db.execute("CREATE TABLE users (id INT, active INT)")
            .unwrap();
        db.execute("INSERT INTO users VALUES (1, 1)").unwrap();
        db.execute("INSERT INTO users VALUES (2, 0)").unwrap();
        db.execute("INSERT INTO users VALUES (3, 1)").unwrap();

        let rows = db
            .query("SELECT COUNT(*) FROM users WHERE active = 1")
            .unwrap();

        match &rows[0].values[0] {
            OwnedValue::Int(count) => {
                assert_eq!(*count, 2, "COUNT(*) WHERE active=1 SHOULD return 2")
            }
            other => panic!("Expected Int for COUNT, got {:?}", other),
        }
    }

    #[test]
    fn sum_returns_total() {
        let dir = tempdir().unwrap();
        let db = Database::create(dir.path().join("test_db")).unwrap();
        db.execute("CREATE TABLE sales (amount INT)").unwrap();
        db.execute("INSERT INTO sales VALUES (10)").unwrap();
        db.execute("INSERT INTO sales VALUES (20)").unwrap();
        db.execute("INSERT INTO sales VALUES (30)").unwrap();

        let rows = db.query("SELECT SUM(amount) FROM sales").unwrap();

        match &rows[0].values[0] {
            OwnedValue::Int(sum) => assert_eq!(*sum, 60, "SUM(amount) SHOULD return 60"),
            other => panic!("Expected Int for SUM, got {:?}", other),
        }
    }

    #[test]
    fn avg_returns_average() {
        let dir = tempdir().unwrap();
        let db = Database::create(dir.path().join("test_db")).unwrap();
        db.execute("CREATE TABLE scores (value INT)").unwrap();
        db.execute("INSERT INTO scores VALUES (10)").unwrap();
        db.execute("INSERT INTO scores VALUES (20)").unwrap();
        db.execute("INSERT INTO scores VALUES (30)").unwrap();

        let rows = db.query("SELECT AVG(value) FROM scores").unwrap();

        match &rows[0].values[0] {
            OwnedValue::Float(avg) => {
                assert!((*avg - 20.0).abs() < 0.001, "AVG(value) SHOULD return 20.0")
            }
            OwnedValue::Int(avg) => assert_eq!(*avg, 20, "AVG(value) SHOULD return 20"),
            other => panic!("Expected Float or Int for AVG, got {:?}", other),
        }
    }

    #[test]
    fn min_returns_minimum() {
        let dir = tempdir().unwrap();
        let db = Database::create(dir.path().join("test_db")).unwrap();
        db.execute("CREATE TABLE nums (n INT)").unwrap();
        db.execute("INSERT INTO nums VALUES (50)").unwrap();
        db.execute("INSERT INTO nums VALUES (10)").unwrap();
        db.execute("INSERT INTO nums VALUES (30)").unwrap();

        let rows = db.query("SELECT MIN(n) FROM nums").unwrap();

        match &rows[0].values[0] {
            OwnedValue::Int(min) => assert_eq!(*min, 10, "MIN(n) SHOULD return 10"),
            other => panic!("Expected Int for MIN, got {:?}", other),
        }
    }

    #[test]
    fn max_returns_maximum() {
        let dir = tempdir().unwrap();
        let db = Database::create(dir.path().join("test_db")).unwrap();
        db.execute("CREATE TABLE nums (n INT)").unwrap();
        db.execute("INSERT INTO nums VALUES (50)").unwrap();
        db.execute("INSERT INTO nums VALUES (10)").unwrap();
        db.execute("INSERT INTO nums VALUES (30)").unwrap();

        let rows = db.query("SELECT MAX(n) FROM nums").unwrap();

        match &rows[0].values[0] {
            OwnedValue::Int(max) => assert_eq!(*max, 50, "MAX(n) SHOULD return 50"),
            other => panic!("Expected Int for MAX, got {:?}", other),
        }
    }
}

mod edge_case_tests {
    use super::*;

    #[test]
    fn empty_string_stores_and_retrieves_correctly() {
        let dir = tempdir().unwrap();
        let db = Database::create(dir.path().join("test_db")).unwrap();
        db.execute("CREATE TABLE data (text_col TEXT)").unwrap();
        db.execute("INSERT INTO data VALUES ('')").unwrap();

        let rows = db.query("SELECT text_col FROM data").unwrap();

        match &rows[0].values[0] {
            OwnedValue::Text(s) => assert_eq!(s, "", "Empty string SHOULD be stored/retrieved"),
            other => panic!("Expected Text, got {:?}", other),
        }
    }

    #[test]
    fn zero_stores_and_retrieves_correctly() {
        let dir = tempdir().unwrap();
        let db = Database::create(dir.path().join("test_db")).unwrap();
        db.execute("CREATE TABLE data (num INT)").unwrap();
        db.execute("INSERT INTO data VALUES (0)").unwrap();

        let rows = db.query("SELECT num FROM data").unwrap();

        match &rows[0].values[0] {
            OwnedValue::Int(n) => assert_eq!(*n, 0, "Zero SHOULD be stored/retrieved"),
            other => panic!("Expected Int, got {:?}", other),
        }
    }

    #[test]
    fn negative_numbers_store_and_retrieve_correctly() {
        let dir = tempdir().unwrap();
        let db = Database::create(dir.path().join("test_db")).unwrap();
        db.execute("CREATE TABLE data (num INT)").unwrap();
        db.execute("INSERT INTO data VALUES (-42)").unwrap();

        let rows = db.query("SELECT num FROM data").unwrap();

        match &rows[0].values[0] {
            OwnedValue::Int(n) => assert_eq!(*n, -42, "Negative number SHOULD be stored/retrieved"),
            other => panic!("Expected Int, got {:?}", other),
        }
    }

    #[test]
    fn special_characters_in_text_store_correctly() {
        let dir = tempdir().unwrap();
        let db = Database::create(dir.path().join("test_db")).unwrap();
        db.execute("CREATE TABLE data (text_col TEXT)").unwrap();
        db.execute("INSERT INTO data VALUES ('Hello''World')")
            .unwrap();

        let rows = db.query("SELECT text_col FROM data").unwrap();

        match &rows[0].values[0] {
            OwnedValue::Text(s) => {
                assert_eq!(s, "Hello'World", "Escaped quote SHOULD be stored correctly")
            }
            other => panic!("Expected Text, got {:?}", other),
        }
    }

    #[test]
    fn large_number_of_rows_handles_correctly() {
        let dir = tempdir().unwrap();
        let db = Database::create(dir.path().join("test_db")).unwrap();
        db.execute("CREATE TABLE data (id INT)").unwrap();

        for i in 0..1000 {
            db.execute(&format!("INSERT INTO data VALUES ({})", i))
                .unwrap();
        }

        let rows = db.query("SELECT COUNT(*) FROM data").unwrap();

        match &rows[0].values[0] {
            OwnedValue::Int(count) => assert_eq!(*count, 1000, "SHOULD handle 1000 rows"),
            other => panic!("Expected Int for COUNT, got {:?}", other),
        }
    }

    #[test]
    fn table_with_many_columns_works() {
        let dir = tempdir().unwrap();
        let db = Database::create(dir.path().join("test_db")).unwrap();
        db.execute(
            "CREATE TABLE wide (c1 INT, c2 INT, c3 INT, c4 INT, c5 INT,
             c6 INT, c7 INT, c8 INT, c9 INT, c10 INT)",
        )
        .unwrap();
        db.execute("INSERT INTO wide VALUES (1, 2, 3, 4, 5, 6, 7, 8, 9, 10)")
            .unwrap();

        let rows = db.query("SELECT * FROM wide").unwrap();

        assert_eq!(rows[0].values.len(), 10, "SHOULD have 10 columns");
        match &rows[0].values[9] {
            OwnedValue::Int(v) => assert_eq!(*v, 10, "10th column SHOULD be 10"),
            other => panic!("Expected Int, got {:?}", other),
        }
    }
}

mod window_function_tests {
    use super::*;

    #[test]
    fn row_number_assigns_sequential_integers_starting_at_one() {
        let dir = tempdir().unwrap();
        let db = Database::create(dir.path().join("test_db")).unwrap();
        db.execute("CREATE TABLE employees (id INT, name TEXT, salary INT)")
            .unwrap();
        db.execute("INSERT INTO employees VALUES (1, 'Alice', 50000)")
            .unwrap();
        db.execute("INSERT INTO employees VALUES (2, 'Bob', 60000)")
            .unwrap();
        db.execute("INSERT INTO employees VALUES (3, 'Charlie', 55000)")
            .unwrap();

        let rows = db
            .query("SELECT name, ROW_NUMBER() OVER (ORDER BY salary DESC) AS rn FROM employees")
            .unwrap();

        assert_eq!(rows.len(), 3, "SHOULD return 3 rows");

        let mut name_to_rn: Vec<(String, i64)> = rows
            .iter()
            .map(|r| {
                let name = match &r.values[0] {
                    OwnedValue::Text(s) => s.clone(),
                    other => panic!("Expected Text for name, got {:?}", other),
                };
                let rn = match &r.values[1] {
                    OwnedValue::Int(n) => *n,
                    other => panic!("Expected Int for ROW_NUMBER, got {:?}", other),
                };
                (name, rn)
            })
            .collect();
        name_to_rn.sort_by_key(|(name, _)| name.clone());

        assert_eq!(
            name_to_rn,
            vec![
                ("Alice".to_string(), 3),
                ("Bob".to_string(), 1),
                ("Charlie".to_string(), 2)
            ],
            "ROW_NUMBER() SHOULD assign ranks by salary DESC: \
             Bob (60000) -> 1, Charlie (55000) -> 2, Alice (50000) -> 3"
        );
    }

    #[test]
    fn row_number_with_partition_resets_for_each_group() {
        let dir = tempdir().unwrap();
        let db = Database::create(dir.path().join("test_db")).unwrap();
        db.execute("CREATE TABLE employees (id INT, dept TEXT, name TEXT, salary INT)")
            .unwrap();
        db.execute("INSERT INTO employees VALUES (1, 'Sales', 'Alice', 50000)")
            .unwrap();
        db.execute("INSERT INTO employees VALUES (2, 'Sales', 'Bob', 60000)")
            .unwrap();
        db.execute("INSERT INTO employees VALUES (3, 'Engineering', 'Charlie', 70000)")
            .unwrap();
        db.execute("INSERT INTO employees VALUES (4, 'Engineering', 'Diana', 65000)")
            .unwrap();

        let rows = db
            .query(
                "SELECT dept, name, ROW_NUMBER() OVER (PARTITION BY dept ORDER BY salary DESC) AS rn \
                 FROM employees",
            )
            .unwrap();

        assert_eq!(rows.len(), 4, "SHOULD return 4 rows");

        let mut results: Vec<(String, String, i64)> = rows
            .iter()
            .map(|r| {
                let dept = match &r.values[0] {
                    OwnedValue::Text(s) => s.clone(),
                    other => panic!("Expected Text for dept, got {:?}", other),
                };
                let name = match &r.values[1] {
                    OwnedValue::Text(s) => s.clone(),
                    other => panic!("Expected Text for name, got {:?}", other),
                };
                let rn = match &r.values[2] {
                    OwnedValue::Int(n) => *n,
                    other => panic!("Expected Int for ROW_NUMBER, got {:?}", other),
                };
                (dept, name, rn)
            })
            .collect();

        results.sort_by(|a, b| a.1.cmp(&b.1));

        assert_eq!(
            results,
            vec![
                ("Sales".to_string(), "Alice".to_string(), 2),
                ("Sales".to_string(), "Bob".to_string(), 1),
                ("Engineering".to_string(), "Charlie".to_string(), 1),
                ("Engineering".to_string(), "Diana".to_string(), 2),
            ],
            "ROW_NUMBER() SHOULD reset to 1 for each partition (dept). \
             Engineering: Charlie -> 1 (highest $70k), Diana -> 2 ($65k). \
             Sales: Bob -> 1 (highest $60k), Alice -> 2 ($50k)"
        );
    }

    #[test]
    fn rank_assigns_same_rank_to_ties_with_gaps() {
        let dir = tempdir().unwrap();
        let db = Database::create(dir.path().join("test_db")).unwrap();
        db.execute("CREATE TABLE scores (name TEXT, score INT)")
            .unwrap();
        db.execute("INSERT INTO scores VALUES ('Alice', 100)")
            .unwrap();
        db.execute("INSERT INTO scores VALUES ('Bob', 100)")
            .unwrap();
        db.execute("INSERT INTO scores VALUES ('Charlie', 90)")
            .unwrap();
        db.execute("INSERT INTO scores VALUES ('Diana', 80)")
            .unwrap();

        let rows = db
            .query("SELECT name, RANK() OVER (ORDER BY score DESC) AS rnk FROM scores")
            .unwrap();

        assert_eq!(rows.len(), 4, "SHOULD return 4 rows");

        let ranks: Vec<i64> = rows
            .iter()
            .map(|r| match &r.values[1] {
                OwnedValue::Int(n) => *n,
                other => panic!("Expected Int for RANK, got {:?}", other),
            })
            .collect();

        assert_eq!(
            ranks,
            vec![1, 1, 3, 4],
            "RANK() SHOULD assign 1, 1, 3, 4 (skips rank 2 after tie). \
             Alice=100 -> 1, Bob=100 -> 1, Charlie=90 -> 3, Diana=80 -> 4"
        );
    }

    #[test]
    fn dense_rank_assigns_same_rank_to_ties_without_gaps() {
        let dir = tempdir().unwrap();
        let db = Database::create(dir.path().join("test_db")).unwrap();
        db.execute("CREATE TABLE scores (name TEXT, score INT)")
            .unwrap();
        db.execute("INSERT INTO scores VALUES ('Alice', 100)")
            .unwrap();
        db.execute("INSERT INTO scores VALUES ('Bob', 100)")
            .unwrap();
        db.execute("INSERT INTO scores VALUES ('Charlie', 90)")
            .unwrap();
        db.execute("INSERT INTO scores VALUES ('Diana', 80)")
            .unwrap();

        let rows = db
            .query("SELECT name, DENSE_RANK() OVER (ORDER BY score DESC) AS drnk FROM scores")
            .unwrap();

        assert_eq!(rows.len(), 4, "SHOULD return 4 rows");

        let ranks: Vec<i64> = rows
            .iter()
            .map(|r| match &r.values[1] {
                OwnedValue::Int(n) => *n,
                other => panic!("Expected Int for DENSE_RANK, got {:?}", other),
            })
            .collect();

        assert_eq!(
            ranks,
            vec![1, 1, 2, 3],
            "DENSE_RANK() SHOULD assign 1, 1, 2, 3 (no gaps after tie). \
             Alice=100 -> 1, Bob=100 -> 1, Charlie=90 -> 2, Diana=80 -> 3"
        );
    }

    #[test]
    fn window_function_without_partition_treats_all_rows_as_one_partition() {
        let dir = tempdir().unwrap();
        let db = Database::create(dir.path().join("test_db")).unwrap();
        db.execute("CREATE TABLE data (val INT)").unwrap();
        db.execute("INSERT INTO data VALUES (30)").unwrap();
        db.execute("INSERT INTO data VALUES (10)").unwrap();
        db.execute("INSERT INTO data VALUES (20)").unwrap();

        let rows = db
            .query("SELECT val, ROW_NUMBER() OVER (ORDER BY val) AS rn FROM data")
            .unwrap();

        let mut results: Vec<(i64, i64)> = rows
            .iter()
            .map(|r| {
                let val = match &r.values[0] {
                    OwnedValue::Int(v) => *v,
                    other => panic!("Expected Int for val, got {:?}", other),
                };
                let rn = match &r.values[1] {
                    OwnedValue::Int(n) => *n,
                    other => panic!("Expected Int for ROW_NUMBER, got {:?}", other),
                };
                (val, rn)
            })
            .collect();

        results.sort_by_key(|(val, _)| *val);

        assert_eq!(
            results,
            vec![(10, 1), (20, 2), (30, 3)],
            "Without PARTITION BY, all rows SHOULD be in one partition. \
             Ordered by val: 10 -> 1, 20 -> 2, 30 -> 3"
        );
    }

    #[test]
    fn multiple_window_functions_in_same_query() {
        let dir = tempdir().unwrap();
        let db = Database::create(dir.path().join("test_db")).unwrap();
        db.execute("CREATE TABLE scores (name TEXT, score INT)")
            .unwrap();
        db.execute("INSERT INTO scores VALUES ('Alice', 100)")
            .unwrap();
        db.execute("INSERT INTO scores VALUES ('Bob', 100)")
            .unwrap();
        db.execute("INSERT INTO scores VALUES ('Charlie', 90)")
            .unwrap();

        let rows = db
            .query(
                "SELECT name, \
                 ROW_NUMBER() OVER (ORDER BY score DESC) AS rn, \
                 RANK() OVER (ORDER BY score DESC) AS rnk, \
                 DENSE_RANK() OVER (ORDER BY score DESC) AS drnk \
                 FROM scores",
            )
            .unwrap();

        assert_eq!(rows.len(), 3, "SHOULD return 3 rows");

        let first_row_rn = match &rows[0].values[1] {
            OwnedValue::Int(n) => *n,
            other => panic!("Expected Int for ROW_NUMBER, got {:?}", other),
        };
        let first_row_rank = match &rows[0].values[2] {
            OwnedValue::Int(n) => *n,
            other => panic!("Expected Int for RANK, got {:?}", other),
        };
        let first_row_drank = match &rows[0].values[3] {
            OwnedValue::Int(n) => *n,
            other => panic!("Expected Int for DENSE_RANK, got {:?}", other),
        };

        assert_eq!(first_row_rn, 1, "First row ROW_NUMBER SHOULD be 1");
        assert_eq!(first_row_rank, 1, "First row RANK SHOULD be 1");
        assert_eq!(first_row_drank, 1, "First row DENSE_RANK SHOULD be 1");

        let third_row_rn = match &rows[2].values[1] {
            OwnedValue::Int(n) => *n,
            other => panic!("Expected Int for ROW_NUMBER, got {:?}", other),
        };
        let third_row_rank = match &rows[2].values[2] {
            OwnedValue::Int(n) => *n,
            other => panic!("Expected Int for RANK, got {:?}", other),
        };
        let third_row_drank = match &rows[2].values[3] {
            OwnedValue::Int(n) => *n,
            other => panic!("Expected Int for DENSE_RANK, got {:?}", other),
        };

        assert_eq!(third_row_rn, 3, "Third row ROW_NUMBER SHOULD be 3");
        assert_eq!(
            third_row_rank, 3,
            "Third row RANK SHOULD be 3 (skips 2 after tie)"
        );
        assert_eq!(
            third_row_drank, 2,
            "Third row DENSE_RANK SHOULD be 2 (no gap)"
        );
    }

    #[test]
    fn window_function_preserves_original_row_order_in_output() {
        let dir = tempdir().unwrap();
        let db = Database::create(dir.path().join("test_db")).unwrap();
        db.execute("CREATE TABLE data (id INT, val INT)").unwrap();
        db.execute("INSERT INTO data VALUES (1, 30)").unwrap();
        db.execute("INSERT INTO data VALUES (2, 10)").unwrap();
        db.execute("INSERT INTO data VALUES (3, 20)").unwrap();

        let rows = db
            .query("SELECT id, val, ROW_NUMBER() OVER (ORDER BY val) AS rn FROM data ORDER BY id")
            .unwrap();

        let results: Vec<(i64, i64, i64)> = rows
            .iter()
            .map(|r| {
                let id = match &r.values[0] {
                    OwnedValue::Int(v) => *v,
                    other => panic!("Expected Int for id, got {:?}", other),
                };
                let val = match &r.values[1] {
                    OwnedValue::Int(v) => *v,
                    other => panic!("Expected Int for val, got {:?}", other),
                };
                let rn = match &r.values[2] {
                    OwnedValue::Int(n) => *n,
                    other => panic!("Expected Int for ROW_NUMBER, got {:?}", other),
                };
                (id, val, rn)
            })
            .collect();

        assert_eq!(
            results,
            vec![(1, 30, 3), (2, 10, 1), (3, 20, 2)],
            "Window function should compute rank by val order, but final ORDER BY id \
             should preserve id ordering. id=1, val=30 -> rn=3; id=2, val=10 -> rn=1; id=3, val=20 -> rn=2"
        );
    }
}

mod insert_select_tests {
    use super::*;
    use turdb::ExecuteResult;

    #[test]
    fn insert_select_copies_all_rows_from_source_table() {
        let dir = tempdir().unwrap();
        let db = Database::create(dir.path().join("test_db")).unwrap();

        db.execute("CREATE TABLE source (id INT, name TEXT, value INT)")
            .unwrap();
        db.execute("INSERT INTO source VALUES (1, 'Alice', 100)")
            .unwrap();
        db.execute("INSERT INTO source VALUES (2, 'Bob', 200)")
            .unwrap();
        db.execute("INSERT INTO source VALUES (3, 'Charlie', 300)")
            .unwrap();

        db.execute("CREATE TABLE target (id INT, name TEXT, value INT)")
            .unwrap();

        let result = db
            .execute("INSERT INTO target SELECT * FROM source")
            .unwrap();

        let rows_affected = match result {
            ExecuteResult::Insert { rows_affected, .. } => rows_affected,
            other => panic!("Expected Insert result, got {:?}", other),
        };
        assert_eq!(
            rows_affected, 3,
            "INSERT...SELECT SHOULD report 3 rows affected when copying 3 rows"
        );

        let rows = db.query("SELECT * FROM target ORDER BY id").unwrap();
        assert_eq!(
            rows.len(),
            3,
            "Target table SHOULD have 3 rows after INSERT...SELECT"
        );

        let first_id = match &rows[0].values[0] {
            OwnedValue::Int(n) => *n,
            other => panic!("Expected Int, got {:?}", other),
        };
        let first_name = match &rows[0].values[1] {
            OwnedValue::Text(s) => s.clone(),
            other => panic!("Expected Text, got {:?}", other),
        };
        assert_eq!(first_id, 1);
        assert_eq!(first_name, "Alice");
    }

    #[test]
    fn insert_select_with_specific_columns() {
        let dir = tempdir().unwrap();
        let db = Database::create(dir.path().join("test_db")).unwrap();

        db.execute("CREATE TABLE source (id INT, name TEXT, value INT)")
            .unwrap();
        db.execute("INSERT INTO source VALUES (1, 'Alice', 100)")
            .unwrap();
        db.execute("INSERT INTO source VALUES (2, 'Bob', 200)")
            .unwrap();

        db.execute("CREATE TABLE target (name TEXT, value INT)")
            .unwrap();

        let result = db
            .execute("INSERT INTO target (name, value) SELECT name, value FROM source")
            .unwrap();

        let rows_affected = match result {
            ExecuteResult::Insert { rows_affected, .. } => rows_affected,
            other => panic!("Expected Insert result, got {:?}", other),
        };
        assert_eq!(rows_affected, 2, "SHOULD insert 2 rows");

        let rows = db.query("SELECT * FROM target ORDER BY value").unwrap();
        assert_eq!(rows.len(), 2);

        let first_name = match &rows[0].values[0] {
            OwnedValue::Text(s) => s.clone(),
            other => panic!("Expected Text, got {:?}", other),
        };
        let first_value = match &rows[0].values[1] {
            OwnedValue::Int(n) => *n,
            other => panic!("Expected Int, got {:?}", other),
        };
        assert_eq!(first_name, "Alice");
        assert_eq!(first_value, 100);
    }

    #[test]
    fn insert_select_with_where_clause_filters_rows() {
        let dir = tempdir().unwrap();
        let db = Database::create(dir.path().join("test_db")).unwrap();

        db.execute("CREATE TABLE source (id INT, value INT)")
            .unwrap();
        db.execute("INSERT INTO source VALUES (1, 50)").unwrap();
        db.execute("INSERT INTO source VALUES (2, 150)").unwrap();
        db.execute("INSERT INTO source VALUES (3, 250)").unwrap();

        db.execute("CREATE TABLE target (id INT, value INT)")
            .unwrap();

        let result = db
            .execute("INSERT INTO target SELECT * FROM source WHERE value > 100")
            .unwrap();

        let rows_affected = match result {
            ExecuteResult::Insert { rows_affected, .. } => rows_affected,
            other => panic!("Expected Insert result, got {:?}", other),
        };
        assert_eq!(
            rows_affected, 2,
            "INSERT...SELECT with WHERE SHOULD only insert rows matching the predicate"
        );

        let rows = db.query("SELECT * FROM target ORDER BY id").unwrap();
        assert_eq!(rows.len(), 2);

        let ids: Vec<i64> = rows
            .iter()
            .map(|r| match &r.values[0] {
                OwnedValue::Int(n) => *n,
                other => panic!("Expected Int, got {:?}", other),
            })
            .collect();
        assert_eq!(
            ids,
            vec![2, 3],
            "Only rows with value > 100 SHOULD be inserted"
        );
    }

    #[test]
    fn insert_select_from_empty_table_inserts_nothing() {
        let dir = tempdir().unwrap();
        let db = Database::create(dir.path().join("test_db")).unwrap();

        db.execute("CREATE TABLE source (id INT, name TEXT)")
            .unwrap();
        db.execute("CREATE TABLE target (id INT, name TEXT)")
            .unwrap();

        let result = db
            .execute("INSERT INTO target SELECT * FROM source")
            .unwrap();

        let rows_affected = match result {
            ExecuteResult::Insert { rows_affected, .. } => rows_affected,
            other => panic!("Expected Insert result, got {:?}", other),
        };
        assert_eq!(
            rows_affected, 0,
            "INSERT...SELECT from empty table SHOULD report 0 rows affected"
        );

        let rows = db.query("SELECT * FROM target").unwrap();
        assert_eq!(rows.len(), 0, "Target table SHOULD remain empty");
    }

    #[test]
    fn insert_select_with_expression_transforms_data() {
        let dir = tempdir().unwrap();
        let db = Database::create(dir.path().join("test_db")).unwrap();

        db.execute("CREATE TABLE source (id INT, value INT)")
            .unwrap();
        db.execute("INSERT INTO source VALUES (1, 10)").unwrap();
        db.execute("INSERT INTO source VALUES (2, 20)").unwrap();

        db.execute("CREATE TABLE target (id INT, doubled_value INT)")
            .unwrap();

        let result = db
            .execute("INSERT INTO target SELECT id, value * 2 FROM source")
            .unwrap();

        let rows_affected = match result {
            ExecuteResult::Insert { rows_affected, .. } => rows_affected,
            other => panic!("Expected Insert result, got {:?}", other),
        };
        assert_eq!(rows_affected, 2);

        let rows = db.query("SELECT * FROM target ORDER BY id").unwrap();

        let first_doubled = match &rows[0].values[1] {
            OwnedValue::Int(n) => *n,
            other => panic!("Expected Int, got {:?}", other),
        };
        let second_doubled = match &rows[1].values[1] {
            OwnedValue::Int(n) => *n,
            other => panic!("Expected Int, got {:?}", other),
        };

        assert_eq!(first_doubled, 20, "value * 2 for 10 SHOULD be 20");
        assert_eq!(second_doubled, 40, "value * 2 for 20 SHOULD be 40");
    }
}

mod insert_on_conflict_tests {
    use super::*;
    use turdb::ExecuteResult;

    #[test]
    fn insert_on_conflict_do_nothing_skips_duplicate_primary_key() {
        let dir = tempdir().unwrap();
        let db = Database::create(dir.path().join("test_db")).unwrap();

        db.execute("CREATE TABLE users (id INT PRIMARY KEY, name TEXT)")
            .unwrap();
        db.execute("INSERT INTO users VALUES (1, 'Alice')").unwrap();

        let result = db
            .execute("INSERT INTO users VALUES (1, 'Bob') ON CONFLICT DO NOTHING")
            .unwrap();

        let rows_affected = match result {
            ExecuteResult::Insert { rows_affected, .. } => rows_affected,
            other => panic!("Expected Insert result, got {:?}", other),
        };
        assert_eq!(
            rows_affected, 0,
            "ON CONFLICT DO NOTHING SHOULD report 0 rows affected when conflict occurs"
        );

        let rows = db.query("SELECT name FROM users WHERE id = 1").unwrap();
        let name = match &rows[0].values[0] {
            OwnedValue::Text(s) => s.clone(),
            other => panic!("Expected Text, got {:?}", other),
        };
        assert_eq!(
            name, "Alice",
            "Original row SHOULD be preserved when ON CONFLICT DO NOTHING"
        );
    }

    #[test]
    fn insert_on_conflict_do_nothing_skips_duplicate_unique_column() {
        let dir = tempdir().unwrap();
        let db = Database::create(dir.path().join("test_db")).unwrap();

        db.execute("CREATE TABLE users (id INT, email TEXT UNIQUE)")
            .unwrap();
        db.execute("INSERT INTO users VALUES (1, 'alice@example.com')")
            .unwrap();

        let result = db
            .execute("INSERT INTO users VALUES (2, 'alice@example.com') ON CONFLICT DO NOTHING")
            .unwrap();

        let rows_affected = match result {
            ExecuteResult::Insert { rows_affected, .. } => rows_affected,
            other => panic!("Expected Insert result, got {:?}", other),
        };
        assert_eq!(
            rows_affected, 0,
            "ON CONFLICT DO NOTHING SHOULD skip row with duplicate UNIQUE column"
        );

        let rows = db.query("SELECT * FROM users").unwrap();
        assert_eq!(
            rows.len(),
            1,
            "Table SHOULD still have only 1 row after conflict"
        );
    }

    #[test]
    fn insert_on_conflict_do_nothing_inserts_non_conflicting_rows() {
        let dir = tempdir().unwrap();
        let db = Database::create(dir.path().join("test_db")).unwrap();

        db.execute("CREATE TABLE users (id INT PRIMARY KEY, name TEXT)")
            .unwrap();
        db.execute("INSERT INTO users VALUES (1, 'Alice')").unwrap();

        let result = db
            .execute("INSERT INTO users VALUES (2, 'Bob') ON CONFLICT DO NOTHING")
            .unwrap();

        let rows_affected = match result {
            ExecuteResult::Insert { rows_affected, .. } => rows_affected,
            other => panic!("Expected Insert result, got {:?}", other),
        };
        assert_eq!(
            rows_affected, 1,
            "Non-conflicting row SHOULD be inserted even with ON CONFLICT clause"
        );

        let rows = db.query("SELECT * FROM users ORDER BY id").unwrap();
        assert_eq!(
            rows.len(),
            2,
            "Table SHOULD have 2 rows after non-conflicting insert"
        );
    }

    #[test]
    fn insert_on_conflict_do_update_updates_existing_row() {
        let dir = tempdir().unwrap();
        let db = Database::create(dir.path().join("test_db")).unwrap();

        db.execute("CREATE TABLE users (id INT PRIMARY KEY, name TEXT, score INT)")
            .unwrap();
        db.execute("INSERT INTO users VALUES (1, 'Alice', 100)")
            .unwrap();

        let result = db
            .execute("INSERT INTO users VALUES (1, 'Bob', 200) ON CONFLICT (id) DO UPDATE SET name = 'Bob', score = 200")
            .unwrap();

        let rows_affected = match result {
            ExecuteResult::Insert { rows_affected, .. } => rows_affected,
            other => panic!("Expected Insert result, got {:?}", other),
        };
        assert_eq!(
            rows_affected, 1,
            "ON CONFLICT DO UPDATE SHOULD report 1 row affected (the updated row)"
        );

        let rows = db.query("SELECT name, score FROM users WHERE id = 1").unwrap();
        let name = match &rows[0].values[0] {
            OwnedValue::Text(s) => s.clone(),
            other => panic!("Expected Text, got {:?}", other),
        };
        let score = match &rows[0].values[1] {
            OwnedValue::Int(n) => *n,
            other => panic!("Expected Int, got {:?}", other),
        };

        assert_eq!(name, "Bob", "Name SHOULD be updated to 'Bob' on conflict");
        assert_eq!(score, 200, "Score SHOULD be updated to 200 on conflict");
    }

    #[test]
    fn insert_on_conflict_do_update_inserts_when_no_conflict() {
        let dir = tempdir().unwrap();
        let db = Database::create(dir.path().join("test_db")).unwrap();

        db.execute("CREATE TABLE users (id INT PRIMARY KEY, name TEXT)")
            .unwrap();
        db.execute("INSERT INTO users VALUES (1, 'Alice')").unwrap();

        let result = db
            .execute("INSERT INTO users VALUES (2, 'Bob') ON CONFLICT (id) DO UPDATE SET name = 'Updated'")
            .unwrap();

        let rows_affected = match result {
            ExecuteResult::Insert { rows_affected, .. } => rows_affected,
            other => panic!("Expected Insert result, got {:?}", other),
        };
        assert_eq!(
            rows_affected, 1,
            "When no conflict, row SHOULD be inserted normally"
        );

        let rows = db.query("SELECT name FROM users WHERE id = 2").unwrap();
        let name = match &rows[0].values[0] {
            OwnedValue::Text(s) => s.clone(),
            other => panic!("Expected Text, got {:?}", other),
        };
        assert_eq!(
            name, "Bob",
            "Name SHOULD be 'Bob' (from INSERT), not 'Updated' (from ON CONFLICT)"
        );
    }
}

mod pk_select_test {
    use super::*;

    #[test]
    fn select_where_on_primary_key_works() {
        let dir = tempdir().unwrap();
        let db = Database::create(dir.path().join("test_db")).unwrap();

        db.execute("CREATE TABLE users (id INT PRIMARY KEY, name TEXT)")
            .unwrap();
        db.execute("INSERT INTO users VALUES (1, 'Alice')").unwrap();

        let rows = db.query("SELECT name FROM users WHERE id = 1").unwrap();
        assert_eq!(rows.len(), 1, "Should find 1 row");

        let name = match &rows[0].values[0] {
            OwnedValue::Text(s) => s.clone(),
            other => panic!("Expected Text, got {:?}", other),
        };
        assert_eq!(name, "Alice", "Name should be Alice");
    }
}

mod returning_clause_tests {
    use super::*;

    #[test]
    fn insert_returning_star_returns_all_columns() {
        let dir = tempdir().unwrap();
        let db = Database::create(dir.path().join("test_db")).unwrap();

        db.execute("CREATE TABLE users (id INT, name TEXT, score INT)")
            .unwrap();

        let result = db
            .execute("INSERT INTO users VALUES (1, 'Alice', 100) RETURNING *")
            .unwrap();

        let (rows_affected, returned) = match result {
            ExecuteResult::Insert {
                rows_affected,
                returned,
            } => (rows_affected, returned),
            other => panic!("Expected Insert result, got {:?}", other),
        };

        assert_eq!(rows_affected, 1, "INSERT SHOULD affect 1 row");

        let returned_rows = returned.expect("RETURNING * SHOULD return rows");
        assert_eq!(returned_rows.len(), 1, "RETURNING * SHOULD return 1 row");

        let row = &returned_rows[0];
        assert_eq!(row.values.len(), 3, "Returned row SHOULD have 3 columns");

        assert_eq!(
            row.values[0],
            OwnedValue::Int(1),
            "First column (id) SHOULD be 1"
        );
        match &row.values[1] {
            OwnedValue::Text(s) => assert_eq!(s, "Alice", "Second column (name) SHOULD be 'Alice'"),
            other => panic!("Expected Text for name, got {:?}", other),
        }
        assert_eq!(
            row.values[2],
            OwnedValue::Int(100),
            "Third column (score) SHOULD be 100"
        );
    }

    #[test]
    fn insert_returning_specific_columns() {
        let dir = tempdir().unwrap();
        let db = Database::create(dir.path().join("test_db")).unwrap();

        db.execute("CREATE TABLE users (id INT, name TEXT, score INT)")
            .unwrap();

        let result = db
            .execute("INSERT INTO users VALUES (1, 'Bob', 200) RETURNING id, name")
            .unwrap();

        let returned = match result {
            ExecuteResult::Insert { returned, .. } => returned,
            other => panic!("Expected Insert result, got {:?}", other),
        };

        let returned_rows = returned.expect("RETURNING id, name SHOULD return rows");
        assert_eq!(returned_rows.len(), 1, "RETURNING SHOULD return 1 row");

        let row = &returned_rows[0];
        assert_eq!(
            row.values.len(),
            2,
            "RETURNING id, name SHOULD return 2 columns"
        );

        assert_eq!(
            row.values[0],
            OwnedValue::Int(1),
            "First returned column (id) SHOULD be 1"
        );
        match &row.values[1] {
            OwnedValue::Text(s) => {
                assert_eq!(s, "Bob", "Second returned column (name) SHOULD be 'Bob'")
            }
            other => panic!("Expected Text for name, got {:?}", other),
        }
    }

    #[test]
    fn insert_multiple_rows_returning_returns_all_rows() {
        let dir = tempdir().unwrap();
        let db = Database::create(dir.path().join("test_db")).unwrap();

        db.execute("CREATE TABLE users (id INT, name TEXT)")
            .unwrap();

        db.execute("CREATE TABLE source (id INT, name TEXT)")
            .unwrap();
        db.execute("INSERT INTO source VALUES (1, 'Alice')")
            .unwrap();
        db.execute("INSERT INTO source VALUES (2, 'Bob')").unwrap();
        db.execute("INSERT INTO source VALUES (3, 'Carol')")
            .unwrap();

        let result = db
            .execute("INSERT INTO users SELECT * FROM source RETURNING *")
            .unwrap();

        let (rows_affected, returned) = match result {
            ExecuteResult::Insert {
                rows_affected,
                returned,
            } => (rows_affected, returned),
            other => panic!("Expected Insert result, got {:?}", other),
        };

        assert_eq!(
            rows_affected, 3,
            "INSERT...SELECT SHOULD report 3 rows affected"
        );

        let returned_rows = returned.expect("RETURNING * SHOULD return rows");
        assert_eq!(
            returned_rows.len(),
            3,
            "RETURNING * SHOULD return all 3 inserted rows"
        );

        let ids: Vec<i64> = returned_rows
            .iter()
            .map(|r| match &r.values[0] {
                OwnedValue::Int(n) => *n,
                other => panic!("Expected Int, got {:?}", other),
            })
            .collect();
        assert!(ids.contains(&1), "Returned rows SHOULD include id=1");
        assert!(ids.contains(&2), "Returned rows SHOULD include id=2");
        assert!(ids.contains(&3), "Returned rows SHOULD include id=3");
    }

    #[test]
    fn update_returning_star_returns_updated_values() {
        let dir = tempdir().unwrap();
        let db = Database::create(dir.path().join("test_db")).unwrap();

        db.execute("CREATE TABLE users (id INT, name TEXT, score INT)")
            .unwrap();
        db.execute("INSERT INTO users VALUES (1, 'Alice', 100)")
            .unwrap();
        db.execute("INSERT INTO users VALUES (2, 'Bob', 200)")
            .unwrap();

        let result = db
            .execute("UPDATE users SET score = 999 WHERE id = 1 RETURNING *")
            .unwrap();

        let (rows_affected, returned) = match result {
            ExecuteResult::Update {
                rows_affected,
                returned,
            } => (rows_affected, returned),
            other => panic!("Expected Update result, got {:?}", other),
        };

        assert_eq!(rows_affected, 1, "UPDATE SHOULD affect 1 row");

        let returned_rows = returned.expect("RETURNING * SHOULD return rows");
        assert_eq!(returned_rows.len(), 1, "RETURNING * SHOULD return 1 row");

        let row = &returned_rows[0];
        assert_eq!(row.values[0], OwnedValue::Int(1), "id SHOULD still be 1");
        assert_eq!(
            row.values[2],
            OwnedValue::Int(999),
            "score SHOULD be the NEW value (999), not the old value (100)"
        );
    }

    #[test]
    fn delete_returning_star_returns_deleted_row() {
        let dir = tempdir().unwrap();
        let db = Database::create(dir.path().join("test_db")).unwrap();

        db.execute("CREATE TABLE users (id INT, name TEXT, score INT)")
            .unwrap();
        db.execute("INSERT INTO users VALUES (1, 'Alice', 100)")
            .unwrap();
        db.execute("INSERT INTO users VALUES (2, 'Bob', 200)")
            .unwrap();

        let result = db
            .execute("DELETE FROM users WHERE id = 1 RETURNING *")
            .unwrap();

        let (rows_affected, returned) = match result {
            ExecuteResult::Delete {
                rows_affected,
                returned,
            } => (rows_affected, returned),
            other => panic!("Expected Delete result, got {:?}", other),
        };

        assert_eq!(rows_affected, 1, "DELETE SHOULD affect 1 row");

        let returned_rows = returned.expect("RETURNING * SHOULD return deleted rows");
        assert_eq!(
            returned_rows.len(),
            1,
            "RETURNING * SHOULD return 1 deleted row"
        );

        let row = &returned_rows[0];
        assert_eq!(
            row.values[0],
            OwnedValue::Int(1),
            "Deleted row's id SHOULD be 1"
        );
        match &row.values[1] {
            OwnedValue::Text(s) => {
                assert_eq!(s, "Alice", "Deleted row's name SHOULD be 'Alice'")
            }
            other => panic!("Expected Text for name, got {:?}", other),
        }
        assert_eq!(
            row.values[2],
            OwnedValue::Int(100),
            "Deleted row's score SHOULD be 100"
        );

        let remaining = db.query("SELECT * FROM users").unwrap();
        assert_eq!(remaining.len(), 1, "Only 1 row SHOULD remain after delete");
        assert_eq!(
            remaining[0].values[0],
            OwnedValue::Int(2),
            "The remaining row SHOULD be id=2"
        );
    }
}
