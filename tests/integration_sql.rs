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

        let result = db.execute("CREATE TABLE users (id INT, name TEXT)").unwrap();

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

        let result = db
            .execute("INSERT INTO users VALUES (1, 'Alice')")
            .unwrap();

        assert!(
            matches!(result, ExecuteResult::Insert { rows_affected: 1 }),
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
        assert_eq!(
            rows[0].values.len(),
            2,
            "Row SHOULD have exactly 2 columns"
        );

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
            matches!(result, ExecuteResult::Update { rows_affected: 2 }),
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
            matches!(result, ExecuteResult::Update { rows_affected: 0 }),
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
            matches!(result, ExecuteResult::Delete { rows_affected: 2 }),
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
            matches!(result, ExecuteResult::Delete { rows_affected: 3 }),
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
        db.execute("INSERT INTO users VALUES (1, 'Original')").unwrap();

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

        assert!(
            result.is_err(),
            "PRIMARY KEY SHOULD reject duplicate id=1"
        );
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

        assert!(
            result.is_ok(),
            "CHECK(age >= 0) SHOULD accept age=25"
        );
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
        db.execute("INSERT INTO docs VALUES ('Hello, World!')").unwrap();

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
        db.execute("INSERT INTO data VALUES ('Hello''World')").unwrap();

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

mod subquery_tests {
    use super::*;

    #[test]
    fn select_from_subquery_returns_subquery_results() {
        let dir = tempdir().unwrap();
        let db = Database::create(dir.path().join("test_db")).unwrap();
        db.execute("CREATE TABLE users (id INT, name TEXT, age INT)")
            .unwrap();
        db.execute("INSERT INTO users VALUES (1, 'Alice', 30)")
            .unwrap();
        db.execute("INSERT INTO users VALUES (2, 'Bob', 25)").unwrap();
        db.execute("INSERT INTO users VALUES (3, 'Carol', 35)")
            .unwrap();

        let rows = db
            .query("SELECT * FROM (SELECT id, name FROM users WHERE age > 26) AS older_users")
            .unwrap();

        assert_eq!(
            rows.len(),
            2,
            "Subquery SHOULD return 2 rows (Alice and Carol, age > 26)"
        );
    }

    #[test]
    fn subquery_can_be_filtered_by_outer_query() {
        let dir = tempdir().unwrap();
        let db = Database::create(dir.path().join("test_db")).unwrap();
        db.execute("CREATE TABLE nums (n INT)").unwrap();
        for i in 1..=10 {
            db.execute(&format!("INSERT INTO nums VALUES ({})", i))
                .unwrap();
        }

        let rows = db
            .query("SELECT * FROM (SELECT n FROM nums WHERE n <= 5) AS small WHERE n > 2")
            .unwrap();

        assert_eq!(
            rows.len(),
            3,
            "Outer WHERE SHOULD filter subquery results: 3, 4, 5"
        );
    }

    #[test]
    fn subquery_with_aggregation_works() {
        let dir = tempdir().unwrap();
        let db = Database::create(dir.path().join("test_db")).unwrap();
        db.execute("CREATE TABLE sales (region TEXT, amount INT)")
            .unwrap();
        db.execute("INSERT INTO sales VALUES ('North', 100)")
            .unwrap();
        db.execute("INSERT INTO sales VALUES ('North', 200)")
            .unwrap();
        db.execute("INSERT INTO sales VALUES ('South', 150)")
            .unwrap();

        let rows = db
            .query(
                "SELECT region, total FROM (SELECT region, SUM(amount) AS total FROM sales GROUP BY region) AS region_totals WHERE total > 200",
            )
            .unwrap();

        assert_eq!(
            rows.len(),
            1,
            "Only North (300) SHOULD have total > 200"
        );
        match &rows[0].values[1] {
            OwnedValue::Int(total) => {
                assert_eq!(*total, 300, "North total SHOULD be 300")
            }
            other => panic!("Expected Int for total, got {:?}", other),
        }
    }

    #[test]
    fn nested_subqueries_work() {
        let dir = tempdir().unwrap();
        let db = Database::create(dir.path().join("test_db")).unwrap();
        db.execute("CREATE TABLE data (x INT)").unwrap();
        for i in 1..=20 {
            db.execute(&format!("INSERT INTO data VALUES ({})", i))
                .unwrap();
        }

        let rows = db
            .query(
                "SELECT * FROM (SELECT * FROM (SELECT x FROM data WHERE x <= 10) AS inner_sq WHERE x > 5) AS outer_sq",
            )
            .unwrap();

        assert_eq!(
            rows.len(),
            5,
            "Nested subqueries SHOULD return 6, 7, 8, 9, 10"
        );
    }

    #[test]
    fn subquery_with_limit_and_offset_works() {
        let dir = tempdir().unwrap();
        let db = Database::create(dir.path().join("test_db")).unwrap();
        db.execute("CREATE TABLE items (id INT)").unwrap();
        for i in 1..=10 {
            db.execute(&format!("INSERT INTO items VALUES ({})", i))
                .unwrap();
        }

        let rows = db
            .query("SELECT * FROM (SELECT id FROM items LIMIT 5 OFFSET 2) AS paged")
            .unwrap();

        assert_eq!(
            rows.len(),
            5,
            "Subquery with LIMIT 5 OFFSET 2 SHOULD return 5 rows"
        );
    }

    #[test]
    fn join_with_subquery_works() {
        let dir = tempdir().unwrap();
        let db = Database::create(dir.path().join("test_db")).unwrap();
        db.execute("CREATE TABLE orders (id INT, user_id INT)")
            .unwrap();
        db.execute("CREATE TABLE users (id INT, name TEXT)")
            .unwrap();
        db.execute("INSERT INTO users VALUES (1, 'Alice')").unwrap();
        db.execute("INSERT INTO users VALUES (2, 'Bob')").unwrap();
        db.execute("INSERT INTO orders VALUES (100, 1)").unwrap();
        db.execute("INSERT INTO orders VALUES (101, 1)").unwrap();

        let rows = db
            .query(
                "SELECT o.id, u.name FROM orders AS o
                 JOIN (SELECT id, name FROM users) AS u ON o.user_id = u.id",
            )
            .unwrap();

        assert_eq!(rows.len(), 2, "Join with subquery SHOULD return 2 rows");
    }
}

mod cte_tests {
    use super::*;

    #[test]
    fn simple_cte_returns_results() {
        let dir = tempdir().unwrap();
        let db = Database::create(dir.path().join("test_db")).unwrap();
        db.execute("CREATE TABLE users (id INT, name TEXT)")
            .unwrap();
        db.execute("INSERT INTO users VALUES (1, 'Alice')").unwrap();
        db.execute("INSERT INTO users VALUES (2, 'Bob')").unwrap();

        let rows = db
            .query("WITH user_names AS (SELECT name FROM users) SELECT * FROM user_names")
            .unwrap();

        assert_eq!(rows.len(), 2, "CTE SHOULD return 2 user names");
    }

    #[test]
    fn cte_can_reference_another_cte() {
        let dir = tempdir().unwrap();
        let db = Database::create(dir.path().join("test_db")).unwrap();
        db.execute("CREATE TABLE nums (n INT)").unwrap();
        for i in 1..=10 {
            db.execute(&format!("INSERT INTO nums VALUES ({})", i))
                .unwrap();
        }

        let rows = db
            .query(
                "WITH
                 small AS (SELECT n FROM nums WHERE n <= 5),
                 tiny AS (SELECT n FROM small WHERE n <= 3)
                 SELECT * FROM tiny",
            )
            .unwrap();

        assert_eq!(
            rows.len(),
            3,
            "Chained CTEs SHOULD return 3 rows (1, 2, 3)"
        );
    }

    #[test]
    fn cte_with_column_aliases() {
        let dir = tempdir().unwrap();
        let db = Database::create(dir.path().join("test_db")).unwrap();
        db.execute("CREATE TABLE sales (amount INT)").unwrap();
        db.execute("INSERT INTO sales VALUES (100)").unwrap();
        db.execute("INSERT INTO sales VALUES (200)").unwrap();

        let rows = db
            .query(
                "WITH totals (total_amount) AS (SELECT SUM(amount) FROM sales)
                 SELECT total_amount FROM totals",
            )
            .unwrap();

        assert_eq!(rows.len(), 1);
        match &rows[0].values[0] {
            OwnedValue::Int(total) => {
                assert_eq!(*total, 300, "CTE column alias SHOULD work, total=300")
            }
            other => panic!("Expected Int, got {:?}", other),
        }
    }

    #[test]
    fn cte_used_multiple_times_in_query() {
        let dir = tempdir().unwrap();
        let db = Database::create(dir.path().join("test_db")).unwrap();
        db.execute("CREATE TABLE items (id INT, value INT)")
            .unwrap();
        db.execute("INSERT INTO items VALUES (1, 10)").unwrap();
        db.execute("INSERT INTO items VALUES (2, 20)").unwrap();

        let rows = db
            .query(
                "WITH item_values AS (SELECT id, value FROM items)
                 SELECT a.id, b.value
                 FROM item_values a
                 JOIN item_values b ON a.id = b.id",
            )
            .unwrap();

        assert_eq!(
            rows.len(),
            2,
            "CTE used twice in same query SHOULD work"
        );
    }

    #[test]
    fn cte_with_aggregation() {
        let dir = tempdir().unwrap();
        let db = Database::create(dir.path().join("test_db")).unwrap();
        db.execute("CREATE TABLE orders (customer_id INT, amount INT)")
            .unwrap();
        db.execute("INSERT INTO orders VALUES (1, 100)").unwrap();
        db.execute("INSERT INTO orders VALUES (1, 50)").unwrap();
        db.execute("INSERT INTO orders VALUES (2, 200)").unwrap();

        let rows = db
            .query(
                "WITH customer_totals AS (
                    SELECT customer_id, SUM(amount) AS total
                    FROM orders
                    GROUP BY customer_id
                 )
                 SELECT * FROM customer_totals WHERE total >= 150",
            )
            .unwrap();

        assert_eq!(
            rows.len(),
            2,
            "Both customers SHOULD have total >= 150"
        );
    }
}

mod set_operation_tests {
    use super::*;

    #[test]
    fn union_combines_results_without_duplicates() {
        let dir = tempdir().unwrap();
        let db = Database::create(dir.path().join("test_db")).unwrap();
        db.execute("CREATE TABLE t1 (n INT)").unwrap();
        db.execute("CREATE TABLE t2 (n INT)").unwrap();
        db.execute("INSERT INTO t1 VALUES (1)").unwrap();
        db.execute("INSERT INTO t1 VALUES (2)").unwrap();
        db.execute("INSERT INTO t2 VALUES (2)").unwrap();
        db.execute("INSERT INTO t2 VALUES (3)").unwrap();

        let rows = db
            .query("SELECT n FROM t1 UNION SELECT n FROM t2")
            .unwrap();

        assert_eq!(
            rows.len(),
            3,
            "UNION SHOULD return 3 unique values (1, 2, 3)"
        );
    }

    #[test]
    fn union_all_preserves_duplicates() {
        let dir = tempdir().unwrap();
        let db = Database::create(dir.path().join("test_db")).unwrap();
        db.execute("CREATE TABLE t1 (n INT)").unwrap();
        db.execute("CREATE TABLE t2 (n INT)").unwrap();
        db.execute("INSERT INTO t1 VALUES (1)").unwrap();
        db.execute("INSERT INTO t1 VALUES (2)").unwrap();
        db.execute("INSERT INTO t2 VALUES (2)").unwrap();
        db.execute("INSERT INTO t2 VALUES (3)").unwrap();

        let rows = db
            .query("SELECT n FROM t1 UNION ALL SELECT n FROM t2")
            .unwrap();

        assert_eq!(
            rows.len(),
            4,
            "UNION ALL SHOULD return 4 values including duplicate 2"
        );
    }

    #[test]
    fn intersect_returns_common_rows() {
        let dir = tempdir().unwrap();
        let db = Database::create(dir.path().join("test_db")).unwrap();
        db.execute("CREATE TABLE t1 (n INT)").unwrap();
        db.execute("CREATE TABLE t2 (n INT)").unwrap();
        db.execute("INSERT INTO t1 VALUES (1)").unwrap();
        db.execute("INSERT INTO t1 VALUES (2)").unwrap();
        db.execute("INSERT INTO t1 VALUES (3)").unwrap();
        db.execute("INSERT INTO t2 VALUES (2)").unwrap();
        db.execute("INSERT INTO t2 VALUES (3)").unwrap();
        db.execute("INSERT INTO t2 VALUES (4)").unwrap();

        let rows = db
            .query("SELECT n FROM t1 INTERSECT SELECT n FROM t2")
            .unwrap();

        assert_eq!(
            rows.len(),
            2,
            "INTERSECT SHOULD return 2 common values (2, 3)"
        );
    }

    #[test]
    fn except_returns_difference() {
        let dir = tempdir().unwrap();
        let db = Database::create(dir.path().join("test_db")).unwrap();
        db.execute("CREATE TABLE t1 (n INT)").unwrap();
        db.execute("CREATE TABLE t2 (n INT)").unwrap();
        db.execute("INSERT INTO t1 VALUES (1)").unwrap();
        db.execute("INSERT INTO t1 VALUES (2)").unwrap();
        db.execute("INSERT INTO t1 VALUES (3)").unwrap();
        db.execute("INSERT INTO t2 VALUES (2)").unwrap();

        let rows = db
            .query("SELECT n FROM t1 EXCEPT SELECT n FROM t2")
            .unwrap();

        assert_eq!(
            rows.len(),
            2,
            "EXCEPT SHOULD return 2 values (1, 3) not in t2"
        );
    }

    #[test]
    fn union_with_order_by() {
        let dir = tempdir().unwrap();
        let db = Database::create(dir.path().join("test_db")).unwrap();
        db.execute("CREATE TABLE t1 (n INT)").unwrap();
        db.execute("CREATE TABLE t2 (n INT)").unwrap();
        db.execute("INSERT INTO t1 VALUES (3)").unwrap();
        db.execute("INSERT INTO t1 VALUES (1)").unwrap();
        db.execute("INSERT INTO t2 VALUES (2)").unwrap();

        let rows = db
            .query("SELECT n FROM t1 UNION SELECT n FROM t2 ORDER BY n")
            .unwrap();

        assert_eq!(rows.len(), 3);
        match &rows[0].values[0] {
            OwnedValue::Int(n) => assert_eq!(*n, 1, "First row SHOULD be 1 after ORDER BY"),
            other => panic!("Expected Int, got {:?}", other),
        }
        match &rows[2].values[0] {
            OwnedValue::Int(n) => assert_eq!(*n, 3, "Last row SHOULD be 3 after ORDER BY"),
            other => panic!("Expected Int, got {:?}", other),
        }
    }

    #[test]
    fn chained_set_operations() {
        let dir = tempdir().unwrap();
        let db = Database::create(dir.path().join("test_db")).unwrap();
        db.execute("CREATE TABLE t1 (n INT)").unwrap();
        db.execute("CREATE TABLE t2 (n INT)").unwrap();
        db.execute("CREATE TABLE t3 (n INT)").unwrap();
        db.execute("INSERT INTO t1 VALUES (1)").unwrap();
        db.execute("INSERT INTO t1 VALUES (2)").unwrap();
        db.execute("INSERT INTO t2 VALUES (2)").unwrap();
        db.execute("INSERT INTO t2 VALUES (3)").unwrap();
        db.execute("INSERT INTO t3 VALUES (3)").unwrap();
        db.execute("INSERT INTO t3 VALUES (4)").unwrap();

        let rows = db
            .query("SELECT n FROM t1 UNION SELECT n FROM t2 UNION SELECT n FROM t3")
            .unwrap();

        assert_eq!(
            rows.len(),
            4,
            "Chained UNION SHOULD return 4 unique values (1, 2, 3, 4)"
        );
    }

    #[test]
    fn set_operation_with_different_column_names() {
        let dir = tempdir().unwrap();
        let db = Database::create(dir.path().join("test_db")).unwrap();
        db.execute("CREATE TABLE t1 (x INT)").unwrap();
        db.execute("CREATE TABLE t2 (y INT)").unwrap();
        db.execute("INSERT INTO t1 VALUES (1)").unwrap();
        db.execute("INSERT INTO t2 VALUES (2)").unwrap();

        let rows = db.query("SELECT x FROM t1 UNION SELECT y FROM t2").unwrap();

        assert_eq!(
            rows.len(),
            2,
            "UNION SHOULD work with different column names"
        );
    }
}

mod prepared_statement_tests {
    use super::*;

    #[test]
    fn prepare_returns_prepared_statement() {
        let dir = tempdir().unwrap();
        let db = Database::create(dir.path().join("test_db")).unwrap();
        db.execute("CREATE TABLE users (id INT, name TEXT)")
            .unwrap();

        let stmt = db.prepare("SELECT * FROM users WHERE id = ?");

        assert!(
            stmt.is_ok(),
            "prepare() SHOULD return Ok for valid SQL with parameter"
        );
    }

    #[test]
    fn prepared_statement_query_returns_matching_rows() {
        let dir = tempdir().unwrap();
        let db = Database::create(dir.path().join("test_db")).unwrap();
        db.execute("CREATE TABLE users (id INT, name TEXT)")
            .unwrap();
        db.execute("INSERT INTO users VALUES (1, 'Alice')").unwrap();
        db.execute("INSERT INTO users VALUES (2, 'Bob')").unwrap();
        db.execute("INSERT INTO users VALUES (3, 'Carol')").unwrap();

        let stmt = db
            .prepare("SELECT id, name FROM users WHERE id = ?")
            .unwrap();
        let rows = stmt.bind(2i64).query(&db).unwrap();

        assert_eq!(rows.len(), 1, "SHOULD return exactly 1 row for id=2");
        match &rows[0].values[0] {
            OwnedValue::Int(id) => assert_eq!(id, &2, "SHOULD return row with id=2"),
            other => panic!("Expected Int for id, got {:?}", other),
        }
        match &rows[0].values[1] {
            OwnedValue::Text(name) => assert_eq!(name, "Bob", "SHOULD return Bob for id=2"),
            other => panic!("Expected Text for name, got {:?}", other),
        }
    }

    #[test]
    fn prepared_statement_with_multiple_params() {
        let dir = tempdir().unwrap();
        let db = Database::create(dir.path().join("test_db")).unwrap();
        db.execute("CREATE TABLE products (id INT, name TEXT, price INT)")
            .unwrap();
        db.execute("INSERT INTO products VALUES (1, 'Apple', 100)")
            .unwrap();
        db.execute("INSERT INTO products VALUES (2, 'Banana', 50)")
            .unwrap();
        db.execute("INSERT INTO products VALUES (3, 'Cherry', 200)")
            .unwrap();

        let stmt = db
            .prepare("SELECT name FROM products WHERE price > ? AND price < ?")
            .unwrap();
        let rows = stmt.bind(50i64).bind(200i64).query(&db).unwrap();

        assert_eq!(
            rows.len(),
            1,
            "SHOULD return exactly 1 product in price range"
        );
        match &rows[0].values[0] {
            OwnedValue::Text(name) => {
                assert_eq!(name, "Apple", "SHOULD return Apple (price=100)")
            }
            other => panic!("Expected Text for name, got {:?}", other),
        }
    }

    #[test]
    fn prepared_statement_can_be_reused_with_different_params() {
        let dir = tempdir().unwrap();
        let db = Database::create(dir.path().join("test_db")).unwrap();
        db.execute("CREATE TABLE users (id INT, name TEXT)")
            .unwrap();
        db.execute("INSERT INTO users VALUES (1, 'Alice')").unwrap();
        db.execute("INSERT INTO users VALUES (2, 'Bob')").unwrap();
        db.execute("INSERT INTO users VALUES (3, 'Carol')").unwrap();

        let stmt = db.prepare("SELECT name FROM users WHERE id = ?").unwrap();

        let rows1 = stmt.bind(1i64).query(&db).unwrap();
        assert_eq!(rows1.len(), 1, "First query SHOULD return 1 row");
        match &rows1[0].values[0] {
            OwnedValue::Text(name) => assert_eq!(name, "Alice"),
            other => panic!("Expected Text, got {:?}", other),
        }

        let rows2 = stmt.bind(2i64).query(&db).unwrap();
        assert_eq!(rows2.len(), 1, "Second query SHOULD return 1 row");
        match &rows2[0].values[0] {
            OwnedValue::Text(name) => assert_eq!(name, "Bob"),
            other => panic!("Expected Text, got {:?}", other),
        }

        let rows3 = stmt.bind(3i64).query(&db).unwrap();
        assert_eq!(rows3.len(), 1, "Third query SHOULD return 1 row");
        match &rows3[0].values[0] {
            OwnedValue::Text(name) => assert_eq!(name, "Carol"),
            other => panic!("Expected Text, got {:?}", other),
        }
    }

    #[test]
    fn prepared_insert_executes_correctly() {
        let dir = tempdir().unwrap();
        let db = Database::create(dir.path().join("test_db")).unwrap();
        db.execute("CREATE TABLE items (id INT, name TEXT)")
            .unwrap();

        let insert_stmt = db.prepare("INSERT INTO items VALUES (?, ?)").unwrap();
        insert_stmt.bind(1i64).bind("Widget").execute(&db).unwrap();
        insert_stmt.bind(2i64).bind("Gadget").execute(&db).unwrap();

        let rows = db.query("SELECT id, name FROM items ORDER BY id").unwrap();
        assert_eq!(rows.len(), 2, "SHOULD have 2 inserted rows");
        match (&rows[0].values[0], &rows[0].values[1]) {
            (OwnedValue::Int(id), OwnedValue::Text(name)) => {
                assert_eq!(id, &1);
                assert_eq!(name, "Widget");
            }
            other => panic!("Unexpected types: {:?}", other),
        }
        match (&rows[1].values[0], &rows[1].values[1]) {
            (OwnedValue::Int(id), OwnedValue::Text(name)) => {
                assert_eq!(id, &2);
                assert_eq!(name, "Gadget");
            }
            other => panic!("Unexpected types: {:?}", other),
        }
    }

    #[test]
    fn prepared_statement_escapes_single_quotes_in_strings() {
        let dir = tempdir().unwrap();
        let db = Database::create(dir.path().join("test_db")).unwrap();
        db.execute("CREATE TABLE notes (id INT, content TEXT)")
            .unwrap();

        let stmt = db.prepare("INSERT INTO notes VALUES (?, ?)").unwrap();
        stmt.bind(1i64).bind("It's a test").execute(&db).unwrap();
        stmt.bind(2i64).bind("Say \"Hello\"").execute(&db).unwrap();

        let rows = db.query("SELECT content FROM notes ORDER BY id").unwrap();
        assert_eq!(rows.len(), 2);
        match &rows[0].values[0] {
            OwnedValue::Text(s) => assert_eq!(s, "It's a test"),
            other => panic!("Expected Text, got {:?}", other),
        }
        match &rows[1].values[0] {
            OwnedValue::Text(s) => assert_eq!(s, "Say \"Hello\""),
            other => panic!("Expected Text, got {:?}", other),
        }
    }

    #[test]
    fn prepared_statement_error_on_param_count_mismatch() {
        let dir = tempdir().unwrap();
        let db = Database::create(dir.path().join("test_db")).unwrap();
        db.execute("CREATE TABLE users (id INT, name TEXT)")
            .unwrap();

        let stmt = db
            .prepare("SELECT * FROM users WHERE id = ? AND name = ?")
            .unwrap();
        let result = stmt.bind(1i64).query(&db);

        assert!(
            result.is_err(),
            "SHOULD error when fewer params than expected"
        );
        let err_msg = result.unwrap_err().to_string();
        assert!(
            err_msg.contains("parameter count mismatch"),
            "Error should mention parameter count mismatch: {}",
            err_msg
        );
    }
}
