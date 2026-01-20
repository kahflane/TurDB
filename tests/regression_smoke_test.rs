//! # Regression Smoke Test
//!
//! This test file serves as the source of truth for TurDB feature correctness.
//! It covers real-life database scenarios including:
//!
//! - DDL: CREATE TABLE, DROP TABLE, ALTER TABLE, CREATE INDEX, DROP INDEX
//! - DML: INSERT, UPDATE, DELETE, SELECT
//! - Constraints: PRIMARY KEY, UNIQUE, FOREIGN KEY, CHECK, NOT NULL
//! - Constraint violations and proper error handling
//! - Transactions: BEGIN, COMMIT, ROLLBACK, SAVEPOINT
//! - Concurrent operations
//! - Edge cases and boundary conditions
//!
//! If any test fails after making changes, it indicates a regression.
//! Do NOT modify expected values to make tests pass - fix the underlying issue.

use std::sync::{Arc, Barrier};
use std::thread;
use tempfile::tempdir;
use turdb::{Database, ExecuteResult, OwnedValue};

fn create_test_db() -> (Database, tempfile::TempDir) {
    let dir = tempdir().unwrap();
    let db = Database::create(dir.path().join("test_db")).unwrap();
    (db, dir)
}

mod ddl_tests {
    use super::*;

    #[test]
    fn create_table_with_all_column_types() {
        let (db, _dir) = create_test_db();

        let result = db
            .execute(
                "CREATE TABLE all_types (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                age INTEGER,
                salary REAL,
                is_active BOOLEAN,
                data BLOB,
                created_at TIMESTAMP,
                unique_code TEXT UNIQUE
            )",
            )
            .unwrap();

        assert!(matches!(result, ExecuteResult::CreateTable { created: true }));

        let rows = db.query("SELECT COUNT(*) FROM all_types").unwrap();
        assert_eq!(rows.len(), 1);
    }

    #[test]
    fn create_table_with_composite_primary_key() {
        let (db, _dir) = create_test_db();

        db.execute(
            "CREATE TABLE order_items (
                order_id INTEGER,
                product_id INTEGER,
                quantity INTEGER NOT NULL,
                PRIMARY KEY (order_id, product_id)
            )",
        )
        .unwrap();

        db.execute("INSERT INTO order_items (order_id, product_id, quantity) VALUES (1, 100, 5)")
            .unwrap();
        db.execute("INSERT INTO order_items (order_id, product_id, quantity) VALUES (1, 101, 3)")
            .unwrap();
        db.execute("INSERT INTO order_items (order_id, product_id, quantity) VALUES (2, 100, 1)")
            .unwrap();

        let rows = db
            .query("SELECT COUNT(*) as cnt FROM order_items")
            .unwrap();
        assert_eq!(rows.len(), 1);
        match &rows[0].values[0] {
            OwnedValue::Int(cnt) => assert_eq!(*cnt, 3),
            other => panic!("Expected Int, got {:?}", other),
        }
    }

    #[test]
    fn drop_table_removes_table() {
        let (db, _dir) = create_test_db();

        db.execute("CREATE TABLE temp_table (id INTEGER PRIMARY KEY)")
            .unwrap();
        db.execute("INSERT INTO temp_table (id) VALUES (1)")
            .unwrap();
        db.execute("DROP TABLE temp_table").unwrap();

        let result = db.query("SELECT * FROM temp_table");
        assert!(result.is_err());
    }

    #[test]
    fn create_index_and_query() {
        let (db, _dir) = create_test_db();

        db.execute(
            "CREATE TABLE indexed_table (
                id INTEGER PRIMARY KEY,
                email TEXT,
                status INTEGER
            )",
        )
        .unwrap();

        for i in 0..100 {
            db.execute(&format!(
                "INSERT INTO indexed_table (id, email, status) VALUES ({}, 'user{}@test.com', {})",
                i,
                i,
                i % 3
            ))
            .unwrap();
        }

        db.execute("CREATE INDEX idx_email ON indexed_table (email)")
            .unwrap();
        db.execute("CREATE INDEX idx_status ON indexed_table (status)")
            .unwrap();

        let rows = db
            .query("SELECT * FROM indexed_table WHERE email = 'user50@test.com'")
            .unwrap();
        assert_eq!(rows.len(), 1);
        match &rows[0].values[0] {
            OwnedValue::Int(id) => assert_eq!(*id, 50),
            other => panic!("Expected Int, got {:?}", other),
        }
    }

    #[test]
    fn drop_index_removes_index() {
        let (db, _dir) = create_test_db();

        db.execute("CREATE TABLE drop_idx_test (id INTEGER PRIMARY KEY, value TEXT)")
            .unwrap();
        db.execute("CREATE INDEX idx_value ON drop_idx_test (value)")
            .unwrap();
        db.execute("DROP INDEX idx_value").unwrap();

        db.execute("INSERT INTO drop_idx_test (id, value) VALUES (1, 'test')")
            .unwrap();
        let rows = db
            .query("SELECT * FROM drop_idx_test WHERE value = 'test'")
            .unwrap();
        assert_eq!(rows.len(), 1);
    }

    #[test]
    fn alter_table_add_column() {
        let (db, _dir) = create_test_db();

        db.execute("CREATE TABLE alter_test (id INTEGER PRIMARY KEY, name TEXT)")
            .unwrap();
        db.execute("INSERT INTO alter_test (id, name) VALUES (1, 'Alice')")
            .unwrap();

        db.execute("ALTER TABLE alter_test ADD COLUMN age INTEGER")
            .unwrap();

        db.execute("UPDATE alter_test SET age = 30 WHERE id = 1")
            .unwrap();

        let rows = db
            .query("SELECT id, name, age FROM alter_test WHERE id = 1")
            .unwrap();
        assert_eq!(rows.len(), 1);
        match &rows[0].values[2] {
            OwnedValue::Int(age) => assert_eq!(*age, 30),
            other => panic!("Expected Int for age, got {:?}", other),
        }
    }

    #[test]
    fn alter_table_drop_column() {
        let (db, _dir) = create_test_db();

        db.execute(
            "CREATE TABLE drop_col_test (id INTEGER PRIMARY KEY, name TEXT, temp_col INTEGER)",
        )
        .unwrap();
        db.execute("INSERT INTO drop_col_test (id, name, temp_col) VALUES (1, 'Bob', 999)")
            .unwrap();

        db.execute("ALTER TABLE drop_col_test DROP COLUMN temp_col")
            .unwrap();

        let rows = db
            .query("SELECT * FROM drop_col_test WHERE id = 1")
            .unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].values.len(), 2);
    }
}

mod constraint_violation_tests {
    use super::*;

    #[test]
    fn primary_key_violation_returns_error() {
        let (db, _dir) = create_test_db();

        db.execute("CREATE TABLE pk_test (id INTEGER PRIMARY KEY, name TEXT)")
            .unwrap();
        db.execute("INSERT INTO pk_test (id, name) VALUES (1, 'First')")
            .unwrap();

        let result = db.execute("INSERT INTO pk_test (id, name) VALUES (1, 'Duplicate')");

        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(
            err_msg.contains("PRIMARY KEY") || err_msg.contains("constraint"),
            "Error should mention PRIMARY KEY constraint: {}",
            err_msg
        );
    }

    #[test]
    fn unique_constraint_violation_returns_error() {
        let (db, _dir) = create_test_db();

        db.execute("CREATE TABLE unique_test (id INTEGER PRIMARY KEY, email TEXT UNIQUE)")
            .unwrap();
        db.execute("INSERT INTO unique_test (id, email) VALUES (1, 'test@example.com')")
            .unwrap();

        let result =
            db.execute("INSERT INTO unique_test (id, email) VALUES (2, 'test@example.com')");

        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(
            err_msg.contains("UNIQUE") || err_msg.contains("constraint"),
            "Error should mention UNIQUE constraint: {}",
            err_msg
        );
    }

    #[test]
    fn foreign_key_violation_on_insert_returns_error() {
        let (db, _dir) = create_test_db();

        db.execute("CREATE TABLE departments (id INTEGER PRIMARY KEY, name TEXT)")
            .unwrap();
        db.execute(
            "CREATE TABLE employees (
                id INTEGER PRIMARY KEY,
                name TEXT,
                dept_id INTEGER REFERENCES departments(id)
            )",
        )
        .unwrap();

        db.execute("INSERT INTO departments (id, name) VALUES (1, 'Engineering')")
            .unwrap();

        let result =
            db.execute("INSERT INTO employees (id, name, dept_id) VALUES (1, 'Alice', 999)");

        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(
            err_msg.contains("FOREIGN KEY") || err_msg.contains("constraint"),
            "Error should mention FOREIGN KEY constraint: {}",
            err_msg
        );
    }

    #[test]
    fn foreign_key_violation_on_delete_returns_error() {
        let (db, _dir) = create_test_db();

        db.execute("CREATE TABLE parent_table (id INTEGER PRIMARY KEY, name TEXT)")
            .unwrap();
        db.execute(
            "CREATE TABLE child_table (
                id INTEGER PRIMARY KEY,
                parent_id INTEGER REFERENCES parent_table(id)
            )",
        )
        .unwrap();

        db.execute("INSERT INTO parent_table (id, name) VALUES (1, 'Parent')")
            .unwrap();
        db.execute("INSERT INTO child_table (id, parent_id) VALUES (1, 1)")
            .unwrap();

        let result = db.execute("DELETE FROM parent_table WHERE id = 1");

        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(
            err_msg.contains("FOREIGN KEY")
                || err_msg.contains("constraint")
                || err_msg.contains("referenced"),
            "Error should mention foreign key constraint: {}",
            err_msg
        );
    }

    #[test]
    fn not_null_constraint_violation_returns_error() {
        let (db, _dir) = create_test_db();

        db.execute(
            "CREATE TABLE not_null_test (id INTEGER PRIMARY KEY, required_field TEXT NOT NULL)",
        )
        .unwrap();

        let result =
            db.execute("INSERT INTO not_null_test (id, required_field) VALUES (1, NULL)");

        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(
            err_msg.contains("NOT NULL") || err_msg.contains("null") || err_msg.contains("NULL"),
            "Error should mention NOT NULL constraint: {}",
            err_msg
        );
    }

    #[test]
    fn check_constraint_violation_returns_error() {
        let (db, _dir) = create_test_db();

        db.execute(
            "CREATE TABLE check_test (
                id INTEGER PRIMARY KEY,
                age INTEGER CHECK (age >= 0 AND age <= 150)
            )",
        )
        .unwrap();

        let result = db.execute("INSERT INTO check_test (id, age) VALUES (1, -5)");

        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(
            err_msg.contains("CHECK") || err_msg.contains("constraint"),
            "Error should mention CHECK constraint: {}",
            err_msg
        );
    }
}

mod dml_tests {
    use super::*;

    #[test]
    fn insert_and_select_various_data_types() {
        let (db, _dir) = create_test_db();

        db.execute(
            "CREATE TABLE data_types (
                id INTEGER PRIMARY KEY,
                int_col INTEGER,
                real_col REAL,
                text_col TEXT,
                bool_col BOOLEAN
            )",
        )
        .unwrap();

        db.execute(
            "INSERT INTO data_types (id, int_col, real_col, text_col, bool_col)
             VALUES (1, 42, 1.23, 'Hello, World!', TRUE)",
        )
        .unwrap();

        let rows = db.query("SELECT * FROM data_types WHERE id = 1").unwrap();

        assert_eq!(rows.len(), 1);
        match &rows[0].values[1] {
            OwnedValue::Int(v) => assert_eq!(*v, 42),
            other => panic!("Expected Int, got {:?}", other),
        }
        match &rows[0].values[2] {
            OwnedValue::Float(v) => assert!((*v - 1.23).abs() < 0.01),
            other => panic!("Expected Float, got {:?}", other),
        }
        match &rows[0].values[3] {
            OwnedValue::Text(v) => assert_eq!(v, "Hello, World!"),
            other => panic!("Expected Text, got {:?}", other),
        }
        match &rows[0].values[4] {
            OwnedValue::Bool(v) => assert!(*v),
            other => panic!("Expected Bool, got {:?}", other),
        }
    }

    #[test]
    fn insert_with_default_values() {
        let (db, _dir) = create_test_db();

        db.execute(
            "CREATE TABLE default_test (
                id INTEGER PRIMARY KEY,
                name TEXT,
                status INTEGER DEFAULT 0
            )",
        )
        .unwrap();

        db.execute("INSERT INTO default_test (id, name) VALUES (1, 'Test')")
            .unwrap();

        let rows = db
            .query("SELECT * FROM default_test WHERE id = 1")
            .unwrap();

        assert_eq!(rows.len(), 1);
        match &rows[0].values[2] {
            OwnedValue::Int(v) => assert_eq!(*v, 0),
            other => panic!("Expected Int(0), got {:?}", other),
        }
    }

    #[test]
    fn update_single_row() {
        let (db, _dir) = create_test_db();

        db.execute("CREATE TABLE update_test (id INTEGER PRIMARY KEY, value INTEGER)")
            .unwrap();
        db.execute("INSERT INTO update_test (id, value) VALUES (1, 100)")
            .unwrap();
        db.execute("INSERT INTO update_test (id, value) VALUES (2, 200)")
            .unwrap();

        let result = db.execute("UPDATE update_test SET value = 150 WHERE id = 1");
        match result.unwrap() {
            ExecuteResult::Update { rows_affected, .. } => assert_eq!(rows_affected, 1),
            other => panic!("Expected Update result, got {:?}", other),
        }

        let rows = db
            .query("SELECT value FROM update_test WHERE id = 1")
            .unwrap();
        match &rows[0].values[0] {
            OwnedValue::Int(v) => assert_eq!(*v, 150),
            other => panic!("Expected Int(150), got {:?}", other),
        }

        let rows = db
            .query("SELECT value FROM update_test WHERE id = 2")
            .unwrap();
        match &rows[0].values[0] {
            OwnedValue::Int(v) => assert_eq!(*v, 200),
            other => panic!("Expected Int(200), got {:?}", other),
        }
    }

    #[test]
    fn update_multiple_rows() {
        let (db, _dir) = create_test_db();

        db.execute(
            "CREATE TABLE batch_update (id INTEGER PRIMARY KEY, category TEXT, value INTEGER)",
        )
        .unwrap();

        for i in 1..=10 {
            let category = if i % 2 == 0 { "even" } else { "odd" };
            db.execute(&format!(
                "INSERT INTO batch_update (id, category, value) VALUES ({}, '{}', {})",
                i,
                category,
                i * 10
            ))
            .unwrap();
        }

        let result = db.execute("UPDATE batch_update SET value = value * 2 WHERE category = 'even'");
        match result.unwrap() {
            ExecuteResult::Update { rows_affected, .. } => assert_eq!(rows_affected, 5),
            other => panic!("Expected Update result, got {:?}", other),
        }

        let rows = db
            .query("SELECT SUM(value) as total FROM batch_update WHERE category = 'even'")
            .unwrap();
        match &rows[0].values[0] {
            OwnedValue::Int(v) => assert_eq!(*v, 600),
            other => panic!("Expected Int(600), got {:?}", other),
        }
    }

    #[test]
    fn delete_single_row() {
        let (db, _dir) = create_test_db();

        db.execute("CREATE TABLE delete_test (id INTEGER PRIMARY KEY, name TEXT)")
            .unwrap();
        db.execute("INSERT INTO delete_test (id, name) VALUES (1, 'Alice')")
            .unwrap();
        db.execute("INSERT INTO delete_test (id, name) VALUES (2, 'Bob')")
            .unwrap();

        let result = db.execute("DELETE FROM delete_test WHERE id = 1");
        match result.unwrap() {
            ExecuteResult::Delete { rows_affected, .. } => assert_eq!(rows_affected, 1),
            other => panic!("Expected Delete result, got {:?}", other),
        }

        let rows = db.query("SELECT COUNT(*) as cnt FROM delete_test").unwrap();
        match &rows[0].values[0] {
            OwnedValue::Int(v) => assert_eq!(*v, 1),
            other => panic!("Expected Int(1), got {:?}", other),
        }
    }

    #[test]
    fn delete_with_complex_where_clause() {
        let (db, _dir) = create_test_db();

        db.execute(
            "CREATE TABLE complex_delete (id INTEGER PRIMARY KEY, status TEXT, priority INTEGER)",
        )
        .unwrap();

        for i in 1..=20 {
            let status = if i % 3 == 0 { "done" } else { "pending" };
            db.execute(&format!(
                "INSERT INTO complex_delete (id, status, priority) VALUES ({}, '{}', {})",
                i,
                status,
                i % 5
            ))
            .unwrap();
        }

        let result = db.execute("DELETE FROM complex_delete WHERE status = 'done' AND priority < 3");
        match result.unwrap() {
            ExecuteResult::Delete { rows_affected, .. } => assert!(rows_affected > 0),
            other => panic!("Expected Delete result, got {:?}", other),
        }

        let rows = db
            .query("SELECT COUNT(*) as cnt FROM complex_delete WHERE status = 'done' AND priority < 3")
            .unwrap();
        match &rows[0].values[0] {
            OwnedValue::Int(v) => assert_eq!(*v, 0),
            other => panic!("Expected Int(0), got {:?}", other),
        }
    }

    #[test]
    fn insert_on_conflict_do_nothing() {
        let (db, _dir) = create_test_db();

        db.execute("CREATE TABLE conflict_nothing (id INTEGER PRIMARY KEY, value TEXT)")
            .unwrap();
        db.execute("INSERT INTO conflict_nothing (id, value) VALUES (1, 'original')")
            .unwrap();

        db.execute(
            "INSERT INTO conflict_nothing (id, value) VALUES (1, 'duplicate') ON CONFLICT DO NOTHING",
        )
        .unwrap();

        let rows = db
            .query("SELECT value FROM conflict_nothing WHERE id = 1")
            .unwrap();
        match &rows[0].values[0] {
            OwnedValue::Text(v) => assert_eq!(v, "original"),
            other => panic!("Expected Text('original'), got {:?}", other),
        }
    }

    #[test]
    fn insert_on_conflict_do_update() {
        let (db, _dir) = create_test_db();

        db.execute("CREATE TABLE conflict_update (id INTEGER PRIMARY KEY, counter INTEGER)")
            .unwrap();
        db.execute("INSERT INTO conflict_update (id, counter) VALUES (1, 1)")
            .unwrap();

        db.execute(
            "INSERT INTO conflict_update (id, counter) VALUES (1, 1)
             ON CONFLICT (id) DO UPDATE SET counter = counter + 1",
        )
        .unwrap();

        let rows = db
            .query("SELECT counter FROM conflict_update WHERE id = 1")
            .unwrap();
        match &rows[0].values[0] {
            OwnedValue::Int(v) => assert_eq!(*v, 2),
            other => panic!("Expected Int(2), got {:?}", other),
        }
    }

    #[test]
    fn insert_returning_clause() {
        let (db, _dir) = create_test_db();

        db.execute(
            "CREATE TABLE returning_test (id INTEGER PRIMARY KEY AUTO_INCREMENT, name TEXT)",
        )
        .unwrap();

        let result = db
            .execute("INSERT INTO returning_test (name) VALUES ('Alice') RETURNING id, name")
            .unwrap();

        match result {
            ExecuteResult::Insert {
                rows_affected,
                returned,
            } => {
                assert_eq!(rows_affected, 1);
                let rows = returned.unwrap();
                assert_eq!(rows.len(), 1);
                match &rows[0].values[0] {
                    OwnedValue::Int(id) => assert_eq!(*id, 1),
                    other => panic!("Expected Int(1), got {:?}", other),
                }
            }
            other => panic!("Expected Insert result, got {:?}", other),
        }
    }

    #[test]
    fn bulk_insert_multiple_rows() {
        let (db, _dir) = create_test_db();

        db.execute("CREATE TABLE bulk_insert (id INTEGER PRIMARY KEY, value INTEGER)")
            .unwrap();

        db.execute(
            "INSERT INTO bulk_insert (id, value) VALUES (1, 10), (2, 20), (3, 30), (4, 40), (5, 50)",
        )
        .unwrap();

        let rows = db
            .query("SELECT SUM(value) as total FROM bulk_insert")
            .unwrap();
        match &rows[0].values[0] {
            OwnedValue::Int(v) => assert_eq!(*v, 150),
            other => panic!("Expected Int(150), got {:?}", other),
        }
    }
}

mod query_tests {
    use super::*;

    #[test]
    fn select_with_where_clause() {
        let (db, _dir) = create_test_db();

        db.execute("CREATE TABLE where_test (id INTEGER PRIMARY KEY, name TEXT, age INTEGER)")
            .unwrap();
        db.execute(
            "INSERT INTO where_test VALUES (1, 'Alice', 30), (2, 'Bob', 25), (3, 'Charlie', 35)",
        )
        .unwrap();

        let rows = db
            .query("SELECT name FROM where_test WHERE age > 28")
            .unwrap();

        assert_eq!(rows.len(), 2);
    }

    #[test]
    fn select_with_order_by() {
        let (db, _dir) = create_test_db();

        db.execute("CREATE TABLE order_test (id INTEGER PRIMARY KEY, value INTEGER)")
            .unwrap();
        db.execute("INSERT INTO order_test VALUES (1, 30), (2, 10), (3, 20)")
            .unwrap();

        let rows = db
            .query("SELECT value FROM order_test ORDER BY value ASC")
            .unwrap();

        match &rows[0].values[0] {
            OwnedValue::Int(v) => assert_eq!(*v, 10),
            other => panic!("Expected Int(10), got {:?}", other),
        }
        match &rows[1].values[0] {
            OwnedValue::Int(v) => assert_eq!(*v, 20),
            other => panic!("Expected Int(20), got {:?}", other),
        }
        match &rows[2].values[0] {
            OwnedValue::Int(v) => assert_eq!(*v, 30),
            other => panic!("Expected Int(30), got {:?}", other),
        }
    }

    #[test]
    fn select_with_limit_and_offset() {
        let (db, _dir) = create_test_db();

        db.execute("CREATE TABLE limit_test (id INTEGER PRIMARY KEY, value INTEGER)")
            .unwrap();

        for i in 1..=10 {
            db.execute(&format!(
                "INSERT INTO limit_test (id, value) VALUES ({}, {})",
                i,
                i * 10
            ))
            .unwrap();
        }

        let rows = db
            .query("SELECT value FROM limit_test ORDER BY id LIMIT 3 OFFSET 2")
            .unwrap();

        assert_eq!(rows.len(), 3);
        match &rows[0].values[0] {
            OwnedValue::Int(v) => assert_eq!(*v, 30),
            other => panic!("Expected Int(30), got {:?}", other),
        }
    }

    #[test]
    fn having_sum_gt_250_returns_only_groups_above_threshold() {
        let (db, _dir) = create_test_db();

        db.execute(
            "CREATE TABLE group_test (id INTEGER PRIMARY KEY, category TEXT, amount INTEGER)",
        )
        .unwrap();
        db.execute(
            "INSERT INTO group_test VALUES
             (1, 'A', 100), (2, 'A', 150), (3, 'B', 200),
             (4, 'B', 50), (5, 'C', 300)",
        )
        .unwrap();

        let rows = db
            .query(
                "SELECT category, SUM(amount) as total FROM group_test
                 GROUP BY category HAVING SUM(amount) > 250",
            )
            .unwrap();

        assert_eq!(
            rows.len(),
            1,
            "HAVING SUM(amount) > 250 should return only C (sum=300). A=250 and B=250 excluded"
        );

        match &rows[0].values[0] {
            OwnedValue::Text(cat) => assert_eq!(cat, "C", "Only category C should pass HAVING"),
            other => panic!("Expected Text(C), got {:?}", other),
        }
        match &rows[0].values[1] {
            OwnedValue::Int(total) => assert_eq!(*total, 300, "SUM for C should be 300"),
            other => panic!("Expected Int(300), got {:?}", other),
        }
    }

    #[test]
    fn having_count_ge_2_excludes_single_item_groups() {
        let (db, _dir) = create_test_db();

        db.execute(
            "CREATE TABLE items (id INTEGER PRIMARY KEY, category TEXT, value INTEGER)",
        )
        .unwrap();
        db.execute(
            "INSERT INTO items VALUES
             (1, 'A', 10), (2, 'A', 20),
             (3, 'B', 30), (4, 'B', 40),
             (5, 'C', 50)",
        )
        .unwrap();

        let rows = db
            .query(
                "SELECT category, COUNT(*) as cnt
                 FROM items
                 GROUP BY category
                 HAVING COUNT(*) >= 2",
            )
            .unwrap();

        assert_eq!(
            rows.len(),
            2,
            "HAVING COUNT(*) >= 2 should filter to only A and B (each has 2 items), but got {} rows",
            rows.len()
        );

        let mut found_a = false;
        let mut found_b = false;
        let mut found_c = false;

        for row in &rows {
            match &row.values[0] {
                OwnedValue::Text(s) => {
                    if s == "A" {
                        found_a = true;
                    } else if s == "B" {
                        found_b = true;
                    } else if s == "C" {
                        found_c = true;
                    }
                }
                other => panic!("Expected Text category, got {:?}", other),
            }
            match &row.values[1] {
                OwnedValue::Int(cnt) => assert_eq!(*cnt, 2, "Each group should have COUNT=2"),
                other => panic!("Expected Int(2), got {:?}", other),
            }
        }

        assert!(found_a && found_b, "Should find categories A and B");
        assert!(!found_c, "Should NOT find category C (only 1 item)");
    }

    #[test]
    fn select_with_join() {
        let (db, _dir) = create_test_db();

        db.execute("CREATE TABLE join_users (id INTEGER PRIMARY KEY, name TEXT)")
            .unwrap();
        db.execute(
            "CREATE TABLE join_orders (id INTEGER PRIMARY KEY, user_id INTEGER, amount INTEGER)",
        )
        .unwrap();

        db.execute("INSERT INTO join_users VALUES (1, 'Alice'), (2, 'Bob')")
            .unwrap();
        db.execute("INSERT INTO join_orders VALUES (1, 1, 100), (2, 1, 200), (3, 2, 150)")
            .unwrap();

        let rows = db
            .query(
                "SELECT u.name, SUM(o.amount) as total
                 FROM join_users u
                 JOIN join_orders o ON u.id = o.user_id
                 GROUP BY u.name",
            )
            .unwrap();

        assert_eq!(rows.len(), 2);
    }

    #[test]
    fn select_with_subquery() {
        let (db, _dir) = create_test_db();

        db.execute("CREATE TABLE subquery_test (id INTEGER PRIMARY KEY, value INTEGER)")
            .unwrap();
        db.execute("INSERT INTO subquery_test VALUES (1, 10), (2, 20), (3, 30), (4, 40), (5, 50)")
            .unwrap();

        let rows = db
            .query(
                "SELECT * FROM subquery_test WHERE value > (SELECT AVG(value) FROM subquery_test)",
            )
            .unwrap();

        assert_eq!(rows.len(), 2);
    }

    #[test]
    fn select_distinct() {
        let (db, _dir) = create_test_db();

        db.execute("CREATE TABLE distinct_test (id INTEGER PRIMARY KEY, category TEXT)")
            .unwrap();
        db.execute("INSERT INTO distinct_test VALUES (1, 'A'), (2, 'B'), (3, 'A'), (4, 'C'), (5, 'B')")
            .unwrap();

        let rows = db
            .query("SELECT DISTINCT category FROM distinct_test ORDER BY category")
            .unwrap();

        assert_eq!(rows.len(), 3);
    }

    #[test]
    fn aggregate_functions() {
        let (db, _dir) = create_test_db();

        db.execute("CREATE TABLE agg_test (id INTEGER PRIMARY KEY, value INTEGER)")
            .unwrap();
        db.execute("INSERT INTO agg_test VALUES (1, 10), (2, 20), (3, 30), (4, 40), (5, 50)")
            .unwrap();

        let rows = db
            .query(
                "SELECT COUNT(*) as cnt, SUM(value) as sum_val,
                        MIN(value) as min_val, MAX(value) as max_val
                 FROM agg_test",
            )
            .unwrap();

        match &rows[0].values[0] {
            OwnedValue::Int(v) => assert_eq!(*v, 5),
            other => panic!("Expected Int(5), got {:?}", other),
        }
        match &rows[0].values[1] {
            OwnedValue::Int(v) => assert_eq!(*v, 150),
            other => panic!("Expected Int(150), got {:?}", other),
        }
    }
}

mod transaction_tests {
    use super::*;

    #[test]
    fn transaction_commit_persists_changes() {
        let (db, _dir) = create_test_db();

        db.execute("CREATE TABLE txn_commit (id INTEGER PRIMARY KEY, value INTEGER)")
            .unwrap();

        db.execute("BEGIN").unwrap();
        db.execute("INSERT INTO txn_commit (id, value) VALUES (1, 100)")
            .unwrap();
        db.execute("INSERT INTO txn_commit (id, value) VALUES (2, 200)")
            .unwrap();
        db.execute("COMMIT").unwrap();

        let rows = db
            .query("SELECT SUM(value) as total FROM txn_commit")
            .unwrap();
        match &rows[0].values[0] {
            OwnedValue::Int(v) => assert_eq!(*v, 300),
            other => panic!("Expected Int(300), got {:?}", other),
        }
    }

    #[test]
    fn transaction_rollback_reverts_changes() {
        let (db, _dir) = create_test_db();

        db.execute("CREATE TABLE txn_rollback (id INTEGER PRIMARY KEY, value INTEGER)")
            .unwrap();
        db.execute("INSERT INTO txn_rollback (id, value) VALUES (1, 100)")
            .unwrap();

        db.execute("BEGIN").unwrap();
        db.execute("UPDATE txn_rollback SET value = 999 WHERE id = 1")
            .unwrap();
        db.execute("INSERT INTO txn_rollback (id, value) VALUES (2, 200)")
            .unwrap();
        db.execute("ROLLBACK").unwrap();

        let rows = db.query("SELECT * FROM txn_rollback").unwrap();

        assert_eq!(rows.len(), 1);
        match &rows[0].values[1] {
            OwnedValue::Int(v) => assert_eq!(*v, 100),
            other => panic!("Expected Int(100), got {:?}", other),
        }
    }

    #[test]
    fn savepoint_allows_partial_rollback() {
        let (db, _dir) = create_test_db();

        db.execute("CREATE TABLE savepoint_test (id INTEGER PRIMARY KEY, value INTEGER)")
            .unwrap();

        db.execute("BEGIN").unwrap();
        db.execute("INSERT INTO savepoint_test (id, value) VALUES (1, 100)")
            .unwrap();
        db.execute("SAVEPOINT sp1").unwrap();
        db.execute("INSERT INTO savepoint_test (id, value) VALUES (2, 200)")
            .unwrap();
        db.execute("ROLLBACK TO sp1").unwrap();
        db.execute("INSERT INTO savepoint_test (id, value) VALUES (3, 300)")
            .unwrap();
        db.execute("COMMIT").unwrap();

        let rows = db
            .query("SELECT id FROM savepoint_test ORDER BY id")
            .unwrap();

        assert_eq!(rows.len(), 2);
        match &rows[0].values[0] {
            OwnedValue::Int(v) => assert_eq!(*v, 1),
            other => panic!("Expected Int(1), got {:?}", other),
        }
        match &rows[1].values[0] {
            OwnedValue::Int(v) => assert_eq!(*v, 3),
            other => panic!("Expected Int(3), got {:?}", other),
        }
    }
}

mod concurrent_tests {
    use super::*;

    #[test]
    fn concurrent_reads_do_not_block() {
        let dir = tempdir().unwrap();
        let db = Arc::new(Database::create(dir.path().join("test_db")).unwrap());

        db.execute("CREATE TABLE concurrent_read (id INTEGER PRIMARY KEY, value INTEGER)")
            .unwrap();

        for i in 1..=100 {
            db.execute(&format!(
                "INSERT INTO concurrent_read (id, value) VALUES ({}, {})",
                i,
                i * 10
            ))
            .unwrap();
        }

        let num_threads = 4;
        let barrier = Arc::new(Barrier::new(num_threads));
        let mut handles = vec![];

        for thread_id in 0..num_threads {
            let db_clone = Arc::clone(&db);
            let barrier_clone = Arc::clone(&barrier);

            let handle = thread::spawn(move || {
                barrier_clone.wait();

                for _ in 0..10 {
                    let rows = db_clone
                        .query(&format!(
                            "SELECT * FROM concurrent_read WHERE id = {}",
                            (thread_id % 100) + 1
                        ))
                        .unwrap();
                    assert_eq!(rows.len(), 1);
                }
            });

            handles.push(handle);
        }

        for handle in handles {
            handle.join().unwrap();
        }
    }

    #[test]
    fn concurrent_writes_to_different_rows() {
        let dir = tempdir().unwrap();
        let db = Arc::new(Database::create(dir.path().join("test_db")).unwrap());

        db.execute("CREATE TABLE concurrent_write (id INTEGER PRIMARY KEY, counter INTEGER)")
            .unwrap();

        for i in 1..=4 {
            db.execute(&format!(
                "INSERT INTO concurrent_write (id, counter) VALUES ({}, 0)",
                i
            ))
            .unwrap();
        }

        let num_threads = 4;
        let barrier = Arc::new(Barrier::new(num_threads));
        let mut handles = vec![];

        for thread_id in 0..num_threads {
            let db_clone = Arc::clone(&db);
            let barrier_clone = Arc::clone(&barrier);

            let handle = thread::spawn(move || {
                barrier_clone.wait();

                for _ in 0..10 {
                    db_clone
                        .execute(&format!(
                            "UPDATE concurrent_write SET counter = counter + 1 WHERE id = {}",
                            thread_id + 1
                        ))
                        .unwrap();
                }
            });

            handles.push(handle);
        }

        for handle in handles {
            handle.join().unwrap();
        }

        let rows = db
            .query("SELECT SUM(counter) as total FROM concurrent_write")
            .unwrap();
        match &rows[0].values[0] {
            OwnedValue::Int(v) => assert_eq!(*v, 40),
            other => panic!("Expected Int(40), got {:?}", other),
        }
    }
}

mod edge_cases {
    use super::*;

    #[test]
    fn empty_table_operations() {
        let (db, _dir) = create_test_db();

        db.execute("CREATE TABLE empty_table (id INTEGER PRIMARY KEY, value TEXT)")
            .unwrap();

        let rows = db.query("SELECT COUNT(*) as cnt FROM empty_table").unwrap();
        match &rows[0].values[0] {
            OwnedValue::Int(v) => assert_eq!(*v, 0),
            other => panic!("Expected Int(0), got {:?}", other),
        }

        let result = db.execute("UPDATE empty_table SET value = 'test'");
        match result.unwrap() {
            ExecuteResult::Update { rows_affected, .. } => assert_eq!(rows_affected, 0),
            other => panic!("Expected Update result, got {:?}", other),
        }

        let result = db.execute("DELETE FROM empty_table WHERE id = 1");
        match result.unwrap() {
            ExecuteResult::Delete { rows_affected, .. } => assert_eq!(rows_affected, 0),
            other => panic!("Expected Delete result, got {:?}", other),
        }
    }

    #[test]
    fn count_with_where_matching_no_rows_returns_one_row_with_zero() {
        let (db, _dir) = create_test_db();

        db.execute("CREATE TABLE count_test (id INTEGER PRIMARY KEY, status TEXT)")
            .unwrap();
        db.execute("INSERT INTO count_test VALUES (1, 'active')").unwrap();
        db.execute("INSERT INTO count_test VALUES (2, 'active')").unwrap();

        let rows = db
            .query("SELECT COUNT(*) as cnt FROM count_test WHERE status = 'nonexistent'")
            .unwrap();
        assert_eq!(rows.len(), 1, "COUNT with WHERE matching no rows should return 1 row");
        match &rows[0].values[0] {
            OwnedValue::Int(v) => assert_eq!(*v, 0, "COUNT should be 0 when no rows match"),
            other => panic!("Expected Int(0), got {:?}", other),
        }
    }

    #[test]
    fn null_handling() {
        let (db, _dir) = create_test_db();

        db.execute("CREATE TABLE null_test (id INTEGER PRIMARY KEY, nullable_col TEXT)")
            .unwrap();
        db.execute("INSERT INTO null_test (id, nullable_col) VALUES (1, NULL)")
            .unwrap();
        db.execute("INSERT INTO null_test (id, nullable_col) VALUES (2, 'not null')")
            .unwrap();

        let rows = db
            .query("SELECT * FROM null_test WHERE nullable_col IS NULL")
            .unwrap();
        assert_eq!(rows.len(), 1);
        match &rows[0].values[0] {
            OwnedValue::Int(v) => assert_eq!(*v, 1),
            other => panic!("Expected Int(1), got {:?}", other),
        }

        let rows = db
            .query("SELECT * FROM null_test WHERE nullable_col IS NOT NULL")
            .unwrap();
        assert_eq!(rows.len(), 1);
        match &rows[0].values[0] {
            OwnedValue::Int(v) => assert_eq!(*v, 2),
            other => panic!("Expected Int(2), got {:?}", other),
        }
    }

    #[test]
    fn auto_increment_generates_sequential_ids() {
        let (db, _dir) = create_test_db();

        db.execute("CREATE TABLE auto_inc (id INTEGER PRIMARY KEY AUTO_INCREMENT, name TEXT)")
            .unwrap();

        for name in &["Alice", "Bob", "Charlie"] {
            db.execute(&format!("INSERT INTO auto_inc (name) VALUES ('{}')", name))
                .unwrap();
        }

        let rows = db.query("SELECT id, name FROM auto_inc ORDER BY id").unwrap();

        assert_eq!(rows.len(), 3);
        match &rows[0].values[0] {
            OwnedValue::Int(v) => assert_eq!(*v, 1),
            other => panic!("Expected Int(1), got {:?}", other),
        }
        match &rows[1].values[0] {
            OwnedValue::Int(v) => assert_eq!(*v, 2),
            other => panic!("Expected Int(2), got {:?}", other),
        }
        match &rows[2].values[0] {
            OwnedValue::Int(v) => assert_eq!(*v, 3),
            other => panic!("Expected Int(3), got {:?}", other),
        }
    }

    #[test]
    fn truncate_table_removes_all_rows() {
        let (db, _dir) = create_test_db();

        db.execute("CREATE TABLE truncate_test (id INTEGER PRIMARY KEY, value INTEGER)")
            .unwrap();

        for i in 1..=100 {
            db.execute(&format!(
                "INSERT INTO truncate_test (id, value) VALUES ({}, {})",
                i, i
            ))
            .unwrap();
        }

        db.execute("TRUNCATE TABLE truncate_test").unwrap();

        let rows = db
            .query("SELECT COUNT(*) as cnt FROM truncate_test")
            .unwrap();
        match &rows[0].values[0] {
            OwnedValue::Int(v) => assert_eq!(*v, 0),
            other => panic!("Expected Int(0), got {:?}", other),
        }
    }
}

mod real_world_scenario {
    use super::*;

    #[test]
    fn e_commerce_order_flow() {
        let (db, _dir) = create_test_db();

        db.execute(
            "CREATE TABLE customers (
                id INTEGER PRIMARY KEY AUTO_INCREMENT,
                email TEXT UNIQUE NOT NULL,
                name TEXT NOT NULL
            )",
        )
        .unwrap();

        db.execute(
            "CREATE TABLE products (
                id INTEGER PRIMARY KEY AUTO_INCREMENT,
                name TEXT NOT NULL,
                price REAL NOT NULL CHECK (price > 0),
                stock INTEGER NOT NULL DEFAULT 0 CHECK (stock >= 0)
            )",
        )
        .unwrap();

        db.execute(
            "CREATE TABLE orders (
                id INTEGER PRIMARY KEY AUTO_INCREMENT,
                customer_id INTEGER NOT NULL REFERENCES customers(id),
                status TEXT NOT NULL DEFAULT 'pending',
                total REAL NOT NULL DEFAULT 0
            )",
        )
        .unwrap();

        db.execute(
            "CREATE TABLE order_items (
                id INTEGER PRIMARY KEY AUTO_INCREMENT,
                order_id INTEGER NOT NULL REFERENCES orders(id),
                product_id INTEGER NOT NULL REFERENCES products(id),
                quantity INTEGER NOT NULL CHECK (quantity > 0),
                unit_price REAL NOT NULL
            )",
        )
        .unwrap();

        db.execute("CREATE INDEX idx_orders_customer ON orders(customer_id)")
            .unwrap();
        db.execute("CREATE INDEX idx_order_items_order ON order_items(order_id)")
            .unwrap();

        db.execute("INSERT INTO customers (email, name) VALUES ('alice@example.com', 'Alice Smith')")
            .unwrap();
        db.execute("INSERT INTO customers (email, name) VALUES ('bob@example.com', 'Bob Jones')")
            .unwrap();

        db.execute("INSERT INTO products (name, price, stock) VALUES ('Widget', 29.99, 100)")
            .unwrap();
        db.execute("INSERT INTO products (name, price, stock) VALUES ('Gadget', 49.99, 50)")
            .unwrap();
        db.execute("INSERT INTO products (name, price, stock) VALUES ('Gizmo', 19.99, 200)")
            .unwrap();

        db.execute("BEGIN").unwrap();

        db.execute("INSERT INTO orders (customer_id, status) VALUES (1, 'pending')")
            .unwrap();

        db.execute(
            "INSERT INTO order_items (order_id, product_id, quantity, unit_price)
             VALUES (1, 1, 2, 29.99)",
        )
        .unwrap();
        db.execute(
            "INSERT INTO order_items (order_id, product_id, quantity, unit_price)
             VALUES (1, 2, 1, 49.99)",
        )
        .unwrap();

        db.execute(
            "UPDATE orders SET total = (
                SELECT SUM(quantity * unit_price) FROM order_items WHERE order_id = 1
            ) WHERE id = 1",
        )
        .unwrap();

        db.execute("UPDATE products SET stock = stock - 2 WHERE id = 1")
            .unwrap();
        db.execute("UPDATE products SET stock = stock - 1 WHERE id = 2")
            .unwrap();

        db.execute("COMMIT").unwrap();

        let rows = db.query("SELECT total FROM orders WHERE id = 1").unwrap();
        match &rows[0].values[0] {
            OwnedValue::Float(v) => assert!((*v - 109.97).abs() < 0.01),
            other => panic!("Expected Float near 109.97, got {:?}", other),
        }

        let rows = db
            .query("SELECT stock FROM products WHERE id = 1")
            .unwrap();
        match &rows[0].values[0] {
            OwnedValue::Int(v) => assert_eq!(*v, 98),
            other => panic!("Expected Int(98), got {:?}", other),
        }

        let duplicate_email =
            db.execute("INSERT INTO customers (email, name) VALUES ('alice@example.com', 'Alice Clone')");
        assert!(duplicate_email.is_err());

        let invalid_customer =
            db.execute("INSERT INTO orders (customer_id, status) VALUES (999, 'pending')");
        assert!(invalid_customer.is_err());

        let invalid_price =
            db.execute("INSERT INTO products (name, price, stock) VALUES ('Bad Product', -10, 5)");
        assert!(invalid_price.is_err());

        db.execute("UPDATE orders SET status = 'shipped' WHERE id = 1")
            .unwrap();

        let rows = db
            .query(
                "SELECT c.name as customer, o.status, o.total
             FROM orders o
             JOIN customers c ON o.customer_id = c.id
             WHERE o.id = 1",
            )
            .unwrap();
        match &rows[0].values[0] {
            OwnedValue::Text(v) => assert_eq!(v, "Alice Smith"),
            other => panic!("Expected Text('Alice Smith'), got {:?}", other),
        }
        match &rows[0].values[1] {
            OwnedValue::Text(v) => assert_eq!(v, "shipped"),
            other => panic!("Expected Text('shipped'), got {:?}", other),
        }
    }

    #[test]
    fn update_with_subquery_sum_computes_total_correctly() {
        let (db, _dir) = create_test_db();

        db.execute("CREATE TABLE orders (id INTEGER PRIMARY KEY, total REAL)")
            .unwrap();
        db.execute("CREATE TABLE order_items (order_id INTEGER, amount REAL)")
            .unwrap();

        db.execute("INSERT INTO orders (id, total) VALUES (1, 0.0)")
            .unwrap();
        db.execute("INSERT INTO order_items (order_id, amount) VALUES (1, 25.0)")
            .unwrap();
        db.execute("INSERT INTO order_items (order_id, amount) VALUES (1, 75.0)")
            .unwrap();

        db.execute(
            "UPDATE orders SET total = (SELECT SUM(amount) FROM order_items) WHERE id = 1",
        )
        .unwrap();

        let rows = db.query("SELECT total FROM orders WHERE id = 1").unwrap();
        match &rows[0].values[0] {
            OwnedValue::Float(v) => assert!((*v - 100.0).abs() < 0.01, "Expected 100.0, got {}", v),
            other => panic!("Expected Float, got {:?}", other),
        }
    }

    #[test]
    fn update_with_subquery_avg_computes_average_correctly() {
        let (db, _dir) = create_test_db();

        db.execute("CREATE TABLE stats (id INTEGER PRIMARY KEY, avg_value REAL)")
            .unwrap();
        db.execute("CREATE TABLE values_table (value REAL)")
            .unwrap();

        db.execute("INSERT INTO stats (id, avg_value) VALUES (1, 0.0)")
            .unwrap();
        db.execute("INSERT INTO values_table (value) VALUES (10.0)")
            .unwrap();
        db.execute("INSERT INTO values_table (value) VALUES (20.0)")
            .unwrap();
        db.execute("INSERT INTO values_table (value) VALUES (30.0)")
            .unwrap();

        db.execute(
            "UPDATE stats SET avg_value = (SELECT AVG(value) FROM values_table) WHERE id = 1",
        )
        .unwrap();

        let rows = db
            .query("SELECT avg_value FROM stats WHERE id = 1")
            .unwrap();
        match &rows[0].values[0] {
            OwnedValue::Float(v) => assert!((*v - 20.0).abs() < 0.01, "Expected 20.0, got {}", v),
            other => panic!("Expected Float, got {:?}", other),
        }
    }

    #[test]
    fn update_with_multiple_subqueries_in_set_clause() {
        let (db, _dir) = create_test_db();

        db.execute("CREATE TABLE summary (id INTEGER PRIMARY KEY, min_val REAL, max_val REAL, sum_val REAL)")
            .unwrap();
        db.execute("CREATE TABLE data_points (value REAL)")
            .unwrap();

        db.execute("INSERT INTO summary (id, min_val, max_val, sum_val) VALUES (1, 0, 0, 0)")
            .unwrap();
        db.execute("INSERT INTO data_points (value) VALUES (10.0)")
            .unwrap();
        db.execute("INSERT INTO data_points (value) VALUES (20.0)")
            .unwrap();
        db.execute("INSERT INTO data_points (value) VALUES (30.0)")
            .unwrap();

        db.execute(
            "UPDATE summary SET
                min_val = (SELECT MIN(value) FROM data_points),
                max_val = (SELECT MAX(value) FROM data_points),
                sum_val = (SELECT SUM(value) FROM data_points)
             WHERE id = 1",
        )
        .unwrap();

        let rows = db
            .query("SELECT min_val, max_val, sum_val FROM summary WHERE id = 1")
            .unwrap();

        match &rows[0].values[0] {
            OwnedValue::Float(v) => assert!((*v - 10.0).abs() < 0.01, "Expected min 10.0, got {}", v),
            other => panic!("Expected Float for min_val, got {:?}", other),
        }
        match &rows[0].values[1] {
            OwnedValue::Float(v) => assert!((*v - 30.0).abs() < 0.01, "Expected max 30.0, got {}", v),
            other => panic!("Expected Float for max_val, got {:?}", other),
        }
        match &rows[0].values[2] {
            OwnedValue::Float(v) => assert!((*v - 60.0).abs() < 0.01, "Expected sum 60.0, got {}", v),
            other => panic!("Expected Float for sum_val, got {:?}", other),
        }
    }

    #[test]
    fn blog_platform_scenario() {
        let (db, _dir) = create_test_db();

        db.execute(
            "CREATE TABLE users (
                id INTEGER PRIMARY KEY AUTO_INCREMENT,
                username TEXT UNIQUE NOT NULL,
                email TEXT UNIQUE NOT NULL
            )",
        )
        .unwrap();

        db.execute(
            "CREATE TABLE posts (
                id INTEGER PRIMARY KEY AUTO_INCREMENT,
                author_id INTEGER NOT NULL REFERENCES users(id),
                title TEXT NOT NULL,
                content TEXT,
                published BOOLEAN DEFAULT FALSE,
                view_count INTEGER DEFAULT 0
            )",
        )
        .unwrap();

        db.execute(
            "CREATE TABLE comments (
                id INTEGER PRIMARY KEY AUTO_INCREMENT,
                post_id INTEGER NOT NULL REFERENCES posts(id),
                author_id INTEGER NOT NULL REFERENCES users(id),
                content TEXT NOT NULL
            )",
        )
        .unwrap();

        db.execute(
            "CREATE TABLE tags (
                id INTEGER PRIMARY KEY AUTO_INCREMENT,
                name TEXT UNIQUE NOT NULL
            )",
        )
        .unwrap();

        db.execute(
            "CREATE TABLE post_tags (
                post_id INTEGER REFERENCES posts(id),
                tag_id INTEGER REFERENCES tags(id),
                PRIMARY KEY (post_id, tag_id)
            )",
        )
        .unwrap();

        db.execute("INSERT INTO users (username, email) VALUES ('john_doe', 'john@example.com')")
            .unwrap();
        db.execute("INSERT INTO users (username, email) VALUES ('jane_doe', 'jane@example.com')")
            .unwrap();

        db.execute(
            "INSERT INTO posts (author_id, title, content, published)
             VALUES (1, 'First Post', 'Hello World!', TRUE)",
        )
        .unwrap();
        db.execute(
            "INSERT INTO posts (author_id, title, content, published)
             VALUES (1, 'Draft Post', 'Work in progress', FALSE)",
        )
        .unwrap();
        db.execute(
            "INSERT INTO posts (author_id, title, content, published)
             VALUES (2, 'Janes Post', 'Content here', TRUE)",
        )
        .unwrap();

        db.execute("INSERT INTO tags (name) VALUES ('rust')")
            .unwrap();
        db.execute("INSERT INTO tags (name) VALUES ('database')")
            .unwrap();
        db.execute("INSERT INTO tags (name) VALUES ('tutorial')")
            .unwrap();

        db.execute("INSERT INTO post_tags VALUES (1, 1), (1, 2)")
            .unwrap();
        db.execute("INSERT INTO post_tags VALUES (3, 1), (3, 3)")
            .unwrap();

        db.execute("INSERT INTO comments (post_id, author_id, content) VALUES (1, 2, 'Great post!')")
            .unwrap();
        db.execute("INSERT INTO comments (post_id, author_id, content) VALUES (1, 1, 'Thanks!')")
            .unwrap();

        let rows = db
            .query(
                "SELECT p.title, u.username as author
             FROM posts p
             JOIN users u ON p.author_id = u.id
             WHERE p.published = TRUE
             ORDER BY p.id",
            )
            .unwrap();
        assert_eq!(rows.len(), 2);

        let rows = db
            .query(
                "SELECT p.title
             FROM posts p
             JOIN post_tags pt ON p.id = pt.post_id
             JOIN tags t ON pt.tag_id = t.id
             WHERE t.name = 'rust'",
            )
            .unwrap();
        assert_eq!(rows.len(), 2);

        let rows = db
            .query(
                "SELECT p.title, COUNT(c.id) as comment_count
             FROM posts p
             LEFT JOIN comments c ON p.id = c.post_id
             GROUP BY p.id, p.title
             ORDER BY comment_count DESC",
            )
            .unwrap();
        match &rows[0].values[1] {
            OwnedValue::Int(v) => assert_eq!(*v, 2),
            other => panic!("Expected Int(2), got {:?}", other),
        }

        db.execute("UPDATE posts SET view_count = view_count + 1 WHERE id = 1")
            .unwrap();
        db.execute("UPDATE posts SET view_count = view_count + 1 WHERE id = 1")
            .unwrap();

        let rows = db
            .query("SELECT view_count FROM posts WHERE id = 1")
            .unwrap();
        match &rows[0].values[0] {
            OwnedValue::Int(v) => assert_eq!(*v, 2),
            other => panic!("Expected Int(2), got {:?}", other),
        }

        let delete_user_with_posts = db.execute("DELETE FROM users WHERE id = 1");
        assert!(delete_user_with_posts.is_err());

        db.execute("DELETE FROM comments WHERE post_id = 1")
            .unwrap();
        db.execute("DELETE FROM post_tags WHERE post_id = 1")
            .unwrap();
        db.execute("DELETE FROM posts WHERE id = 1").unwrap();

        let rows = db
            .query("SELECT COUNT(*) as cnt FROM posts WHERE author_id = 1")
            .unwrap();
        match &rows[0].values[0] {
            OwnedValue::Int(v) => assert_eq!(*v, 1),
            other => panic!("Expected Int(1), got {:?}", other),
        }
    }
}
