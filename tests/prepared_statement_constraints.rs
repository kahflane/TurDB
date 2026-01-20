//! # Prepared Statement Constraint Validation Test Suite
//!
//! This module tests that prepared statement INSERTs properly enforce all
//! constraints: PRIMARY KEY, UNIQUE, FOREIGN KEY, CHECK, NOT NULL, DEFAULT,
//! and AUTO_INCREMENT.
//!
//! ## Bug Reference
//!
//! GitHub Issue #12: Prepared statement INSERT skips constraint validation.
//! The `insert_cached()` function in batch.rs bypasses validation that
//! `execute_insert_internal()` properly performs.
//!
//! ## Test Categories
//!
//! 1. **PRIMARY KEY**: Duplicate key rejection
//! 2. **UNIQUE**: Duplicate value rejection
//! 3. **FOREIGN KEY**: Reference validation when enabled
//! 4. **CHECK**: Expression evaluation
//! 5. **NOT NULL**: Null value rejection
//! 6. **DEFAULT**: Default value application
//! 7. **AUTO_INCREMENT**: Automatic value generation
//!
//! ## Usage
//!
//! ```sh
//! cargo test --test prepared_statement_constraints --release -- --nocapture
//! ```

use tempfile::tempdir;
use turdb::{Database, OwnedValue};

fn create_test_db() -> (tempfile::TempDir, Database) {
    let dir = tempdir().expect("Failed to create temp dir");
    let db = Database::create(dir.path().join("test_db")).expect("Failed to create database");
    (dir, db)
}

mod primary_key_tests {
    use super::*;

    #[test]
    fn prepared_insert_rejects_duplicate_primary_key() {
        let (_dir, db) = create_test_db();

        db.execute("CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT)")
            .expect("Failed to create table");

        db.execute("INSERT INTO users VALUES (1, 'Alice')")
            .expect("Initial insert should succeed");

        let stmt = db
            .prepare("INSERT INTO users VALUES (?, ?)")
            .expect("Failed to prepare statement");

        let result = stmt
            .bind(OwnedValue::Int(1))
            .bind(OwnedValue::Text("Bob".to_string()))
            .execute(&db);

        assert!(
            result.is_err(),
            "Prepared INSERT with duplicate PK should fail, but got: {:?}",
            result
        );

        let err_msg = result.unwrap_err().to_string().to_lowercase();
        assert!(
            err_msg.contains("primary key") || err_msg.contains("unique") || err_msg.contains("constraint"),
            "Error message should mention PRIMARY KEY constraint violation, got: {}",
            err_msg
        );

        let rows = db.query("SELECT COUNT(*) FROM users").unwrap();
        let count = match rows[0].get(0) {
            Some(OwnedValue::Int(n)) => *n,
            _ => panic!("Expected int count"),
        };
        assert_eq!(count, 1, "Only original row should exist");
    }

    #[test]
    fn prepared_insert_allows_different_primary_keys() {
        let (_dir, db) = create_test_db();

        db.execute("CREATE TABLE items (id INTEGER PRIMARY KEY, name TEXT)")
            .expect("Failed to create table");

        let stmt = db
            .prepare("INSERT INTO items VALUES (?, ?)")
            .expect("Failed to prepare statement");

        stmt.bind(OwnedValue::Int(1))
            .bind(OwnedValue::Text("First".to_string()))
            .execute(&db)
            .expect("First insert should succeed");

        let stmt2 = db
            .prepare("INSERT INTO items VALUES (?, ?)")
            .expect("Failed to prepare statement");

        stmt2
            .bind(OwnedValue::Int(2))
            .bind(OwnedValue::Text("Second".to_string()))
            .execute(&db)
            .expect("Second insert with different PK should succeed");

        let rows = db.query("SELECT COUNT(*) FROM items").unwrap();
        let count = match rows[0].get(0) {
            Some(OwnedValue::Int(n)) => *n,
            _ => panic!("Expected int count"),
        };
        assert_eq!(count, 2, "Both rows should exist");
    }
}

mod unique_constraint_tests {
    use super::*;

    #[test]
    fn prepared_insert_rejects_unique_violation() {
        let (_dir, db) = create_test_db();

        db.execute("CREATE TABLE accounts (id INTEGER PRIMARY KEY, email TEXT UNIQUE, name TEXT)")
            .expect("Failed to create table");

        db.execute("INSERT INTO accounts VALUES (1, 'alice@example.com', 'Alice')")
            .expect("Initial insert should succeed");

        let stmt = db
            .prepare("INSERT INTO accounts VALUES (?, ?, ?)")
            .expect("Failed to prepare statement");

        let result = stmt
            .bind(OwnedValue::Int(2))
            .bind(OwnedValue::Text("alice@example.com".to_string()))
            .bind(OwnedValue::Text("Another Alice".to_string()))
            .execute(&db);

        assert!(
            result.is_err(),
            "Prepared INSERT with duplicate UNIQUE value should fail, but got: {:?}",
            result
        );

        let err_msg = result.unwrap_err().to_string().to_lowercase();
        assert!(
            err_msg.contains("unique") || err_msg.contains("constraint"),
            "Error message should mention UNIQUE constraint violation, got: {}",
            err_msg
        );
    }

    #[test]
    fn prepared_insert_allows_null_in_unique_column() {
        let (_dir, db) = create_test_db();

        db.execute("CREATE TABLE profiles (id INTEGER PRIMARY KEY, nickname TEXT UNIQUE)")
            .expect("Failed to create table");

        let stmt1 = db
            .prepare("INSERT INTO profiles VALUES (?, ?)")
            .expect("Failed to prepare statement");

        stmt1
            .bind(OwnedValue::Int(1))
            .bind(OwnedValue::Null)
            .execute(&db)
            .expect("First NULL should be allowed");

        let stmt2 = db
            .prepare("INSERT INTO profiles VALUES (?, ?)")
            .expect("Failed to prepare statement");

        stmt2
            .bind(OwnedValue::Int(2))
            .bind(OwnedValue::Null)
            .execute(&db)
            .expect("Multiple NULLs should be allowed in UNIQUE column");

        let rows = db.query("SELECT COUNT(*) FROM profiles").unwrap();
        let count = match rows[0].get(0) {
            Some(OwnedValue::Int(n)) => *n,
            _ => panic!("Expected int count"),
        };
        assert_eq!(count, 2, "Both rows with NULL should exist");
    }
}

mod foreign_key_tests {
    use super::*;

    #[test]
    fn prepared_insert_validates_foreign_key_when_enabled() {
        let (_dir, db) = create_test_db();

        db.execute("CREATE TABLE departments (id INTEGER PRIMARY KEY, name TEXT)")
            .expect("Failed to create departments");

        db.execute(
            "CREATE TABLE employees (
                id INTEGER PRIMARY KEY,
                name TEXT,
                dept_id INTEGER REFERENCES departments(id)
            )",
        )
        .expect("Failed to create employees");

        db.execute("INSERT INTO departments VALUES (1, 'Engineering')")
            .expect("Insert department should succeed");

        // FK is enabled by default in TurDB

        let stmt = db
            .prepare("INSERT INTO employees VALUES (?, ?, ?)")
            .expect("Failed to prepare statement");

        let result = stmt
            .bind(OwnedValue::Int(1))
            .bind(OwnedValue::Text("Bob".to_string()))
            .bind(OwnedValue::Int(999))
            .execute(&db);

        assert!(
            result.is_err(),
            "Prepared INSERT with invalid FK reference should fail when FK enabled, but got: {:?}",
            result
        );

        let err_msg = result.unwrap_err().to_string().to_lowercase();
        assert!(
            err_msg.contains("foreign key") || err_msg.contains("referenced") || err_msg.contains("constraint"),
            "Error message should mention FOREIGN KEY constraint violation, got: {}",
            err_msg
        );
    }

    #[test]
    fn prepared_insert_allows_valid_foreign_key() {
        let (_dir, db) = create_test_db();

        db.execute("CREATE TABLE categories (id INTEGER PRIMARY KEY, name TEXT)")
            .expect("Failed to create categories");

        db.execute(
            "CREATE TABLE products (
                id INTEGER PRIMARY KEY,
                name TEXT,
                category_id INTEGER REFERENCES categories(id)
            )",
        )
        .expect("Failed to create products");

        db.execute("INSERT INTO categories VALUES (1, 'Electronics')")
            .expect("Insert category should succeed");

        // FK is enabled by default in TurDB

        let stmt = db
            .prepare("INSERT INTO products VALUES (?, ?, ?)")
            .expect("Failed to prepare statement");

        let result = stmt
            .bind(OwnedValue::Int(1))
            .bind(OwnedValue::Text("Laptop".to_string()))
            .bind(OwnedValue::Int(1))
            .execute(&db);

        assert!(
            result.is_ok(),
            "Prepared INSERT with valid FK reference should succeed, but got: {:?}",
            result.unwrap_err()
        );
    }

    #[test]
    fn prepared_insert_allows_null_foreign_key() {
        let (_dir, db) = create_test_db();

        db.execute("CREATE TABLE teams (id INTEGER PRIMARY KEY, name TEXT)")
            .expect("Failed to create teams");

        db.execute(
            "CREATE TABLE members (
                id INTEGER PRIMARY KEY,
                name TEXT,
                team_id INTEGER REFERENCES teams(id)
            )",
        )
        .expect("Failed to create members");

        // FK is enabled by default in TurDB

        let stmt = db
            .prepare("INSERT INTO members VALUES (?, ?, ?)")
            .expect("Failed to prepare statement");

        let result = stmt
            .bind(OwnedValue::Int(1))
            .bind(OwnedValue::Text("Freelancer".to_string()))
            .bind(OwnedValue::Null)
            .execute(&db);

        assert!(
            result.is_ok(),
            "Prepared INSERT with NULL FK should succeed, but got: {:?}",
            result.unwrap_err()
        );
    }
}

mod check_constraint_tests {
    use super::*;

    #[test]
    fn prepared_insert_validates_check_constraint() {
        let (_dir, db) = create_test_db();

        db.execute(
            "CREATE TABLE persons (
                id INTEGER PRIMARY KEY,
                name TEXT,
                age INTEGER CHECK(age >= 0)
            )",
        )
        .expect("Failed to create table");

        let stmt = db
            .prepare("INSERT INTO persons VALUES (?, ?, ?)")
            .expect("Failed to prepare statement");

        let result = stmt
            .bind(OwnedValue::Int(1))
            .bind(OwnedValue::Text("Negative Age".to_string()))
            .bind(OwnedValue::Int(-5))
            .execute(&db);

        assert!(
            result.is_err(),
            "Prepared INSERT violating CHECK constraint should fail, but got: {:?}",
            result
        );

        let err_msg = result.unwrap_err().to_string().to_lowercase();
        assert!(
            err_msg.contains("check") || err_msg.contains("constraint") || err_msg.contains("violated"),
            "Error message should mention CHECK constraint violation, got: {}",
            err_msg
        );
    }

    #[test]
    fn prepared_insert_allows_value_passing_check() {
        let (_dir, db) = create_test_db();

        db.execute(
            "CREATE TABLE scores (
                id INTEGER PRIMARY KEY,
                value INTEGER CHECK(value >= 0 AND value <= 100)
            )",
        )
        .expect("Failed to create table");

        let stmt = db
            .prepare("INSERT INTO scores VALUES (?, ?)")
            .expect("Failed to prepare statement");

        let result = stmt
            .bind(OwnedValue::Int(1))
            .bind(OwnedValue::Int(85))
            .execute(&db);

        assert!(
            result.is_ok(),
            "Prepared INSERT with valid CHECK value should succeed, but got: {:?}",
            result.unwrap_err()
        );
    }

    #[test]
    fn check_constraint_handles_negative_numbers() {
        let (_dir, db) = create_test_db();

        db.execute(
            "CREATE TABLE temperatures (
                id INTEGER PRIMARY KEY,
                temp REAL CHECK(temp >= -273.15)
            )",
        )
        .expect("Failed to create table");

        let stmt = db
            .prepare("INSERT INTO temperatures VALUES (?, ?)")
            .expect("Failed to prepare statement");

        let result_valid = stmt
            .bind(OwnedValue::Int(1))
            .bind(OwnedValue::Float(-100.0))
            .execute(&db);

        assert!(
            result_valid.is_ok(),
            "INSERT with -100.0 (above -273.15) should succeed"
        );

        let stmt2 = db
            .prepare("INSERT INTO temperatures VALUES (?, ?)")
            .expect("Failed to prepare statement");

        let result_invalid = stmt2
            .bind(OwnedValue::Int(2))
            .bind(OwnedValue::Float(-300.0))
            .execute(&db);

        assert!(
            result_invalid.is_err(),
            "INSERT with -300.0 (below -273.15) should fail"
        );
    }

    #[test]
    fn check_constraint_handles_float_range() {
        let (_dir, db) = create_test_db();

        db.execute(
            "CREATE TABLE ratios (
                id INTEGER PRIMARY KEY,
                ratio REAL CHECK(ratio >= 0.0 AND ratio <= 1.0)
            )",
        )
        .expect("Failed to create table");

        let stmt = db
            .prepare("INSERT INTO ratios VALUES (?, ?)")
            .expect("Failed to prepare statement");

        let result_valid = stmt
            .bind(OwnedValue::Int(1))
            .bind(OwnedValue::Float(0.5))
            .execute(&db);

        assert!(
            result_valid.is_ok(),
            "INSERT with 0.5 should succeed"
        );

        let stmt2 = db
            .prepare("INSERT INTO ratios VALUES (?, ?)")
            .expect("Failed to prepare statement");

        let result_invalid = stmt2
            .bind(OwnedValue::Int(2))
            .bind(OwnedValue::Float(1.5))
            .execute(&db);

        assert!(
            result_invalid.is_err(),
            "INSERT with 1.5 (above 1.0) should fail"
        );
    }

    #[test]
    fn check_constraint_handles_deeply_nested_parentheses() {
        let (_dir, db) = create_test_db();

        db.execute(
            "CREATE TABLE nested_check (
                id INTEGER PRIMARY KEY,
                age INTEGER CHECK((((age >= 0))))
            )",
        )
        .expect("Failed to create table");

        let stmt = db
            .prepare("INSERT INTO nested_check VALUES (?, ?)")
            .expect("Failed to prepare statement");

        let result_valid = stmt
            .bind(OwnedValue::Int(1))
            .bind(OwnedValue::Int(25))
            .execute(&db);

        assert!(
            result_valid.is_ok(),
            "INSERT with valid value through nested parens should succeed"
        );

        let stmt2 = db
            .prepare("INSERT INTO nested_check VALUES (?, ?)")
            .expect("Failed to prepare statement");

        let result_invalid = stmt2
            .bind(OwnedValue::Int(2))
            .bind(OwnedValue::Int(-5))
            .execute(&db);

        assert!(
            result_invalid.is_err(),
            "INSERT with invalid value through nested parens should fail"
        );
    }

    #[test]
    fn check_constraint_handles_mixed_case_operators() {
        let (_dir, db) = create_test_db();

        db.execute(
            "CREATE TABLE mixed_case (
                id INTEGER PRIMARY KEY,
                age INTEGER CHECK(age >= 0 AnD age <= 150)
            )",
        )
        .expect("Failed to create table");

        let stmt = db
            .prepare("INSERT INTO mixed_case VALUES (?, ?)")
            .expect("Failed to prepare statement");

        let result_valid = stmt
            .bind(OwnedValue::Int(1))
            .bind(OwnedValue::Int(75))
            .execute(&db);

        assert!(
            result_valid.is_ok(),
            "INSERT should succeed with mixed case AND operator"
        );

        let stmt2 = db
            .prepare("INSERT INTO mixed_case VALUES (?, ?)")
            .expect("Failed to prepare statement");

        let result_high = stmt2
            .bind(OwnedValue::Int(2))
            .bind(OwnedValue::Int(200))
            .execute(&db);

        assert!(
            result_high.is_err(),
            "INSERT with value above 150 should fail"
        );
    }

    #[test]
    fn check_constraint_handles_multiple_or_conditions() {
        let (_dir, db) = create_test_db();

        db.execute(
            "CREATE TABLE status_codes (
                id INTEGER PRIMARY KEY,
                code INTEGER CHECK(code >= 200 AND code <= 299 OR code >= 400 AND code <= 499)
            )",
        )
        .expect("Failed to create table");

        let stmt = db
            .prepare("INSERT INTO status_codes VALUES (?, ?)")
            .expect("Failed to prepare statement");

        let result_200 = stmt
            .bind(OwnedValue::Int(1))
            .bind(OwnedValue::Int(200))
            .execute(&db);

        assert!(
            result_200.is_ok(),
            "INSERT with 200 should succeed"
        );

        let stmt2 = db
            .prepare("INSERT INTO status_codes VALUES (?, ?)")
            .expect("Failed to prepare statement");

        let result_404 = stmt2
            .bind(OwnedValue::Int(2))
            .bind(OwnedValue::Int(404))
            .execute(&db);

        assert!(
            result_404.is_ok(),
            "INSERT with 404 should succeed"
        );

        let stmt3 = db
            .prepare("INSERT INTO status_codes VALUES (?, ?)")
            .expect("Failed to prepare statement");

        let result_300 = stmt3
            .bind(OwnedValue::Int(3))
            .bind(OwnedValue::Int(300))
            .execute(&db);

        assert!(
            result_300.is_err(),
            "INSERT with 300 should fail (not in valid ranges)"
        );
    }

    #[test]
    fn check_constraint_max_depth_returns_error() {
        let (_dir, db) = create_test_db();

        let mut nested_expr = "age >= 0".to_string();
        for i in 1..35 {
            nested_expr = format!("({} AND age <= {})", nested_expr, 100 + i);
        }

        let create_sql = format!(
            "CREATE TABLE deep_nesting (
                id INTEGER PRIMARY KEY,
                age INTEGER CHECK({})
            )",
            nested_expr
        );

        db.execute(&create_sql)
            .expect("Failed to create table");

        let stmt = db
            .prepare("INSERT INTO deep_nesting VALUES (?, ?)")
            .expect("Failed to prepare statement");

        let result = stmt
            .bind(OwnedValue::Int(1))
            .bind(OwnedValue::Int(25))
            .execute(&db);

        assert!(
            result.is_err(),
            "INSERT with deeply nested AND expression (>32 levels) should return error"
        );

        let err_msg = result.unwrap_err().to_string().to_lowercase();
        assert!(
            err_msg.contains("depth") || err_msg.contains("nesting") || err_msg.contains("exceed"),
            "Error should mention depth/nesting exceeded, got: {}",
            err_msg
        );
    }
}

mod not_null_tests {
    use super::*;

    #[test]
    fn prepared_insert_rejects_null_for_not_null_column() {
        let (_dir, db) = create_test_db();

        db.execute("CREATE TABLE required_fields (id INTEGER PRIMARY KEY, name TEXT NOT NULL)")
            .expect("Failed to create table");

        let stmt = db
            .prepare("INSERT INTO required_fields VALUES (?, ?)")
            .expect("Failed to prepare statement");

        let result = stmt
            .bind(OwnedValue::Int(1))
            .bind(OwnedValue::Null)
            .execute(&db);

        assert!(
            result.is_err(),
            "Prepared INSERT with NULL in NOT NULL column should fail, but got: {:?}",
            result
        );

        let err_msg = result.unwrap_err().to_string().to_lowercase();
        assert!(
            err_msg.contains("not null") || err_msg.contains("null") || err_msg.contains("constraint"),
            "Error message should mention NOT NULL constraint violation, got: {}",
            err_msg
        );
    }
}

mod default_value_tests {
    use super::*;

    #[test]
    fn prepared_insert_applies_default_values() {
        let (_dir, db) = create_test_db();

        db.execute(
            "CREATE TABLE configs (
                id INTEGER PRIMARY KEY,
                setting TEXT,
                value TEXT DEFAULT 'default_value'
            )",
        )
        .expect("Failed to create table");

        let stmt = db
            .prepare("INSERT INTO configs (id, setting) VALUES (?, ?)")
            .expect("Failed to prepare statement");

        stmt.bind(OwnedValue::Int(1))
            .bind(OwnedValue::Text("test_setting".to_string()))
            .execute(&db)
            .expect("Insert should succeed");

        let rows = db.query("SELECT value FROM configs WHERE id = 1").unwrap();
        assert_eq!(rows.len(), 1, "Should have one row");

        let value = match rows[0].get(0) {
            Some(OwnedValue::Text(s)) => s.clone(),
            other => panic!("Expected Text, got {:?}", other),
        };

        assert_eq!(
            value, "default_value",
            "DEFAULT value should be applied when column not specified"
        );
    }

    #[test]
    fn prepared_insert_applies_default_for_explicit_null() {
        let (_dir, db) = create_test_db();

        db.execute(
            "CREATE TABLE settings (
                id INTEGER PRIMARY KEY,
                value INTEGER DEFAULT 42
            )",
        )
        .expect("Failed to create table");

        let stmt = db
            .prepare("INSERT INTO settings VALUES (?, ?)")
            .expect("Failed to prepare statement");

        stmt.bind(OwnedValue::Int(1))
            .bind(OwnedValue::Null)
            .execute(&db)
            .expect("Insert should succeed");

        let rows = db.query("SELECT value FROM settings WHERE id = 1").unwrap();
        let value = match rows[0].get(0) {
            Some(OwnedValue::Int(n)) => *n,
            Some(OwnedValue::Null) => -1,
            other => panic!("Expected Int or Null, got {:?}", other),
        };

        assert_eq!(
            value, 42,
            "DEFAULT value should be applied when NULL is explicitly provided"
        );
    }
}

mod auto_increment_tests {
    use super::*;

    #[test]
    fn prepared_insert_handles_auto_increment() {
        let (_dir, db) = create_test_db();

        db.execute(
            "CREATE TABLE logs (
                id INTEGER PRIMARY KEY AUTO_INCREMENT,
                message TEXT
            )",
        )
        .expect("Failed to create table");

        let stmt1 = db
            .prepare("INSERT INTO logs (message) VALUES (?)")
            .expect("Failed to prepare statement");

        stmt1
            .bind(OwnedValue::Text("First log".to_string()))
            .execute(&db)
            .expect("First insert should succeed");

        let stmt2 = db
            .prepare("INSERT INTO logs (message) VALUES (?)")
            .expect("Failed to prepare statement");

        stmt2
            .bind(OwnedValue::Text("Second log".to_string()))
            .execute(&db)
            .expect("Second insert should succeed");

        let rows = db.query("SELECT id FROM logs ORDER BY id").unwrap();
        assert_eq!(rows.len(), 2, "Should have two rows");

        let id1 = match rows[0].get(0) {
            Some(OwnedValue::Int(n)) => *n,
            other => panic!("Expected Int, got {:?}", other),
        };

        let id2 = match rows[1].get(0) {
            Some(OwnedValue::Int(n)) => *n,
            other => panic!("Expected Int, got {:?}", other),
        };

        assert!(id1 > 0, "First AUTO_INCREMENT value should be positive");
        assert_eq!(id2, id1 + 1, "Second AUTO_INCREMENT should be first + 1");
    }

    #[test]
    fn prepared_insert_respects_explicit_auto_increment_value() {
        let (_dir, db) = create_test_db();

        db.execute(
            "CREATE TABLE events (
                id INTEGER PRIMARY KEY AUTO_INCREMENT,
                name TEXT
            )",
        )
        .expect("Failed to create table");

        let stmt = db
            .prepare("INSERT INTO events VALUES (?, ?)")
            .expect("Failed to prepare statement");

        stmt.bind(OwnedValue::Int(100))
            .bind(OwnedValue::Text("Explicit ID".to_string()))
            .execute(&db)
            .expect("Insert with explicit ID should succeed");

        let stmt2 = db
            .prepare("INSERT INTO events (name) VALUES (?)")
            .expect("Failed to prepare statement");

        stmt2
            .bind(OwnedValue::Text("Auto ID".to_string()))
            .execute(&db)
            .expect("Insert with auto ID should succeed");

        let rows = db.query("SELECT id FROM events ORDER BY id").unwrap();
        assert_eq!(rows.len(), 2, "Should have two rows");

        let id1 = match rows[0].get(0) {
            Some(OwnedValue::Int(n)) => *n,
            other => panic!("Expected Int, got {:?}", other),
        };

        let id2 = match rows[1].get(0) {
            Some(OwnedValue::Int(n)) => *n,
            other => panic!("Expected Int, got {:?}", other),
        };

        assert_eq!(id1, 100, "First ID should be 100 (explicit)");
        assert!(id2 > 100, "Second ID should be > 100 (auto-generated after explicit)");
    }
}

mod composite_pk_tests {
    use super::*;

    #[test]
    fn prepared_insert_validates_composite_primary_key() {
        let (_dir, db) = create_test_db();

        db.execute(
            "CREATE TABLE subscriptions (
                user_id INTEGER,
                product_id INTEGER,
                created_at TEXT,
                PRIMARY KEY (user_id, product_id)
            )",
        )
        .expect("Failed to create table");

        db.execute("INSERT INTO subscriptions VALUES (1, 100, '2024-01-01')")
            .expect("Initial insert should succeed");

        let stmt = db
            .prepare("INSERT INTO subscriptions VALUES (?, ?, ?)")
            .expect("Failed to prepare statement");

        let result = stmt
            .bind(OwnedValue::Int(1))
            .bind(OwnedValue::Int(100))
            .bind(OwnedValue::Text("2024-02-01".to_string()))
            .execute(&db);

        assert!(
            result.is_err(),
            "Prepared INSERT with duplicate composite PK should fail, but got: {:?}",
            result
        );
    }

    #[test]
    fn prepared_insert_allows_partial_composite_pk_match() {
        let (_dir, db) = create_test_db();

        db.execute(
            "CREATE TABLE assignments (
                teacher_id INTEGER,
                class_id INTEGER,
                subject TEXT,
                PRIMARY KEY (teacher_id, class_id)
            )",
        )
        .expect("Failed to create table");

        db.execute("INSERT INTO assignments VALUES (1, 100, 'Math')")
            .expect("Initial insert should succeed");

        let stmt = db
            .prepare("INSERT INTO assignments VALUES (?, ?, ?)")
            .expect("Failed to prepare statement");

        let result = stmt
            .bind(OwnedValue::Int(1))
            .bind(OwnedValue::Int(200))
            .bind(OwnedValue::Text("Science".to_string()))
            .execute(&db);

        assert!(
            result.is_ok(),
            "Prepared INSERT with different composite key should succeed, but got: {:?}",
            result.unwrap_err()
        );
    }
}

mod composite_unique_tests {
    use super::*;

    #[test]
    fn prepared_insert_validates_composite_unique_constraint() {
        let (_dir, db) = create_test_db();

        db.execute(
            "CREATE TABLE reservations (
                id INTEGER PRIMARY KEY,
                user_id INTEGER,
                resource_id INTEGER,
                date TEXT,
                UNIQUE (user_id, resource_id)
            )",
        )
        .expect("Failed to create table");

        db.execute("INSERT INTO reservations VALUES (1, 10, 100, '2024-01-01')")
            .expect("Initial insert should succeed");

        let stmt = db
            .prepare("INSERT INTO reservations VALUES (?, ?, ?, ?)")
            .expect("Failed to prepare statement");

        let result = stmt
            .bind(OwnedValue::Int(2))
            .bind(OwnedValue::Int(10))
            .bind(OwnedValue::Int(100))
            .bind(OwnedValue::Text("2024-02-01".to_string()))
            .execute(&db);

        assert!(
            result.is_err(),
            "Prepared INSERT with duplicate composite UNIQUE should fail, but got: {:?}",
            result
        );
    }

    #[test]
    fn prepared_insert_allows_partial_composite_unique_match() {
        let (_dir, db) = create_test_db();

        db.execute(
            "CREATE TABLE bookings (
                id INTEGER PRIMARY KEY,
                room_id INTEGER,
                date TEXT,
                UNIQUE (room_id, date)
            )",
        )
        .expect("Failed to create table");

        db.execute("INSERT INTO bookings VALUES (1, 100, '2024-01-01')")
            .expect("Initial insert should succeed");

        let stmt = db
            .prepare("INSERT INTO bookings VALUES (?, ?, ?)")
            .expect("Failed to prepare statement");

        let result = stmt
            .bind(OwnedValue::Int(2))
            .bind(OwnedValue::Int(100))
            .bind(OwnedValue::Text("2024-01-02".to_string()))
            .execute(&db);

        assert!(
            result.is_ok(),
            "Prepared INSERT with different composite UNIQUE key should succeed, but got: {:?}",
            result.unwrap_err()
        );
    }
}
