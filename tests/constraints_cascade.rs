//! # Constraints and CASCADE Operations Test Suite
//!
//! This module provides comprehensive tests for TurDB's constraint enforcement
//! and referential action handling (CASCADE, RESTRICT, SET NULL, etc.).
//!
//! ## Test Categories
//!
//! 1. **Schema Setup**: Create tables with various constraints
//! 2. **Index Tests**: Create and verify secondary indexes
//! 3. **INSERT Tests**: Constraint validation during inserts
//! 4. **UPDATE Tests**: Constraint validation and CASCADE during updates
//! 5. **DELETE Tests**: Constraint validation and CASCADE during deletes
//! 6. **Complex Scenarios**: Multi-table relationships, chained cascades
//!
//! ## Usage
//!
//! ```sh
//! cargo test --test constraints_cascade --release -- --nocapture
//! ```

use tempfile::tempdir;
use turdb::{Database, OwnedValue};

// ============================================================================
// HELPER FUNCTIONS
// ============================================================================

fn create_test_db() -> (tempfile::TempDir, Database) {
    let dir = tempdir().expect("Failed to create temp dir");
    let db = Database::create(dir.path().join("test_db")).expect("Failed to create database");
    (dir, db)
}

fn get_int(row: &turdb::Row, idx: usize) -> i64 {
    match row.get(idx) {
        Some(OwnedValue::Int(v)) => *v,
        other => panic!("Expected Int at index {}, got {:?}", idx, other),
    }
}

fn get_text(row: &turdb::Row, idx: usize) -> String {
    match row.get(idx) {
        Some(OwnedValue::Text(s)) => s.clone(),
        other => panic!("Expected Text at index {}, got {:?}", idx, other),
    }
}

// ============================================================================
// SCHEMA SETUP TESTS
// ============================================================================

mod schema_tests {
    use super::*;

    #[test]
    fn create_tables_with_all_constraint_types() {
        let (_dir, db) = create_test_db();

        // Parent table with PRIMARY KEY and UNIQUE constraints
        db.execute(
            "CREATE TABLE departments (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL UNIQUE,
                budget REAL CHECK(budget >= 0)
            )",
        )
        .expect("Failed to create departments table");

        // Child table with FOREIGN KEY constraints
        db.execute(
            "CREATE TABLE employees (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                email TEXT UNIQUE,
                dept_id INTEGER REFERENCES departments(id) ON DELETE CASCADE ON UPDATE CASCADE,
                salary REAL CHECK(salary > 0)
            )",
        )
        .expect("Failed to create employees table");

        // Grandchild table for chained CASCADE testing
        db.execute(
            "CREATE TABLE projects (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                lead_id INTEGER REFERENCES employees(id) ON DELETE SET NULL ON UPDATE CASCADE,
                budget REAL
            )",
        )
        .expect("Failed to create projects table");

        // Verify tables exist
        let dept_rows = db.query("SELECT * FROM departments").unwrap();
        assert_eq!(dept_rows.len(), 0, "departments should be empty");

        let emp_rows = db.query("SELECT * FROM employees").unwrap();
        assert_eq!(emp_rows.len(), 0, "employees should be empty");

        let proj_rows = db.query("SELECT * FROM projects").unwrap();
        assert_eq!(proj_rows.len(), 0, "projects should be empty");
    }

    #[test]
    fn create_secondary_indexes() {
        let (_dir, db) = create_test_db();

        db.execute(
            "CREATE TABLE products (
                id INTEGER PRIMARY KEY,
                sku TEXT NOT NULL,
                name TEXT,
                category TEXT,
                price REAL,
                stock INTEGER
            )",
        )
        .unwrap();

        // Create various indexes
        db.execute("CREATE INDEX idx_products_category ON products(category)")
            .expect("Failed to create category index");

        db.execute("CREATE UNIQUE INDEX idx_products_sku ON products(sku)")
            .expect("Failed to create unique SKU index");

        db.execute("CREATE INDEX idx_products_price_stock ON products(price, stock)")
            .expect("Failed to create composite index");

        // Insert test data
        db.execute("INSERT INTO products VALUES (1, 'SKU001', 'Widget', 'Electronics', 29.99, 100)")
            .unwrap();
        db.execute("INSERT INTO products VALUES (2, 'SKU002', 'Gadget', 'Electronics', 49.99, 50)")
            .unwrap();
        db.execute("INSERT INTO products VALUES (3, 'SKU003', 'Tool', 'Hardware', 19.99, 200)")
            .unwrap();

        // Query using indexes
        let electronics = db
            .query("SELECT * FROM products WHERE category = 'Electronics'")
            .unwrap();
        assert_eq!(electronics.len(), 2, "Should find 2 electronics products");

        let by_sku = db
            .query("SELECT * FROM products WHERE sku = 'SKU002'")
            .unwrap();
        assert_eq!(by_sku.len(), 1, "Should find exactly 1 product by SKU");
    }
}

// ============================================================================
// INSERT CONSTRAINT TESTS
// ============================================================================

mod insert_tests {
    use super::*;

    #[test]
    fn insert_with_primary_key_constraint() {
        let (_dir, db) = create_test_db();

        db.execute("CREATE TABLE items (id INTEGER PRIMARY KEY, name TEXT)")
            .unwrap();

        db.execute("INSERT INTO items VALUES (1, 'First')").unwrap();
        db.execute("INSERT INTO items VALUES (2, 'Second')").unwrap();

        // Duplicate PK should fail
        let result = db.execute("INSERT INTO items VALUES (1, 'Duplicate')");
        assert!(result.is_err(), "Duplicate PRIMARY KEY should be rejected");
    }

    #[test]
    fn insert_with_unique_constraint() {
        let (_dir, db) = create_test_db();

        db.execute("CREATE TABLE users (id INTEGER PRIMARY KEY, email TEXT UNIQUE)")
            .unwrap();

        db.execute("INSERT INTO users VALUES (1, 'alice@test.com')")
            .unwrap();
        db.execute("INSERT INTO users VALUES (2, 'bob@test.com')")
            .unwrap();

        // Duplicate email should fail
        let result = db.execute("INSERT INTO users VALUES (3, 'alice@test.com')");
        assert!(result.is_err(), "Duplicate UNIQUE value should be rejected");

        // NULL values should be allowed multiple times
        db.execute("INSERT INTO users VALUES (4, NULL)").unwrap();
        db.execute("INSERT INTO users VALUES (5, NULL)").unwrap();

        // Verify we have 4 users total (2 with emails, 2 with NULL)
        let all_rows = db.query("SELECT * FROM users").unwrap();
        assert_eq!(all_rows.len(), 4, "Should have 4 users total");
    }

    #[test]
    fn insert_with_not_null_constraint() {
        let (_dir, db) = create_test_db();

        db.execute("CREATE TABLE contacts (id INTEGER PRIMARY KEY, name TEXT NOT NULL)")
            .unwrap();

        db.execute("INSERT INTO contacts VALUES (1, 'John')").unwrap();

        // NULL name should fail
        let result = db.execute("INSERT INTO contacts VALUES (2, NULL)");
        assert!(result.is_err(), "NULL in NOT NULL column should be rejected");
    }

    #[test]
    fn insert_with_check_constraint() {
        let (_dir, db) = create_test_db();

        db.execute(
            "CREATE TABLE accounts (
                id INTEGER PRIMARY KEY,
                balance REAL CHECK(balance >= 0)
            )",
        )
        .unwrap();

        db.execute("INSERT INTO accounts VALUES (1, 100.0)").unwrap();
        db.execute("INSERT INTO accounts VALUES (2, 0.0)").unwrap();

        // Negative balance should fail
        let result = db.execute("INSERT INTO accounts VALUES (3, -50.0)");
        assert!(result.is_err(), "CHECK constraint violation should be rejected");
    }

    #[test]
    fn insert_with_foreign_key_constraint() {
        let (_dir, db) = create_test_db();

        db.execute("CREATE TABLE categories (id INTEGER PRIMARY KEY, name TEXT)")
            .unwrap();
        db.execute(
            "CREATE TABLE products (
                id INTEGER PRIMARY KEY,
                name TEXT,
                category_id INTEGER REFERENCES categories(id)
            )",
        )
        .unwrap();

        db.execute("INSERT INTO categories VALUES (1, 'Electronics')")
            .unwrap();
        db.execute("INSERT INTO categories VALUES (2, 'Books')")
            .unwrap();

        // Valid FK reference
        db.execute("INSERT INTO products VALUES (1, 'Laptop', 1)")
            .unwrap();

        // Invalid FK reference should fail
        let result = db.execute("INSERT INTO products VALUES (2, 'Widget', 999)");
        assert!(result.is_err(), "Invalid FOREIGN KEY reference should be rejected");

        // NULL FK is allowed
        db.execute("INSERT INTO products VALUES (3, 'Unknown', NULL)")
            .unwrap();
    }
}

// ============================================================================
// UPDATE CONSTRAINT TESTS
// ============================================================================

mod update_tests {
    use super::*;

    #[test]
    fn update_with_unique_constraint() {
        let (_dir, db) = create_test_db();

        db.execute("CREATE TABLE users (id INTEGER PRIMARY KEY, email TEXT UNIQUE)")
            .unwrap();

        db.execute("INSERT INTO users VALUES (1, 'alice@test.com')")
            .unwrap();
        db.execute("INSERT INTO users VALUES (2, 'bob@test.com')")
            .unwrap();

        // Update to unique value should succeed
        db.execute("UPDATE users SET email = 'alice.new@test.com' WHERE id = 1")
            .unwrap();

        // Update to existing value should fail
        let result = db.execute("UPDATE users SET email = 'bob@test.com' WHERE id = 1");
        assert!(
            result.is_err(),
            "UPDATE to duplicate UNIQUE value should be rejected"
        );
    }

    #[test]
    fn update_with_check_constraint() {
        let (_dir, db) = create_test_db();

        db.execute(
            "CREATE TABLE accounts (
                id INTEGER PRIMARY KEY,
                balance REAL CHECK(balance >= 0)
            )",
        )
        .unwrap();

        db.execute("INSERT INTO accounts VALUES (1, 100.0)").unwrap();

        // Valid update
        db.execute("UPDATE accounts SET balance = 50.0 WHERE id = 1")
            .unwrap();

        // Invalid update should fail
        let result = db.execute("UPDATE accounts SET balance = -10.0 WHERE id = 1");
        assert!(
            result.is_err(),
            "UPDATE violating CHECK constraint should be rejected"
        );

        // Verify balance unchanged
        let rows = db.query("SELECT balance FROM accounts WHERE id = 1").unwrap();
        match rows[0].get(0) {
            Some(OwnedValue::Float(f)) => assert!((f - 50.0).abs() < 0.001),
            other => panic!("Expected Float, got {:?}", other),
        }
    }

    #[test]
    fn update_with_foreign_key_constraint() {
        let (_dir, db) = create_test_db();

        db.execute("CREATE TABLE departments (id INTEGER PRIMARY KEY, name TEXT)")
            .unwrap();
        db.execute(
            "CREATE TABLE employees (
                id INTEGER PRIMARY KEY,
                dept_id INTEGER REFERENCES departments(id)
            )",
        )
        .unwrap();

        db.execute("INSERT INTO departments VALUES (1, 'Engineering')")
            .unwrap();
        db.execute("INSERT INTO departments VALUES (2, 'Sales')")
            .unwrap();
        db.execute("INSERT INTO employees VALUES (1, 1)").unwrap();

        // Valid FK update
        db.execute("UPDATE employees SET dept_id = 2 WHERE id = 1")
            .unwrap();

        // Verify update worked
        let rows = db
            .query("SELECT dept_id FROM employees WHERE id = 1")
            .unwrap();
        assert_eq!(get_int(&rows[0], 0), 2, "dept_id should be updated to 2");
    }
}

// ============================================================================
// DELETE CONSTRAINT TESTS
// ============================================================================

mod delete_tests {
    use super::*;

    #[test]
    fn delete_with_foreign_key_restrict() {
        let (_dir, db) = create_test_db();

        db.execute("CREATE TABLE departments (id INTEGER PRIMARY KEY, name TEXT)")
            .unwrap();
        db.execute(
            "CREATE TABLE employees (
                id INTEGER PRIMARY KEY,
                dept_id INTEGER REFERENCES departments(id)
            )",
        )
        .unwrap();

        db.execute("INSERT INTO departments VALUES (1, 'Engineering')")
            .unwrap();
        db.execute("INSERT INTO employees VALUES (1, 1)").unwrap();

        // Delete referenced row should fail (default is RESTRICT/NO ACTION)
        let result = db.execute("DELETE FROM departments WHERE id = 1");
        assert!(
            result.is_err(),
            "DELETE of referenced row should be rejected with RESTRICT"
        );

        // Delete non-referenced row should succeed
        db.execute("INSERT INTO departments VALUES (2, 'Marketing')")
            .unwrap();
        db.execute("DELETE FROM departments WHERE id = 2")
            .expect("DELETE of non-referenced row should succeed");
    }
}

// ============================================================================
// CASCADE TESTS
// ============================================================================

mod cascade_tests {
    use super::*;

    #[test]
    fn on_delete_cascade_single_level() {
        let (_dir, db) = create_test_db();

        db.execute("CREATE TABLE departments (id INTEGER PRIMARY KEY, name TEXT)")
            .unwrap();
        db.execute(
            "CREATE TABLE employees (
                id INTEGER PRIMARY KEY,
                name TEXT,
                dept_id INTEGER REFERENCES departments(id) ON DELETE CASCADE
            )",
        )
        .unwrap();

        // Setup data
        db.execute("INSERT INTO departments VALUES (1, 'Engineering')")
            .unwrap();
        db.execute("INSERT INTO departments VALUES (2, 'Sales')")
            .unwrap();
        db.execute("INSERT INTO employees VALUES (100, 'Alice', 1)")
            .unwrap();
        db.execute("INSERT INTO employees VALUES (101, 'Bob', 1)")
            .unwrap();
        db.execute("INSERT INTO employees VALUES (102, 'Charlie', 2)")
            .unwrap();

        // Verify initial state
        let employees = db.query("SELECT * FROM employees").unwrap();
        assert_eq!(employees.len(), 3, "Should have 3 employees initially");

        // Delete parent row - should cascade to children
        let result = db.execute("DELETE FROM departments WHERE id = 1");
        assert!(result.is_ok(), "ON DELETE CASCADE should allow deleting parent");

        // Verify cascade
        let remaining = db.query("SELECT * FROM employees").unwrap();
        assert_eq!(
            remaining.len(),
            1,
            "Should have 1 employee after cascade delete"
        );
        assert_eq!(get_int(&remaining[0], 0), 102, "Charlie should remain");
    }

    #[test]
    fn on_update_cascade_single_level() {
        let (_dir, db) = create_test_db();

        db.execute("CREATE TABLE departments (id INTEGER PRIMARY KEY, name TEXT)")
            .unwrap();
        db.execute(
            "CREATE TABLE employees (
                id INTEGER PRIMARY KEY,
                name TEXT,
                dept_id INTEGER REFERENCES departments(id) ON UPDATE CASCADE
            )",
        )
        .unwrap();

        // Setup data
        db.execute("INSERT INTO departments VALUES (1, 'Engineering')")
            .unwrap();
        db.execute("INSERT INTO employees VALUES (100, 'Alice', 1)")
            .unwrap();
        db.execute("INSERT INTO employees VALUES (101, 'Bob', 1)")
            .unwrap();

        // Update parent PK - should cascade to children
        let result = db.execute("UPDATE departments SET id = 10 WHERE id = 1");
        assert!(
            result.is_ok(),
            "ON UPDATE CASCADE should allow updating parent PK"
        );

        // Verify cascade
        let with_new_dept = db
            .query("SELECT * FROM employees WHERE dept_id = 10")
            .unwrap();
        assert_eq!(
            with_new_dept.len(),
            2,
            "Both employees should have dept_id = 10"
        );

        let with_old_dept = db
            .query("SELECT * FROM employees WHERE dept_id = 1")
            .unwrap();
        assert_eq!(
            with_old_dept.len(),
            0,
            "No employees should have old dept_id = 1"
        );
    }

    #[test]
    fn on_update_restrict_blocks_update() {
        let (_dir, db) = create_test_db();

        db.execute("CREATE TABLE categories (id INTEGER PRIMARY KEY, name TEXT)")
            .unwrap();
        db.execute(
            "CREATE TABLE products (
                id INTEGER PRIMARY KEY,
                category_id INTEGER REFERENCES categories(id) ON UPDATE RESTRICT
            )",
        )
        .unwrap();

        db.execute("INSERT INTO categories VALUES (1, 'Electronics')")
            .unwrap();
        db.execute("INSERT INTO products VALUES (1, 1)").unwrap();

        // Update of referenced PK should fail
        let result = db.execute("UPDATE categories SET id = 100 WHERE id = 1");
        assert!(
            result.is_err(),
            "ON UPDATE RESTRICT should block update of referenced PK"
        );
    }

    #[test]
    fn cascade_with_multiple_children() {
        let (_dir, db) = create_test_db();

        db.execute("CREATE TABLE customers (id INTEGER PRIMARY KEY, name TEXT)")
            .unwrap();
        db.execute(
            "CREATE TABLE orders (
                id INTEGER PRIMARY KEY,
                customer_id INTEGER REFERENCES customers(id) ON DELETE CASCADE ON UPDATE CASCADE,
                total REAL
            )",
        )
        .unwrap();
        db.execute(
            "CREATE TABLE reviews (
                id INTEGER PRIMARY KEY,
                customer_id INTEGER REFERENCES customers(id) ON DELETE CASCADE ON UPDATE CASCADE,
                rating INTEGER
            )",
        )
        .unwrap();

        // Setup data
        db.execute("INSERT INTO customers VALUES (1, 'John')").unwrap();
        db.execute("INSERT INTO orders VALUES (1, 1, 99.99)").unwrap();
        db.execute("INSERT INTO orders VALUES (2, 1, 149.99)").unwrap();
        db.execute("INSERT INTO reviews VALUES (1, 1, 5)").unwrap();

        // Update customer ID - should cascade to both orders and reviews
        db.execute("UPDATE customers SET id = 100 WHERE id = 1")
            .unwrap();

        let orders = db
            .query("SELECT * FROM orders WHERE customer_id = 100")
            .unwrap();
        assert_eq!(orders.len(), 2, "Both orders should have customer_id = 100");

        let reviews = db
            .query("SELECT * FROM reviews WHERE customer_id = 100")
            .unwrap();
        assert_eq!(reviews.len(), 1, "Review should have customer_id = 100");

        // Delete customer - should cascade to both orders and reviews
        db.execute("DELETE FROM customers WHERE id = 100").unwrap();

        let remaining_orders = db.query("SELECT * FROM orders").unwrap();
        assert_eq!(remaining_orders.len(), 0, "All orders should be deleted");

        let remaining_reviews = db.query("SELECT * FROM reviews").unwrap();
        assert_eq!(remaining_reviews.len(), 0, "All reviews should be deleted");
    }
}

// ============================================================================
// COMPLEX SCENARIO TESTS
// ============================================================================

mod complex_tests {
    use super::*;

    #[test]
    fn full_crud_with_constraints() {
        let (_dir, db) = create_test_db();

        // Create schema with multiple constraints
        db.execute(
            "CREATE TABLE categories (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL UNIQUE,
                description TEXT
            )",
        )
        .unwrap();

        db.execute(
            "CREATE TABLE products (
                id INTEGER PRIMARY KEY,
                sku TEXT NOT NULL UNIQUE,
                name TEXT NOT NULL,
                category_id INTEGER REFERENCES categories(id) ON DELETE CASCADE,
                price REAL CHECK(price > 0),
                stock INTEGER CHECK(stock >= 0)
            )",
        )
        .unwrap();

        db.execute("CREATE INDEX idx_products_category ON products(category_id)")
            .unwrap();

        // INSERT test data
        db.execute("INSERT INTO categories VALUES (1, 'Electronics', 'Electronic devices')")
            .unwrap();
        db.execute("INSERT INTO categories VALUES (2, 'Books', 'Reading materials')")
            .unwrap();

        db.execute("INSERT INTO products VALUES (1, 'ELEC-001', 'Laptop', 1, 999.99, 50)")
            .unwrap();
        db.execute("INSERT INTO products VALUES (2, 'ELEC-002', 'Phone', 1, 599.99, 100)")
            .unwrap();
        db.execute("INSERT INTO products VALUES (3, 'BOOK-001', 'Rust Book', 2, 49.99, 200)")
            .unwrap();

        // Verify initial data
        let all_products = db.query("SELECT * FROM products").unwrap();
        assert_eq!(all_products.len(), 3, "Should have 3 products");

        // UPDATE tests
        db.execute("UPDATE products SET price = 899.99 WHERE sku = 'ELEC-001'")
            .unwrap();

        let updated = db
            .query("SELECT price FROM products WHERE sku = 'ELEC-001'")
            .unwrap();
        match updated[0].get(0) {
            Some(OwnedValue::Float(f)) => assert!((f - 899.99).abs() < 0.001),
            other => panic!("Expected Float, got {:?}", other),
        }

        // Test CASCADE DELETE
        db.execute("DELETE FROM categories WHERE id = 1").unwrap();

        let remaining = db.query("SELECT * FROM products").unwrap();
        assert_eq!(
            remaining.len(),
            1,
            "Only 1 product (Rust Book) should remain after cascade delete"
        );

        // Verify the remaining product
        assert_eq!(get_text(&remaining[0], 1), "BOOK-001");

        // DELETE remaining data
        db.execute("DELETE FROM products WHERE id = 3").unwrap();
        db.execute("DELETE FROM categories WHERE id = 2").unwrap();

        let final_products = db.query("SELECT * FROM products").unwrap();
        assert_eq!(final_products.len(), 0, "All products should be deleted");

        let final_categories = db.query("SELECT * FROM categories").unwrap();
        assert_eq!(final_categories.len(), 0, "All categories should be deleted");
    }

    #[test]
    fn transaction_with_constraints() {
        let (_dir, db) = create_test_db();

        db.execute("CREATE TABLE accounts (id INTEGER PRIMARY KEY, balance REAL CHECK(balance >= 0))")
            .unwrap();

        db.execute("INSERT INTO accounts VALUES (1, 1000.0)").unwrap();
        db.execute("INSERT INTO accounts VALUES (2, 500.0)").unwrap();

        // Start transaction
        db.execute("BEGIN").unwrap();

        // Transfer money (using literal values since expressions not supported)
        db.execute("UPDATE accounts SET balance = 800.0 WHERE id = 1")
            .unwrap();
        db.execute("UPDATE accounts SET balance = 700.0 WHERE id = 2")
            .unwrap();

        // Commit
        db.execute("COMMIT").unwrap();

        // Verify balances
        let acc1 = db
            .query("SELECT balance FROM accounts WHERE id = 1")
            .unwrap();
        let acc2 = db
            .query("SELECT balance FROM accounts WHERE id = 2")
            .unwrap();

        match (acc1[0].get(0), acc2[0].get(0)) {
            (Some(OwnedValue::Float(b1)), Some(OwnedValue::Float(b2))) => {
                assert!((b1 - 800.0).abs() < 0.001, "Account 1 should have 800.0");
                assert!((b2 - 700.0).abs() < 0.001, "Account 2 should have 700.0");
            }
            other => panic!("Expected Float values, got {:?}", other),
        }
    }

    #[test]
    fn bulk_operations_with_cascade() {
        let (_dir, db) = create_test_db();

        db.execute("CREATE TABLE groups (id INTEGER PRIMARY KEY, name TEXT)")
            .unwrap();
        db.execute(
            "CREATE TABLE members (
                id INTEGER PRIMARY KEY,
                name TEXT,
                group_id INTEGER REFERENCES groups(id) ON DELETE CASCADE
            )",
        )
        .unwrap();

        // Bulk insert
        db.execute("BEGIN").unwrap();
        for i in 1..=5 {
            db.execute(&format!("INSERT INTO groups VALUES ({}, 'Group {}')", i, i))
                .unwrap();
            for j in 1..=10 {
                let member_id = (i - 1) * 10 + j;
                db.execute(&format!(
                    "INSERT INTO members VALUES ({}, 'Member {}', {})",
                    member_id, member_id, i
                ))
                .unwrap();
            }
        }
        db.execute("COMMIT").unwrap();

        // Verify counts
        let groups = db.query("SELECT * FROM groups").unwrap();
        assert_eq!(groups.len(), 5, "Should have 5 groups");

        let members = db.query("SELECT * FROM members").unwrap();
        assert_eq!(members.len(), 50, "Should have 50 members (10 per group)");

        // Delete a group - should cascade delete 10 members
        db.execute("DELETE FROM groups WHERE id = 3").unwrap();

        let remaining_members = db.query("SELECT * FROM members").unwrap();
        assert_eq!(
            remaining_members.len(),
            40,
            "Should have 40 members after cascade delete"
        );

        let group3_members = db
            .query("SELECT * FROM members WHERE group_id = 3")
            .unwrap();
        assert_eq!(
            group3_members.len(),
            0,
            "No members should belong to deleted group 3"
        );
    }
}

// ============================================================================
// INDEX USAGE TESTS
// ============================================================================

mod index_tests {
    use super::*;

    #[test]
    fn index_used_for_equality_lookup() {
        let (_dir, db) = create_test_db();

        db.execute(
            "CREATE TABLE orders (
                id INTEGER PRIMARY KEY,
                customer_id INTEGER,
                status TEXT,
                total REAL
            )",
        )
        .unwrap();

        db.execute("CREATE INDEX idx_orders_customer ON orders(customer_id)")
            .unwrap();
        db.execute("CREATE INDEX idx_orders_status ON orders(status)")
            .unwrap();

        // Insert test data
        for i in 1..=100 {
            let customer_id = (i % 10) + 1;
            let status = if i % 3 == 0 {
                "completed"
            } else if i % 3 == 1 {
                "pending"
            } else {
                "shipped"
            };
            db.execute(&format!(
                "INSERT INTO orders VALUES ({}, {}, '{}', {})",
                i,
                customer_id,
                status,
                i as f64 * 10.5
            ))
            .unwrap();
        }

        // Query by indexed column
        let customer_orders = db
            .query("SELECT * FROM orders WHERE customer_id = 5")
            .unwrap();
        assert_eq!(
            customer_orders.len(),
            10,
            "Should find 10 orders for customer 5"
        );

        let pending = db
            .query("SELECT * FROM orders WHERE status = 'pending'")
            .unwrap();
        assert_eq!(pending.len(), 34, "Should find ~34 pending orders");

        let completed = db
            .query("SELECT * FROM orders WHERE status = 'completed'")
            .unwrap();
        assert_eq!(completed.len(), 33, "Should find ~33 completed orders");
    }

    #[test]
    fn unique_index_enforcement() {
        let (_dir, db) = create_test_db();

        db.execute(
            "CREATE TABLE inventory (
                id INTEGER PRIMARY KEY,
                sku TEXT,
                quantity INTEGER
            )",
        )
        .unwrap();

        db.execute("CREATE UNIQUE INDEX idx_inventory_sku ON inventory(sku)")
            .unwrap();

        db.execute("INSERT INTO inventory VALUES (1, 'WIDGET-A', 100)")
            .unwrap();
        db.execute("INSERT INTO inventory VALUES (2, 'WIDGET-B', 50)")
            .unwrap();

        // Duplicate SKU should fail due to unique index
        let result = db.execute("INSERT INTO inventory VALUES (3, 'WIDGET-A', 75)");
        assert!(
            result.is_err(),
            "Duplicate value in UNIQUE index should be rejected"
        );

        // Valid update should succeed
        db.execute("UPDATE inventory SET sku = 'WIDGET-C' WHERE id = 2")
            .unwrap();

        let rows = db
            .query("SELECT sku FROM inventory WHERE id = 2")
            .unwrap();
        assert_eq!(get_text(&rows[0], 0), "WIDGET-C", "SKU should be updated");
    }
}
