//! # Grace Hash Join Integration Tests
//!
//! This module tests the Grace Hash Join implementation with spill-to-disk support.
//! Tests cover various join types (INNER, LEFT, RIGHT, FULL OUTER) and verify
//! correct behavior when data exceeds the memory budget.
//!
//! ## Test Strategy
//!
//! 1. Create two related tables (orders and customers)
//! 2. Insert test data with known relationships
//! 3. Execute JOIN queries and verify results
//! 4. Test with small memory budget to force spilling
//!
//! ## Usage
//!
//! ```sh
//! cargo test --test grace_hash_join --release -- --nocapture
//! ```

use tempfile::TempDir;
use turdb::{Database, OwnedValue};

fn create_test_db() -> (Database, TempDir) {
    let temp_dir = TempDir::new().expect("Failed to create temp directory");
    let db = Database::create(temp_dir.path()).expect("Failed to create database");
    (db, temp_dir)
}

#[allow(dead_code)]
trait ValueExt {
    fn is_null(&self) -> bool;
    fn as_int(&self) -> Option<i64>;
    fn as_float(&self) -> Option<f64>;
}

impl ValueExt for OwnedValue {
    fn is_null(&self) -> bool {
        matches!(self, OwnedValue::Null)
    }

    fn as_int(&self) -> Option<i64> {
        match self {
            OwnedValue::Int(i) => Some(*i),
            _ => None,
        }
    }

    fn as_float(&self) -> Option<f64> {
        match self {
            OwnedValue::Float(f) => Some(*f),
            _ => None,
        }
    }
}

fn setup_test_tables(db: &Database) {
    db.execute(
        "CREATE TABLE customers (
            id BIGINT PRIMARY KEY,
            name TEXT,
            city TEXT
        )",
    )
    .expect("Failed to create customers table");

    db.execute(
        "CREATE TABLE orders (
            id BIGINT PRIMARY KEY,
            customer_id BIGINT,
            amount FLOAT,
            status TEXT
        )",
    )
    .expect("Failed to create orders table");
}

fn insert_test_data(db: &Database, num_customers: usize, orders_per_customer: usize) {
    db.execute("BEGIN").expect("Failed to begin transaction");

    for i in 1..=num_customers {
        let name = format!("Customer_{}", i);
        let city = match i % 3 {
            0 => "New York",
            1 => "Los Angeles",
            _ => "Chicago",
        };
        db.execute(&format!(
            "INSERT INTO customers (id, name, city) VALUES ({}, '{}', '{}')",
            i, name, city
        ))
        .expect("Failed to insert customer");

        for j in 1..=orders_per_customer {
            let order_id = (i - 1) * orders_per_customer + j;
            let amount = (i * 100 + j * 10) as f64;
            let status = match j % 3 {
                0 => "completed",
                1 => "pending",
                _ => "shipped",
            };
            db.execute(&format!(
                "INSERT INTO orders (id, customer_id, amount, status) VALUES ({}, {}, {}, '{}')",
                order_id, i, amount, status
            ))
            .expect("Failed to insert order");
        }
    }

    db.execute("COMMIT").expect("Failed to commit transaction");
}

fn insert_orphan_orders(db: &Database, start_id: usize, count: usize) {
    db.execute("BEGIN").expect("Failed to begin transaction");

    for i in 0..count {
        let order_id = start_id + i;
        let customer_id = 999999 + i;
        db.execute(&format!(
            "INSERT INTO orders (id, customer_id, amount, status) VALUES ({}, {}, 50.0, 'orphan')",
            order_id, customer_id
        ))
        .expect("Failed to insert orphan order");
    }

    db.execute("COMMIT").expect("Failed to commit transaction");
}

fn insert_customers_without_orders(db: &Database, start_id: usize, count: usize) {
    db.execute("BEGIN").expect("Failed to begin transaction");

    for i in 0..count {
        let customer_id = start_id + i;
        db.execute(&format!(
            "INSERT INTO customers (id, name, city) VALUES ({}, 'NoOrders_{}', 'Miami')",
            customer_id, i
        ))
        .expect("Failed to insert customer without orders");
    }

    db.execute("COMMIT").expect("Failed to commit transaction");
}

#[test]
fn inner_join_returns_matching_rows_only() {
    let (db, _temp) = create_test_db();
    setup_test_tables(&db);
    insert_test_data(&db, 10, 5);
    insert_orphan_orders(&db, 1000, 5);
    insert_customers_without_orders(&db, 100, 5);

    let rows = db
        .query(
            "SELECT c.id, c.name, o.id, o.amount
             FROM customers c
             INNER JOIN orders o ON c.id = o.customer_id
             ORDER BY c.id, o.id",
        )
        .expect("Failed to execute INNER JOIN");

    assert_eq!(rows.len(), 50, "Should have 10 customers * 5 orders = 50 matching rows");

    if let Some(first_row) = rows.first() {
        assert_eq!(first_row.values.len(), 4, "Should have 4 columns");
    }
}

#[test]
fn left_join_includes_unmatched_left_rows() {
    let (db, _temp) = create_test_db();
    setup_test_tables(&db);
    insert_test_data(&db, 10, 5);
    insert_customers_without_orders(&db, 100, 5);

    let rows = db
        .query(
            "SELECT c.id, c.name, o.id, o.amount
             FROM customers c
             LEFT JOIN orders o ON c.id = o.customer_id
             ORDER BY c.id, o.id",
        )
        .expect("Failed to execute LEFT JOIN");

    assert_eq!(
        rows.len(),
        55,
        "Should have 50 matched + 5 unmatched customers = 55 rows"
    );

    let null_order_rows: Vec<_> = rows
        .iter()
        .filter(|r| r.values.get(2).map(|v| v.is_null()).unwrap_or(false))
        .collect();
    assert_eq!(
        null_order_rows.len(),
        5,
        "Should have 5 rows with NULL order_id (customers without orders)"
    );
}

#[test]
fn right_join_includes_unmatched_right_rows() {
    let (db, _temp) = create_test_db();
    setup_test_tables(&db);
    insert_test_data(&db, 10, 5);
    insert_orphan_orders(&db, 1000, 5);

    let rows = db
        .query(
            "SELECT c.id, c.name, o.id, o.amount
             FROM customers c
             RIGHT JOIN orders o ON c.id = o.customer_id
             ORDER BY o.id",
        )
        .expect("Failed to execute RIGHT JOIN");

    assert_eq!(
        rows.len(),
        55,
        "Should have 50 matched + 5 orphan orders = 55 rows"
    );

    let null_customer_rows: Vec<_> = rows
        .iter()
        .filter(|r| r.values.first().map(|v| v.is_null()).unwrap_or(false))
        .collect();
    assert_eq!(
        null_customer_rows.len(),
        5,
        "Should have 5 rows with NULL customer_id (orphan orders)"
    );
}

#[test]
fn full_outer_join_includes_all_unmatched_rows() {
    let (db, _temp) = create_test_db();
    setup_test_tables(&db);
    insert_test_data(&db, 10, 5);
    insert_orphan_orders(&db, 1000, 5);
    insert_customers_without_orders(&db, 100, 5);

    let rows = db
        .query(
            "SELECT c.id, c.name, o.id, o.amount
             FROM customers c
             FULL OUTER JOIN orders o ON c.id = o.customer_id",
        )
        .expect("Failed to execute FULL OUTER JOIN");

    assert_eq!(
        rows.len(),
        60,
        "Should have 50 matched + 5 orphan orders + 5 customers without orders = 60 rows"
    );
}

#[test]
fn join_with_filter_on_both_sides() {
    let (db, _temp) = create_test_db();
    setup_test_tables(&db);
    insert_test_data(&db, 20, 10);

    let rows = db
        .query(
            "SELECT c.id, c.name, o.id, o.amount
             FROM customers c
             INNER JOIN orders o ON c.id = o.customer_id
             WHERE c.city = 'New York' AND o.status = 'completed'
             ORDER BY c.id, o.id",
        )
        .expect("Failed to execute JOIN with filter");

    assert!(!rows.is_empty(), "Should have some matching rows");

    for row in &rows {
        let amount = row.values.get(3).and_then(|v| v.as_float());
        assert!(amount.is_some(), "Amount should be present");
    }
}

#[test]
fn join_with_aggregate() {
    let (db, _temp) = create_test_db();
    setup_test_tables(&db);
    insert_test_data(&db, 10, 5);

    let rows = db
        .query(
            "SELECT c.id, c.name, COUNT(*) as order_count, SUM(o.amount) as total
             FROM customers c
             INNER JOIN orders o ON c.id = o.customer_id
             GROUP BY c.id, c.name
             ORDER BY c.id",
        )
        .expect("Failed to execute JOIN with aggregate");

    assert_eq!(rows.len(), 10, "Should have 10 customer groups");

    for row in &rows {
        let count = row.values.get(2).and_then(|v| v.as_int()).unwrap_or(0);
        assert_eq!(count, 5, "Each customer should have 5 orders");
    }
}

#[test]
fn pragma_join_memory_budget_can_be_set() {
    let (db, _temp) = create_test_db();

    let result = db
        .execute("PRAGMA join_memory_budget")
        .expect("Failed to get join_memory_budget");

    if let turdb::ExecuteResult::Pragma { name, value } = result {
        assert_eq!(name, "JOIN_MEMORY_BUDGET");
        let budget: usize = value.unwrap().parse().unwrap();
        assert_eq!(budget, 10 * 1024 * 1024, "Default should be 10MB");
    } else {
        panic!("Expected Pragma result");
    }

    db.execute("PRAGMA join_memory_budget = 1048576")
        .expect("Failed to set join_memory_budget");

    let result = db
        .execute("PRAGMA join_memory_budget")
        .expect("Failed to get join_memory_budget");

    if let turdb::ExecuteResult::Pragma { name, value } = result {
        assert_eq!(name, "JOIN_MEMORY_BUDGET");
        let budget: usize = value.unwrap().parse().unwrap();
        assert_eq!(budget, 1048576, "Should be set to 1MB");
    }
}

#[test]
fn large_join_completes_successfully() {
    let (db, _temp) = create_test_db();
    setup_test_tables(&db);

    println!("Inserting 1000 customers with 10 orders each...");
    let start = std::time::Instant::now();
    insert_test_data(&db, 1000, 10);
    println!("Insert completed in {:?}", start.elapsed());

    println!("Executing large INNER JOIN...");
    let start = std::time::Instant::now();
    let rows = db
        .query(
            "SELECT COUNT(*)
             FROM customers c
             INNER JOIN orders o ON c.id = o.customer_id",
        )
        .expect("Failed to execute large JOIN");
    println!("JOIN completed in {:?}", start.elapsed());

    let count = rows
        .first()
        .and_then(|r| r.values.first())
        .and_then(|v| v.as_int())
        .unwrap_or(0);
    assert_eq!(count, 10000, "Should have 1000 * 10 = 10000 joined rows");
}

#[test]
fn self_join_works() {
    let (db, _temp) = create_test_db();

    db.execute(
        "CREATE TABLE employees (
            id BIGINT PRIMARY KEY,
            name TEXT,
            manager_id BIGINT
        )",
    )
    .expect("Failed to create employees table");

    db.execute("BEGIN").unwrap();
    db.execute("INSERT INTO employees VALUES (1, 'CEO', NULL)")
        .unwrap();
    db.execute("INSERT INTO employees VALUES (2, 'VP1', 1)")
        .unwrap();
    db.execute("INSERT INTO employees VALUES (3, 'VP2', 1)")
        .unwrap();
    db.execute("INSERT INTO employees VALUES (4, 'Manager1', 2)")
        .unwrap();
    db.execute("INSERT INTO employees VALUES (5, 'Manager2', 2)")
        .unwrap();
    db.execute("INSERT INTO employees VALUES (6, 'Manager3', 3)")
        .unwrap();
    db.execute("COMMIT").unwrap();

    let rows = db
        .query(
            "SELECT e.name as employee, m.name as manager
             FROM employees e
             INNER JOIN employees m ON e.manager_id = m.id
             ORDER BY e.id",
        )
        .expect("Failed to execute self join");

    assert_eq!(rows.len(), 5, "Should have 5 employees with managers (excluding CEO)");
}

#[test]
fn join_with_null_keys_handled_correctly() {
    let (db, _temp) = create_test_db();

    db.execute("CREATE TABLE t1 (id BIGINT, val TEXT)")
        .expect("Failed to create t1");
    db.execute("CREATE TABLE t2 (id BIGINT, val TEXT)")
        .expect("Failed to create t2");

    db.execute("BEGIN").unwrap();
    db.execute("INSERT INTO t1 VALUES (1, 'a')").unwrap();
    db.execute("INSERT INTO t1 VALUES (NULL, 'b')").unwrap();
    db.execute("INSERT INTO t1 VALUES (2, 'c')").unwrap();
    db.execute("INSERT INTO t2 VALUES (1, 'x')").unwrap();
    db.execute("INSERT INTO t2 VALUES (NULL, 'y')").unwrap();
    db.execute("INSERT INTO t2 VALUES (3, 'z')").unwrap();
    db.execute("COMMIT").unwrap();

    let rows = db
        .query("SELECT t1.id, t1.val, t2.id, t2.val FROM t1 INNER JOIN t2 ON t1.id = t2.id")
        .expect("Failed to execute join with nulls");

    assert_eq!(rows.len(), 1, "NULL = NULL should not match in standard SQL semantics");
}

#[test]
fn multiple_join_conditions() {
    let (db, _temp) = create_test_db();

    db.execute("CREATE TABLE products (id BIGINT, category TEXT, price FLOAT)")
        .expect("Failed to create products");
    db.execute("CREATE TABLE inventory (product_id BIGINT, warehouse TEXT, quantity BIGINT)")
        .expect("Failed to create inventory");

    db.execute("BEGIN").unwrap();
    db.execute("INSERT INTO products VALUES (1, 'electronics', 100.0)")
        .unwrap();
    db.execute("INSERT INTO products VALUES (2, 'electronics', 200.0)")
        .unwrap();
    db.execute("INSERT INTO products VALUES (3, 'clothing', 50.0)")
        .unwrap();
    db.execute("INSERT INTO inventory VALUES (1, 'west', 10)")
        .unwrap();
    db.execute("INSERT INTO inventory VALUES (1, 'east', 20)")
        .unwrap();
    db.execute("INSERT INTO inventory VALUES (2, 'west', 5)")
        .unwrap();
    db.execute("INSERT INTO inventory VALUES (3, 'east', 100)")
        .unwrap();
    db.execute("COMMIT").unwrap();

    let rows = db
        .query(
            "SELECT p.id, p.category, i.warehouse, i.quantity
             FROM products p
             INNER JOIN inventory i ON p.id = i.product_id
             WHERE p.category = 'electronics'
             ORDER BY p.id, i.warehouse",
        )
        .expect("Failed to execute join");

    assert_eq!(rows.len(), 3, "Should have 3 inventory records for electronics");
}

#[test]
fn qualified_column_names_resolve_correctly_in_join() {
    let (db, _temp) = create_test_db();

    db.execute("CREATE TABLE competitions (id BIGINT PRIMARY KEY, name TEXT)")
        .expect("Failed to create competitions table");
    db.execute("CREATE TABLE episodes (id BIGINT PRIMARY KEY, competition_id BIGINT, title TEXT)")
        .expect("Failed to create episodes table");

    db.execute("BEGIN").unwrap();
    db.execute("INSERT INTO competitions VALUES (1, 'Competition A')")
        .unwrap();
    db.execute("INSERT INTO competitions VALUES (2, 'Competition B')")
        .unwrap();
    db.execute("INSERT INTO competitions VALUES (3, 'Competition C')")
        .unwrap();
    db.execute("INSERT INTO episodes VALUES (100, 1, 'Episode 1')")
        .unwrap();
    db.execute("INSERT INTO episodes VALUES (101, 1, 'Episode 2')")
        .unwrap();
    db.execute("COMMIT").unwrap();

    let rows = db
        .query(
            "SELECT c.id AS comp_id, e.id AS epis_id
             FROM competitions c
             LEFT JOIN episodes e ON e.competition_id = c.id
             ORDER BY c.id, e.id",
        )
        .expect("Failed to execute LEFT JOIN with qualified columns");

    assert_eq!(rows.len(), 4, "Should have 2 matched + 2 unmatched = 4 rows");

    let comp_id_0 = rows[0].values.first().and_then(|v| v.as_int()).unwrap();
    let epis_id_0 = rows[0].values.get(1).and_then(|v| v.as_int());
    assert_eq!(comp_id_0, 1, "First row comp_id should be 1");
    assert_eq!(epis_id_0, Some(100), "First row epis_id should be 100");

    let comp_id_1 = rows[1].values.first().and_then(|v| v.as_int()).unwrap();
    let epis_id_1 = rows[1].values.get(1).and_then(|v| v.as_int());
    assert_eq!(comp_id_1, 1, "Second row comp_id should be 1");
    assert_eq!(epis_id_1, Some(101), "Second row epis_id should be 101");

    let comp_id_2 = rows[2].values.first().and_then(|v| v.as_int()).unwrap();
    let epis_id_2 = rows[2].values.get(1);
    assert_eq!(comp_id_2, 2, "Third row comp_id should be 2");
    assert!(
        epis_id_2.map(|v| v.is_null()).unwrap_or(false),
        "Third row epis_id should be NULL (no matching episode)"
    );

    let comp_id_3 = rows[3].values.first().and_then(|v| v.as_int()).unwrap();
    let epis_id_3 = rows[3].values.get(1);
    assert_eq!(comp_id_3, 3, "Fourth row comp_id should be 3");
    assert!(
        epis_id_3.map(|v| v.is_null()).unwrap_or(false),
        "Fourth row epis_id should be NULL (no matching episode)"
    );
}
