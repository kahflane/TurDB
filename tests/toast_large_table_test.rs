//! Test for TOAST with large tables
//!
//! This test reproduces the "TOAST chunk not found" error that occurs
//! with large tables (hundreds of thousands or millions of rows).

use tempfile::tempdir;
use turdb::database::Database;
use turdb::types::OwnedValue;

#[test]
fn test_toast_large_table_simulation() {
    let dir = tempdir().unwrap();
    let db = Database::create(dir.path().join("toast_large")).unwrap();

    db.execute("PRAGMA WAL=ON").unwrap();

    // Create table with a TEXT column (toastable)
    db.execute("CREATE TABLE large_toast (
        id BIGINT PRIMARY KEY,
        data TEXT
    )").unwrap();

    // Generate large text that exceeds TOAST_THRESHOLD (1000 bytes)
    let large_text = "x".repeat(2000);

    // Insert many rows - start with a smaller number to find the issue
    let num_rows = 50_000;

    println!("Inserting {} rows with large text...", num_rows);
    for i in 0..num_rows {
        let sql = format!("INSERT INTO large_toast VALUES ({}, '{}')", i, large_text);
        if let Err(e) = db.execute(&sql) {
            panic!("Insert failed at row {}: {}", i, e);
        }
        if i % 10_000 == 0 && i > 0 {
            println!("Inserted {} rows", i);
        }
    }
    println!("All {} rows inserted successfully", num_rows);

    // Now try to read all rows - this should trigger detoasting
    println!("Reading all rows...");
    let result = db.query("SELECT id, data FROM large_toast ORDER BY id");

    match result {
        Ok(rows) => {
            assert_eq!(rows.len(), num_rows as usize, "Should have all rows");
            println!("Successfully read {} rows", rows.len());

            // Verify some random rows
            for row in rows.iter().take(10) {
                if let OwnedValue::Text(text) = &row.values[1] {
                    assert_eq!(text.len(), 2000, "Text should be 2000 bytes");
                } else {
                    panic!("Expected Text value, got {:?}", row.values[1]);
                }
            }
        }
        Err(e) => {
            panic!("Query failed: {}", e);
        }
    }
}

#[test]
fn test_toast_with_db_reopen() {
    let dir = tempdir().unwrap();
    let db_path = dir.path().join("toast_reopen");

    // First session: create and insert
    {
        let db = Database::create(&db_path).unwrap();
        db.execute("PRAGMA WAL=ON").unwrap();

        db.execute("CREATE TABLE test_reopen (
            id BIGINT PRIMARY KEY,
            data TEXT
        )").unwrap();

        let large_text = "y".repeat(2000);
        for i in 0..1000 {
            let sql = format!("INSERT INTO test_reopen VALUES ({}, '{}')", i, large_text);
            db.execute(&sql).unwrap();
        }

        // Verify we can read in the same session
        let rows = db.query("SELECT * FROM test_reopen").unwrap();
        assert_eq!(rows.len(), 1000);
    }

    // Second session: reopen and read
    {
        let db = Database::open(&db_path).unwrap();

        // This is where the issue might occur - row_id might reset
        let rows = db.query("SELECT id, data FROM test_reopen ORDER BY id");
        match rows {
            Ok(r) => {
                assert_eq!(r.len(), 1000, "Should still have 1000 rows after reopen");
                for row in r.iter().take(10) {
                    if let OwnedValue::Text(text) = &row.values[1] {
                        assert_eq!(text.len(), 2000);
                    } else {
                        panic!("Expected Text value");
                    }
                }
            }
            Err(e) => {
                panic!("Query failed after reopen: {}", e);
            }
        }
    }
}

#[test]
fn test_toast_insert_after_reopen() {
    // This test specifically checks for row_id collision issues
    let dir = tempdir().unwrap();
    let db_path = dir.path().join("toast_collision");

    // First session: create and insert some rows
    {
        let db = Database::create(&db_path).unwrap();
        db.execute("PRAGMA WAL=ON").unwrap();

        db.execute("CREATE TABLE collision_test (
            id BIGINT PRIMARY KEY,
            data TEXT
        )").unwrap();

        let large_text = "a".repeat(2000);
        for i in 0..100 {
            let sql = format!("INSERT INTO collision_test VALUES ({}, '{}')", i, large_text);
            db.execute(&sql).unwrap();
        }
    }

    // Second session: insert more rows (row_id might reset and collide!)
    {
        let db = Database::open(&db_path).unwrap();
        db.execute("PRAGMA WAL=ON").unwrap();

        let large_text = "b".repeat(2000);
        for i in 100..200 {
            let sql = format!("INSERT INTO collision_test VALUES ({}, '{}')", i, large_text);
            db.execute(&sql).unwrap();
        }

        // Verify all 200 rows can be read with correct data
        let rows = db.query("SELECT id, data FROM collision_test ORDER BY id").unwrap();
        assert_eq!(rows.len(), 200);

        // First 100 should have 'a' repeated
        for (idx, row) in rows.iter().take(100).enumerate() {
            if let OwnedValue::Text(text) = &row.values[1] {
                if !text.chars().all(|c| c == 'a') {
                    panic!("Row {} should have 'a' repeated, got data starting with: {:?}",
                           idx, text.chars().take(10).collect::<String>());
                }
            } else {
                panic!("Expected Text value at row {}", idx);
            }
        }

        // Next 100 should have 'b' repeated
        for (idx, row) in rows.iter().skip(100).take(100).enumerate() {
            if let OwnedValue::Text(text) = &row.values[1] {
                if !text.chars().all(|c| c == 'b') {
                    panic!("Row {} should have 'b' repeated, got data starting with: {:?}",
                           100 + idx, text.chars().take(10).collect::<String>());
                }
            } else {
                panic!("Expected Text value at row {}", 100 + idx);
            }
        }
    }
}
