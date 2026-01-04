use turdb::Database;
use tempfile::tempdir;

#[test]
fn test_toast_consistency() {
    let dir = tempdir().unwrap();
    let db_path = dir.path().join("test_db");

    // 1. Create DB and Table
    {
        let db = Database::create(&db_path).unwrap();
        db.execute("PRAGMA WAL=ON").unwrap();
        
        db.execute("CREATE TABLE test (id INT PRIMARY KEY, data TEXT)").unwrap();
        
        // 2. Insert Large Value
        let large_string = "a".repeat(10_000); // 10KB > 2KB threshold
        let sql = format!("INSERT INTO test VALUES (1, '{}')", large_string);
        db.execute(&sql).unwrap();
        
        // 3. Commit (implicitly done by execute in autocommit mode, but let's be explicit)
        db.execute("BEGIN").unwrap();
        db.execute(&format!("INSERT INTO test VALUES (2, '{}')", large_string)).unwrap();
        db.execute("COMMIT").unwrap();
    }

    // 4. Re-open DB (simulates restart)
    {
        let db = Database::open(&db_path).unwrap();
        
        // 5. Query
        let rows = db.query("SELECT data FROM test WHERE id = 1").unwrap();
        assert_eq!(rows.len(), 1);
        let val = rows[0].get(0).unwrap();
        if let turdb::OwnedValue::Text(s) = val {
            assert_eq!(s.len(), 10_000);
        } else {
            panic!("Expected Text value");
        }

        let rows = db.query("SELECT data FROM test WHERE id = 2").unwrap();
        assert_eq!(rows.len(), 1);
        let val = rows[0].get(0).unwrap();
        if let turdb::OwnedValue::Text(s) = val {
            assert_eq!(s.len(), 10_000);
        } else {
            panic!("Expected Text value");
        }
    }
}
