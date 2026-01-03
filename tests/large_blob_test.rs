//! Large blob stress test
//! Tests storing 10MB+ blobs across 100k rows

use tempfile::tempdir;
use turdb::Database;
use turdb::types::OwnedValue;

#[test]
fn test_large_blob_storage_100k_rows() {
    let dir = tempdir().unwrap();
    let db = Database::create(dir.path().join("large_blob_test")).unwrap();
    
    // Enable WAL for better performance
    db.execute("PRAGMA WAL=ON").unwrap();
    
    // Create table with blob column
    db.execute("CREATE TABLE blobs (id BIGINT PRIMARY KEY, data BLOB, size_kb INT)").unwrap();
    
    // We'll insert 100k rows with varying blob sizes
    // To get 10MB+ total, we need average ~100 bytes per row minimum
    // But to really test large blobs, let's use bigger sizes
    
    let total_rows = 100_000;
    let batch_size = 1000;
    
    // Create blobs of different sizes (100 bytes to 10KB)
    // This will give us roughly 500MB-1GB total data
    
    println!("Inserting {} rows...", total_rows);
    let start = std::time::Instant::now();
    
    for batch in 0..(total_rows / batch_size) {
        db.execute("BEGIN TRANSACTION").unwrap();
        
        for i in 0..batch_size {
            let id = batch * batch_size + i;
            // Vary blob size: 100 bytes to 10KB based on id
            let size = 100 + (id % 100) * 100; // 100 to 10000 bytes
            let blob_data: Vec<u8> = (0..size).map(|j| ((id + j) % 256) as u8).collect();
            
            // Use hex encoding for blob
            let hex_blob: String = blob_data.iter().map(|b| format!("{:02x}", b)).collect();
            let sql = format!(
                "INSERT INTO blobs VALUES ({}, X'{}', {})",
                id, hex_blob, size / 1024
            );
            db.execute(&sql).unwrap();
        }
        
        db.execute("COMMIT").unwrap();
        
        if (batch + 1) % 10 == 0 {
            println!("  Inserted {} rows...", (batch + 1) * batch_size);
        }
    }
    
    let insert_time = start.elapsed();
    println!("Insert completed in {:?}", insert_time);
    
    // Verify count
    let count_result = db.query("SELECT COUNT(*) FROM blobs").unwrap();
    assert_eq!(count_result[0].values[0], OwnedValue::Int(total_rows as i64));
    println!("Verified row count: {}", total_rows);
    
    // Query some specific rows
    println!("Querying specific rows...");
    let query_start = std::time::Instant::now();
    
    for test_id in [0, 1000, 50000, 99999] {
        let rows = db.query(&format!("SELECT id, data, size_kb FROM blobs WHERE id = {}", test_id)).unwrap();
        assert_eq!(rows.len(), 1, "Should find row {}", test_id);
        assert_eq!(rows[0].values[0], OwnedValue::Int(test_id as i64));
        
        if let OwnedValue::Blob(blob) = &rows[0].values[1] {
            let expected_size = 100 + (test_id % 100) * 100;
            assert_eq!(blob.len(), expected_size, "Blob size mismatch for id {}", test_id);
            
            // Verify blob content
            for j in 0..blob.len() {
                let expected = ((test_id + j) % 256) as u8;
                assert_eq!(blob[j], expected, "Blob content mismatch at position {} for id {}", j, test_id);
            }
        } else {
            panic!("Expected Blob value for id {}", test_id);
        }
    }
    
    let query_time = query_start.elapsed();
    println!("Query verification completed in {:?}", query_time);
    
    // Query with range scan
    println!("Running range scan...");
    let range_start = std::time::Instant::now();
    let range_rows = db.query("SELECT id, LENGTH(data) as len FROM blobs WHERE id >= 50000 AND id < 51000").unwrap();
    assert_eq!(range_rows.len(), 1000);
    let range_time = range_start.elapsed();
    println!("Range scan (1000 rows) completed in {:?}", range_time);
    
    // Aggregate query
    println!("Running aggregate query...");
    let agg_start = std::time::Instant::now();
    let agg_result = db.query("SELECT SUM(size_kb), AVG(size_kb), MAX(size_kb), MIN(size_kb) FROM blobs").unwrap();
    let agg_time = agg_start.elapsed();
    println!("Aggregate query completed in {:?}", agg_time);
    println!("Aggregates: {:?}", agg_result[0].values);
    
    println!("\n=== Test Summary ===");
    println!("Total rows: {}", total_rows);
    println!("Insert time: {:?}", insert_time);
    println!("Query time: {:?}", query_time);
    println!("Range scan time: {:?}", range_time);
    println!("Aggregate time: {:?}", agg_time);
}

#[test]
fn test_single_10mb_blob() {
    let dir = tempdir().unwrap();
    let db = Database::create(dir.path().join("single_large_blob")).unwrap();
    
    db.execute("PRAGMA WAL=ON").unwrap();
    db.execute("CREATE TABLE large_data (id BIGINT PRIMARY KEY, data BLOB)").unwrap();
    
    // Create a 10MB blob
    let blob_size = 10 * 1024 * 1024; // 10MB
    println!("Creating {}MB blob...", blob_size / 1024 / 1024);
    
    let blob_data: Vec<u8> = (0..blob_size).map(|i| (i % 256) as u8).collect();
    let hex_blob: String = blob_data.iter().map(|b| format!("{:02x}", b)).collect();
    
    println!("Inserting 10MB blob...");
    let start = std::time::Instant::now();
    let sql = format!("INSERT INTO large_data VALUES (1, X'{}')", hex_blob);
    db.execute(&sql).unwrap();
    let insert_time = start.elapsed();
    println!("Insert completed in {:?}", insert_time);
    
    // Read it back
    println!("Reading 10MB blob...");
    let read_start = std::time::Instant::now();
    let rows = db.query("SELECT id, data FROM large_data WHERE id = 1").unwrap();
    let read_time = read_start.elapsed();
    println!("Read completed in {:?}", read_time);
    
    assert_eq!(rows.len(), 1);
    if let OwnedValue::Blob(blob) = &rows[0].values[1] {
        assert_eq!(blob.len(), blob_size, "Blob size mismatch");
        
        // Verify content (check samples)
        for i in [0, 1000, 500000, blob_size - 1] {
            assert_eq!(blob[i], (i % 256) as u8, "Content mismatch at position {}", i);
        }
        println!("Blob content verified!");
    } else {
        panic!("Expected Blob value");
    }
}
