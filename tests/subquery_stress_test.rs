//! Subquery Stress Test Suite
//!
//! This test suite validates complex subquery behavior including:
//! - ORDER BY with LIMIT in subqueries
//! - Nested subqueries at multiple levels
//! - Aggregations within subqueries
//! - Various ORDER BY directions (ASC/DESC)
//! - Edge cases like empty results, OFFSET, etc.
//!
//! Each query is timed and results are verified against expected row counts.

use std::fs;
use std::time::{Duration, Instant};
use turdb::Database;

struct QueryResult {
    query_num: usize,
    query: String,
    row_count: usize,
    expected: ExpectedResult,
    passed: bool,
    duration: Duration,
    error: Option<String>,
}

#[derive(Debug, Clone)]
enum ExpectedResult {
    ExactRows(usize),
    MinRows(usize),
    MaxId,
    MinId,
    Any,
}

fn parse_expected_result(comment: &str) -> ExpectedResult {
    if comment.contains("EXPECT_MAX_ID") {
        ExpectedResult::MaxId
    } else if comment.contains("EXPECT_MIN_ID") {
        ExpectedResult::MinId
    } else if comment.contains("EXPECT_ROWS: >=") {
        let num = comment
            .split(">=")
            .nth(1)
            .and_then(|s| s.trim().parse().ok())
            .unwrap_or(0);
        ExpectedResult::MinRows(num)
    } else if comment.contains("EXPECT_ROWS:") {
        let num = comment
            .split("EXPECT_ROWS:")
            .nth(1)
            .and_then(|s| s.trim().parse().ok())
            .unwrap_or(0);
        ExpectedResult::ExactRows(num)
    } else {
        ExpectedResult::Any
    }
}

fn parse_sql_file(content: &str) -> Vec<(usize, String, ExpectedResult)> {
    let mut queries = Vec::new();
    let mut current_query = String::new();
    let mut query_num = 0;
    let mut expected = ExpectedResult::Any;

    for line in content.lines() {
        let trimmed = line.trim();

        if trimmed.starts_with("-- Q") && trimmed.contains(":") {
            if !current_query.is_empty() {
                queries.push((query_num, current_query.trim().to_string(), expected.clone()));
                current_query.clear();
            }
            query_num += 1;
            expected = ExpectedResult::Any;
        } else if trimmed.starts_with("-- EXPECT") {
            expected = parse_expected_result(trimmed);
        } else if trimmed.starts_with("--") {
            continue;
        } else if !trimmed.is_empty() {
            if !current_query.is_empty() {
                current_query.push(' ');
            }
            current_query.push_str(trimmed);
        }
    }

    if !current_query.is_empty() {
        queries.push((query_num, current_query.trim().to_string(), expected));
    }

    queries
}

fn verify_result(
    rows: &[turdb::Row],
    expected: &ExpectedResult,
    db: &Database,
) -> (bool, Option<String>) {
    match expected {
        ExpectedResult::ExactRows(n) => {
            if rows.len() == *n {
                (true, None)
            } else {
                (
                    false,
                    Some(format!("Expected {} rows, got {}", n, rows.len())),
                )
            }
        }
        ExpectedResult::MinRows(n) => {
            if rows.len() >= *n {
                (true, None)
            } else {
                (
                    false,
                    Some(format!("Expected >= {} rows, got {}", n, rows.len())),
                )
            }
        }
        ExpectedResult::MaxId => {
            let max_result = db.query("SELECT MAX(id) FROM organizations").unwrap();
            if max_result.is_empty() || rows.is_empty() {
                return (false, Some("Empty result for MAX check".to_string()));
            }
            let max_id = match &max_result[0].values[0] {
                turdb::OwnedValue::Int(v) => *v,
                _ => return (false, Some("MAX id not an integer".to_string())),
            };
            let got_id = match &rows[0].values[0] {
                turdb::OwnedValue::Int(v) => *v,
                _ => return (false, Some("Result id not an integer".to_string())),
            };
            if got_id == max_id {
                (true, None)
            } else {
                (
                    false,
                    Some(format!("Expected max id {}, got {}", max_id, got_id)),
                )
            }
        }
        ExpectedResult::MinId => {
            let min_result = db.query("SELECT MIN(id) FROM organizations").unwrap();
            if min_result.is_empty() || rows.is_empty() {
                return (false, Some("Empty result for MIN check".to_string()));
            }
            let min_id = match &min_result[0].values[0] {
                turdb::OwnedValue::Int(v) => *v,
                _ => return (false, Some("MIN id not an integer".to_string())),
            };
            let got_id = match &rows[0].values[0] {
                turdb::OwnedValue::Int(v) => *v,
                _ => return (false, Some("Result id not an integer".to_string())),
            };
            if got_id == min_id {
                (true, None)
            } else {
                (
                    false,
                    Some(format!("Expected min id {}, got {}", min_id, got_id)),
                )
            }
        }
        ExpectedResult::Any => (true, None),
    }
}

#[test]
fn test_subquery_stress_suite() {
    let db_path = std::path::Path::new("./.worktrees/bismillah");
    if !db_path.exists() {
        eprintln!("Test database not found at {:?}, skipping test", db_path);
        return;
    }

    let db = Database::open(db_path).expect("Failed to open database");

    let sql_path = std::path::Path::new("tests/queries/subquery_stress_test.sql");
    let sql_content = fs::read_to_string(sql_path).expect("Failed to read SQL file");

    let queries = parse_sql_file(&sql_content);
    let mut results: Vec<QueryResult> = Vec::new();
    let mut total_duration = Duration::ZERO;
    let mut passed_count = 0;
    let mut failed_count = 0;

    println!("\n{}", "=".repeat(80));
    println!("SUBQUERY STRESS TEST SUITE");
    println!("{}", "=".repeat(80));
    println!("Running {} queries...\n", queries.len());

    for (query_num, query, expected) in &queries {
        let start = Instant::now();
        let result = db.query(query);
        let duration = start.elapsed();
        total_duration += duration;

        let (row_count, passed, error) = match result {
            Ok(rows) => {
                let (passed, err) = verify_result(&rows, expected, &db);
                (rows.len(), passed, err)
            }
            Err(e) => {
                let is_expected_error = matches!(expected, ExpectedResult::ExactRows(0));
                (0, is_expected_error, Some(e.to_string()))
            }
        };

        if passed {
            passed_count += 1;
            print!(".");
        } else {
            failed_count += 1;
            print!("F");
        }

        results.push(QueryResult {
            query_num: *query_num,
            query: query.clone(),
            row_count,
            expected: expected.clone(),
            passed,
            duration,
            error,
        });
    }

    println!("\n\n{}", "=".repeat(80));
    println!("RESULTS SUMMARY");
    println!("{}", "=".repeat(80));

    println!("\nTiming Statistics:");
    println!("  Total time: {:?}", total_duration);
    println!(
        "  Average per query: {:?}",
        total_duration / results.len() as u32
    );

    let mut sorted_by_time: Vec<_> = results.iter().collect();
    sorted_by_time.sort_by(|a, b| b.duration.cmp(&a.duration));

    println!("\nTop 5 Slowest Queries:");
    for r in sorted_by_time.iter().take(5) {
        println!(
            "  Q{}: {:?} - {} rows",
            r.query_num, r.duration, r.row_count
        );
    }

    if failed_count > 0 {
        println!("\n{}", "-".repeat(80));
        println!("FAILED QUERIES:");
        println!("{}", "-".repeat(80));
        for r in &results {
            if !r.passed {
                println!("\nQ{}: FAILED", r.query_num);
                println!("  Query: {}", truncate_string(&r.query, 70));
                println!("  Expected: {:?}", r.expected);
                println!("  Got: {} rows", r.row_count);
                if let Some(ref err) = r.error {
                    println!("  Error: {}", err);
                }
            }
        }
    }

    println!("\n{}", "=".repeat(80));
    println!(
        "FINAL: {} passed, {} failed out of {} queries",
        passed_count,
        failed_count,
        results.len()
    );
    println!("{}", "=".repeat(80));

    assert_eq!(
        failed_count, 0,
        "{} queries failed. See output above for details.",
        failed_count
    );
}

fn truncate_string(s: &str, max_len: usize) -> String {
    if s.len() <= max_len {
        s.to_string()
    } else {
        format!("{}...", &s[..max_len])
    }
}

#[test]
fn test_order_by_limit_correctness() {
    let db_path = std::path::Path::new("./.worktrees/bismillah");
    if !db_path.exists() {
        eprintln!("Test database not found, skipping test");
        return;
    }

    let db = Database::open(db_path).expect("Failed to open database");

    println!("\nTesting ORDER BY + LIMIT correctness in subqueries...\n");

    let max_result = db.query("SELECT MAX(id) FROM organizations").unwrap();
    let max_id: i64 = match &max_result[0].values[0] {
        turdb::OwnedValue::Int(v) => *v,
        _ => panic!("Expected integer for MAX(id)"),
    };

    let min_result = db.query("SELECT MIN(id) FROM organizations").unwrap();
    let min_id: i64 = match &min_result[0].values[0] {
        turdb::OwnedValue::Int(v) => *v,
        _ => panic!("Expected integer for MIN(id)"),
    };

    println!("Table stats: MIN(id)={}, MAX(id)={}", min_id, max_id);

    let desc_subquery = db
        .query("SELECT id FROM (SELECT id FROM organizations ORDER BY id DESC LIMIT 1)")
        .unwrap();
    assert_eq!(desc_subquery.len(), 1, "DESC subquery should return 1 row");
    let got_desc: i64 = match &desc_subquery[0].values[0] {
        turdb::OwnedValue::Int(v) => *v,
        _ => panic!("Expected integer"),
    };
    assert_eq!(
        got_desc, max_id,
        "DESC LIMIT 1 in subquery should return MAX id"
    );
    println!("  ORDER BY DESC LIMIT 1 in subquery: {} (expected {})", got_desc, max_id);

    let asc_subquery = db
        .query("SELECT id FROM (SELECT id FROM organizations ORDER BY id ASC LIMIT 1)")
        .unwrap();
    assert_eq!(asc_subquery.len(), 1, "ASC subquery should return 1 row");
    let got_asc: i64 = match &asc_subquery[0].values[0] {
        turdb::OwnedValue::Int(v) => *v,
        _ => panic!("Expected integer"),
    };
    assert_eq!(
        got_asc, min_id,
        "ASC LIMIT 1 in subquery should return MIN id"
    );
    println!("  ORDER BY ASC LIMIT 1 in subquery: {} (expected {})", got_asc, min_id);

    let desc_direct = db
        .query("SELECT id FROM organizations ORDER BY id DESC LIMIT 1")
        .unwrap();
    let got_desc_direct: i64 = match &desc_direct[0].values[0] {
        turdb::OwnedValue::Int(v) => *v,
        _ => panic!("Expected integer"),
    };
    assert_eq!(
        got_desc, got_desc_direct,
        "Subquery result should match direct query"
    );
    println!("  Direct vs Subquery (DESC): {} == {} ", got_desc_direct, got_desc);

    let nested = db
        .query("SELECT id FROM (SELECT id FROM (SELECT id FROM organizations ORDER BY id DESC LIMIT 10) ORDER BY id ASC LIMIT 1)")
        .unwrap();
    let got_nested: i64 = match &nested[0].values[0] {
        turdb::OwnedValue::Int(v) => *v,
        _ => panic!("Expected integer"),
    };
    println!("  Nested subquery (DESC 10, then ASC 1): {}", got_nested);

    let top10 = db
        .query("SELECT id FROM organizations ORDER BY id DESC LIMIT 10")
        .unwrap();
    let min_of_top10: i64 = top10
        .iter()
        .map(|r| match &r.values[0] {
            turdb::OwnedValue::Int(v) => *v,
            _ => 0,
        })
        .min()
        .unwrap();
    assert_eq!(
        got_nested, min_of_top10,
        "Nested query should return smallest of top 10"
    );

    println!("\nAll ORDER BY + LIMIT correctness tests passed!");
}

#[test]
fn test_debug_q84() {
    let db_path = std::path::Path::new("./.worktrees/bismillah");
    if !db_path.exists() {
        eprintln!("Test database not found, skipping test");
        return;
    }

    println!("\nDebugging Q84...\n");

    // First test direct BTreeReader reverse iteration BEFORE opening the database
    {
        use turdb::storage::{MmapStorage, TableFileHeader};
        use turdb::btree::BTreeReader;

        let storage_path = db_path.join("root").join("organizations.tbd");
        let storage = MmapStorage::open(&storage_path).expect("Failed to open storage");

        let root_page = {
            let page0 = storage.page(0).expect("Failed to read page 0");
            let header = TableFileHeader::from_bytes(page0).expect("Failed to parse header");
            header.root_page()
        };
        println!("File header root_page: {}", root_page);

        let reader = BTreeReader::new(&storage, root_page).expect("Failed to create reader");

        println!("Testing cursor_last()...");
        let mut cursor = reader.cursor_last().expect("Failed to get cursor_last");
        println!("cursor_last() returned, valid: {}", cursor.valid());

        if cursor.valid() {
            let key = cursor.key().expect("Failed to get key");
            println!("Last key (first 20 bytes): {:?}", &key[..key.len().min(20)]);
        }

        println!("Testing prev() calls (max 10)...");
        let mut count = 0;
        let max_count = 10;
        while count < max_count && cursor.valid() {
            let result = cursor.prev();
            match result {
                Ok(true) => {
                    count += 1;
                    println!("  prev() #{} succeeded", count);
                }
                Ok(false) => {
                    println!("  reached beginning");
                    break;
                }
                Err(e) => {
                    println!("  error: {}", e);
                    break;
                }
            }
        }
        println!("Completed {} prev() calls", count);
    }

    let db = Database::open(db_path).expect("Failed to open database");

    let count_result = db.query("SELECT COUNT(*) FROM organizations").unwrap();
    let total_rows: i64 = match &count_result[0].values[0] {
        turdb::OwnedValue::Int(v) => *v,
        _ => panic!("Expected integer for COUNT(*)"),
    };
    println!("Total rows in organizations: {}", total_rows);

    // Test various cases to understand the issue
    let test_cases = [
        ("No ORDER BY, no LIMIT", "SELECT id FROM organizations"),
        ("Just LIMIT 100", "SELECT id FROM organizations LIMIT 100"),
        ("ORDER BY ASC", "SELECT id FROM organizations ORDER BY id ASC"),
        ("ORDER BY ASC LIMIT 100", "SELECT id FROM organizations ORDER BY id ASC LIMIT 100"),
        ("ORDER BY DESC", "SELECT id FROM organizations ORDER BY id DESC"),
        ("ORDER BY DESC LIMIT 100", "SELECT id FROM organizations ORDER BY id DESC LIMIT 100"),
        ("LIMIT 10 OFFSET 0", "SELECT id FROM organizations ORDER BY id DESC LIMIT 10 OFFSET 0"),
        ("LIMIT 10 OFFSET 50", "SELECT id FROM organizations ORDER BY id DESC LIMIT 10 OFFSET 50"),
        ("LIMIT 10 OFFSET 60", "SELECT id FROM organizations ORDER BY id DESC LIMIT 10 OFFSET 60"),
        ("LIMIT 10 OFFSET 100", "SELECT id FROM organizations ORDER BY id DESC LIMIT 10 OFFSET 100"),
    ];

    for (name, query) in test_cases {
        let result = db.query(query);
        match result {
            Ok(rows) => {
                let first_id = rows.first().map(|r| format!("{:?}", r.values[0])).unwrap_or_default();
                println!("{}: {} rows (first: {})", name, rows.len(), first_id);
            }
            Err(e) => println!("{}: ERROR - {}", name, e),
        }
    }

    // Original tests
    let inner = db.query("SELECT id FROM organizations ORDER BY id DESC LIMIT 50 OFFSET 700");
    match &inner {
        Ok(rows) => {
            println!("\nInner query (no wrapper): {} rows", rows.len());
            if let Some(first) = rows.first() {
                println!("  First row: {:?}", first.values);
            }
        }
        Err(e) => println!("Inner query error: {}", e),
    }

    let q84 = db.query("SELECT * FROM (SELECT id FROM organizations ORDER BY id DESC LIMIT 50 OFFSET 700)");
    match &q84 {
        Ok(rows) => {
            println!("Q84 (wrapped): {} rows", rows.len());
            if let Some(first) = rows.first() {
                println!("  First row: {:?}", first.values);
            }
        }
        Err(e) => println!("Q84 error: {}", e),
    }

    let q83 = db.query("SELECT * FROM (SELECT id, name FROM organizations ORDER BY id DESC LIMIT 10 OFFSET 5)");
    match &q83 {
        Ok(rows) => println!("Q83 (smaller offset): {} rows", rows.len()),
        Err(e) => println!("Q83 error: {}", e),
    }

    // Test raw cursor behavior
    // Test direct reverse iteration with BTreeReader
    println!("\n--- Testing direct BTreeReader reverse iteration ---");
    {
        use turdb::storage::MmapStorage;
        use turdb::btree::BTreeReader;
        use turdb::storage::TableFileHeader;

        let storage_path = db_path.join("root").join("organizations.tbd");
        let storage = MmapStorage::open(&storage_path).expect("Failed to open storage");

        let root_page = {
            let page0 = storage.page(0).expect("Failed to read page 0");
            let header = TableFileHeader::from_bytes(page0).expect("Failed to parse header");
            header.root_page()
        };
        println!("Using root_page: {}", root_page);

        let reader = BTreeReader::new(&storage, root_page).expect("Failed to create reader");

        // Test cursor_last
        println!("Testing cursor_last()...");
        let mut cursor = reader.cursor_last().expect("Failed to get cursor_last");
        println!("cursor_last() returned, valid: {}", cursor.valid());

        if cursor.valid() {
            let key = cursor.key().expect("Failed to get key");
            println!("Last key: {:?}", &key[..key.len().min(20)]);
        }

        // Test a few prev() calls with timeout protection
        println!("Testing prev() calls...");
        let mut count = 0;
        let max_count = 10;
        while count < max_count && cursor.valid() {
            println!("  prev() call {}...", count);
            let result = cursor.prev();
            match result {
                Ok(true) => {
                    count += 1;
                    let key = cursor.key().expect("Failed to get key");
                    println!("    success, new key: {:?}", &key[..key.len().min(20)]);
                }
                Ok(false) => {
                    println!("    reached beginning");
                    break;
                }
                Err(e) => {
                    println!("    error: {}", e);
                    break;
                }
            }
        }
        println!("Completed {} prev() calls", count);
    }

    println!("\n--- Testing raw cursor reverse iteration ---");
    use turdb::storage::MmapStorage;
    use turdb::btree::BTreeReader;
    use turdb::storage::TableFileHeader;

    let storage_path = db_path.join("root").join("organizations.tbd");
    if storage_path.exists() {
        let storage = MmapStorage::open(&storage_path).expect("Failed to open storage");

        // Read the actual root page from file header
        let root_page = {
            let page0 = storage.page(0).expect("Failed to read page 0");
            let header = TableFileHeader::from_bytes(page0).expect("Failed to read header");
            header.root_page()
        };
        println!("Actual root_page from header: {}", root_page);
        println!("Total page count: {}", storage.page_count());

        // Check page types
        for page_no in 1..storage.page_count().min(10) {
            let page_data = storage.page(page_no).expect("Failed to read page");
            let page_type = page_data[0];
            let cell_count = u16::from_le_bytes([page_data[2], page_data[3]]);
            let next_leaf = u32::from_le_bytes([page_data[12], page_data[13], page_data[14], page_data[15]]);
            println!("Page {}: type={}, cell_count={}, next_leaf={}", page_no, page_type, cell_count, next_leaf);
        }

        let reader = BTreeReader::new(&storage, root_page).expect("Failed to create reader");

        // Test cursor_last and prev()
        let mut cursor = reader.cursor_last().expect("Failed to get cursor_last");
        let mut count = 0;
        let max_count = 100; // Just count first 100 to debug

        if cursor.valid() {
            count += 1;
            while count < max_count && cursor.prev().expect("prev failed") {
                count += 1;
            }
        }
        println!("Raw cursor reverse count (max {}): {}", max_count, count);

        // Try full reverse iteration
        let mut cursor = reader.cursor_last().expect("Failed to get cursor_last");
        let mut full_count = 0;
        if cursor.valid() {
            full_count += 1;
            while cursor.prev().expect("prev failed") {
                full_count += 1;
            }
        }
        println!("Raw cursor full reverse count: {}", full_count);
    }

    assert!(q84.is_ok() && !q84.unwrap().is_empty(), "Q84 should return rows");
}

#[test]
fn test_trace_leaf_chain() {
    use turdb::storage::MmapStorage;

    let db_path = std::path::Path::new("./.worktrees/bismillah");
    if !db_path.exists() {
        eprintln!("Test database not found, skipping test");
        return;
    }

    let storage_path = db_path.join("root").join("organizations.tbd");
    let storage = MmapStorage::open(&storage_path).expect("Failed to open storage");
    let page_count = storage.page_count();

    // Trace from page 1 following next_leaf
    println!("\nTracing leaf chain from page 1...");
    let mut current_page: u32 = 1;
    let mut total_cells = 0u64;
    let mut chain: Vec<(u32, u16)> = Vec::new();

    while current_page != 0 && current_page < page_count {
        let page_data = storage.page(current_page).expect("Failed to read page");
        let page_type = page_data[0];
        let cell_count = u16::from_le_bytes([page_data[2], page_data[3]]);
        let next_leaf = u32::from_le_bytes([page_data[12], page_data[13], page_data[14], page_data[15]]);

        if page_type != 2 {
            println!("  Page {} is not a leaf (type={}), stopping", current_page, page_type);
            break;
        }

        chain.push((current_page, cell_count));
        total_cells += cell_count as u64;

        if chain.len() <= 30 {
            println!("  Page {}: {} cells -> next_leaf={}", current_page, cell_count, next_leaf);
        }

        current_page = next_leaf;
    }

    println!("Total leaf pages in chain: {}", chain.len());
    println!("Total cells: {}", total_cells);
}

#[test]
fn test_order_by_desc_fresh_database() {
    // Create a fresh temporary database to test ORDER BY DESC with proper structure
    let temp_dir = tempfile::tempdir().expect("Failed to create temp dir");
    let db_path = temp_dir.path();

    // Create database and insert test data
    {
        let db = Database::create(db_path).expect("Failed to create database");

        db.execute("CREATE TABLE test_table (id INTEGER PRIMARY KEY, value TEXT)")
            .expect("Failed to create table");

        // Insert 200 rows to ensure multiple pages and B-tree splits
        for i in 1..=200 {
            db.execute(&format!("INSERT INTO test_table (id, value) VALUES ({}, 'value_{}')", i, i))
                .expect("Failed to insert row");
        }
    }

    // Reopen and test
    let db = Database::open(db_path).expect("Failed to open database");

    // Test forward scan
    let forward = db.query("SELECT id FROM test_table ORDER BY id ASC").unwrap();
    println!("Forward scan: {} rows", forward.len());
    assert_eq!(forward.len(), 200, "Forward scan should return 200 rows");

    let first_forward = match &forward[0].values[0] {
        turdb::OwnedValue::Int(v) => *v,
        _ => panic!("Expected Int"),
    };
    assert_eq!(first_forward, 1, "First row should be id=1");

    // Test reverse scan (ORDER BY DESC)
    let reverse = db.query("SELECT id FROM test_table ORDER BY id DESC").unwrap();
    println!("Reverse scan: {} rows", reverse.len());
    assert_eq!(reverse.len(), 200, "Reverse scan should return 200 rows");

    let first_reverse = match &reverse[0].values[0] {
        turdb::OwnedValue::Int(v) => *v,
        _ => panic!("Expected Int"),
    };
    assert_eq!(first_reverse, 200, "First row of DESC should be id=200");

    // Test with LIMIT and OFFSET
    let limited = db.query("SELECT id FROM test_table ORDER BY id DESC LIMIT 10 OFFSET 50").unwrap();
    println!("Limited reverse: {} rows", limited.len());
    assert_eq!(limited.len(), 10, "Should return 10 rows");

    let first_limited = match &limited[0].values[0] {
        turdb::OwnedValue::Int(v) => *v,
        _ => panic!("Expected Int"),
    };
    // ORDER BY DESC LIMIT 10 OFFSET 50 should return ids 150, 149, 148, ...
    assert_eq!(first_limited, 150, "First row of DESC with OFFSET 50 should be id=150");

    println!("All ORDER BY DESC tests passed with fresh database!");
}

#[test]
fn test_subquery_performance_benchmark() {
    let db_path = std::path::Path::new("./.worktrees/bismillah");
    if !db_path.exists() {
        eprintln!("Test database not found, skipping test");
        return;
    }

    let db = Database::open(db_path).expect("Failed to open database");

    println!("\nSubquery Performance Benchmark");
    println!("{}", "=".repeat(60));

    let benchmarks = vec![
        ("Direct query LIMIT 1", "SELECT * FROM organizations LIMIT 1"),
        (
            "Subquery LIMIT 1",
            "SELECT * FROM (SELECT * FROM organizations LIMIT 1)",
        ),
        (
            "Direct ORDER BY DESC LIMIT 1",
            "SELECT * FROM organizations ORDER BY id DESC LIMIT 1",
        ),
        (
            "Subquery ORDER BY DESC LIMIT 1",
            "SELECT * FROM (SELECT * FROM organizations ORDER BY id DESC LIMIT 1)",
        ),
        (
            "Nested 2-level subquery",
            "SELECT * FROM (SELECT * FROM (SELECT * FROM organizations LIMIT 100) LIMIT 10)",
        ),
        (
            "Subquery with WHERE",
            "SELECT * FROM (SELECT * FROM organizations WHERE id > 1000 LIMIT 10)",
        ),
        (
            "Subquery with aggregate",
            "SELECT * FROM (SELECT COUNT(*) FROM organizations)",
        ),
    ];

    let iterations = 10;

    for (name, query) in benchmarks {
        let mut total_time = Duration::ZERO;
        let mut row_count = 0;

        for _ in 0..iterations {
            let start = Instant::now();
            let result = db.query(query).unwrap();
            total_time += start.elapsed();
            row_count = result.len();
        }

        let avg_time = total_time / iterations as u32;
        println!(
            "{:40} avg: {:>10?}  rows: {}",
            name, avg_time, row_count
        );
    }

    println!("{}", "=".repeat(60));
}

#[test]
#[ignore] // Run manually with: cargo test test_repair_root_page --release -- --nocapture --ignored
fn test_repair_root_page() {
    use turdb::storage::MmapStorage;
    use turdb::btree::BTreeReader;
    use turdb::storage::TableFileHeader;

    let db_path = std::path::Path::new("./.worktrees/bismillah");
    let storage_path = db_path.join("root").join("organizations.tbd");

    if !storage_path.exists() {
        eprintln!("Storage file not found");
        return;
    }

    // Open storage (mutable operations available via page_mut)
    let mut storage = MmapStorage::open(&storage_path).expect("Failed to open storage");
    let page_count = storage.page_count();
    println!("Total pages: {}", page_count);

    // Find all interior pages (type=1)
    let mut interior_pages: Vec<u32> = Vec::new();
    for page_no in 1..page_count {
        let page_data = storage.page(page_no).expect("Failed to read page");
        if page_data[0] == 1 { // Interior node
            interior_pages.push(page_no);
        }
    }
    println!("Found {} interior pages: {:?}", interior_pages.len(), &interior_pages[..interior_pages.len().min(10)]);

    // With the linear split pattern observed in this B-tree, the root is the highest interior page
    // Each split creates: new_leaf, new_interior where new_interior becomes the new root
    let best_root = *interior_pages.last().expect("No interior pages found");
    println!("Using highest interior page as root: {}", best_root);

    // Verify with forward iteration (which works with any interior page)
    let reader = BTreeReader::new(&storage, best_root).expect("Failed to create reader");
    let mut cursor = reader.cursor_first().expect("Failed to get cursor");
    let mut best_count = 0;
    while cursor.valid() {
        best_count += 1;
        if cursor.advance().is_err() { break; }
    }
    println!("Forward iteration count with root {}: {}", best_root, best_count);

    // Update the header with the correct root page
    if best_count > 0 {
        let page0 = storage.page_mut(0).expect("Failed to get page 0 for writing");
        let header = TableFileHeader::from_bytes_mut(page0).expect("Failed to parse header");
        let old_root = header.root_page();
        header.set_root_page(best_root);
        storage.sync().expect("Failed to sync storage");
        println!("Updated root_page from {} to {}", old_root, best_root);
    }
}
