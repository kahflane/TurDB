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
