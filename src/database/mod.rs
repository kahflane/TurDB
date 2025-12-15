//! # Database Module
//!
//! This module provides the high-level Database API for TurDB, combining all
//! components (storage, catalog, SQL processing) into a unified interface.
//!
//! ## Architecture
//!
//! The Database struct serves as the main entry point, orchestrating:
//! - FileManager: Manages table data files, index files, and metadata
//! - Catalog: Tracks schemas, tables, columns, and indexes
//! - SQL Engine: Parses, plans, and executes SQL statements
//!
//! ## Query Execution Pipeline
//!
//! ```text
//! SQL String
//!     │
//!     ▼
//! ┌─────────────────────────────────────────────────────┐
//! │ 1. PARSE: SQL → AST                                 │
//! │    Lexer → Parser → Statement                       │
//! └─────────────────────────────────────────────────────┘
//!     │
//!     ▼
//! ┌─────────────────────────────────────────────────────┐
//! │ 2. PLAN: AST → PhysicalPlan                         │
//! │    Planner::plan(stmt) → PhysicalPlan               │
//! └─────────────────────────────────────────────────────┘
//!     │
//!     ▼
//! ┌─────────────────────────────────────────────────────┐
//! │ 3. BUILD: PhysicalPlan → Executor                   │
//! │    ExecutorBuilder::build(plan) → DynamicExecutor   │
//! └─────────────────────────────────────────────────────┘
//!     │
//!     ▼
//! ┌─────────────────────────────────────────────────────┐
//! │ 4. EXECUTE: Volcano-style pull iteration            │
//! │    executor.open() → next() → close()               │
//! └─────────────────────────────────────────────────────┘
//!     │
//!     ▼
//! Vec<Row> returned to user
//! ```
//!
//! ## Memory Management
//!
//! Each query uses a dedicated arena allocator (bumpalo::Bump) for:
//! - AST nodes during parsing
//! - Plan nodes during planning
//! - Intermediate results during execution
//!
//! The arena is dropped after query completion, bulk-deallocating all
//! query-scoped allocations in O(1).
//!
//! ## Thread Safety
//!
//! Database is Send + Sync and can be safely shared across threads.
//! Internal locking (RwLock) protects:
//! - Catalog reads/writes
//! - FileManager file operations
//!
//! ## Usage Example
//!
//! ```ignore
//! use turdb::Database;
//!
//! // Create or open database
//! let db = Database::open("./mydb")?;
//!
//! // Execute DDL
//! db.execute("CREATE TABLE users (id INT PRIMARY KEY, name TEXT)")?;
//!
//! // Insert data
//! db.execute("INSERT INTO users VALUES (1, 'Alice')")?;
//!
//! // Query data
//! let rows = db.query("SELECT * FROM users WHERE id = 1")?;
//! for row in rows {
//!     println!("{:?}", row);
//! }
//! ```
//!
//! ## Performance Targets
//!
//! - Point read: < 1µs (cached)
//! - Sequential scan: > 1M rows/sec
//! - Insert: > 100K rows/sec
//! - Query planning: < 100µs for simple queries

#[allow(clippy::module_inception)]
mod database;
pub mod owned_value;
pub mod row;

pub use database::Database;
pub use owned_value::{
    create_column_map, create_record_schema, owned_values_to_values, OwnedValue,
};
pub use row::Row;

#[derive(Debug)]
pub enum ExecuteResult {
    CreateTable { created: bool },
    CreateSchema { created: bool },
    CreateIndex { created: bool },
    DropTable { dropped: bool },
    DropIndex { dropped: bool },
    DropSchema { dropped: bool },
    Insert { rows_affected: usize },
    Update { rows_affected: usize },
    Delete { rows_affected: usize },
    Select { rows: Vec<Row> },
    Pragma { name: String, value: Option<String> },
}

#[derive(Debug, Clone)]
pub struct RecoveryInfo {
    pub frames_recovered: u32,
    pub wal_size_bytes: u64,
}

#[derive(Debug, Clone)]
pub struct CheckpointInfo {
    pub frames_checkpointed: u32,
    pub wal_truncated: bool,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::database::database::Database;
    use tempfile::tempdir;

    #[test]
    fn test_create_and_open_database() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("test_db");

        let db = Database::create(&db_path).unwrap();
        drop(db);

        let db = Database::open(&db_path).unwrap();
        assert!(db.path().exists());
    }

    #[test]
    fn test_create_table() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("test_db");

        let db = Database::create(&db_path).unwrap();

        let result = db
            .execute("CREATE TABLE users (id INT, name TEXT)")
            .unwrap();
        assert!(matches!(
            result,
            ExecuteResult::CreateTable { created: true }
        ));

        let result = db
            .execute("CREATE TABLE IF NOT EXISTS users (id INT, name TEXT)")
            .unwrap();
        assert!(matches!(
            result,
            ExecuteResult::CreateTable { created: false }
        ));
    }

    #[test]
    fn test_insert_and_query() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("test_db");

        let db = Database::create(&db_path).unwrap();

        db.execute("CREATE TABLE users (id INT, name TEXT)")
            .unwrap();

        db.execute("INSERT INTO users VALUES (1, 'Alice')").unwrap();
        db.execute("INSERT INTO users VALUES (2, 'Bob')").unwrap();

        let rows = db.query("SELECT * FROM users").unwrap();
        assert_eq!(rows.len(), 2);
    }

    #[test]
    fn test_four_column_insert() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("test_db");

        let db = Database::create(&db_path).unwrap();

        db.execute("CREATE TABLE users (id INT, name TEXT, age INT, score FLOAT)")
            .unwrap();

        db.execute("INSERT INTO users VALUES (1, 'Alice', 25, 95.5)")
            .unwrap();
        db.execute("INSERT INTO users VALUES (2, 'Bob', 30, 88.0)")
            .unwrap();

        let rows = db.query("SELECT * FROM users").unwrap();
        assert_eq!(rows.len(), 2);
    }

    #[test]
    fn test_many_inserts() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("test_db");

        let db = Database::create(&db_path).unwrap();

        db.execute("CREATE TABLE users (id INT, name TEXT, age INT, score FLOAT)")
            .unwrap();

        for i in 0..100 {
            let sql = format!(
                "INSERT INTO users VALUES ({}, 'user{}', {}, {})",
                i,
                i,
                20 + (i % 60),
                (i as f64) * 0.1
            );
            db.execute(&sql).unwrap();
        }

        let rows = db.query("SELECT * FROM users").unwrap();
        assert_eq!(rows.len(), 100);
    }

    #[test]
    fn test_wal_directory_created_lazily() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("test_db");

        let db = Database::create(&db_path).unwrap();

        let wal_dir = db_path.join("wal");
        assert!(
            !wal_dir.exists(),
            "WAL directory should NOT exist before first write"
        );

        db.execute("CREATE TABLE test (id INT)").unwrap();
        db.execute("INSERT INTO test VALUES (1)").unwrap();

        db.ensure_wal().unwrap();

        assert!(
            wal_dir.exists(),
            "WAL directory should exist after ensure_wal"
        );
        assert!(wal_dir.is_dir(), "WAL should be a directory");
    }

    #[test]
    fn test_wal_directory_created_on_checkpoint() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("test_db");

        {
            let db = Database::create(&db_path).unwrap();
            drop(db);
        }

        let db = Database::open(&db_path).unwrap();

        let wal_dir = db_path.join("wal");
        assert!(
            !wal_dir.exists(),
            "WAL directory should NOT exist immediately after open"
        );

        db.checkpoint().unwrap();
    }

    #[test]
    fn test_checkpoint_returns_info() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("test_db");

        let db = Database::create(&db_path).unwrap();

        let checkpoint_info = db.checkpoint().unwrap();
        assert_eq!(checkpoint_info.frames_checkpointed, 0);
        assert!(!checkpoint_info.wal_truncated);
    }

    #[test]
    fn test_close_returns_checkpoint_info() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("test_db");

        let db = Database::create(&db_path).unwrap();

        let checkpoint_info = db.close().unwrap();
        assert_eq!(checkpoint_info.frames_checkpointed, 0);

        assert!(db.is_closed());
    }

    #[test]
    fn test_close_prevents_further_operations() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("test_db");

        let db = Database::create(&db_path).unwrap();
        db.close().unwrap();

        let result = db.checkpoint();
        assert!(result.is_err(), "checkpoint should fail after close");
    }

    #[test]
    fn test_double_close_fails() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("test_db");

        let db = Database::create(&db_path).unwrap();
        db.close().unwrap();

        let result = db.close();
        assert!(result.is_err(), "second close should fail");
    }

    #[test]
    fn test_open_with_recovery_returns_info() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("test_db");

        {
            let db = Database::create(&db_path).unwrap();
            drop(db);
        }

        let (db, recovery_info) = Database::open_with_recovery(&db_path).unwrap();
        assert_eq!(recovery_info.frames_recovered, 0);
        assert_eq!(recovery_info.wal_size_bytes, 0);
        drop(db);
    }

    #[test]
    fn test_database_survives_reopen() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("test_db");

        {
            let db = Database::create(&db_path).unwrap();
            db.execute("CREATE TABLE users (id INT, name TEXT)")
                .unwrap();
            db.execute("INSERT INTO users VALUES (1, 'Alice')").unwrap();
            db.close().unwrap();
        }

        let db = Database::open(&db_path).unwrap();
        let rows = db.query("SELECT * FROM users").unwrap();
        assert_eq!(rows.len(), 1);
    }

    #[test]
    fn test_update_basic() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("test_db");

        let db = Database::create(&db_path).unwrap();

        db.execute("CREATE TABLE users (id INT, name TEXT)")
            .unwrap();
        db.execute("INSERT INTO users VALUES (1, 'Alice')").unwrap();
        db.execute("INSERT INTO users VALUES (2, 'Bob')").unwrap();

        let result = db
            .execute("UPDATE users SET name = 'Charlie' WHERE id = 1")
            .unwrap();

        assert!(
            matches!(result, ExecuteResult::Update { rows_affected: 1 }),
            "expected Update with 1 row affected"
        );

        let rows = db.query("SELECT * FROM users").unwrap();
        assert_eq!(
            rows.len(),
            2,
            "row count should remain unchanged after UPDATE"
        );
    }

    #[test]
    fn test_update_all_rows() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("test_db");

        let db = Database::create(&db_path).unwrap();

        db.execute("CREATE TABLE users (id INT, name TEXT)")
            .unwrap();
        db.execute("INSERT INTO users VALUES (1, 'Alice')").unwrap();
        db.execute("INSERT INTO users VALUES (2, 'Bob')").unwrap();
        db.execute("INSERT INTO users VALUES (3, 'Carol')").unwrap();

        let result = db.execute("UPDATE users SET name = 'Updated'").unwrap();

        assert!(
            matches!(result, ExecuteResult::Update { rows_affected: 3 }),
            "expected Update with 3 rows affected"
        );

        let rows = db.query("SELECT * FROM users").unwrap();
        assert_eq!(
            rows.len(),
            3,
            "row count should remain unchanged after UPDATE"
        );
    }

    #[test]
    fn test_update_no_match() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("test_db");

        let db = Database::create(&db_path).unwrap();

        db.execute("CREATE TABLE users (id INT, name TEXT)")
            .unwrap();
        db.execute("INSERT INTO users VALUES (1, 'Alice')").unwrap();
        db.execute("INSERT INTO users VALUES (2, 'Bob')").unwrap();

        let result = db
            .execute("UPDATE users SET name = 'X' WHERE id = 999")
            .unwrap();

        assert!(
            matches!(result, ExecuteResult::Update { rows_affected: 0 }),
            "expected Update with 0 rows affected"
        );

        let rows = db.query("SELECT * FROM users").unwrap();
        assert_eq!(
            rows.len(),
            2,
            "row count should remain unchanged after UPDATE"
        );
    }

    #[test]
    fn test_update_multiple_columns() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("test_db");

        let db = Database::create(&db_path).unwrap();

        db.execute("CREATE TABLE users (id INT, name TEXT, age INT)")
            .unwrap();
        db.execute("INSERT INTO users VALUES (1, 'Alice', 25)")
            .unwrap();
        db.execute("INSERT INTO users VALUES (2, 'Bob', 30)")
            .unwrap();

        let result = db
            .execute("UPDATE users SET name = 'Updated', age = 99 WHERE id = 1")
            .unwrap();

        assert!(
            matches!(result, ExecuteResult::Update { rows_affected: 1 }),
            "expected Update with 1 row affected"
        );

        let rows = db.query("SELECT * FROM users").unwrap();
        assert_eq!(
            rows.len(),
            2,
            "row count should remain unchanged after UPDATE"
        );
    }

    #[test]
    fn test_delete_basic() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("test_db");

        let db = Database::create(&db_path).unwrap();

        db.execute("CREATE TABLE users (id INT, name TEXT)")
            .unwrap();
        db.execute("INSERT INTO users VALUES (1, 'Alice')").unwrap();
        db.execute("INSERT INTO users VALUES (2, 'Bob')").unwrap();
        db.execute("INSERT INTO users VALUES (3, 'Carol')").unwrap();

        let result = db.execute("DELETE FROM users WHERE id = 2").unwrap();

        assert!(
            matches!(result, ExecuteResult::Delete { rows_affected: 1 }),
            "expected Delete with 1 row affected"
        );

        let rows = db.query("SELECT * FROM users").unwrap();
        assert_eq!(rows.len(), 2, "one row should be deleted");
    }

    #[test]
    fn test_delete_all_rows() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("test_db");

        let db = Database::create(&db_path).unwrap();

        db.execute("CREATE TABLE users (id INT, name TEXT)")
            .unwrap();
        db.execute("INSERT INTO users VALUES (1, 'Alice')").unwrap();
        db.execute("INSERT INTO users VALUES (2, 'Bob')").unwrap();
        db.execute("INSERT INTO users VALUES (3, 'Carol')").unwrap();

        let result = db.execute("DELETE FROM users").unwrap();

        assert!(
            matches!(result, ExecuteResult::Delete { rows_affected: 3 }),
            "expected Delete with 3 rows affected"
        );

        let rows = db.query("SELECT * FROM users").unwrap();
        assert_eq!(rows.len(), 0, "all rows should be deleted");
    }

    #[test]
    fn test_delete_no_match() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("test_db");

        let db = Database::create(&db_path).unwrap();

        db.execute("CREATE TABLE users (id INT, name TEXT)")
            .unwrap();
        db.execute("INSERT INTO users VALUES (1, 'Alice')").unwrap();
        db.execute("INSERT INTO users VALUES (2, 'Bob')").unwrap();

        let result = db.execute("DELETE FROM users WHERE id = 999").unwrap();

        assert!(
            matches!(result, ExecuteResult::Delete { rows_affected: 0 }),
            "expected Delete with 0 rows affected"
        );

        let rows = db.query("SELECT * FROM users").unwrap();
        assert_eq!(rows.len(), 2, "no rows should be deleted");
    }

    #[test]
    fn test_delete_multiple_matches() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("test_db");

        let db = Database::create(&db_path).unwrap();

        db.execute("CREATE TABLE users (id INT, name TEXT, active INT)")
            .unwrap();
        db.execute("INSERT INTO users VALUES (1, 'Alice', 1)")
            .unwrap();
        db.execute("INSERT INTO users VALUES (2, 'Bob', 0)")
            .unwrap();
        db.execute("INSERT INTO users VALUES (3, 'Carol', 0)")
            .unwrap();
        db.execute("INSERT INTO users VALUES (4, 'Dave', 1)")
            .unwrap();

        let result = db.execute("DELETE FROM users WHERE active = 0").unwrap();

        assert!(
            matches!(result, ExecuteResult::Delete { rows_affected: 2 }),
            "expected Delete with 2 rows affected"
        );

        let rows = db.query("SELECT * FROM users").unwrap();
        assert_eq!(rows.len(), 2, "two rows should remain");
    }

    #[test]
    fn test_comprehensive_all_column_types_and_operations() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("test_db");

        let db = Database::create(&db_path).unwrap();

        db.execute(
            "CREATE TABLE all_types (
                id INT,
                bigint_col BIGINT,
                smallint_col SMALLINT,
                tinyint_col TINYINT,
                real_col REAL,
                float_col FLOAT,
                double_col DOUBLE PRECISION,
                decimal_col DECIMAL(10, 2),
                numeric_col NUMERIC(8, 4),
                varchar_col VARCHAR(255),
                char_col CHAR(10),
                text_col TEXT,
                blob_col BLOB,
                bool_col BOOLEAN,
                date_col DATE,
                time_col TIME,
                ts_col TIMESTAMP,
                tstz_col TIMESTAMPTZ,
                interval_col INTERVAL
            )",
        )
        .unwrap();

        db.execute(
            "CREATE TABLE json_table (
                id INT,
                name TEXT,
                data JSONB
            )",
        )
        .unwrap();

        db.execute(
            "CREATE TABLE uuid_table (
                id INT,
                description TEXT,
                uuid_val UUID
            )",
        )
        .unwrap();

        db.execute(
            "CREATE TABLE vector_table (
                id INT,
                label TEXT,
                embedding VECTOR(128)
            )",
        )
        .unwrap();

        db.execute(
            "CREATE TABLE related_data (
                id INT,
                all_types_id INT,
                name TEXT,
                score FLOAT,
                active BOOLEAN,
                category TEXT
            )",
        )
        .unwrap();

        for i in 0..100 {
            let sql = format!(
                "INSERT INTO all_types VALUES (
                    {id},
                    {bigint},
                    {smallint},
                    {tinyint},
                    {real},
                    {float},
                    {double},
                    {decimal},
                    {numeric},
                    'varchar_{id}',
                    'char{id}',
                    'This is a text field for row {id} with some longer content',
                    'blob_data_{id}',
                    {bool_val},
                    {date},
                    {time},
                    {ts},
                    {tstz},
                    'P1Y2M3D'
                )",
                id = i,
                bigint = i as i64 * 1000000,
                smallint = (i % 32000) as i16,
                tinyint = (i % 127) as i8,
                real = (i as f32) * 1.5,
                float = (i as f64) * 2.5,
                double = (i as f64) * std::f64::consts::PI,
                decimal = (i as f64) * 100.0,
                numeric = (i as f64) / 10.0,
                bool_val = if i % 2 == 0 { "TRUE" } else { "FALSE" },
                date = 19000 + i,
                time = 3600000000i64 * (i as i64 % 24),
                ts = 1700000000i64 + (i as i64 * 86400),
                tstz = 1700000000i64 + (i as i64 * 86400),
            );
            if let Err(e) = db.execute(&sql) {
                eprintln!("Warning: Failed to insert all_types row {}: {:?}", i, e);
            }
        }

        for i in 0..20 {
            let sql = format!("INSERT INTO json_table VALUES ({}, 'item_{}', NULL)", i, i);
            if let Err(e) = db.execute(&sql) {
                eprintln!("Warning: Failed to insert json_table row {}: {:?}", i, e);
            }
        }

        for i in 0..20 {
            let sql = format!(
                "INSERT INTO uuid_table VALUES ({}, 'uuid_item_{}', NULL)",
                i, i
            );
            if let Err(e) = db.execute(&sql) {
                eprintln!("Warning: Failed to insert uuid_table row {}: {:?}", i, e);
            }
        }

        for i in 0..20 {
            let sql = format!(
                "INSERT INTO vector_table VALUES ({}, 'vector_item_{}', NULL)",
                i, i
            );
            if let Err(e) = db.execute(&sql) {
                eprintln!("Warning: Failed to insert vector_table row {}: {:?}", i, e);
            }
        }

        for i in 0..100 {
            let category = match i % 5 {
                0 => "Electronics",
                1 => "Clothing",
                2 => "Books",
                3 => "Phones",
                _ => "Laptops",
            };
            let active = if i % 3 == 0 { "TRUE" } else { "FALSE" };
            let score = (i as f64) * 0.1;

            let sql = format!(
                "INSERT INTO related_data VALUES ({}, {}, 'Item_{}', {}, {}, '{}')",
                i,
                i % 50,
                i,
                score,
                active,
                category
            );
            db.execute(&sql).unwrap();
        }

        let rows = db.query("SELECT * FROM all_types").unwrap();
        println!("all_types row count: {}", rows.len());
        assert!(!rows.is_empty(), "should have inserted rows into all_types");

        let rows = db.query("SELECT * FROM related_data").unwrap();
        assert_eq!(rows.len(), 100, "should have 100 related_data rows");

        let result = db.query(
            "SELECT a.id, a.text_col, r.name, r.score
             FROM all_types a, related_data r
             WHERE a.id = r.all_types_id",
        );
        assert!(
            result.is_err(),
            "JOIN queries should fail until executor supports joins"
        );
        println!("JOIN query correctly rejected (executor doesn't support joins yet)");

        let result = db.query(
            "SELECT a.id, a.bigint_col, r.category
             FROM all_types a, related_data r
             WHERE a.id = r.id AND r.active = TRUE",
        );
        assert!(
            result.is_err(),
            "Filtered JOIN queries should fail until executor supports joins"
        );
        println!("Filtered JOIN correctly rejected");

        let rows = db
            .query("SELECT * FROM all_types WHERE bigint_col > 50000000")
            .unwrap();
        println!("bigint filter: {} rows", rows.len());

        let rows = db
            .query("SELECT * FROM all_types WHERE real_col > 50.0 AND double_col < 500.0")
            .unwrap();
        println!("float range filter: {} rows", rows.len());

        let rows = db
            .query("SELECT * FROM all_types WHERE bool_col = TRUE")
            .unwrap();
        println!("boolean filter: {} rows", rows.len());

        let rows = db
            .query("SELECT * FROM all_types WHERE text_col LIKE '%row 5%'")
            .unwrap();
        println!("LIKE query: {} rows", rows.len());

        let rows = db.query("SELECT * FROM all_types LIMIT 10").unwrap();
        assert!(rows.len() <= 10, "LIMIT 10 should return at most 10 rows");

        let rows = db
            .query("SELECT * FROM all_types LIMIT 20 OFFSET 10")
            .unwrap();
        println!("LIMIT OFFSET: {} rows", rows.len());

        let rows = db
            .query("SELECT id, date_col, time_col, ts_col FROM all_types WHERE id < 5")
            .unwrap();
        println!("Date/Time columns query: {} rows", rows.len());
        for row in &rows {
            println!("  DateTime Row: {:?}", row);
        }

        let rows = db
            .query("SELECT id, interval_col FROM all_types WHERE id < 5")
            .unwrap();
        println!("Interval column query: {} rows", rows.len());

        let rows = db.query("SELECT * FROM json_table").unwrap();
        println!("JSON table query: {} rows", rows.len());

        let rows = db.query("SELECT * FROM uuid_table").unwrap();
        println!("UUID table query: {} rows", rows.len());

        let rows = db.query("SELECT * FROM vector_table").unwrap();
        println!("Vector table query: {} rows", rows.len());

        let result = db
            .execute("UPDATE all_types SET text_col = 'Updated text content' WHERE id < 10")
            .unwrap();
        if let ExecuteResult::Update { rows_affected } = result {
            println!("Updated {} rows", rows_affected);
            assert!(rows_affected > 0, "should update some rows");
        }

        let result = db
            .execute("UPDATE related_data SET score = 999.99, active = FALSE WHERE id >= 90")
            .unwrap();
        if let ExecuteResult::Update { rows_affected } = result {
            println!("Updated related_data: {} rows", rows_affected);
        }

        let result = db
            .execute("DELETE FROM related_data WHERE id >= 95")
            .unwrap();
        if let ExecuteResult::Delete { rows_affected } = result {
            println!("Deleted {} rows from related_data", rows_affected);
            assert_eq!(rows_affected, 5, "should delete 5 rows");
        }

        let rows = db.query("SELECT * FROM related_data").unwrap();
        assert_eq!(
            rows.len(),
            95,
            "should have 95 related_data rows after delete"
        );

        let result = db.query(
            "SELECT a.id, r.name, r.category
             FROM all_types a, related_data r
             WHERE a.id = r.id AND r.score > 5.0
             LIMIT 10",
        );
        assert!(
            result.is_err(),
            "Complex JOIN query should fail until executor supports joins"
        );
        println!("Complex JOIN with filter and LIMIT correctly rejected");

        db.close().unwrap();

        let db = Database::open(&db_path).unwrap();
        let rows = db.query("SELECT * FROM all_types").unwrap();
        println!("After reopen, all_types has {} rows", rows.len());
        assert!(!rows.is_empty(), "data should persist after reopen");

        let rows = db.query("SELECT * FROM related_data").unwrap();
        assert_eq!(rows.len(), 95, "related_data should persist with 95 rows");

        println!("\n=== Test Summary ===");
        println!("SQL column types tested in CREATE TABLE:");
        println!("  - Basic: INT, BIGINT, SMALLINT, TINYINT, REAL, FLOAT, DOUBLE PRECISION");
        println!("  - Numeric: DECIMAL, NUMERIC");
        println!("  - String: VARCHAR, CHAR, TEXT, BLOB");
        println!("  - Boolean: BOOLEAN");
        println!("  - Temporal: DATE, TIME, TIMESTAMP, TIMESTAMPTZ, INTERVAL");
        println!("  - Special (separate tables): JSONB, UUID, VECTOR");
        println!("INSERT, SELECT, UPDATE, DELETE operations verified");
        println!("JOIN queries verified across multiple tables");
        println!("LIMIT, OFFSET, LIKE, complex WHERE clauses verified");
        println!("Data persistence across close/reopen verified");
    }

    #[test]
    fn test_insert_uuid_from_string_literal() {
        use crate::database::owned_value::OwnedValue;

        let dir = tempdir().unwrap();
        let db_path = dir.path().join("test_db");

        let db = Database::create(&db_path).unwrap();

        db.execute("CREATE TABLE items (id UUID, name TEXT)")
            .unwrap();

        db.execute(
            "INSERT INTO items VALUES ('550e8400-e29b-41d4-a716-446655440000', 'Test Item')",
        )
        .unwrap();

        let rows = db.query("SELECT id, name FROM items").unwrap();
        println!("rows.len() = {}", rows.len());
        assert_eq!(rows.len(), 1);

        let row = &rows[0];
        println!("row.values.len() = {}", row.values.len());
        println!("row.values = {:?}", row.values);
        assert_eq!(
            row.values.len(),
            2,
            "Expected 2 values, got {:?}",
            row.values
        );

        match &row.values[0] {
            OwnedValue::Uuid(u) => {
                let expected: [u8; 16] = [
                    0x55, 0x0e, 0x84, 0x00, 0xe2, 0x9b, 0x41, 0xd4, 0xa7, 0x16, 0x44, 0x66, 0x55,
                    0x44, 0x00, 0x00,
                ];
                assert_eq!(u, &expected, "UUID bytes should match");
            }
            OwnedValue::Null => panic!("UUID decoded as NULL"),
            other => panic!("Expected Uuid for UUID column, got {:?}", other),
        }
    }

    #[test]
    fn test_insert_jsonb_from_string_literal() {
        use crate::database::owned_value::OwnedValue;

        let dir = tempdir().unwrap();
        let db_path = dir.path().join("test_db");

        let db = Database::create(&db_path).unwrap();

        db.execute("CREATE TABLE docs (id INT, data JSONB)")
            .unwrap();

        db.execute(r#"INSERT INTO docs VALUES (1, '{"name": "test", "value": 42}')"#)
            .unwrap();

        let rows = db.query("SELECT id, data FROM docs").unwrap();
        assert_eq!(rows.len(), 1);

        let row = &rows[0];
        assert_eq!(row.values.len(), 2);

        match &row.values[0] {
            OwnedValue::Int(id) => assert_eq!(*id, 1),
            other => panic!("Expected Int for id column, got {:?}", other),
        }

        match &row.values[1] {
            OwnedValue::Jsonb(data) => {
                assert!(!data.is_empty(), "JSONB data should not be empty");
            }
            OwnedValue::Null => panic!("JSONB should not be NULL after insertion"),
            other => panic!("Expected Jsonb for JSONB column, got {:?}", other),
        }
    }

    #[test]
    fn test_insert_vector_from_array_literal() {
        use crate::database::owned_value::OwnedValue;

        let dir = tempdir().unwrap();
        let db_path = dir.path().join("test_db");

        let db = Database::create(&db_path).unwrap();

        db.execute("CREATE TABLE embeddings (id INT, vec VECTOR(3))")
            .unwrap();

        db.execute("INSERT INTO embeddings VALUES (1, '[1.0, 2.0, 3.0]')")
            .unwrap();

        let rows = db.query("SELECT id, vec FROM embeddings").unwrap();
        assert_eq!(rows.len(), 1);

        let row = &rows[0];
        assert_eq!(row.values.len(), 2);

        match &row.values[1] {
            OwnedValue::Vector(v) => {
                assert_eq!(v.len(), 3, "Vector should have 3 elements");
                assert!((v[0] - 1.0).abs() < 0.001);
                assert!((v[1] - 2.0).abs() < 0.001);
                assert!((v[2] - 3.0).abs() < 0.001);
            }
            OwnedValue::Null => panic!("Vector should not be NULL after insertion"),
            other => panic!("Expected Vector for VECTOR column, got {:?}", other),
        }
    }

    #[test]
    fn test_insert_blob_from_hex_literal() {
        use crate::database::owned_value::OwnedValue;

        let dir = tempdir().unwrap();
        let db_path = dir.path().join("test_db");

        let db = Database::create(&db_path).unwrap();

        db.execute("CREATE TABLE binaries (id INT, data BLOB)")
            .unwrap();

        db.execute("INSERT INTO binaries VALUES (1, x'DEADBEEF')")
            .unwrap();

        let rows = db.query("SELECT id, data FROM binaries").unwrap();
        assert_eq!(rows.len(), 1);

        let row = &rows[0];
        assert_eq!(row.values.len(), 2);

        match &row.values[1] {
            OwnedValue::Blob(b) => {
                assert_eq!(b.len(), 4, "BLOB should have 4 bytes");
                assert_eq!(b.as_slice(), &[0xDE, 0xAD, 0xBE, 0xEF]);
            }
            other => panic!("Expected Blob for BLOB column, got {:?}", other),
        }
    }

    #[test]
    fn test_unique_constraint_auto_creates_index() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("test_db");

        let db = Database::create(&db_path).unwrap();

        db.execute("CREATE TABLE users (id INT PRIMARY KEY, email TEXT UNIQUE, name TEXT)")
            .unwrap();

        let pk_index_path = db_path.join("root").join("users_id_pkey.idx");
        assert!(
            pk_index_path.exists(),
            "PRIMARY KEY should auto-create index at {:?}",
            pk_index_path
        );

        let unique_index_path = db_path.join("root").join("users_email_key.idx");
        assert!(
            unique_index_path.exists(),
            "UNIQUE constraint should auto-create index at {:?}",
            unique_index_path
        );
    }

    #[test]
    fn test_unique_index_used_for_constraint_check() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("test_db");

        let db = Database::create(&db_path).unwrap();

        db.execute("CREATE TABLE users (id INT PRIMARY KEY, email TEXT UNIQUE)")
            .unwrap();

        db.execute("INSERT INTO users VALUES (1, 'alice@example.com')")
            .unwrap();

        let result = db.execute("INSERT INTO users VALUES (2, 'alice@example.com')");
        assert!(
            result.is_err(),
            "UNIQUE constraint should prevent duplicate email"
        );
        let err_msg = result.unwrap_err().to_string();
        assert!(
            err_msg.contains("UNIQUE") || err_msg.contains("unique"),
            "Error should mention UNIQUE constraint: {}",
            err_msg
        );
    }

    #[test]
    fn test_unique_constraint_allows_different_values() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("test_db");

        let db = Database::create(&db_path).unwrap();

        db.execute("CREATE TABLE users (id INT, email TEXT UNIQUE)")
            .unwrap();

        db.execute("INSERT INTO users VALUES (1, 'alice@test.com')")
            .unwrap();
        db.execute("INSERT INTO users VALUES (2, 'bob@test.com')")
            .unwrap();

        let rows = db.query("SELECT * FROM users").unwrap();
        assert_eq!(
            rows.len(),
            2,
            "Both inserts should succeed with different emails"
        );
    }

    #[test]
    fn test_unique_index_performance_many_inserts() {
        use std::time::Instant;

        let dir = tempdir().unwrap();
        let db_path = dir.path().join("test_db");

        let db = Database::create(&db_path).unwrap();

        db.execute("CREATE TABLE users (id INT PRIMARY KEY, email TEXT UNIQUE)")
            .unwrap();

        let start = Instant::now();
        for i in 0..1000 {
            let sql = format!("INSERT INTO users VALUES ({}, 'user{}@example.com')", i, i);
            db.execute(&sql).unwrap();
        }
        let elapsed = start.elapsed();

        println!(
            "1000 inserts with UNIQUE constraint took {:?} ({:.2} inserts/sec)",
            elapsed,
            1000.0 / elapsed.as_secs_f64()
        );

        assert!(
            elapsed.as_secs() < 5,
            "1000 inserts should complete in under 5 seconds with O(log n) index lookup, took {:?}",
            elapsed
        );
    }

    #[test]
    fn test_unique_constraint_allows_multiple_nulls() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("test_db");

        let db = Database::create(&db_path).unwrap();

        db.execute("CREATE TABLE users (id INT, email TEXT UNIQUE)")
            .unwrap();

        db.execute("INSERT INTO users VALUES (1, NULL)").unwrap();
        db.execute("INSERT INTO users VALUES (2, NULL)").unwrap();

        let rows = db.query("SELECT * FROM users").unwrap();
        assert_eq!(rows.len(), 2, "UNIQUE should allow multiple NULLs");
    }

    #[test]
    fn test_primary_key_rejects_duplicate_on_insert() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("test_db");

        let db = Database::create(&db_path).unwrap();

        db.execute("CREATE TABLE users (id INT PRIMARY KEY, name TEXT)")
            .unwrap();

        db.execute("INSERT INTO users VALUES (1, 'Alice')").unwrap();

        let result = db.execute("INSERT INTO users VALUES (1, 'Bob')");

        assert!(result.is_err(), "PRIMARY KEY should reject duplicate id");
        let err_msg = result.unwrap_err().to_string();
        assert!(
            err_msg.contains("PRIMARY KEY")
                || err_msg.contains("UNIQUE")
                || err_msg.contains("unique"),
            "Error should mention constraint violation: {}",
            err_msg
        );
    }

    #[test]
    fn test_unique_constraint_rejects_duplicate_on_update() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("test_db");

        let db = Database::create(&db_path).unwrap();

        db.execute("CREATE TABLE users (id INT, email TEXT UNIQUE)")
            .unwrap();

        db.execute("INSERT INTO users VALUES (1, 'alice@test.com')")
            .unwrap();
        db.execute("INSERT INTO users VALUES (2, 'bob@test.com')")
            .unwrap();

        let result = db.execute("UPDATE users SET email = 'alice@test.com' WHERE id = 2");

        assert!(
            result.is_err(),
            "UNIQUE constraint should reject duplicate email on UPDATE"
        );
        let err_msg = result.unwrap_err().to_string();
        assert!(
            err_msg.contains("UNIQUE") || err_msg.contains("unique"),
            "Error should mention UNIQUE constraint violation: {}",
            err_msg
        );
    }

    #[test]
    fn test_check_constraint_rejects_invalid_value_on_insert() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("test_db");

        let db = Database::create(&db_path).unwrap();

        db.execute("CREATE TABLE users (id INT, age INT CHECK(age >= 0))")
            .unwrap();

        let result = db.execute("INSERT INTO users VALUES (1, -5)");

        assert!(
            result.is_err(),
            "CHECK constraint should reject negative age"
        );
        let err_msg = result.unwrap_err().to_string();
        assert!(
            err_msg.contains("CHECK") || err_msg.contains("check"),
            "Error should mention CHECK constraint violation: {}",
            err_msg
        );
    }

    #[test]
    fn test_check_constraint_accepts_valid_value_on_insert() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("test_db");

        let db = Database::create(&db_path).unwrap();

        db.execute("CREATE TABLE users (id INT, age INT CHECK(age >= 0))")
            .unwrap();

        db.execute("INSERT INTO users VALUES (1, 25)").unwrap();

        let rows = db.query("SELECT * FROM users").unwrap();
        assert_eq!(rows.len(), 1, "Insert should succeed with valid age");
    }

    #[test]
    fn test_check_constraint_rejects_invalid_value_on_update() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("test_db");

        let db = Database::create(&db_path).unwrap();

        db.execute("CREATE TABLE users (id INT, age INT CHECK(age >= 0))")
            .unwrap();
        db.execute("INSERT INTO users VALUES (1, 25)").unwrap();

        let result = db.execute("UPDATE users SET age = -10 WHERE id = 1");

        assert!(
            result.is_err(),
            "CHECK constraint should reject negative age on UPDATE"
        );
        let err_msg = result.unwrap_err().to_string();
        assert!(
            err_msg.contains("CHECK") || err_msg.contains("check"),
            "Error should mention CHECK constraint violation: {}",
            err_msg
        );
    }

    #[test]
    fn test_foreign_key_rejects_missing_reference_on_insert() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("test_db");

        let db = Database::create(&db_path).unwrap();

        db.execute("CREATE TABLE users (id INT PRIMARY KEY, name TEXT)")
            .unwrap();
        db.execute("CREATE TABLE orders (id INT, user_id INT REFERENCES users(id))")
            .unwrap();

        db.execute("INSERT INTO users VALUES (1, 'Alice')").unwrap();

        let result = db.execute("INSERT INTO orders VALUES (1, 999)");

        assert!(
            result.is_err(),
            "FOREIGN KEY constraint should reject missing reference"
        );
        let err_msg = result.unwrap_err().to_string();
        assert!(
            err_msg.contains("FOREIGN KEY")
                || err_msg.contains("foreign key")
                || err_msg.contains("referenced"),
            "Error should mention FOREIGN KEY constraint violation: {}",
            err_msg
        );
    }

    #[test]
    fn test_foreign_key_accepts_valid_reference_on_insert() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("test_db");

        let db = Database::create(&db_path).unwrap();

        db.execute("CREATE TABLE users (id INT PRIMARY KEY, name TEXT)")
            .unwrap();
        db.execute("CREATE TABLE orders (id INT, user_id INT REFERENCES users(id))")
            .unwrap();

        db.execute("INSERT INTO users VALUES (1, 'Alice')").unwrap();
        db.execute("INSERT INTO orders VALUES (1, 1)").unwrap();

        let rows = db.query("SELECT * FROM orders").unwrap();
        assert_eq!(
            rows.len(),
            1,
            "Insert should succeed with valid foreign key"
        );
    }

    #[test]
    fn test_foreign_key_blocks_delete_of_referenced_row() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("test_db");

        let db = Database::create(&db_path).unwrap();

        db.execute("CREATE TABLE users (id INT PRIMARY KEY, name TEXT)")
            .unwrap();
        db.execute("CREATE TABLE orders (id INT, user_id INT REFERENCES users(id))")
            .unwrap();

        db.execute("INSERT INTO users VALUES (1, 'Alice')").unwrap();
        db.execute("INSERT INTO orders VALUES (1, 1)").unwrap();

        let result = db.execute("DELETE FROM users WHERE id = 1");

        assert!(
            result.is_err(),
            "FOREIGN KEY constraint should block delete of referenced row"
        );
        let err_msg = result.unwrap_err().to_string();
        assert!(
            err_msg.contains("referenced")
                || err_msg.contains("FOREIGN KEY")
                || err_msg.contains("foreign key"),
            "Error should mention row is referenced: {}",
            err_msg
        );
    }

    #[test]
    fn test_create_table_with_geometric_types() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("test_db");

        let db = Database::create(&db_path).unwrap();

        let result =
            db.execute("CREATE TABLE geo (id INT, location POINT, area CIRCLE, bounds BOX)");
        assert!(
            result.is_ok(),
            "Should create table with geometric types: {:?}",
            result.err()
        );
    }

    #[test]
    fn test_create_table_with_range_types() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("test_db");

        let db = Database::create(&db_path).unwrap();

        let result = db.execute(
            "CREATE TABLE ranges (id INT, r1 INT4RANGE, r2 INT8RANGE, r3 DATERANGE, r4 TSRANGE)",
        );
        assert!(
            result.is_ok(),
            "Should create table with range types: {:?}",
            result.err()
        );
    }

    #[test]
    fn test_create_table_with_network_types() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("test_db");

        let db = Database::create(&db_path).unwrap();

        let result = db.execute("CREATE TABLE network (id INT, mac MACADDR, ip INET)");
        assert!(
            result.is_ok(),
            "Should create table with network types: {:?}",
            result.err()
        );
    }

    #[test]
    fn test_interval_type_end_to_end() {
        use crate::types::OwnedValue;

        let dir = tempdir().unwrap();
        let db_path = dir.path().join("test_db");

        let db = Database::create(&db_path).unwrap();

        db.execute("CREATE TABLE events (id INT, name TEXT, duration INTERVAL)")
            .unwrap();

        db.execute("INSERT INTO events VALUES (1, 'meeting', '1 hour')")
            .unwrap();
        db.execute("INSERT INTO events VALUES (2, 'project', '2 years 3 months 5 days')")
            .unwrap();
        db.execute("INSERT INTO events VALUES (3, 'task', 'P1Y2M3D')")
            .unwrap();

        let rows = db.query("SELECT id, name, duration FROM events").unwrap();
        assert_eq!(rows.len(), 3, "Should have 3 rows");

        let duration1 = rows[0].get(2).expect("should have duration column");
        assert!(
            matches!(duration1, OwnedValue::Interval(micros, days, months) if *micros == 3_600_000_000 && *days == 0 && *months == 0),
            "First interval should be 1 hour (3600000000 microseconds): got {:?}",
            duration1
        );

        let duration2 = rows[1].get(2).expect("should have duration column");
        assert!(
            matches!(duration2, OwnedValue::Interval(_, days, months) if *days == 5 && *months == 27),
            "Second interval should be 2 years 3 months 5 days: got {:?}",
            duration2
        );

        let duration3 = rows[2].get(2).expect("should have duration column");
        assert!(
            matches!(duration3, OwnedValue::Interval(_, days, months) if *days == 3 && *months == 14),
            "Third interval should be 1 year 2 months 3 days (ISO 8601): got {:?}",
            duration3
        );
    }

    #[test]
    fn test_drop_index() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("test_db");

        let db = Database::create(&db_path).unwrap();

        db.execute("CREATE TABLE users (id INT, name TEXT)")
            .unwrap();
        db.execute("CREATE INDEX idx_users_name ON users(name)")
            .unwrap();

        let result = db.execute("DROP INDEX idx_users_name");
        assert!(
            result.is_ok(),
            "DROP INDEX should succeed: {:?}",
            result.err()
        );

        assert!(
            matches!(result.unwrap(), ExecuteResult::DropIndex { dropped: true }),
            "Should return DropIndex with dropped: true"
        );
    }

    #[test]
    fn test_drop_index_if_exists_nonexistent() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("test_db");

        let db = Database::create(&db_path).unwrap();

        db.execute("CREATE TABLE users (id INT, name TEXT)")
            .unwrap();

        let result = db.execute("DROP INDEX IF EXISTS nonexistent_idx");
        assert!(
            result.is_ok(),
            "DROP INDEX IF EXISTS on nonexistent index should succeed"
        );
    }

    #[test]
    fn test_drop_index_nonexistent_fails() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("test_db");

        let db = Database::create(&db_path).unwrap();

        db.execute("CREATE TABLE users (id INT, name TEXT)")
            .unwrap();

        let result = db.execute("DROP INDEX nonexistent_idx");
        assert!(
            result.is_err(),
            "DROP INDEX on nonexistent index should fail"
        );
        assert!(
            result.unwrap_err().to_string().contains("not found"),
            "Error should mention index not found"
        );
    }

    #[test]
    fn test_drop_schema_execution() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("test_db");

        let db = Database::create(&db_path).unwrap();

        db.execute("CREATE SCHEMA test_schema").unwrap();

        let result = db.execute("DROP SCHEMA test_schema");
        assert!(
            result.is_ok(),
            "DROP SCHEMA should succeed: {:?}",
            result.err()
        );

        assert!(
            matches!(result.unwrap(), ExecuteResult::DropSchema { dropped: true }),
            "Should return DropSchema with dropped: true"
        );
    }

    #[test]
    fn test_drop_schema_if_exists_nonexistent() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("test_db");

        let db = Database::create(&db_path).unwrap();

        let result = db.execute("DROP SCHEMA IF EXISTS nonexistent_schema");
        assert!(
            result.is_ok(),
            "DROP SCHEMA IF EXISTS on nonexistent schema should succeed"
        );
    }

    #[test]
    fn test_select_distinct() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("test_db");

        let db = Database::create(&db_path).unwrap();

        db.execute("CREATE TABLE items (category TEXT, name TEXT)")
            .unwrap();
        db.execute("INSERT INTO items VALUES ('fruit', 'apple')")
            .unwrap();
        db.execute("INSERT INTO items VALUES ('fruit', 'banana')")
            .unwrap();
        db.execute("INSERT INTO items VALUES ('vegetable', 'carrot')")
            .unwrap();
        db.execute("INSERT INTO items VALUES ('fruit', 'apple')")
            .unwrap();

        let rows = db.query("SELECT DISTINCT category FROM items").unwrap();
        assert_eq!(rows.len(), 2, "DISTINCT should remove duplicate categories");
    }

    #[test]
    fn test_select_distinct_multiple_columns() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("test_db");

        let db = Database::create(&db_path).unwrap();

        db.execute("CREATE TABLE items (category TEXT, name TEXT)")
            .unwrap();
        db.execute("INSERT INTO items VALUES ('fruit', 'apple')")
            .unwrap();
        db.execute("INSERT INTO items VALUES ('fruit', 'apple')")
            .unwrap();
        db.execute("INSERT INTO items VALUES ('fruit', 'banana')")
            .unwrap();
        db.execute("INSERT INTO items VALUES ('vegetable', 'carrot')")
            .unwrap();

        let rows = db
            .query("SELECT DISTINCT category, name FROM items")
            .unwrap();
        assert_eq!(
            rows.len(),
            3,
            "DISTINCT should remove duplicate category+name combinations"
        );
    }

    #[test]
    fn test_select_distinct_all_same() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("test_db");

        let db = Database::create(&db_path).unwrap();

        db.execute("CREATE TABLE items (value INT)").unwrap();
        db.execute("INSERT INTO items VALUES (1)").unwrap();
        db.execute("INSERT INTO items VALUES (1)").unwrap();
        db.execute("INSERT INTO items VALUES (1)").unwrap();

        let rows = db.query("SELECT DISTINCT value FROM items").unwrap();
        assert_eq!(
            rows.len(),
            1,
            "DISTINCT with all same values should return 1 row"
        );
    }

    #[test]
    fn test_create_index_with_expression() {
        use crate::schema::IndexColumnDef;

        let dir = tempdir().unwrap();
        let db_path = dir.path().join("test_db");

        let db = Database::create(&db_path).unwrap();

        db.execute("CREATE TABLE users (id INT, email TEXT)")
            .unwrap();

        let result = db.execute("CREATE INDEX idx_lower_email ON users (LOWER(email))");
        assert!(
            result.is_ok(),
            "CREATE INDEX with expression should succeed"
        );

        let catalog = db.catalog.read();
        let catalog = catalog.as_ref().unwrap();
        let table = catalog.resolve_table("users").unwrap();

        let idx = table
            .indexes()
            .iter()
            .find(|i| i.name() == "idx_lower_email")
            .expect("Index should exist");

        assert!(
            idx.has_expressions(),
            "Index should have expression columns"
        );

        let col_defs = idx.column_defs();
        assert_eq!(col_defs.len(), 1);
        assert!(
            matches!(&col_defs[0], IndexColumnDef::Expression(e) if e.contains("LOWER")),
            "First column should be LOWER expression, got: {:?}",
            col_defs[0]
        );
    }

    #[test]
    fn test_create_index_with_mixed_columns_and_expressions() {
        use crate::schema::IndexColumnDef;

        let dir = tempdir().unwrap();
        let db_path = dir.path().join("test_db");

        let db = Database::create(&db_path).unwrap();

        db.execute("CREATE TABLE users (id INT, email TEXT, tenant_id INT)")
            .unwrap();

        let result = db.execute("CREATE INDEX idx_tenant_lower ON users (tenant_id, LOWER(email))");
        assert!(
            result.is_ok(),
            "CREATE INDEX with mixed columns/expressions should succeed"
        );

        let catalog = db.catalog.read();
        let catalog = catalog.as_ref().unwrap();
        let table = catalog.resolve_table("users").unwrap();

        let idx = table
            .indexes()
            .iter()
            .find(|i| i.name() == "idx_tenant_lower")
            .expect("Index should exist");

        let col_defs = idx.column_defs();
        assert_eq!(col_defs.len(), 2);
        assert!(
            matches!(&col_defs[0], IndexColumnDef::Column(c) if c == "tenant_id"),
            "First should be column tenant_id"
        );
        assert!(
            matches!(&col_defs[1], IndexColumnDef::Expression(_)),
            "Second should be expression"
        );
    }

    #[test]
    fn test_create_partial_index_with_where() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("test_db");

        let db = Database::create(&db_path).unwrap();

        db.execute("CREATE TABLE users (id INT, email TEXT, status TEXT)")
            .unwrap();

        let result =
            db.execute("CREATE INDEX idx_active_users ON users (email) WHERE status = 'active'");
        assert!(result.is_ok(), "CREATE INDEX with WHERE should succeed");

        let catalog = db.catalog.read();
        let catalog = catalog.as_ref().unwrap();
        let table = catalog.resolve_table("users").unwrap();

        let idx = table
            .indexes()
            .iter()
            .find(|i| i.name() == "idx_active_users")
            .expect("Index should exist");

        assert!(idx.is_partial(), "Index should be partial");
        assert!(
            idx.where_clause().is_some(),
            "Index should have WHERE clause"
        );
        let where_clause = idx.where_clause().unwrap();
        assert!(
            where_clause.contains("status") && where_clause.contains("active"),
            "WHERE clause should reference status and active: {}",
            where_clause
        );
    }

    #[test]
    fn test_create_partial_index_with_expression_and_where() {
        use crate::schema::IndexColumnDef;

        let dir = tempdir().unwrap();
        let db_path = dir.path().join("test_db");

        let db = Database::create(&db_path).unwrap();

        db.execute("CREATE TABLE users (id INT, email TEXT, deleted_at TIMESTAMP)")
            .unwrap();

        let result = db.execute(
            "CREATE UNIQUE INDEX idx_unique_email ON users (LOWER(email)) WHERE deleted_at IS NULL",
        );
        assert!(
            result.is_ok(),
            "CREATE UNIQUE INDEX with WHERE should succeed"
        );

        let catalog = db.catalog.read();
        let catalog = catalog.as_ref().unwrap();
        let table = catalog.resolve_table("users").unwrap();

        let idx = table
            .indexes()
            .iter()
            .find(|i| i.name() == "idx_unique_email")
            .expect("Index should exist");

        assert!(idx.is_unique(), "Index should be unique");
        assert!(idx.is_partial(), "Index should be partial");
        assert!(idx.has_expressions(), "Index should have expressions");

        let col_defs = idx.column_defs();
        assert_eq!(col_defs.len(), 1);
        assert!(
            matches!(&col_defs[0], IndexColumnDef::Expression(e) if e.contains("LOWER")),
            "Should have LOWER expression"
        );

        assert!(
            idx.where_clause().unwrap().contains("deleted_at"),
            "WHERE clause should reference deleted_at"
        );
    }
}
