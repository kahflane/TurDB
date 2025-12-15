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
pub mod row;

pub use database::Database;
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
    Begin,
    Commit,
    Rollback,
    Savepoint { name: String },
    Release { name: String },
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
