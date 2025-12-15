//! # TurDB - High-Performance Embedded Database
//!
//! TurDB is an embedded database combining SQLite-inspired row storage with
//! native HNSW vector search. This Rust implementation prioritizes:
//!
//! - **Zero-copy data access**: Direct mmap slices, no intermediate buffers
//! - **Zero allocation during CRUD**: Pre-allocated buffers, arena allocators
//! - **Extreme memory efficiency**: 1MB minimum RAM for basic operations
//!
//! ## Quick Start
//!
//! ```ignore
//! use turdb::Database;
//!
//! let db = Database::builder()
//!     .path("./mydb")
//!     .page_cache_size(64)
//!     .open()?;
//!
//! db.execute("CREATE TABLE users (id INT PRIMARY KEY, name TEXT)")?;
//! db.execute("INSERT INTO users VALUES (1, 'Alice')")?;
//!
//! let rows = db.query("SELECT * FROM users WHERE id = ?")
//!     .bind(1)
//!     .fetch_all()?;
//! ```
//!
//! ## Architecture
//!
//! TurDB uses a layered architecture:
//!
//! ```text
//! ┌─────────────────────────────────────┐
//! │         Public API (Database)        │
//! ├─────────────────────────────────────┤
//! │     SQL Layer (Parser/Executor)      │
//! ├─────────────────────────────────────┤
//! │  Schema & Catalog │ MVCC Transactions│
//! ├───────────────────┼─────────────────┤
//! │   B-Tree Index    │   HNSW Index     │
//! ├─────────────────────────────────────┤
//! │     Record Serialization Layer       │
//! ├─────────────────────────────────────┤
//! │      Storage Layer (Pager/Cache)     │
//! ├─────────────────────────────────────┤
//! │    Memory-Mapped File I/O + WAL      │
//! └─────────────────────────────────────┘
//! ```
//!
//! ## File Layout
//!
//! TurDB uses MySQL-style file-per-table architecture:
//!
//! ```text
//! database_dir/
//! ├── turdb.meta           # Global metadata and catalog
//! ├── root/                # Default schema
//! │   ├── table_name.tbd   # Table data file
//! │   ├── table_name.idx   # B-tree indexes
//! │   └── table_name.hnsw  # HNSW vector indexes
//! └── wal/
//!     └── wal.000001       # Write-ahead log segments
//! ```
//!
//! ## Performance Targets
//!
//! - Point read: < 1µs (cached), < 50µs (disk)
//! - Sequential scan: > 1M rows/sec
//! - Insert: > 100K rows/sec
//! - k-NN search (1M vectors, k=10): < 10ms
//!
//! ## Module Overview
//!
//! - [`storage`]: Memory-mapped storage, page cache, freelist
//! - `btree`: B+tree index with slot arrays and prefix hints
//! - `hnsw`: HNSW vector index with SIMD distance functions
//! - `record`: Zero-copy row serialization
//! - `schema`: Multi-schema catalog management
//! - `sql`: Lexer, parser, planner, executor
//! - `transaction`: MVCC and WAL

#[macro_use]
mod macros;

pub mod btree;
pub mod constraints;
pub mod database;
pub mod encoding;
pub mod hnsw;
pub mod mvcc;
pub mod parsing;
pub mod records;
pub mod schema;
pub mod sql;
pub mod storage;
pub mod types;

pub use database::{CheckpointInfo, Database, ExecuteResult, OwnedValue, RecoveryInfo, Row};
