//! # TurDB CLI Module
//!
//! This module provides an interactive command-line interface for TurDB,
//! similar to the MySQL CLI. It supports:
//!
//! - Interactive SQL execution with query history
//! - ASCII table-formatted result display
//! - Dot commands for database introspection
//!
//! ## Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────┐
//! │                      CLI Entry Point                        │
//! │                      (bin/turdb.rs)                         │
//! ├─────────────────────────────────────────────────────────────┤
//! │                         REPL Loop                           │
//! │  - Reads input via rustyline                                │
//! │  - Dispatches to command handler or SQL executor            │
//! │  - Formats and displays results                             │
//! ├─────────────────────────────────────────────────────────────┤
//! │     Commands          │    Table Formatter    │   History   │
//! │  (.quit, .tables,     │  ASCII box drawing    │  Persistent │
//! │   .schema, .help)     │  for query results    │  ~/.turdb_* │
//! └─────────────────────────────────────────────────────────────┘
//! ```
//!
//! ## Usage
//!
//! ```bash
//! # Open existing database
//! turdb ./mydb
//!
//! # Create new database
//! turdb --create ./newdb
//!
//! # Show version
//! turdb --version
//! ```
//!
//! ## Dot Commands
//!
//! The CLI supports SQLite-style dot commands:
//!
//! | Command              | Description                              |
//! |----------------------|------------------------------------------|
//! | `.quit` / `.exit`    | Exit the CLI                             |
//! | `.tables`            | List all tables in current schema        |
//! | `.schema [table]`    | Show CREATE statement for table(s)       |
//! | `.indexes [table]`   | List indexes, optionally for a table     |
//! | `.help`              | Show available commands                  |
//!
//! ## Table Display
//!
//! Query results are displayed in ASCII box format:
//!
//! ```text
//! +----+-------+-----+
//! | id | name  | age |
//! +----+-------+-----+
//! |  1 | Alice |  30 |
//! |  2 | Bob   |  25 |
//! +----+-------+-----+
//! 2 rows in set
//! ```
//!
//! ## History
//!
//! Command history is persisted to `~/.turdb_history` by default.
//! This can be overridden with the `TURDB_HISTORY` environment variable.
//!
//! ## Module Organization
//!
//! - `repl`: Main read-eval-print loop with rustyline integration
//! - `commands`: Dot command parsing and execution
//! - `table`: ASCII table formatter for query results
//! - `history`: History file path resolution and management

pub mod commands;
pub mod history;
pub mod repl;
pub mod table;

pub use repl::Repl;
