//! # SQL Processing Module
//!
//! This module provides SQL parsing and execution capabilities for TurDB.
//! The implementation follows a zero-copy design where possible, with tokens
//! and AST nodes borrowing from the original input string.
//!
//! ## Module Structure
//!
//! - `token`: Token and keyword definitions
//! - `lexer`: Zero-copy SQL tokenizer
//! - `parser`: Recursive descent parser
//! - `planner`: Query planning
//! - `executor`: Query execution
//!
//! ## Design Philosophy
//!
//! The SQL layer prioritizes:
//!
//! 1. **Zero-copy parsing**: Tokens borrow from input, no string allocation
//! 2. **Arena allocation**: AST nodes allocated in bump allocator
//! 3. **Streaming execution**: Volcano model with pull-based iteration
//! 4. **Memory efficiency**: Fits within 256KB query budget
//!
//! ## Supported SQL
//!
//! TurDB supports a PostgreSQL-compatible SQL dialect including:
//!
//! - DDL: CREATE/ALTER/DROP TABLE, INDEX, SCHEMA, VIEW, FUNCTION, PROCEDURE
//! - DML: SELECT, INSERT, UPDATE, DELETE, MERGE/UPSERT
//! - Queries: Subqueries, CTEs, window functions, set operations
//! - Transactions: BEGIN, COMMIT, ROLLBACK, SAVEPOINT
//! - Analysis: EXPLAIN, ANALYZE
//!
//! ## Example
//!
//! ```ignore
//! use turdb::sql::{Lexer, Token, Keyword};
//!
//! let sql = "SELECT id, name FROM users WHERE active = true";
//! let mut lexer = Lexer::new(sql);
//!
//! while let token = lexer.next_token() {
//!     if matches!(token, Token::Eof) { break; }
//!     println!("{:?}", token);
//! }
//! ```

pub mod adapter;
pub mod ast;
pub mod builder;
pub mod context;
pub mod decoder;
pub mod executor;
pub mod expr;
pub mod functions;
pub mod lexer;
pub mod mvcc_scan;
pub mod optimizer;
pub mod parser;
pub mod partition_spiller;
pub mod planner;
pub mod predicate;
pub mod row_serde;
pub mod state;
pub mod subquery;
pub mod token;
pub mod util;

pub use ast::*;
pub use lexer::Lexer;
pub use mvcc_scan::{check_row_visibility, strip_mvcc_header, MvccStreamingSource};
pub use parser::{ParseError, Parser};
pub use token::{Keyword, Parameter, Span, Token};
