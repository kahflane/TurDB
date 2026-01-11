//! # Subquery Processing Module
//!
//! This module provides comprehensive subquery support for TurDB, including
//! classification, context management, and execution strategies.
//!
//! ## Subquery Types
//!
//! TurDB supports all standard SQL subquery patterns:
//!
//! | Type | Example | Execution Strategy |
//! |------|---------|-------------------|
//! | Scalar | `SELECT (SELECT MAX(x) FROM t)` | Execute once, cache |
//! | Derived | `FROM (SELECT ...) AS sub` | Materialize or stream |
//! | EXISTS | `WHERE EXISTS (SELECT ...)` | Short-circuit |
//! | IN | `WHERE x IN (SELECT ...)` | Hash set lookup |
//! | ANY/ALL | `WHERE x > ANY (SELECT ...)` | Quantified comparison |
//! | Lateral | `FROM t1, LATERAL (SELECT ...)` | Per-row execution |
//!
//! ## Correlation Detection
//!
//! Subqueries may reference columns from outer queries (correlated subqueries).
//! The classifier detects these references and tracks them for:
//!
//! 1. **Decorrelation**: Converting correlated subqueries to joins when possible
//! 2. **Execution planning**: Choosing between caching and re-execution
//! 3. **Context binding**: Providing outer row values during execution
//!
//! ## Memory Management
//!
//! Subquery results respect the 256KB query memory budget:
//!
//! - Uncorrelated subqueries: Execute once, materialize result
//! - Large results: Spill to temporary files via `SpillableBuffer`
//! - Correlated subqueries: Stream results, re-execute per outer row
//!
//! ## Architecture
//!
//! ```text
//! ┌─────────────────┐
//! │ SubqueryClassifier │  Analyzes AST, detects type and correlation
//! └────────┬────────┘
//!          │
//!          ▼
//! ┌─────────────────┐
//! │ SubqueryContext │  Tracks outer scope for correlated execution
//! └────────┬────────┘
//!          │
//!          ▼
//! ┌─────────────────┐
//! │ SubqueryExecutor│  Executes with appropriate strategy
//! └─────────────────┘
//! ```
//!
//! ## Usage
//!
//! ```ignore
//! use turdb::sql::subquery::{SubqueryClassifier, SubqueryType, SubqueryInfo};
//!
//! let classifier = SubqueryClassifier::new(&outer_tables);
//! let info = classifier.classify(subquery_expr)?;
//!
//! match info.subquery_type {
//!     SubqueryType::Scalar => { /* handle scalar */ }
//!     SubqueryType::Exists { negated } => { /* handle EXISTS */ }
//!     // ...
//! }
//! ```

mod classifier;
mod context;
pub mod spill;

pub use classifier::{
    CorrelationRef, SubqueryClassifier, SubqueryInfo, SubqueryPosition, SubqueryType,
};
pub use context::SubqueryContext;
pub use spill::{MaterializedRow, SpillableBuffer};
