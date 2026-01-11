//! # Optimization Rules
//!
//! Individual transformation rules for the query optimizer.
//!
//! ## Rule Categories
//!
//! | Rule | Purpose | Priority |
//! |------|---------|----------|
//! | ConstantFolding | Evaluate constants at plan time | 1 (first) |
//! | PredicatePushdown | Move filters to data sources | 2 |
//! | ProjectionPruning | Remove unused columns | 3 |
//! | SubqueryDecorrelation | Convert correlated subqueries to joins | 4 (last) |
//!
//! ## Rule Implementation Guidelines
//!
//! 1. Rules must be idempotent - applying twice should have no effect
//! 2. Rules return `None` if no transformation is possible
//! 3. Rules allocate new nodes in the provided arena
//! 4. Rules preserve query semantics (same results)

mod constant_folding;
mod decorrelate;
mod predicate_pushdown;
mod projection_pruning;

pub use constant_folding::ConstantFoldingRule;
pub use decorrelate::SubqueryDecorrelationRule;
pub use predicate_pushdown::PredicatePushdownRule;
pub use projection_pruning::ProjectionPruningRule;
