//! # Optimization Rules
//!
//! Individual transformation rules for the query optimizer.
//!
//! ## Rule Categories
//!
//! | Rule | Purpose | Priority |
//! |------|---------|----------|
//! | ConstantFolding | Evaluate constants at plan time | 1 (first) |
//! | JoinConditionExtraction | Extract join conditions from WHERE | 2 |
//! | JoinReordering | Reorder joins by cardinality | 3 |
//! | PredicatePushdown | Move filters to data sources | 4 |
//! | ProjectionPruning | Remove unused columns | 5 |
//! | SubqueryDecorrelation | Convert correlated subqueries to joins | 6 (last) |
//!
//! ## Rule Implementation Guidelines
//!
//! 1. Rules must be idempotent - applying twice should have no effect
//! 2. Rules return `None` if no transformation is possible
//! 3. Rules allocate new nodes in the provided arena
//! 4. Rules preserve query semantics (same results)

mod constant_folding;
mod decorrelate;
mod join_condition_extraction;
mod join_reordering;
mod predicate_pushdown;
mod projection_pruning;

pub use constant_folding::ConstantFoldingRule;
pub use decorrelate::SubqueryDecorrelationRule;
pub use join_condition_extraction::JoinConditionExtractionRule;
pub use join_reordering::JoinReorderingRule;
pub use predicate_pushdown::PredicatePushdownRule;
pub use projection_pruning::ProjectionPruningRule;
