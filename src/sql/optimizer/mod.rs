//! # Query Optimizer Module
//!
//! This module implements TurDB's rule-based query optimizer, transforming logical
//! plans into more efficient equivalent plans through a series of rewrite rules.
//!
//! ## Architecture
//!
//! The optimizer uses a multi-pass approach where each pass applies one category
//! of optimization rules:
//!
//! ```text
//! Logical Plan → [Pass 1: Fold] → [Pass 2: Pushdown] → [Pass 3: Prune] → [Pass 4: Decorrelate] → Optimized Plan
//! ```
//!
//! ## Optimization Categories
//!
//! | Category | Rules | Purpose |
//! |----------|-------|---------|
//! | Constant Folding | fold | Evaluate constant expressions at plan time |
//! | Predicate Pushdown | pushdown | Move filters closer to data sources |
//! | Projection Pruning | prune | Remove unused columns early |
//! | Subquery Decorrelation | decorrelate | Convert correlated subqueries to joins |
//!
//! ## Rule Application Strategy
//!
//! Rules are applied in a fixed-point iteration until no more changes occur:
//!
//! 1. Apply all rules in order
//! 2. If any rule modified the plan, repeat from step 1
//! 3. Stop when no rule modifies the plan or max iterations reached
//!
//! ## Memory Model
//!
//! The optimizer operates on arena-allocated plans. When a rule transforms a plan,
//! it allocates new nodes in the same arena. Old nodes are not deallocated but
//! become unreachable.
//!
//! ## Cost Model Integration
//!
//! This optimizer is rule-based (heuristic). Cost-based optimization is handled
//! separately in the physical planner which selects access paths and join algorithms.
//!
//! ## Usage
//!
//! ```ignore
//! use turdb::sql::optimizer::Optimizer;
//!
//! let optimizer = Optimizer::new();
//! let optimized = optimizer.optimize(logical_plan, arena)?;
//! ```

pub mod rules;

use crate::sql::planner::LogicalOperator;
use bumpalo::Bump;
use eyre::Result;

pub trait OptimizationRule {
    fn name(&self) -> &'static str;

    fn apply<'a>(
        &self,
        plan: &'a LogicalOperator<'a>,
        arena: &'a Bump,
    ) -> Result<Option<&'a LogicalOperator<'a>>>;
}

pub struct Optimizer {
    rules: Vec<Box<dyn OptimizationRule + Send + Sync>>,
    max_iterations: usize,
}

impl Optimizer {
    pub fn new() -> Self {
        Self {
            rules: vec![
                Box::new(rules::ConstantFoldingRule),
                Box::new(rules::PredicatePushdownRule),
                Box::new(rules::ProjectionPruningRule),
                Box::new(rules::SubqueryDecorrelationRule),
            ],
            max_iterations: 10,
        }
    }

    pub fn with_rules(rules: Vec<Box<dyn OptimizationRule + Send + Sync>>) -> Self {
        Self {
            rules,
            max_iterations: 10,
        }
    }

    pub fn with_max_iterations(mut self, max: usize) -> Self {
        self.max_iterations = max;
        self
    }

    pub fn optimize<'a>(
        &self,
        plan: &'a LogicalOperator<'a>,
        arena: &'a Bump,
    ) -> Result<&'a LogicalOperator<'a>> {
        let mut current = plan;

        for iteration in 0..self.max_iterations {
            let mut changed = false;

            for rule in &self.rules {
                if let Some(new_plan) = rule.apply(current, arena)? {
                    current = new_plan;
                    changed = true;
                }
            }

            if !changed {
                break;
            }

            if iteration == self.max_iterations - 1 {
                eprintln!(
                    "[warn] optimizer reached max iterations ({}), stopping",
                    self.max_iterations
                );
            }
        }

        Ok(current)
    }

    pub fn add_rule(&mut self, rule: Box<dyn OptimizationRule + Send + Sync>) {
        self.rules.push(rule);
    }
}

impl Default for Optimizer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct NoOpRule;

    impl OptimizationRule for NoOpRule {
        fn name(&self) -> &'static str {
            "noop"
        }

        fn apply<'a>(
            &self,
            _plan: &'a LogicalOperator<'a>,
            _arena: &'a Bump,
        ) -> Result<Option<&'a LogicalOperator<'a>>> {
            Ok(None)
        }
    }

    #[test]
    fn test_optimizer_no_changes() {
        let optimizer = Optimizer::with_rules(vec![Box::new(NoOpRule)]);
        let arena = Bump::new();
        let plan = arena.alloc(LogicalOperator::DualScan);

        let result = optimizer.optimize(plan, &arena).unwrap();
        assert!(std::ptr::eq(result, plan));
    }

    #[test]
    fn test_optimizer_default() {
        let optimizer = Optimizer::default();
        assert_eq!(optimizer.max_iterations, 10);
        assert_eq!(optimizer.rules.len(), 4);
    }
}
