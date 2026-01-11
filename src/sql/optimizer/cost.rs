//! # Cost Estimation Module
//!
//! This module provides cost estimation functions for query optimization.
//! The cost model is heuristic-based and considers I/O costs, CPU costs,
//! and selectivity estimates.
//!
//! ## Cost Model Parameters
//!
//! - Page size: 16KB
//! - Average row size: 100 bytes
//! - I/O cost per page: 1.0
//! - CPU cost per row: 0.01
//!
//! ## Selectivity Estimates
//!
//! - Equality selectivity: 1% (0.01)
//! - Unique index selectivity: 0.1% (0.001)
//! - Filter selectivity: 10% (0.1)
//! - Join selectivity: 10% (0.1)
//!
//! ## Usage
//!
//! ```ignore
//! let cost_estimator = CostEstimator::new(catalog, arena);
//! let index_cost = cost_estimator.estimate_index_access_cost(index, &columns);
//! let scan_cost = cost_estimator.estimate_table_scan_cost(1000);
//! ```

use crate::schema::{Catalog, IndexDef, TableDef};
use crate::sql::planner::{LogicalOperator, SetOpKind};
use bumpalo::Bump;

pub struct CostEstimator<'a> {
    #[allow(dead_code)]
    catalog: &'a Catalog,
    #[allow(dead_code)]
    arena: &'a Bump,
}

impl<'a> CostEstimator<'a> {
    pub fn new(catalog: &'a Catalog, arena: &'a Bump) -> Self {
        Self { catalog, arena }
    }

    pub fn estimate_index_access_cost(&self, index: &IndexDef, filter_columns: &[&str]) -> f64 {
        const TABLE_CARDINALITY: f64 = 1000.0;
        const PAGE_SIZE: f64 = 16384.0;
        const AVG_ROW_SIZE: f64 = 100.0;
        const IO_COST_PER_PAGE: f64 = 1.0;
        const CPU_COST_PER_ROW: f64 = 0.01;

        let matched_columns: usize = index
            .columns()
            .filter(|c| filter_columns.contains(c))
            .count();

        let selectivity = self.estimate_index_selectivity(index, matched_columns);
        let estimated_rows = (TABLE_CARDINALITY * selectivity).max(1.0);

        let rows_per_page = (PAGE_SIZE / AVG_ROW_SIZE).max(1.0);
        let data_pages = (estimated_rows / rows_per_page).ceil();

        let index_height = 3.0;
        let index_io = index_height + (estimated_rows / rows_per_page).ceil().min(10.0);

        let total_io = index_io + data_pages;
        let io_cost = total_io * IO_COST_PER_PAGE;
        let cpu_cost = estimated_rows * CPU_COST_PER_ROW;

        io_cost + cpu_cost
    }

    pub fn estimate_index_selectivity(&self, index: &IndexDef, matched_columns: usize) -> f64 {
        const EQUALITY_SELECTIVITY: f64 = 0.01;
        const UNIQUE_SELECTIVITY: f64 = 0.001;

        if matched_columns == 0 {
            return 1.0;
        }

        if index.is_unique() && matched_columns >= index.columns().count() {
            return UNIQUE_SELECTIVITY;
        }

        let base_selectivity = if matched_columns == 1 {
            EQUALITY_SELECTIVITY
        } else {
            EQUALITY_SELECTIVITY
                .powf(matched_columns as f64)
                .max(UNIQUE_SELECTIVITY)
        };

        if index.is_unique() {
            base_selectivity * 0.5
        } else {
            base_selectivity
        }
    }

    pub fn estimate_table_scan_cost(&self, table_cardinality: u64) -> f64 {
        const PAGE_SIZE: f64 = 16384.0;
        const AVG_ROW_SIZE: f64 = 100.0;
        const IO_COST_PER_PAGE: f64 = 1.0;
        const CPU_COST_PER_ROW: f64 = 0.01;

        let rows_per_page = (PAGE_SIZE / AVG_ROW_SIZE).max(1.0);
        let total_pages = (table_cardinality as f64 / rows_per_page).ceil();

        let io_cost = total_pages * IO_COST_PER_PAGE;
        let cpu_cost = table_cardinality as f64 * CPU_COST_PER_ROW;

        io_cost + cpu_cost
    }

    pub fn should_use_index(
        &self,
        index: &IndexDef,
        filter_columns: &[&str],
        table_cardinality: u64,
    ) -> bool {
        let index_cost = self.estimate_index_access_cost(index, filter_columns);
        let scan_cost = self.estimate_table_scan_cost(table_cardinality);
        index_cost < scan_cost
    }

    pub fn estimate_cardinality(&self, op: &LogicalOperator<'a>) -> u64 {
        const DEFAULT_TABLE_CARDINALITY: u64 = 1000;
        const FILTER_SELECTIVITY: f64 = 0.1;
        const JOIN_SELECTIVITY: f64 = 0.1;

        match op {
            LogicalOperator::Scan(_) => DEFAULT_TABLE_CARDINALITY,
            LogicalOperator::DualScan => 1,
            LogicalOperator::Filter(filter) => {
                let input_card = self.estimate_cardinality(filter.input);
                ((input_card as f64) * FILTER_SELECTIVITY).max(1.0) as u64
            }
            LogicalOperator::Project(project) => self.estimate_cardinality(project.input),
            LogicalOperator::Aggregate(agg) => {
                let input_card = self.estimate_cardinality(agg.input);
                if agg.group_by.is_empty() {
                    1
                } else {
                    (input_card / 10).max(1)
                }
            }
            LogicalOperator::Join(join) => {
                let left_card = self.estimate_cardinality(join.left);
                let right_card = self.estimate_cardinality(join.right);
                if join.condition.is_some() {
                    ((left_card as f64 * right_card as f64 * JOIN_SELECTIVITY) as u64).max(1)
                } else {
                    left_card * right_card
                }
            }
            LogicalOperator::Sort(sort) => self.estimate_cardinality(sort.input),
            LogicalOperator::Limit(limit) => {
                let input_card = self.estimate_cardinality(limit.input);
                match (limit.limit, limit.offset) {
                    (Some(l), Some(o)) => input_card.saturating_sub(o).min(l),
                    (Some(l), None) => input_card.min(l),
                    (None, Some(o)) => input_card.saturating_sub(o),
                    (None, None) => input_card,
                }
            }
            LogicalOperator::Values(values) => values.rows.len() as u64,
            LogicalOperator::Insert(_) => 0,
            LogicalOperator::Update(_) => 0,
            LogicalOperator::Delete(_) => 0,
            LogicalOperator::Subquery(subq) => self.estimate_cardinality(subq.plan),
            LogicalOperator::SetOp(set_op) => {
                let left_card = self.estimate_cardinality(set_op.left);
                let right_card = self.estimate_cardinality(set_op.right);
                match set_op.kind {
                    SetOpKind::Union => {
                        if set_op.all {
                            left_card + right_card
                        } else {
                            left_card + right_card / 2
                        }
                    }
                    SetOpKind::Intersect => left_card.min(right_card) / 2,
                    SetOpKind::Except => left_card / 2,
                }
            }
            LogicalOperator::Window(window) => self.estimate_cardinality(window.input),
        }
    }
}

pub fn select_best_index<'a>(
    table: &'a TableDef,
    filter_columns: &[&str],
    catalog: &'a Catalog,
    arena: &'a Bump,
) -> Option<&'a IndexDef> {
    let cost_estimator = CostEstimator::new(catalog, arena);
    let candidates = find_applicable_indexes(table, filter_columns);

    if candidates.is_empty() {
        return None;
    }

    candidates.into_iter().min_by(|a, b| {
        let cost_a = cost_estimator.estimate_index_access_cost(a, filter_columns);
        let cost_b = cost_estimator.estimate_index_access_cost(b, filter_columns);
        cost_a
            .partial_cmp(&cost_b)
            .unwrap_or(std::cmp::Ordering::Equal)
    })
}

pub fn find_applicable_indexes<'a>(
    table: &'a TableDef,
    filter_columns: &[&str],
) -> Vec<&'a IndexDef> {
    table
        .indexes()
        .iter()
        .filter(|idx| {
            idx.columns()
                .next()
                .map(|first_col| filter_columns.contains(&first_col))
                .unwrap_or(false)
        })
        .collect()
}
