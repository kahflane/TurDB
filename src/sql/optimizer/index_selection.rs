//! # Index Selection Module
//!
//! This module provides utilities for selecting optimal indexes for query execution.
//! It analyzes predicates and sort requirements to determine when index scans
//! can be used instead of table scans.
//!
//! ## Index Selection Criteria
//!
//! - Column matching: Index prefix must match filter columns
//! - Partial index compatibility: Query predicate must imply index WHERE clause
//! - Cost comparison: Index scan must be cheaper than table scan
//!
//! ## Supported Optimizations
//!
//! - Filter to index scan: Equality predicates on indexed columns
//! - Sort to index scan: ORDER BY on indexed columns
//! - TopK optimization: LIMIT with ORDER BY using index order
//!
//! ## Usage
//!
//! ```ignore
//! let selector = IndexSelector::new(catalog, arena);
//! if let Some(index) = selector.select_best_index(table, &columns) {
//!     // Use index scan
//! }
//! ```

use crate::schema::{Catalog, IndexDef, IndexType, TableDef};
use crate::sql::ast::{BinaryOperator, Expr};
use crate::sql::planner::{
    LogicalFilter, LogicalOperator, LogicalProject, LogicalScan, LogicalSort,
    PhysicalFilterExec, PhysicalOperator, PhysicalProjectExec, PhysicalSecondaryIndexScan,
    PhysicalTableScan, ScanRange,
};
use bumpalo::Bump;
use super::cost::CostEstimator;

pub struct IndexSelector<'a> {
    catalog: &'a Catalog,
    arena: &'a Bump,
}

impl<'a> IndexSelector<'a> {
    pub fn new(catalog: &'a Catalog, arena: &'a Bump) -> Self {
        Self { catalog, arena }
    }

    pub fn find_applicable_indexes(
        &self,
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

    pub fn find_applicable_indexes_with_predicate(
        &self,
        table: &'a TableDef,
        filter_columns: &[&str],
        query_predicate: Option<&str>,
    ) -> Vec<&'a IndexDef> {
        table
            .indexes()
            .iter()
            .filter(|idx| {
                let column_matches = idx
                    .columns()
                    .next()
                    .map(|first_col| filter_columns.contains(&first_col))
                    .unwrap_or(false);

                if !column_matches {
                    return false;
                }

                if let Some(where_clause) = idx.where_clause() {
                    match query_predicate {
                        Some(pred) => predicate_implies_where_clause(pred, where_clause),
                        None => false,
                    }
                } else {
                    true
                }
            })
            .collect()
    }

    pub fn select_best_index(
        &self,
        table: &'a TableDef,
        filter_columns: &[&str],
    ) -> Option<&'a IndexDef> {
        let candidates = self.find_applicable_indexes(table, filter_columns);
        if candidates.is_empty() {
            return None;
        }

        let cost_estimator = CostEstimator::new(self.catalog, self.arena);
        candidates.into_iter().min_by(|a, b| {
            let cost_a = cost_estimator.estimate_index_access_cost(a, filter_columns);
            let cost_b = cost_estimator.estimate_index_access_cost(b, filter_columns);
            cost_a
                .partial_cmp(&cost_b)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
    }

    pub fn try_optimize_filter_to_index_scan(
        &self,
        filter: &LogicalFilter<'a>,
        encode_fn: impl Fn(&Expr<'a>) -> Option<&'a [u8]>,
    ) -> Option<&'a PhysicalOperator<'a>> {
        fn find_scan<'b>(op: &'b LogicalOperator<'b>) -> Option<&'b LogicalScan<'b>> {
            match op {
                LogicalOperator::Scan(scan) => Some(scan),
                LogicalOperator::Project(proj) => find_scan(proj.input),
                _ => None,
            }
        }

        fn find_project<'b>(op: &'b LogicalOperator<'b>) -> Option<&'b LogicalProject<'b>> {
            match op {
                LogicalOperator::Project(proj) => Some(proj),
                _ => None,
            }
        }

        let scan = find_scan(filter.input)?;
        let table_def = self.catalog.resolve_table(scan.table).ok()?;

        let (col_name, literal_expr) = extract_equality_predicate(filter.predicate)?;

        let matching_index = table_def.indexes().iter().find(|idx| {
            if idx.has_expressions() || idx.is_partial() {
                return false;
            }
            if idx.index_type() != IndexType::BTree {
                return false;
            }
            idx.columns()
                .next()
                .map(|first_col| first_col.eq_ignore_ascii_case(col_name))
                .unwrap_or(false)
        })?;

        let key_bytes = encode_fn(literal_expr)?;

        let index_name = self.arena.alloc_str(matching_index.name());
        let table_def_alloc = self.arena.alloc(table_def.clone());

        let covered_columns = vec![col_name.to_string()];
        let residual = compute_residual_filter(self.arena, filter.predicate, &covered_columns);

        let index_scan = self.arena.alloc(PhysicalOperator::SecondaryIndexScan(
            PhysicalSecondaryIndexScan {
                schema: scan.schema,
                table: scan.table,
                index_name,
                table_def: Some(table_def_alloc),
                reverse: false,
                is_unique_index: matching_index.is_unique(),
                key_range: Some(ScanRange::PrefixScan { prefix: key_bytes }),
                limit: None,
            },
        ));

        let with_residual = if let Some(residual_predicate) = residual {
            self.arena.alloc(PhysicalOperator::FilterExec(PhysicalFilterExec {
                input: index_scan,
                predicate: residual_predicate,
            }))
        } else {
            index_scan
        };

        let project = find_project(filter.input);
        if let Some(proj) = project {
            let physical_proj = self.arena.alloc(PhysicalOperator::ProjectExec(PhysicalProjectExec {
                input: with_residual,
                expressions: proj.expressions,
                aliases: proj.aliases,
            }));
            return Some(physical_proj);
        }

        Some(with_residual)
    }

    pub fn try_optimize_sort_to_index_scan(
        &self,
        sort: &LogicalSort<'a>,
    ) -> Option<&'a PhysicalOperator<'a>> {
        self.try_optimize_sort_to_index_scan_with_limit(sort, None)
    }

    pub fn try_optimize_sort_to_index_scan_with_limit(
        &self,
        sort: &LogicalSort<'a>,
        limit: Option<usize>,
    ) -> Option<&'a PhysicalOperator<'a>> {
        if sort.order_by.len() != 1 {
            return None;
        }

        let sort_key = &sort.order_by[0];

        let col_name = match sort_key.expr {
            Expr::Column(col_ref) => col_ref.column,
            _ => return None,
        };

        fn find_scan<'b>(op: &'b LogicalOperator<'b>) -> Option<&'b LogicalScan<'b>> {
            match op {
                LogicalOperator::Scan(scan) => Some(scan),
                LogicalOperator::Project(proj) => find_scan(proj.input),
                LogicalOperator::Filter(filter) => find_scan(filter.input),
                _ => None,
            }
        }

        fn has_filter(op: &LogicalOperator<'_>) -> bool {
            match op {
                LogicalOperator::Filter(_) => true,
                LogicalOperator::Project(proj) => has_filter(proj.input),
                _ => false,
            }
        }

        if has_filter(sort.input) {
            return None;
        }

        let scan = find_scan(sort.input)?;
        let table_def = self.catalog.resolve_table(scan.table).ok()?;

        let reverse = !sort_key.ascending;

        let pk_col = table_def
            .columns()
            .iter()
            .find(|c| c.has_constraint(&crate::schema::table::Constraint::PrimaryKey));

        if let Some(pk) = pk_col {
            if pk.name().eq_ignore_ascii_case(col_name) {
                let table_scan = self
                    .arena
                    .alloc(PhysicalOperator::TableScan(PhysicalTableScan {
                        schema: scan.schema,
                        table: scan.table,
                        alias: scan.alias,
                        post_scan_filter: None,
                        table_def: Some(table_def),
                        reverse,
                    }));

                return match sort.input {
                    LogicalOperator::Scan(_) => Some(table_scan),
                    LogicalOperator::Project(proj) => {
                        let physical_proj =
                            self.arena
                                .alloc(PhysicalOperator::ProjectExec(PhysicalProjectExec {
                                    input: table_scan,
                                    expressions: proj.expressions,
                                    aliases: proj.aliases,
                                }));
                        Some(physical_proj)
                    }
                    _ => None,
                };
            }
        }

        let matching_index = table_def.indexes().iter().find(|idx| {
            if idx.has_expressions() || idx.is_partial() {
                return false;
            }
            idx.columns()
                .next()
                .map(|first_col| first_col.eq_ignore_ascii_case(col_name))
                .unwrap_or(false)
        });

        if let Some(idx) = matching_index {
            let index_name = self.arena.alloc_str(idx.name());
            let table_def_alloc = self.arena.alloc(table_def.clone());

            let index_scan = self.arena.alloc(PhysicalOperator::SecondaryIndexScan(
                PhysicalSecondaryIndexScan {
                    schema: scan.schema,
                    table: scan.table,
                    index_name,
                    table_def: Some(table_def_alloc),
                    reverse,
                    is_unique_index: idx.is_unique(),
                    key_range: None,
                    limit,
                },
            ));

            return match sort.input {
                LogicalOperator::Scan(_) => Some(index_scan),
                LogicalOperator::Project(proj) => {
                    let physical_proj =
                        self.arena
                            .alloc(PhysicalOperator::ProjectExec(PhysicalProjectExec {
                                input: index_scan,
                                expressions: proj.expressions,
                                aliases: proj.aliases,
                            }));
                    Some(physical_proj)
                }
                _ => None,
            };
        }

        None
    }
}

pub fn extract_equality_predicate<'a>(expr: &'a Expr<'a>) -> Option<(&'a str, &'a Expr<'a>)> {
    match expr {
        Expr::BinaryOp { left, op: BinaryOperator::Eq, right } => {
            match (left, right) {
                (Expr::Column(col_ref), lit @ Expr::Literal(_)) => {
                    Some((col_ref.column, lit))
                }
                (lit @ Expr::Literal(_), Expr::Column(col_ref)) => {
                    Some((col_ref.column, lit))
                }
                _ => None,
            }
        }
        Expr::BinaryOp { left, op: BinaryOperator::And, right } => {
            extract_equality_predicate(left)
                .or_else(|| extract_equality_predicate(right))
        }
        _ => None,
    }
}

pub fn compute_residual_filter<'a>(
    arena: &'a Bump,
    predicate: &'a Expr<'a>,
    index_columns: &[String],
) -> Option<&'a Expr<'a>> {
    match predicate {
        Expr::BinaryOp { left, op, right } => match op {
            BinaryOperator::And => {
                let left_residual = compute_residual_filter(arena, left, index_columns);
                let right_residual = compute_residual_filter(arena, right, index_columns);

                match (left_residual, right_residual) {
                    (Some(l), Some(r)) => Some(arena.alloc(Expr::BinaryOp {
                        left: l,
                        op: BinaryOperator::And,
                        right: r,
                    })),
                    (Some(l), None) => Some(l),
                    (None, Some(r)) => Some(r),
                    (None, None) => None,
                }
            }
            BinaryOperator::Eq
            | BinaryOperator::Lt
            | BinaryOperator::LtEq
            | BinaryOperator::Gt
            | BinaryOperator::GtEq => {
                if predicate_uses_index_column(predicate, index_columns) {
                    None
                } else {
                    Some(predicate)
                }
            }
            _ => Some(predicate),
        },
        Expr::Between { expr, .. } => {
            if predicate_uses_index_column(expr, index_columns) {
                None
            } else {
                Some(predicate)
            }
        }
        _ => Some(predicate),
    }
}

pub fn predicate_uses_index_column(expr: &Expr<'_>, index_columns: &[String]) -> bool {
    match expr {
        Expr::Column(col_ref) => index_columns
            .iter()
            .any(|c| c.eq_ignore_ascii_case(col_ref.column)),
        Expr::BinaryOp { left, right, .. } => {
            predicate_uses_index_column(left, index_columns)
                || predicate_uses_index_column(right, index_columns)
        }
        Expr::Between { expr, .. } => predicate_uses_index_column(expr, index_columns),
        _ => false,
    }
}

pub fn predicate_implies_where_clause(query_predicate: &str, index_where: &str) -> bool {
    let norm_query = normalize_predicate(query_predicate);
    let norm_index = normalize_predicate(index_where);

    norm_query.contains(&norm_index)
}

pub fn normalize_predicate(predicate: &str) -> String {
    predicate
        .replace("(", "")
        .replace(")", "")
        .replace(" Eq ", " = ")
        .replace(" = ", "=")
        .replace("'", "")
        .to_lowercase()
}
