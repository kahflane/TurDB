//! # Query Planner and Optimizer
//!
//! This module implements TurDB's query planner and optimizer, transforming SQL AST
//! into executable query plans through a two-phase planning process.
//!
//! ## Architecture
//!
//! The planner uses a two-phase approach for clarity and maintainability:
//!
//! 1. **Logical Planning (AST → Logical Plan)**
//!    - Validates the query against the catalog
//!    - Resolves names and binds column references
//!    - Performs initial transformations (subquery flattening)
//!    - Generates an optimized logical plan (what to do)
//!
//! 2. **Physical Planning (Logical Plan → Physical Plan)**
//!    - Selects physical operators (how to do it)
//!    - Chooses access paths (index vs sequential scan)
//!    - Applies optimization rules based on cost
//!    - Selects join algorithms (nested loop, grace hash)
//!
//! ## Module Structure
//!
//! - `logical`: Logical operator definitions
//! - `physical`: Physical operator definitions
//! - `schema`: Output schema and table source types
//! - `types`: Helper types (ScanRange, PlanNode)
//! - `select`: SELECT statement planning
//! - `dml`: INSERT/UPDATE/DELETE planning
//! - `convert`: Logical to physical conversion
//! - `encoding`: Key encoding utilities
//!
//! ## Memory Constraints
//!
//! All operations respect the 256KB working memory budget:
//!
//! - Grace Hash Join: 16 partitions, spill to partition/ on overflow
//! - Hash Aggregate: Fail fast on overflow (future: spill)
//! - Sort: External merge sort on overflow

pub mod convert;
pub mod dml;
pub mod encoding;
pub mod logical;
pub mod physical;
pub mod schema;
pub mod select;
pub mod type_inference;
pub mod types;

pub use logical::{
    InsertSource, LogicalAggregate, LogicalDelete, LogicalFilter, LogicalInsert, LogicalJoin,
    LogicalLimit, LogicalOperator, LogicalPlan, LogicalProject, LogicalScan, LogicalSetOp,
    LogicalSort, LogicalSubquery, LogicalUpdate, LogicalValues, LogicalWindow, SetOpKind,
    SortKey, UpdateAssignment, WindowFunctionDef, WindowFunctionType,
};
pub use physical::{
    AggregateExpr, AggregateFunction, PhysicalExistsSubqueryExec, PhysicalFilterExec,
    PhysicalGraceHashJoin, PhysicalHashAggregate, PhysicalHashAntiJoin, PhysicalHashSemiJoin,
    PhysicalIndexScan, PhysicalInListSubqueryExec, PhysicalLimitExec, PhysicalNestedLoopJoin,
    PhysicalOperator, PhysicalPlan, PhysicalProjectExec, PhysicalScalarSubqueryExec,
    PhysicalSecondaryIndexScan, PhysicalSetOpExec, PhysicalSortExec, PhysicalSortedAggregate,
    PhysicalSubqueryExec, PhysicalTableScan, PhysicalTopKExec, PhysicalWindowExec,
};
pub use schema::{CteContext, OutputColumn, OutputSchema, PlannedCte, TableSource};
pub use types::{PlanNode, ScanRange};

use crate::schema::Catalog;
use crate::sql::ast::{Expr, Statement};
use bumpalo::Bump;
use eyre::{bail, Result};

pub struct Planner<'a> {
    catalog: &'a Catalog,
    arena: &'a Bump,
}

impl<'a> Planner<'a> {
    pub fn new(catalog: &'a Catalog, arena: &'a Bump) -> Self {
        Self { catalog, arena }
    }

    pub fn catalog(&self) -> &'a Catalog {
        self.catalog
    }

    pub fn arena(&self) -> &'a Bump {
        self.arena
    }

    pub fn create_logical_plan(&self, stmt: &'a Statement<'a>) -> Result<LogicalPlan<'a>> {
        match stmt {
            Statement::Select(select) => self.plan_select(select),
            Statement::Insert(insert) => self.plan_insert(insert),
            Statement::Update(update) => self.plan_update(update),
            Statement::Delete(delete) => self.plan_delete(delete),
            _ => bail!("unsupported statement type for logical planning"),
        }
    }

    pub fn create_physical_plan(&self, stmt: &'a Statement<'a>) -> Result<PhysicalPlan<'a>> {
        let logical = self.create_logical_plan(stmt)?;
        self.optimize_to_physical(&logical)
    }

    pub(crate) fn validate_table_exists(&self, schema: Option<&str>, table: &str) -> Result<()> {
        let table_name = if let Some(s) = schema {
            self.arena.alloc_str(&format!("{}.{}", s, table))
        } else {
            table
        };

        self.catalog.resolve_table(table_name).map(|_| ())
    }

    pub(crate) fn collect_tables_in_scope(&self, op: &'a LogicalOperator<'a>) -> Vec<TableSource<'a>> {
        let mut tables = Vec::new();
        self.collect_tables_recursive(op, &mut tables);
        tables
    }

    fn collect_tables_recursive(
        &self,
        op: &'a LogicalOperator<'a>,
        tables: &mut Vec<TableSource<'a>>,
    ) {
        match op {
            LogicalOperator::Scan(scan) => {
                let table_name = if let Some(schema) = scan.schema {
                    self.arena.alloc_str(&format!("{}.{}", schema, scan.table))
                } else {
                    scan.table
                };
                if let Ok(table_def) = self.catalog.resolve_table(table_name) {
                    tables.push(TableSource::Table {
                        schema: scan.schema,
                        name: scan.table,
                        alias: scan.alias,
                        def: table_def,
                    });
                }
            }
            LogicalOperator::Join(join) => {
                self.collect_tables_recursive(join.left, tables);
                self.collect_tables_recursive(join.right, tables);
            }
            LogicalOperator::Filter(filter) => {
                self.collect_tables_recursive(filter.input, tables);
            }
            LogicalOperator::Project(project) => {
                self.collect_tables_recursive(project.input, tables);
            }
            LogicalOperator::Aggregate(agg) => {
                self.collect_tables_recursive(agg.input, tables);
            }
            LogicalOperator::Sort(sort) => {
                self.collect_tables_recursive(sort.input, tables);
            }
            LogicalOperator::Limit(limit) => {
                self.collect_tables_recursive(limit.input, tables);
            }
            LogicalOperator::Subquery(subq) => {
                tables.push(TableSource::Subquery {
                    alias: subq.alias,
                    output_schema: subq.output_schema.clone(),
                });
            }
            _ => {}
        }
    }

    pub(crate) fn validate_column_in_scope(
        &self,
        col_ref: &crate::sql::ast::ColumnRef<'a>,
        tables: &[TableSource<'a>],
    ) -> Result<()> {
        if col_ref.table.is_some() {
            let col_table = col_ref.table.unwrap();
            for source in tables {
                let matches_table = source.effective_name().eq_ignore_ascii_case(col_table);
                if matches_table && source.has_column(col_ref.column) {
                    return Ok(());
                }
            }

            bail!(
                "column '{}' not found in table '{}'",
                col_ref.column,
                col_ref.table.unwrap()
            )
        } else {
            let mut matching_tables: Vec<&str> = Vec::new();
            for source in tables {
                if source.has_column(col_ref.column) {
                    matching_tables.push(source.effective_name());
                }
            }

            match matching_tables.len() {
                0 => bail!(
                    "column '{}' not found in any table in scope",
                    col_ref.column
                ),
                1 => Ok(()),
                _ => bail!(
                    "column '{}' is ambiguous (found in tables: {})",
                    col_ref.column,
                    matching_tables.join(", ")
                ),
            }
        }
    }

    pub(crate) fn validate_expr_columns(&self, expr: &Expr<'a>, tables: &[TableSource<'a>]) -> Result<()> {
        match expr {
            Expr::Column(col_ref) => self.validate_column_in_scope(col_ref, tables),
            Expr::BinaryOp { left, right, .. } => {
                self.validate_expr_columns(left, tables)?;
                self.validate_expr_columns(right, tables)
            }
            Expr::UnaryOp { expr, .. } => self.validate_expr_columns(expr, tables),
            Expr::Function(func) => {
                use crate::sql::ast::FunctionArgs;
                if let FunctionArgs::Args(args) = func.args {
                    for arg in args.iter() {
                        self.validate_expr_columns(arg.value, tables)?;
                    }
                }
                Ok(())
            }
            Expr::Case {
                operand,
                conditions,
                else_result,
            } => {
                if let Some(op) = operand {
                    self.validate_expr_columns(op, tables)?;
                }
                for when_clause in conditions.iter() {
                    self.validate_expr_columns(when_clause.condition, tables)?;
                    self.validate_expr_columns(when_clause.result, tables)?;
                }
                if let Some(el) = else_result {
                    self.validate_expr_columns(el, tables)?;
                }
                Ok(())
            }
            Expr::Cast { expr, .. } => self.validate_expr_columns(expr, tables),
            Expr::Between {
                expr, low, high, ..
            } => {
                self.validate_expr_columns(expr, tables)?;
                self.validate_expr_columns(low, tables)?;
                self.validate_expr_columns(high, tables)
            }
            Expr::InList { expr, list, .. } => {
                self.validate_expr_columns(expr, tables)?;
                for item in list.iter() {
                    self.validate_expr_columns(item, tables)?;
                }
                Ok(())
            }
            Expr::IsNull { expr, .. } => self.validate_expr_columns(expr, tables),
            Expr::Like { expr, pattern, .. } => {
                self.validate_expr_columns(expr, tables)?;
                self.validate_expr_columns(pattern, tables)
            }
            _ => Ok(()),
        }
    }

    pub(crate) fn validate_select_columns(
        &self,
        columns: &'a [crate::sql::ast::SelectColumn<'a>],
        tables: &[TableSource<'a>],
    ) -> Result<()> {
        use crate::sql::ast::SelectColumn;

        for col in columns {
            match col {
                SelectColumn::Expr { expr, .. } => {
                    self.validate_expr_columns(expr, tables)?;
                }
                SelectColumn::AllColumns => {}
                SelectColumn::TableAllColumns(table_name) => {
                    let found = tables
                        .iter()
                        .any(|source| source.effective_name().eq_ignore_ascii_case(table_name));
                    if !found {
                        bail!("table '{}' not found in FROM clause", table_name);
                    }
                }
            }
        }
        Ok(())
    }

    pub fn extract_filter_columns(&self, expr: &Expr<'a>) -> &'a [&'a str] {
        let mut columns = bumpalo::collections::Vec::new_in(self.arena);
        self.collect_columns_from_expr(expr, &mut columns);
        columns.into_bump_slice()
    }

    fn collect_columns_from_expr(
        &self,
        expr: &Expr<'a>,
        columns: &mut bumpalo::collections::Vec<'a, &'a str>,
    ) {
        match expr {
            Expr::Column(col_ref) => {
                columns.push(col_ref.column);
            }
            Expr::BinaryOp { left, op, right } => match op {
                crate::sql::ast::BinaryOperator::And | crate::sql::ast::BinaryOperator::Or => {
                    self.collect_columns_from_expr(left, columns);
                    self.collect_columns_from_expr(right, columns);
                }
                crate::sql::ast::BinaryOperator::Eq
                | crate::sql::ast::BinaryOperator::NotEq
                | crate::sql::ast::BinaryOperator::Lt
                | crate::sql::ast::BinaryOperator::LtEq
                | crate::sql::ast::BinaryOperator::Gt
                | crate::sql::ast::BinaryOperator::GtEq => {
                    self.collect_columns_from_expr(left, columns);
                    self.collect_columns_from_expr(right, columns);
                }
                _ => {}
            },
            Expr::IsNull { expr, .. } => {
                self.collect_columns_from_expr(expr, columns);
            }
            Expr::Between {
                expr, low, high, ..
            } => {
                self.collect_columns_from_expr(expr, columns);
                self.collect_columns_from_expr(low, columns);
                self.collect_columns_from_expr(high, columns);
            }
            Expr::InList { expr, list, .. } => {
                self.collect_columns_from_expr(expr, columns);
                for item in *list {
                    self.collect_columns_from_expr(item, columns);
                }
            }
            _ => {}
        }
    }

    pub fn find_applicable_indexes<'t>(
        &self,
        table: &'t crate::schema::TableDef,
        filter_columns: &[&str],
    ) -> Vec<&'t crate::schema::IndexDef> {
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

    pub fn find_applicable_indexes_with_predicate<'t>(
        &self,
        table: &'t crate::schema::TableDef,
        filter_columns: &[&str],
        query_predicate: Option<&str>,
    ) -> Vec<&'t crate::schema::IndexDef> {
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
                        Some(pred) => self.predicate_implies_where_clause(pred, where_clause),
                        None => false,
                    }
                } else {
                    true
                }
            })
            .collect()
    }

    fn predicate_implies_where_clause(&self, query_predicate: &str, index_where: &str) -> bool {
        let norm_query = Self::normalize_predicate(query_predicate);
        let norm_index = Self::normalize_predicate(index_where);

        norm_query.contains(&norm_index)
    }

    fn normalize_predicate(predicate: &str) -> String {
        predicate
            .replace("(", "")
            .replace(")", "")
            .replace(" Eq ", " = ")
            .replace(" = ", "=")
            .replace("'", "")
            .to_lowercase()
    }

    pub fn select_best_index<'t>(
        &self,
        table: &'t crate::schema::TableDef,
        filter_columns: &[&str],
    ) -> Option<&'t crate::schema::IndexDef> {
        let candidates = self.find_applicable_indexes(table, filter_columns);
        if candidates.is_empty() {
            return None;
        }

        candidates.into_iter().min_by(|a, b| {
            let cost_a = self.estimate_index_access_cost(a, filter_columns);
            let cost_b = self.estimate_index_access_cost(b, filter_columns);
            cost_a
                .partial_cmp(&cost_b)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
    }

    pub fn estimate_index_access_cost(
        &self,
        index: &crate::schema::IndexDef,
        filter_columns: &[&str],
    ) -> f64 {
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

    pub fn estimate_index_selectivity(
        &self,
        index: &crate::schema::IndexDef,
        matched_columns: usize,
    ) -> f64 {
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
        index: &crate::schema::IndexDef,
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

    pub fn extract_join_tables(
        &self,
        op: &'a LogicalOperator<'a>,
    ) -> &'a [&'a LogicalOperator<'a>] {
        let mut tables = bumpalo::collections::Vec::new_in(self.arena);
        self.collect_join_tables(op, &mut tables);
        tables.into_bump_slice()
    }

    fn collect_join_tables(
        &self,
        op: &'a LogicalOperator<'a>,
        tables: &mut bumpalo::collections::Vec<'a, &'a LogicalOperator<'a>>,
    ) {
        match op {
            LogicalOperator::Join(join) => {
                self.collect_join_tables(join.left, tables);
                self.collect_join_tables(join.right, tables);
            }
            LogicalOperator::Scan(_) => {
                tables.push(op);
            }
            LogicalOperator::Filter(filter) => {
                self.collect_join_tables(filter.input, tables);
            }
            _ => {
                tables.push(op);
            }
        }
    }

    pub fn order_tables_by_cardinality(
        &self,
        tables_with_card: &[(&'a LogicalOperator<'a>, u64)],
    ) -> &'a [&'a LogicalOperator<'a>] {
        let mut sorted: bumpalo::collections::Vec<(&'a LogicalOperator<'a>, u64)> =
            bumpalo::collections::Vec::from_iter_in(tables_with_card.iter().copied(), self.arena);
        sorted.sort_by_key(|(_, card)| *card);

        let mut result = bumpalo::collections::Vec::new_in(self.arena);
        for (op, _) in sorted {
            result.push(op);
        }
        result.into_bump_slice()
    }

    pub fn extract_equi_join_keys(&self, condition: Option<&'a Expr<'a>>) -> &'a [EquiJoinKey<'a>] {
        let condition = match condition {
            Some(c) => c,
            None => return &[],
        };

        let mut keys = bumpalo::collections::Vec::new_in(self.arena);
        self.collect_equi_join_keys(condition, &mut keys);
        keys.into_bump_slice()
    }

    fn collect_equi_join_keys(
        &self,
        expr: &'a Expr<'a>,
        keys: &mut bumpalo::collections::Vec<'a, EquiJoinKey<'a>>,
    ) {
        use crate::sql::ast::BinaryOperator;

        if let Expr::BinaryOp { left, op, right } = expr {
            match op {
                BinaryOperator::And => {
                    self.collect_equi_join_keys(left, keys);
                    self.collect_equi_join_keys(right, keys);
                }
                BinaryOperator::Eq => {
                    if let (Expr::Column(left_col), Expr::Column(right_col)) = (*left, *right) {
                        keys.push(EquiJoinKey {
                            left_expr: left,
                            right_expr: right,
                            left_column: left_col.column,
                            right_column: right_col.column,
                            left_table: left_col.table,
                            right_table: right_col.table,
                        });
                    }
                }
                _ => {}
            }
        }
    }

    pub fn has_equi_join_keys(&self, condition: Option<&'a Expr<'a>>) -> bool {
        !self.extract_equi_join_keys(condition).is_empty()
    }

    pub fn convert_equi_keys_to_join_keys(
        &self,
        equi_keys: &'a [EquiJoinKey<'a>],
    ) -> &'a [(&'a Expr<'a>, &'a Expr<'a>)] {
        let mut join_keys = bumpalo::collections::Vec::new_in(self.arena);
        for key in equi_keys {
            join_keys.push((key.left_expr, key.right_expr));
        }
        join_keys.into_bump_slice()
    }

    pub fn extract_non_equi_conditions(
        &self,
        condition: Option<&'a Expr<'a>>,
    ) -> Option<&'a Expr<'a>> {
        let condition = condition?;
        self.collect_non_equi_conditions(condition)
    }

    fn collect_non_equi_conditions(&self, expr: &'a Expr<'a>) -> Option<&'a Expr<'a>> {
        use crate::sql::ast::BinaryOperator;

        match expr {
            Expr::BinaryOp { left, op, right } => match op {
                BinaryOperator::And => {
                    let left_non_equi = self.collect_non_equi_conditions(left);
                    let right_non_equi = self.collect_non_equi_conditions(right);

                    match (left_non_equi, right_non_equi) {
                        (Some(l), Some(r)) => Some(self.arena.alloc(Expr::BinaryOp {
                            left: l,
                            op: BinaryOperator::And,
                            right: r,
                        })),
                        (Some(l), None) => Some(l),
                        (None, Some(r)) => Some(r),
                        (None, None) => None,
                    }
                }
                BinaryOperator::Eq => {
                    if matches!((*left, *right), (Expr::Column(_), Expr::Column(_))) {
                        None
                    } else {
                        Some(expr)
                    }
                }
                _ => Some(expr),
            },
            _ => Some(expr),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EquiJoinKey<'a> {
    pub left_expr: &'a Expr<'a>,
    pub right_expr: &'a Expr<'a>,
    pub left_column: &'a str,
    pub right_column: &'a str,
    pub left_table: Option<&'a str>,
    pub right_table: Option<&'a str>,
}
