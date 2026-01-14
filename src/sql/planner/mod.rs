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

pub use crate::sql::optimizer::EquiJoinKey;

use crate::schema::Catalog;
use crate::sql::ast::{Expr, Statement};
use crate::sql::optimizer::JoinAnalyzer;
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

    pub fn has_equi_join_keys(&self, condition: Option<&'a Expr<'a>>) -> bool {
        let analyzer = JoinAnalyzer::new(self.arena);
        analyzer.has_equi_join_keys(condition)
    }

    pub fn extract_equi_join_keys(&self, condition: Option<&'a Expr<'a>>) -> &'a [EquiJoinKey<'a>] {
        let analyzer = JoinAnalyzer::new(self.arena);
        analyzer.extract_equi_join_keys(condition)
    }

    pub fn extract_equi_join_keys_for_join(
        &self,
        condition: Option<&'a Expr<'a>>,
        left: &'a LogicalOperator<'a>,
        right: &'a LogicalOperator<'a>,
    ) -> &'a [EquiJoinKey<'a>] {
        let analyzer = JoinAnalyzer::new(self.arena);
        analyzer.extract_equi_join_keys_for_join(condition, left, right)
    }

    pub fn convert_equi_keys_to_join_keys(
        &self,
        equi_keys: &'a [EquiJoinKey<'a>],
    ) -> &'a [(&'a Expr<'a>, &'a Expr<'a>)] {
        let analyzer = JoinAnalyzer::new(self.arena);
        analyzer.convert_equi_keys_to_join_keys(equi_keys)
    }
}
