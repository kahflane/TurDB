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
//! ## Logical Operators
//!
//! The logical plan is a tree of operators focused on relational semantics:
//!
//! | Operator | SQL Correspondence |
//! |----------|-------------------|
//! | Scan | Base table access |
//! | Project | SELECT list |
//! | Filter | WHERE clause |
//! | Aggregate | GROUP BY + aggregates |
//! | Join | JOIN operations |
//! | Sort | ORDER BY |
//! | Limit | LIMIT/OFFSET |
//! | Values | INSERT VALUES |
//!
//! ## Physical Operators
//!
//! Physical operators specify execution algorithms:
//!
//! | Operator | Description | Memory Strategy |
//! |----------|-------------|-----------------|
//! | TableScan | Sequential page iteration | Streaming |
//! | IndexScan | B-tree range scan | Streaming |
//! | FilterExec | Predicate evaluation | Zero-copy |
//! | ProjectExec | Column projection | Zero-copy |
//! | NestedLoopJoin | Simple join | O(1) memory |
//! | GraceHashJoin | Partitioned hash join | 256KB budget |
//! | HashAggregate | Hash-based grouping | Spill on overflow |
//! | SortedAggregate | Pre-sorted input | Streaming |
//! | SortExec | In-memory/external sort | 256KB budget |
//!
//! ## Cost Model
//!
//! The planner uses a heuristic-based cost model initially:
//!
//! - **Cardinality estimation**: Fixed selectivity factors (1/10 for filters)
//! - **Operator costs**: Sequential I/O (low) vs Random I/O (high)
//! - **Index selection**: Cost-based comparison of scan vs index
//!
//! ## Optimization Rules
//!
//! Key optimization rules (applied as PlanRewriter functions):
//!
//! 1. **Predicate Pushdown**: Push filters to data sources
//! 2. **Join Reordering**: Greedy smallest-first heuristic
//! 3. **Constant Folding**: Evaluate constant expressions during planning
//! 4. **Index Selection**: Choose indexes based on cost
//! 5. **Sort/Limit Combination**: Convert to Top-K sort when possible
//!
//! ## Memory Constraints
//!
//! All operations respect the 256KB working memory budget:
//!
//! - Grace Hash Join: 16 partitions, spill to partition/ on overflow
//! - Hash Aggregate: Fail fast on overflow (future: spill)
//! - Sort: External merge sort on overflow
//!
//! ## Arena Allocation
//!
//! All plan nodes are arena-allocated using bumpalo, matching the AST design:
//!
//! ```text
//! Planner<'a>
//!     ├── catalog: &'a Catalog
//!     ├── arena: &'a Bump
//!     └── create_physical_plan() -> Result<PhysicalPlan<'a>>
//! ```
//!
//! ## Performance Targets
//!
//! - Simple queries: < 100µs planning time
//! - Complex queries: < 10ms planning time
//! - Achieved through arena allocation and heuristic optimization

use crate::records::types::DataType;
use crate::schema::{Catalog, TableDef};
use crate::sql::ast::{Expr, JoinType, Literal, Statement};
use bumpalo::Bump;
use eyre::{bail, Result};

#[derive(Debug, Clone)]
pub enum TableSource<'a> {
    Table {
        schema: Option<&'a str>,
        name: &'a str,
        alias: Option<&'a str>,
        def: &'a TableDef,
    },
    Subquery {
        alias: &'a str,
        output_schema: OutputSchema<'a>,
    },
}

impl<'a> TableSource<'a> {
    pub fn effective_name(&self) -> &'a str {
        match self {
            TableSource::Table { alias, name, .. } => alias.unwrap_or(name),
            TableSource::Subquery { alias, .. } => alias,
        }
    }

    pub fn has_column(&self, col_name: &str) -> bool {
        match self {
            TableSource::Table { def, .. } => def
                .columns()
                .iter()
                .any(|c| c.name().eq_ignore_ascii_case(col_name)),
            TableSource::Subquery { output_schema, .. } => output_schema
                .columns
                .iter()
                .any(|c| c.name.eq_ignore_ascii_case(col_name)),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct OutputColumn<'a> {
    pub name: &'a str,
    pub data_type: DataType,
    pub nullable: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub struct OutputSchema<'a> {
    pub columns: &'a [OutputColumn<'a>],
}

impl<'a> OutputSchema<'a> {
    pub fn empty() -> Self {
        Self { columns: &[] }
    }

    pub fn column_count(&self) -> usize {
        self.columns.len()
    }

    pub fn get_column(&self, name: &str) -> Option<&OutputColumn<'a>> {
        self.columns.iter().find(|c| c.name == name)
    }

    pub fn get_column_by_index(&self, idx: usize) -> Option<&OutputColumn<'a>> {
        self.columns.get(idx)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum LogicalOperator<'a> {
    Scan(LogicalScan<'a>),
    DualScan,
    Project(LogicalProject<'a>),
    Filter(LogicalFilter<'a>),
    Aggregate(LogicalAggregate<'a>),
    Join(LogicalJoin<'a>),
    Sort(LogicalSort<'a>),
    Limit(LogicalLimit<'a>),
    Values(LogicalValues<'a>),
    Insert(LogicalInsert<'a>),
    Update(LogicalUpdate<'a>),
    Delete(LogicalDelete<'a>),
    Subquery(LogicalSubquery<'a>),
    SetOp(LogicalSetOp<'a>),
    Window(LogicalWindow<'a>),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SetOpKind {
    Union,
    Intersect,
    Except,
}

#[derive(Debug, Clone, PartialEq)]
pub struct LogicalSetOp<'a> {
    pub left: &'a LogicalOperator<'a>,
    pub right: &'a LogicalOperator<'a>,
    pub kind: SetOpKind,
    pub all: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub struct LogicalScan<'a> {
    pub schema: Option<&'a str>,
    pub table: &'a str,
    pub alias: Option<&'a str>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct LogicalProject<'a> {
    pub input: &'a LogicalOperator<'a>,
    pub expressions: &'a [&'a Expr<'a>],
    pub aliases: &'a [Option<&'a str>],
}

#[derive(Debug, Clone, PartialEq)]
pub struct LogicalFilter<'a> {
    pub input: &'a LogicalOperator<'a>,
    pub predicate: &'a Expr<'a>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct LogicalAggregate<'a> {
    pub input: &'a LogicalOperator<'a>,
    pub group_by: &'a [&'a Expr<'a>],
    pub aggregates: &'a [&'a Expr<'a>],
}

#[derive(Debug, Clone, PartialEq)]
pub struct LogicalJoin<'a> {
    pub left: &'a LogicalOperator<'a>,
    pub right: &'a LogicalOperator<'a>,
    pub join_type: JoinType,
    pub condition: Option<&'a Expr<'a>>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct LogicalSort<'a> {
    pub input: &'a LogicalOperator<'a>,
    pub order_by: &'a [SortKey<'a>],
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SortKey<'a> {
    pub expr: &'a Expr<'a>,
    pub ascending: bool,
    pub nulls_first: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub struct LogicalLimit<'a> {
    pub input: &'a LogicalOperator<'a>,
    pub limit: Option<u64>,
    pub offset: Option<u64>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct LogicalValues<'a> {
    pub rows: &'a [&'a [&'a Expr<'a>]],
}

#[derive(Debug, Clone, PartialEq)]
pub struct LogicalInsert<'a> {
    pub schema: Option<&'a str>,
    pub table: &'a str,
    pub columns: Option<&'a [&'a str]>,
    pub source: InsertSource<'a>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum InsertSource<'a> {
    Values(&'a [&'a [&'a Expr<'a>]]),
    Select(&'a LogicalOperator<'a>),
    Default,
}

#[derive(Debug, Clone, PartialEq)]
pub struct LogicalUpdate<'a> {
    pub schema: Option<&'a str>,
    pub table: &'a str,
    pub assignments: &'a [UpdateAssignment<'a>],
    pub filter: Option<&'a Expr<'a>>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct UpdateAssignment<'a> {
    pub column: &'a str,
    pub value: &'a Expr<'a>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct LogicalDelete<'a> {
    pub schema: Option<&'a str>,
    pub table: &'a str,
    pub filter: Option<&'a Expr<'a>>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct LogicalSubquery<'a> {
    pub plan: &'a LogicalOperator<'a>,
    pub alias: &'a str,
    pub output_schema: OutputSchema<'a>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct LogicalWindow<'a> {
    pub input: &'a LogicalOperator<'a>,
    pub window_functions: &'a [WindowFunctionDef<'a>],
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WindowFunctionType {
    RowNumber,
    Rank,
    DenseRank,
    Count,
    Sum,
    Avg,
    Min,
    Max,
}

impl WindowFunctionType {
    pub fn from_name(name: &str) -> Option<Self> {
        match name.to_uppercase().as_str() {
            "ROW_NUMBER" => Some(Self::RowNumber),
            "RANK" => Some(Self::Rank),
            "DENSE_RANK" => Some(Self::DenseRank),
            "COUNT" => Some(Self::Count),
            "SUM" => Some(Self::Sum),
            "AVG" => Some(Self::Avg),
            "MIN" => Some(Self::Min),
            "MAX" => Some(Self::Max),
            _ => None,
        }
    }

    pub fn returns_integer(&self) -> bool {
        matches!(
            self,
            Self::RowNumber | Self::Rank | Self::DenseRank | Self::Count
        )
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct WindowFunctionDef<'a> {
    pub function_name: &'a str,
    pub function_type: WindowFunctionType,
    pub args: &'a [&'a Expr<'a>],
    pub partition_by: &'a [&'a Expr<'a>],
    pub order_by: &'a [SortKey<'a>],
    pub alias: Option<&'a str>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct LogicalPlan<'a> {
    pub root: &'a LogicalOperator<'a>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum PhysicalOperator<'a> {
    TableScan(PhysicalTableScan<'a>),
    DualScan,
    IndexScan(PhysicalIndexScan<'a>),
    SecondaryIndexScan(PhysicalSecondaryIndexScan<'a>),
    FilterExec(PhysicalFilterExec<'a>),
    ProjectExec(PhysicalProjectExec<'a>),
    NestedLoopJoin(PhysicalNestedLoopJoin<'a>),
    GraceHashJoin(PhysicalGraceHashJoin<'a>),
    HashAggregate(PhysicalHashAggregate<'a>),
    SortedAggregate(PhysicalSortedAggregate<'a>),
    SortExec(PhysicalSortExec<'a>),
    LimitExec(PhysicalLimitExec<'a>),
    TopKExec(PhysicalTopKExec<'a>),
    SubqueryExec(PhysicalSubqueryExec<'a>),
    SetOpExec(PhysicalSetOpExec<'a>),
    WindowExec(PhysicalWindowExec<'a>),
}

#[derive(Debug, Clone, PartialEq)]
pub struct PhysicalSetOpExec<'a> {
    pub left: &'a PhysicalOperator<'a>,
    pub right: &'a PhysicalOperator<'a>,
    pub kind: SetOpKind,
    pub all: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub struct PhysicalTableScan<'a> {
    pub schema: Option<&'a str>,
    pub table: &'a str,
    pub alias: Option<&'a str>,
    pub post_scan_filter: Option<&'a Expr<'a>>,
    pub table_def: Option<&'a TableDef>,
    pub reverse: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ScanRange<'a> {
    FullScan,
    PrefixScan {
        prefix: &'a [u8],
    },
    RangeScan {
        start: Option<&'a [u8]>,
        end: Option<&'a [u8]>,
    },
}

#[derive(Debug, Clone, PartialEq)]
pub struct PhysicalIndexScan<'a> {
    pub schema: Option<&'a str>,
    pub table: &'a str,
    pub index_name: &'a str,
    pub key_range: ScanRange<'a>,
    pub residual_filter: Option<&'a Expr<'a>>,
    pub is_covering: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub struct PhysicalSecondaryIndexScan<'a> {
    pub schema: Option<&'a str>,
    pub table: &'a str,
    pub index_name: &'a str,
    pub table_def: Option<&'a TableDef>,
    pub reverse: bool,
    pub is_unique_index: bool,
    pub key_range: Option<ScanRange<'a>>,
    pub limit: Option<usize>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct PhysicalFilterExec<'a> {
    pub input: &'a PhysicalOperator<'a>,
    pub predicate: &'a Expr<'a>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct PhysicalProjectExec<'a> {
    pub input: &'a PhysicalOperator<'a>,
    pub expressions: &'a [&'a Expr<'a>],
    pub aliases: &'a [Option<&'a str>],
}

#[derive(Debug, Clone, PartialEq)]
pub struct PhysicalNestedLoopJoin<'a> {
    pub left: &'a PhysicalOperator<'a>,
    pub right: &'a PhysicalOperator<'a>,
    pub join_type: JoinType,
    pub condition: Option<&'a Expr<'a>>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct PhysicalGraceHashJoin<'a> {
    pub left: &'a PhysicalOperator<'a>,
    pub right: &'a PhysicalOperator<'a>,
    pub join_type: JoinType,
    pub join_keys: &'a [(&'a Expr<'a>, &'a Expr<'a>)],
    pub num_partitions: u8,
}

#[derive(Debug, Clone, PartialEq)]
pub struct PhysicalHashAggregate<'a> {
    pub input: &'a PhysicalOperator<'a>,
    pub group_by: &'a [&'a Expr<'a>],
    pub aggregates: &'a [AggregateExpr<'a>],
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AggregateExpr<'a> {
    pub function: AggregateFunction,
    pub argument: Option<&'a Expr<'a>>,
    pub distinct: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AggregateFunction {
    Count,
    Sum,
    Avg,
    Min,
    Max,
}

#[derive(Debug, Clone, PartialEq)]
pub struct PhysicalSortedAggregate<'a> {
    pub input: &'a PhysicalOperator<'a>,
    pub group_by: &'a [&'a Expr<'a>],
    pub aggregates: &'a [AggregateExpr<'a>],
}

#[derive(Debug, Clone, PartialEq)]
pub struct PhysicalSortExec<'a> {
    pub input: &'a PhysicalOperator<'a>,
    pub order_by: &'a [SortKey<'a>],
}

#[derive(Debug, Clone, PartialEq)]
pub struct PhysicalLimitExec<'a> {
    pub input: &'a PhysicalOperator<'a>,
    pub limit: Option<u64>,
    pub offset: Option<u64>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct PhysicalTopKExec<'a> {
    pub input: &'a PhysicalOperator<'a>,
    pub order_by: &'a [SortKey<'a>],
    pub limit: u64,
    pub offset: Option<u64>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct PhysicalSubqueryExec<'a> {
    pub child_plan: &'a PhysicalOperator<'a>,
    pub alias: &'a str,
    pub output_schema: OutputSchema<'a>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct PhysicalWindowExec<'a> {
    pub input: &'a PhysicalOperator<'a>,
    pub window_functions: &'a [WindowFunctionDef<'a>],
}

#[derive(Debug, Clone, PartialEq)]
pub struct PhysicalPlan<'a> {
    pub root: &'a PhysicalOperator<'a>,
    pub output_schema: OutputSchema<'a>,
}

impl<'a> PhysicalPlan<'a> {
    pub fn explain(&self) -> String {
        let mut output = String::new();
        self.format_operator(self.root, 0, &mut output);
        output
    }

    fn format_operator(&self, op: &PhysicalOperator<'a>, indent: usize, output: &mut String) {
        use std::fmt::Write;
        let prefix = "  ".repeat(indent);

        match op {
            PhysicalOperator::TableScan(scan) => {
                let _ = writeln!(
                    output,
                    "{}-> TableScan on {} (reverse={})",
                    prefix, scan.table, scan.reverse
                );
            }
            PhysicalOperator::DualScan => {
                let _ = writeln!(output, "{}-> DualScan", prefix);
            }
            PhysicalOperator::IndexScan(scan) => {
                let _ = writeln!(
                    output,
                    "{}-> IndexScan on {} using {}",
                    prefix, scan.table, scan.index_name
                );
            }
            PhysicalOperator::SecondaryIndexScan(scan) => {
                let _ = writeln!(
                    output,
                    "{}-> SecondaryIndexScan on {} using {} (reverse={}, limit={:?})",
                    prefix, scan.table, scan.index_name, scan.reverse, scan.limit
                );
            }
            PhysicalOperator::FilterExec(filter) => {
                let _ = writeln!(output, "{}-> Filter", prefix);
                self.format_operator(filter.input, indent + 1, output);
            }
            PhysicalOperator::ProjectExec(proj) => {
                let _ = writeln!(output, "{}-> Project", prefix);
                self.format_operator(proj.input, indent + 1, output);
            }
            PhysicalOperator::NestedLoopJoin(join) => {
                let _ = writeln!(output, "{}-> NestedLoopJoin ({:?})", prefix, join.join_type);
                self.format_operator(join.left, indent + 1, output);
                self.format_operator(join.right, indent + 1, output);
            }
            PhysicalOperator::GraceHashJoin(join) => {
                let _ = writeln!(output, "{}-> GraceHashJoin ({:?})", prefix, join.join_type);
                self.format_operator(join.left, indent + 1, output);
                self.format_operator(join.right, indent + 1, output);
            }
            PhysicalOperator::HashAggregate(agg) => {
                let _ = writeln!(output, "{}-> HashAggregate", prefix);
                self.format_operator(agg.input, indent + 1, output);
            }
            PhysicalOperator::SortedAggregate(agg) => {
                let _ = writeln!(output, "{}-> SortedAggregate", prefix);
                self.format_operator(agg.input, indent + 1, output);
            }
            PhysicalOperator::SortExec(sort) => {
                let _ = writeln!(output, "{}-> Sort", prefix);
                self.format_operator(sort.input, indent + 1, output);
            }
            PhysicalOperator::LimitExec(limit) => {
                let _ = writeln!(
                    output,
                    "{}-> Limit (limit={:?}, offset={:?})",
                    prefix, limit.limit, limit.offset
                );
                self.format_operator(limit.input, indent + 1, output);
            }
            PhysicalOperator::TopKExec(topk) => {
                let _ = writeln!(
                    output,
                    "{}-> TopK (limit={}, offset={:?})",
                    prefix, topk.limit, topk.offset
                );
                self.format_operator(topk.input, indent + 1, output);
            }
            PhysicalOperator::SubqueryExec(subq) => {
                let _ = writeln!(output, "{}-> Subquery (alias={:?})", prefix, subq.alias);
                self.format_operator(subq.child_plan, indent + 1, output);
            }
            PhysicalOperator::SetOpExec(set_op) => {
                let _ = writeln!(output, "{}-> SetOp ({:?})", prefix, set_op.kind);
                self.format_operator(set_op.left, indent + 1, output);
                self.format_operator(set_op.right, indent + 1, output);
            }
            PhysicalOperator::WindowExec(window) => {
                let _ = writeln!(output, "{}-> Window", prefix);
                self.format_operator(window.input, indent + 1, output);
            }
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum PlanNode<'a> {
    Logical(&'a LogicalOperator<'a>),
    Physical(&'a PhysicalOperator<'a>),
}

impl<'a> PlanNode<'a> {
    pub fn is_logical(&self) -> bool {
        matches!(self, PlanNode::Logical(_))
    }

    pub fn is_physical(&self) -> bool {
        matches!(self, PlanNode::Physical(_))
    }

    pub fn as_logical(&self) -> Option<&'a LogicalOperator<'a>> {
        match self {
            PlanNode::Logical(op) => Some(op),
            PlanNode::Physical(_) => None,
        }
    }

    pub fn as_physical(&self) -> Option<&'a PhysicalOperator<'a>> {
        match self {
            PlanNode::Logical(_) => None,
            PlanNode::Physical(op) => Some(op),
        }
    }
}

pub struct CteContext<'a> {
    pub ctes: hashbrown::HashMap<&'a str, PlannedCte<'a>>,
}

pub struct PlannedCte<'a> {
    pub plan: &'a LogicalOperator<'a>,
    pub output_schema: OutputSchema<'a>,
    pub columns: Option<&'a [&'a str]>,
}

impl<'a> CteContext<'a> {
    pub fn new() -> Self {
        Self {
            ctes: hashbrown::HashMap::new(),
        }
    }

    pub fn get(&self, name: &str) -> Option<&PlannedCte<'a>> {
        self.ctes
            .iter()
            .find(|(k, _)| k.eq_ignore_ascii_case(name))
            .map(|(_, v)| v)
    }

    pub fn insert(&mut self, name: &'a str, cte: PlannedCte<'a>) {
        self.ctes.insert(name, cte);
    }
}

impl<'a> Default for CteContext<'a> {
    fn default() -> Self {
        Self::new()
    }
}

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

    fn plan_select(&self, select: &crate::sql::ast::SelectStmt<'a>) -> Result<LogicalPlan<'a>> {
        let cte_context = self.build_cte_context(select.with)?;
        self.plan_select_with_ctes(select, &cte_context)
    }

    fn build_cte_context(
        &self,
        with_clause: Option<&crate::sql::ast::WithClause<'a>>,
    ) -> Result<CteContext<'a>> {
        let mut ctx = CteContext::new();

        if let Some(with) = with_clause {
            for cte in with.ctes {
                let cte_plan = self.plan_select_with_ctes(cte.query, &ctx)?;
                let mut output_schema = self.compute_logical_output_schema(cte_plan.root)?;

                if let Some(col_names) = cte.columns {
                    if col_names.len() != output_schema.columns.len() {
                        bail!(
                            "CTE '{}' has {} column names but query returns {} columns",
                            cte.name,
                            col_names.len(),
                            output_schema.columns.len()
                        );
                    }
                    let mut new_cols = bumpalo::collections::Vec::new_in(self.arena);
                    for (i, col) in output_schema.columns.iter().enumerate() {
                        new_cols.push(OutputColumn {
                            name: col_names[i],
                            data_type: col.data_type,
                            nullable: col.nullable,
                        });
                    }
                    output_schema = OutputSchema {
                        columns: new_cols.into_bump_slice(),
                    };
                }

                ctx.insert(
                    cte.name,
                    PlannedCte {
                        plan: cte_plan.root,
                        output_schema,
                        columns: cte.columns,
                    },
                );
            }
        }

        Ok(ctx)
    }

    fn plan_select_with_ctes(
        &self,
        select: &crate::sql::ast::SelectStmt<'a>,
        cte_context: &CteContext<'a>,
    ) -> Result<LogicalPlan<'a>> {
        let mut current: &'a LogicalOperator<'a> = match select.from {
            Some(from) => self.plan_from_clause_with_ctes(from, cte_context)?,
            None => self.arena.alloc(LogicalOperator::DualScan),
        };

        let tables_in_scope = self.collect_tables_in_scope(current);
        if select.from.is_some() {
            self.validate_select_columns(select.columns, &tables_in_scope)?;
        }

        if let Some(predicate) = select.where_clause {
            self.validate_expr_columns(predicate, &tables_in_scope)?;
            let filter = self.arena.alloc(LogicalOperator::Filter(LogicalFilter {
                input: current,
                predicate,
            }));
            current = filter;
        }

        let has_aggregates = self.select_has_aggregates(select.columns);
        if !select.group_by.is_empty() || has_aggregates {
            for group_expr in select.group_by.iter() {
                self.validate_expr_columns(group_expr, &tables_in_scope)?;
            }

            let aggregates = self.extract_aggregates(select.columns);
            let agg = self
                .arena
                .alloc(LogicalOperator::Aggregate(LogicalAggregate {
                    input: current,
                    group_by: select.group_by,
                    aggregates,
                }));
            current = agg;

            if let Some(having) = select.having {
                self.validate_expr_columns(having, &tables_in_scope)?;
                let having_filter = self.arena.alloc(LogicalOperator::Filter(LogicalFilter {
                    input: current,
                    predicate: having,
                }));
                current = having_filter;
            }
        }

        let has_window_functions = self.select_has_window_functions(select.columns);

        if has_window_functions {
            let window_functions = self.extract_window_functions(select.columns)?;
            let window = self.arena.alloc(LogicalOperator::Window(LogicalWindow {
                input: current,
                window_functions,
            }));
            current = window;
        }

        let (exprs, aliases) = self.extract_select_expressions(select.columns);
        let project = self.arena.alloc(LogicalOperator::Project(LogicalProject {
            input: current,
            expressions: exprs,
            aliases,
        }));
        current = project;

        if let Some(set_op) = select.set_op {
            let right_select = set_op.right;
            let right_order_by = right_select.order_by;
            let right_limit = right_select.limit;
            let right_offset = right_select.offset;

            let right_plan = self.plan_select_core(right_select, cte_context)?;
            let kind = match set_op.op {
                crate::sql::ast::SetOperator::Union => SetOpKind::Union,
                crate::sql::ast::SetOperator::Intersect => SetOpKind::Intersect,
                crate::sql::ast::SetOperator::Except => SetOpKind::Except,
            };
            let set_op_node = self.arena.alloc(LogicalOperator::SetOp(LogicalSetOp {
                left: current,
                right: right_plan.root,
                kind,
                all: set_op.all,
            }));
            current = set_op_node;

            if !right_order_by.is_empty() {
                let sort_keys = self.convert_order_by(right_order_by);
                let sort = self.arena.alloc(LogicalOperator::Sort(LogicalSort {
                    input: current,
                    order_by: sort_keys,
                }));
                current = sort;
            }

            if right_limit.is_some() || right_offset.is_some() {
                let limit_val = right_limit.and_then(|e| self.eval_const_u64(e));
                let offset_val = right_offset.and_then(|e| self.eval_const_u64(e));
                let limit_op = self.arena.alloc(LogicalOperator::Limit(LogicalLimit {
                    input: current,
                    limit: limit_val,
                    offset: offset_val,
                }));
                current = limit_op;
            }
        } else {
            if !select.order_by.is_empty() {
                for order_item in select.order_by.iter() {
                    let is_alias = if let crate::sql::ast::Expr::Column(col_ref) = order_item.expr {
                        col_ref.table.is_none()
                            && aliases.contains(&Some(col_ref.column))
                    } else {
                        false
                    };

                    if !is_alias {
                        self.validate_expr_columns(order_item.expr, &tables_in_scope)?;
                    }
                }
                let sort_keys = self.convert_order_by(select.order_by);
                let sort = self.arena.alloc(LogicalOperator::Sort(LogicalSort {
                    input: current,
                    order_by: sort_keys,
                }));
                current = sort;
            }

            if select.limit.is_some() || select.offset.is_some() {
                let limit_val = select.limit.and_then(|e| self.eval_const_u64(e));
                let offset_val = select.offset.and_then(|e| self.eval_const_u64(e));

                let limit_op = self.arena.alloc(LogicalOperator::Limit(LogicalLimit {
                    input: current,
                    limit: limit_val,
                    offset: offset_val,
                }));
                current = limit_op;
            }
        }

        Ok(LogicalPlan { root: current })
    }

    fn plan_select_core(
        &self,
        select: &crate::sql::ast::SelectStmt<'a>,
        cte_context: &CteContext<'a>,
    ) -> Result<LogicalPlan<'a>> {
        let mut current: &'a LogicalOperator<'a> = match select.from {
            Some(from) => self.plan_from_clause_with_ctes(from, cte_context)?,
            None => self.arena.alloc(LogicalOperator::DualScan),
        };

        let tables_in_scope = self.collect_tables_in_scope(current);
        if select.from.is_some() {
            self.validate_select_columns(select.columns, &tables_in_scope)?;
        }

        if let Some(predicate) = select.where_clause {
            self.validate_expr_columns(predicate, &tables_in_scope)?;
            let filter = self.arena.alloc(LogicalOperator::Filter(LogicalFilter {
                input: current,
                predicate,
            }));
            current = filter;
        }

        let has_aggregates = self.select_has_aggregates(select.columns);
        if !select.group_by.is_empty() || has_aggregates {
            for group_expr in select.group_by.iter() {
                self.validate_expr_columns(group_expr, &tables_in_scope)?;
            }

            let aggregates = self.extract_aggregates(select.columns);
            let agg = self
                .arena
                .alloc(LogicalOperator::Aggregate(LogicalAggregate {
                    input: current,
                    group_by: select.group_by,
                    aggregates,
                }));
            current = agg;

            if let Some(having) = select.having {
                self.validate_expr_columns(having, &tables_in_scope)?;
                let having_filter = self.arena.alloc(LogicalOperator::Filter(LogicalFilter {
                    input: current,
                    predicate: having,
                }));
                current = having_filter;
            }
        }

        let (exprs, aliases) = self.extract_select_expressions(select.columns);
        let project = self.arena.alloc(LogicalOperator::Project(LogicalProject {
            input: current,
            expressions: exprs,
            aliases,
        }));
        current = project;

        if let Some(set_op) = select.set_op {
            let right_plan = self.plan_select_core(set_op.right, cte_context)?;
            let kind = match set_op.op {
                crate::sql::ast::SetOperator::Union => SetOpKind::Union,
                crate::sql::ast::SetOperator::Intersect => SetOpKind::Intersect,
                crate::sql::ast::SetOperator::Except => SetOpKind::Except,
            };
            let set_op_node = self.arena.alloc(LogicalOperator::SetOp(LogicalSetOp {
                left: current,
                right: right_plan.root,
                kind,
                all: set_op.all,
            }));
            current = set_op_node;
        }

        Ok(LogicalPlan { root: current })
    }

    fn plan_from_clause_with_ctes(
        &self,
        from: &'a crate::sql::ast::FromClause<'a>,
        cte_context: &CteContext<'a>,
    ) -> Result<&'a LogicalOperator<'a>> {
        use crate::sql::ast::FromClause;

        match from {
            FromClause::Table(table_ref) => {
                if table_ref.schema.is_none() {
                    if let Some(cte) = cte_context.get(table_ref.name) {
                        let alias = table_ref.alias.unwrap_or(table_ref.name);
                        let subquery_op =
                            self.arena.alloc(LogicalOperator::Subquery(LogicalSubquery {
                                plan: cte.plan,
                                alias,
                                output_schema: cte.output_schema.clone(),
                            }));
                        return Ok(subquery_op);
                    }
                }

                self.validate_table_exists(table_ref.schema, table_ref.name)?;

                let scan = self.arena.alloc(LogicalOperator::Scan(LogicalScan {
                    schema: table_ref.schema,
                    table: table_ref.name,
                    alias: table_ref.alias,
                }));
                Ok(scan)
            }
            FromClause::Join(join) => self.plan_join_with_ctes(join, cte_context),
            FromClause::Subquery { query, alias } => {
                let subquery_plan = self.plan_select_with_ctes(query, cte_context)?;
                let output_schema = self.compute_logical_output_schema(subquery_plan.root)?;
                let subquery_op = self.arena.alloc(LogicalOperator::Subquery(LogicalSubquery {
                    plan: subquery_plan.root,
                    alias,
                    output_schema,
                }));
                Ok(subquery_op)
            }
            FromClause::Lateral {
                subquery: _,
                alias: _,
            } => {
                bail!("LATERAL subqueries not yet implemented")
            }
        }
    }

    fn plan_join_with_ctes(
        &self,
        join: &'a crate::sql::ast::JoinClause<'a>,
        cte_context: &CteContext<'a>,
    ) -> Result<&'a LogicalOperator<'a>> {
        let left = self.plan_from_clause_with_ctes(join.left, cte_context)?;
        let right = self.plan_from_clause_with_ctes(join.right, cte_context)?;

        let condition = match join.condition {
            crate::sql::ast::JoinCondition::On(expr) => Some(expr),
            crate::sql::ast::JoinCondition::Using(_) => {
                bail!("USING clause in joins not yet implemented")
            }
            crate::sql::ast::JoinCondition::Natural => {
                bail!("NATURAL joins not yet implemented")
            }
            crate::sql::ast::JoinCondition::None => None,
        };

        let join_op = self.arena.alloc(LogicalOperator::Join(LogicalJoin {
            left,
            right,
            join_type: join.join_type,
            condition,
        }));

        Ok(join_op)
    }

    fn extract_aggregates(
        &self,
        columns: &'a [crate::sql::ast::SelectColumn<'a>],
    ) -> &'a [&'a Expr<'a>] {
        use crate::sql::ast::SelectColumn;

        let mut aggregates = bumpalo::collections::Vec::new_in(self.arena);

        for col in columns {
            if let SelectColumn::Expr { expr, .. } = col {
                self.collect_aggregates_from_expr(expr, &mut aggregates);
            }
        }

        aggregates.into_bump_slice()
    }

    fn traverse_expr_for_aggregates<F>(&self, expr: &'a Expr<'a>, on_aggregate: &mut F) -> bool
    where
        F: FnMut(&'a Expr<'a>) -> bool,
    {
        use crate::sql::ast::{FunctionArgs, FunctionCall};

        match expr {
            Expr::Function(FunctionCall {
                name, over, args, ..
            }) => {
                if over.is_none() {
                    let func_name = name.name.to_ascii_lowercase();
                    if matches!(func_name.as_str(), "count" | "sum" | "avg" | "min" | "max") {
                        if on_aggregate(expr) {
                            return true;
                        }
                        return false;
                    }
                }
                if let FunctionArgs::Args(fn_args) = args {
                    for arg in *fn_args {
                        if self.traverse_expr_for_aggregates(arg.value, on_aggregate) {
                            return true;
                        }
                    }
                }
                false
            }
            Expr::BinaryOp { left, right, .. } => {
                self.traverse_expr_for_aggregates(left, on_aggregate)
                    || self.traverse_expr_for_aggregates(right, on_aggregate)
            }
            Expr::UnaryOp { expr: inner, .. } => {
                self.traverse_expr_for_aggregates(inner, on_aggregate)
            }
            Expr::Between {
                expr, low, high, ..
            } => {
                self.traverse_expr_for_aggregates(expr, on_aggregate)
                    || self.traverse_expr_for_aggregates(low, on_aggregate)
                    || self.traverse_expr_for_aggregates(high, on_aggregate)
            }
            Expr::Like {
                expr,
                pattern,
                escape,
                ..
            } => {
                self.traverse_expr_for_aggregates(expr, on_aggregate)
                    || self.traverse_expr_for_aggregates(pattern, on_aggregate)
                    || escape
                        .map(|e| self.traverse_expr_for_aggregates(e, on_aggregate))
                        .unwrap_or(false)
            }
            Expr::InList { expr, list, .. } => {
                self.traverse_expr_for_aggregates(expr, on_aggregate)
                    || list
                        .iter()
                        .any(|e| self.traverse_expr_for_aggregates(e, on_aggregate))
            }
            Expr::IsNull { expr, .. } => self.traverse_expr_for_aggregates(expr, on_aggregate),
            Expr::IsDistinctFrom { left, right, .. } => {
                self.traverse_expr_for_aggregates(left, on_aggregate)
                    || self.traverse_expr_for_aggregates(right, on_aggregate)
            }
            Expr::Case {
                operand,
                conditions,
                else_result,
            } => {
                operand
                    .map(|o| self.traverse_expr_for_aggregates(o, on_aggregate))
                    .unwrap_or(false)
                    || conditions.iter().any(|c| {
                        self.traverse_expr_for_aggregates(c.condition, on_aggregate)
                            || self.traverse_expr_for_aggregates(c.result, on_aggregate)
                    })
                    || else_result
                        .map(|e| self.traverse_expr_for_aggregates(e, on_aggregate))
                        .unwrap_or(false)
            }
            Expr::Cast { expr, .. } => self.traverse_expr_for_aggregates(expr, on_aggregate),
            Expr::ArraySubscript { array, index, .. } => {
                self.traverse_expr_for_aggregates(array, on_aggregate)
                    || self.traverse_expr_for_aggregates(index, on_aggregate)
            }
            Expr::ArraySlice {
                array,
                lower,
                upper,
            } => {
                self.traverse_expr_for_aggregates(array, on_aggregate)
                    || lower
                        .map(|e| self.traverse_expr_for_aggregates(e, on_aggregate))
                        .unwrap_or(false)
                    || upper
                        .map(|e| self.traverse_expr_for_aggregates(e, on_aggregate))
                        .unwrap_or(false)
            }
            Expr::Array(items) => items
                .iter()
                .any(|e| self.traverse_expr_for_aggregates(e, on_aggregate)),
            Expr::Row(items) => items
                .iter()
                .any(|e| self.traverse_expr_for_aggregates(e, on_aggregate)),
            Expr::Literal(_)
            | Expr::Column(_)
            | Expr::Parameter(_)
            | Expr::Subquery(_)
            | Expr::Exists { .. }
            | Expr::InSubquery { .. } => false,
        }
    }

    fn collect_aggregates_from_expr(
        &self,
        expr: &'a Expr<'a>,
        aggregates: &mut bumpalo::collections::Vec<&'a Expr<'a>>,
    ) {
        self.traverse_expr_for_aggregates(expr, &mut |agg| {
            aggregates.push(agg);
            false
        });
    }

    fn contains_aggregate(&self, expr: &Expr<'a>) -> bool {
        self.traverse_expr_for_aggregates(expr, &mut |_| true)
    }

    fn select_has_aggregates(&self, columns: &'a [crate::sql::ast::SelectColumn<'a>]) -> bool {
        use crate::sql::ast::SelectColumn;

        for col in columns {
            if let SelectColumn::Expr { expr, .. } = col {
                if self.contains_aggregate(expr) {
                    return true;
                }
            }
        }
        false
    }

    fn is_window_function(&self, expr: &Expr<'a>) -> bool {
        if let Expr::Function(func) = expr {
            func.over.is_some()
        } else {
            false
        }
    }

    fn select_has_window_functions(
        &self,
        columns: &'a [crate::sql::ast::SelectColumn<'a>],
    ) -> bool {
        use crate::sql::ast::SelectColumn;

        for col in columns {
            if let SelectColumn::Expr { expr, .. } = col {
                if self.is_window_function(expr) {
                    return true;
                }
            }
        }
        false
    }

    fn extract_window_functions(
        &self,
        columns: &'a [crate::sql::ast::SelectColumn<'a>],
    ) -> Result<&'a [WindowFunctionDef<'a>]> {
        use crate::sql::ast::{NullsOrder, OrderDirection, SelectColumn};

        let mut window_funcs = bumpalo::collections::Vec::new_in(self.arena);

        for col in columns {
            if let SelectColumn::Expr {
                expr: Expr::Function(func),
                alias,
            } = col
            {
                if let Some(window_spec) = &func.over {
                    let order_by_keys: &[SortKey<'a>] = {
                        let mut keys = bumpalo::collections::Vec::new_in(self.arena);
                        for item in window_spec.order_by.iter() {
                            keys.push(SortKey {
                                expr: item.expr,
                                ascending: matches!(item.direction, OrderDirection::Asc),
                                nulls_first: matches!(item.nulls, NullsOrder::First),
                            });
                        }
                        keys.into_bump_slice()
                    };

                    let args: &[&Expr<'a>] = match &func.args {
                        crate::sql::ast::FunctionArgs::Args(func_args) => {
                            let mut arg_exprs = bumpalo::collections::Vec::new_in(self.arena);
                            for arg in func_args.iter() {
                                arg_exprs.push(arg.value);
                            }
                            arg_exprs.into_bump_slice()
                        }
                        crate::sql::ast::FunctionArgs::Star => &[],
                        crate::sql::ast::FunctionArgs::None => &[],
                    };

                    let function_type =
                        WindowFunctionType::from_name(func.name.name).ok_or_else(|| {
                            eyre::eyre!(
                                "unsupported window function: '{}'. Supported: ROW_NUMBER, RANK, \
                                 DENSE_RANK, COUNT, SUM, AVG, MIN, MAX",
                                func.name.name
                            )
                        })?;
                    window_funcs.push(WindowFunctionDef {
                        function_name: func.name.name,
                        function_type,
                        args,
                        partition_by: window_spec.partition_by,
                        order_by: order_by_keys,
                        alias: *alias,
                    });
                }
            }
        }

        Ok(window_funcs.into_bump_slice())
    }

    fn extract_select_expressions(
        &self,
        columns: &'a [crate::sql::ast::SelectColumn<'a>],
    ) -> (&'a [&'a Expr<'a>], &'a [Option<&'a str>]) {
        use crate::sql::ast::SelectColumn;

        let mut exprs = bumpalo::collections::Vec::new_in(self.arena);
        let mut aliases = bumpalo::collections::Vec::new_in(self.arena);

        for col in columns {
            match col {
                SelectColumn::Expr { expr, alias } => {
                    exprs.push(*expr);
                    aliases.push(*alias);
                }
                SelectColumn::AllColumns => {}
                SelectColumn::TableAllColumns(_) => {}
            }
        }

        (exprs.into_bump_slice(), aliases.into_bump_slice())
    }

    fn convert_order_by(
        &self,
        order_by: &'a [crate::sql::ast::OrderByItem<'a>],
    ) -> &'a [SortKey<'a>] {
        use crate::sql::ast::{NullsOrder, OrderDirection};

        let mut keys = bumpalo::collections::Vec::new_in(self.arena);

        for item in order_by {
            keys.push(SortKey {
                expr: item.expr,
                ascending: matches!(item.direction, OrderDirection::Asc),
                nulls_first: matches!(item.nulls, NullsOrder::First),
            });
        }

        keys.into_bump_slice()
    }

    fn eval_const_u64(&self, expr: &Expr<'a>) -> Option<u64> {
        match expr {
            Expr::Literal(crate::sql::ast::Literal::Integer(n)) => n.parse().ok(),
            _ => None,
        }
    }

    fn validate_table_exists(&self, schema: Option<&str>, table: &str) -> Result<()> {
        let table_name = if let Some(s) = schema {
            self.arena.alloc_str(&format!("{}.{}", s, table))
        } else {
            table
        };

        self.catalog.resolve_table(table_name).map(|_| ())
    }

    fn collect_tables_in_scope(&self, op: &'a LogicalOperator<'a>) -> Vec<TableSource<'a>> {
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

    fn validate_column_in_scope(
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

    fn validate_expr_columns(&self, expr: &Expr<'a>, tables: &[TableSource<'a>]) -> Result<()> {
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

    fn validate_select_columns(
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

    fn plan_insert(&self, insert: &crate::sql::ast::InsertStmt<'a>) -> Result<LogicalPlan<'a>> {
        self.validate_table_exists(insert.table.schema, insert.table.name)?;

        let table_name = if let Some(schema) = insert.table.schema {
            self.arena
                .alloc_str(&format!("{}.{}", schema, insert.table.name))
        } else {
            insert.table.name
        };
        let table_def = self.catalog.resolve_table(table_name)?;

        let expected_column_count = if let Some(columns) = insert.columns {
            for col_name in columns.iter() {
                let col_exists = table_def
                    .columns()
                    .iter()
                    .any(|c| c.name().eq_ignore_ascii_case(col_name));
                if !col_exists {
                    bail!(
                        "column '{}' not found in table '{}'",
                        col_name,
                        insert.table.name
                    );
                }
            }
            columns.len()
        } else {
            table_def.columns().len()
        };

        if let crate::sql::ast::InsertSource::Values(rows) = insert.source {
            for (row_idx, row) in rows.iter().enumerate() {
                if row.len() != expected_column_count {
                    bail!(
                        "INSERT has {} values but {} columns were expected (row {})",
                        row.len(),
                        expected_column_count,
                        row_idx + 1
                    );
                }
            }
        }

        let source = match insert.source {
            crate::sql::ast::InsertSource::Values(rows) => InsertSource::Values(rows),
            crate::sql::ast::InsertSource::Select(select) => {
                let select_plan = self.plan_select(select)?;
                InsertSource::Select(select_plan.root)
            }
            crate::sql::ast::InsertSource::Default => InsertSource::Default,
        };

        let insert_op = self.arena.alloc(LogicalOperator::Insert(LogicalInsert {
            schema: insert.table.schema,
            table: insert.table.name,
            columns: insert.columns,
            source,
        }));

        Ok(LogicalPlan { root: insert_op })
    }

    fn plan_update(&self, update: &crate::sql::ast::UpdateStmt<'a>) -> Result<LogicalPlan<'a>> {
        self.validate_table_exists(update.table.schema, update.table.name)?;

        let table_name = if let Some(schema) = update.table.schema {
            self.arena
                .alloc_str(&format!("{}.{}", schema, update.table.name))
        } else {
            update.table.name
        };
        let table_def = self.catalog.resolve_table(table_name)?;

        let tables_in_scope: Vec<TableSource<'a>> = vec![TableSource::Table {
            schema: update.table.schema,
            name: update.table.name,
            alias: update.table.alias,
            def: table_def,
        }];

        for assign in update.assignments {
            let col_exists = table_def
                .columns()
                .iter()
                .any(|c| c.name().eq_ignore_ascii_case(assign.column.column));
            if !col_exists {
                bail!(
                    "column '{}' not found in table '{}'",
                    assign.column.column,
                    update.table.name
                );
            }

            self.validate_expr_columns(assign.value, &tables_in_scope)?;
        }

        if let Some(predicate) = update.where_clause {
            self.validate_expr_columns(predicate, &tables_in_scope)?;
        }

        let mut assignments = bumpalo::collections::Vec::new_in(self.arena);
        for assign in update.assignments {
            assignments.push(UpdateAssignment {
                column: assign.column.column,
                value: assign.value,
            });
        }

        let update_op = self.arena.alloc(LogicalOperator::Update(LogicalUpdate {
            schema: update.table.schema,
            table: update.table.name,
            assignments: assignments.into_bump_slice(),
            filter: update.where_clause,
        }));

        Ok(LogicalPlan { root: update_op })
    }

    fn plan_delete(&self, delete: &crate::sql::ast::DeleteStmt<'a>) -> Result<LogicalPlan<'a>> {
        self.validate_table_exists(delete.table.schema, delete.table.name)?;

        let delete_op = self.arena.alloc(LogicalOperator::Delete(LogicalDelete {
            schema: delete.table.schema,
            table: delete.table.name,
            filter: delete.where_clause,
        }));

        Ok(LogicalPlan { root: delete_op })
    }

    fn optimize_to_physical(&self, logical: &LogicalPlan<'a>) -> Result<PhysicalPlan<'a>> {
        let physical_root = self.logical_to_physical(logical.root)?;
        let output_schema = self.compute_output_schema(physical_root)?;
        Ok(PhysicalPlan {
            root: physical_root,
            output_schema,
        })
    }

    fn compute_output_schema(&self, op: &'a PhysicalOperator<'a>) -> Result<OutputSchema<'a>> {
        match op {
            PhysicalOperator::TableScan(scan) => {
                let table_def = if let Some(td) = scan.table_def {
                    td
                } else {
                    let table_name = if let Some(schema) = scan.schema {
                        self.arena.alloc_str(&format!("{}.{}", schema, scan.table))
                    } else {
                        scan.table
                    };
                    self.catalog.resolve_table(table_name)?
                };

                let mut columns = bumpalo::collections::Vec::new_in(self.arena);

                for col in table_def.columns() {
                    columns.push(OutputColumn {
                        name: self.arena.alloc_str(col.name()),
                        data_type: col.data_type(),
                        nullable: col.is_nullable(),
                    });
                }

                Ok(OutputSchema {
                    columns: columns.into_bump_slice(),
                })
            }
            PhysicalOperator::DualScan => Ok(OutputSchema { columns: &[] }),
            PhysicalOperator::IndexScan(scan) => {
                let table_name = if let Some(schema) = scan.schema {
                    self.arena.alloc_str(&format!("{}.{}", schema, scan.table))
                } else {
                    scan.table
                };

                let table_def = self.catalog.resolve_table(table_name)?;
                let mut columns = bumpalo::collections::Vec::new_in(self.arena);

                for col in table_def.columns() {
                    columns.push(OutputColumn {
                        name: self.arena.alloc_str(col.name()),
                        data_type: col.data_type(),
                        nullable: col.is_nullable(),
                    });
                }

                Ok(OutputSchema {
                    columns: columns.into_bump_slice(),
                })
            }
            PhysicalOperator::SecondaryIndexScan(scan) => {
                let table_def = scan.table_def.ok_or_else(|| {
                    eyre::eyre!("SecondaryIndexScan missing table_def for {}", scan.table)
                })?;
                let mut columns = bumpalo::collections::Vec::new_in(self.arena);

                for col in table_def.columns() {
                    columns.push(OutputColumn {
                        name: self.arena.alloc_str(col.name()),
                        data_type: col.data_type(),
                        nullable: col.is_nullable(),
                    });
                }

                Ok(OutputSchema {
                    columns: columns.into_bump_slice(),
                })
            }
            PhysicalOperator::FilterExec(filter) => self.compute_output_schema(filter.input),
            PhysicalOperator::LimitExec(limit) => self.compute_output_schema(limit.input),
            PhysicalOperator::SortExec(sort) => self.compute_output_schema(sort.input),
            PhysicalOperator::TopKExec(topk) => self.compute_output_schema(topk.input),
            PhysicalOperator::ProjectExec(project) => {
                let input_schema = self.compute_output_schema(project.input)?;

                if project.expressions.is_empty() {
                    return Ok(input_schema);
                }

                let mut columns = bumpalo::collections::Vec::new_in(self.arena);

                for (i, expr) in project.expressions.iter().enumerate() {
                    let (name, data_type, nullable) = self.infer_expr_type(expr, &input_schema)?;
                    let col_name = if let Some(alias) = project.aliases.get(i).and_then(|a| *a) {
                        alias
                    } else {
                        name
                    };
                    columns.push(OutputColumn {
                        name: col_name,
                        data_type,
                        nullable,
                    });
                }

                Ok(OutputSchema {
                    columns: columns.into_bump_slice(),
                })
            }
            PhysicalOperator::NestedLoopJoin(join) => {
                let left_schema = self.compute_output_schema(join.left)?;
                let right_schema = self.compute_output_schema(join.right)?;
                let mut columns = bumpalo::collections::Vec::new_in(self.arena);

                for col in left_schema.columns {
                    columns.push(*col);
                }
                for col in right_schema.columns {
                    columns.push(*col);
                }

                Ok(OutputSchema {
                    columns: columns.into_bump_slice(),
                })
            }
            PhysicalOperator::GraceHashJoin(join) => {
                let left_schema = self.compute_output_schema(join.left)?;
                let right_schema = self.compute_output_schema(join.right)?;
                let mut columns = bumpalo::collections::Vec::new_in(self.arena);

                for col in left_schema.columns {
                    columns.push(*col);
                }
                for col in right_schema.columns {
                    columns.push(*col);
                }

                Ok(OutputSchema {
                    columns: columns.into_bump_slice(),
                })
            }
            PhysicalOperator::HashAggregate(agg) => {
                let mut columns = bumpalo::collections::Vec::new_in(self.arena);
                let input_schema = self.compute_output_schema(agg.input)?;

                for group_expr in agg.group_by.iter() {
                    let (name, data_type, nullable) =
                        self.infer_expr_type(group_expr, &input_schema)?;
                    columns.push(OutputColumn {
                        name,
                        data_type,
                        nullable,
                    });
                }

                for agg_expr in agg.aggregates.iter() {
                    let (name, data_type) = self.infer_aggregate_type(agg_expr, &input_schema)?;
                    columns.push(OutputColumn {
                        name,
                        data_type,
                        nullable: true,
                    });
                }

                Ok(OutputSchema {
                    columns: columns.into_bump_slice(),
                })
            }
            PhysicalOperator::SortedAggregate(agg) => {
                let mut columns = bumpalo::collections::Vec::new_in(self.arena);
                let input_schema = self.compute_output_schema(agg.input)?;

                for group_expr in agg.group_by.iter() {
                    let (name, data_type, nullable) =
                        self.infer_expr_type(group_expr, &input_schema)?;
                    columns.push(OutputColumn {
                        name,
                        data_type,
                        nullable,
                    });
                }

                for agg_expr in agg.aggregates.iter() {
                    let (name, data_type) = self.infer_aggregate_type(agg_expr, &input_schema)?;
                    columns.push(OutputColumn {
                        name,
                        data_type,
                        nullable: true,
                    });
                }

                Ok(OutputSchema {
                    columns: columns.into_bump_slice(),
                })
            }
            PhysicalOperator::SubqueryExec(subq) => Ok(subq.output_schema.clone()),
            PhysicalOperator::SetOpExec(set_op) => self.compute_output_schema(set_op.left),
            PhysicalOperator::WindowExec(window) => {
                let input_schema = self.compute_output_schema(window.input)?;
                let mut columns = bumpalo::collections::Vec::new_in(self.arena);

                for col in input_schema.columns {
                    columns.push(*col);
                }

                for window_func in window.window_functions.iter() {
                    let name = window_func.alias.unwrap_or(window_func.function_name);
                    columns.push(OutputColumn {
                        name,
                        data_type: crate::records::types::DataType::Int8,
                        nullable: false,
                    });
                }

                Ok(OutputSchema {
                    columns: columns.into_bump_slice(),
                })
            }
        }
    }

    fn compute_logical_output_schema(
        &self,
        op: &'a LogicalOperator<'a>,
    ) -> Result<OutputSchema<'a>> {
        match op {
            LogicalOperator::Scan(scan) => {
                let table_name = if let Some(schema) = scan.schema {
                    self.arena.alloc_str(&format!("{}.{}", schema, scan.table))
                } else {
                    scan.table
                };
                let table_def = self.catalog.resolve_table(table_name)?;
                let mut columns = bumpalo::collections::Vec::new_in(self.arena);

                for col in table_def.columns() {
                    columns.push(OutputColumn {
                        name: self.arena.alloc_str(col.name()),
                        data_type: col.data_type(),
                        nullable: col.is_nullable(),
                    });
                }

                Ok(OutputSchema {
                    columns: columns.into_bump_slice(),
                })
            }
            LogicalOperator::DualScan => Ok(OutputSchema { columns: &[] }),
            LogicalOperator::Project(project) => {
                let input_schema = self.compute_logical_output_schema(project.input)?;

                if project.expressions.is_empty() {
                    return Ok(input_schema);
                }

                let mut columns = bumpalo::collections::Vec::new_in(self.arena);

                for (i, expr) in project.expressions.iter().enumerate() {
                    let (name, data_type, nullable) = self.infer_expr_type(expr, &input_schema)?;
                    let col_name = if let Some(alias) = project.aliases.get(i).and_then(|a| *a) {
                        alias
                    } else {
                        name
                    };
                    columns.push(OutputColumn {
                        name: col_name,
                        data_type,
                        nullable,
                    });
                }

                Ok(OutputSchema {
                    columns: columns.into_bump_slice(),
                })
            }
            LogicalOperator::Filter(filter) => self.compute_logical_output_schema(filter.input),
            LogicalOperator::Sort(sort) => self.compute_logical_output_schema(sort.input),
            LogicalOperator::Limit(limit) => self.compute_logical_output_schema(limit.input),
            LogicalOperator::Join(join) => {
                let left_schema = self.compute_logical_output_schema(join.left)?;
                let right_schema = self.compute_logical_output_schema(join.right)?;
                let mut columns = bumpalo::collections::Vec::new_in(self.arena);

                for col in left_schema.columns {
                    columns.push(*col);
                }
                for col in right_schema.columns {
                    columns.push(*col);
                }

                Ok(OutputSchema {
                    columns: columns.into_bump_slice(),
                })
            }
            LogicalOperator::Aggregate(agg) => {
                let mut columns = bumpalo::collections::Vec::new_in(self.arena);
                let input_schema = self.compute_logical_output_schema(agg.input)?;

                for group_expr in agg.group_by.iter() {
                    let (name, data_type, nullable) =
                        self.infer_expr_type(group_expr, &input_schema)?;
                    columns.push(OutputColumn {
                        name,
                        data_type,
                        nullable,
                    });
                }

                for agg_expr in agg.aggregates.iter() {
                    if let Expr::Function(func) = agg_expr {
                        let name = self.arena.alloc_str(func.name.name);
                        let data_type = match func.name.name.to_lowercase().as_str() {
                            "count" => DataType::Int8,
                            "avg" => DataType::Float8,
                            "sum" | "min" | "max" => {
                                if let crate::sql::ast::FunctionArgs::Args(args) = func.args {
                                    if let Some(first_arg) = args.first() {
                                        let (_, dt, _) =
                                            self.infer_expr_type(first_arg.value, &input_schema)?;
                                        dt
                                    } else {
                                        DataType::Int8
                                    }
                                } else {
                                    DataType::Int8
                                }
                            }
                            _ => DataType::Text,
                        };
                        columns.push(OutputColumn {
                            name,
                            data_type,
                            nullable: true,
                        });
                    }
                }

                Ok(OutputSchema {
                    columns: columns.into_bump_slice(),
                })
            }
            LogicalOperator::Subquery(subq) => Ok(subq.output_schema.clone()),
            LogicalOperator::SetOp(set_op) => self.compute_logical_output_schema(set_op.left),
            LogicalOperator::Window(window) => {
                let input_schema = self.compute_logical_output_schema(window.input)?;
                let mut columns = bumpalo::collections::Vec::new_in(self.arena);

                for col in input_schema.columns {
                    columns.push(*col);
                }

                for window_func in window.window_functions.iter() {
                    let name = window_func.alias.unwrap_or(window_func.function_name);
                    columns.push(OutputColumn {
                        name,
                        data_type: DataType::Int8,
                        nullable: false,
                    });
                }

                Ok(OutputSchema {
                    columns: columns.into_bump_slice(),
                })
            }
            LogicalOperator::Values(_)
            | LogicalOperator::Insert(_)
            | LogicalOperator::Update(_)
            | LogicalOperator::Delete(_) => Ok(OutputSchema::empty()),
        }
    }

    fn infer_expr_type(
        &self,
        expr: &'a Expr<'a>,
        input_schema: &OutputSchema<'a>,
    ) -> Result<(&'a str, DataType, bool)> {
        use crate::sql::ast::Literal;

        match expr {
            Expr::Column(col_ref) => {
                if let Some(col) = input_schema.get_column(col_ref.column) {
                    Ok((col.name, col.data_type, col.nullable))
                } else {
                    Ok((col_ref.column, DataType::Text, true))
                }
            }
            Expr::Literal(lit) => {
                let (name, data_type) = match lit {
                    Literal::Null => ("?column?", DataType::Text),
                    Literal::Boolean(_) => ("?column?", DataType::Bool),
                    Literal::Integer(_) => ("?column?", DataType::Int8),
                    Literal::Float(_) => ("?column?", DataType::Float8),
                    Literal::String(_) => ("?column?", DataType::Text),
                    Literal::HexNumber(_) => ("?column?", DataType::Int8),
                    Literal::BinaryNumber(_) => ("?column?", DataType::Int8),
                };
                Ok((self.arena.alloc_str(name), data_type, true))
            }
            Expr::BinaryOp { left, op, right } => {
                use crate::sql::ast::BinaryOperator;
                let data_type = match op {
                    BinaryOperator::Plus
                    | BinaryOperator::Minus
                    | BinaryOperator::Multiply
                    | BinaryOperator::Divide
                    | BinaryOperator::Modulo
                    | BinaryOperator::Power
                    | BinaryOperator::BitwiseAnd
                    | BinaryOperator::BitwiseOr
                    | BinaryOperator::BitwiseXor
                    | BinaryOperator::LeftShift
                    | BinaryOperator::RightShift => {
                        let (_, left_type, _) = self.infer_expr_type(left, input_schema)?;
                        let (_, right_type, _) = self.infer_expr_type(right, input_schema)?;
                        if left_type == DataType::Float8 || right_type == DataType::Float8 {
                            DataType::Float8
                        } else {
                            left_type
                        }
                    }
                    BinaryOperator::Concat => DataType::Text,
                    BinaryOperator::VectorL2Distance
                    | BinaryOperator::VectorCosineDistance
                    | BinaryOperator::VectorInnerProduct => DataType::Float8,
                    _ => DataType::Bool,
                };
                Ok((self.arena.alloc_str("?column?"), data_type, true))
            }
            Expr::UnaryOp { expr, .. } => self.infer_expr_type(expr, input_schema),
            Expr::Function(func) => {
                let name = self.arena.alloc_str(func.name.name);
                Ok((name, DataType::Text, true))
            }
            _ => Ok((self.arena.alloc_str("?column?"), DataType::Text, true)),
        }
    }

    fn infer_aggregate_type(
        &self,
        agg: &AggregateExpr<'a>,
        input_schema: &OutputSchema<'a>,
    ) -> Result<(&'a str, DataType)> {
        let func_name = match agg.function {
            AggregateFunction::Count => "count",
            AggregateFunction::Sum => "sum",
            AggregateFunction::Avg => "avg",
            AggregateFunction::Min => "min",
            AggregateFunction::Max => "max",
        };

        let data_type = match agg.function {
            AggregateFunction::Count => DataType::Int8,
            AggregateFunction::Avg => DataType::Float8,
            AggregateFunction::Sum | AggregateFunction::Min | AggregateFunction::Max => {
                if let Some(arg) = agg.argument {
                    let (_, dt, _) = self.infer_expr_type(arg, input_schema)?;
                    dt
                } else {
                    DataType::Int8
                }
            }
        };

        Ok((self.arena.alloc_str(func_name), data_type))
    }

    fn logical_to_physical(&self, op: &'a LogicalOperator<'a>) -> Result<&'a PhysicalOperator<'a>> {
        match op {
            LogicalOperator::Scan(scan) => {
                let table_name = if let Some(schema) = scan.schema {
                    self.arena.alloc_str(&format!("{}.{}", schema, scan.table))
                } else {
                    scan.table
                };
                let table_def = self.catalog.resolve_table(table_name).ok();

                let physical = self
                    .arena
                    .alloc(PhysicalOperator::TableScan(PhysicalTableScan {
                        schema: scan.schema,
                        table: scan.table,
                        alias: scan.alias,
                        post_scan_filter: None,
                        table_def,
                        reverse: false,
                    }));
                Ok(physical)
            }
            LogicalOperator::DualScan => Ok(self.arena.alloc(PhysicalOperator::DualScan)),
            LogicalOperator::Filter(filter) => {
                if let Some(index_scan) = self.try_optimize_filter_to_index_scan(filter) {
                    return Ok(index_scan);
                }

                let input = self.logical_to_physical(filter.input)?;
                let physical = self
                    .arena
                    .alloc(PhysicalOperator::FilterExec(PhysicalFilterExec {
                        input,
                        predicate: filter.predicate,
                    }));
                Ok(physical)
            }
            LogicalOperator::Project(project) => {
                let input = self.logical_to_physical(project.input)?;
                let physical =
                    self.arena
                        .alloc(PhysicalOperator::ProjectExec(PhysicalProjectExec {
                            input,
                            expressions: project.expressions,
                            aliases: project.aliases,
                        }));
                Ok(physical)
            }
            LogicalOperator::Join(join) => {
                let left = self.logical_to_physical(join.left)?;
                let right = self.logical_to_physical(join.right)?;

                if self.has_equi_join_keys(join.condition) {
                    let equi_keys = self.extract_equi_join_keys(join.condition);
                    let join_keys = self.convert_equi_keys_to_join_keys(equi_keys);
                    let physical =
                        self.arena
                            .alloc(PhysicalOperator::GraceHashJoin(PhysicalGraceHashJoin {
                                left,
                                right,
                                join_type: join.join_type,
                                join_keys,
                                num_partitions: 16,
                            }));
                    Ok(physical)
                } else {
                    let physical = self.arena.alloc(PhysicalOperator::NestedLoopJoin(
                        PhysicalNestedLoopJoin {
                            left,
                            right,
                            join_type: join.join_type,
                            condition: join.condition,
                        },
                    ));
                    Ok(physical)
                }
            }
            LogicalOperator::Sort(sort) => {
                if let Some(optimized) = self.try_optimize_sort_to_index_scan(sort) {
                    return Ok(optimized);
                }
                let input = self.logical_to_physical(sort.input)?;
                let physical = self
                    .arena
                    .alloc(PhysicalOperator::SortExec(PhysicalSortExec {
                        input,
                        order_by: sort.order_by,
                    }));
                Ok(physical)
            }
            LogicalOperator::Limit(limit) => {
                if let Some(limit_val) = limit.limit {
                    if let LogicalOperator::Sort(sort) = limit.input {
                        let effective_limit = (limit_val + limit.offset.unwrap_or(0)) as usize;
                        if let Some(optimized) =
                            self.try_optimize_sort_to_index_scan_with_limit(sort, Some(effective_limit))
                        {
                            if limit.offset.unwrap_or(0) > 0 {
                                let physical = self
                                    .arena
                                    .alloc(PhysicalOperator::LimitExec(PhysicalLimitExec {
                                        input: optimized,
                                        limit: Some(limit_val),
                                        offset: limit.offset,
                                    }));
                                return Ok(physical);
                            }
                            return Ok(optimized);
                        }

                        let sort_input = self.logical_to_physical(sort.input)?;
                        let physical = self
                            .arena
                            .alloc(PhysicalOperator::TopKExec(PhysicalTopKExec {
                                input: sort_input,
                                order_by: sort.order_by,
                                limit: limit_val,
                                offset: limit.offset,
                            }));
                        return Ok(physical);
                    }
                }

                let input = self.logical_to_physical(limit.input)?;
                let physical = self
                    .arena
                    .alloc(PhysicalOperator::LimitExec(PhysicalLimitExec {
                        input,
                        limit: limit.limit,
                        offset: limit.offset,
                    }));
                Ok(physical)
            }
            LogicalOperator::Aggregate(agg) => {
                let input = self.logical_to_physical(agg.input)?;
                let aggregates = self.convert_aggregates_to_physical(agg.aggregates);
                let physical =
                    self.arena
                        .alloc(PhysicalOperator::HashAggregate(PhysicalHashAggregate {
                            input,
                            group_by: agg.group_by,
                            aggregates,
                        }));
                Ok(physical)
            }
            LogicalOperator::Values(_) => {
                bail!("Values operator cannot be directly converted to physical - only valid as INSERT source")
            }
            LogicalOperator::Insert(_) => {
                bail!(
                    "Insert operator cannot be converted to physical plan - DML handled separately"
                )
            }
            LogicalOperator::Update(_) => {
                bail!(
                    "Update operator cannot be converted to physical plan - DML handled separately"
                )
            }
            LogicalOperator::Delete(_) => {
                bail!(
                    "Delete operator cannot be converted to physical plan - DML handled separately"
                )
            }
            LogicalOperator::Subquery(subq) => {
                let child_plan = self.logical_to_physical(subq.plan)?;
                let physical =
                    self.arena
                        .alloc(PhysicalOperator::SubqueryExec(PhysicalSubqueryExec {
                            child_plan,
                            alias: subq.alias,
                            output_schema: subq.output_schema.clone(),
                        }));
                Ok(physical)
            }
            LogicalOperator::SetOp(set_op) => {
                let left = self.logical_to_physical(set_op.left)?;
                let right = self.logical_to_physical(set_op.right)?;
                let physical = self
                    .arena
                    .alloc(PhysicalOperator::SetOpExec(PhysicalSetOpExec {
                        left,
                        right,
                        kind: set_op.kind,
                        all: set_op.all,
                    }));
                Ok(physical)
            }
            LogicalOperator::Window(window) => {
                let input = self.logical_to_physical(window.input)?;
                let physical = self
                    .arena
                    .alloc(PhysicalOperator::WindowExec(PhysicalWindowExec {
                        input,
                        window_functions: window.window_functions,
                    }));
                Ok(physical)
            }
        }
    }

    fn convert_aggregates_to_physical(
        &self,
        aggregates: &'a [&'a Expr<'a>],
    ) -> &'a [AggregateExpr<'a>] {
        use crate::sql::ast::{FunctionArgs, FunctionCall};

        let mut result = bumpalo::collections::Vec::new_in(self.arena);

        for expr in aggregates {
            if let Expr::Function(FunctionCall {
                name,
                args,
                distinct,
                ..
            }) = expr
            {
                let func_name = name.name.to_ascii_lowercase();
                let function = match func_name.as_str() {
                    "count" => AggregateFunction::Count,
                    "sum" => AggregateFunction::Sum,
                    "avg" => AggregateFunction::Avg,
                    "min" => AggregateFunction::Min,
                    "max" => AggregateFunction::Max,
                    _ => continue,
                };

                let argument = match args {
                    FunctionArgs::Star => None,
                    FunctionArgs::None => None,
                    FunctionArgs::Args(func_args) => {
                        if func_args.is_empty() {
                            None
                        } else {
                            Some(func_args[0].value)
                        }
                    }
                };

                result.push(AggregateExpr {
                    function,
                    argument,
                    distinct: *distinct,
                });
            }
        }

        result.into_bump_slice()
    }

    fn try_optimize_filter_to_index_scan(
        &self,
        filter: &LogicalFilter<'a>,
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

        let (col_name, literal_expr) = self.extract_equality_predicate(filter.predicate)?;

        let matching_index = table_def.indexes().iter().find(|idx| {
            if idx.has_expressions() || idx.is_partial() {
                return false;
            }
            if idx.index_type() != crate::schema::IndexType::BTree {
                return false;
            }
            idx.columns()
                .next()
                .map(|first_col| first_col.eq_ignore_ascii_case(col_name))
                .unwrap_or(false)
        })?;

        let key_bytes = self.encode_literal_to_bytes(literal_expr)?;

        let index_name = self.arena.alloc_str(matching_index.name());
        let table_def_alloc = self.arena.alloc(table_def.clone());
        let _index_columns: Vec<String> = matching_index.columns().map(|s| s.to_string()).collect();

        let covered_columns = vec![col_name.to_string()];
        let residual = self.compute_residual_filter(filter.predicate, &covered_columns);

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

    fn extract_equality_predicate(&self, expr: &'a Expr<'a>) -> Option<(&'a str, &'a Expr<'a>)> {
        use crate::sql::ast::BinaryOperator;

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
                self.extract_equality_predicate(left)
                    .or_else(|| self.extract_equality_predicate(right))
            }
            _ => None,
        }
    }

    fn try_optimize_sort_to_index_scan(
        &self,
        sort: &LogicalSort<'a>,
    ) -> Option<&'a PhysicalOperator<'a>> {
        self.try_optimize_sort_to_index_scan_with_limit(sort, None)
    }

    fn try_optimize_sort_to_index_scan_with_limit(
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

    fn compute_residual_filter(
        &self,
        predicate: &'a Expr<'a>,
        index_columns: &[String],
    ) -> Option<&'a Expr<'a>> {
        use crate::sql::ast::BinaryOperator;

        match predicate {
            Expr::BinaryOp { left, op, right } => match op {
                BinaryOperator::And => {
                    let left_residual = self.compute_residual_filter(left, index_columns);
                    let right_residual = self.compute_residual_filter(right, index_columns);

                    match (left_residual, right_residual) {
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
                BinaryOperator::Eq
                | BinaryOperator::Lt
                | BinaryOperator::LtEq
                | BinaryOperator::Gt
                | BinaryOperator::GtEq => {
                    if self.predicate_uses_index_column(predicate, index_columns) {
                        None
                    } else {
                        Some(predicate)
                    }
                }
                _ => Some(predicate),
            },
            Expr::Between { expr, .. } => {
                if self.predicate_uses_index_column(expr, index_columns) {
                    None
                } else {
                    Some(predicate)
                }
            }
            _ => Some(predicate),
        }
    }

    fn predicate_uses_index_column(&self, expr: &Expr<'a>, index_columns: &[String]) -> bool {
        match expr {
            Expr::Column(col_ref) => index_columns
                .iter()
                .any(|c| c.eq_ignore_ascii_case(col_ref.column)),
            Expr::BinaryOp { left, right, .. } => {
                self.predicate_uses_index_column(left, index_columns)
                    || self.predicate_uses_index_column(right, index_columns)
            }
            Expr::Between { expr, .. } => self.predicate_uses_index_column(expr, index_columns),
            _ => false,
        }
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

    pub fn extract_scan_bounds_for_column(
        &self,
        predicate: &'a Expr<'a>,
        target_column: &str,
    ) -> ColumnScanBounds<'a> {
        let mut bounds = ColumnScanBounds {
            lower: None,
            upper: None,
            point_value: None,
        };

        self.collect_bounds_from_expr(predicate, target_column, &mut bounds);
        bounds
    }

    fn collect_bounds_from_expr(
        &self,
        expr: &'a Expr<'a>,
        target_column: &str,
        bounds: &mut ColumnScanBounds<'a>,
    ) {
        use crate::sql::ast::BinaryOperator;

        match expr {
            Expr::BinaryOp { left, op, right } => match op {
                BinaryOperator::And => {
                    self.collect_bounds_from_expr(left, target_column, bounds);
                    self.collect_bounds_from_expr(right, target_column, bounds);
                }
                BinaryOperator::Eq => {
                    if self.expr_references_column(left, target_column) {
                        bounds.point_value = Some(ColumnBound {
                            value: right,
                            inclusive: true,
                        });
                    } else if self.expr_references_column(right, target_column) {
                        bounds.point_value = Some(ColumnBound {
                            value: left,
                            inclusive: true,
                        });
                    }
                }
                BinaryOperator::Lt => {
                    if self.expr_references_column(left, target_column) {
                        bounds.upper = Some(ColumnBound {
                            value: right,
                            inclusive: false,
                        });
                    } else if self.expr_references_column(right, target_column) {
                        bounds.lower = Some(ColumnBound {
                            value: left,
                            inclusive: false,
                        });
                    }
                }
                BinaryOperator::LtEq => {
                    if self.expr_references_column(left, target_column) {
                        bounds.upper = Some(ColumnBound {
                            value: right,
                            inclusive: true,
                        });
                    } else if self.expr_references_column(right, target_column) {
                        bounds.lower = Some(ColumnBound {
                            value: left,
                            inclusive: true,
                        });
                    }
                }
                BinaryOperator::Gt => {
                    if self.expr_references_column(left, target_column) {
                        bounds.lower = Some(ColumnBound {
                            value: right,
                            inclusive: false,
                        });
                    } else if self.expr_references_column(right, target_column) {
                        bounds.upper = Some(ColumnBound {
                            value: left,
                            inclusive: false,
                        });
                    }
                }
                BinaryOperator::GtEq => {
                    if self.expr_references_column(left, target_column) {
                        bounds.lower = Some(ColumnBound {
                            value: right,
                            inclusive: true,
                        });
                    } else if self.expr_references_column(right, target_column) {
                        bounds.upper = Some(ColumnBound {
                            value: left,
                            inclusive: true,
                        });
                    }
                }
                _ => {}
            },
            Expr::Between {
                expr: between_expr,
                negated,
                low,
                high,
            } => {
                if !negated && self.expr_references_column(between_expr, target_column) {
                    bounds.lower = Some(ColumnBound {
                        value: low,
                        inclusive: true,
                    });
                    bounds.upper = Some(ColumnBound {
                        value: high,
                        inclusive: true,
                    });
                }
            }
            _ => {}
        }
    }

    fn expr_references_column(&self, expr: &Expr<'a>, target_column: &str) -> bool {
        match expr {
            Expr::Column(col_ref) => col_ref.column.eq_ignore_ascii_case(target_column),
            _ => false,
        }
    }

    pub fn bounds_to_scan_type(&self, bounds: &ColumnScanBounds<'a>) -> ScanBoundType {
        if bounds.point_value.is_some() {
            ScanBoundType::Point
        } else if bounds.lower.is_some() || bounds.upper.is_some() {
            ScanBoundType::Range
        } else {
            ScanBoundType::Full
        }
    }

    fn encode_literal_to_bytes(&self, expr: &Expr<'a>) -> Option<&'a [u8]> {
        match expr {
            Expr::Literal(lit) => {
                let mut buf = bumpalo::collections::Vec::with_capacity_in(32, self.arena);
                match lit {
                    Literal::Null => return None,
                    Literal::Boolean(b) => {
                        buf.push(if *b { 0x03 } else { 0x02 });
                    }
                    Literal::Integer(s) => {
                        if let Ok(n) = s.parse::<i64>() {
                            self.encode_int_to_arena(n, &mut buf);
                        } else {
                            return None;
                        }
                    }
                    Literal::Float(s) => {
                        if let Ok(f) = s.parse::<f64>() {
                            self.encode_float_to_arena(f, &mut buf);
                        } else {
                            return None;
                        }
                    }
                    Literal::String(s) => {
                        self.encode_text_to_arena(s, &mut buf);
                    }
                    Literal::HexNumber(s) => {
                        if let Ok(n) = i64::from_str_radix(s.trim_start_matches("0x"), 16) {
                            self.encode_int_to_arena(n, &mut buf);
                        } else {
                            return None;
                        }
                    }
                    Literal::BinaryNumber(s) => {
                        if let Ok(n) = i64::from_str_radix(s.trim_start_matches("0b"), 2) {
                            self.encode_int_to_arena(n, &mut buf);
                        } else {
                            return None;
                        }
                    }
                }
                Some(buf.into_bump_slice())
            }
            _ => None,
        }
    }

    fn encode_int_to_arena(&self, n: i64, buf: &mut bumpalo::collections::Vec<'a, u8>) {
        use crate::encoding::key::type_prefix;
        if n < 0 {
            buf.push(type_prefix::NEG_INT);
            buf.extend((n as u64).to_be_bytes());
        } else if n == 0 {
            buf.push(type_prefix::ZERO);
        } else {
            buf.push(type_prefix::POS_INT);
            buf.extend((n as u64).to_be_bytes());
        }
    }

    fn encode_float_to_arena(&self, f: f64, buf: &mut bumpalo::collections::Vec<'a, u8>) {
        use crate::encoding::key::type_prefix;
        if f.is_nan() {
            buf.push(type_prefix::NAN);
        } else if f == f64::NEG_INFINITY {
            buf.push(type_prefix::NEG_INFINITY);
        } else if f == f64::INFINITY {
            buf.push(type_prefix::POS_INFINITY);
        } else if f < 0.0 {
            buf.push(type_prefix::NEG_FLOAT);
            buf.extend((!f.to_bits()).to_be_bytes());
        } else if f == 0.0 {
            buf.push(type_prefix::ZERO);
        } else {
            buf.push(type_prefix::POS_FLOAT);
            buf.extend((f.to_bits() ^ (1u64 << 63)).to_be_bytes());
        }
    }

    fn encode_text_to_arena(&self, s: &str, buf: &mut bumpalo::collections::Vec<'a, u8>) {
        use crate::encoding::key::type_prefix;
        buf.push(type_prefix::TEXT);
        for &byte in s.as_bytes() {
            match byte {
                0x00 => {
                    buf.push(0x00);
                    buf.push(0xFF);
                }
                0xFF => {
                    buf.push(0xFF);
                    buf.push(0x00);
                }
                b => buf.push(b),
            }
        }
        buf.push(0x00);
        buf.push(0x00);
    }

    #[allow(dead_code)]
    fn encode_scan_bounds(&self, bounds: &ColumnScanBounds<'a>) -> ScanRange<'a> {
        if let Some(point) = bounds.point_value {
            if let Some(encoded) = self.encode_literal_to_bytes(point.value) {
                return ScanRange::PrefixScan { prefix: encoded };
            }
        }

        if bounds.lower.is_some() || bounds.upper.is_some() {
            let start = bounds
                .lower
                .and_then(|b| self.encode_literal_to_bytes(b.value));
            let end = bounds
                .upper
                .and_then(|b| self.encode_literal_to_bytes(b.value));
            return ScanRange::RangeScan { start, end };
        }

        ScanRange::FullScan
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

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ColumnBound<'a> {
    pub value: &'a Expr<'a>,
    pub inclusive: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct ColumnScanBounds<'a> {
    pub lower: Option<ColumnBound<'a>>,
    pub upper: Option<ColumnBound<'a>>,
    pub point_value: Option<ColumnBound<'a>>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScanBoundType {
    Point,
    Range,
    Full,
}
