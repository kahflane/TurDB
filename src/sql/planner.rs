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
//! - Grace Hash Join: 16 partitions, spill to wal/ on overflow
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

use crate::schema::Catalog;
use crate::sql::ast::{Expr, JoinType, Statement};
use bumpalo::Bump;
use eyre::{bail, Result};

#[derive(Debug, Clone, PartialEq)]
pub enum LogicalOperator<'a> {
    Scan(LogicalScan<'a>),
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
pub struct LogicalPlan<'a> {
    pub root: &'a LogicalOperator<'a>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum PhysicalOperator<'a> {
    TableScan(PhysicalTableScan<'a>),
    IndexScan(PhysicalIndexScan<'a>),
    FilterExec(PhysicalFilterExec<'a>),
    ProjectExec(PhysicalProjectExec<'a>),
    NestedLoopJoin(PhysicalNestedLoopJoin<'a>),
    GraceHashJoin(PhysicalGraceHashJoin<'a>),
    HashAggregate(PhysicalHashAggregate<'a>),
    SortedAggregate(PhysicalSortedAggregate<'a>),
    SortExec(PhysicalSortExec<'a>),
    LimitExec(PhysicalLimitExec<'a>),
}

#[derive(Debug, Clone, PartialEq)]
pub struct PhysicalTableScan<'a> {
    pub schema: Option<&'a str>,
    pub table: &'a str,
    pub alias: Option<&'a str>,
    pub post_scan_filter: Option<&'a Expr<'a>>,
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
pub struct PhysicalPlan<'a> {
    pub root: &'a PhysicalOperator<'a>,
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
        let mut current: &'a LogicalOperator<'a> = match select.from {
            Some(from) => self.plan_from_clause(from)?,
            None => bail!("SELECT without FROM clause not yet supported"),
        };

        if let Some(predicate) = select.where_clause {
            let filter = self.arena.alloc(LogicalOperator::Filter(LogicalFilter {
                input: current,
                predicate,
            }));
            current = filter;
        }

        if !select.group_by.is_empty() {
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

        if !select.order_by.is_empty() {
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

        Ok(LogicalPlan { root: current })
    }

    fn plan_from_clause(
        &self,
        from: &'a crate::sql::ast::FromClause<'a>,
    ) -> Result<&'a LogicalOperator<'a>> {
        use crate::sql::ast::FromClause;

        match from {
            FromClause::Table(table_ref) => {
                self.validate_table_exists(table_ref.schema, table_ref.name)?;

                let scan = self.arena.alloc(LogicalOperator::Scan(LogicalScan {
                    schema: table_ref.schema,
                    table: table_ref.name,
                    alias: table_ref.alias,
                }));
                Ok(scan)
            }
            FromClause::Join(join) => self.plan_join(join),
            FromClause::Subquery { query: _, alias: _ } => {
                bail!("subqueries in FROM clause not yet implemented")
            }
            FromClause::Lateral {
                subquery: _,
                alias: _,
            } => {
                bail!("LATERAL subqueries not yet implemented")
            }
        }
    }

    fn plan_join(
        &self,
        join: &'a crate::sql::ast::JoinClause<'a>,
    ) -> Result<&'a LogicalOperator<'a>> {
        let left = self.plan_from_clause(join.left)?;
        let right = self.plan_from_clause(join.right)?;

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
                if self.is_aggregate_function(expr) {
                    aggregates.push(*expr);
                }
            }
        }

        aggregates.into_bump_slice()
    }

    fn is_aggregate_function(&self, expr: &Expr<'a>) -> bool {
        use crate::sql::ast::FunctionCall;

        if let Expr::Function(FunctionCall { name, over, .. }) = expr {
            if over.is_some() {
                return false;
            }
            let func_name = name.name.to_ascii_lowercase();
            matches!(
                func_name.as_str(),
                "count" | "sum" | "avg" | "min" | "max"
            )
        } else {
            false
        }
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
            format!("{}.{}", s, table)
        } else {
            table.to_string()
        };

        self.catalog
            .resolve_table(&table_name)
            .map(|_| ())
    }

    fn plan_insert(&self, insert: &crate::sql::ast::InsertStmt<'a>) -> Result<LogicalPlan<'a>> {
        self.validate_table_exists(insert.table.schema, insert.table.name)?;

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
        Ok(PhysicalPlan { root: physical_root })
    }

    fn logical_to_physical(
        &self,
        op: &'a LogicalOperator<'a>,
    ) -> Result<&'a PhysicalOperator<'a>> {
        match op {
            LogicalOperator::Scan(scan) => {
                let physical = self.arena.alloc(PhysicalOperator::TableScan(PhysicalTableScan {
                    schema: scan.schema,
                    table: scan.table,
                    alias: scan.alias,
                    post_scan_filter: None,
                }));
                Ok(physical)
            }
            LogicalOperator::Filter(filter) => {
                if let Some(index_scan) = self.try_optimize_filter_to_index_scan(filter) {
                    return Ok(index_scan);
                }

                let input = self.logical_to_physical(filter.input)?;
                let physical = self.arena.alloc(PhysicalOperator::FilterExec(PhysicalFilterExec {
                    input,
                    predicate: filter.predicate,
                }));
                Ok(physical)
            }
            LogicalOperator::Project(project) => {
                let input = self.logical_to_physical(project.input)?;
                let physical = self.arena.alloc(PhysicalOperator::ProjectExec(PhysicalProjectExec {
                    input,
                    expressions: project.expressions,
                    aliases: project.aliases,
                }));
                Ok(physical)
            }
            LogicalOperator::Join(join) => {
                let left = self.logical_to_physical(join.left)?;
                let right = self.logical_to_physical(join.right)?;
                let physical = self.arena.alloc(PhysicalOperator::NestedLoopJoin(PhysicalNestedLoopJoin {
                    left,
                    right,
                    join_type: join.join_type,
                    condition: join.condition,
                }));
                Ok(physical)
            }
            LogicalOperator::Sort(sort) => {
                let input = self.logical_to_physical(sort.input)?;
                let physical = self.arena.alloc(PhysicalOperator::SortExec(PhysicalSortExec {
                    input,
                    order_by: sort.order_by,
                }));
                Ok(physical)
            }
            LogicalOperator::Limit(limit) => {
                let input = self.logical_to_physical(limit.input)?;
                let physical = self.arena.alloc(PhysicalOperator::LimitExec(PhysicalLimitExec {
                    input,
                    limit: limit.limit,
                    offset: limit.offset,
                }));
                Ok(physical)
            }
            LogicalOperator::Aggregate(agg) => {
                let input = self.logical_to_physical(agg.input)?;
                let aggregates = self.convert_aggregates_to_physical(agg.aggregates);
                let physical = self.arena.alloc(PhysicalOperator::HashAggregate(PhysicalHashAggregate {
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
                bail!("Insert operator cannot be converted to physical plan - DML handled separately")
            }
            LogicalOperator::Update(_) => {
                bail!("Update operator cannot be converted to physical plan - DML handled separately")
            }
            LogicalOperator::Delete(_) => {
                bail!("Delete operator cannot be converted to physical plan - DML handled separately")
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
        let scan = match filter.input {
            LogicalOperator::Scan(s) => s,
            _ => return None,
        };

        let table_name = if let Some(schema) = scan.schema {
            let full_name = self
                .arena
                .alloc_str(&format!("{}.{}", schema, scan.table));
            full_name
        } else {
            scan.table
        };

        let table_def = self.catalog.resolve_table(table_name).ok()?;

        let filter_columns = self.extract_filter_columns(filter.predicate);
        let best_index = self.select_best_index(table_def, &filter_columns)?;

        let first_index_col = best_index.columns().first()?;
        let bounds = self.extract_scan_bounds_for_column(filter.predicate, first_index_col);
        let scan_type = self.bounds_to_scan_type(&bounds);

        if scan_type == ScanBoundType::Full {
            return None;
        }

        let key_range = match scan_type {
            ScanBoundType::Point => ScanRange::PrefixScan {
                prefix: &[],
            },
            ScanBoundType::Range => ScanRange::RangeScan {
                start: None,
                end: None,
            },
            ScanBoundType::Full => ScanRange::FullScan,
        };

        let residual = self.compute_residual_filter(filter.predicate, best_index.columns());

        let index_scan =
            self.arena
                .alloc(PhysicalOperator::IndexScan(PhysicalIndexScan {
                    schema: scan.schema,
                    table: scan.table,
                    index_name: best_index.name(),
                    key_range,
                    residual_filter: residual,
                }));

        Some(index_scan)
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

    pub fn extract_filter_columns(&self, expr: &Expr<'a>) -> Vec<&'a str> {
        let mut columns = Vec::new();
        self.collect_columns_from_expr(expr, &mut columns);
        columns
    }

    fn collect_columns_from_expr(&self, expr: &Expr<'a>, columns: &mut Vec<&'a str>) {
        match expr {
            Expr::Column(col_ref) => {
                columns.push(col_ref.column);
            }
            Expr::BinaryOp { left, op, right } => {
                match op {
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
                }
            }
            Expr::IsNull { expr, .. } => {
                self.collect_columns_from_expr(expr, columns);
            }
            Expr::Between { expr, low, high, .. } => {
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
                if idx.columns().is_empty() {
                    return false;
                }
                idx.columns()
                    .first()
                    .map(|first_col| filter_columns.contains(&first_col.as_str()))
                    .unwrap_or(false)
            })
            .collect()
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

        candidates.into_iter().max_by(|a, b| {
            let a_matched = a
                .columns()
                .iter()
                .filter(|c| filter_columns.contains(&c.as_str()))
                .count();
            let b_matched = b
                .columns()
                .iter()
                .filter(|c| filter_columns.contains(&c.as_str()))
                .count();

            match a_matched.cmp(&b_matched) {
                std::cmp::Ordering::Equal => {
                    if a.is_unique() && !b.is_unique() {
                        std::cmp::Ordering::Greater
                    } else if !a.is_unique() && b.is_unique() {
                        std::cmp::Ordering::Less
                    } else {
                        std::cmp::Ordering::Equal
                    }
                }
                ord => ord,
            }
        })
    }

    pub fn estimate_cardinality(&self, op: &LogicalOperator<'a>) -> u64 {
        const DEFAULT_TABLE_CARDINALITY: u64 = 1000;
        const FILTER_SELECTIVITY: f64 = 0.1;
        const JOIN_SELECTIVITY: f64 = 0.1;

        match op {
            LogicalOperator::Scan(_) => DEFAULT_TABLE_CARDINALITY,
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
        }
    }

    pub fn extract_join_tables(&self, op: &'a LogicalOperator<'a>) -> Vec<&'a LogicalOperator<'a>> {
        let mut tables = Vec::new();
        self.collect_join_tables(op, &mut tables);
        tables
    }

    fn collect_join_tables(
        &self,
        op: &'a LogicalOperator<'a>,
        tables: &mut Vec<&'a LogicalOperator<'a>>,
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
        tables_with_card: Vec<(&'a LogicalOperator<'a>, u64)>,
    ) -> Vec<&'a LogicalOperator<'a>> {
        let mut sorted = tables_with_card;
        sorted.sort_by_key(|(_, card)| *card);
        sorted.into_iter().map(|(op, _)| op).collect()
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
            Expr::BinaryOp { left, op, right } => {
                match op {
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
                }
            }
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

    pub fn extract_equi_join_keys(
        &self,
        condition: Option<&'a Expr<'a>>,
    ) -> &'a [EquiJoinKey<'a>] {
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
                        (Some(l), Some(r)) => {
                            Some(self.arena.alloc(Expr::BinaryOp {
                                left: l,
                                op: BinaryOperator::And,
                                right: r,
                            }))
                        }
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

#[cfg(test)]
mod tests {
    use super::*;
    use bumpalo::Bump;

    fn create_test_catalog() -> Catalog {
        use crate::records::types::DataType;
        use crate::schema::ColumnDef;

        let mut catalog = Catalog::new();

        let users_columns = vec![
            ColumnDef::new("id", DataType::Int8),
            ColumnDef::new("name", DataType::Text),
            ColumnDef::new("email", DataType::Text),
            ColumnDef::new("active", DataType::Bool),
        ];
        catalog.create_table("root", "users", users_columns).unwrap();

        let orders_columns = vec![
            ColumnDef::new("id", DataType::Int8),
            ColumnDef::new("user_id", DataType::Int8),
            ColumnDef::new("product_id", DataType::Int8),
            ColumnDef::new("quantity", DataType::Int4),
            ColumnDef::new("total", DataType::Float8),
        ];
        catalog.create_table("root", "orders", orders_columns).unwrap();

        let products_columns = vec![
            ColumnDef::new("id", DataType::Int8),
            ColumnDef::new("name", DataType::Text),
            ColumnDef::new("price", DataType::Float8),
        ];
        catalog.create_table("root", "products", products_columns).unwrap();

        catalog
    }

    #[test]
    fn logical_scan_has_table_and_schema() {
        let scan = LogicalScan {
            schema: Some("public"),
            table: "users",
            alias: Some("u"),
        };
        assert_eq!(scan.schema, Some("public"));
        assert_eq!(scan.table, "users");
        assert_eq!(scan.alias, Some("u"));
    }

    #[test]
    fn logical_operator_scan_variant_exists() {
        let scan = LogicalScan {
            schema: None,
            table: "orders",
            alias: None,
        };
        let op = LogicalOperator::Scan(scan);
        assert!(matches!(op, LogicalOperator::Scan(_)));
    }

    #[test]
    fn logical_operator_all_variants_exist() {
        let bump = Bump::new();

        let scan = LogicalScan {
            schema: None,
            table: "t",
            alias: None,
        };
        let scan_op = bump.alloc(LogicalOperator::Scan(scan));

        assert!(matches!(
            LogicalOperator::Scan(LogicalScan {
                schema: None,
                table: "t",
                alias: None
            }),
            LogicalOperator::Scan(_)
        ));

        let filter = LogicalFilter {
            input: scan_op,
            predicate: bump.alloc(Expr::Literal(crate::sql::ast::Literal::Boolean(true))),
        };
        assert!(matches!(
            LogicalOperator::Filter(filter),
            LogicalOperator::Filter(_)
        ));

        let values = LogicalValues { rows: &[] };
        assert!(matches!(
            LogicalOperator::Values(values),
            LogicalOperator::Values(_)
        ));
    }

    #[test]
    fn logical_limit_has_limit_and_offset() {
        let bump = Bump::new();
        let scan = bump.alloc(LogicalOperator::Scan(LogicalScan {
            schema: None,
            table: "t",
            alias: None,
        }));

        let limit = LogicalLimit {
            input: scan,
            limit: Some(10),
            offset: Some(5),
        };

        assert_eq!(limit.limit, Some(10));
        assert_eq!(limit.offset, Some(5));
    }

    #[test]
    fn physical_table_scan_has_table_and_filter() {
        let bump = Bump::new();
        let filter = bump.alloc(Expr::Literal(crate::sql::ast::Literal::Boolean(true)));

        let scan = PhysicalTableScan {
            schema: Some("root"),
            table: "users",
            alias: None,
            post_scan_filter: Some(filter),
        };

        assert_eq!(scan.schema, Some("root"));
        assert_eq!(scan.table, "users");
        assert!(scan.post_scan_filter.is_some());
    }

    #[test]
    fn physical_index_scan_has_key_range() {
        let scan = PhysicalIndexScan {
            schema: None,
            table: "users",
            index_name: "users_pk",
            key_range: ScanRange::FullScan,
            residual_filter: None,
        };

        assert!(matches!(scan.key_range, ScanRange::FullScan));

        let prefix: &[u8] = &[0x01, 0x02];
        let scan_prefix = PhysicalIndexScan {
            schema: None,
            table: "users",
            index_name: "users_pk",
            key_range: ScanRange::PrefixScan { prefix },
            residual_filter: None,
        };

        assert!(matches!(
            scan_prefix.key_range,
            ScanRange::PrefixScan { .. }
        ));
    }

    #[test]
    fn physical_operator_all_variants_exist() {
        let bump = Bump::new();

        let table_scan = PhysicalTableScan {
            schema: None,
            table: "t",
            alias: None,
            post_scan_filter: None,
        };
        let table_scan_op = bump.alloc(PhysicalOperator::TableScan(table_scan));
        assert!(matches!(table_scan_op, PhysicalOperator::TableScan(_)));

        let filter = PhysicalFilterExec {
            input: table_scan_op,
            predicate: bump.alloc(Expr::Literal(crate::sql::ast::Literal::Boolean(true))),
        };
        assert!(matches!(
            PhysicalOperator::FilterExec(filter),
            PhysicalOperator::FilterExec(_)
        ));

        let limit = PhysicalLimitExec {
            input: table_scan_op,
            limit: Some(100),
            offset: None,
        };
        assert!(matches!(
            PhysicalOperator::LimitExec(limit),
            PhysicalOperator::LimitExec(_)
        ));
    }

    #[test]
    fn physical_grace_hash_join_has_partitions() {
        let bump = Bump::new();

        let left_scan = bump.alloc(PhysicalOperator::TableScan(PhysicalTableScan {
            schema: None,
            table: "orders",
            alias: None,
            post_scan_filter: None,
        }));

        let right_scan = bump.alloc(PhysicalOperator::TableScan(PhysicalTableScan {
            schema: None,
            table: "customers",
            alias: None,
            post_scan_filter: None,
        }));

        let join = PhysicalGraceHashJoin {
            left: left_scan,
            right: right_scan,
            join_type: JoinType::Inner,
            join_keys: &[],
            num_partitions: 16,
        };

        assert_eq!(join.num_partitions, 16);
        assert_eq!(join.join_type, JoinType::Inner);
    }

    #[test]
    fn aggregate_function_variants_exist() {
        assert!(matches!(AggregateFunction::Count, AggregateFunction::Count));
        assert!(matches!(AggregateFunction::Sum, AggregateFunction::Sum));
        assert!(matches!(AggregateFunction::Avg, AggregateFunction::Avg));
        assert!(matches!(AggregateFunction::Min, AggregateFunction::Min));
        assert!(matches!(AggregateFunction::Max, AggregateFunction::Max));
    }

    #[test]
    fn plan_node_logical_variant() {
        let bump = Bump::new();

        let scan = bump.alloc(LogicalOperator::Scan(LogicalScan {
            schema: None,
            table: "users",
            alias: None,
        }));

        let node = PlanNode::Logical(scan);
        assert!(node.is_logical());
        assert!(!node.is_physical());
        assert!(node.as_logical().is_some());
        assert!(node.as_physical().is_none());
    }

    #[test]
    fn plan_node_physical_variant() {
        let bump = Bump::new();

        let scan = bump.alloc(PhysicalOperator::TableScan(PhysicalTableScan {
            schema: None,
            table: "users",
            alias: None,
            post_scan_filter: None,
        }));

        let node = PlanNode::Physical(scan);
        assert!(node.is_physical());
        assert!(!node.is_logical());
        assert!(node.as_physical().is_some());
        assert!(node.as_logical().is_none());
    }

    #[test]
    fn plan_node_as_logical_returns_operator() {
        let bump = Bump::new();

        let logical_scan = bump.alloc(LogicalOperator::Scan(LogicalScan {
            schema: Some("public"),
            table: "orders",
            alias: Some("o"),
        }));

        let node = PlanNode::Logical(logical_scan);
        let op = node.as_logical().unwrap();

        if let LogicalOperator::Scan(scan) = op {
            assert_eq!(scan.table, "orders");
            assert_eq!(scan.alias, Some("o"));
        } else {
            panic!("Expected Scan operator");
        }
    }

    #[test]
    fn planner_new_creates_instance() {
        let catalog = Catalog::new();
        let arena = Bump::new();
        let planner = Planner::new(&catalog, &arena);

        assert!(std::ptr::eq(planner.catalog(), &catalog));
        assert!(std::ptr::eq(planner.arena(), &arena));
    }

    #[test]
    fn planner_create_logical_plan_returns_error_for_unsupported() {
        let catalog = Catalog::new();
        let arena = Bump::new();
        let planner = Planner::new(&catalog, &arena);

        let stmt = Statement::Commit;
        let result = planner.create_logical_plan(&stmt);
        assert!(result.is_err());
    }

    #[test]
    fn planner_create_physical_plan_dispatches_to_logical_first() {
        let catalog = Catalog::new();
        let arena = Bump::new();
        let planner = Planner::new(&catalog, &arena);

        let stmt = Statement::Commit;
        let result = planner.create_physical_plan(&stmt);
        assert!(result.is_err());
    }

    #[test]
    fn plan_select_simple_from_table() {
        use crate::sql::ast::{Distinct, FromClause, SelectColumn, SelectStmt, TableRef};

        let arena = Bump::new();
        let catalog = create_test_catalog();
        let planner = Planner::new(&catalog, &arena);

        let table_ref = TableRef {
            schema: None,
            name: "users",
            alias: None,
        };
        let from = arena.alloc(FromClause::Table(table_ref));

        let select = arena.alloc(SelectStmt {
            with: None,
            distinct: Distinct::All,
            columns: &[SelectColumn::AllColumns],
            from: Some(from),
            where_clause: None,
            group_by: &[],
            having: None,
            order_by: &[],
            limit: None,
            offset: None,
            set_op: None,
            for_clause: None,
        });

        let stmt = Statement::Select(select);
        let result = planner.create_logical_plan(&stmt);
        assert!(result.is_ok());

        let plan = result.unwrap();
        assert!(matches!(plan.root, LogicalOperator::Project(_)));

        if let LogicalOperator::Project(project) = plan.root {
            assert!(matches!(project.input, LogicalOperator::Scan(_)));
            if let LogicalOperator::Scan(scan) = project.input {
                assert_eq!(scan.table, "users");
            }
        }
    }

    #[test]
    fn plan_select_with_where_clause() {
        use crate::sql::ast::{Distinct, FromClause, Literal, SelectColumn, SelectStmt, TableRef};

        let arena = Bump::new();
        let catalog = create_test_catalog();
        let planner = Planner::new(&catalog, &arena);

        let table_ref = TableRef {
            schema: None,
            name: "users",
            alias: None,
        };
        let from = arena.alloc(FromClause::Table(table_ref));
        let predicate = arena.alloc(Expr::Literal(Literal::Boolean(true)));

        let select = arena.alloc(SelectStmt {
            with: None,
            distinct: Distinct::All,
            columns: &[SelectColumn::AllColumns],
            from: Some(from),
            where_clause: Some(predicate),
            group_by: &[],
            having: None,
            order_by: &[],
            limit: None,
            offset: None,
            set_op: None,
            for_clause: None,
        });

        let stmt = Statement::Select(select);
        let result = planner.create_logical_plan(&stmt);
        assert!(result.is_ok());

        let plan = result.unwrap();
        if let LogicalOperator::Project(project) = plan.root {
            assert!(matches!(project.input, LogicalOperator::Filter(_)));
        }
    }

    #[test]
    fn plan_select_with_limit() {
        use crate::sql::ast::{Distinct, FromClause, Literal, SelectColumn, SelectStmt, TableRef};

        let arena = Bump::new();
        let catalog = create_test_catalog();
        let planner = Planner::new(&catalog, &arena);

        let table_ref = TableRef {
            schema: None,
            name: "users",
            alias: None,
        };
        let from = arena.alloc(FromClause::Table(table_ref));
        let limit_expr = arena.alloc(Expr::Literal(Literal::Integer("10")));

        let select = arena.alloc(SelectStmt {
            with: None,
            distinct: Distinct::All,
            columns: &[SelectColumn::AllColumns],
            from: Some(from),
            where_clause: None,
            group_by: &[],
            having: None,
            order_by: &[],
            limit: Some(limit_expr),
            offset: None,
            set_op: None,
            for_clause: None,
        });

        let stmt = Statement::Select(select);
        let result = planner.create_logical_plan(&stmt);
        assert!(result.is_ok());

        let plan = result.unwrap();
        assert!(matches!(plan.root, LogicalOperator::Limit(_)));

        if let LogicalOperator::Limit(limit) = plan.root {
            assert_eq!(limit.limit, Some(10));
        }
    }

    #[test]
    fn plan_select_with_join() {
        use crate::sql::ast::{
            Distinct, FromClause, JoinClause, JoinCondition, Literal, SelectColumn, SelectStmt,
            TableRef,
        };

        let arena = Bump::new();
        let catalog = create_test_catalog();
        let planner = Planner::new(&catalog, &arena);

        let left_table = TableRef {
            schema: None,
            name: "orders",
            alias: None,
        };
        let left_from = arena.alloc(FromClause::Table(left_table));

        let right_table = TableRef {
            schema: None,
            name: "users",
            alias: None,
        };
        let right_from = arena.alloc(FromClause::Table(right_table));

        let condition = arena.alloc(Expr::Literal(Literal::Boolean(true)));
        let join_clause = arena.alloc(JoinClause {
            left: left_from,
            join_type: JoinType::Inner,
            right: right_from,
            condition: JoinCondition::On(condition),
        });
        let from = arena.alloc(FromClause::Join(join_clause));

        let select = arena.alloc(SelectStmt {
            with: None,
            distinct: Distinct::All,
            columns: &[SelectColumn::AllColumns],
            from: Some(from),
            where_clause: None,
            group_by: &[],
            having: None,
            order_by: &[],
            limit: None,
            offset: None,
            set_op: None,
            for_clause: None,
        });

        let stmt = Statement::Select(select);
        let result = planner.create_logical_plan(&stmt);
        assert!(result.is_ok());

        let plan = result.unwrap();
        if let LogicalOperator::Project(project) = plan.root {
            assert!(matches!(project.input, LogicalOperator::Join(_)));
            if let LogicalOperator::Join(join) = project.input {
                assert_eq!(join.join_type, JoinType::Inner);
            }
        }
    }

    #[test]
    fn plan_insert_values_basic() {
        use crate::sql::ast::{InsertStmt, Literal, TableRef};

        let arena = Bump::new();
        let catalog = create_test_catalog();
        let planner = Planner::new(&catalog, &arena);

        let val1 = arena.alloc(Expr::Literal(Literal::Integer("1")));
        let val2 = arena.alloc(Expr::Literal(Literal::String("Alice")));
        let row: &[&Expr] = arena.alloc_slice_copy(&[val1 as &Expr, val2 as &Expr]);
        let rows: &[&[&Expr]] = arena.alloc_slice_copy(&[row]);

        let insert = arena.alloc(InsertStmt {
            table: TableRef {
                schema: None,
                name: "users",
                alias: None,
            },
            columns: None,
            source: crate::sql::ast::InsertSource::Values(rows),
            on_conflict: None,
            returning: None,
        });

        let stmt = Statement::Insert(insert);
        let result = planner.create_logical_plan(&stmt);
        assert!(result.is_ok());

        let plan = result.unwrap();
        assert!(matches!(plan.root, LogicalOperator::Insert(_)));

        if let LogicalOperator::Insert(insert_op) = plan.root {
            assert_eq!(insert_op.table, "users");
            assert!(insert_op.schema.is_none());
            assert!(insert_op.columns.is_none());
            assert!(matches!(insert_op.source, InsertSource::Values(_)));
        }
    }

    #[test]
    fn plan_insert_values_with_columns() {
        use crate::sql::ast::{InsertStmt, Literal, TableRef};

        let arena = Bump::new();
        let catalog = create_test_catalog();
        let planner = Planner::new(&catalog, &arena);

        let val1 = arena.alloc(Expr::Literal(Literal::Integer("1")));
        let row: &[&Expr] = arena.alloc_slice_copy(&[val1 as &Expr]);
        let rows: &[&[&Expr]] = arena.alloc_slice_copy(&[row]);
        let columns: &[&str] = arena.alloc_slice_copy(&["id"]);

        let insert = arena.alloc(InsertStmt {
            table: TableRef {
                schema: Some("root"),
                name: "users",
                alias: None,
            },
            columns: Some(columns),
            source: crate::sql::ast::InsertSource::Values(rows),
            on_conflict: None,
            returning: None,
        });

        let stmt = Statement::Insert(insert);
        let result = planner.create_logical_plan(&stmt);
        assert!(result.is_ok());

        let plan = result.unwrap();
        if let LogicalOperator::Insert(insert_op) = plan.root {
            assert_eq!(insert_op.table, "users");
            assert_eq!(insert_op.schema, Some("root"));
            assert_eq!(insert_op.columns, Some(&["id"][..]));
        }
    }

    #[test]
    fn plan_insert_default_values() {
        use crate::sql::ast::{InsertStmt, TableRef};

        let arena = Bump::new();
        let catalog = create_test_catalog();
        let planner = Planner::new(&catalog, &arena);

        let insert = arena.alloc(InsertStmt {
            table: TableRef {
                schema: None,
                name: "users",
                alias: None,
            },
            columns: None,
            source: crate::sql::ast::InsertSource::Default,
            on_conflict: None,
            returning: None,
        });

        let stmt = Statement::Insert(insert);
        let result = planner.create_logical_plan(&stmt);
        assert!(result.is_ok());

        let plan = result.unwrap();
        if let LogicalOperator::Insert(insert_op) = plan.root {
            assert!(matches!(insert_op.source, InsertSource::Default));
        }
    }

    #[test]
    fn plan_insert_select() {
        use crate::sql::ast::{Distinct, FromClause, InsertStmt, SelectColumn, SelectStmt, TableRef};

        let arena = Bump::new();
        let catalog = create_test_catalog();
        let planner = Planner::new(&catalog, &arena);

        let table_ref = TableRef {
            schema: None,
            name: "users",
            alias: None,
        };
        let from = arena.alloc(FromClause::Table(table_ref));

        let select = arena.alloc(SelectStmt {
            with: None,
            distinct: Distinct::All,
            columns: &[SelectColumn::AllColumns],
            from: Some(from),
            where_clause: None,
            group_by: &[],
            having: None,
            order_by: &[],
            limit: None,
            offset: None,
            set_op: None,
            for_clause: None,
        });

        let insert = arena.alloc(InsertStmt {
            table: TableRef {
                schema: None,
                name: "orders",
                alias: None,
            },
            columns: None,
            source: crate::sql::ast::InsertSource::Select(select),
            on_conflict: None,
            returning: None,
        });

        let stmt = Statement::Insert(insert);
        let result = planner.create_logical_plan(&stmt);
        assert!(result.is_ok());

        let plan = result.unwrap();
        if let LogicalOperator::Insert(insert_op) = plan.root {
            assert_eq!(insert_op.table, "orders");
            assert!(matches!(insert_op.source, InsertSource::Select(_)));
        }
    }

    #[test]
    fn plan_update_basic() {
        use crate::sql::ast::{Assignment, ColumnRef, Literal, TableRef, UpdateStmt};

        let arena = Bump::new();
        let catalog = create_test_catalog();
        let planner = Planner::new(&catalog, &arena);

        let value = arena.alloc(Expr::Literal(Literal::Integer("42")));
        let assignments: &[Assignment] = arena.alloc_slice_copy(&[Assignment {
            column: ColumnRef {
                schema: None,
                table: None,
                column: "age",
            },
            value,
        }]);

        let update = arena.alloc(UpdateStmt {
            table: TableRef {
                schema: None,
                name: "users",
                alias: None,
            },
            assignments,
            from: None,
            where_clause: None,
            returning: None,
        });

        let stmt = Statement::Update(update);
        let result = planner.create_logical_plan(&stmt);
        assert!(result.is_ok());

        let plan = result.unwrap();
        assert!(matches!(plan.root, LogicalOperator::Update(_)));

        if let LogicalOperator::Update(update_op) = plan.root {
            assert_eq!(update_op.table, "users");
            assert!(update_op.schema.is_none());
            assert_eq!(update_op.assignments.len(), 1);
            assert_eq!(update_op.assignments[0].column, "age");
            assert!(update_op.filter.is_none());
        }
    }

    #[test]
    fn plan_update_with_where() {
        use crate::sql::ast::{Assignment, ColumnRef, Literal, TableRef, UpdateStmt};

        let arena = Bump::new();
        let catalog = create_test_catalog();
        let planner = Planner::new(&catalog, &arena);

        let value = arena.alloc(Expr::Literal(Literal::String("inactive")));
        let assignments: &[Assignment] = arena.alloc_slice_copy(&[Assignment {
            column: ColumnRef {
                schema: None,
                table: None,
                column: "status",
            },
            value,
        }]);

        let where_clause = arena.alloc(Expr::Literal(Literal::Boolean(true)));

        let update = arena.alloc(UpdateStmt {
            table: TableRef {
                schema: Some("root"),
                name: "users",
                alias: None,
            },
            assignments,
            from: None,
            where_clause: Some(where_clause),
            returning: None,
        });

        let stmt = Statement::Update(update);
        let result = planner.create_logical_plan(&stmt);
        assert!(result.is_ok());

        let plan = result.unwrap();
        if let LogicalOperator::Update(update_op) = plan.root {
            assert_eq!(update_op.table, "users");
            assert_eq!(update_op.schema, Some("root"));
            assert!(update_op.filter.is_some());
        }
    }

    #[test]
    fn plan_update_multiple_assignments() {
        use crate::sql::ast::{Assignment, ColumnRef, Literal, TableRef, UpdateStmt};

        let arena = Bump::new();
        let catalog = create_test_catalog();
        let planner = Planner::new(&catalog, &arena);

        let val1 = arena.alloc(Expr::Literal(Literal::String("John")));
        let val2 = arena.alloc(Expr::Literal(Literal::Integer("30")));
        let assignments: &[Assignment] = arena.alloc_slice_copy(&[
            Assignment {
                column: ColumnRef {
                    schema: None,
                    table: None,
                    column: "name",
                },
                value: val1,
            },
            Assignment {
                column: ColumnRef {
                    schema: None,
                    table: None,
                    column: "age",
                },
                value: val2,
            },
        ]);

        let update = arena.alloc(UpdateStmt {
            table: TableRef {
                schema: None,
                name: "users",
                alias: None,
            },
            assignments,
            from: None,
            where_clause: None,
            returning: None,
        });

        let stmt = Statement::Update(update);
        let result = planner.create_logical_plan(&stmt);
        assert!(result.is_ok());

        let plan = result.unwrap();
        if let LogicalOperator::Update(update_op) = plan.root {
            assert_eq!(update_op.assignments.len(), 2);
            assert_eq!(update_op.assignments[0].column, "name");
            assert_eq!(update_op.assignments[1].column, "age");
        }
    }

    #[test]
    fn plan_delete_basic() {
        use crate::sql::ast::{DeleteStmt, TableRef};

        let arena = Bump::new();
        let catalog = create_test_catalog();
        let planner = Planner::new(&catalog, &arena);

        let delete = arena.alloc(DeleteStmt {
            table: TableRef {
                schema: None,
                name: "users",
                alias: None,
            },
            using: None,
            where_clause: None,
            returning: None,
        });

        let stmt = Statement::Delete(delete);
        let result = planner.create_logical_plan(&stmt);
        assert!(result.is_ok());

        let plan = result.unwrap();
        assert!(matches!(plan.root, LogicalOperator::Delete(_)));

        if let LogicalOperator::Delete(delete_op) = plan.root {
            assert_eq!(delete_op.table, "users");
            assert!(delete_op.schema.is_none());
            assert!(delete_op.filter.is_none());
        }
    }

    #[test]
    fn plan_delete_with_where() {
        use crate::sql::ast::{DeleteStmt, Literal, TableRef};

        let arena = Bump::new();
        let catalog = create_test_catalog();
        let planner = Planner::new(&catalog, &arena);

        let where_clause = arena.alloc(Expr::Literal(Literal::Boolean(true)));

        let delete = arena.alloc(DeleteStmt {
            table: TableRef {
                schema: Some("root"),
                name: "users",
                alias: None,
            },
            using: None,
            where_clause: Some(where_clause),
            returning: None,
        });

        let stmt = Statement::Delete(delete);
        let result = planner.create_logical_plan(&stmt);
        assert!(result.is_ok());

        let plan = result.unwrap();
        if let LogicalOperator::Delete(delete_op) = plan.root {
            assert_eq!(delete_op.table, "users");
            assert_eq!(delete_op.schema, Some("root"));
            assert!(delete_op.filter.is_some());
        }
    }

    #[test]
    fn extract_column_refs_from_equality() {
        use crate::sql::ast::{BinaryOperator, ColumnRef, Literal};

        let arena = Bump::new();
        let catalog = Catalog::new();
        let planner = Planner::new(&catalog, &arena);

        let col = arena.alloc(Expr::Column(ColumnRef {
            schema: None,
            table: None,
            column: "id",
        }));
        let val = arena.alloc(Expr::Literal(Literal::Integer("1")));
        let eq_expr = Expr::BinaryOp {
            left: col,
            op: BinaryOperator::Eq,
            right: val,
        };

        let columns = planner.extract_filter_columns(&eq_expr);
        assert_eq!(columns.len(), 1);
        assert!(columns.contains(&"id"));
    }

    #[test]
    fn extract_column_refs_from_and() {
        use crate::sql::ast::{BinaryOperator, ColumnRef, Literal};

        let arena = Bump::new();
        let catalog = Catalog::new();
        let planner = Planner::new(&catalog, &arena);

        let col1 = arena.alloc(Expr::Column(ColumnRef {
            schema: None,
            table: None,
            column: "id",
        }));
        let val1 = arena.alloc(Expr::Literal(Literal::Integer("1")));
        let eq1 = arena.alloc(Expr::BinaryOp {
            left: col1,
            op: BinaryOperator::Eq,
            right: val1,
        });

        let col2 = arena.alloc(Expr::Column(ColumnRef {
            schema: None,
            table: None,
            column: "status",
        }));
        let val2 = arena.alloc(Expr::Literal(Literal::String("active")));
        let eq2 = arena.alloc(Expr::BinaryOp {
            left: col2,
            op: BinaryOperator::Eq,
            right: val2,
        });

        let and_expr = Expr::BinaryOp {
            left: eq1,
            op: BinaryOperator::And,
            right: eq2,
        };

        let columns = planner.extract_filter_columns(&and_expr);
        assert_eq!(columns.len(), 2);
        assert!(columns.contains(&"id"));
        assert!(columns.contains(&"status"));
    }

    #[test]
    fn find_applicable_indexes_single_column() {
        use crate::schema::{IndexDef, IndexType, TableDef};

        let arena = Bump::new();
        let catalog = Catalog::new();
        let planner = Planner::new(&catalog, &arena);

        let table = TableDef::new(1, "users", vec![])
            .with_index(IndexDef::new("idx_id", vec!["id"], true, IndexType::BTree))
            .with_index(IndexDef::new("idx_email", vec!["email"], true, IndexType::BTree));

        let filter_columns: Vec<&str> = vec!["id"];
        let candidates = planner.find_applicable_indexes(&table, &filter_columns);

        assert_eq!(candidates.len(), 1);
        assert_eq!(candidates[0].name(), "idx_id");
    }

    #[test]
    fn find_applicable_indexes_multi_column() {
        use crate::schema::{IndexDef, IndexType, TableDef};

        let arena = Bump::new();
        let catalog = Catalog::new();
        let planner = Planner::new(&catalog, &arena);

        let table = TableDef::new(1, "users", vec![])
            .with_index(IndexDef::new("idx_id", vec!["id"], true, IndexType::BTree))
            .with_index(IndexDef::new(
                "idx_name_email",
                vec!["name", "email"],
                false,
                IndexType::BTree,
            ));

        let filter_columns: Vec<&str> = vec!["name", "email"];
        let candidates = planner.find_applicable_indexes(&table, &filter_columns);

        assert!(candidates.iter().any(|idx| idx.name() == "idx_name_email"));
    }

    #[test]
    fn select_best_index_prefers_unique() {
        use crate::schema::{IndexDef, IndexType, TableDef};

        let arena = Bump::new();
        let catalog = Catalog::new();
        let planner = Planner::new(&catalog, &arena);

        let table = TableDef::new(1, "users", vec![])
            .with_index(IndexDef::new("idx_id", vec!["id"], true, IndexType::BTree))
            .with_index(IndexDef::new("idx_id_nonunique", vec!["id"], false, IndexType::BTree));

        let filter_columns: Vec<&str> = vec!["id"];
        let best = planner.select_best_index(&table, &filter_columns);

        assert!(best.is_some());
        assert_eq!(best.unwrap().name(), "idx_id");
    }

    #[test]
    fn select_best_index_prefers_more_columns() {
        use crate::schema::{IndexDef, IndexType, TableDef};

        let arena = Bump::new();
        let catalog = Catalog::new();
        let planner = Planner::new(&catalog, &arena);

        let table = TableDef::new(1, "users", vec![])
            .with_index(IndexDef::new("idx_id", vec!["id"], false, IndexType::BTree))
            .with_index(IndexDef::new(
                "idx_id_status",
                vec!["id", "status"],
                false,
                IndexType::BTree,
            ));

        let filter_columns: Vec<&str> = vec!["id", "status"];
        let best = planner.select_best_index(&table, &filter_columns);

        assert!(best.is_some());
        assert_eq!(best.unwrap().name(), "idx_id_status");
    }

    #[test]
    fn estimate_scan_cardinality() {
        let arena = Bump::new();
        let catalog = Catalog::new();
        let planner = Planner::new(&catalog, &arena);

        let scan = arena.alloc(LogicalOperator::Scan(LogicalScan {
            schema: None,
            table: "users",
            alias: None,
        }));

        let est = planner.estimate_cardinality(scan);
        assert!(est > 0);
    }

    #[test]
    fn estimate_filter_reduces_cardinality() {
        let arena = Bump::new();
        let catalog = Catalog::new();
        let planner = Planner::new(&catalog, &arena);

        let scan = arena.alloc(LogicalOperator::Scan(LogicalScan {
            schema: None,
            table: "users",
            alias: None,
        }));

        let predicate = arena.alloc(Expr::Literal(crate::sql::ast::Literal::Boolean(true)));
        let filter = arena.alloc(LogicalOperator::Filter(LogicalFilter {
            input: scan,
            predicate,
        }));

        let scan_est = planner.estimate_cardinality(scan);
        let filter_est = planner.estimate_cardinality(filter);
        assert!(filter_est <= scan_est);
    }

    #[test]
    fn estimate_join_multiplies_cardinalities() {
        let arena = Bump::new();
        let catalog = Catalog::new();
        let planner = Planner::new(&catalog, &arena);

        let left = arena.alloc(LogicalOperator::Scan(LogicalScan {
            schema: None,
            table: "orders",
            alias: None,
        }));
        let right = arena.alloc(LogicalOperator::Scan(LogicalScan {
            schema: None,
            table: "customers",
            alias: None,
        }));

        let join = arena.alloc(LogicalOperator::Join(LogicalJoin {
            left,
            right,
            join_type: JoinType::Inner,
            condition: None,
        }));

        let join_est = planner.estimate_cardinality(join);
        assert!(join_est > 0);
    }

    #[test]
    fn extract_join_tables() {
        let arena = Bump::new();
        let catalog = Catalog::new();
        let planner = Planner::new(&catalog, &arena);

        let t1 = arena.alloc(LogicalOperator::Scan(LogicalScan {
            schema: None,
            table: "orders",
            alias: None,
        }));
        let t2 = arena.alloc(LogicalOperator::Scan(LogicalScan {
            schema: None,
            table: "customers",
            alias: None,
        }));
        let t3 = arena.alloc(LogicalOperator::Scan(LogicalScan {
            schema: None,
            table: "products",
            alias: None,
        }));

        let join1 = arena.alloc(LogicalOperator::Join(LogicalJoin {
            left: t1,
            right: t2,
            join_type: JoinType::Inner,
            condition: None,
        }));
        let join2 = arena.alloc(LogicalOperator::Join(LogicalJoin {
            left: join1,
            right: t3,
            join_type: JoinType::Inner,
            condition: None,
        }));

        let tables = planner.extract_join_tables(join2);
        assert_eq!(tables.len(), 3);
    }

    #[test]
    fn reorder_joins_smallest_first() {
        let arena = Bump::new();
        let catalog = Catalog::new();
        let planner = Planner::new(&catalog, &arena);

        let scans: Vec<(&LogicalOperator, u64)> = vec![
            (
                arena.alloc(LogicalOperator::Scan(LogicalScan {
                    schema: None,
                    table: "large_table",
                    alias: None,
                })) as &LogicalOperator,
                1000,
            ),
            (
                arena.alloc(LogicalOperator::Scan(LogicalScan {
                    schema: None,
                    table: "small_table",
                    alias: None,
                })) as &LogicalOperator,
                10,
            ),
            (
                arena.alloc(LogicalOperator::Scan(LogicalScan {
                    schema: None,
                    table: "medium_table",
                    alias: None,
                })) as &LogicalOperator,
                100,
            ),
        ];

        let ordered = planner.order_tables_by_cardinality(scans);
        assert_eq!(ordered.len(), 3);
        if let LogicalOperator::Scan(s) = ordered[0] {
            assert_eq!(s.table, "small_table");
        }
    }

    #[test]
    fn optimize_scan_to_table_scan() {
        let arena = Bump::new();
        let catalog = Catalog::new();
        let planner = Planner::new(&catalog, &arena);

        let scan = arena.alloc(LogicalOperator::Scan(LogicalScan {
            schema: None,
            table: "users",
            alias: None,
        }));
        let logical = LogicalPlan { root: scan };

        let physical = planner.optimize_to_physical(&logical);
        assert!(physical.is_ok());

        let plan = physical.unwrap();
        assert!(matches!(plan.root, PhysicalOperator::TableScan(_)));

        if let PhysicalOperator::TableScan(ts) = plan.root {
            assert_eq!(ts.table, "users");
        }
    }

    #[test]
    fn optimize_filter_to_filter_exec() {
        let arena = Bump::new();
        let catalog = Catalog::new();
        let planner = Planner::new(&catalog, &arena);

        let scan = arena.alloc(LogicalOperator::Scan(LogicalScan {
            schema: None,
            table: "users",
            alias: None,
        }));
        let predicate = arena.alloc(Expr::Literal(crate::sql::ast::Literal::Boolean(true)));
        let filter = arena.alloc(LogicalOperator::Filter(LogicalFilter {
            input: scan,
            predicate,
        }));
        let logical = LogicalPlan { root: filter };

        let physical = planner.optimize_to_physical(&logical);
        assert!(physical.is_ok());

        let plan = physical.unwrap();
        assert!(matches!(plan.root, PhysicalOperator::FilterExec(_)));
    }

    #[test]
    fn optimize_project_to_project_exec() {
        let arena = Bump::new();
        let catalog = Catalog::new();
        let planner = Planner::new(&catalog, &arena);

        let scan = arena.alloc(LogicalOperator::Scan(LogicalScan {
            schema: None,
            table: "users",
            alias: None,
        }));
        let project = arena.alloc(LogicalOperator::Project(LogicalProject {
            input: scan,
            expressions: &[],
            aliases: &[],
        }));
        let logical = LogicalPlan { root: project };

        let physical = planner.optimize_to_physical(&logical);
        assert!(physical.is_ok());

        let plan = physical.unwrap();
        assert!(matches!(plan.root, PhysicalOperator::ProjectExec(_)));
    }

    #[test]
    fn optimize_join_to_nested_loop_join() {
        let arena = Bump::new();
        let catalog = Catalog::new();
        let planner = Planner::new(&catalog, &arena);

        let left = arena.alloc(LogicalOperator::Scan(LogicalScan {
            schema: None,
            table: "orders",
            alias: None,
        }));
        let right = arena.alloc(LogicalOperator::Scan(LogicalScan {
            schema: None,
            table: "customers",
            alias: None,
        }));
        let join = arena.alloc(LogicalOperator::Join(LogicalJoin {
            left,
            right,
            join_type: JoinType::Inner,
            condition: None,
        }));
        let logical = LogicalPlan { root: join };

        let physical = planner.optimize_to_physical(&logical);
        assert!(physical.is_ok());

        let plan = physical.unwrap();
        assert!(matches!(plan.root, PhysicalOperator::NestedLoopJoin(_)));
    }

    #[test]
    fn optimize_sort_to_sort_exec() {
        let arena = Bump::new();
        let catalog = Catalog::new();
        let planner = Planner::new(&catalog, &arena);

        let scan = arena.alloc(LogicalOperator::Scan(LogicalScan {
            schema: None,
            table: "users",
            alias: None,
        }));
        let sort = arena.alloc(LogicalOperator::Sort(LogicalSort {
            input: scan,
            order_by: &[],
        }));
        let logical = LogicalPlan { root: sort };

        let physical = planner.optimize_to_physical(&logical);
        assert!(physical.is_ok());

        let plan = physical.unwrap();
        assert!(matches!(plan.root, PhysicalOperator::SortExec(_)));
    }

    #[test]
    fn optimize_limit_to_limit_exec() {
        let arena = Bump::new();
        let catalog = Catalog::new();
        let planner = Planner::new(&catalog, &arena);

        let scan = arena.alloc(LogicalOperator::Scan(LogicalScan {
            schema: None,
            table: "users",
            alias: None,
        }));
        let limit = arena.alloc(LogicalOperator::Limit(LogicalLimit {
            input: scan,
            limit: Some(10),
            offset: None,
        }));
        let logical = LogicalPlan { root: limit };

        let physical = planner.optimize_to_physical(&logical);
        assert!(physical.is_ok());

        let plan = physical.unwrap();
        assert!(matches!(plan.root, PhysicalOperator::LimitExec(_)));

        if let PhysicalOperator::LimitExec(le) = plan.root {
            assert_eq!(le.limit, Some(10));
        }
    }

    #[test]
    fn optimize_aggregate_to_hash_aggregate() {
        let arena = Bump::new();
        let catalog = Catalog::new();
        let planner = Planner::new(&catalog, &arena);

        let scan = arena.alloc(LogicalOperator::Scan(LogicalScan {
            schema: None,
            table: "users",
            alias: None,
        }));
        let agg = arena.alloc(LogicalOperator::Aggregate(LogicalAggregate {
            input: scan,
            group_by: &[],
            aggregates: &[],
        }));
        let logical = LogicalPlan { root: agg };

        let physical = planner.optimize_to_physical(&logical);
        assert!(physical.is_ok());

        let plan = physical.unwrap();
        assert!(matches!(plan.root, PhysicalOperator::HashAggregate(_)));
    }

    #[test]
    fn extract_aggregates_count_star() {
        use crate::sql::ast::{FunctionArgs, FunctionCall, FunctionName, SelectColumn};

        let arena = Bump::new();
        let catalog = Catalog::new();
        let planner = Planner::new(&catalog, &arena);

        let count_star = arena.alloc(Expr::Function(FunctionCall {
            name: FunctionName {
                schema: None,
                name: "count",
            },
            args: FunctionArgs::Star,
            distinct: false,
            filter: None,
            over: None,
        }));

        let columns: &[SelectColumn] = arena.alloc_slice_copy(&[SelectColumn::Expr {
            expr: count_star,
            alias: None,
        }]);

        let aggregates = planner.extract_aggregates(columns);
        assert_eq!(aggregates.len(), 1);
    }

    #[test]
    fn extract_aggregates_sum() {
        use crate::sql::ast::{
            ColumnRef, FunctionArg, FunctionArgs, FunctionCall, FunctionName, SelectColumn,
        };

        let arena = Bump::new();
        let catalog = Catalog::new();
        let planner = Planner::new(&catalog, &arena);

        let col_expr = arena.alloc(Expr::Column(ColumnRef {
            schema: None,
            table: None,
            column: "amount",
        }));

        let args: &[FunctionArg] = arena.alloc_slice_copy(&[FunctionArg {
            name: None,
            value: col_expr,
        }]);

        let sum_expr = arena.alloc(Expr::Function(FunctionCall {
            name: FunctionName {
                schema: None,
                name: "sum",
            },
            args: FunctionArgs::Args(args),
            distinct: false,
            filter: None,
            over: None,
        }));

        let columns: &[SelectColumn] = arena.alloc_slice_copy(&[SelectColumn::Expr {
            expr: sum_expr,
            alias: None,
        }]);

        let aggregates = planner.extract_aggregates(columns);
        assert_eq!(aggregates.len(), 1);
    }

    #[test]
    fn extract_aggregates_multiple() {
        use crate::sql::ast::{
            ColumnRef, FunctionArg, FunctionArgs, FunctionCall, FunctionName, SelectColumn,
        };

        let arena = Bump::new();
        let catalog = Catalog::new();
        let planner = Planner::new(&catalog, &arena);

        let count_star = arena.alloc(Expr::Function(FunctionCall {
            name: FunctionName {
                schema: None,
                name: "count",
            },
            args: FunctionArgs::Star,
            distinct: false,
            filter: None,
            over: None,
        }));

        let col_expr = arena.alloc(Expr::Column(ColumnRef {
            schema: None,
            table: None,
            column: "price",
        }));

        let args: &[FunctionArg] = arena.alloc_slice_copy(&[FunctionArg {
            name: None,
            value: col_expr,
        }]);

        let avg_expr = arena.alloc(Expr::Function(FunctionCall {
            name: FunctionName {
                schema: None,
                name: "avg",
            },
            args: FunctionArgs::Args(args),
            distinct: false,
            filter: None,
            over: None,
        }));

        let columns: &[SelectColumn] = arena.alloc_slice_copy(&[
            SelectColumn::Expr {
                expr: count_star,
                alias: None,
            },
            SelectColumn::Expr {
                expr: avg_expr,
                alias: None,
            },
        ]);

        let aggregates = planner.extract_aggregates(columns);
        assert_eq!(aggregates.len(), 2);
    }

    #[test]
    fn extract_aggregates_non_aggregate_ignored() {
        use crate::sql::ast::{ColumnRef, Literal, SelectColumn};

        let arena = Bump::new();
        let catalog = Catalog::new();
        let planner = Planner::new(&catalog, &arena);

        let col_expr = arena.alloc(Expr::Column(ColumnRef {
            schema: None,
            table: None,
            column: "name",
        }));

        let lit_expr = arena.alloc(Expr::Literal(Literal::Integer("42")));

        let columns: &[SelectColumn] = arena.alloc_slice_copy(&[
            SelectColumn::Expr {
                expr: col_expr,
                alias: None,
            },
            SelectColumn::Expr {
                expr: lit_expr,
                alias: None,
            },
        ]);

        let aggregates = planner.extract_aggregates(columns);
        assert_eq!(aggregates.len(), 0);
    }

    #[test]
    fn convert_aggregates_count_star() {
        use crate::sql::ast::{FunctionArgs, FunctionCall, FunctionName};

        let arena = Bump::new();
        let catalog = Catalog::new();
        let planner = Planner::new(&catalog, &arena);

        let count_star = arena.alloc(Expr::Function(FunctionCall {
            name: FunctionName {
                schema: None,
                name: "count",
            },
            args: FunctionArgs::Star,
            distinct: false,
            filter: None,
            over: None,
        }));

        let exprs: &[&Expr] = arena.alloc_slice_copy(&[count_star as &Expr]);
        let aggregates = planner.convert_aggregates_to_physical(exprs);

        assert_eq!(aggregates.len(), 1);
        assert_eq!(aggregates[0].function, AggregateFunction::Count);
        assert!(aggregates[0].argument.is_none());
        assert!(!aggregates[0].distinct);
    }

    #[test]
    fn convert_aggregates_sum_with_column() {
        use crate::sql::ast::{
            ColumnRef, FunctionArg, FunctionArgs, FunctionCall, FunctionName,
        };

        let arena = Bump::new();
        let catalog = Catalog::new();
        let planner = Planner::new(&catalog, &arena);

        let col_expr = arena.alloc(Expr::Column(ColumnRef {
            schema: None,
            table: None,
            column: "amount",
        }));

        let args: &[FunctionArg] = arena.alloc_slice_copy(&[FunctionArg {
            name: None,
            value: col_expr,
        }]);

        let sum_expr = arena.alloc(Expr::Function(FunctionCall {
            name: FunctionName {
                schema: None,
                name: "sum",
            },
            args: FunctionArgs::Args(args),
            distinct: false,
            filter: None,
            over: None,
        }));

        let exprs: &[&Expr] = arena.alloc_slice_copy(&[sum_expr as &Expr]);
        let aggregates = planner.convert_aggregates_to_physical(exprs);

        assert_eq!(aggregates.len(), 1);
        assert_eq!(aggregates[0].function, AggregateFunction::Sum);
        assert!(aggregates[0].argument.is_some());
    }

    #[test]
    fn convert_aggregates_distinct_count() {
        use crate::sql::ast::{
            ColumnRef, FunctionArg, FunctionArgs, FunctionCall, FunctionName,
        };

        let arena = Bump::new();
        let catalog = Catalog::new();
        let planner = Planner::new(&catalog, &arena);

        let col_expr = arena.alloc(Expr::Column(ColumnRef {
            schema: None,
            table: None,
            column: "user_id",
        }));

        let args: &[FunctionArg] = arena.alloc_slice_copy(&[FunctionArg {
            name: None,
            value: col_expr,
        }]);

        let count_distinct = arena.alloc(Expr::Function(FunctionCall {
            name: FunctionName {
                schema: None,
                name: "COUNT",
            },
            args: FunctionArgs::Args(args),
            distinct: true,
            filter: None,
            over: None,
        }));

        let exprs: &[&Expr] = arena.alloc_slice_copy(&[count_distinct as &Expr]);
        let aggregates = planner.convert_aggregates_to_physical(exprs);

        assert_eq!(aggregates.len(), 1);
        assert_eq!(aggregates[0].function, AggregateFunction::Count);
        assert!(aggregates[0].distinct);
    }

    #[test]
    fn convert_aggregates_all_functions() {
        use crate::sql::ast::{
            ColumnRef, FunctionArg, FunctionArgs, FunctionCall, FunctionName,
        };

        let arena = Bump::new();
        let catalog = Catalog::new();
        let planner = Planner::new(&catalog, &arena);

        let make_agg = |name: &'static str| {
            let col = arena.alloc(Expr::Column(ColumnRef {
                schema: None,
                table: None,
                column: "val",
            }));
            let args: &[FunctionArg] = arena.alloc_slice_copy(&[FunctionArg {
                name: None,
                value: col,
            }]);
            arena.alloc(Expr::Function(FunctionCall {
                name: FunctionName { schema: None, name },
                args: FunctionArgs::Args(args),
                distinct: false,
                filter: None,
                over: None,
            }))
        };

        let count = make_agg("count");
        let sum = make_agg("sum");
        let avg = make_agg("avg");
        let min = make_agg("min");
        let max = make_agg("max");

        let exprs: &[&Expr] = arena.alloc_slice_copy(&[
            count as &Expr,
            sum as &Expr,
            avg as &Expr,
            min as &Expr,
            max as &Expr,
        ]);
        let aggregates = planner.convert_aggregates_to_physical(exprs);

        assert_eq!(aggregates.len(), 5);
        assert_eq!(aggregates[0].function, AggregateFunction::Count);
        assert_eq!(aggregates[1].function, AggregateFunction::Sum);
        assert_eq!(aggregates[2].function, AggregateFunction::Avg);
        assert_eq!(aggregates[3].function, AggregateFunction::Min);
        assert_eq!(aggregates[4].function, AggregateFunction::Max);
    }

    #[test]
    fn extract_scan_bounds_equality() {
        use crate::sql::ast::{BinaryOperator, ColumnRef, Literal};

        let arena = Bump::new();
        let catalog = Catalog::new();
        let planner = Planner::new(&catalog, &arena);

        let col = arena.alloc(Expr::Column(ColumnRef {
            schema: None,
            table: None,
            column: "id",
        }));
        let val = arena.alloc(Expr::Literal(Literal::Integer("42")));
        let predicate = arena.alloc(Expr::BinaryOp {
            left: col,
            op: BinaryOperator::Eq,
            right: val,
        });

        let bounds = planner.extract_scan_bounds_for_column(predicate, "id");

        assert!(bounds.point_value.is_some());
        assert!(bounds.lower.is_none());
        assert!(bounds.upper.is_none());
        assert_eq!(planner.bounds_to_scan_type(&bounds), ScanBoundType::Point);
    }

    #[test]
    fn extract_scan_bounds_greater_than() {
        use crate::sql::ast::{BinaryOperator, ColumnRef, Literal};

        let arena = Bump::new();
        let catalog = Catalog::new();
        let planner = Planner::new(&catalog, &arena);

        let col = arena.alloc(Expr::Column(ColumnRef {
            schema: None,
            table: None,
            column: "age",
        }));
        let val = arena.alloc(Expr::Literal(Literal::Integer("18")));
        let predicate = arena.alloc(Expr::BinaryOp {
            left: col,
            op: BinaryOperator::Gt,
            right: val,
        });

        let bounds = planner.extract_scan_bounds_for_column(predicate, "age");

        assert!(bounds.lower.is_some());
        assert!(!bounds.lower.unwrap().inclusive);
        assert!(bounds.upper.is_none());
        assert!(bounds.point_value.is_none());
        assert_eq!(planner.bounds_to_scan_type(&bounds), ScanBoundType::Range);
    }

    #[test]
    fn extract_scan_bounds_less_than_or_equal() {
        use crate::sql::ast::{BinaryOperator, ColumnRef, Literal};

        let arena = Bump::new();
        let catalog = Catalog::new();
        let planner = Planner::new(&catalog, &arena);

        let col = arena.alloc(Expr::Column(ColumnRef {
            schema: None,
            table: None,
            column: "price",
        }));
        let val = arena.alloc(Expr::Literal(Literal::Float("100.00")));
        let predicate = arena.alloc(Expr::BinaryOp {
            left: col,
            op: BinaryOperator::LtEq,
            right: val,
        });

        let bounds = planner.extract_scan_bounds_for_column(predicate, "price");

        assert!(bounds.upper.is_some());
        assert!(bounds.upper.unwrap().inclusive);
        assert!(bounds.lower.is_none());
        assert_eq!(planner.bounds_to_scan_type(&bounds), ScanBoundType::Range);
    }

    #[test]
    fn extract_scan_bounds_between() {
        use crate::sql::ast::{ColumnRef, Literal};

        let arena = Bump::new();
        let catalog = Catalog::new();
        let planner = Planner::new(&catalog, &arena);

        let col = arena.alloc(Expr::Column(ColumnRef {
            schema: None,
            table: None,
            column: "score",
        }));
        let low = arena.alloc(Expr::Literal(Literal::Integer("60")));
        let high = arena.alloc(Expr::Literal(Literal::Integer("100")));
        let predicate = arena.alloc(Expr::Between {
            expr: col,
            negated: false,
            low,
            high,
        });

        let bounds = planner.extract_scan_bounds_for_column(predicate, "score");

        assert!(bounds.lower.is_some());
        assert!(bounds.upper.is_some());
        assert!(bounds.lower.unwrap().inclusive);
        assert!(bounds.upper.unwrap().inclusive);
        assert_eq!(planner.bounds_to_scan_type(&bounds), ScanBoundType::Range);
    }

    #[test]
    fn extract_scan_bounds_combined_and() {
        use crate::sql::ast::{BinaryOperator, ColumnRef, Literal};

        let arena = Bump::new();
        let catalog = Catalog::new();
        let planner = Planner::new(&catalog, &arena);

        let col1 = arena.alloc(Expr::Column(ColumnRef {
            schema: None,
            table: None,
            column: "age",
        }));
        let val1 = arena.alloc(Expr::Literal(Literal::Integer("18")));
        let cond1 = arena.alloc(Expr::BinaryOp {
            left: col1,
            op: BinaryOperator::GtEq,
            right: val1,
        });

        let col2 = arena.alloc(Expr::Column(ColumnRef {
            schema: None,
            table: None,
            column: "age",
        }));
        let val2 = arena.alloc(Expr::Literal(Literal::Integer("65")));
        let cond2 = arena.alloc(Expr::BinaryOp {
            left: col2,
            op: BinaryOperator::Lt,
            right: val2,
        });

        let predicate = arena.alloc(Expr::BinaryOp {
            left: cond1,
            op: BinaryOperator::And,
            right: cond2,
        });

        let bounds = planner.extract_scan_bounds_for_column(predicate, "age");

        assert!(bounds.lower.is_some());
        assert!(bounds.upper.is_some());
        assert!(bounds.lower.unwrap().inclusive);
        assert!(!bounds.upper.unwrap().inclusive);
        assert_eq!(planner.bounds_to_scan_type(&bounds), ScanBoundType::Range);
    }

    #[test]
    fn extract_scan_bounds_no_applicable_predicate() {
        use crate::sql::ast::{BinaryOperator, ColumnRef, Literal};

        let arena = Bump::new();
        let catalog = Catalog::new();
        let planner = Planner::new(&catalog, &arena);

        let col = arena.alloc(Expr::Column(ColumnRef {
            schema: None,
            table: None,
            column: "name",
        }));
        let val = arena.alloc(Expr::Literal(Literal::String("John")));
        let predicate = arena.alloc(Expr::BinaryOp {
            left: col,
            op: BinaryOperator::Eq,
            right: val,
        });

        let bounds = planner.extract_scan_bounds_for_column(predicate, "id");

        assert!(bounds.lower.is_none());
        assert!(bounds.upper.is_none());
        assert!(bounds.point_value.is_none());
        assert_eq!(planner.bounds_to_scan_type(&bounds), ScanBoundType::Full);
    }

    #[test]
    fn extract_equi_join_keys_simple() {
        use crate::sql::ast::{BinaryOperator, ColumnRef};

        let arena = Bump::new();
        let catalog = Catalog::new();
        let planner = Planner::new(&catalog, &arena);

        let left_col = arena.alloc(Expr::Column(ColumnRef {
            schema: None,
            table: Some("users"),
            column: "id",
        }));
        let right_col = arena.alloc(Expr::Column(ColumnRef {
            schema: None,
            table: Some("orders"),
            column: "user_id",
        }));
        let condition = arena.alloc(Expr::BinaryOp {
            left: left_col,
            op: BinaryOperator::Eq,
            right: right_col,
        });

        let keys = planner.extract_equi_join_keys(Some(condition));

        assert_eq!(keys.len(), 1);
        assert_eq!(keys[0].left_column, "id");
        assert_eq!(keys[0].right_column, "user_id");
        assert_eq!(keys[0].left_table, Some("users"));
        assert_eq!(keys[0].right_table, Some("orders"));
    }

    #[test]
    fn extract_equi_join_keys_multiple() {
        use crate::sql::ast::{BinaryOperator, ColumnRef};

        let arena = Bump::new();
        let catalog = Catalog::new();
        let planner = Planner::new(&catalog, &arena);

        let left_col1 = arena.alloc(Expr::Column(ColumnRef {
            schema: None,
            table: Some("a"),
            column: "id",
        }));
        let right_col1 = arena.alloc(Expr::Column(ColumnRef {
            schema: None,
            table: Some("b"),
            column: "a_id",
        }));
        let cond1 = arena.alloc(Expr::BinaryOp {
            left: left_col1,
            op: BinaryOperator::Eq,
            right: right_col1,
        });

        let left_col2 = arena.alloc(Expr::Column(ColumnRef {
            schema: None,
            table: Some("a"),
            column: "type",
        }));
        let right_col2 = arena.alloc(Expr::Column(ColumnRef {
            schema: None,
            table: Some("b"),
            column: "type",
        }));
        let cond2 = arena.alloc(Expr::BinaryOp {
            left: left_col2,
            op: BinaryOperator::Eq,
            right: right_col2,
        });

        let condition = arena.alloc(Expr::BinaryOp {
            left: cond1,
            op: BinaryOperator::And,
            right: cond2,
        });

        let keys = planner.extract_equi_join_keys(Some(condition));

        assert_eq!(keys.len(), 2);
        assert_eq!(keys[0].left_column, "id");
        assert_eq!(keys[0].right_column, "a_id");
        assert_eq!(keys[1].left_column, "type");
        assert_eq!(keys[1].right_column, "type");
    }

    #[test]
    fn extract_equi_join_keys_no_equi() {
        use crate::sql::ast::{BinaryOperator, ColumnRef, Literal};

        let arena = Bump::new();
        let catalog = Catalog::new();
        let planner = Planner::new(&catalog, &arena);

        let col = arena.alloc(Expr::Column(ColumnRef {
            schema: None,
            table: Some("users"),
            column: "age",
        }));
        let val = arena.alloc(Expr::Literal(Literal::Integer("18")));
        let condition = arena.alloc(Expr::BinaryOp {
            left: col,
            op: BinaryOperator::Gt,
            right: val,
        });

        let keys = planner.extract_equi_join_keys(Some(condition));
        assert_eq!(keys.len(), 0);
    }

    #[test]
    fn extract_equi_join_keys_none_condition() {
        let arena = Bump::new();
        let catalog = Catalog::new();
        let planner = Planner::new(&catalog, &arena);

        let keys = planner.extract_equi_join_keys(None);
        assert_eq!(keys.len(), 0);
    }

    #[test]
    fn has_equi_join_keys_returns_true() {
        use crate::sql::ast::{BinaryOperator, ColumnRef};

        let arena = Bump::new();
        let catalog = Catalog::new();
        let planner = Planner::new(&catalog, &arena);

        let left_col = arena.alloc(Expr::Column(ColumnRef {
            schema: None,
            table: Some("a"),
            column: "id",
        }));
        let right_col = arena.alloc(Expr::Column(ColumnRef {
            schema: None,
            table: Some("b"),
            column: "a_id",
        }));
        let condition = arena.alloc(Expr::BinaryOp {
            left: left_col,
            op: BinaryOperator::Eq,
            right: right_col,
        });

        assert!(planner.has_equi_join_keys(Some(condition)));
    }

    #[test]
    fn extract_non_equi_conditions_mixed() {
        use crate::sql::ast::{BinaryOperator, ColumnRef, Literal};

        let arena = Bump::new();
        let catalog = Catalog::new();
        let planner = Planner::new(&catalog, &arena);

        let left_col = arena.alloc(Expr::Column(ColumnRef {
            schema: None,
            table: Some("a"),
            column: "id",
        }));
        let right_col = arena.alloc(Expr::Column(ColumnRef {
            schema: None,
            table: Some("b"),
            column: "a_id",
        }));
        let equi_cond = arena.alloc(Expr::BinaryOp {
            left: left_col,
            op: BinaryOperator::Eq,
            right: right_col,
        });

        let filter_col = arena.alloc(Expr::Column(ColumnRef {
            schema: None,
            table: Some("a"),
            column: "age",
        }));
        let filter_val = arena.alloc(Expr::Literal(Literal::Integer("18")));
        let filter_cond = arena.alloc(Expr::BinaryOp {
            left: filter_col,
            op: BinaryOperator::Gt,
            right: filter_val,
        });

        let combined = arena.alloc(Expr::BinaryOp {
            left: equi_cond,
            op: BinaryOperator::And,
            right: filter_cond,
        });

        let non_equi = planner.extract_non_equi_conditions(Some(combined));
        assert!(non_equi.is_some());

        if let Some(Expr::BinaryOp { op, .. }) = non_equi {
            assert_eq!(*op, BinaryOperator::Gt);
        } else {
            panic!("Expected BinaryOp");
        }
    }

    #[test]
    fn optimize_filter_scan_with_index() {
        use crate::records::types::DataType;
        use crate::schema::table::{ColumnDef, IndexDef, IndexType, TableDef};
        use crate::sql::ast::{BinaryOperator, ColumnRef, Literal};

        let arena = Bump::new();
        let mut catalog = Catalog::new();

        let columns = vec![
            ColumnDef::new("id", DataType::Int8),
            ColumnDef::new("name", DataType::Text),
        ];
        let mut table = TableDef::new(1, "users", columns);
        let index = IndexDef::new("idx_id", vec!["id"], true, IndexType::BTree);
        table = table.with_index(index);

        let schema = catalog.get_schema_mut("root").unwrap();
        schema.add_table(table);

        let planner = Planner::new(&catalog, &arena);

        let scan = arena.alloc(LogicalOperator::Scan(LogicalScan {
            schema: None,
            table: "users",
            alias: None,
        }));

        let id_col = arena.alloc(Expr::Column(ColumnRef {
            schema: None,
            table: None,
            column: "id",
        }));
        let val = arena.alloc(Expr::Literal(Literal::Integer("42")));
        let predicate = arena.alloc(Expr::BinaryOp {
            left: id_col,
            op: BinaryOperator::Eq,
            right: val,
        });

        let filter = LogicalFilter {
            input: scan,
            predicate,
        };

        let result = planner.try_optimize_filter_to_index_scan(&filter);
        assert!(result.is_some());

        if let Some(PhysicalOperator::IndexScan(index_scan)) = result {
            assert_eq!(index_scan.index_name, "idx_id");
            assert_eq!(index_scan.table, "users");
        } else {
            panic!("Expected IndexScan");
        }
    }

    #[test]
    fn optimize_filter_scan_no_index_fallback() {
        use crate::sql::ast::{BinaryOperator, ColumnRef, Literal};

        let arena = Bump::new();
        let catalog = Catalog::new();
        let planner = Planner::new(&catalog, &arena);

        let scan = arena.alloc(LogicalOperator::Scan(LogicalScan {
            schema: None,
            table: "nonexistent",
            alias: None,
        }));

        let id_col = arena.alloc(Expr::Column(ColumnRef {
            schema: None,
            table: None,
            column: "id",
        }));
        let val = arena.alloc(Expr::Literal(Literal::Integer("42")));
        let predicate = arena.alloc(Expr::BinaryOp {
            left: id_col,
            op: BinaryOperator::Eq,
            right: val,
        });

        let filter = LogicalFilter {
            input: scan,
            predicate,
        };

        let result = planner.try_optimize_filter_to_index_scan(&filter);
        assert!(result.is_none());
    }

    #[test]
    fn compute_residual_filter_removes_indexed_predicate() {
        use crate::sql::ast::{BinaryOperator, ColumnRef, Literal};

        let arena = Bump::new();
        let catalog = Catalog::new();
        let planner = Planner::new(&catalog, &arena);

        let id_col = arena.alloc(Expr::Column(ColumnRef {
            schema: None,
            table: None,
            column: "id",
        }));
        let val = arena.alloc(Expr::Literal(Literal::Integer("42")));
        let predicate = arena.alloc(Expr::BinaryOp {
            left: id_col,
            op: BinaryOperator::Eq,
            right: val,
        });

        let index_columns = vec!["id".to_string()];
        let residual = planner.compute_residual_filter(predicate, &index_columns);
        assert!(residual.is_none());
    }

    #[test]
    fn compute_residual_filter_keeps_non_indexed_predicate() {
        use crate::sql::ast::{BinaryOperator, ColumnRef, Literal};

        let arena = Bump::new();
        let catalog = Catalog::new();
        let planner = Planner::new(&catalog, &arena);

        let name_col = arena.alloc(Expr::Column(ColumnRef {
            schema: None,
            table: None,
            column: "name",
        }));
        let val = arena.alloc(Expr::Literal(Literal::String("John")));
        let predicate = arena.alloc(Expr::BinaryOp {
            left: name_col,
            op: BinaryOperator::Eq,
            right: val,
        });

        let index_columns = vec!["id".to_string()];
        let residual = planner.compute_residual_filter(predicate, &index_columns);
        assert!(residual.is_some());
    }

    #[test]
    fn validate_table_exists_in_from_clause() {
        use crate::records::types::DataType;
        use crate::schema::ColumnDef;
        use crate::sql::ast::{Distinct, FromClause, SelectColumn, SelectStmt, TableRef};

        let arena = Bump::new();
        let mut catalog = Catalog::new();

        let columns = vec![
            ColumnDef::new("id", DataType::Int8),
            ColumnDef::new("name", DataType::Text),
        ];
        catalog.create_table("root", "users", columns).unwrap();

        let planner = Planner::new(&catalog, &arena);

        let select = arena.alloc(SelectStmt {
            with: None,
            distinct: Distinct::All,
            columns: arena.alloc_slice_copy(&[SelectColumn::AllColumns]),
            from: Some(arena.alloc(FromClause::Table(TableRef {
                schema: None,
                name: "users",
                alias: None,
            }))),
            where_clause: None,
            group_by: &[],
            having: None,
            order_by: &[],
            limit: None,
            offset: None,
            set_op: None,
            for_clause: None,
        });

        let result = planner.plan_select(select);
        assert!(result.is_ok());
    }

    #[test]
    fn validate_table_not_found_error() {
        use crate::sql::ast::{Distinct, FromClause, SelectColumn, SelectStmt, TableRef};

        let arena = Bump::new();
        let catalog = Catalog::new();
        let planner = Planner::new(&catalog, &arena);

        let select = arena.alloc(SelectStmt {
            with: None,
            distinct: Distinct::All,
            columns: arena.alloc_slice_copy(&[SelectColumn::AllColumns]),
            from: Some(arena.alloc(FromClause::Table(TableRef {
                schema: None,
                name: "nonexistent_table",
                alias: None,
            }))),
            where_clause: None,
            group_by: &[],
            having: None,
            order_by: &[],
            limit: None,
            offset: None,
            set_op: None,
            for_clause: None,
        });

        let result = planner.plan_select(select);
        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(err_msg.contains("not found") || err_msg.contains("does not exist"));
    }

    #[test]
    fn validate_table_in_qualified_name() {
        use crate::records::types::DataType;
        use crate::schema::ColumnDef;
        use crate::sql::ast::{Distinct, FromClause, SelectColumn, SelectStmt, TableRef};

        let arena = Bump::new();
        let mut catalog = Catalog::new();

        catalog.create_schema("analytics").unwrap();
        let schema = catalog.get_schema_mut("analytics").unwrap();
        schema.add_table(crate::schema::TableDef::new(
            1,
            "events",
            vec![ColumnDef::new("id", DataType::Int8)],
        ));

        let planner = Planner::new(&catalog, &arena);

        let select = arena.alloc(SelectStmt {
            with: None,
            distinct: Distinct::All,
            columns: arena.alloc_slice_copy(&[SelectColumn::AllColumns]),
            from: Some(arena.alloc(FromClause::Table(TableRef {
                schema: Some("analytics"),
                name: "events",
                alias: None,
            }))),
            where_clause: None,
            group_by: &[],
            having: None,
            order_by: &[],
            limit: None,
            offset: None,
            set_op: None,
            for_clause: None,
        });

        let result = planner.plan_select(select);
        assert!(result.is_ok());
    }

    #[test]
    fn validate_schema_not_found_error() {
        use crate::sql::ast::{Distinct, FromClause, SelectColumn, SelectStmt, TableRef};

        let arena = Bump::new();
        let catalog = Catalog::new();
        let planner = Planner::new(&catalog, &arena);

        let select = arena.alloc(SelectStmt {
            with: None,
            distinct: Distinct::All,
            columns: arena.alloc_slice_copy(&[SelectColumn::AllColumns]),
            from: Some(arena.alloc(FromClause::Table(TableRef {
                schema: Some("nonexistent_schema"),
                name: "table",
                alias: None,
            }))),
            where_clause: None,
            group_by: &[],
            having: None,
            order_by: &[],
            limit: None,
            offset: None,
            set_op: None,
            for_clause: None,
        });

        let result = planner.plan_select(select);
        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(err_msg.contains("not found") || err_msg.contains("schema"));
    }
}
