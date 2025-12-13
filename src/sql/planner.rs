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
        _columns: &'a [crate::sql::ast::SelectColumn<'a>],
    ) -> &'a [&'a Expr<'a>] {
        &[]
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

    fn plan_insert(&self, insert: &crate::sql::ast::InsertStmt<'a>) -> Result<LogicalPlan<'a>> {
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
        let delete_op = self.arena.alloc(LogicalOperator::Delete(LogicalDelete {
            schema: delete.table.schema,
            table: delete.table.name,
            filter: delete.where_clause,
        }));

        Ok(LogicalPlan { root: delete_op })
    }

    fn optimize_to_physical(&self, _logical: &LogicalPlan<'a>) -> Result<PhysicalPlan<'a>> {
        bail!("optimize_to_physical not yet implemented")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bumpalo::Bump;

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
        let catalog = Catalog::new();
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
        let catalog = Catalog::new();
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
        let catalog = Catalog::new();
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
        let catalog = Catalog::new();
        let planner = Planner::new(&catalog, &arena);

        let left_table = TableRef {
            schema: None,
            name: "orders",
            alias: None,
        };
        let left_from = arena.alloc(FromClause::Table(left_table));

        let right_table = TableRef {
            schema: None,
            name: "customers",
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
        let catalog = Catalog::new();
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
        let catalog = Catalog::new();
        let planner = Planner::new(&catalog, &arena);

        let val1 = arena.alloc(Expr::Literal(Literal::Integer("1")));
        let row: &[&Expr] = arena.alloc_slice_copy(&[val1 as &Expr]);
        let rows: &[&[&Expr]] = arena.alloc_slice_copy(&[row]);
        let columns: &[&str] = arena.alloc_slice_copy(&["id"]);

        let insert = arena.alloc(InsertStmt {
            table: TableRef {
                schema: Some("public"),
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
            assert_eq!(insert_op.schema, Some("public"));
            assert_eq!(insert_op.columns, Some(&["id"][..]));
        }
    }

    #[test]
    fn plan_insert_default_values() {
        use crate::sql::ast::{InsertStmt, TableRef};

        let arena = Bump::new();
        let catalog = Catalog::new();
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
        let catalog = Catalog::new();
        let planner = Planner::new(&catalog, &arena);

        let table_ref = TableRef {
            schema: None,
            name: "source_table",
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
                name: "target_table",
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
            assert_eq!(insert_op.table, "target_table");
            assert!(matches!(insert_op.source, InsertSource::Select(_)));
        }
    }

    #[test]
    fn plan_update_basic() {
        use crate::sql::ast::{Assignment, ColumnRef, Literal, TableRef, UpdateStmt};

        let arena = Bump::new();
        let catalog = Catalog::new();
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
        let catalog = Catalog::new();
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
                schema: Some("public"),
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
            assert_eq!(update_op.schema, Some("public"));
            assert!(update_op.filter.is_some());
        }
    }

    #[test]
    fn plan_update_multiple_assignments() {
        use crate::sql::ast::{Assignment, ColumnRef, Literal, TableRef, UpdateStmt};

        let arena = Bump::new();
        let catalog = Catalog::new();
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
        let catalog = Catalog::new();
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
        let catalog = Catalog::new();
        let planner = Planner::new(&catalog, &arena);

        let where_clause = arena.alloc(Expr::Literal(Literal::Boolean(true)));

        let delete = arena.alloc(DeleteStmt {
            table: TableRef {
                schema: Some("public"),
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
            assert_eq!(delete_op.schema, Some("public"));
            assert!(delete_op.filter.is_some());
        }
    }
}
