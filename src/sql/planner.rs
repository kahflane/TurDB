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

use crate::sql::ast::{Expr, JoinType};

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
}
