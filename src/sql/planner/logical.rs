//! # Logical Operators
//!
//! This module defines the logical operators that form the basis of query plans.
//! Logical operators represent relational algebra operations without specifying
//! physical implementation details.
//!
//! ## Operator Categories
//!
//! - **Scan**: Base table access (LogicalScan, DualScan)
//! - **Projection**: Column selection and expression evaluation (LogicalProject)
//! - **Selection**: Row filtering (LogicalFilter)
//! - **Aggregation**: Grouping and aggregate functions (LogicalAggregate)
//! - **Joining**: Combining tables (LogicalJoin)
//! - **Ordering**: Sorting results (LogicalSort)
//! - **Limiting**: Restricting output (LogicalLimit)
//! - **DML**: Data modification (LogicalInsert, LogicalUpdate, LogicalDelete)
//! - **Set Operations**: UNION, INTERSECT, EXCEPT (LogicalSetOp)
//! - **Window Functions**: Analytic functions (LogicalWindow)
//!
//! ## Design
//!
//! All operators use arena allocation for memory efficiency. The lifetime
//! parameter 'a represents the planning arena's lifetime.

use crate::sql::ast::{Expr, JoinType};
use super::schema::OutputSchema;

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
