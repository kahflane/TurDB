//! # Physical Operators
//!
//! This module defines the physical operators that implement query execution.
//! Physical operators specify concrete algorithms for executing relational
//! operations, including access paths, join algorithms, and aggregation strategies.
//!
//! ## Operator Categories
//!
//! - **Scans**: TableScan, IndexScan, SecondaryIndexScan
//! - **Filtering**: FilterExec
//! - **Projection**: ProjectExec
//! - **Joins**: NestedLoopJoin, GraceHashJoin, StreamingHashJoin, HashSemiJoin, HashAntiJoin
//! - **Aggregation**: HashAggregate, SortedAggregate
//! - **Ordering**: SortExec, TopKExec
//! - **Limiting**: LimitExec
//! - **Set Operations**: SetOpExec
//! - **Window Functions**: WindowExec
//! - **Subqueries**: SubqueryExec, ScalarSubqueryExec, ExistsSubqueryExec, InListSubqueryExec
//!
//! ## Memory Model
//!
//! Physical operators are designed to work within TurDB's 256KB memory budget.
//! Operators like GraceHashJoin use partitioning to handle large datasets.

use crate::schema::TableDef;
use crate::sql::ast::{Expr, JoinType};
use super::logical::{SetOpKind, SortKey, WindowFunctionDef};
use super::schema::OutputSchema;
use super::types::ScanRange;

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
    StreamingHashJoin(PhysicalStreamingHashJoin<'a>),
    HashSemiJoin(PhysicalHashSemiJoin<'a>),
    HashAntiJoin(PhysicalHashAntiJoin<'a>),
    HashAggregate(PhysicalHashAggregate<'a>),
    SortedAggregate(PhysicalSortedAggregate<'a>),
    SortExec(PhysicalSortExec<'a>),
    LimitExec(PhysicalLimitExec<'a>),
    TopKExec(PhysicalTopKExec<'a>),
    SubqueryExec(PhysicalSubqueryExec<'a>),
    SetOpExec(PhysicalSetOpExec<'a>),
    WindowExec(PhysicalWindowExec<'a>),
    ScalarSubqueryExec(PhysicalScalarSubqueryExec<'a>),
    ExistsSubqueryExec(PhysicalExistsSubqueryExec<'a>),
    InListSubqueryExec(PhysicalInListSubqueryExec<'a>),
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
pub struct PhysicalStreamingHashJoin<'a> {
    pub build: &'a PhysicalOperator<'a>,
    pub probe: &'a PhysicalOperator<'a>,
    pub join_type: JoinType,
    pub join_keys: &'a [(&'a Expr<'a>, &'a Expr<'a>)],
    pub swapped: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub struct PhysicalHashSemiJoin<'a> {
    pub left: &'a PhysicalOperator<'a>,
    pub right: &'a PhysicalOperator<'a>,
    pub join_keys: &'a [(&'a Expr<'a>, &'a Expr<'a>)],
}

#[derive(Debug, Clone, PartialEq)]
pub struct PhysicalHashAntiJoin<'a> {
    pub left: &'a PhysicalOperator<'a>,
    pub right: &'a PhysicalOperator<'a>,
    pub join_keys: &'a [(&'a Expr<'a>, &'a Expr<'a>)],
}

#[derive(Debug, Clone, PartialEq)]
pub struct PhysicalScalarSubqueryExec<'a> {
    pub subquery: &'a PhysicalOperator<'a>,
    pub is_correlated: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub struct PhysicalExistsSubqueryExec<'a> {
    pub subquery: &'a PhysicalOperator<'a>,
    pub negated: bool,
    pub is_correlated: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub struct PhysicalInListSubqueryExec<'a> {
    pub expr: &'a Expr<'a>,
    pub subquery: &'a PhysicalOperator<'a>,
    pub negated: bool,
    pub is_correlated: bool,
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
            PhysicalOperator::StreamingHashJoin(join) => {
                let _ = writeln!(output, "{}-> StreamingHashJoin ({:?})", prefix, join.join_type);
                let _ = writeln!(output, "{}  Build:", prefix);
                self.format_operator(join.build, indent + 2, output);
                let _ = writeln!(output, "{}  Probe (streaming):", prefix);
                self.format_operator(join.probe, indent + 2, output);
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
            PhysicalOperator::HashSemiJoin(join) => {
                let _ = writeln!(output, "{}-> HashSemiJoin", prefix);
                self.format_operator(join.left, indent + 1, output);
                self.format_operator(join.right, indent + 1, output);
            }
            PhysicalOperator::HashAntiJoin(join) => {
                let _ = writeln!(output, "{}-> HashAntiJoin", prefix);
                self.format_operator(join.left, indent + 1, output);
                self.format_operator(join.right, indent + 1, output);
            }
            PhysicalOperator::ScalarSubqueryExec(subq) => {
                let _ = writeln!(
                    output,
                    "{}-> ScalarSubquery (correlated={})",
                    prefix, subq.is_correlated
                );
                self.format_operator(subq.subquery, indent + 1, output);
            }
            PhysicalOperator::ExistsSubqueryExec(subq) => {
                let _ = writeln!(
                    output,
                    "{}-> ExistsSubquery (negated={}, correlated={})",
                    prefix, subq.negated, subq.is_correlated
                );
                self.format_operator(subq.subquery, indent + 1, output);
            }
            PhysicalOperator::InListSubqueryExec(subq) => {
                let _ = writeln!(
                    output,
                    "{}-> InListSubquery (negated={}, correlated={})",
                    prefix, subq.negated, subq.is_correlated
                );
                self.format_operator(subq.subquery, indent + 1, output);
            }
        }
    }
}
