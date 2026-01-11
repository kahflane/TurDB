//! # Planner Type Definitions
//!
//! This module contains helper types and enums used throughout the query planner.
//! These types are fundamental building blocks that don't fit into the logical
//! or physical operator categories.
//!
//! ## Types
//!
//! - `ScanRange`: Describes the range of keys to scan in an index or table
//! - `PlanNode`: A union type that can hold either a logical or physical operator
//!
//! ## Usage
//!
//! These types are primarily used internally by the planner and optimizer
//! to represent intermediate states and scan configurations.

use super::logical::LogicalOperator;
use super::physical::PhysicalOperator;

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
