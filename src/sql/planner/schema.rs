//! # Schema Types for Query Planning
//!
//! This module contains types that describe the schema of data flowing through
//! query operators. These types are essential for type checking, column resolution,
//! and query validation.
//!
//! ## Types
//!
//! - `TableSource`: Represents a table or subquery in the FROM clause
//! - `OutputColumn`: Describes a single column in a result set
//! - `OutputSchema`: Describes the complete schema of a query result
//! - `CteContext`: Tracks Common Table Expressions (WITH clauses) during planning
//! - `PlannedCte`: A planned CTE with its output schema
//!
//! ## Design
//!
//! All types use arena allocation via bumpalo for efficient memory management.
//! The lifetime parameter 'a represents the lifetime of the planning arena.

use crate::records::types::DataType;
use crate::schema::TableDef;

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

pub struct CteContext<'a> {
    pub ctes: hashbrown::HashMap<&'a str, PlannedCte<'a>>,
}

pub struct PlannedCte<'a> {
    pub plan: &'a super::logical::LogicalOperator<'a>,
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
