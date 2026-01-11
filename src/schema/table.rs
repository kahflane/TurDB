//! # Table Definition Module
//!
//! This module provides the core schema definition types for TurDB tables,
//! columns, indexes, and constraints. These types represent the metadata
//! that describes the structure of database objects.
//!
//! ## Overview
//!
//! The schema system supports:
//! - **Tables**: Named collections of columns with optional primary keys and indexes
//! - **Columns**: Typed fields with optional constraints and default values
//! - **Indexes**: B-tree or HNSW indexes on one or more columns
//! - **Constraints**: NOT NULL, PRIMARY KEY, UNIQUE, FOREIGN KEY, CHECK
//!
//! ## Type System
//!
//! Column types are defined in `records::types::DataType` and include:
//! - Fixed-width: bool, int2, int4, int8, float4, float8, date, time, timestamp, uuid
//! - Variable-width: text, blob, vector, jsonb
//!
//! ## Table Definition Example
//!
//! ```rust,ignore
//! use turdb::schema::{TableDef, ColumnDef, IndexDef, Constraint, IndexType};
//! use turdb::records::types::DataType;
//!
//! let columns = vec![
//!     ColumnDef::new("id", DataType::Int8)
//!         .with_constraint(Constraint::PrimaryKey),
//!     ColumnDef::new("email", DataType::Text)
//!         .with_constraint(Constraint::NotNull)
//!         .with_constraint(Constraint::Unique),
//!     ColumnDef::new("created_at", DataType::Timestamp)
//!         .with_default("CURRENT_TIMESTAMP"),
//! ];
//!
//! let table = TableDef::new(1, "users", columns)
//!     .with_primary_key(vec!["id"])
//!     .with_index(IndexDef::new("idx_email", vec!["email"], true, IndexType::BTree));
//! ```
//!
//! ## Index Types
//!
//! TurDB supports two index types:
//! - **BTree**: Traditional B+tree for range queries and exact matches
//! - **Hnsw**: Hierarchical Navigable Small World graph for vector similarity search
//!
//! ## Constraints
//!
//! | Constraint | Description |
//! |------------|-------------|
//! | NotNull | Column cannot contain NULL values |
//! | PrimaryKey | Column is part of the primary key |
//! | Unique | Column values must be unique |
//! | ForeignKey | References a column in another table |
//! | Check | Custom expression that must evaluate to true |
//!
//! ## Memory Layout
//!
//! Schema definitions are stored in memory as Rust structs during runtime.
//! For persistence, they are serialized to the turdb.meta file using a
//! compact binary format (see schema::persistence module).
//!
//! ## Thread Safety
//!
//! Schema types are Clone but not Sync. The Catalog wraps schemas in
//! appropriate locks for concurrent access.

use crate::records::types::DataType;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IndexType {
    BTree,
    Hnsw,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ReferentialAction {
    Cascade,
    Restrict,
    #[default]
    NoAction,
    SetNull,
    SetDefault,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Constraint {
    NotNull,
    PrimaryKey,
    Unique,
    AutoIncrement,
    ForeignKey {
        table: String,
        column: String,
        on_delete: Option<ReferentialAction>,
        on_update: Option<ReferentialAction>,
    },
    Check(String),
}

#[derive(Debug, Clone, PartialEq)]
pub struct ColumnDef {
    name: String,
    data_type: DataType,
    constraints: Vec<Constraint>,
    default_value: Option<String>,
    max_length: Option<u32>,
}

impl ColumnDef {
    pub fn new(name: impl Into<String>, data_type: DataType) -> Self {
        Self {
            name: name.into(),
            data_type,
            constraints: Vec::new(),
            default_value: None,
            max_length: None,
        }
    }

    pub fn with_constraint(mut self, constraint: Constraint) -> Self {
        self.constraints.push(constraint);
        self
    }

    pub fn with_default(mut self, default: impl Into<String>) -> Self {
        self.default_value = Some(default.into());
        self
    }

    pub fn with_max_length(mut self, max_length: u32) -> Self {
        self.max_length = Some(max_length);
        self
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn data_type(&self) -> DataType {
        self.data_type
    }

    pub fn constraints(&self) -> &[Constraint] {
        &self.constraints
    }

    pub fn has_constraint(&self, constraint: &Constraint) -> bool {
        self.constraints.iter().any(|c| {
            std::mem::discriminant(c) == std::mem::discriminant(constraint)
                && match (c, constraint) {
                    (Constraint::NotNull, Constraint::NotNull) => true,
                    (Constraint::PrimaryKey, Constraint::PrimaryKey) => true,
                    (Constraint::Unique, Constraint::Unique) => true,
                    (Constraint::AutoIncrement, Constraint::AutoIncrement) => true,
                    (
                        Constraint::ForeignKey {
                            table: t1,
                            column: c1,
                            on_delete: d1,
                            on_update: u1,
                        },
                        Constraint::ForeignKey {
                            table: t2,
                            column: c2,
                            on_delete: d2,
                            on_update: u2,
                        },
                    ) => t1 == t2 && c1 == c2 && d1 == d2 && u1 == u2,
                    (Constraint::Check(e1), Constraint::Check(e2)) => e1 == e2,
                    _ => false,
                }
        })
    }

    pub fn default_value(&self) -> Option<&str> {
        self.default_value.as_deref()
    }

    pub fn is_nullable(&self) -> bool {
        !self.has_constraint(&Constraint::NotNull)
    }

    pub fn max_length(&self) -> Option<u32> {
        self.max_length
    }

    pub fn set_name(&mut self, name: impl Into<String>) {
        self.name = name.into();
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SortDirection {
    #[default]
    Asc,
    Desc,
}

#[derive(Debug, Clone, PartialEq)]
pub struct IndexColumnDef {
    pub column_or_expr: IndexColumnKind,
    pub direction: SortDirection,
}

#[derive(Debug, Clone, PartialEq)]
pub enum IndexColumnKind {
    Column(String),
    Expression(String),
}

impl IndexColumnDef {
    pub fn column(name: impl Into<String>) -> Self {
        Self {
            column_or_expr: IndexColumnKind::Column(name.into()),
            direction: SortDirection::Asc,
        }
    }

    pub fn column_desc(name: impl Into<String>) -> Self {
        Self {
            column_or_expr: IndexColumnKind::Column(name.into()),
            direction: SortDirection::Desc,
        }
    }

    pub fn expression(expr: impl Into<String>) -> Self {
        Self {
            column_or_expr: IndexColumnKind::Expression(expr.into()),
            direction: SortDirection::Asc,
        }
    }

    pub fn with_direction(mut self, direction: SortDirection) -> Self {
        self.direction = direction;
        self
    }

    pub fn is_column(&self) -> bool {
        matches!(self.column_or_expr, IndexColumnKind::Column(_))
    }

    pub fn is_expression(&self) -> bool {
        matches!(self.column_or_expr, IndexColumnKind::Expression(_))
    }

    pub fn as_column(&self) -> Option<&str> {
        match &self.column_or_expr {
            IndexColumnKind::Column(c) => Some(c),
            IndexColumnKind::Expression(_) => None,
        }
    }

    pub fn as_expression(&self) -> Option<&str> {
        match &self.column_or_expr {
            IndexColumnKind::Column(_) => None,
            IndexColumnKind::Expression(e) => Some(e),
        }
    }

    pub fn is_desc(&self) -> bool {
        self.direction == SortDirection::Desc
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct IndexDef {
    name: String,
    column_defs: Vec<IndexColumnDef>,
    is_unique: bool,
    index_type: IndexType,
    where_clause: Option<String>,
}

impl IndexDef {
    pub fn new(
        name: impl Into<String>,
        columns: Vec<impl Into<String>>,
        is_unique: bool,
        index_type: IndexType,
    ) -> Self {
        Self {
            name: name.into(),
            column_defs: columns
                .into_iter()
                .map(|c| IndexColumnDef::column(c.into()))
                .collect(),
            is_unique,
            index_type,
            where_clause: None,
        }
    }

    pub fn new_expression(
        name: impl Into<String>,
        column_defs: Vec<IndexColumnDef>,
        is_unique: bool,
        index_type: IndexType,
    ) -> Self {
        Self {
            name: name.into(),
            column_defs,
            is_unique,
            index_type,
            where_clause: None,
        }
    }

    pub fn with_where_clause(mut self, predicate: String) -> Self {
        self.where_clause = Some(predicate);
        self
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn columns(&self) -> impl Iterator<Item = &str> + '_ {
        self.column_defs.iter().filter_map(|cd| cd.as_column())
    }

    pub fn column_defs(&self) -> &[IndexColumnDef] {
        &self.column_defs
    }

    pub fn is_unique(&self) -> bool {
        self.is_unique
    }

    pub fn index_type(&self) -> IndexType {
        self.index_type
    }

    pub fn has_expressions(&self) -> bool {
        self.column_defs.iter().any(|cd| cd.is_expression())
    }

    pub fn is_partial(&self) -> bool {
        self.where_clause.is_some()
    }

    pub fn where_clause(&self) -> Option<&str> {
        self.where_clause.as_deref()
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct TableDef {
    id: u64,
    name: String,
    columns: Vec<ColumnDef>,
    primary_key: Option<Vec<String>>,
    indexes: Vec<IndexDef>,
    toast_id: Option<u64>,
}

impl TableDef {
    pub fn new(id: u64, name: impl Into<String>, columns: Vec<ColumnDef>) -> Self {
        Self {
            id,
            name: name.into(),
            columns,
            primary_key: None,
            indexes: Vec::new(),
            toast_id: None,
        }
    }

    pub fn with_primary_key(mut self, columns: Vec<impl Into<String>>) -> Self {
        self.primary_key = Some(columns.into_iter().map(|c| c.into()).collect());
        self
    }

    pub fn with_index(mut self, index: IndexDef) -> Self {
        self.indexes.push(index);
        self
    }

    pub fn with_toast_id(mut self, toast_id: u64) -> Self {
        self.toast_id = Some(toast_id);
        self
    }

    pub fn has_toast(&self) -> bool {
        self.toast_id.is_some()
    }

    pub fn toast_id(&self) -> Option<u64> {
        self.toast_id
    }

    pub fn set_toast_id(&mut self, toast_id: Option<u64>) {
        self.toast_id = toast_id;
    }

    pub fn id(&self) -> u64 {
        self.id
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn columns(&self) -> &[ColumnDef] {
        &self.columns
    }

    pub fn primary_key(&self) -> Option<&[String]> {
        self.primary_key.as_deref()
    }

    pub fn indexes(&self) -> &[IndexDef] {
        &self.indexes
    }

    pub fn get_column(&self, name: &str) -> Option<&ColumnDef> {
        self.columns.iter().find(|c| c.name() == name)
    }

    pub fn column_index(&self, name: &str) -> Option<usize> {
        self.columns.iter().position(|c| c.name() == name)
    }

    pub fn get_index(&self, name: &str) -> Option<&IndexDef> {
        self.indexes.iter().find(|i| i.name() == name)
    }

    pub fn remove_index(&mut self, name: &str) -> Option<IndexDef> {
        if let Some(pos) = self.indexes.iter().position(|i| i.name() == name) {
            Some(self.indexes.remove(pos))
        } else {
            None
        }
    }

    pub fn add_index(&mut self, index: IndexDef) {
        self.indexes.push(index);
    }

    pub fn drop_column(&mut self, name: &str) -> Option<ColumnDef> {
        if let Some(pos) = self.columns.iter().position(|c| c.name() == name) {
            Some(self.columns.remove(pos))
        } else {
            None
        }
    }

    pub fn rename(&mut self, new_name: impl Into<String>) {
        self.name = new_name.into();
    }

    pub fn add_column(&mut self, column: ColumnDef) {
        self.columns.push(column);
    }

    pub fn rename_column(&mut self, old_name: &str, new_name: &str) -> bool {
        if let Some(col) = self.columns.iter_mut().find(|c| c.name() == old_name) {
            col.set_name(new_name);
            true
        } else {
            false
        }
    }
}
