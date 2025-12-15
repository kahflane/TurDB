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

#[derive(Debug, Clone, PartialEq)]
pub enum Constraint {
    NotNull,
    PrimaryKey,
    Unique,
    ForeignKey { table: String, column: String },
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
                    (
                        Constraint::ForeignKey {
                            table: t1,
                            column: c1,
                        },
                        Constraint::ForeignKey {
                            table: t2,
                            column: c2,
                        },
                    ) => t1 == t2 && c1 == c2,
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
}

#[derive(Debug, Clone, PartialEq)]
pub enum IndexColumnDef {
    Column(String),
    Expression(String),
}

impl IndexColumnDef {
    pub fn is_column(&self) -> bool {
        matches!(self, IndexColumnDef::Column(_))
    }

    pub fn is_expression(&self) -> bool {
        matches!(self, IndexColumnDef::Expression(_))
    }

    pub fn as_column(&self) -> Option<&str> {
        match self {
            IndexColumnDef::Column(c) => Some(c),
            IndexColumnDef::Expression(_) => None,
        }
    }

    pub fn as_expression(&self) -> Option<&str> {
        match self {
            IndexColumnDef::Column(_) => None,
            IndexColumnDef::Expression(e) => Some(e),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct IndexDef {
    name: String,
    column_defs: Vec<IndexColumnDef>,
    is_unique: bool,
    index_type: IndexType,
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
                .map(|c| IndexColumnDef::Column(c.into()))
                .collect(),
            is_unique,
            index_type,
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
        }
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn columns(&self) -> Vec<String> {
        self.column_defs
            .iter()
            .filter_map(|cd| cd.as_column().map(|s| s.to_string()))
            .collect()
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
}

#[derive(Debug, Clone, PartialEq)]
pub struct TableDef {
    id: u64,
    name: String,
    columns: Vec<ColumnDef>,
    primary_key: Option<Vec<String>>,
    indexes: Vec<IndexDef>,
}

impl TableDef {
    pub fn new(id: u64, name: impl Into<String>, columns: Vec<ColumnDef>) -> Self {
        Self {
            id,
            name: name.into(),
            columns,
            primary_key: None,
            indexes: Vec::new(),
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
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn index_def_supports_expression_columns() {
        let expr_index = IndexDef::new_expression(
            "idx_lower_email",
            vec![IndexColumnDef::Expression("LOWER(email)".to_string())],
            false,
            IndexType::BTree,
        );

        assert_eq!(expr_index.name(), "idx_lower_email");
        assert!(!expr_index.is_unique());
        assert_eq!(expr_index.index_type(), IndexType::BTree);

        let cols = expr_index.column_defs();
        assert_eq!(cols.len(), 1);
        assert!(matches!(&cols[0], IndexColumnDef::Expression(e) if e == "LOWER(email)"));
    }

    #[test]
    fn index_def_supports_mixed_columns_and_expressions() {
        let mixed_index = IndexDef::new_expression(
            "idx_mixed",
            vec![
                IndexColumnDef::Column("tenant_id".to_string()),
                IndexColumnDef::Expression("LOWER(email)".to_string()),
            ],
            true,
            IndexType::BTree,
        );

        assert_eq!(mixed_index.name(), "idx_mixed");
        assert!(mixed_index.is_unique());

        let cols = mixed_index.column_defs();
        assert_eq!(cols.len(), 2);
        assert!(matches!(&cols[0], IndexColumnDef::Column(c) if c == "tenant_id"));
        assert!(matches!(&cols[1], IndexColumnDef::Expression(e) if e == "LOWER(email)"));
    }

    #[test]
    fn index_def_columns_returns_column_names_only() {
        let mixed_index = IndexDef::new_expression(
            "idx_mixed",
            vec![
                IndexColumnDef::Column("tenant_id".to_string()),
                IndexColumnDef::Expression("LOWER(email)".to_string()),
                IndexColumnDef::Column("status".to_string()),
            ],
            false,
            IndexType::BTree,
        );

        let column_names = mixed_index.columns();
        assert_eq!(column_names.len(), 2);
        assert_eq!(column_names[0], "tenant_id");
        assert_eq!(column_names[1], "status");
    }

    #[test]
    fn index_def_backward_compatibility() {
        let simple_index = IndexDef::new("idx_email", vec!["email"], false, IndexType::BTree);

        assert_eq!(simple_index.name(), "idx_email");
        assert_eq!(simple_index.columns(), &["email".to_string()]);

        let cols = simple_index.column_defs();
        assert_eq!(cols.len(), 1);
        assert!(matches!(&cols[0], IndexColumnDef::Column(c) if c == "email"));
    }
}
