//! # Table Definition Module
//!
//! Provides structures for defining tables, columns, indexes, and constraints.

#[derive(Debug, Clone)]
pub struct TableDef {
    id: u64,
    name: String,
    columns: Vec<ColumnDef>,
}

impl TableDef {
    pub fn new(id: u64, name: impl Into<String>, columns: Vec<ColumnDef>) -> Self {
        Self {
            id,
            name: name.into(),
            columns,
        }
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
}

#[derive(Debug, Clone)]
pub struct ColumnDef {
    name: String,
}

impl ColumnDef {
    pub fn new(name: impl Into<String>) -> Self {
        Self { name: name.into() }
    }

    pub fn name(&self) -> &str {
        &self.name
    }
}

#[derive(Debug, Clone)]
pub struct IndexDef {
    name: String,
}

impl IndexDef {
    pub fn new(name: impl Into<String>) -> Self {
        Self { name: name.into() }
    }

    pub fn name(&self) -> &str {
        &self.name
    }
}

#[derive(Debug, Clone)]
pub enum Constraint {
    NotNull,
    PrimaryKey,
    Unique,
}
