//! # Multi-Schema Catalog
//!
//! This module implements TurDB's multi-schema catalog system, supporting
//! PostgreSQL-style schema organization with namespaced tables.
//!
//! ## Architecture
//!
//! The catalog manages multiple independent schemas (namespaces) within a
//! single database. Each schema contains its own set of tables and can be
//! accessed using fully-qualified names (schema.table) or implicitly via
//! the default schema.
//!
//! ## Default Schemas
//!
//! Every database has two built-in schemas:
//! - `root`: The default user schema for application tables
//! - `turdb_catalog`: System schema for internal metadata (indexes, statistics, etc.)
//!
//! ## Schema Hierarchy
//!
//! ```text
//! Database
//! ├── Schema "root" (default)
//! │   ├── Table "users"
//! │   ├── Table "posts"
//! │   └── ...
//! ├── Schema "turdb_catalog" (system)
//! │   ├── Table "tables"
//! │   ├── Table "indexes"
//! │   └── ...
//! └── Schema "analytics" (user-created)
//!     └── Table "events"
//! ```
//!
//! ## Name Resolution
//!
//! Table names are resolved in this order:
//! 1. If fully-qualified (schema.table), look in that schema
//! 2. Otherwise, look in the default schema (usually "root")
//! 3. If not found, return an error
//!
//! This matches PostgreSQL's search_path behavior with a simplified
//! single-schema default.
//!
//! ## File Layout
//!
//! Each schema maps to a directory on disk:
//! ```
//! database_dir/
//! ├── turdb.meta           # Catalog metadata
//! ├── root/                # Default schema directory
//! │   ├── users.tbd
//! │   └── users_pk.idx
//! ├── turdb_catalog/       # System schema directory
//! │   └── ...
//! └── analytics/           # User schema directory
//!     └── events.tbd
//! ```
//!
//! ## Concurrency
//!
//! - Catalog uses a single RwLock (read-heavy workload)
//! - Schema creation/deletion is rare compared to table lookups
//! - Lock ordering: Catalog -> Schema -> Table
//!
//! ## Memory Efficiency
//!
//! - Schemas are eagerly loaded (all schemas loaded at startup)
//! - Table definitions within schemas are also eagerly loaded
//! - For very large catalogs (1000+ tables), consider lazy loading
//!   in a future optimization
//!
//! ## Schema Identifiers
//!
//! - Schema names: case-sensitive, alphanumeric + underscore
//! - Reserved names: "turdb_catalog" (system use only)
//! - Maximum name length: 255 bytes UTF-8
//!
//! ## Usage Example
//!
//! ```rust,ignore
//! let mut catalog = Catalog::new();
//!
//! // Create a new schema
//! catalog.create_schema("analytics")?;
//!
//! // Resolve table with fully-qualified name
//! let table = catalog.resolve_table("analytics.events")?;
//!
//! // Resolve table in default schema
//! let table = catalog.resolve_table("users")?;  // Looks in "root"
//!
//! // Drop a schema
//! catalog.drop_schema("analytics")?;
//! ```

pub mod catalog;
pub mod table;

pub use catalog::Catalog;
pub use table::{ColumnDef, Constraint, IndexDef, TableDef};

pub type SchemaId = u32;
pub type TableId = u64;
pub type IndexId = u64;

use std::collections::HashMap;

#[derive(Debug)]
pub struct Schema {
    id: SchemaId,
    name: String,
    tables: HashMap<String, TableDef>,
}

impl Schema {
    pub fn new(id: SchemaId, name: impl Into<String>) -> Self {
        Self {
            id,
            name: name.into(),
            tables: HashMap::new(),
        }
    }

    pub fn id(&self) -> SchemaId {
        self.id
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn add_table(&mut self, table: TableDef) {
        self.tables.insert(table.name().to_string(), table);
    }

    pub fn get_table(&self, name: &str) -> Option<&TableDef> {
        self.tables.get(name)
    }

    pub fn remove_table(&mut self, name: &str) -> Option<TableDef> {
        self.tables.remove(name)
    }

    pub fn table_exists(&self, name: &str) -> bool {
        self.tables.contains_key(name)
    }

    pub fn tables(&self) -> &HashMap<String, TableDef> {
        &self.tables
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_catalog_new() {
        let catalog = Catalog::new();

        assert_eq!(catalog.default_schema(), "root");
        assert!(catalog.schema_exists("root"));
        assert!(catalog.schema_exists("turdb_catalog"));
    }

    #[test]
    fn test_create_schema() {
        let mut catalog = Catalog::new();

        catalog.create_schema("analytics").unwrap();
        assert!(catalog.schema_exists("analytics"));
    }

    #[test]
    fn test_create_duplicate_schema() {
        let mut catalog = Catalog::new();

        catalog.create_schema("analytics").unwrap();
        let result = catalog.create_schema("analytics");

        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("already exists"));
    }

    #[test]
    fn test_drop_schema() {
        let mut catalog = Catalog::new();

        catalog.create_schema("temp").unwrap();
        assert!(catalog.schema_exists("temp"));

        catalog.drop_schema("temp").unwrap();
        assert!(!catalog.schema_exists("temp"));
    }

    #[test]
    fn test_drop_nonexistent_schema() {
        let mut catalog = Catalog::new();

        let result = catalog.drop_schema("nonexistent");
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("not found"));
    }

    #[test]
    fn test_drop_system_schema() {
        let mut catalog = Catalog::new();

        let result = catalog.drop_schema("turdb_catalog");
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("system schema"));
    }

    #[test]
    fn test_resolve_table_qualified() {
        let mut catalog = Catalog::new();
        catalog.create_schema("analytics").unwrap();

        let schema = catalog.get_schema_mut("analytics").unwrap();
        schema.add_table(TableDef::new(1, "events", vec![]));

        let table = catalog.resolve_table("analytics.events").unwrap();
        assert_eq!(table.name(), "events");
    }

    #[test]
    fn test_resolve_table_default_schema() {
        let mut catalog = Catalog::new();

        let schema = catalog.get_schema_mut("root").unwrap();
        schema.add_table(TableDef::new(1, "users", vec![]));

        let table = catalog.resolve_table("users").unwrap();
        assert_eq!(table.name(), "users");
    }

    #[test]
    fn test_resolve_table_not_found() {
        let catalog = Catalog::new();

        let result = catalog.resolve_table("nonexistent");
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("not found"));
    }

    #[test]
    fn test_resolve_table_schema_not_found() {
        let catalog = Catalog::new();

        let result = catalog.resolve_table("nonexistent.table");
        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(err_msg.contains("schema") || err_msg.contains("not found"));
    }
}
