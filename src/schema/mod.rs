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
//! ```text
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
pub mod persistence;
pub mod table;

pub use catalog::Catalog;
pub use table::{ColumnDef, Constraint, IndexColumnDef, IndexDef, IndexType, TableDef};

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
    use crate::records::types::DataType;

    #[test]
    fn test_column_def_with_type_and_constraints() {
        let col = ColumnDef::new("email", DataType::Text)
            .with_constraint(Constraint::NotNull)
            .with_constraint(Constraint::Unique);

        assert_eq!(col.name(), "email");
        assert_eq!(col.data_type(), DataType::Text);
        assert!(col.has_constraint(&Constraint::NotNull));
        assert!(col.has_constraint(&Constraint::Unique));
        assert!(!col.has_constraint(&Constraint::PrimaryKey));
    }

    #[test]
    fn test_column_def_with_default() {
        let col =
            ColumnDef::new("created_at", DataType::Timestamp).with_default("CURRENT_TIMESTAMP");

        assert_eq!(col.default_value(), Some("CURRENT_TIMESTAMP"));
    }

    #[test]
    fn test_constraint_foreign_key() {
        let fk = Constraint::ForeignKey {
            table: "users".to_string(),
            column: "id".to_string(),
        };

        if let Constraint::ForeignKey { table, column } = &fk {
            assert_eq!(table, "users");
            assert_eq!(column, "id");
        } else {
            panic!("Expected ForeignKey constraint");
        }
    }

    #[test]
    fn test_constraint_check() {
        let check = Constraint::Check("age >= 0".to_string());

        if let Constraint::Check(expr) = &check {
            assert_eq!(expr, "age >= 0");
        } else {
            panic!("Expected Check constraint");
        }
    }

    #[test]
    fn test_table_def_with_primary_key() {
        let columns = vec![
            ColumnDef::new("id", DataType::Int8),
            ColumnDef::new("name", DataType::Text),
        ];

        let table = TableDef::new(1, "users", columns).with_primary_key(vec!["id"]);

        assert_eq!(table.primary_key(), Some(&["id".to_string()][..]));
    }

    #[test]
    fn test_table_def_with_composite_primary_key() {
        let columns = vec![
            ColumnDef::new("user_id", DataType::Int8),
            ColumnDef::new("role_id", DataType::Int8),
            ColumnDef::new("assigned_at", DataType::Timestamp),
        ];

        let table =
            TableDef::new(1, "user_roles", columns).with_primary_key(vec!["user_id", "role_id"]);

        let pk = table.primary_key().unwrap();
        assert_eq!(pk.len(), 2);
        assert_eq!(pk[0], "user_id");
        assert_eq!(pk[1], "role_id");
    }

    #[test]
    fn test_table_def_with_indexes() {
        let columns = vec![
            ColumnDef::new("id", DataType::Int8),
            ColumnDef::new("email", DataType::Text),
        ];

        let idx = IndexDef::new("idx_users_email", vec!["email"], true, IndexType::BTree);

        let table = TableDef::new(1, "users", columns).with_index(idx);

        assert_eq!(table.indexes().len(), 1);
        assert_eq!(table.indexes()[0].name(), "idx_users_email");
    }

    #[test]
    fn test_index_def_btree() {
        let idx = IndexDef::new("idx_email", vec!["email"], true, IndexType::BTree);

        assert_eq!(idx.name(), "idx_email");
        assert_eq!(idx.columns(), &["email".to_string()]);
        assert!(idx.is_unique());
        assert_eq!(idx.index_type(), IndexType::BTree);
    }

    #[test]
    fn test_index_def_hnsw() {
        let idx = IndexDef::new("idx_embedding", vec!["embedding"], false, IndexType::Hnsw);

        assert_eq!(idx.index_type(), IndexType::Hnsw);
        assert!(!idx.is_unique());
    }

    #[test]
    fn test_index_def_multi_column() {
        let idx = IndexDef::new(
            "idx_name_email",
            vec!["last_name", "first_name"],
            false,
            IndexType::BTree,
        );

        assert_eq!(idx.columns().len(), 2);
        assert_eq!(idx.columns()[0], "last_name");
        assert_eq!(idx.columns()[1], "first_name");
    }

    #[test]
    fn test_catalog_create_table() {
        let mut catalog = Catalog::new();

        let columns = vec![
            ColumnDef::new("id", DataType::Int8),
            ColumnDef::new("name", DataType::Text),
        ];

        catalog.create_table("root", "users", columns).unwrap();

        let table = catalog.get_table("root", "users").unwrap();
        assert_eq!(table.name(), "users");
        assert_eq!(table.columns().len(), 2);
    }

    #[test]
    fn test_catalog_create_table_default_schema() {
        let mut catalog = Catalog::new();

        let columns = vec![ColumnDef::new("id", DataType::Int8)];

        catalog.create_table("", "items", columns).unwrap();

        let table = catalog.get_table("root", "items").unwrap();
        assert_eq!(table.name(), "items");
    }

    #[test]
    fn test_catalog_create_table_duplicate_fails() {
        let mut catalog = Catalog::new();

        let columns = vec![ColumnDef::new("id", DataType::Int8)];
        catalog
            .create_table("root", "users", columns.clone())
            .unwrap();

        let result = catalog.create_table("root", "users", columns);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("already exists"));
    }

    #[test]
    fn test_catalog_drop_table() {
        let mut catalog = Catalog::new();

        let columns = vec![ColumnDef::new("id", DataType::Int8)];
        catalog.create_table("root", "temp", columns).unwrap();

        assert!(catalog.get_table("root", "temp").is_some());

        catalog.drop_table("root", "temp").unwrap();
        assert!(catalog.get_table("root", "temp").is_none());
    }

    #[test]
    fn test_catalog_drop_nonexistent_table() {
        let mut catalog = Catalog::new();

        let result = catalog.drop_table("root", "nonexistent");
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("not found"));
    }

    #[test]
    fn test_catalog_get_table() {
        let mut catalog = Catalog::new();

        let columns = vec![
            ColumnDef::new("id", DataType::Int8),
            ColumnDef::new("email", DataType::Text),
        ];
        catalog.create_table("root", "accounts", columns).unwrap();

        let table = catalog.get_table("root", "accounts");
        assert!(table.is_some());

        let table = table.unwrap();
        assert_eq!(table.columns().len(), 2);
        assert_eq!(table.columns()[0].name(), "id");
        assert_eq!(table.columns()[1].name(), "email");
    }

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
