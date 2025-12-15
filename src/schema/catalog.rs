//! # Catalog Module
//!
//! The catalog manages all schemas within a database.

use super::table::{ColumnDef, TableDef};
use super::{Schema, SchemaId, TableId};
use eyre::{bail, ensure, Result};
use std::collections::HashMap;

#[derive(Debug)]
pub struct Catalog {
    schemas: HashMap<String, Schema>,
    default_schema: String,
    next_schema_id: SchemaId,
    next_table_id: TableId,
}

impl Catalog {
    pub fn new() -> Self {
        let mut catalog = Self {
            schemas: HashMap::new(),
            default_schema: "root".to_string(),
            next_schema_id: 1,
            next_table_id: 1,
        };

        catalog
            .schemas
            .insert("root".to_string(), Schema::new(0, "root"));

        catalog
            .schemas
            .insert("turdb_catalog".to_string(), Schema::new(1, "turdb_catalog"));

        catalog.next_schema_id = 2;
        catalog
    }

    pub fn default_schema(&self) -> &str {
        &self.default_schema
    }

    pub fn schema_exists(&self, name: &str) -> bool {
        self.schemas.contains_key(name)
    }

    pub fn create_schema(&mut self, name: impl Into<String>) -> Result<SchemaId> {
        let name_str = name.into();

        ensure!(
            !self.schemas.contains_key(&name_str),
            "schema '{}' already exists",
            name_str
        );

        let id = self.next_schema_id;
        self.next_schema_id += 1;

        self.schemas
            .insert(name_str.clone(), Schema::new(id, name_str));

        Ok(id)
    }

    pub fn drop_schema(&mut self, name: &str) -> Result<()> {
        ensure!(
            name != "turdb_catalog",
            "cannot drop system schema 'turdb_catalog'"
        );

        ensure!(
            self.schemas.contains_key(name),
            "schema '{}' not found",
            name
        );

        self.schemas.remove(name);
        Ok(())
    }

    pub fn get_schema(&self, name: &str) -> Option<&Schema> {
        self.schemas.get(name)
    }

    pub fn get_schema_mut(&mut self, name: &str) -> Option<&mut Schema> {
        self.schemas.get_mut(name)
    }

    pub fn resolve_table(&self, name: &str) -> Result<&TableDef> {
        if let Some((schema_name, table_name)) = name.split_once('.') {
            let schema = self
                .schemas
                .get(schema_name)
                .ok_or_else(|| eyre::eyre!("schema '{}' not found", schema_name))?;

            schema.get_table(table_name).ok_or_else(|| {
                eyre::eyre!(
                    "table '{}' not found in schema '{}'",
                    table_name,
                    schema_name
                )
            })
        } else {
            let schema = self
                .schemas
                .get(&self.default_schema)
                .ok_or_else(|| eyre::eyre!("default schema '{}' not found", self.default_schema))?;

            schema.get_table(name).ok_or_else(|| {
                eyre::eyre!(
                    "table '{}' not found in default schema '{}'",
                    name,
                    self.default_schema
                )
            })
        }
    }

    pub fn schemas(&self) -> &HashMap<String, Schema> {
        &self.schemas
    }

    pub fn create_table(
        &mut self,
        schema_name: &str,
        table_name: &str,
        columns: Vec<ColumnDef>,
    ) -> Result<TableId> {
        let table_id = self.next_table_id;
        self.next_table_id += 1;
        self.create_table_with_id(schema_name, table_name, columns, table_id)
    }

    pub fn create_table_with_id(
        &mut self,
        schema_name: &str,
        table_name: &str,
        columns: Vec<ColumnDef>,
        table_id: TableId,
    ) -> Result<TableId> {
        let schema_name = if schema_name.is_empty() {
            &self.default_schema
        } else {
            schema_name
        };

        let schema = self
            .schemas
            .get_mut(schema_name)
            .ok_or_else(|| eyre::eyre!("schema '{}' not found", schema_name))?;

        ensure!(
            !schema.table_exists(table_name),
            "table '{}' already exists in schema '{}'",
            table_name,
            schema_name
        );

        let table = TableDef::new(table_id, table_name, columns);
        schema.add_table(table);

        Ok(table_id)
    }

    pub fn drop_table(&mut self, schema_name: &str, table_name: &str) -> Result<()> {
        let schema_name = if schema_name.is_empty() {
            &self.default_schema
        } else {
            schema_name
        };

        let schema = self
            .schemas
            .get_mut(schema_name)
            .ok_or_else(|| eyre::eyre!("schema '{}' not found", schema_name))?;

        ensure!(
            schema.table_exists(table_name),
            "table '{}' not found in schema '{}'",
            table_name,
            schema_name
        );

        schema.remove_table(table_name);
        Ok(())
    }

    pub fn get_table(&self, schema_name: &str, table_name: &str) -> Option<&TableDef> {
        let schema_name = if schema_name.is_empty() {
            &self.default_schema
        } else {
            schema_name
        };

        self.schemas
            .get(schema_name)
            .and_then(|s| s.get_table(table_name))
    }

    pub fn get_table_mut(&mut self, schema_name: &str, table_name: &str) -> Option<&mut TableDef> {
        let schema_name = if schema_name.is_empty() {
            &self.default_schema
        } else {
            schema_name
        };

        self.schemas
            .get_mut(schema_name)
            .and_then(|s| s.get_table_mut(table_name))
    }

    pub fn find_index(&self, index_name: &str) -> Option<(&str, &str, &crate::schema::IndexDef)> {
        for (schema_name, schema) in &self.schemas {
            for (table_name, table) in schema.tables() {
                if let Some(index) = table.get_index(index_name) {
                    return Some((schema_name, table_name, index));
                }
            }
        }
        None
    }

    pub fn remove_index(&mut self, index_name: &str) -> Result<()> {
        for schema in self.schemas.values_mut() {
            for table in schema.tables_mut().values_mut() {
                if table.remove_index(index_name).is_some() {
                    return Ok(());
                }
            }
        }
        bail!("index '{}' not found", index_name)
    }

    pub fn table_by_id(&self, table_id: u64) -> Option<&TableDef> {
        for schema in self.schemas.values() {
            for table in schema.tables().values() {
                if table.id() == table_id {
                    return Some(table);
                }
            }
        }
        None
    }
}

impl Default for Catalog {
    fn default() -> Self {
        Self::new()
    }
}
