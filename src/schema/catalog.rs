//! # Catalog Module
//!
//! The catalog manages all schemas within a database.

use super::table::TableDef;
use super::{Schema, SchemaId};
use eyre::{ensure, Result};
use std::collections::HashMap;

#[derive(Debug)]
pub struct Catalog {
    schemas: HashMap<String, Schema>,
    default_schema: String,
    next_schema_id: SchemaId,
}

impl Catalog {
    pub fn new() -> Self {
        let mut catalog = Self {
            schemas: HashMap::new(),
            default_schema: "root".to_string(),
            next_schema_id: 1,
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
}

impl Default for Catalog {
    fn default() -> Self {
        Self::new()
    }
}
