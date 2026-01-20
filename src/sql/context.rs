use bumpalo::Bump;
use crate::memory::MemoryBudget;
use crate::types::OwnedValue;
use hashbrown::HashMap;
use std::sync::Arc;

pub type ScalarSubqueryResults = HashMap<usize, OwnedValue>;

pub struct ExecutionContext<'a> {
    pub arena: &'a Bump,
    pub file_manager: Option<&'a mut crate::storage::FileManager>,
    pub catalog: Option<&'a crate::schema::Catalog>,
    pub scalar_subquery_results: Option<&'a ScalarSubqueryResults>,
    pub memory_budget: Option<&'a Arc<MemoryBudget>>,
}

impl<'a> ExecutionContext<'a> {
    pub fn new(arena: &'a Bump) -> Self {
        Self {
            arena,
            file_manager: None,
            catalog: None,
            scalar_subquery_results: None,
            memory_budget: None,
        }
    }

    pub fn with_memory_budget(arena: &'a Bump, memory_budget: &'a Arc<MemoryBudget>) -> Self {
        Self {
            arena,
            file_manager: None,
            catalog: None,
            scalar_subquery_results: None,
            memory_budget: Some(memory_budget),
        }
    }

    pub fn with_scalar_subqueries(arena: &'a Bump, results: &'a ScalarSubqueryResults) -> Self {
        Self {
            arena,
            file_manager: None,
            catalog: None,
            scalar_subquery_results: Some(results),
            memory_budget: None,
        }
    }

    pub fn with_scalar_subqueries_and_budget(
        arena: &'a Bump,
        results: &'a ScalarSubqueryResults,
        memory_budget: &'a Arc<MemoryBudget>,
    ) -> Self {
        Self {
            arena,
            file_manager: None,
            catalog: None,
            scalar_subquery_results: Some(results),
            memory_budget: Some(memory_budget),
        }
    }

    pub fn with_storage(
        arena: &'a Bump,
        file_manager: &'a mut crate::storage::FileManager,
        catalog: &'a crate::schema::Catalog,
    ) -> Self {
        Self {
            arena,
            file_manager: Some(file_manager),
            catalog: Some(catalog),
            scalar_subquery_results: None,
            memory_budget: None,
        }
    }

    pub fn get_table_storage(
        &mut self,
        schema: &str,
        table: &str,
    ) -> eyre::Result<std::sync::Arc<parking_lot::RwLock<crate::storage::MmapStorage>>> {
        let fm = self
            .file_manager
            .as_mut()
            .ok_or_else(|| eyre::eyre!("file manager not available in execution context"))?;
        fm.table_data_mut(schema, table)
    }

    pub fn get_table_def(&self, table_name: &str) -> eyre::Result<&crate::schema::TableDef> {
        let catalog = self
            .catalog
            .ok_or_else(|| eyre::eyre!("catalog not available in execution context"))?;
        catalog.resolve_table(table_name)
    }
}
