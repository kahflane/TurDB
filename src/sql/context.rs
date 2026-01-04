use bumpalo::Bump;

pub struct ExecutionContext<'a> {
    pub arena: &'a Bump,
    pub file_manager: Option<&'a mut crate::storage::FileManager>,
    pub catalog: Option<&'a crate::schema::Catalog>,
}

impl<'a> ExecutionContext<'a> {
    pub fn new(arena: &'a Bump) -> Self {
        Self {
            arena,
            file_manager: None,
            catalog: None,
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
