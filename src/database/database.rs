use crate::database::owned_value::OwnedValue;
use crate::database::row::Row;
use crate::database::{CheckpointInfo, ExecuteResult, RecoveryInfo};
use crate::schema::{Catalog, ColumnDef as SchemaColumnDef};
use crate::sql::executor::{Executor, StreamingBTreeSource};
use crate::sql::planner::Planner;
use crate::sql::Parser;
use crate::storage::{FileManager, Wal, WalStorage};
use bumpalo::Bump;
use eyre::{bail, ensure, Result, WrapErr};
use hashbrown::HashSet;
use parking_lot::{Mutex, RwLock};
use std::path::{Path, PathBuf};
use crate::sql::builder::ExecutorBuilder;
use crate::sql::context::ExecutionContext;
use crate::sql::predicate::CompiledPredicate;

pub struct Database {
    path: PathBuf,
    file_manager: RwLock<Option<FileManager>>,
    catalog: RwLock<Option<Catalog>>,
    wal: Mutex<Option<Wal>>,
    wal_dir: PathBuf,
    next_row_id: std::sync::atomic::AtomicU64,
    next_table_id: std::sync::atomic::AtomicU64,
    next_index_id: std::sync::atomic::AtomicU64,
    closed: std::sync::atomic::AtomicBool,
    wal_enabled: std::sync::atomic::AtomicBool,
    dirty_pages: Mutex<HashSet<u32>>,
}

impl Database {
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self> {
        Self::open_with_recovery(path).map(|(db, _)| db)
    }

    pub fn open_with_recovery<P: AsRef<Path>>(path: P) -> Result<(Self, RecoveryInfo)> {
        use crate::storage::{MetaFileHeader, FILE_HEADER_SIZE};
        use std::fs::File;
        use std::io::Read;
        use std::sync::atomic::{AtomicBool, AtomicU64};

        let path = path.as_ref().to_path_buf();

        let meta_path = path.join("turdb.meta");
        ensure!(meta_path.exists(), "database not found at {:?}", path);

        let mut header_bytes = [0u8; FILE_HEADER_SIZE];
        File::open(&meta_path)
            .wrap_err_with(|| format!("failed to open metadata file at {:?}", meta_path))?
            .read_exact(&mut header_bytes)
            .wrap_err("failed to read database header")?;

        let header = MetaFileHeader::from_bytes(&header_bytes)
            .wrap_err("failed to parse database metadata header")?;

        let next_table_id = header.next_table_id();
        let next_index_id = header.next_index_id();

        let wal_dir = path.join("wal");

        let wal_size_bytes = if wal_dir.exists() {
            let segment_path = wal_dir.join("wal.000001");
            if segment_path.exists() {
                std::fs::metadata(&segment_path)
                    .map(|m| m.len())
                    .unwrap_or(0)
            } else {
                0
            }
        } else {
            0
        };

        let db = Self {
            path,
            file_manager: RwLock::new(None),
            catalog: RwLock::new(None),
            wal: Mutex::new(None),
            wal_dir,
            next_row_id: AtomicU64::new(1),
            next_table_id: AtomicU64::new(next_table_id),
            next_index_id: AtomicU64::new(next_index_id),
            closed: AtomicBool::new(false),
            wal_enabled: AtomicBool::new(false),
            dirty_pages: Mutex::new(HashSet::new()),
        };

        let recovery_info = RecoveryInfo {
            frames_recovered: 0,
            wal_size_bytes,
        };

        Ok((db, recovery_info))
    }

    pub fn create<P: AsRef<Path>>(path: P) -> Result<Self> {
        use crate::storage::{MetaFileHeader, PAGE_SIZE};
        use std::fs::File;
        use std::io::Write;
        use std::sync::atomic::{AtomicBool, AtomicU64};
        use zerocopy::IntoBytes;

        let path = path.as_ref().to_path_buf();

        std::fs::create_dir_all(&path)
            .wrap_err_with(|| format!("failed to create database directory at {:?}", path))?;

        let root_dir = path.join("root");
        std::fs::create_dir_all(&root_dir).wrap_err_with(|| {
            format!("failed to create root schema directory at {:?}", root_dir)
        })?;

        let meta_path = path.join("turdb.meta");
        let mut page = vec![0u8; PAGE_SIZE];
        let header = MetaFileHeader::new();
        page[..128].copy_from_slice(header.as_bytes());

        let mut file = File::create(&meta_path)
            .wrap_err_with(|| format!("failed to create metadata file at {:?}", meta_path))?;
        file.write_all(&page)
            .wrap_err("failed to write database header")?;

        let wal_dir = path.join("wal");

        Ok(Self {
            path,
            file_manager: RwLock::new(None),
            catalog: RwLock::new(None),
            wal: Mutex::new(None),
            wal_dir,
            next_row_id: AtomicU64::new(1),
            next_table_id: AtomicU64::new(1),
            next_index_id: AtomicU64::new(1),
            closed: AtomicBool::new(false),
            wal_enabled: AtomicBool::new(false),
            dirty_pages: Mutex::new(HashSet::new()),
        })
    }

    pub fn open_or_create<P: AsRef<Path>>(path: P) -> Result<Self> {
        let path = path.as_ref();
        let meta_path = path.join("turdb.meta");

        if meta_path.exists() {
            Self::open(path)
        } else {
            Self::create(path)
        }
    }

    fn ensure_file_manager(&self) -> Result<()> {
        let mut guard = self.file_manager.write();
        if guard.is_none() {
            let fm = FileManager::open(&self.path, 64)
                .wrap_err_with(|| format!("failed to open file manager at {:?}", self.path))?;
            *guard = Some(fm);
        }
        Ok(())
    }

    fn ensure_catalog(&self) -> Result<()> {
        let mut guard = self.catalog.write();
        if guard.is_none() {
            let catalog = Self::load_catalog(&self.path)?;
            *guard = Some(catalog);
        }
        Ok(())
    }

    pub fn ensure_wal(&self) -> Result<()> {
        let mut guard = self.wal.lock();
        if guard.is_none() {
            let wal = if self.wal_dir.exists() {
                Wal::open(&self.wal_dir)
                    .wrap_err_with(|| format!("failed to open WAL at {:?}", self.wal_dir))?
            } else {
                Wal::create(&self.wal_dir)
                    .wrap_err_with(|| format!("failed to create WAL at {:?}", self.wal_dir))?
            };
            *guard = Some(wal);
        }
        Ok(())
    }

    fn load_catalog(path: &Path) -> Result<Catalog> {
        use crate::schema::persistence::CatalogPersistence;

        let catalog_path = path.join("turdb.catalog");
        let mut catalog = Catalog::new();
        if catalog_path.exists() {
            CatalogPersistence::load(&catalog_path, &mut catalog)
                .wrap_err_with(|| format!("failed to load catalog from {:?}", catalog_path))?;
        }
        Ok(catalog)
    }

    fn save_catalog(&self) -> Result<()> {
        use crate::schema::persistence::CatalogPersistence;

        let catalog_path = self.path.join("turdb.catalog");
        let catalog_guard = self.catalog.read();
        if let Some(ref catalog) = *catalog_guard {
            CatalogPersistence::save(catalog, &catalog_path)
                .wrap_err_with(|| format!("failed to save catalog to {:?}", catalog_path))?;
        }
        Ok(())
    }

    fn save_meta(&self) -> Result<()> {
        use crate::storage::{MetaFileHeader, PAGE_SIZE};
        use std::fs::OpenOptions;
        use std::io::{Read, Seek, SeekFrom, Write};
        use std::sync::atomic::Ordering;
        use zerocopy::IntoBytes;

        let meta_path = self.path.join("turdb.meta");

        let mut file = OpenOptions::new()
            .read(true)
            .write(true)
            .open(&meta_path)
            .wrap_err_with(|| format!("failed to open metadata file at {:?}", meta_path))?;

        let mut page = vec![0u8; PAGE_SIZE];
        file.read_exact(&mut page)
            .wrap_err("failed to read metadata page")?;

        let header = MetaFileHeader::from_bytes(&page)?;
        let mut new_header = *header;
        new_header.set_next_table_id(self.next_table_id.load(Ordering::Acquire));
        new_header.set_next_index_id(self.next_index_id.load(Ordering::Acquire));

        page[..128].copy_from_slice(new_header.as_bytes());

        file.seek(SeekFrom::Start(0))
            .wrap_err("failed to seek to start of metadata file")?;
        file.write_all(&page)
            .wrap_err("failed to write metadata header")?;
        file.sync_all().wrap_err("failed to sync metadata file")?;

        Ok(())
    }

    fn allocate_table_id(&self) -> u64 {
        use std::sync::atomic::Ordering;
        self.next_table_id.fetch_add(1, Ordering::AcqRel)
    }

    fn allocate_index_id(&self) -> u64 {
        use std::sync::atomic::Ordering;
        self.next_index_id.fetch_add(1, Ordering::AcqRel)
    }

    pub fn query(&self, sql: &str) -> Result<Vec<Row>> {
        self.ensure_catalog()?;
        self.ensure_file_manager()?;

        let arena = Bump::new();

        let mut parser = Parser::new(sql, &arena);
        let stmt = parser
            .parse_statement()
            .wrap_err("failed to parse SQL statement")?;

        let catalog_guard = self.catalog.read();
        let catalog = catalog_guard.as_ref().unwrap();
        let planner = Planner::new(catalog, &arena);
        let physical_plan = planner
            .create_physical_plan(&stmt)
            .wrap_err("failed to create query plan")?;

        let mut file_manager_guard = self.file_manager.write();
        let file_manager = file_manager_guard.as_mut().unwrap();

        fn find_table_scan<'a>(
            op: &'a crate::sql::planner::PhysicalOperator<'a>,
        ) -> Option<&'a crate::sql::planner::PhysicalTableScan<'a>> {
            use crate::sql::planner::PhysicalOperator;
            match op {
                PhysicalOperator::TableScan(scan) => Some(scan),
                PhysicalOperator::FilterExec(filter) => find_table_scan(filter.input),
                PhysicalOperator::ProjectExec(project) => find_table_scan(project.input),
                PhysicalOperator::LimitExec(limit) => find_table_scan(limit.input),
                PhysicalOperator::SortExec(sort) => find_table_scan(sort.input),
                _ => None,
            }
        }

        fn find_projections<'a>(
            op: &'a crate::sql::planner::PhysicalOperator<'a>,
            table_def: &crate::schema::TableDef,
        ) -> Option<Vec<usize>> {
            use crate::sql::ast::Expr;
            use crate::sql::planner::PhysicalOperator;

            match op {
                PhysicalOperator::ProjectExec(project) => {
                    let mut indices = Vec::new();
                    for expr in project.expressions.iter() {
                        if let Expr::Column(col_ref) = expr {
                            for (idx, col) in table_def.columns().iter().enumerate() {
                                if col.name().eq_ignore_ascii_case(col_ref.column) {
                                    indices.push(idx);
                                    break;
                                }
                            }
                        }
                    }
                    if indices.is_empty() || indices.len() != project.expressions.len() {
                        None
                    } else {
                        Some(indices)
                    }
                }
                PhysicalOperator::FilterExec(filter) => find_projections(filter.input, table_def),
                PhysicalOperator::LimitExec(limit) => find_projections(limit.input, table_def),
                PhysicalOperator::SortExec(sort) => find_projections(sort.input, table_def),
                _ => None,
            }
        }

        let table_scan = find_table_scan(physical_plan.root);

        let rows = if let Some(scan) = table_scan {
            let schema_name = scan.schema.unwrap_or("root");
            let table_name = scan.table;

            let table_def = catalog
                .resolve_table(table_name)
                .wrap_err_with(|| format!("table '{}' not found", table_name))?;

            let column_types: Vec<_> = table_def.columns().iter().map(|c| c.data_type()).collect();

            let projections = find_projections(physical_plan.root, table_def);

            let storage = file_manager
                .table_data(schema_name, table_name)
                .wrap_err_with(|| {
                    format!(
                        "failed to open table storage for {}.{}",
                        schema_name, table_name
                    )
                })?;

            let root_page = 1u32;
            let source = StreamingBTreeSource::from_btree_scan_with_projections(
                storage,
                root_page,
                column_types,
                projections,
            )
            .wrap_err("failed to create table scan")?;

            let ctx = ExecutionContext::new(&arena);
            let builder = ExecutorBuilder::new(&ctx);
            let mut executor = builder
                .build_with_source(&physical_plan, source)
                .wrap_err("failed to build executor")?;

            let mut rows = Vec::new();
            executor.open()?;
            while let Some(row) = executor.next()? {
                #[cfg(test)]
                eprintln!(
                    "DEBUG query: row from executor has {} values",
                    row.values.len()
                );
                let owned: Vec<OwnedValue> = row.values.iter().map(OwnedValue::from).collect();
                rows.push(Row::new(owned));
            }
            executor.close()?;
            rows
        } else {
            bail!("unsupported query plan type - only table scans currently supported")
        };

        Ok(rows)
    }

    pub fn execute(&self, sql: &str) -> Result<ExecuteResult> {
        let arena = Bump::new();

        let mut parser = Parser::new(sql, &arena);
        let stmt = parser
            .parse_statement()
            .wrap_err("failed to parse SQL statement")?;

        use crate::sql::ast::Statement;

        match stmt {
            Statement::CreateTable(create) => self.execute_create_table(create, &arena),
            Statement::CreateSchema(create) => self.execute_create_schema(create),
            Statement::CreateIndex(create) => self.execute_create_index(create, &arena),
            Statement::Insert(insert) => self.execute_insert(insert, &arena),
            Statement::Update(update) => self.execute_update(update, &arena),
            Statement::Delete(delete) => self.execute_delete(delete, &arena),
            Statement::Select(_) => {
                let rows = self.query(sql)?;
                Ok(ExecuteResult::Select { rows })
            }
            Statement::Drop(drop) => {
                use crate::sql::ast::ObjectType;
                match drop.object_type {
                    ObjectType::Table => self.execute_drop_table(drop),
                    _ => bail!("unsupported DROP statement type: {:?}", drop.object_type),
                }
            }
            Statement::Pragma(pragma) => self.execute_pragma(pragma),
            _ => bail!("unsupported statement type"),
        }
    }

    fn execute_create_table(
        &self,
        create: &crate::sql::ast::CreateTableStmt<'_>,
        _arena: &Bump,
    ) -> Result<ExecuteResult> {
        self.ensure_catalog()?;
        self.ensure_file_manager()?;

        let mut catalog_guard = self.catalog.write();
        let catalog = catalog_guard.as_mut().unwrap();

        let schema_name = create.schema.unwrap_or("root");
        let table_name = create.name;

        if catalog
            .get_schema(schema_name)
            .is_some_and(|s| s.table_exists(table_name))
        {
            if create.if_not_exists {
                return Ok(ExecuteResult::CreateTable { created: false });
            }
            bail!(
                "table '{}' already exists in schema '{}'",
                table_name,
                schema_name
            );
        }

        let columns: Vec<SchemaColumnDef> = create
            .columns
            .iter()
            .map(|col| {
                let data_type = Self::convert_data_type(&col.data_type);
                let mut column = SchemaColumnDef::new(col.name.to_string(), data_type);

                for constraint in col.constraints {
                    use crate::schema::table::Constraint as SchemaConstraint;
                    use crate::sql::ast::ColumnConstraint;

                    match constraint {
                        ColumnConstraint::NotNull => {
                            column = column.with_constraint(SchemaConstraint::NotNull);
                        }
                        ColumnConstraint::Unique => {
                            column = column.with_constraint(SchemaConstraint::Unique);
                        }
                        ColumnConstraint::PrimaryKey => {
                            column = column.with_constraint(SchemaConstraint::PrimaryKey);
                            column = column.with_constraint(SchemaConstraint::NotNull);
                        }
                        ColumnConstraint::Default(expr) => {
                            if let Some(default_str) = Self::expr_to_default_string(expr) {
                                column = column.with_default(default_str);
                            }
                        }
                        ColumnConstraint::Check(expr) => {
                            if let Some(check_str) = Self::expr_to_string(expr) {
                                column =
                                    column.with_constraint(SchemaConstraint::Check(check_str));
                            }
                        }
                        ColumnConstraint::References {
                            table,
                            column: ref_col,
                            ..
                        } => {
                            let fk_column = ref_col.unwrap_or(col.name);
                            column = column.with_constraint(SchemaConstraint::ForeignKey {
                                table: table.to_string(),
                                column: fk_column.to_string(),
                            });
                        }
                        ColumnConstraint::Null | ColumnConstraint::Generated { .. } => {}
                    }
                }
                column
            })
            .collect();

        let column_count = columns.len() as u32;
        let table_id = self.allocate_table_id();
        catalog.create_table_with_id(schema_name, table_name, columns, table_id)?;

        drop(catalog_guard);

        let mut file_manager_guard = self.file_manager.write();
        let file_manager = file_manager_guard.as_mut().unwrap();
        file_manager.create_table(schema_name, table_name, table_id, column_count)?;

        let storage = file_manager.table_data_mut(schema_name, table_name)?;
        storage.grow(2)?;
        crate::btree::BTree::create(storage, 1)?;

        self.save_catalog()?;
        self.save_meta()?;

        Ok(ExecuteResult::CreateTable { created: true })
    }

    fn execute_create_schema(
        &self,
        create: &crate::sql::ast::CreateSchemaStmt<'_>,
    ) -> Result<ExecuteResult> {
        self.ensure_catalog()?;

        let mut catalog_guard = self.catalog.write();
        let catalog = catalog_guard.as_mut().unwrap();

        if catalog.schema_exists(create.name) {
            if create.if_not_exists {
                return Ok(ExecuteResult::CreateSchema { created: false });
            }
            bail!("schema '{}' already exists", create.name);
        }

        catalog.create_schema(create.name)?;

        drop(catalog_guard);

        let schema_dir = self.path.join(create.name);
        std::fs::create_dir_all(&schema_dir)?;

        self.save_catalog()?;

        Ok(ExecuteResult::CreateSchema { created: true })
    }

    fn execute_create_index(
        &self,
        create: &crate::sql::ast::CreateIndexStmt<'_>,
        _arena: &Bump,
    ) -> Result<ExecuteResult> {
        use crate::sql::ast::Expr;

        self.ensure_catalog()?;
        self.ensure_file_manager()?;

        let schema_name = create.table.schema.unwrap_or("root");
        let table_name = create.table.name;
        let index_name = create.name;

        let columns: Vec<String> = create
            .columns
            .iter()
            .filter_map(|c| {
                if let Expr::Column(ref col) = c.expr {
                    Some(col.column.to_string())
                } else {
                    None
                }
            })
            .collect();
        let key_column_count = columns.len() as u32;

        let index_def = crate::schema::table::IndexDef::new(
            index_name.to_string(),
            columns,
            create.unique,
            crate::schema::table::IndexType::BTree,
        );

        let mut catalog_guard = self.catalog.write();
        let catalog = catalog_guard.as_mut().unwrap();

        if let Some(schema) = catalog.get_schema_mut(schema_name) {
            if let Some(table) = schema.get_table(table_name) {
                let table_with_index = table.clone().with_index(index_def);
                schema.remove_table(table_name);
                schema.add_table(table_with_index);
            }
        }

        let table_id = catalog.resolve_table(table_name)?.id();

        drop(catalog_guard);

        let index_id = self.allocate_index_id();

        let mut file_manager_guard = self.file_manager.write();
        let file_manager = file_manager_guard.as_mut().unwrap();
        file_manager.create_index(
            schema_name,
            table_name,
            index_name,
            index_id,
            table_id,
            key_column_count,
            create.unique,
        )?;

        self.save_catalog()?;
        self.save_meta()?;

        Ok(ExecuteResult::CreateIndex { created: true })
    }

    fn execute_insert(
        &self,
        insert: &crate::sql::ast::InsertStmt<'_>,
        _arena: &Bump,
    ) -> Result<ExecuteResult> {
        use crate::btree::BTree;
        use crate::constraints::ConstraintValidator;
        use crate::database::owned_value::create_record_schema;
        use std::sync::atomic::Ordering;

        self.ensure_catalog()?;
        self.ensure_file_manager()?;

        let wal_enabled = self.wal_enabled.load(Ordering::Acquire);
        if wal_enabled {
            self.ensure_wal()?;
        }

        let catalog_guard = self.catalog.read();
        let catalog = catalog_guard.as_ref().unwrap();

        let schema_name = insert.table.schema.unwrap_or("root");
        let table_name = insert.table.name;

        let table_def = catalog.resolve_table(table_name)?;
        let columns = table_def.columns().to_vec();
        let table_def_for_validator = table_def.clone();

        let schema = create_record_schema(&columns);

        let rows = match &insert.source {
            crate::sql::ast::InsertSource::Values(values) => values,
            _ => bail!("only VALUES insert supported"),
        };

        let root_page = 1u32;

        let column_types: Vec<crate::records::types::DataType> =
            columns.iter().map(|c| c.data_type()).collect();
        let validator = ConstraintValidator::new(&table_def_for_validator);

        use crate::schema::table::Constraint;
        let fk_constraints: Vec<(usize, String, String)> = columns
            .iter()
            .enumerate()
            .flat_map(|(idx, col)| {
                col.constraints().iter().filter_map(move |c| {
                    if let Constraint::ForeignKey { table, column } = c {
                        Some((idx, table.clone(), column.clone()))
                    } else {
                        None
                    }
                })
            })
            .collect();

        if !fk_constraints.is_empty() {
            let catalog_guard = self.catalog.read();
            let catalog = catalog_guard.as_ref().unwrap();

            let mut file_manager_guard = self.file_manager.write();
            let file_manager = file_manager_guard.as_mut().unwrap();

            for row_exprs in rows.iter() {
                let parsed_values: Vec<OwnedValue> = row_exprs
                    .iter()
                    .zip(column_types.iter())
                    .map(|(expr, data_type)| {
                        Database::eval_literal_with_type(expr, Some(data_type))
                    })
                    .collect::<Result<Vec<_>>>()?;

                for (col_idx, fk_table, fk_column) in &fk_constraints {
                    if let Some(value) = parsed_values.get(*col_idx) {
                        if value.is_null() {
                            continue;
                        }

                        let referenced_table = catalog.resolve_table(fk_table)?;
                        let ref_columns = referenced_table.columns();
                        let ref_col_idx = ref_columns
                            .iter()
                            .position(|c| c.name().eq_ignore_ascii_case(fk_column));

                        if ref_col_idx.is_none() {
                            bail!(
                                "FOREIGN KEY constraint: column '{}' not found in table '{}'",
                                fk_column,
                                fk_table
                            );
                        }

                        let ref_schema_name = "root";
                        let ref_storage =
                            file_manager.table_data_mut(ref_schema_name, fk_table)?;
                        let ref_btree = BTree::new(ref_storage, 1)?;
                        let ref_schema = create_record_schema(ref_columns);
                        let mut ref_cursor = ref_btree.cursor_first()?;

                        let mut found = false;
                        while ref_cursor.valid() {
                            let existing_value = ref_cursor.value()?;
                            let existing_record =
                                crate::records::RecordView::new(existing_value, &ref_schema)?;
                            let existing_values =
                                OwnedValue::extract_row_from_record(&existing_record, ref_columns)?;

                            if let Some(ref_val) = existing_values.get(ref_col_idx.unwrap()) {
                                if !ref_val.is_null() && ref_val == value {
                                    found = true;
                                    break;
                                }
                            }
                            ref_cursor.advance()?;
                        }

                        if !found {
                            bail!(
                                "FOREIGN KEY constraint violated on column '{}' in table '{}': referenced value not found in {}.{}",
                                columns[*col_idx].name(),
                                table_name,
                                fk_table,
                                fk_column
                            );
                        }
                    }
                }
            }

            drop(file_manager_guard);
        }

        let mut file_manager_guard = self.file_manager.write();
        let file_manager = file_manager_guard.as_mut().unwrap();
        let storage = file_manager.table_data_mut(schema_name, table_name)?;

        fn insert_rows<'a, S: crate::storage::Storage>(
            btree: &mut BTree<'_, S>,
            rows: &[&[&crate::sql::ast::Expr<'a>]],
            schema: &crate::records::Schema,
            column_types: &[crate::records::types::DataType],
            next_row_id: &std::sync::atomic::AtomicU64,
            validator: &ConstraintValidator<'_>,
            columns: &[SchemaColumnDef],
        ) -> Result<usize> {
            use crate::records::RecordView;
            use crate::schema::table::Constraint;

            let unique_col_indices: Vec<usize> = columns
                .iter()
                .enumerate()
                .filter(|(_, col)| {
                    col.has_constraint(&Constraint::Unique)
                        || col.has_constraint(&Constraint::PrimaryKey)
                })
                .map(|(idx, _)| idx)
                .collect();

            let mut count = 0;
            for row_exprs in rows.iter() {
                let mut values: Vec<OwnedValue> = row_exprs
                    .iter()
                    .zip(column_types.iter())
                    .map(|(expr, data_type)| {
                        Database::eval_literal_with_type(expr, Some(data_type))
                    })
                    .collect::<Result<Vec<_>>>()?;
                validator.validate_insert(&mut values)?;

                for (col_idx, col) in columns.iter().enumerate() {
                    for constraint in col.constraints() {
                        if let Constraint::Check(expr_str) = constraint {
                            let col_value = values.get(col_idx);
                            if !Database::evaluate_check_expression(
                                expr_str,
                                col.name(),
                                col_value,
                            ) {
                                bail!(
                                    "CHECK constraint violated on column '{}' in table '{}': {}",
                                    col.name(),
                                    validator.table().name(),
                                    expr_str
                                );
                            }
                        }
                    }
                }

                if !unique_col_indices.is_empty() {
                    let mut cursor = btree.cursor_first()?;
                    while cursor.valid() {
                        let existing_value = cursor.value()?;
                        let existing_record = RecordView::new(existing_value, schema)?;
                        let existing_values =
                            OwnedValue::extract_row_from_record(&existing_record, columns)?;

                        for &col_idx in &unique_col_indices {
                            let new_val = values.get(col_idx);
                            let existing_val = existing_values.get(col_idx);

                            if let (Some(new_v), Some(existing_v)) = (new_val, existing_val) {
                                if !new_v.is_null() && !existing_v.is_null() && new_v == existing_v {
                                    let col_name = &columns[col_idx].name();
                                    let constraint_type =
                                        if columns[col_idx].has_constraint(&Constraint::PrimaryKey) {
                                            "PRIMARY KEY"
                                        } else {
                                            "UNIQUE"
                                        };
                                    bail!(
                                        "{} constraint violated on column '{}' in table '{}': value already exists",
                                        constraint_type,
                                        col_name,
                                        validator.table().name()
                                    );
                                }
                            }
                        }
                        cursor.advance()?;
                    }
                }

                let record_data = OwnedValue::build_record_from_values(&values, schema)?;
                let row_id = next_row_id.fetch_add(1, Ordering::Relaxed);
                let key = Database::generate_row_key(row_id);
                btree.insert(&key, &record_data)?;
                count += 1;
            }
            Ok(count)
        }

        let rows_affected = if wal_enabled {
            let mut wal_storage = WalStorage::new(storage, &self.dirty_pages);
            let mut btree = BTree::new(&mut wal_storage, root_page)?;
            insert_rows(
                &mut btree,
                rows,
                &schema,
                &column_types,
                &self.next_row_id,
                &validator,
                &columns,
            )?
        } else {
            let mut btree = BTree::new(storage, root_page)?;
            insert_rows(
                &mut btree,
                rows,
                &schema,
                &column_types,
                &self.next_row_id,
                &validator,
                &columns,
            )?
        };

        Ok(ExecuteResult::Insert { rows_affected })
    }

    fn execute_update(
        &self,
        update: &crate::sql::ast::UpdateStmt<'_>,
        arena: &Bump,
    ) -> Result<ExecuteResult> {
        use crate::btree::BTree;
        use crate::database::owned_value::{
            create_column_map, create_record_schema, owned_values_to_values,
        };
        use crate::records::RecordView;

        self.ensure_catalog()?;
        self.ensure_file_manager()?;

        let catalog_guard = self.catalog.read();
        let catalog = catalog_guard.as_ref().unwrap();

        let schema_name = update.table.schema.unwrap_or("root");
        let table_name = update.table.name;

        let table_def = catalog.resolve_table(table_name)?.clone();
        let columns = table_def.columns().to_vec();

        drop(catalog_guard);

        let schema = create_record_schema(&columns);
        let column_map = create_column_map(&columns);

        let predicate = update
            .where_clause
            .map(|expr| CompiledPredicate::new(expr, column_map));

        let assignment_indices: Vec<(usize, &crate::sql::ast::Expr<'_>)> = update
            .assignments
            .iter()
            .filter_map(|a| {
                columns
                    .iter()
                    .position(|c| c.name().eq_ignore_ascii_case(a.column.column))
                    .map(|idx| (idx, a.value))
            })
            .collect();

        let mut file_manager_guard = self.file_manager.write();
        let file_manager = file_manager_guard.as_mut().unwrap();
        let storage = file_manager.table_data_mut(schema_name, table_name)?;

        let root_page = 1u32;
        let btree = BTree::new(storage, root_page)?;
        let mut cursor = btree.cursor_first()?;

        let mut rows_to_update: Vec<(Vec<u8>, Vec<OwnedValue>)> = Vec::new();

        while cursor.valid() {
            let key = cursor.key()?;
            let value = cursor.value()?;

            let record = RecordView::new(value, &schema)?;
            let mut row_values = OwnedValue::extract_row_from_record(&record, &columns)?;

            let should_update = if let Some(ref pred) = predicate {
                use crate::sql::executor::ExecutorRow;

                let values = owned_values_to_values(&row_values);
                let values_slice = arena.alloc_slice_fill_iter(values.into_iter());
                let exec_row = ExecutorRow::new(values_slice);
                pred.evaluate(&exec_row)
            } else {
                true
            };

            if should_update {
                for (col_idx, value_expr) in &assignment_indices {
                    let new_value = Self::eval_literal(value_expr)?;
                    row_values[*col_idx] = new_value;
                }

                let validator = crate::constraints::ConstraintValidator::new(&table_def);
                validator.validate_update(&row_values)?;

                for (col_idx, col) in columns.iter().enumerate() {
                    for constraint in col.constraints() {
                        if let crate::schema::table::Constraint::Check(expr_str) = constraint {
                            let col_value = row_values.get(col_idx);
                            if !Self::evaluate_check_expression(expr_str, col.name(), col_value) {
                                bail!(
                                    "CHECK constraint violated on column '{}' in table '{}': {}",
                                    col.name(),
                                    table_name,
                                    expr_str
                                );
                            }
                        }
                    }
                }

                rows_to_update.push((key.to_vec(), row_values));
            }

            cursor.advance()?;
        }

        use crate::schema::table::Constraint;

        let unique_col_indices: Vec<usize> = columns
            .iter()
            .enumerate()
            .filter(|(_, col)| {
                col.has_constraint(&Constraint::Unique)
                    || col.has_constraint(&Constraint::PrimaryKey)
            })
            .map(|(idx, _)| idx)
            .collect();

        if !unique_col_indices.is_empty() {
            let storage_for_check = file_manager.table_data_mut(schema_name, table_name)?;
            let btree_for_check = BTree::new(storage_for_check, root_page)?;
            let mut check_cursor = btree_for_check.cursor_first()?;

            for (update_key, updated_values) in &rows_to_update {
                while check_cursor.valid() {
                    let existing_key = check_cursor.key()?;

                    if existing_key != update_key.as_slice() {
                        let existing_value = check_cursor.value()?;
                        let existing_record = RecordView::new(existing_value, &schema)?;
                        let existing_values =
                            OwnedValue::extract_row_from_record(&existing_record, &columns)?;

                        for &col_idx in &unique_col_indices {
                            let new_val = updated_values.get(col_idx);
                            let existing_val = existing_values.get(col_idx);

                            if let (Some(new_v), Some(existing_v)) = (new_val, existing_val) {
                                if !new_v.is_null() && !existing_v.is_null() && new_v == existing_v
                                {
                                    let col_name = &columns[col_idx].name();
                                    bail!(
                                        "UNIQUE constraint violated on column '{}' in table '{}': value already exists",
                                        col_name,
                                        table_name
                                    );
                                }
                            }
                        }
                    }
                    check_cursor.advance()?;
                }
                check_cursor = btree_for_check.cursor_first()?;
            }
        }

        let rows_affected = rows_to_update.len();
        let storage = file_manager.table_data_mut(schema_name, table_name)?;
        let mut btree_mut = BTree::new(storage, root_page)?;

        for (key, updated_values) in rows_to_update {
            btree_mut.delete(&key)?;
            let record_data = OwnedValue::build_record_from_values(&updated_values, &schema)?;
            btree_mut.insert(&key, &record_data)?;
        }

        Ok(ExecuteResult::Update { rows_affected })
    }

    fn execute_delete(
        &self,
        delete: &crate::sql::ast::DeleteStmt<'_>,
        arena: &Bump,
    ) -> Result<ExecuteResult> {
        use crate::btree::BTree;
        use crate::database::owned_value::{
            create_column_map, create_record_schema, owned_values_to_values,
        };
        use crate::records::RecordView;
        use crate::schema::table::Constraint;

        self.ensure_catalog()?;
        self.ensure_file_manager()?;

        let catalog_guard = self.catalog.read();
        let catalog = catalog_guard.as_ref().unwrap();

        let schema_name = delete.table.schema.unwrap_or("root");
        let table_name = delete.table.name;

        let table_def = catalog.resolve_table(table_name)?;
        let columns = table_def.columns().to_vec();

        let mut fk_references: Vec<(String, String, String, usize)> = Vec::new();
        for (schema_key, schema_val) in catalog.schemas() {
            for (child_table_name, child_table_def) in schema_val.tables() {
                for col in child_table_def.columns().iter() {
                    for constraint in col.constraints() {
                        if let Constraint::ForeignKey { table, column } = constraint {
                            if table == table_name {
                                let ref_col_idx = columns
                                    .iter()
                                    .position(|c| c.name() == column)
                                    .unwrap_or(0);
                                fk_references.push((
                                    schema_key.clone(),
                                    child_table_name.clone(),
                                    col.name().to_string(),
                                    ref_col_idx,
                                ));
                            }
                        }
                    }
                }
            }
        }

        let child_table_schemas: Vec<(String, String, Vec<crate::schema::table::ColumnDef>, usize)> =
            fk_references
                .iter()
                .map(|(schema_key, child_name, fk_col_name, _ref_col_idx)| {
                    let child_def = catalog.schemas().get(schema_key).unwrap().tables().get(child_name).unwrap();
                    let fk_col_idx = child_def
                        .columns()
                        .iter()
                        .position(|c| c.name() == fk_col_name)
                        .unwrap_or(0);
                    (
                        schema_key.clone(),
                        child_name.clone(),
                        child_def.columns().to_vec(),
                        fk_col_idx,
                    )
                })
                .collect();

        drop(catalog_guard);

        let schema = create_record_schema(&columns);
        let column_map = create_column_map(&columns);

        let predicate = delete
            .where_clause
            .map(|expr| CompiledPredicate::new(expr, column_map));

        let mut file_manager_guard = self.file_manager.write();
        let file_manager = file_manager_guard.as_mut().unwrap();
        let storage = file_manager.table_data_mut(schema_name, table_name)?;

        let root_page = 1u32;
        let btree = BTree::new(storage, root_page)?;
        let mut cursor = btree.cursor_first()?;

        let mut keys_to_delete: Vec<Vec<u8>> = Vec::new();
        let mut values_to_check: Vec<(usize, OwnedValue)> = Vec::new();

        while cursor.valid() {
            let key = cursor.key()?;
            let value = cursor.value()?;

            let should_delete = if let Some(ref pred) = predicate {
                let record = RecordView::new(value, &schema)?;
                let row_values = OwnedValue::extract_row_from_record(&record, &columns)?;

                use crate::sql::executor::ExecutorRow;

                let values = owned_values_to_values(&row_values);
                let values_slice = arena.alloc_slice_fill_iter(values.into_iter());
                let exec_row = ExecutorRow::new(values_slice);
                pred.evaluate(&exec_row)
            } else {
                true
            };

            if should_delete {
                keys_to_delete.push(key.to_vec());

                if !fk_references.is_empty() {
                    let record = RecordView::new(value, &schema)?;
                    let row_values = OwnedValue::extract_row_from_record(&record, &columns)?;
                    for (_, _, _, ref_col_idx) in &fk_references {
                        if let Some(v) = row_values.get(*ref_col_idx) {
                            values_to_check.push((*ref_col_idx, v.clone()));
                        }
                    }
                }
            }

            cursor.advance()?;
        }

        if !values_to_check.is_empty() {
            for (child_schema, child_name, child_columns, fk_col_idx) in &child_table_schemas {
                let child_storage = file_manager.table_data_mut(child_schema, child_name)?;
                let child_btree = BTree::new(child_storage, root_page)?;
                let mut child_cursor = child_btree.cursor_first()?;
                let child_record_schema = create_record_schema(child_columns);

                while child_cursor.valid() {
                    let child_value = child_cursor.value()?;
                    let child_record = RecordView::new(child_value, &child_record_schema)?;
                    let child_row = OwnedValue::extract_row_from_record(&child_record, child_columns)?;

                    if let Some(child_fk_val) = child_row.get(*fk_col_idx) {
                        for (ref_col_idx, del_val) in &values_to_check {
                            if let Some((_, _, _, matching_ref_idx)) = fk_references.iter().find(|(s, n, _, r)| {
                                s == child_schema && n == child_name && r == ref_col_idx
                            }) {
                                if matching_ref_idx == ref_col_idx && child_fk_val == del_val {
                                    bail!(
                                        "FOREIGN KEY constraint violated: row in '{}' is still referenced by '{}'",
                                        table_name,
                                        child_name
                                    );
                                }
                            }
                        }
                    }

                    child_cursor.advance()?;
                }
            }
        }

        let rows_affected = keys_to_delete.len();
        let storage = file_manager.table_data_mut(schema_name, table_name)?;
        let mut btree_mut = BTree::new(storage, root_page)?;

        for key in keys_to_delete {
            btree_mut.delete(&key)?;
        }

        Ok(ExecuteResult::Delete { rows_affected })
    }

    fn execute_drop_table(
        &self,
        drop_stmt: &crate::sql::ast::DropStmt<'_>,
    ) -> Result<ExecuteResult> {
        self.ensure_catalog()?;
        self.ensure_file_manager()?;

        let mut catalog_guard = self.catalog.write();
        let catalog = catalog_guard.as_mut().unwrap();

        for table_ref in drop_stmt.names.iter() {
            let schema_name = table_ref.schema.unwrap_or("root");
            let table_name = table_ref.name;

            if let Some(schema) = catalog.get_schema_mut(schema_name) {
                if schema.table_exists(table_name) {
                    schema.remove_table(table_name);
                } else if !drop_stmt.if_exists {
                    bail!(
                        "table '{}' not found in schema '{}'",
                        table_name,
                        schema_name
                    );
                }
            } else if !drop_stmt.if_exists {
                bail!("schema '{}' not found", schema_name);
            }

            let mut file_manager_guard = self.file_manager.write();
            let file_manager = file_manager_guard.as_mut().unwrap();
            let _ = file_manager.drop_table(schema_name, table_name);
        }

        drop(catalog_guard);
        self.save_catalog()?;

        Ok(ExecuteResult::DropTable { dropped: true })
    }

    fn execute_pragma(&self, pragma: &crate::sql::ast::PragmaStmt<'_>) -> Result<ExecuteResult> {
        use std::sync::atomic::Ordering;

        let name = pragma.name.to_uppercase();
        let value = pragma.value.map(|v| v.to_uppercase());

        match name.as_str() {
            "WAL" => {
                if let Some(ref val) = value {
                    match val.as_str() {
                        "ON" | "TRUE" | "1" => {
                            self.wal_enabled.store(true, Ordering::Release);
                            self.ensure_wal()?;
                        }
                        "OFF" | "FALSE" | "0" => {
                            self.wal_enabled.store(false, Ordering::Release);
                        }
                        _ => bail!("invalid PRAGMA WAL value: {}", val),
                    }
                }
                let current = self.wal_enabled.load(Ordering::Acquire);
                Ok(ExecuteResult::Pragma {
                    name: name.clone(),
                    value: Some(if current {
                        "ON".to_string()
                    } else {
                        "OFF".to_string()
                    }),
                })
            }
            _ => bail!("unknown PRAGMA: {}", name),
        }
    }

    fn evaluate_check_expression(
        expr_str: &str,
        col_name: &str,
        col_value: Option<&OwnedValue>,
    ) -> bool {
        let Some(value) = col_value else {
            return true;
        };

        if value.is_null() {
            return true;
        }

        let expr_lower = expr_str.to_lowercase();
        let col_lower = col_name.to_lowercase();

        if expr_lower.contains(&col_lower) {
            if let Some(op_idx) = expr_str.find(">=") {
                let right_part = expr_str[op_idx + 2..].trim();
                if let Ok(threshold) = right_part.parse::<i64>() {
                    if let OwnedValue::Int(v) = value {
                        return *v >= threshold;
                    }
                }
                if let Ok(threshold) = right_part.parse::<f64>() {
                    if let OwnedValue::Float(v) = value {
                        return *v >= threshold;
                    }
                }
            } else if let Some(op_idx) = expr_str.find("<=") {
                let right_part = expr_str[op_idx + 2..].trim();
                if let Ok(threshold) = right_part.parse::<i64>() {
                    if let OwnedValue::Int(v) = value {
                        return *v <= threshold;
                    }
                }
                if let Ok(threshold) = right_part.parse::<f64>() {
                    if let OwnedValue::Float(v) = value {
                        return *v <= threshold;
                    }
                }
            } else if let Some(op_idx) = expr_str.find('>') {
                let right_part = expr_str[op_idx + 1..].trim();
                if let Ok(threshold) = right_part.parse::<i64>() {
                    if let OwnedValue::Int(v) = value {
                        return *v > threshold;
                    }
                }
                if let Ok(threshold) = right_part.parse::<f64>() {
                    if let OwnedValue::Float(v) = value {
                        return *v > threshold;
                    }
                }
            } else if let Some(op_idx) = expr_str.find('<') {
                let right_part = expr_str[op_idx + 1..].trim();
                if let Ok(threshold) = right_part.parse::<i64>() {
                    if let OwnedValue::Int(v) = value {
                        return *v < threshold;
                    }
                }
                if let Ok(threshold) = right_part.parse::<f64>() {
                    if let OwnedValue::Float(v) = value {
                        return *v < threshold;
                    }
                }
            }
        }

        true
    }

    fn convert_data_type(sql_type: &crate::sql::ast::DataType) -> crate::records::types::DataType {
        use crate::records::types::DataType;
        use crate::sql::ast::DataType as SqlType;

        match sql_type {
            SqlType::Integer | SqlType::BigInt => DataType::Int8,
            SqlType::SmallInt => DataType::Int2,
            SqlType::TinyInt => DataType::Int2,
            SqlType::Real | SqlType::DoublePrecision => DataType::Float8,
            SqlType::Decimal(_, _) | SqlType::Numeric(_, _) => DataType::Float8,
            SqlType::Varchar(_) | SqlType::Text => DataType::Text,
            SqlType::Char(_) => DataType::Text,
            SqlType::Blob => DataType::Blob,
            SqlType::Boolean => DataType::Bool,
            SqlType::Date => DataType::Date,
            SqlType::Time => DataType::Time,
            SqlType::Timestamp => DataType::Timestamp,
            SqlType::TimestampTz => DataType::TimestampTz,
            SqlType::Uuid => DataType::Uuid,
            SqlType::Json | SqlType::Jsonb => DataType::Jsonb,
            SqlType::Vector(_) => DataType::Vector,
            SqlType::Array(_) => DataType::Array,
            SqlType::Interval => DataType::Text,
            _ => DataType::Text,
        }
    }

    fn expr_to_default_string(expr: &crate::sql::ast::Expr<'_>) -> Option<String> {
        use crate::sql::ast::Expr;

        match expr {
            Expr::Literal(lit) => match lit {
                crate::sql::ast::Literal::Integer(n) => Some(n.to_string()),
                crate::sql::ast::Literal::Float(f) => Some(f.to_string()),
                crate::sql::ast::Literal::String(s) => Some(s.to_string()),
                crate::sql::ast::Literal::Boolean(b) => Some(b.to_string()),
                crate::sql::ast::Literal::Null => None,
                _ => None,
            },
            _ => None,
        }
    }

    fn expr_to_string(expr: &crate::sql::ast::Expr<'_>) -> Option<String> {
        use crate::sql::ast::{BinaryOperator, Expr};

        match expr {
            Expr::BinaryOp { left, op, right } => {
                let left_str = Self::expr_to_string(left)?;
                let right_str = Self::expr_to_string(right)?;
                let op_str = match op {
                    BinaryOperator::Plus => "+",
                    BinaryOperator::Minus => "-",
                    BinaryOperator::Multiply => "*",
                    BinaryOperator::Divide => "/",
                    BinaryOperator::Modulo => "%",
                    BinaryOperator::Eq => "=",
                    BinaryOperator::NotEq => "!=",
                    BinaryOperator::Lt => "<",
                    BinaryOperator::LtEq => "<=",
                    BinaryOperator::Gt => ">",
                    BinaryOperator::GtEq => ">=",
                    BinaryOperator::And => "AND",
                    BinaryOperator::Or => "OR",
                    _ => "?",
                };
                Some(format!("{} {} {}", left_str, op_str, right_str))
            }
            Expr::Column(col_ref) => Some(col_ref.column.to_string()),
            Expr::Literal(lit) => match lit {
                crate::sql::ast::Literal::Integer(n) => Some(n.to_string()),
                crate::sql::ast::Literal::Float(f) => Some(f.to_string()),
                crate::sql::ast::Literal::String(s) => Some(format!("'{}'", s)),
                crate::sql::ast::Literal::Boolean(b) => Some(b.to_string()),
                _ => None,
            },
            _ => None,
        }
    }

    fn eval_literal(expr: &crate::sql::ast::Expr<'_>) -> Result<OwnedValue> {
        Self::eval_literal_with_type(expr, None)
    }

    fn eval_literal_with_type(
        expr: &crate::sql::ast::Expr<'_>,
        target_type: Option<&crate::records::types::DataType>,
    ) -> Result<OwnedValue> {
        use crate::records::types::DataType;
        use crate::sql::ast::{Expr, Literal, UnaryOperator};

        match expr {
            Expr::Literal(lit) => match lit {
                Literal::Null => Ok(OwnedValue::Null),
                Literal::Integer(s) => {
                    let i: i64 = s
                        .parse()
                        .wrap_err_with(|| format!("failed to parse integer: {}", s))?;
                    Ok(OwnedValue::Int(i))
                }
                Literal::Float(s) => {
                    let f: f64 = s
                        .parse()
                        .wrap_err_with(|| format!("failed to parse float: {}", s))?;
                    Ok(OwnedValue::Float(f))
                }
                Literal::String(s) => match target_type {
                    Some(DataType::Uuid) => Self::parse_uuid_string(s),
                    Some(DataType::Jsonb) => Self::parse_json_string(s),
                    Some(DataType::Vector) => Self::parse_vector_string(s),
                    _ => Ok(OwnedValue::Text(s.to_string())),
                },
                Literal::Boolean(b) => Ok(OwnedValue::Bool(*b)),
                Literal::HexNumber(s) => Self::parse_hex_to_blob(s),
                Literal::BinaryNumber(s) => Self::parse_binary_to_blob(s),
            },
            Expr::UnaryOp { op, expr: inner } => {
                let inner_val = Self::eval_literal_with_type(inner, target_type)?;
                match (op, inner_val) {
                    (UnaryOperator::Minus, OwnedValue::Int(i)) => Ok(OwnedValue::Int(-i)),
                    (UnaryOperator::Minus, OwnedValue::Float(f)) => Ok(OwnedValue::Float(-f)),
                    (UnaryOperator::Plus, val) => Ok(val),
                    (UnaryOperator::Not, OwnedValue::Bool(b)) => Ok(OwnedValue::Bool(!b)),
                    _ => bail!("unsupported unary operation"),
                }
            }
            _ => bail!("expected literal expression, got {:?}", expr),
        }
    }

    fn parse_uuid_string(s: &str) -> Result<OwnedValue> {
        let s = s.trim();
        let hex_only: String = s.chars().filter(|c| *c != '-').collect();

        if hex_only.len() != 32 {
            bail!(
                "invalid UUID format '{}': expected 32 hex chars, got {}",
                s,
                hex_only.len()
            );
        }

        let mut bytes = [0u8; 16];
        for (i, chunk) in hex_only.as_bytes().chunks(2).enumerate() {
            let hex_pair = std::str::from_utf8(chunk)
                .wrap_err_with(|| format!("invalid UTF-8 in UUID hex: {:?}", chunk))?;
            bytes[i] = u8::from_str_radix(hex_pair, 16)
                .wrap_err_with(|| format!("invalid hex in UUID: '{}'", hex_pair))?;
        }

        Ok(OwnedValue::Uuid(bytes))
    }

    fn parse_json_string(s: &str) -> Result<OwnedValue> {
        let value = Self::parse_json_to_value(s.trim())?;
        let bytes = Self::jsonb_value_to_bytes(&value);
        Ok(OwnedValue::Jsonb(bytes))
    }

    fn parse_json_to_value(s: &str) -> Result<crate::records::jsonb::JsonbBuilderValue> {
        use crate::records::jsonb::JsonbBuilderValue;
        let s = s.trim();

        if s == "null" {
            Ok(JsonbBuilderValue::Null)
        } else if s == "true" {
            Ok(JsonbBuilderValue::Bool(true))
        } else if s == "false" {
            Ok(JsonbBuilderValue::Bool(false))
        } else if s.starts_with('"') && s.ends_with('"') {
            let inner = &s[1..s.len() - 1];
            let unescaped = Self::unescape_json_string(inner)?;
            Ok(JsonbBuilderValue::String(unescaped))
        } else if s.starts_with('{') && s.ends_with('}') {
            Self::parse_json_object_to_value(&s[1..s.len() - 1])
        } else if s.starts_with('[') && s.ends_with(']') {
            Self::parse_json_array_to_value(&s[1..s.len() - 1])
        } else if let Ok(n) = s.parse::<f64>() {
            Ok(JsonbBuilderValue::Number(n))
        } else {
            bail!("invalid JSON value: '{}'", s)
        }
    }

    fn jsonb_value_to_bytes(value: &crate::records::jsonb::JsonbBuilderValue) -> Vec<u8> {
        use crate::records::jsonb::{JsonbBuilder, JsonbBuilderValue};

        fn build_from_value(value: &JsonbBuilderValue) -> JsonbBuilder {
            match value {
                JsonbBuilderValue::Null => JsonbBuilder::new_null(),
                JsonbBuilderValue::Bool(b) => JsonbBuilder::new_bool(*b),
                JsonbBuilderValue::Number(n) => JsonbBuilder::new_number(*n),
                JsonbBuilderValue::String(s) => JsonbBuilder::new_string(s.clone()),
                JsonbBuilderValue::Array(elements) => {
                    let mut builder = JsonbBuilder::new_array();
                    for elem in elements {
                        builder.push(elem.clone());
                    }
                    builder
                }
                JsonbBuilderValue::Object(entries) => {
                    let mut builder = JsonbBuilder::new_object();
                    for (key, val) in entries {
                        builder.set(key.clone(), val.clone());
                    }
                    builder
                }
            }
        }

        build_from_value(value).build()
    }

    fn unescape_json_string(s: &str) -> Result<String> {
        let mut result = String::with_capacity(s.len());
        let mut chars = s.chars().peekable();

        while let Some(c) = chars.next() {
            if c == '\\' {
                match chars.next() {
                    Some('n') => result.push('\n'),
                    Some('r') => result.push('\r'),
                    Some('t') => result.push('\t'),
                    Some('\\') => result.push('\\'),
                    Some('"') => result.push('"'),
                    Some('/') => result.push('/'),
                    Some('u') => {
                        let hex: String = chars.by_ref().take(4).collect();
                        if hex.len() != 4 {
                            bail!("invalid unicode escape in JSON string");
                        }
                        let cp =
                            u32::from_str_radix(&hex, 16).wrap_err("invalid unicode escape")?;
                        if let Some(ch) = char::from_u32(cp) {
                            result.push(ch);
                        } else {
                            bail!("invalid unicode codepoint: {}", cp);
                        }
                    }
                    Some(other) => bail!("invalid escape sequence: \\{}", other),
                    None => bail!("unexpected end of string after escape"),
                }
            } else {
                result.push(c);
            }
        }

        Ok(result)
    }

    fn parse_json_object_to_value(s: &str) -> Result<crate::records::jsonb::JsonbBuilderValue> {
        use crate::records::jsonb::JsonbBuilderValue;
        let s = s.trim();

        if s.is_empty() {
            return Ok(JsonbBuilderValue::Object(Vec::new()));
        }

        let mut entries = Vec::new();
        let mut depth = 0;
        let mut in_string = false;
        let mut escape_next = false;
        let mut current_start = 0;

        for (i, c) in s.char_indices() {
            if escape_next {
                escape_next = false;
                continue;
            }

            match c {
                '\\' if in_string => escape_next = true,
                '"' => in_string = !in_string,
                '{' | '[' if !in_string => depth += 1,
                '}' | ']' if !in_string => depth -= 1,
                ',' if !in_string && depth == 0 => {
                    let (key, value) = Self::parse_json_kv_pair_to_value(&s[current_start..i])?;
                    entries.push((key, value));
                    current_start = i + 1;
                }
                _ => {}
            }
        }

        if current_start < s.len() {
            let (key, value) = Self::parse_json_kv_pair_to_value(&s[current_start..])?;
            entries.push((key, value));
        }

        Ok(JsonbBuilderValue::Object(entries))
    }

    fn parse_json_kv_pair_to_value(
        s: &str,
    ) -> Result<(String, crate::records::jsonb::JsonbBuilderValue)> {
        let s = s.trim();
        let colon_pos = Self::find_json_colon(s)?;

        let key_part = s[..colon_pos].trim();
        let value_part = s[colon_pos + 1..].trim();

        if !key_part.starts_with('"') || !key_part.ends_with('"') {
            bail!("JSON object key must be a string: '{}'", key_part);
        }

        let key = Self::unescape_json_string(&key_part[1..key_part.len() - 1])?;
        let value = Self::parse_json_to_value(value_part)?;

        Ok((key, value))
    }

    fn find_json_colon(s: &str) -> Result<usize> {
        let mut in_string = false;
        let mut escape_next = false;

        for (i, c) in s.char_indices() {
            if escape_next {
                escape_next = false;
                continue;
            }

            match c {
                '\\' if in_string => escape_next = true,
                '"' => in_string = !in_string,
                ':' if !in_string => return Ok(i),
                _ => {}
            }
        }

        bail!("no colon found in JSON key-value pair: '{}'", s)
    }

    fn parse_json_array_to_value(s: &str) -> Result<crate::records::jsonb::JsonbBuilderValue> {
        use crate::records::jsonb::JsonbBuilderValue;
        let s = s.trim();

        if s.is_empty() {
            return Ok(JsonbBuilderValue::Array(Vec::new()));
        }

        let mut elements = Vec::new();
        let mut depth = 0;
        let mut in_string = false;
        let mut escape_next = false;
        let mut current_start = 0;

        for (i, c) in s.char_indices() {
            if escape_next {
                escape_next = false;
                continue;
            }

            match c {
                '\\' if in_string => escape_next = true,
                '"' => in_string = !in_string,
                '{' | '[' if !in_string => depth += 1,
                '}' | ']' if !in_string => depth -= 1,
                ',' if !in_string && depth == 0 => {
                    elements.push(Self::parse_json_to_value(&s[current_start..i])?);
                    current_start = i + 1;
                }
                _ => {}
            }
        }

        if current_start < s.len() {
            elements.push(Self::parse_json_to_value(&s[current_start..])?);
        }

        Ok(JsonbBuilderValue::Array(elements))
    }

    fn parse_vector_string(s: &str) -> Result<OwnedValue> {
        let s = s.trim();

        let inner = if s.starts_with('[') && s.ends_with(']') {
            &s[1..s.len() - 1]
        } else {
            s
        };

        let values: Vec<f32> = inner
            .split(',')
            .map(|part| {
                part.trim()
                    .parse::<f32>()
                    .wrap_err_with(|| format!("failed to parse vector element: '{}'", part.trim()))
            })
            .collect::<Result<Vec<_>>>()?;

        Ok(OwnedValue::Vector(values))
    }

    fn parse_hex_to_blob(s: &str) -> Result<OwnedValue> {
        if !s.len().is_multiple_of(2) {
            bail!("hex string must have even length, got {}", s.len());
        }

        let bytes: Vec<u8> = (0..s.len())
            .step_by(2)
            .map(|i| {
                u8::from_str_radix(&s[i..i + 2], 16)
                    .wrap_err_with(|| format!("invalid hex byte: '{}'", &s[i..i + 2]))
            })
            .collect::<Result<Vec<_>>>()?;

        Ok(OwnedValue::Blob(bytes))
    }

    fn parse_binary_to_blob(s: &str) -> Result<OwnedValue> {
        if !s.len().is_multiple_of(8) {
            let bytes: Vec<u8> = (0..s.len())
                .step_by(8)
                .map(|i| {
                    let end = std::cmp::min(i + 8, s.len());
                    u8::from_str_radix(&s[i..end], 2)
                        .wrap_err_with(|| format!("invalid binary byte: '{}'", &s[i..end]))
                })
                .collect::<Result<Vec<_>>>()?;
            return Ok(OwnedValue::Blob(bytes));
        }

        let bytes: Vec<u8> = (0..s.len())
            .step_by(8)
            .map(|i| {
                u8::from_str_radix(&s[i..i + 8], 2)
                    .wrap_err_with(|| format!("invalid binary byte: '{}'", &s[i..i + 8]))
            })
            .collect::<Result<Vec<_>>>()?;

        Ok(OwnedValue::Blob(bytes))
    }

    fn generate_row_key(row_id: u64) -> Vec<u8> {
        row_id.to_be_bytes().to_vec()
    }

    pub fn checkpoint(&self) -> Result<CheckpointInfo> {
        self.checkpoint_table("root", "users")
    }

    pub fn checkpoint_table(&self, schema_name: &str, table_name: &str) -> Result<CheckpointInfo> {
        use std::sync::atomic::Ordering;

        if self.closed.load(Ordering::Acquire) {
            bail!("database is closed");
        }

        let dirty_count = self.dirty_pages.lock().len();
        if dirty_count == 0 {
            return Ok(CheckpointInfo {
                frames_checkpointed: 0,
                wal_truncated: false,
            });
        }

        self.ensure_file_manager()?;

        let mut wal_guard = self.wal.lock();
        let wal = match wal_guard.as_mut() {
            Some(w) => w,
            None => {
                self.dirty_pages.lock().clear();
                return Ok(CheckpointInfo {
                    frames_checkpointed: 0,
                    wal_truncated: false,
                });
            }
        };

        let mut file_manager_guard = self.file_manager.write();
        let file_manager = match file_manager_guard.as_mut() {
            Some(fm) => fm,
            None => {
                self.dirty_pages.lock().clear();
                return Ok(CheckpointInfo {
                    frames_checkpointed: 0,
                    wal_truncated: false,
                });
            }
        };

        let frames_written = if let Ok(storage) = file_manager.table_data(schema_name, table_name) {
            WalStorage::flush_wal(&self.dirty_pages, storage, wal)
                .wrap_err("failed to flush dirty pages to WAL")?
        } else {
            self.dirty_pages.lock().clear();
            0
        };

        let current_offset = wal.current_offset();
        let had_frames = current_offset > 0;

        if had_frames {
            wal.truncate()?;
        }

        Ok(CheckpointInfo {
            frames_checkpointed: frames_written,
            wal_truncated: had_frames,
        })
    }

    pub fn close(&self) -> Result<CheckpointInfo> {
        use std::sync::atomic::Ordering;

        if self.closed.load(Ordering::Acquire) {
            bail!("database already closed");
        }

        let checkpoint_result = self.checkpoint();

        self.closed.store(true, Ordering::Release);

        let mut wal_guard = self.wal.lock();
        *wal_guard = None;

        self.save_catalog()?;

        let mut file_manager_guard = self.file_manager.write();
        if let Some(ref mut file_manager) = *file_manager_guard {
            file_manager.sync_all()?;
        }

        checkpoint_result
    }

    pub fn is_closed(&self) -> bool {
        use std::sync::atomic::Ordering;
        self.closed.load(Ordering::Acquire)
    }

    pub fn path(&self) -> &Path {
        &self.path
    }
}

impl Drop for Database {
    fn drop(&mut self) {
        use std::sync::atomic::Ordering;

        if self.closed.load(Ordering::Acquire) {
            return;
        }

        let _ = self.checkpoint();

        let mut wal_guard = self.wal.lock();
        *wal_guard = None;

        let _ = self.save_catalog();

        if let Some(mut file_manager_guard) = self.file_manager.try_write() {
            if let Some(ref mut file_manager) = *file_manager_guard {
                let _ = file_manager.sync_all();
            }
        }
    }
}
