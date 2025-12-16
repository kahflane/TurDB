use crate::database::row::Row;
use crate::database::{CheckpointInfo, ExecuteResult, RecoveryInfo};
use crate::mvcc::{TransactionManager, TxnId, TxnState, WriteEntry};
use crate::parsing::{parse_binary_blob, parse_hex_blob, parse_interval, parse_uuid, parse_vector};
use crate::schema::{Catalog, ColumnDef as SchemaColumnDef};
use crate::sql::ast::IsolationLevel;
use crate::sql::builder::ExecutorBuilder;
use crate::sql::context::ExecutionContext;
use crate::sql::executor::{Executor, MaterializedRowSource, StreamingBTreeSource};
use crate::sql::planner::Planner;
use crate::sql::predicate::CompiledPredicate;
use crate::sql::Parser;
use crate::storage::{
    FileManager, MmapStorage, TableFileHeader, Wal, WalStorage, FILE_HEADER_SIZE,
};
use crate::types::{
    create_column_map, create_record_schema, owned_values_to_values, DataType, OwnedValue, Value,
};
use bumpalo::Bump;
use eyre::{bail, ensure, Result, WrapErr};
use hashbrown::HashSet;
use parking_lot::{Mutex, RwLock};
use smallvec::SmallVec;
use std::path::{Path, PathBuf};

fn convert_value_with_type(val: &Value<'_>, col_type: DataType) -> OwnedValue {
    match (val, col_type) {
        (Value::Int(i), DataType::Bool) => OwnedValue::Bool(*i != 0),
        (Value::Int(i), DataType::Date) => OwnedValue::Date(*i as i32),
        (Value::Int(i), DataType::Time) => OwnedValue::Time(*i),
        (Value::Int(i), DataType::Timestamp) => OwnedValue::Timestamp(*i),
        _ => OwnedValue::from(val),
    }
}

#[derive(Debug, Clone)]
pub struct Savepoint {
    pub name: String,
    pub write_entry_idx: usize,
}

#[derive(Debug)]
#[allow(dead_code)]
pub struct ActiveTransaction {
    pub txn_id: TxnId,
    pub slot_idx: usize,
    pub state: TxnState,
    pub isolation_level: Option<IsolationLevel>,
    pub read_only: bool,
    pub savepoints: SmallVec<[Savepoint; 4]>,
    pub write_entries: SmallVec<[WriteEntry; 16]>,
    pub undo_data: SmallVec<[Option<Vec<u8>>; 16]>,
}

impl ActiveTransaction {
    pub fn new(
        txn_id: TxnId,
        slot_idx: usize,
        isolation_level: Option<IsolationLevel>,
        read_only: bool,
    ) -> Self {
        Self {
            txn_id,
            slot_idx,
            state: TxnState::Active,
            isolation_level,
            read_only,
            savepoints: SmallVec::new(),
            write_entries: SmallVec::new(),
            undo_data: SmallVec::new(),
        }
    }

    pub fn create_savepoint(&mut self, name: String) {
        self.savepoints.push(Savepoint {
            name,
            write_entry_idx: self.write_entries.len(),
        });
    }

    pub fn find_savepoint(&self, name: &str) -> Option<usize> {
        self.savepoints.iter().position(|sp| sp.name == name)
    }

    pub fn add_write_entry(&mut self, entry: WriteEntry) {
        self.write_entries.push(entry);
        self.undo_data.push(None);
    }

    pub fn add_write_entry_with_undo(&mut self, entry: WriteEntry, undo: Vec<u8>) {
        self.write_entries.push(entry);
        self.undo_data.push(Some(undo));
    }

    pub fn rollback_to_savepoint(&mut self, idx: usize) -> (Vec<WriteEntry>, Vec<Option<Vec<u8>>>) {
        let savepoint = &self.savepoints[idx];
        let target_idx = savepoint.write_entry_idx;
        let entries_to_undo: Vec<WriteEntry> = self.write_entries.drain(target_idx..).collect();
        let undo_to_apply: Vec<Option<Vec<u8>>> = self.undo_data.drain(target_idx..).collect();
        self.savepoints.truncate(idx + 1);
        (entries_to_undo, undo_to_apply)
    }

    pub fn release_savepoint(&mut self, idx: usize) {
        self.savepoints.remove(idx);
    }

    #[allow(clippy::type_complexity)]
    pub fn take_write_entries(
        &mut self,
    ) -> (SmallVec<[WriteEntry; 16]>, SmallVec<[Option<Vec<u8>>; 16]>) {
        (
            std::mem::take(&mut self.write_entries),
            std::mem::take(&mut self.undo_data),
        )
    }
}

pub struct Database {
    path: PathBuf,
    file_manager: RwLock<Option<FileManager>>,
    pub(crate) catalog: RwLock<Option<Catalog>>,
    wal: Mutex<Option<Wal>>,
    wal_dir: PathBuf,
    next_row_id: std::sync::atomic::AtomicU64,
    next_table_id: std::sync::atomic::AtomicU64,
    next_index_id: std::sync::atomic::AtomicU64,
    closed: std::sync::atomic::AtomicBool,
    wal_enabled: std::sync::atomic::AtomicBool,
    dirty_pages: Mutex<HashSet<u32>>,
    txn_manager: TransactionManager,
    active_txn: Mutex<Option<ActiveTransaction>>,
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

        let segment_path = wal_dir.join("wal.000001");
        let wal_size_bytes = if segment_path.exists() {
            std::fs::metadata(&segment_path)
                .map(|m| m.len())
                .unwrap_or(0)
        } else {
            0
        };

        let frames_recovered = if wal_size_bytes > 0 {
            Self::recover_all_tables(&path, &wal_dir)?
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
            txn_manager: TransactionManager::new(),
            active_txn: Mutex::new(None),
        };

        let recovery_info = RecoveryInfo {
            frames_recovered,
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
            txn_manager: TransactionManager::new(),
            active_txn: Mutex::new(None),
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

    fn recover_all_tables(db_path: &Path, wal_dir: &Path) -> Result<u32> {
        use std::fs;

        let mut wal = Wal::open(wal_dir)
            .wrap_err_with(|| format!("failed to open WAL for recovery at {:?}", wal_dir))?;

        let mut total_frames = 0u32;

        for entry in fs::read_dir(db_path).wrap_err("failed to read database directory")? {
            let entry = entry.wrap_err("failed to read directory entry")?;
            let path = entry.path();

            if path.is_dir() {
                total_frames += Self::recover_schema_tables(&path, &mut wal)
                    .wrap_err_with(|| format!("failed to recover tables in schema {:?}", path))?;
            }
        }

        if total_frames > 0 {
            wal.truncate()
                .wrap_err("failed to truncate WAL after recovery")?;
        }

        Ok(total_frames)
    }

    fn recover_schema_tables(schema_path: &Path, wal: &mut Wal) -> Result<u32> {
        use std::fs;
        use std::io::Read;

        let mut total_frames = 0u32;

        for entry in fs::read_dir(schema_path).wrap_err("failed to read schema directory")? {
            let entry = entry.wrap_err("failed to read directory entry")?;
            let path = entry.path();

            if path.extension().map(|e| e == "tbd").unwrap_or(false) {
                let mut header_bytes = [0u8; FILE_HEADER_SIZE];
                let mut file = fs::File::open(&path).wrap_err_with(|| {
                    format!("failed to open table file {:?} for recovery", path)
                })?;

                if file
                    .read(&mut header_bytes)
                    .wrap_err("failed to read table header")?
                    < FILE_HEADER_SIZE
                {
                    continue;
                }

                let header = match TableFileHeader::from_bytes(&header_bytes) {
                    Ok(h) => h,
                    Err(_) => continue,
                };

                let table_id = header.table_id();
                if table_id == 0 {
                    continue;
                }

                let mut storage = MmapStorage::open(&path).wrap_err_with(|| {
                    format!("failed to open storage {:?} for WAL recovery", path)
                })?;

                let frames = wal
                    .recover_for_file(&mut storage, table_id)
                    .wrap_err_with(|| {
                        format!(
                            "failed to recover WAL frames for table_id={} from {:?}",
                            table_id, path
                        )
                    })?;

                if frames > 0 {
                    storage.sync().wrap_err_with(|| {
                        format!("failed to sync storage {:?} after recovery", path)
                    })?;
                }

                total_frames += frames;
            }
        }

        Ok(total_frames)
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
        use crate::sql::ast::{Distinct, Statement};

        self.ensure_catalog()?;
        self.ensure_file_manager()?;

        let arena = Bump::new();

        let mut parser = Parser::new(sql, &arena);
        let stmt = parser
            .parse_statement()
            .wrap_err("failed to parse SQL statement")?;

        let is_distinct =
            matches!(&stmt, Statement::Select(select) if select.distinct == Distinct::Distinct);

        let catalog_guard = self.catalog.read();
        let catalog = catalog_guard.as_ref().unwrap();
        let planner = Planner::new(catalog, &arena);
        let physical_plan = planner
            .create_physical_plan(&stmt)
            .wrap_err("failed to create query plan")?;

        let mut file_manager_guard = self.file_manager.write();
        let file_manager = file_manager_guard.as_mut().unwrap();

        enum PlanSource<'a> {
            TableScan(&'a crate::sql::planner::PhysicalTableScan<'a>),
            IndexScan(&'a crate::sql::planner::PhysicalIndexScan<'a>),
            Subquery(&'a crate::sql::planner::PhysicalSubqueryExec<'a>),
        }

        fn find_plan_source<'a>(
            op: &'a crate::sql::planner::PhysicalOperator<'a>,
        ) -> Option<PlanSource<'a>> {
            use crate::sql::planner::PhysicalOperator;
            match op {
                PhysicalOperator::TableScan(scan) => Some(PlanSource::TableScan(scan)),
                PhysicalOperator::IndexScan(scan) => Some(PlanSource::IndexScan(scan)),
                PhysicalOperator::SubqueryExec(subq) => Some(PlanSource::Subquery(subq)),
                PhysicalOperator::FilterExec(filter) => find_plan_source(filter.input),
                PhysicalOperator::ProjectExec(project) => find_plan_source(project.input),
                PhysicalOperator::LimitExec(limit) => find_plan_source(limit.input),
                PhysicalOperator::SortExec(sort) => find_plan_source(sort.input),
                PhysicalOperator::HashAggregate(agg) => find_plan_source(agg.input),
                PhysicalOperator::SortedAggregate(agg) => find_plan_source(agg.input),
                PhysicalOperator::WindowExec(window) => find_plan_source(window.input),
                _ => None,
            }
        }

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
                PhysicalOperator::HashAggregate(agg) => find_table_scan(agg.input),
                PhysicalOperator::SortedAggregate(agg) => find_table_scan(agg.input),
                PhysicalOperator::WindowExec(window) => find_table_scan(window.input),
                _ => None,
            }
        }

        fn has_filter<'a>(op: &'a crate::sql::planner::PhysicalOperator<'a>) -> bool {
            use crate::sql::planner::PhysicalOperator;
            match op {
                PhysicalOperator::FilterExec(_) => true,
                PhysicalOperator::ProjectExec(project) => has_filter(project.input),
                PhysicalOperator::LimitExec(limit) => has_filter(limit.input),
                PhysicalOperator::SortExec(sort) => has_filter(sort.input),
                PhysicalOperator::WindowExec(window) => has_filter(window.input),
                _ => false,
            }
        }

        fn has_aggregate<'a>(op: &'a crate::sql::planner::PhysicalOperator<'a>) -> bool {
            use crate::sql::planner::PhysicalOperator;
            match op {
                PhysicalOperator::HashAggregate(_) | PhysicalOperator::SortedAggregate(_) => true,
                PhysicalOperator::ProjectExec(project) => has_aggregate(project.input),
                PhysicalOperator::LimitExec(limit) => has_aggregate(limit.input),
                PhysicalOperator::SortExec(sort) => has_aggregate(sort.input),
                PhysicalOperator::FilterExec(filter) => has_aggregate(filter.input),
                PhysicalOperator::WindowExec(window) => has_aggregate(window.input),
                _ => false,
            }
        }

        fn has_window<'a>(op: &'a crate::sql::planner::PhysicalOperator<'a>) -> bool {
            use crate::sql::planner::PhysicalOperator;
            match op {
                PhysicalOperator::WindowExec(_) => true,
                PhysicalOperator::ProjectExec(project) => has_window(project.input),
                PhysicalOperator::LimitExec(limit) => has_window(limit.input),
                PhysicalOperator::SortExec(sort) => has_window(sort.input),
                PhysicalOperator::FilterExec(filter) => has_window(filter.input),
                _ => false,
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

        let plan_source = find_plan_source(physical_plan.root);

        let rows = match plan_source {
            Some(PlanSource::TableScan(scan)) => {
                let schema_name = scan.schema.unwrap_or("root");
                let table_name = scan.table;

                let table_def = catalog
                    .resolve_table(table_name)
                    .wrap_err_with(|| format!("table '{}' not found", table_name))?;

                let column_types: Vec<_> =
                    table_def.columns().iter().map(|c| c.data_type()).collect();

                let plan_has_filter = has_filter(physical_plan.root);
                let plan_has_aggregate = has_aggregate(physical_plan.root);
                let plan_has_window = has_window(physical_plan.root);
                let needs_all_columns = plan_has_filter || plan_has_aggregate || plan_has_window;
                let projections = if needs_all_columns {
                    None
                } else {
                    find_projections(physical_plan.root, table_def)
                };

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

                let all_columns_map: Vec<(String, usize)> = table_def
                    .columns()
                    .iter()
                    .enumerate()
                    .map(|(idx, col)| (col.name().to_lowercase(), idx))
                    .collect();

                let mut executor = if needs_all_columns {
                    builder
                        .build_with_source_and_column_map(&physical_plan, source, &all_columns_map)
                        .wrap_err("failed to build executor")?
                } else {
                    builder
                        .build_with_source(&physical_plan, source)
                        .wrap_err("failed to build executor")?
                };

                let output_columns = physical_plan.output_schema.columns;

                let mut rows = Vec::new();
                executor.open()?;
                while let Some(row) = executor.next()? {
                    let owned: Vec<OwnedValue> = row
                        .values
                        .iter()
                        .enumerate()
                        .map(|(idx, val)| {
                            let col_type = output_columns
                                .get(idx)
                                .map(|c| c.data_type)
                                .unwrap_or(DataType::Int8);
                            convert_value_with_type(val, col_type)
                        })
                        .collect();
                    rows.push(Row::new(owned));
                }
                executor.close()?;
                rows
            }
            Some(PlanSource::IndexScan(scan)) => {
                let schema_name = scan.schema.unwrap_or("root");
                let table_name = scan.table;

                let table_def = catalog
                    .resolve_table(table_name)
                    .wrap_err_with(|| format!("table '{}' not found", table_name))?;

                let column_types: Vec<_> =
                    table_def.columns().iter().map(|c| c.data_type()).collect();

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
                    None,
                )
                .wrap_err("failed to create table scan for index scan fallback")?;

                let ctx = ExecutionContext::new(&arena);
                let builder = ExecutorBuilder::new(&ctx);

                let all_columns_map: Vec<(String, usize)> = table_def
                    .columns()
                    .iter()
                    .enumerate()
                    .map(|(idx, col)| (col.name().to_lowercase(), idx))
                    .collect();

                let mut executor = builder
                    .build_with_source_and_column_map(&physical_plan, source, &all_columns_map)
                    .wrap_err("failed to build executor for index scan")?;

                let output_columns = physical_plan.output_schema.columns;

                let mut rows = Vec::new();
                executor.open()?;
                while let Some(row) = executor.next()? {
                    let owned: Vec<OwnedValue> = row
                        .values
                        .iter()
                        .enumerate()
                        .map(|(idx, val)| {
                            let col_type = output_columns
                                .get(idx)
                                .map(|c| c.data_type)
                                .unwrap_or(DataType::Int8);
                            convert_value_with_type(val, col_type)
                        })
                        .collect();
                    rows.push(Row::new(owned));
                }
                executor.close()?;
                rows
            }
            Some(PlanSource::Subquery(subq)) => {
                let inner_table_scan = find_table_scan(subq.child_plan);

                let inner_rows = if let Some(inner_scan) = inner_table_scan {
                    let schema_name = inner_scan.schema.unwrap_or("root");
                    let table_name = inner_scan.table;

                    let inner_table_def = catalog
                        .resolve_table(table_name)
                        .wrap_err_with(|| format!("table '{}' not found", table_name))?;

                    let column_types: Vec<_> = inner_table_def
                        .columns()
                        .iter()
                        .map(|c| c.data_type())
                        .collect();

                    let storage = file_manager
                        .table_data(schema_name, table_name)
                        .wrap_err_with(|| {
                            format!(
                                "failed to open table storage for {}.{}",
                                schema_name, table_name
                            )
                        })?;

                    let root_page = 1u32;
                    let inner_source = StreamingBTreeSource::from_btree_scan_with_projections(
                        storage,
                        root_page,
                        column_types,
                        None,
                    )
                    .wrap_err("failed to create inner table scan")?;

                    let inner_arena = Bump::new();
                    let inner_plan = crate::sql::planner::PhysicalPlan {
                        root: subq.child_plan,
                        output_schema: subq.output_schema.clone(),
                    };

                    let inner_ctx = ExecutionContext::new(&inner_arena);
                    let inner_builder = ExecutorBuilder::new(&inner_ctx);
                    let mut inner_executor = inner_builder
                        .build_with_source(&inner_plan, inner_source)
                        .wrap_err("failed to build inner executor")?;

                    let mut materialized_rows: Vec<Vec<OwnedValue>> = Vec::new();
                    inner_executor.open()?;
                    while let Some(row) = inner_executor.next()? {
                        let owned_values: Vec<OwnedValue> =
                            row.values.iter().map(OwnedValue::from).collect();
                        materialized_rows.push(owned_values);
                    }
                    inner_executor.close()?;
                    materialized_rows
                } else {
                    bail!("subquery inner plan must have a table scan")
                };

                let materialized_source = MaterializedRowSource::new(inner_rows);

                let ctx = ExecutionContext::new(&arena);
                let builder = ExecutorBuilder::new(&ctx);
                let mut executor = builder
                    .build_with_source(&physical_plan, materialized_source)
                    .wrap_err("failed to build executor with subquery source")?;

                let output_columns = physical_plan.output_schema.columns;

                let mut rows = Vec::new();
                executor.open()?;
                while let Some(row) = executor.next()? {
                    let owned: Vec<OwnedValue> = row
                        .values
                        .iter()
                        .enumerate()
                        .map(|(idx, val)| {
                            let col_type = output_columns
                                .get(idx)
                                .map(|c| c.data_type)
                                .unwrap_or(DataType::Int8);
                            convert_value_with_type(val, col_type)
                        })
                        .collect();
                    rows.push(Row::new(owned));
                }
                executor.close()?;
                rows
            }
            None => {
                bail!("unsupported query plan type - no table scan or subquery found")
            }
        };

        let rows = if is_distinct {
            let mut seen: std::collections::HashSet<Vec<u64>> = std::collections::HashSet::new();
            rows.into_iter()
                .filter(|row| {
                    let key: Vec<u64> = row
                        .values
                        .iter()
                        .map(|v| {
                            use std::hash::{Hash, Hasher};
                            let mut hasher = std::collections::hash_map::DefaultHasher::new();
                            format!("{:?}", v).hash(&mut hasher);
                            hasher.finish()
                        })
                        .collect();
                    seen.insert(key)
                })
                .collect()
        } else {
            rows
        };

        Ok(rows)
    }

    fn execute_select_internal(
        &self,
        select_stmt: &crate::sql::ast::SelectStmt<'_>,
    ) -> Result<Vec<Row>> {
        use crate::sql::ast::{Distinct, Statement};

        self.ensure_catalog()?;
        self.ensure_file_manager()?;

        let arena = Bump::new();
        let stmt = Statement::Select(select_stmt);

        let is_distinct = select_stmt.distinct == Distinct::Distinct;

        let catalog_guard = self.catalog.read();
        let catalog = catalog_guard.as_ref().unwrap();
        let planner = Planner::new(catalog, &arena);
        let physical_plan = planner
            .create_physical_plan(&stmt)
            .wrap_err("failed to create query plan for INSERT...SELECT")?;

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
                PhysicalOperator::HashAggregate(agg) => find_table_scan(agg.input),
                PhysicalOperator::SortedAggregate(agg) => find_table_scan(agg.input),
                PhysicalOperator::WindowExec(window) => find_table_scan(window.input),
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
                None,
            )
            .wrap_err("failed to create table scan for INSERT...SELECT")?;

            let output_column_types: Vec<DataType> = physical_plan
                .output_schema
                .columns
                .iter()
                .map(|c| c.data_type)
                .collect();

            let source_column_map: Vec<(String, usize)> = table_def
                .columns()
                .iter()
                .enumerate()
                .map(|(idx, col)| (col.name().to_string(), idx))
                .collect();

            let ctx = ExecutionContext::new(&arena);
            let builder = ExecutorBuilder::new(&ctx);
            let mut executor = builder
                .build_with_source_and_column_map(&physical_plan, source, &source_column_map)
                .wrap_err("failed to build executor for INSERT...SELECT")?;

            let mut rows = Vec::new();
            executor.open()?;
            while let Some(row) = executor.next()? {
                let owned: Vec<OwnedValue> = row
                    .values
                    .iter()
                    .enumerate()
                    .map(|(idx, val)| {
                        let col_type = output_column_types
                            .get(idx)
                            .copied()
                            .unwrap_or(DataType::Int8);
                        convert_value_with_type(val, col_type)
                    })
                    .collect();
                rows.push(Row::new(owned));
            }
            executor.close()?;
            rows
        } else {
            bail!("INSERT...SELECT query plan must have a table scan")
        };

        let rows = if is_distinct {
            let mut seen: std::collections::HashSet<Vec<u64>> = std::collections::HashSet::new();
            rows.into_iter()
                .filter(|row| {
                    let key: Vec<u64> = row
                        .values
                        .iter()
                        .map(|v| {
                            use std::hash::{Hash, Hasher};
                            let mut hasher = std::collections::hash_map::DefaultHasher::new();
                            format!("{:?}", v).hash(&mut hasher);
                            hasher.finish()
                        })
                        .collect();
                    seen.insert(key)
                })
                .collect()
        } else {
            rows
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
                    ObjectType::Index => self.execute_drop_index(drop),
                    ObjectType::Schema => self.execute_drop_schema_stmt(drop),
                    _ => bail!("unsupported DROP statement type: {:?}", drop.object_type),
                }
            }
            Statement::Pragma(pragma) => self.execute_pragma(pragma),
            Statement::Begin(begin) => self.execute_begin(begin),
            Statement::Commit => self.execute_commit(),
            Statement::Rollback(rollback) => self.execute_rollback(rollback),
            Statement::Savepoint(savepoint) => self.execute_savepoint(savepoint),
            Statement::Release(release) => self.execute_release(release),
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

        let mut unique_columns: Vec<(String, bool)> = Vec::new();

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
                            unique_columns.push((col.name.to_string(), false));
                        }
                        ColumnConstraint::PrimaryKey => {
                            column = column.with_constraint(SchemaConstraint::PrimaryKey);
                            column = column.with_constraint(SchemaConstraint::NotNull);
                            unique_columns.push((col.name.to_string(), true));
                        }
                        ColumnConstraint::Default(expr) => {
                            if let Some(default_str) = Self::expr_to_default_string(expr) {
                                column = column.with_default(default_str);
                            }
                        }
                        ColumnConstraint::Check(expr) => {
                            if let Some(check_str) = Self::expr_to_string(expr) {
                                column = column.with_constraint(SchemaConstraint::Check(check_str));
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

        for (col_name, is_primary_key) in &unique_columns {
            let index_name = if *is_primary_key {
                format!("{}_pkey", col_name)
            } else {
                format!("{}_key", col_name)
            };

            let index_id = self.allocate_index_id();
            file_manager.create_index(
                schema_name,
                table_name,
                &index_name,
                index_id,
                table_id,
                1,
                true,
            )?;

            let index_storage =
                file_manager.index_data_mut(schema_name, table_name, &index_name)?;
            index_storage.grow(2)?;
            crate::btree::BTree::create(index_storage, 1)?;

            let index_def = crate::schema::table::IndexDef::new(
                index_name.clone(),
                vec![col_name.clone()],
                true,
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
        }

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

    fn format_expr(expr: &crate::sql::ast::Expr<'_>) -> String {
        use crate::sql::ast::{Expr, FunctionArgs, Literal};

        match expr {
            Expr::Column(col) => {
                if let Some(table) = col.table {
                    format!("{}.{}", table, col.column)
                } else {
                    col.column.to_string()
                }
            }
            Expr::Literal(lit) => match lit {
                Literal::Null => "NULL".to_string(),
                Literal::Boolean(b) => {
                    if *b {
                        "TRUE".to_string()
                    } else {
                        "FALSE".to_string()
                    }
                }
                Literal::Integer(i) => i.to_string(),
                Literal::Float(f) => f.to_string(),
                Literal::String(s) => format!("'{}'", s),
                _ => format!("{:?}", lit),
            },
            Expr::Function(func) => {
                let name = func.name.name;
                let args = match func.args {
                    FunctionArgs::None => String::new(),
                    FunctionArgs::Star => "*".to_string(),
                    FunctionArgs::Args(args) => args
                        .iter()
                        .map(|arg| Self::format_expr(arg.value))
                        .collect::<Vec<_>>()
                        .join(", "),
                };
                format!("{}({})", name, args)
            }
            Expr::BinaryOp { left, op, right } => {
                format!(
                    "({} {:?} {})",
                    Self::format_expr(left),
                    op,
                    Self::format_expr(right)
                )
            }
            Expr::UnaryOp { op, expr } => {
                format!("{:?} {}", op, Self::format_expr(expr))
            }
            _ => format!("{:?}", expr),
        }
    }

    fn execute_create_index(
        &self,
        create: &crate::sql::ast::CreateIndexStmt<'_>,
        _arena: &Bump,
    ) -> Result<ExecuteResult> {
        use crate::schema::IndexColumnDef;
        use crate::sql::ast::Expr;

        self.ensure_catalog()?;
        self.ensure_file_manager()?;

        let schema_name = create.table.schema.unwrap_or("root");
        let table_name = create.table.name;
        let index_name = create.name;

        let column_defs: Vec<IndexColumnDef> = create
            .columns
            .iter()
            .map(|c| {
                if let Expr::Column(ref col) = c.expr {
                    IndexColumnDef::Column(col.column.to_string())
                } else {
                    IndexColumnDef::Expression(Self::format_expr(c.expr))
                }
            })
            .collect();
        let key_column_count = column_defs.len() as u32;

        let mut index_def = crate::schema::table::IndexDef::new_expression(
            index_name.to_string(),
            column_defs,
            create.unique,
            crate::schema::table::IndexType::BTree,
        );

        if let Some(where_clause) = create.where_clause {
            index_def = index_def.with_where_clause(Self::format_expr(where_clause));
        }

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

        let index_storage = file_manager.index_data_mut(schema_name, table_name, index_name)?;
        index_storage.grow(2)?;
        crate::btree::BTree::create(index_storage, 1)?;

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
        use crate::schema::table::Constraint;
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
        let table_id = table_def.id();
        let columns = table_def.columns().to_vec();
        let table_def_for_validator = table_def.clone();

        let unique_columns: Vec<(usize, String, bool)> = columns
            .iter()
            .enumerate()
            .filter_map(|(idx, col)| {
                let is_pk = col.has_constraint(&Constraint::PrimaryKey);
                let is_unique = col.has_constraint(&Constraint::Unique);
                if is_pk || is_unique {
                    let index_name = if is_pk {
                        format!("{}_pkey", col.name())
                    } else {
                        format!("{}_key", col.name())
                    };
                    Some((idx, index_name, is_pk))
                } else {
                    None
                }
            })
            .collect();

        let unique_column_index_names: HashSet<&str> = unique_columns
            .iter()
            .map(|(_, name, _)| name.as_str())
            .collect();

        let unique_indexes: Vec<(Vec<usize>, String)> = table_def
            .indexes()
            .iter()
            .filter(|idx| idx.is_unique())
            .filter(|idx| !unique_column_index_names.contains(idx.name()))
            .filter_map(|idx| {
                let col_indices: Vec<usize> = idx
                    .columns()
                    .iter()
                    .filter_map(|col_name| {
                        columns
                            .iter()
                            .position(|c| c.name().eq_ignore_ascii_case(col_name))
                    })
                    .collect();
                if col_indices.is_empty() {
                    None
                } else {
                    Some((col_indices, idx.name().to_string()))
                }
            })
            .collect();

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

        let schema = create_record_schema(&columns);

        let column_types: Vec<crate::records::types::DataType> =
            columns.iter().map(|c| c.data_type()).collect();

        let rows_to_insert: Vec<Vec<OwnedValue>> = match &insert.source {
            crate::sql::ast::InsertSource::Values(values) => {
                let mut result = Vec::with_capacity(values.len());
                for row_exprs in values.iter() {
                    let row: Vec<OwnedValue> = row_exprs
                        .iter()
                        .zip(column_types.iter())
                        .map(|(expr, data_type)| {
                            Database::eval_literal_with_type(expr, Some(data_type))
                        })
                        .collect::<Result<Vec<_>>>()?;
                    result.push(row);
                }
                result
            }
            crate::sql::ast::InsertSource::Select(select_stmt) => {
                drop(catalog_guard);
                let select_rows = self.execute_select_internal(select_stmt)?;
                select_rows.into_iter().map(|row| row.values).collect()
            }
            crate::sql::ast::InsertSource::Default => {
                bail!("DEFAULT VALUES insert not supported")
            }
        };

        let catalog_guard = self.catalog.read();
        let root_page = 1u32;
        let validator = ConstraintValidator::new(&table_def_for_validator);

        drop(catalog_guard);

        let mut file_manager_guard = self.file_manager.write();
        let file_manager = file_manager_guard.as_mut().unwrap();

        let mut count = 0;
        let mut key_buf: SmallVec<[u8; 64]> = SmallVec::new();
        let mut returned_rows: Option<Vec<Row>> = insert.returning.map(|_| Vec::new());

        for row_values in rows_to_insert.iter() {
            let mut values: Vec<OwnedValue> = row_values.clone();

            validator.validate_insert(&mut values)?;

            for (col_idx, col) in columns.iter().enumerate() {
                for constraint in col.constraints() {
                    if let Constraint::Check(expr_str) = constraint {
                        let col_value = values.get(col_idx);
                        if !Database::evaluate_check_expression(expr_str, col.name(), col_value) {
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

            if !fk_constraints.is_empty() {
                let catalog_guard = self.catalog.read();
                let catalog = catalog_guard.as_ref().unwrap();

                for (col_idx, fk_table, fk_column) in &fk_constraints {
                    if let Some(value) = values.get(*col_idx) {
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
                        let ref_storage = file_manager.table_data_mut(ref_schema_name, fk_table)?;
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

            let mut has_conflict = false;
            let mut conflicting_key: Option<Vec<u8>> = None;

            for (col_idx, index_name, is_pk) in &unique_columns {
                if let Some(value) = values.get(*col_idx) {
                    if value.is_null() {
                        continue;
                    }

                    if file_manager.index_exists(schema_name, table_name, index_name) {
                        let index_storage =
                            file_manager.index_data_mut(schema_name, table_name, index_name)?;
                        let index_btree = BTree::new(index_storage, root_page)?;

                        key_buf.clear();
                        Self::encode_value_as_key(value, &mut key_buf);

                        if let Some(handle) = index_btree.search(&key_buf)? {
                            if insert.on_conflict.is_some() {
                                has_conflict = true;
                                let row_key_bytes = index_btree.get_value(&handle)?;
                                conflicting_key = Some(row_key_bytes.to_vec());
                                break;
                            } else {
                                let constraint_type = if *is_pk { "PRIMARY KEY" } else { "UNIQUE" };
                                bail!(
                                    "{} constraint violated on column '{}' in table '{}': value already exists",
                                    constraint_type,
                                    columns[*col_idx].name(),
                                    table_name
                                );
                            }
                        }
                    }
                }
            }

            if !has_conflict {
                for (col_indices, index_name) in &unique_indexes {
                    let all_non_null = col_indices
                        .iter()
                        .all(|&idx| values.get(idx).map(|v| !v.is_null()).unwrap_or(false));

                    if all_non_null
                        && file_manager.index_exists(schema_name, table_name, index_name)
                    {
                        let index_storage =
                            file_manager.index_data_mut(schema_name, table_name, index_name)?;
                        let index_btree = BTree::new(index_storage, root_page)?;

                        key_buf.clear();
                        for &col_idx in col_indices {
                            if let Some(value) = values.get(col_idx) {
                                Self::encode_value_as_key(value, &mut key_buf);
                            }
                        }

                        if let Some(handle) = index_btree.search(&key_buf)? {
                            if insert.on_conflict.is_some() {
                                has_conflict = true;
                                let row_key_bytes = index_btree.get_value(&handle)?;
                                conflicting_key = Some(row_key_bytes.to_vec());
                                break;
                            } else {
                                let col_names: SmallVec<[&str; 8]> = col_indices
                                    .iter()
                                    .filter_map(|&idx| columns.get(idx).map(|c| c.name()))
                                    .collect();
                                bail!(
                                    "UNIQUE constraint violated on index '{}' (columns: {}) in table '{}': value already exists",
                                    index_name,
                                    col_names.join(", "),
                                    table_name
                                );
                            }
                        }
                    }
                }
            }

            if has_conflict {
                use crate::sql::ast::OnConflictAction;

                if let Some(on_conflict) = insert.on_conflict {
                    match on_conflict.action {
                        OnConflictAction::DoNothing => {
                            continue;
                        }
                        OnConflictAction::DoUpdate(assignments) => {
                            if let Some(existing_key) = conflicting_key {
                                let table_storage =
                                    file_manager.table_data_mut(schema_name, table_name)?;
                                let btree = BTree::new(table_storage, root_page)?;

                                if let Some(handle) = btree.search(&existing_key)? {
                                    let existing_value = btree.get_value(&handle)?;
                                    let record = crate::records::RecordView::new(existing_value, &schema)?;
                                    let mut existing_values =
                                        OwnedValue::extract_row_from_record(&record, &columns)?;

                                    for assignment in assignments.iter() {
                                        if let Some(col_idx) = columns
                                            .iter()
                                            .position(|c| {
                                                c.name()
                                                    .eq_ignore_ascii_case(assignment.column.column)
                                            })
                                        {
                                            let new_value = Self::eval_literal(assignment.value)?;
                                            existing_values[col_idx] = new_value;
                                        }
                                    }

                                    let updated_record =
                                        OwnedValue::build_record_from_values(&existing_values, &schema)?;

                                    let table_storage =
                                        file_manager.table_data_mut(schema_name, table_name)?;
                                    let mut btree_mut = BTree::new(table_storage, root_page)?;
                                    btree_mut.delete(&existing_key)?;
                                    btree_mut.insert(&existing_key, &updated_record)?;

                                    count += 1;

                                    if let Some(ref mut rows) = returned_rows {
                                        let returning_cols = insert.returning.unwrap();
                                        let row_values: Vec<OwnedValue> = returning_cols
                                            .iter()
                                            .flat_map(|col| match col {
                                                crate::sql::ast::SelectColumn::AllColumns => {
                                                    existing_values.clone()
                                                }
                                                crate::sql::ast::SelectColumn::TableAllColumns(_) => {
                                                    existing_values.clone()
                                                }
                                                crate::sql::ast::SelectColumn::Expr { expr, .. } => {
                                                    if let crate::sql::ast::Expr::Column(col_ref) =
                                                        expr
                                                    {
                                                        columns
                                                            .iter()
                                                            .position(|c| {
                                                                c.name().eq_ignore_ascii_case(
                                                                    col_ref.column,
                                                                )
                                                            })
                                                            .and_then(|idx| {
                                                                existing_values.get(idx).cloned()
                                                            })
                                                            .map(|v| vec![v])
                                                            .unwrap_or_default()
                                                    } else {
                                                        vec![]
                                                    }
                                                }
                                            })
                                            .collect();
                                        rows.push(Row::new(row_values));
                                    }
                                }
                            }
                            continue;
                        }
                    }
                }
            }

            let table_storage = file_manager.table_data_mut(schema_name, table_name)?;
            let row_id = self.next_row_id.fetch_add(1, Ordering::Relaxed);
            let row_key = Self::generate_row_key(row_id);
            let record_data = OwnedValue::build_record_from_values(&values, &schema)?;

            if wal_enabled {
                let mut wal_storage = WalStorage::new(table_storage, &self.dirty_pages);
                let mut btree = BTree::new(&mut wal_storage, root_page)?;
                btree.insert(&row_key, &record_data)?;
            } else {
                let mut btree = BTree::new(table_storage, root_page)?;
                btree.insert(&row_key, &record_data)?;
            }

            {
                let mut active_txn = self.active_txn.lock();
                if let Some(ref mut txn) = *active_txn {
                    txn.add_write_entry(WriteEntry {
                        table_id: table_id as u32,
                        key: row_key.clone(),
                        page_id: 0,
                        offset: 0,
                        undo_page_id: None,
                        undo_offset: None,
                        is_insert: true,
                    });
                }
            }

            for (col_idx, index_name, _) in &unique_columns {
                if let Some(value) = values.get(*col_idx) {
                    if value.is_null() {
                        continue;
                    }

                    if file_manager.index_exists(schema_name, table_name, index_name) {
                        let index_storage =
                            file_manager.index_data_mut(schema_name, table_name, index_name)?;

                        key_buf.clear();
                        Self::encode_value_as_key(value, &mut key_buf);

                        let row_id_bytes = row_id.to_be_bytes();

                        let mut index_btree = BTree::new(index_storage, root_page)?;
                        index_btree.insert(&key_buf, &row_id_bytes)?;
                    }
                }
            }

            for (col_indices, index_name) in &unique_indexes {
                let all_non_null = col_indices
                    .iter()
                    .all(|&idx| values.get(idx).map(|v| !v.is_null()).unwrap_or(false));

                if all_non_null && file_manager.index_exists(schema_name, table_name, index_name) {
                    let index_storage =
                        file_manager.index_data_mut(schema_name, table_name, index_name)?;

                    key_buf.clear();
                    for &col_idx in col_indices {
                        if let Some(value) = values.get(col_idx) {
                            Self::encode_value_as_key(value, &mut key_buf);
                        }
                    }

                    let row_id_bytes = row_id.to_be_bytes();

                    let mut index_btree = BTree::new(index_storage, root_page)?;
                    index_btree.insert(&key_buf, &row_id_bytes)?;
                }
            }

            count += 1;

            if let Some(ref mut rows) = returned_rows {
                let returning_cols = insert.returning.unwrap();
                let row_values: Vec<OwnedValue> = returning_cols
                    .iter()
                    .flat_map(|col| match col {
                        crate::sql::ast::SelectColumn::AllColumns => values.clone(),
                        crate::sql::ast::SelectColumn::TableAllColumns(_) => values.clone(),
                        crate::sql::ast::SelectColumn::Expr { expr, .. } => {
                            if let crate::sql::ast::Expr::Column(col_ref) = expr {
                                columns
                                    .iter()
                                    .position(|c| c.name().eq_ignore_ascii_case(col_ref.column))
                                    .and_then(|idx| values.get(idx).cloned())
                                    .map(|v| vec![v])
                                    .unwrap_or_default()
                            } else {
                                vec![]
                            }
                        }
                    })
                    .collect();
                rows.push(Row::new(row_values));
            }
        }

        Ok(ExecuteResult::Insert {
            rows_affected: count,
            returned: returned_rows,
        })
    }

    fn encode_value_as_key<B: crate::encoding::key::KeyBuffer>(value: &OwnedValue, buf: &mut B) {
        use crate::encoding::key;

        match value {
            OwnedValue::Null => key::encode_null(buf),
            OwnedValue::Bool(b) => key::encode_bool(*b, buf),
            OwnedValue::Int(n) => key::encode_int(*n, buf),
            OwnedValue::Float(f) => key::encode_float(*f, buf),
            OwnedValue::Text(s) => key::encode_text(s, buf),
            OwnedValue::Blob(b) => key::encode_blob(b, buf),
            OwnedValue::Uuid(u) => key::encode_uuid(u, buf),
            OwnedValue::Jsonb(j) => key::encode_blob(j, buf),
            OwnedValue::Vector(v) => {
                for f in v {
                    key::encode_float(*f as f64, buf);
                }
            }
            OwnedValue::Date(days) => key::encode_date(*days, buf),
            OwnedValue::Time(micros) => key::encode_time(*micros, buf),
            OwnedValue::Timestamp(micros) => key::encode_timestamp(*micros, buf),
            OwnedValue::TimestampTz(micros, tz_offset_mins) => {
                key::encode_timestamptz(*micros, *tz_offset_mins as i16, buf)
            }
            OwnedValue::Interval(micros, days, months) => {
                key::encode_interval(*months, *days, *micros, buf)
            }
            OwnedValue::MacAddr(m) => key::encode_blob(m, buf),
            OwnedValue::Inet4(a) => key::encode_blob(a, buf),
            OwnedValue::Inet6(a) => key::encode_blob(a, buf),
            OwnedValue::Point(x, y) => {
                key::encode_float(*x, buf);
                key::encode_float(*y, buf);
            }
            OwnedValue::Box((x1, y1), (x2, y2)) => {
                key::encode_float(*x1, buf);
                key::encode_float(*y1, buf);
                key::encode_float(*x2, buf);
                key::encode_float(*y2, buf);
            }
            OwnedValue::Circle((x, y), r) => {
                key::encode_float(*x, buf);
                key::encode_float(*y, buf);
                key::encode_float(*r, buf);
            }
            OwnedValue::Decimal(digits, scale) => {
                let divisor = 10i128.pow(*scale as u32);
                let float_val = *digits as f64 / divisor as f64;
                key::encode_float(float_val, buf);
            }
            OwnedValue::Enum(type_id, ordinal) => {
                key::encode_enum(*type_id as u32, *ordinal as u32, buf);
            }
        }
    }

    fn execute_update(
        &self,
        update: &crate::sql::ast::UpdateStmt<'_>,
        arena: &Bump,
    ) -> Result<ExecuteResult> {
        use crate::btree::BTree;
        use crate::records::RecordView;

        self.ensure_catalog()?;
        self.ensure_file_manager()?;

        let catalog_guard = self.catalog.read();
        let catalog = catalog_guard.as_ref().unwrap();

        let schema_name = update.table.schema.unwrap_or("root");
        let table_name = update.table.name;

        let table_def = catalog.resolve_table(table_name)?.clone();
        let table_id = table_def.id();
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

        let mut rows_to_update: Vec<(Vec<u8>, Vec<u8>, Vec<OwnedValue>)> = Vec::new();

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
                let old_value = value.to_vec();

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

                rows_to_update.push((key.to_vec(), old_value, row_values));
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

            for (update_key, _old_value, updated_values) in &rows_to_update {
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

        let returned_rows: Option<Vec<Row>> = update.returning.map(|returning_cols| {
            rows_to_update
                .iter()
                .map(|(_key, _old_value, updated_values)| {
                    let row_values: Vec<OwnedValue> = returning_cols
                        .iter()
                        .flat_map(|col| match col {
                            crate::sql::ast::SelectColumn::AllColumns => updated_values.clone(),
                            crate::sql::ast::SelectColumn::TableAllColumns(_) => {
                                updated_values.clone()
                            }
                            crate::sql::ast::SelectColumn::Expr { expr, .. } => {
                                if let crate::sql::ast::Expr::Column(col_ref) = expr {
                                    columns
                                        .iter()
                                        .position(|c| c.name().eq_ignore_ascii_case(col_ref.column))
                                        .and_then(|idx| updated_values.get(idx).cloned())
                                        .map(|v| vec![v])
                                        .unwrap_or_default()
                                } else {
                                    vec![]
                                }
                            }
                        })
                        .collect();
                    Row::new(row_values)
                })
                .collect()
        });

        let storage = file_manager.table_data_mut(schema_name, table_name)?;
        let mut btree_mut = BTree::new(storage, root_page)?;

        for (key, _old_value, updated_values) in &rows_to_update {
            btree_mut.delete(key)?;
            let record_data = OwnedValue::build_record_from_values(updated_values, &schema)?;
            btree_mut.insert(key, &record_data)?;
        }

        drop(file_manager_guard);

        {
            let mut active_txn = self.active_txn.lock();
            if let Some(ref mut txn) = *active_txn {
                for (key, old_value, _updated_values) in rows_to_update {
                    txn.add_write_entry_with_undo(
                        WriteEntry {
                            table_id: table_id as u32,
                            key,
                            page_id: 0,
                            offset: 0,
                            undo_page_id: None,
                            undo_offset: None,
                            is_insert: false,
                        },
                        old_value,
                    );
                }
            }
        }

        Ok(ExecuteResult::Update {
            rows_affected,
            returned: returned_rows,
        })
    }

    fn execute_delete(
        &self,
        delete: &crate::sql::ast::DeleteStmt<'_>,
        arena: &Bump,
    ) -> Result<ExecuteResult> {
        use crate::btree::BTree;
        use crate::records::RecordView;
        use crate::schema::table::Constraint;

        self.ensure_catalog()?;
        self.ensure_file_manager()?;

        let catalog_guard = self.catalog.read();
        let catalog = catalog_guard.as_ref().unwrap();

        let schema_name = delete.table.schema.unwrap_or("root");
        let table_name = delete.table.name;

        let table_def = catalog.resolve_table(table_name)?;
        let table_id = table_def.id();
        let columns = table_def.columns().to_vec();

        let mut fk_references: Vec<(String, String, String, usize)> = Vec::new();
        for (schema_key, schema_val) in catalog.schemas() {
            for (child_table_name, child_table_def) in schema_val.tables() {
                for col in child_table_def.columns().iter() {
                    for constraint in col.constraints() {
                        if let Constraint::ForeignKey { table, column } = constraint {
                            if table == table_name {
                                let ref_col_idx =
                                    columns.iter().position(|c| c.name() == column).unwrap_or(0);
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

        let child_table_schemas: Vec<(
            String,
            String,
            Vec<crate::schema::table::ColumnDef>,
            usize,
        )> = fk_references
            .iter()
            .map(|(schema_key, child_name, fk_col_name, _ref_col_idx)| {
                let child_def = catalog
                    .schemas()
                    .get(schema_key)
                    .unwrap()
                    .tables()
                    .get(child_name)
                    .unwrap();
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

        let mut rows_to_delete: Vec<(Vec<u8>, Vec<u8>, Vec<OwnedValue>)> = Vec::new();
        let mut values_to_check: Vec<(usize, OwnedValue)> = Vec::new();

        while cursor.valid() {
            let key = cursor.key()?;
            let value = cursor.value()?;

            let record = RecordView::new(value, &schema)?;
            let row_values = OwnedValue::extract_row_from_record(&record, &columns)?;

            let should_delete = if let Some(ref pred) = predicate {
                use crate::sql::executor::ExecutorRow;

                let values = owned_values_to_values(&row_values);
                let values_slice = arena.alloc_slice_fill_iter(values.into_iter());
                let exec_row = ExecutorRow::new(values_slice);
                pred.evaluate(&exec_row)
            } else {
                true
            };

            if should_delete {
                if !fk_references.is_empty() {
                    for (_, _, _, ref_col_idx) in &fk_references {
                        if let Some(v) = row_values.get(*ref_col_idx) {
                            values_to_check.push((*ref_col_idx, v.clone()));
                        }
                    }
                }
                rows_to_delete.push((key.to_vec(), value.to_vec(), row_values));
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
                    let child_row =
                        OwnedValue::extract_row_from_record(&child_record, child_columns)?;

                    if let Some(child_fk_val) = child_row.get(*fk_col_idx) {
                        for (ref_col_idx, del_val) in &values_to_check {
                            if let Some((_, _, _, matching_ref_idx)) =
                                fk_references.iter().find(|(s, n, _, r)| {
                                    s == child_schema && n == child_name && r == ref_col_idx
                                })
                            {
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

        let rows_affected = rows_to_delete.len();

        let returned_rows: Option<Vec<Row>> = if let Some(returning_cols) = delete.returning {
            let mut rows = Vec::with_capacity(rows_to_delete.len());
            for (_key, _old_value, deleted_values) in &rows_to_delete {
                let row_values: Vec<OwnedValue> = returning_cols
                    .iter()
                    .flat_map(|col| match col {
                        crate::sql::ast::SelectColumn::AllColumns => deleted_values.clone(),
                        crate::sql::ast::SelectColumn::TableAllColumns(_) => deleted_values.clone(),
                        crate::sql::ast::SelectColumn::Expr { expr, .. } => {
                            if let crate::sql::ast::Expr::Column(col_ref) = expr {
                                columns
                                    .iter()
                                    .position(|c| c.name().eq_ignore_ascii_case(col_ref.column))
                                    .and_then(|idx| deleted_values.get(idx).cloned())
                                    .map(|v| vec![v])
                                    .unwrap_or_default()
                            } else {
                                vec![]
                            }
                        }
                    })
                    .collect();
                rows.push(Row::new(row_values));
            }
            Some(rows)
        } else {
            None
        };

        let storage = file_manager.table_data_mut(schema_name, table_name)?;
        let mut btree_mut = BTree::new(storage, root_page)?;

        for (key, _old_value, _row_values) in &rows_to_delete {
            btree_mut.delete(key)?;
        }

        drop(file_manager_guard);

        {
            let mut active_txn = self.active_txn.lock();
            if let Some(ref mut txn) = *active_txn {
                for (key, old_value, _row_values) in rows_to_delete {
                    txn.add_write_entry_with_undo(
                        WriteEntry {
                            table_id: table_id as u32,
                            key,
                            page_id: 0,
                            offset: 0,
                            undo_page_id: None,
                            undo_offset: None,
                            is_insert: false,
                        },
                        old_value,
                    );
                }
            }
        }

        Ok(ExecuteResult::Delete {
            rows_affected,
            returned: returned_rows,
        })
    }

    fn execute_drop_table(
        &self,
        drop_stmt: &crate::sql::ast::DropStmt<'_>,
    ) -> Result<ExecuteResult> {
        self.ensure_catalog()?;
        self.ensure_file_manager()?;

        let mut catalog_guard = self.catalog.write();
        let catalog = catalog_guard.as_mut().unwrap();

        let mut actually_dropped = false;

        for table_ref in drop_stmt.names.iter() {
            let schema_name = table_ref.schema.unwrap_or("root");
            let table_name = table_ref.name;

            if let Some(schema) = catalog.get_schema_mut(schema_name) {
                if schema.table_exists(table_name) {
                    schema.remove_table(table_name);
                    actually_dropped = true;
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

            if actually_dropped {
                let mut file_manager_guard = self.file_manager.write();
                let file_manager = file_manager_guard.as_mut().unwrap();
                let _ = file_manager.drop_table(schema_name, table_name);
            }
        }

        drop(catalog_guard);
        if actually_dropped {
            self.save_catalog()?;
        }

        Ok(ExecuteResult::DropTable {
            dropped: actually_dropped,
        })
    }

    fn execute_drop_index(
        &self,
        drop_stmt: &crate::sql::ast::DropStmt<'_>,
    ) -> Result<ExecuteResult> {
        self.ensure_catalog()?;

        let mut catalog_guard = self.catalog.write();
        let catalog = catalog_guard.as_mut().unwrap();

        for index_ref in drop_stmt.names.iter() {
            let index_name = index_ref.name;

            if catalog.find_index(index_name).is_some() {
                catalog.remove_index(index_name)?;
            } else if !drop_stmt.if_exists {
                bail!("index '{}' not found", index_name);
            }
        }

        drop(catalog_guard);
        self.save_catalog()?;

        Ok(ExecuteResult::DropIndex { dropped: true })
    }

    fn execute_drop_schema_stmt(
        &self,
        drop_stmt: &crate::sql::ast::DropStmt<'_>,
    ) -> Result<ExecuteResult> {
        self.ensure_catalog()?;
        self.ensure_file_manager()?;

        let mut catalog_guard = self.catalog.write();
        let catalog = catalog_guard.as_mut().unwrap();

        for schema_ref in drop_stmt.names.iter() {
            let schema_name = schema_ref.name;

            if catalog.schema_exists(schema_name) {
                catalog.drop_schema(schema_name)?;
            } else if !drop_stmt.if_exists {
                bail!("schema '{}' not found", schema_name);
            }
        }

        drop(catalog_guard);
        self.save_catalog()?;

        Ok(ExecuteResult::DropSchema { dropped: true })
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

    fn execute_begin(&self, begin: &crate::sql::ast::BeginStmt) -> Result<ExecuteResult> {
        let mut active_txn = self.active_txn.lock();
        if active_txn.is_some() {
            bail!("transaction already in progress, use SAVEPOINT for nested transactions");
        }

        let mvcc_txn = self
            .txn_manager
            .begin_txn()
            .wrap_err("failed to begin MVCC transaction")?;

        let read_only = begin.read_only.unwrap_or(false);
        *active_txn = Some(ActiveTransaction::new(
            mvcc_txn.id(),
            mvcc_txn.slot_idx(),
            begin.isolation_level,
            read_only,
        ));

        mvcc_txn.commit();

        Ok(ExecuteResult::Begin)
    }

    fn execute_commit(&self) -> Result<ExecuteResult> {
        let mut active_txn = self.active_txn.lock();
        let txn = active_txn
            .take()
            .ok_or_else(|| eyre::eyre!("no transaction in progress"))?;

        self.finalize_transaction_commit(txn)?;

        Ok(ExecuteResult::Commit)
    }

    fn finalize_transaction_commit(&self, mut txn: ActiveTransaction) -> Result<()> {
        let (write_entries, _undo_data) = txn.take_write_entries();

        for entry in write_entries.iter() {
            self.finalize_write_entry_commit(entry)?;
        }

        Ok(())
    }

    fn finalize_write_entry_commit(&self, _entry: &WriteEntry) -> Result<()> {
        Ok(())
    }

    fn execute_rollback(
        &self,
        rollback: &crate::sql::ast::RollbackStmt<'_>,
    ) -> Result<ExecuteResult> {
        let mut active_txn = self.active_txn.lock();

        if let Some(savepoint_name) = rollback.savepoint {
            let txn = active_txn
                .as_mut()
                .ok_or_else(|| eyre::eyre!("no transaction in progress"))?;

            let sp_idx = txn
                .find_savepoint(savepoint_name)
                .ok_or_else(|| eyre::eyre!("savepoint '{}' does not exist", savepoint_name))?;

            let (entries_to_undo, undo_data) = txn.rollback_to_savepoint(sp_idx);

            drop(active_txn);
            self.undo_write_entries(&entries_to_undo, &undo_data)?;

            return Ok(ExecuteResult::Rollback);
        }

        let txn = active_txn
            .take()
            .ok_or_else(|| eyre::eyre!("no transaction in progress"))?;

        let write_entries: Vec<WriteEntry> = txn.write_entries.iter().cloned().collect();
        let undo_data: Vec<Option<Vec<u8>>> = txn.undo_data.iter().cloned().collect();

        drop(active_txn);
        self.undo_write_entries(&write_entries, &undo_data)?;

        Ok(ExecuteResult::Rollback)
    }

    fn undo_write_entries(
        &self,
        entries: &[WriteEntry],
        undo_data: &[Option<Vec<u8>>],
    ) -> Result<()> {
        for (i, entry) in entries.iter().enumerate().rev() {
            let undo = undo_data.get(i).and_then(|o| o.as_ref());
            self.undo_write_entry(entry, undo)?;
        }
        Ok(())
    }

    fn undo_write_entry(&self, entry: &WriteEntry, undo_data: Option<&Vec<u8>>) -> Result<()> {
        self.ensure_file_manager()?;

        let mut file_manager_guard = self.file_manager.write();
        let file_manager = file_manager_guard.as_mut().unwrap();

        let catalog_guard = self.catalog.read();
        let catalog = catalog_guard.as_ref().unwrap();

        let table_id = entry.table_id;
        let table_def = catalog.table_by_id(table_id as u64);

        if table_def.is_none() {
            return Ok(());
        }

        let table_def = table_def.unwrap();
        let schema_name = "root";
        let table_name = table_def.name();

        let table_storage = file_manager.table_data_mut(schema_name, table_name)?;

        use crate::btree::BTree;
        let mut btree = BTree::new(table_storage, 1)?;

        if entry.is_insert {
            btree.delete(&entry.key)?;
        } else if let Some(old_value) = undo_data {
            btree.delete(&entry.key)?;
            btree.insert(&entry.key, old_value)?;
        }

        Ok(())
    }

    fn execute_savepoint(
        &self,
        savepoint: &crate::sql::ast::SavepointStmt<'_>,
    ) -> Result<ExecuteResult> {
        let mut active_txn = self.active_txn.lock();
        let txn = active_txn
            .as_mut()
            .ok_or_else(|| eyre::eyre!("no transaction in progress"))?;

        let name = savepoint.name.to_string();
        txn.create_savepoint(name.clone());
        Ok(ExecuteResult::Savepoint { name })
    }

    fn execute_release(&self, release: &crate::sql::ast::ReleaseStmt<'_>) -> Result<ExecuteResult> {
        let mut active_txn = self.active_txn.lock();
        let txn = active_txn
            .as_mut()
            .ok_or_else(|| eyre::eyre!("no transaction in progress"))?;

        let sp_idx = txn
            .find_savepoint(release.name)
            .ok_or_else(|| eyre::eyre!("savepoint '{}' does not exist", release.name))?;

        txn.release_savepoint(sp_idx);
        Ok(ExecuteResult::Release {
            name: release.name.to_string(),
        })
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
            SqlType::Interval => DataType::Interval,
            SqlType::Point => DataType::Point,
            SqlType::Box => DataType::Box,
            SqlType::Circle => DataType::Circle,
            SqlType::MacAddr => DataType::MacAddr,
            SqlType::Inet => DataType::Inet6,
            SqlType::Int4Range => DataType::Int4Range,
            SqlType::Int8Range => DataType::Int8Range,
            SqlType::DateRange => DataType::DateRange,
            SqlType::TsRange => DataType::TimestampRange,
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
                    Some(DataType::Uuid) => parse_uuid(s),
                    Some(DataType::Jsonb) => Self::parse_json_string(s),
                    Some(DataType::Vector) => parse_vector(s),
                    Some(DataType::Interval) => parse_interval(s),
                    _ => Ok(OwnedValue::Text(s.to_string())),
                },
                Literal::Boolean(b) => Ok(OwnedValue::Bool(*b)),
                Literal::HexNumber(s) => parse_hex_blob(s),
                Literal::BinaryNumber(s) => parse_binary_blob(s),
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
