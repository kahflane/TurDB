use crate::database::dirty_tracker::ShardedDirtyTracker;
use crate::database::row::Row;
use crate::database::{CheckpointInfo, ExecuteResult, RecoveryInfo};
use crate::mvcc::{TransactionManager, TxnId, TxnState, WriteEntry};
use crate::parsing::{
    parse_binary_blob, parse_date, parse_hex_blob, parse_interval, parse_time, parse_timestamp,
    parse_uuid, parse_vector,
};
use crate::schema::{Catalog, ColumnDef as SchemaColumnDef};
use crate::sql::ast::IsolationLevel;
use crate::sql::builder::ExecutorBuilder;
use crate::sql::context::ExecutionContext;
use crate::sql::executor::{BTreeSource, Executor, ExecutorRow, MaterializedRowSource, ReverseBTreeSource, RowSource, StreamingBTreeSource};
use crate::sql::planner::Planner;
use crate::sql::predicate::CompiledPredicate;
use crate::sql::Parser;
use crate::storage::{
    FileManager, MmapStorage, TableFileHeader, Wal, WalStoragePerTable, FILE_HEADER_SIZE,
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
use std::sync::atomic::{AtomicU64, Ordering as AtomicOrdering};

static PARSE_TIME_NS: AtomicU64 = AtomicU64::new(0);
static INSERT_TIME_NS: AtomicU64 = AtomicU64::new(0);

static RECORD_BUILD_NS: AtomicU64 = AtomicU64::new(0);
static BTREE_INSERT_NS: AtomicU64 = AtomicU64::new(0);

pub fn reset_timing_stats() {
    PARSE_TIME_NS.store(0, AtomicOrdering::Relaxed);
    INSERT_TIME_NS.store(0, AtomicOrdering::Relaxed);
    RECORD_BUILD_NS.store(0, AtomicOrdering::Relaxed);
    BTREE_INSERT_NS.store(0, AtomicOrdering::Relaxed);
}

pub fn get_timing_stats() -> (u64, u64) {
    (PARSE_TIME_NS.load(AtomicOrdering::Relaxed), INSERT_TIME_NS.load(AtomicOrdering::Relaxed))
}

pub fn get_batch_timing_stats() -> (u64, u64) {
    (RECORD_BUILD_NS.load(AtomicOrdering::Relaxed), BTREE_INSERT_NS.load(AtomicOrdering::Relaxed))
}

/// Macro to execute BTree operations with or without WAL tracking.
/// This eliminates code duplication across UPDATE and DELETE operations where we need
/// to branch between WalStorage (when WAL is enabled) and direct storage access.
///
/// # Arguments
/// * `$wal_enabled` - boolean indicating if WAL mode is active
macro_rules! with_btree_storage {
    ($wal_enabled:expr, $storage:expr, $dirty_tracker:expr, $table_id:expr, $root_page:expr, $btree_ops:expr) => {{
        use crate::btree::BTree;
        if $wal_enabled {
            let mut wal_storage = WalStoragePerTable::new($storage, $dirty_tracker, $table_id);
            let mut btree_mut = BTree::new(&mut wal_storage, $root_page)?;
            $btree_ops(&mut btree_mut)?;
        } else {
            let mut btree_mut = BTree::new($storage, $root_page)?;
            $btree_ops(&mut btree_mut)?;
        }
    }};
}

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

/// Default page cache size (number of 16KB pages)
const DEFAULT_CACHE_SIZE: u32 = 256;

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
    dirty_tracker: ShardedDirtyTracker,
    txn_manager: TransactionManager,
    active_txn: Mutex<Option<ActiveTransaction>>,
    /// Session setting: whether foreign key constraints are checked (default: true)
    foreign_keys_enabled: std::sync::atomic::AtomicBool,
    /// Session setting: page cache size in number of pages (default: 256 = 4MB)
    cache_size: std::sync::atomic::AtomicU32,
    table_id_lookup: RwLock<hashbrown::HashMap<u32, (String, String)>>,
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
            dirty_tracker: ShardedDirtyTracker::new(),
            txn_manager: TransactionManager::new(),
            active_txn: Mutex::new(None),
            foreign_keys_enabled: AtomicBool::new(true),
            cache_size: std::sync::atomic::AtomicU32::new(DEFAULT_CACHE_SIZE),
            table_id_lookup: RwLock::new(hashbrown::HashMap::new()),
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
            dirty_tracker: ShardedDirtyTracker::new(),
            txn_manager: TransactionManager::new(),
            active_txn: Mutex::new(None),
            foreign_keys_enabled: AtomicBool::new(true),
            cache_size: std::sync::atomic::AtomicU32::new(DEFAULT_CACHE_SIZE),
            table_id_lookup: RwLock::new(hashbrown::HashMap::new()),
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
            self.populate_table_id_cache(&catalog);
            *guard = Some(catalog);
        }
        Ok(())
    }

    fn populate_table_id_cache(&self, catalog: &Catalog) {
        let mut lookup = self.table_id_lookup.write();
        for (schema_name, schema) in catalog.schemas() {
            for (table_name, table_def) in schema.tables() {
                lookup.insert(
                    table_def.id() as u32,
                    (schema_name.to_string(), table_name.to_string()),
                );
                if let Some(toast_id) = table_def.toast_id() {
                    let toast_table_name = crate::storage::toast::toast_table_name(table_name);
                    lookup.insert(toast_id as u32, (schema_name.to_string(), toast_table_name));
                }
            }
        }
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

    /// Flushes dirty pages to WAL if WAL is enabled and not in an explicit transaction.
    /// In autocommit mode (no explicit transaction), this flushes immediately.
    /// In explicit transaction mode, WAL flush is deferred to commit.
    ///
    /// Returns the number of frames written, or 0 if no flush was needed.
    fn flush_wal_if_autocommit(
        &self,
        file_manager: &mut crate::storage::FileManager,
        schema_name: &str,
        table_name: &str,
        table_id: u32,
    ) -> Result<usize> {
        use std::sync::atomic::Ordering;

        let wal_enabled = self.wal_enabled.load(Ordering::Acquire);
        if !wal_enabled {
            return Ok(0);
        }

        let _txn_guard = self.active_txn.lock();
        if _txn_guard.is_some() {
            return Ok(0);
        }

        if !self.dirty_tracker.has_dirty_pages(table_id) {
            return Ok(0);
        }

        let table_storage = file_manager.table_data(schema_name, table_name)?;
        let mut wal_guard = self.wal.lock();
        let wal = wal_guard.as_mut().ok_or_else(|| {
            eyre::eyre!("WAL is enabled but not initialized - this is a bug")
        })?;
        let frames_written =
            WalStoragePerTable::flush_wal_for_table(&self.dirty_tracker, table_storage, wal, table_id)
                .wrap_err("failed to flush WAL")?;
        Ok(frames_written as usize)
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

    /// Prepares a SQL statement for execution with parameters.
    ///
    /// The SQL can contain parameter placeholders:
    /// - `?` for anonymous parameters (bound in order)
    /// - `$1`, `$2`, etc. for positional parameters
    ///
    /// Returns a `PreparedStatement` that can be bound with values and executed.
    pub fn prepare(&self, sql: &str) -> Result<super::PreparedStatement> {
        use super::prepared::count_parameters;

        let arena = Bump::new();
        let mut parser = Parser::new(sql, &arena);
        parser
            .parse_statement()
            .wrap_err_with(|| format!("failed to parse SQL for prepared statement: {}", sql))?;

        let param_count = count_parameters(sql);
        Ok(super::PreparedStatement::new(sql.to_string(), param_count))
    }

    pub fn insert_batch(&self, table: &str, rows: &[Vec<OwnedValue>]) -> Result<usize> {
        let (schema_name, table_name) = if let Some(dot_pos) = table.find('.') {
            (&table[..dot_pos], &table[dot_pos + 1..])
        } else {
            ("root", table)
        };
        self.insert_batch_into_schema(schema_name, table_name, rows)
    }

    pub fn insert_batch_into_schema(
        &self,
        schema_name: &str,
        table_name: &str,
        rows: &[Vec<OwnedValue>],
    ) -> Result<usize> {
        use crate::btree::BTree;
        use std::sync::atomic::Ordering;

        if rows.is_empty() {
            return Ok(0);
        }

        self.ensure_catalog()?;
        self.ensure_file_manager()?;

        let wal_enabled = self.wal_enabled.load(Ordering::Acquire);
        if wal_enabled {
            self.ensure_wal()?;
        }

        let catalog_guard = self.catalog.read();
        let catalog = catalog_guard.as_ref().unwrap();

        let table_def = catalog.resolve_table(table_name)?;
        let table_id = table_def.id();
        let columns = table_def.columns().to_vec();

        let schema = create_record_schema(&columns);

        drop(catalog_guard);

        let mut file_manager_guard = self.file_manager.write();
        let file_manager = file_manager_guard.as_mut().unwrap();

        let (mut root_page, mut rightmost_hint): (u32, Option<u32>) = {
            let storage = file_manager.table_data_mut(schema_name, table_name)?;
            let page = storage.page(0)?;
            let header = TableFileHeader::from_bytes(page)?;
            let stored_root = header.root_page();
            let hint = header.rightmost_hint();
            let root = if stored_root > 0 { stored_root } else { 1 };
            (root, if hint > 0 { Some(hint) } else { Some(root) })
        };

        let table_file_key = crate::storage::FileManager::make_table_key(schema_name, table_name);
        let mut record_builder = crate::records::RecordBuilder::new(&schema);
        let mut record_buffer = Vec::with_capacity(256);

        let count;

        if wal_enabled {
            let table_storage = file_manager.table_data_mut_with_key(&table_file_key)
                .ok_or_else(|| eyre::eyre!("table storage not found in cache"))?;
            let mut wal_storage =
                WalStoragePerTable::new(table_storage, &self.dirty_tracker, table_id as u32);
            let mut btree = BTree::with_rightmost_hint(&mut wal_storage, root_page, rightmost_hint)?;

            for row_values in rows {
                let row_id = self.next_row_id.fetch_add(1, Ordering::Relaxed);
                let row_key = Self::generate_row_key(row_id);
                OwnedValue::build_record_into_buffer(row_values, &mut record_builder, &mut record_buffer)?;
                btree.insert_append(&row_key, &record_buffer)?;
            }

            root_page = btree.root_page();
            rightmost_hint = btree.rightmost_hint();
            count = rows.len();
        } else {
            let table_storage = file_manager.table_data_mut_with_key(&table_file_key)
                .ok_or_else(|| eyre::eyre!("table storage not found in cache"))?;
            let mut btree = BTree::with_rightmost_hint(table_storage, root_page, rightmost_hint)?;

            for row_values in rows {
                let row_id = self.next_row_id.fetch_add(1, Ordering::Relaxed);
                let row_key = Self::generate_row_key(row_id);
                OwnedValue::build_record_into_buffer(row_values, &mut record_builder, &mut record_buffer)?;
                btree.insert_append(&row_key, &record_buffer)?;
            }

            root_page = btree.root_page();
            rightmost_hint = btree.rightmost_hint();
            count = rows.len();
        }

        {
            let storage = file_manager.table_data_mut(schema_name, table_name)?;
            let page = storage.page_mut(0)?;
            let header = TableFileHeader::from_bytes_mut(page)?;
            header.set_root_page(root_page);
            if let Some(hint) = rightmost_hint {
                header.set_rightmost_hint(hint);
            }
            let new_row_count = header.row_count().saturating_add(count as u64);
            header.set_row_count(new_row_count);
        }

        self.flush_wal_if_autocommit(file_manager, schema_name, table_name, table_id as u32)?;

        Ok(count)
    }

    pub fn query(&self, sql: &str) -> Result<Vec<Row>> {
        let (_columns, rows) = self.query_with_columns(sql)?;
        Ok(rows)
    }

    pub fn query_with_columns(&self, sql: &str) -> Result<(Vec<String>, Vec<Row>)> {
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

        let column_names: Vec<String> = physical_plan
            .output_schema
            .columns
            .iter()
            .map(|c| c.name.to_string())
            .collect();

        let mut file_manager_guard = self.file_manager.write();
        let file_manager = file_manager_guard.as_mut().unwrap();

        enum PlanSource<'a> {
            TableScan(&'a crate::sql::planner::PhysicalTableScan<'a>),
            IndexScan(&'a crate::sql::planner::PhysicalIndexScan<'a>),
            SecondaryIndexScan(&'a crate::sql::planner::PhysicalSecondaryIndexScan<'a>),
            Subquery(&'a crate::sql::planner::PhysicalSubqueryExec<'a>),
            NestedLoopJoin(&'a crate::sql::planner::PhysicalNestedLoopJoin<'a>),
            GraceHashJoin(&'a crate::sql::planner::PhysicalGraceHashJoin<'a>),
            SetOp(&'a crate::sql::planner::PhysicalSetOpExec<'a>),
            DualScan,
        }

        fn find_plan_source<'a>(
            op: &'a crate::sql::planner::PhysicalOperator<'a>,
        ) -> Option<PlanSource<'a>> {
            use crate::sql::planner::PhysicalOperator;
            match op {
                PhysicalOperator::TableScan(scan) => Some(PlanSource::TableScan(scan)),
                PhysicalOperator::DualScan => Some(PlanSource::DualScan),
                PhysicalOperator::IndexScan(scan) => Some(PlanSource::IndexScan(scan)),
                PhysicalOperator::SecondaryIndexScan(scan) => {
                    Some(PlanSource::SecondaryIndexScan(scan))
                }
                PhysicalOperator::SubqueryExec(subq) => Some(PlanSource::Subquery(subq)),
                PhysicalOperator::NestedLoopJoin(join) => Some(PlanSource::NestedLoopJoin(join)),
                PhysicalOperator::GraceHashJoin(join) => Some(PlanSource::GraceHashJoin(join)),
                PhysicalOperator::SetOpExec(set_op) => Some(PlanSource::SetOp(set_op)),
                PhysicalOperator::FilterExec(filter) => find_plan_source(filter.input),
                PhysicalOperator::ProjectExec(project) => find_plan_source(project.input),
                PhysicalOperator::LimitExec(limit) => find_plan_source(limit.input),
                PhysicalOperator::SortExec(sort) => find_plan_source(sort.input),
                PhysicalOperator::HashAggregate(agg) => find_plan_source(agg.input),
                PhysicalOperator::SortedAggregate(agg) => find_plan_source(agg.input),
                PhysicalOperator::WindowExec(window) => find_plan_source(window.input),
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
                PhysicalOperator::SubqueryExec(subq) => find_table_scan(subq.child_plan),
                PhysicalOperator::WindowExec(window) => find_table_scan(window.input),
                _ => None,
            }
        }

        fn find_nested_subquery<'a>(
            op: &'a crate::sql::planner::PhysicalOperator<'a>,
        ) -> Option<&'a crate::sql::planner::PhysicalSubqueryExec<'a>> {
            use crate::sql::planner::PhysicalOperator;
            match op {
                PhysicalOperator::SubqueryExec(subq) => Some(subq),
                PhysicalOperator::FilterExec(filter) => find_nested_subquery(filter.input),
                PhysicalOperator::ProjectExec(project) => find_nested_subquery(project.input),
                PhysicalOperator::LimitExec(limit) => find_nested_subquery(limit.input),
                PhysicalOperator::SortExec(sort) => find_nested_subquery(sort.input),
                PhysicalOperator::HashAggregate(agg) => find_nested_subquery(agg.input),
                PhysicalOperator::SortedAggregate(agg) => find_nested_subquery(agg.input),
                PhysicalOperator::WindowExec(window) => find_nested_subquery(window.input),
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

        fn find_limit<'a>(
            op: &'a crate::sql::planner::PhysicalOperator<'a>,
        ) -> Option<(Option<u64>, Option<u64>)> {
            use crate::sql::planner::PhysicalOperator;
            match op {
                PhysicalOperator::LimitExec(limit) => Some((limit.limit, limit.offset)),
                PhysicalOperator::ProjectExec(project) => find_limit(project.input),
                PhysicalOperator::FilterExec(filter) => find_limit(filter.input),
                PhysicalOperator::SortExec(sort) => find_limit(sort.input),
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

        fn execute_subquery_recursive<'a>(
            subq: &'a crate::sql::planner::PhysicalSubqueryExec<'a>,
            catalog: &crate::schema::catalog::Catalog,
            file_manager: &mut FileManager,
        ) -> Result<Vec<Vec<OwnedValue>>> {
            if let Some(nested_subq) = find_nested_subquery(subq.child_plan) {
                let nested_rows = execute_subquery_recursive(nested_subq, catalog, file_manager)?;

                let nested_source = MaterializedRowSource::new(nested_rows);
                let nested_arena = Bump::new();

                let nested_column_map: Vec<(String, usize)> = nested_subq
                    .output_schema
                    .columns
                    .iter()
                    .enumerate()
                    .map(|(idx, col)| (col.name.to_lowercase(), idx))
                    .collect();

                let inner_plan = crate::sql::planner::PhysicalPlan {
                    root: subq.child_plan,
                    output_schema: subq.output_schema.clone(),
                };

                let inner_ctx = ExecutionContext::new(&nested_arena);
                let inner_builder = ExecutorBuilder::new(&inner_ctx);
                let mut inner_executor = inner_builder
                    .build_with_source_and_column_map(&inner_plan, nested_source, &nested_column_map)
                    .wrap_err("failed to build executor for nested subquery")?;

                let mut materialized_rows: Vec<Vec<OwnedValue>> = Vec::new();
                inner_executor.open()?;
                while let Some(row) = inner_executor.next()? {
                    let owned_values: Vec<OwnedValue> =
                        row.values.iter().map(OwnedValue::from).collect();
                    materialized_rows.push(owned_values);
                }
                inner_executor.close()?;
                return Ok(materialized_rows);
            }

            let inner_table_scan = find_table_scan(subq.child_plan);

            if let Some(inner_scan) = inner_table_scan {
                let schema_name = inner_scan.schema.unwrap_or("root");
                let table_name = inner_scan.table;

                let inner_table_def = catalog
                    .resolve_table(table_name)
                    .wrap_err_with(|| format!("table '{}' not found", table_name))?;

                let column_types: Vec<_> =
                    inner_table_def.columns().iter().map(|c| c.data_type()).collect();

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

                let inner_table_column_map: Vec<(String, usize)> = inner_table_def
                    .columns()
                    .iter()
                    .enumerate()
                    .map(|(idx, col)| (col.name().to_lowercase(), idx))
                    .collect();

                let inner_ctx = ExecutionContext::new(&inner_arena);
                let inner_builder = ExecutorBuilder::new(&inner_ctx);
                let mut inner_executor = inner_builder
                    .build_with_source_and_column_map(&inner_plan, inner_source, &inner_table_column_map)
                    .wrap_err("failed to build inner executor")?;

                let mut materialized_rows: Vec<Vec<OwnedValue>> = Vec::new();
                inner_executor.open()?;
                while let Some(row) = inner_executor.next()? {
                    let owned_values: Vec<OwnedValue> =
                        row.values.iter().map(OwnedValue::from).collect();
                    materialized_rows.push(owned_values);
                }
                inner_executor.close()?;
                Ok(materialized_rows)
            } else {
                bail!("subquery inner plan must have a table scan")
            }
        }

        let plan_source = find_plan_source(physical_plan.root);

        fn is_simple_count_star<'a>(
            op: &'a crate::sql::planner::PhysicalOperator<'a>,
        ) -> Option<&'a crate::sql::planner::PhysicalTableScan<'a>> {
            use crate::sql::planner::{AggregateFunction, PhysicalOperator};
            match op {
                PhysicalOperator::HashAggregate(agg) => {
                    if !agg.group_by.is_empty() {
                        return None;
                    }
                    if agg.aggregates.len() != 1 {
                        return None;
                    }
                    let agg_expr = &agg.aggregates[0];
                    if agg_expr.function != AggregateFunction::Count || agg_expr.distinct {
                        return None;
                    }
                    match agg.input {
                        PhysicalOperator::TableScan(scan) => {
                            if scan.post_scan_filter.is_none() {
                                Some(scan)
                            } else {
                                None
                            }
                        }
                        _ => None,
                    }
                }
                PhysicalOperator::ProjectExec(proj) => {
                    fn is_simple_aggregate_projection(
                        expressions: &[&crate::sql::ast::Expr<'_>],
                    ) -> bool {
                        use crate::sql::ast::{Expr, FunctionArgs};
                        if expressions.len() != 1 {
                            return false;
                        }
                        match expressions[0] {
                            Expr::Function(func) => {
                                let name = func.name.name.to_uppercase();
                                if !matches!(
                                    name.as_str(),
                                    "COUNT" | "SUM" | "AVG" | "MIN" | "MAX"
                                ) {
                                    return false;
                                }
                                matches!(func.args, FunctionArgs::Star | FunctionArgs::None)
                                    || matches!(&func.args, FunctionArgs::Args(args) if args.len() <= 1)
                            }
                            _ => false,
                        }
                    }
                    if is_simple_aggregate_projection(proj.expressions) {
                        is_simple_count_star(proj.input)
                    } else {
                        None
                    }
                }
                _ => None,
            }
        }

        if let Some(scan) = is_simple_count_star(physical_plan.root) {
            let schema_name = scan.schema.unwrap_or("root");
            let table_name = scan.table;

            let storage = file_manager
                .table_data(schema_name, table_name)
                .wrap_err_with(|| {
                    format!(
                        "failed to open table storage for {}.{}",
                        schema_name, table_name
                    )
                })?;

            let page = storage.page(0)?;
            let header = TableFileHeader::from_bytes(page)?;
            let count = header.row_count() as i64;

            return Ok((column_names, vec![Row::new(vec![OwnedValue::Int(count)])]));
        }

        let mut toast_table_info: Option<(String, String)> = None;

        let rows = match plan_source {
            Some(PlanSource::TableScan(scan)) => {
                let schema_name = scan.schema.unwrap_or("root");
                let table_name = scan.table;

                toast_table_info = Some((schema_name.to_string(), table_name.to_string()));

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
                let source: BTreeSource = if scan.reverse {
                    BTreeSource::Reverse(
                        ReverseBTreeSource::from_btree_scan_reverse_with_projections(
                            storage,
                            root_page,
                            column_types,
                            projections,
                        )
                        .wrap_err("failed to create reverse table scan")?,
                    )
                } else {
                    BTreeSource::Forward(
                        StreamingBTreeSource::from_btree_scan_with_projections(
                            storage,
                            root_page,
                            column_types,
                            projections,
                        )
                        .wrap_err("failed to create table scan")?,
                    )
                };

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

                toast_table_info = Some((schema_name.to_string(), table_name.to_string()));

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
            Some(PlanSource::SecondaryIndexScan(scan)) => {
                use crate::btree::BTreeReader;
                use crate::records::RecordView;

                let schema_name = scan.schema.unwrap_or("root");
                let table_name = scan.table;
                let index_name = scan.index_name;

                toast_table_info = Some((schema_name.to_string(), table_name.to_string()));

                let table_def = scan.table_def.ok_or_else(|| {
                    eyre::eyre!("SecondaryIndexScan missing table_def for {}", table_name)
                })?;

                let columns = table_def.columns();
                let schema = create_record_schema(columns);

                let row_id_suffix_len = if scan.is_unique_index { 0 } else { 8 };

                let row_keys: Vec<[u8; 8]> = {
                    let index_storage = file_manager
                        .index_data(schema_name, table_name, index_name)
                        .wrap_err_with(|| {
                            format!(
                                "failed to open index storage for {}.{}.{}",
                                schema_name, table_name, index_name
                            )
                        })?;

                    let root_page = 1u32;
                    let index_reader = BTreeReader::new(index_storage, root_page)?;

                    let mut keys = Vec::new();

                    if scan.reverse {
                        let mut cursor = index_reader.cursor_last()?;
                        if cursor.valid() {
                            loop {
                                let index_key = cursor.key()?;
                                let row_id_bytes = if scan.is_unique_index {
                                    cursor.value()?
                                } else {
                                    &index_key[index_key.len().saturating_sub(row_id_suffix_len)..]
                                };

                                if row_id_bytes.len() == 8 {
                                    let row_key: [u8; 8] = row_id_bytes.try_into().unwrap();
                                    keys.push(row_key);
                                }

                                if !cursor.prev()? {
                                    break;
                                }
                            }
                        }
                    } else {
                        let mut cursor = index_reader.cursor_first()?;
                        if cursor.valid() {
                            loop {
                                let index_key = cursor.key()?;
                                let row_id_bytes = if scan.is_unique_index {
                                    cursor.value()?
                                } else {
                                    &index_key[index_key.len().saturating_sub(row_id_suffix_len)..]
                                };

                                if row_id_bytes.len() == 8 {
                                    let row_key: [u8; 8] = row_id_bytes.try_into().unwrap();
                                    keys.push(row_key);
                                }

                                if !cursor.advance()? {
                                    break;
                                }
                            }
                        }
                    }
                    keys
                };

                let mut materialized_rows: Vec<Vec<OwnedValue>> = Vec::with_capacity(row_keys.len());

                {
                    let table_storage = file_manager
                        .table_data(schema_name, table_name)
                        .wrap_err_with(|| {
                            format!(
                                "failed to open table storage for {}.{}",
                                schema_name, table_name
                            )
                        })?;

                    let root_page = 1u32;
                    let table_reader = BTreeReader::new(table_storage, root_page)?;

                    for row_key in &row_keys {
                        if let Some(row_data) = table_reader.get(row_key)? {
                            let record = RecordView::new(row_data, &schema)?;
                            let row_values =
                                OwnedValue::extract_row_from_record(&record, columns)?;
                            materialized_rows.push(row_values);
                        }
                    }
                }

                let materialized_source = MaterializedRowSource::new(materialized_rows);

                let ctx = ExecutionContext::new(&arena);
                let builder = ExecutorBuilder::new(&ctx);

                let all_columns_map: Vec<(String, usize)> = table_def
                    .columns()
                    .iter()
                    .enumerate()
                    .map(|(idx, col)| (col.name().to_lowercase(), idx))
                    .collect();

                let mut executor = builder
                    .build_with_source_and_column_map(
                        &physical_plan,
                        materialized_source,
                        &all_columns_map,
                    )
                    .wrap_err("failed to build executor for secondary index scan")?;

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
                let inner_rows = execute_subquery_recursive(subq, catalog, file_manager)?;

                let materialized_source = MaterializedRowSource::new(inner_rows);

                let subq_column_map: Vec<(String, usize)> = subq
                    .output_schema
                    .columns
                    .iter()
                    .enumerate()
                    .map(|(idx, col)| (col.name.to_lowercase(), idx))
                    .collect();

                let ctx = ExecutionContext::new(&arena);
                let builder = ExecutorBuilder::new(&ctx);
                let mut executor = builder
                    .build_with_source_and_column_map(&physical_plan, materialized_source, &subq_column_map)
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
            Some(PlanSource::NestedLoopJoin(_)) | Some(PlanSource::GraceHashJoin(_)) => {
                fn find_subquery_in_join<'a>(
                    op: &'a crate::sql::planner::PhysicalOperator<'a>,
                ) -> Option<&'a crate::sql::planner::PhysicalSubqueryExec<'a>> {
                    use crate::sql::planner::PhysicalOperator;
                    match op {
                        PhysicalOperator::SubqueryExec(subq) => Some(subq),
                        PhysicalOperator::FilterExec(f) => find_subquery_in_join(f.input),
                        PhysicalOperator::ProjectExec(p) => find_subquery_in_join(p.input),
                        PhysicalOperator::HashAggregate(agg) => find_subquery_in_join(agg.input),
                        PhysicalOperator::SortedAggregate(agg) => find_subquery_in_join(agg.input),
                        PhysicalOperator::SortExec(s) => find_subquery_in_join(s.input),
                        PhysicalOperator::LimitExec(l) => find_subquery_in_join(l.input),
                        PhysicalOperator::WindowExec(w) => find_subquery_in_join(w.input),
                        _ => None,
                    }
                }

                enum JoinScanInfo<'a> {
                    TableScan(&'a crate::sql::planner::PhysicalTableScan<'a>),
                    IndexScan(&'a crate::sql::planner::PhysicalIndexScan<'a>),
                    SecondaryIndexScan(&'a crate::sql::planner::PhysicalSecondaryIndexScan<'a>),
                }

                fn find_scan_in_join<'a>(
                    op: &'a crate::sql::planner::PhysicalOperator<'a>,
                ) -> Option<JoinScanInfo<'a>> {
                    use crate::sql::planner::PhysicalOperator;
                    match op {
                        PhysicalOperator::TableScan(scan) => Some(JoinScanInfo::TableScan(scan)),
                        PhysicalOperator::IndexScan(scan) => Some(JoinScanInfo::IndexScan(scan)),
                        PhysicalOperator::SecondaryIndexScan(scan) => Some(JoinScanInfo::SecondaryIndexScan(scan)),
                        PhysicalOperator::FilterExec(f) => find_scan_in_join(f.input),
                        PhysicalOperator::ProjectExec(p) => find_scan_in_join(p.input),
                        PhysicalOperator::HashAggregate(agg) => find_scan_in_join(agg.input),
                        PhysicalOperator::SortedAggregate(agg) => find_scan_in_join(agg.input),
                        PhysicalOperator::SortExec(s) => find_scan_in_join(s.input),
                        PhysicalOperator::LimitExec(l) => find_scan_in_join(l.input),
                        PhysicalOperator::WindowExec(w) => find_scan_in_join(w.input),
                        _ => None,
                    }
                }

                let (left_op, right_op, join_type, condition, join_keys) = match plan_source {
                    Some(PlanSource::NestedLoopJoin(j)) => (j.left, j.right, j.join_type, j.condition, &[][..]),
                    Some(PlanSource::GraceHashJoin(j)) => (j.left, j.right, j.join_type, None, j.join_keys),
                    _ => unreachable!(),
                };

                let left_subq = find_subquery_in_join(left_op);
                let right_subq = find_subquery_in_join(right_op);
                let left_scan = find_scan_in_join(left_op);
                let right_scan = find_scan_in_join(right_op);

                let mut left_rows: Vec<Vec<OwnedValue>> = Vec::new();
                let mut right_rows: Vec<Vec<OwnedValue>> = Vec::new();
                let mut right_col_count = 0usize;

                let mut left_table_name: Option<&str> = None;
                let mut left_alias: Option<&str> = None;
                let mut right_table_name: Option<&str> = None;
                let mut right_alias: Option<&str> = None;

                if let Some(subq) = left_subq {
                    left_rows = execute_subquery_recursive(subq, catalog, file_manager)?;
                } else if let Some(scan_info) = &left_scan {
                    let (schema_name, table_name, alias) = match scan_info {
                        JoinScanInfo::TableScan(scan) => (scan.schema.unwrap_or("root"), scan.table, scan.alias),
                        JoinScanInfo::IndexScan(scan) => (scan.schema.unwrap_or("root"), scan.table, None),
                        JoinScanInfo::SecondaryIndexScan(scan) => (scan.schema.unwrap_or("root"), scan.table, None),
                    };
                    left_table_name = Some(table_name);
                    left_alias = alias;
                    let table_def = catalog.resolve_table(table_name)?;
                    let column_types: Vec<_> = table_def.columns().iter().map(|c| c.data_type()).collect();
                    let storage = file_manager.table_data(schema_name, table_name)?;
                    let source = StreamingBTreeSource::from_btree_scan_with_projections(
                        storage, 1, column_types.clone(), None,
                    )?;
                    let mut cursor = source;
                    while let Some(row) = cursor.next_row()? {
                        left_rows.push(row.iter().map(OwnedValue::from).collect());
                    }
                }

                if let Some(subq) = right_subq {
                    right_rows = execute_subquery_recursive(subq, catalog, file_manager)?;
                    right_col_count = subq.output_schema.columns.len();
                } else if let Some(scan_info) = &right_scan {
                    let (schema_name, table_name, alias) = match scan_info {
                        JoinScanInfo::TableScan(scan) => (scan.schema.unwrap_or("root"), scan.table, scan.alias),
                        JoinScanInfo::IndexScan(scan) => (scan.schema.unwrap_or("root"), scan.table, None),
                        JoinScanInfo::SecondaryIndexScan(scan) => (scan.schema.unwrap_or("root"), scan.table, None),
                    };
                    right_table_name = Some(table_name);
                    right_alias = alias;
                    let table_def = catalog.resolve_table(table_name)?;
                    let column_types: Vec<_> = table_def.columns().iter().map(|c| c.data_type()).collect();
                    let storage = file_manager.table_data(schema_name, table_name)?;
                    let source = StreamingBTreeSource::from_btree_scan_with_projections(
                        storage, 1, column_types.clone(), None,
                    )?;
                    let mut cursor = source;
                    while let Some(row) = cursor.next_row()? {
                        right_rows.push(row.iter().map(OwnedValue::from).collect());
                    }
                    right_col_count = column_types.len();
                }

                let mut join_column_map: Vec<(String, usize)> = Vec::new();
                let mut idx = 0usize;

                if let Some(subq) = left_subq {
                    for col in subq.output_schema.columns {
                        join_column_map.push((col.name.to_lowercase(), idx));
                        join_column_map.push((format!("{}.{}", subq.alias, col.name).to_lowercase(), idx));
                        idx += 1;
                    }
                } else if let Some(table_name) = left_table_name {
                    let table_def = catalog.resolve_table(table_name)?;
                    for col in table_def.columns() {
                        join_column_map.push((col.name().to_lowercase(), idx));
                        join_column_map.push((format!("{}.{}", table_name, col.name()).to_lowercase(), idx));
                        if let Some(alias) = left_alias {
                            join_column_map.push((format!("{}.{}", alias, col.name()).to_lowercase(), idx));
                        }
                        idx += 1;
                    }
                }

                if let Some(subq) = right_subq {
                    for col in subq.output_schema.columns {
                        join_column_map.push((col.name.to_lowercase(), idx));
                        join_column_map.push((format!("{}.{}", subq.alias, col.name).to_lowercase(), idx));
                        idx += 1;
                    }
                } else if let Some(table_name) = right_table_name {
                    let table_def = catalog.resolve_table(table_name)?;
                    for col in table_def.columns() {
                        join_column_map.push((col.name().to_lowercase(), idx));
                        join_column_map.push((format!("{}.{}", table_name, col.name()).to_lowercase(), idx));
                        if let Some(alias) = right_alias {
                            join_column_map.push((format!("{}.{}", alias, col.name()).to_lowercase(), idx));
                        }
                        idx += 1;
                    }
                }

                let condition_predicate = condition.map(|c| {
                    crate::sql::predicate::CompiledPredicate::new(c, join_column_map.clone())
                });

                let key_indices: Vec<(usize, usize)> = join_keys.iter().filter_map(|(left_expr, right_expr)| {
                    use crate::sql::ast::Expr;

                    let left_idx = if let Expr::Column(col) = left_expr {
                        let qualified = col.table.map(|t| format!("{}.{}", t, col.column).to_lowercase());
                        qualified.as_ref()
                            .and_then(|q| join_column_map.iter().find(|(name, _)| name == q).map(|(_, idx)| *idx))
                            .or_else(|| join_column_map.iter().find(|(name, _)| name.eq_ignore_ascii_case(col.column)).map(|(_, idx)| *idx))
                    } else { None };

                    let right_idx = if let Expr::Column(col) = right_expr {
                        let qualified = col.table.map(|t| format!("{}.{}", t, col.column).to_lowercase());
                        qualified.as_ref()
                            .and_then(|q| join_column_map.iter().find(|(name, _)| name == q).map(|(_, idx)| *idx))
                            .or_else(|| join_column_map.iter().find(|(name, _)| name.eq_ignore_ascii_case(col.column)).map(|(_, idx)| *idx))
                    } else { None };

                    match (left_idx, right_idx) {
                        (Some(l), Some(r)) => Some((l, r)),
                        _ => None,
                    }
                }).collect();

                let limit_info = find_limit(physical_plan.root);
                let offset = limit_info.and_then(|(_, o)| o).unwrap_or(0) as usize;
                let limit = limit_info.and_then(|(l, _)| l).map(|l| l as usize);

                let mut result_rows: Vec<Row> = Vec::new();
                let mut skipped = 0usize;
                let mut seen: std::collections::HashSet<Vec<u64>> = std::collections::HashSet::new();

                'outer: for left_row in &left_rows {
                    let mut matched = false;
                    for right_row in &right_rows {
                        let mut combined: Vec<OwnedValue> = left_row.clone();
                        combined.extend(right_row.clone());

                        let should_include = if let Some(ref pred) = condition_predicate {
                            let values: Vec<Value<'_>> = combined.iter().map(|v| v.to_value()).collect();
                            let row_ref = ExecutorRow::new(&values);
                            pred.evaluate(&row_ref)
                        } else if !key_indices.is_empty() {
                            key_indices.iter().all(|(left_idx, right_idx)| {
                                combined.get(*left_idx) == combined.get(*right_idx)
                            })
                        } else {
                            true
                        };

                        if should_include {
                            matched = true;

                            let output_columns = physical_plan.output_schema.columns;
                            let owned: Vec<OwnedValue> = output_columns.iter().map(|col| {
                                let col_name = col.name.to_lowercase();
                                let source_idx = join_column_map.iter()
                                    .find(|(name, _)| name == &col_name)
                                    .map(|(_, idx)| *idx)
                                    .unwrap_or(0);
                                let val = combined.get(source_idx).cloned().unwrap_or(OwnedValue::Null);
                                convert_value_with_type(&val.to_value(), col.data_type)
                            }).collect();

                            if is_distinct {
                                let key: Vec<u64> = owned
                                    .iter()
                                    .map(|v| {
                                        use std::hash::{Hash, Hasher};
                                        let mut hasher = std::collections::hash_map::DefaultHasher::new();
                                        format!("{:?}", v).hash(&mut hasher);
                                        hasher.finish()
                                    })
                                    .collect();
                                if !seen.insert(key) {
                                    continue;
                                }
                            }

                            if skipped < offset {
                                skipped += 1;
                                continue;
                            }

                            result_rows.push(Row::new(owned));

                            if let Some(lim) = limit {
                                if result_rows.len() >= lim {
                                    break 'outer;
                                }
                            }
                        }
                    }
                    if !matched && matches!(join_type, crate::sql::ast::JoinType::Left | crate::sql::ast::JoinType::Full) {
                        let mut combined: Vec<OwnedValue> = left_row.clone();
                        combined.extend(std::iter::repeat_n(OwnedValue::Null, right_col_count));
                        let output_columns = physical_plan.output_schema.columns;
                        let owned: Vec<OwnedValue> = output_columns.iter().map(|col| {
                            let col_name = col.name.to_lowercase();
                            let source_idx = join_column_map.iter()
                                .find(|(name, _)| name == &col_name)
                                .map(|(_, idx)| *idx)
                                .unwrap_or(0);
                            let val = combined.get(source_idx).cloned().unwrap_or(OwnedValue::Null);
                            convert_value_with_type(&val.to_value(), col.data_type)
                        }).collect();

                        if is_distinct {
                            let key: Vec<u64> = owned
                                .iter()
                                .map(|v| {
                                    use std::hash::{Hash, Hasher};
                                    let mut hasher = std::collections::hash_map::DefaultHasher::new();
                                    format!("{:?}", v).hash(&mut hasher);
                                    hasher.finish()
                                })
                                .collect();
                            if !seen.insert(key) {
                                continue;
                            }
                        }

                        if skipped < offset {
                            skipped += 1;
                            continue;
                        }

                        result_rows.push(Row::new(owned));

                        if let Some(lim) = limit {
                            if result_rows.len() >= lim {
                                break 'outer;
                            }
                        }
                    }
                }
                result_rows
            }
            Some(PlanSource::SetOp(_set_op)) => {
                drop(file_manager_guard);
                let rows = self.execute_physical_plan_recursive(physical_plan.root, &arena)?;
                return Ok((column_names, rows));
            }
            Some(PlanSource::DualScan) => {
                let ctx = ExecutionContext::new(&arena);
                let builder = ExecutorBuilder::new(&ctx);

                let source = crate::sql::executor::DualSource::default();
                let mut executor = builder
                    .build_with_source(&physical_plan, source)
                    .wrap_err("failed to build executor for dual scan")?;

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
            let limit_info = find_limit(physical_plan.root);
            let offset = limit_info.and_then(|(_, o)| o).unwrap_or(0) as usize;
            let limit = limit_info.and_then(|(l, _)| l);

            let mut seen: std::collections::HashSet<Vec<u64>> = std::collections::HashSet::new();
            let deduplicated: Vec<Row> = rows
                .into_iter()
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
                .collect();

            if let Some(lim) = limit {
                deduplicated.into_iter().skip(offset).take(lim as usize).collect()
            } else if offset > 0 {
                deduplicated.into_iter().skip(offset).collect()
            } else {
                deduplicated
            }
        } else {
            rows
        };

        let rows = if let Some((schema_name, table_name)) = toast_table_info {
            self.detoast_rows(file_manager, &schema_name, &table_name, rows)?
        } else {
            rows
        };

        Ok((column_names, rows))
    }

    fn execute_physical_plan_recursive<'a>(
        &self,
        op: &'a crate::sql::planner::PhysicalOperator<'a>,
        _arena: &'a Bump,
    ) -> Result<Vec<Row>> {
        use crate::sql::planner::{PhysicalOperator, SetOpKind};

        let catalog_guard = self.catalog.read();
        let catalog = catalog_guard.as_ref().unwrap();

        let mut file_manager_guard = self.file_manager.write();
        let file_manager = file_manager_guard.as_mut().unwrap();

        fn execute_branch_for_set_op<'a>(
            _db_path: &std::path::Path,
            op: &'a PhysicalOperator<'a>,
            catalog: &crate::schema::catalog::Catalog,
            file_manager: &mut crate::storage::FileManager,
        ) -> Result<Vec<Row>> {
            fn find_table_scan_for_set<'a>(
                op: &'a PhysicalOperator<'a>,
            ) -> Option<&'a crate::sql::planner::PhysicalTableScan<'a>> {
                match op {
                    PhysicalOperator::TableScan(scan) => Some(scan),
                    PhysicalOperator::FilterExec(f) => find_table_scan_for_set(f.input),
                    PhysicalOperator::ProjectExec(p) => find_table_scan_for_set(p.input),
                    PhysicalOperator::LimitExec(l) => find_table_scan_for_set(l.input),
                    PhysicalOperator::SortExec(s) => find_table_scan_for_set(s.input),
                    PhysicalOperator::SubqueryExec(sub) => find_table_scan_for_set(sub.child_plan),
                    PhysicalOperator::SetOpExec(set) => find_table_scan_for_set(set.left),
                    PhysicalOperator::WindowExec(w) => find_table_scan_for_set(w.input),
                    _ => None,
                }
            }

            match op {
                PhysicalOperator::SortExec(sort) => {
                    let mut rows = execute_branch_for_set_op(_db_path, sort.input, catalog, file_manager)?;
                    if !sort.order_by.is_empty() {
                        let first_key = &sort.order_by[0];
                        let ascending = first_key.ascending;
                        rows.sort_by(|a, b| {
                            let a_val = a.values.first();
                            let b_val = b.values.first();
                            let cmp = match (a_val, b_val) {
                                (Some(OwnedValue::Int(a_i)), Some(OwnedValue::Int(b_i))) => a_i.cmp(b_i),
                                (Some(OwnedValue::Text(a_t)), Some(OwnedValue::Text(b_t))) => a_t.cmp(b_t),
                                (Some(OwnedValue::Float(a_f)), Some(OwnedValue::Float(b_f))) => {
                                    a_f.partial_cmp(b_f).unwrap_or(std::cmp::Ordering::Equal)
                                }
                                _ => std::cmp::Ordering::Equal,
                            };
                            if ascending { cmp } else { cmp.reverse() }
                        });
                    }
                    Ok(rows)
                }
                PhysicalOperator::LimitExec(limit) => {
                    let rows = execute_branch_for_set_op(_db_path, limit.input, catalog, file_manager)?;
                    let offset = limit.offset.unwrap_or(0) as usize;
                    let count = limit.limit.unwrap_or(usize::MAX as u64) as usize;
                    Ok(rows.into_iter().skip(offset).take(count).collect())
                }
                PhysicalOperator::SetOpExec(set_op) => {
                    let left_rows = execute_branch_for_set_op(_db_path, set_op.left, catalog, file_manager)?;
                    let right_rows = execute_branch_for_set_op(_db_path, set_op.right, catalog, file_manager)?;

                    fn row_to_key(row: &Row) -> Vec<u64> {
                        use std::hash::{Hash, Hasher};
                        row.values
                            .iter()
                            .map(|v| {
                                let mut hasher = std::collections::hash_map::DefaultHasher::new();
                                format!("{:?}", v).hash(&mut hasher);
                                hasher.finish()
                            })
                            .collect()
                    }

                    let result = match set_op.kind {
                        SetOpKind::Union => {
                            if set_op.all {
                                let mut all = left_rows;
                                all.extend(right_rows);
                                all
                            } else {
                                let mut seen: std::collections::HashSet<Vec<u64>> =
                                    std::collections::HashSet::new();
                                let mut result = Vec::new();
                                for row in left_rows.into_iter().chain(right_rows.into_iter()) {
                                    let key = row_to_key(&row);
                                    if seen.insert(key) {
                                        result.push(row);
                                    }
                                }
                                result
                            }
                        }
                        SetOpKind::Intersect => {
                            let right_keys: std::collections::HashSet<Vec<u64>> =
                                right_rows.iter().map(row_to_key).collect();
                            if set_op.all {
                                left_rows
                                    .into_iter()
                                    .filter(|row| right_keys.contains(&row_to_key(row)))
                                    .collect()
                            } else {
                                let mut seen: std::collections::HashSet<Vec<u64>> =
                                    std::collections::HashSet::new();
                                left_rows
                                    .into_iter()
                                    .filter(|row| {
                                        let key = row_to_key(row);
                                        right_keys.contains(&key) && seen.insert(key)
                                    })
                                    .collect()
                            }
                        }
                        SetOpKind::Except => {
                            let right_keys: std::collections::HashSet<Vec<u64>> =
                                right_rows.iter().map(row_to_key).collect();
                            if set_op.all {
                                left_rows
                                    .into_iter()
                                    .filter(|row| !right_keys.contains(&row_to_key(row)))
                                    .collect()
                            } else {
                                let mut seen: std::collections::HashSet<Vec<u64>> =
                                    std::collections::HashSet::new();
                                left_rows
                                    .into_iter()
                                    .filter(|row| {
                                        let key = row_to_key(row);
                                        !right_keys.contains(&key) && seen.insert(key)
                                    })
                                    .collect()
                            }
                        }
                    };
                    Ok(result)
                }
                _ => {
                    let scan = find_table_scan_for_set(op)
                        .ok_or_else(|| eyre::eyre!("set operation branch must have a table scan"))?;

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
                        column_types.clone(),
                        None,
                    )
                    .wrap_err("failed to create table scan")?;

                    let branch_arena = Bump::new();
                    let output_schema = crate::sql::planner::OutputSchema {
                        columns: branch_arena.alloc_slice_fill_iter(
                            table_def.columns().iter().map(|col| {
                                crate::sql::planner::OutputColumn {
                                    name: branch_arena.alloc_str(col.name()),
                                    data_type: col.data_type(),
                                    nullable: col.is_nullable(),
                                }
                            })
                        ),
                    };

                    let branch_plan = crate::sql::planner::PhysicalPlan {
                        root: op,
                        output_schema,
                    };

                    let column_map: Vec<(String, usize)> = table_def
                        .columns()
                        .iter()
                        .enumerate()
                        .map(|(idx, col)| (col.name().to_lowercase(), idx))
                        .collect();

                    let ctx = ExecutionContext::new(&branch_arena);
                    let builder = ExecutorBuilder::new(&ctx);
                    let mut executor = builder
                        .build_with_source_and_column_map(&branch_plan, source, &column_map)
                        .wrap_err("failed to build executor")?;

                    let mut rows = Vec::new();
                    executor.open()?;
                    while let Some(row) = executor.next()? {
                        let owned: Vec<OwnedValue> =
                            row.values.iter().map(OwnedValue::from).collect();
                        rows.push(Row::new(owned));
                    }
                    executor.close()?;
                    Ok(rows)
                }
            }
        }

        execute_branch_for_set_op(&self.path, op, catalog, file_manager)
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
        let parse_start = std::time::Instant::now();
        let arena = Bump::new();
        let mut parser = Parser::new(sql, &arena);
        let stmt = parser
            .parse_statement()
            .wrap_err("failed to parse SQL statement")?;
        PARSE_TIME_NS.fetch_add(parse_start.elapsed().as_nanos() as u64, AtomicOrdering::Relaxed);

        use crate::sql::ast::Statement;
        match stmt {
            Statement::CreateTable(create) => self.execute_create_table(create, &arena),
            Statement::CreateSchema(create) => self.execute_create_schema(create),
            Statement::CreateIndex(create) => self.execute_create_index(create, &arena),
            Statement::Insert(insert) => {
                let insert_start = std::time::Instant::now();
                let result = self.execute_insert(insert, &arena);
                INSERT_TIME_NS.fetch_add(insert_start.elapsed().as_nanos() as u64, AtomicOrdering::Relaxed);
                result
            }
            Statement::Update(update) => self.execute_update(update, &arena),
            Statement::Delete(delete) => self.execute_delete(delete, &arena),
            Statement::Select(_) => {
                let (columns, rows) = self.query_with_columns(sql)?;
                Ok(ExecuteResult::Select { columns, rows })
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
            Statement::Truncate(truncate) => self.execute_truncate(truncate),
            Statement::AlterTable(alter) => self.execute_alter_table(alter),
            Statement::Set(set) => self.execute_set(set),
            _ => bail!("unsupported statement type"),
        }
    }

    fn execute_set(&self, set: &crate::sql::ast::SetStmt<'_>) -> Result<ExecuteResult> {
        use std::sync::atomic::Ordering;

        let name = set.name.to_lowercase();
        let value = set.value.first().ok_or_else(|| eyre::eyre!("SET requires a value"))?;

        match name.as_str() {
            "foreign_keys" => {
                let enabled = match value {
                    crate::sql::ast::Expr::Literal(crate::sql::ast::Literal::Boolean(b)) => *b,
                    crate::sql::ast::Expr::Literal(crate::sql::ast::Literal::Integer(i)) => {
                        i.parse::<i64>().unwrap_or(0) != 0
                    }
                    crate::sql::ast::Expr::Literal(crate::sql::ast::Literal::String(s)) => {
                        matches!(s.to_lowercase().as_str(), "on" | "true" | "1" | "yes")
                    }
                    crate::sql::ast::Expr::Column(col) => {
                        matches!(col.column.to_lowercase().as_str(), "on" | "true" | "yes")
                    }
                    _ => bail!("invalid value for foreign_keys: expected ON/OFF, TRUE/FALSE, or 1/0"),
                };
                self.foreign_keys_enabled.store(enabled, Ordering::Release);
                Ok(ExecuteResult::Set {
                    name: "foreign_keys".to_string(),
                    value: if enabled { "ON".to_string() } else { "OFF".to_string() },
                })
            }
            "cache_size" => {
                let size = match value {
                    crate::sql::ast::Expr::Literal(crate::sql::ast::Literal::Integer(i)) => {
                        let parsed: i64 = i.parse().wrap_err("cache_size must be a valid integer")?;
                        if parsed <= 0 {
                            bail!("cache_size must be a positive integer");
                        }
                        parsed as u32
                    }
                    _ => bail!("invalid value for cache_size: expected a positive integer"),
                };
                self.cache_size.store(size, Ordering::Release);
                Ok(ExecuteResult::Set {
                    name: "cache_size".to_string(),
                    value: size.to_string(),
                })
            }
            _ => bail!("unknown setting: {}", set.name),
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
                use crate::schema::table::Constraint as SchemaConstraint;
                use crate::sql::ast::ColumnConstraint;
                use crate::sql::ast::DataType as SqlDataType;

                let data_type = Self::convert_data_type(&col.data_type);
                let mut column = SchemaColumnDef::new(col.name.to_string(), data_type);

                if let Some(max_len) = Self::extract_type_length(&col.data_type) {
                    column = column.with_max_length(max_len);
                }

                for constraint in col.constraints {
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
                        ColumnConstraint::AutoIncrement => {
                            column = column.with_constraint(SchemaConstraint::AutoIncrement);
                        }
                        ColumnConstraint::Null | ColumnConstraint::Generated { .. } => {}
                    }
                }

                if matches!(col.data_type, SqlDataType::Serial | SqlDataType::BigSerial | SqlDataType::SmallSerial) {
                    column = column.with_constraint(SchemaConstraint::AutoIncrement);
                    column = column.with_constraint(SchemaConstraint::NotNull);
                }

                column
            })
            .collect();

        let column_count = columns.len() as u32;
        let table_id = self.allocate_table_id();
        catalog.create_table_with_id(schema_name, table_name, columns, table_id)?;

        drop(catalog_guard);

        self.table_id_lookup.write().insert(
            table_id as u32,
            (schema_name.to_string(), table_name.to_string()),
        );

        let mut file_manager_guard = self.file_manager.write();
        let file_manager = file_manager_guard.as_mut().unwrap();
        file_manager.create_table(schema_name, table_name, table_id, column_count)?;

        let storage = file_manager.table_data_mut(schema_name, table_name)?;
        storage.grow(2)?;
        crate::btree::BTree::create(storage, 1)?;

        let needs_toast = {
            let catalog_guard = self.catalog.read();
            let catalog = catalog_guard.as_ref().unwrap();
            catalog
                .get_schema(schema_name)
                .and_then(|s| s.get_table(table_name))
                .map(|t| t.columns().iter().any(|c| c.data_type().is_toastable()))
                .unwrap_or(false)
        };

        if needs_toast {
            let toast_table_name = crate::storage::toast::toast_table_name(table_name);
            let toast_id = self.allocate_table_id();
            file_manager.create_table(schema_name, &toast_table_name, toast_id, 3)?;

            let toast_storage = file_manager.table_data_mut(schema_name, &toast_table_name)?;
            toast_storage.grow(2)?;
            crate::btree::BTree::create(toast_storage, 1)?;

            self.table_id_lookup.write().insert(
                toast_id as u32,
                (schema_name.to_string(), toast_table_name.clone()),
            );

            let mut catalog_guard = self.catalog.write();
            let catalog = catalog_guard.as_mut().unwrap();
            if let Some(schema) = catalog.get_schema_mut(schema_name) {
                if let Some(table) = schema.get_table(table_name) {
                    let mut table_clone = table.clone();
                    table_clone.set_toast_id(Some(toast_id));
                    schema.remove_table(table_name);
                    schema.add_table(table_clone);
                }
            }
        }

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
        use crate::btree::BTree;
        use crate::records::RecordView;
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

        let has_expressions = column_defs.iter().any(|cd| cd.is_expression());

        let mut index_def = crate::schema::table::IndexDef::new_expression(
            index_name.to_string(),
            column_defs.clone(),
            create.unique,
            crate::schema::table::IndexType::BTree,
        );

        if let Some(where_clause) = create.where_clause {
            index_def = index_def.with_where_clause(Self::format_expr(where_clause));
        }

        let mut catalog_guard = self.catalog.write();
        let catalog = catalog_guard.as_mut().unwrap();

        let table_def = catalog.resolve_table(table_name)?;
        let columns = table_def.columns().to_vec();
        let schema = create_record_schema(&columns);

        let index_col_indices: Vec<usize> = column_defs
            .iter()
            .filter_map(|cd| {
                cd.as_column().and_then(|col_name| {
                    columns
                        .iter()
                        .position(|c| c.name().eq_ignore_ascii_case(col_name))
                })
            })
            .collect();

        let can_populate = !has_expressions && index_col_indices.len() == column_defs.len();

        if let Some(schema_obj) = catalog.get_schema_mut(schema_name) {
            if let Some(table) = schema_obj.get_table(table_name) {
                let table_with_index = table.clone().with_index(index_def);
                schema_obj.remove_table(table_name);
                schema_obj.add_table(table_with_index);
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
        BTree::create(index_storage, 1)?;

        if can_populate {
            let root_page = 1u32;

            let index_entries: Vec<(SmallVec<[u8; 64]>, u64)> = {
                let table_storage = file_manager.table_data_mut(schema_name, table_name)?;
                let table_btree = BTree::new(table_storage, root_page)?;
                let mut cursor = table_btree.cursor_first()?;

                let mut entries = Vec::new();
                let mut key_buf: SmallVec<[u8; 64]> = SmallVec::new();

                if cursor.valid() {
                    loop {
                        let row_key = cursor.key()?;
                        let row_data = cursor.value()?;

                        let row_id = u64::from_be_bytes(
                            row_key
                                .try_into()
                                .wrap_err("Invalid row key length in table")?,
                        );

                        let record = RecordView::new(row_data, &schema)?;

                        key_buf.clear();
                        let mut all_non_null = true;
                        for &col_idx in &index_col_indices {
                            let col_def = &columns[col_idx];
                            let value = OwnedValue::from_record_column(
                                &record,
                                col_idx,
                                col_def.data_type(),
                            )?;
                            if value.is_null() {
                                all_non_null = false;
                                break;
                            }
                            Self::encode_value_as_key(&value, &mut key_buf);
                        }

                        if all_non_null {
                            entries.push((key_buf.clone(), row_id));
                        }

                        if !cursor.advance()? {
                            break;
                        }
                    }
                }
                entries
            };

            for (mut key_buf, row_id) in index_entries {
                let row_id_bytes = row_id.to_be_bytes();
                if !create.unique {
                    key_buf.extend_from_slice(&row_id_bytes);
                }
                let index_storage =
                    file_manager.index_data_mut(schema_name, table_name, index_name)?;
                let mut index_btree = BTree::new(index_storage, root_page)?;
                index_btree.insert(&key_buf, &row_id_bytes)?;
            }
        }

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
        let has_toast = table_def.has_toast();

        let unique_columns: Vec<(usize, String, bool, bool)> = columns
            .iter()
            .enumerate()
            .filter_map(|(idx, col)| {
                let is_pk = col.has_constraint(&Constraint::PrimaryKey);
                let is_unique = col.has_constraint(&Constraint::Unique);
                let is_auto_increment = col.has_constraint(&Constraint::AutoIncrement);
                if is_pk || is_unique {
                    let index_name = if is_pk {
                        format!("{}_pkey", col.name())
                    } else {
                        format!("{}_key", col.name())
                    };
                    Some((idx, index_name, is_pk, is_auto_increment))
                } else {
                    None
                }
            })
            .collect();

        let unique_column_index_names: HashSet<&str> = unique_columns
            .iter()
            .map(|(_, name, _, _)| name.as_str())
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

        let insert_col_indices: Option<Vec<usize>> = insert.columns.map(|cols| {
            cols.iter()
                .filter_map(|col_name| {
                    columns.iter().position(|c| c.name().eq_ignore_ascii_case(col_name))
                })
                .collect()
        });

        let auto_increment_col_idx: Option<usize> = columns
            .iter()
            .position(|c| c.has_constraint(&Constraint::AutoIncrement));

        let rows_to_insert: Vec<Vec<OwnedValue>> = match &insert.source {
            crate::sql::ast::InsertSource::Values(values) => {
                let mut result = Vec::with_capacity(values.len());
                for row_exprs in values.iter() {
                    let mut row = vec![OwnedValue::Null; columns.len()];

                    if let Some(ref col_indices) = insert_col_indices {
                        for (val_idx, &col_idx) in col_indices.iter().enumerate() {
                            if let Some(expr) = row_exprs.get(val_idx) {
                                let data_type = column_types.get(col_idx);
                                row[col_idx] = Database::eval_literal_with_type(expr, data_type)?;
                            }
                        }
                    } else {
                        for (idx, expr) in row_exprs.iter().enumerate() {
                            if idx < columns.len() {
                                let data_type = column_types.get(idx);
                                row[idx] = Database::eval_literal_with_type(expr, data_type)?;
                            }
                        }
                    }

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

        let mut auto_increment_current = if auto_increment_col_idx.is_some() {
            let storage = file_manager.table_data_mut(schema_name, table_name)?;
            let page = storage.page(0)?;
            let header = TableFileHeader::from_bytes(page)?;
            header.auto_increment()
        } else {
            0
        };
        let mut auto_increment_max = auto_increment_current;

        let mut rightmost_hint: Option<u32> = {
            let storage = file_manager.table_data_mut(schema_name, table_name)?;
            let page = storage.page(0)?;
            let header = TableFileHeader::from_bytes(page)?;
            let hint = header.rightmost_hint();
            if hint > 0 { Some(hint) } else { None }
        };

        let mut toast_rightmost_hints: SmallVec<[Option<u32>; 8]> = SmallVec::new();
        let toastable_col_indices: SmallVec<[usize; 8]> = if has_toast {
            toast_rightmost_hints.resize(columns.len(), None);
            columns.iter()
                .enumerate()
                .filter(|(_, col)| col.data_type().is_toastable())
                .map(|(idx, _)| idx)
                .collect()
        } else {
            SmallVec::new()
        };

        let table_file_key = crate::storage::FileManager::make_table_key(schema_name, table_name);
        let mut record_builder = crate::records::RecordBuilder::new(&schema);

        // Pre-compute toast table info for performance
        use crate::storage::FileKey;
        let toast_file_key: Option<FileKey>;
        let mut toast_table_id: u32 = 0;
        let mut toast_root_page: u32 = 1;
        let mut toast_initial_root_page: u32 = 1;
        let mut toast_rightmost_hint: Option<u32> = None;
        if has_toast {
            let toast_table_name_owned = crate::storage::toast::toast_table_name(table_name);
            toast_file_key = Some(crate::storage::FileManager::make_table_key(schema_name, &toast_table_name_owned));
            let toast_storage = file_manager.table_data_mut(schema_name, &toast_table_name_owned)?;
            let page0 = toast_storage.page(0)?;
            let header = crate::storage::TableFileHeader::from_bytes(page0)?;
            toast_table_id = header.table_id() as u32;
            toast_root_page = header.root_page();
            toast_initial_root_page = toast_root_page;
            let hint = header.rightmost_hint();
            toast_rightmost_hint = if hint > 0 { Some(hint) } else { None };
        } else {
            toast_file_key = None;
        }

        let unique_column_keys: Vec<(usize, FileKey, bool, bool)> = unique_columns
            .iter()
            .filter(|(_, _, _, is_auto_increment)| !is_auto_increment)
            .filter_map(|(col_idx, index_name, is_pk, _is_auto_increment)| {
                if file_manager.index_exists(schema_name, table_name, index_name) {
                    // Pre-open the index to ensure it's in the cache
                    let _ = file_manager.index_data_mut(schema_name, table_name, index_name);
                    let key = crate::storage::FileManager::make_index_key(schema_name, table_name, index_name);
                    Some((*col_idx, key, *is_pk, false))
                } else {
                    None
                }
            })
            .collect();

        for row_values in rows_to_insert.iter() {
            let mut values: Vec<OwnedValue> = row_values.clone();

            if let Some(auto_col_idx) = auto_increment_col_idx {
                if values.get(auto_col_idx).is_none_or(|v| v.is_null()) {
                    auto_increment_current = auto_increment_current.checked_add(1).ok_or_else(|| {
                        eyre::eyre!("auto_increment overflow: exceeded maximum value")
                    })?;
                    values[auto_col_idx] = OwnedValue::Int(auto_increment_current as i64);
                    if auto_increment_current > auto_increment_max {
                        auto_increment_max = auto_increment_current;
                    }
                } else if let Some(OwnedValue::Int(provided_val)) = values.get(auto_col_idx) {
                    if *provided_val < 0 {
                        bail!(
                            "auto_increment column cannot have negative value: {}",
                            provided_val
                        );
                    }
                    if (*provided_val as u64) > auto_increment_max {
                        auto_increment_max = *provided_val as u64;
                    }
                }
            }

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

            // Only check FK constraints if foreign_keys_enabled is true
            let fk_enabled = self.foreign_keys_enabled.load(Ordering::Acquire);
            if fk_enabled && !fk_constraints.is_empty() {
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

            for (col_idx, index_key, is_pk, _) in &unique_column_keys {
                if let Some(value) = values.get(*col_idx) {
                    if value.is_null() {
                        continue;
                    }

                    let index_storage = file_manager.index_data_mut_with_key(index_key)
                        .ok_or_else(|| eyre::eyre!("index storage not found"))?;
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

            let row_id = self.next_row_id.fetch_add(1, Ordering::Relaxed);
            let row_key = Self::generate_row_key(row_id);

            if !toastable_col_indices.is_empty() {
                use crate::storage::toast::{needs_toast, make_chunk_key, ToastPointer, TOAST_CHUNK_SIZE};

                for &col_idx in &toastable_col_indices {
                    let value = &values[col_idx];
                    let data = match value {
                        OwnedValue::Text(s) => Some(s.as_bytes()),
                        OwnedValue::Blob(b) => Some(b.as_slice()),
                        _ => None,
                    };

                    if let Some(data) = data {
                        if needs_toast(data) {
                            let toast_key = toast_file_key.as_ref().unwrap();
                            let toast_storage = file_manager.table_data_mut_with_key(toast_key)
                                .ok_or_else(|| eyre::eyre!("toast storage not found"))?;

                            let pointer = ToastPointer::new(row_id, col_idx as u16, data.len() as u64);
                            let chunk_id = pointer.chunk_id;

                            let (new_hint, new_root) = if wal_enabled {
                                let mut wal_storage =
                                    WalStoragePerTable::new(toast_storage, &self.dirty_tracker, toast_table_id);
                                let mut btree = BTree::with_rightmost_hint(&mut wal_storage, toast_root_page, toast_rightmost_hint)?;
                                for (seq, chunk) in data.chunks(TOAST_CHUNK_SIZE).enumerate() {
                                    let chunk_key = make_chunk_key(chunk_id, seq as u32);
                                    btree.insert(&chunk_key, chunk)?;
                                }
                                (btree.rightmost_hint(), btree.root_page())
                            } else {
                                let mut btree = BTree::with_rightmost_hint(toast_storage, toast_root_page, toast_rightmost_hint)?;
                                for (seq, chunk) in data.chunks(TOAST_CHUNK_SIZE).enumerate() {
                                    let chunk_key = make_chunk_key(chunk_id, seq as u32);
                                    btree.insert(&chunk_key, chunk)?;
                                }
                                (btree.rightmost_hint(), btree.root_page())
                            };

                            toast_rightmost_hint = new_hint;
                            if new_root != toast_root_page {
                                toast_root_page = new_root;
                            }

                            values[col_idx] = OwnedValue::Blob(pointer.encode().to_vec());
                        }
                    }
                }
            }

            let table_storage = file_manager.table_data_mut_with_key(&table_file_key)
                .ok_or_else(|| eyre::eyre!("table storage not found in cache"))?;

            let record_data = OwnedValue::build_record_with_builder(&values, &mut record_builder)?;

            if wal_enabled {
                let mut wal_storage =
                    WalStoragePerTable::new(table_storage, &self.dirty_tracker, table_id as u32);
                let mut btree = BTree::with_rightmost_hint(&mut wal_storage, root_page, rightmost_hint)?;
                btree.insert(&row_key, &record_data)?;
                rightmost_hint = btree.rightmost_hint();
            } else {
                let mut btree = BTree::with_rightmost_hint(table_storage, root_page, rightmost_hint)?;
                btree.insert(&row_key, &record_data)?;
                rightmost_hint = btree.rightmost_hint();
            }

            {
                let mut active_txn = self.active_txn.lock();
                if let Some(ref mut txn) = *active_txn {
                    txn.add_write_entry(WriteEntry {
                        table_id: table_id as u32,
                        key: row_key.to_vec(),
                        page_id: 0,
                        offset: 0,
                        undo_page_id: None,
                        undo_offset: None,
                        is_insert: true,
                    });
                }
            }

            for (col_idx, index_key, _, _) in &unique_column_keys {
                if let Some(value) = values.get(*col_idx) {
                    if value.is_null() {
                        continue;
                    }

                    let index_storage = file_manager.index_data_mut_with_key(index_key)
                        .ok_or_else(|| eyre::eyre!("index storage not found"))?;

                    key_buf.clear();
                    Self::encode_value_as_key(value, &mut key_buf);

                    let row_id_bytes = row_id.to_be_bytes();

                    let mut index_btree = BTree::new(index_storage, root_page)?;
                    index_btree.insert(&key_buf, &row_id_bytes)?;
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

        if auto_increment_col_idx.is_some() && auto_increment_max > 0 {
            let storage = file_manager.table_data_mut(schema_name, table_name)?;
            let page = storage.page_mut(0)?;
            let header = TableFileHeader::from_bytes_mut(page)?;
            if auto_increment_max > header.auto_increment() {
                header.set_auto_increment(auto_increment_max);
            }
        }

        if let Some(hint) = rightmost_hint {
            let storage = file_manager.table_data_mut(schema_name, table_name)?;
            let page = storage.page_mut(0)?;
            let header = TableFileHeader::from_bytes_mut(page)?;
            header.set_rightmost_hint(hint);
        }

        if count > 0 {
            let storage = file_manager.table_data_mut(schema_name, table_name)?;
            let page = storage.page_mut(0)?;
            let header = TableFileHeader::from_bytes_mut(page)?;
            let new_row_count = header.row_count().saturating_add(count as u64);
            header.set_row_count(new_row_count);
        }

        if has_toast && toast_root_page != toast_initial_root_page {
            let toast_table_name_owned = crate::storage::toast::toast_table_name(table_name);
            let toast_storage = file_manager.table_data_mut(schema_name, &toast_table_name_owned)?;
            let page0 = toast_storage.page_mut(0)?;
            let header = crate::storage::TableFileHeader::from_bytes_mut(page0)?;
            header.set_root_page(toast_root_page);
        }

        // Flush WAL in autocommit mode; deferred to commit in explicit transactions
        self.flush_wal_if_autocommit(file_manager, schema_name, table_name, table_id as u32)?;

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
            OwnedValue::ToastPointer(b) => key::encode_blob(b, buf),
        }
    }

    fn execute_update(
        &self,
        update: &crate::sql::ast::UpdateStmt<'_>,
        arena: &Bump,
    ) -> Result<ExecuteResult> {
        use crate::btree::BTree;
        use crate::sql::decoder::RecordDecoder;

        self.ensure_catalog()?;
        self.ensure_file_manager()?;

        let catalog_guard = self.catalog.read();
        let catalog = catalog_guard.as_ref().unwrap();

        let schema_name = update.table.schema.unwrap_or("root");
        let table_name = update.table.name;
        let table_alias = update.table.alias;

        let table_def = catalog.resolve_table(table_name)?.clone();
        let table_id = table_def.id();
        let columns = table_def.columns().to_vec();
        let has_toast = table_def.has_toast();

        #[allow(clippy::type_complexity)]
        let from_tables_data: Option<Vec<(
            String,
            String,
            Option<&str>,
            Vec<crate::schema::table::ColumnDef>,
        )>> = if let Some(from_clause) = update.from {
            let mut tables = Vec::new();
            Self::extract_tables_from_clause(*from_clause, catalog, &mut tables)?;
            Some(tables)
        } else {
            None
        };

        drop(catalog_guard);

        let schema = create_record_schema(&columns);

        if let Some(from_tables) = from_tables_data {
            return self.execute_update_with_from(
                update,
                arena,
                schema_name,
                table_name,
                table_alias,
                &table_def,
                table_id as usize,
                &columns,
                &schema,
                from_tables,
            );
        }

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

        let column_types: Vec<crate::records::types::DataType> =
            columns.iter().map(|c| c.data_type()).collect();
        let decoder = crate::sql::decoder::SimpleDecoder::new(column_types);

        let root_page = 1u32;
        let btree = BTree::new(storage, root_page)?;
        let mut cursor = btree.cursor_first()?;

        #[allow(clippy::type_complexity)]
        let mut rows_to_update: Vec<(Vec<u8>, Vec<u8>, Vec<OwnedValue>, Vec<(usize, OwnedValue)>)> = Vec::new();

        while cursor.valid() {
            let key = cursor.key()?;
            let value = cursor.value()?;

            let values = decoder.decode(key, value)?;
            let mut row_values: Vec<OwnedValue> =
                values.into_iter().map(OwnedValue::from).collect();

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

                let mut old_toast_values: Vec<(usize, OwnedValue)> = Vec::new();
                for (col_idx, value_expr) in &assignment_indices {
                    if let OwnedValue::ToastPointer(_) = &row_values[*col_idx] {
                        old_toast_values.push((*col_idx, row_values[*col_idx].clone()));
                    }
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

                rows_to_update.push((key.to_vec(), old_value, row_values, old_toast_values));
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

            for (update_key, _old_value, updated_values, _old_toast) in &rows_to_update {
                while check_cursor.valid() {
                    let existing_key = check_cursor.key()?;

                    if existing_key != update_key.as_slice() {
                        let existing_value = check_cursor.value()?;
                        let existing_values_raw = decoder.decode(existing_key, existing_value)?;
                        let existing_values: Vec<OwnedValue> =
                            existing_values_raw.into_iter().map(OwnedValue::from).collect();

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
                .map(|(_key, _old_value, updated_values, _old_toast)| {
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

        let mut processed_rows: Vec<(Vec<u8>, Vec<OwnedValue>)> = Vec::with_capacity(rows_to_update.len());

        // WAL must be initialized before TOAST operations because toast_value() uses
        // WalStoragePerTable when WAL is enabled. If we delay this, TOAST chunks
        // would bypass WAL and be lost on crash recovery.
        let wal_enabled = self.wal_enabled.load(std::sync::atomic::Ordering::Acquire);
        if wal_enabled {
            self.ensure_wal()?;
        }

        if has_toast {
            use crate::storage::toast::ToastPointer;
            for (key, _old_value, mut updated_values, old_toast_values) in rows_to_update.clone() {
                for (_col_idx, old_val) in old_toast_values {
                    if let OwnedValue::ToastPointer(ptr) = old_val {
                        if let Ok(pointer) = ToastPointer::decode(&ptr) {
                            let _ = self.delete_toast_chunks(
                                file_manager,
                                schema_name,
                                table_name,
                                pointer.row_id(),
                                pointer.column_index(),
                                pointer.total_size,
                            );
                        }
                    }
                }

                let pk_value = if let Some(pk_idx) = columns.iter().position(|c| c.has_constraint(&crate::schema::table::Constraint::PrimaryKey)) {
                    if let OwnedValue::Int(id) = &updated_values[pk_idx] {
                        *id as u64
                    } else {
                        0
                    }
                } else {
                    0
                };

                for (col_idx, val) in updated_values.iter_mut().enumerate() {
                    if columns[col_idx].data_type().is_toastable() {
                        let needs_toast = match val {
                            OwnedValue::Text(s) => crate::storage::toast::needs_toast(s.as_bytes()),
                            OwnedValue::Blob(b) => crate::storage::toast::needs_toast(b),
                            _ => false,
                        };
                        if needs_toast {
                            let data = match val {
                                OwnedValue::Text(s) => s.as_bytes().to_vec(),
                                OwnedValue::Blob(b) => b.clone(),
                                _ => continue,
                            };
                            let (pointer, _) = self.toast_value(
                                file_manager,
                                schema_name,
                                table_name,
                                pk_value,
                                col_idx as u16,
                                &data,
                                wal_enabled,
                                None,
                            )?;
                            *val = OwnedValue::ToastPointer(pointer);
                        }
                    }
                }
                processed_rows.push((key, updated_values));
            }
        } else {
            for (key, _old_value, updated_values, _old_toast) in rows_to_update.clone() {
                processed_rows.push((key, updated_values));
            }
        }

        let storage = file_manager.table_data_mut(schema_name, table_name)?;

        with_btree_storage!(wal_enabled, storage, &self.dirty_tracker, table_id as u32, root_page, |btree_mut: &mut crate::btree::BTree<_>| {
            for (key, updated_values) in &processed_rows {
                btree_mut.delete(key)?;
                let record_data = OwnedValue::build_record_from_values(updated_values, &schema)?;
                btree_mut.insert(key, &record_data)?;
            }
            Ok::<_, eyre::Report>(())
        });

        // Flush WAL in autocommit mode; deferred to commit in explicit transactions
        self.flush_wal_if_autocommit(file_manager, schema_name, table_name, table_id as u32)?;

        drop(file_manager_guard);

        {
            let mut active_txn = self.active_txn.lock();
            if let Some(ref mut txn) = *active_txn {
                for (key, old_value, _updated_values, _old_toast) in rows_to_update {
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

    #[allow(clippy::too_many_arguments)]
    fn execute_update_with_from(
        &self,
        update: &crate::sql::ast::UpdateStmt<'_>,
        arena: &Bump,
        schema_name: &str,
        table_name: &str,
        table_alias: Option<&str>,
        table_def: &crate::schema::table::TableDef,
        table_id: usize,
        columns: &[crate::schema::table::ColumnDef],
        schema: &crate::records::Schema,
        from_tables: Vec<(
            String,
            String,
            Option<&str>,
            Vec<crate::schema::table::ColumnDef>,
        )>,
    ) -> Result<ExecuteResult> {
        use crate::btree::BTree;
        use crate::records::RecordView;
        use crate::sql::decoder::RecordDecoder;
        use crate::sql::executor::ExecutorRow;
        use std::borrow::Cow;

        let mut combined_column_map: Vec<(String, usize)> = Vec::new();
        for (idx, col) in columns.iter().enumerate() {
            combined_column_map.push((col.name().to_string(), idx));
            combined_column_map.push((
                format!("{}.{}", table_name, col.name()),
                idx,
            ));
            if let Some(alias) = table_alias {
                combined_column_map.push((
                    format!("{}.{}", alias, col.name()),
                    idx,
                ));
            }
        }

        let mut current_col_offset = columns.len();
        let mut from_schemas: Vec<crate::records::Schema> = Vec::new();
        for (_, from_table_name, from_alias, from_columns) in &from_tables {
            for (idx, col) in from_columns.iter().enumerate() {
                combined_column_map.push((col.name().to_string(), current_col_offset + idx));
                combined_column_map.push((
                    format!("{}.{}", from_table_name, col.name()),
                    current_col_offset + idx,
                ));
                if let Some(alias) = from_alias {
                    combined_column_map.push((
                        format!("{}.{}", alias, col.name()),
                        current_col_offset + idx,
                    ));
                }
            }
            current_col_offset += from_columns.len();
            from_schemas.push(create_record_schema(from_columns));
        }

        let predicate = update
            .where_clause
            .map(|expr| CompiledPredicate::new(expr, combined_column_map.clone()));

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

        let mut all_from_rows: Vec<Vec<Vec<OwnedValue>>> = Vec::new();
        for (i, (from_schema_name, from_table_name, _, from_columns)) in from_tables.iter().enumerate() {
            let from_storage = file_manager.table_data_mut(from_schema_name, from_table_name)?;
            let from_btree = BTree::new(from_storage, 1u32)?;
            let mut from_cursor = from_btree.cursor_first()?;

            let mut table_rows: Vec<Vec<OwnedValue>> = Vec::new();
            while from_cursor.valid() {
                let value = from_cursor.value()?;
                let record = RecordView::new(value, &from_schemas[i])?;
                let row_values = OwnedValue::extract_row_from_record(&record, from_columns)?;
                table_rows.push(row_values);
                from_cursor.advance()?;
            }
            all_from_rows.push(table_rows);
        }

        let combined_from_rows = Self::cartesian_product(&all_from_rows);

        let storage = file_manager.table_data_mut(schema_name, table_name)?;
        let root_page = 1u32;
        let btree = BTree::new(storage, root_page)?;
        let mut cursor = btree.cursor_first()?;

        let column_types: Vec<crate::records::types::DataType> =
            columns.iter().map(|c| c.data_type()).collect();
        let decoder = crate::sql::decoder::SimpleDecoder::new(column_types);

        let mut rows_to_update: Vec<(Vec<u8>, Vec<u8>, Vec<OwnedValue>)> = Vec::new();
        let mut updated_keys: HashSet<Vec<u8>> = HashSet::new();

        while cursor.valid() {
            let key = cursor.key()?;
            let value = cursor.value()?;

            let values = decoder.decode(key, value)?;
            let target_row_values: Vec<OwnedValue> =
                values.into_iter().map(OwnedValue::from).collect();

            for from_row in &combined_from_rows {
                let mut combined_values: Vec<Value<'_>> = Vec::with_capacity(
                    target_row_values.len() + from_row.len(),
                );

                for val in &target_row_values {
                    combined_values.push(match val {
                        OwnedValue::Null => Value::Null,
                        OwnedValue::Bool(b) => Value::Int(if *b { 1 } else { 0 }),
                        OwnedValue::Int(i) => Value::Int(*i),
                        OwnedValue::Float(f) => Value::Float(*f),
                        OwnedValue::Text(s) => Value::Text(Cow::Borrowed(s.as_str())),
                        OwnedValue::Blob(b) => Value::Blob(Cow::Borrowed(b.as_slice())),
                        _ => Value::Null,
                    });
                }

                for val in from_row {
                    combined_values.push(match val {
                        OwnedValue::Null => Value::Null,
                        OwnedValue::Bool(b) => Value::Int(if *b { 1 } else { 0 }),
                        OwnedValue::Int(i) => Value::Int(*i),
                        OwnedValue::Float(f) => Value::Float(*f),
                        OwnedValue::Text(s) => Value::Text(Cow::Borrowed(s.as_str())),
                        OwnedValue::Blob(b) => Value::Blob(Cow::Borrowed(b.as_slice())),
                        _ => Value::Null,
                    });
                }

                let values_slice = arena.alloc_slice_fill_iter(combined_values.into_iter());
                let exec_row = ExecutorRow::new(values_slice);

                let should_update = if let Some(ref pred) = predicate {
                    pred.evaluate(&exec_row)
                } else {
                    true
                };

                if should_update && !updated_keys.contains(&key.to_vec()) {
                    let old_value = value.to_vec();
                    let mut row_values = target_row_values.clone();

                    for (col_idx, value_expr) in &assignment_indices {
                        let new_value = self.eval_expr_with_row(
                            value_expr,
                            &exec_row,
                            &combined_column_map,
                        )?;
                        row_values[*col_idx] = new_value;
                    }

                    let validator = crate::constraints::ConstraintValidator::new(table_def);
                    validator.validate_update(&row_values)?;

                    for (col_idx, col) in columns.iter().enumerate() {
                        for constraint in col.constraints() {
                            if let crate::schema::table::Constraint::Check(expr_str) = constraint {
                                let col_value = row_values.get(col_idx);
                                if !Self::evaluate_check_expression(
                                    expr_str,
                                    col.name(),
                                    col_value,
                                ) {
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

                    updated_keys.insert(key.to_vec());
                    rows_to_update.push((key.to_vec(), old_value, row_values));
                }
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
                        let existing_record = RecordView::new(existing_value, schema)?;
                        let existing_values =
                            OwnedValue::extract_row_from_record(&existing_record, columns)?;

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

        // Check if WAL is enabled
        let wal_enabled = self.wal_enabled.load(std::sync::atomic::Ordering::Acquire);
        if wal_enabled {
            self.ensure_wal()?;
        }

        let storage = file_manager.table_data_mut(schema_name, table_name)?;

        with_btree_storage!(wal_enabled, storage, &self.dirty_tracker, table_id as u32, root_page, |btree_mut: &mut crate::btree::BTree<_>| {
            for (key, _old_value, updated_values) in &rows_to_update {
                btree_mut.delete(key)?;
                let record_data = OwnedValue::build_record_from_values(updated_values, schema)?;
                btree_mut.insert(key, &record_data)?;
            }
            Ok::<_, eyre::Report>(())
        });

        // Flush WAL in autocommit mode; deferred to commit in explicit transactions
        self.flush_wal_if_autocommit(file_manager, schema_name, table_name, table_id as u32)?;

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

    fn cartesian_product(tables: &[Vec<Vec<OwnedValue>>]) -> Vec<Vec<OwnedValue>> {
        if tables.is_empty() {
            return vec![vec![]];
        }

        let mut result: Vec<Vec<OwnedValue>> = vec![vec![]];

        for table_rows in tables {
            let mut new_result: Vec<Vec<OwnedValue>> = Vec::new();
            for existing in &result {
                for row in table_rows {
                    let mut combined = existing.clone();
                    combined.extend(row.clone());
                    new_result.push(combined);
                }
            }
            result = new_result;
        }

        result
    }

    fn eval_expr_with_row(
        &self,
        expr: &crate::sql::ast::Expr<'_>,
        row: &ExecutorRow<'_>,
        column_map: &[(String, usize)],
    ) -> Result<OwnedValue> {
        use crate::sql::ast::{BinaryOperator, Expr, UnaryOperator};

        match expr {
            Expr::Literal(_) => Self::eval_literal(expr),
            Expr::Column(col_ref) => {
                let col_name = if let Some(table) = col_ref.table {
                    format!("{}.{}", table, col_ref.column)
                } else {
                    col_ref.column.to_string()
                };

                let col_idx = column_map
                    .iter()
                    .find(|(name, _)| name.eq_ignore_ascii_case(&col_name))
                    .map(|(_, idx)| *idx)
                    .or_else(|| {
                        column_map
                            .iter()
                            .find(|(name, _)| name.eq_ignore_ascii_case(col_ref.column))
                            .map(|(_, idx)| *idx)
                    });

                if let Some(idx) = col_idx {
                    if let Some(val) = row.get(idx) {
                        Ok(OwnedValue::from(val))
                    } else {
                        Ok(OwnedValue::Null)
                    }
                } else {
                    bail!(
                        "column '{}' not found in UPDATE...FROM context",
                        col_name
                    )
                }
            }
            Expr::BinaryOp { left, op, right } => {
                let left_val = self.eval_expr_with_row(left, row, column_map)?;
                let right_val = self.eval_expr_with_row(right, row, column_map)?;

                match op {
                    BinaryOperator::Plus => match (&left_val, &right_val) {
                        (OwnedValue::Int(a), OwnedValue::Int(b)) => Ok(OwnedValue::Int(a + b)),
                        (OwnedValue::Float(a), OwnedValue::Float(b)) => {
                            Ok(OwnedValue::Float(a + b))
                        }
                        (OwnedValue::Int(a), OwnedValue::Float(b)) => {
                            Ok(OwnedValue::Float(*a as f64 + b))
                        }
                        (OwnedValue::Float(a), OwnedValue::Int(b)) => {
                            Ok(OwnedValue::Float(a + *b as f64))
                        }
                        _ => bail!("unsupported types for addition"),
                    },
                    BinaryOperator::Minus => match (&left_val, &right_val) {
                        (OwnedValue::Int(a), OwnedValue::Int(b)) => Ok(OwnedValue::Int(a - b)),
                        (OwnedValue::Float(a), OwnedValue::Float(b)) => {
                            Ok(OwnedValue::Float(a - b))
                        }
                        (OwnedValue::Int(a), OwnedValue::Float(b)) => {
                            Ok(OwnedValue::Float(*a as f64 - b))
                        }
                        (OwnedValue::Float(a), OwnedValue::Int(b)) => {
                            Ok(OwnedValue::Float(a - *b as f64))
                        }
                        _ => bail!("unsupported types for subtraction"),
                    },
                    BinaryOperator::Multiply => match (&left_val, &right_val) {
                        (OwnedValue::Int(a), OwnedValue::Int(b)) => Ok(OwnedValue::Int(a * b)),
                        (OwnedValue::Float(a), OwnedValue::Float(b)) => {
                            Ok(OwnedValue::Float(a * b))
                        }
                        (OwnedValue::Int(a), OwnedValue::Float(b)) => {
                            Ok(OwnedValue::Float(*a as f64 * b))
                        }
                        (OwnedValue::Float(a), OwnedValue::Int(b)) => {
                            Ok(OwnedValue::Float(a * *b as f64))
                        }
                        _ => bail!("unsupported types for multiplication"),
                    },
                    BinaryOperator::Divide => match (&left_val, &right_val) {
                        (OwnedValue::Int(a), OwnedValue::Int(b)) if *b != 0 => {
                            Ok(OwnedValue::Int(a / b))
                        }
                        (OwnedValue::Float(a), OwnedValue::Float(b)) if *b != 0.0 => {
                            Ok(OwnedValue::Float(a / b))
                        }
                        (OwnedValue::Int(a), OwnedValue::Float(b)) if *b != 0.0 => {
                            Ok(OwnedValue::Float(*a as f64 / b))
                        }
                        (OwnedValue::Float(a), OwnedValue::Int(b)) if *b != 0 => {
                            Ok(OwnedValue::Float(a / *b as f64))
                        }
                        _ => bail!("division by zero or unsupported types"),
                    },
                    BinaryOperator::Concat => match (&left_val, &right_val) {
                        (OwnedValue::Text(a), OwnedValue::Text(b)) => {
                            Ok(OwnedValue::Text(format!("{}{}", a, b)))
                        }
                        _ => bail!("unsupported types for concatenation"),
                    },
                    _ => bail!("unsupported binary operator in UPDATE...FROM SET expression"),
                }
            }
            Expr::UnaryOp { op, expr: inner } => {
                let inner_val = self.eval_expr_with_row(inner, row, column_map)?;
                match (op, inner_val) {
                    (UnaryOperator::Minus, OwnedValue::Int(i)) => Ok(OwnedValue::Int(-i)),
                    (UnaryOperator::Minus, OwnedValue::Float(f)) => Ok(OwnedValue::Float(-f)),
                    (UnaryOperator::Plus, val) => Ok(val),
                    (UnaryOperator::Not, OwnedValue::Bool(b)) => Ok(OwnedValue::Bool(!b)),
                    _ => bail!("unsupported unary operation"),
                }
            }
            _ => Self::eval_literal(expr),
        }
    }

    fn extract_tables_from_clause<'a>(
        from_clause: crate::sql::ast::FromClause<'a>,
        catalog: &Catalog,
        tables: &mut Vec<(
            String,
            String,
            Option<&'a str>,
            Vec<crate::schema::table::ColumnDef>,
        )>,
    ) -> Result<()> {
        use crate::sql::ast::FromClause;

        match from_clause {
            FromClause::Table(table_ref) => {
                let schema = table_ref.schema.unwrap_or("root");
                let table_name = table_ref.name;
                let alias = table_ref.alias;
                let table_def = catalog.resolve_table(table_name)?;
                let columns = table_def.columns().to_vec();
                tables.push((schema.to_string(), table_name.to_string(), alias, columns));
            }
            FromClause::Join(join_clause) => {
                Self::extract_tables_from_clause(*join_clause.left, catalog, tables)?;
                Self::extract_tables_from_clause(*join_clause.right, catalog, tables)?;
            }
            FromClause::Subquery { .. } => {
                bail!("UPDATE...FROM does not support subqueries in FROM clause")
            }
            FromClause::Lateral { .. } => {
                bail!("UPDATE...FROM does not support LATERAL in FROM clause")
            }
        }
        Ok(())
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
        let has_toast = table_def.has_toast();

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

        if has_toast {
            use crate::storage::toast::ToastPointer;
            for (_key, _old_value, row_values) in &rows_to_delete {
                for val in row_values.iter() {
                    if let OwnedValue::ToastPointer(ptr) = val {
                        if let Ok(pointer) = ToastPointer::decode(ptr) {
                            let _ = self.delete_toast_chunks(
                                file_manager,
                                schema_name,
                                table_name,
                                pointer.row_id(),
                                pointer.column_index(),
                                pointer.total_size,
                            );
                        }
                    }
                }
            }
        }

        let wal_enabled = self.wal_enabled.load(std::sync::atomic::Ordering::Acquire);
        if wal_enabled {
            self.ensure_wal()?;
        }

        let storage = file_manager.table_data_mut(schema_name, table_name)?;

        with_btree_storage!(wal_enabled, storage, &self.dirty_tracker, table_id as u32, root_page, |btree_mut: &mut crate::btree::BTree<_>| {
            for (key, _old_value, _row_values) in &rows_to_delete {
                btree_mut.delete(key)?;
            }
            Ok::<_, eyre::Report>(())
        });

        // Flush WAL in autocommit mode; deferred to commit in explicit transactions
        self.flush_wal_if_autocommit(file_manager, schema_name, table_name, table_id as u32)?;

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

        if rows_affected > 0 {
            let mut file_manager_guard = self.file_manager.write();
            let file_manager = file_manager_guard.as_mut().unwrap();
            let storage = file_manager.table_data_mut(schema_name, table_name)?;
            let page = storage.page_mut(0)?;
            let header = TableFileHeader::from_bytes_mut(page)?;
            let new_row_count = header.row_count().saturating_sub(rows_affected as u64);
            header.set_row_count(new_row_count);
        }

        Ok(ExecuteResult::Delete {
            rows_affected,
            returned: returned_rows,
        })
    }

    fn execute_truncate(
        &self,
        truncate: &crate::sql::ast::TruncateStmt<'_>,
    ) -> Result<ExecuteResult> {
        use crate::btree::BTree;

        self.ensure_catalog()?;
        self.ensure_file_manager()?;

        let tables_info: Vec<(String, String)> = {
            let catalog_guard = self.catalog.read();
            let catalog = catalog_guard.as_ref().unwrap();

            let mut info = Vec::new();
            for table_ref in truncate.tables {
                let schema_name = table_ref.schema.unwrap_or("root");
                let table_name = table_ref.name;

                catalog.resolve_table(table_name)?;
                info.push((schema_name.to_string(), table_name.to_string()));
            }
            info
        };

        let mut total_rows_affected: usize = 0;

        for (schema_name, table_name) in &tables_info {
            let mut file_manager_guard = self.file_manager.write();
            let file_manager = file_manager_guard.as_mut().unwrap();
            let storage = file_manager.table_data_mut(schema_name, table_name)?;

            let root_page = 1u32;
            let btree = BTree::new(storage, root_page)?;
            let mut cursor = btree.cursor_first()?;

            let mut keys_to_delete: Vec<Vec<u8>> = Vec::new();
            while cursor.valid() {
                keys_to_delete.push(cursor.key()?.to_vec());
                cursor.advance()?;
            }

            let rows_affected = keys_to_delete.len();
            total_rows_affected += rows_affected;

            let mut btree_mut = BTree::new(storage, root_page)?;
            for key in &keys_to_delete {
                btree_mut.delete(key)?;
            }

            let page = storage.page_mut(0)?;
            let header = TableFileHeader::from_bytes_mut(page)?;

            header.set_row_count(0);

            if truncate.restart_identity {
                header.set_auto_increment(0);
            }

            storage.sync()?;

            let catalog_guard = self.catalog.read();
            let catalog = catalog_guard.as_ref().unwrap();
            let table_def = catalog.resolve_table(table_name)?;
            let indexes: Vec<String> = table_def.indexes().iter().map(|i| i.name().to_string()).collect();
            drop(catalog_guard);

            for index_name in indexes {
                if file_manager.index_exists(schema_name, table_name, &index_name) {
                    let index_storage = file_manager.index_data_mut(schema_name, table_name, &index_name)?;
                    let index_btree = BTree::new(index_storage, root_page)?;
                    let mut index_cursor = index_btree.cursor_first()?;

                    let mut index_keys_to_delete: Vec<Vec<u8>> = Vec::new();
                    while index_cursor.valid() {
                        index_keys_to_delete.push(index_cursor.key()?.to_vec());
                        index_cursor.advance()?;
                    }

                    let mut index_btree_mut = BTree::new(index_storage, root_page)?;
                    for key in &index_keys_to_delete {
                        index_btree_mut.delete(key)?;
                    }

                    index_storage.sync()?;
                }
            }
        }

        Ok(ExecuteResult::Truncate {
            rows_affected: total_rows_affected,
        })
    }

    fn execute_alter_table(
        &self,
        alter: &crate::sql::ast::AlterTableStmt<'_>,
    ) -> Result<ExecuteResult> {
        use crate::sql::ast::AlterTableAction;

        self.ensure_catalog()?;
        self.ensure_file_manager()?;

        let schema_name = alter.table.schema.unwrap_or("root");
        let table_name = alter.table.name;

        let action_desc = match &alter.action {
            AlterTableAction::RenameTable(new_name) => {
                {
                    let mut file_manager_guard = self.file_manager.write();
                    let file_manager = file_manager_guard.as_mut().unwrap();
                    file_manager.rename_table(schema_name, table_name, new_name)?;
                }

                {
                    let mut catalog_guard = self.catalog.write();
                    let catalog = catalog_guard.as_mut().unwrap();
                    let schema = catalog
                        .get_schema_mut(schema_name)
                        .ok_or_else(|| eyre::eyre!("schema '{}' not found", schema_name))?;
                    if !schema.table_exists(table_name) {
                        bail!("table '{}' not found in schema '{}'", table_name, schema_name);
                    }
                    schema.rename_table(table_name, new_name);
                }

                format!("renamed table to '{}'", new_name)
            }
            AlterTableAction::RenameColumn { old_name, new_name } => {
                let mut catalog_guard = self.catalog.write();
                let catalog = catalog_guard.as_mut().unwrap();
                let schema = catalog
                    .get_schema_mut(schema_name)
                    .ok_or_else(|| eyre::eyre!("schema '{}' not found", schema_name))?;
                if !schema.table_exists(table_name) {
                    bail!("table '{}' not found in schema '{}'", table_name, schema_name);
                }
                let table = schema.get_table_mut(table_name).unwrap();
                if !table.rename_column(old_name, new_name) {
                    bail!("column '{}' not found in table '{}'", old_name, table_name);
                }
                format!("renamed column '{}' to '{}'", old_name, new_name)
            }
            AlterTableAction::AddColumn(col_def) => {
                let mut catalog_guard = self.catalog.write();
                let catalog = catalog_guard.as_mut().unwrap();
                let schema = catalog
                    .get_schema_mut(schema_name)
                    .ok_or_else(|| eyre::eyre!("schema '{}' not found", schema_name))?;
                if !schema.table_exists(table_name) {
                    bail!("table '{}' not found in schema '{}'", table_name, schema_name);
                }
                let table = schema.get_table_mut(table_name).unwrap();
                let column = Self::ast_column_to_schema_column(col_def)?;
                let col_name = column.name().to_string();
                table.add_column(column);
                format!("added column '{}'", col_name)
            }
            AlterTableAction::DropColumn { name, if_exists, .. } => {
                self.migrate_table_drop_column(schema_name, table_name, name, *if_exists)?
            }
            _ => bail!("ALTER TABLE action not yet supported"),
        };

        self.save_catalog()?;

        Ok(ExecuteResult::AlterTable { action: action_desc })
    }

    fn migrate_table_drop_column(
        &self,
        schema_name: &str,
        table_name: &str,
        column_name: &str,
        if_exists: bool,
    ) -> Result<String> {
        use crate::btree::BTree;
        use crate::sql::decoder::{RecordDecoder, SimpleDecoder};
        use crate::types::{create_record_schema, OwnedValue};

        const BATCH_SIZE: usize = 10_000;

        let (old_columns, drop_idx, indexes_to_drop) = {
            let catalog_guard = self.catalog.read();
            let catalog = catalog_guard.as_ref().unwrap();
            let schema = catalog
                .get_schema(schema_name)
                .ok_or_else(|| eyre::eyre!("schema '{}' not found", schema_name))?;
            let table = schema
                .get_table(table_name)
                .ok_or_else(|| eyre::eyre!("table '{}' not found", table_name))?;

            let old_columns: Vec<crate::schema::ColumnDef> = table.columns().to_vec();
            let drop_idx = old_columns
                .iter()
                .position(|c| c.name().eq_ignore_ascii_case(column_name));

            if drop_idx.is_none() && !if_exists {
                bail!(
                    "column '{}' not found in table '{}'",
                    column_name,
                    table_name
                );
            }

            let indexes_to_drop: Vec<String> = table
                .indexes()
                .iter()
                .filter(|idx| {
                    idx.columns()
                        .iter()
                        .any(|c| c.eq_ignore_ascii_case(column_name))
                })
                .map(|idx| idx.name().to_string())
                .collect();

            (old_columns, drop_idx, indexes_to_drop)
        };

        let Some(drop_idx) = drop_idx else {
            return Ok(format!("column '{}' does not exist (skipped)", column_name));
        };

        for index_name in &indexes_to_drop {
            let mut file_manager_guard = self.file_manager.write();
            let file_manager = file_manager_guard.as_mut().unwrap();

            if file_manager.index_exists(schema_name, table_name, index_name) {
                let index_storage =
                    file_manager.index_data_mut(schema_name, table_name, index_name)?;
                let root_page = 1u32;
                let index_btree = BTree::new(index_storage, root_page)?;
                let mut index_cursor = index_btree.cursor_first()?;

                let mut keys_to_delete: Vec<Vec<u8>> = Vec::new();
                while index_cursor.valid() {
                    keys_to_delete.push(index_cursor.key()?.to_vec());
                    index_cursor.advance()?;
                }

                let mut index_btree_mut = BTree::new(index_storage, root_page)?;
                for key in &keys_to_delete {
                    index_btree_mut.delete(key)?;
                }
                index_storage.sync()?;
            }
        }

        let old_column_types: Vec<crate::records::types::DataType> =
            old_columns.iter().map(|c| c.data_type()).collect();
        let decoder = SimpleDecoder::new(old_column_types);

        let mut new_columns = old_columns.clone();
        new_columns.remove(drop_idx);
        let new_schema = create_record_schema(&new_columns);

        {
            let mut file_manager_guard = self.file_manager.write();
            let file_manager = file_manager_guard.as_mut().unwrap();
            let storage = file_manager.table_data_mut(schema_name, table_name)?;
            let root_page = 1u32;

            let all_keys: Vec<Vec<u8>> = {
                let btree = BTree::new(storage, root_page)?;
                let mut cursor = btree.cursor_first()?;
                let mut keys = Vec::new();
                while cursor.valid() {
                    keys.push(cursor.key()?.to_vec());
                    cursor.advance()?;
                }
                keys
            };

            for chunk in all_keys.chunks(BATCH_SIZE) {
                let mut batch: Vec<(Vec<u8>, Vec<u8>)> = Vec::with_capacity(chunk.len());

                {
                    let btree = BTree::new(storage, root_page)?;
                    for key in chunk {
                        if let Some(handle) = btree.search(key)? {
                            let value = btree.get_value(&handle)?;
                            let values = decoder.decode(key, value)?;
                            let mut owned_values: Vec<OwnedValue> =
                                values.into_iter().map(OwnedValue::from).collect();
                            owned_values.remove(drop_idx);

                            let new_record =
                                OwnedValue::build_record_from_values(&owned_values, &new_schema)?;
                            batch.push((key.clone(), new_record));
                        }
                    }
                }

                let mut btree_mut = BTree::new(storage, root_page)?;
                for (key, _) in &batch {
                    btree_mut.delete(key)?;
                }
                for (key, new_value) in &batch {
                    btree_mut.insert(key, new_value)?;
                }
            }

            storage.sync()?;
        }

        {
            let mut catalog_guard = self.catalog.write();
            let catalog = catalog_guard.as_mut().unwrap();
            let schema = catalog.get_schema_mut(schema_name).unwrap();
            let table = schema.get_table_mut(table_name).unwrap();

            for index_name in &indexes_to_drop {
                table.remove_index(index_name);
            }

            table.drop_column(column_name);
        }

        Ok(format!("dropped column '{}'", column_name))
    }

    fn ast_column_to_schema_column(
        col_def: &crate::sql::ast::ColumnDef<'_>,
    ) -> Result<crate::schema::table::ColumnDef> {
        use crate::schema::table::{ColumnDef as SchemaColumnDef, Constraint as SchemaConstraint};

        let data_type = Self::convert_data_type(&col_def.data_type);
        let mut column = SchemaColumnDef::new(col_def.name, data_type);

        for constraint in col_def.constraints.iter() {
            match constraint {
                crate::sql::ast::ColumnConstraint::NotNull => {
                    column = column.with_constraint(SchemaConstraint::NotNull);
                }
                crate::sql::ast::ColumnConstraint::PrimaryKey => {
                    column = column.with_constraint(SchemaConstraint::PrimaryKey);
                    column = column.with_constraint(SchemaConstraint::NotNull);
                }
                crate::sql::ast::ColumnConstraint::Unique => {
                    column = column.with_constraint(SchemaConstraint::Unique);
                }
                crate::sql::ast::ColumnConstraint::AutoIncrement => {
                    column = column.with_constraint(SchemaConstraint::AutoIncrement);
                }
                crate::sql::ast::ColumnConstraint::Default(expr) => {
                    if let Some(default_str) = Self::expr_to_default_string(expr) {
                        column = column.with_default(default_str);
                    }
                }
                _ => {}
            }
        }

        Ok(column)
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
                if let Some(table_def) = schema.get_table(table_name) {
                    let table_id = table_def.id() as u32;
                    schema.remove_table(table_name);
                    self.table_id_lookup.write().remove(&table_id);
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
        use crate::storage::SyncMode;
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
            "SYNCHRONOUS" => {
                if let Some(ref val) = value {
                    let mode = match val.as_str() {
                        "OFF" | "0" => SyncMode::Off,
                        "NORMAL" | "1" => SyncMode::Normal,
                        "FULL" | "2" => SyncMode::Full,
                        _ => bail!("invalid PRAGMA synchronous value: {} (use OFF, NORMAL, or FULL)", val),
                    };
                    let wal_guard = self.wal.lock();
                    if let Some(ref wal) = *wal_guard {
                        wal.set_sync_mode(mode);
                    }
                    drop(wal_guard);
                }
                let current_mode = {
                    let wal_guard = self.wal.lock();
                    wal_guard.as_ref().map(|w| w.sync_mode()).unwrap_or(SyncMode::Full)
                };
                let mode_str = match current_mode {
                    SyncMode::Off => "OFF",
                    SyncMode::Normal => "NORMAL",
                    SyncMode::Full => "FULL",
                };
                Ok(ExecuteResult::Pragma {
                    name: name.clone(),
                    value: Some(mode_str.to_string()),
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
        let wal_enabled = self
            .wal_enabled
            .load(std::sync::atomic::Ordering::Acquire);

        {
            let active_txn = self.active_txn.lock();
            active_txn
                .as_ref()
                .ok_or_else(|| eyre::eyre!("no transaction in progress"))?;
        }

        let mut active_txn = self.active_txn.lock();
        let txn = active_txn
            .take()
            .ok_or_else(|| eyre::eyre!("no transaction in progress"))?;

        self.finalize_transaction_commit(txn)?;

        if wal_enabled {
            let dirty_table_ids = self.dirty_tracker.all_dirty_table_ids();
            if dirty_table_ids.is_empty() {
                return Ok(ExecuteResult::Commit);
            }

            let table_infos: Vec<(u32, String, String)> = {
                let lookup = self.table_id_lookup.read();
                dirty_table_ids
                    .iter()
                    .filter_map(|&table_id| {
                        lookup
                            .get(&table_id)
                            .map(|(s, t)| (table_id, s.clone(), t.clone()))
                    })
                    .collect()
            };

            let mut file_manager_guard = self.file_manager.write();
            let file_manager = file_manager_guard
                .as_mut()
                .ok_or_else(|| eyre::eyre!("file manager not available for WAL flush"))?;

            let mut wal_guard = self.wal.lock();
            let wal = wal_guard
                .as_mut()
                .ok_or_else(|| eyre::eyre!("WAL not initialized but WAL mode is enabled"))?;

            for (table_id, schema_name, table_name) in table_infos {
                let storage = file_manager
                    .table_data(&schema_name, &table_name)
                    .wrap_err_with(|| {
                        format!(
                            "failed to get storage for table {}.{} during WAL flush",
                            schema_name, table_name
                        )
                    })?;

                WalStoragePerTable::flush_wal_for_table(&self.dirty_tracker, storage, wal, table_id)
                    .wrap_err_with(|| {
                        format!(
                            "failed to flush WAL for table {}.{} on commit",
                            schema_name, table_name
                        )
                    })?;
            }
        }

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
            SqlType::Integer => DataType::Int4,
            SqlType::BigInt => DataType::Int8,
            SqlType::SmallInt => DataType::Int2,
            SqlType::TinyInt => DataType::Int2,
            SqlType::Serial => DataType::Int4,
            SqlType::BigSerial => DataType::Int8,
            SqlType::SmallSerial => DataType::Int2,
            SqlType::Real | SqlType::DoublePrecision => DataType::Float8,
            SqlType::Decimal(_, _) | SqlType::Numeric(_, _) => DataType::Float8,
            SqlType::Varchar(_) => DataType::Varchar,
            SqlType::Text => DataType::Text,
            SqlType::Char(_) => DataType::Char,
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

    fn extract_type_length(sql_type: &crate::sql::ast::DataType) -> Option<u32> {
        use crate::sql::ast::DataType as SqlType;

        match sql_type {
            SqlType::Varchar(len) => *len,
            SqlType::Char(len) => *len,
            _ => None,
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
            Expr::Function(func) => {
                let name = func.name.name.to_uppercase();
                match name.as_str() {
                    "CURRENT_TIMESTAMP" | "NOW" | "CURRENT_DATE" | "CURRENT_TIME" 
                    | "LOCALTIME" | "LOCALTIMESTAMP" => Some(name),
                    _ => None,
                }
            }
            Expr::Column(col) => {
                let name = col.column.to_uppercase();
                match name.as_str() {
                    "CURRENT_TIMESTAMP" | "NOW" | "CURRENT_DATE" | "CURRENT_TIME"
                    | "LOCALTIME" | "LOCALTIMESTAMP" => Some(name),
                    _ => None,
                }
            }
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
                    Some(DataType::Timestamp) => parse_timestamp(s),
                    Some(DataType::TimestampTz) => parse_timestamp(s),
                    Some(DataType::Date) => parse_date(s),
                    Some(DataType::Time) => parse_time(s),
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

    fn generate_row_key(row_id: u64) -> [u8; 8] {
        row_id.to_be_bytes()
    }

    pub fn checkpoint(&self) -> Result<CheckpointInfo> {
        use std::sync::atomic::Ordering;

        if self.closed.load(Ordering::Acquire) {
            bail!("database is closed");
        }

        let mut wal_guard = self.wal.lock();
        let wal = match wal_guard.as_mut() {
            Some(w) => w,
            None => {
                self.dirty_tracker.clear_all();
                return Ok(CheckpointInfo {
                    frames_checkpointed: 0,
                    wal_truncated: false,
                });
            }
        };

        if self.dirty_tracker.is_empty() {
            wal.cleanup_old_segments()?;
            return Ok(CheckpointInfo {
                frames_checkpointed: 0,
                wal_truncated: false,
            });
        }
        let table_ids = self.dirty_tracker.all_dirty_table_ids();

        self.ensure_file_manager()?;

        let mut file_manager_guard = self.file_manager.write();
        let file_manager = match file_manager_guard.as_mut() {
            Some(fm) => fm,
            None => {
                self.dirty_tracker.clear_all();
                return Ok(CheckpointInfo {
                    frames_checkpointed: 0,
                    wal_truncated: false,
                });
            }
        };

        let table_infos: Vec<(u32, String, String)> = {
            let lookup = self.table_id_lookup.read();
            table_ids
                .iter()
                .filter_map(|&table_id| {
                    lookup
                        .get(&table_id)
                        .map(|(s, t)| (table_id, s.clone(), t.clone()))
                })
                .collect()
        };

        let mut total_frames = 0u32;
        for (table_id, schema_name, table_name) in &table_infos {
            if let Ok(storage) = file_manager.table_data(schema_name, table_name) {
                let frames =
                    WalStoragePerTable::flush_wal_for_table(&self.dirty_tracker, storage, wal, *table_id)
                        .wrap_err_with(|| {
                            format!(
                                "failed to flush dirty pages for table {}.{}",
                                schema_name, table_name
                            )
                        })?;
                total_frames += frames;
            }
        }

        let current_offset = wal.current_offset();
        let had_frames = current_offset > 0;

        if had_frames {
            wal.truncate()?;
        }

        Ok(CheckpointInfo {
            frames_checkpointed: total_frames,
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

    #[allow(clippy::too_many_arguments)]
    fn toast_value(
        &self,
        file_manager: &mut crate::storage::FileManager,
        schema_name: &str,
        table_name: &str,
        row_id: u64,
        column_index: u16,
        data: &[u8],
        wal_enabled: bool,
        hint: Option<u32>,
    ) -> Result<(Vec<u8>, Option<u32>)> {
        use crate::btree::BTree;
        use crate::storage::toast::{make_chunk_key, ToastPointer, TOAST_CHUNK_SIZE};

        let toast_table_name = crate::storage::toast::toast_table_name(table_name);
        let toast_storage = file_manager.table_data_mut(schema_name, &toast_table_name)?;

        let (toast_table_id, root_page) = {
            let page0 = toast_storage.page(0)?;
            let header = crate::storage::TableFileHeader::from_bytes(page0)?;
            (header.table_id() as u32, header.root_page())
        };

        let pointer = ToastPointer::new(row_id, column_index, data.len() as u64);
        let chunk_id = pointer.chunk_id;

        let (new_hint, new_root) = if wal_enabled {
            let mut wal_storage =
                WalStoragePerTable::new(toast_storage, &self.dirty_tracker, toast_table_id);
            let mut btree = BTree::with_rightmost_hint(&mut wal_storage, root_page, hint)?;
            for (seq, chunk) in data.chunks(TOAST_CHUNK_SIZE).enumerate() {
                let chunk_key = make_chunk_key(chunk_id, seq as u32);
                btree.insert(&chunk_key, chunk)?;
            }
            (btree.rightmost_hint(), btree.root_page())
        } else {
            let mut btree = BTree::with_rightmost_hint(toast_storage, root_page, hint)?;
            for (seq, chunk) in data.chunks(TOAST_CHUNK_SIZE).enumerate() {
                let chunk_key = make_chunk_key(chunk_id, seq as u32);
                btree.insert(&chunk_key, chunk)?;
            }
            (btree.rightmost_hint(), btree.root_page())
        };

        if new_root != root_page {
            let toast_storage = file_manager.table_data_mut(schema_name, &toast_table_name)?;
            let page0 = toast_storage.page_mut(0)?;
            let header = crate::storage::TableFileHeader::from_bytes_mut(page0)?;
            header.set_root_page(new_root);
        }

        Ok((pointer.encode().to_vec(), new_hint))
    }

    fn detoast_value(
        &self,
        file_manager: &mut crate::storage::FileManager,
        schema_name: &str,
        table_name: &str,
        toast_pointer: &[u8],
    ) -> Result<Vec<u8>> {
        use crate::btree::BTree;
        use crate::storage::toast::{chunk_count, make_chunk_key, ToastPointer};

        let pointer = ToastPointer::decode(toast_pointer)?;
        let chunk_id = pointer.chunk_id;
        let total_size = pointer.total_size as usize;
        let num_chunks = chunk_count(total_size);

        let toast_table_name = crate::storage::toast::toast_table_name(table_name);
        let toast_storage = file_manager.table_data_mut(schema_name, &toast_table_name)?;

        let root_page = {
            let page0 = toast_storage.page(0)?;
            crate::storage::TableFileHeader::from_bytes(page0)?.root_page()
        };

        let btree = BTree::new(toast_storage, root_page)?;

        let mut result = Vec::with_capacity(total_size);

        for seq in 0..num_chunks {
            let chunk_key = make_chunk_key(chunk_id, seq as u32);
            let handle = btree
                .search(&chunk_key)?
                .ok_or_else(|| eyre::eyre!("TOAST chunk not found: {:?}", chunk_key))?;
            let chunk_data = btree.get_value(&handle)?;
            result.extend_from_slice(chunk_data);
        }

        result.truncate(total_size);
        Ok(result)
    }

    fn detoast_rows(
        &self,
        file_manager: &mut crate::storage::FileManager,
        schema_name: &str,
        table_name: &str,
        rows: Vec<crate::database::Row>,
    ) -> Result<Vec<crate::database::Row>> {
        let mut result = Vec::with_capacity(rows.len());
        for row in rows {
            let mut new_values = Vec::with_capacity(row.values.len());
            for val in row.values {
                let new_val = match val {
                    OwnedValue::ToastPointer(ptr) => {
                        let data = self.detoast_value(file_manager, schema_name, table_name, &ptr)?;
                        if let Ok(s) = String::from_utf8(data.clone()) {
                            OwnedValue::Text(s)
                        } else {
                            OwnedValue::Blob(data)
                        }
                    }
                    other => other,
                };
                new_values.push(new_val);
            }
            result.push(crate::database::Row::new(new_values));
        }
        Ok(result)
    }

    fn delete_toast_chunks(
        &self,
        file_manager: &mut crate::storage::FileManager,
        schema_name: &str,
        table_name: &str,
        row_id: u64,
        column_index: u16,
        total_size: u64,
    ) -> Result<()> {
        use crate::btree::BTree;
        use crate::storage::toast::{chunk_count, make_chunk_key, ToastPointer};

        let pointer = ToastPointer::new(row_id, column_index, total_size);
        let chunk_id = pointer.chunk_id;
        let num_chunks = chunk_count(total_size as usize);

        let toast_table_name = crate::storage::toast::toast_table_name(table_name);
        let toast_storage = file_manager.table_data_mut(schema_name, &toast_table_name)?;

        let root_page = {
            let page0 = toast_storage.page(0)?;
            crate::storage::TableFileHeader::from_bytes(page0)?.root_page()
        };

        let mut btree = BTree::new(toast_storage, root_page)?;

        for seq in 0..num_chunks {
            let chunk_key = make_chunk_key(chunk_id, seq as u32);
            let _ = btree.delete(&chunk_key);
        }

        Ok(())
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
