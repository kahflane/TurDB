use crate::database::convert::convert_value_with_type;
use crate::database::dirty_tracker::ShardedDirtyTracker;
use crate::database::row::Row;
use crate::database::{CheckpointInfo, ExecuteResult, RecoveryInfo};
use crate::database::transaction::ActiveTransaction;
use crate::mvcc::TransactionManager;

use crate::schema::Catalog;

use crate::sql::builder::ExecutorBuilder;
use crate::sql::context::ExecutionContext;
use crate::sql::executor::{BTreeSource, Executor, ExecutorRow, MaterializedRowSource, ReverseBTreeSource, RowSource, StreamingBTreeSource};
use crate::sql::planner::Planner;

use crate::sql::Parser;
use crate::storage::{FileManager, TableFileHeader, Wal, WalStoragePerTable, DEFAULT_SCHEMA};
use crate::types::{create_record_schema, DataType, OwnedValue, Value};
use bumpalo::Bump;
use eyre::{bail, ensure, Result, WrapErr};

use parking_lot::{Mutex, RwLock};

use std::path::{Path, PathBuf};
use std::sync::atomic::Ordering as AtomicOrdering;

use crate::database::timing::{INSERT_TIME_NS, PARSE_TIME_NS};

pub struct Database {
    path: PathBuf,
    pub(crate) file_manager: RwLock<Option<FileManager>>,
    pub(crate) catalog: RwLock<Option<Catalog>>,
    pub(crate) wal: Mutex<Option<Wal>>,
    wal_dir: PathBuf,
    pub(crate) next_row_id: std::sync::atomic::AtomicU64,
    next_table_id: std::sync::atomic::AtomicU64,
    next_index_id: std::sync::atomic::AtomicU64,
    closed: std::sync::atomic::AtomicBool,
    pub(crate) wal_enabled: std::sync::atomic::AtomicBool,
    pub(crate) dirty_tracker: ShardedDirtyTracker,
    pub(crate) txn_manager: TransactionManager,
    pub(crate) active_txn: Mutex<Option<ActiveTransaction>>,
    /// Session setting: whether foreign key constraints are checked (default: true)
    pub(crate) foreign_keys_enabled: std::sync::atomic::AtomicBool,
    pub(crate) table_id_lookup: RwLock<hashbrown::HashMap<u32, (String, String)>>,
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
            table_id_lookup: RwLock::new(hashbrown::HashMap::new()),
        };

        db.ensure_catalog()?;

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

        let root_dir = path.join(DEFAULT_SCHEMA);
        std::fs::create_dir_all(&root_dir).wrap_err_with(|| {
            format!("failed to create default schema directory at {:?}", root_dir)
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

    pub(crate) fn ensure_file_manager(&self) -> Result<()> {
        if self.file_manager.read().is_some() {
            return Ok(());
        }
        let mut guard = self.file_manager.write();
        if guard.is_none() {
            let fm = FileManager::open(&self.path, 64)
                .wrap_err_with(|| format!("failed to open file manager at {:?}", self.path))?;
            *guard = Some(fm);
        }
        Ok(())
    }

    pub(crate) fn ensure_catalog(&self) -> Result<()> {
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
    pub(crate) fn flush_wal_if_autocommit(
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

        let table_storage_arc = file_manager.table_data(schema_name, table_name)?;
        let table_storage = table_storage_arc.read();
        let mut wal_guard = self.wal.lock();
        let wal = wal_guard.as_mut().ok_or_else(|| {
            eyre::eyre!("WAL is enabled but not initialized - this is a bug")
        })?;
        let frames_written =
            WalStoragePerTable::flush_wal_for_table(&self.dirty_tracker, &table_storage, wal, table_id)
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

    pub(crate) fn save_catalog(&self) -> Result<()> {
        use crate::schema::persistence::CatalogPersistence;

        let catalog_path = self.path.join("turdb.catalog");
        let catalog_guard = self.catalog.read();
        if let Some(ref catalog) = *catalog_guard {
            CatalogPersistence::save(catalog, &catalog_path)
                .wrap_err_with(|| format!("failed to save catalog to {:?}", catalog_path))?;
        }
        Ok(())
    }

    pub(crate) fn save_meta(&self) -> Result<()> {
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

    pub(crate) fn allocate_table_id(&self) -> u64 {
        use std::sync::atomic::Ordering;
        self.next_table_id.fetch_add(1, Ordering::AcqRel)
    }

    pub(crate) fn allocate_index_id(&self) -> u64 {
        use std::sync::atomic::Ordering;
        self.next_index_id.fetch_add(1, Ordering::AcqRel)
    }

    pub(crate) fn encode_value_as_key<B: crate::encoding::key::KeyBuffer>(value: &OwnedValue, buf: &mut B) {
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
            let storage_arc = file_manager.table_data_mut(schema_name, table_name)?;
            let storage = storage_arc.read();
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
            let table_storage_arc = file_manager.table_data_mut_with_key(&table_file_key)
                .ok_or_else(|| eyre::eyre!("table storage not found in cache"))?;
            let mut table_storage = table_storage_arc.write();
            let mut wal_storage =
                WalStoragePerTable::new(&mut *table_storage, &self.dirty_tracker, table_id as u32);
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
            let table_storage_arc = file_manager.table_data_mut_with_key(&table_file_key)
                .ok_or_else(|| eyre::eyre!("table storage not found in cache"))?;
            let mut table_storage = table_storage_arc.write();
            let mut btree = BTree::with_rightmost_hint(&mut *table_storage, root_page, rightmost_hint)?;

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
            let storage_arc = file_manager.table_data_mut(schema_name, table_name)?;
            let mut storage = storage_arc.write();
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

    pub fn insert_cached(
        &self,
        plan: &crate::database::prepared::CachedInsertPlan,
        params: &[OwnedValue],
    ) -> Result<usize> {
        use crate::btree::BTree;
        use crate::storage::WalStoragePerTable;
        use std::sync::atomic::Ordering;

        self.ensure_file_manager()?;

        let wal_enabled = self.wal_enabled.load(Ordering::Acquire);
        if wal_enabled {
            self.ensure_wal()?;
        }

        // Try to get storage from cache or acquire it
        let storage_arc = if let Some(weak) = plan.storage.borrow().as_ref() {
            weak.upgrade()
        } else {
            None
        };

        let storage_arc = if let Some(arc) = storage_arc {
            arc
        } else {
            // Cold path: Lock FileManager and get/cache storage
            let mut file_manager_guard = self.file_manager.write();
            let file_manager = file_manager_guard.as_mut().unwrap();
            let arc = file_manager.table_data_mut(&plan.schema_name, &plan.table_name)?;
            
            // Update weak cache
            *plan.storage.borrow_mut() = Some(std::sync::Arc::downgrade(&arc));
            arc
        };

        // Acquire lock on the specific table storage
        let mut storage_guard = storage_arc.write();

        // Read metadata from page 0 directly (fast, no FM lock)
        let (mut root_page, mut rightmost_hint) = {
            let page = storage_guard.page(0)?;
            let header = TableFileHeader::from_bytes(page)?;
            let stored_root = header.root_page();
            let hint = header.rightmost_hint();
            let root = if stored_root > 0 { stored_root } else { 1 };
            (root, if hint > 0 { Some(hint) } else { Some(root) })
        };

        let mut record_builder = crate::records::RecordBuilder::new(&plan.record_schema);
        
        let mut buffer_guard = plan.record_buffer.borrow_mut();
        buffer_guard.clear();
        OwnedValue::build_record_into_buffer(params, &mut record_builder, &mut *buffer_guard)?;

        let row_id = self.next_row_id.fetch_add(1, Ordering::Relaxed);
        let row_key = Self::generate_row_key(row_id);

        if wal_enabled {
            let mut wal_storage =
                WalStoragePerTable::new(&mut *storage_guard, &self.dirty_tracker, plan.table_id as u32);
            let mut btree = BTree::with_rightmost_hint(&mut wal_storage, root_page, rightmost_hint)?;
            btree.insert_append(&row_key, &buffer_guard)?;
            root_page = btree.root_page();
            rightmost_hint = btree.rightmost_hint();
        } else {
            let mut btree = BTree::with_rightmost_hint(&mut *storage_guard, root_page, rightmost_hint)?;
            btree.insert_append(&row_key, &buffer_guard)?;
            root_page = btree.root_page();
            rightmost_hint = btree.rightmost_hint();
        }

        // Update header
        {
            let page = storage_guard.page_mut(0)?;
            let header = TableFileHeader::from_bytes_mut(page)?;
            header.set_root_page(root_page);
            if let Some(hint) = rightmost_hint {
                header.set_rightmost_hint(hint);
            }
            let new_row_count = header.row_count().saturating_add(1);
            header.set_row_count(new_row_count);
        }

        // Handle WAL flush for autocommit without relocking FileManager
        if wal_enabled {
             let txn_active = self.active_txn.lock().is_some();
             if !txn_active && self.dirty_tracker.has_dirty_pages(plan.table_id as u32) {
                 let mut wal_guard = self.wal.lock();
                 if let Some(wal) = wal_guard.as_mut() {
                     WalStoragePerTable::flush_wal_for_table(
                        &self.dirty_tracker, 
                        &mut *storage_guard, 
                        wal, 
                        plan.table_id as u32
                    )?;
                 }
             }
        }

        Ok(1)
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

                let storage_arc = file_manager
                    .table_data(schema_name, table_name)
                    .wrap_err_with(|| {
                        format!(
                            "failed to open table storage for {}.{}",
                            schema_name, table_name
                        )
                    })?;
                let storage = storage_arc.read();

                let root_page = 1u32;
                let inner_source = StreamingBTreeSource::from_btree_scan_with_projections(
                    &storage,
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

            let storage_arc = file_manager
                .table_data(schema_name, table_name)
                .wrap_err_with(|| {
                    format!(
                        "failed to open table storage for {}.{}",
                        schema_name, table_name
                    )
                })?;
            let storage = storage_arc.read();

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

                let storage_arc = file_manager
                    .table_data(schema_name, table_name)
                    .wrap_err_with(|| {
                        format!(
                            "failed to open table storage for {}.{}",
                            schema_name, table_name
                        )
                    })?;
                let storage = storage_arc.read();

                let root_page = 1u32;
                let source: BTreeSource = if scan.reverse {
                    BTreeSource::Reverse(
                        ReverseBTreeSource::from_btree_scan_reverse_with_projections(
                            &storage,
                            root_page,
                            column_types,
                            projections,
                        )
                        .wrap_err("failed to create reverse table scan")?,
                    )
                } else {
                    BTreeSource::Forward(
                        StreamingBTreeSource::from_btree_scan_with_projections(
                            &storage,
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
                use crate::sql::planner::ScanRange;

                let schema_name = scan.schema.unwrap_or("root");
                let table_name = scan.table;

                toast_table_info = Some((schema_name.to_string(), table_name.to_string()));

                let table_def = catalog
                    .resolve_table(table_name)
                    .wrap_err_with(|| format!("table '{}' not found", table_name))?;

                let column_types: Vec<_> =
                    table_def.columns().iter().map(|c| c.data_type()).collect();

                let storage_arc = file_manager
                    .table_data(schema_name, table_name)
                    .wrap_err_with(|| {
                        format!(
                            "failed to open table storage for {}.{}",
                            schema_name, table_name
                        )
                    })?;
                let storage = storage_arc.read();

                let root_page = 1u32;

                let (start_key, end_key): (Option<&[u8]>, Option<&[u8]>) = match &scan.key_range {
                    ScanRange::FullScan => (None, None),
                    ScanRange::PrefixScan { prefix } => {
                        let mut end = prefix.to_vec();
                        let mut carry = true;
                        for byte in end.iter_mut().rev() {
                            if carry {
                                if *byte == 0xFF {
                                    *byte = 0x00;
                                } else {
                                    *byte += 1;
                                    carry = false;
                                    break;
                                }
                            }
                        }
                        if carry {
                            (Some(*prefix), None)
                        } else {
                            let end_slice = arena.alloc_slice_copy(&end);
                            (Some(*prefix), Some(end_slice as &[u8]))
                        }
                    }
                    ScanRange::RangeScan { start, end } => (*start, *end),
                };

                let source = StreamingBTreeSource::from_btree_range_scan_with_projections(
                    &storage,
                    root_page,
                    start_key,
                    end_key,
                    column_types,
                    None,
                )
                .wrap_err("failed to create range scan for index scan")?;

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
                    let index_storage_arc = file_manager
                        .index_data(schema_name, table_name, index_name)
                        .wrap_err_with(|| {
                            format!(
                                "failed to open index storage for {}.{}.{}",
                                schema_name, table_name, index_name
                            )
                        })?;
                    let index_storage = index_storage_arc.read();

                    let root_page = 1u32;
                    let index_reader = BTreeReader::new(&index_storage, root_page)?;

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
                    let table_storage_arc = file_manager
                        .table_data(schema_name, table_name)
                        .wrap_err_with(|| {
                            format!(
                                "failed to open table storage for {}.{}",
                                schema_name, table_name
                            )
                        })?;
                    let table_storage = table_storage_arc.read();

                    let root_page = 1u32;
                    let table_reader = BTreeReader::new(&table_storage, root_page)?;

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
                    let storage_arc = file_manager.table_data(schema_name, table_name)?;
                    let storage = storage_arc.read();
                    let source = StreamingBTreeSource::from_btree_scan_with_projections(
                        &storage, 1, column_types.clone(), None,
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
                    let storage_arc = file_manager.table_data(schema_name, table_name)?;
                    let storage = storage_arc.read();
                    let source = StreamingBTreeSource::from_btree_scan_with_projections(
                        &storage, 1, column_types.clone(), None,
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

                    let storage_arc = file_manager
                        .table_data(schema_name, table_name)
                        .wrap_err_with(|| {
                            format!(
                                "failed to open table storage for {}.{}",
                                schema_name, table_name
                            )
                        })?;
                    let storage = storage_arc.read();

                    let root_page = 1u32;
                    let source = StreamingBTreeSource::from_btree_scan_with_projections(
                        &storage,
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

    pub(crate) fn execute_select_internal(
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

            let storage_arc = file_manager
                .table_data(schema_name, table_name)
                .wrap_err_with(|| {
                    format!(
                        "failed to open table storage for {}.{}",
                        schema_name, table_name
                    )
                })?;
            let storage = storage_arc.read();

            let root_page = 1u32;
            let source = StreamingBTreeSource::from_btree_scan_with_projections(
                &storage,
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

        self.execute_statement(&stmt, sql, &arena, None)
    }

    pub fn execute_with_params(&self, sql: &str, params: &[OwnedValue]) -> Result<ExecuteResult> {
        let parse_start = std::time::Instant::now();
        let arena = Bump::new();
        let mut parser = Parser::new(sql, &arena);
        let stmt = parser
            .parse_statement()
            .wrap_err("failed to parse SQL statement")?;
        PARSE_TIME_NS.fetch_add(parse_start.elapsed().as_nanos() as u64, AtomicOrdering::Relaxed);

        self.execute_statement(&stmt, sql, &arena, Some(params))
    }

    pub fn execute_with_cached_plan(
        &self,
        prepared: &super::PreparedStatement,
        params: &[OwnedValue],
    ) -> Result<ExecuteResult> {
        if let Some(result) = prepared.with_cached_plan(|plan| {
            let insert_start = std::time::Instant::now();
            let result = self.execute_insert_cached(plan, params);
            INSERT_TIME_NS.fetch_add(insert_start.elapsed().as_nanos() as u64, AtomicOrdering::Relaxed);
            result
        }) {
            return result;
        }

        let parse_start = std::time::Instant::now();
        let arena = Bump::new();
        let mut parser = Parser::new(prepared.sql(), &arena);
        let stmt = parser
            .parse_statement()
            .wrap_err("failed to parse SQL statement")?;
        PARSE_TIME_NS.fetch_add(parse_start.elapsed().as_nanos() as u64, AtomicOrdering::Relaxed);

        if let crate::sql::ast::Statement::Insert(insert) = &stmt {
            self.ensure_catalog()?;
            self.ensure_file_manager()?;
            
            let catalog_guard = self.catalog.read();
            let catalog = catalog_guard.as_ref().unwrap();
            
            let table_name = insert.table.name;
            let schema_name = insert.table.schema.unwrap_or(crate::storage::DEFAULT_SCHEMA);
            
            if let Ok(table_def) = catalog.resolve_table(table_name) {
                let column_types: Vec<_> = table_def.columns().iter().map(|c| c.data_type()).collect();
                
                if let Ok(storage_arc) = {
                    let mut fm_guard = self.file_manager.write();
                    let fm = fm_guard.as_mut().unwrap();
                    fm.table_data(schema_name, table_name)
                } {
                    let cached_plan = super::prepared::CachedInsertPlan {
                        table_id: table_def.id(),
                        schema_name: schema_name.to_string(),
                        table_name: table_name.to_string(),
                        column_count: column_types.len(),
                        column_types,
                        record_schema: create_record_schema(table_def.columns()),
                        root_page: std::cell::Cell::new(0),
                        rightmost_hint: std::cell::Cell::new(None),
                        row_count: std::cell::Cell::new(None),
                         storage: std::cell::RefCell::new(Some(std::sync::Arc::downgrade(&storage_arc))),
                         record_buffer: std::cell::RefCell::new(Vec::with_capacity(256)),
                    };
                    
                    prepared.set_cached_insert_plan(cached_plan);
                }
            }
            drop(catalog_guard);
        }

        self.execute_statement(&stmt, prepared.sql(), &arena, Some(params))
    }

    fn execute_statement<'a>(
        &self,
        stmt: &crate::sql::ast::Statement<'a>,
        sql: &str,
        arena: &Bump,
        params: Option<&[OwnedValue]>,
    ) -> Result<ExecuteResult> {
        use crate::sql::ast::Statement;
        match stmt {
            Statement::CreateTable(create) => self.execute_create_table(create, arena),
            Statement::CreateSchema(create) => self.execute_create_schema(create),
            Statement::CreateIndex(create) => self.execute_create_index(create, arena),
            Statement::Insert(insert) => {
                let insert_start = std::time::Instant::now();
                let result = self.execute_insert_with_params(insert, arena, params);
                INSERT_TIME_NS.fetch_add(insert_start.elapsed().as_nanos() as u64, AtomicOrdering::Relaxed);
                result
            }
            Statement::Update(update) => self.execute_update(update, arena),
            Statement::Delete(delete) => self.execute_delete(delete, arena),
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
            _ => bail!("unknown setting: {}", set.name),
        }
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


    pub(crate) fn evaluate_check_expression(
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
            if let Ok(storage_arc) = file_manager.table_data(schema_name, table_name) {
                let storage = storage_arc.read();
                let frames =
                    WalStoragePerTable::flush_wal_for_table(&self.dirty_tracker, &*storage, wal, *table_id)
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
