//! # Database Module
//!
//! This module provides the central `Database` struct and `SharedDatabase` state
//! that coordinates all components of TurDB (SQL engine, storage, catalog, etc.).
//!
//! ## key Components
//!
//! - **Database**: The public API handle. Cloneable and thread-safe.
//! - **SharedDatabase**: Internal state shared across all `Database` handles.
//! - **Catalog**: Manages schema metadata (tables, columns, indexes).
//! - **FileManager**: Manages B-Tree storage files.
//! - **Wal**: Write-Ahead Log for durability and atomic commits.
//!
//! ## Concurrency Model
//!
//! The database uses a hybrid concurrency model:
//! - **MVCC**: Multi-Version Concurrency Control for reader-writer isolation.
//! - **Sharded Read/Write Locks**: For reducing contention on hot structures.
//! - **Page-Level Locking**: For granular write access to B-Tree pages.
//!
//! ## Query Execution
//!
//! Queries are processed in stages:
//! 1. **Parse**: SQL -> AST
//! 2. **Plan**: AST -> Logical Plan -> Physical Plan
//! 3. **Execute**: Physical Plan -> Volcano-style Iterator -> Results
//!
//! ## Example
//!
//! ```rust
//! # use turdb::Database;
//! # use tempfile::tempdir;
//! # let dir = tempdir().unwrap();
//! # let path = dir.path().join("mydb");
//! let db = Database::create(&path).unwrap();
//! db.execute("CREATE TABLE foo (id int)").unwrap();
//! db.execute("INSERT INTO foo VALUES (1)").unwrap();
//! ```

use crate::database::convert::convert_value_with_type;
use crate::database::dirty_tracker::ShardedDirtyTracker;
use crate::database::query::{
    build_simple_column_map, compare_owned_values, find_limit, find_nested_subquery,
    find_plan_source, find_projections, find_sort_exec, find_table_scan, has_aggregate,
    has_filter, has_order_by_expression, has_window, is_simple_count_star, materialize_table_rows,
    materialize_table_rows_with_def, PlanSource,
};
use crate::database::row::Row;
use crate::database::transaction::ActiveTransaction;
use crate::database::{ExecuteResult, RecoveryInfo};
use crate::memory::{MemoryBudget, PageBufferPool, Pool};
use crate::mvcc::TransactionManager;

use crate::schema::Catalog;

use crate::sql::builder::ExecutorBuilder;
use crate::sql::context::ExecutionContext;
use crate::sql::executor::{
    BTreeSource, Executor, ExecutorRow, MaterializedRowSource, ReverseBTreeSource, RowSource,
    StreamingBTreeSource,
};
use crate::sql::planner::Planner;

use crate::sql::Parser;
use crate::storage::{
    FileKey, FileManager, TableFileHeader, Wal, WalStoragePerTable, CATALOG_FILE_NAME,
    DEFAULT_SCHEMA,
};
use crate::types::{create_record_schema, DataType, OwnedValue, Value};
use bumpalo::Bump;
use eyre::{bail, ensure, Result, WrapErr};

use parking_lot::{Mutex, RwLock};

use std::path::{Path, PathBuf};
use std::sync::atomic::Ordering as AtomicOrdering;
use std::collections::HashMap;
use std::sync::Arc;

use crate::database::timing::{INSERT_TIME_NS, PARSE_TIME_NS};

/// Aggregate group state: group key hashes -> (accumulated group values, (count, sum) pairs for AVG)
type AggregateGroups = HashMap<Vec<u64>, (Vec<OwnedValue>, Vec<(i64, f64)>)>;

/// Result type for subquery materialization: (rows, column_map)
type MaterializedSubqueryResult = (Vec<Vec<OwnedValue>>, Vec<(String, usize)>);

/// Default number of pre-allocated page buffers in the buffer pool.
/// Sized to handle typical concurrent commit workloads without allocation.
const DEFAULT_BUFFER_POOL_SIZE: usize = 16;

pub(crate) struct SharedDatabase {
    pub(crate) path: PathBuf,
    pub(crate) file_manager: RwLock<Option<FileManager>>,
    pub(crate) catalog: RwLock<Option<Catalog>>,
    pub(crate) wal: Mutex<Option<Wal>>,
    pub(crate) wal_dir: PathBuf,

    pub(crate) next_row_id: std::sync::atomic::AtomicU64,
    pub(crate) next_table_id: std::sync::atomic::AtomicU64,
    pub(crate) next_index_id: std::sync::atomic::AtomicU64,
    pub(crate) closed: std::sync::atomic::AtomicBool,
    pub(crate) wal_enabled: std::sync::atomic::AtomicBool,
    pub(crate) dirty_tracker: ShardedDirtyTracker,
    pub(crate) txn_manager: TransactionManager,
    pub(crate) table_id_lookup: RwLock<hashbrown::HashMap<u32, (String, String)>>,
    /// Group commit queue for batching WAL flushes across concurrent transactions
    pub(crate) group_commit_queue: super::group_commit::GroupCommitQueue,
    /// Fine-grained page-level lock manager for write concurrency
    pub(crate) page_locks: super::page_locks::PageLockManager,
    /// Memory budget for Grace Hash Join spill-to-disk (default: 256KB)
    pub(crate) join_memory_budget: std::sync::atomic::AtomicUsize,
    /// Cached HNSW indexes for vector operations
    pub(crate) hnsw_indexes:
        RwLock<hashbrown::HashMap<FileKey, Arc<RwLock<crate::hnsw::PersistentHnswIndex>>>>,
    /// Global memory budget for the database (default: 25% of system RAM, minimum 4MB)
    pub(crate) memory_budget: Arc<MemoryBudget>,
    /// Current database operating mode (ReadWrite or ReadOnlyDegraded)
    pub(crate) mode: RwLock<super::DatabaseMode>,
    /// Pre-allocated buffer pool for zero-allocation commit operations
    pub(crate) page_buffer_pool: PageBufferPool,
}

pub struct Database {
    pub(crate) shared: Arc<SharedDatabase>,
    pub(crate) active_txn: Mutex<Option<ActiveTransaction>>,
    pub(crate) foreign_keys_enabled: std::sync::atomic::AtomicBool,
}

impl Clone for Database {
    fn clone(&self) -> Self {
        Self {
            shared: Arc::clone(&self.shared),
            active_txn: Mutex::new(None),
            foreign_keys_enabled: std::sync::atomic::AtomicBool::new(
                self.foreign_keys_enabled.load(AtomicOrdering::Acquire),
            ),
        }
    }
}

impl SharedDatabase {
    pub(crate) fn save_catalog(&self) -> Result<()> {
        use crate::schema::persistence::CatalogPersistence;

        let catalog_path = self.path.join(CATALOG_FILE_NAME);
        let catalog_guard = self.catalog.read();
        if let Some(ref catalog) = *catalog_guard {
            CatalogPersistence::save(catalog, &catalog_path)
                .wrap_err_with(|| format!("failed to save catalog to {:?}", catalog_path))?;
        }
        Ok(())
    }

    pub(crate) fn checkpoint(&self) -> Result<u32> {
        let closed_segments = {
            let guard = self.wal.lock();
            if let Some(wal) = guard.as_ref() {
                wal.rotate_segment()?;
                wal.get_closed_segments()
            } else {
                return Ok(0);
            }
        };

        if closed_segments.is_empty() {
            return Ok(0);
        }

        let root_dir = self.path.join(crate::storage::DEFAULT_SCHEMA);
        let frames = Database::replay_schema_tables_from_segments(&root_dir, &closed_segments)?;

        {
            let guard = self.wal.lock();
            if let Some(wal) = guard.as_ref() {
                wal.remove_closed_segments(&closed_segments)?;
            }
        }

        Ok(frames)
    }
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

        let memory_budget = Arc::new(MemoryBudget::auto_detect());
        let recovery_available = memory_budget.available(Pool::Recovery);

        let estimate = Self::estimate_recovery_cost(&wal_dir)?;

        let (frames_recovered, mode) = if estimate.frame_count > 0 {

            if estimate.estimated_bytes <= recovery_available {
                let frames = Self::recover_all_tables(&path, &wal_dir)?;
                (frames, super::DatabaseMode::ReadWrite)
            } else {
                eprintln!(
                    "[turdb] WAL recovery requires ~{}MB but only {}MB available in recovery pool.",
                    estimate.estimated_bytes / 1_000_000,
                    recovery_available / 1_000_000
                );
                eprintln!(
                    "[turdb] Opening in read-only degraded mode. Run PRAGMA recover_wal to recover."
                );
                (
                    0,
                    super::DatabaseMode::ReadOnlyDegraded {
                        pending_wal_frames: estimate.frame_count,
                    },
                )
            }
        } else {
            (0, super::DatabaseMode::ReadWrite)
        };

        let shared = Arc::new(SharedDatabase {
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
            table_id_lookup: RwLock::new(hashbrown::HashMap::new()),
            group_commit_queue: super::group_commit::GroupCommitQueue::with_default_config(),
            page_locks: super::page_locks::PageLockManager::new(),
            join_memory_budget: std::sync::atomic::AtomicUsize::new(10 * 1024 * 1024),
            hnsw_indexes: RwLock::new(hashbrown::HashMap::new()),
            memory_budget,
            mode: RwLock::new(mode),
            page_buffer_pool: PageBufferPool::new(DEFAULT_BUFFER_POOL_SIZE),
        });

        let db = Self {
            shared,
            active_txn: Mutex::new(None),
            foreign_keys_enabled: AtomicBool::new(true),
        };

        db.ensure_catalog()?;
        db.ensure_system_tables()?;

        let recovery_info = RecoveryInfo {
            frames_recovered,
            wal_size_bytes: estimate.wal_size_bytes,
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
            format!(
                "failed to create default schema directory at {:?}",
                root_dir
            )
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


        let shared = Arc::new(SharedDatabase {
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
            table_id_lookup: RwLock::new(hashbrown::HashMap::new()),
            group_commit_queue: super::group_commit::GroupCommitQueue::with_default_config(),
            page_locks: super::page_locks::PageLockManager::new(),
            join_memory_budget: std::sync::atomic::AtomicUsize::new(10 * 1024 * 1024),
            hnsw_indexes: RwLock::new(hashbrown::HashMap::new()),
            memory_budget: Arc::new(MemoryBudget::auto_detect()),
            mode: RwLock::new(super::DatabaseMode::ReadWrite),
            page_buffer_pool: PageBufferPool::new(DEFAULT_BUFFER_POOL_SIZE),
        });

        let db = Self {
            shared,
            active_txn: Mutex::new(None),
            foreign_keys_enabled: AtomicBool::new(true),
        };

        db.ensure_system_tables()?;

        Ok(db)
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
        if self.shared.file_manager.read().is_some() {
            return Ok(());
        }
        let mut guard = self.shared.file_manager.write();
        if guard.is_none() {
            let fm = FileManager::open(&self.shared.path, 64).wrap_err_with(|| {
                format!("failed to open file manager at {:?}", self.shared.path)
            })?;
            *guard = Some(fm);
        }
        Ok(())
    }

    pub fn ensure_catalog(&self) -> Result<()> {
        if self.shared.catalog.read().is_some() {
            return Ok(());
        }
        let mut guard = self.shared.catalog.write();
        if guard.is_none() {
            let catalog = Self::load_catalog(&self.shared.path)?;

            self.populate_table_id_cache(&catalog);

            let table_count = catalog
                .schemas()
                .values()
                .map(|schema| schema.tables().len())
                .sum::<usize>();
            let estimated_schema_memory = table_count * 1024;
            let _ = self
                .shared
                .memory_budget
                .allocate(Pool::Schema, estimated_schema_memory);

            *guard = Some(catalog);
        }
        Ok(())
    }

    /// Ensures system tables exist in both catalog and storage.
    /// Creates memory_stats and wal_stats tables in the turdb_catalog schema.
    fn ensure_system_tables(&self) -> Result<()> {
        use crate::schema::system_tables::{
            create_memory_stats_table_def, create_wal_stats_table_def, MEMORY_STATS_TABLE,
            SYSTEM_SCHEMA, WAL_STATS_TABLE,
        };

        self.ensure_catalog()?;
        self.ensure_file_manager()?;

        // Create schema directory if it doesn't exist
        {
            let mut file_manager_guard = self.shared.file_manager.write();
            let file_manager = file_manager_guard.as_mut().unwrap();
            if !file_manager.schema_exists(SYSTEM_SCHEMA) {
                file_manager.create_schema(SYSTEM_SCHEMA)?;
            }
        }

        // Collect info about which tables need to be created
        let tables_to_create: Vec<(u64, String, usize)> = {
            let mut catalog_guard = self.shared.catalog.write();
            let catalog = catalog_guard.as_mut().unwrap();

            let schema = catalog.get_schema(SYSTEM_SCHEMA).ok_or_else(|| {
                eyre::eyre!("system schema '{}' not found", SYSTEM_SCHEMA)
            })?;

            let mut tables = Vec::new();

            if !schema.table_exists(MEMORY_STATS_TABLE) {
                // Use allocate_table_id() to stay in sync with SharedDatabase's counter
                let table_id = self.allocate_table_id();
                let table_def = create_memory_stats_table_def(table_id);
                let column_count = table_def.columns().len();
                catalog
                    .get_schema_mut(SYSTEM_SCHEMA)
                    .unwrap()
                    .add_table(table_def);
                tables.push((table_id, MEMORY_STATS_TABLE.to_string(), column_count));
            }

            // Re-check schema after potential modification
            let schema = catalog.get_schema(SYSTEM_SCHEMA).unwrap();
            if !schema.table_exists(WAL_STATS_TABLE) {
                let table_id = self.allocate_table_id();
                let table_def = create_wal_stats_table_def(table_id);
                let column_count = table_def.columns().len();
                catalog
                    .get_schema_mut(SYSTEM_SCHEMA)
                    .unwrap()
                    .add_table(table_def);
                tables.push((table_id, WAL_STATS_TABLE.to_string(), column_count));
            }

            tables
        };

        // Create storage for each new system table
        if !tables_to_create.is_empty() {
            let mut file_manager_guard = self.shared.file_manager.write();
            let file_manager = file_manager_guard.as_mut().unwrap();

            for (table_id, table_name, column_count) in &tables_to_create {
                // Only create storage if it doesn't already exist
                if !file_manager.table_exists(SYSTEM_SCHEMA, table_name) {
                    // Create storage file
                    file_manager.create_table(
                        SYSTEM_SCHEMA,
                        table_name,
                        *table_id,
                        *column_count as u32,
                    )?;

                    // Initialize storage with BTree
                    let storage_arc = file_manager.table_data_mut(SYSTEM_SCHEMA, table_name)?;
                    let mut storage = storage_arc.write();
                    storage.grow(2)?;
                    crate::btree::BTree::create(&mut *storage, 1)?;
                }

                // Add to table ID lookup
                self.shared.table_id_lookup.write().insert(
                    *table_id as u32,
                    (SYSTEM_SCHEMA.to_string(), table_name.clone()),
                );
            }

            // Persist catalog with system tables
            drop(file_manager_guard);
            self.save_catalog()?;
        }

        Ok(())
    }

    fn populate_table_id_cache(&self, catalog: &Catalog) {
        let mut lookup = self.shared.table_id_lookup.write();
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
        let mut guard = self.shared.wal.lock();
        if guard.is_none() {
            let wal = if self.shared.wal_dir.exists() {
                Wal::open(&self.shared.wal_dir)
                    .wrap_err_with(|| format!("failed to open WAL at {:?}", self.shared.wal_dir))?
            } else {
                Wal::create(&self.shared.wal_dir).wrap_err_with(|| {
                    format!("failed to create WAL at {:?}", self.shared.wal_dir)
                })?
            };
            *guard = Some(wal);
        }
        Ok(())
    }

    pub(crate) fn flush_wal_if_autocommit(
        &self,
        file_manager: &mut crate::storage::FileManager,
        schema_name: &str,
        table_name: &str,
        table_id: u32,
    ) -> Result<usize> {
        use std::sync::atomic::Ordering;

        let wal_enabled = self.shared.wal_enabled.load(Ordering::Acquire);
        if !wal_enabled {
            return Ok(0);
        }

        let _txn_guard = self.active_txn.lock();
        if _txn_guard.is_some() {
            return Ok(0);
        }

        if !self.shared.dirty_tracker.has_dirty_pages(table_id) {
            return Ok(0);
        }

        let table_storage_arc = file_manager.table_data(schema_name, table_name)?;
        let table_storage = table_storage_arc.read();
        let mut wal_guard = self.shared.wal.lock();
        let wal = wal_guard
            .as_mut()
            .ok_or_else(|| eyre::eyre!("WAL is enabled but not initialized - this is a bug"))?;
        let frames_written = WalStoragePerTable::flush_wal_for_table(
            &self.shared.dirty_tracker,
            &table_storage,
            wal,
            table_id,
        )
        .wrap_err("failed to flush WAL")?;

        Ok(frames_written as usize)
    }

    fn load_catalog(path: &Path) -> Result<Catalog> {
        use crate::schema::persistence::CatalogPersistence;

        let catalog_path = path.join(CATALOG_FILE_NAME);
        let mut catalog = Catalog::new();
        if catalog_path.exists() {
            CatalogPersistence::load(&catalog_path, &mut catalog)
                .wrap_err_with(|| format!("failed to load catalog from {:?}", catalog_path))?;
        }
        Ok(catalog)
    }

    pub(crate) fn save_catalog(&self) -> Result<()> {
        self.shared.save_catalog()
    }

    pub(crate) fn save_meta(&self) -> Result<()> {
        use crate::storage::{MetaFileHeader, PAGE_SIZE};
        use std::fs::OpenOptions;
        use std::io::{Read, Seek, SeekFrom, Write};
        use std::sync::atomic::Ordering;
        use zerocopy::IntoBytes;

        let meta_path = self.shared.path.join("turdb.meta");

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
        new_header.set_next_table_id(self.shared.next_table_id.load(Ordering::Acquire));
        new_header.set_next_index_id(self.shared.next_index_id.load(Ordering::Acquire));

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
        self.shared.next_table_id.fetch_add(1, Ordering::AcqRel)
    }

    pub(crate) fn allocate_index_id(&self) -> u64 {
        use std::sync::atomic::Ordering;
        self.shared.next_index_id.fetch_add(1, Ordering::AcqRel)
    }

    pub(crate) fn encode_value_as_key<B: crate::encoding::key::KeyBuffer>(
        value: &OwnedValue,
        buf: &mut B,
    ) {
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

        let catalog_guard = self.shared.catalog.read();
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

        let mut file_manager_guard = self.shared.file_manager.write();
        let file_manager = file_manager_guard.as_mut().unwrap();

        fn collect_scalar_subqueries<'a>(
            expr: &'a crate::sql::ast::Expr<'a>,
            subqueries: &mut smallvec::SmallVec<[&'a crate::sql::ast::SelectStmt<'a>; 4]>,
        ) {
            use crate::sql::ast::Expr;
            match expr {
                Expr::Subquery(subq) => {
                    subqueries.push(subq);
                }
                Expr::BinaryOp { left, right, .. } => {
                    collect_scalar_subqueries(left, subqueries);
                    collect_scalar_subqueries(right, subqueries);
                }
                Expr::UnaryOp { expr, .. } => {
                    collect_scalar_subqueries(expr, subqueries);
                }
                Expr::IsNull { expr, .. } => {
                    collect_scalar_subqueries(expr, subqueries);
                }
                Expr::InList { expr, list, .. } => {
                    collect_scalar_subqueries(expr, subqueries);
                    for item in list.iter() {
                        collect_scalar_subqueries(item, subqueries);
                    }
                }
                Expr::Between { expr, low, high, .. } => {
                    collect_scalar_subqueries(expr, subqueries);
                    collect_scalar_subqueries(low, subqueries);
                    collect_scalar_subqueries(high, subqueries);
                }
                Expr::Case { operand, conditions, else_result } => {
                    if let Some(op) = operand {
                        collect_scalar_subqueries(op, subqueries);
                    }
                    for cond in conditions.iter() {
                        collect_scalar_subqueries(cond.condition, subqueries);
                        collect_scalar_subqueries(cond.result, subqueries);
                    }
                    if let Some(else_e) = else_result {
                        collect_scalar_subqueries(else_e, subqueries);
                    }
                }
                Expr::Function(func) => {
                    if let crate::sql::ast::FunctionArgs::Args(args) = &func.args {
                        for arg in args.iter() {
                            collect_scalar_subqueries(arg.value, subqueries);
                        }
                    }
                }
                Expr::Cast { expr, .. } => {
                    collect_scalar_subqueries(expr, subqueries);
                }
                _ => {}
            }
        }

        fn find_filter_predicates<'a>(
            op: &'a crate::sql::planner::PhysicalOperator<'a>,
            predicates: &mut smallvec::SmallVec<[&'a crate::sql::ast::Expr<'a>; 8]>,
        ) {
            use crate::sql::planner::PhysicalOperator;
            match op {
                PhysicalOperator::FilterExec(f) => {
                    predicates.push(f.predicate);
                    find_filter_predicates(f.input, predicates);
                }
                PhysicalOperator::ProjectExec(p) => find_filter_predicates(p.input, predicates),
                PhysicalOperator::LimitExec(l) => find_filter_predicates(l.input, predicates),
                PhysicalOperator::SortExec(s) => find_filter_predicates(s.input, predicates),
                PhysicalOperator::TopKExec(t) => find_filter_predicates(t.input, predicates),
                PhysicalOperator::HashAggregate(a) => find_filter_predicates(a.input, predicates),
                PhysicalOperator::NestedLoopJoin(j) => {
                    find_filter_predicates(j.left, predicates);
                    find_filter_predicates(j.right, predicates);
                }
                PhysicalOperator::GraceHashJoin(j) => {
                    find_filter_predicates(j.left, predicates);
                    find_filter_predicates(j.right, predicates);
                }
                PhysicalOperator::SubqueryExec(s) => find_filter_predicates(s.child_plan, predicates),
                _ => {}
            }
        }

        fn execute_scalar_subquery<'a>(
            subq: &'a crate::sql::ast::SelectStmt<'a>,
            catalog: &crate::schema::catalog::Catalog,
            file_manager: &mut FileManager,
            arena: &'a Bump,
        ) -> Result<OwnedValue> {
            let planner = Planner::new(catalog, arena);
            let stmt = crate::sql::ast::Statement::Select(subq);
            let subq_plan = planner.create_physical_plan(&stmt)?;

            let plan_source = find_plan_source(subq_plan.root);

            if let Some(PlanSource::TableScan(scan)) = plan_source {
                let schema_name = scan.schema.unwrap_or(DEFAULT_SCHEMA);
                let table_name = scan.table;

                let table_def = catalog.resolve_table_in_schema(scan.schema, table_name)?;
                let column_types: Vec<_> = table_def.columns().iter().map(|c| c.data_type()).collect();

                let storage_arc = file_manager.table_data(schema_name, table_name)?;
                let storage = storage_arc.read();

                let root_page = {
                    use crate::storage::TableFileHeader;
                    let page = storage.page(0)?;
                    TableFileHeader::from_bytes(page)?.root_page()
                };
                let source = StreamingBTreeSource::from_btree_scan_with_projections(
                    &storage,
                    root_page,
                    column_types,
                    None,
                )?;

                let column_map = build_simple_column_map(table_def);

                let ctx = ExecutionContext::new(arena);
                let builder = ExecutorBuilder::new(&ctx);
                let mut executor = builder
                    .build_with_source_and_column_map(&subq_plan, source, &column_map)?;

                executor.open()?;
                let result = if let Some(row) = executor.next()? {
                    row.values.first().map(OwnedValue::from).unwrap_or(OwnedValue::Null)
                } else {
                    OwnedValue::Null
                };
                executor.close()?;
                Ok(result)
            } else {
                Ok(OwnedValue::Null)
            }
        }

        let mut scalar_subquery_results: Option<hashbrown::HashMap<usize, OwnedValue>> = None;
        {
            let mut filter_predicates: smallvec::SmallVec<[&crate::sql::ast::Expr; 8]> =
                smallvec::SmallVec::new();
            find_filter_predicates(physical_plan.root, &mut filter_predicates);

            for predicate in filter_predicates {
                let mut subqueries: smallvec::SmallVec<[&crate::sql::ast::SelectStmt; 4]> =
                    smallvec::SmallVec::new();
                collect_scalar_subqueries(predicate, &mut subqueries);

                for subq in subqueries {
                    let key = std::ptr::from_ref(subq) as usize;
                    let map = scalar_subquery_results.get_or_insert_with(hashbrown::HashMap::new);
                    if let hashbrown::hash_map::Entry::Vacant(entry) = map.entry(key) {
                        let result = execute_scalar_subquery(subq, catalog, file_manager, &arena)?;
                        entry.insert(result);
                    }
                }
            }
        }

        fn execute_subquery_recursive<'a>(
            subq: &'a crate::sql::planner::PhysicalSubqueryExec<'a>,
            catalog: &crate::schema::catalog::Catalog,
            file_manager: &mut FileManager,
            arena: &'a Bump,
        ) -> Result<Vec<Vec<OwnedValue>>> {
            if let Some(nested_subq) = find_nested_subquery(subq.child_plan) {
                let nested_rows = execute_subquery_recursive(nested_subq, catalog, file_manager, arena)?;

                let nested_source = MaterializedRowSource::new(nested_rows);

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

                let inner_ctx = ExecutionContext::new(arena);
                let inner_builder = ExecutorBuilder::new(&inner_ctx);
                let mut inner_executor = inner_builder
                    .build_with_source_and_column_map(
                        &inner_plan,
                        nested_source,
                        &nested_column_map,
                    )
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
                let schema_name = inner_scan.schema.unwrap_or(DEFAULT_SCHEMA);
                let table_name = inner_scan.table;

                let inner_table_def = catalog
                    .resolve_table_in_schema(inner_scan.schema, table_name)
                    .wrap_err_with(|| format!("table '{}' not found", table_name))?;

                let column_types: Vec<_> = inner_table_def
                    .columns()
                    .iter()
                    .map(|c| c.data_type())
                    .collect();

                let storage_arc = file_manager
                    .table_data(schema_name, table_name)
                    .wrap_err_with(|| {
                        format!(
                            "failed to open table storage for {}.{}",
                            schema_name, table_name
                        )
                    })?;
                let storage = storage_arc.read();

                let root_page = {
                    use crate::storage::TableFileHeader;
                    let page = storage.page(0)?;
                    TableFileHeader::from_bytes(page)?.root_page()
                };
                let inner_source = if inner_scan.reverse {
                    BTreeSource::Reverse(
                        ReverseBTreeSource::from_btree_scan_reverse_with_projections(
                            &storage,
                            root_page,
                            column_types,
                            None,
                        )
                        .wrap_err("failed to create reverse inner table scan")?,
                    )
                } else {
                    BTreeSource::Forward(
                        StreamingBTreeSource::from_btree_scan_with_projections(
                            &storage,
                            root_page,
                            column_types,
                            None,
                        )
                        .wrap_err("failed to create inner table scan")?,
                    )
                };

                let inner_arena = Bump::new();
                let inner_plan = crate::sql::planner::PhysicalPlan {
                    root: subq.child_plan,
                    output_schema: subq.output_schema.clone(),
                };

                let inner_table_column_map = build_simple_column_map(inner_table_def);

                let inner_ctx = ExecutionContext::new(&inner_arena);
                let inner_builder = ExecutorBuilder::new(&inner_ctx);
                let mut inner_executor = inner_builder
                    .build_with_source_and_column_map(
                        &inner_plan,
                        inner_source,
                        &inner_table_column_map,
                    )
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
            } else if let Some(inner_join) = find_join_in_subquery(subq.child_plan) {
                let (mut join_rows, join_column_map) =
                    execute_join_in_subquery(inner_join, subq.child_plan, &subq.output_schema, catalog, file_manager)?;

                fn get_limit_from_plan<'a>(
                    op: &'a crate::sql::planner::PhysicalOperator<'a>,
                ) -> (Option<usize>, usize) {
                    use crate::sql::planner::PhysicalOperator;
                    match op {
                        PhysicalOperator::LimitExec(l) => (l.limit.map(|v| v as usize), l.offset.unwrap_or(0) as usize),
                        PhysicalOperator::FilterExec(f) => get_limit_from_plan(f.input),
                        PhysicalOperator::ProjectExec(p) => get_limit_from_plan(p.input),
                        PhysicalOperator::SortExec(s) => get_limit_from_plan(s.input),
                        PhysicalOperator::TopKExec(t) => (Some(t.limit as usize), t.offset.unwrap_or(0) as usize),
                        _ => (None, 0),
                    }
                }

                let (limit, offset) = get_limit_from_plan(subq.child_plan);

                if offset > 0 {
                    join_rows = join_rows.into_iter().skip(offset).collect();
                }

                if let Some(lim) = limit {
                    join_rows.truncate(lim);
                }

                fn get_project_from_plan<'a>(
                    op: &'a crate::sql::planner::PhysicalOperator<'a>,
                ) -> Option<&'a [&'a crate::sql::ast::Expr<'a>]> {
                    use crate::sql::planner::PhysicalOperator;
                    match op {
                        PhysicalOperator::ProjectExec(p) => Some(p.expressions),
                        PhysicalOperator::LimitExec(l) => get_project_from_plan(l.input),
                        PhysicalOperator::SortExec(s) => get_project_from_plan(s.input),
                        PhysicalOperator::TopKExec(t) => get_project_from_plan(t.input),
                        _ => None,
                    }
                }

                if let Some(projections) = get_project_from_plan(subq.child_plan) {
                    let project_arena = Bump::new();
                    let compiled_projections: Vec<crate::sql::predicate::CompiledPredicate> = projections
                        .iter()
                        .map(|expr| crate::sql::predicate::CompiledPredicate::new(expr, join_column_map.clone()))
                        .collect();

                    let _ctx = ExecutionContext::new(&project_arena);

                    let mut projected_rows: Vec<Vec<OwnedValue>> = Vec::new();
                    for row_owned in &join_rows {
                        let values: smallvec::SmallVec<[Value<'_>; 16]> =
                            row_owned.iter().map(|v| v.to_value()).collect();
                        let row_ref = ExecutorRow::new(&values);

                        let projected: Vec<OwnedValue> = compiled_projections
                            .iter()
                            .map(|pred| {
                                pred.evaluate_to_value(&row_ref)
                                    .map(|v| OwnedValue::from(&v))
                                    .unwrap_or(OwnedValue::Null)
                            })
                            .collect();
                        projected_rows.push(projected);
                    }
                    Ok(projected_rows)
                } else {
                    Ok(join_rows)
                }
            } else {
                bail!("subquery inner plan must have a table scan")
            }
        }

        fn find_join_in_subquery<'a>(
            op: &'a crate::sql::planner::PhysicalOperator<'a>,
        ) -> Option<&'a crate::sql::planner::PhysicalNestedLoopJoin<'a>> {
            use crate::sql::planner::PhysicalOperator;
            match op {
                PhysicalOperator::NestedLoopJoin(join) => Some(join),
                PhysicalOperator::FilterExec(f) => find_join_in_subquery(f.input),
                PhysicalOperator::ProjectExec(p) => find_join_in_subquery(p.input),
                PhysicalOperator::LimitExec(l) => find_join_in_subquery(l.input),
                PhysicalOperator::SortExec(s) => find_join_in_subquery(s.input),
                PhysicalOperator::TopKExec(t) => find_join_in_subquery(t.input),
                PhysicalOperator::HashAggregate(a) => find_join_in_subquery(a.input),
                _ => None,
            }
        }

        fn execute_join_in_subquery<'a>(
            join: &'a crate::sql::planner::PhysicalNestedLoopJoin<'a>,
            _full_plan: &'a crate::sql::planner::PhysicalOperator<'a>,
            _output_schema: &crate::sql::planner::OutputSchema<'a>,
            catalog: &crate::schema::catalog::Catalog,
            file_manager: &mut FileManager,
        ) -> Result<MaterializedSubqueryResult> {
            fn get_table_scan<'a>(
                op: &'a crate::sql::planner::PhysicalOperator<'a>,
            ) -> Option<&'a crate::sql::planner::PhysicalTableScan<'a>> {
                use crate::sql::planner::PhysicalOperator;
                match op {
                    PhysicalOperator::TableScan(scan) => Some(scan),
                    PhysicalOperator::FilterExec(f) => get_table_scan(f.input),
                    PhysicalOperator::ProjectExec(p) => get_table_scan(p.input),
                    PhysicalOperator::LimitExec(l) => get_table_scan(l.input),
                    _ => None,
                }
            }

            let left_scan = get_table_scan(join.left);
            let right_scan = get_table_scan(join.right);

            let mut left_rows: Vec<Vec<OwnedValue>> = Vec::new();
            let mut right_rows: Vec<Vec<OwnedValue>> = Vec::new();
            let mut column_map: Vec<(String, usize)> = Vec::new();
            let mut idx = 0usize;

            if let Some(scan) = left_scan {
                let schema_name = scan.schema.unwrap_or(DEFAULT_SCHEMA);
                let table_name = scan.table;
                let alias = scan.alias.unwrap_or(table_name);
                let table_def = catalog.resolve_table_in_schema(scan.schema, table_name)?;
                let column_types: Vec<_> =
                    table_def.columns().iter().map(|c| c.data_type()).collect();
                let storage_arc = file_manager.table_data(schema_name, table_name)?;
                let storage = storage_arc.read();
                let root_page = {
                    use crate::storage::TableFileHeader;
                    let page = storage.page(0)?;
                    TableFileHeader::from_bytes(page)?.root_page()
                };
                let mut source = StreamingBTreeSource::from_btree_scan_with_projections(
                    &storage,
                    root_page,
                    column_types,
                    None,
                )?;
                while let Some(row) = source.next_row()? {
                    left_rows.push(row.iter().map(OwnedValue::from).collect());
                }
                for col in table_def.columns() {
                    column_map.push((col.name().to_lowercase(), idx));
                    column_map.push((format!("{}.{}", alias, col.name()).to_lowercase(), idx));
                    if scan.alias.is_some() {
                        column_map.push((format!("{}.{}", table_name, col.name()).to_lowercase(), idx));
                    }
                    idx += 1;
                }
            }

            if let Some(scan) = right_scan {
                let schema_name = scan.schema.unwrap_or(DEFAULT_SCHEMA);
                let table_name = scan.table;
                let alias = scan.alias.unwrap_or(table_name);
                let table_def = catalog.resolve_table_in_schema(scan.schema, table_name)?;
                let column_types: Vec<_> =
                    table_def.columns().iter().map(|c| c.data_type()).collect();
                let storage_arc = file_manager.table_data(schema_name, table_name)?;
                let storage = storage_arc.read();
                let root_page = {
                    use crate::storage::TableFileHeader;
                    let page = storage.page(0)?;
                    TableFileHeader::from_bytes(page)?.root_page()
                };
                let mut source = StreamingBTreeSource::from_btree_scan_with_projections(
                    &storage,
                    root_page,
                    column_types,
                    None,
                )?;
                while let Some(row) = source.next_row()? {
                    right_rows.push(row.iter().map(OwnedValue::from).collect());
                }
                for col in table_def.columns() {
                    column_map.push((col.name().to_lowercase(), idx));
                    column_map.push((format!("{}.{}", alias, col.name()).to_lowercase(), idx));
                    if scan.alias.is_some() {
                        column_map.push((format!("{}.{}", table_name, col.name()).to_lowercase(), idx));
                    }
                    idx += 1;
                }
            }

            let condition_predicate = join.condition.map(|c| {
                crate::sql::predicate::CompiledPredicate::new(c, column_map.clone())
            });

            let mut result_rows: Vec<Vec<OwnedValue>> = Vec::new();
            for left_row in &left_rows {
                for right_row in &right_rows {
                    let mut combined: Vec<OwnedValue> = left_row.clone();
                    combined.extend(right_row.iter().cloned());

                    let passes = if let Some(ref pred) = condition_predicate {
                        let values: smallvec::SmallVec<[Value<'_>; 16]> =
                            combined.iter().map(|v| v.to_value()).collect();
                        let row_ref = ExecutorRow::new(&values);
                        pred.evaluate(&row_ref)
                    } else {
                        true
                    };

                    if passes {
                        result_rows.push(combined);
                    }
                }
            }

            Ok((result_rows, column_map))
        }

        let plan_source = find_plan_source(physical_plan.root);

        if let Some(scan) = is_simple_count_star(physical_plan.root) {
            let schema_name = scan.schema.unwrap_or(DEFAULT_SCHEMA);
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
                let schema_name = scan.schema.unwrap_or(DEFAULT_SCHEMA);
                let table_name = scan.table;

                toast_table_info = Some((schema_name.to_string(), table_name.to_string()));

                let table_def = catalog
                    .resolve_table_in_schema(scan.schema, table_name)
                    .wrap_err_with(|| format!("table '{}' not found", table_name))?;

                let column_types: Vec<_> =
                    table_def.columns().iter().map(|c| c.data_type()).collect();

                let plan_has_filter = has_filter(physical_plan.root);
                let plan_has_aggregate = has_aggregate(physical_plan.root);
                let plan_has_window = has_window(physical_plan.root);
                let plan_has_order_by_expr = has_order_by_expression(physical_plan.root);
                let needs_all_columns = plan_has_filter || plan_has_aggregate || plan_has_window || plan_has_order_by_expr;
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

                let root_page = {
                    use crate::storage::TableFileHeader;
                    let page = storage.page(0)?;
                    TableFileHeader::from_bytes(page)?.root_page()
                };
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

                let ctx = match scalar_subquery_results.as_ref() {
                    None => ExecutionContext::new(&arena),
                    Some(results) => ExecutionContext::with_scalar_subqueries(&arena, results.clone()),
                };
                let builder = ExecutorBuilder::new(&ctx);

                let all_columns_map = build_simple_column_map(table_def);

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

                let schema_name = scan.schema.unwrap_or(DEFAULT_SCHEMA);
                let table_name = scan.table;

                toast_table_info = Some((schema_name.to_string(), table_name.to_string()));

                let table_def = catalog
                    .resolve_table_in_schema(scan.schema, table_name)
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

                let root_page = {
                    use crate::storage::TableFileHeader;
                    let page = storage.page(0)?;
                    TableFileHeader::from_bytes(page)?.root_page()
                };

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

                let ctx = match scalar_subquery_results.as_ref() {
                    None => ExecutionContext::new(&arena),
                    Some(results) => ExecutionContext::with_scalar_subqueries(&arena, results.clone()),
                };
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
                use crate::sql::planner::ScanRange;

                let schema_name = scan.schema.unwrap_or(DEFAULT_SCHEMA);
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

                    let root_page = {
                        use crate::storage::IndexFileHeader;
                        let page = index_storage.page(0)?;
                        let header = IndexFileHeader::from_bytes(page)?;
                        header.root_page()
                    };
                    let index_reader = BTreeReader::new(&index_storage, root_page)?;

                    let mut keys = Vec::new();

                    match &scan.key_range {
                        Some(ScanRange::PrefixScan { prefix }) => {
                            let mut cursor = index_reader.cursor_seek(prefix)?;
                            if cursor.valid() {
                                loop {
                                    let index_key = cursor.key()?;
                                    if !index_key.starts_with(prefix) {
                                        break;
                                    }

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
                        Some(ScanRange::RangeScan { start, end }) => {
                            let mut cursor = if let Some(start_key) = start {
                                index_reader.cursor_seek(start_key)?
                            } else {
                                index_reader.cursor_first()?
                            };

                            if cursor.valid() {
                                loop {
                                    let index_key = cursor.key()?;
                                    if let Some(end_key) = end {
                                        if index_key >= *end_key {
                                            break;
                                        }
                                    }

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
                        Some(ScanRange::FullScan) | None => {
                            let scan_limit = scan.limit;
                            if scan.reverse {
                                let mut cursor = index_reader.cursor_last()?;
                                if cursor.valid() {
                                    loop {
                                        if let Some(limit) = scan_limit {
                                            if keys.len() >= limit {
                                                break;
                                            }
                                        }

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
                                        if let Some(limit) = scan_limit {
                                            if keys.len() >= limit {
                                                break;
                                            }
                                        }

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
                        }
                    }
                    keys
                };

                let mut materialized_rows: Vec<Vec<OwnedValue>> =
                    Vec::with_capacity(row_keys.len());

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

                    let root_page = {
                        use crate::storage::TableFileHeader;
                        let page = table_storage.page(0)?;
                        let header = TableFileHeader::from_bytes(page)?;
                        header.root_page()
                    };
                    let table_reader = BTreeReader::new(&table_storage, root_page)?;

                    for row_key in &row_keys {
                        if let Some(row_data) = table_reader.get(row_key)? {
                            let user_data =
                                crate::database::dml::mvcc_helpers::get_user_data(row_data);
                            let record = RecordView::new(user_data, &schema)?;
                            let row_values = OwnedValue::extract_row_from_record(&record, columns)?;
                            materialized_rows.push(row_values);
                        }
                    }
                }

                let materialized_source = MaterializedRowSource::new(materialized_rows);

                let ctx = match scalar_subquery_results.as_ref() {
                    None => ExecutionContext::new(&arena),
                    Some(results) => ExecutionContext::with_scalar_subqueries(&arena, results.clone()),
                };
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
                let inner_rows = execute_subquery_recursive(subq, catalog, file_manager, &arena)?;

                let materialized_source = MaterializedRowSource::new(inner_rows);

                let subq_column_map: Vec<(String, usize)> = subq
                    .output_schema
                    .columns
                    .iter()
                    .enumerate()
                    .map(|(idx, col)| (col.name.to_lowercase(), idx))
                    .collect();

                let ctx = match scalar_subquery_results.as_ref() {
                    None => ExecutionContext::new(&arena),
                    Some(results) => ExecutionContext::with_scalar_subqueries(&arena, results.clone()),
                };
                let builder = ExecutorBuilder::new(&ctx);
                let mut executor = builder
                    .build_with_source_and_column_map(
                        &physical_plan,
                        materialized_source,
                        &subq_column_map,
                    )
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
            Some(PlanSource::NestedLoopJoin(_))
            | Some(PlanSource::GraceHashJoin(_))
            | Some(PlanSource::HashSemiJoin(_))
            | Some(PlanSource::HashAntiJoin(_)) => {
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
                        PhysicalOperator::TopKExec(t) => find_subquery_in_join(t.input),
                        PhysicalOperator::LimitExec(l) => find_subquery_in_join(l.input),
                        PhysicalOperator::WindowExec(w) => find_subquery_in_join(w.input),
                        _ => None,
                    }
                }

                enum JoinScan<'a> {
                    Table(&'a crate::sql::planner::PhysicalTableScan<'a>),
                    Index(&'a crate::sql::planner::PhysicalIndexScan<'a>),
                    SecondaryIndex(&'a crate::sql::planner::PhysicalSecondaryIndexScan<'a>),
                }

                fn find_scan_in_join<'a>(
                    op: &'a crate::sql::planner::PhysicalOperator<'a>,
                ) -> Option<JoinScan<'a>> {
                    use crate::sql::planner::PhysicalOperator;
                    match op {
                        PhysicalOperator::TableScan(scan) => Some(JoinScan::Table(scan)),
                        PhysicalOperator::IndexScan(scan) => Some(JoinScan::Index(scan)),
                        PhysicalOperator::SecondaryIndexScan(scan) => {
                            Some(JoinScan::SecondaryIndex(scan))
                        }
                        PhysicalOperator::FilterExec(f) => find_scan_in_join(f.input),
                        PhysicalOperator::ProjectExec(p) => find_scan_in_join(p.input),
                        PhysicalOperator::HashAggregate(agg) => find_scan_in_join(agg.input),
                        PhysicalOperator::SortedAggregate(agg) => find_scan_in_join(agg.input),
                        PhysicalOperator::SortExec(s) => find_scan_in_join(s.input),
                        PhysicalOperator::TopKExec(t) => find_scan_in_join(t.input),
                        PhysicalOperator::LimitExec(l) => find_scan_in_join(l.input),
                        PhysicalOperator::WindowExec(w) => find_scan_in_join(w.input),
                        _ => None,
                    }
                }

                fn find_nested_join<'a>(
                    op: &'a crate::sql::planner::PhysicalOperator<'a>,
                ) -> Option<&'a crate::sql::planner::PhysicalNestedLoopJoin<'a>> {
                    use crate::sql::planner::PhysicalOperator;
                    match op {
                        PhysicalOperator::NestedLoopJoin(join) => Some(join),
                        PhysicalOperator::FilterExec(f) => find_nested_join(f.input),
                        PhysicalOperator::ProjectExec(p) => find_nested_join(p.input),
                        PhysicalOperator::LimitExec(l) => find_nested_join(l.input),
                        PhysicalOperator::SortExec(s) => find_nested_join(s.input),
                        PhysicalOperator::TopKExec(t) => find_nested_join(t.input),
                        _ => None,
                    }
                }

                fn execute_nested_join_recursive<'a>(
                    nested_join: &'a crate::sql::planner::PhysicalNestedLoopJoin<'a>,
                    catalog: &crate::schema::catalog::Catalog,
                    file_manager: &mut FileManager,
                ) -> Result<MaterializedSubqueryResult> {
                    let mut left_rows: Vec<Vec<OwnedValue>> = Vec::new();
                    let mut right_rows: Vec<Vec<OwnedValue>> = Vec::new();
                    let mut column_map: Vec<(String, usize)> = Vec::new();
                    let mut idx = 0usize;

                    fn get_scan_from_op<'a>(
                        op: &'a crate::sql::planner::PhysicalOperator<'a>,
                    ) -> Option<&'a crate::sql::planner::PhysicalTableScan<'a>> {
                        use crate::sql::planner::PhysicalOperator;
                        match op {
                            PhysicalOperator::TableScan(scan) => Some(scan),
                            PhysicalOperator::FilterExec(f) => get_scan_from_op(f.input),
                            PhysicalOperator::ProjectExec(p) => get_scan_from_op(p.input),
                            PhysicalOperator::LimitExec(l) => get_scan_from_op(l.input),
                            _ => None,
                        }
                    }

                    fn get_nested_join_from_op<'a>(
                        op: &'a crate::sql::planner::PhysicalOperator<'a>,
                    ) -> Option<&'a crate::sql::planner::PhysicalNestedLoopJoin<'a>> {
                        use crate::sql::planner::PhysicalOperator;
                        match op {
                            PhysicalOperator::NestedLoopJoin(j) => Some(j),
                            PhysicalOperator::FilterExec(f) => get_nested_join_from_op(f.input),
                            PhysicalOperator::ProjectExec(p) => get_nested_join_from_op(p.input),
                            PhysicalOperator::LimitExec(l) => get_nested_join_from_op(l.input),
                            _ => None,
                        }
                    }

                    let left_scan = get_scan_from_op(nested_join.left);
                    let right_scan = get_scan_from_op(nested_join.right);
                    let left_nested = get_nested_join_from_op(nested_join.left);
                    let right_nested = get_nested_join_from_op(nested_join.right);

                    if let Some(scan) = left_scan {
                        let table_name = scan.table;
                        let alias = scan.alias.unwrap_or(table_name);
                        let (rows, _, table_def) = materialize_table_rows_with_def(catalog, file_manager, scan.schema, table_name)?;
                        left_rows = rows;
                        for col in table_def.columns() {
                            column_map.push((col.name().to_lowercase(), idx));
                            column_map.push((format!("{}.{}", alias, col.name()).to_lowercase(), idx));
                            if scan.alias.is_some() {
                                column_map.push((format!("{}.{}", table_name, col.name()).to_lowercase(), idx));
                            }
                            idx += 1;
                        }
                    } else if let Some(nested) = left_nested {
                        let (rows, nested_col_map) = execute_nested_join_recursive(nested, catalog, file_manager)?;
                        left_rows = rows;
                        let row_width = left_rows.first().map(|r| r.len()).unwrap_or(0);
                        for (name, nested_idx) in nested_col_map {
                            eyre::ensure!(
                                nested_idx < row_width || row_width == 0,
                                "column index {} out of bounds for row width {} in nested join",
                                nested_idx,
                                row_width
                            );
                            column_map.push((name, idx + nested_idx));
                        }
                        idx += row_width;
                    }

                    if let Some(scan) = right_scan {
                        let table_name = scan.table;
                        let alias = scan.alias.unwrap_or(table_name);
                        let (rows, _, table_def) = materialize_table_rows_with_def(catalog, file_manager, scan.schema, table_name)?;
                        right_rows = rows;
                        for col in table_def.columns() {
                            column_map.push((col.name().to_lowercase(), idx));
                            column_map.push((format!("{}.{}", alias, col.name()).to_lowercase(), idx));
                            if scan.alias.is_some() {
                                column_map.push((format!("{}.{}", table_name, col.name()).to_lowercase(), idx));
                            }
                            idx += 1;
                        }
                    } else if let Some(nested) = right_nested {
                        let (rows, nested_col_map) = execute_nested_join_recursive(nested, catalog, file_manager)?;
                        right_rows = rows;
                        let row_width = right_rows.first().map(|r| r.len()).unwrap_or(0);
                        for (name, nested_idx) in nested_col_map {
                            eyre::ensure!(
                                nested_idx < row_width || row_width == 0,
                                "column index {} out of bounds for row width {} in nested join",
                                nested_idx,
                                row_width
                            );
                            column_map.push((name, idx + nested_idx));
                        }
                    }

                    let condition_predicate = nested_join.condition.map(|c| {
                        crate::sql::predicate::CompiledPredicate::new(c, column_map.clone())
                    });

                    let mut result_rows: Vec<Vec<OwnedValue>> = Vec::new();
                    for left_row in &left_rows {
                        for right_row in &right_rows {
                            let mut combined: Vec<OwnedValue> = left_row.clone();
                            combined.extend(right_row.iter().cloned());

                            let passes = if let Some(ref pred) = condition_predicate {
                                let values: smallvec::SmallVec<[Value<'_>; 16]> =
                                    combined.iter().map(|v| v.to_value()).collect();
                                let row_ref = ExecutorRow::new(&values);
                                pred.evaluate(&row_ref)
                            } else {
                                true
                            };

                            if passes {
                                result_rows.push(combined);
                            }
                        }
                    }

                    Ok((result_rows, column_map))
                }

                let (left_op, right_op, join_type, condition, join_keys) = match plan_source {
                    Some(PlanSource::NestedLoopJoin(j)) => {
                        (j.left, j.right, j.join_type, j.condition, &[][..])
                    }
                    Some(PlanSource::GraceHashJoin(j)) => {
                        (j.left, j.right, j.join_type, None, j.join_keys)
                    }
                    Some(PlanSource::HashSemiJoin(j)) => {
                        (j.left, j.right, crate::sql::ast::JoinType::Semi, None, j.join_keys)
                    }
                    Some(PlanSource::HashAntiJoin(j)) => {
                        (j.left, j.right, crate::sql::ast::JoinType::Anti, None, j.join_keys)
                    }
                    _ => unreachable!(),
                };

                let left_subq = find_subquery_in_join(left_op);
                let right_subq = find_subquery_in_join(right_op);
                let left_scan = find_scan_in_join(left_op);
                let right_scan = find_scan_in_join(right_op);
                let left_nested_join = find_nested_join(left_op);
                let right_nested_join = find_nested_join(right_op);

                let mut left_rows: Vec<Vec<OwnedValue>> = Vec::new();
                let mut right_rows: Vec<Vec<OwnedValue>> = Vec::new();
                let mut right_col_count = 0usize;

                let mut left_table_name: Option<&str> = None;
                let mut left_alias: Option<&str> = None;
                let mut left_schema: Option<&str> = None;
                let mut right_table_name: Option<&str> = None;
                let mut right_alias: Option<&str> = None;
                let mut right_schema: Option<&str> = None;

                let mut left_nested_join_column_map: Vec<(String, usize)> = Vec::new();
                let mut right_nested_join_column_map: Vec<(String, usize)> = Vec::new();

                if let Some(subq) = left_subq {
                    left_rows = execute_subquery_recursive(subq, catalog, file_manager, &arena)?;
                } else if let Some(scan_info) = &left_scan {
                    let (schema_opt, table_name, alias) = match scan_info {
                        JoinScan::Table(scan) => (scan.schema, scan.table, scan.alias),
                        JoinScan::Index(scan) => (scan.schema, scan.table, None),
                        JoinScan::SecondaryIndex(scan) => (scan.schema, scan.table, None),
                    };
                    left_table_name = Some(table_name);
                    left_alias = alias;
                    left_schema = schema_opt;
                    let (rows, _) = materialize_table_rows(catalog, file_manager, schema_opt, table_name)?;
                    left_rows = rows;
                } else if let Some(nested_join) = left_nested_join {
                    (left_rows, left_nested_join_column_map) =
                        execute_nested_join_recursive(nested_join, catalog, file_manager)?;
                }

                if let Some(subq) = right_subq {
                    right_rows = execute_subquery_recursive(subq, catalog, file_manager, &arena)?;
                    right_col_count = subq.output_schema.columns.len();
                } else if let Some(scan_info) = &right_scan {
                    let (schema_opt, table_name, alias) = match scan_info {
                        JoinScan::Table(scan) => (scan.schema, scan.table, scan.alias),
                        JoinScan::Index(scan) => (scan.schema, scan.table, None),
                        JoinScan::SecondaryIndex(scan) => (scan.schema, scan.table, None),
                    };
                    right_table_name = Some(table_name);
                    right_alias = alias;
                    right_schema = schema_opt;
                    let (rows, column_types) = materialize_table_rows(catalog, file_manager, schema_opt, table_name)?;
                    right_rows = rows;
                    right_col_count = column_types.len();
                } else if let Some(nested_join) = right_nested_join {
                    (right_rows, right_nested_join_column_map) =
                        execute_nested_join_recursive(nested_join, catalog, file_manager)?;
                    right_col_count = right_rows.first().map(|r| r.len()).unwrap_or(0);
                }

                let mut join_column_map: Vec<(String, usize)> = Vec::new();
                let mut idx = 0usize;

                if let Some(subq) = left_subq {
                    for col in subq.output_schema.columns {
                        join_column_map.push((col.name.to_lowercase(), idx));
                        join_column_map
                            .push((format!("{}.{}", subq.alias, col.name).to_lowercase(), idx));
                        idx += 1;
                    }
                } else if let Some(table_name) = left_table_name {
                    let table_def = catalog.resolve_table_in_schema(left_schema, table_name)?;
                    for col in table_def.columns() {
                        join_column_map.push((col.name().to_lowercase(), idx));
                        join_column_map
                            .push((format!("{}.{}", table_name, col.name()).to_lowercase(), idx));
                        if let Some(alias) = left_alias {
                            join_column_map
                                .push((format!("{}.{}", alias, col.name()).to_lowercase(), idx));
                        }
                        idx += 1;
                    }
                } else if !left_nested_join_column_map.is_empty() {
                    for (name, nested_idx) in &left_nested_join_column_map {
                        join_column_map.push((name.clone(), idx + nested_idx));
                    }
                    idx += left_rows.first().map(|r| r.len()).unwrap_or(0);
                }

                if let Some(subq) = right_subq {
                    for col in subq.output_schema.columns {
                        join_column_map.push((col.name.to_lowercase(), idx));
                        join_column_map
                            .push((format!("{}.{}", subq.alias, col.name).to_lowercase(), idx));
                        idx += 1;
                    }
                } else if let Some(table_name) = right_table_name {
                    let table_def = catalog.resolve_table_in_schema(right_schema, table_name)?;
                    for col in table_def.columns() {
                        join_column_map.push((col.name().to_lowercase(), idx));
                        join_column_map
                            .push((format!("{}.{}", table_name, col.name()).to_lowercase(), idx));
                        if let Some(alias) = right_alias {
                            join_column_map
                                .push((format!("{}.{}", alias, col.name()).to_lowercase(), idx));
                        }
                        idx += 1;
                    }
                } else if !right_nested_join_column_map.is_empty() {
                    for (name, nested_idx) in &right_nested_join_column_map {
                        join_column_map.push((name.clone(), idx + nested_idx));
                    }
                }

                let condition_predicate = condition.map(|c| {
                    crate::sql::predicate::CompiledPredicate::new(c, join_column_map.clone())
                });

                fn find_filter_for_join<'a>(
                    op: &'a crate::sql::planner::PhysicalOperator<'a>,
                ) -> Option<&'a crate::sql::ast::Expr<'a>> {
                    use crate::sql::planner::PhysicalOperator;
                    match op {
                        PhysicalOperator::FilterExec(f) => Some(f.predicate),
                        PhysicalOperator::ProjectExec(p) => find_filter_for_join(p.input),
                        PhysicalOperator::LimitExec(l) => find_filter_for_join(l.input),
                        PhysicalOperator::SortExec(s) => find_filter_for_join(s.input),
                        PhysicalOperator::HashAggregate(a) => find_filter_for_join(a.input),
                        _ => None,
                    }
                }

                let left_col_count = if let Some(subq) = left_subq {
                    subq.output_schema.columns.len()
                } else if let Some(table_name) = left_table_name {
                    catalog.resolve_table_in_schema(left_schema, table_name).map(|t| t.columns().len()).unwrap_or(0)
                } else if !left_nested_join_column_map.is_empty() {
                    left_rows.first().map(|r| r.len()).unwrap_or(0)
                } else {
                    0
                };

                let where_predicate = find_filter_for_join(physical_plan.root).map(|expr| {
                    crate::sql::predicate::CompiledPredicate::new(expr, join_column_map.clone())
                });

                let key_indices: Vec<(usize, usize)> = join_keys
                    .iter()
                    .filter_map(|(expr_a, expr_b)| {
                        use crate::sql::ast::Expr;

                        fn find_column_idx(
                            expr: &Expr,
                            column_map: &[(String, usize)],
                        ) -> Option<usize> {
                            if let Expr::Column(col) = expr {
                                let qualified = col
                                    .table
                                    .map(|t| format!("{}.{}", t, col.column).to_lowercase());
                                qualified
                                    .as_ref()
                                    .and_then(|q| {
                                        column_map
                                            .iter()
                                            .find(|(name, _)| name == q)
                                            .map(|(_, idx)| *idx)
                                    })
                                    .or_else(|| {
                                        column_map
                                            .iter()
                                            .find(|(name, _)| {
                                                name.eq_ignore_ascii_case(col.column)
                                            })
                                            .map(|(_, idx)| *idx)
                                    })
                            } else {
                                None
                            }
                        }

                        let idx_a = find_column_idx(expr_a, &join_column_map)?;
                        let idx_b = find_column_idx(expr_b, &join_column_map)?;

                        let a_is_left = idx_a < left_col_count;
                        let b_is_left = idx_b < left_col_count;

                        match (a_is_left, b_is_left) {
                            (true, false) => Some((idx_a, idx_b)),
                            (false, true) => Some((idx_b, idx_a)),
                            _ => None,
                        }
                    })
                    .collect();

                let limit_info = find_limit(physical_plan.root);
                let offset = limit_info.and_then(|(_, o)| o).unwrap_or(0) as usize;
                let limit = limit_info.and_then(|(l, _)| l).map(|l| l as usize);

                let mut result_rows: Vec<Row> = Vec::new();
                let mut skipped = 0usize;
                let mut seen: std::collections::HashSet<Vec<u64>> =
                    std::collections::HashSet::new();
                let mut left_matched: Vec<bool> = vec![false; left_rows.len()];
                let mut right_matched: Vec<bool> = vec![false; right_rows.len()];

                let output_columns = physical_plan.output_schema.columns;
                let output_source_indices: Vec<(usize, crate::types::DataType)> = {
                    let mut name_occurrence_count: hashbrown::HashMap<String, usize> =
                        hashbrown::HashMap::new();
                    output_columns
                        .iter()
                        .map(|col| {
                            let col_name = col.name.to_lowercase();
                            let occurrence = *name_occurrence_count.get(&col_name).unwrap_or(&0);
                            *name_occurrence_count.entry(col_name.clone()).or_insert(0) += 1;
                            let source_idx = join_column_map
                                .iter()
                                .filter(|(name, _)| name == &col_name)
                                .nth(occurrence)
                                .map(|(_, idx)| *idx)
                                .unwrap_or(0);
                            (source_idx, col.data_type)
                        })
                        .collect()
                };

                fn hash_join_key(row: &[OwnedValue], key_indices: &[usize]) -> u64 {
                    use std::hash::{Hash, Hasher};
                    let mut hasher = std::collections::hash_map::DefaultHasher::new();
                    for &idx in key_indices {
                        if let Some(val) = row.get(idx) {
                            match val {
                                OwnedValue::Null => 0u8.hash(&mut hasher),
                                OwnedValue::Bool(b) => b.hash(&mut hasher),
                                OwnedValue::Int(i) => i.hash(&mut hasher),
                                OwnedValue::Float(f) => f.to_bits().hash(&mut hasher),
                                OwnedValue::Text(s) => s.hash(&mut hasher),
                                OwnedValue::Blob(b) => b.hash(&mut hasher),
                                OwnedValue::Vector(v) => {
                                    for f in v {
                                        f.to_bits().hash(&mut hasher);
                                    }
                                }
                                OwnedValue::Date(d) => d.hash(&mut hasher),
                                OwnedValue::Time(t) => t.hash(&mut hasher),
                                OwnedValue::Timestamp(ts) => ts.hash(&mut hasher),
                                OwnedValue::TimestampTz(ts, tz) => {
                                    ts.hash(&mut hasher);
                                    tz.hash(&mut hasher);
                                }
                                OwnedValue::Interval(a, b, c) => {
                                    a.hash(&mut hasher);
                                    b.hash(&mut hasher);
                                    c.hash(&mut hasher);
                                }
                                OwnedValue::Uuid(u) => u.hash(&mut hasher),
                                OwnedValue::Inet4(addr) => addr.hash(&mut hasher),
                                OwnedValue::Inet6(addr) => addr.hash(&mut hasher),
                                OwnedValue::MacAddr(m) => m.hash(&mut hasher),
                                OwnedValue::Jsonb(j) => j.hash(&mut hasher),
                                OwnedValue::Decimal(d, scale) => {
                                    d.hash(&mut hasher);
                                    scale.hash(&mut hasher);
                                }
                                OwnedValue::Point(x, y) => {
                                    x.to_bits().hash(&mut hasher);
                                    y.to_bits().hash(&mut hasher);
                                }
                                OwnedValue::Box(p1, p2) => {
                                    p1.0.to_bits().hash(&mut hasher);
                                    p1.1.to_bits().hash(&mut hasher);
                                    p2.0.to_bits().hash(&mut hasher);
                                    p2.1.to_bits().hash(&mut hasher);
                                }
                                OwnedValue::Circle(center, radius) => {
                                    center.0.to_bits().hash(&mut hasher);
                                    center.1.to_bits().hash(&mut hasher);
                                    radius.to_bits().hash(&mut hasher);
                                }
                                OwnedValue::Enum(a, b) => {
                                    a.hash(&mut hasher);
                                    b.hash(&mut hasher);
                                }
                                OwnedValue::ToastPointer(p) => p.hash(&mut hasher),
                            }
                        }
                    }
                    hasher.finish()
                }

                fn has_null_key(row: &[OwnedValue], key_indices: &[usize]) -> bool {
                    key_indices.iter().any(|&idx| {
                        row.get(idx).is_none_or(|v| matches!(v, OwnedValue::Null))
                    })
                }

                let use_hash_join = !key_indices.is_empty() && condition_predicate.is_none();

                let left_key_indices: Vec<usize> = key_indices.iter().map(|(l, _)| *l).collect();
                let right_key_indices: Vec<usize> = key_indices
                    .iter()
                    .map(|(_, r)| *r - left_col_count)
                    .collect();

                let mut hash_table: hashbrown::HashMap<u64, smallvec::SmallVec<[usize; 4]>> =
                    hashbrown::HashMap::with_capacity(left_rows.len());

                if use_hash_join {
                    for (left_idx, left_row) in left_rows.iter().enumerate() {
                        if has_null_key(left_row, &left_key_indices) {
                            continue;
                        }
                        let hash = hash_join_key(left_row, &left_key_indices);
                        hash_table.entry(hash).or_default().push(left_idx);
                    }
                }

                let mut combined_buf: smallvec::SmallVec<[OwnedValue; 16]> =
                    smallvec::SmallVec::new();

                'outer: {
                    if use_hash_join {
                        for (right_idx, right_row) in right_rows.iter().enumerate() {
                            if has_null_key(right_row, &right_key_indices) {
                                continue;
                            }
                            let hash = hash_join_key(right_row, &right_key_indices);

                            if let Some(left_indices) = hash_table.get(&hash) {
                                for &left_idx in left_indices {
                                    let left_row = &left_rows[left_idx];

                                    let keys_match = left_key_indices
                                        .iter()
                                        .zip(right_key_indices.iter())
                                        .all(|(&li, &ri)| {
                                            left_row.get(li) == right_row.get(ri)
                                        });

                                    if !keys_match {
                                        continue;
                                    }

                                    combined_buf.clear();
                                    combined_buf.extend(left_row.iter().cloned());
                                    combined_buf.extend(right_row.iter().cloned());

                                    let passes_where = if let Some(ref pred) = where_predicate {
                                        let values: smallvec::SmallVec<[Value<'_>; 16]> =
                                            combined_buf.iter().map(|v| v.to_value()).collect();
                                        let row_ref = ExecutorRow::new(&values);
                                        pred.evaluate(&row_ref)
                                    } else {
                                        true
                                    };

                                    if !passes_where {
                                        continue;
                                    }

                                    let was_matched = left_matched[left_idx];
                                    left_matched[left_idx] = true;
                                    right_matched[right_idx] = true;

                                    // For Semi join: skip if already matched (output once per left row)
                                    // For Anti join: don't output during matching phase
                                    if matches!(join_type, crate::sql::ast::JoinType::Semi) {
                                        if was_matched {
                                            continue;
                                        }
                                    } else if matches!(join_type, crate::sql::ast::JoinType::Anti) {
                                        continue;
                                    }

                                    let owned: Vec<OwnedValue> = output_source_indices
                                        .iter()
                                        .map(|(source_idx, data_type)| {
                                            let val = combined_buf
                                                .get(*source_idx)
                                                .cloned()
                                                .unwrap_or(OwnedValue::Null);
                                            convert_value_with_type(&val.to_value(), *data_type)
                                        })
                                        .collect();

                                    if is_distinct {
                                        let key: Vec<u64> = owned
                                            .iter()
                                            .map(|v| {
                                                use std::hash::{Hash, Hasher};
                                                let mut hasher =
                                                    std::collections::hash_map::DefaultHasher::new();
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
                        }
                    } else {
                        for (left_idx, left_row) in left_rows.iter().enumerate() {
                            for (right_idx, right_row) in right_rows.iter().enumerate() {
                                combined_buf.clear();
                                combined_buf.extend(left_row.iter().cloned());
                                combined_buf.extend(right_row.iter().cloned());

                                let should_include = if let Some(ref pred) = condition_predicate {
                                    let values: smallvec::SmallVec<[Value<'_>; 16]> =
                                        combined_buf.iter().map(|v| v.to_value()).collect();
                                    let row_ref = ExecutorRow::new(&values);
                                    pred.evaluate(&row_ref)
                                } else {
                                    true
                                };

                                if !should_include {
                                    continue;
                                }

                                let passes_where = if let Some(ref pred) = where_predicate {
                                    let values: smallvec::SmallVec<[Value<'_>; 16]> =
                                        combined_buf.iter().map(|v| v.to_value()).collect();
                                    let row_ref = ExecutorRow::new(&values);
                                    pred.evaluate(&row_ref)
                                } else {
                                    true
                                };

                                if !passes_where {
                                    continue;
                                }

                                let was_matched = left_matched[left_idx];
                                left_matched[left_idx] = true;
                                right_matched[right_idx] = true;

                                // For Semi join: skip if already matched (output once per left row)
                                // For Anti join: don't output during matching phase
                                if matches!(join_type, crate::sql::ast::JoinType::Semi) {
                                    if was_matched {
                                        continue;
                                    }
                                } else if matches!(join_type, crate::sql::ast::JoinType::Anti) {
                                    continue;
                                }

                                let owned: Vec<OwnedValue> = output_source_indices
                                    .iter()
                                    .map(|(source_idx, data_type)| {
                                        let val = combined_buf
                                            .get(*source_idx)
                                            .cloned()
                                            .unwrap_or(OwnedValue::Null);
                                        convert_value_with_type(&val.to_value(), *data_type)
                                    })
                                    .collect();

                                if is_distinct {
                                    let key: Vec<u64> = owned
                                        .iter()
                                        .map(|v| {
                                            use std::hash::{Hash, Hasher};
                                            let mut hasher =
                                                std::collections::hash_map::DefaultHasher::new();
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
                    }

                    if matches!(
                        join_type,
                        crate::sql::ast::JoinType::Left | crate::sql::ast::JoinType::Full
                    ) {
                        for (left_idx, left_row) in left_rows.iter().enumerate() {
                            if left_matched[left_idx] {
                                continue;
                            }

                            combined_buf.clear();
                            combined_buf.extend(left_row.iter().cloned());
                            combined_buf.extend(std::iter::repeat_n(OwnedValue::Null, right_col_count));

                            let owned: Vec<OwnedValue> = output_source_indices
                                .iter()
                                .map(|(source_idx, data_type)| {
                                    let val = combined_buf
                                        .get(*source_idx)
                                        .cloned()
                                        .unwrap_or(OwnedValue::Null);
                                    convert_value_with_type(&val.to_value(), *data_type)
                                })
                                .collect();

                            if is_distinct {
                                let key: Vec<u64> = owned
                                    .iter()
                                    .map(|v| {
                                        use std::hash::{Hash, Hasher};
                                        let mut hasher =
                                            std::collections::hash_map::DefaultHasher::new();
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
                }

                if matches!(
                    join_type,
                    crate::sql::ast::JoinType::Right | crate::sql::ast::JoinType::Full
                ) {
                    for (right_idx, right_row) in right_rows.iter().enumerate() {
                        if right_matched[right_idx] {
                            continue;
                        }

                        let mut combined: Vec<OwnedValue> =
                            std::iter::repeat_n(OwnedValue::Null, left_col_count).collect();
                        combined.extend(right_row.clone());

                        let output_columns = physical_plan.output_schema.columns;
                        let mut name_occurrence_count: std::collections::HashMap<String, usize> =
                            std::collections::HashMap::new();
                        let owned: Vec<OwnedValue> = output_columns
                            .iter()
                            .map(|col| {
                                let col_name = col.name.to_lowercase();
                                let occurrence =
                                    *name_occurrence_count.get(&col_name).unwrap_or(&0);
                                *name_occurrence_count.entry(col_name.clone()).or_insert(0) += 1;
                                let source_idx = join_column_map
                                    .iter()
                                    .filter(|(name, _)| name == &col_name)
                                    .nth(occurrence)
                                    .map(|(_, idx)| *idx)
                                    .unwrap_or(0);
                                let val = combined
                                    .get(source_idx)
                                    .cloned()
                                    .unwrap_or(OwnedValue::Null);
                                convert_value_with_type(&val.to_value(), col.data_type)
                            })
                            .collect();

                        if is_distinct {
                            let key: Vec<u64> = owned
                                .iter()
                                .map(|v| {
                                    use std::hash::{Hash, Hasher};
                                    let mut hasher =
                                        std::collections::hash_map::DefaultHasher::new();
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
                                break;
                            }
                        }
                    }
                }

                // Anti join: output left rows that weren't matched
                if matches!(join_type, crate::sql::ast::JoinType::Anti) {
                    for (left_idx, left_row) in left_rows.iter().enumerate() {
                        if left_matched[left_idx] {
                            continue;
                        }

                        combined_buf.clear();
                        combined_buf.extend(left_row.iter().cloned());

                        let owned: Vec<OwnedValue> = output_source_indices
                            .iter()
                            .map(|(source_idx, data_type)| {
                                let val = combined_buf
                                    .get(*source_idx)
                                    .cloned()
                                    .unwrap_or(OwnedValue::Null);
                                convert_value_with_type(&val.to_value(), *data_type)
                            })
                            .collect();

                        if is_distinct {
                            let key: Vec<u64> = owned
                                .iter()
                                .map(|v| {
                                    use std::hash::{Hash, Hasher};
                                    let mut hasher =
                                        std::collections::hash_map::DefaultHasher::new();
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
                                break;
                            }
                        }
                    }
                }

                fn find_hash_aggregate<'a>(
                    op: &'a crate::sql::planner::PhysicalOperator<'a>,
                ) -> Option<&'a crate::sql::planner::PhysicalHashAggregate<'a>> {
                    use crate::sql::planner::PhysicalOperator;
                    match op {
                        PhysicalOperator::HashAggregate(agg) => Some(agg),
                        PhysicalOperator::ProjectExec(p) => find_hash_aggregate(p.input),
                        PhysicalOperator::LimitExec(l) => find_hash_aggregate(l.input),
                        PhysicalOperator::SortExec(s) => find_hash_aggregate(s.input),
                        PhysicalOperator::FilterExec(f) => find_hash_aggregate(f.input),
                        _ => None,
                    }
                }

                if let Some(hash_agg) = find_hash_aggregate(physical_plan.root) {
                    use std::collections::HashMap;

                    let group_by_indices: Vec<usize> = hash_agg
                        .group_by
                        .iter()
                        .filter_map(|expr| {
                            if let crate::sql::ast::Expr::Column(col) = expr {
                                let col_name = col.column.to_lowercase();
                                let qualified =
                                    col.table.map(|t| format!("{}.{}", t, col.column).to_lowercase());
                                qualified
                                    .as_ref()
                                    .and_then(|q| {
                                        join_column_map
                                            .iter()
                                            .find(|(name, _)| name == q)
                                            .map(|(_, idx)| *idx)
                                    })
                                    .or_else(|| {
                                        join_column_map
                                            .iter()
                                            .find(|(name, _)| name == &col_name)
                                            .map(|(_, idx)| *idx)
                                    })
                            } else {
                                None
                            }
                        })
                        .collect();

                    let mut groups: AggregateGroups = HashMap::new();

                    for row in &result_rows {
                        let group_key: Vec<u64> = group_by_indices
                            .iter()
                            .map(|&idx| {
                                use std::hash::{Hash, Hasher};
                                let mut hasher = std::collections::hash_map::DefaultHasher::new();
                                if let Some(val) = row.values.get(idx) {
                                    format!("{:?}", val).hash(&mut hasher);
                                }
                                hasher.finish()
                            })
                            .collect();

                        let group_vals: Vec<OwnedValue> = group_by_indices
                            .iter()
                            .filter_map(|&idx| row.values.get(idx).cloned())
                            .collect();

                        let entry = groups
                            .entry(group_key)
                            .or_insert_with(|| (group_vals, vec![(0, 0.0); hash_agg.aggregates.len()]));

                        for (agg_idx, agg_expr) in hash_agg.aggregates.iter().enumerate() {
                            use crate::sql::planner::AggregateFunction;
                            match agg_expr.function {
                                AggregateFunction::Count => {
                                    entry.1[agg_idx].0 += 1;
                                }
                                AggregateFunction::Sum => {
                                    if let Some(crate::sql::ast::Expr::Column(col)) = agg_expr.argument {
                                        let col_name = col.column.to_lowercase();
                                        let qualified = col
                                            .table
                                            .map(|t| format!("{}.{}", t, col.column).to_lowercase());
                                        let arg_idx = qualified
                                            .as_ref()
                                            .and_then(|q| {
                                                join_column_map
                                                    .iter()
                                                    .find(|(name, _)| name == q)
                                                    .map(|(_, idx)| *idx)
                                            })
                                            .or_else(|| {
                                                join_column_map
                                                    .iter()
                                                    .find(|(name, _)| name == &col_name)
                                                    .map(|(_, idx)| *idx)
                                            });
                                        if let Some(idx) = arg_idx {
                                            if let Some(val) = row.values.get(idx) {
                                                match val {
                                                    OwnedValue::Int(i) => {
                                                        entry.1[agg_idx].1 += *i as f64
                                                    }
                                                    OwnedValue::Float(f) => {
                                                        entry.1[agg_idx].1 += *f
                                                    }
                                                    _ => {}
                                                }
                                            }
                                        }
                                    }
                                }
                                _ => {}
                            }
                        }
                    }

                    let mut aggregated_rows: Vec<Row> = groups
                        .into_values()
                        .map(|(group_vals, agg_states)| {
                            let mut values = group_vals;
                            for (agg_idx, agg_expr) in hash_agg.aggregates.iter().enumerate() {
                                use crate::sql::planner::AggregateFunction;
                                let agg_val = match agg_expr.function {
                                    AggregateFunction::Count => OwnedValue::Int(agg_states[agg_idx].0),
                                    AggregateFunction::Sum => OwnedValue::Float(agg_states[agg_idx].1),
                                    _ => OwnedValue::Null,
                                };
                                values.push(agg_val);
                            }
                            Row::new(values)
                        })
                        .collect();

                    aggregated_rows.sort_by(|a, b| {
                        for (i, _) in group_by_indices.iter().enumerate() {
                            if i < a.values.len() && i < b.values.len() {
                                match (&a.values[i], &b.values[i]) {
                                    (OwnedValue::Int(a_val), OwnedValue::Int(b_val)) => {
                                        match a_val.cmp(b_val) {
                                            std::cmp::Ordering::Equal => continue,
                                            other => return other,
                                        }
                                    }
                                    _ => continue,
                                }
                            }
                        }
                        std::cmp::Ordering::Equal
                    });

                    return Ok((column_names, aggregated_rows));
                }

                if let Some(sort_exec) = find_sort_exec(physical_plan.root) {
                    if !sort_exec.order_by.is_empty() {
                        let output_column_map: Vec<(String, usize)> = output_columns
                            .iter()
                            .enumerate()
                            .map(|(idx, col)| (col.name.to_lowercase(), idx))
                            .collect();

                        let mut extended_output_map: Vec<(String, usize)> = output_column_map.clone();
                        for (name, idx) in &output_column_map {
                            if let Some((_full_name, join_idx)) = join_column_map.iter().find(|(n, _)| n == name) {
                                for (alias_name, alias_join_idx) in &join_column_map {
                                    if *alias_join_idx == *join_idx && alias_name != name {
                                        extended_output_map.push((alias_name.clone(), *idx));
                                    }
                                }
                            }
                        }

                        let sort_key_indices: Vec<(usize, bool)> = sort_exec
                            .order_by
                            .iter()
                            .filter_map(|key| {
                                if let crate::sql::ast::Expr::Column(col) = key.expr {
                                    let col_name = col.column.to_lowercase();
                                    let qualified = col
                                        .table
                                        .map(|t| format!("{}.{}", t, col.column).to_lowercase());
                                    let idx = qualified
                                        .as_ref()
                                        .and_then(|q| {
                                            extended_output_map.iter().find(|(n, _)| n == q).map(|(_, i)| *i)
                                        })
                                        .or_else(|| {
                                            extended_output_map
                                                .iter()
                                                .find(|(n, _)| n == &col_name)
                                                .map(|(_, i)| *i)
                                        });
                                    idx.map(|i| (i, key.ascending))
                                } else {
                                    None
                                }
                            })
                            .collect();

                        if !sort_key_indices.is_empty() {
                            result_rows.sort_by(|a, b| {
                                for &(idx, ascending) in &sort_key_indices {
                                    if idx < a.values.len() && idx < b.values.len() {
                                        let cmp = compare_owned_values(&a.values[idx], &b.values[idx]);
                                        if cmp != std::cmp::Ordering::Equal {
                                            return if ascending { cmp } else { cmp.reverse() };
                                        }
                                    }
                                }
                                std::cmp::Ordering::Equal
                            });
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
                deduplicated
                    .into_iter()
                    .skip(offset)
                    .take(lim as usize)
                    .collect()
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

        let catalog_guard = self.shared.catalog.read();
        let catalog = catalog_guard.as_ref().unwrap();
        let planner = Planner::new(catalog, &arena);
        let physical_plan = planner
            .create_physical_plan(&stmt)
            .wrap_err("failed to create query plan for INSERT...SELECT")?;

        let mut file_manager_guard = self.shared.file_manager.write();
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
            let schema_name = scan.schema.unwrap_or(DEFAULT_SCHEMA);
            let table_name = scan.table;

            let table_def = catalog
                .resolve_table_in_schema(scan.schema, table_name)
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

            let root_page = {
                use crate::storage::TableFileHeader;
                let page = storage.page(0)?;
                TableFileHeader::from_bytes(page)?.root_page()
            };
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
        PARSE_TIME_NS.fetch_add(
            parse_start.elapsed().as_nanos() as u64,
            AtomicOrdering::Relaxed,
        );

        self.execute_statement(&stmt, sql, &arena, None)
    }

    pub fn execute_with_params(&self, sql: &str, params: &[OwnedValue]) -> Result<ExecuteResult> {
        let parse_start = std::time::Instant::now();
        let arena = Bump::new();
        let mut parser = Parser::new(sql, &arena);
        let stmt = parser
            .parse_statement()
            .wrap_err("failed to parse SQL statement")?;
        PARSE_TIME_NS.fetch_add(
            parse_start.elapsed().as_nanos() as u64,
            AtomicOrdering::Relaxed,
        );

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
            INSERT_TIME_NS.fetch_add(
                insert_start.elapsed().as_nanos() as u64,
                AtomicOrdering::Relaxed,
            );
            result
        }) {
            return result;
        }

        if let Some(result) =
            prepared.with_cached_update_plan(|plan| self.execute_update_cached(plan, params))
        {
            return result;
        }

        let parse_start = std::time::Instant::now();
        let arena = Bump::new();
        let mut parser = Parser::new(prepared.sql(), &arena);
        let stmt = parser
            .parse_statement()
            .wrap_err("failed to parse SQL statement")?;
        PARSE_TIME_NS.fetch_add(
            parse_start.elapsed().as_nanos() as u64,
            AtomicOrdering::Relaxed,
        );

        if let crate::sql::ast::Statement::Insert(insert) = &stmt {
            self.ensure_catalog()?;
            self.ensure_file_manager()?;

            let catalog_guard = self.shared.catalog.read();
            let catalog = catalog_guard.as_ref().unwrap();

            let table_name = insert.table.name;
            let schema_name = insert
                .table
                .schema
                .unwrap_or(crate::storage::DEFAULT_SCHEMA);

            if let Ok(table_def) = catalog.resolve_table_in_schema(insert.table.schema, table_name) {
                let column_types: Vec<_> =
                    table_def.columns().iter().map(|c| c.data_type()).collect();

                if let Ok(storage_arc) = {
                    let mut fm_guard = self.shared.file_manager.write();
                    let fm = fm_guard.as_mut().unwrap();
                    fm.table_data(schema_name, table_name)
                } {
                    use crate::schema::Constraint;
                    use std::collections::HashSet;

                    let mut indexes = Vec::new();
                    let mut seen_names = HashSet::new();

                    for (idx, col) in table_def.columns().iter().enumerate() {
                        let is_pk = col.has_constraint(&Constraint::PrimaryKey);
                        let is_unique = col.has_constraint(&Constraint::Unique);
                        if is_pk || is_unique {
                            let name = if is_pk {
                                format!("{}_pkey", col.name())
                            } else {
                                format!("{}_key", col.name())
                            };
                            if seen_names.insert(name.clone()) {
                                indexes.push(super::prepared::CachedIndexPlan {
                                    name,
                                    is_pk,
                                    is_unique: true,
                                    col_indices: vec![idx],
                                    storage: std::cell::RefCell::new(None),
                                });
                            }
                        }
                    }

                    for idx_def in table_def.indexes() {
                        let name = idx_def.name().to_string();
                        if seen_names.contains(&name) {
                            continue;
                        }

                        let col_indices: Vec<usize> = idx_def
                            .columns()
                            .filter_map(|cname| {
                                table_def
                                    .columns()
                                    .iter()
                                    .position(|c| c.name().eq_ignore_ascii_case(cname))
                            })
                            .collect();

                        if !col_indices.is_empty() {
                            seen_names.insert(name.clone());
                            indexes.push(super::prepared::CachedIndexPlan {
                                name,
                                is_pk: false,
                                is_unique: idx_def.is_unique(),
                                col_indices,
                                storage: std::cell::RefCell::new(None),
                            });
                        }
                    }

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
                        storage: std::cell::RefCell::new(Some(std::sync::Arc::downgrade(
                            &storage_arc,
                        ))),
                        record_buffer: std::cell::RefCell::new(Vec::with_capacity(256)),
                        indexes,
                    };

                    prepared.set_cached_insert_plan(cached_plan);
                }
            }
            drop(catalog_guard);
        }

        if let crate::sql::ast::Statement::Update(update) = &stmt {
            self.ensure_catalog()?;
            self.ensure_file_manager()?;

            let catalog_guard = self.shared.catalog.read();
            let catalog = catalog_guard.as_ref().unwrap();

            let table_name = update.table.name;
            let schema_name = update
                .table
                .schema
                .unwrap_or(crate::storage::DEFAULT_SCHEMA);

            if let Ok(table_def) = catalog.resolve_table_in_schema(update.table.schema, table_name) {
                let column_types: Vec<_> =
                    table_def.columns().iter().map(|c| c.data_type()).collect();

                if let Ok(storage_arc) = {
                    let mut fm_guard = self.shared.file_manager.write();
                    let fm = fm_guard.as_mut().unwrap();
                    fm.table_data(schema_name, table_name)
                } {
                    use crate::schema::Constraint;
                    let unique_col_indices: Vec<usize> = table_def
                        .columns()
                        .iter()
                        .enumerate()
                        .filter(|(_, col)| col.has_constraint(&Constraint::Unique))
                        .map(|(idx, _)| idx)
                        .collect();

                    let assignment_indices: Vec<(usize, String)> = update
                        .assignments
                        .iter()
                        .filter_map(|assignment| {
                            table_def
                                .columns()
                                .iter()
                                .position(|col| {
                                    col.name().eq_ignore_ascii_case(assignment.column.column)
                                })
                                .map(|idx| (idx, format!("{:?}", assignment.value)))
                        })
                        .collect();

                    let where_clause_str = update.where_clause.as_ref().map(|w| format!("{:?}", w));

                    let all_params = update
                        .assignments
                        .iter()
                        .all(|a| matches!(a.value, crate::sql::ast::Expr::Parameter(_)));

                    let (is_simple_pk_update, pk_column_index) =
                        if let Some(ref where_clause) = update.where_clause {
                            use crate::schema::Constraint;
                            use crate::sql::ast::{BinaryOperator, Expr};

                            if let Expr::BinaryOp {
                                left,
                                op: BinaryOperator::Eq,
                                right,
                            } = where_clause
                            {
                                let pk_idx = table_def
                                    .columns()
                                    .iter()
                                    .position(|c| c.has_constraint(&Constraint::PrimaryKey));

                                if let Some(pk_idx) = pk_idx {
                                    let pk_col_name = table_def.columns()[pk_idx].name();

                                    let is_pk_match = match (&**left, &**right) {
                                        (Expr::Column(c), Expr::Parameter(_))
                                            if c.column.eq_ignore_ascii_case(pk_col_name) =>
                                        {
                                            true
                                        }
                                        (Expr::Parameter(_), Expr::Column(c))
                                            if c.column.eq_ignore_ascii_case(pk_col_name) =>
                                        {
                                            true
                                        }
                                        _ => false,
                                    };

                                    (
                                        is_pk_match && all_params,
                                        if is_pk_match { Some(pk_idx) } else { None },
                                    )
                                } else {
                                    (false, None)
                                }
                            } else {
                                (false, None)
                            }
                        } else {
                            (false, None)
                        };

                    let cached_plan = super::prepared::CachedUpdatePlan {
                        table_id: table_def.id(),
                        schema_name: schema_name.to_string(),
                        table_name: table_name.to_string(),
                        column_count: column_types.len(),
                        column_types,
                        record_schema: create_record_schema(table_def.columns()),
                        assignment_indices,
                        where_clause_str,
                        unique_col_indices,
                        root_page: std::cell::Cell::new(0),
                        storage: std::cell::RefCell::new(Some(std::sync::Arc::downgrade(
                            &storage_arc,
                        ))),
                        original_sql: prepared.sql().to_string(),
                        row_buffer: std::cell::RefCell::new(Vec::new()),
                        key_buffer: std::cell::RefCell::new(Vec::new()),
                        record_buffer: std::cell::RefCell::new(Vec::new()),
                        is_simple_pk_update,
                        pk_column_index,
                        all_params,
                    };

                    prepared.set_cached_update_plan(cached_plan);
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
            Statement::CreateTable(create) => {
                self.check_writable()?;
                self.execute_create_table(create, arena)
            }
            Statement::CreateSchema(create) => {
                self.check_writable()?;
                self.execute_create_schema(create)
            }
            Statement::CreateIndex(create) => {
                self.check_writable()?;
                self.execute_create_index(create, arena)
            }
            Statement::Insert(insert) => {
                self.check_writable()?;
                self.execute_insert(insert, arena, params)
            }
            Statement::Update(update) => {
                self.check_writable()?;
                self.execute_update(update, params.unwrap_or(&[]), arena)
            }
            Statement::Delete(delete) => {
                self.check_writable()?;
                self.execute_delete(delete, params.unwrap_or(&[]), arena)
            }
            Statement::Select(_) => {
                let (columns, rows) = self.query_with_columns(sql)?;
                Ok(ExecuteResult::Select { columns, rows })
            }
            Statement::Drop(drop) => {
                self.check_writable()?;
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
            Statement::Truncate(truncate) => {
                self.check_writable()?;
                self.execute_truncate(truncate)
            }
            Statement::AlterTable(alter) => {
                self.check_writable()?;
                self.execute_alter_table(alter)
            }
            Statement::Set(set) => self.execute_set(set),
            Statement::Explain(explain) => self.execute_explain(explain, arena),
            _ => bail!("unsupported statement type"),
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

    pub(crate) fn get_or_create_hnsw_index(
        &self,
        schema: &str,
        table: &str,
        index_name: &str,
    ) -> Result<Arc<RwLock<crate::hnsw::PersistentHnswIndex>>> {
        let key = FileKey::Hnsw {
            schema: schema.to_string(),
            table: table.to_string(),
            index_name: index_name.to_string(),
        };

        {
            let cache = self.shared.hnsw_indexes.read();
            if let Some(index) = cache.get(&key) {
                return Ok(Arc::clone(index));
            }
        }

        let file_manager_guard = self.shared.file_manager.read();
        let file_manager = file_manager_guard
            .as_ref()
            .ok_or_else(|| eyre::eyre!("file manager not initialized"))?;

        let hnsw_path = file_manager
            .base_path()
            .join(schema)
            .join(format!("{}_{}.hnsw", table, index_name));

        ensure!(
            hnsw_path.exists(),
            "HNSW index file does not exist: {}",
            hnsw_path.display()
        );

        drop(file_manager_guard);

        let index = crate::hnsw::PersistentHnswIndex::open(&hnsw_path)
            .wrap_err_with(|| format!("failed to open HNSW index at {}", hnsw_path.display()))?;

        let index_arc = Arc::new(RwLock::new(index));

        {
            let mut cache = self.shared.hnsw_indexes.write();
            cache.insert(key, Arc::clone(&index_arc));
        }

        Ok(index_arc)
    }
}

impl Drop for Database {
    fn drop(&mut self) {
        self.abort_active_transaction();
    }
}

impl Drop for SharedDatabase {
    fn drop(&mut self) {
        use std::sync::atomic::Ordering;

        if self.closed.load(Ordering::Acquire) {
            return;
        }

        // On clean close, do a full checkpoint (apply WAL + truncate)
        // This ensures all data is persisted to storage files before closing
        if !std::thread::panicking() {
            // Checkpoint: rotate current segment and apply all closed segments
            let closed_segments = {
                let guard = self.wal.lock();
                if let Some(wal) = guard.as_ref() {
                    let _ = wal.rotate_segment();
                    wal.get_closed_segments()
                } else {
                    vec![]
                }
            };

            if !closed_segments.is_empty() {
                let root_dir = self.path.join(crate::storage::DEFAULT_SCHEMA);
                let _ = Database::replay_schema_tables_from_segments(&root_dir, &closed_segments);

                // Remove closed segments (full checkpoint = truncate)
                let guard = self.wal.lock();
                if let Some(wal) = guard.as_ref() {
                    let _ = wal.remove_closed_segments(&closed_segments);
                }
            }
        }

        let _ = self.save_catalog();

        {
            let mut wal_guard = self.wal.lock();
            *wal_guard = None;
        }

        if let Some(mut file_manager_guard) = self.file_manager.try_write() {
            if let Some(ref mut file_manager) = *file_manager_guard {
                let _ = file_manager.sync_all();
            }
        }
    }
}
