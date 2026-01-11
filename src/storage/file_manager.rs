//! # Multi-File Manager
//!
//! This module implements MySQL-style file-per-table architecture for TurDB.
//! Each table has its own set of files (.tbd for data, .idx for indexes,
//! .hnsw for vector indexes), organized into schema directories.
//!
//! ## Directory Structure
//!
//! ```text
//! database_dir/
//! ├── turdb.meta           # Global catalog (16KB paged, always open)
//! ├── root/                # Default schema directory
//! │   ├── users.tbd        # Table data
//! │   ├── users_pk.idx     # Primary key index
//! │   └── users_email.idx  # Secondary index
//! ├── analytics/           # User-created schema
//! │   └── events.tbd
//! └── wal/                 # WAL directory (separate component)
//! ```
//!
//! ## LRU File Management
//!
//! To avoid exhausting file descriptors, the FileManager maintains an LRU
//! cache of open files. When the limit is reached, least-recently-used files
//! are synced and closed. Critical files (turdb.meta) are kept outside the
//! LRU cache and are always open.
//!
//! ## File Limits by Environment
//!
//! | Environment      | Default max_open_files | Rationale                    |
//! |------------------|------------------------|------------------------------|
//! | Embedded/IoT     | 16-32                  | Minimal footprint            |
//! | Desktop app      | 64                     | Safe on macOS (256 limit)    |
//! | Server (default) | 256                    | Handles 25-50 tables         |
//! | Server (tuned)   | 1,024+                 | Large deployments            |
//!
//! ## File Types
//!
//! - `.tbd` (Table Data): 128-byte header with magic, table_id, row_count, etc.
//! - `.idx` (B-tree Index): 128-byte header with index_id, table_id, root_page
//! - `.hnsw` (Vector Index): 128-byte header with dimension, M, ef_construction
//! - `turdb.meta`: Global catalog using 16KB paged format
//!
//! ## File Header Formats
//!
//! ### Table Data File (.tbd)
//! ```text
//! Offset  Size  Description
//! 0       16    Magic: "TurDB Table\x00\x00\x00\x00"
//! 16      8     Table ID
//! 24      8     Row count
//! 32      4     Root page number
//! 36      4     Column count
//! 40      8     First free page
//! 48      8     Auto-increment value
//! 56      72    Reserved
//! ```
//!
//! ### Index File (.idx)
//! ```text
//! Offset  Size  Description
//! 0       16    Magic: "TurDB Index\x00\x00\x00\x00"
//! 16      8     Index ID
//! 24      8     Table ID
//! 32      4     Root page number
//! 36      4     Key column count
//! 40      1     Is unique
//! 41      1     Index type (0=btree)
//! 42      86    Reserved
//! ```
//!
//! ## Thread Safety
//!
//! `FileManager` is designed for single-threaded use within a database
//! connection. For concurrent access, wrap in appropriate synchronization
//! at the Database level.
//!
//! ## Usage Example
//!
//! ```ignore
//! let mut fm = FileManager::open("./mydb")?;
//!
//! // Create a schema
//! fm.create_schema("analytics")?;
//!
//! // Create a table (schema, table_name, table_id, column_count)
//! fm.create_table("analytics", "events", 1, 3)?;
//!
//! // Access table data
//! let storage = fm.table_data("analytics", "events")?;
//! ```
//!
//! ## Performance Characteristics
//!
//! - Schema/table lookup: O(1) hash map access
//! - File open: O(1) if cached, O(1) disk otherwise
//! - LRU eviction: O(1) using linked list
//!
//! ## Safety Considerations
//!
//! Files are synced before closing to ensure durability. The Drop
//! implementation ensures all files are properly closed and synced.

use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

use eyre::{ensure, Result, WrapErr};

use super::driver::StorageKind;
use super::{MmapStorage, PAGE_SIZE};
use parking_lot::RwLock;
use std::sync::Arc;

pub const DEFAULT_MAX_OPEN_FILES: usize = 64;
pub const MIN_MAX_OPEN_FILES: usize = 8;

pub const TABLE_FILE_EXTENSION: &str = "tbd";
pub const INDEX_FILE_EXTENSION: &str = "idx";
pub const HNSW_FILE_EXTENSION: &str = "hnsw";
pub const META_FILE_NAME: &str = "turdb.meta";
pub const CATALOG_FILE_NAME: &str = "turdb.catalog";

pub const HNSW_MAGIC: &[u8; 16] = b"TurDB HNSW\x00\x00\x00\x00\x00\x00";

pub use super::headers::META_MAGIC;
#[cfg(test)]
#[allow(unused_imports)]
pub use super::headers::{INDEX_MAGIC, TABLE_MAGIC};

pub const DEFAULT_SCHEMA: &str = "root";

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum FileKey {
    TableData {
        schema: String,
        table: String,
    },
    Index {
        schema: String,
        table: String,
        index_name: String,
    },
    Hnsw {
        schema: String,
        table: String,
        index_name: String,
    },
}

#[derive(Debug, Default)]
pub struct TableFiles {
    data: Option<PathBuf>,
    indexes: HashMap<String, PathBuf>,
    hnsw_indexes: HashMap<String, PathBuf>,
}

impl TableFiles {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn data(&self) -> Option<&Path> {
        self.data.as_deref()
    }

    pub fn set_data(&mut self, path: PathBuf) {
        self.data = Some(path);
    }

    pub fn indexes(&self) -> &HashMap<String, PathBuf> {
        &self.indexes
    }

    pub fn add_index(&mut self, name: String, path: PathBuf) {
        self.indexes.insert(name, path);
    }

    pub fn hnsw_indexes(&self) -> &HashMap<String, PathBuf> {
        &self.hnsw_indexes
    }

    pub fn add_hnsw_index(&mut self, name: String, path: PathBuf) {
        self.hnsw_indexes.insert(name, path);
    }
}

pub struct LruFileCache<K, V> {
    capacity: usize,
    order: Vec<K>,
    map: HashMap<K, V>,
}

impl<K: Clone + Eq + std::hash::Hash, V> LruFileCache<K, V> {
    pub fn new(capacity: usize) -> Self {
        Self {
            capacity,
            order: Vec::with_capacity(capacity),
            map: HashMap::with_capacity(capacity),
        }
    }

    pub fn get(&mut self, key: &K) -> Option<&V> {
        if self.map.contains_key(key) {
            self.touch(key);
            self.map.get(key)
        } else {
            None
        }
    }

    pub fn get_mut(&mut self, key: &K) -> Option<&mut V> {
        if self.map.contains_key(key) {
            self.touch(key);
            self.map.get_mut(key)
        } else {
            None
        }
    }

    pub fn insert(&mut self, key: K, value: V) -> Option<(K, V)> {
        if self.map.contains_key(&key) {
            self.touch(&key);
            self.map.insert(key, value);
            return None;
        }

        let evicted = if self.order.len() >= self.capacity {
            self.pop_lru()
        } else {
            None
        };

        self.order.push(key.clone());
        self.map.insert(key, value);

        evicted
    }

    pub fn pop_lru(&mut self) -> Option<(K, V)> {
        if self.order.is_empty() {
            return None;
        }

        let key = self.order.remove(0);
        let value = self.map.remove(&key)?;
        Some((key, value))
    }

    pub fn len(&self) -> usize {
        self.map.len()
    }

    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }

    pub fn remove(&mut self, key: &K) -> Option<V> {
        if let Some(pos) = self.order.iter().position(|k| k == key) {
            self.order.remove(pos);
        }
        self.map.remove(key)
    }

    fn touch(&mut self, key: &K) {
        if let Some(pos) = self.order.iter().position(|k| k == key) {
            let k = self.order.remove(pos);
            self.order.push(k);
        }
    }
}

pub struct FileManager {
    base_path: PathBuf,
    max_open_files: usize,
    storage_kind: StorageKind,
    meta_storage: MmapStorage,
    open_files: LruFileCache<FileKey, Arc<RwLock<MmapStorage>>>,
}

impl FileManager {
    pub fn open<P: AsRef<Path>>(path: P, max_open_files: usize) -> Result<Self> {
        Self::open_with_storage(StorageKind::mmap(path.as_ref()), max_open_files)
    }

    /// Opens an existing database with the specified storage kind.
    ///
    /// Currently only `StorageKind::Mmap` is supported on native platforms.
    /// WASM support via `StorageKind::Opfs` will be added in a future release.
    pub fn open_with_storage(storage_kind: StorageKind, max_open_files: usize) -> Result<Self> {
        let max_open_files = max_open_files.max(MIN_MAX_OPEN_FILES);

        let base_path = match &storage_kind {
            StorageKind::Mmap { path } => path.clone(),
            #[cfg(target_arch = "wasm32")]
            StorageKind::Opfs { name } => PathBuf::from(name),
        };

        ensure!(
            base_path.exists(),
            "database directory '{}' does not exist",
            base_path.display()
        );

        let meta_path = base_path.join(META_FILE_NAME);
        ensure!(
            meta_path.exists(),
            "metadata file '{}' does not exist",
            meta_path.display()
        );

        let meta_storage = MmapStorage::open(&meta_path)?;

        let page = meta_storage.page(0)?;
        ensure!(
            &page[..16] == META_MAGIC,
            "invalid database: magic bytes mismatch"
        );

        let version = u32::from_le_bytes(page[16..20].try_into().unwrap());
        ensure!(version == 1, "unsupported database version: {}", version);

        Ok(Self {
            base_path,
            max_open_files,
            storage_kind,
            meta_storage,
            open_files: LruFileCache::new(max_open_files),
        })
    }

    pub fn create<P: AsRef<Path>>(path: P, max_open_files: usize) -> Result<Self> {
        Self::create_with_storage(StorageKind::mmap(path.as_ref()), max_open_files)
    }

    /// Creates a new database with the specified storage kind.
    ///
    /// Currently only `StorageKind::Mmap` is supported on native platforms.
    /// WASM support via `StorageKind::Opfs` will be added in a future release.
    pub fn create_with_storage(storage_kind: StorageKind, max_open_files: usize) -> Result<Self> {
        let max_open_files = max_open_files.max(MIN_MAX_OPEN_FILES);

        let base_path = match &storage_kind {
            StorageKind::Mmap { path } => path.clone(),
            #[cfg(target_arch = "wasm32")]
            StorageKind::Opfs { name } => PathBuf::from(name),
        };

        fs::create_dir_all(&base_path).wrap_err_with(|| {
            format!(
                "failed to create database directory '{}'",
                base_path.display()
            )
        })?;

        let default_schema_path = base_path.join(DEFAULT_SCHEMA);
        fs::create_dir_all(&default_schema_path).wrap_err_with(|| {
            format!(
                "failed to create default schema directory '{}'",
                default_schema_path.display()
            )
        })?;

        let meta_path = base_path.join(META_FILE_NAME);
        let mut meta_storage = MmapStorage::create(&meta_path, 1)?;

        let page = meta_storage.page_mut(0)?;
        page[..16].copy_from_slice(META_MAGIC);

        let version: u32 = 1;
        page[16..20].copy_from_slice(&version.to_le_bytes());

        let page_size: u32 = PAGE_SIZE as u32;
        page[20..24].copy_from_slice(&page_size.to_le_bytes());

        meta_storage.sync()?;

        Ok(Self {
            base_path,
            max_open_files,
            storage_kind,
            meta_storage,
            open_files: LruFileCache::new(max_open_files),
        })
    }

    /// Returns the storage kind used by this FileManager.
    pub fn storage_kind(&self) -> &StorageKind {
        &self.storage_kind
    }

    pub fn base_path(&self) -> &Path {
        &self.base_path
    }

    pub fn max_open_files(&self) -> usize {
        self.max_open_files
    }

    pub fn open_file_count(&self) -> usize {
        self.open_files.len()
    }

    pub fn sync_all(&mut self) -> Result<()> {
        self.meta_storage.sync()?;

        for storage_lock in self.open_files.map.values() {
            storage_lock.write().sync()?;
        }

        Ok(())
    }

    pub fn meta_storage(&self) -> &MmapStorage {
        &self.meta_storage
    }

    pub fn meta_storage_mut(&mut self) -> &mut MmapStorage {
        &mut self.meta_storage
    }

    pub fn create_schema(&mut self, name: &str) -> Result<()> {
        Self::validate_name(name)?;

        let schema_path = self.base_path.join(name);

        ensure!(!schema_path.exists(), "schema '{}' already exists", name);

        fs::create_dir(&schema_path).wrap_err_with(|| {
            format!(
                "failed to create schema directory '{}'",
                schema_path.display()
            )
        })?;

        Ok(())
    }

    pub fn schema_exists(&self, name: &str) -> bool {
        let schema_path = self.base_path.join(name);
        schema_path.exists() && schema_path.is_dir()
    }

    pub fn create_table(
        &mut self,
        schema: &str,
        table: &str,
        table_id: u64,
        column_count: u32,
    ) -> Result<()> {
        use super::headers::TableFileHeader;
        use zerocopy::IntoBytes;

        Self::validate_name(table)?;

        ensure!(
            self.schema_exists(schema),
            "schema '{}' does not exist",
            schema
        );

        let table_path = self.table_file_path(schema, table);

        ensure!(
            !table_path.exists(),
            "table '{}.{}' already exists",
            schema,
            table
        );

        let mut storage = MmapStorage::create(&table_path, 1)?;

        let header = TableFileHeader::new(table_id, 0, 1, column_count, 0, 0);
        let page = storage.page_mut(0)?;
        page[..128].copy_from_slice(header.as_bytes());

        storage.sync()?;

        Ok(())
    }

    pub fn table_exists(&self, schema: &str, table: &str) -> bool {
        let table_path = self.table_file_path(schema, table);
        table_path.exists()
    }

    pub fn drop_table(&mut self, schema: &str, table: &str) -> Result<()> {
        ensure!(
            self.table_exists(schema, table),
            "table '{}.{}' does not exist",
            schema,
            table
        );

        for index_name in self.list_indexes(schema, table)? {
            let index_path = self.index_file_path(schema, table, &index_name);
            fs::remove_file(&index_path).wrap_err_with(|| {
                format!("failed to remove index file '{}'", index_path.display())
            })?;
        }

        let table_path = self.table_file_path(schema, table);
        fs::remove_file(&table_path)
            .wrap_err_with(|| format!("failed to remove table file '{}'", table_path.display()))?;

        Ok(())
    }

    pub fn rename_table(&mut self, schema: &str, old_name: &str, new_name: &str) -> Result<()> {
        ensure!(
            self.table_exists(schema, old_name),
            "table '{}.{}' does not exist",
            schema,
            old_name
        );

        ensure!(
            !self.table_exists(schema, new_name),
            "table '{}.{}' already exists",
            schema,
            new_name
        );

        let old_table_path = self.table_file_path(schema, old_name);
        let new_table_path = self.table_file_path(schema, new_name);

        fs::rename(&old_table_path, &new_table_path).wrap_err_with(|| {
            format!(
                "failed to rename table file from '{}' to '{}'",
                old_table_path.display(),
                new_table_path.display()
            )
        })?;

        for index_name in self.list_indexes(schema, old_name).unwrap_or_default() {
            let old_index_path = self.index_file_path(schema, old_name, &index_name);
            let new_index_path = self.index_file_path(schema, new_name, &index_name);
            if old_index_path.exists() {
                fs::rename(&old_index_path, &new_index_path).wrap_err_with(|| {
                    format!(
                        "failed to rename index file from '{}' to '{}'",
                        old_index_path.display(),
                        new_index_path.display()
                    )
                })?;
            }
        }

        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub fn create_index(
        &mut self,
        schema: &str,
        table: &str,
        index_name: &str,
        index_id: u64,
        table_id: u64,
        key_column_count: u32,
        is_unique: bool,
    ) -> Result<()> {
        use super::headers::{IndexFileHeader, INDEX_TYPE_BTREE};
        use zerocopy::IntoBytes;

        Self::validate_name(index_name)?;

        ensure!(
            self.table_exists(schema, table),
            "table '{}.{}' does not exist",
            schema,
            table
        );

        let index_path = self.index_file_path(schema, table, index_name);

        ensure!(
            !index_path.exists(),
            "index '{}.{}.{}' already exists",
            schema,
            table,
            index_name
        );

        let mut storage = MmapStorage::create(&index_path, 1)?;

        let header = IndexFileHeader::new(
            index_id,
            table_id,
            1,
            key_column_count,
            is_unique,
            INDEX_TYPE_BTREE,
        );
        let page = storage.page_mut(0)?;
        page[..128].copy_from_slice(header.as_bytes());

        storage.sync()?;

        Ok(())
    }

    pub fn index_exists(&self, schema: &str, table: &str, index_name: &str) -> bool {
        let index_path = self.index_file_path(schema, table, index_name);
        index_path.exists()
    }

    pub fn drop_index(&mut self, schema: &str, table: &str, index_name: &str) -> Result<()> {
        let index_path = self.index_file_path(schema, table, index_name);

        if !index_path.exists() {
            return Ok(());
        }

        let key = FileKey::Index {
            schema: schema.to_string(),
            table: table.to_string(),
            index_name: index_name.to_string(),
        };
        if let Some(lock) = self.open_files.remove(&key) {
            drop(lock);
        }

        fs::remove_file(&index_path).wrap_err_with(|| {
            format!("failed to remove index file '{}'", index_path.display())
        })?;

        Ok(())
    }

    pub fn table_data(&mut self, schema: &str, table: &str) -> Result<Arc<RwLock<MmapStorage>>> {
        ensure!(
            self.table_exists(schema, table),
            "table '{}.{}' does not exist",
            schema,
            table
        );

        let key = FileKey::TableData {
            schema: schema.to_string(),
            table: table.to_string(),
        };

        if self.open_files.get(&key).is_none() {
            let path = self.table_file_path(schema, table);
            let storage = MmapStorage::open(&path)?;
            let storage_arc = Arc::new(RwLock::new(storage));
            if let Some((_, lock)) = self.open_files.insert(key.clone(), storage_arc) {
                lock.write().sync()?;
            }
        }

        Ok(self.open_files.get(&key).unwrap().clone())
    }

    pub fn table_data_mut(
        &mut self,
        schema: &str,
        table: &str,
    ) -> Result<Arc<RwLock<MmapStorage>>> {
        ensure!(
            self.table_exists(schema, table),
            "table '{}.{}' does not exist",
            schema,
            table
        );

        let key = FileKey::TableData {
            schema: schema.to_string(),
            table: table.to_string(),
        };

        if self.open_files.get(&key).is_none() {
            let path = self.table_file_path(schema, table);
            let storage = MmapStorage::open(&path)?;
            let storage_arc = Arc::new(RwLock::new(storage));
            if let Some((_, lock)) = self.open_files.insert(key.clone(), storage_arc) {
                lock.write().sync()?;
            }
        }

        Ok(self.open_files.get(&key).unwrap().clone())
    }

    pub fn table_data_mut_with_key(&mut self, key: &FileKey) -> Option<Arc<RwLock<MmapStorage>>> {
        self.open_files.get_mut(key).cloned()
    }

    pub fn make_table_key(schema: &str, table: &str) -> FileKey {
        FileKey::TableData {
            schema: schema.to_string(),
            table: table.to_string(),
        }
    }

    pub fn make_index_key(schema: &str, table: &str, index_name: &str) -> FileKey {
        FileKey::Index {
            schema: schema.to_string(),
            table: table.to_string(),
            index_name: index_name.to_string(),
        }
    }

    pub fn index_data_mut_with_key(&mut self, key: &FileKey) -> Option<Arc<RwLock<MmapStorage>>> {
        self.open_files.get_mut(key).cloned()
    }

    pub fn index_data(
        &mut self,
        schema: &str,
        table: &str,
        index_name: &str,
    ) -> Result<Arc<RwLock<MmapStorage>>> {
        ensure!(
            self.index_exists(schema, table, index_name),
            "index '{}.{}.{}' does not exist",
            schema,
            table,
            index_name
        );

        let key = FileKey::Index {
            schema: schema.to_string(),
            table: table.to_string(),
            index_name: index_name.to_string(),
        };

        if self.open_files.get(&key).is_none() {
            let path = self.index_file_path(schema, table, index_name);
            let storage = MmapStorage::open(&path)?;
            let storage_arc = Arc::new(RwLock::new(storage));
            if let Some((_, lock)) = self.open_files.insert(key.clone(), storage_arc) {
                lock.write().sync()?;
            }
        }

        Ok(self.open_files.get(&key).unwrap().clone())
    }

    pub fn index_data_mut(
        &mut self,
        schema: &str,
        table: &str,
        index_name: &str,
    ) -> Result<Arc<RwLock<MmapStorage>>> {
        ensure!(
            self.index_exists(schema, table, index_name),
            "index '{}.{}.{}' does not exist",
            schema,
            table,
            index_name
        );

        let key = FileKey::Index {
            schema: schema.to_string(),
            table: table.to_string(),
            index_name: index_name.to_string(),
        };

        if self.open_files.get(&key).is_none() {
            let path = self.index_file_path(schema, table, index_name);
            let storage = MmapStorage::open(&path)?;
            let storage_arc = Arc::new(RwLock::new(storage));
            if let Some((_, lock)) = self.open_files.insert(key.clone(), storage_arc) {
                lock.write().sync()?;
            }
        }

        Ok(self.open_files.get_mut(&key).unwrap().clone())
    }

    #[allow(clippy::too_many_arguments)]
    pub fn create_hnsw_index(
        &mut self,
        schema: &str,
        table: &str,
        index_name: &str,
        index_id: u64,
        dimensions: u16,
        m: u16,
        ef_construction: u16,
    ) -> Result<()> {
        Self::validate_name(index_name)?;

        ensure!(
            self.table_exists(schema, table),
            "table '{}.{}' does not exist",
            schema,
            table
        );

        let hnsw_path = self.hnsw_file_path(schema, table, index_name);

        ensure!(
            !hnsw_path.exists(),
            "HNSW index '{}.{}.{}' already exists",
            schema,
            table,
            index_name
        );

        let mut storage = MmapStorage::create(&hnsw_path, 1)?;

        let page = storage.page_mut(0)?;
        page[..16].copy_from_slice(HNSW_MAGIC);
        page[16..24].copy_from_slice(&index_id.to_le_bytes());
        page[32..34].copy_from_slice(&dimensions.to_le_bytes());
        page[34..36].copy_from_slice(&m.to_le_bytes());
        page[36..38].copy_from_slice(&(m * 2).to_le_bytes());
        page[38..40].copy_from_slice(&ef_construction.to_le_bytes());
        page[40..42].copy_from_slice(&32u16.to_le_bytes());
        page[44..48].copy_from_slice(&u32::MAX.to_le_bytes());
        page[48..50].copy_from_slice(&u16::MAX.to_le_bytes());

        storage.sync()?;

        Ok(())
    }

    pub fn hnsw_exists(&self, schema: &str, table: &str, index_name: &str) -> bool {
        let hnsw_path = self.hnsw_file_path(schema, table, index_name);
        hnsw_path.exists()
    }

    pub fn hnsw_data(
        &mut self,
        schema: &str,
        table: &str,
        index_name: &str,
    ) -> Result<Arc<RwLock<MmapStorage>>> {
        ensure!(
            self.hnsw_exists(schema, table, index_name),
            "HNSW index '{}.{}.{}' does not exist",
            schema,
            table,
            index_name
        );

        let key = FileKey::Hnsw {
            schema: schema.to_string(),
            table: table.to_string(),
            index_name: index_name.to_string(),
        };

        if self.open_files.get(&key).is_none() {
            let path = self.hnsw_file_path(schema, table, index_name);
            let storage = MmapStorage::open(&path)?;
            let storage_arc = Arc::new(RwLock::new(storage));
            if let Some((_, lock)) = self.open_files.insert(key.clone(), storage_arc) {
                lock.write().sync()?;
            }
        }

        Ok(self.open_files.get(&key).unwrap().clone())
    }

    pub fn hnsw_data_mut(
        &mut self,
        schema: &str,
        table: &str,
        index_name: &str,
    ) -> Result<Arc<RwLock<MmapStorage>>> {
        ensure!(
            self.hnsw_exists(schema, table, index_name),
            "HNSW index '{}.{}.{}' does not exist",
            schema,
            table,
            index_name
        );

        let key = FileKey::Hnsw {
            schema: schema.to_string(),
            table: table.to_string(),
            index_name: index_name.to_string(),
        };

        if self.open_files.get(&key).is_none() {
            let path = self.hnsw_file_path(schema, table, index_name);
            let storage = MmapStorage::open(&path)?;
            let storage_arc = Arc::new(RwLock::new(storage));
            if let Some((_, lock)) = self.open_files.insert(key.clone(), storage_arc) {
                lock.write().sync()?;
            }
        }

        Ok(self.open_files.get_mut(&key).unwrap().clone())
    }

    fn hnsw_file_path(&self, schema: &str, table: &str, index_name: &str) -> PathBuf {
        self.base_path
            .join(schema)
            .join(format!("{}_{}.{}", table, index_name, HNSW_FILE_EXTENSION))
    }

    fn list_indexes(&self, schema: &str, table: &str) -> Result<Vec<String>> {
        let schema_path = self.base_path.join(schema);
        let prefix = format!("{}_{}", table, "");
        let suffix = format!(".{}", INDEX_FILE_EXTENSION);

        let mut indexes = Vec::new();

        if let Ok(entries) = fs::read_dir(&schema_path) {
            for entry in entries.flatten() {
                if let Some(name) = entry.file_name().to_str() {
                    if name.starts_with(&prefix) && name.ends_with(&suffix) {
                        let index_name = &name[prefix.len()..name.len() - suffix.len()];
                        indexes.push(index_name.to_string());
                    }
                }
            }
        }

        Ok(indexes)
    }

    fn table_file_path(&self, schema: &str, table: &str) -> PathBuf {
        self.base_path
            .join(schema)
            .join(format!("{}.{}", table, TABLE_FILE_EXTENSION))
    }

    fn index_file_path(&self, schema: &str, table: &str, index_name: &str) -> PathBuf {
        self.base_path
            .join(schema)
            .join(format!("{}_{}.{}", table, index_name, INDEX_FILE_EXTENSION))
    }

    fn validate_name(name: &str) -> Result<()> {
        ensure!(!name.is_empty(), "name cannot be empty");
        ensure!(
            !name.contains('/') && !name.contains('\\'),
            "name cannot contain path separators"
        );
        ensure!(
            !name.contains(".."),
            "name cannot contain parent directory references"
        );
        ensure!(
            name.chars()
                .all(|c| c.is_alphanumeric() || c == '_' || c == '-'),
            "name can only contain alphanumeric characters, underscores, and hyphens"
        );
        Ok(())
    }
}
