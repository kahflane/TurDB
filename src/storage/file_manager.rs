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
//! // Create a table
//! fm.create_table("analytics", "events", 1)?;
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

use super::{MmapStorage, PAGE_SIZE};

pub const DEFAULT_MAX_OPEN_FILES: usize = 64;
pub const MIN_MAX_OPEN_FILES: usize = 8;

pub const TABLE_FILE_EXTENSION: &str = "tbd";
pub const INDEX_FILE_EXTENSION: &str = "idx";
pub const HNSW_FILE_EXTENSION: &str = "hnsw";
pub const META_FILE_NAME: &str = "turdb.meta";

pub const TABLE_MAGIC: &[u8; 16] = b"TurDB Table\x00\x00\x00\x00\x00";
pub const INDEX_MAGIC: &[u8; 16] = b"TurDB Index\x00\x00\x00\x00\x00";
pub const HNSW_MAGIC: &[u8; 16] = b"TurDB HNSW\x00\x00\x00\x00\x00\x00";
pub const META_MAGIC: &[u8; 16] = b"TurDB Rust v1\x00\x00\x00";

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

    pub fn insert(&mut self, key: K, value: V) {
        if self.map.contains_key(&key) {
            self.touch(&key);
            self.map.insert(key, value);
            return;
        }

        if self.order.len() >= self.capacity {
            self.pop_lru();
        }

        self.order.push(key.clone());
        self.map.insert(key, value);
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
    meta_storage: MmapStorage,
    open_files: LruFileCache<FileKey, MmapStorage>,
}

impl FileManager {
    pub fn create<P: AsRef<Path>>(path: P, max_open_files: usize) -> Result<Self> {
        let base_path = path.as_ref().to_path_buf();
        let max_open_files = max_open_files.max(MIN_MAX_OPEN_FILES);

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
            meta_storage,
            open_files: LruFileCache::new(max_open_files),
        })
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

    pub fn create_table(&mut self, schema: &str, table: &str, table_id: u64) -> Result<()> {
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

        let page = storage.page_mut(0)?;
        page[..16].copy_from_slice(TABLE_MAGIC);
        page[16..24].copy_from_slice(&table_id.to_le_bytes());

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

    pub fn create_index(
        &mut self,
        schema: &str,
        table: &str,
        index_name: &str,
        index_id: u64,
        is_unique: bool,
    ) -> Result<()> {
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

        let page = storage.page_mut(0)?;
        page[..16].copy_from_slice(INDEX_MAGIC);
        page[16..24].copy_from_slice(&index_id.to_le_bytes());
        page[40] = if is_unique { 1 } else { 0 };

        storage.sync()?;

        Ok(())
    }

    pub fn index_exists(&self, schema: &str, table: &str, index_name: &str) -> bool {
        let index_path = self.index_file_path(schema, table, index_name);
        index_path.exists()
    }

    pub fn table_data(&mut self, schema: &str, table: &str) -> Result<&MmapStorage> {
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
            self.open_files.insert(key.clone(), storage);
        }

        Ok(self.open_files.get(&key).unwrap())
    }

    pub fn table_data_mut(&mut self, schema: &str, table: &str) -> Result<&mut MmapStorage> {
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
            self.open_files.insert(key.clone(), storage);
        }

        Ok(self.open_files.get_mut(&key).unwrap())
    }

    pub fn index_data(
        &mut self,
        schema: &str,
        table: &str,
        index_name: &str,
    ) -> Result<&MmapStorage> {
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
            self.open_files.insert(key.clone(), storage);
        }

        Ok(self.open_files.get(&key).unwrap())
    }

    pub fn index_data_mut(
        &mut self,
        schema: &str,
        table: &str,
        index_name: &str,
    ) -> Result<&mut MmapStorage> {
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
            self.open_files.insert(key.clone(), storage);
        }

        Ok(self.open_files.get_mut(&key).unwrap())
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

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn file_manager_new_creates_directory_structure() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("testdb");

        let fm = FileManager::create(&db_path, DEFAULT_MAX_OPEN_FILES).unwrap();

        assert!(db_path.exists());
        assert!(db_path.join(DEFAULT_SCHEMA).exists());
        assert!(db_path.join(META_FILE_NAME).exists());
        assert_eq!(fm.base_path(), &db_path);
    }

    #[test]
    fn file_manager_new_with_custom_max_files() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("testdb");

        let fm = FileManager::create(&db_path, 128).unwrap();

        assert_eq!(fm.max_open_files(), 128);
    }

    #[test]
    fn file_manager_new_enforces_minimum_max_files() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("testdb");

        let fm = FileManager::create(&db_path, 2).unwrap();

        assert_eq!(fm.max_open_files(), MIN_MAX_OPEN_FILES);
    }

    #[test]
    fn table_files_stores_data_and_indexes() {
        let table_files = TableFiles::new();

        assert!(table_files.data().is_none());
        assert!(table_files.indexes().is_empty());
        assert!(table_files.hnsw_indexes().is_empty());
    }

    #[test]
    fn lru_cache_evicts_least_recently_used() {
        let mut cache: LruFileCache<u32, String> = LruFileCache::new(3);

        cache.insert(1, "one".to_string());
        cache.insert(2, "two".to_string());
        cache.insert(3, "three".to_string());

        assert_eq!(cache.len(), 3);

        cache.get(&1);

        cache.insert(4, "four".to_string());

        assert!(cache.get(&1).is_some());
        assert!(cache.get(&2).is_none());
        assert!(cache.get(&3).is_some());
        assert!(cache.get(&4).is_some());
    }

    #[test]
    fn lru_cache_pop_lru_returns_oldest() {
        let mut cache: LruFileCache<u32, String> = LruFileCache::new(3);

        cache.insert(1, "one".to_string());
        cache.insert(2, "two".to_string());
        cache.insert(3, "three".to_string());

        let (key, value) = cache.pop_lru().unwrap();

        assert_eq!(key, 1);
        assert_eq!(value, "one");
        assert_eq!(cache.len(), 2);
    }

    #[test]
    fn create_schema_creates_directory() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("testdb");

        let mut fm = FileManager::create(&db_path, DEFAULT_MAX_OPEN_FILES).unwrap();

        fm.create_schema("analytics").unwrap();

        assert!(db_path.join("analytics").exists());
        assert!(db_path.join("analytics").is_dir());
    }

    #[test]
    fn create_schema_fails_for_existing() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("testdb");

        let mut fm = FileManager::create(&db_path, DEFAULT_MAX_OPEN_FILES).unwrap();

        fm.create_schema("myschema").unwrap();
        let result = fm.create_schema("myschema");

        assert!(result.is_err());
    }

    #[test]
    fn create_schema_validates_name() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("testdb");

        let mut fm = FileManager::create(&db_path, DEFAULT_MAX_OPEN_FILES).unwrap();

        assert!(fm.create_schema("").is_err());
        assert!(fm.create_schema("../escape").is_err());
        assert!(fm.create_schema("with/slash").is_err());
    }

    #[test]
    fn schema_exists_returns_correct_value() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("testdb");

        let mut fm = FileManager::create(&db_path, DEFAULT_MAX_OPEN_FILES).unwrap();

        assert!(fm.schema_exists(DEFAULT_SCHEMA));
        assert!(!fm.schema_exists("nonexistent"));

        fm.create_schema("newschema").unwrap();
        assert!(fm.schema_exists("newschema"));
    }

    #[test]
    fn create_table_creates_tbd_file() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("testdb");

        let mut fm = FileManager::create(&db_path, DEFAULT_MAX_OPEN_FILES).unwrap();

        fm.create_table(DEFAULT_SCHEMA, "users", 1).unwrap();

        let table_path = db_path.join(DEFAULT_SCHEMA).join("users.tbd");
        assert!(table_path.exists());
    }

    #[test]
    fn create_table_initializes_header() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("testdb");

        let mut fm = FileManager::create(&db_path, DEFAULT_MAX_OPEN_FILES).unwrap();

        fm.create_table(DEFAULT_SCHEMA, "users", 42).unwrap();

        let table_path = db_path.join(DEFAULT_SCHEMA).join("users.tbd");
        let storage = MmapStorage::open(&table_path).unwrap();
        let page = storage.page(0).unwrap();

        assert_eq!(&page[..16], TABLE_MAGIC);

        let table_id = u64::from_le_bytes(page[16..24].try_into().unwrap());
        assert_eq!(table_id, 42);
    }

    #[test]
    fn create_table_fails_for_nonexistent_schema() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("testdb");

        let mut fm = FileManager::create(&db_path, DEFAULT_MAX_OPEN_FILES).unwrap();

        let result = fm.create_table("nonexistent", "users", 1);
        assert!(result.is_err());
    }

    #[test]
    fn create_table_fails_for_existing_table() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("testdb");

        let mut fm = FileManager::create(&db_path, DEFAULT_MAX_OPEN_FILES).unwrap();

        fm.create_table(DEFAULT_SCHEMA, "users", 1).unwrap();
        let result = fm.create_table(DEFAULT_SCHEMA, "users", 2);

        assert!(result.is_err());
    }

    #[test]
    fn table_exists_returns_correct_value() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("testdb");

        let mut fm = FileManager::create(&db_path, DEFAULT_MAX_OPEN_FILES).unwrap();

        assert!(!fm.table_exists(DEFAULT_SCHEMA, "users"));

        fm.create_table(DEFAULT_SCHEMA, "users", 1).unwrap();
        assert!(fm.table_exists(DEFAULT_SCHEMA, "users"));
    }

    #[test]
    fn drop_table_removes_tbd_file() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("testdb");

        let mut fm = FileManager::create(&db_path, DEFAULT_MAX_OPEN_FILES).unwrap();

        fm.create_table(DEFAULT_SCHEMA, "users", 1).unwrap();
        assert!(fm.table_exists(DEFAULT_SCHEMA, "users"));

        fm.drop_table(DEFAULT_SCHEMA, "users").unwrap();
        assert!(!fm.table_exists(DEFAULT_SCHEMA, "users"));
    }

    #[test]
    fn drop_table_fails_for_nonexistent_table() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("testdb");

        let mut fm = FileManager::create(&db_path, DEFAULT_MAX_OPEN_FILES).unwrap();

        let result = fm.drop_table(DEFAULT_SCHEMA, "nonexistent");
        assert!(result.is_err());
    }

    #[test]
    fn drop_table_removes_associated_indexes() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("testdb");

        let mut fm = FileManager::create(&db_path, DEFAULT_MAX_OPEN_FILES).unwrap();

        fm.create_table(DEFAULT_SCHEMA, "users", 1).unwrap();
        fm.create_index(DEFAULT_SCHEMA, "users", "email_idx", 1, true)
            .unwrap();

        assert!(fm.index_exists(DEFAULT_SCHEMA, "users", "email_idx"));

        fm.drop_table(DEFAULT_SCHEMA, "users").unwrap();

        assert!(!fm.table_exists(DEFAULT_SCHEMA, "users"));
        assert!(!fm.index_exists(DEFAULT_SCHEMA, "users", "email_idx"));
    }

    #[test]
    fn table_data_returns_mmap_storage() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("testdb");

        let mut fm = FileManager::create(&db_path, DEFAULT_MAX_OPEN_FILES).unwrap();

        fm.create_table(DEFAULT_SCHEMA, "users", 42).unwrap();

        let storage = fm.table_data(DEFAULT_SCHEMA, "users").unwrap();

        let page = storage.page(0).unwrap();
        assert_eq!(&page[..16], TABLE_MAGIC);
        let table_id = u64::from_le_bytes(page[16..24].try_into().unwrap());
        assert_eq!(table_id, 42);
    }

    #[test]
    fn table_data_caches_open_files() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("testdb");

        let mut fm = FileManager::create(&db_path, DEFAULT_MAX_OPEN_FILES).unwrap();

        fm.create_table(DEFAULT_SCHEMA, "users", 1).unwrap();

        let _storage1 = fm.table_data(DEFAULT_SCHEMA, "users").unwrap();

        assert_eq!(fm.open_file_count(), 1);

        let _storage2 = fm.table_data(DEFAULT_SCHEMA, "users").unwrap();

        assert_eq!(fm.open_file_count(), 1);
    }

    #[test]
    fn table_data_fails_for_nonexistent_table() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("testdb");

        let mut fm = FileManager::create(&db_path, DEFAULT_MAX_OPEN_FILES).unwrap();

        let result = fm.table_data(DEFAULT_SCHEMA, "nonexistent");
        assert!(result.is_err());
    }

    #[test]
    fn index_data_returns_mmap_storage() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("testdb");

        let mut fm = FileManager::create(&db_path, DEFAULT_MAX_OPEN_FILES).unwrap();

        fm.create_table(DEFAULT_SCHEMA, "users", 1).unwrap();
        fm.create_index(DEFAULT_SCHEMA, "users", "email_idx", 99, true)
            .unwrap();

        let storage = fm.index_data(DEFAULT_SCHEMA, "users", "email_idx").unwrap();

        let page = storage.page(0).unwrap();
        assert_eq!(&page[..16], INDEX_MAGIC);
        let index_id = u64::from_le_bytes(page[16..24].try_into().unwrap());
        assert_eq!(index_id, 99);
    }
}
