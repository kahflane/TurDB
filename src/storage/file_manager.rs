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

use std::fs;
use std::path::{Path, PathBuf};

use eyre::{Result, WrapErr};

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

#[derive(Debug)]
pub struct FileManager {
    base_path: PathBuf,
    max_open_files: usize,
    meta_storage: MmapStorage,
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
        })
    }

    pub fn base_path(&self) -> &Path {
        &self.base_path
    }

    pub fn max_open_files(&self) -> usize {
        self.max_open_files
    }

    pub fn meta_storage(&self) -> &MmapStorage {
        &self.meta_storage
    }

    pub fn meta_storage_mut(&mut self) -> &mut MmapStorage {
        &mut self.meta_storage
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
}
