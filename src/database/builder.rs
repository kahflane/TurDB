//! # Database Builder
//!
//! This module provides the `DatabaseBuilder` API for configuring and opening TurDB
//! databases with fine-grained control over storage backends, memory budgets, and
//! other settings.
//!
//! ## Builder Pattern
//!
//! `DatabaseBuilder` uses the builder pattern to provide a fluent, type-safe API
//! for database configuration. Settings can be chained together before calling
//! `open()` to create the database.
//!
//! ## Storage Backend Selection
//!
//! The builder supports multiple storage backends through `StorageKind`:
//!
//! - **Mmap (native)**: Memory-mapped files with zero-copy page access. This is
//!   the default and recommended backend for native applications.
//!
//! - **OPFS (WASM)**: Origin Private File System for browser environments.
//!   Requires running in a Web Worker context. WAL is automatically disabled.
//!
//! ## Configuration Options
//!
//! | Option          | Default           | Description                              |
//! |-----------------|-------------------|------------------------------------------|
//! | memory_budget   | 25% of RAM (4MB+) | Total memory budget for database         |
//! | page_cache_size | Auto (from budget)| Number of pages in cache                 |
//! | max_open_files  | 64                | Maximum open file handles in LRU cache   |
//! | wal_enabled     | true (mmap only)  | Whether to use Write-Ahead Log           |
//!
//! ## Usage Examples
//!
//! ### Native (Mmap) with defaults:
//!
//! ```ignore
//! let db = Database::builder()
//!     .path("./mydb")
//!     .open()?;
//! ```
//!
//! ### Native with custom memory budget:
//!
//! ```ignore
//! let db = Database::builder()
//!     .path("./mydb")
//!     .memory_budget(64 * 1024 * 1024)  // 64 MB
//!     .max_open_files(128)
//!     .open()?;
//! ```
//!
//! ### WASM (OPFS in Worker):
//!
//! ```ignore
//! let db = Database::builder()
//!     .opfs("my-database")
//!     .open()?;  // WAL auto-disabled
//! ```
//!
//! ## Platform Considerations
//!
//! ### Native Platforms
//!
//! On Linux, macOS, and Windows, the builder defaults to mmap storage with
//! WAL enabled for durability. The memory budget is auto-detected as 25% of
//! system RAM (minimum 4MB).
//!
//! ### WASM (Browser)
//!
//! On WASM targets, the builder requires explicit OPFS configuration via
//! `opfs()`. This storage backend:
//!
//! - Requires a dedicated Web Worker context (not main thread)
//! - Uses synchronous FileSystemSyncAccessHandle API
//! - Disables WAL (OPFS provides its own durability)
//! - Does not support zero-copy page access
//!
//! ## Thread Safety
//!
//! `DatabaseBuilder` is not `Send` or `Sync` and should be used from a single
//! thread. Once `open()` is called, the resulting `Database` is thread-safe.

use std::path::Path;

use eyre::Result;

use crate::storage::StorageKind;

/// Builder for configuring and opening a TurDB database.
///
/// Use `Database::builder()` to create a new builder, then chain configuration
/// methods before calling `open()` to create the database.
pub struct DatabaseBuilder {
    storage_kind: Option<StorageKind>,
    memory_budget: Option<usize>,
    max_open_files: Option<usize>,
    wal_enabled: Option<bool>,
}

impl Default for DatabaseBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl DatabaseBuilder {
    /// Creates a new DatabaseBuilder with default settings.
    pub fn new() -> Self {
        Self {
            storage_kind: None,
            memory_budget: None,
            max_open_files: None,
            wal_enabled: None,
        }
    }

    /// Configures the database to use mmap storage at the specified path.
    ///
    /// This is the default storage backend for native platforms. It provides
    /// zero-copy page access through memory-mapped files.
    ///
    /// # Arguments
    ///
    /// * `path` - Path to the database directory. Will be created if it doesn't exist.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let db = Database::builder()
    ///     .path("./mydb")
    ///     .open()?;
    /// ```
    pub fn path<P: AsRef<Path>>(mut self, path: P) -> Self {
        self.storage_kind = Some(StorageKind::mmap(path.as_ref()));
        self
    }

    /// Configures the database to use OPFS storage with the specified name.
    ///
    /// This storage backend is only available on WASM targets and requires
    /// running in a dedicated Web Worker context. WAL is automatically
    /// disabled for OPFS storage.
    ///
    /// # Arguments
    ///
    /// * `name` - Name of the OPFS directory for this database.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let db = Database::builder()
    ///     .opfs("my-database")
    ///     .open()?;
    /// ```
    #[cfg(target_arch = "wasm32")]
    pub fn opfs(mut self, name: &str) -> Self {
        self.storage_kind = Some(StorageKind::opfs(name));
        self.wal_enabled = Some(false);
        self
    }

    /// Sets the storage kind directly.
    ///
    /// This is an advanced method for cases where you need to construct
    /// the `StorageKind` separately.
    pub fn storage_kind(mut self, kind: StorageKind) -> Self {
        #[cfg(target_arch = "wasm32")]
        if matches!(kind, StorageKind::Opfs { .. }) {
            self.wal_enabled = Some(false);
        }
        self.storage_kind = Some(kind);
        self
    }

    /// Sets the total memory budget for the database.
    ///
    /// The memory budget controls how much RAM the database can use for
    /// page caching, query execution, recovery, and schema storage.
    ///
    /// If not specified, defaults to 25% of system RAM with a minimum of 4MB.
    ///
    /// # Arguments
    ///
    /// * `bytes` - Total memory budget in bytes.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let db = Database::builder()
    ///     .path("./mydb")
    ///     .memory_budget(64 * 1024 * 1024)  // 64 MB
    ///     .open()?;
    /// ```
    pub fn memory_budget(mut self, bytes: usize) -> Self {
        self.memory_budget = Some(bytes);
        self
    }

    /// Sets the maximum number of open file handles in the LRU cache.
    ///
    /// TurDB uses a file-per-table architecture. When the limit is reached,
    /// least-recently-used files are synced and closed.
    ///
    /// If not specified, defaults to 64.
    ///
    /// # Arguments
    ///
    /// * `count` - Maximum number of open file handles.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let db = Database::builder()
    ///     .path("./mydb")
    ///     .max_open_files(128)
    ///     .open()?;
    /// ```
    pub fn max_open_files(mut self, count: usize) -> Self {
        self.max_open_files = Some(count);
        self
    }

    /// Enables or disables the Write-Ahead Log (WAL).
    ///
    /// WAL provides durability and atomic commits. It is enabled by default
    /// for mmap storage and disabled for OPFS storage (which has its own
    /// durability guarantees).
    ///
    /// # Arguments
    ///
    /// * `enabled` - Whether to enable WAL.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let db = Database::builder()
    ///     .path("./mydb")
    ///     .wal_enabled(false)  // Disable WAL for testing
    ///     .open()?;
    /// ```
    pub fn wal_enabled(mut self, enabled: bool) -> Self {
        self.wal_enabled = Some(enabled);
        self
    }

    /// Opens or creates the database with the configured settings.
    ///
    /// If the database exists at the specified path, it will be opened.
    /// Otherwise, a new database will be created.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - No storage kind was specified (call `path()` or `opfs()` first)
    /// - The path is invalid or inaccessible
    /// - Database metadata is corrupted
    /// - OPFS is used outside a Web Worker context (WASM only)
    ///
    /// # Example
    ///
    /// ```ignore
    /// let db = Database::builder()
    ///     .path("./mydb")
    ///     .open()?;
    /// ```
    pub fn open(self) -> Result<super::Database> {
        let storage_kind = self.storage_kind.as_ref().ok_or_else(|| {
            eyre::eyre!("storage kind not specified: call .path() or .opfs() first")
        })?;

        match storage_kind {
            StorageKind::Mmap { path } => {
                let meta_path = path.join("turdb.meta");
                if meta_path.exists() {
                    self.open_existing(path)
                } else {
                    self.create_new(path)
                }
            }
            #[cfg(target_arch = "wasm32")]
            StorageKind::Opfs { .. } => {
                eyre::bail!("OPFS storage is not yet implemented; use mmap storage on native platforms")
            }
        }
    }

    fn open_existing(&self, path: &std::path::Path) -> Result<super::Database> {
        super::Database::open_with_config(
            path,
            self.memory_budget,
            self.max_open_files,
            self.wal_enabled,
        )
    }

    fn create_new(&self, path: &std::path::Path) -> Result<super::Database> {
        super::Database::create_with_config(
            path,
            self.memory_budget,
            self.max_open_files,
            self.wal_enabled,
        )
    }

    /// Returns the configured storage kind, if any.
    pub fn get_storage_kind(&self) -> Option<&StorageKind> {
        self.storage_kind.as_ref()
    }

    /// Returns the configured memory budget, if any.
    pub fn get_memory_budget(&self) -> Option<usize> {
        self.memory_budget
    }

    /// Returns the configured max open files, if any.
    pub fn get_max_open_files(&self) -> Option<usize> {
        self.max_open_files
    }

    /// Returns whether WAL is enabled, if configured.
    pub fn get_wal_enabled(&self) -> Option<bool> {
        self.wal_enabled
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;
    use tempfile::tempdir;

    #[test]
    fn builder_path_sets_storage_kind() {
        let builder = DatabaseBuilder::new().path("/tmp/test");

        let kind = builder.get_storage_kind().unwrap();
        assert!(kind.is_mmap());
        assert_eq!(kind.path(), Some(&PathBuf::from("/tmp/test")));
    }

    #[test]
    fn builder_memory_budget_sets_value() {
        let builder = DatabaseBuilder::new()
            .path("/tmp/test")
            .memory_budget(64 * 1024 * 1024);

        assert_eq!(builder.get_memory_budget(), Some(64 * 1024 * 1024));
    }

    #[test]
    fn builder_max_open_files_sets_value() {
        let builder = DatabaseBuilder::new()
            .path("/tmp/test")
            .max_open_files(128);

        assert_eq!(builder.get_max_open_files(), Some(128));
    }

    #[test]
    fn builder_wal_enabled_sets_value() {
        let builder = DatabaseBuilder::new()
            .path("/tmp/test")
            .wal_enabled(false);

        assert_eq!(builder.get_wal_enabled(), Some(false));
    }

    #[test]
    fn builder_open_without_storage_kind_fails() {
        let result = DatabaseBuilder::new().open();

        assert!(result.is_err());
        match result {
            Err(e) => assert!(e.to_string().contains("storage kind not specified")),
            Ok(_) => panic!("expected error"),
        }
    }

    #[test]
    fn builder_open_creates_new_database() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test_db");

        let db = DatabaseBuilder::new()
            .path(&path)
            .open()
            .unwrap();

        assert!(path.join("turdb.meta").exists());
        drop(db);
    }

    #[test]
    fn builder_open_opens_existing_database() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test_db");

        let db1 = DatabaseBuilder::new()
            .path(&path)
            .open()
            .unwrap();
        drop(db1);

        let db2 = DatabaseBuilder::new()
            .path(&path)
            .open()
            .unwrap();

        drop(db2);
    }

    #[test]
    fn builder_chaining_works() {
        let builder = DatabaseBuilder::new()
            .path("/tmp/test")
            .memory_budget(32 * 1024 * 1024)
            .max_open_files(64)
            .wal_enabled(true);

        assert!(builder.get_storage_kind().is_some());
        assert_eq!(builder.get_memory_budget(), Some(32 * 1024 * 1024));
        assert_eq!(builder.get_max_open_files(), Some(64));
        assert_eq!(builder.get_wal_enabled(), Some(true));
    }
}
