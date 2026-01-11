//! # Database Lifecycle Operations
//!
//! This module implements lifecycle management for TurDB databases, including
//! checkpoint and close operations.
//!
//! ## Checkpoint
//!
//! Checkpointing is the process of flushing dirty pages from the WAL (Write-Ahead Log)
//! to the main database files. This is important for:
//! - Reducing WAL size and recovery time
//! - Persisting changes to durable storage
//! - Freeing up memory used by dirty page tracking
//!
//! ### Checkpoint Types
//!
//! - `checkpoint_wal` - Simple WAL flush via SharedDatabase
//! - `checkpoint_wal_with_stats` - Checkpoint with memory/WAL stats persistence
//! - `checkpoint` - Full checkpoint with detailed CheckpointInfo
//!
//! ## Close
//!
//! Closing the database performs a final checkpoint and marks the database
//! as closed to prevent further operations. The Drop implementation ensures
//! cleanup happens even if close() isn't explicitly called.
//!
//! ## Usage
//!
//! ```ignore
//! let db = Database::open("./mydb")?;
//! // ... use database ...
//! db.checkpoint()?;  // Optional: force checkpoint
//! db.close()?;       // Recommended: explicit close
//! ```

use crate::database::{CheckpointInfo, Database};
use crate::storage::WalStoragePerTable;
use eyre::{bail, Result, WrapErr};
use std::path::Path;

impl Database {
    pub fn checkpoint_wal(&self) -> Result<u32> {
        self.shared.checkpoint()
    }

    pub fn checkpoint_wal_with_stats(&self) -> Result<u32> {
        let frames = self.shared.checkpoint()?;
        if frames > 0 {
            let _ = self.persist_memory_stats();
            let _ = self.persist_wal_stats();
        }
        Ok(frames)
    }

    pub fn checkpoint(&self) -> Result<CheckpointInfo> {
        use std::sync::atomic::Ordering;

        if self.shared.closed.load(Ordering::Acquire) {
            bail!("database is closed");
        }

        let mut wal_guard = self.shared.wal.lock();
        let wal = match wal_guard.as_mut() {
            Some(w) => w,
            None => {
                self.shared.dirty_tracker.clear_all();
                return Ok(CheckpointInfo {
                    frames_checkpointed: 0,
                    wal_truncated: false,
                });
            }
        };

        if self.shared.dirty_tracker.is_empty() {
            wal.cleanup_old_segments()?;
            return Ok(CheckpointInfo {
                frames_checkpointed: 0,
                wal_truncated: false,
            });
        }
        let table_ids = self.shared.dirty_tracker.all_dirty_table_ids();

        self.ensure_file_manager()?;

        let mut file_manager_guard = self.shared.file_manager.write();
        let file_manager = match file_manager_guard.as_mut() {
            Some(fm) => fm,
            None => {
                self.shared.dirty_tracker.clear_all();
                return Ok(CheckpointInfo {
                    frames_checkpointed: 0,
                    wal_truncated: false,
                });
            }
        };

        let table_infos: Vec<(u32, String, String)> = {
            let lookup = self.shared.table_id_lookup.read();
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
                let frames = WalStoragePerTable::flush_wal_for_table(
                    &self.shared.dirty_tracker,
                    &storage,
                    wal,
                    *table_id,
                )
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
        if self.is_closed() {
            bail!("database already closed");
        }

        self.abort_active_transaction();

        let _ = self.checkpoint();

        self.shared
            .closed
            .store(true, std::sync::atomic::Ordering::Release);

        Ok(CheckpointInfo {
            frames_checkpointed: 0,
            wal_truncated: false,
        })
    }

    pub fn is_closed(&self) -> bool {
        use std::sync::atomic::Ordering;
        self.shared.closed.load(Ordering::Acquire)
    }

    pub fn path(&self) -> &Path {
        &self.shared.path
    }
}
