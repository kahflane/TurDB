//! # WAL Recovery Module
//!
//! This module implements Write-Ahead Log (WAL) recovery for TurDB. During database
//! startup, the recovery process replays uncommitted changes from the WAL to restore
//! the database to a consistent state after a crash.
//!
//! ## Purpose
//!
//! WAL recovery ensures durability and crash consistency:
//!
//! 1. **Crash Recovery**: Replays changes that were written to WAL but not yet
//!    flushed to the main data files
//! 2. **Consistency**: Restores the database to the last known consistent state
//! 3. **Atomicity**: Ensures partially written transactions are either completed
//!    or rolled back
//!
//! ## Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────────────┐
//! │                    WAL Recovery Flow                                     │
//! ├─────────────────────────────────────────────────────────────────────────┤
//! │                                                                         │
//! │   Database Open                                                         │
//! │       │                                                                 │
//! │       ▼                                                                 │
//! │   ┌─────────────────────┐                                               │
//! │   │ Check WAL exists?   │                                               │
//! │   └─────────┬───────────┘                                               │
//! │             │ yes                                                       │
//! │             ▼                                                           │
//! │   ┌─────────────────────┐     ┌─────────────────────┐                   │
//! │   │ recover_all_tables  │────►│ For each schema dir │                   │
//! │   └─────────────────────┘     └──────────┬──────────┘                   │
//! │                                          │                              │
//! │                                          ▼                              │
//! │                               ┌─────────────────────────┐               │
//! │                               │ recover_schema_tables   │               │
//! │                               └──────────┬──────────────┘               │
//! │                                          │                              │
//! │                                          ▼                              │
//! │                               ┌─────────────────────────┐               │
//! │                               │ For each .tbd file:     │               │
//! │                               │ - Read table_id header  │               │
//! │                               │ - Replay WAL frames     │               │
//! │                               │ - Sync storage          │               │
//! │                               └──────────┬──────────────┘               │
//! │                                          │                              │
//! │                                          ▼                              │
//! │                               ┌─────────────────────────┐               │
//! │                               │ Truncate WAL            │               │
//! │                               └─────────────────────────┘               │
//! │                                                                         │
//! └─────────────────────────────────────────────────────────────────────────┘
//! ```
//!
//! ## Recovery Process
//!
//! 1. **WAL Detection**: Check if WAL segment files exist and have content
//! 2. **Schema Iteration**: Process each schema directory (root, custom schemas)
//! 3. **Table Recovery**: For each table file:
//!    - Read the table header to get table_id
//!    - Replay WAL frames for that table_id
//!    - Sync changes to disk
//! 4. **WAL Cleanup**: Truncate WAL after successful recovery
//!
//! ## Performance Characteristics
//!
//! - Recovery time is proportional to WAL size
//! - Each table file is opened and recovered independently
//! - Changes are synced to disk after each table recovery
//!
//! ## Error Handling
//!
//! Recovery is robust to:
//! - Missing or corrupt table files (skipped)
//! - Invalid headers (skipped)
//! - Empty files (skipped)
//!
//! If recovery fails for a table, the error is propagated and the database
//! will not open, ensuring no data corruption from partial recovery.

use crate::storage::{MmapStorage, TableFileHeader, Wal, FILE_HEADER_SIZE};
use eyre::{Result, WrapErr};
use std::path::Path;

use super::Database;

impl Database {
    pub(crate) fn recover_all_tables(db_path: &Path, wal_dir: &Path) -> Result<u32> {
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

    pub(crate) fn recover_schema_tables(schema_path: &Path, wal: &mut Wal) -> Result<u32> {
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
}
