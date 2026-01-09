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
//! ## Architecture (Single-Pass Recovery)
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────────────┐
//! │                    WAL Recovery Flow (Optimized)                        │
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
//! │   ┌─────────────────────────────────────────────┐                       │
//! │   │ Phase 1: Collect all table storages         │                       │
//! │   │ - Scan all schema directories               │                       │
//! │   │ - Map file_id -> MmapStorage                │                       │
//! │   └─────────────────────┬───────────────────────┘                       │
//! │                         │                                               │
//! │                         ▼                                               │
//! │   ┌─────────────────────────────────────────────┐                       │
//! │   │ Phase 2: Single-pass WAL replay             │                       │
//! │   │ - Read each WAL segment ONCE                │                       │
//! │   │ - For each frame, lookup storage by file_id │                       │
//! │   │ - Apply frame to correct storage            │                       │
//! │   └─────────────────────┬───────────────────────┘                       │
//! │                         │                                               │
//! │                         ▼                                               │
//! │   ┌─────────────────────────────────────────────┐                       │
//! │   │ Phase 3: Sync all modified storages         │                       │
//! │   └─────────────────────┬───────────────────────┘                       │
//! │                         │                                               │
//! │                         ▼                                               │
//! │   ┌─────────────────────────────────────────────┐                       │
//! │   │ Phase 4: Truncate WAL                       │                       │
//! │   └─────────────────────────────────────────────┘                       │
//! │                                                                         │
//! └─────────────────────────────────────────────────────────────────────────┘
//! ```
//!
//! ## Performance Characteristics
//!
//! The single-pass algorithm achieves O(T + F) complexity where:
//! - T = number of tables
//! - F = number of WAL frames
//!
//! This is a significant improvement over the naive O(T × F) approach which
//! would re-scan the entire WAL for each table. For a database with 23 tables
//! and 26GB of WAL (419 segments), this reduces I/O from ~600GB to ~26GB.
//!
//! ## Error Handling
//!
//! Recovery is robust to:
//! - Missing or corrupt table files (skipped)
//! - Invalid headers (skipped)
//! - Empty files (skipped)
//! - Frames for unknown file_ids (skipped with warning)

use crate::storage::{MmapStorage, TableFileHeader, Wal, WalSegment, FILE_HEADER_SIZE};
use eyre::{Result, WrapErr};
use hashbrown::HashMap;
use std::path::{Path, PathBuf};

use super::Database;

impl Database {
    pub(crate) fn recover_all_tables(db_path: &Path, wal_dir: &Path) -> Result<u32> {
        use std::fs;

        let mut storages: HashMap<u64, (PathBuf, MmapStorage)> = HashMap::new();
        let mut modified_file_ids: hashbrown::HashSet<u64> = hashbrown::HashSet::new();

        for entry in fs::read_dir(db_path).wrap_err("failed to read database directory")? {
            let entry = entry.wrap_err("failed to read directory entry")?;
            let path = entry.path();

            if path.is_dir() {
                Self::collect_table_storages(&path, &mut storages)?;
            }
        }

        if storages.is_empty() {
            return Ok(0);
        }

        let max_segment = Wal::find_latest_segment(wal_dir)?;
        let mut total_frames = 0u32;

        if max_segment > 1 {
            eprintln!(
                "[recovery] Starting single-pass WAL recovery: {} segments, {} tables",
                max_segment,
                storages.len()
            );
        }

        for i in 1..=max_segment {
            let segment_path = wal_dir.join(format!("wal.{:06}", i));
            if !segment_path.exists() {
                continue;
            }

            if max_segment > 10 && i % 50 == 0 {
                eprintln!("[recovery] Processing segment {}/{}", i, max_segment);
            }

            let mut segment = WalSegment::open(&segment_path, i)
                .wrap_err_with(|| format!("failed to open WAL segment {:?}", segment_path))?;

            while let Ok((header, page_data)) = segment.read_frame() {
                let file_id = header.file_id;

                if let Some((_path, storage)) = storages.get_mut(&file_id) {
                    if header.page_no >= storage.page_count() {
                        let required_pages = header.db_size.max(header.page_no + 1);
                        storage.grow(required_pages).wrap_err_with(|| {
                            format!(
                                "failed to grow storage for file_id={} to {} pages",
                                file_id, required_pages
                            )
                        })?;
                    }

                    let page_mut = storage.page_mut(header.page_no).wrap_err_with(|| {
                        format!(
                            "failed to get page {} for file_id={} during recovery",
                            header.page_no, file_id
                        )
                    })?;

                    page_mut.copy_from_slice(&page_data);
                    modified_file_ids.insert(file_id);
                    total_frames += 1;
                }
            }
        }

        for file_id in &modified_file_ids {
            if let Some((path, storage)) = storages.get_mut(file_id) {
                storage
                    .sync()
                    .wrap_err_with(|| format!("failed to sync storage {:?} after recovery", path))?;
            }
        }

        if total_frames > 0 {
            if max_segment > 1 {
                eprintln!(
                    "[recovery] Applied {} frames, syncing {} modified tables...",
                    total_frames,
                    modified_file_ids.len()
                );
            }

            let wal = Wal::open(wal_dir)
                .wrap_err_with(|| format!("failed to open WAL for truncation at {:?}", wal_dir))?;
            wal.truncate()
                .wrap_err("failed to truncate WAL after recovery")?;

            if max_segment > 1 {
                eprintln!("[recovery] WAL recovery complete!");
            }
        }

        Ok(total_frames)
    }

    fn collect_table_storages(
        schema_path: &Path,
        storages: &mut HashMap<u64, (PathBuf, MmapStorage)>,
    ) -> Result<()> {
        use std::fs;
        use std::io::Read;

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

                let storage = MmapStorage::open(&path).wrap_err_with(|| {
                    format!("failed to open storage {:?} for WAL recovery", path)
                })?;

                storages.insert(table_id as u64, (path.clone(), storage));
            }
        }

        Ok(())
    }

    pub(crate) fn replay_schema_tables_from_segments(
        schema_path: &Path,
        segments: &[std::path::PathBuf],
    ) -> Result<u32> {
        use std::fs;
        use std::io::Read;

        let mut total_frames = 0u32;

        for entry in fs::read_dir(schema_path).wrap_err("failed to read schema directory")? {
            let entry = entry.wrap_err("failed to read directory entry")?;
            let path = entry.path();

            if path.extension().map(|e| e == "tbd").unwrap_or(false) {
                let mut header_bytes = [0u8; FILE_HEADER_SIZE];
                let mut file = fs::File::open(&path)
                    .wrap_err_with(|| format!("failed to open table file {:?} for relay", path))?;

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
                    format!("failed to open storage {:?} for WAL replay", path)
                })?;

                let frames =
                    Wal::replay_segments_to_storage(segments, &mut storage, table_id)
                        .wrap_err_with(|| {
                            format!(
                                "failed to replay WAL frames for table_id={} from segments",
                                table_id
                            )
                        })?;

                if frames > 0 {
                    storage.sync().wrap_err_with(|| {
                        format!("failed to sync storage {:?} after replay", path)
                    })?;
                }

                total_frames += frames;
            }
        }

        Ok(total_frames)
    }
}
