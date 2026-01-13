//! # Transaction Management Module
//!
//! This module implements transaction support for TurDB, providing ACID-compliant
//! transaction control with savepoint support and rollback capabilities.
//!
//! ## Purpose
//!
//! Transactions ensure data integrity by grouping multiple operations into atomic
//! units that either fully complete or fully roll back. This module provides:
//!
//! 1. Transaction lifecycle management (BEGIN, COMMIT, ROLLBACK)
//! 2. Savepoint support for partial rollback within transactions
//! 3. Write entry tracking for undo operations
//! 4. Integration with the MVCC transaction manager
//!
//! ## Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────────────┐
//! │                     Transaction Flow                                    │
//! ├─────────────────────────────────────────────────────────────────────────┤
//! │                                                                         │
//! │   User              Database                ActiveTransaction           │
//! │   ────              ────────                ─────────────────           │
//! │                                                                         │
//! │   BEGIN ──────────► execute_begin() ──────► new()                       │
//! │                                             │                           │
//! │   INSERT/UPDATE ───► (DML ops) ───────────► add_write_entry()           │
//! │                                             │                           │
//! │   SAVEPOINT ──────► execute_savepoint() ──► create_savepoint()          │
//! │                                             │                           │
//! │   ROLLBACK TO ────► execute_rollback() ───► rollback_to_savepoint()     │
//! │       ▼                                     │                           │
//! │   undo_write_entries() ◄──────────────────┘                             │
//! │                                                                         │
//! │   COMMIT ─────────► execute_commit() ──────► finalize_commit()          │
//! │       │                                                                 │
//! │       └───────────► WAL flush (if enabled)                             │
//! │                                                                         │
//! └─────────────────────────────────────────────────────────────────────────┘
//! ```
//!
//! ## Key Data Structures
//!
//! - **Savepoint**: Named checkpoint within a transaction for partial rollback
//! - **ActiveTransaction**: Full transaction state including write log and undo data
//! - **WriteEntry**: Record of a single write operation for potential undo
//!
//! ## Transaction States
//!
//! ```text
//! ┌────────┐   BEGIN    ┌────────┐   COMMIT   ┌───────────┐
//! │  None  │ ─────────► │ Active │ ─────────► │ Committed │
//! └────────┘            └────────┘            └───────────┘
//!                            │
//!                            │ ROLLBACK
//!                            ▼
//!                       ┌────────────┐
//!                       │ Rolled Back│
//!                       └────────────┘
//! ```
//!
//! ## Savepoint Semantics
//!
//! Savepoints create named checkpoints within a transaction:
//! - SAVEPOINT sp1 - Creates checkpoint at current position
//! - ROLLBACK TO sp1 - Undoes all changes after sp1, keeps sp1 active
//! - RELEASE sp1 - Removes savepoint (changes become permanent in transaction)
//!
//! ## Undo Mechanism
//!
//! Each write operation stores:
//! 1. The operation type (insert/update/delete)
//! 2. The affected key
//! 3. Optional undo data (old value for updates, None for inserts)
//!
//! On rollback, operations are undone in reverse order:
//! - INSERT: Delete the inserted key
//! - UPDATE: Restore the old value
//! - DELETE: Reinsert with undo data
//!
//! ## WAL Integration
//!
//! Transaction commit coordinates with Write-Ahead Logging:
//! 1. All dirty pages are identified via dirty_tracker
//! 2. Changes are flushed to WAL before commit returns
//! 3. This ensures durability even on crash after commit
//!
//! ## Group Commit
//!
//! When group_commit_queue is enabled, multiple concurrent transactions batch
//! their WAL flushes together for better throughput:
//! 1. Transaction submits dirty table IDs to the queue and waits
//! 2. One transaction becomes the flush leader and performs the batched flush
//! 3. All waiting transactions are notified on completion or failure
//!
//! ## Page-Level Locking
//!
//! During WAL flush, fine-grained page locks are acquired to enable concurrent
//! writes to different pages:
//! 1. Acquire table intent-exclusive lock for each dirty table
//! 2. Acquire page-write locks for all dirty pages (sorted order for deadlock prevention)
//! 3. Perform the WAL flush while holding locks
//! 4. Release locks (RAII via guard drop)
//!
//! ## Thread Safety
//!
//! - `active_txn` is protected by a Mutex (one transaction per connection)
//! - Write entries are owned by the transaction (no sharing)
//! - WAL flush acquires file_manager and WAL locks atomically
//! - Page locks use 256-shard design to minimize contention

use crate::mvcc::{TxnId, TxnState, UndoRegistry, WriteEntry};
use crate::schema::table::{Constraint, IndexType};
use crate::sql::ast::IsolationLevel;
use crate::storage::IndexFileHeader;
use eyre::{bail, Result, WrapErr};
use smallvec::SmallVec;

use super::group_commit::CommitPayload;
use super::{Database, ExecuteResult};

const COMMIT_BATCH_SIZE: usize = 12;

/// Named checkpoint within a transaction for partial rollback.
#[derive(Debug, Clone)]
pub struct Savepoint {
    pub name: String,
    pub write_entry_idx: usize,
}

/// Active transaction state with write log and undo information.
///
/// MVCC is always enabled - all transactions use versioned records with
/// RecordHeader for visibility control and snapshot isolation.
pub struct ActiveTransaction {
    pub txn_id: TxnId,
    pub read_ts: TxnId,
    pub slot_idx: usize,
    pub state: TxnState,
    pub isolation_level: Option<IsolationLevel>,
    pub read_only: bool,
    pub savepoints: SmallVec<[Savepoint; 4]>,
    pub write_entries: SmallVec<[WriteEntry; 16]>,
    pub undo_data: SmallVec<[Option<Vec<u8>>; 16]>,
    pub undo_registry: UndoRegistry,
}

impl std::fmt::Debug for ActiveTransaction {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ActiveTransaction")
            .field("txn_id", &self.txn_id)
            .field("read_ts", &self.read_ts)
            .field("slot_idx", &self.slot_idx)
            .field("state", &self.state)
            .field("isolation_level", &self.isolation_level)
            .field("read_only", &self.read_only)
            .field("savepoints", &self.savepoints)
            .field("write_entries_count", &self.write_entries.len())
            .finish()
    }
}

impl ActiveTransaction {
    /// Creates a new transaction with MVCC enabled.
    ///
    /// All transactions use MVCC for versioning and snapshot isolation.
    /// The `read_ts` is set to `txn_id` for snapshot isolation semantics.
    pub fn new(
        txn_id: TxnId,
        slot_idx: usize,
        isolation_level: Option<IsolationLevel>,
        read_only: bool,
    ) -> Self {
        Self {
            txn_id,
            read_ts: txn_id,
            slot_idx,
            state: TxnState::Active,
            isolation_level,
            read_only,
            savepoints: SmallVec::new(),
            write_entries: SmallVec::new(),
            undo_data: SmallVec::new(),
            undo_registry: UndoRegistry::new(),
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

    pub fn add_write_entries_batch(&mut self, entries: impl IntoIterator<Item = WriteEntry>) {
        for entry in entries {
            self.write_entries.push(entry);
            self.undo_data.push(None);
        }
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

impl Database {
    pub(crate) fn execute_begin(
        &self,
        begin: &crate::sql::ast::BeginStmt,
    ) -> Result<ExecuteResult> {
        let mut active_txn = self.active_txn.lock();
        if active_txn.is_some() {
            bail!("transaction already in progress, use SAVEPOINT for nested transactions");
        }

        let mvcc_txn = self
            .shared
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

    pub(crate) fn execute_commit(&self) -> Result<ExecuteResult> {
        let wal_enabled = self
            .shared
            .wal_enabled
            .load(std::sync::atomic::Ordering::Acquire);

        let mut active_txn = self.active_txn.lock();
        let txn = active_txn
            .take()
            .ok_or_else(|| eyre::eyre!("no transaction in progress"))?;

        self.finalize_transaction_commit(txn)?;

        if wal_enabled {
            let dirty_table_ids = self.shared.dirty_tracker.all_dirty_table_ids();

            if !dirty_table_ids.is_empty() {
                let total_dirty_pages: u64 = dirty_table_ids
                    .iter()
                    .map(|&tid| self.shared.dirty_tracker.dirty_count(tid))
                    .sum();

                if total_dirty_pages as usize > COMMIT_BATCH_SIZE {
                    self.execute_chunked_wal_commit(&dirty_table_ids)?;
                } else {
                    self.execute_small_commit(&dirty_table_ids)?;
                }
            }

            self.maybe_auto_checkpoint();
        }

        Ok(ExecuteResult::Commit)
    }

    fn execute_small_commit(&self, dirty_table_ids: &[u32]) -> Result<()> {
        let mut payload: CommitPayload = SmallVec::new();

        let mut file_manager_guard = self.shared.file_manager.write();
        let file_manager = file_manager_guard
            .as_mut()
            .ok_or_else(|| eyre::eyre!("file manager not available for commit"))?;

        let mut table_infos: Vec<(u32, String, String)> = {
            let lookup = self.shared.table_id_lookup.read();
            dirty_table_ids
                .iter()
                .filter_map(|&table_id| {
                    lookup
                        .get(&table_id)
                        .map(|(s, t)| (table_id, s.clone(), t.clone()))
                })
                .collect()
        };
        table_infos.sort_by_key(|(id, _, _)| *id);

        for (table_id, schema_name, table_name) in table_infos {
            let _table_lock = self.shared.page_locks.table_intent_exclusive(table_id);
            let dirty_pages = self.shared.dirty_tracker.dirty_pages_for_table(table_id);

            if dirty_pages.is_empty() {
                continue;
            }

            let page_tuples: Vec<(u32, u32)> =
                dirty_pages.iter().map(|&p| (table_id, p)).collect();
            let _page_locks = self.shared.page_locks.page_write_multi(&page_tuples);

            if let Ok(storage_arc) = file_manager.table_data(&schema_name, &table_name) {
                let storage = storage_arc.read();
                let db_size = storage.page_count();
                let pages_to_flush = self.shared.dirty_tracker.drain_for_table(table_id);

                for page_no in pages_to_flush {
                    if let Ok(data) = storage.page(page_no) {
                        let mut buffer = self.shared.page_buffer_pool.acquire_blocking();
                        buffer.copy_from_page(data);
                        payload.push((table_id, page_no, buffer, db_size));
                    }
                }
            }
        }

        drop(file_manager_guard);

        if self.shared.group_commit_queue.is_enabled() {
            match self.shared.group_commit_queue.submit_and_wait(payload) {
                Ok(_batch_id) => {
                    if let Some(pending_commits) = self.shared.group_commit_queue.take_pending() {
                        let result = self.execute_group_wal_flush(&pending_commits);
                        match &result {
                            Ok(()) => self
                                .shared
                                .group_commit_queue
                                .complete_batch(&pending_commits),
                            Err(e) => self
                                .shared
                                .group_commit_queue
                                .fail_batch(&pending_commits, &e.to_string()),
                        }
                        result?;
                    }
                }
                Err(e) => {
                    bail!("group commit failed: {}", e);
                }
            }
        } else if !payload.is_empty() {
            let mut wal_guard = self.shared.wal.lock();
            let wal = wal_guard
                .as_mut()
                .ok_or_else(|| eyre::eyre!("WAL not initialized but WAL mode is enabled"))?;

            for (table_id, page_no, buffer, db_size) in payload {
                wal.write_frame_with_file_id(page_no, db_size, buffer.as_slice(), table_id as u64)
                    .wrap_err("failed to write WAL frame in direct commit")?;
            }
        }

        Ok(())
    }

    fn execute_chunked_wal_commit(&self, dirty_table_ids: &[u32]) -> Result<()> {
        let mut wal_guard = self.shared.wal.lock();
        let wal = wal_guard
            .as_mut()
            .ok_or_else(|| eyre::eyre!("WAL not initialized but WAL mode is enabled"))?;

        let mut file_manager_guard = self.shared.file_manager.write();
        let file_manager = file_manager_guard
            .as_mut()
            .ok_or_else(|| eyre::eyre!("file manager not available for commit"))?;

        let mut table_infos: Vec<(u32, String, String)> = {
            let lookup = self.shared.table_id_lookup.read();
            dirty_table_ids
                .iter()
                .filter_map(|&table_id| {
                    lookup
                        .get(&table_id)
                        .map(|(s, t)| (table_id, s.clone(), t.clone()))
                })
                .collect()
        };
        table_infos.sort_by_key(|(id, _, _)| *id);

        for (table_id, schema_name, table_name) in table_infos {
            let pages_to_flush = self.shared.dirty_tracker.drain_for_table(table_id);

            if pages_to_flush.is_empty() {
                continue;
            }

            let storage_arc = file_manager
                .table_data(&schema_name, &table_name)
                .wrap_err_with(|| {
                    format!(
                        "failed to get storage for table '{}.{}'",
                        schema_name, table_name
                    )
                })?;

            for chunk in pages_to_flush.chunks(COMMIT_BATCH_SIZE) {
                let _table_lock = self.shared.page_locks.table_intent_exclusive(table_id);

                let page_tuples: Vec<(u32, u32)> =
                    chunk.iter().map(|&p| (table_id, p)).collect();
                let _page_locks = self.shared.page_locks.page_write_multi(&page_tuples);

                let storage = storage_arc.read();
                let db_size = storage.page_count();

                let mut chunk_payload: CommitPayload = SmallVec::new();

                for &page_no in chunk {
                    if let Ok(data) = storage.page(page_no) {
                        let mut buffer = self.shared.page_buffer_pool.acquire_blocking();
                        buffer.copy_from_page(data);
                        chunk_payload.push((table_id, page_no, buffer, db_size));
                    }
                }

                drop(storage);

                for (tid, page_no, buffer, db_sz) in chunk_payload {
                    wal.write_frame_with_file_id(page_no, db_sz, buffer.as_slice(), tid as u64)
                        .wrap_err("failed to write WAL frame in chunked commit")?;
                }
            }
        }

        Ok(())
    }

    fn execute_group_wal_flush(
        &self,
        pending_commits: &[std::sync::Arc<super::group_commit::PendingCommit>],
    ) -> Result<()> {
        let mut wal_guard = self.shared.wal.lock();
        let wal = wal_guard
            .as_mut()
            .ok_or_else(|| eyre::eyre!("WAL not initialized but WAL mode is enabled"))?;

        for commit in pending_commits {
            for (table_id, page_no, buffer, db_size) in &commit.payload {
                wal.write_frame_with_file_id(*page_no, *db_size, buffer.as_slice(), *table_id as u64)
                    .wrap_err("failed to write WAL frame in group commit")?;
            }
        }

        Ok(())
    }

    fn finalize_transaction_commit(&self, mut txn: ActiveTransaction) -> Result<()> {
        let commit_ts = self.shared.txn_manager.commit_txn(txn.slot_idx);
        let (write_entries, _undo_data) = txn.take_write_entries();

        let mut value_buffer = Vec::with_capacity(256);
        for entry in write_entries.iter() {
            self.finalize_write_entry_commit(entry, commit_ts, &mut value_buffer)?;
        }

        Ok(())
    }

    fn finalize_write_entry_commit(
        &self,
        entry: &WriteEntry,
        commit_ts: TxnId,
        value_buffer: &mut Vec<u8>,
    ) -> Result<()> {
        use crate::btree::BTree;
        use crate::mvcc::RecordHeader;
        use crate::storage::DEFAULT_SCHEMA;

        self.ensure_file_manager()?;

        let mut file_manager_guard = self.shared.file_manager.write();
        let file_manager = file_manager_guard.as_mut().unwrap();

        let catalog_guard = self.shared.catalog.read();
        let catalog = catalog_guard.as_ref().unwrap();

        let table_id = entry.table_id;
        let table_def = catalog.table_by_id(table_id as u64);

        if table_def.is_none() {
            return Ok(());
        }

        let table_def = table_def.unwrap();
        let schema_name = DEFAULT_SCHEMA;
        let table_name = table_def.name();

        let table_storage_arc = file_manager.table_data_mut(schema_name, table_name)?;
        let mut table_storage = table_storage_arc.write();

        let btree = BTree::new(&mut *table_storage, 1)?;
        if let Some(raw_value) = btree.get(&entry.key)? {
            if raw_value.len() >= RecordHeader::SIZE {
                let mut header = RecordHeader::from_bytes(raw_value);
                header.set_locked(false);
                header.txn_id = commit_ts;

                value_buffer.clear();
                value_buffer.extend_from_slice(raw_value);
                header.write_to(&mut value_buffer[..RecordHeader::SIZE]);

                let mut btree_mut = BTree::new(&mut *table_storage, 1)?;
                btree_mut.update(&entry.key, value_buffer)?;
            }
        }

        Ok(())
    }

    fn maybe_auto_checkpoint(&self) {
        let needs_checkpoint = {
            let wal_guard = self.shared.wal.lock();
            wal_guard
                .as_ref()
                .map(|wal| wal.needs_checkpoint())
                .unwrap_or(false)
        };

        if needs_checkpoint {
            let _ = self.shared.checkpoint();
        }
    }

    pub(crate) fn execute_rollback(
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
        use crate::btree::BTree;
        use crate::database::dml::mvcc_helpers::get_user_data;
        use crate::records::RecordView;
        use crate::storage::TableFileHeader;
        use crate::types::{create_record_schema, OwnedValue};

        self.ensure_file_manager()?;

        let mut file_manager_guard = self.shared.file_manager.write();
        let file_manager = file_manager_guard.as_mut().unwrap();

        let catalog_guard = self.shared.catalog.read();
        let catalog = catalog_guard.as_ref().unwrap();

        let table_id = entry.table_id;
        let result = catalog.table_with_schema_by_id(table_id as u64);

        if result.is_none() {
            return Ok(());
        }

        let (schema_name, table_def) = result.unwrap();
        let schema_name = schema_name.to_string();
        let table_name = table_def.name().to_string();
        let columns = table_def.columns().to_vec();

        let secondary_indexes: Vec<(String, Vec<usize>)> = table_def
            .indexes()
            .iter()
            .filter(|idx| idx.index_type() == IndexType::BTree)
            .map(|idx| {
                let col_indices: Vec<usize> = idx
                    .columns()
                    .filter_map(|col_name| columns.iter().position(|c| c.name() == col_name))
                    .collect();
                (idx.name().to_string(), col_indices)
            })
            .collect();

        let unique_columns: Vec<(usize, String, bool)> = columns
            .iter()
            .enumerate()
            .filter_map(|(idx, col)| {
                let is_pk = col.has_constraint(&Constraint::PrimaryKey);
                let is_unique = col.has_constraint(&Constraint::Unique);
                if is_pk || is_unique {
                    let index_name = if is_pk {
                        format!("{}_pkey", col.name())
                    } else {
                        format!("{}_key", col.name())
                    };
                    Some((idx, index_name, is_pk))
                } else {
                    None
                }
            })
            .collect();

        drop(catalog_guard);

        let schema = create_record_schema(&columns);

        let table_storage_arc = file_manager.table_data_mut(&schema_name, &table_name)?;
        let mut table_storage = table_storage_arc.write();

        let mut btree = BTree::new(&mut *table_storage, 1)?;

        if entry.is_insert {
            let row_values: Option<Vec<OwnedValue>> =
                if let Some(raw_value) = btree.get(&entry.key)? {
                    let user_data = get_user_data(raw_value);
                    if let Ok(record) = RecordView::new(user_data, &schema) {
                        OwnedValue::extract_row_from_record(&record, &columns).ok()
                    } else {
                        None
                    }
                } else {
                    None
                };

            let deleted = btree.delete(&entry.key)?;

            if deleted {
                let page = table_storage.page_mut(0)?;
                let header = TableFileHeader::from_bytes_mut(page)?;
                let new_count = header.row_count().saturating_sub(1);
                header.set_row_count(new_count);
            }
            drop(table_storage);

            if let Some(row_values) = row_values {
                let mut key_buf: SmallVec<[u8; 64]> = SmallVec::new();

                for (col_idx, index_name, _is_pk) in &unique_columns {
                    if file_manager.index_exists(&schema_name, &table_name, index_name) {
                        if let Some(value) = row_values.get(*col_idx) {
                            if !value.is_null() {
                                let index_storage_arc = file_manager.index_data_mut(
                                    &schema_name,
                                    &table_name,
                                    index_name,
                                )?;
                                let mut index_storage = index_storage_arc.write();

                                let index_root_page = {
                                    let page0 = index_storage.page(0)?;
                                    let header = IndexFileHeader::from_bytes(page0)?;
                                    header.root_page()
                                };

                                let mut index_btree =
                                    BTree::new(&mut *index_storage, index_root_page)?;
                                key_buf.clear();
                                Self::encode_value_as_key(value, &mut key_buf);
                                let _ = index_btree.delete(&key_buf);
                            }
                        }
                    }
                }

                for (index_name, col_indices) in &secondary_indexes {
                    if col_indices.is_empty() {
                        continue;
                    }
                    if file_manager.index_exists(&schema_name, &table_name, index_name) {
                        let all_non_null = col_indices
                            .iter()
                            .all(|&idx| row_values.get(idx).is_some_and(|v| !v.is_null()));

                        if all_non_null {
                            let index_storage_arc = file_manager.index_data_mut(
                                &schema_name,
                                &table_name,
                                index_name,
                            )?;
                            let mut index_storage = index_storage_arc.write();

                            let index_root_page = {
                                let page0 = index_storage.page(0)?;
                                let header = IndexFileHeader::from_bytes(page0)?;
                                header.root_page()
                            };

                            let mut index_btree = BTree::new(&mut *index_storage, index_root_page)?;
                            key_buf.clear();
                            for &col_idx in col_indices {
                                if let Some(value) = row_values.get(col_idx) {
                                    Self::encode_value_as_key(value, &mut key_buf);
                                }
                            }
                            let _ = index_btree.delete(&key_buf);
                        }
                    }
                }
            }
        } else if let Some(old_value) = undo_data {
            btree.delete(&entry.key)?;
            btree.insert(&entry.key, old_value)?;
            drop(table_storage);

            let old_row_values: Option<Vec<OwnedValue>> = {
                let user_data = get_user_data(old_value);
                if let Ok(record) = RecordView::new(user_data, &schema) {
                    OwnedValue::extract_row_from_record(&record, &columns).ok()
                } else {
                    None
                }
            };

            if let Some(row_values) = old_row_values {
                let mut key_buf: SmallVec<[u8; 64]> = SmallVec::new();

                for (col_idx, index_name, _is_pk) in &unique_columns {
                    if file_manager.index_exists(&schema_name, &table_name, index_name) {
                        if let Some(value) = row_values.get(*col_idx) {
                            if !value.is_null() {
                                let index_storage_arc = file_manager.index_data_mut(
                                    &schema_name,
                                    &table_name,
                                    index_name,
                                )?;
                                let mut index_storage = index_storage_arc.write();

                                let index_root_page = {
                                    let page0 = index_storage.page(0)?;
                                    let header = IndexFileHeader::from_bytes(page0)?;
                                    header.root_page()
                                };

                                let mut index_btree =
                                    BTree::new(&mut *index_storage, index_root_page)?;
                                key_buf.clear();
                                Self::encode_value_as_key(value, &mut key_buf);

                                let pk_idx = columns
                                    .iter()
                                    .position(|c| c.has_constraint(&Constraint::PrimaryKey));
                                if let Some(pk_idx) = pk_idx {
                                    if let Some(OwnedValue::Int(pk_val)) = row_values.get(pk_idx) {
                                        let row_id_bytes = (*pk_val as u64).to_be_bytes();
                                        let _ = index_btree.insert(&key_buf, &row_id_bytes);
                                    }
                                }
                            }
                        }
                    }
                }

                for (index_name, col_indices) in &secondary_indexes {
                    if col_indices.is_empty() {
                        continue;
                    }
                    if file_manager.index_exists(&schema_name, &table_name, index_name) {
                        let all_non_null = col_indices
                            .iter()
                            .all(|&idx| row_values.get(idx).is_some_and(|v| !v.is_null()));

                        if all_non_null {
                            let index_storage_arc = file_manager.index_data_mut(
                                &schema_name,
                                &table_name,
                                index_name,
                            )?;
                            let mut index_storage = index_storage_arc.write();

                            let index_root_page = {
                                let page0 = index_storage.page(0)?;
                                let header = IndexFileHeader::from_bytes(page0)?;
                                header.root_page()
                            };

                            let mut index_btree = BTree::new(&mut *index_storage, index_root_page)?;
                            key_buf.clear();
                            for &col_idx in col_indices {
                                if let Some(value) = row_values.get(col_idx) {
                                    Self::encode_value_as_key(value, &mut key_buf);
                                }
                            }

                            let pk_idx = columns
                                .iter()
                                .position(|c| c.has_constraint(&Constraint::PrimaryKey));
                            if let Some(pk_idx) = pk_idx {
                                if let Some(OwnedValue::Int(pk_val)) = row_values.get(pk_idx) {
                                    let row_id_bytes = (*pk_val as u64).to_be_bytes();
                                    let _ = index_btree.insert(&key_buf, &row_id_bytes);
                                }
                            }
                        }
                    }
                }
            }
        }

        Ok(())
    }

    pub(crate) fn execute_savepoint(
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

    pub(crate) fn execute_release(
        &self,
        release: &crate::sql::ast::ReleaseStmt<'_>,
    ) -> Result<ExecuteResult> {
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

    pub(crate) fn abort_active_transaction(&self) {
        let txn = {
            let mut active_txn = self.active_txn.lock();
            active_txn.take()
        };

        if let Some(txn) = txn {
            let write_entries: Vec<WriteEntry> = txn.write_entries.iter().cloned().collect();
            let undo_data: Vec<Option<Vec<u8>>> = txn.undo_data.iter().cloned().collect();

            let _ = self.undo_write_entries(&write_entries, &undo_data);
        }
    }
}
