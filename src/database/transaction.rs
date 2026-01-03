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
//! ## Thread Safety
//!
//! - `active_txn` is protected by a Mutex (one transaction per connection)
//! - Write entries are owned by the transaction (no sharing)
//! - WAL flush acquires file_manager and WAL locks atomically

use crate::mvcc::{TxnId, TxnState, WriteEntry};
use crate::sql::ast::IsolationLevel;
use crate::storage::{WalStoragePerTable, DEFAULT_SCHEMA};
use eyre::{bail, Result, WrapErr};
use smallvec::SmallVec;

use super::{Database, ExecuteResult};

/// Named checkpoint within a transaction for partial rollback.
#[derive(Debug, Clone)]
pub struct Savepoint {
    pub name: String,
    pub write_entry_idx: usize,
}

/// Active transaction state with write log and undo information.
#[derive(Debug)]
pub struct ActiveTransaction {
    pub txn_id: TxnId,
    pub slot_idx: usize,
    pub state: TxnState,
    pub isolation_level: Option<IsolationLevel>,
    pub read_only: bool,
    pub savepoints: SmallVec<[Savepoint; 4]>,
    pub write_entries: SmallVec<[WriteEntry; 16]>,
    pub undo_data: SmallVec<[Option<Vec<u8>>; 16]>,
}

impl ActiveTransaction {
    pub fn new(
        txn_id: TxnId,
        slot_idx: usize,
        isolation_level: Option<IsolationLevel>,
        read_only: bool,
    ) -> Self {
        Self {
            txn_id,
            slot_idx,
            state: TxnState::Active,
            isolation_level,
            read_only,
            savepoints: SmallVec::new(),
            write_entries: SmallVec::new(),
            undo_data: SmallVec::new(),
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
            .wal_enabled
            .load(std::sync::atomic::Ordering::Acquire);

        {
            let active_txn = self.active_txn.lock();
            active_txn
                .as_ref()
                .ok_or_else(|| eyre::eyre!("no transaction in progress"))?;
        }

        let mut active_txn = self.active_txn.lock();
        let txn = active_txn
            .take()
            .ok_or_else(|| eyre::eyre!("no transaction in progress"))?;

        self.finalize_transaction_commit(txn)?;

        if wal_enabled {
            let dirty_table_ids = self.dirty_tracker.all_dirty_table_ids();
            if dirty_table_ids.is_empty() {
                return Ok(ExecuteResult::Commit);
            }

            let table_infos: Vec<(u32, String, String)> = {
                let lookup = self.table_id_lookup.read();
                dirty_table_ids
                    .iter()
                    .filter_map(|&table_id| {
                        lookup
                            .get(&table_id)
                            .map(|(s, t)| (table_id, s.clone(), t.clone()))
                    })
                    .collect()
            };

            let mut file_manager_guard = self.file_manager.write();
            let file_manager = file_manager_guard
                .as_mut()
                .ok_or_else(|| eyre::eyre!("file manager not available for WAL flush"))?;

            let mut wal_guard = self.wal.lock();
            let wal = wal_guard
                .as_mut()
                .ok_or_else(|| eyre::eyre!("WAL not initialized but WAL mode is enabled"))?;

            for (table_id, schema_name, table_name) in table_infos {
                let storage = file_manager
                    .table_data(&schema_name, &table_name)
                    .wrap_err_with(|| {
                        format!(
                            "failed to get storage for table {}.{} during WAL flush",
                            schema_name, table_name
                        )
                    })?;

                WalStoragePerTable::flush_wal_for_table(&self.dirty_tracker, storage, wal, table_id)
                    .wrap_err_with(|| {
                        format!(
                            "failed to flush WAL for table {}.{} on commit",
                            schema_name, table_name
                        )
                    })?;
            }
        }

        Ok(ExecuteResult::Commit)
    }

    fn finalize_transaction_commit(&self, mut txn: ActiveTransaction) -> Result<()> {
        let (write_entries, _undo_data) = txn.take_write_entries();

        for entry in write_entries.iter() {
            self.finalize_write_entry_commit(entry)?;
        }

        Ok(())
    }

    fn finalize_write_entry_commit(&self, _entry: &WriteEntry) -> Result<()> {
        Ok(())
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
        self.ensure_file_manager()?;

        let mut file_manager_guard = self.file_manager.write();
        let file_manager = file_manager_guard.as_mut().unwrap();

        let catalog_guard = self.catalog.read();
        let catalog = catalog_guard.as_ref().unwrap();

        let table_id = entry.table_id;
        let table_def = catalog.table_by_id(table_id as u64);

        if table_def.is_none() {
            return Ok(());
        }

        let table_def = table_def.unwrap();
        let schema_name = DEFAULT_SCHEMA;
        let table_name = table_def.name();

        let table_storage = file_manager.table_data_mut(schema_name, table_name)?;

        use crate::btree::BTree;
        let mut btree = BTree::new(table_storage, 1)?;

        if entry.is_insert {
            btree.delete(&entry.key)?;
        } else if let Some(old_value) = undo_data {
            btree.delete(&entry.key)?;
            btree.insert(&entry.key, old_value)?;
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
}
