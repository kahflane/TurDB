//! # Transaction Management
//!
//! This module provides the core transaction primitives for TurDB's MVCC
//! implementation. The design follows a Single-Writer / Multi-Reader model
//! with row-level locking stored directly in record headers.
//!
//! ## Transaction Identifiers
//!
//! Transaction IDs (`TxnId`) are 64-bit monotonically increasing integers
//! allocated from a global atomic counter. This provides:
//! - Unique identification for each transaction
//! - Natural ordering for visibility checks
//! - Effectively unlimited transaction space (centuries at 100K/sec)
//!
//! Special values:
//! - `TxnId = 0`: Reserved for "always visible" bootstrapped data
//! - `TxnId = u64::MAX`: Sentinel value (never used for real transactions)
//!
//! ## Transaction States
//!
//! ```text
//! ┌─────────┐     commit()     ┌───────────┐
//! │ Active  │ ───────────────> │ Committed │
//! └─────────┘                  └───────────┘
//!      │
//!      │ rollback()
//!      v
//! ┌─────────┐
//! │ Aborted │
//! └─────────┘
//! ```
//!
//! ## Memory Layout
//!
//! The TransactionManager uses a fixed-size slot array to track active
//! transactions, avoiding any dynamic allocation:
//!
//! ```text
//! TransactionManager {
//!     global_ts: AtomicU64,           // 8 bytes
//!     active_slots: [AtomicU64; 64],  // 512 bytes
//!     slot_lock: Mutex<()>,           // ~40 bytes (parking_lot)
//! }
//! Total: ~560 bytes
//! ```
//!
//! ## Concurrency Model
//!
//! - `global_ts`: Lock-free increment via `fetch_add`
//! - `active_slots`: Lock-free reads, mutex-protected slot allocation
//! - Watermark calculation: Lock-free iteration over slots
//!
//! ## Slot Array Design
//!
//! Each slot holds the start timestamp of an active transaction:
//! - Value 0: Slot is empty (available)
//! - Value > 0: Start timestamp of active transaction
//!
//! Maximum concurrent transactions: 64 (hard limit for 1MB budget)
//! Memory overhead: 64 * 8 bytes = 512 bytes
//!
//! ## Watermark Calculation
//!
//! The global watermark is the minimum of:
//! - Current global timestamp
//! - All non-zero values in active_slots
//!
//! Any version with txn_id < watermark is safe for garbage collection.
//!
//! ## Write Set Tracking
//!
//! Transactions track modified keys using `SmallVec<[(TableId, Key); 16]>`:
//! - 90% of embedded transactions touch <16 rows
//! - Stack-allocated for small transactions
//! - Spills to heap only for large transactions
//!
//! ## Safety Invariants
//!
//! 1. A transaction ID is never reused
//! 2. Slots are released on commit/rollback (enforced via Drop)
//! 3. Watermark is always <= global_ts
//! 4. Only one transaction can hold a given slot

pub type TxnId = u64;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum TxnState {
    #[default]
    Active,
    Committed,
    Aborted,
}

use eyre::{bail, Result};
use parking_lot::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};

pub const MAX_CONCURRENT_TXNS: usize = 64;

pub struct TransactionManager {
    pub(crate) global_ts: AtomicU64,
    pub(crate) active_slots: [AtomicU64; MAX_CONCURRENT_TXNS],
    pub(crate) slot_lock: Mutex<()>,
}

impl TransactionManager {
    #[allow(clippy::declare_interior_mutable_const)]
    pub fn new() -> Self {
        const INIT: AtomicU64 = AtomicU64::new(0);
        Self {
            global_ts: AtomicU64::new(1),
            #[allow(clippy::borrow_interior_mutable_const)]
            active_slots: [INIT; MAX_CONCURRENT_TXNS],
            slot_lock: Mutex::new(()),
        }
    }

    pub fn begin_txn(&self) -> Result<Transaction<'_>> {
        let _guard = self.slot_lock.lock();
        let start_ts = self.global_ts.fetch_add(1, Ordering::SeqCst);
        for (idx, slot) in self.active_slots.iter().enumerate() {
            if slot.load(Ordering::Relaxed) == 0 {
                slot.store(start_ts, Ordering::SeqCst);
                return Ok(Transaction::new(self, start_ts, idx));
            }
        }
        bail!(
            "too many concurrent transactions (max {})",
            MAX_CONCURRENT_TXNS
        )
    }

    pub fn commit_txn(&self, slot_idx: usize) -> TxnId {
        let commit_ts = self.global_ts.fetch_add(1, Ordering::SeqCst);
        self.active_slots[slot_idx].store(0, Ordering::SeqCst);
        commit_ts
    }

    pub fn abort_txn(&self, slot_idx: usize) {
        self.active_slots[slot_idx].store(0, Ordering::SeqCst);
    }

    pub fn get_global_watermark(&self) -> TxnId {
        let mut min_ts = self.global_ts.load(Ordering::Relaxed);
        for slot in &self.active_slots {
            let ts = slot.load(Ordering::Relaxed);
            if ts != 0 && ts < min_ts {
                min_ts = ts;
            }
        }
        min_ts
    }
}

impl Default for TransactionManager {
    fn default() -> Self {
        Self::new()
    }
}

pub type TableId = u32;
pub type PageId = u64;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WriteKey {
    pub table_id: TableId,
    pub key: Vec<u8>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WriteEntry {
    pub table_id: TableId,
    pub key: Vec<u8>,
    pub page_id: PageId,
    pub offset: u16,
    pub undo_page_id: Option<PageId>,
    pub undo_offset: Option<u16>,
    pub is_insert: bool,
}

pub struct Transaction<'a> {
    id: TxnId,
    slot_idx: usize,
    state: TxnState,
    write_set: smallvec::SmallVec<[WriteKey; 16]>,
    write_entries: smallvec::SmallVec<[WriteEntry; 16]>,
    manager: &'a TransactionManager,
    committed: bool,
}

impl<'a> Transaction<'a> {
    pub fn new(manager: &'a TransactionManager, id: TxnId, slot_idx: usize) -> Self {
        Self {
            id,
            slot_idx,
            state: TxnState::Active,
            write_set: smallvec::SmallVec::new(),
            write_entries: smallvec::SmallVec::new(),
            manager,
            committed: false,
        }
    }

    pub fn id(&self) -> TxnId {
        self.id
    }

    pub fn slot_idx(&self) -> usize {
        self.slot_idx
    }

    pub fn state(&self) -> TxnState {
        self.state
    }

    pub fn write_set(&self) -> &[WriteKey] {
        &self.write_set
    }

    pub fn write_entries(&self) -> &[WriteEntry] {
        &self.write_entries
    }

    pub fn add_to_write_set(&mut self, table_id: TableId, key: Vec<u8>) {
        self.write_set.push(WriteKey { table_id, key });
    }

    pub fn add_write_entry(&mut self, entry: WriteEntry) {
        self.write_entries.push(entry);
    }

    pub fn commit(mut self) -> TxnId {
        self.state = TxnState::Committed;
        self.committed = true;
        self.manager.commit_txn(self.slot_idx)
    }

    pub fn commit_with_finalize<F, E>(mut self, mut finalize: F) -> Result<TxnId, E>
    where
        F: FnMut(&WriteEntry, TxnId) -> Result<(), E>,
    {
        let commit_ts = self.manager.commit_txn(self.slot_idx);

        for entry in &self.write_entries {
            finalize(entry, commit_ts)?;
        }

        self.state = TxnState::Committed;
        self.committed = true;
        Ok(commit_ts)
    }

    pub fn rollback(mut self) {
        self.state = TxnState::Aborted;
        self.committed = true;
        self.manager.abort_txn(self.slot_idx);
    }

    pub fn rollback_with_undo<F, E>(mut self, mut undo: F) -> Result<(), E>
    where
        F: FnMut(&WriteEntry) -> Result<(), E>,
    {
        for entry in self.write_entries.iter().rev() {
            undo(entry)?;
        }

        self.state = TxnState::Aborted;
        self.committed = true;
        self.manager.abort_txn(self.slot_idx);
        Ok(())
    }
}

impl<'a> Drop for Transaction<'a> {
    fn drop(&mut self) {
        if !self.committed {
            self.manager.abort_txn(self.slot_idx);
        }
    }
}
