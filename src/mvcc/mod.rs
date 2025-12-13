//! # Multi-Version Concurrency Control (MVCC)
//!
//! This module implements Snapshot Isolation (SI) for TurDB, providing
//! concurrent read/write access without blocking readers. The design
//! prioritizes zero-allocation during CRUD and fits within a 1MB memory budget.
//!
//! ## Design Philosophy
//!
//! ### Single-Writer / Multi-Reader with Row-Level Locking
//!
//! TurDB uses a hybrid approach optimized for embedded use cases:
//! - Writers acquire row-level locks stored directly in record headers
//! - Readers never block - they traverse version chains to find visible data
//! - Conflict detection happens at commit time via write-write checking
//!
//! ### Version Storage: Inline Newest + Undo Log
//!
//! The newest version lives in the B-tree leaf for O(1) access to current data.
//! Older versions are stored in undo pages, linked via prev_version pointers.
//! This optimizes the common case: reading the latest committed data.
//!
//! ```text
//!        [ B-Tree Leaf Page ]
//!        +-----------------------------+
//!        | Key="user:1"                |
//!        | Value:                      |
//!        |  [RecordHeader]             |
//!        |    Flags: 0x00              |
//!        |    TxnId: 105               |
//!        |    Prev:  Page 4, Off 64  -----\
//!        |  [Body: "Alice", 30]        |     \
//!        +-----------------------------+      \
//!                                              \
//!                                               \   [ Undo Page 4 ]
//!                                                \  +------------------+
//!                                                 ->| Offset 64:       |
//!                                                   |  TxnId: 100      |
//!                                                   |  Prev: NULL      |
//!                                                   |  Body: "Alice",29|
//!                                                   +------------------+
//! ```
//!
//! ## Memory Budget Constraints
//!
//! To fit within 1MB total RAM:
//! - Fixed-size slot array for active transactions (64 slots = 512 bytes)
//! - SmallVec<16> for write sets (90% of txns touch <16 rows)
//! - No dynamic allocation in TransactionManager
//! - Watermark calculation is O(64) - effectively constant time
//!
//! ## Transaction Lifecycle
//!
//! ```text
//! begin() ─────> Active ─────> commit() ─────> Committed
//!                  │                              │
//!                  │                              v
//!                  └──> rollback() ───> Aborted   └──> Versions visible
//! ```
//!
//! ## Visibility Rules (Snapshot Isolation)
//!
//! A version V is visible to transaction T if:
//! 1. V.txn_id <= T.read_ts (created before snapshot)
//! 2. V.txn_id is committed (not from an in-progress transaction)
//! 3. V is not deleted, or deleted by a transaction > T.read_ts
//!
//! ## Lock-Free Design
//!
//! - Global timestamp: AtomicU64 for allocation without locking
//! - Slot array: Fixed-size, only needs lock for slot claim/release
//! - Watermark: Computed by iterating slots without global lock
//! - Record headers: Locked via CAS on flags byte
//!
//! ## Key Structures
//!
//! - `TxnId`: 8-byte transaction identifier (u64)
//! - `TxnState`: Transaction lifecycle state
//! - `TransactionManager`: Global coordinator with timestamp allocation
//! - `Transaction`: Per-transaction context with write set
//! - `RecordHeader`: 17-byte header prepended to each row version
//!
//! ## Safety Considerations
//!
//! The RecordHeader uses manual byte parsing to handle potentially unaligned
//! memory from mmap. This avoids UB on architectures requiring alignment.

pub mod transaction;

pub use transaction::{Transaction, TransactionManager, TxnId, TxnState, MAX_CONCURRENT_TXNS};

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::Ordering;

    #[test]
    fn txn_id_is_u64() {
        let id: TxnId = 42;
        assert_eq!(id, 42u64);
    }

    #[test]
    fn txn_state_default_is_active() {
        let state = TxnState::default();
        assert!(matches!(state, TxnState::Active));
    }

    #[test]
    fn txn_state_transitions_to_committed() {
        let state = TxnState::Committed;
        assert!(matches!(state, TxnState::Committed));
    }

    #[test]
    fn txn_state_transitions_to_aborted() {
        let state = TxnState::Aborted;
        assert!(matches!(state, TxnState::Aborted));
    }

    #[test]
    fn txn_state_is_copy() {
        let state1 = TxnState::Active;
        let state2 = state1;
        assert!(matches!(state1, TxnState::Active));
        assert!(matches!(state2, TxnState::Active));
    }

    #[test]
    fn transaction_manager_new_initializes_global_ts_to_one() {
        let mgr = TransactionManager::new();
        assert_eq!(mgr.global_ts.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn transaction_manager_new_initializes_all_slots_to_zero() {
        let mgr = TransactionManager::new();
        for i in 0..MAX_CONCURRENT_TXNS {
            assert_eq!(mgr.active_slots[i].load(Ordering::Relaxed), 0);
        }
    }

    #[test]
    fn transaction_manager_has_64_slots() {
        assert_eq!(MAX_CONCURRENT_TXNS, 64);
        let mgr = TransactionManager::new();
        assert_eq!(mgr.active_slots.len(), 64);
    }

    #[test]
    fn transaction_manager_default_equals_new() {
        let mgr1 = TransactionManager::new();
        let mgr2 = TransactionManager::default();
        assert_eq!(
            mgr1.global_ts.load(Ordering::SeqCst),
            mgr2.global_ts.load(Ordering::SeqCst)
        );
    }

    #[test]
    fn transaction_has_id_and_slot_idx() {
        let mgr = TransactionManager::new();
        let txn = Transaction::new(&mgr, 42, 5);
        assert_eq!(txn.id(), 42);
        assert_eq!(txn.slot_idx(), 5);
    }

    #[test]
    fn transaction_initial_state_is_active() {
        let mgr = TransactionManager::new();
        let txn = Transaction::new(&mgr, 1, 0);
        assert!(matches!(txn.state(), TxnState::Active));
    }

    #[test]
    fn transaction_write_set_starts_empty() {
        let mgr = TransactionManager::new();
        let txn = Transaction::new(&mgr, 1, 0);
        assert!(txn.write_set().is_empty());
    }

    #[test]
    fn transaction_can_add_to_write_set() {
        let mgr = TransactionManager::new();
        let mut txn = Transaction::new(&mgr, 1, 0);
        txn.add_to_write_set(100, vec![1, 2, 3]);
        assert_eq!(txn.write_set().len(), 1);
    }

    #[test]
    fn begin_txn_returns_transaction_with_start_timestamp() {
        let mgr = TransactionManager::new();
        let txn = mgr.begin_txn().unwrap();
        assert_eq!(txn.id(), 1);
    }

    #[test]
    fn begin_txn_increments_timestamp_for_each_transaction() {
        let mgr = TransactionManager::new();
        let txn1 = mgr.begin_txn().unwrap();
        let txn2 = mgr.begin_txn().unwrap();
        assert_eq!(txn1.id(), 1);
        assert_eq!(txn2.id(), 2);
    }

    #[test]
    fn begin_txn_assigns_unique_slot() {
        let mgr = TransactionManager::new();
        let txn1 = mgr.begin_txn().unwrap();
        let txn2 = mgr.begin_txn().unwrap();
        assert_ne!(txn1.slot_idx(), txn2.slot_idx());
    }

    #[test]
    fn begin_txn_marks_slot_as_active() {
        let mgr = TransactionManager::new();
        let txn = mgr.begin_txn().unwrap();
        let slot_val = mgr.active_slots[txn.slot_idx()].load(Ordering::SeqCst);
        assert_eq!(slot_val, txn.id());
    }

    #[test]
    fn begin_txn_fails_when_all_slots_full() {
        let mgr = TransactionManager::new();
        let mut txns = Vec::new();
        for _ in 0..MAX_CONCURRENT_TXNS {
            txns.push(mgr.begin_txn().unwrap());
        }
        assert!(mgr.begin_txn().is_err());
    }
}
