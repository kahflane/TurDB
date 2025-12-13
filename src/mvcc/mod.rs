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

pub mod record_header;
pub mod transaction;
pub mod undo_page;
pub mod version;

pub use record_header::RecordHeader;
pub use transaction::{
    PageId, TableId, Transaction, TransactionManager, TxnId, TxnState, WriteEntry, WriteKey,
    MAX_CONCURRENT_TXNS,
};
pub use undo_page::{UndoHeader, UndoPageReader, UndoPageWriter, UndoRecord};
pub use version::{
    VersionChainReader, VersionChainWriter, VisibilityResult, VisibleVersion, WriteCheckResult,
};

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

    #[test]
    fn commit_releases_slot() {
        let mgr = TransactionManager::new();
        let txn = mgr.begin_txn().unwrap();
        let slot_idx = txn.slot_idx();
        let commit_ts = txn.commit();
        assert!(commit_ts > 0);
        assert_eq!(mgr.active_slots[slot_idx].load(Ordering::SeqCst), 0);
    }

    #[test]
    fn commit_returns_commit_timestamp() {
        let mgr = TransactionManager::new();
        let txn1 = mgr.begin_txn().unwrap();
        let txn2 = mgr.begin_txn().unwrap();
        let commit_ts1 = txn1.commit();
        let commit_ts2 = txn2.commit();
        assert!(commit_ts2 > commit_ts1);
    }

    #[test]
    fn rollback_releases_slot() {
        let mgr = TransactionManager::new();
        let txn = mgr.begin_txn().unwrap();
        let slot_idx = txn.slot_idx();
        txn.rollback();
        assert_eq!(mgr.active_slots[slot_idx].load(Ordering::SeqCst), 0);
    }

    #[test]
    fn slot_can_be_reused_after_commit() {
        let mgr = TransactionManager::new();
        let txn1 = mgr.begin_txn().unwrap();
        let slot1 = txn1.slot_idx();
        txn1.commit();
        let txn2 = mgr.begin_txn().unwrap();
        assert_eq!(txn2.slot_idx(), slot1);
    }

    #[test]
    fn slot_can_be_reused_after_rollback() {
        let mgr = TransactionManager::new();
        let txn1 = mgr.begin_txn().unwrap();
        let slot1 = txn1.slot_idx();
        txn1.rollback();
        let txn2 = mgr.begin_txn().unwrap();
        assert_eq!(txn2.slot_idx(), slot1);
    }

    #[test]
    fn drop_without_commit_releases_slot() {
        let mgr = TransactionManager::new();
        let slot_idx;
        {
            let txn = mgr.begin_txn().unwrap();
            slot_idx = txn.slot_idx();
        }
        assert_eq!(mgr.active_slots[slot_idx].load(Ordering::SeqCst), 0);
    }

    #[test]
    fn watermark_equals_global_ts_when_no_active_transactions() {
        let mgr = TransactionManager::new();
        assert_eq!(mgr.get_global_watermark(), 1);
    }

    #[test]
    fn watermark_equals_oldest_active_transaction() {
        let mgr = TransactionManager::new();
        let _txn1 = mgr.begin_txn().unwrap();
        let _txn2 = mgr.begin_txn().unwrap();
        let _txn3 = mgr.begin_txn().unwrap();
        assert_eq!(mgr.get_global_watermark(), 1);
    }

    #[test]
    fn watermark_advances_after_oldest_commits() {
        let mgr = TransactionManager::new();
        let txn1 = mgr.begin_txn().unwrap();
        let _txn2 = mgr.begin_txn().unwrap();
        assert_eq!(mgr.get_global_watermark(), 1);
        txn1.commit();
        assert_eq!(mgr.get_global_watermark(), 2);
    }

    #[test]
    fn watermark_stays_at_oldest_uncommitted() {
        let mgr = TransactionManager::new();
        let txn1 = mgr.begin_txn().unwrap();
        let txn2 = mgr.begin_txn().unwrap();
        let txn3 = mgr.begin_txn().unwrap();
        txn2.commit();
        assert_eq!(mgr.get_global_watermark(), 1);
        txn3.commit();
        assert_eq!(mgr.get_global_watermark(), 1);
        txn1.commit();
        assert!(mgr.get_global_watermark() > 3);
    }

    #[test]
    fn transaction_write_entries_starts_empty() {
        let mgr = TransactionManager::new();
        let txn = Transaction::new(&mgr, 1, 0);
        assert!(txn.write_entries().is_empty());
    }

    #[test]
    fn transaction_can_add_write_entry() {
        use crate::mvcc::transaction::WriteEntry;
        let mgr = TransactionManager::new();
        let mut txn = Transaction::new(&mgr, 1, 0);

        txn.add_write_entry(WriteEntry {
            table_id: 1,
            key: b"key1".to_vec(),
            page_id: 100,
            offset: 50,
            undo_page_id: Some(200),
            undo_offset: Some(60),
            is_insert: false,
        });

        assert_eq!(txn.write_entries().len(), 1);
        assert_eq!(txn.write_entries()[0].page_id, 100);
    }

    #[test]
    fn commit_with_finalize_calls_callback_for_each_entry() {
        use crate::mvcc::transaction::WriteEntry;
        use std::cell::RefCell;

        let mgr = TransactionManager::new();
        let mut txn = mgr.begin_txn().unwrap();

        txn.add_write_entry(WriteEntry {
            table_id: 1,
            key: b"key1".to_vec(),
            page_id: 100,
            offset: 50,
            undo_page_id: None,
            undo_offset: None,
            is_insert: true,
        });
        txn.add_write_entry(WriteEntry {
            table_id: 1,
            key: b"key2".to_vec(),
            page_id: 101,
            offset: 60,
            undo_page_id: None,
            undo_offset: None,
            is_insert: true,
        });

        let finalized = RefCell::new(Vec::new());

        let commit_ts = txn
            .commit_with_finalize(|entry, ts| -> Result<(), ()> {
                finalized.borrow_mut().push((entry.page_id, ts));
                Ok(())
            })
            .unwrap();

        let calls = finalized.borrow();
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].0, 100);
        assert_eq!(calls[1].0, 101);
        assert_eq!(calls[0].1, commit_ts);
    }

    #[test]
    fn rollback_with_undo_calls_callback_in_reverse() {
        use crate::mvcc::transaction::WriteEntry;
        use std::cell::RefCell;

        let mgr = TransactionManager::new();
        let mut txn = mgr.begin_txn().unwrap();

        txn.add_write_entry(WriteEntry {
            table_id: 1,
            key: b"key1".to_vec(),
            page_id: 100,
            offset: 50,
            undo_page_id: Some(200),
            undo_offset: Some(10),
            is_insert: false,
        });
        txn.add_write_entry(WriteEntry {
            table_id: 1,
            key: b"key2".to_vec(),
            page_id: 101,
            offset: 60,
            undo_page_id: Some(200),
            undo_offset: Some(20),
            is_insert: false,
        });

        let undone = RefCell::new(Vec::new());

        txn.rollback_with_undo(|entry| -> Result<(), ()> {
            undone.borrow_mut().push(entry.page_id);
            Ok(())
        })
        .unwrap();

        let calls = undone.borrow();
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0], 101);
        assert_eq!(calls[1], 100);
    }

    #[test]
    fn concurrent_transactions_get_unique_ids() {
        use std::sync::Arc;
        use std::thread;

        let mgr = Arc::new(TransactionManager::new());
        let mut handles = vec![];

        for _ in 0..10 {
            let mgr_clone = Arc::clone(&mgr);
            handles.push(thread::spawn(move || {
                let txn = mgr_clone.begin_txn().unwrap();
                let id = txn.id();
                txn.commit();
                id
            }));
        }

        let mut ids: Vec<TxnId> = handles.into_iter().map(|h| h.join().unwrap()).collect();
        ids.sort();
        ids.dedup();
        assert_eq!(ids.len(), 10);
    }

    #[test]
    fn concurrent_transactions_get_unique_slots() {
        use std::sync::Arc;
        use std::thread;

        let mgr = Arc::new(TransactionManager::new());
        let mut handles = vec![];

        for _ in 0..10 {
            let mgr_clone = Arc::clone(&mgr);
            handles.push(thread::spawn(move || {
                let txn = mgr_clone.begin_txn().unwrap();
                let slot = txn.slot_idx();
                std::thread::sleep(std::time::Duration::from_millis(10));
                txn.commit();
                slot
            }));
        }

        let slots: Vec<usize> = handles.into_iter().map(|h| h.join().unwrap()).collect();
        let mut unique_slots = slots.clone();
        unique_slots.sort();
        unique_slots.dedup();
        assert_eq!(unique_slots.len(), 10);
    }

    #[test]
    fn concurrent_watermark_remains_consistent() {
        use std::sync::Arc;
        use std::thread;

        let mgr = Arc::new(TransactionManager::new());

        let txn1 = mgr.begin_txn().unwrap();
        let txn1_id = txn1.id();
        assert_eq!(mgr.get_global_watermark(), txn1_id);

        let mgr_clone = Arc::clone(&mgr);
        let handle = thread::spawn(move || {
            let txn2 = mgr_clone.begin_txn().unwrap();
            let wm = mgr_clone.get_global_watermark();
            txn2.commit();
            wm
        });

        let wm_from_thread = handle.join().unwrap();
        assert_eq!(wm_from_thread, txn1_id);

        assert_eq!(mgr.get_global_watermark(), txn1_id);
        txn1.commit();
        assert!(mgr.get_global_watermark() > txn1_id);
    }

    #[test]
    fn slots_released_on_thread_panic() {
        use std::sync::Arc;
        use std::thread;

        let mgr = Arc::new(TransactionManager::new());
        let mgr_clone = Arc::clone(&mgr);

        let handle = thread::spawn(move || {
            let txn = mgr_clone.begin_txn().unwrap();
            let _slot = txn.slot_idx();
            panic!("simulated panic");
        });

        let _ = handle.join();

        let txn = mgr.begin_txn().unwrap();
        assert_eq!(txn.slot_idx(), 0);
    }

    #[test]
    fn max_concurrent_enforced_across_threads() {
        use std::sync::atomic::{AtomicUsize, Ordering as AtomicOrdering};
        use std::sync::{Arc, Barrier};
        use std::thread;

        let mgr = Arc::new(TransactionManager::new());
        let barrier = Arc::new(Barrier::new(MAX_CONCURRENT_TXNS + 5));
        let success_count = Arc::new(AtomicUsize::new(0));
        let mut handles = vec![];

        for _ in 0..MAX_CONCURRENT_TXNS + 5 {
            let mgr_clone = Arc::clone(&mgr);
            let barrier_clone = Arc::clone(&barrier);
            let success_clone = Arc::clone(&success_count);

            handles.push(thread::spawn(move || {
                barrier_clone.wait();

                if let Ok(txn) = mgr_clone.begin_txn() {
                    success_clone.fetch_add(1, AtomicOrdering::SeqCst);
                    thread::sleep(std::time::Duration::from_millis(50));
                    txn.commit();
                }
            }));
        }

        for handle in handles {
            handle.join().unwrap();
        }

        assert_eq!(
            success_count.load(AtomicOrdering::SeqCst),
            MAX_CONCURRENT_TXNS
        );
    }
}
