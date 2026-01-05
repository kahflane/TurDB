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
pub mod table;
pub mod transaction;
pub mod undo_manager;
pub mod undo_page;
pub mod version;

pub use record_header::RecordHeader;
pub use table::{MvccTable, MvccValue};
pub use transaction::{
    PageId, TableId, Transaction, TransactionManager, TxnId, TxnState, WriteEntry, WriteKey,
    MAX_CONCURRENT_TXNS,
};
pub use undo_manager::{UndoPageManager, UndoRegistry};
pub use undo_page::{UndoHeader, UndoPageReader, UndoPageWriter, UndoRecord};
pub use version::{
    VersionChainReader, VersionChainWriter, VisibilityResult, VisibleVersion, WriteCheckResult,
};
