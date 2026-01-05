//! # DML Operations Module
//!
//! This module contains Data Manipulation Language (DML) operations for TurDB:
//! INSERT, UPDATE, and DELETE. These operations modify table data while respecting
//! constraints, triggering TOAST operations, and integrating with transactions.
//!
//! ## Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────────────┐
//! │                        DML Operation Flow                               │
//! ├─────────────────────────────────────────────────────────────────────────┤
//! │                                                                         │
//! │   SQL Statement                                                         │
//! │       │                                                                 │
//! │       ▼                                                                 │
//! │   ┌─────────────────────┐                                               │
//! │   │ Parse & Validate    │                                               │
//! │   │ - Check table exists│                                               │
//! │   │ - Resolve columns   │                                               │
//! │   └─────────┬───────────┘                                               │
//! │             │                                                           │
//! │             ▼                                                           │
//! │   ┌─────────────────────┐                                               │
//! │   │ Constraint Check    │                                               │
//! │   │ - PRIMARY KEY       │                                               │
//! │   │ - UNIQUE            │                                               │
//! │   │ - CHECK             │                                               │
//! │   │ - FOREIGN KEY       │                                               │
//! │   └─────────┬───────────┘                                               │
//! │             │                                                           │
//! │             ▼                                                           │
//! │   ┌─────────────────────┐                                               │
//! │   │ MVCC Processing     │                                               │
//! │   │ - Visibility check  │                                               │
//! │   │ - Undo logging      │                                               │
//! │   │ - Header wrapping   │                                               │
//! │   └─────────┬───────────┘                                               │
//! │             │                                                           │
//! │             ▼                                                           │
//! │   ┌─────────────────────┐                                               │
//! │   │ TOAST Processing    │                                               │
//! │   │ - Large values      │                                               │
//! │   │ - Chunk storage     │                                               │
//! │   └─────────┬───────────┘                                               │
//! │             │                                                           │
//! │             ▼                                                           │
//! │   ┌─────────────────────┐                                               │
//! │   │ BTree Operation     │                                               │
//! │   │ - Insert/Update/Del │                                               │
//! │   │ - WAL tracking      │                                               │
//! │   └─────────┬───────────┘                                               │
//! │             │                                                           │
//! │             ▼                                                           │
//! │   ┌─────────────────────┐                                               │
//! │   │ Index Maintenance   │                                               │
//! │   │ - Update indexes    │                                               │
//! │   │ - Constraint idx    │                                               │
//! │   └─────────┬───────────┘                                               │
//! │             │                                                           │
//! │             ▼                                                           │
//! │   ┌─────────────────────┐                                               │
//! │   │ Transaction Log     │                                               │
//! │   │ - Write entry       │                                               │
//! │   │ - Undo data         │                                               │
//! │   └─────────────────────┘                                               │
//! │                                                                         │
//! └─────────────────────────────────────────────────────────────────────────┘
//! ```
//!
//! ## Module Structure
//!
//! - `insert`: INSERT operations including ON CONFLICT handling
//! - `update`: UPDATE operations including UPDATE...FROM syntax
//! - `delete`: DELETE operations with FK constraint checking
//! - `mvcc_helpers`: MVCC integration helpers for wrapping/unwrapping records
//!
//! ## Constraint Handling
//!
//! All DML operations respect these constraints:
//!
//! 1. **PRIMARY KEY**: Checked via unique index, rejects duplicates
//! 2. **UNIQUE**: Checked via unique index or full scan for multi-column
//! 3. **CHECK**: Expression evaluated for each modified row
//! 4. **FOREIGN KEY**: Reference existence verified, delete restrictions
//! 5. **NOT NULL**: Enforced during value assignment
//!
//! ## WAL Integration
//!
//! DML operations use `with_btree_storage!` macro to transparently handle:
//! - Direct writes when WAL is disabled
//! - Dirty page tracking when WAL is enabled
//! - Automatic flush in autocommit mode
//! - Deferred flush in explicit transactions
//!
//! ## Transaction Support
//!
//! Each DML operation records write entries for transaction support:
//! - INSERT: Records inserted key for rollback delete
//! - UPDATE: Records old value for rollback restore
//! - DELETE: Records old value for rollback reinsert
//!
//! ## MVCC Support
//!
//! MVCC is always enabled. All operations use versioned records:
//! - INSERT: Prepends RecordHeader with LOCK_BIT and txn_id
//! - UPDATE: Writes old version to undo log, updates with prev_version pointer
//! - DELETE: Sets DELETE_BIT instead of physical delete
//! - COMMIT: Clears LOCK_BIT and updates to commit_ts
//! - ROLLBACK: Restores old versions from undo log
//!
//! ## Performance Characteristics
//!
//! - INSERT: O(log n) per row for BTree insert + index updates
//! - UPDATE: O(n) scan + O(log n) per update for BTree modify
//! - DELETE: O(n) scan + O(log n) per delete for BTree remove
//! - Constraint checks add overhead proportional to constraints defined

mod delete;
mod insert;
pub mod mvcc_helpers;
mod update;
