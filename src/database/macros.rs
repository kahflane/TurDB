//! # Database Macros Module
//!
//! This module contains internal macros used across the database implementation.
//! Macros are separated into their own module to improve code organization and
//! to make them easily reusable across the DML operation modules.
//!
//! ## Purpose
//!
//! The primary macro in this module (`with_btree_storage!`) abstracts over the
//! choice between WAL-enabled and direct storage access. This pattern appears
//! frequently in UPDATE, DELETE, and other DML operations that need to modify
//! BTree data while optionally tracking changes in the Write-Ahead Log.
//!
//! ## Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────┐
//! │                  BTree Operation Flow                        │
//! ├─────────────────────────────────────────────────────────────┤
//! │                                                             │
//! │   with_btree_storage!(wal_enabled, ...)                     │
//! │           │                                                 │
//! │           ├── if WAL enabled ──► WalStoragePerTable         │
//! │           │                         │                       │
//! │           │                         ▼                       │
//! │           │                    BTree::new(&mut wal_storage) │
//! │           │                         │                       │
//! │           │                         ▼                       │
//! │           │                    Execute btree_ops closure    │
//! │           │                         │                       │
//! │           │                         ▼                       │
//! │           │                    Dirty pages tracked          │
//! │           │                                                 │
//! │           └── else ──────────► BTree::new(storage)          │
//! │                                     │                       │
//! │                                     ▼                       │
//! │                                Execute btree_ops closure    │
//! │                                     │                       │
//! │                                     ▼                       │
//! │                                Direct write to storage      │
//! │                                                             │
//! └─────────────────────────────────────────────────────────────┘
//! ```
//!
//! ## Key Design Decisions
//!
//! 1. **Closure-based API**: The macro takes a closure for the BTree operations,
//!    ensuring the BTree is properly scoped and dropped after use.
//!
//! 2. **Conditional WAL wrapping**: Rather than always using WalStorage (which
//!    has overhead even when WAL is disabled), we branch at runtime based on
//!    the `wal_enabled` flag.
//!
//! 3. **Type-level flexibility**: The closure receives `&mut BTree<S>` where S
//!    is either the direct storage or WalStoragePerTable, allowing the same
//!    operations to work with either.
//!
//! ## Usage Patterns
//!
//! ```ignore
//! with_btree_storage!(
//!     wal_enabled,
//!     storage,
//!     &dirty_tracker,
//!     table_id,
//!     root_page,
//!     |btree: &mut BTree<_>| {
//!         btree.delete(&key)?;
//!         Ok(())
//!     }
//! );
//! ```
//!
//! ## Dependencies
//!
//! The macro expands code that uses:
//! - `crate::btree::BTree` (imported inside macro)
//! - `WalStoragePerTable` (must be in scope at call site)
//!
//! The caller must have `WalStoragePerTable` imported since the macro does not
//! use a fully-qualified path for it (to keep the macro definition cleaner).
//!
//! ## Performance Characteristics
//!
//! - Zero cost when WAL is disabled (direct BTree access)
//! - Minimal overhead when WAL is enabled (thin wrapper that tracks dirty pages)
//! - Branch prediction typically optimizes the wal_enabled check after first call
//!
//! ## Thread Safety
//!
//! The macro itself has no threading implications. Thread safety depends on
//! how the storage and dirty_tracker are synchronized at the call site.

/// Executes BTree operations with or without WAL tracking.
///
/// Eliminates code duplication across UPDATE and DELETE operations where
/// branching between WalStorage (WAL enabled) and direct storage access is needed.
#[macro_export]
macro_rules! with_btree_storage {
    ($wal_enabled:expr, $storage:expr, $dirty_tracker:expr, $table_id:expr, $root_page:expr, $btree_ops:expr) => {{
        use $crate::btree::BTree;
        use $crate::storage::WalStoragePerTable;
        if $wal_enabled {
            let mut wal_storage = WalStoragePerTable::new($storage, $dirty_tracker, $table_id);
            let mut btree_mut = BTree::new(&mut wal_storage, $root_page)?;
            $btree_ops(&mut btree_mut)?;
        } else {
            let mut btree_mut = BTree::new($storage, $root_page)?;
            $btree_ops(&mut btree_mut)?;
        }
    }};
}

pub(crate) use with_btree_storage;
