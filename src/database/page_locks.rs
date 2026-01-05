//! # Fine-Grained Page-Level Locking
//!
//! This module implements page-level locking for TurDB, enabling concurrent
//! writes to different pages within the same database.
//!
//! ## Why Page-Level Locks?
//!
//! A global database lock serializes all writes, limiting throughput to a
//! single writer at a time. Page-level locks allow:
//!
//! - Multiple writers to different tables (different pages)
//! - Multiple writers to different pages of the same large table
//! - Readers concurrent with writers on different pages
//!
//! ## Lock Hierarchy
//!
//! To prevent deadlocks, locks are acquired in a consistent order:
//!
//! ```text
//! 1. Table intent lock (shared for reads, exclusive for schema changes)
//! 2. Page locks in ascending page number order
//! ```
//!
//! ## Lock Types
//!
//! - **TableIntentShared (IS)**: Intend to read pages - blocks exclusive table locks
//! - **TableIntentExclusive (IX)**: Intend to write pages - blocks exclusive table locks
//! - **TableExclusive (X)**: Full table lock for DDL - blocks all access
//! - **PageShared (S)**: Read lock on a specific page
//! - **PageExclusive (X)**: Write lock on a specific page
//!
//! ## Lock Sharding
//!
//! To reduce contention on the lock manager itself, page locks are sharded
//! by table_id. This allows concurrent lock acquisition for different tables
//! without any contention.
//!
//! ## Performance Characteristics
//!
//! - Lock acquisition: O(1) hash lookup + lock contention
//! - Memory: ~64 bytes per locked page (cleaned up on unlock)
//! - Shards: 64 for table locks, reduces lock manager contention

use parking_lot::{Mutex, RwLock};
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

const TABLE_SHARD_COUNT: usize = 64;
const PAGE_SHARD_COUNT: usize = 256;

/// Statistics for monitoring lock performance
#[derive(Debug, Default)]
pub struct LockStats {
    pub page_locks_acquired: AtomicU64,
    pub page_locks_contended: AtomicU64,
    pub table_locks_acquired: AtomicU64,
}

impl LockStats {
    pub fn record_page_lock(&self, contended: bool) {
        self.page_locks_acquired.fetch_add(1, Ordering::Relaxed);
        if contended {
            self.page_locks_contended.fetch_add(1, Ordering::Relaxed);
        }
    }

    pub fn record_table_lock(&self) {
        self.table_locks_acquired.fetch_add(1, Ordering::Relaxed);
    }
}

/// A unique identifier for a page within a table
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PageId {
    pub table_id: u32,
    pub page_no: u32,
}

impl PageId {
    pub fn new(table_id: u32, page_no: u32) -> Self {
        Self { table_id, page_no }
    }

    fn shard_index(&self) -> usize {
        let hash = (self.table_id as usize)
            .wrapping_mul(31)
            .wrapping_add(self.page_no as usize);
        hash % PAGE_SHARD_COUNT
    }
}

/// Per-page lock entry with reference counting for cleanup
struct PageLockEntry {
    lock: RwLock<()>,
    ref_count: AtomicU64,
}

impl PageLockEntry {
    fn new() -> Self {
        Self {
            lock: RwLock::new(()),
            ref_count: AtomicU64::new(1),
        }
    }

    fn acquire(&self) {
        self.ref_count.fetch_add(1, Ordering::AcqRel);
    }

    fn release(&self) -> bool {
        self.ref_count.fetch_sub(1, Ordering::AcqRel) == 1
    }
}

/// A shard of page locks
struct PageLockShard {
    locks: Mutex<HashMap<PageId, Arc<PageLockEntry>>>,
}

impl PageLockShard {
    fn new() -> Self {
        Self {
            locks: Mutex::new(HashMap::new()),
        }
    }

    fn get_or_create(&self, page_id: PageId) -> Arc<PageLockEntry> {
        let mut map = self.locks.lock();
        if let Some(entry) = map.get(&page_id) {
            entry.acquire();
            return Arc::clone(entry);
        }
        let entry = Arc::new(PageLockEntry::new());
        map.insert(page_id, Arc::clone(&entry));
        entry
    }

    fn try_cleanup(&self, page_id: PageId, entry: &PageLockEntry) {
        if entry.release() {
            let mut map = self.locks.lock();
            // Double-check ref_count is still 0 under lock
            if entry.ref_count.load(Ordering::Acquire) == 0 {
                map.remove(&page_id);
            }
        }
    }
}

/// Table-level intent lock
struct TableLockShard {
    locks: RwLock<HashMap<u32, TableLockState>>,
}

#[derive(Default)]
struct TableLockState {
    intent_shared: u32,
    intent_exclusive: u32,
    exclusive: bool,
}

impl TableLockShard {
    fn new() -> Self {
        Self {
            locks: RwLock::new(HashMap::new()),
        }
    }
}

/// Guard for a page read lock - uses RAII to release lock on drop
pub struct PageReadGuard {
    shard: *const PageLockShard,
    page_id: PageId,
    entry: Arc<PageLockEntry>,
}

// SAFETY: PageReadGuard only holds a pointer to PageLockShard which is part of PageLockManager.
// The PageLockManager is always stored in Arc<SharedDatabase> which ensures it outlives all guards.
// The entry Arc keeps the lock data alive.
unsafe impl Send for PageReadGuard {}
unsafe impl Sync for PageReadGuard {}

impl Drop for PageReadGuard {
    fn drop(&mut self) {
        // Release the read lock
        // SAFETY: We acquired a read lock in page_read, so we must have a read lock to release.
        // parking_lot RwLock allows force_unlock which releases the lock held by this thread.
        unsafe { self.entry.lock.force_unlock_read() };

        // Try to clean up the entry if we're the last reference
        // SAFETY: shard pointer is valid for the lifetime of PageLockManager which outlives this guard
        unsafe { (*self.shard).try_cleanup(self.page_id, &self.entry) };
    }
}

/// Guard for a page write lock - uses RAII to release lock on drop
pub struct PageWriteGuard {
    shard: *const PageLockShard,
    page_id: PageId,
    entry: Arc<PageLockEntry>,
}

// SAFETY: Same reasoning as PageReadGuard
unsafe impl Send for PageWriteGuard {}
unsafe impl Sync for PageWriteGuard {}

impl Drop for PageWriteGuard {
    fn drop(&mut self) {
        // Release the write lock
        // SAFETY: We acquired a write lock in page_write, so we must have a write lock to release.
        unsafe { self.entry.lock.force_unlock_write() };

        // Try to clean up the entry if we're the last reference
        // SAFETY: shard pointer is valid for the lifetime of PageLockManager which outlives this guard
        unsafe { (*self.shard).try_cleanup(self.page_id, &self.entry) };
    }
}

/// Guard for table intent-shared lock
pub struct TableIntentSharedGuard<'a> {
    lock_manager: &'a PageLockManager,
    table_id: u32,
}

impl Drop for TableIntentSharedGuard<'_> {
    fn drop(&mut self) {
        self.lock_manager.release_table_intent_shared(self.table_id);
    }
}

/// Guard for table intent-exclusive lock
pub struct TableIntentExclusiveGuard<'a> {
    lock_manager: &'a PageLockManager,
    table_id: u32,
}

impl Drop for TableIntentExclusiveGuard<'_> {
    fn drop(&mut self) {
        self.lock_manager.release_table_intent_exclusive(self.table_id);
    }
}

/// The main page lock manager
pub struct PageLockManager {
    page_shards: Vec<PageLockShard>,
    table_shards: Vec<TableLockShard>,
    pub stats: LockStats,
}

impl Default for PageLockManager {
    fn default() -> Self {
        Self::new()
    }
}

impl PageLockManager {
    pub fn new() -> Self {
        let page_shards = (0..PAGE_SHARD_COUNT)
            .map(|_| PageLockShard::new())
            .collect();
        let table_shards = (0..TABLE_SHARD_COUNT)
            .map(|_| TableLockShard::new())
            .collect();

        Self {
            page_shards,
            table_shards,
            stats: LockStats::default(),
        }
    }

    fn table_shard_index(&self, table_id: u32) -> usize {
        table_id as usize % TABLE_SHARD_COUNT
    }

    /// Acquire an intent-shared lock on a table (for reading pages)
    pub fn table_intent_shared(&self, table_id: u32) -> TableIntentSharedGuard<'_> {
        let shard_idx = self.table_shard_index(table_id);
        let shard = &self.table_shards[shard_idx];

        loop {
            {
                let mut map = shard.locks.write();
                let state = map.entry(table_id).or_default();

                if !state.exclusive {
                    state.intent_shared += 1;
                    self.stats.record_table_lock();
                    return TableIntentSharedGuard {
                        lock_manager: self,
                        table_id,
                    };
                }
            }
            std::thread::yield_now();
        }
    }

    /// Acquire an intent-exclusive lock on a table (for writing pages)
    pub fn table_intent_exclusive(&self, table_id: u32) -> TableIntentExclusiveGuard<'_> {
        let shard_idx = self.table_shard_index(table_id);
        let shard = &self.table_shards[shard_idx];

        loop {
            {
                let mut map = shard.locks.write();
                let state = map.entry(table_id).or_default();

                if !state.exclusive {
                    state.intent_exclusive += 1;
                    self.stats.record_table_lock();
                    return TableIntentExclusiveGuard {
                        lock_manager: self,
                        table_id,
                    };
                }
            }
            std::thread::yield_now();
        }
    }

    fn release_table_intent_shared(&self, table_id: u32) {
        let shard_idx = self.table_shard_index(table_id);
        let shard = &self.table_shards[shard_idx];

        let mut map = shard.locks.write();
        if let Some(state) = map.get_mut(&table_id) {
            state.intent_shared = state.intent_shared.saturating_sub(1);

            // Clean up if no locks held
            if state.intent_shared == 0 && state.intent_exclusive == 0 && !state.exclusive {
                map.remove(&table_id);
            }
        }
    }

    fn release_table_intent_exclusive(&self, table_id: u32) {
        let shard_idx = self.table_shard_index(table_id);
        let shard = &self.table_shards[shard_idx];

        let mut map = shard.locks.write();
        if let Some(state) = map.get_mut(&table_id) {
            state.intent_exclusive = state.intent_exclusive.saturating_sub(1);

            // Clean up if no locks held
            if state.intent_shared == 0 && state.intent_exclusive == 0 && !state.exclusive {
                map.remove(&table_id);
            }
        }
    }

    /// Acquire a read lock on a page (blocking)
    ///
    /// Should be called while holding a table intent-shared lock.
    pub fn page_read(&self, table_id: u32, page_no: u32) -> PageReadGuard {
        let page_id = PageId::new(table_id, page_no);
        let shard = &self.page_shards[page_id.shard_index()];
        let entry = shard.get_or_create(page_id);

        // Check if lock was contended
        let contended = entry.lock.try_read().is_none();

        // Acquire the read lock (blocking)
        let guard = entry.lock.read();
        // Forget the guard to prevent automatic unlock - we'll manually unlock in Drop
        std::mem::forget(guard);

        self.stats.record_page_lock(contended);

        PageReadGuard {
            shard: shard as *const PageLockShard,
            page_id,
            entry,
        }
    }

    /// Acquire a write lock on a page (blocking)
    ///
    /// Should be called while holding a table intent-exclusive lock.
    pub fn page_write(&self, table_id: u32, page_no: u32) -> PageWriteGuard {
        let page_id = PageId::new(table_id, page_no);
        let shard = &self.page_shards[page_id.shard_index()];
        let entry = shard.get_or_create(page_id);

        // Check if lock was contended
        let contended = entry.lock.try_write().is_none();

        // Acquire the write lock (blocking)
        let guard = entry.lock.write();
        // Forget the guard to prevent automatic unlock - we'll manually unlock in Drop
        std::mem::forget(guard);

        self.stats.record_page_lock(contended);

        PageWriteGuard {
            shard: shard as *const PageLockShard,
            page_id,
            entry,
        }
    }

    /// Acquire multiple page write locks in order (deadlock-safe)
    ///
    /// Pages are sorted by (table_id, page_no) before locking to ensure
    /// consistent ordering across all transactions.
    pub fn page_write_multi(&self, pages: &[(u32, u32)]) -> Vec<PageWriteGuard> {
        let mut sorted: Vec<_> = pages.to_vec();
        sorted.sort();

        sorted.into_iter()
            .map(|(table_id, page_no)| self.page_write(table_id, page_no))
            .collect()
    }

    /// Get current lock statistics
    pub fn stats(&self) -> &LockStats {
        &self.stats
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::thread;

    #[test]
    fn test_page_read_lock() {
        let manager = PageLockManager::new();

        let guard = manager.page_read(1, 100);
        drop(guard);

        assert_eq!(manager.stats.page_locks_acquired.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn test_page_write_lock() {
        let manager = PageLockManager::new();

        let guard = manager.page_write(1, 100);
        drop(guard);

        assert_eq!(manager.stats.page_locks_acquired.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn test_concurrent_read_locks() {
        let manager = Arc::new(PageLockManager::new());

        let guard1 = manager.page_read(1, 100);
        let guard2 = manager.page_read(1, 100);

        drop(guard1);
        drop(guard2);

        assert_eq!(manager.stats.page_locks_acquired.load(Ordering::Relaxed), 2);
    }

    #[test]
    fn test_different_pages_concurrent() {
        let manager = Arc::new(PageLockManager::new());
        let manager2 = Arc::clone(&manager);

        let guard1 = manager.page_write(1, 100);

        let handle = thread::spawn(move || {
            // Different page should succeed immediately
            manager2.page_write(1, 200)
        });

        let guard2 = handle.join().unwrap();

        drop(guard1);
        drop(guard2);

        assert_eq!(manager.stats.page_locks_acquired.load(Ordering::Relaxed), 2);
    }

    #[test]
    fn test_table_intent_lock() {
        let manager = PageLockManager::new();

        let table_guard = manager.table_intent_shared(1);
        let page_guard = manager.page_read(1, 100);

        drop(page_guard);
        drop(table_guard);
    }

    #[test]
    fn test_multi_page_lock_ordering() {
        let manager = PageLockManager::new();

        // Acquire in reverse order - should still work
        let guards = manager.page_write_multi(&[(1, 300), (1, 100), (1, 200)]);
        assert_eq!(guards.len(), 3);
    }

    #[test]
    fn test_lock_cleanup() {
        let manager = PageLockManager::new();

        {
            let _guard = manager.page_write(1, 100);
        }

        // Lock should be cleaned up
        let shard = &manager.page_shards[PageId::new(1, 100).shard_index()];
        let map = shard.locks.lock();
        assert!(map.is_empty());
    }
}
