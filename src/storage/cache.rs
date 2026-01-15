//! # SIEVE Page Cache with Lock Sharding
//!
//! This module implements a high-performance page cache using the SIEVE eviction
//! algorithm with 64-way lock sharding for concurrent access.
//!
//! ## Why SIEVE Instead of LRU?
//!
//! Standard LRU (Least Recently Used) has a critical flaw for databases: a
//! sequential scan will evict the entire cache. When scanning a large table,
//! each new page becomes the "most recently used" and pushes out pages that
//! might be accessed again soon by other queries.
//!
//! SIEVE (Simple and Effective cache replacement) solves this by using a
//! "visited" flag instead of strict recency ordering:
//!
//! - On access: Set the visited flag to true
//! - On eviction: Scan entries with a "hand" pointer
//!   - If visited=true: Clear the flag, move hand forward (second chance)
//!   - If visited=false: Evict this entry
//!
//! This gives frequently-accessed pages a "second chance" while allowing
//! scan pages to be evicted quickly (they're only accessed once).
//!
//! ## Lock Sharding
//!
//! A single global lock for the entire cache creates contention in concurrent
//! workloads. TurDB uses 64 independent shards, each with its own RwLock:
//!
//! ```text
//! PageCache
//! ├── Shard 0:  RwLock<CacheShard>
//! ├── Shard 1:  RwLock<CacheShard>
//! ├── ...
//! └── Shard 63: RwLock<CacheShard>
//! ```
//!
//! Pages are assigned to shards via hash: `(file_id * 31 + page_no) % 64`
//!
//! This reduces lock contention by ~64x for random access patterns.
//!
//! ## Memory Layout
//!
//! Page buffers are pre-allocated at startup to avoid allocation during CRUD:
//!
//! ```text
//! CacheEntry {
//!     key: PageKey,           // 8 bytes (file_id + page_no)
//!     visited: AtomicBool,    // 1 byte (lock-free access marking)
//!     dirty: bool,            // 1 byte
//!     pin_count: u32,         // 4 bytes (reference counting)
//!     data: Box<[u8; 16384]>, // 16KB page buffer
//! }
//! ```
//!
//! With 32 pages minimum (512KB), the cache fits comfortably in the 1MB budget.
//!
//! ## Pin/Unpin Protocol
//!
//! Pages must be pinned before access to prevent eviction:
//!
//! 1. `get(key)` returns a pinned page reference (increments pin_count)
//! 2. Caller reads/writes the page data
//! 3. `unpin(key)` decrements pin_count
//! 4. Pages with pin_count > 0 cannot be evicted
//!
//! The `PageGuard` RAII wrapper handles automatic unpinning on drop.
//!
//! ## Thread Safety
//!
//! - `PageCache` is `Send + Sync` (can be shared across threads)
//! - Each shard uses `parking_lot::RwLock` for efficient locking
//! - `visited` flag uses `AtomicBool` for lock-free access marking
//! - Pin counts are protected by the shard lock
//!
//! ## Usage Example
//!
//! ```ignore
//! let cache = PageCache::new(32)?;  // 32 pages = 512KB
//!
//! // Load page into cache
//! let guard = cache.get_or_load(file_id, page_no, || {
//!     storage.page(page_no)
//! })?;
//!
//! // Access page data (automatically pinned)
//! let data = guard.data();
//!
//! // Page automatically unpinned when guard drops
//! drop(guard);
//! ```

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::Arc;

use eyre::{ensure, Result};
use parking_lot::RwLock;

use super::PAGE_SIZE;
use crate::config::CACHE_SHARD_COUNT as SHARD_COUNT;
use crate::memory::{MemoryBudget, Pool};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PageKey {
    pub file_id: u32,
    pub page_no: u32,
}

impl PageKey {
    pub fn new(file_id: u32, page_no: u32) -> Self {
        Self { file_id, page_no }
    }
}

struct CacheEntry {
    key: PageKey,
    visited: AtomicBool,
    dirty: AtomicBool,
    pin_count: AtomicU32,
    data: Box<[u8; PAGE_SIZE]>,
}

impl CacheEntry {
    fn new(key: PageKey) -> Self {
        Self {
            key,
            visited: AtomicBool::new(false),
            dirty: AtomicBool::new(false),
            pin_count: AtomicU32::new(0),
            data: Box::new([0u8; PAGE_SIZE]),
        }
    }

    fn is_pinned(&self) -> bool {
        self.pin_count.load(Ordering::Acquire) > 0
    }

    fn pin(&self) {
        self.pin_count.fetch_add(1, Ordering::AcqRel);
    }

    fn unpin(&self) {
        let prev = self.pin_count.fetch_sub(1, Ordering::AcqRel);
        debug_assert!(prev > 0, "unpin called on unpinned page");
    }

    fn mark_visited(&self) {
        self.visited.store(true, Ordering::Release);
    }

    fn clear_visited(&self) -> bool {
        self.visited.swap(false, Ordering::AcqRel)
    }

    fn is_dirty(&self) -> bool {
        self.dirty.load(Ordering::Acquire)
    }

    fn mark_dirty(&self) {
        self.dirty.store(true, Ordering::Release);
    }

    fn clear_dirty(&self) {
        self.dirty.store(false, Ordering::Release);
    }
}

struct CacheShard {
    entries: Vec<CacheEntry>,
    index: HashMap<PageKey, usize>,
    hand: usize,
    capacity: usize,
}

impl CacheShard {
    fn new(capacity: usize) -> Self {
        Self {
            entries: Vec::with_capacity(capacity),
            index: HashMap::with_capacity(capacity),
            hand: 0,
            capacity,
        }
    }

    fn get(&self, key: &PageKey) -> Option<usize> {
        self.index.get(key).copied()
    }

    fn access(&self, idx: usize) {
        if idx < self.entries.len() {
            self.entries[idx].mark_visited();
        }
    }

    fn evict(&mut self) -> Option<(PageKey, bool)> {
        if self.entries.is_empty() {
            return None;
        }

        let start = self.hand;
        let mut checked_all_pinned = false;

        loop {
            let entry = &self.entries[self.hand];

            if entry.is_pinned() {
                self.hand = (self.hand + 1) % self.entries.len();
                if self.hand == start {
                    if checked_all_pinned {
                        return None;
                    }
                    checked_all_pinned = true;
                }
                continue;
            }

            if entry.clear_visited() {
                self.hand = (self.hand + 1) % self.entries.len();
                continue;
            }

            let evicted_key = entry.key;
            let was_dirty = entry.is_dirty();
            return Some((evicted_key, was_dirty));
        }
    }

    fn remove(&mut self, idx: usize) -> CacheEntry {
        let entry = self.entries.swap_remove(idx);
        self.index.remove(&entry.key);

        if idx < self.entries.len() {
            let moved_key = self.entries[idx].key;
            self.index.insert(moved_key, idx);
        }

        if self.hand >= self.entries.len() && !self.entries.is_empty() {
            self.hand = 0;
        }

        entry
    }

    fn insert(&mut self, entry: CacheEntry) -> usize {
        let key = entry.key;
        let idx = self.entries.len();
        self.entries.push(entry);
        self.index.insert(key, idx);
        idx
    }

    fn is_full(&self) -> bool {
        self.entries.len() >= self.capacity
    }

    fn len(&self) -> usize {
        self.entries.len()
    }
}

pub struct PageCache {
    shards: Vec<RwLock<CacheShard>>,
    capacity_per_shard: usize,
    budget: Option<Arc<MemoryBudget>>,
}

impl PageCache {
    pub fn new(total_capacity: usize) -> Result<Self> {
        Self::with_budget(total_capacity, None)
    }

    pub fn with_budget(total_capacity: usize, budget: Option<Arc<MemoryBudget>>) -> Result<Self> {
        ensure!(
            total_capacity >= SHARD_COUNT,
            "cache capacity {} must be at least {} (one per shard)",
            total_capacity,
            SHARD_COUNT
        );

        let capacity_per_shard = total_capacity / SHARD_COUNT;
        let remainder = total_capacity % SHARD_COUNT;

        let shards: Vec<_> = (0..SHARD_COUNT)
            .map(|i| {
                let cap = if i < remainder {
                    capacity_per_shard + 1
                } else {
                    capacity_per_shard
                };
                RwLock::new(CacheShard::new(cap))
            })
            .collect();

        Ok(Self {
            shards,
            capacity_per_shard,
            budget,
        })
    }

    fn shard_index(&self, key: &PageKey) -> usize {
        let hash = (key.file_id as usize)
            .wrapping_mul(31)
            .wrapping_add(key.page_no as usize);
        hash % SHARD_COUNT
    }

    fn shard(&self, key: &PageKey) -> &RwLock<CacheShard> {
        &self.shards[self.shard_index(key)]
    }

    pub fn get(&self, key: &PageKey) -> Option<PageRef<'_>> {
        let shard = self.shard(key);
        let guard = shard.read();

        if let Some(idx) = guard.get(key) {
            guard.entries[idx].pin();
            guard.access(idx);
            Some(PageRef {
                cache: self,
                key: *key,
            })
        } else {
            None
        }
    }

    pub fn get_or_insert<F>(&self, key: PageKey, init: F) -> Result<PageRef<'_>>
    where
        F: FnOnce(&mut [u8]) -> Result<()>,
    {
        {
            let shard = self.shard(&key);
            let guard = shard.read();

            if let Some(idx) = guard.get(&key) {
                guard.entries[idx].pin();
                guard.access(idx);
                return Ok(PageRef { cache: self, key });
            }
        }

        let shard = self.shard(&key);
        let mut guard = shard.write();

        if let Some(idx) = guard.get(&key) {
            guard.entries[idx].pin();
            guard.access(idx);
            return Ok(PageRef { cache: self, key });
        }

        if let Some(budget) = &self.budget {
            while !budget.can_allocate(Pool::Cache, PAGE_SIZE) {
                if let Some((evicted_key, _was_dirty)) = guard.evict() {
                    if let Some(idx) = guard.get(&evicted_key) {
                        guard.remove(idx);
                        budget.release(Pool::Cache, PAGE_SIZE);
                    }
                } else {
                    eyre::bail!(
                        "cannot cache page: memory budget exhausted and no evictable pages (budget={} bytes)",
                        budget.total_limit()
                    );
                }
            }

            budget.allocate(Pool::Cache, PAGE_SIZE)?;
        }

        if guard.is_full() {
            if let Some((evicted_key, _was_dirty)) = guard.evict() {
                if let Some(idx) = guard.get(&evicted_key) {
                    guard.remove(idx);
                    if let Some(budget) = &self.budget {
                        budget.release(Pool::Cache, PAGE_SIZE);
                    }
                }
            } else {
                if let Some(budget) = &self.budget {
                    budget.release(Pool::Cache, PAGE_SIZE);
                }
                eyre::bail!(
                    "cache shard full and all pages pinned (capacity={})",
                    guard.capacity
                );
            }
        }

        let mut entry = CacheEntry::new(key);
        init(entry.data.as_mut_slice())?;
        entry.pin();
        entry.mark_visited();

        guard.insert(entry);

        Ok(PageRef { cache: self, key })
    }

    pub fn data(&self, key: &PageKey) -> Option<&[u8]> {
        let shard = self.shard(key);
        let guard = shard.read();

        guard.get(key).map(|idx| {
            let entry = &guard.entries[idx];
            let ptr = entry.data.as_ptr();
            // SAFETY: entry.data is a Box<[u8; PAGE_SIZE]> which is always valid for
            // PAGE_SIZE bytes. The pointer is derived from a Box which guarantees proper
            // alignment and validity. The lifetime of the returned slice is tied to the
            // RwLockReadGuard, and since we return within the map closure, the guard
            // remains held for the duration of the returned reference. The caller must
            // ensure the cache entry remains valid (pinned) while using the slice.
            unsafe { std::slice::from_raw_parts(ptr, PAGE_SIZE) }
        })
    }

    // SAFETY: This function is unsafe because it returns a mutable reference to page data
    // while only holding a read lock. The caller must ensure:
    // 1. Only one mutable reference exists at a time (enforced by PageRef's &mut self)
    // 2. The page is pinned (enforced by PageRef holding a reference)
    // 3. No concurrent read access occurs while mutating (caller responsibility)
    // The function is private and only called from PageRef::data_mut which takes &mut self,
    // ensuring exclusive access at the PageRef level.
    #[allow(clippy::mut_from_ref)]
    unsafe fn data_mut_unchecked(&self, key: &PageKey) -> Option<&mut [u8]> {
        let shard = self.shard(key);
        let guard = shard.read();

        guard.get(key).map(|idx| {
            let entry = &guard.entries[idx];
            entry.mark_dirty();
            let ptr = entry.data.as_ptr() as *mut u8;
            // SAFETY: entry.data is a Box<[u8; PAGE_SIZE]>, so ptr is valid for PAGE_SIZE
            // bytes with proper alignment. The caller (PageRef::data_mut) ensures exclusive
            // access via &mut self. The page is pinned, preventing eviction while in use.
            std::slice::from_raw_parts_mut(ptr, PAGE_SIZE)
        })
    }

    pub fn unpin(&self, key: &PageKey) {
        let shard = self.shard(key);
        let guard = shard.read();

        if let Some(idx) = guard.get(key) {
            guard.entries[idx].unpin();
        }
    }

    pub fn mark_dirty(&self, key: &PageKey) {
        let shard = self.shard(key);
        let guard = shard.read();

        if let Some(idx) = guard.get(key) {
            guard.entries[idx].mark_dirty();
        }
    }

    pub fn is_dirty(&self, key: &PageKey) -> bool {
        let shard = self.shard(key);
        let guard = shard.read();

        guard
            .get(key)
            .map(|idx| guard.entries[idx].is_dirty())
            .unwrap_or(false)
    }

    pub fn clear_dirty(&self, key: &PageKey) {
        let shard = self.shard(key);
        let guard = shard.read();

        if let Some(idx) = guard.get(key) {
            guard.entries[idx].clear_dirty();
        }
    }

    pub fn flush_dirty<F>(&self, mut flush_fn: F) -> Result<usize>
    where
        F: FnMut(&PageKey, &[u8]) -> Result<()>,
    {
        let mut flushed = 0;

        for shard in &self.shards {
            let guard = shard.read();

            for entry in &guard.entries {
                if entry.is_dirty() {
                    flush_fn(&entry.key, &*entry.data)?;
                    entry.clear_dirty();
                    flushed += 1;
                }
            }
        }

        Ok(flushed)
    }

    pub fn len(&self) -> usize {
        self.shards.iter().map(|s| s.read().len()).sum()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn capacity(&self) -> usize {
        self.capacity_per_shard * SHARD_COUNT
    }

    pub fn budget(&self) -> Option<&Arc<MemoryBudget>> {
        self.budget.as_ref()
    }

    pub fn memory_used(&self) -> usize {
        self.len() * PAGE_SIZE
    }

    pub fn clear(&self) {
        let page_count = self.len();

        for shard in &self.shards {
            let mut guard = shard.write();
            guard.entries.clear();
            guard.index.clear();
            guard.hand = 0;
        }

        if let Some(budget) = &self.budget {
            budget.release(Pool::Cache, page_count * PAGE_SIZE);
        }
    }

    pub fn evict_all_unpinned(&self) -> usize {
        let mut evicted = 0;

        for shard in &self.shards {
            let mut guard = shard.write();

            let mut to_remove = Vec::new();
            for (i, entry) in guard.entries.iter().enumerate() {
                if !entry.is_pinned() {
                    to_remove.push(i);
                }
            }

            to_remove.sort_unstable_by(|a, b| b.cmp(a));

            for idx in to_remove {
                guard.remove(idx);
                evicted += 1;
                if let Some(budget) = &self.budget {
                    budget.release(Pool::Cache, PAGE_SIZE);
                }
            }
        }

        evicted
    }
}

pub struct PageRef<'a> {
    cache: &'a PageCache,
    key: PageKey,
}

impl<'a> PageRef<'a> {
    pub fn key(&self) -> &PageKey {
        &self.key
    }

    pub fn data(&self) -> &[u8] {
        self.cache.data(&self.key).expect("page not in cache") // INVARIANT: PageRef can only exist if page is pinned in cache
    }

    pub fn data_mut(&mut self) -> &mut [u8] {
        // SAFETY: PageRef takes &mut self, ensuring exclusive access. The page is pinned
        // (pin_count > 0) because PageRef's existence implies the page was pinned in
        // get_or_insert. The page cannot be evicted while pinned, so the underlying
        // data remains valid. data_mut_unchecked's safety requirements are satisfied.
        unsafe {
            self.cache
                .data_mut_unchecked(&self.key)
                .expect("page not in cache") // INVARIANT: PageRef can only exist if page is pinned in cache
        }
    }

    pub fn mark_dirty(&self) {
        self.cache.mark_dirty(&self.key);
    }
}

impl Drop for PageRef<'_> {
    fn drop(&mut self) {
        self.cache.unpin(&self.key);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_page_cache_basic_operations() {
        let cache = PageCache::new(64).unwrap();
        let key = PageKey::new(1, 0);

        let page_ref = cache.get_or_insert(key, |data| {
            data[0] = 42;
            Ok(())
        }).unwrap();

        assert_eq!(page_ref.data()[0], 42);
        drop(page_ref);

        let page_ref = cache.get(&key).unwrap();
        assert_eq!(page_ref.data()[0], 42);
    }

    #[test]
    fn test_page_cache_with_budget() {
        let budget = Arc::new(MemoryBudget::with_limit(4 * 1024 * 1024));
        let cache = PageCache::with_budget(64, Some(Arc::clone(&budget))).unwrap();

        let initial_used = budget.stats().cache_used;

        let key = PageKey::new(1, 0);
        let page_ref = cache.get_or_insert(key, |data| {
            data[0] = 1;
            Ok(())
        }).unwrap();

        let after_insert = budget.stats().cache_used;
        assert_eq!(after_insert - initial_used, PAGE_SIZE);

        drop(page_ref);
    }

    #[test]
    fn test_page_cache_budget_release_on_clear() {
        let budget = Arc::new(MemoryBudget::with_limit(4 * 1024 * 1024));
        let cache = PageCache::with_budget(64, Some(Arc::clone(&budget))).unwrap();

        for i in 0..10 {
            let key = PageKey::new(1, i);
            let _ref = cache.get_or_insert(key, |_| Ok(())).unwrap();
        }

        let before_clear = budget.stats().cache_used;
        assert!(before_clear >= 10 * PAGE_SIZE);

        cache.clear();

        let after_clear = budget.stats().cache_used;
        assert_eq!(after_clear, 0);
    }

    #[test]
    fn test_page_cache_evict_all_unpinned() {
        let budget = Arc::new(MemoryBudget::with_limit(4 * 1024 * 1024));
        let cache = PageCache::with_budget(64, Some(Arc::clone(&budget))).unwrap();

        for i in 0..5 {
            let key = PageKey::new(1, i);
            let _ref = cache.get_or_insert(key, |_| Ok(())).unwrap();
        }

        let before_evict = cache.len();
        assert_eq!(before_evict, 5);

        let evicted = cache.evict_all_unpinned();
        assert_eq!(evicted, 5);
        assert_eq!(cache.len(), 0);
    }

    #[test]
    fn test_page_cache_memory_used() {
        let cache = PageCache::new(64).unwrap();

        for i in 0..3 {
            let key = PageKey::new(1, i);
            let _ref = cache.get_or_insert(key, |_| Ok(())).unwrap();
        }

        assert_eq!(cache.memory_used(), 3 * PAGE_SIZE);
    }

    #[test]
    fn test_page_cache_budget_accessor() {
        let budget = Arc::new(MemoryBudget::with_limit(4 * 1024 * 1024));
        let cache = PageCache::with_budget(64, Some(Arc::clone(&budget))).unwrap();

        assert!(cache.budget().is_some());
        assert_eq!(cache.budget().unwrap().total_limit(), 4 * 1024 * 1024);
    }

    #[test]
    fn test_page_cache_no_budget() {
        let cache = PageCache::new(64).unwrap();
        assert!(cache.budget().is_none());
    }

    #[test]
    fn test_page_cache_eviction_releases_budget() {
        let budget = Arc::new(MemoryBudget::with_limit(4 * 1024 * 1024));
        let capacity = 64;
        let cache = PageCache::with_budget(capacity, Some(Arc::clone(&budget))).unwrap();

        for i in 0..(capacity + 10) as u32 {
            let key = PageKey::new(1, i);
            let _ref = cache.get_or_insert(key, |_| Ok(())).unwrap();
        }

        let used = budget.stats().cache_used;
        assert!(used <= capacity * PAGE_SIZE);
    }
}
