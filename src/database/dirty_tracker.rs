//! # Sharded Dirty Page Tracker
//!
//! High-performance dirty page tracking using:
//! - 16-way lock sharding to reduce contention
//! - RoaringBitmap for memory-efficient page ID storage
//!
//! This replaces the global `Mutex<HashMap<u32, HashSet<u32>>>` which caused
//! severe performance degradation during large transactions.
//!
//! ## Problem Statement
//!
//! The original dirty page tracking used a single global mutex protecting a
//! `HashMap<u32, HashSet<u32>>` where the key is table_id and value is the set
//! of dirty page numbers. Every `page_mut()` call locked this mutex:
//!
//! ```ignore
//! fn page_mut(&mut self, page_no: u32) -> Result<&mut [u8]> {
//!     let mut guard = self.dirty_pages.lock();  // GLOBAL LOCK
//!     guard.entry(self.table_id).or_default().insert(page_no);
//!     drop(guard);
//!     self.storage.page_mut(page_no)
//! }
//! ```
//!
//! With 963K rows and ~3-5 page operations per row, that's **3-5 million mutex
//! lock/unlock cycles**. Combined with HashSet growth overhead, this caused
//! significant performance degradation.
//!
//! ## Solution
//!
//! 1. **16-way sharding**: Split the single mutex into 16 shards indexed by
//!    `table_id % 16`. Different tables access different shards, eliminating
//!    contention for multi-table workloads.
//!
//! 2. **RoaringBitmap**: Replace `HashSet<u32>` with `RoaringBitmap`. For 100K
//!    dirty pages, this reduces memory from ~3.2MB to ~12KB (256x improvement).
//!    Bit operations are also faster than hash computations.
//!
//! ## Performance Characteristics
//!
//! | Metric              | Before (HashMap+HashSet) | After (Sharded+Roaring) |
//! |---------------------|--------------------------|-------------------------|
//! | Lock contention     | 100% (global)            | ~6% (1/16 shards)       |
//! | Memory (100K pages) | ~3.2MB                   | ~12KB                   |
//! | Insert speed        | O(1) amortized + hash    | O(1) bit operation      |
//!
//! ## Thread Safety
//!
//! Each shard is protected by a `parking_lot::Mutex`. The shard selection is
//! deterministic based on table_id, ensuring all operations for a given table
//! go to the same shard.

use hashbrown::HashMap;
use parking_lot::Mutex;
use roaring::RoaringBitmap;

const DIRTY_SHARD_COUNT: usize = 16;

/// Sharded dirty page tracker for high-performance dirty page management.
///
/// Each shard contains a HashMap mapping table_id to a RoaringBitmap of dirty
/// page numbers. Sharding by table_id ensures that operations on different
/// tables don't contend.
pub struct ShardedDirtyTracker {
    shards: [Mutex<HashMap<u32, RoaringBitmap>>; DIRTY_SHARD_COUNT],
}

impl ShardedDirtyTracker {
    pub fn new() -> Self {
        Self {
            shards: std::array::from_fn(|_| Mutex::new(HashMap::new())),
        }
    }

    #[inline]
    fn shard_for(&self, table_id: u32) -> &Mutex<HashMap<u32, RoaringBitmap>> {
        &self.shards[(table_id as usize) % DIRTY_SHARD_COUNT]
    }

    #[inline]
    pub fn mark_dirty(&self, table_id: u32, page_no: u32) {
        let mut shard = self.shard_for(table_id).lock();
        shard.entry(table_id).or_default().insert(page_no);
    }

    pub fn has_dirty_pages(&self, table_id: u32) -> bool {
        let shard = self.shard_for(table_id).lock();
        shard
            .get(&table_id)
            .map(|bm| !bm.is_empty())
            .unwrap_or(false)
    }

    pub fn dirty_count(&self, table_id: u32) -> u64 {
        let shard = self.shard_for(table_id).lock();
        shard.get(&table_id).map(|bm| bm.len()).unwrap_or(0)
    }

    pub fn drain_for_table(&self, table_id: u32) -> Vec<u32> {
        let mut shard = self.shard_for(table_id).lock();
        match shard.get_mut(&table_id) {
            Some(pages) => {
                let result: Vec<u32> = pages.iter().collect();
                pages.clear();
                result
            }
            None => Vec::new(),
        }
    }

    pub fn clear_for_table(&self, table_id: u32) {
        let mut shard = self.shard_for(table_id).lock();
        if let Some(pages) = shard.get_mut(&table_id) {
            pages.clear();
        }
    }

    pub fn clear_all(&self) {
        for shard in &self.shards {
            shard.lock().clear();
        }
    }

    pub fn total_dirty_count(&self) -> u64 {
        let mut total = 0;
        for shard in &self.shards {
            let guard = shard.lock();
            for bm in guard.values() {
                total += bm.len();
            }
        }
        total
    }

    pub fn is_empty(&self) -> bool {
        for shard in &self.shards {
            let guard = shard.lock();
            if !guard.is_empty() {
                return false;
            }
        }
        true
    }

    pub fn all_dirty_table_ids(&self) -> Vec<u32> {
        let mut table_ids = Vec::new();
        for shard in &self.shards {
            let guard = shard.lock();
            for (&table_id, bm) in guard.iter() {
                if !bm.is_empty() {
                    table_ids.push(table_id);
                }
            }
        }
        table_ids
    }
}

impl Default for ShardedDirtyTracker {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mark_dirty_and_drain() {
        let tracker = ShardedDirtyTracker::new();

        tracker.mark_dirty(1, 10);
        tracker.mark_dirty(1, 20);
        tracker.mark_dirty(1, 30);

        assert_eq!(tracker.dirty_count(1), 3);
        assert!(tracker.has_dirty_pages(1));

        let pages = tracker.drain_for_table(1);
        assert_eq!(pages.len(), 3);
        assert!(pages.contains(&10));
        assert!(pages.contains(&20));
        assert!(pages.contains(&30));

        assert_eq!(tracker.dirty_count(1), 0);
        assert!(!tracker.has_dirty_pages(1));
    }

    #[test]
    fn different_tables_independent() {
        let tracker = ShardedDirtyTracker::new();

        tracker.mark_dirty(1, 100);
        tracker.mark_dirty(2, 200);

        assert_eq!(tracker.dirty_count(1), 1);
        assert_eq!(tracker.dirty_count(2), 1);

        tracker.clear_for_table(1);

        assert_eq!(tracker.dirty_count(1), 0);
        assert_eq!(tracker.dirty_count(2), 1);
    }

    #[test]
    fn duplicate_pages_not_counted_twice() {
        let tracker = ShardedDirtyTracker::new();

        tracker.mark_dirty(1, 100);
        tracker.mark_dirty(1, 100);
        tracker.mark_dirty(1, 100);

        assert_eq!(tracker.dirty_count(1), 1);
    }

    #[test]
    fn clear_all_clears_all_tables() {
        let tracker = ShardedDirtyTracker::new();

        for table_id in 0..100 {
            tracker.mark_dirty(table_id, table_id * 10);
        }

        assert!(tracker.total_dirty_count() > 0);

        tracker.clear_all();

        assert_eq!(tracker.total_dirty_count(), 0);
    }

    #[test]
    fn tables_in_same_shard_still_independent() {
        let tracker = ShardedDirtyTracker::new();

        let table_a = 1;
        let table_b = 1 + DIRTY_SHARD_COUNT as u32;

        tracker.mark_dirty(table_a, 10);
        tracker.mark_dirty(table_b, 20);

        assert_eq!(tracker.dirty_count(table_a), 1);
        assert_eq!(tracker.dirty_count(table_b), 1);

        tracker.clear_for_table(table_a);

        assert_eq!(tracker.dirty_count(table_a), 0);
        assert_eq!(tracker.dirty_count(table_b), 1);
    }

    #[test]
    fn drain_returns_empty_for_nonexistent_table() {
        let tracker = ShardedDirtyTracker::new();

        let pages = tracker.drain_for_table(999);
        assert!(pages.is_empty());
    }

    #[test]
    fn large_page_set_performance() {
        let tracker = ShardedDirtyTracker::new();

        for page_no in 0..100_000 {
            tracker.mark_dirty(1, page_no);
        }

        assert_eq!(tracker.dirty_count(1), 100_000);

        let pages = tracker.drain_for_table(1);
        assert_eq!(pages.len(), 100_000);
    }
}
