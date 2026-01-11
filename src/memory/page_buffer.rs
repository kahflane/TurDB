//! # Page Buffer Pool
//!
//! Pre-allocated pool of page-sized buffers for zero-allocation commit operations.
//!
//! ## Purpose
//!
//! During transaction commit, dirty page data must be copied from mmap'd storage
//! before locks are released. This pool provides reusable buffers to avoid
//! heap allocation on every commit, supporting the zero-allocation CRUD goal.
//!
//! ## Usage
//!
//! ```ignore
//! let pool = PageBufferPool::new(16); // Pre-allocate 16 buffers
//!
//! // Acquire a buffer (returns None if pool exhausted)
//! let mut buffer = pool.acquire().expect("buffer pool exhausted");
//! buffer.copy_from_slice(page_data);
//!
//! // Buffer automatically returns to pool when dropped
//! drop(buffer);
//! ```
//!
//! ## Design
//!
//! The pool uses lock sharding (16 shards) to reduce contention under high
//! concurrency, similar to the dirty tracker sharding strategy.
//!
//! `PooledPageBuffer` uses `ManuallyDrop` instead of `Option` to make invalid
//! states unrepresentable at the type level, eliminating potential panics.
//!
//! ## Zero-Allocation Guarantee
//!
//! Unlike traditional pools that allocate on exhaustion, this pool returns `None`
//! when all buffers are in use. This enforces the zero-allocation requirement
//! during CRUD operations. Size the pool appropriately at startup based on
//! expected concurrent commit load.

use crate::storage::PAGE_SIZE;
use parking_lot::Mutex;
use std::mem::ManuallyDrop;
use std::ops::{Deref, DerefMut};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

/// Number of shards for the buffer pool to reduce lock contention.
const BUFFER_POOL_SHARD_COUNT: usize = 16;

/// A pool of reusable page-sized buffers.
///
/// Buffers are returned to the pool when dropped, enabling zero-allocation
/// commit operations after initial pool creation.
///
/// Uses 16-way lock sharding to reduce contention under high concurrency.
pub struct PageBufferPool {
    inner: Arc<PageBufferPoolInner>,
}

struct PageBufferPoolInner {
    shards: [Mutex<Vec<Box<[u8; PAGE_SIZE]>>>; BUFFER_POOL_SHARD_COUNT],
    /// Round-robin counter for distributing acquire requests across shards
    next_shard: AtomicUsize,
}

impl PageBufferPool {
    /// Create a new pool with the specified number of pre-allocated buffers.
    ///
    /// Buffers are distributed evenly across shards.
    ///
    /// # Arguments
    /// * `initial_capacity` - Number of buffers to pre-allocate
    pub fn new(initial_capacity: usize) -> Self {
        let shards: [Mutex<Vec<Box<[u8; PAGE_SIZE]>>>; BUFFER_POOL_SHARD_COUNT] =
            std::array::from_fn(|_| Mutex::new(Vec::new()));

        // Distribute buffers evenly across shards
        let per_shard = initial_capacity / BUFFER_POOL_SHARD_COUNT;
        let remainder = initial_capacity % BUFFER_POOL_SHARD_COUNT;

        for (i, shard) in shards.iter().enumerate() {
            let count = per_shard + if i < remainder { 1 } else { 0 };
            let mut guard = shard.lock();
            for _ in 0..count {
                guard.push(Box::new([0u8; PAGE_SIZE]));
            }
        }

        Self {
            inner: Arc::new(PageBufferPoolInner {
                shards,
                next_shard: AtomicUsize::new(0),
            }),
        }
    }

    /// Acquire a buffer from the pool.
    ///
    /// Uses round-robin shard selection to distribute load. If the selected
    /// shard is empty, attempts work-stealing from other shards before giving up.
    ///
    /// Returns `None` if all shards are empty (pool exhausted). This enforces
    /// the zero-allocation guarantee - callers should handle exhaustion as an
    /// error rather than silently allocating.
    ///
    /// The buffer is automatically returned to its originating shard when dropped.
    pub fn acquire(&self) -> Option<PooledPageBuffer> {
        // Round-robin shard selection
        let start_shard = self.inner.next_shard.fetch_add(1, Ordering::Relaxed) % BUFFER_POOL_SHARD_COUNT;

        // Try the primary shard first
        if let Some(buffer) = self.inner.shards[start_shard].lock().pop() {
            return Some(PooledPageBuffer {
                buffer: ManuallyDrop::new(buffer),
                pool: Arc::clone(&self.inner),
                shard_idx: start_shard,
            });
        }

        // Work-stealing: try all other shards before giving up
        for offset in 1..BUFFER_POOL_SHARD_COUNT {
            let shard_idx = (start_shard + offset) % BUFFER_POOL_SHARD_COUNT;
            if let Some(buffer) = self.inner.shards[shard_idx].lock().pop() {
                return Some(PooledPageBuffer {
                    buffer: ManuallyDrop::new(buffer),
                    pool: Arc::clone(&self.inner),
                    shard_idx,
                });
            }
        }

        // Pool exhausted - fail fast rather than allocate
        None
    }

    /// Returns the current number of available buffers in the pool (across all shards).
    ///
    /// **Performance note**: This method acquires all 16 shard locks sequentially.
    /// It is intended for monitoring/debugging only, not for hot paths.
    pub fn available(&self) -> usize {
        self.inner
            .shards
            .iter()
            .map(|s| s.lock().len())
            .sum()
    }
}

impl Clone for PageBufferPool {
    fn clone(&self) -> Self {
        Self {
            inner: Arc::clone(&self.inner),
        }
    }
}

/// A page buffer that returns to its pool when dropped.
///
/// Provides `Deref` and `DerefMut` to the underlying `[u8; PAGE_SIZE]`.
///
/// Uses `ManuallyDrop` instead of `Option` to make invalid states unrepresentable.
/// The buffer is always valid until `Drop` is called, eliminating potential panics.
pub struct PooledPageBuffer {
    /// The buffer itself. Always valid until Drop.
    /// ManuallyDrop is used so we can take ownership in Drop without moving out of self.
    buffer: ManuallyDrop<Box<[u8; PAGE_SIZE]>>,
    pool: Arc<PageBufferPoolInner>,
    /// The shard index this buffer should return to
    shard_idx: usize,
}

impl std::fmt::Debug for PooledPageBuffer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PooledPageBuffer")
            .field("shard_idx", &self.shard_idx)
            .finish()
    }
}

impl PooledPageBuffer {
    /// Copy data into the buffer, truncating or zero-padding to PAGE_SIZE.
    pub fn copy_from_page(&mut self, data: &[u8]) {
        let buf = &mut *self.buffer;
        let copy_len = data.len().min(PAGE_SIZE);
        buf[..copy_len].copy_from_slice(&data[..copy_len]);
        // Zero remaining bytes if source is smaller than PAGE_SIZE
        if copy_len < PAGE_SIZE {
            buf[copy_len..].fill(0);
        }
    }

    /// Returns the buffer contents as a slice.
    pub fn as_slice(&self) -> &[u8] {
        self.buffer.as_slice()
    }
}

impl Deref for PooledPageBuffer {
    type Target = [u8; PAGE_SIZE];

    fn deref(&self) -> &Self::Target {
        &self.buffer
    }
}

impl DerefMut for PooledPageBuffer {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.buffer
    }
}

impl Drop for PooledPageBuffer {
    fn drop(&mut self) {
        // SAFETY: We only call this once in drop, and the buffer is always valid until drop.
        // ManuallyDrop::take moves the value out, so after this point self.buffer is invalid.
        // This is safe because drop() is only called once.
        let buffer = unsafe { ManuallyDrop::take(&mut self.buffer) };
        self.pool.shards[self.shard_idx].lock().push(buffer);
    }
}

// SAFETY: PooledPageBuffer is Send because:
// - `buffer` (ManuallyDrop<Box<[u8; PAGE_SIZE]>>) is Send (owned heap data)
// - `pool` (Arc<PageBufferPoolInner>) is Send (Arc of Send+Sync inner)
// - `shard_idx` (usize) is Send (Copy type)
// The automatic impl is blocked by the raw pointer-like semantics of ManuallyDrop,
// but our usage is safe: we only call ManuallyDrop::take once in Drop.
unsafe impl Send for PooledPageBuffer {}

// SAFETY: PooledPageBuffer is Sync because:
// - All fields are Sync: ManuallyDrop<Box<T>> where T: Sync, Arc<T> where T: Sync, usize
// - PageBufferPoolInner contains only Mutex<Vec<...>> which is Sync
// - No interior mutability is exposed through &PooledPageBuffer (Deref returns &[u8])
unsafe impl Sync for PooledPageBuffer {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pool_acquire_and_return() {
        let pool = PageBufferPool::new(2);
        assert_eq!(pool.available(), 2);

        let buf1 = pool.acquire().expect("should have buffer");
        assert_eq!(pool.available(), 1);

        let buf2 = pool.acquire().expect("should have buffer");
        assert_eq!(pool.available(), 0);

        // Pool exhausted - returns None (zero-allocation guarantee)
        assert!(pool.acquire().is_none());
        assert_eq!(pool.available(), 0);

        drop(buf1);
        assert_eq!(pool.available(), 1);

        drop(buf2);
        assert_eq!(pool.available(), 2);
    }

    #[test]
    fn test_buffer_copy_from_page() {
        let pool = PageBufferPool::new(1);
        let mut buf = pool.acquire().expect("should have buffer");

        let data = [0xABu8; 100];
        buf.copy_from_page(&data);

        assert_eq!(&buf[..100], &data);
        assert_eq!(&buf[100..200], &[0u8; 100]);
    }

    #[test]
    fn test_pool_clone_shares_buffers() {
        let pool1 = PageBufferPool::new(2);
        let pool2 = pool1.clone();

        let _buf = pool1.acquire().expect("should have buffer");
        assert_eq!(pool2.available(), 1);
    }

    #[test]
    fn test_work_stealing() {
        // Create pool with buffers distributed across shards
        let pool = PageBufferPool::new(BUFFER_POOL_SHARD_COUNT);

        // Acquire all buffers - work-stealing should find them across shards
        let mut buffers = Vec::new();
        for _ in 0..BUFFER_POOL_SHARD_COUNT {
            buffers.push(pool.acquire().expect("work-stealing should find buffer"));
        }

        // Now pool should be exhausted
        assert!(pool.acquire().is_none());
        assert_eq!(pool.available(), 0);

        // Return all buffers
        drop(buffers);
        assert_eq!(pool.available(), BUFFER_POOL_SHARD_COUNT);
    }
}
