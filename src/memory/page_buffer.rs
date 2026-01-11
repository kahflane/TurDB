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
//! // Acquire a buffer (from pool or newly allocated if pool empty)
//! let mut buffer = pool.acquire();
//! buffer.copy_from_slice(page_data);
//!
//! // Buffer automatically returns to pool when dropped
//! drop(buffer);
//! ```

use crate::storage::PAGE_SIZE;
use parking_lot::Mutex;
use std::ops::{Deref, DerefMut};
use std::sync::Arc;

/// A pool of reusable page-sized buffers.
///
/// Buffers are returned to the pool when dropped, enabling zero-allocation
/// commit operations after initial pool creation.
pub struct PageBufferPool {
    inner: Arc<PageBufferPoolInner>,
}

struct PageBufferPoolInner {
    buffers: Mutex<Vec<Box<[u8; PAGE_SIZE]>>>,
}

impl PageBufferPool {
    /// Create a new pool with the specified number of pre-allocated buffers.
    ///
    /// # Arguments
    /// * `initial_capacity` - Number of buffers to pre-allocate
    pub fn new(initial_capacity: usize) -> Self {
        let buffers: Vec<Box<[u8; PAGE_SIZE]>> = (0..initial_capacity)
            .map(|_| Box::new([0u8; PAGE_SIZE]))
            .collect();

        Self {
            inner: Arc::new(PageBufferPoolInner {
                buffers: Mutex::new(buffers),
            }),
        }
    }

    /// Acquire a buffer from the pool.
    ///
    /// If the pool is empty, allocates a new buffer. The buffer is automatically
    /// returned to the pool when dropped.
    pub fn acquire(&self) -> PooledPageBuffer {
        let buffer = {
            let mut pool = self.inner.buffers.lock();
            pool.pop()
        };

        let buffer = buffer.unwrap_or_else(|| Box::new([0u8; PAGE_SIZE]));

        PooledPageBuffer {
            buffer: Some(buffer),
            pool: Arc::clone(&self.inner),
        }
    }

    /// Returns the current number of available buffers in the pool.
    pub fn available(&self) -> usize {
        self.inner.buffers.lock().len()
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
pub struct PooledPageBuffer {
    buffer: Option<Box<[u8; PAGE_SIZE]>>,
    pool: Arc<PageBufferPoolInner>,
}

impl std::fmt::Debug for PooledPageBuffer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PooledPageBuffer")
            .field("has_buffer", &self.buffer.is_some())
            .finish()
    }
}

impl PooledPageBuffer {
    /// Copy data into the buffer, truncating or zero-padding to PAGE_SIZE.
    pub fn copy_from_page(&mut self, data: &[u8]) {
        let buf = self.buffer.as_mut().expect("buffer already taken");
        let copy_len = data.len().min(PAGE_SIZE);
        buf[..copy_len].copy_from_slice(&data[..copy_len]);
        // Zero remaining bytes if source is smaller than PAGE_SIZE
        if copy_len < PAGE_SIZE {
            buf[copy_len..].fill(0);
        }
    }

    /// Returns the buffer contents as a slice.
    pub fn as_slice(&self) -> &[u8] {
        self.buffer.as_ref().expect("buffer already taken").as_slice()
    }
}

impl Deref for PooledPageBuffer {
    type Target = [u8; PAGE_SIZE];

    fn deref(&self) -> &Self::Target {
        self.buffer.as_ref().expect("buffer already taken")
    }
}

impl DerefMut for PooledPageBuffer {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.buffer.as_mut().expect("buffer already taken")
    }
}

impl Drop for PooledPageBuffer {
    fn drop(&mut self) {
        if let Some(buffer) = self.buffer.take() {
            self.pool.buffers.lock().push(buffer);
        }
    }
}

// PooledPageBuffer cannot be Send/Sync automatically due to the Arc<PageBufferPoolInner>
// but since PageBufferPoolInner only contains Mutex<Vec<...>>, it's safe.
unsafe impl Send for PooledPageBuffer {}
unsafe impl Sync for PooledPageBuffer {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pool_acquire_and_return() {
        let pool = PageBufferPool::new(2);
        assert_eq!(pool.available(), 2);

        let buf1 = pool.acquire();
        assert_eq!(pool.available(), 1);

        let buf2 = pool.acquire();
        assert_eq!(pool.available(), 0);

        // Pool empty, this will allocate
        let _buf3 = pool.acquire();
        assert_eq!(pool.available(), 0);

        drop(buf1);
        assert_eq!(pool.available(), 1);

        drop(buf2);
        assert_eq!(pool.available(), 2);
    }

    #[test]
    fn test_buffer_copy_from_page() {
        let pool = PageBufferPool::new(1);
        let mut buf = pool.acquire();

        let data = [0xABu8; 100];
        buf.copy_from_page(&data);

        assert_eq!(&buf[..100], &data);
        assert_eq!(&buf[100..200], &[0u8; 100]);
    }

    #[test]
    fn test_pool_clone_shares_buffers() {
        let pool1 = PageBufferPool::new(2);
        let pool2 = pool1.clone();

        let _buf = pool1.acquire();
        assert_eq!(pool2.available(), 1);
    }
}
