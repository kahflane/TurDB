//! # Group Commit Implementation
//!
//! This module implements a "Push-Based" group commit mechanism for TurDB.
//! Transactions buffer their dirty pages and "push" them into the commit queue,
//! allowing the leader to write to WAL without acquiring any storage locks.
//!
//! ## Design Overview
//!
//! ```text
//! Thread 1 (Buffer Data) ──┐
//! Thread 2 (Buffer Data) ──┼──► GroupCommitQueue (Data Payloads) ──► Write WAL
//! Thread 3 (Buffer Data) ──┘
//! ```
//!
//! ## Locking Model (Deadlock Free)
//!
//! 1. **Transaction Phase**:
//!    - Acquire Table/Page Locks.
//!    - Update Memory.
//!    - Read Dirty Pages -> Buffer.
//!    - **Release Locks**.
//! 2. **Commit Phase**:
//!    - Submit Buffer to Queue.
//!    - Wait for Flush.
//! 3. **Flush Phase** (Leader):
//!    - Buffer -> WAL File.
//!    - No Storage Locks required.

use crate::memory::FallbackBuffer;
use parking_lot::{Condvar, Mutex};
use smallvec::SmallVec;
use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::{Duration, Instant};

/// Payload type for dirty pages: (table_id, page_id, buffer, db_size)
///
/// Uses `FallbackBuffer` which prefers pooled buffers but falls back to heap
/// allocation when the pool is exhausted. This prevents commit failures under
/// high concurrency while still benefiting from pooling in the common case.
pub type CommitPayload = SmallVec<[(u32, u32, FallbackBuffer, u32); 4]>;

/// Configuration for group commit behavior
#[derive(Debug, Clone, Copy)]
pub struct GroupCommitConfig {
    /// Maximum number of commits to batch before forcing a flush
    pub max_batch_size: usize,
    /// Maximum time to wait for more commits before flushing (microseconds)
    pub max_wait_us: u64,
    /// Minimum commits to wait for before considering a flush
    pub min_batch_size: usize,
}

impl Default for GroupCommitConfig {
    fn default() -> Self {
        Self {
            max_batch_size: 64,
            max_wait_us: 1000,
            min_batch_size: 1,
        }
    }
}

impl GroupCommitConfig {
    /// Configuration optimized for high throughput (more batching)
    pub fn high_throughput() -> Self {
        Self {
            max_batch_size: 128,
            max_wait_us: 5000,
            min_batch_size: 4,
        }
    }

    /// Configuration optimized for low latency (less batching)
    pub fn low_latency() -> Self {
        Self {
            max_batch_size: 16,
            max_wait_us: 100,
            min_batch_size: 1,
        }
    }
}

/// A pending commit request waiting to be flushed
#[derive(Debug)]
pub struct PendingCommit {
    /// Unique batch ID for this commit
    pub batch_id: u64,
    /// Buffered dirty pages: (table_id, page_id, data)
    pub payload: CommitPayload,
    /// Completion flag - set to true when WAL flush completes
    pub completed: AtomicBool,
    /// Error message if commit failed (empty string means success)
    pub error: Mutex<Option<String>>,
}

impl PendingCommit {
    pub fn new(batch_id: u64, payload: CommitPayload) -> Self {
        Self {
            batch_id,
            payload,
            completed: AtomicBool::new(false),
            error: Mutex::new(None),
        }
    }

    pub fn mark_completed(&self) {
        self.completed.store(true, Ordering::Release);
    }

    pub fn mark_failed(&self, error: String) {
        *self.error.lock() = Some(error);
        self.completed.store(true, Ordering::Release);
    }

    pub fn is_completed(&self) -> bool {
        self.completed.load(Ordering::Acquire)
    }

    pub fn take_error(&self) -> Option<String> {
        self.error.lock().take()
    }
}

/// Statistics for monitoring group commit performance
#[derive(Debug, Default)]
pub struct GroupCommitStats {
    /// Total number of commits processed
    pub total_commits: AtomicU64,
    /// Total number of WAL flushes (batches)
    pub total_flushes: AtomicU64,
    /// Total commits that were batched with others
    pub batched_commits: AtomicU64,
    /// Maximum batch size observed
    pub max_batch_size: AtomicU64,
}

impl GroupCommitStats {
    pub fn record_flush(&self, batch_size: usize) {
        self.total_commits
            .fetch_add(batch_size as u64, Ordering::Relaxed);
        self.total_flushes.fetch_add(1, Ordering::Relaxed);
        if batch_size > 1 {
            self.batched_commits
                .fetch_add(batch_size as u64 - 1, Ordering::Relaxed);
        }

        let mut current_max = self.max_batch_size.load(Ordering::Relaxed);
        while batch_size as u64 > current_max {
            match self.max_batch_size.compare_exchange_weak(
                current_max,
                batch_size as u64,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => break,
                Err(x) => current_max = x,
            }
        }
    }

    pub fn average_batch_size(&self) -> f64 {
        let commits = self.total_commits.load(Ordering::Relaxed);
        let flushes = self.total_flushes.load(Ordering::Relaxed);
        if flushes == 0 {
            0.0
        } else {
            commits as f64 / flushes as f64
        }
    }
}

/// Internal state for the commit queue
struct QueueState {
    /// Pending commits waiting to be flushed
    pending: VecDeque<std::sync::Arc<PendingCommit>>,
    /// Current batch ID counter
    next_batch_id: u64,
    /// Timestamp of first commit in current batch
    batch_start: Option<Instant>,
    /// Whether a flush is currently in progress
    flush_in_progress: bool,
}

impl QueueState {
    fn new() -> Self {
        Self {
            pending: VecDeque::with_capacity(64),
            next_batch_id: 1,
            batch_start: None,
            flush_in_progress: false,
        }
    }
}

/// The group commit queue that batches multiple commits
pub struct GroupCommitQueue {
    config: GroupCommitConfig,
    state: Mutex<QueueState>,
    /// Condition variable for waiting threads
    commit_ready: Condvar,
    /// Condition variable for flush completion
    flush_complete: Condvar,
    /// Statistics for monitoring
    pub stats: GroupCommitStats,
    /// Whether the queue is enabled (can be disabled for testing)
    enabled: AtomicBool,
}

impl GroupCommitQueue {
    pub fn new(config: GroupCommitConfig) -> Self {
        Self {
            config,
            state: Mutex::new(QueueState::new()),
            commit_ready: Condvar::new(),
            flush_complete: Condvar::new(),
            stats: GroupCommitStats::default(),
            enabled: AtomicBool::new(true),
        }
    }

    pub fn with_default_config() -> Self {
        Self::new(GroupCommitConfig::default())
    }

    /// Enable or disable group commit (useful for testing)
    pub fn set_enabled(&self, enabled: bool) {
        self.enabled.store(enabled, Ordering::Release);
    }

    pub fn is_enabled(&self) -> bool {
        self.enabled.load(Ordering::Acquire)
    }

    /// Submit a commit request and wait for it to complete
    ///
    /// Returns the batch ID this commit was part of, or an error message
    pub fn submit_and_wait(
        &self,
        payload: CommitPayload,
    ) -> Result<u64, String> {
        if !self.is_enabled() || payload.is_empty() {
            return Ok(0);
        }

        let pending = {
            let mut state = self.state.lock();
            let batch_id = state.next_batch_id;
            state.next_batch_id += 1;

            if state.batch_start.is_none() {
                state.batch_start = Some(Instant::now());
            }

            let pending = std::sync::Arc::new(PendingCommit::new(batch_id, payload));
            state.pending.push_back(std::sync::Arc::clone(&pending));

            let should_flush = state.pending.len() >= self.config.max_batch_size;
            if should_flush && !state.flush_in_progress {
                self.commit_ready.notify_one();
            }

            pending
        };

        self.wait_for_completion(&pending)?;

        Ok(pending.batch_id)
    }

    /// Submit a commit request without waiting (for async usage)
    pub fn submit_async(
        &self,
        payload: CommitPayload,
    ) -> std::sync::Arc<PendingCommit> {
        let mut state = self.state.lock();
        let batch_id = state.next_batch_id;
        state.next_batch_id += 1;

        if state.batch_start.is_none() {
            state.batch_start = Some(Instant::now());
        }

        let pending = std::sync::Arc::new(PendingCommit::new(batch_id, payload));
        state.pending.push_back(std::sync::Arc::clone(&pending));

        if state.pending.len() >= self.config.max_batch_size && !state.flush_in_progress {
            self.commit_ready.notify_one();
        }

        pending
    }

    fn wait_for_completion(&self, pending: &PendingCommit) -> Result<(), String> {
        // Use a generous timeout for the actual flush operation, as disk I/O can be slow
        // especially under load or with large batches. 10ms (old) was deemed too short.
        let timeout = Duration::from_secs(30);
        let start = Instant::now();

        while !pending.is_completed() {
            let mut state = self.state.lock();

            if pending.is_completed() {
                break;
            }

            let should_flush = !state.flush_in_progress && self.should_flush(&state);

            if should_flush {
                state.flush_in_progress = true;
                drop(state);
                return Ok(());
            } else {
                let remaining = timeout.saturating_sub(start.elapsed());
                if remaining.is_zero() {
                    return Err("group commit timeout".to_string());
                }
                self.flush_complete.wait_for(&mut state, remaining);
            }
        }

        if let Some(error) = pending.take_error() {
            Err(error)
        } else {
            Ok(())
        }
    }

    fn should_flush(&self, state: &QueueState) -> bool {
        if state.pending.is_empty() {
            return false;
        }

        if state.pending.len() >= self.config.max_batch_size {
            return true;
        }

        if let Some(batch_start) = state.batch_start {
            if batch_start.elapsed() >= Duration::from_micros(self.config.max_wait_us) {
                return true;
            }
        }

        state.pending.len() >= self.config.min_batch_size
    }

    /// Take all pending commits for flushing
    /// Returns None if no commits are pending or flush is already in progress
    pub fn take_pending(&self) -> Option<Vec<std::sync::Arc<PendingCommit>>> {
        let mut state = self.state.lock();

        if state.pending.is_empty() {
            return None;
        }

        state.flush_in_progress = true;
        let pending: Vec<_> = state.pending.drain(..).collect();
        state.batch_start = None;

        Some(pending)
    }

    /// Mark a batch of commits as completed
    pub fn complete_batch(&self, commits: &[std::sync::Arc<PendingCommit>]) {
        let batch_size = commits.len();

        for commit in commits {
            commit.mark_completed();
        }

        self.stats.record_flush(batch_size);

        {
            let mut state = self.state.lock();
            state.flush_in_progress = false;
        }
        self.flush_complete.notify_all();
    }

    /// Mark a batch of commits as failed
    pub fn fail_batch(&self, commits: &[std::sync::Arc<PendingCommit>], error: &str) {
        for commit in commits {
            commit.mark_failed(error.to_string());
        }

        {
            let mut state = self.state.lock();
            state.flush_in_progress = false;
        }
        self.flush_complete.notify_all();
    }

    /// Get the number of pending commits
    pub fn pending_count(&self) -> usize {
        self.state.lock().pending.len()
    }

    /// Force a flush of all pending commits (for testing or shutdown)
    pub fn force_flush(&self) -> Option<Vec<std::sync::Arc<PendingCommit>>> {
        self.take_pending()
    }
}

impl Default for GroupCommitQueue {
    fn default() -> Self {
        Self::with_default_config()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::PageBufferPool;
    use smallvec::smallvec;

    /// Helper to create a test payload with the given data byte
    fn test_payload(pool: &PageBufferPool, table_id: u32, page_no: u32, data_byte: u8, db_size: u32) -> CommitPayload {
        let mut buffer = pool.acquire_or_alloc();
        buffer.as_mut_slice()[0] = data_byte;
        smallvec![(table_id, page_no, buffer, db_size)]
    }

    #[test]
    fn test_group_commit_config_default() {
        let config = GroupCommitConfig::default();
        assert_eq!(config.max_batch_size, 64);
        assert_eq!(config.max_wait_us, 1000);
        assert_eq!(config.min_batch_size, 1);
    }

    #[test]
    fn test_pending_commit_lifecycle() {
        let pool = PageBufferPool::new(4);
        let payload = test_payload(&pool, 1, 100, 1, 10);
        let pending = PendingCommit::new(1, payload);
        assert!(!pending.is_completed());
        assert!(pending.take_error().is_none());

        pending.mark_completed();
        assert!(pending.is_completed());
    }

    #[test]
    fn test_pending_commit_failure() {
        let pending = PendingCommit::new(2, SmallVec::new());
        pending.mark_failed("test error".to_string());

        assert!(pending.is_completed());
        assert_eq!(pending.take_error(), Some("test error".to_string()));
    }

    #[test]
    fn test_stats_recording() {
        let stats = GroupCommitStats::default();

        stats.record_flush(1);
        assert_eq!(stats.total_commits.load(Ordering::Relaxed), 1);
        assert_eq!(stats.total_flushes.load(Ordering::Relaxed), 1);
        assert_eq!(stats.batched_commits.load(Ordering::Relaxed), 0);

        stats.record_flush(5);
        assert_eq!(stats.total_commits.load(Ordering::Relaxed), 6);
        assert_eq!(stats.total_flushes.load(Ordering::Relaxed), 2);
        assert_eq!(stats.batched_commits.load(Ordering::Relaxed), 4);
        assert_eq!(stats.max_batch_size.load(Ordering::Relaxed), 5);
    }

    #[test]
    fn test_average_batch_size() {
        let stats = GroupCommitStats::default();
        assert_eq!(stats.average_batch_size(), 0.0);

        stats.record_flush(4);
        stats.record_flush(6);
        assert_eq!(stats.average_batch_size(), 5.0);
    }

    #[test]
    fn test_queue_submit_async() {
        let pool = PageBufferPool::new(4);
        let queue = GroupCommitQueue::with_default_config();

        let pending = queue.submit_async(test_payload(&pool, 1, 0, 1, 10));
        assert_eq!(pending.batch_id, 1);
        assert_eq!(queue.pending_count(), 1);

        let pending2 = queue.submit_async(test_payload(&pool, 1, 0, 2, 10));
        assert_eq!(pending2.batch_id, 2);
        assert_eq!(queue.pending_count(), 2);
    }

    #[test]
    fn test_queue_take_pending() {
        let pool = PageBufferPool::new(4);
        let queue = GroupCommitQueue::with_default_config();

        queue.submit_async(test_payload(&pool, 1, 0, 1, 10));
        queue.submit_async(test_payload(&pool, 1, 0, 2, 10));

        let pending = queue.take_pending().unwrap();
        assert_eq!(pending.len(), 2);
        assert_eq!(queue.pending_count(), 0);
    }

    #[test]
    fn test_queue_complete_batch() {
        let pool = PageBufferPool::new(4);
        let queue = GroupCommitQueue::with_default_config();

        let p1 = queue.submit_async(test_payload(&pool, 1, 0, 1, 10));
        let p2 = queue.submit_async(test_payload(&pool, 1, 0, 2, 10));

        let pending = queue.take_pending().unwrap();
        queue.complete_batch(&pending);

        assert!(p1.is_completed());
        assert!(p2.is_completed());
        assert_eq!(queue.stats.total_flushes.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn test_queue_disabled() {
        let pool = PageBufferPool::new(4);
        let queue = GroupCommitQueue::with_default_config();
        queue.set_enabled(false);

        let result = queue.submit_and_wait(test_payload(&pool, 1, 0, 1, 10));
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), 0);
        assert_eq!(queue.pending_count(), 0);
    }

    #[test]
    fn test_queue_empty_payload() {
        let queue = GroupCommitQueue::with_default_config();

        let result = queue.submit_and_wait(SmallVec::new());
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), 0);
    }
}
