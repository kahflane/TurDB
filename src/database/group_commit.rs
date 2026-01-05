//! # Group Commit Implementation
//!
//! This module implements group commit for TurDB, a critical optimization that
//! batches multiple transaction commits into a single WAL flush operation.
//! This dramatically reduces I/O overhead, especially on IoT devices where
//! storage write latency is high.
//!
//! ## Design Overview
//!
//! ```text
//! Thread 1 ──┐
//! Thread 2 ──┼──► GroupCommitQueue ──► Single WAL Flush ──► Notify All
//! Thread 3 ──┘
//! ```
//!
//! ## How It Works
//!
//! 1. Multiple threads submit commit requests to the queue
//! 2. A configurable timeout or batch size triggers the flush
//! 3. All pending commits are flushed to WAL in a single I/O operation
//! 4. All waiting threads are notified of completion
//!
//! ## Performance Benefits
//!
//! - Reduces fsync() calls from N to 1 for N concurrent commits
//! - Amortizes WAL write overhead across multiple transactions
//! - Typical speedup: 5-10x for write-heavy workloads
//!
//! ## Thread Safety
//!
//! `GroupCommitQueue` is `Send + Sync`. Internal synchronization uses
//! `parking_lot::Mutex` and `Condvar` for efficient waiting.

use parking_lot::{Condvar, Mutex};
use smallvec::SmallVec;
use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::{Duration, Instant};

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
            max_batch_size: 64,      // Match transaction slot limit
            max_wait_us: 1000,       // 1ms max latency
            min_batch_size: 1,       // Flush immediately if at least 1 commit
        }
    }
}

impl GroupCommitConfig {
    /// Configuration optimized for high throughput (more batching)
    pub fn high_throughput() -> Self {
        Self {
            max_batch_size: 128,
            max_wait_us: 5000,   // 5ms
            min_batch_size: 4,
        }
    }

    /// Configuration optimized for low latency (less batching)
    pub fn low_latency() -> Self {
        Self {
            max_batch_size: 16,
            max_wait_us: 100,   // 100us
            min_batch_size: 1,
        }
    }
}

/// A pending commit request waiting to be flushed
#[derive(Debug)]
pub struct PendingCommit {
    /// Unique batch ID for this commit
    pub batch_id: u64,
    /// Table IDs with dirty pages to flush
    pub dirty_table_ids: SmallVec<[u32; 8]>,
    /// Completion flag - set to true when WAL flush completes
    pub completed: AtomicBool,
    /// Error message if commit failed (empty string means success)
    pub error: Mutex<Option<String>>,
}

impl PendingCommit {
    pub fn new(batch_id: u64, dirty_table_ids: SmallVec<[u32; 8]>) -> Self {
        Self {
            batch_id,
            dirty_table_ids,
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
        self.total_commits.fetch_add(batch_size as u64, Ordering::Relaxed);
        self.total_flushes.fetch_add(1, Ordering::Relaxed);
        if batch_size > 1 {
            self.batched_commits.fetch_add(batch_size as u64 - 1, Ordering::Relaxed);
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
    pub fn submit_and_wait(&self, dirty_table_ids: SmallVec<[u32; 8]>) -> Result<u64, String> {
        if !self.is_enabled() || dirty_table_ids.is_empty() {
            // Bypass batching if disabled or no dirty pages
            return Ok(0);
        }

        let pending = {
            let mut state = self.state.lock();
            let batch_id = state.next_batch_id;
            state.next_batch_id += 1;

            if state.batch_start.is_none() {
                state.batch_start = Some(Instant::now());
            }

            let pending = std::sync::Arc::new(PendingCommit::new(batch_id, dirty_table_ids));
            state.pending.push_back(std::sync::Arc::clone(&pending));

            // Check if we should trigger a flush
            let should_flush = state.pending.len() >= self.config.max_batch_size;

            if should_flush && !state.flush_in_progress {
                // Signal that a flush should happen
                self.commit_ready.notify_one();
            }

            pending
        };

        // Wait for completion
        self.wait_for_completion(&pending)?;

        Ok(pending.batch_id)
    }

    /// Submit a commit request without waiting (for async usage)
    pub fn submit_async(&self, dirty_table_ids: SmallVec<[u32; 8]>) -> std::sync::Arc<PendingCommit> {
        let mut state = self.state.lock();
        let batch_id = state.next_batch_id;
        state.next_batch_id += 1;

        if state.batch_start.is_none() {
            state.batch_start = Some(Instant::now());
        }

        let pending = std::sync::Arc::new(PendingCommit::new(batch_id, dirty_table_ids));
        state.pending.push_back(std::sync::Arc::clone(&pending));

        // Check if we should trigger a flush
        if state.pending.len() >= self.config.max_batch_size && !state.flush_in_progress {
            self.commit_ready.notify_one();
        }

        pending
    }

    fn wait_for_completion(&self, pending: &PendingCommit) -> Result<(), String> {
        let timeout = Duration::from_micros(self.config.max_wait_us * 10); // 10x max wait as absolute timeout
        let start = Instant::now();

        while !pending.is_completed() {
            let mut state = self.state.lock();

            // Double-check completion with lock held
            if pending.is_completed() {
                break;
            }

            // Check if we should initiate a flush
            let should_flush = !state.flush_in_progress && self.should_flush(&state);

            if should_flush {
                state.flush_in_progress = true;
                drop(state);
                // Signal that flush should happen (handled by flush_pending)
                self.commit_ready.notify_all();
            } else {
                // Wait for flush to complete
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

        // Flush if batch size reached
        if state.pending.len() >= self.config.max_batch_size {
            return true;
        }

        // Flush if timeout reached
        if let Some(batch_start) = state.batch_start {
            if batch_start.elapsed() >= Duration::from_micros(self.config.max_wait_us) {
                return true;
            }
        }

        // Flush if we have minimum batch size
        state.pending.len() >= self.config.min_batch_size
    }

    /// Take all pending commits for flushing
    /// Returns None if no commits are pending or flush is already in progress
    pub fn take_pending(&self) -> Option<Vec<std::sync::Arc<PendingCommit>>> {
        let mut state = self.state.lock();

        if state.pending.is_empty() {
            return None;
        }

        // Mark flush as in progress
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

        // Update stats
        self.stats.record_flush(batch_size);

        // Clear flush_in_progress and notify waiters
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

        // Clear flush_in_progress and notify waiters
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

/// Collect unique dirty table IDs from a batch of pending commits
pub fn collect_dirty_tables(commits: &[std::sync::Arc<PendingCommit>]) -> SmallVec<[u32; 16]> {
    use hashbrown::HashSet;

    let mut seen: HashSet<u32> = HashSet::with_capacity(commits.len() * 2);
    let mut result: SmallVec<[u32; 16]> = SmallVec::new();

    for commit in commits {
        for &table_id in commit.dirty_table_ids.iter() {
            if seen.insert(table_id) {
                result.push(table_id);
            }
        }
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_group_commit_config_default() {
        let config = GroupCommitConfig::default();
        assert_eq!(config.max_batch_size, 64);
        assert_eq!(config.max_wait_us, 1000);
        assert_eq!(config.min_batch_size, 1);
    }

    #[test]
    fn test_pending_commit_lifecycle() {
        let pending = PendingCommit::new(1, SmallVec::from_slice(&[1, 2, 3]));
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
        let queue = GroupCommitQueue::with_default_config();

        let pending = queue.submit_async(SmallVec::from_slice(&[1]));
        assert_eq!(pending.batch_id, 1);
        assert_eq!(queue.pending_count(), 1);

        let pending2 = queue.submit_async(SmallVec::from_slice(&[2]));
        assert_eq!(pending2.batch_id, 2);
        assert_eq!(queue.pending_count(), 2);
    }

    #[test]
    fn test_queue_take_pending() {
        let queue = GroupCommitQueue::with_default_config();

        queue.submit_async(SmallVec::from_slice(&[1]));
        queue.submit_async(SmallVec::from_slice(&[2]));

        let pending = queue.take_pending().unwrap();
        assert_eq!(pending.len(), 2);
        assert_eq!(queue.pending_count(), 0);
    }

    #[test]
    fn test_queue_complete_batch() {
        let queue = GroupCommitQueue::with_default_config();

        let p1 = queue.submit_async(SmallVec::from_slice(&[1]));
        let p2 = queue.submit_async(SmallVec::from_slice(&[2]));

        let pending = queue.take_pending().unwrap();
        queue.complete_batch(&pending);

        assert!(p1.is_completed());
        assert!(p2.is_completed());
        assert_eq!(queue.stats.total_flushes.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn test_collect_dirty_tables() {
        let commits = vec![
            std::sync::Arc::new(PendingCommit::new(1, SmallVec::from_slice(&[1, 2]))),
            std::sync::Arc::new(PendingCommit::new(2, SmallVec::from_slice(&[2, 3]))),
            std::sync::Arc::new(PendingCommit::new(3, SmallVec::from_slice(&[1, 4]))),
        ];

        let dirty = collect_dirty_tables(&commits);
        assert_eq!(dirty.len(), 4); // 1, 2, 3, 4 (deduplicated)
    }

    #[test]
    fn test_queue_disabled() {
        let queue = GroupCommitQueue::with_default_config();
        queue.set_enabled(false);

        let result = queue.submit_and_wait(SmallVec::from_slice(&[1]));
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), 0); // Bypassed
        assert_eq!(queue.pending_count(), 0);
    }

    #[test]
    fn test_queue_empty_dirty_tables() {
        let queue = GroupCommitQueue::with_default_config();

        let result = queue.submit_and_wait(SmallVec::new());
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), 0); // Bypassed
    }
}
