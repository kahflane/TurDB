//! # Memory Budget Implementation
//!
//! This module implements the core memory budget tracking and enforcement.
//!
//! ## Design Principles
//!
//! 1. **Hard Limits**: Allocations that would exceed the budget fail immediately
//! 2. **Reserved Pools**: Each subsystem has a guaranteed minimum allocation
//! 3. **Shared Overflow**: When reserved is exhausted, shared pool is used
//! 4. **Thread Safety**: All counters use atomics for lock-free operation
//!
//! ## Pool Allocation Strategy
//!
//! When a subsystem requests memory:
//! 1. Check if request fits in subsystem's reserved pool
//! 2. If not, check if request fits in shared pool
//! 3. If neither, return OutOfMemory error
//!
//! ## Memory Accounting
//!
//! Tracked memory includes:
//! - Page cache entries (16KB per page)
//! - Query execution buffers (hash tables, sort buffers)
//! - WAL recovery buffers
//! - Schema/catalog metadata
//!
//! Untracked (small/fixed):
//! - Stack allocations
//! - Small temporary buffers (< 1KB)
//! - Static data structures
//!
//! ## Query Memory Estimates
//!
//! Query operators use conservative estimates for memory tracking since exact
//! allocation sizes would require traversing all stored values. The estimates
//! are designed to slightly over-count rather than under-count to prevent OOM:
//!
//! | Context | Estimate | Rationale |
//! |---------|----------|-----------|
//! | Row materialization | 128 bytes/row | Covers typical row with 8-10 columns |
//! | Hash aggregate entry | 256 bytes/entry | Includes key, group values, and aggregate states |
//! | Hash table bucket | 24 bytes/entry | Hash key (8) + indices vec overhead (16) |
//!
//! These estimates may significantly over-count for tables with few small columns
//! or under-count for tables with many large text/blob columns. The periodic sync
//! pattern (every 64KB) amortizes the overhead of atomic operations.
//!
//! Actual memory usage may differ from tracked amounts. The goal is preventing
//! runaway queries from exhausting system memory, not precise accounting.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::OnceLock;

use eyre::{bail, Result};
use sysinfo::System;

pub use crate::config::{
    CACHE_RESERVED, DEFAULT_BUDGET_PERCENT, MIN_BUDGET_FLOOR, QUERY_RESERVED, RECOVERY_RESERVED,
    SCHEMA_RESERVED, TOTAL_RESERVED,
};

static SYSTEM_TOTAL_MEMORY: OnceLock<usize> = OnceLock::new();

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Pool {
    Cache,
    Query,
    Recovery,
    Schema,
    Shared,
}

impl Pool {
    pub fn reserved_size(&self) -> usize {
        match self {
            Pool::Cache => CACHE_RESERVED,
            Pool::Query => QUERY_RESERVED,
            Pool::Recovery => RECOVERY_RESERVED,
            Pool::Schema => SCHEMA_RESERVED,
            Pool::Shared => 0,
        }
    }

    pub fn name(&self) -> &'static str {
        match self {
            Pool::Cache => "cache",
            Pool::Query => "query",
            Pool::Recovery => "recovery",
            Pool::Schema => "schema",
            Pool::Shared => "shared",
        }
    }
}

#[derive(Debug, Clone)]
pub struct BudgetStats {
    pub total_limit: usize,
    pub total_used: usize,
    pub cache_used: usize,
    pub cache_reserved: usize,
    pub query_used: usize,
    pub query_reserved: usize,
    pub recovery_used: usize,
    pub recovery_reserved: usize,
    pub schema_used: usize,
    pub schema_reserved: usize,
    pub shared_used: usize,
    pub shared_available: usize,
}

impl BudgetStats {
    pub fn available(&self) -> usize {
        self.total_limit.saturating_sub(self.total_used)
    }

    pub fn utilization_percent(&self) -> f64 {
        if self.total_limit == 0 {
            return 0.0;
        }
        (self.total_used as f64 / self.total_limit as f64) * 100.0
    }
}

impl std::fmt::Display for BudgetStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "cache:{}/{},query:{}/{},recovery:{}/{},schema:{}/{},shared:{}/{}",
            self.cache_used,
            self.cache_reserved,
            self.query_used,
            self.query_reserved,
            self.recovery_used,
            self.recovery_reserved,
            self.schema_used,
            self.schema_reserved,
            self.shared_used,
            self.shared_available
        )
    }
}

#[derive(Debug)]
pub struct MemoryError {
    pub pool: Pool,
    pub requested: usize,
    pub available: usize,
}

impl std::fmt::Display for MemoryError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "memory budget exceeded: {} pool requested {} bytes but only {} available",
            self.pool.name(),
            self.requested,
            self.available
        )
    }
}

impl std::error::Error for MemoryError {}

#[derive(Debug)]
pub struct MemoryBudget {
    total_limit: AtomicUsize,
    cache_used: AtomicUsize,
    query_used: AtomicUsize,
    recovery_used: AtomicUsize,
    schema_used: AtomicUsize,
    shared_used: AtomicUsize,
}

impl MemoryBudget {
    pub fn auto_detect() -> Self {
        let total_memory = *SYSTEM_TOTAL_MEMORY.get_or_init(|| {
            let mut sys = System::new();
            sys.refresh_memory();
            sys.total_memory() as usize
        });

        let budget = (total_memory * DEFAULT_BUDGET_PERCENT) / 100;
        let budget = budget.max(MIN_BUDGET_FLOOR);

        Self::with_limit(budget)
    }

    pub fn with_limit(limit: usize) -> Self {
        let limit = limit.max(MIN_BUDGET_FLOOR);

        Self {
            total_limit: AtomicUsize::new(limit),
            cache_used: AtomicUsize::new(0),
            query_used: AtomicUsize::new(0),
            recovery_used: AtomicUsize::new(0),
            schema_used: AtomicUsize::new(0),
            shared_used: AtomicUsize::new(0),
        }
    }

    pub fn total_limit(&self) -> usize {
        self.total_limit.load(Ordering::Acquire)
    }

    pub fn total_used(&self) -> usize {
        self.cache_used.load(Ordering::Acquire)
            + self.query_used.load(Ordering::Acquire)
            + self.recovery_used.load(Ordering::Acquire)
            + self.schema_used.load(Ordering::Acquire)
            + self.shared_used.load(Ordering::Acquire)
    }

    pub fn available(&self, pool: Pool) -> usize {
        let pool_used = self.pool_used(pool);
        let reserved = pool.reserved_size();

        let reserved_available = reserved.saturating_sub(pool_used);
        let shared_available = self.shared_available();

        reserved_available + shared_available
    }

    pub fn shared_available(&self) -> usize {
        let total = self.total_limit();
        let used = self.total_used();
        let shared_pool_size = total.saturating_sub(TOTAL_RESERVED);

        shared_pool_size.saturating_sub(used.saturating_sub(TOTAL_RESERVED).max(0))
    }

    fn pool_used(&self, pool: Pool) -> usize {
        match pool {
            Pool::Cache => self.cache_used.load(Ordering::Acquire),
            Pool::Query => self.query_used.load(Ordering::Acquire),
            Pool::Recovery => self.recovery_used.load(Ordering::Acquire),
            Pool::Schema => self.schema_used.load(Ordering::Acquire),
            Pool::Shared => self.shared_used.load(Ordering::Acquire),
        }
    }

    fn pool_counter(&self, pool: Pool) -> &AtomicUsize {
        match pool {
            Pool::Cache => &self.cache_used,
            Pool::Query => &self.query_used,
            Pool::Recovery => &self.recovery_used,
            Pool::Schema => &self.schema_used,
            Pool::Shared => &self.shared_used,
        }
    }

    pub fn can_allocate(&self, pool: Pool, bytes: usize) -> bool {
        self.available(pool) >= bytes
    }

    pub fn allocate(&self, pool: Pool, bytes: usize) -> Result<()> {
        if bytes == 0 {
            return Ok(());
        }

        let pool_counter = self.pool_counter(pool);
        let reserved = pool.reserved_size();

        loop {
            let current_pool_used = pool_counter.load(Ordering::Acquire);
            let current_total_used = self.total_used();
            let total_limit = self.total_limit();

            let new_pool_used = current_pool_used + bytes;
            let new_total_used = current_total_used + bytes;

            if new_total_used > total_limit {
                bail!(MemoryError {
                    pool,
                    requested: bytes,
                    available: total_limit.saturating_sub(current_total_used),
                });
            }

            if pool != Pool::Shared && new_pool_used > reserved {
                let overflow = new_pool_used - reserved;
                let shared_available = self.shared_available();

                if overflow > shared_available {
                    bail!(MemoryError {
                        pool,
                        requested: bytes,
                        available: reserved.saturating_sub(current_pool_used) + shared_available,
                    });
                }
            }

            match pool_counter.compare_exchange_weak(
                current_pool_used,
                new_pool_used,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => return Ok(()),
                Err(_) => continue,
            }
        }
    }

    pub fn release(&self, pool: Pool, bytes: usize) {
        if bytes == 0 {
            return;
        }

        let pool_counter = self.pool_counter(pool);

        loop {
            let current = pool_counter.load(Ordering::Acquire);
            let new_value = current.saturating_sub(bytes);

            match pool_counter.compare_exchange_weak(
                current,
                new_value,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => return,
                Err(_) => continue,
            }
        }
    }

    pub fn try_allocate(&self, pool: Pool, bytes: usize) -> bool {
        self.allocate(pool, bytes).is_ok()
    }

    pub fn stats(&self) -> BudgetStats {
        let total_limit = self.total_limit();
        let cache_used = self.cache_used.load(Ordering::Acquire);
        let query_used = self.query_used.load(Ordering::Acquire);
        let recovery_used = self.recovery_used.load(Ordering::Acquire);
        let schema_used = self.schema_used.load(Ordering::Acquire);
        let shared_used = self.shared_used.load(Ordering::Acquire);
        let total_used = cache_used + query_used + recovery_used + schema_used + shared_used;

        BudgetStats {
            total_limit,
            total_used,
            cache_used,
            cache_reserved: CACHE_RESERVED,
            query_used,
            query_reserved: QUERY_RESERVED,
            recovery_used,
            recovery_reserved: RECOVERY_RESERVED,
            schema_used,
            schema_reserved: SCHEMA_RESERVED,
            shared_used,
            shared_available: self.shared_available(),
        }
    }

    pub fn reset(&self) {
        self.cache_used.store(0, Ordering::Release);
        self.query_used.store(0, Ordering::Release);
        self.recovery_used.store(0, Ordering::Release);
        self.schema_used.store(0, Ordering::Release);
        self.shared_used.store(0, Ordering::Release);
    }
}

impl Default for MemoryBudget {
    fn default() -> Self {
        Self::auto_detect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_auto_detect_respects_floor() {
        let budget = MemoryBudget::auto_detect();
        assert!(budget.total_limit() >= MIN_BUDGET_FLOOR);
    }

    #[test]
    fn test_with_limit_respects_floor() {
        let budget = MemoryBudget::with_limit(1000);
        assert_eq!(budget.total_limit(), MIN_BUDGET_FLOOR);
    }

    #[test]
    fn test_allocate_within_reserved() {
        let budget = MemoryBudget::with_limit(MIN_BUDGET_FLOOR);
        assert!(budget.allocate(Pool::Cache, 256 * 1024).is_ok());
        assert_eq!(budget.pool_used(Pool::Cache), 256 * 1024);
    }

    #[test]
    fn test_allocate_exceeds_total_budget() {
        let budget = MemoryBudget::with_limit(MIN_BUDGET_FLOOR);
        assert!(budget.allocate(Pool::Cache, MIN_BUDGET_FLOOR + 1).is_err());
    }

    #[test]
    fn test_release_memory() {
        let budget = MemoryBudget::with_limit(MIN_BUDGET_FLOOR);
        budget.allocate(Pool::Cache, 256 * 1024).unwrap();
        budget.release(Pool::Cache, 128 * 1024);
        assert_eq!(budget.pool_used(Pool::Cache), 128 * 1024);
    }

    #[test]
    fn test_release_underflow_protection() {
        let budget = MemoryBudget::with_limit(MIN_BUDGET_FLOOR);
        budget.release(Pool::Cache, 1000);
        assert_eq!(budget.pool_used(Pool::Cache), 0);
    }

    #[test]
    fn test_can_allocate() {
        let budget = MemoryBudget::with_limit(MIN_BUDGET_FLOOR);
        assert!(budget.can_allocate(Pool::Cache, 256 * 1024));
        assert!(!budget.can_allocate(Pool::Cache, MIN_BUDGET_FLOOR + 1));
    }

    #[test]
    fn test_stats_accuracy() {
        let budget = MemoryBudget::with_limit(MIN_BUDGET_FLOOR);
        budget.allocate(Pool::Cache, 100_000).unwrap();
        budget.allocate(Pool::Query, 50_000).unwrap();

        let stats = budget.stats();
        assert_eq!(stats.cache_used, 100_000);
        assert_eq!(stats.query_used, 50_000);
        assert_eq!(stats.total_used, 150_000);
    }

    #[test]
    fn test_multiple_pools_independent() {
        let budget = MemoryBudget::with_limit(MIN_BUDGET_FLOOR);
        budget.allocate(Pool::Cache, 256 * 1024).unwrap();
        budget.allocate(Pool::Query, 128 * 1024).unwrap();
        budget.allocate(Pool::Recovery, 64 * 1024).unwrap();

        assert_eq!(budget.pool_used(Pool::Cache), 256 * 1024);
        assert_eq!(budget.pool_used(Pool::Query), 128 * 1024);
        assert_eq!(budget.pool_used(Pool::Recovery), 64 * 1024);
    }

    #[test]
    fn test_shared_pool_overflow() {
        let budget = MemoryBudget::with_limit(MIN_BUDGET_FLOOR);

        budget.allocate(Pool::Cache, CACHE_RESERVED).unwrap();
        assert!(budget.allocate(Pool::Cache, 100_000).is_ok());
    }

    #[test]
    fn test_zero_allocation() {
        let budget = MemoryBudget::with_limit(MIN_BUDGET_FLOOR);
        assert!(budget.allocate(Pool::Cache, 0).is_ok());
        assert_eq!(budget.pool_used(Pool::Cache), 0);
    }

    #[test]
    fn test_reset() {
        let budget = MemoryBudget::with_limit(MIN_BUDGET_FLOOR);
        budget.allocate(Pool::Cache, 100_000).unwrap();
        budget.allocate(Pool::Query, 50_000).unwrap();

        budget.reset();

        assert_eq!(budget.pool_used(Pool::Cache), 0);
        assert_eq!(budget.pool_used(Pool::Query), 0);
        assert_eq!(budget.total_used(), 0);
    }

    #[test]
    fn test_stats_display() {
        let budget = MemoryBudget::with_limit(MIN_BUDGET_FLOOR);
        budget.allocate(Pool::Cache, 100).unwrap();

        let stats = budget.stats();
        let display = stats.to_string();

        assert!(display.contains("cache:100/"));
    }
}
