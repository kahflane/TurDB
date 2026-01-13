//! # Memory Budget Management
//!
//! This module provides memory budget tracking and enforcement for TurDB.
//! It ensures the database operates within configurable memory limits,
//! making it suitable for embedded devices, mobile, and IoT applications.
//!
//! ## Architecture
//!
//! The memory budget system uses a **reserved minimums + shared pool** model:
//!
//! ```text
//! +----------------------------------------------------------+
//! |                  Total Memory Budget                      |
//! |  (default: 25% of system RAM, minimum floor: 4 MB)       |
//! +----------------------------------------------------------+
//! |                                                          |
//! |  Reserved Pools (guaranteed minimums):                   |
//! |  +----------+ +----------+ +----------+ +----------+     |
//! |  | Cache    | | Query    | | Recovery | | Schema   |     |
//! |  | 512 KB   | | 256 KB   | | 256 KB   | | 128 KB   |     |
//! |  +----------+ +----------+ +----------+ +----------+     |
//! |                                                          |
//! |  Shared Pool (remainder):                                |
//! |  +----------------------------------------------------+  |
//! |  | Available for any subsystem when reserved exceeded |  |
//! |  +----------------------------------------------------+  |
//! |                                                          |
//! +----------------------------------------------------------+
//! ```
//!
//! ## Enforcement Model
//!
//! This implementation uses **hard limits** - operations that would exceed
//! the budget are refused with an error. This is the safest approach for
//! resource-constrained devices where memory exhaustion could be fatal.
//!
//! ## Subsystem Integration
//!
//! Each subsystem (page cache, query executor, recovery, schema) requests
//! memory from its designated pool. If the reserved pool is exhausted,
//! allocation falls back to the shared pool. If both are exhausted, the
//! allocation fails.
//!
//! ## Safety Net
//!
//! The `cap` crate is used as a global allocator wrapper to provide a
//! hard backstop. Even if subsystem accounting drifts, the `cap` allocator
//! will prevent actual memory exhaustion.
//!
//! ## Configuration
//!
//! ```rust,ignore
//! // Auto-detect (25% of system RAM, 4MB floor)
//! let budget = MemoryBudget::auto_detect();
//!
//! // Explicit limit
//! let budget = MemoryBudget::with_limit(16 * 1024 * 1024); // 16 MB
//!
//! // Via database builder
//! let db = Database::builder()
//!     .memory_budget(16 * 1024 * 1024)
//!     .open()?;
//! ```
//!
//! ## PRAGMA Commands
//!
//! - `PRAGMA memory_budget` - Query total budget in bytes
//! - `PRAGMA memory_stats` - Query per-pool usage
//! - `PRAGMA database_mode` - Check if in degraded mode
//! - `PRAGMA recover_wal` - Trigger streaming recovery (degraded mode)
//! - `PRAGMA wal_checkpoint` - Force checkpoint now
//! - `PRAGMA wal_checkpoint_threshold` - Get/set auto-checkpoint frame count

mod budget;
mod page_buffer;

pub use budget::{BudgetStats, MemoryBudget, MemoryError, Pool};
pub use page_buffer::{FallbackBuffer, PageBufferPool, PooledPageBuffer};
