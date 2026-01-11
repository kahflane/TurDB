//! # System Tables Module
//!
//! This module defines the system tables that live in the `turdb_catalog` schema.
//! These tables store internal metadata and statistics that are accessible via SQL.
//!
//! ## System Schema
//!
//! The `turdb_catalog` schema is automatically created when a database is opened
//! and cannot be dropped by users. It contains:
//!
//! - `memory_stats`: Current memory pool usage statistics
//! - `wal_stats`: WAL status and checkpoint information
//!
//! ## Cross-Session Visibility
//!
//! Unlike in-memory counters, data in system tables persists across database
//! sessions. This allows monitoring tools to query stats from any connection:
//!
//! ```sql
//! SELECT * FROM turdb_catalog.memory_stats;
//! SELECT * FROM turdb_catalog.wal_stats;
//! ```
//!
//! ## Update Frequency
//!
//! System tables are updated:
//! - On checkpoint operations
//! - On explicit PRAGMA calls
//! - Optionally on commit (configurable)
//!
//! ## Thread Safety
//!
//! System table updates use the same transaction semantics as user tables.
//! Concurrent reads are always safe; writes are serialized via table locks.

use super::{ColumnDef, Constraint, TableDef};
use crate::records::types::DataType;

/// System schema name - this schema is created automatically and cannot be dropped.
pub const SYSTEM_SCHEMA: &str = "turdb_catalog";

/// Memory statistics table - stores per-pool memory usage.
pub const MEMORY_STATS_TABLE: &str = "memory_stats";

/// WAL statistics table - stores WAL frame count and size.
pub const WAL_STATS_TABLE: &str = "wal_stats";

/// Creates the column definitions for the memory_stats table.
///
/// Schema:
/// ```sql
/// CREATE TABLE turdb_catalog.memory_stats (
///     stat_name TEXT PRIMARY KEY,
///     stat_value BIGINT,
///     updated_at TEXT
/// );
/// ```
pub fn memory_stats_columns() -> Vec<ColumnDef> {
    vec![
        ColumnDef::new("stat_name", DataType::Text)
            .with_constraint(Constraint::PrimaryKey)
            .with_constraint(Constraint::NotNull),
        ColumnDef::new("stat_value", DataType::Int8),
        ColumnDef::new("updated_at", DataType::Text),
    ]
}

/// Creates the column definitions for the wal_stats table.
///
/// Schema:
/// ```sql
/// CREATE TABLE turdb_catalog.wal_stats (
///     stat_name TEXT PRIMARY KEY,
///     stat_value BIGINT,
///     updated_at TEXT
/// );
/// ```
pub fn wal_stats_columns() -> Vec<ColumnDef> {
    vec![
        ColumnDef::new("stat_name", DataType::Text)
            .with_constraint(Constraint::PrimaryKey)
            .with_constraint(Constraint::NotNull),
        ColumnDef::new("stat_value", DataType::Int8),
        ColumnDef::new("updated_at", DataType::Text),
    ]
}

pub mod memory_stat_names {
    pub const BUDGET_TOTAL: &str = "memory_budget_total";
    pub const USED_CACHE: &str = "memory_used_cache";
    pub const USED_QUERY: &str = "memory_used_query";
    pub const USED_RECOVERY: &str = "memory_used_recovery";
    pub const USED_SCHEMA: &str = "memory_used_schema";
    pub const USED_SHARED: &str = "memory_used_shared";
    pub const USED_TOTAL: &str = "memory_used_total";
    pub const AVAILABLE_SHARED: &str = "memory_available_shared";
}

pub mod wal_stat_names {
    pub const FRAME_COUNT: &str = "wal_frame_count";
    pub const SIZE_BYTES: &str = "wal_size_bytes";
    pub const CHECKPOINT_THRESHOLD: &str = "wal_checkpoint_threshold";
    pub const LAST_CHECKPOINT_FRAMES: &str = "wal_last_checkpoint_frames";
}

pub fn create_memory_stats_table_def(table_id: u64) -> TableDef {
    TableDef::new(table_id, MEMORY_STATS_TABLE, memory_stats_columns())
}

pub fn create_wal_stats_table_def(table_id: u64) -> TableDef {
    TableDef::new(table_id, WAL_STATS_TABLE, wal_stats_columns())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memory_stats_columns() {
        let cols = memory_stats_columns();
        assert_eq!(cols.len(), 3);
        assert_eq!(cols[0].name(), "stat_name");
        assert_eq!(cols[1].name(), "stat_value");
        assert_eq!(cols[2].name(), "updated_at");
    }

    #[test]
    fn test_wal_stats_columns() {
        let cols = wal_stats_columns();
        assert_eq!(cols.len(), 3);
        assert_eq!(cols[0].name(), "stat_name");
        assert_eq!(cols[1].name(), "stat_value");
        assert_eq!(cols[2].name(), "updated_at");
    }

    #[test]
    fn test_system_schema_constant() {
        assert_eq!(SYSTEM_SCHEMA, "turdb_catalog");
    }
}
