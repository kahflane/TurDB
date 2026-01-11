//! # Database Configuration and Statistics
//!
//! This module implements configuration and statistics functions for TurDB.
//! It provides runtime configuration through SET statements, memory statistics
//! tracking, WAL metrics, and database mode queries.
//!
//! ## Categories
//!
//! ### Statement Execution
//! - `execute_explain` - Generate and display query execution plans
//! - `execute_set` - Handle SET statements for runtime configuration
//!
//! ### Memory Statistics
//! - `join_memory_budget` - Query join operator memory limit
//! - `memory_budget` - Get memory budget Arc reference
//! - `memory_budget_ref` - Get memory budget reference
//! - `memory_stats` - Query per-pool memory usage
//! - `persist_memory_stats` - Write memory stats to system table
//!
//! ### WAL Statistics
//! - `wal_frame_count` - Query current WAL frame count
//! - `wal_size_bytes` - Query total WAL size
//! - `persist_wal_stats` - Write WAL stats to system table
//!
//! ### Database Mode
//! - `mode` - Query current operating mode
//! - `is_read_write` - Check if database is writable
//! - `is_degraded` - Check if in degraded read-only mode
//! - `check_writable` - Guard function for write operations
//!
//! ## Usage
//!
//! ```sql
//! EXPLAIN SELECT * FROM users;           -- Show query plan
//! EXPLAIN VERBOSE SELECT * FROM users;   -- Show detailed plan with indexes
//! SET foreign_keys = ON;                 -- Enable foreign key checks
//! ```

use crate::database::{Database, ExecuteResult};
use crate::memory::MemoryBudget;
use crate::sql::planner::Planner;
use bumpalo::Bump;
use eyre::{bail, Result, WrapErr};
use std::sync::Arc;

impl Database {
    pub(crate) fn execute_explain(
        &self,
        explain: &crate::sql::ast::ExplainStmt<'_>,
        arena: &Bump,
    ) -> Result<ExecuteResult> {
        self.ensure_catalog()?;

        let catalog_guard = self.shared.catalog.read();
        let catalog = catalog_guard.as_ref().unwrap();
        let planner = Planner::new(catalog, arena);
        let physical_plan = planner
            .create_physical_plan(explain.statement)
            .wrap_err("failed to create query plan for EXPLAIN")?;

        let mut plan_text = physical_plan.explain();

        if explain.verbose {
            if let crate::sql::ast::Statement::Select(select) = explain.statement {
                if let Some(from_clause) = select.from {
                    let (table_schema, table_name) = match from_clause {
                        crate::sql::ast::FromClause::Table(table_ref) => {
                            (table_ref.schema, Some(table_ref.name))
                        }
                        _ => (None, None),
                    };
                    if let Some(table_name) = table_name {
                        if let Ok(table_def) =
                            catalog.resolve_table_in_schema(table_schema, table_name)
                        {
                            plan_text.push_str("\nTable Info:\n");
                            plan_text.push_str(&format!("  Table: {}\n", table_name));
                            plan_text.push_str("  Indexes:\n");
                            for idx in table_def.indexes() {
                                let cols: Vec<_> = idx.columns().collect();
                                plan_text.push_str(&format!(
                                    "    - {} (columns: {:?}, unique: {}, partial: {}, expressions: {})\n",
                                    idx.name(),
                                    cols,
                                    idx.is_unique(),
                                    idx.is_partial(),
                                    idx.has_expressions()
                                ));
                            }
                            if table_def.indexes().is_empty() {
                                plan_text.push_str("    (no indexes)\n");
                            }
                        }
                    }
                }
            }
        }

        Ok(ExecuteResult::Explain { plan: plan_text })
    }

    pub(crate) fn execute_set(&self, set: &crate::sql::ast::SetStmt<'_>) -> Result<ExecuteResult> {
        use std::sync::atomic::Ordering;

        let name = set.name.to_lowercase();
        let value = set
            .value
            .first()
            .ok_or_else(|| eyre::eyre!("SET requires a value"))?;

        match name.as_str() {
            "foreign_keys" => {
                let enabled = match value {
                    crate::sql::ast::Expr::Literal(crate::sql::ast::Literal::Boolean(b)) => *b,
                    crate::sql::ast::Expr::Literal(crate::sql::ast::Literal::Integer(i)) => {
                        i.parse::<i64>().unwrap_or(0) != 0
                    }
                    crate::sql::ast::Expr::Literal(crate::sql::ast::Literal::String(s)) => {
                        matches!(s.to_lowercase().as_str(), "on" | "true" | "1" | "yes")
                    }
                    crate::sql::ast::Expr::Column(col) => {
                        matches!(col.column.to_lowercase().as_str(), "on" | "true" | "yes")
                    }
                    _ => {
                        bail!("invalid value for foreign_keys: expected ON/OFF, TRUE/FALSE, or 1/0")
                    }
                };
                self.foreign_keys_enabled.store(enabled, Ordering::Release);
                Ok(ExecuteResult::Set {
                    name: "foreign_keys".to_string(),
                    value: if enabled {
                        "ON".to_string()
                    } else {
                        "OFF".to_string()
                    },
                })
            }
            _ => bail!("unknown setting: {}", set.name),
        }
    }

    pub fn join_memory_budget(&self) -> usize {
        self.shared
            .join_memory_budget
            .load(std::sync::atomic::Ordering::Acquire)
    }

    pub fn memory_budget(&self) -> Arc<MemoryBudget> {
        Arc::clone(&self.shared.memory_budget)
    }

    pub fn memory_budget_ref(&self) -> &MemoryBudget {
        &self.shared.memory_budget
    }

    pub fn memory_stats(&self) -> crate::memory::BudgetStats {
        self.shared.memory_budget.stats()
    }

    pub fn persist_memory_stats(&self) -> Result<()> {
        use crate::schema::system_tables::{memory_stat_names, MEMORY_STATS_TABLE, SYSTEM_SCHEMA};
        use std::time::{SystemTime, UNIX_EPOCH};

        let stats = self.shared.memory_budget.stats();
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs().to_string())
            .unwrap_or_else(|_| "0".to_string());

        let stats_to_write = [
            (memory_stat_names::BUDGET_TOTAL, stats.total_limit as i64),
            (memory_stat_names::USED_CACHE, stats.cache_used as i64),
            (memory_stat_names::USED_QUERY, stats.query_used as i64),
            (memory_stat_names::USED_RECOVERY, stats.recovery_used as i64),
            (memory_stat_names::USED_SCHEMA, stats.schema_used as i64),
            (memory_stat_names::USED_SHARED, stats.shared_used as i64),
            (memory_stat_names::USED_TOTAL, stats.total_used as i64),
            (memory_stat_names::AVAILABLE_SHARED, stats.shared_available as i64),
        ];

        for (stat_name, stat_value) in &stats_to_write {
            let delete_sql = format!(
                "DELETE FROM {}.{} WHERE stat_name = '{}'",
                SYSTEM_SCHEMA, MEMORY_STATS_TABLE, stat_name
            );
            let _ = self.execute(&delete_sql);

            let insert_sql = format!(
                "INSERT INTO {}.{} (stat_name, stat_value, updated_at) VALUES ('{}', {}, '{}')",
                SYSTEM_SCHEMA, MEMORY_STATS_TABLE, stat_name, stat_value, now
            );
            self.execute(&insert_sql)?;
        }

        Ok(())
    }

    pub fn persist_wal_stats(&self) -> Result<()> {
        use crate::schema::system_tables::{wal_stat_names, SYSTEM_SCHEMA, WAL_STATS_TABLE};
        use std::time::{SystemTime, UNIX_EPOCH};

        let frame_count = self.wal_frame_count()? as i64;
        let size_bytes = self.wal_size_bytes()? as i64;
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs().to_string())
            .unwrap_or_else(|_| "0".to_string());

        let stats_to_write = [
            (wal_stat_names::FRAME_COUNT, frame_count),
            (wal_stat_names::SIZE_BYTES, size_bytes),
        ];

        for (stat_name, stat_value) in &stats_to_write {
            let delete_sql = format!(
                "DELETE FROM {}.{} WHERE stat_name = '{}'",
                SYSTEM_SCHEMA, WAL_STATS_TABLE, stat_name
            );
            let _ = self.execute(&delete_sql);

            let insert_sql = format!(
                "INSERT INTO {}.{} (stat_name, stat_value, updated_at) VALUES ('{}', {}, '{}')",
                SYSTEM_SCHEMA, WAL_STATS_TABLE, stat_name, stat_value, now
            );
            self.execute(&insert_sql)?;
        }

        Ok(())
    }

    pub fn wal_frame_count(&self) -> Result<u32> {
        let wal_guard = self.shared.wal.lock();
        if let Some(ref wal) = *wal_guard {
            Ok(wal.frame_count())
        } else {
            Ok(0)
        }
    }

    pub fn wal_size_bytes(&self) -> Result<u64> {
        let wal_guard = self.shared.wal.lock();
        if let Some(ref wal) = *wal_guard {
            Ok(wal.total_wal_size_bytes())
        } else {
            Ok(0)
        }
    }

    /// Returns the current database operating mode.
    pub fn mode(&self) -> super::DatabaseMode {
        *self.shared.mode.read()
    }

    /// Returns true if the database is in read-write mode.
    pub fn is_read_write(&self) -> bool {
        matches!(self.mode(), super::DatabaseMode::ReadWrite)
    }

    /// Returns true if the database is in degraded read-only mode.
    pub fn is_degraded(&self) -> bool {
        matches!(self.mode(), super::DatabaseMode::ReadOnlyDegraded { .. })
    }

    /// Check if the database is writable. Returns an error if in degraded mode.
    pub(crate) fn check_writable(&self) -> Result<()> {
        match self.mode() {
            super::DatabaseMode::ReadWrite => Ok(()),
            super::DatabaseMode::ReadOnlyDegraded { pending_wal_frames } => {
                bail!(
                    "Database is in read-only degraded mode with {} pending WAL frames. \
                     Run PRAGMA recover_wal to recover and enable writes.",
                    pending_wal_frames
                )
            }
        }
    }
}
