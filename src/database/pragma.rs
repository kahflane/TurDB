//! # PRAGMA Statement Handling
//!
//! This module implements PRAGMA statement execution for TurDB. PRAGMAs are used
//! to query and modify database configuration at runtime.
//!
//! ## Supported PRAGMAs
//!
//! ### WAL Configuration
//! - `PRAGMA wal` - Enable/disable write-ahead logging (ON/OFF)
//! - `PRAGMA synchronous` - Set sync mode (OFF, NORMAL, FULL)
//! - `PRAGMA wal_checkpoint` - Force WAL checkpoint
//! - `PRAGMA wal_checkpoint_stats` - Checkpoint with stats persistence
//! - `PRAGMA wal_checkpoint_threshold` - Get/set auto-checkpoint threshold
//! - `PRAGMA wal_frame_count` - Query current WAL frame count
//! - `PRAGMA wal_size` - Query total WAL size in bytes
//!
//! ### Memory Configuration
//! - `PRAGMA memory_budget` - Query total memory budget
//! - `PRAGMA memory_stats` - Query per-pool memory usage
//! - `PRAGMA persisted_memory_stats` - Query persisted stats from system table
//! - `PRAGMA join_memory_budget` - Get/set join operator memory limit
//!
//! ### Database State
//! - `PRAGMA database_mode` - Query current mode (read_write or read_only_degraded)
//! - `PRAGMA recover_wal` - Trigger streaming WAL recovery in degraded mode
//!
//! ## Usage
//!
//! ```sql
//! PRAGMA wal = ON;                    -- Enable WAL
//! PRAGMA memory_stats;                -- Show memory usage
//! PRAGMA wal_checkpoint;              -- Force checkpoint
//! PRAGMA join_memory_budget = 65536;  -- Set join memory to 64KB
//! ```

use crate::database::{Database, ExecuteResult};
use crate::storage::SyncMode;
use eyre::{bail, Result};
use std::sync::atomic::Ordering;
use std::sync::Arc;

impl Database {
    pub(crate) fn execute_pragma(
        &self,
        pragma: &crate::sql::ast::PragmaStmt<'_>,
    ) -> Result<ExecuteResult> {
        let name = pragma.name.to_uppercase();
        let value = pragma.value.map(|v| v.to_uppercase());

        match name.as_str() {
            "WAL" => self.pragma_wal(&name, value.as_deref()),
            "WAL_AUTOFLUSH" => self.pragma_wal_autoflush(&name, value.as_deref()),
            "SYNCHRONOUS" => self.pragma_synchronous(&name, value.as_deref()),
            "JOIN_MEMORY_BUDGET" => self.pragma_join_memory_budget(&name, value.as_deref()),
            "MEMORY_BUDGET" => self.pragma_memory_budget(&name),
            "MEMORY_STATS" => self.pragma_memory_stats(&name),
            "PERSISTED_MEMORY_STATS" => self.pragma_persisted_memory_stats(&name),
            "WAL_CHECKPOINT" => self.pragma_wal_checkpoint(&name),
            "WAL_CHECKPOINT_STATS" => self.pragma_wal_checkpoint_stats(&name),
            "WAL_CHECKPOINT_THRESHOLD" => {
                self.pragma_wal_checkpoint_threshold(&name, value.as_deref())
            }
            "WAL_FRAME_COUNT" => self.pragma_wal_frame_count(&name),
            "WAL_SIZE" => self.pragma_wal_size(&name),
            "DATABASE_MODE" => self.pragma_database_mode(&name),
            "RECOVER_WAL" => self.pragma_recover_wal(&name),
            _ => bail!("unknown PRAGMA: {}", name),
        }
    }

    fn pragma_wal(&self, name: &str, value: Option<&str>) -> Result<ExecuteResult> {
        if let Some(val) = value {
            match val {
                "ON" | "TRUE" | "1" => {
                    self.ensure_wal()?;
                    self.shared.wal_enabled.store(true, Ordering::Release);
                }
                "OFF" | "FALSE" | "0" => {
                    self.shared.wal_enabled.store(false, Ordering::Release);
                }
                _ => bail!("invalid PRAGMA WAL value: {}", val),
            }
        }
        let current = self.shared.wal_enabled.load(Ordering::Acquire);
        Ok(ExecuteResult::Pragma {
            name: name.to_string(),
            value: Some(if current { "ON" } else { "OFF" }.to_string()),
        })
    }

    fn pragma_wal_autoflush(&self, name: &str, value: Option<&str>) -> Result<ExecuteResult> {
        if let Some(val) = value {
            match val {
                "ON" | "TRUE" | "1" => {
                    self.shared.wal_autoflush.store(true, Ordering::Release);
                }
                "OFF" | "FALSE" | "0" => {
                    self.shared.wal_autoflush.store(false, Ordering::Release);
                }
                _ => bail!("invalid PRAGMA WAL_AUTOFLUSH value: {} (use ON or OFF)", val),
            }
        }
        let current = self.shared.wal_autoflush.load(Ordering::Acquire);
        Ok(ExecuteResult::Pragma {
            name: name.to_string(),
            value: Some(if current { "ON" } else { "OFF" }.to_string()),
        })
    }

    fn pragma_synchronous(&self, name: &str, value: Option<&str>) -> Result<ExecuteResult> {
        if let Some(val) = value {
            let mode = match val {
                "OFF" | "0" => SyncMode::Off,
                "NORMAL" | "1" => SyncMode::Normal,
                "FULL" | "2" => SyncMode::Full,
                _ => bail!(
                    "invalid PRAGMA synchronous value: {} (use OFF, NORMAL, or FULL)",
                    val
                ),
            };
            self.ensure_wal()?;
            let wal_guard = self.shared.wal.lock();
            if let Some(ref wal) = *wal_guard {
                wal.set_sync_mode(mode);
            }
        }
        let current_mode = {
            let wal_guard = self.shared.wal.lock();
            wal_guard
                .as_ref()
                .map(|w| w.sync_mode())
                .unwrap_or(SyncMode::Full)
        };
        let mode_str = match current_mode {
            SyncMode::Off => "OFF",
            SyncMode::Normal => "NORMAL",
            SyncMode::Full => "FULL",
        };
        Ok(ExecuteResult::Pragma {
            name: name.to_string(),
            value: Some(mode_str.to_string()),
        })
    }

    fn pragma_join_memory_budget(&self, name: &str, value: Option<&str>) -> Result<ExecuteResult> {
        if let Some(val) = value {
            let budget: usize = val.parse().map_err(|_| {
                eyre::eyre!(
                    "invalid PRAGMA join_memory_budget value: {} (use a number in bytes)",
                    val
                )
            })?;
            self.shared
                .join_memory_budget
                .store(budget, Ordering::Release);
        }
        let current = self.shared.join_memory_budget.load(Ordering::Acquire);
        Ok(ExecuteResult::Pragma {
            name: name.to_string(),
            value: Some(current.to_string()),
        })
    }

    fn pragma_memory_budget(&self, name: &str) -> Result<ExecuteResult> {
        let budget = self.shared.memory_budget.total_limit();
        Ok(ExecuteResult::Pragma {
            name: name.to_string(),
            value: Some(budget.to_string()),
        })
    }

    fn pragma_memory_stats(&self, name: &str) -> Result<ExecuteResult> {
        let stats = self.shared.memory_budget.stats();
        Ok(ExecuteResult::Pragma {
            name: name.to_string(),
            value: Some(stats.to_string()),
        })
    }

    fn pragma_persisted_memory_stats(&self, name: &str) -> Result<ExecuteResult> {
        use crate::schema::system_tables::{MEMORY_STATS_TABLE, SYSTEM_SCHEMA};

        let result = self.execute(&format!(
            "SELECT stat_name, stat_value, updated_at FROM {}.{}",
            SYSTEM_SCHEMA, MEMORY_STATS_TABLE
        ));

        match result {
            Ok(ExecuteResult::Select { rows, .. }) => {
                if rows.is_empty() {
                    Ok(ExecuteResult::Pragma {
                        name: name.to_string(),
                        value: Some("(no persisted stats - run checkpoint first)".to_string()),
                    })
                } else {
                    let stats_str: Vec<String> = rows
                        .iter()
                        .map(|r| {
                            let stat_name = r
                                .values
                                .first()
                                .map(|v| format!("{:?}", v))
                                .unwrap_or_default();
                            let stat_value = r
                                .values
                                .get(1)
                                .map(|v| format!("{:?}", v))
                                .unwrap_or_default();
                            format!("{}={}", stat_name, stat_value)
                        })
                        .collect();
                    Ok(ExecuteResult::Pragma {
                        name: name.to_string(),
                        value: Some(stats_str.join(",")),
                    })
                }
            }
            _ => Ok(ExecuteResult::Pragma {
                name: name.to_string(),
                value: Some("(unable to read persisted stats)".to_string()),
            }),
        }
    }

    fn pragma_wal_checkpoint(&self, name: &str) -> Result<ExecuteResult> {
        let frames = self.shared.checkpoint()?;
        Ok(ExecuteResult::Pragma {
            name: name.to_string(),
            value: Some(format!("checkpointed {} frames", frames)),
        })
    }

    fn pragma_wal_checkpoint_stats(&self, name: &str) -> Result<ExecuteResult> {
        let frames = self.checkpoint_wal_with_stats()?;
        Ok(ExecuteResult::Pragma {
            name: name.to_string(),
            value: Some(format!("checkpointed {} frames (stats persisted)", frames)),
        })
    }

    fn pragma_wal_checkpoint_threshold(
        &self,
        name: &str,
        value: Option<&str>,
    ) -> Result<ExecuteResult> {
        let wal_guard = self.shared.wal.lock();
        if let Some(ref wal) = *wal_guard {
            if let Some(val) = value {
                let threshold: u32 = val.parse().map_err(|_| {
                    eyre::eyre!(
                        "invalid PRAGMA wal_checkpoint_threshold value: {} (use a number)",
                        val
                    )
                })?;
                wal.set_checkpoint_threshold(threshold);
            }
            let current = wal.checkpoint_threshold();
            drop(wal_guard);
            Ok(ExecuteResult::Pragma {
                name: name.to_string(),
                value: Some(current.to_string()),
            })
        } else {
            drop(wal_guard);
            Ok(ExecuteResult::Pragma {
                name: name.to_string(),
                value: Some("1000".to_string()),
            })
        }
    }

    fn pragma_wal_frame_count(&self, name: &str) -> Result<ExecuteResult> {
        let wal_guard = self.shared.wal.lock();
        let frame_count = if let Some(ref wal) = *wal_guard {
            wal.frame_count()
        } else {
            0
        };
        drop(wal_guard);
        Ok(ExecuteResult::Pragma {
            name: name.to_string(),
            value: Some(frame_count.to_string()),
        })
    }

    fn pragma_wal_size(&self, name: &str) -> Result<ExecuteResult> {
        let wal_guard = self.shared.wal.lock();
        let wal_size = if let Some(ref wal) = *wal_guard {
            wal.total_wal_size_bytes()
        } else {
            0
        };
        drop(wal_guard);
        Ok(ExecuteResult::Pragma {
            name: name.to_string(),
            value: Some(wal_size.to_string()),
        })
    }

    fn pragma_database_mode(&self, name: &str) -> Result<ExecuteResult> {
        let mode = *self.shared.mode.read();
        let mode_str = match mode {
            super::DatabaseMode::ReadWrite => "read_write".to_string(),
            super::DatabaseMode::ReadOnlyDegraded { pending_wal_frames } => {
                format!("read_only_degraded (pending {} WAL frames)", pending_wal_frames)
            }
        };
        Ok(ExecuteResult::Pragma {
            name: name.to_string(),
            value: Some(mode_str),
        })
    }

    fn pragma_recover_wal(&self, name: &str) -> Result<ExecuteResult> {
        let mode = *self.shared.mode.read();
        match mode {
            super::DatabaseMode::ReadWrite => Ok(ExecuteResult::Pragma {
                name: name.to_string(),
                value: Some("already in read_write mode, no recovery needed".to_string()),
            }),
            super::DatabaseMode::ReadOnlyDegraded { pending_wal_frames } => {
                eprintln!(
                    "[turdb] Starting streaming WAL recovery for {} frames...",
                    pending_wal_frames
                );

                let frames = Self::streaming_recovery(
                    &self.shared.path,
                    &self.shared.wal_dir,
                    super::recovery::DEFAULT_RECOVERY_BATCH_SIZE,
                    Some(Arc::clone(&self.shared.memory_budget)),
                )?;

                *self.shared.mode.write() = super::DatabaseMode::ReadWrite;

                eprintln!("[turdb] WAL recovery complete! Recovered {} frames.", frames);

                Ok(ExecuteResult::Pragma {
                    name: name.to_string(),
                    value: Some(format!("recovered {} frames, now in read_write mode", frames)),
                })
            }
        }
    }
}
