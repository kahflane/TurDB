//! # TurDB Configuration Module
//!
//! This module centralizes all configuration constants for TurDB. Constants are
//! grouped by their functional area and interdependencies are documented and
//! enforced through compile-time assertions.
//!
//! ## Why Centralization?
//!
//! Scattered constants across multiple files led to bugs where interdependent
//! values became mismatched. For example, `COMMIT_BATCH_SIZE` must never exceed
//! `DEFAULT_BUFFER_POOL_SIZE` or a deadlock occurs during commit. By co-locating
//! these constants and adding compile-time checks, we prevent such issues.
//!
//! ## Module Organization
//!
//! - [`constants`]: All numeric configuration values with dependency documentation

pub mod constants;
pub use constants::*;
