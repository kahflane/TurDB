//! # Encoding Module
//!
//! This module provides encoding utilities for TurDB, including:
//!
//! - **Key encoding**: Big-endian byte-comparable encoding for B-tree indexes
//! - Future: Varint encoding, record serialization

pub mod key;

pub use key::type_prefix;
