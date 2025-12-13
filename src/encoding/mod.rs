//! # Encoding Module
//!
//! This module provides encoding utilities for TurDB, including:
//!
//! - **Key encoding**: Big-endian byte-comparable encoding for B-tree indexes
//! - **Varint encoding**: Variable-length integer encoding for record lengths and cell sizes

pub mod key;
pub mod varint;

pub use key::type_prefix;
pub use varint::{decode_varint, encode_varint, varint_len};
