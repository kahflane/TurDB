//! # B-Tree Index Implementation
//!
//! This module implements a B+tree index structure optimized for TurDB's storage layer.
//! The design prioritizes zero-copy access and fast key lookup through prefix hints.
//!
//! ## Architecture Overview
//!
//! TurDB's B+tree uses a slot array design where each node stores key prefixes inline
//! in the slot array, enabling fast prefix-based filtering before reading full keys.
//! This reduces cache misses during key searches.
//!
//! ## Node Types
//!
//! - **Leaf Nodes**: Store actual key-value pairs. Keys are stored in sorted order.
//!   Values are either inline (small) or overflow page references (large).
//!
//! - **Interior Nodes**: Store separator keys and child page pointers. The separator
//!   key at position i is the smallest key in the subtree rooted at child i+1.
//!
//! ## Page Layout (Leaf Node)
//!
//! ```text
//! +----------------------+
//! | Page Header (16B)    |  Standard TurDB page header
//! +----------------------+
//! | Leaf Header (8B)     |  Leaf-specific metadata
//! +----------------------+
//! | Slot Array           |  Array of Slot structs (8B each)
//! | [Slot 0]             |  - prefix: [u8; 4] (first 4 bytes of key)
//! | [Slot 1]             |  - offset: u16 (cell content offset)
//! | ...                  |  - key_len: u16 (key length for validation)
//! +----------------------+
//! | Free Space           |
//! +----------------------+
//! | Cell Content         |  Grows upward from page end
//! | (key | value_len |   |  - key: [u8; key_len]
//! |  value)              |  - value_len: varint
//! |                      |  - value: [u8; value_len]
//! +----------------------+
//! ```
//!
//! ## Slot Array Design
//!
//! Each slot contains:
//! - **prefix** (4 bytes): First 4 bytes of the key for fast comparison
//! - **offset** (2 bytes): Offset to cell content within the page
//! - **key_len** (2 bytes): Length of the full key
//!
//! This design enables:
//! 1. Fast prefix-based binary search (compare 4-byte prefixes)
//! 2. Early rejection of non-matching keys
//! 3. Cache-efficient sequential access during scans
//!
//! ## Zero-Copy Access
//!
//! All key and value access returns `&[u8]` slices pointing directly into
//! the page buffer. No data is copied during reads:
//!
//! ```text
//! let node = LeafNode::from_page(page_data);
//! let key: &[u8] = node.key_at(0)?;    // Points into page_data
//! let value: &[u8] = node.value_at(0)?; // Points into page_data
//! ```
//!
//! ## Key Search Algorithm
//!
//! 1. Extract 4-byte prefix from search key
//! 2. Binary search slot array comparing prefixes (fast integer comparison)
//! 3. On prefix match, compare full keys (handles prefix collisions)
//! 4. Return Found(index) or NotFound(insertion_point)
//!
//! ## Memory Layout
//!
//! The Slot struct is designed for efficient memory access:
//! - Total size: 8 bytes (fits in a single cache line with 8 slots)
//! - Aligned for efficient access
//! - Uses zerocopy for safe transmutation from page bytes
//!
//! ## Capacity Calculation
//!
//! With 16KB pages:
//! - Page header: 16 bytes
//! - Leaf header: 8 bytes
//! - Usable for slots + cells: 16360 bytes
//! - Minimum cell size: ~10 bytes (small key + small value + varint)
//! - Maximum slots: ~900 (with average 10-byte cells)
//!
//! ## Thread Safety
//!
//! LeafNode borrows from page data and is `!Send + !Sync`. Thread safety
//! is provided by the page cache layer which controls access to page buffers.

mod interior;
mod leaf;
mod tree;

pub use interior::{
    separator_len, InteriorNode, InteriorNodeMut, InteriorSlot, INTERIOR_CONTENT_START,
    INTERIOR_SLOT_SIZE,
};
pub use leaf::{
    extract_prefix, LeafNode, LeafNodeMut, SearchResult, Slot, LEAF_CONTENT_START,
    LEAF_HEADER_SIZE, SLOT_SIZE,
};
pub use tree::{BTree, InsertResult};
