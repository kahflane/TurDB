//! # Overflow Value Encoding
//!
//! This module handles the encoding and decoding of values that may be stored
//! partially in overflow pages. Large values (BLOBs, TEXT, etc.) that exceed
//! the inline threshold are stored as:
//!
//! 1. An inline prefix (first N bytes for index lookups)
//! 2. An overflow pointer to the remaining data
//!
//! ## Value Format
//!
//! Values use a type prefix to distinguish inline from overflow:
//!
//! ```text
//! Inline value (type = 0x00):
//! +------+----------+------------+
//! | 0x00 | len:var  | data:bytes |
//! +------+----------+------------+
//!
//! Overflow value (type = 0x01):
//! +------+------------+------------+--------------+
//! | 0x01 | total:u32  | page:u32   | prefix:bytes |
//! +------+------------+------------+--------------+
//! ```
//!
//! The overflow format stores:
//! - total_size: Total size of the complete value
//! - first_page: Page number of the first overflow page
//! - prefix: Inline prefix for index comparison (up to threshold)

use eyre::{ensure, Result};

use crate::encoding::varint::{decode_varint, encode_varint, varint_len};
use crate::storage::{free_overflow_chain, read_overflow, write_overflow, Storage};

/// Marker byte for inline values
pub const VALUE_TYPE_INLINE: u8 = 0x00;

/// Marker byte for overflow values
pub const VALUE_TYPE_OVERFLOW: u8 = 0x01;

/// Overhead for overflow pointer (type + total_size + first_page)
pub const OVERFLOW_POINTER_OVERHEAD: usize = 1 + 4 + 4; // 9 bytes

/// Maximum inline value size before overflow is required.
/// This leaves room for the key, slot, and cell overhead in a single page.
/// Values larger than this will use overflow pages.
pub const MAX_INLINE_VALUE: usize = 4096;

/// Minimum inline prefix to store for overflow values.
/// This helps with index lookups without needing to read overflow pages.
pub const MIN_INLINE_PREFIX: usize = 32;

/// Encodes a value for storage in a B-tree cell.
///
/// Small values are stored inline. Large values are written to overflow
/// pages and an overflow pointer is stored inline.
///
/// Returns the encoded value (either inline format or overflow pointer).
pub fn encode_value<S: Storage>(
    storage: &mut S,
    value: &[u8],
    available_space: usize,
    allocate_page: impl FnMut(&mut S) -> Result<u32>,
) -> Result<Vec<u8>> {
    // Calculate space needed for inline encoding
    let inline_len = 1 + varint_len(value.len() as u64) + value.len();

    // If it fits inline, use inline encoding
    if inline_len <= available_space && value.len() <= MAX_INLINE_VALUE {
        let mut encoded = Vec::with_capacity(inline_len);
        encoded.push(VALUE_TYPE_INLINE);
        let mut len_buf = [0u8; 10];
        let len_size = encode_varint(value.len() as u64, &mut len_buf);
        encoded.extend_from_slice(&len_buf[..len_size]);
        encoded.extend_from_slice(value);
        return Ok(encoded);
    }

    // Value needs overflow - calculate inline prefix size
    let max_prefix = available_space.saturating_sub(OVERFLOW_POINTER_OVERHEAD);
    let inline_prefix_len = max_prefix.min(value.len()).max(MIN_INLINE_PREFIX).min(value.len());

    // Calculate overflow data (everything after inline prefix)
    let overflow_data = &value[inline_prefix_len..];

    // Write overflow pages
    let (first_page, _pages) = write_overflow(storage, overflow_data, allocate_page)?;

    // Encode overflow pointer with inline prefix
    let mut encoded = Vec::with_capacity(OVERFLOW_POINTER_OVERHEAD + inline_prefix_len);
    encoded.push(VALUE_TYPE_OVERFLOW);
    encoded.extend_from_slice(&(value.len() as u32).to_le_bytes());
    encoded.extend_from_slice(&first_page.to_le_bytes());
    encoded.extend_from_slice(&value[..inline_prefix_len]);

    Ok(encoded)
}

/// Decodes a value from B-tree cell storage.
///
/// For raw/inline values (no prefix), returns the data directly.
/// For overflow values (0x01 prefix), reads from overflow pages and reconstructs.
/// For legacy encoded inline values (0x00 prefix), decodes and returns data.
pub fn decode_value<S: Storage + ?Sized>(storage: &S, encoded: &[u8]) -> Result<Vec<u8>> {
    ensure!(!encoded.is_empty(), "empty encoded value");

    match encoded[0] {
        VALUE_TYPE_INLINE => {
            // Legacy format: 0x00 prefix with varint length
            ensure!(encoded.len() > 1, "inline value too short");
            let (len, varint_size) = decode_varint(&encoded[1..])?;
            let data_start = 1 + varint_size;
            ensure!(
                data_start + len as usize <= encoded.len(),
                "inline value truncated"
            );
            Ok(encoded[data_start..data_start + len as usize].to_vec())
        }
        VALUE_TYPE_OVERFLOW => {
            // Overflow format: 0x01 prefix with overflow pointer
            ensure!(
                encoded.len() >= OVERFLOW_POINTER_OVERHEAD,
                "overflow pointer too short"
            );

            let total_size =
                u32::from_le_bytes([encoded[1], encoded[2], encoded[3], encoded[4]]) as usize;
            let first_page =
                u32::from_le_bytes([encoded[5], encoded[6], encoded[7], encoded[8]]);

            let inline_prefix = &encoded[OVERFLOW_POINTER_OVERHEAD..];
            let inline_prefix_len = inline_prefix.len();

            // Reconstruct full value
            let mut result = Vec::with_capacity(total_size);
            result.extend_from_slice(inline_prefix);

            // Read remaining data from overflow pages
            if inline_prefix_len < total_size {
                let overflow_data = read_overflow(storage, first_page, total_size - inline_prefix_len)?;
                result.extend_from_slice(&overflow_data);
            }

            Ok(result)
        }
        _ => {
            // Raw format: no prefix, value stored directly
            // This is the fast path for inline values <= MAX_INLINE_VALUE
            Ok(encoded.to_vec())
        }
    }
}

/// Returns a reference to inline value data without allocation.
///
/// For inline/raw values, returns Some with the data slice.
/// For overflow values, returns None (caller must use decode_value).
pub fn get_inline_value(encoded: &[u8]) -> Result<Option<&[u8]>> {
    ensure!(!encoded.is_empty(), "empty encoded value");

    match encoded[0] {
        VALUE_TYPE_INLINE => {
            // Legacy format: 0x00 prefix with varint length
            ensure!(encoded.len() > 1, "inline value too short");
            let (len, varint_size) = decode_varint(&encoded[1..])?;
            let data_start = 1 + varint_size;
            ensure!(
                data_start + len as usize <= encoded.len(),
                "inline value truncated"
            );
            Ok(Some(&encoded[data_start..data_start + len as usize]))
        }
        VALUE_TYPE_OVERFLOW => Ok(None),
        _ => {
            // Raw format: no prefix, value stored directly
            Ok(Some(encoded))
        }
    }
}

/// Checks if an encoded value uses overflow pages.
pub fn is_overflow_value(encoded: &[u8]) -> bool {
    !encoded.is_empty() && encoded[0] == VALUE_TYPE_OVERFLOW
}

/// Frees overflow pages associated with an encoded value.
///
/// Call this when deleting a row to reclaim overflow page space.
pub fn free_overflow_value<S: Storage>(
    storage: &mut S,
    encoded: &[u8],
    free_page: impl FnMut(&mut S, u32) -> Result<()>,
) -> Result<()> {
    if encoded.is_empty() || encoded[0] != VALUE_TYPE_OVERFLOW {
        return Ok(()); // Inline value, nothing to free
    }

    ensure!(
        encoded.len() >= OVERFLOW_POINTER_OVERHEAD,
        "overflow pointer too short"
    );

    let first_page = u32::from_le_bytes([encoded[5], encoded[6], encoded[7], encoded[8]]);

    if first_page != 0 {
        free_overflow_chain(storage, first_page, free_page)?;
    }

    Ok(())
}

/// Returns the total size of the value (without reading overflow pages).
pub fn get_value_size(encoded: &[u8]) -> Result<usize> {
    ensure!(!encoded.is_empty(), "empty encoded value");

    match encoded[0] {
        VALUE_TYPE_INLINE => {
            ensure!(encoded.len() > 1, "inline value too short");
            let (len, _) = decode_varint(&encoded[1..])?;
            Ok(len as usize)
        }
        VALUE_TYPE_OVERFLOW => {
            ensure!(
                encoded.len() >= OVERFLOW_POINTER_OVERHEAD,
                "overflow pointer too short"
            );
            let total_size =
                u32::from_le_bytes([encoded[1], encoded[2], encoded[3], encoded[4]]) as usize;
            Ok(total_size)
        }
        other => {
            eyre::bail!("unknown value type marker: 0x{:02x}", other);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::MmapStorage;
    use tempfile::tempdir;

    #[test]
    fn test_inline_value_roundtrip() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.db");
        let mut storage = MmapStorage::create(&path, 1).unwrap();

        let value = b"hello world";
        let encoded = encode_value(&mut storage, value, 1000, |s| {
            let page_count = s.page_count();
            s.grow(page_count + 1)?;
            Ok(page_count)
        })
        .unwrap();

        assert_eq!(encoded[0], VALUE_TYPE_INLINE);

        let decoded = decode_value(&storage, &encoded).unwrap();
        assert_eq!(decoded, value);
    }

    #[test]
    fn test_overflow_value_roundtrip() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.db");
        let mut storage = MmapStorage::create(&path, 10).unwrap();

        // Create a value larger than MAX_INLINE_VALUE
        let value: Vec<u8> = (0..10000).map(|i| (i % 256) as u8).collect();

        let encoded = encode_value(&mut storage, &value, 100, |s| {
            let page_count = s.page_count();
            s.grow(page_count + 1)?;
            Ok(page_count)
        })
        .unwrap();

        assert_eq!(encoded[0], VALUE_TYPE_OVERFLOW);
        assert!(is_overflow_value(&encoded));

        let decoded = decode_value(&storage, &encoded).unwrap();
        assert_eq!(decoded, value);
    }

    #[test]
    fn test_get_inline_value() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.db");
        let mut storage = MmapStorage::create(&path, 1).unwrap();

        let value = b"test data";
        let encoded = encode_value(&mut storage, value, 1000, |s| {
            let page_count = s.page_count();
            s.grow(page_count + 1)?;
            Ok(page_count)
        })
        .unwrap();

        let inline = get_inline_value(&encoded).unwrap();
        assert_eq!(inline, Some(value.as_slice()));
    }

    #[test]
    fn test_get_value_size() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.db");
        let mut storage = MmapStorage::create(&path, 10).unwrap();

        // Test inline
        let value = b"hello";
        let encoded = encode_value(&mut storage, value, 1000, |s| {
            let page_count = s.page_count();
            s.grow(page_count + 1)?;
            Ok(page_count)
        })
        .unwrap();
        assert_eq!(get_value_size(&encoded).unwrap(), 5);

        // Test overflow
        let large_value: Vec<u8> = (0..10000).map(|i| (i % 256) as u8).collect();
        let encoded = encode_value(&mut storage, &large_value, 100, |s| {
            let page_count = s.page_count();
            s.grow(page_count + 1)?;
            Ok(page_count)
        })
        .unwrap();
        assert_eq!(get_value_size(&encoded).unwrap(), 10000);
    }
}
