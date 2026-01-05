//! # SIMD-Accelerated B-Tree Leaf Scanning
//!
//! This module provides SIMD-optimized functions for B-Tree leaf page operations,
//! specifically targeting the "Slot Array with Prefix Hints" optimization.
//!
//! ## Design
//!
//! The key insight is that B-Tree leaf pages store 4-byte prefix hints in a
//! contiguous slot array. This layout is ideal for SIMD acceleration:
//!
//! ```text
//! Slot Array (contiguous in memory):
//! +--------+--------+--------+--------+--------+--------+
//! | Slot 0 | Slot 1 | Slot 2 | Slot 3 | Slot 4 | ...    |
//! +--------+--------+--------+--------+--------+--------+
//!
//! Each Slot (8 bytes):
//! +---------------+----------+-----------+
//! | prefix (4B)   | offset   | key_len   |
//! +---------------+----------+-----------+
//! ```
//!
//! ## Optimization Strategy
//!
//! 1. **SIMD Prefix Search**: Compare target prefix against 4-8 slot prefixes
//!    simultaneously using AVX2 (8 x 32-bit) or NEON (4 x 32-bit)
//!
//! 2. **Batch Slot Decoding**: Load and decode multiple slots in parallel for
//!    range scans, amortizing memory access overhead
//!
//! 3. **Prefetch Hints**: Issue prefetch instructions for upcoming slots
//!    during sequential scans
//!
//! ## Performance Targets
//!
//! - Point lookup: Reduce prefix comparisons from O(log n) serial to O(log n / 4-8) SIMD
//! - Range scan: Decode 4-8 slots per SIMD operation instead of 1
//!
//! ## Thread Safety
//!
//! All functions are pure and operate on borrowed data, making them safe
//! for concurrent use across threads.

use super::leaf::{extract_prefix, SearchResult, LEAF_CONTENT_START, SLOT_SIZE};
use crate::storage::PAGE_SIZE;

/// Number of slots that can be processed in one AVX2 operation (8 x 32-bit prefixes)
#[cfg(target_arch = "x86_64")]
pub const AVX2_BATCH_SIZE: usize = 8;

/// Number of slots that can be processed in one NEON operation (4 x 32-bit prefixes)
#[cfg(target_arch = "aarch64")]
pub const NEON_BATCH_SIZE: usize = 4;

/// Scalar fallback batch size (process 4 at a time for cache efficiency)
pub const SCALAR_BATCH_SIZE: usize = 4;

/// Batch of decoded slots for efficient iteration
#[derive(Debug, Default)]
pub struct SlotBatch {
    /// Prefixes as u32 (big-endian for comparison)
    pub prefixes: [u32; 8],
    /// Offsets to cell content
    pub offsets: [u16; 8],
    /// Key lengths
    pub key_lens: [u16; 8],
    /// Number of valid entries in this batch
    pub count: usize,
}

impl SlotBatch {
    /// Create a new empty batch
    pub fn new() -> Self {
        Self::default()
    }

    /// Load a batch of slots starting at the given index
    ///
    /// # Arguments
    /// * `page_data` - The page data buffer
    /// * `start_index` - Starting slot index
    /// * `cell_count` - Total number of cells in the page
    pub fn load_from_page(page_data: &[u8], start_index: usize, cell_count: usize) -> Self {
        let mut batch = Self::new();
        let end_index = (start_index + 8).min(cell_count);
        batch.count = end_index - start_index;

        for i in 0..batch.count {
            let slot_offset = LEAF_CONTENT_START + (start_index + i) * SLOT_SIZE;
            if slot_offset + SLOT_SIZE <= PAGE_SIZE {
                // Read prefix as big-endian u32 for comparison
                let prefix_bytes: [u8; 4] = page_data[slot_offset..slot_offset + 4]
                    .try_into()
                    .unwrap_or([0; 4]);
                batch.prefixes[i] = u32::from_be_bytes(prefix_bytes);

                // Read offset (little-endian)
                let offset_bytes: [u8; 2] = page_data[slot_offset + 4..slot_offset + 6]
                    .try_into()
                    .unwrap_or([0; 2]);
                batch.offsets[i] = u16::from_le_bytes(offset_bytes);

                // Read key_len (little-endian)
                let key_len_bytes: [u8; 2] = page_data[slot_offset + 6..slot_offset + 8]
                    .try_into()
                    .unwrap_or([0; 2]);
                batch.key_lens[i] = u16::from_le_bytes(key_len_bytes);
            }
        }

        batch
    }
}

/// SIMD-accelerated binary search for a prefix in the slot array
///
/// This function uses SIMD to compare the target prefix against multiple
/// slot prefixes simultaneously, reducing the number of iterations needed.
///
/// # Arguments
/// * `page_data` - The page data buffer
/// * `target_prefix` - The prefix to search for (as big-endian u32)
/// * `cell_count` - Number of cells in the page
///
/// # Returns
/// * `(left_bound, exact_matches)` - The narrowed search range and a bitmask of exact prefix matches
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe fn simd_prefix_search_avx2(
    page_data: &[u8],
    target_prefix: u32,
    cell_count: usize,
) -> (usize, usize, u32) {
    use std::arch::x86_64::*;

    if cell_count == 0 {
        return (0, 0, 0);
    }

    let mut left = 0usize;
    let mut right = cell_count;

    // Broadcast target prefix to all lanes
    let target = _mm256_set1_epi32(target_prefix as i32);

    // Process 8 slots at a time when range is large enough
    while right - left >= AVX2_BATCH_SIZE {
        let mid_start = left + (right - left) / 2;
        // Align to batch boundary
        let batch_start = mid_start.saturating_sub(AVX2_BATCH_SIZE / 2);
        let batch_start = batch_start.min(right.saturating_sub(AVX2_BATCH_SIZE));

        if batch_start + AVX2_BATCH_SIZE > cell_count {
            break; // Fall back to scalar for remainder
        }

        // Load 8 prefixes (need to gather from non-contiguous memory due to slot layout)
        let mut prefixes = [0i32; 8];
        for i in 0..8 {
            let slot_offset = LEAF_CONTENT_START + (batch_start + i) * SLOT_SIZE;
            let prefix_bytes: [u8; 4] = page_data[slot_offset..slot_offset + 4]
                .try_into()
                .unwrap_or([0; 4]);
            prefixes[i] = u32::from_be_bytes(prefix_bytes) as i32;
        }

        let batch = _mm256_loadu_si256(prefixes.as_ptr() as *const __m256i);

        // Compare: find positions where prefix < target (we want the first >= target)
        let cmp_lt = _mm256_cmpgt_epi32(target, batch); // target > batch means batch < target
        let lt_mask = _mm256_movemask_epi8(cmp_lt) as u32;

        // Find first slot where prefix >= target using trailing zeros
        // Each comparison produces 4 bytes in the mask
        let first_ge_idx = if lt_mask == 0xFFFFFFFF {
            // All prefixes < target, search in right half
            left = batch_start + AVX2_BATCH_SIZE;
            continue;
        } else if lt_mask == 0 {
            // All prefixes >= target, search in left half
            right = batch_start;
            continue;
        } else {
            // Mixed: find transition point
            // Each i32 produces 4 bytes in mask, so divide by 4
            (lt_mask.trailing_ones() / 4) as usize
        };

        // Narrow the range
        if first_ge_idx > 0 {
            left = batch_start + first_ge_idx - 1;
        }
        right = batch_start + first_ge_idx.min(7) + 1;

        // Check for exact matches in narrowed range
        let cmp_eq = _mm256_cmpeq_epi32(target, batch);
        let eq_mask = _mm256_movemask_epi8(cmp_eq) as u32;

        return (left, right, eq_mask);
    }

    // Fall back to scalar for small ranges
    (left, right, 0)
}

/// SIMD-accelerated binary search for a prefix in the slot array (NEON version)
#[cfg(target_arch = "aarch64")]
pub unsafe fn simd_prefix_search_neon(
    page_data: &[u8],
    target_prefix: u32,
    cell_count: usize,
) -> (usize, usize, u32) {
    use std::arch::aarch64::*;

    if cell_count == 0 {
        return (0, 0, 0);
    }

    let mut left = 0usize;
    let mut right = cell_count;

    // Broadcast target prefix to all lanes
    let target = vdupq_n_u32(target_prefix);

    // Process 4 slots at a time when range is large enough
    while right - left >= NEON_BATCH_SIZE {
        let mid_start = left + (right - left) / 2;
        let batch_start = mid_start.saturating_sub(NEON_BATCH_SIZE / 2);
        let batch_start = batch_start.min(right.saturating_sub(NEON_BATCH_SIZE));

        if batch_start + NEON_BATCH_SIZE > cell_count {
            break;
        }

        // Load 4 prefixes
        let mut prefixes = [0u32; 4];
        for i in 0..4 {
            let slot_offset = LEAF_CONTENT_START + (batch_start + i) * SLOT_SIZE;
            let prefix_bytes: [u8; 4] = page_data[slot_offset..slot_offset + 4]
                .try_into()
                .unwrap_or([0; 4]);
            prefixes[i] = u32::from_be_bytes(prefix_bytes);
        }

        let batch = vld1q_u32(prefixes.as_ptr());

        // Compare for less-than (batch < target)
        let cmp_lt = vcltq_u32(batch, target);

        // Convert comparison result to mask
        let cmp_narrow = vmovn_u32(cmp_lt);
        let mask_val = vget_lane_u64(vreinterpret_u64_u16(cmp_narrow), 0);

        let first_ge_idx = if mask_val == 0xFFFFFFFFFFFFFFFF {
            left = batch_start + NEON_BATCH_SIZE;
            continue;
        } else if mask_val == 0 {
            right = batch_start;
            continue;
        } else {
            (mask_val.trailing_ones() / 16) as usize
        };

        if first_ge_idx > 0 {
            left = batch_start + first_ge_idx - 1;
        }
        right = batch_start + first_ge_idx.min(3) + 1;

        // Check for exact matches
        let cmp_eq = vceqq_u32(batch, target);
        let eq_narrow = vmovn_u32(cmp_eq);
        let eq_mask = vget_lane_u64(vreinterpret_u64_u16(eq_narrow), 0) as u32;

        return (left, right, eq_mask);
    }

    (left, right, 0)
}

/// Scalar fallback for prefix search (used when SIMD is not available or for small ranges)
pub fn simd_prefix_search_scalar(
    page_data: &[u8],
    target_prefix: u32,
    cell_count: usize,
) -> (usize, usize, u32) {
    if cell_count == 0 {
        return (0, 0, 0);
    }

    let mut left = 0usize;
    let mut right = cell_count;
    let mut eq_mask = 0u32;

    // Scalar binary search with batch processing
    while right - left >= SCALAR_BATCH_SIZE {
        let mid = left + (right - left) / 2;
        let slot_offset = LEAF_CONTENT_START + mid * SLOT_SIZE;

        if slot_offset + SLOT_SIZE > PAGE_SIZE {
            break;
        }

        let prefix_bytes: [u8; 4] = page_data[slot_offset..slot_offset + 4]
            .try_into()
            .unwrap_or([0; 4]);
        let slot_prefix = u32::from_be_bytes(prefix_bytes);

        match slot_prefix.cmp(&target_prefix) {
            std::cmp::Ordering::Less => left = mid + 1,
            std::cmp::Ordering::Greater => right = mid,
            std::cmp::Ordering::Equal => {
                eq_mask = 1 << (mid % 32);
                // Found equal prefix, narrow range to find exact position
                return (mid.saturating_sub(1), mid + 1, eq_mask);
            }
        }
    }

    (left, right, eq_mask)
}

/// SIMD-accelerated key search in a leaf page
///
/// Uses SIMD prefix comparison to narrow down the search range, then falls
/// back to scalar comparison for the final key match.
pub fn find_key_simd(page_data: &[u8], key: &[u8], cell_count: usize) -> SearchResult {
    if cell_count == 0 {
        return SearchResult::NotFound(0);
    }

    let target_prefix = u32::from_be_bytes(extract_prefix(key));

    // Use SIMD to narrow down the search range
    #[cfg(target_arch = "x86_64")]
    let (mut left, mut right, _eq_mask) = {
        if is_x86_feature_detected!("avx2") {
            // SAFETY: We've checked for AVX2 support
            unsafe { simd_prefix_search_avx2(page_data, target_prefix, cell_count) }
        } else {
            simd_prefix_search_scalar(page_data, target_prefix, cell_count)
        }
    };

    #[cfg(target_arch = "aarch64")]
    let (mut left, mut right, _eq_mask) = {
        // NEON is always available on aarch64
        // SAFETY: NEON is guaranteed on aarch64
        unsafe { simd_prefix_search_neon(page_data, target_prefix, cell_count) }
    };

    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    let (mut left, mut right, _eq_mask) =
        simd_prefix_search_scalar(page_data, target_prefix, cell_count);

    // Ensure bounds are valid
    right = right.min(cell_count);

    // Scalar binary search in the narrowed range
    while left < right {
        let mid = left + (right - left) / 2;
        let slot_offset = LEAF_CONTENT_START + mid * SLOT_SIZE;

        if slot_offset + SLOT_SIZE > PAGE_SIZE {
            return SearchResult::NotFound(mid);
        }

        // Read prefix
        let prefix_bytes: [u8; 4] = page_data[slot_offset..slot_offset + 4]
            .try_into()
            .unwrap_or([0; 4]);
        let slot_prefix = u32::from_be_bytes(prefix_bytes);

        match slot_prefix.cmp(&target_prefix) {
            std::cmp::Ordering::Less => left = mid + 1,
            std::cmp::Ordering::Greater => right = mid,
            std::cmp::Ordering::Equal => {
                // Prefix matches, need to compare full key
                let offset_bytes: [u8; 2] = page_data[slot_offset + 4..slot_offset + 6]
                    .try_into()
                    .unwrap_or([0; 2]);
                let cell_offset = u16::from_le_bytes(offset_bytes) as usize;

                let key_len_bytes: [u8; 2] = page_data[slot_offset + 6..slot_offset + 8]
                    .try_into()
                    .unwrap_or([0; 2]);
                let key_len = u16::from_le_bytes(key_len_bytes) as usize;

                if cell_offset + key_len > PAGE_SIZE {
                    return SearchResult::NotFound(mid);
                }

                let full_key = &page_data[cell_offset..cell_offset + key_len];

                match full_key.cmp(key) {
                    std::cmp::Ordering::Equal => return SearchResult::Found(mid),
                    std::cmp::Ordering::Less => left = mid + 1,
                    std::cmp::Ordering::Greater => right = mid,
                }
            }
        }
    }

    SearchResult::NotFound(left)
}

/// Batch iterator for efficient range scans
///
/// This iterator decodes slots in batches of 8, amortizing the overhead
/// of reading from the slot array.
pub struct BatchSlotIterator<'a> {
    page_data: &'a [u8],
    cell_count: usize,
    current_index: usize,
    current_batch: SlotBatch,
    batch_index: usize,
}

impl<'a> BatchSlotIterator<'a> {
    pub fn new(page_data: &'a [u8], cell_count: usize) -> Self {
        let current_batch = if cell_count > 0 {
            SlotBatch::load_from_page(page_data, 0, cell_count)
        } else {
            SlotBatch::new()
        };

        Self {
            page_data,
            cell_count,
            current_index: 0,
            current_batch,
            batch_index: 0,
        }
    }

    pub fn from_index(page_data: &'a [u8], cell_count: usize, start_index: usize) -> Self {
        let aligned_start = (start_index / 8) * 8;
        let current_batch = if start_index < cell_count {
            SlotBatch::load_from_page(page_data, aligned_start, cell_count)
        } else {
            SlotBatch::new()
        };

        Self {
            page_data,
            cell_count,
            current_index: start_index,
            current_batch,
            batch_index: start_index - aligned_start,
        }
    }

    /// Get the current slot without advancing
    pub fn current(&self) -> Option<(u32, u16, u16)> {
        if self.current_index >= self.cell_count {
            return None;
        }

        if self.batch_index < self.current_batch.count {
            Some((
                self.current_batch.prefixes[self.batch_index],
                self.current_batch.offsets[self.batch_index],
                self.current_batch.key_lens[self.batch_index],
            ))
        } else {
            None
        }
    }

    /// Advance to the next slot
    pub fn advance(&mut self) {
        self.current_index += 1;
        self.batch_index += 1;

        // Load next batch if needed
        if self.batch_index >= 8 && self.current_index < self.cell_count {
            let aligned_start = (self.current_index / 8) * 8;
            self.current_batch =
                SlotBatch::load_from_page(self.page_data, aligned_start, self.cell_count);
            self.batch_index = self.current_index - aligned_start;
        }
    }

    /// Get remaining count
    pub fn remaining(&self) -> usize {
        self.cell_count.saturating_sub(self.current_index)
    }
}

impl<'a> Iterator for BatchSlotIterator<'a> {
    type Item = (u32, u16, u16); // (prefix, offset, key_len)

    fn next(&mut self) -> Option<Self::Item> {
        let result = self.current();
        if result.is_some() {
            self.advance();
        }
        result
    }
}

/// Prefetch upcoming slots for better cache utilization during range scans
#[cfg(target_arch = "x86_64")]
#[inline]
pub fn prefetch_slots(page_data: &[u8], start_index: usize, _cell_count: usize) {
    use std::arch::x86_64::*;

    let slot_offset = LEAF_CONTENT_START + start_index * SLOT_SIZE;
    if slot_offset + 64 <= PAGE_SIZE {
        // SAFETY: The pointer is valid (within page bounds) and aligned for prefetch
        unsafe {
            _mm_prefetch(page_data.as_ptr().add(slot_offset) as *const i8, _MM_HINT_T0);
        }
    }
}

#[cfg(target_arch = "aarch64")]
#[inline]
pub fn prefetch_slots(page_data: &[u8], start_index: usize, _cell_count: usize) {
    use std::arch::aarch64::*;

    let slot_offset = LEAF_CONTENT_START + start_index * SLOT_SIZE;
    if slot_offset + 64 <= PAGE_SIZE {
        // SAFETY: The pointer is valid (within page bounds)
        unsafe {
            _prefetch(page_data.as_ptr().add(slot_offset) as *const i8, _PREFETCH_READ, _PREFETCH_LOCALITY3);
        }
    }
}

#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
#[inline]
pub fn prefetch_slots(_page_data: &[u8], _start_index: usize, _cell_count: usize) {
    // No-op on unsupported architectures
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_slot_batch_empty() {
        let batch = SlotBatch::new();
        assert_eq!(batch.count, 0);
    }

    #[test]
    fn test_simd_prefix_search_scalar_empty() {
        let page_data = [0u8; PAGE_SIZE];
        let (left, right, eq_mask) = simd_prefix_search_scalar(&page_data, 0x12345678, 0);
        assert_eq!(left, 0);
        assert_eq!(right, 0);
        assert_eq!(eq_mask, 0);
    }

    #[test]
    fn test_find_key_simd_empty() {
        let page_data = [0u8; PAGE_SIZE];
        let result = find_key_simd(&page_data, b"test", 0);
        assert!(matches!(result, SearchResult::NotFound(0)));
    }

    #[test]
    fn test_batch_iterator_empty() {
        let page_data = [0u8; PAGE_SIZE];
        let mut iter = BatchSlotIterator::new(&page_data, 0);
        assert!(iter.next().is_none());
        assert_eq!(iter.remaining(), 0);
    }

    #[test]
    fn test_extract_prefix_short_key() {
        let key = b"ab";
        let prefix = extract_prefix(key);
        assert_eq!(prefix, [b'a', b'b', 0, 0]);
    }

    #[test]
    fn test_extract_prefix_long_key() {
        let key = b"hello world";
        let prefix = extract_prefix(key);
        assert_eq!(prefix, [b'h', b'e', b'l', b'l']);
    }

    #[test]
    fn test_prefix_comparison_order() {
        // Test that big-endian comparison gives correct lexicographic order
        let key_a = b"aaaa";
        let key_b = b"aaab";
        let key_c = b"aaba";

        let prefix_a = u32::from_be_bytes(extract_prefix(key_a));
        let prefix_b = u32::from_be_bytes(extract_prefix(key_b));
        let prefix_c = u32::from_be_bytes(extract_prefix(key_c));

        assert!(prefix_a < prefix_b);
        assert!(prefix_b < prefix_c);
    }
}
