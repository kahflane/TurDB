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
//!
//! ## Platform Notes
//!
//! - **x86_64**: Full AVX2 SIMD support with prefetch hints
//! - **aarch64**: NEON SIMD support; prefetch disabled (requires nightly Rust)
//! - **Other**: Scalar fallback implementation

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
/// * `(left_bound, right_bound, exact_match_mask)` - The narrowed search range and a bitmask of exact prefix matches
///
/// # Safety
/// Caller must verify AVX2 support via `is_x86_feature_detected!("avx2")`.
/// The caller must ensure page_data is valid and cell_count does not exceed
/// the actual number of slots in the page.
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

    let sign_bit = _mm256_set1_epi32(0x80000000u32 as i32);
    let target_unsigned = _mm256_xor_si256(_mm256_set1_epi32(target_prefix as i32), sign_bit);

    while right - left >= AVX2_BATCH_SIZE {
        let mid_start = left + (right - left) / 2;
        let batch_start = mid_start.saturating_sub(AVX2_BATCH_SIZE / 2);
        let batch_start = batch_start.min(right.saturating_sub(AVX2_BATCH_SIZE));

        if batch_start + AVX2_BATCH_SIZE > cell_count {
            break;
        }

        let mut prefixes = [0i32; 8];
        for i in 0..8 {
            let slot_offset = LEAF_CONTENT_START + (batch_start + i) * SLOT_SIZE;
            let prefix_bytes: [u8; 4] = page_data[slot_offset..slot_offset + 4]
                .try_into()
                .unwrap_or([0; 4]);
            prefixes[i] = u32::from_be_bytes(prefix_bytes) as i32;
        }

        let batch = _mm256_loadu_si256(prefixes.as_ptr() as *const __m256i);
        let batch_unsigned = _mm256_xor_si256(batch, sign_bit);

        let cmp_lt = _mm256_cmpgt_epi32(target_unsigned, batch_unsigned);
        let lt_mask = _mm256_movemask_epi8(cmp_lt) as u32;

        let cmp_eq = _mm256_cmpeq_epi32(_mm256_set1_epi32(target_prefix as i32), batch);
        let eq_mask = _mm256_movemask_epi8(cmp_eq) as u32;

        if lt_mask == 0xFFFFFFFF {
            if eq_mask != 0 {
                let eq_idx = (eq_mask.trailing_zeros() / 4) as usize;
                let last_eq_idx = (32 - eq_mask.leading_zeros()) as usize / 4;
                return (batch_start + eq_idx, batch_start + last_eq_idx, eq_mask);
            }
            left = batch_start + AVX2_BATCH_SIZE;
            continue;
        } else if lt_mask == 0 {
            if eq_mask != 0 {
                let eq_idx = (eq_mask.trailing_zeros() / 4) as usize;
                let last_eq_idx = (32 - eq_mask.leading_zeros()) as usize / 4;
                return (batch_start + eq_idx, batch_start + last_eq_idx, eq_mask);
            }
            right = batch_start;
            continue;
        }

        let first_ge_idx = (lt_mask.trailing_ones() / 4) as usize;

        if first_ge_idx > 0 {
            left = batch_start + first_ge_idx - 1;
        }
        right = batch_start + first_ge_idx.min(7) + 1;

        // If there are equal matches, ensure we include all of them
        if eq_mask != 0 {
            let last_eq_idx = (32 - eq_mask.leading_zeros()) as usize / 4;
             if batch_start + last_eq_idx > right {
                 right = batch_start + last_eq_idx;
             }
        }

        return (left, right, eq_mask);
    }

    (left, right, 0)
}

/// SIMD-accelerated binary search for a prefix in the slot array (NEON version)
///
/// # Safety
/// NEON is guaranteed on aarch64. The caller must ensure page_data is valid
/// and cell_count does not exceed the actual number of slots in the page.
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

    let target = vdupq_n_u32(target_prefix);

    while right - left >= NEON_BATCH_SIZE {
        let mid_start = left + (right - left) / 2;
        let batch_start = mid_start.saturating_sub(NEON_BATCH_SIZE / 2);
        let batch_start = batch_start.min(right.saturating_sub(NEON_BATCH_SIZE));

        if batch_start + NEON_BATCH_SIZE > cell_count {
            break;
        }

        let mut prefixes = [0u32; 4];
        for (i, prefix) in prefixes.iter_mut().enumerate() {
            let slot_offset = LEAF_CONTENT_START + (batch_start + i) * SLOT_SIZE;
            let prefix_bytes: [u8; 4] = page_data[slot_offset..slot_offset + 4]
                .try_into()
                .unwrap_or([0; 4]);
            *prefix = u32::from_be_bytes(prefix_bytes);
        }

        let batch = vld1q_u32(prefixes.as_ptr());
        let cmp_lt = vcltq_u32(batch, target);
        let cmp_narrow = vmovn_u32(cmp_lt);
        let lt_mask_val = vget_lane_u64(vreinterpret_u64_u16(cmp_narrow), 0);

        let cmp_eq = vceqq_u32(batch, target);
        let eq_narrow = vmovn_u32(cmp_eq);
        let eq_mask_val = vget_lane_u64(vreinterpret_u64_u16(eq_narrow), 0);

        // Calculate count of LT (Less Than) items
        let num_lt = (lt_mask_val.trailing_ones() / 16) as usize;
        
        let new_left = if num_lt > 0 {
            batch_start + num_lt
        } else {
            left
        };
        
        // Calculate index of first GT (Greater Than) item
        let lt_eq_mask = lt_mask_val | eq_mask_val;
        let gt_mask = !lt_eq_mask;
        
        let mut new_right = right;
        if gt_mask != 0 {
             let first_gt = (gt_mask.trailing_zeros() / 16) as usize;
             if first_gt < 4 {
                new_right = batch_start + first_gt;
             }
        }
        
        if new_left <= left && new_right >= right {
             break; 
        }
        
        left = new_left;
        right = new_right;
        
        if left >= right {
            break;
        }
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
    let mut start_eq_mask = 0u32;

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
                // Found equal prefix.
                // We cannot simply return [mid-1, mid+1] because neighbors might also be equal.
                // We should break and let the standard binary search handle strict range.
                // Or try to expand locally?
                // Safest is to just break and return current range, assuming binary search is fast enough.
                start_eq_mask = 1; 
                break;
            }
        }
    }

    (left, right, start_eq_mask)
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
pub fn prefetch_slots(_page_data: &[u8], _start_index: usize, _cell_count: usize) {
}

#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
#[inline]
pub fn prefetch_slots(_page_data: &[u8], _start_index: usize, _cell_count: usize) {
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
        let key_a = b"aaaa";
        let key_b = b"aaab";
        let key_c = b"aaba";

        let prefix_a = u32::from_be_bytes(extract_prefix(key_a));
        let prefix_b = u32::from_be_bytes(extract_prefix(key_b));
        let prefix_c = u32::from_be_bytes(extract_prefix(key_c));

        assert!(prefix_a < prefix_b);
        assert!(prefix_b < prefix_c);
    }

    fn create_test_page_with_keys(keys: &[&[u8]]) -> Vec<u8> {
        use crate::storage::{PageHeader, PageType};

        let mut page = vec![0u8; PAGE_SIZE];

        let header = PageHeader::from_bytes_mut(&mut page).unwrap();
        header.set_page_type(PageType::BTreeLeaf);
        header.set_cell_count(keys.len() as u16);
        header.set_free_start((LEAF_CONTENT_START + keys.len() * SLOT_SIZE) as u16);

        let mut cell_end = PAGE_SIZE;
        for (i, key) in keys.iter().enumerate() {
            let cell_size = key.len() + 2;
            cell_end -= cell_size;

            page[cell_end..cell_end + key.len()].copy_from_slice(key);
            page[cell_end + key.len()] = 0;
            page[cell_end + key.len() + 1] = 0;

            let slot_offset = LEAF_CONTENT_START + i * SLOT_SIZE;
            let prefix = extract_prefix(key);
            page[slot_offset..slot_offset + 4].copy_from_slice(&prefix);
            page[slot_offset + 4..slot_offset + 6]
                .copy_from_slice(&(cell_end as u16).to_le_bytes());
            page[slot_offset + 6..slot_offset + 8]
                .copy_from_slice(&(key.len() as u16).to_le_bytes());
        }

        let header = PageHeader::from_bytes_mut(&mut page).unwrap();
        header.set_free_end(cell_end as u16);

        page
    }

    #[test]
    fn test_high_byte_prefix_ordering() {
        let key_low = b"aaa";
        let key_high = b"\x80\x00\x00";

        let prefix_low = u32::from_be_bytes(extract_prefix(key_low));
        let prefix_high = u32::from_be_bytes(extract_prefix(key_high));

        assert!(
            prefix_low < prefix_high,
            "0x{:08X} should be < 0x{:08X} for correct lexicographic order",
            prefix_low,
            prefix_high
        );
    }

    #[test]
    fn test_simd_bug_repro() {
        let mut keys: Vec<Vec<u8>> = Vec::new();
        // Create 200 keys: 0, 2, ...
        // Note: loop limit 500 to match reproduction test scale if needed, but 250 (125 items) is enough
        for i in (0..500).step_by(2) {
             let k = (i as u64).to_be_bytes().to_vec();
             keys.push(k);
        }
        
        // Setup: Indices 0..120 exist. Key at 120 is 240.
        // We want to insert 242.
        
        let limit = 121; // include 240
        let keys_slice = &keys[0..limit];
        let keys_refs: Vec<&[u8]> = keys_slice.iter().map(|k| k.as_slice()).collect();
        
        let page = create_test_page_with_keys(&keys_refs);
        
        let target_key = &keys[limit]; // 242
        println!("Target Key: {:?}", target_key);
        
        let result = find_key_simd(&page, target_key, keys_refs.len());
         match result {
             SearchResult::NotFound(idx) => {
                 println!("Found insertion point at {}", idx);
                 assert_eq!(idx, limit, "Should insert at {} (after 240)", limit);
             },
             _ => panic!("Should result in NotFound"),
        }
    }


    #[test]
    fn test_find_key_simd_with_high_byte_keys() {
        let keys: Vec<&[u8]> = vec![
            b"aaa",
            b"bbb",
            b"ccc",
            b"\x80\x00\x00",
            b"\x90\x00\x00",
            b"\xff\x00\x00",
        ];

        let page = create_test_page_with_keys(&keys);

        for (expected_idx, key) in keys.iter().enumerate() {
            let result = find_key_simd(&page, key, keys.len());
            assert_eq!(
                result,
                SearchResult::Found(expected_idx),
                "Failed to find key {:?} (expected at index {})",
                key,
                expected_idx
            );
        }
    }

    #[test]
    fn test_find_key_simd_not_found_high_byte() {
        let keys: Vec<&[u8]> = vec![
            b"aaa",
            b"ccc",
            b"\x80\x00\x00",
            b"\xff\x00\x00",
        ];

        let page = create_test_page_with_keys(&keys);

        let result = find_key_simd(&page, b"bbb", keys.len());
        assert_eq!(
            result,
            SearchResult::NotFound(1),
            "bbb should be inserted at index 1"
        );

        let result = find_key_simd(&page, b"\x85\x00\x00", keys.len());
        assert_eq!(
            result,
            SearchResult::NotFound(3),
            "0x85... should be inserted at index 3"
        );
    }

    #[test]
    fn test_find_key_simd_matches_scalar_for_many_keys() {
        let mut keys: Vec<Vec<u8>> = Vec::new();
        for i in 0u8..64 {
            keys.push(vec![i, 0, 0, 0]);
        }
        for i in 0x80u8..0xC0 {
            keys.push(vec![i, 0, 0, 0]);
        }

        keys.sort();
        let key_refs: Vec<&[u8]> = keys.iter().map(|k| k.as_slice()).collect();
        let page = create_test_page_with_keys(&key_refs);

        for (expected_idx, key) in key_refs.iter().enumerate() {
            let simd_result = find_key_simd(&page, key, key_refs.len());
            assert_eq!(
                simd_result,
                SearchResult::Found(expected_idx),
                "SIMD failed to find key {:02X?} at index {}",
                key,
                expected_idx
            );
        }

        let not_found_key = vec![0x50u8, 0, 0, 0];
        let simd_result = find_key_simd(&page, &not_found_key, key_refs.len());
        if let SearchResult::NotFound(idx) = simd_result {
            assert!(
                idx <= key_refs.len(),
                "Insertion point {} should be <= {}",
                idx,
                key_refs.len()
            );
            if idx > 0 {
                assert!(
                    key_refs[idx - 1] < not_found_key.as_slice(),
                    "Key at idx-1 should be less than search key"
                );
            }
            if idx < key_refs.len() {
                assert!(
                    key_refs[idx] > not_found_key.as_slice(),
                    "Key at idx should be greater than search key"
                );
            }
        } else {
            panic!("Key 0x50... should not be found");
        }
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_avx2_unsigned_comparison() {
        if !is_x86_feature_detected!("avx2") {
            return;
        }

        let keys: Vec<&[u8]> = vec![
            b"\x00\x00\x00\x00",
            b"\x40\x00\x00\x00",
            b"\x7f\xff\xff\xff",
            b"\x80\x00\x00\x00",
            b"\xc0\x00\x00\x00",
            b"\xff\xff\xff\xff",
        ];

        let page = create_test_page_with_keys(&keys);

        let target_prefix_mid = 0x80000000u32;
        let (left, right, _) = unsafe {
            simd_prefix_search_avx2(&page, target_prefix_mid, keys.len())
        };

        assert!(
            left <= 3 && right >= 4,
            "Range [{}, {}) should include index 3 (where 0x80... is)",
            left,
            right
        );
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_neon_prefix_search() {
        let keys: Vec<&[u8]> = vec![
            b"\x00\x00\x00\x00",
            b"\x40\x00\x00\x00",
            b"\x80\x00\x00\x00",
            b"\xc0\x00\x00\x00",
        ];

        let page = create_test_page_with_keys(&keys);

        let target_prefix = 0x80000000u32;
        let (left, right, _) = unsafe {
            simd_prefix_search_neon(&page, target_prefix, keys.len())
        };

        assert!(
            left <= 2 && right >= 3,
            "Range [{}, {}) should include index 2 (where 0x80... is)",
            left,
            right
        );
    }

    #[test]
    fn test_scalar_handles_signed_boundary() {
        let keys: Vec<&[u8]> = vec![
            b"\x7f\xff\xff\xff",
            b"\x80\x00\x00\x00",
        ];

        let page = create_test_page_with_keys(&keys);

        let result = find_key_simd(&page, b"\x80\x00\x00\x00", keys.len());
        assert_eq!(
            result,
            SearchResult::Found(1),
            "0x80000000 should be found at index 1"
        );

        let result = find_key_simd(&page, b"\x7f\xff\xff\xff", keys.len());
        assert_eq!(
            result,
            SearchResult::Found(0),
            "0x7FFFFFFF should be found at index 0"
        );
    }
}
