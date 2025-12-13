//! # B+Tree Interior Node Implementation
//!
//! This module implements interior (internal) nodes for TurDB's B+tree index.
//! Interior nodes store separator keys and child page pointers, enabling
//! efficient tree traversal from root to leaf.
//!
//! ## Design Goals
//!
//! 1. **Zero-copy access**: All key reads return slices into page memory
//! 2. **Fast navigation**: 4-byte prefix hints enable early child selection
//! 3. **Space efficiency**: Suffix truncation minimizes separator key storage
//! 4. **Inline child pointers**: Child page numbers stored in slot array
//!
//! ## Interior Slot Architecture
//!
//! Unlike leaf nodes (8-byte slots), interior nodes use 12-byte slots with
//! inline child pointers for faster navigation:
//!
//! ```text
//! InteriorSlot (12 bytes):
//! +--------+--------+--------+--------+--------+--------+--------+--------+
//! |      prefix (4 bytes)             |     child_page (4 bytes)          |
//! +--------+--------+--------+--------+--------+--------+--------+--------+
//! |    offset (2B)  |   key_len (2B)  |
//! +--------+--------+--------+--------+
//! ```
//!
//! - **prefix**: First 4 bytes of separator key for fast comparison
//! - **child_page**: Left child page number (keys < separator go here)
//! - **offset**: Byte offset to full separator key in cell area
//! - **key_len**: Length of full separator key
//!
//! ## Page Layout
//!
//! ```text
//! +----------------------------------+
//! | PageHeader (16 bytes)            |
//! |   - page_type = BTreeInterior    |
//! |   - cell_count = N               |
//! |   - right_child = rightmost ptr  |
//! +----------------------------------+
//! | InteriorSlot[0]  (12 bytes)      |  ← keys < separator[0] go to child[0]
//! | InteriorSlot[1]  (12 bytes)      |  ← separator[0] <= keys < separator[1]
//! | ...                              |
//! | InteriorSlot[N-1] (12 bytes)     |
//! +----------------------------------+
//! | Free space                       |
//! +----------------------------------+
//! | Cell content (separator keys)    |  ← grows upward from page end
//! +----------------------------------+
//! ```
//!
//! ## Navigation Semantics
//!
//! For a search key K:
//! - If K < separator[0]: go to slot[0].child_page
//! - If separator[i-1] <= K < separator[i]: go to slot[i].child_page
//! - If K >= separator[N-1]: go to header.right_child
//!
//! The `right_child` in PageHeader handles keys >= all separators.
//!
//! ## Suffix Truncation
//!
//! When splitting a leaf node, the separator key promoted to the parent
//! need only be long enough to distinguish the two children. The
//! `separator_len()` function computes the minimal prefix of right_min
//! that is strictly greater than left_max:
//!
//! ```text
//! left_max = "apple", right_min = "banana" → separator = "b" (1 byte)
//! left_max = "abc",   right_min = "abd"    → separator = "abd" (3 bytes)
//! left_max = "test",  right_min = "testing"→ separator = "testing" (7 bytes)
//! ```
//!
//! This optimization significantly reduces interior node space usage,
//! allowing more children per interior node and shallower trees.
//!
//! ## Capacity Calculation
//!
//! With 16KB pages and 12-byte slots:
//! - Page header: 16 bytes
//! - Usable for slots + cells: 16368 bytes
//! - With average 4-byte truncated separators: ~800+ children per node
//! - With full 100-byte keys: ~100 children per node
//!
//! ## Zero-Copy Guarantees
//!
//! ```text
//! struct InteriorNode<'a> {
//!     data: &'a [u8],  // Borrowed from page buffer
//! }
//! ```
//!
//! Methods return &'a [u8] slices pointing directly into the page.
//!
//! ## Thread Safety
//!
//! InteriorNode is not thread-safe. External synchronization required
//! (typically via PageCache's sharded RwLocks).

use eyre::{bail, ensure, Result};
use zerocopy::{
    byteorder::{LittleEndian, U16, U32},
    FromBytes, Immutable, IntoBytes, KnownLayout, Unaligned,
};

use crate::btree::extract_prefix;
use crate::storage::{PageHeader, PageType, PAGE_HEADER_SIZE, PAGE_SIZE};

pub const INTERIOR_SLOT_SIZE: usize = 12;
pub const INTERIOR_CONTENT_START: usize = PAGE_HEADER_SIZE;

#[repr(C)]
#[derive(
    Debug, Clone, Copy, FromBytes, IntoBytes, Immutable, KnownLayout, Unaligned, PartialEq, Eq,
)]
pub struct InteriorSlot {
    pub prefix: [u8; 4],
    pub child_page: U32<LittleEndian>,
    pub offset: U16<LittleEndian>,
    pub key_len: U16<LittleEndian>,
}

impl InteriorSlot {
    pub fn new(key: &[u8], child_page: u32, offset: u16) -> Self {
        Self {
            prefix: extract_prefix(key),
            child_page: U32::new(child_page),
            offset: U16::new(offset),
            key_len: U16::new(key.len() as u16),
        }
    }

    pub fn prefix_as_u32(&self) -> u32 {
        u32::from_be_bytes(self.prefix)
    }

    pub fn child_page(&self) -> u32 {
        self.child_page.get()
    }

    pub fn offset(&self) -> u16 {
        self.offset.get()
    }

    pub fn key_len(&self) -> u16 {
        self.key_len.get()
    }
}

pub fn separator_len(left_max: &[u8], right_min: &[u8]) -> usize {
    for len in 1..=right_min.len() {
        if &right_min[..len] > left_max {
            return len;
        }
    }
    right_min.len()
}

#[derive(Debug)]
pub struct InteriorNode<'a> {
    data: &'a [u8],
}

pub struct InteriorNodeMut<'a> {
    data: &'a mut [u8],
}

impl<'a> InteriorNode<'a> {
    pub fn from_page(data: &'a [u8]) -> Result<Self> {
        ensure!(
            data.len() == PAGE_SIZE,
            "invalid page size: {} != {}",
            data.len(),
            PAGE_SIZE
        );
        let header = PageHeader::from_bytes(data)?;
        ensure!(
            header.page_type() == PageType::BTreeInterior,
            "expected BTreeInterior page, got {:?}",
            header.page_type()
        );
        Ok(Self { data })
    }

    pub fn cell_count(&self) -> u16 {
        let header = PageHeader::from_bytes(self.data).unwrap();
        header.cell_count()
    }

    pub fn right_child(&self) -> u32 {
        let header = PageHeader::from_bytes(self.data).unwrap();
        header.right_child()
    }

    fn slot_offset(&self, index: usize) -> usize {
        INTERIOR_CONTENT_START + index * INTERIOR_SLOT_SIZE
    }

    pub fn slot_at(&self, index: usize) -> Result<&InteriorSlot> {
        ensure!(
            index < self.cell_count() as usize,
            "slot index {} out of bounds (cell_count={})",
            index,
            self.cell_count()
        );
        let offset = self.slot_offset(index);
        InteriorSlot::ref_from_bytes(&self.data[offset..offset + INTERIOR_SLOT_SIZE])
            .map_err(|e| eyre::eyre!("failed to read interior slot at index {}: {:?}", index, e))
    }

    pub fn key_at(&self, index: usize) -> Result<&'a [u8]> {
        let slot = self.slot_at(index)?;
        let cell_offset = slot.offset() as usize;
        let key_len = slot.key_len() as usize;

        ensure!(
            cell_offset + key_len <= PAGE_SIZE,
            "key extends beyond page boundary: offset={}, key_len={}",
            cell_offset,
            key_len
        );

        Ok(&self.data[cell_offset..cell_offset + key_len])
    }

    pub fn find_child(&self, key: &[u8]) -> Result<(u32, Option<usize>)> {
        let count = self.cell_count() as usize;
        let key_prefix = u32::from_be_bytes(extract_prefix(key));

        for i in 0..count {
            let slot = self.slot_at(i)?;
            let slot_prefix = slot.prefix_as_u32();

            if key_prefix < slot_prefix {
                return Ok((slot.child_page(), Some(i)));
            }
            if key_prefix == slot_prefix {
                let separator = self.key_at(i)?;
                if key < separator {
                    return Ok((slot.child_page(), Some(i)));
                }
            }
        }
        Ok((self.right_child(), None))
    }
}

impl<'a> InteriorNodeMut<'a> {
    pub fn from_page(data: &'a mut [u8]) -> Result<Self> {
        ensure!(
            data.len() == PAGE_SIZE,
            "invalid page size: {} != {}",
            data.len(),
            PAGE_SIZE
        );
        let header = PageHeader::from_bytes(data)?;
        ensure!(
            header.page_type() == PageType::BTreeInterior,
            "expected BTreeInterior page, got {:?}",
            header.page_type()
        );
        Ok(Self { data })
    }

    pub fn init(data: &'a mut [u8], right_child: u32) -> Result<Self> {
        ensure!(
            data.len() == PAGE_SIZE,
            "invalid page size: {} != {}",
            data.len(),
            PAGE_SIZE
        );

        let header = PageHeader::from_bytes_mut(data)?;
        header.set_page_type(PageType::BTreeInterior);
        header.set_cell_count(0);
        header.set_free_start(INTERIOR_CONTENT_START as u16);
        header.set_free_end(PAGE_SIZE as u16);
        header.set_frag_bytes(0);
        header.set_right_child(right_child);

        Ok(Self { data })
    }

    pub fn cell_count(&self) -> u16 {
        let header = PageHeader::from_bytes(self.data).unwrap();
        header.cell_count()
    }

    pub fn right_child(&self) -> u32 {
        let header = PageHeader::from_bytes(self.data).unwrap();
        header.right_child()
    }

    fn free_start(&self) -> u16 {
        let header = PageHeader::from_bytes(self.data).unwrap();
        header.free_start()
    }

    fn free_end(&self) -> u16 {
        let header = PageHeader::from_bytes(self.data).unwrap();
        header.free_end()
    }

    pub fn free_space(&self) -> u16 {
        self.free_end() - self.free_start()
    }

    fn slot_offset(&self, index: usize) -> usize {
        INTERIOR_CONTENT_START + index * INTERIOR_SLOT_SIZE
    }

    pub fn slot_at(&self, index: usize) -> Result<&InteriorSlot> {
        ensure!(
            index < self.cell_count() as usize,
            "slot index {} out of bounds (cell_count={})",
            index,
            self.cell_count()
        );
        let offset = self.slot_offset(index);
        InteriorSlot::ref_from_bytes(&self.data[offset..offset + INTERIOR_SLOT_SIZE])
            .map_err(|e| eyre::eyre!("failed to read interior slot at index {}: {:?}", index, e))
    }

    pub fn key_at(&self, index: usize) -> Result<&[u8]> {
        let slot = self.slot_at(index)?;
        let cell_offset = slot.offset() as usize;
        let key_len = slot.key_len() as usize;

        ensure!(
            cell_offset + key_len <= PAGE_SIZE,
            "key extends beyond page boundary"
        );

        Ok(&self.data[cell_offset..cell_offset + key_len])
    }

    pub fn find_child(&self, key: &[u8]) -> Result<(u32, Option<usize>)> {
        let count = self.cell_count() as usize;
        let key_prefix = u32::from_be_bytes(extract_prefix(key));

        for i in 0..count {
            let slot = self.slot_at(i)?;
            let slot_prefix = slot.prefix_as_u32();

            if key_prefix < slot_prefix {
                return Ok((slot.child_page(), Some(i)));
            }
            if key_prefix == slot_prefix {
                let separator = self.key_at(i)?;
                if key < separator {
                    return Ok((slot.child_page(), Some(i)));
                }
            }
        }
        Ok((self.right_child(), None))
    }

    pub fn insert_separator(&mut self, key: &[u8], left_child: u32) -> Result<()> {
        let space_needed = key.len() + INTERIOR_SLOT_SIZE;

        ensure!(
            self.free_space() as usize >= space_needed,
            "not enough free space: need {}, have {}",
            space_needed,
            self.free_space()
        );

        let insert_pos = self.find_insert_position(key)?;

        let new_free_end = self.free_end() as usize - key.len();
        self.data[new_free_end..new_free_end + key.len()].copy_from_slice(key);

        let cell_count = self.cell_count() as usize;
        for i in (insert_pos..cell_count).rev() {
            let src_offset = self.slot_offset(i);
            let dst_offset = self.slot_offset(i + 1);
            self.data
                .copy_within(src_offset..src_offset + INTERIOR_SLOT_SIZE, dst_offset);
        }

        let slot = InteriorSlot::new(key, left_child, new_free_end as u16);
        let slot_offset = self.slot_offset(insert_pos);
        self.data[slot_offset..slot_offset + INTERIOR_SLOT_SIZE].copy_from_slice(slot.as_bytes());

        let header = PageHeader::from_bytes_mut(self.data)?;
        header.set_cell_count(cell_count as u16 + 1);
        header.set_free_start(header.free_start() + INTERIOR_SLOT_SIZE as u16);
        header.set_free_end(new_free_end as u16);

        Ok(())
    }

    fn find_insert_position(&self, key: &[u8]) -> Result<usize> {
        let count = self.cell_count() as usize;
        let key_prefix = u32::from_be_bytes(extract_prefix(key));

        for i in 0..count {
            let slot = self.slot_at(i)?;
            let slot_prefix = slot.prefix_as_u32();

            if key_prefix < slot_prefix {
                return Ok(i);
            }
            if key_prefix == slot_prefix {
                let separator = self.key_at(i)?;
                if key < separator {
                    return Ok(i);
                }
                if key == separator {
                    bail!("separator key already exists");
                }
            }
        }
        Ok(count)
    }

    pub fn set_right_child(&mut self, page_no: u32) -> Result<()> {
        let header = PageHeader::from_bytes_mut(self.data)?;
        header.set_right_child(page_no);
        Ok(())
    }

    pub fn as_ref(&self) -> InteriorNode<'_> {
        InteriorNode { data: self.data }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_page() -> Vec<u8> {
        vec![0u8; PAGE_SIZE]
    }

    #[test]
    fn interior_slot_size_is_12_bytes() {
        assert_eq!(size_of::<InteriorSlot>(), 12);
    }

    #[test]
    fn interior_slot_new_extracts_prefix() {
        let slot = InteriorSlot::new(b"hello", 42, 100);

        assert_eq!(slot.prefix, [b'h', b'e', b'l', b'l']);
        assert_eq!(slot.child_page(), 42);
        assert_eq!(slot.offset(), 100);
        assert_eq!(slot.key_len(), 5);
    }

    #[test]
    fn interior_slot_short_key_pads_with_zeros() {
        let slot = InteriorSlot::new(b"ab", 10, 200);

        assert_eq!(slot.prefix, [b'a', b'b', 0, 0]);
        assert_eq!(slot.key_len(), 2);
    }

    #[test]
    fn separator_len_no_common_prefix() {
        assert_eq!(separator_len(b"apple", b"banana"), 1);
    }

    #[test]
    fn separator_len_common_prefix() {
        assert_eq!(separator_len(b"abc", b"abd"), 3);
    }

    #[test]
    fn separator_len_prefix_relationship() {
        assert_eq!(separator_len(b"test", b"testing"), 5);
    }

    #[test]
    fn separator_len_single_char_difference() {
        assert_eq!(separator_len(b"a", b"b"), 1);
    }

    #[test]
    fn separator_len_longer_common_prefix() {
        assert_eq!(separator_len(b"prefix_aaa", b"prefix_bbb"), 8);
    }

    #[test]
    fn interior_node_init_sets_correct_header() {
        let mut page = make_page();
        let node = InteriorNodeMut::init(&mut page, 99).unwrap();

        assert_eq!(node.cell_count(), 0);
        assert_eq!(node.right_child(), 99);
        assert_eq!(node.free_start(), INTERIOR_CONTENT_START as u16);
        assert_eq!(node.free_end(), PAGE_SIZE as u16);
    }

    #[test]
    fn interior_node_insert_single_separator() {
        let mut page = make_page();
        let mut node = InteriorNodeMut::init(&mut page, 100).unwrap();

        node.insert_separator(b"middle", 50).unwrap();

        assert_eq!(node.cell_count(), 1);
        assert_eq!(node.key_at(0).unwrap(), b"middle");
        assert_eq!(node.slot_at(0).unwrap().child_page(), 50);
        assert_eq!(node.right_child(), 100);
    }

    #[test]
    fn interior_node_insert_maintains_sorted_order() {
        let mut page = make_page();
        let mut node = InteriorNodeMut::init(&mut page, 100).unwrap();

        node.insert_separator(b"charlie", 30).unwrap();
        node.insert_separator(b"alpha", 10).unwrap();
        node.insert_separator(b"bravo", 20).unwrap();

        assert_eq!(node.cell_count(), 3);
        assert_eq!(node.key_at(0).unwrap(), b"alpha");
        assert_eq!(node.key_at(1).unwrap(), b"bravo");
        assert_eq!(node.key_at(2).unwrap(), b"charlie");

        assert_eq!(node.slot_at(0).unwrap().child_page(), 10);
        assert_eq!(node.slot_at(1).unwrap().child_page(), 20);
        assert_eq!(node.slot_at(2).unwrap().child_page(), 30);
    }

    #[test]
    fn interior_node_find_child_less_than_first_separator() {
        let mut page = make_page();
        let mut node = InteriorNodeMut::init(&mut page, 100).unwrap();

        node.insert_separator(b"delta", 40).unwrap();
        node.insert_separator(b"bravo", 20).unwrap();

        let (child, slot_idx) = node.find_child(b"alpha").unwrap();
        assert_eq!(child, 20);
        assert_eq!(slot_idx, Some(0));
    }

    #[test]
    fn interior_node_find_child_between_separators() {
        let mut page = make_page();
        let mut node = InteriorNodeMut::init(&mut page, 100).unwrap();

        node.insert_separator(b"alpha", 10).unwrap();
        node.insert_separator(b"charlie", 30).unwrap();

        let (child, slot_idx) = node.find_child(b"bravo").unwrap();
        assert_eq!(child, 30);
        assert_eq!(slot_idx, Some(1));
    }

    #[test]
    fn interior_node_find_child_greater_than_all_separators() {
        let mut page = make_page();
        let mut node = InteriorNodeMut::init(&mut page, 100).unwrap();

        node.insert_separator(b"alpha", 10).unwrap();
        node.insert_separator(b"bravo", 20).unwrap();

        let (child, slot_idx) = node.find_child(b"zulu").unwrap();
        assert_eq!(child, 100);
        assert_eq!(slot_idx, None);
    }

    #[test]
    fn interior_node_find_child_exact_match_goes_to_right() {
        let mut page = make_page();
        let mut node = InteriorNodeMut::init(&mut page, 100).unwrap();

        node.insert_separator(b"bravo", 20).unwrap();

        let (child, slot_idx) = node.find_child(b"bravo").unwrap();
        assert_eq!(child, 100);
        assert_eq!(slot_idx, None);
    }

    #[test]
    fn interior_node_find_child_with_prefix_collision() {
        let mut page = make_page();
        let mut node = InteriorNodeMut::init(&mut page, 100).unwrap();

        node.insert_separator(b"test1", 10).unwrap();
        node.insert_separator(b"test3", 30).unwrap();

        let (child, slot_idx) = node.find_child(b"test2").unwrap();
        assert_eq!(child, 30);
        assert_eq!(slot_idx, Some(1));
    }

    #[test]
    fn interior_node_from_page_validates_page_type() {
        let mut page = make_page();

        PageHeader::new(PageType::BTreeLeaf)
            .write_to(&mut page)
            .unwrap();

        let result = InteriorNode::from_page(&page);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("BTreeInterior"));
    }

    #[test]
    fn interior_node_zero_copy_key_access() {
        let mut page = make_page();
        let mut node = InteriorNodeMut::init(&mut page, 100).unwrap();

        node.insert_separator(b"mykey", 50).unwrap();

        let key = node.key_at(0).unwrap();
        let key_ptr = key.as_ptr();
        let page_ptr = page.as_ptr();

        assert!(key_ptr >= page_ptr && key_ptr < unsafe { page_ptr.add(PAGE_SIZE) });
    }

    #[test]
    fn interior_node_set_right_child() {
        let mut page = make_page();
        let mut node = InteriorNodeMut::init(&mut page, 100).unwrap();

        node.set_right_child(999).unwrap();

        assert_eq!(node.right_child(), 999);
    }

    #[test]
    fn interior_node_insert_duplicate_fails() {
        let mut page = make_page();
        let mut node = InteriorNodeMut::init(&mut page, 100).unwrap();

        node.insert_separator(b"key", 10).unwrap();
        let result = node.insert_separator(b"key", 20);

        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("already exists"));
    }

    #[test]
    fn interior_node_many_separators() {
        let mut page = make_page();
        let mut node = InteriorNodeMut::init(&mut page, 1000).unwrap();

        for i in 0..100 {
            let key = format!("key{:03}", i);
            node.insert_separator(key.as_bytes(), i).unwrap();
        }

        assert_eq!(node.cell_count(), 100);

        for i in 0..100 {
            let expected_key = format!("key{:03}", i);
            assert_eq!(node.key_at(i).unwrap(), expected_key.as_bytes());
            assert_eq!(node.slot_at(i).unwrap().child_page(), i as u32);
        }
    }

    #[test]
    fn interior_node_immutable_from_mutable() {
        let mut page = make_page();
        let mut node = InteriorNodeMut::init(&mut page, 100).unwrap();
        node.insert_separator(b"test", 50).unwrap();

        let immutable = node.as_ref();
        assert_eq!(immutable.cell_count(), 1);
        assert_eq!(immutable.key_at(0).unwrap(), b"test");
    }
}
