//! # B+Tree Leaf Node Implementation
//!
//! This module implements leaf nodes for TurDB's B+tree index. Leaf nodes store
//! the actual key-value pairs and are linked together for efficient range scans.
//!
//! ## Design Goals
//!
//! 1. **Zero-copy access**: All key/value reads return slices into page memory
//! 2. **Fast key search**: 4-byte prefix hints enable early rejection
//! 3. **Cache efficiency**: Slot array is compact and sequential
//! 4. **Minimal fragmentation**: Cell compaction on delete
//!
//! ## Slot Array Architecture
//!
//! The slot array stores metadata about each key-value cell:
//!
//! ```text
//! Slot (8 bytes):
//! +--------+--------+--------+--------+--------+--------+--------+--------+
//! |      prefix (4 bytes)             | offset (2B)     | key_len (2B)    |
//! +--------+--------+--------+--------+--------+--------+--------+--------+
//! ```
//!
//! - **prefix**: First 4 bytes of the key, enabling fast comparison
//! - **offset**: Byte offset to cell content within the page
//! - **key_len**: Length of the full key (for bounds checking)
//!
//! ## Cell Content Layout
//!
//! Each cell stores key and value data:
//!
//! ```text
//! Cell:
//! +------------------+------------------+------------------+
//! | key (key_len B)  | value_len (var)  | value (N bytes)  |
//! +------------------+------------------+------------------+
//! ```
//!
//! The key length is stored in the slot, not the cell, to avoid reading
//! cell content during prefix-only searches.
//!
//! ## Search Algorithm
//!
//! The find_key algorithm proceeds in two phases:
//!
//! ### Phase 1: Prefix Search
//! ```text
//! 1. Extract 4-byte prefix from search key (pad with 0 if shorter)
//! 2. Linear scan through slot prefixes (could be binary search for large nodes)
//! 3. Compare prefix as big-endian u32 for byte-order correctness
//! ```
//!
//! ### Phase 2: Full Key Comparison
//! ```text
//! 1. On prefix match or first prefix > target, read full key from cell
//! 2. Compare full keys byte-by-byte
//! 3. Return Found(i) or NotFound(insertion_point)
//! ```
//!
//! ## Insertion Algorithm
//!
//! ```text
//! 1. Find insertion point via find_key()
//! 2. Check if enough free space exists
//! 3. Allocate cell content from end of page (grows upward)
//! 4. Write key and value to cell
//! 5. Shift slot array to make room at insertion point
//! 6. Write new slot with prefix and offset
//! 7. Update header (cell_count, free_end)
//! ```
//!
//! ## Deletion Algorithm
//!
//! ```text
//! 1. Find key position via find_key()
//! 2. Remove slot from array (shift remaining slots left)
//! 3. Add freed cell space to fragment count
//! 4. If fragmentation exceeds threshold, compact the page
//! 5. Update header (cell_count, frag_bytes)
//! ```
//!
//! ## Compaction
//!
//! When fragmented space exceeds a threshold (e.g., 25% of page), compact:
//!
//! ```text
//! 1. Collect all valid cells referenced by slots
//! 2. Rewrite cells contiguously from end of page
//! 3. Update slot offsets to point to new locations
//! 4. Reset free_end and frag_bytes
//! ```
//!
//! ## Zero-Copy Guarantees
//!
//! The LeafNode struct borrows from page data with lifetime 'a:
//!
//! ```text
//! struct LeafNode<'a> {
//!     data: &'a [u8],  // Borrowed from page buffer
//! }
//! ```
//!
//! Methods return &'a [u8] slices that point directly into the page:
//! - key_at() -> &'a [u8]
//! - value_at() -> &'a [u8]
//!
//! This ensures no data copying during reads.
//!
//! ## Page Layout Example
//!
//! 16KB page with 3 cells:
//!
//! ```text
//! Offset    Content
//! ------    -------
//! 0         Page Header (16 bytes)
//! 16        Leaf Header (8 bytes)
//! 24        Slot 0: [prefix, offset=16350, key_len=5]
//! 32        Slot 1: [prefix, offset=16320, key_len=8]
//! 40        Slot 2: [prefix, offset=16280, key_len=12]
//! 48        Free space starts (free_start = 48)
//! ...
//! 16280     Cell 2: key(12) + value_len + value
//! 16320     Cell 1: key(8) + value_len + value
//! 16350     Cell 0: key(5) + value_len + value
//! 16384     Page end (free_end starts at 16280)
//! ```
//!
//! ## Thread Safety
//!
//! LeafNode is not thread-safe. It borrows from a page buffer that must
//! be protected by external synchronization (typically the PageCache's
//! sharded RwLocks).

use eyre::{bail, ensure, Result};
use zerocopy::{FromBytes, Immutable, IntoBytes, KnownLayout};

use crate::encoding::varint::{decode_varint, encode_varint, varint_len};
use crate::storage::{PageHeader, PageType, PAGE_HEADER_SIZE, PAGE_SIZE};

pub const SLOT_SIZE: usize = 8;
pub const LEAF_HEADER_SIZE: usize = 8;
pub const LEAF_CONTENT_START: usize = PAGE_HEADER_SIZE + LEAF_HEADER_SIZE;

#[repr(C)]
#[derive(Debug, Clone, Copy, FromBytes, IntoBytes, Immutable, KnownLayout, PartialEq, Eq)]
pub struct Slot {
    pub prefix: [u8; 4],
    pub offset: u16,
    pub key_len: u16,
}

impl Slot {
    pub fn new(key: &[u8], offset: u16) -> Self {
        Self {
            prefix: extract_prefix(key),
            offset,
            key_len: key.len() as u16,
        }
    }

    pub fn prefix_as_u32(&self) -> u32 {
        u32::from_be_bytes(self.prefix)
    }
}

pub fn extract_prefix(key: &[u8]) -> [u8; 4] {
    let mut prefix = [0u8; 4];
    let len = key.len().min(4);
    prefix[..len].copy_from_slice(&key[..len]);
    prefix
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SearchResult {
    Found(usize),
    NotFound(usize),
}

#[derive(Debug)]
pub struct LeafNode<'a> {
    data: &'a [u8],
}

pub struct LeafNodeMut<'a> {
    data: &'a mut [u8],
}

impl<'a> LeafNode<'a> {
    pub fn from_page(data: &'a [u8]) -> Result<Self> {
        ensure!(
            data.len() == PAGE_SIZE,
            "invalid page size: {} != {}",
            data.len(),
            PAGE_SIZE
        );
        let header = PageHeader::from_bytes(data)?;
        ensure!(
            header.page_type() == PageType::BTreeLeaf,
            "expected BTreeLeaf page, got {:?}",
            header.page_type()
        );
        Ok(Self { data })
    }

    pub fn cell_count(&self) -> u16 {
        let header = PageHeader::from_bytes(self.data).unwrap();
        header.cell_count()
    }

    pub fn free_space(&self) -> u16 {
        let header = PageHeader::from_bytes(self.data).unwrap();
        header.free_space()
    }

    fn slot_offset(&self, index: usize) -> usize {
        LEAF_CONTENT_START + index * SLOT_SIZE
    }

    pub fn slot_at(&self, index: usize) -> Result<&Slot> {
        ensure!(
            index < self.cell_count() as usize,
            "slot index {} out of bounds (cell_count={})",
            index,
            self.cell_count()
        );
        let offset = self.slot_offset(index);
        Slot::ref_from_bytes(&self.data[offset..offset + SLOT_SIZE])
            .map_err(|e| eyre::eyre!("failed to read slot at index {}: {:?}", index, e))
    }

    pub fn key_at(&self, index: usize) -> Result<&'a [u8]> {
        let slot = self.slot_at(index)?;
        let cell_offset = slot.offset as usize;
        let key_len = slot.key_len as usize;

        ensure!(
            cell_offset + key_len <= PAGE_SIZE,
            "key extends beyond page boundary: offset={}, key_len={}",
            cell_offset,
            key_len
        );

        Ok(&self.data[cell_offset..cell_offset + key_len])
    }

    pub fn value_at(&self, index: usize) -> Result<&'a [u8]> {
        let slot = self.slot_at(index)?;
        let cell_offset = slot.offset as usize;
        let key_len = slot.key_len as usize;
        let value_start = cell_offset + key_len;

        ensure!(
            value_start < PAGE_SIZE,
            "value_len offset beyond page: {}",
            value_start
        );

        let (value_len, varint_size) = decode_varint(&self.data[value_start..])?;
        let value_data_start = value_start + varint_size;

        ensure!(
            value_data_start + value_len as usize <= PAGE_SIZE,
            "value extends beyond page boundary"
        );

        Ok(&self.data[value_data_start..value_data_start + value_len as usize])
    }

    pub fn find_key(&self, key: &[u8]) -> SearchResult {
        let target_prefix = u32::from_be_bytes(extract_prefix(key));
        let count = self.cell_count() as usize;

        for i in 0..count {
            let slot = match self.slot_at(i) {
                Ok(s) => s,
                Err(_) => return SearchResult::NotFound(i),
            };

            let slot_prefix = slot.prefix_as_u32();

            if slot_prefix > target_prefix {
                return SearchResult::NotFound(i);
            }

            if slot_prefix == target_prefix {
                let full_key = match self.key_at(i) {
                    Ok(k) => k,
                    Err(_) => return SearchResult::NotFound(i),
                };

                match full_key.cmp(key) {
                    std::cmp::Ordering::Equal => return SearchResult::Found(i),
                    std::cmp::Ordering::Greater => return SearchResult::NotFound(i),
                    std::cmp::Ordering::Less => continue,
                }
            }
        }

        SearchResult::NotFound(count)
    }

    pub fn next_leaf(&self) -> u32 {
        let header = PageHeader::from_bytes(self.data).unwrap();
        header.next_leaf()
    }
}

impl<'a> LeafNodeMut<'a> {
    pub fn from_page(data: &'a mut [u8]) -> Result<Self> {
        ensure!(
            data.len() == PAGE_SIZE,
            "invalid page size: {} != {}",
            data.len(),
            PAGE_SIZE
        );
        let header = PageHeader::from_bytes(data)?;
        ensure!(
            header.page_type() == PageType::BTreeLeaf,
            "expected BTreeLeaf page, got {:?}",
            header.page_type()
        );
        Ok(Self { data })
    }

    pub fn init(data: &'a mut [u8]) -> Result<Self> {
        ensure!(
            data.len() == PAGE_SIZE,
            "invalid page size: {} != {}",
            data.len(),
            PAGE_SIZE
        );

        let header = PageHeader::from_bytes_mut(data)?;
        header.set_page_type(PageType::BTreeLeaf);
        header.set_cell_count(0);
        header.set_free_start(LEAF_CONTENT_START as u16);
        header.set_free_end(PAGE_SIZE as u16);
        header.set_frag_bytes(0);
        header.set_next_leaf(0);

        Ok(Self { data })
    }

    pub fn cell_count(&self) -> u16 {
        let header = PageHeader::from_bytes(self.data).unwrap();
        header.cell_count()
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

    fn frag_bytes(&self) -> u8 {
        let header = PageHeader::from_bytes(self.data).unwrap();
        header.frag_bytes()
    }

    fn slot_offset(&self, index: usize) -> usize {
        LEAF_CONTENT_START + index * SLOT_SIZE
    }

    pub fn slot_at(&self, index: usize) -> Result<&Slot> {
        ensure!(
            index < self.cell_count() as usize,
            "slot index {} out of bounds (cell_count={})",
            index,
            self.cell_count()
        );
        let offset = self.slot_offset(index);
        Slot::ref_from_bytes(&self.data[offset..offset + SLOT_SIZE])
            .map_err(|e| eyre::eyre!("failed to read slot at index {}: {:?}", index, e))
    }

    pub fn key_at(&self, index: usize) -> Result<&[u8]> {
        let slot = self.slot_at(index)?;
        let cell_offset = slot.offset as usize;
        let key_len = slot.key_len as usize;

        ensure!(
            cell_offset + key_len <= PAGE_SIZE,
            "key extends beyond page boundary"
        );

        Ok(&self.data[cell_offset..cell_offset + key_len])
    }

    pub fn value_at(&self, index: usize) -> Result<&[u8]> {
        let slot = self.slot_at(index)?;
        let cell_offset = slot.offset as usize;
        let key_len = slot.key_len as usize;
        let value_start = cell_offset + key_len;

        ensure!(value_start < PAGE_SIZE, "value_len offset beyond page");

        let (value_len, varint_size) = decode_varint(&self.data[value_start..])?;
        let value_data_start = value_start + varint_size;

        ensure!(
            value_data_start + value_len as usize <= PAGE_SIZE,
            "value extends beyond page boundary"
        );

        Ok(&self.data[value_data_start..value_data_start + value_len as usize])
    }

    pub fn find_key(&self, key: &[u8]) -> SearchResult {
        let target_prefix = u32::from_be_bytes(extract_prefix(key));
        let count = self.cell_count() as usize;

        for i in 0..count {
            let slot = match self.slot_at(i) {
                Ok(s) => s,
                Err(_) => return SearchResult::NotFound(i),
            };

            let slot_prefix = slot.prefix_as_u32();

            if slot_prefix > target_prefix {
                return SearchResult::NotFound(i);
            }

            if slot_prefix == target_prefix {
                let full_key = match self.key_at(i) {
                    Ok(k) => k,
                    Err(_) => return SearchResult::NotFound(i),
                };

                match full_key.cmp(key) {
                    std::cmp::Ordering::Equal => return SearchResult::Found(i),
                    std::cmp::Ordering::Greater => return SearchResult::NotFound(i),
                    std::cmp::Ordering::Less => continue,
                }
            }
        }

        SearchResult::NotFound(count)
    }

    pub fn insert_cell(&mut self, key: &[u8], value: &[u8]) -> Result<()> {
        let value_len_size = varint_len(value.len() as u64);
        let cell_size = key.len() + value_len_size + value.len();
        let space_needed = cell_size + SLOT_SIZE;

        ensure!(
            self.free_space() as usize >= space_needed,
            "not enough free space: need {}, have {}",
            space_needed,
            self.free_space()
        );

        let insert_pos = match self.find_key(key) {
            SearchResult::Found(_) => {
                bail!("key already exists");
            }
            SearchResult::NotFound(pos) => pos,
        };

        let new_free_end = self.free_end() as usize - cell_size;
        let mut offset = new_free_end;

        self.data[offset..offset + key.len()].copy_from_slice(key);
        offset += key.len();

        offset += encode_varint(value.len() as u64, &mut self.data[offset..]);

        self.data[offset..offset + value.len()].copy_from_slice(value);

        let cell_count = self.cell_count() as usize;
        for i in (insert_pos..cell_count).rev() {
            let src_offset = self.slot_offset(i);
            let dst_offset = self.slot_offset(i + 1);
            self.data
                .copy_within(src_offset..src_offset + SLOT_SIZE, dst_offset);
        }

        let slot = Slot::new(key, new_free_end as u16);
        let slot_offset = self.slot_offset(insert_pos);
        self.data[slot_offset..slot_offset + SLOT_SIZE].copy_from_slice(slot.as_bytes());

        let header = PageHeader::from_bytes_mut(self.data)?;
        header.set_cell_count(cell_count as u16 + 1);
        header.set_free_start(header.free_start() + SLOT_SIZE as u16);
        header.set_free_end(new_free_end as u16);

        Ok(())
    }

    pub fn delete_cell(&mut self, index: usize) -> Result<()> {
        let cell_count = self.cell_count() as usize;
        ensure!(
            index < cell_count,
            "delete index {} out of bounds (cell_count={})",
            index,
            cell_count
        );

        let slot = self.slot_at(index)?;
        let cell_offset = slot.offset as usize;
        let key_len = slot.key_len as usize;
        let value_start = cell_offset + key_len;
        let (value_len, varint_size) = decode_varint(&self.data[value_start..])?;
        let cell_size = key_len + varint_size + value_len as usize;

        for i in index..cell_count - 1 {
            let src_offset = self.slot_offset(i + 1);
            let dst_offset = self.slot_offset(i);
            self.data
                .copy_within(src_offset..src_offset + SLOT_SIZE, dst_offset);
        }

        let header = PageHeader::from_bytes_mut(self.data)?;
        header.set_cell_count(cell_count as u16 - 1);
        header.set_free_start(header.free_start() - SLOT_SIZE as u16);

        let new_frag = header.frag_bytes().saturating_add(cell_size as u8);
        header.set_frag_bytes(new_frag);

        if self.should_compact() {
            self.compact()?;
        }

        Ok(())
    }

    fn should_compact(&self) -> bool {
        let frag = self.frag_bytes() as usize;
        let total_space = PAGE_SIZE - LEAF_CONTENT_START;
        frag > total_space / 4
    }

    fn compact(&mut self) -> Result<()> {
        let cell_count = self.cell_count() as usize;
        if cell_count == 0 {
            let header = PageHeader::from_bytes_mut(self.data)?;
            header.set_free_end(PAGE_SIZE as u16);
            header.set_frag_bytes(0);
            return Ok(());
        }

        let mut cells: Vec<(Slot, Vec<u8>)> = Vec::with_capacity(cell_count);

        for i in 0..cell_count {
            let slot = *self.slot_at(i)?;
            let cell_offset = slot.offset as usize;
            let key_len = slot.key_len as usize;
            let value_start = cell_offset + key_len;
            let (value_len, varint_size) = decode_varint(&self.data[value_start..])?;
            let cell_end = value_start + varint_size + value_len as usize;
            let cell_data = self.data[cell_offset..cell_end].to_vec();
            cells.push((slot, cell_data));
        }

        let mut new_free_end = PAGE_SIZE;

        for (i, (mut slot, cell_data)) in cells.into_iter().enumerate() {
            new_free_end -= cell_data.len();
            self.data[new_free_end..new_free_end + cell_data.len()].copy_from_slice(&cell_data);

            slot.offset = new_free_end as u16;
            let slot_offset = self.slot_offset(i);
            self.data[slot_offset..slot_offset + SLOT_SIZE].copy_from_slice(slot.as_bytes());
        }

        let header = PageHeader::from_bytes_mut(self.data)?;
        header.set_free_end(new_free_end as u16);
        header.set_frag_bytes(0);

        Ok(())
    }

    pub fn set_next_leaf(&mut self, page_no: u32) -> Result<()> {
        let header = PageHeader::from_bytes_mut(self.data)?;
        header.set_next_leaf(page_no);
        Ok(())
    }

    pub fn as_ref(&self) -> LeafNode<'_> {
        LeafNode { data: self.data }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_page() -> Vec<u8> {
        vec![0u8; PAGE_SIZE]
    }

    #[test]
    fn slot_size_is_8_bytes() {
        assert_eq!(size_of::<Slot>(), 8);
    }

    #[test]
    fn slot_new_extracts_prefix() {
        let slot = Slot::new(b"hello", 100);

        assert_eq!(slot.prefix, [b'h', b'e', b'l', b'l']);
        assert_eq!(slot.offset, 100);
        assert_eq!(slot.key_len, 5);
    }

    #[test]
    fn slot_new_short_key_pads_with_zeros() {
        let slot = Slot::new(b"ab", 200);

        assert_eq!(slot.prefix, [b'a', b'b', 0, 0]);
        assert_eq!(slot.key_len, 2);
    }

    #[test]
    fn extract_prefix_full_key() {
        assert_eq!(extract_prefix(b"testing"), [b't', b'e', b's', b't']);
    }

    #[test]
    fn extract_prefix_short_key() {
        assert_eq!(extract_prefix(b"xy"), [b'x', b'y', 0, 0]);
    }

    #[test]
    fn extract_prefix_empty_key() {
        assert_eq!(extract_prefix(b""), [0, 0, 0, 0]);
    }

    #[test]
    fn slot_prefix_as_u32_big_endian() {
        let slot = Slot::new(b"abcd", 0);
        let expected = u32::from_be_bytes([b'a', b'b', b'c', b'd']);
        assert_eq!(slot.prefix_as_u32(), expected);
    }

    #[test]
    fn leaf_node_init_sets_correct_header() {
        let mut page = make_page();
        let node = LeafNodeMut::init(&mut page).unwrap();

        assert_eq!(node.cell_count(), 0);
        assert_eq!(node.free_start(), LEAF_CONTENT_START as u16);
        assert_eq!(node.free_end(), PAGE_SIZE as u16);
    }

    #[test]
    fn leaf_node_insert_and_read_single_cell() {
        let mut page = make_page();
        let mut node = LeafNodeMut::init(&mut page).unwrap();

        node.insert_cell(b"key1", b"value1").unwrap();

        assert_eq!(node.cell_count(), 1);
        assert_eq!(node.key_at(0).unwrap(), b"key1");
        assert_eq!(node.value_at(0).unwrap(), b"value1");
    }

    #[test]
    fn leaf_node_insert_maintains_sorted_order() {
        let mut page = make_page();
        let mut node = LeafNodeMut::init(&mut page).unwrap();

        node.insert_cell(b"charlie", b"3").unwrap();
        node.insert_cell(b"alpha", b"1").unwrap();
        node.insert_cell(b"bravo", b"2").unwrap();

        assert_eq!(node.cell_count(), 3);
        assert_eq!(node.key_at(0).unwrap(), b"alpha");
        assert_eq!(node.key_at(1).unwrap(), b"bravo");
        assert_eq!(node.key_at(2).unwrap(), b"charlie");
    }

    #[test]
    fn leaf_node_find_key_found() {
        let mut page = make_page();
        let mut node = LeafNodeMut::init(&mut page).unwrap();

        node.insert_cell(b"apple", b"fruit").unwrap();
        node.insert_cell(b"banana", b"yellow").unwrap();
        node.insert_cell(b"cherry", b"red").unwrap();

        assert_eq!(node.find_key(b"apple"), SearchResult::Found(0));
        assert_eq!(node.find_key(b"banana"), SearchResult::Found(1));
        assert_eq!(node.find_key(b"cherry"), SearchResult::Found(2));
    }

    #[test]
    fn leaf_node_find_key_not_found() {
        let mut page = make_page();
        let mut node = LeafNodeMut::init(&mut page).unwrap();

        node.insert_cell(b"beta", b"2").unwrap();
        node.insert_cell(b"delta", b"4").unwrap();

        assert_eq!(node.find_key(b"alpha"), SearchResult::NotFound(0));
        assert_eq!(node.find_key(b"gamma"), SearchResult::NotFound(2));
        assert_eq!(node.find_key(b"omega"), SearchResult::NotFound(2));
    }

    #[test]
    fn leaf_node_find_key_with_prefix_collision() {
        let mut page = make_page();
        let mut node = LeafNodeMut::init(&mut page).unwrap();

        node.insert_cell(b"test1", b"a").unwrap();
        node.insert_cell(b"test2", b"b").unwrap();
        node.insert_cell(b"test3", b"c").unwrap();

        assert_eq!(node.find_key(b"test1"), SearchResult::Found(0));
        assert_eq!(node.find_key(b"test2"), SearchResult::Found(1));
        assert_eq!(node.find_key(b"test3"), SearchResult::Found(2));
        assert_eq!(node.find_key(b"test0"), SearchResult::NotFound(0));
        assert_eq!(node.find_key(b"test4"), SearchResult::NotFound(3));
    }

    #[test]
    fn leaf_node_delete_cell() {
        let mut page = make_page();
        let mut node = LeafNodeMut::init(&mut page).unwrap();

        node.insert_cell(b"a", b"1").unwrap();
        node.insert_cell(b"b", b"2").unwrap();
        node.insert_cell(b"c", b"3").unwrap();

        node.delete_cell(1).unwrap();

        assert_eq!(node.cell_count(), 2);
        assert_eq!(node.key_at(0).unwrap(), b"a");
        assert_eq!(node.key_at(1).unwrap(), b"c");
    }

    #[test]
    fn leaf_node_delete_first_cell() {
        let mut page = make_page();
        let mut node = LeafNodeMut::init(&mut page).unwrap();

        node.insert_cell(b"first", b"1").unwrap();
        node.insert_cell(b"second", b"2").unwrap();

        node.delete_cell(0).unwrap();

        assert_eq!(node.cell_count(), 1);
        assert_eq!(node.key_at(0).unwrap(), b"second");
    }

    #[test]
    fn leaf_node_delete_last_cell() {
        let mut page = make_page();
        let mut node = LeafNodeMut::init(&mut page).unwrap();

        node.insert_cell(b"first", b"1").unwrap();
        node.insert_cell(b"second", b"2").unwrap();

        node.delete_cell(1).unwrap();

        assert_eq!(node.cell_count(), 1);
        assert_eq!(node.key_at(0).unwrap(), b"first");
    }

    #[test]
    fn leaf_node_insert_duplicate_fails() {
        let mut page = make_page();
        let mut node = LeafNodeMut::init(&mut page).unwrap();

        node.insert_cell(b"key", b"value1").unwrap();
        let result = node.insert_cell(b"key", b"value2");

        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("already exists"));
    }

    #[test]
    fn leaf_node_zero_copy_key_access() {
        let mut page = make_page();
        let mut node = LeafNodeMut::init(&mut page).unwrap();

        node.insert_cell(b"mykey", b"myvalue").unwrap();

        let key = node.key_at(0).unwrap();
        let key_ptr = key.as_ptr();
        let page_ptr = page.as_ptr();

        assert!(key_ptr >= page_ptr && key_ptr < unsafe { page_ptr.add(PAGE_SIZE) });
    }

    #[test]
    fn leaf_node_zero_copy_value_access() {
        let mut page = make_page();
        let mut node = LeafNodeMut::init(&mut page).unwrap();

        node.insert_cell(b"mykey", b"myvalue").unwrap();

        let value = node.value_at(0).unwrap();
        let value_ptr = value.as_ptr();
        let page_ptr = page.as_ptr();

        assert!(value_ptr >= page_ptr && value_ptr < unsafe { page_ptr.add(PAGE_SIZE) });
    }

    #[test]
    fn leaf_node_large_values() {
        let mut page = make_page();
        let mut node = LeafNodeMut::init(&mut page).unwrap();

        let large_value = vec![0xAB; 1000];
        node.insert_cell(b"bigkey", &large_value).unwrap();

        assert_eq!(node.value_at(0).unwrap(), &large_value[..]);
    }

    #[test]
    fn leaf_node_many_small_cells() {
        let mut page = make_page();
        let mut node = LeafNodeMut::init(&mut page).unwrap();

        for i in 0..100 {
            let key = format!("key{:03}", i);
            let value = format!("val{:03}", i);
            node.insert_cell(key.as_bytes(), value.as_bytes()).unwrap();
        }

        assert_eq!(node.cell_count(), 100);

        for i in 0..100 {
            let expected_key = format!("key{:03}", i);
            assert_eq!(node.key_at(i).unwrap(), expected_key.as_bytes());
        }
    }

    #[test]
    fn leaf_node_from_page_validates_page_type() {
        let mut page = make_page();

        PageHeader::new(PageType::BTreeInterior)
            .write_to(&mut page)
            .unwrap();

        let result = LeafNode::from_page(&page);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("BTreeLeaf"));
    }

    #[test]
    fn leaf_node_next_leaf_pointer() {
        let mut page = make_page();
        let mut node = LeafNodeMut::init(&mut page).unwrap();

        node.set_next_leaf(42).unwrap();

        assert_eq!(node.as_ref().next_leaf(), 42);
    }
}
