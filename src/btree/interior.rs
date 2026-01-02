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
//! ## Suffix Truncation (Currently Disabled)
//!
//! NOTE: Suffix truncation is currently disabled because it can produce duplicate
//! separators when two leaf splits have different left_max values that both allow
//! truncation to the same prefix. Full keys are now used as separators instead.
//!
//! The `separator_len()` function (preserved for potential future use) computes 
//! the minimal prefix of right_min that is strictly greater than left_max:
//!
//! ```text
//! left_max = "apple", right_min = "banana" → separator = "b" (1 byte)
//! left_max = "abc",   right_min = "abd"    → separator = "abd" (3 bytes)
//! left_max = "test",  right_min = "testing"→ separator = "testing" (7 bytes)
//! ```
//!
//! This optimization would reduce interior node space usage, but requires
//! additional handling to ensure separator uniqueness across all splits.
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
        let header = PageHeader::from_bytes(self.data).unwrap(); // INVARIANT: page validated in from_page constructor
        header.cell_count()
    }

    pub fn right_child(&self) -> u32 {
        let header = PageHeader::from_bytes(self.data).unwrap(); // INVARIANT: page validated in from_page constructor
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
        if count == 0 {
            return Ok((self.right_child(), None));
        }

        let key_prefix = u32::from_be_bytes(extract_prefix(key));
        let mut left = 0usize;
        let mut right = count;

        while left < right {
            let mid = left + (right - left) / 2;
            let slot = self.slot_at(mid)?;
            let slot_prefix = slot.prefix_as_u32();

            match key_prefix.cmp(&slot_prefix) {
                std::cmp::Ordering::Less => right = mid,
                std::cmp::Ordering::Greater => left = mid + 1,
                std::cmp::Ordering::Equal => {
                    let separator = self.key_at(mid)?;
                    if key < separator {
                        right = mid;
                    } else {
                        left = mid + 1;
                    }
                }
            }
        }

        if left < count {
            let slot = self.slot_at(left)?;
            Ok((slot.child_page(), Some(left)))
        } else {
            Ok((self.right_child(), None))
        }
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
        let header = PageHeader::from_bytes(self.data).unwrap(); // INVARIANT: page validated in from_page/init constructor
        header.cell_count()
    }

    pub fn right_child(&self) -> u32 {
        let header = PageHeader::from_bytes(self.data).unwrap(); // INVARIANT: page validated in from_page/init constructor
        header.right_child()
    }

    fn free_start(&self) -> u16 {
        let header = PageHeader::from_bytes(self.data).unwrap(); // INVARIANT: page validated in from_page/init constructor
        header.free_start()
    }

    fn free_end(&self) -> u16 {
        let header = PageHeader::from_bytes(self.data).unwrap(); // INVARIANT: page validated in from_page/init constructor
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
        if count == 0 {
            return Ok((self.right_child(), None));
        }

        let key_prefix = u32::from_be_bytes(extract_prefix(key));
        let mut left = 0usize;
        let mut right = count;

        while left < right {
            let mid = left + (right - left) / 2;
            let slot = self.slot_at(mid)?;
            let slot_prefix = slot.prefix_as_u32();

            match key_prefix.cmp(&slot_prefix) {
                std::cmp::Ordering::Less => right = mid,
                std::cmp::Ordering::Greater => left = mid + 1,
                std::cmp::Ordering::Equal => {
                    let separator = self.key_at(mid)?;
                    if key < separator {
                        right = mid;
                    } else {
                        left = mid + 1;
                    }
                }
            }
        }

        if left < count {
            let slot = self.slot_at(left)?;
            Ok((slot.child_page(), Some(left)))
        } else {
            Ok((self.right_child(), None))
        }
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
