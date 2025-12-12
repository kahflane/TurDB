//! # Freelist Management
//!
//! This module implements free page tracking and allocation for TurDB. The freelist
//! maintains a linked list of trunk pages, each containing references to free pages
//! that can be reused when new pages are needed.
//!
//! ## Design Overview
//!
//! When pages are deleted (e.g., from a dropped table or B-tree node removal), they
//! are added to the freelist rather than being reclaimed immediately. This allows
//! the database file to reuse space efficiently without file truncation.
//!
//! The freelist uses a trunk page structure where each trunk page contains:
//! - A pointer to the next trunk page (or 0 if this is the last trunk)
//! - A count of free page numbers stored in this trunk
//! - An array of free page numbers
//!
//! ## Trunk Page Layout
//!
//! ```text
//! Offset  Size      Description
//! ------  --------  ----------------------------------------
//! 0       16        Standard PageHeader (type = FreeList)
//! 16      4         next_trunk: Page number of next trunk (0 = none)
//! 20      4         count: Number of page numbers in this trunk
//! 24      4*N       page_numbers: Array of free page numbers
//! ```
//!
//! With 16KB pages and 16-byte header, each trunk can store:
//! - (16384 - 16 - 8) / 4 = 4090 page numbers
//!
//! ## Allocation Strategy
//!
//! When allocating a page:
//! 1. If the current trunk has free pages, pop one from the array
//! 2. If the current trunk is empty but has a next_trunk, move to that trunk
//! 3. If no free pages exist, return None (caller must grow the file)
//!
//! When releasing a page:
//! 1. If the current trunk has space, push the page number
//! 2. If the current trunk is full, create a new trunk page
//!
//! ## Thread Safety
//!
//! The `Freelist` struct is not thread-safe on its own. Thread safety is provided
//! by the higher-level `Pager` which holds a mutex around freelist operations.
//!
//! ## Persistence
//!
//! The freelist head page number is stored in the file header. On database open,
//! the freelist is reconstructed by reading the trunk chain from disk.
//!
//! ## Zero-Copy Design
//!
//! The freelist operates directly on mmap'd page data where possible, using
//! zerocopy for safe transmutation of trunk page headers.
//!
//! ## Memory Efficiency
//!
//! The `Freelist` struct itself is small (16 bytes), holding only:
//! - head_page: The first trunk page number
//! - free_count: Total number of free pages across all trunks
//!
//! Trunk page contents are read/written through the storage layer on demand,
//! not cached in memory beyond the page cache.

use eyre::{ensure, Result};
use zerocopy::{FromBytes, Immutable, IntoBytes, KnownLayout};

use super::{PAGE_HEADER_SIZE, PAGE_SIZE};

pub const TRUNK_HEADER_SIZE: usize = 8;
pub const TRUNK_MAX_ENTRIES: usize = (PAGE_SIZE - PAGE_HEADER_SIZE - TRUNK_HEADER_SIZE) / 4;

#[repr(C)]
#[derive(Debug, Clone, Copy, FromBytes, IntoBytes, Immutable, KnownLayout)]
pub struct TrunkHeader {
    next_trunk: u32,
    count: u32,
}

impl TrunkHeader {
    pub fn new() -> Self {
        Self {
            next_trunk: 0,
            count: 0,
        }
    }

    pub fn with_next(next_trunk: u32) -> Self {
        Self {
            next_trunk,
            count: 0,
        }
    }

    pub fn from_bytes(data: &[u8]) -> Result<&Self> {
        ensure!(
            data.len() >= size_of::<Self>(),
            "buffer too small for TrunkHeader: {} < {}",
            data.len(),
            size_of::<Self>()
        );

        Self::ref_from_bytes(&data[..size_of::<Self>()])
            .map_err(|e| eyre::eyre!("failed to read TrunkHeader: {:?}", e))
    }

    pub fn from_bytes_mut(data: &mut [u8]) -> Result<&mut Self> {
        ensure!(
            data.len() >= size_of::<Self>(),
            "buffer too small for TrunkHeader: {} < {}",
            data.len(),
            size_of::<Self>()
        );

        Self::mut_from_bytes(&mut data[..size_of::<Self>()])
            .map_err(|e| eyre::eyre!("failed to read TrunkHeader: {:?}", e))
    }

    pub fn write_to(&self, data: &mut [u8]) -> Result<()> {
        ensure!(
            data.len() >= size_of::<Self>(),
            "buffer too small for TrunkHeader: {} < {}",
            data.len(),
            size_of::<Self>()
        );

        data[..size_of::<Self>()].copy_from_slice(self.as_bytes());
        Ok(())
    }

    pub fn next_trunk(&self) -> u32 {
        self.next_trunk
    }

    pub fn set_next_trunk(&mut self, page_no: u32) {
        self.next_trunk = page_no;
    }

    pub fn count(&self) -> u32 {
        self.count
    }

    pub fn set_count(&mut self, count: u32) {
        self.count = count;
    }

    pub fn is_full(&self) -> bool {
        self.count as usize >= TRUNK_MAX_ENTRIES
    }

    pub fn is_empty(&self) -> bool {
        self.count == 0
    }
}

impl Default for TrunkHeader {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug)]
pub struct Freelist {
    head_page: u32,
    free_count: u32,
}

impl Freelist {
    pub fn new() -> Self {
        Self {
            head_page: 0,
            free_count: 0,
        }
    }

    pub fn with_head(head_page: u32, free_count: u32) -> Self {
        Self {
            head_page,
            free_count,
        }
    }

    pub fn head_page(&self) -> u32 {
        self.head_page
    }

    pub fn free_count(&self) -> u32 {
        self.free_count
    }

    pub fn is_empty(&self) -> bool {
        self.free_count == 0
    }

    pub fn set_head(&mut self, head_page: u32, free_count: u32) {
        self.head_page = head_page;
        self.free_count = free_count;
    }

    pub fn allocate(&mut self, storage: &mut super::MmapStorage) -> Result<Option<u32>> {
        if self.is_empty() {
            return Ok(None);
        }

        let page_data = storage.page_mut(self.head_page)?;
        let trunk_offset = PAGE_HEADER_SIZE;

        let (count, next_trunk) = {
            let trunk = TrunkHeader::from_bytes(&page_data[trunk_offset..])?;
            (trunk.count(), trunk.next_trunk())
        };

        if count == 0 {
            if next_trunk == 0 {
                self.head_page = 0;
                self.free_count = 0;
                return Ok(None);
            }
            self.head_page = next_trunk;
            return self.allocate(storage);
        }

        let entry_index = (count - 1) as usize;
        let entry_offset = PAGE_HEADER_SIZE + TRUNK_HEADER_SIZE + entry_index * 4;
        let page_no = u32::from_le_bytes(
            page_data[entry_offset..entry_offset + 4]
                .try_into()
                .unwrap(),
        );

        let trunk = TrunkHeader::from_bytes_mut(&mut page_data[trunk_offset..])?;
        trunk.set_count(count - 1);
        self.free_count -= 1;

        if count - 1 == 0 {
            self.head_page = next_trunk;
        }

        Ok(Some(page_no))
    }
}

impl Default for Freelist {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::{PAGE_HEADER_SIZE, PAGE_SIZE};

    #[test]
    fn trunk_header_size_is_8_bytes() {
        assert_eq!(size_of::<TrunkHeader>(), 8);
    }

    #[test]
    fn trunk_header_new_creates_empty_trunk() {
        let trunk = TrunkHeader::new();

        assert_eq!(trunk.next_trunk(), 0);
        assert_eq!(trunk.count(), 0);
    }

    #[test]
    fn trunk_header_with_next_sets_next_trunk() {
        let trunk = TrunkHeader::with_next(42);

        assert_eq!(trunk.next_trunk(), 42);
        assert_eq!(trunk.count(), 0);
    }

    #[test]
    fn trunk_header_set_next_trunk_updates_value() {
        let mut trunk = TrunkHeader::new();

        trunk.set_next_trunk(123);

        assert_eq!(trunk.next_trunk(), 123);
    }

    #[test]
    fn trunk_header_set_count_updates_value() {
        let mut trunk = TrunkHeader::new();

        trunk.set_count(500);

        assert_eq!(trunk.count(), 500);
    }

    #[test]
    fn trunk_header_from_bytes_zero_copy() {
        let mut data = [0u8; 8];
        data[0..4].copy_from_slice(&42u32.to_le_bytes());
        data[4..8].copy_from_slice(&100u32.to_le_bytes());

        let trunk = TrunkHeader::from_bytes(&data).unwrap();

        assert_eq!(trunk.next_trunk(), 42);
        assert_eq!(trunk.count(), 100);
    }

    #[test]
    fn trunk_header_from_bytes_too_small() {
        let data = [0u8; 4];
        let result = TrunkHeader::from_bytes(&data);

        assert!(result.is_err());
    }

    #[test]
    fn trunk_header_write_to() {
        let mut trunk = TrunkHeader::new();
        trunk.set_next_trunk(99);
        trunk.set_count(50);
        let mut data = [0xFFu8; 16];

        trunk.write_to(&mut data).unwrap();

        assert_eq!(&data[0..4], &99u32.to_le_bytes());
        assert_eq!(&data[4..8], &50u32.to_le_bytes());
    }

    #[test]
    fn trunk_max_entries_calculated_correctly() {
        let expected = (PAGE_SIZE - PAGE_HEADER_SIZE - size_of::<TrunkHeader>()) / 4;
        assert_eq!(TRUNK_MAX_ENTRIES, expected);
        assert_eq!(TRUNK_MAX_ENTRIES, 4090);
    }

    #[test]
    fn trunk_header_is_full() {
        let mut trunk = TrunkHeader::new();

        assert!(!trunk.is_full());

        trunk.set_count(TRUNK_MAX_ENTRIES as u32);

        assert!(trunk.is_full());
    }

    #[test]
    fn freelist_new_creates_empty_freelist() {
        let freelist = super::Freelist::new();

        assert_eq!(freelist.head_page(), 0);
        assert_eq!(freelist.free_count(), 0);
    }

    #[test]
    fn freelist_with_head_sets_head_page() {
        let freelist = super::Freelist::with_head(42, 100);

        assert_eq!(freelist.head_page(), 42);
        assert_eq!(freelist.free_count(), 100);
    }

    #[test]
    fn freelist_is_empty_when_free_count_zero() {
        let freelist = super::Freelist::new();

        assert!(freelist.is_empty());
    }

    #[test]
    fn freelist_is_not_empty_when_has_free_pages() {
        let freelist = super::Freelist::with_head(1, 10);

        assert!(!freelist.is_empty());
    }

    #[test]
    fn freelist_set_head_updates_head_page() {
        let mut freelist = super::Freelist::new();

        freelist.set_head(5, 50);

        assert_eq!(freelist.head_page(), 5);
        assert_eq!(freelist.free_count(), 50);
    }

    #[test]
    fn freelist_allocate_returns_none_when_empty() {
        let mut freelist = Freelist::new();
        let mut storage = create_test_storage(10);

        let result = freelist.allocate(&mut storage).unwrap();

        assert!(result.is_none());
    }

    #[test]
    fn freelist_allocate_returns_page_from_trunk() {
        let mut storage = create_test_storage(10);
        let trunk_page = 1;
        setup_trunk_page(&mut storage, trunk_page, 0, &[5, 6, 7]);
        let mut freelist = Freelist::with_head(trunk_page, 3);

        let page = freelist.allocate(&mut storage).unwrap();

        assert_eq!(page, Some(7));
        assert_eq!(freelist.free_count(), 2);
    }

    #[test]
    fn freelist_allocate_decrements_trunk_count() {
        let mut storage = create_test_storage(10);
        let trunk_page = 1;
        setup_trunk_page(&mut storage, trunk_page, 0, &[10, 20, 30]);
        let mut freelist = Freelist::with_head(trunk_page, 3);

        freelist.allocate(&mut storage).unwrap();

        let page_data = storage.page(trunk_page).unwrap();
        let trunk = TrunkHeader::from_bytes(&page_data[PAGE_HEADER_SIZE..]).unwrap();
        assert_eq!(trunk.count(), 2);
    }

    #[test]
    fn freelist_allocate_moves_to_next_trunk_when_empty() {
        let mut storage = create_test_storage(10);
        setup_trunk_page(&mut storage, 1, 2, &[100]);
        setup_trunk_page(&mut storage, 2, 0, &[200, 201, 202]);
        let mut freelist = Freelist::with_head(1, 4);

        let page1 = freelist.allocate(&mut storage).unwrap();
        assert_eq!(page1, Some(100));
        assert_eq!(freelist.head_page(), 2);
        assert_eq!(freelist.free_count(), 3);
    }

    #[test]
    fn freelist_allocate_multiple_pages() {
        let mut storage = create_test_storage(10);
        setup_trunk_page(&mut storage, 1, 0, &[5, 6, 7]);
        let mut freelist = Freelist::with_head(1, 3);

        let p1 = freelist.allocate(&mut storage).unwrap();
        let p2 = freelist.allocate(&mut storage).unwrap();
        let p3 = freelist.allocate(&mut storage).unwrap();
        let p4 = freelist.allocate(&mut storage).unwrap();

        assert_eq!(p1, Some(7));
        assert_eq!(p2, Some(6));
        assert_eq!(p3, Some(5));
        assert_eq!(p4, None);
        assert_eq!(freelist.free_count(), 0);
    }

    fn create_test_storage(page_count: u32) -> crate::storage::MmapStorage {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.keep().join("test.db");
        crate::storage::MmapStorage::create(&path, page_count).unwrap()
    }

    fn setup_trunk_page(
        storage: &mut crate::storage::MmapStorage,
        page_no: u32,
        next_trunk: u32,
        page_numbers: &[u32],
    ) {
        let page = storage.page_mut(page_no).unwrap();

        let header = crate::storage::PageHeader::new(crate::storage::PageType::FreeList);
        header.write_to(page).unwrap();

        let mut trunk = TrunkHeader::new();
        trunk.set_next_trunk(next_trunk);
        trunk.set_count(page_numbers.len() as u32);
        trunk.write_to(&mut page[PAGE_HEADER_SIZE..]).unwrap();

        let entries_offset = PAGE_HEADER_SIZE + TRUNK_HEADER_SIZE;
        for (i, &pn) in page_numbers.iter().enumerate() {
            let offset = entries_offset + i * 4;
            page[offset..offset + 4].copy_from_slice(&pn.to_le_bytes());
        }
    }
}
