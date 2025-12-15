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

use super::{Storage, PAGE_HEADER_SIZE, PAGE_SIZE};

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

    pub fn allocate<S: Storage>(&mut self, storage: &mut S) -> Result<Option<u32>> {
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
        let entries_size = entry_index
            .checked_mul(4)
            .ok_or_else(|| eyre::eyre!("trunk entry index overflow: {}", entry_index))?;
        let entry_offset = (PAGE_HEADER_SIZE + TRUNK_HEADER_SIZE)
            .checked_add(entries_size)
            .ok_or_else(|| eyre::eyre!("entry offset overflow"))?;
        ensure!(
            entry_offset + 4 <= PAGE_SIZE,
            "trunk entry at offset {} exceeds page size {}",
            entry_offset,
            PAGE_SIZE
        );
        let page_no = u32::from_le_bytes(
            page_data[entry_offset..entry_offset + 4]
                .try_into()
                .map_err(|_| eyre::eyre!("failed to read page number from freelist"))?,
        );

        let trunk = TrunkHeader::from_bytes_mut(&mut page_data[trunk_offset..])?;
        trunk.set_count(count - 1);
        self.free_count -= 1;

        if count - 1 == 0 {
            self.head_page = next_trunk;
        }

        Ok(Some(page_no))
    }

    pub fn release<S: Storage>(&mut self, storage: &mut S, page_no: u32) -> Result<()> {
        if self.head_page == 0 {
            self.initialize_trunk(storage, page_no)?;
            return Ok(());
        }

        let is_full = {
            let page_data = storage.page(self.head_page)?;
            let trunk = TrunkHeader::from_bytes(&page_data[PAGE_HEADER_SIZE..])?;
            trunk.is_full()
        };

        if is_full {
            self.create_new_trunk(storage, page_no)?;
            return Ok(());
        }

        let page_data = storage.page_mut(self.head_page)?;
        let trunk_offset = PAGE_HEADER_SIZE;

        let count = {
            let trunk = TrunkHeader::from_bytes(&page_data[trunk_offset..])?;
            trunk.count()
        };

        let entries_size = (count as usize)
            .checked_mul(4)
            .ok_or_else(|| eyre::eyre!("trunk count overflow: {}", count))?;
        let entry_offset = (PAGE_HEADER_SIZE + TRUNK_HEADER_SIZE)
            .checked_add(entries_size)
            .ok_or_else(|| eyre::eyre!("entry offset overflow"))?;
        ensure!(
            entry_offset + 4 <= PAGE_SIZE,
            "trunk entry at offset {} exceeds page size {}",
            entry_offset,
            PAGE_SIZE
        );
        page_data[entry_offset..entry_offset + 4].copy_from_slice(&page_no.to_le_bytes());

        let trunk = TrunkHeader::from_bytes_mut(&mut page_data[trunk_offset..])?;
        trunk.set_count(count + 1);
        self.free_count += 1;

        Ok(())
    }

    fn create_new_trunk<S: Storage>(&mut self, storage: &mut S, page_no: u32) -> Result<()> {
        let old_head = self.head_page;

        let page_data = storage.page_mut(page_no)?;

        let header = super::PageHeader::new(super::PageType::FreeList);
        header.write_to(page_data)?;

        let mut trunk = TrunkHeader::new();
        trunk.set_next_trunk(old_head);
        trunk.write_to(&mut page_data[PAGE_HEADER_SIZE..])?;

        self.head_page = page_no;
        self.free_count += 1;

        Ok(())
    }

    pub fn sync(&self, storage: &super::MmapStorage) -> Result<()> {
        storage.sync()
    }

    fn initialize_trunk<S: Storage>(&mut self, storage: &mut S, page_no: u32) -> Result<()> {
        let page_data = storage.page_mut(page_no)?;

        let header = super::PageHeader::new(super::PageType::FreeList);
        header.write_to(page_data)?;

        let trunk = TrunkHeader::new();
        trunk.write_to(&mut page_data[PAGE_HEADER_SIZE..])?;

        self.head_page = page_no;
        self.free_count = 1;

        Ok(())
    }
}

impl Default for Freelist {
    fn default() -> Self {
        Self::new()
    }
}
