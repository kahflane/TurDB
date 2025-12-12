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
}

impl Default for Freelist {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
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
}
