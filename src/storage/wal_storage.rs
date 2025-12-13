//! # WAL-Aware Storage Wrapper
//!
//! This module provides `WalStorage`, a storage wrapper that integrates
//! Write-Ahead Logging (WAL) at the storage layer. It wraps `MmapStorage`
//! and `Wal` together, automatically tracking dirty pages and providing
//! a method to flush modifications to the WAL.
//!
//! ## Design Philosophy
//!
//! WAL integration at the storage layer provides several benefits:
//! - Transparent logging: callers (like BTree) don't need to know about WAL
//! - Consistent tracking: all page modifications are captured automatically
//! - Clean abstraction: WAL concerns are isolated from data structure logic
//!
//! ## How It Works
//!
//! 1. When `page_mut()` is called, the page number is recorded as dirty
//! 2. The mutable page slice is returned for modification
//! 3. When `flush_wal()` is called, all dirty pages are written to the WAL
//! 4. The dirty page set is cleared after successful flush
//!
//! ## Usage Pattern
//!
//! ```ignore
//! let mut wal_storage = WalStorage::new(&mut storage, &mut wal);
//!
//! // Perform operations that modify pages
//! {
//!     let mut btree = BTree::new(&mut wal_storage, root_page)?;
//!     btree.insert(key, value)?;
//! }
//!
//! // Flush all modifications to WAL
//! wal_storage.flush_wal()?;
//! ```
//!
//! ## Dirty Page Tracking
//!
//! The wrapper uses a `HashSet<u32>` to track which pages have been accessed
//! via `page_mut()`. This is a conservative approach - a page is marked dirty
//! even if no actual modifications occur. This is safe because:
//!
//! - Writing unchanged data to WAL is correct (just wasteful)
//! - The overhead is minimal for typical workloads
//! - It avoids complex change detection logic
//!
//! ## Thread Safety
//!
//! `WalStorage` borrows both storage and WAL mutably, so it cannot be shared
//! across threads. For concurrent access, use higher-level synchronization
//! (e.g., `RwLock<Database>`).
//!
//! ## Recovery
//!
//! On database open, call `Wal::recover()` to replay WAL frames to storage.
//! This ensures durability: even if the process crashes after `flush_wal()`
//! but before the main database file is synced, recovery will restore the data.
//!
//! ## Performance Considerations
//!
//! - `flush_wal()` reads each dirty page and writes to WAL sequentially
//! - WAL writes are synchronous (fsync after each frame) for durability
//! - For bulk inserts, consider batching and calling flush_wal() less frequently
//!
//! ## Checkpoint Integration
//!
//! After WAL grows beyond a threshold, call `Wal::checkpoint()` to:
//! 1. Apply WAL frames to the main database
//! 2. Truncate the WAL file
//! 3. Free up disk space and improve read performance

use super::{MmapStorage, Storage, Wal, PAGE_SIZE};
use eyre::{ensure, Result, WrapErr};
use hashbrown::HashSet;
use parking_lot::Mutex;

pub struct WalStorage<'a> {
    storage: &'a mut MmapStorage,
    dirty_pages: &'a Mutex<HashSet<u32>>,
}

impl<'a> WalStorage<'a> {
    pub fn new(storage: &'a mut MmapStorage, dirty_pages: &'a Mutex<HashSet<u32>>) -> Self {
        Self {
            storage,
            dirty_pages,
        }
    }

    pub fn flush_wal(
        dirty_pages: &Mutex<HashSet<u32>>,
        storage: &MmapStorage,
        wal: &mut Wal,
    ) -> Result<u32> {
        let dirty: Vec<u32> = {
            let mut guard = dirty_pages.lock();
            if guard.is_empty() {
                return Ok(0);
            }
            guard.drain().collect()
        };

        let db_size = storage.page_count();
        let mut frames_written = 0;

        for page_no in dirty {
            let page_data = storage.page(page_no)?;

            ensure!(
                page_data.len() == PAGE_SIZE,
                "page {} has unexpected size {} (expected {})",
                page_no,
                page_data.len(),
                PAGE_SIZE
            );

            wal.write_frame(page_no, db_size, page_data)
                .wrap_err_with(|| format!("failed to write page {} to WAL", page_no))?;

            frames_written += 1;
        }

        Ok(frames_written)
    }

    pub fn dirty_page_count(&self) -> usize {
        self.dirty_pages.lock().len()
    }

    pub fn inner_storage(&self) -> &MmapStorage {
        self.storage
    }
}

impl<'a> Storage for WalStorage<'a> {
    fn page(&self, page_no: u32) -> Result<&[u8]> {
        self.storage.page(page_no)
    }

    fn page_mut(&mut self, page_no: u32) -> Result<&mut [u8]> {
        self.dirty_pages.lock().insert(page_no);
        self.storage.page_mut(page_no)
    }

    fn grow(&mut self, new_page_count: u32) -> Result<()> {
        self.storage.grow(new_page_count)
    }

    fn page_count(&self) -> u32 {
        self.storage.page_count()
    }

    fn sync(&self) -> Result<()> {
        self.storage.sync()
    }

    fn prefetch_pages(&self, start_page: u32, count: u32) {
        self.storage.prefetch_pages(start_page, count)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    fn create_test_storage(dir: &std::path::Path) -> (MmapStorage, Wal) {
        let db_path = dir.join("test.tbd");
        let wal_dir = dir.join("wal");

        let storage = MmapStorage::create(&db_path, 10).expect("should create storage");
        let wal = Wal::create(&wal_dir).expect("should create WAL");

        (storage, wal)
    }

    #[test]
    fn wal_storage_tracks_dirty_pages() {
        let dir = tempdir().expect("should create temp dir");
        let (mut storage, _wal) = create_test_storage(dir.path());
        let dirty_pages = Mutex::new(HashSet::new());

        let mut wal_storage = WalStorage::new(&mut storage, &dirty_pages);

        assert_eq!(wal_storage.dirty_page_count(), 0);

        {
            let page = wal_storage.page_mut(1).expect("should get page");
            page[0] = 42;
        }

        assert_eq!(wal_storage.dirty_page_count(), 1);

        {
            let page = wal_storage.page_mut(2).expect("should get page");
            page[0] = 43;
        }

        assert_eq!(wal_storage.dirty_page_count(), 2);

        {
            let page = wal_storage.page_mut(1).expect("should get same page");
            page[1] = 44;
        }

        assert_eq!(wal_storage.dirty_page_count(), 2);
    }

    #[test]
    fn wal_storage_flush_writes_to_wal() {
        let dir = tempdir().expect("should create temp dir");
        let (mut storage, mut wal) = create_test_storage(dir.path());
        let dirty_pages = Mutex::new(HashSet::new());

        {
            let mut wal_storage = WalStorage::new(&mut storage, &dirty_pages);

            {
                let page = wal_storage.page_mut(1).expect("should get page");
                page[0] = 0xAB;
                page[1] = 0xCD;
            }

            {
                let page = wal_storage.page_mut(3).expect("should get page");
                page[0] = 0xEF;
            }
        }

        let frames_written =
            WalStorage::flush_wal(&dirty_pages, &storage, &mut wal).expect("should flush WAL");

        assert_eq!(frames_written, 2);
        assert_eq!(dirty_pages.lock().len(), 0);
    }

    #[test]
    fn wal_storage_flush_empty_returns_zero() {
        let dir = tempdir().expect("should create temp dir");
        let (storage, mut wal) = create_test_storage(dir.path());
        let dirty_pages = Mutex::new(HashSet::new());

        let frames_written =
            WalStorage::flush_wal(&dirty_pages, &storage, &mut wal).expect("should flush WAL");

        assert_eq!(frames_written, 0);
    }

    #[test]
    fn wal_storage_read_does_not_mark_dirty() {
        let dir = tempdir().expect("should create temp dir");
        let (mut storage, _wal) = create_test_storage(dir.path());
        let dirty_pages = Mutex::new(HashSet::new());

        let wal_storage = WalStorage::new(&mut storage, &dirty_pages);

        let _page = wal_storage.page(1).expect("should read page");

        assert_eq!(wal_storage.dirty_page_count(), 0);
    }

    #[test]
    fn wal_storage_grow_works() {
        let dir = tempdir().expect("should create temp dir");
        let (mut storage, _wal) = create_test_storage(dir.path());
        let dirty_pages = Mutex::new(HashSet::new());

        let mut wal_storage = WalStorage::new(&mut storage, &dirty_pages);

        assert_eq!(wal_storage.page_count(), 10);

        wal_storage.grow(20).expect("should grow");

        assert_eq!(wal_storage.page_count(), 20);
    }

    #[test]
    fn wal_storage_recovery_restores_data() {
        let dir = tempdir().expect("should create temp dir");
        let db_path = dir.path().join("test.tbd");
        let wal_dir = dir.path().join("wal");

        {
            let mut storage = MmapStorage::create(&db_path, 10).expect("should create storage");
            let mut wal = Wal::create(&wal_dir).expect("should create WAL");
            let dirty_pages = Mutex::new(HashSet::new());

            {
                let mut wal_storage = WalStorage::new(&mut storage, &dirty_pages);

                {
                    let page = wal_storage.page_mut(5).expect("should get page");
                    page[100] = 0xDE;
                    page[101] = 0xAD;
                    page[102] = 0xBE;
                    page[103] = 0xEF;
                }
            }

            WalStorage::flush_wal(&dirty_pages, &storage, &mut wal).expect("should flush WAL");
        }

        {
            let mut storage = MmapStorage::create(&db_path, 10).expect("should create new storage");
            let mut wal = Wal::open(&wal_dir).expect("should open WAL");

            let frames_recovered = wal.recover(&mut storage).expect("should recover from WAL");

            assert_eq!(frames_recovered, 1);

            let page = storage.page(5).expect("should read page");
            assert_eq!(page[100], 0xDE);
            assert_eq!(page[101], 0xAD);
            assert_eq!(page[102], 0xBE);
            assert_eq!(page[103], 0xEF);
        }
    }
}
