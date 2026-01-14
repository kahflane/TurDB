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

use super::{MmapStorage, Storage, Wal};
use crate::database::dirty_tracker::ShardedDirtyTracker;
use eyre::{Result, WrapErr};
use hashbrown::{HashMap, HashSet};
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
        let frames_count = dirty.len() as u32;

        let frames = dirty.iter().map(|&page_no| {
            let page_data = storage
                .page(page_no)
                .expect("failed to read page for WAL flush");
            (page_no, db_size, page_data, 0u64)
        });

        wal.write_frames_batch(frames)
            .wrap_err("failed to write frames batch to WAL")?;

        Ok(frames_count)
    }

    pub fn dirty_page_count(&self) -> usize {
        self.dirty_pages.lock().len()
    }

    pub fn inner_storage(&self) -> &MmapStorage {
        self.storage
    }

    pub fn with_file_id(
        storage: &'a mut MmapStorage,
        dirty_pages: &'a Mutex<HashMap<u64, HashSet<u32>>>,
        file_id: u64,
    ) -> WalStorageMulti<'a> {
        WalStorageMulti::new(storage, dirty_pages, file_id)
    }

    pub fn flush_wal_for_file(
        dirty_pages: &Mutex<HashMap<u64, HashSet<u32>>>,
        storage: &MmapStorage,
        wal: &mut Wal,
        file_id: u64,
    ) -> Result<u32> {
        let dirty: Vec<u32> = {
            let mut guard = dirty_pages.lock();
            match guard.get_mut(&file_id) {
                Some(pages) => {
                    if pages.is_empty() {
                        return Ok(0);
                    }
                    pages.drain().collect()
                }
                None => return Ok(0),
            }
        };

        let db_size = storage.page_count();
        let frames_count = dirty.len() as u32;

        let frames = dirty.iter().map(|&page_no| {
            let page_data = storage
                .page(page_no)
                .expect("failed to read page for WAL flush");
            (page_no, db_size, page_data, file_id)
        });

        wal.write_frames_batch(frames)
            .wrap_err_with(|| format!("failed to write frames batch for file_id={}", file_id))?;

        Ok(frames_count)
    }
}

pub struct WalStorageMulti<'a> {
    storage: &'a mut MmapStorage,
    dirty_pages: &'a Mutex<HashMap<u64, HashSet<u32>>>,
    file_id: u64,
}

impl<'a> WalStorageMulti<'a> {
    pub fn new(
        storage: &'a mut MmapStorage,
        dirty_pages: &'a Mutex<HashMap<u64, HashSet<u32>>>,
        file_id: u64,
    ) -> Self {
        Self {
            storage,
            dirty_pages,
            file_id,
        }
    }

    pub fn dirty_page_count(&self) -> usize {
        let guard = self.dirty_pages.lock();
        guard.get(&self.file_id).map(|s| s.len()).unwrap_or(0)
    }

    pub fn inner_storage(&self) -> &MmapStorage {
        self.storage
    }
}

impl<'a> Storage for WalStorageMulti<'a> {
    fn page(&self, page_no: u32) -> Result<&[u8]> {
        self.storage.page(page_no)
    }

    fn page_mut(&mut self, page_no: u32) -> Result<&mut [u8]> {
        let mut guard = self.dirty_pages.lock();
        guard.entry(self.file_id).or_default().insert(page_no);
        drop(guard);
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

pub struct WalStoragePerTable<'a> {
    storage: &'a mut MmapStorage,
    dirty_tracker: &'a ShardedDirtyTracker,
    table_id: u32,
}

impl<'a> WalStoragePerTable<'a> {
    pub fn new(
        storage: &'a mut MmapStorage,
        dirty_tracker: &'a ShardedDirtyTracker,
        table_id: u32,
    ) -> Self {
        Self {
            storage,
            dirty_tracker,
            table_id,
        }
    }

    pub fn dirty_page_count(&self) -> usize {
        self.dirty_tracker.dirty_count(self.table_id) as usize
    }

    pub fn inner_storage(&self) -> &MmapStorage {
        self.storage
    }

    pub fn flush_wal_for_table(
        dirty_tracker: &ShardedDirtyTracker,
        storage: &MmapStorage,
        wal: &mut Wal,
        table_id: u32,
    ) -> Result<u32> {
        let dirty = dirty_tracker.drain_for_table(table_id);
        if dirty.is_empty() {
            return Ok(0);
        }

        let db_size = storage.page_count();
        let frames_count = dirty.len() as u32;

        let frames = dirty.iter().map(|&page_no| {
            let page_data = storage
                .page(page_no)
                .expect("failed to read page for WAL flush");
            (page_no, db_size, page_data, table_id as u64)
        });

        wal.write_frames_batch(frames)
            .wrap_err_with(|| format!("failed to write frames batch for table_id={}", table_id))?;

        Ok(frames_count)
    }
}

impl<'a> Storage for WalStoragePerTable<'a> {
    fn page(&self, page_no: u32) -> Result<&[u8]> {
        self.storage.page(page_no)
    }

    fn page_mut(&mut self, page_no: u32) -> Result<&mut [u8]> {
        self.dirty_tracker.mark_dirty(self.table_id, page_no);
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
    fn wal_storage_multi_file_tracks_pages_separately() {
        use hashbrown::HashMap;

        let dir = tempdir().expect("should create temp dir");
        let db_path1 = dir.path().join("table1.tbd");
        let db_path2 = dir.path().join("table2.tbd");
        let wal_dir = dir.path().join("wal");

        let mut storage1 = MmapStorage::create(&db_path1, 10).expect("should create storage1");
        let mut storage2 = MmapStorage::create(&db_path2, 10).expect("should create storage2");
        let mut wal = Wal::create(&wal_dir).expect("should create WAL");

        let dirty_pages: Mutex<HashMap<u64, HashSet<u32>>> = Mutex::new(HashMap::new());

        let file_id_1 = 1u64;
        let file_id_2 = 2u64;

        {
            let mut wal_storage1 = WalStorage::with_file_id(&mut storage1, &dirty_pages, file_id_1);
            let page = wal_storage1.page_mut(5).expect("should get page");
            page[0] = 0xAA;
        }

        {
            let mut wal_storage2 = WalStorage::with_file_id(&mut storage2, &dirty_pages, file_id_2);
            let page = wal_storage2.page_mut(5).expect("should get page");
            page[0] = 0xBB;
        }

        {
            let guard = dirty_pages.lock();
            assert!(guard
                .get(&file_id_1)
                .map(|s| s.contains(&5))
                .unwrap_or(false));
            assert!(guard
                .get(&file_id_2)
                .map(|s| s.contains(&5))
                .unwrap_or(false));
        }

        WalStorage::flush_wal_for_file(&dirty_pages, &storage1, &mut wal, file_id_1)
            .expect("should flush file 1");
        WalStorage::flush_wal_for_file(&dirty_pages, &storage2, &mut wal, file_id_2)
            .expect("should flush file 2");

        drop(wal);

        let wal = Wal::open(&wal_dir).expect("should reopen WAL");
        let mut new_storage1 = MmapStorage::create(&db_path1, 10).expect("should create storage1");
        let mut new_storage2 = MmapStorage::create(&db_path2, 10).expect("should create storage2");

        wal.recover_for_file(&mut new_storage1, file_id_1)
            .expect("should recover file 1");
        wal.recover_for_file(&mut new_storage2, file_id_2)
            .expect("should recover file 2");

        let page1 = new_storage1
            .page(5)
            .expect("should read page from storage1");
        let page2 = new_storage2
            .page(5)
            .expect("should read page from storage2");

        assert_eq!(page1[0], 0xAA, "storage1 page 5 should have 0xAA");
        assert_eq!(page2[0], 0xBB, "storage2 page 5 should have 0xBB");
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
            let wal = Wal::open(&wal_dir).expect("should open WAL");

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
