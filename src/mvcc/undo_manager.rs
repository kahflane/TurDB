//! # Undo Page Manager
//!
//! This module provides the `UndoPageManager` for managing the lifecycle of undo pages
//! in TurDB's MVCC implementation. It handles allocation, writing, and cleanup of
//! undo records that store old row versions during transactions.
//!
//! ## Purpose
//!
//! The UndoPageManager serves as the central coordinator for undo operations:
//! - Allocates new undo pages when existing pages fill up
//! - Writes old row versions before updates/deletes
//! - Reads old versions for version chain traversal
//! - Manages page chain linking
//!
//! ## Page Allocation Strategy
//!
//! ```text
//! Transaction begins
//!       │
//!       ▼
//! ┌─────────────────────┐
//! │ First write         │
//! │ - Allocate undo page│
//! │ - Link to txn       │
//! └─────────────────────┘
//!       │
//!       ▼
//! ┌─────────────────────┐
//! │ Subsequent writes   │
//! │ - Append to page    │
//! │ - Allocate new page │
//! │   if full           │
//! └─────────────────────┘
//!       │
//!       ▼
//! ┌─────────────────────┐
//! │ Commit/Rollback     │
//! │ - Pages remain for  │
//! │   version chain     │
//! └─────────────────────┘
//! ```
//!
//! ## Memory Management
//!
//! Undo pages use the freelist for allocation and are reclaimed during
//! garbage collection when no active transaction needs them.
//!
//! ## Thread Safety
//!
//! The UndoPageManager itself is not thread-safe. It should be accessed
//! through the Database's transaction management which provides synchronization.

use super::record_header::RecordHeader;
use super::undo_page::{UndoPageReader, UndoPageWriter, UndoRecord};
use super::{PageId, TableId, TxnId};
use crate::storage::{PageType, Storage, PAGE_SIZE};
use eyre::Result;

pub struct UndoPageManager {
    current_page_id: Option<PageId>,
    page_chain_head: Option<PageId>,
    min_txn_id: TxnId,
}

impl UndoPageManager {
    pub fn new() -> Self {
        Self {
            current_page_id: None,
            page_chain_head: None,
            min_txn_id: u64::MAX,
        }
    }

    pub fn with_existing_page(page_id: PageId) -> Self {
        Self {
            current_page_id: Some(page_id),
            page_chain_head: Some(page_id),
            min_txn_id: u64::MAX,
        }
    }

    pub fn current_page_id(&self) -> Option<PageId> {
        self.current_page_id
    }

    pub fn page_chain_head(&self) -> Option<PageId> {
        self.page_chain_head
    }

    pub fn write_undo_record<S: Storage>(
        &mut self,
        storage: &mut S,
        record: &UndoRecord,
    ) -> Result<(PageId, u16)> {
        let record_size = record.serialized_size();

        if let Some(page_id) = self.current_page_id {
            let page = storage.page_mut(page_id as u32)?;
            let mut writer = UndoPageWriter::new(page)?;

            if writer.free_space() >= record_size {
                let offset = writer.append(record)?;
                self.update_min_txn(record.record_header.txn_id);
                return Ok((page_id, offset));
            }
        }

        let new_page_id = self.allocate_undo_page(storage)?;

        if let Some(old_page_id) = self.current_page_id {
            let old_page = storage.page_mut(old_page_id as u32)?;
            let mut old_writer = UndoPageWriter::new(old_page)?;
            old_writer.set_next_page(new_page_id as u32);
        }

        let new_page = storage.page_mut(new_page_id as u32)?;
        let mut writer = UndoPageWriter::init_empty(new_page)?;
        let offset = writer.append(record)?;

        if self.page_chain_head.is_none() {
            self.page_chain_head = Some(new_page_id);
        }
        self.current_page_id = Some(new_page_id);
        self.update_min_txn(record.record_header.txn_id);

        Ok((new_page_id, offset))
    }

    pub fn read_undo_record<S: Storage>(
        &self,
        storage: &S,
        page_id: PageId,
        offset: u16,
    ) -> Result<UndoRecord> {
        let page = storage.page(page_id as u32)?;
        let reader = UndoPageReader::new(page)?;
        reader.read_record_at(offset)
    }

    pub fn read_undo_header_and_data<S: Storage>(
        storage: &S,
        page_id: PageId,
        offset: u16,
    ) -> Result<Option<(RecordHeader, Vec<u8>)>> {
        let page = storage.page(page_id as u32)?;
        let reader = UndoPageReader::new(page)?;

        match reader.read_record_at(offset) {
            Ok(record) => {
                let data_start =
                    offset as usize + super::undo_page::UNDO_RECORD_HEADER_SIZE + record.key.len();
                let data_end = offset as usize + record.serialized_size();

                if data_end <= PAGE_SIZE {
                    let data_slice = &page[data_start..data_end];
                    Ok(Some((record.record_header, data_slice.to_vec())))
                } else {
                    Ok(None)
                }
            }
            Err(_) => Ok(None),
        }
    }

    fn allocate_undo_page<S: Storage>(&self, storage: &mut S) -> Result<PageId> {
        let page_count = storage.page_count();
        let new_page_id = page_count;
        storage.grow(1)?;

        let page = storage.page_mut(new_page_id)?;
        page.fill(0);

        use crate::storage::PageHeader;
        let header = PageHeader::new(PageType::Undo);
        header.write_to(&mut page[..16])?;

        Ok(new_page_id as PageId)
    }

    fn update_min_txn(&mut self, txn_id: TxnId) {
        if txn_id < self.min_txn_id {
            self.min_txn_id = txn_id;
        }
    }

    pub fn min_txn_id(&self) -> TxnId {
        self.min_txn_id
    }
}

impl Default for UndoPageManager {
    fn default() -> Self {
        Self::new()
    }
}

pub struct UndoRegistry {
    managers: hashbrown::HashMap<TableId, UndoPageManager>,
}

impl UndoRegistry {
    pub fn new() -> Self {
        Self {
            managers: hashbrown::HashMap::new(),
        }
    }

    pub fn get_or_create(&mut self, table_id: TableId) -> &mut UndoPageManager {
        self.managers.entry(table_id).or_default()
    }

    pub fn get(&self, table_id: TableId) -> Option<&UndoPageManager> {
        self.managers.get(&table_id)
    }

    pub fn clear(&mut self) {
        self.managers.clear();
    }
}

impl Default for UndoRegistry {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mvcc::RecordHeader;

    struct MockStorage {
        pages: Vec<Vec<u8>>,
    }

    impl MockStorage {
        fn new(page_count: usize) -> Self {
            Self {
                pages: vec![vec![0u8; PAGE_SIZE]; page_count],
            }
        }
    }

    impl Storage for MockStorage {
        fn page(&self, page_no: u32) -> Result<&[u8]> {
            self.pages
                .get(page_no as usize)
                .map(|p| p.as_slice())
                .ok_or_else(|| eyre::eyre!("page not found"))
        }

        fn page_mut(&mut self, page_no: u32) -> Result<&mut [u8]> {
            self.pages
                .get_mut(page_no as usize)
                .map(|p| p.as_mut_slice())
                .ok_or_else(|| eyre::eyre!("page not found"))
        }

        fn page_count(&self) -> u32 {
            self.pages.len() as u32
        }

        fn grow(&mut self, additional: u32) -> Result<()> {
            for _ in 0..additional {
                self.pages.push(vec![0u8; PAGE_SIZE]);
            }
            Ok(())
        }

        fn sync(&self) -> Result<()> {
            Ok(())
        }
    }

    #[test]
    fn test_write_and_read_undo_record() {
        let mut storage = MockStorage::new(1);
        let mut manager = UndoPageManager::new();

        let header = RecordHeader::new(100);
        let record = UndoRecord::new(1, header, vec![1, 2, 3, 4], vec![5, 6, 7, 8]);

        let (page_id, offset) = manager.write_undo_record(&mut storage, &record).unwrap();

        let read_record = manager.read_undo_record(&storage, page_id, offset).unwrap();
        assert_eq!(read_record.table_id, 1);
        assert_eq!(read_record.key, vec![1, 2, 3, 4]);
        assert_eq!(read_record.value, vec![5, 6, 7, 8]);
        assert_eq!(read_record.record_header.txn_id, 100);
    }

    #[test]
    fn test_multiple_records_allocation() {
        let mut storage = MockStorage::new(1);
        let mut manager = UndoPageManager::new();

        let header = RecordHeader::new(100);
        let large_value = vec![0u8; 10000];
        let record = UndoRecord::new(1, header, vec![1, 2, 3, 4], large_value.clone());

        let (page1, _) = manager.write_undo_record(&mut storage, &record).unwrap();
        let (page2, _) = manager.write_undo_record(&mut storage, &record).unwrap();

        assert_ne!(page1, page2);
        assert!(storage.page_count() >= 2);
    }
}
