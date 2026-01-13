//! # Batch Insert Operations
//!
//! This module implements batch and bulk insert operations for TurDB.
//! These operations are optimized for inserting multiple rows efficiently.
//!
//! ## Operations
//!
//! ### insert_batch / insert_batch_into_schema
//! - Batch insert multiple rows with WAL support
//! - Uses rightmost hint for append-only optimization
//! - Handles MVCC wrapping for transaction isolation
//!
//! ### insert_cached
//! - Optimized path for prepared statement inserts
//! - Caches storage references to avoid repeated lookups
//! - Updates secondary indexes atomically
//!
//! ### bulk_insert
//! - High-throughput bulk loading via FastLoader
//! - Bypasses WAL for maximum speed
//! - Best for initial data loading scenarios
//!
//! ## Performance Characteristics
//!
//! - `insert_batch`: Optimized for small-medium batches (1-10K rows)
//! - `insert_cached`: Fastest for repeated single-row inserts
//! - `bulk_insert`: Best throughput for large loads (100K+ rows)

use crate::btree::BTree;
#[cfg(feature = "timing")]
use crate::database::timing::{
    BTREE_INSERT_NS, INDEX_UPDATE_NS, INSERT_COUNT, MVCC_WRAP_NS, PAGE0_READ_NS, PAGE0_UPDATE_NS,
    RECORD_BUILD_NS, STORAGE_LOCK_NS, TXN_LOOKUP_NS, WAL_FLUSH_NS,
};
use crate::database::Database;
use crate::storage::{TableFileHeader, WalStoragePerTable, DEFAULT_SCHEMA};
use crate::types::{create_record_schema, OwnedValue};
use eyre::Result;
use std::sync::atomic::Ordering;

impl Database {
    pub fn insert_batch(&self, table: &str, rows: &[Vec<OwnedValue>]) -> Result<usize> {
        let (schema_name, table_name) = if let Some(dot_pos) = table.find('.') {
            (&table[..dot_pos], &table[dot_pos + 1..])
        } else {
            (DEFAULT_SCHEMA, table)
        };
        self.insert_batch_into_schema(schema_name, table_name, rows)
    }

    pub fn insert_batch_into_schema(
        &self,
        schema_name: &str,
        table_name: &str,
        rows: &[Vec<OwnedValue>],
    ) -> Result<usize> {
        if rows.is_empty() {
            return Ok(0);
        }

        self.ensure_catalog()?;
        self.ensure_file_manager()?;

        let wal_enabled = self.shared.wal_enabled.load(Ordering::Acquire);
        if wal_enabled {
            self.ensure_wal()?;
        }

        let catalog_guard = self.shared.catalog.read();
        let catalog = catalog_guard.as_ref().unwrap();

        let table_def = catalog.resolve_table(table_name)?;
        let table_id = table_def.id();
        let columns = table_def.columns().to_vec();

        let schema = create_record_schema(&columns);

        drop(catalog_guard);

        let mut file_manager_guard = self.shared.file_manager.write();
        let file_manager = file_manager_guard.as_mut().unwrap();

        let (mut root_page, mut rightmost_hint): (u32, Option<u32>) = {
            let storage_arc = file_manager.table_data_mut(schema_name, table_name)?;
            let storage = storage_arc.read();
            let page = storage.page(0)?;
            let header = TableFileHeader::from_bytes(page)?;
            let stored_root = header.root_page();
            let hint = header.rightmost_hint();
            let root = if stored_root > 0 { stored_root } else { 1 };
            (root, if hint > 0 { Some(hint) } else { Some(root) })
        };

        let table_file_key =
            crate::storage::FileManager::make_table_key(schema_name, table_name);
        let mut record_builder = crate::records::RecordBuilder::new(&schema);
        let mut record_buffer = Vec::with_capacity(256);

        let count;

        if wal_enabled {
            let table_storage_arc = file_manager
                .table_data_mut_with_key(&table_file_key)
                .ok_or_else(|| eyre::eyre!("table storage not found in cache"))?;
            let mut table_storage = table_storage_arc.write();
            let mut wal_storage = WalStoragePerTable::new(
                &mut table_storage,
                &self.shared.dirty_tracker,
                table_id as u32,
            );
            let mut btree =
                BTree::with_rightmost_hint(&mut wal_storage, root_page, rightmost_hint)?;

            for row_values in rows {
                let row_id = self.shared.next_row_id.fetch_add(1, Ordering::Relaxed);
                let row_key = Self::generate_row_key(row_id);
                OwnedValue::build_record_into_buffer(
                    row_values,
                    &mut record_builder,
                    &mut record_buffer,
                )?;
                btree.insert_append(&row_key, &record_buffer)?;
            }

            root_page = btree.root_page();
            rightmost_hint = btree.rightmost_hint();
            count = rows.len();
        } else {
            let table_storage_arc = file_manager
                .table_data_mut_with_key(&table_file_key)
                .ok_or_else(|| eyre::eyre!("table storage not found in cache"))?;
            let mut table_storage = table_storage_arc.write();
            let mut btree =
                BTree::with_rightmost_hint(&mut *table_storage, root_page, rightmost_hint)?;

            for row_values in rows {
                let row_id = self.shared.next_row_id.fetch_add(1, Ordering::Relaxed);
                let row_key = Self::generate_row_key(row_id);
                OwnedValue::build_record_into_buffer(
                    row_values,
                    &mut record_builder,
                    &mut record_buffer,
                )?;
                btree.insert_append(&row_key, &record_buffer)?;
            }

            root_page = btree.root_page();
            rightmost_hint = btree.rightmost_hint();
            count = rows.len();
        }

        {
            let storage_arc = file_manager.table_data_mut(schema_name, table_name)?;
            let mut storage = storage_arc.write();
            let page = storage.page_mut(0)?;
            let header = TableFileHeader::from_bytes_mut(page)?;
            header.set_root_page(root_page);
            if let Some(hint) = rightmost_hint {
                header.set_rightmost_hint(hint);
            }
            let new_row_count = header.row_count().saturating_add(count as u64);
            header.set_row_count(new_row_count);
        }

        self.flush_wal_if_autocommit(file_manager, schema_name, table_name, table_id as u32)?;

        Ok(count)
    }

    pub fn insert_cached(
        &self,
        plan: &crate::database::prepared::CachedInsertPlan,
        params: &[OwnedValue],
    ) -> Result<usize> {
        use crate::database::dml::mvcc_helpers::wrap_record_for_insert;

        #[cfg(feature = "timing")]
        INSERT_COUNT.fetch_add(1, Ordering::Relaxed);

        self.ensure_file_manager()?;

        let wal_enabled = self.shared.wal_enabled.load(Ordering::Acquire);
        if wal_enabled {
            self.ensure_wal()?;
        }

        #[cfg(feature = "timing")]
        let storage_lock_start = std::time::Instant::now();

        let storage_arc = if let Some(weak) = plan.storage.borrow().as_ref() {
            weak.upgrade()
        } else {
            None
        };

        let storage_arc = if let Some(arc) = storage_arc {
            arc
        } else {
            let mut file_manager_guard = self.shared.file_manager.write();
            let file_manager = file_manager_guard.as_mut().unwrap();
            let arc = file_manager.table_data_mut(&plan.schema_name, &plan.table_name)?;
            *plan.storage.borrow_mut() = Some(std::sync::Arc::downgrade(&arc));
            arc
        };

        let mut storage_guard = storage_arc.write();

        #[cfg(feature = "timing")]
        STORAGE_LOCK_NS.fetch_add(storage_lock_start.elapsed().as_nanos() as u64, Ordering::Relaxed);

        #[cfg(feature = "timing")]
        let page0_start = std::time::Instant::now();

        let (mut root_page, mut rightmost_hint) = {
            let page = storage_guard.page(0)?;
            let header = TableFileHeader::from_bytes(page)?;
            let stored_root = header.root_page();
            let hint = header.rightmost_hint();
            let root = if stored_root > 0 { stored_root } else { 1 };
            (root, if hint > 0 { Some(hint) } else { Some(root) })
        };

        #[cfg(feature = "timing")]
        PAGE0_READ_NS.fetch_add(page0_start.elapsed().as_nanos() as u64, Ordering::Relaxed);

        #[cfg(feature = "timing")]
        let record_start = std::time::Instant::now();

        let mut record_builder = crate::records::RecordBuilder::new(&plan.record_schema);

        let mut buffer_guard = plan.record_buffer.borrow_mut();
        buffer_guard.clear();
        OwnedValue::build_record_into_buffer(params, &mut record_builder, &mut buffer_guard)?;

        #[cfg(feature = "timing")]
        RECORD_BUILD_NS.fetch_add(record_start.elapsed().as_nanos() as u64, Ordering::Relaxed);

        #[cfg(feature = "timing")]
        let txn_start = std::time::Instant::now();

        let (txn_id, in_transaction) = {
            let active_txn = self.active_txn.lock();
            if let Some(ref txn) = *active_txn {
                (txn.txn_id, true)
            } else {
                (
                    self.shared
                        .txn_manager
                        .global_ts
                        .fetch_add(1, Ordering::SeqCst),
                    false,
                )
            }
        };

        #[cfg(feature = "timing")]
        TXN_LOOKUP_NS.fetch_add(txn_start.elapsed().as_nanos() as u64, Ordering::Relaxed);

        #[cfg(feature = "timing")]
        let mvcc_start = std::time::Instant::now();

        let mvcc_record = wrap_record_for_insert(txn_id, &buffer_guard, in_transaction);

        #[cfg(feature = "timing")]
        MVCC_WRAP_NS.fetch_add(mvcc_start.elapsed().as_nanos() as u64, Ordering::Relaxed);

        let row_id = self.shared.next_row_id.fetch_add(1, Ordering::Relaxed);
        let row_key = Self::generate_row_key(row_id);

        #[cfg(feature = "timing")]
        let btree_start = std::time::Instant::now();

        if wal_enabled {
            let mut wal_storage = WalStoragePerTable::new(
                &mut storage_guard,
                &self.shared.dirty_tracker,
                plan.table_id as u32,
            );
            let mut btree =
                BTree::with_rightmost_hint(&mut wal_storage, root_page, rightmost_hint)?;
            btree.insert_append(&row_key, &mvcc_record)?;
            root_page = btree.root_page();
            rightmost_hint = btree.rightmost_hint();
        } else {
            let mut btree =
                BTree::with_rightmost_hint(&mut storage_guard, root_page, rightmost_hint)?;
            btree.insert_append(&row_key, &mvcc_record)?;
            root_page = btree.root_page();
            rightmost_hint = btree.rightmost_hint();
        }

        #[cfg(feature = "timing")]
        BTREE_INSERT_NS.fetch_add(btree_start.elapsed().as_nanos() as u64, Ordering::Relaxed);

        #[cfg(feature = "timing")]
        let index_start = std::time::Instant::now();

        let row_id_bytes = row_id.to_be_bytes();
        for index_plan in &plan.indexes {
            let index_storage_arc = if let Some(weak) = index_plan.storage.borrow().as_ref() {
                weak.upgrade()
            } else {
                None
            };

            let index_storage_arc = if let Some(arc) = index_storage_arc {
                arc
            } else {
                let mut file_manager_guard = self.shared.file_manager.write();
                let file_manager = file_manager_guard.as_mut().unwrap();
                if let Ok(arc) = file_manager.index_data_mut(
                    &plan.schema_name,
                    &plan.table_name,
                    &index_plan.name,
                ) {
                    *index_plan.storage.borrow_mut() = Some(std::sync::Arc::downgrade(&arc));
                    arc
                } else {
                    continue;
                }
            };

            let mut index_storage_guard = index_storage_arc.write();

            let index_root = {
                let cached_root = index_plan.root_page.get();
                if cached_root > 0 {
                    cached_root
                } else {
                    use crate::storage::IndexFileHeader;
                    let page = index_storage_guard.page(0)?;
                    let header = IndexFileHeader::from_bytes(page)?;
                    let root = header.root_page();
                    index_plan.root_page.set(root);
                    root
                }
            };

            let mut key_buf_guard = index_plan.key_buffer.borrow_mut();
            key_buf_guard.clear();
            for &col_idx in &index_plan.col_indices {
                if let Some(val) = params.get(col_idx) {
                    Self::encode_value_as_key(val, &mut *key_buf_guard);
                }
            }

            let mut index_btree = BTree::new(&mut *index_storage_guard, index_root)?;
            index_btree.insert(&key_buf_guard, &row_id_bytes)?;

            let new_root = index_btree.root_page();
            if new_root != index_root {
                use crate::storage::IndexFileHeader;
                let page = index_storage_guard.page_mut(0)?;
                let header = IndexFileHeader::from_bytes_mut(page)?;
                header.set_root_page(new_root);
                index_plan.root_page.set(new_root);
            }
        }
        #[cfg(feature = "timing")]
        INDEX_UPDATE_NS.fetch_add(index_start.elapsed().as_nanos() as u64, Ordering::Relaxed);

        #[cfg(feature = "timing")]
        let page0_update_start = std::time::Instant::now();

        {
            let page = storage_guard.page_mut(0)?;
            let header = TableFileHeader::from_bytes_mut(page)?;
            header.set_root_page(root_page);
            if let Some(hint) = rightmost_hint {
                header.set_rightmost_hint(hint);
            }
            let new_row_count = header.row_count().saturating_add(1);
            header.set_row_count(new_row_count);
        }

        #[cfg(feature = "timing")]
        PAGE0_UPDATE_NS.fetch_add(page0_update_start.elapsed().as_nanos() as u64, Ordering::Relaxed);

        if wal_enabled && self.shared.wal_autoflush.load(Ordering::Acquire) {
            #[cfg(feature = "timing")]
            let wal_flush_start = std::time::Instant::now();

            let txn_active = self.active_txn.lock().is_some();
            if !txn_active
                && self
                    .shared
                    .dirty_tracker
                    .has_dirty_pages(plan.table_id as u32)
            {
                let mut wal_guard = self.shared.wal.lock();
                if let Some(wal) = wal_guard.as_mut() {
                    WalStoragePerTable::flush_wal_for_table(
                        &self.shared.dirty_tracker,
                        &storage_guard,
                        wal,
                        plan.table_id as u32,
                    )?;
                }
            }

            #[cfg(feature = "timing")]
            WAL_FLUSH_NS.fetch_add(wal_flush_start.elapsed().as_nanos() as u64, Ordering::Relaxed);
        }

        Ok(1)
    }

    pub fn bulk_insert(&self, table_name: &str, rows: Vec<Vec<OwnedValue>>) -> Result<u64> {
        use crate::database::dml::fast_load::FastLoader;

        self.ensure_catalog()?;
        self.ensure_file_manager()?;

        let schema_name = DEFAULT_SCHEMA;

        let record_schema = {
            let catalog_guard = self.shared.catalog.read();
            let catalog = catalog_guard.as_ref().unwrap();
            let table_def = catalog.resolve_table(table_name)?;
            create_record_schema(table_def.columns())
        };

        let mut file_manager_guard = self.shared.file_manager.write();
        let file_manager = file_manager_guard.as_mut().unwrap();
        let storage_arc = file_manager.table_data_mut(schema_name, table_name)?;

        let (root_page, starting_row_id) = {
            let storage = storage_arc.write();
            let page = storage.page(0)?;
            let header = TableFileHeader::from_bytes(page)?;
            (header.root_page(), header.row_count())
        };

        let mut storage = storage_arc.write();
        let mut loader =
            FastLoader::new(&mut *storage, &record_schema, root_page, starting_row_id)?;

        for row in rows {
            loader.insert_unchecked(&row)?;
        }

        let stats = loader.finish()?;

        {
            let page = storage.page_mut(0)?;
            let header = TableFileHeader::from_bytes_mut(page)?;
            let new_row_count = header.row_count().saturating_add(stats.row_count);
            header.set_row_count(new_row_count);
        }

        Ok(stats.row_count)
    }
}
