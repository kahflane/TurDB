//! # INSERT Operation Module
//!
//! This module implements INSERT operations for TurDB, handling single-row and
//! multi-row inserts with full constraint validation and conflict resolution.
//!
//! ## Purpose
//!
//! INSERT operations add new rows to tables while:
//! - Validating all constraints (PK, UNIQUE, CHECK, FK)
//! - Processing ON CONFLICT clauses (DO NOTHING, DO UPDATE)
//! - Handling TOAST for large values
//! - Maintaining secondary indexes
//! - Recording transaction write entries
//!
//! ## Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────────────┐
//! │                        INSERT Operation Flow                            │
//! ├─────────────────────────────────────────────────────────────────────────┤
//! │                                                                         │
//! │   INSERT INTO table (cols) VALUES (vals)                                │
//! │       │                                                                 │
//! │       ▼                                                                 │
//! │   ┌─────────────────────────────────────────────────────────────────┐   │
//! │   │ 1. Resolve table & columns                                      │   │
//! │   │    - Load table definition from catalog                         │   │
//! │   │    - Map column names to indices                                │   │
//! │   │    - Identify constraint columns (PK, UNIQUE)                   │   │
//! │   └─────────────────────────────────────────────────────────────────┘   │
//! │       │                                                                 │
//! │       ▼                                                                 │
//! │   ┌─────────────────────────────────────────────────────────────────┐   │
//! │   │ 2. For each row:                                                │   │
//! │   │    a. Handle AUTO_INCREMENT if present                          │   │
//! │   │    b. Validate NOT NULL and type constraints                    │   │
//! │   │    c. Evaluate CHECK expressions                                │   │
//! │   │    d. Verify FOREIGN KEY references exist                       │   │
//! │   │    e. Check UNIQUE/PK via index lookup                          │   │
//! │   │    f. Handle ON CONFLICT if duplicate found                     │   │
//! │   └─────────────────────────────────────────────────────────────────┘   │
//! │       │                                                                 │
//! │       ▼                                                                 │
//! │   ┌─────────────────────────────────────────────────────────────────┐   │
//! │   │ 3. TOAST large values                                           │   │
//! │   │    - Split into chunks if exceeds threshold                     │   │
//! │   │    - Store in TOAST table                                       │   │
//! │   │    - Replace with TOAST pointer                                 │   │
//! │   └─────────────────────────────────────────────────────────────────┘   │
//! │       │                                                                 │
//! │       ▼                                                                 │
//! │   ┌─────────────────────────────────────────────────────────────────┐   │
//! │   │ 4. Insert into BTree                                            │   │
//! │   │    - Generate row key from row_id                               │   │
//! │   │    - Build record from values                                   │   │
//! │   │    - Insert with WAL tracking if enabled                        │   │
//! │   └─────────────────────────────────────────────────────────────────┘   │
//! │       │                                                                 │
//! │       ▼                                                                 │
//! │   ┌─────────────────────────────────────────────────────────────────┐   │
//! │   │ 5. Update indexes                                               │   │
//! │   │    - Insert into UNIQUE indexes                                 │   │
//! │   │    - Insert into secondary indexes                              │   │
//! │   └─────────────────────────────────────────────────────────────────┘   │
//! │       │                                                                 │
//! │       ▼                                                                 │
//! │   ┌─────────────────────────────────────────────────────────────────┐   │
//! │   │ 6. Record transaction write entry                               │   │
//! │   │    - Store key for potential rollback                           │   │
//! │   └─────────────────────────────────────────────────────────────────┘   │
//! │                                                                         │
//! └─────────────────────────────────────────────────────────────────────────┘
//! ```
//!
//! ## ON CONFLICT Handling
//!
//! INSERT supports ON CONFLICT for upsert operations:
//!
//! - `ON CONFLICT DO NOTHING`: Skip rows that violate constraints
//! - `ON CONFLICT DO UPDATE SET ...`: Update existing row on conflict
//!
//! ## Performance Characteristics
//!
//! - Single row: O(log n) for BTree + O(k * log n) for k indexes
//! - Batch insert: Amortized O(log n) with rightmost hint optimization
//! - Constraint check: O(log n) per unique constraint via index
//! - FK check: O(n) scan of referenced table (could be optimized with index)
//!
//! ## Thread Safety
//!
//! INSERT acquires write lock on file_manager for the duration of the operation.
//! Transaction write entries are recorded under active_txn lock.

use crate::btree::BTree;
use crate::constraints::ConstraintValidator;
use crate::database::dml::fast_load::InsertBuffers;
use crate::database::dml::mvcc_helpers::{wrap_record_for_insert, wrap_record_into_buffer};
use crate::database::row::Row;
use crate::database::{Database, ExecuteResult};
use crate::mvcc::WriteEntry;
use crate::schema::table::{Constraint, IndexType};
use crate::storage::{TableFileHeader, WalStoragePerTable, DEFAULT_SCHEMA};
use crate::types::{create_record_schema, OwnedValue};
use bumpalo::Bump;
use eyre::{bail, Result};
use hashbrown::HashSet;
use smallvec::SmallVec;
use std::sync::atomic::Ordering;

impl Database {
    pub(crate) fn execute_insert_cached(
        &self,
        cached: &crate::database::prepared::CachedInsertPlan,
        params: &[OwnedValue],
    ) -> Result<ExecuteResult> {
        self.ensure_file_manager()?;

        let wal_enabled = self.shared.wal_enabled.load(Ordering::Acquire);
        if wal_enabled {
            self.ensure_wal()?;
        }

        if params.len() != cached.column_count {
            eyre::bail!(
                "parameter count mismatch: expected {} but got {}",
                cached.column_count,
                params.len()
            );
        }

        let count = self.insert_cached(cached, params)?;

        Ok(ExecuteResult::Insert {
            rows_affected: count,
            returned: None,
        })
    }

    pub(crate) fn execute_insert(
        &self,
        insert: &crate::sql::ast::InsertStmt<'_>,
        _arena: &Bump,
        params: Option<&[OwnedValue]>,
    ) -> Result<ExecuteResult> {
        self.execute_insert_internal(insert, _arena, params)
    }

    fn execute_insert_internal(
        &self,
        insert: &crate::sql::ast::InsertStmt<'_>,
        _arena: &Bump,
        params: Option<&[OwnedValue]>,
    ) -> Result<ExecuteResult> {
        self.ensure_catalog()?;
        self.ensure_file_manager()?;

        let wal_enabled = self.shared.wal_enabled.load(Ordering::Acquire);
        if wal_enabled {
            self.ensure_wal()?;
        }

        let catalog_guard = self.shared.catalog.read();
        let catalog = catalog_guard.as_ref().unwrap();

        let schema_name = insert.table.schema.unwrap_or(DEFAULT_SCHEMA);
        let table_name = insert.table.name;

        let table_def = catalog.resolve_table_in_schema(insert.table.schema, table_name)?;
        let table_id = table_def.id();
        let columns = table_def.columns().to_vec();
        let table_def_for_validator = table_def.clone();
        let has_toast = table_def.has_toast();

        let unique_columns: Vec<(usize, String, bool, bool)> = columns
            .iter()
            .enumerate()
            .filter_map(|(idx, col)| {
                let is_pk = col.has_constraint(&Constraint::PrimaryKey);
                let is_unique = col.has_constraint(&Constraint::Unique);
                let is_auto_increment = col.has_constraint(&Constraint::AutoIncrement);
                if is_pk || is_unique {
                    let index_name = if is_pk {
                        format!("{}_pkey", col.name())
                    } else {
                        format!("{}_key", col.name())
                    };
                    Some((idx, index_name, is_pk, is_auto_increment))
                } else {
                    None
                }
            })
            .collect();

        let unique_column_index_names: HashSet<&str> = unique_columns
            .iter()
            .map(|(_, name, _, _)| name.as_str())
            .collect();

        let unique_indexes: Vec<(Vec<usize>, FileKey)> = table_def
            .indexes()
            .iter()
            .filter(|idx| idx.is_unique())
            .filter(|idx| !unique_column_index_names.contains(idx.name()))
            .filter_map(|idx| {
                let col_indices: Vec<usize> = idx
                    .columns()
                    .filter_map(|col_name| {
                        columns
                            .iter()
                            .position(|c| c.name().eq_ignore_ascii_case(col_name))
                    })
                    .collect();
                if col_indices.is_empty() {
                    None
                } else {
                    let key = crate::storage::FileManager::make_index_key(
                        schema_name,
                        table_name,
                        idx.name(),
                    );
                    Some((col_indices, key))
                }
            })
            .collect();

        let secondary_indexes: Vec<(Vec<usize>, FileKey)> = table_def
            .indexes()
            .iter()
            .filter(|idx| !idx.is_unique())
            .filter_map(|idx| {
                let col_indices: Vec<usize> = idx
                    .columns()
                    .filter_map(|col_name| {
                        columns
                            .iter()
                            .position(|c| c.name().eq_ignore_ascii_case(col_name))
                    })
                    .collect();
                if col_indices.is_empty() {
                    None
                } else {
                    let key = crate::storage::FileManager::make_index_key(
                        schema_name,
                        table_name,
                        idx.name(),
                    );
                    Some((col_indices, key))
                }
            })
            .collect();

        let hnsw_indexes: Vec<(String, usize)> = table_def
            .indexes()
            .iter()
            .filter(|idx| idx.index_type() == IndexType::Hnsw)
            .filter_map(|idx| {
                let col_name = idx.columns().next()?;
                let col_idx = columns
                    .iter()
                    .position(|c| c.name().eq_ignore_ascii_case(col_name))?;
                Some((idx.name().to_string(), col_idx))
            })
            .collect();

        let fk_constraints: Vec<(usize, String, String, Option<String>)> = columns
            .iter()
            .enumerate()
            .flat_map(|(idx, col)| {
                col.constraints().iter().filter_map(move |c| {
                    if let Constraint::ForeignKey { table, column, .. } = c {
                        let index_name = catalog
                            .resolve_table(table)
                            .ok()
                            .and_then(|ref_table| {
                                ref_table.indexes().iter().find(|idx_def| {
                                    idx_def.columns().next().is_some_and(|first_col| {
                                        first_col.eq_ignore_ascii_case(column)
                                    })
                                })
                            })
                            .map(|idx_def| idx_def.name().to_string());
                        Some((idx, table.clone(), column.clone(), index_name))
                    } else {
                        None
                    }
                })
            })
            .collect();

        let schema = create_record_schema(&columns);

        let column_types: Vec<crate::records::types::DataType> =
            columns.iter().map(|c| c.data_type()).collect();

        let insert_col_indices: Option<Vec<usize>> = insert.columns.map(|cols| {
            cols.iter()
                .filter_map(|col_name| {
                    columns
                        .iter()
                        .position(|c| c.name().eq_ignore_ascii_case(col_name))
                })
                .collect()
        });

        let auto_increment_col_idx: Option<usize> = columns
            .iter()
            .position(|c| c.has_constraint(&Constraint::AutoIncrement));

        let mut param_idx = 0usize;

        drop(catalog_guard);

        let rows_to_insert: Vec<Vec<OwnedValue>> = match &insert.source {
            crate::sql::ast::InsertSource::Values(values) => {
                let mut result = Vec::with_capacity(values.len());
                for row_exprs in values.iter() {
                    let mut row = vec![OwnedValue::Null; columns.len()];

                    if let Some(ref col_indices) = insert_col_indices {
                        for (val_idx, &col_idx) in col_indices.iter().enumerate() {
                            if let Some(expr) = row_exprs.get(val_idx) {
                                let data_type = column_types.get(col_idx);
                                row[col_idx] = Self::eval_expr_with_params(
                                    expr,
                                    data_type,
                                    params,
                                    &mut param_idx,
                                )?;
                            }
                        }
                    } else {
                        for (idx, expr) in row_exprs.iter().enumerate() {
                            if idx < columns.len() {
                                let data_type = column_types.get(idx);
                                row[idx] = Self::eval_expr_with_params(
                                    expr,
                                    data_type,
                                    params,
                                    &mut param_idx,
                                )?;
                            }
                        }
                    }

                    result.push(row);
                }
                result
            }
            crate::sql::ast::InsertSource::Select(select_stmt) => {
                let select_rows = self.execute_select_internal(select_stmt)?;
                select_rows.into_iter().map(|row| row.values).collect()
            }
            crate::sql::ast::InsertSource::Default => {
                bail!("DEFAULT VALUES insert not supported")
            }
        };

        let mut root_page: u32;
        let initial_root_page: u32;
        let validator = ConstraintValidator::new(&table_def_for_validator);

        let mut file_manager_guard = self.shared.file_manager.write();
        let file_manager = file_manager_guard.as_mut().unwrap();
        let mut storage_map: hashbrown::HashMap<
            crate::storage::FileKey,
            std::sync::Arc<parking_lot::RwLock<crate::storage::MmapStorage>>,
        > = hashbrown::HashMap::new();

        let table_file_key = crate::storage::FileManager::make_table_key(schema_name, table_name);
        let main_storage_arc = file_manager.table_data_mut(schema_name, table_name)?;
        storage_map.insert(table_file_key.clone(), main_storage_arc.clone());

        let (mut auto_increment_current, hint) = {
            let storage = main_storage_arc.write();
            let page = storage.page(0)?;
            let header = TableFileHeader::from_bytes(page)?;
            root_page = header.root_page();
            initial_root_page = root_page;
            let ai = if auto_increment_col_idx.is_some() {
                header.auto_increment()
            } else {
                0
            };
            (ai, header.rightmost_hint())
        };
        let mut auto_increment_max = auto_increment_current;
        let mut rightmost_hint = if hint > 0 { Some(hint) } else { None };

        let mut toast_rightmost_hints: SmallVec<[Option<u32>; 8]> = SmallVec::new();
        let toastable_col_indices: SmallVec<[usize; 8]> = if has_toast {
            toast_rightmost_hints.resize(columns.len(), None);
            columns
                .iter()
                .enumerate()
                .filter(|(_, col)| col.data_type().is_toastable())
                .map(|(idx, _)| idx)
                .collect()
        } else {
            SmallVec::new()
        };

        use crate::storage::FileKey;
        let toast_file_key: Option<FileKey>;
        let mut toast_table_id: u32 = 0;
        let mut toast_root_page: u32 = 1;
        let mut toast_initial_root_page: u32 = 1;
        let mut toast_rightmost_hint: Option<u32> = None;

        if has_toast {
            let toast_table_name_owned = crate::storage::toast::toast_table_name(table_name);
            let key =
                crate::storage::FileManager::make_table_key(schema_name, &toast_table_name_owned);

            if file_manager.table_exists(schema_name, &toast_table_name_owned) {
                let toast_storage_arc =
                    file_manager.table_data_mut(schema_name, &toast_table_name_owned)?;
                let toast_storage = toast_storage_arc.write();
                let page0 = toast_storage.page(0)?;
                let header = crate::storage::TableFileHeader::from_bytes(page0)?;
                toast_table_id = header.table_id() as u32;
                toast_root_page = header.root_page();
                toast_initial_root_page = toast_root_page;
                let hint = header.rightmost_hint();
                toast_rightmost_hint = if hint > 0 { Some(hint) } else { None };

                drop(toast_storage);
                storage_map.insert(key.clone(), toast_storage_arc);
                toast_file_key = Some(key);
            } else {
                toast_file_key = None;
            }
        } else {
            toast_file_key = None;
        }

        let mut unique_column_keys: Vec<(usize, FileKey, bool, bool)> = Vec::new();
        for (col_idx, index_name, is_pk, is_auto_increment) in &unique_columns {
            if *is_auto_increment {
                continue;
            }
            if file_manager.index_exists(schema_name, table_name, index_name) {
                let storage = file_manager.index_data_mut(schema_name, table_name, index_name)?;
                let key = crate::storage::FileManager::make_index_key(
                    schema_name,
                    table_name,
                    index_name,
                );
                storage_map.insert(key.clone(), storage);
                unique_column_keys.push((*col_idx, key, *is_pk, false));
            }
        }

        for (_, key) in &unique_indexes {
            if let FileKey::Index { index_name, .. } = key {
                if file_manager.index_exists(schema_name, table_name, index_name) {
                    let storage = file_manager.index_data_mut(schema_name, table_name, index_name)?;
                    storage_map.insert(key.clone(), storage);
                }
            }
        }

        for (_, key) in &secondary_indexes {
            if let FileKey::Index { index_name, .. } = key {
                if file_manager.index_exists(schema_name, table_name, index_name) {
                    let storage = file_manager.index_data_mut(schema_name, table_name, index_name)?;
                    storage_map.insert(key.clone(), storage);
                }
            }
        }

        for (_, fk_table, _, fk_index_name) in &fk_constraints {
            if file_manager.table_exists(schema_name, fk_table) {
                let storage = file_manager.table_data_mut(schema_name, fk_table)?;
                let key = crate::storage::FileManager::make_table_key(schema_name, fk_table);
                storage_map.insert(key, storage);

                if let Some(idx_name) = fk_index_name {
                    if file_manager.index_exists(schema_name, fk_table, idx_name) {
                        let idx_storage =
                            file_manager.index_data_mut(schema_name, fk_table, idx_name)?;
                        let idx_key = crate::storage::FileManager::make_index_key(
                            schema_name,
                            fk_table,
                            idx_name,
                        );
                        storage_map.insert(idx_key, idx_storage);
                    }
                }
            }
        }

        let mut record_builder = crate::records::RecordBuilder::new(&schema);
        let mut count = 0;
        let mut buffers = InsertBuffers::new();
        let mut returned_rows: Option<Vec<Row>> = insert.returning.map(|_| Vec::new());

        let mut index_rightmost_hints: hashbrown::HashMap<FileKey, Option<u32>> =
            hashbrown::HashMap::new();
        for (_, key, _, _) in &unique_column_keys {
            index_rightmost_hints.insert(key.clone(), None);
        }
        for (_, key) in &unique_indexes {
            index_rightmost_hints.insert(key.clone(), None);
        }
        for (_, key) in &secondary_indexes {
            index_rightmost_hints.insert(key.clone(), None);
        }

        let mut index_root_pages: hashbrown::HashMap<FileKey, u32> = hashbrown::HashMap::new();
        let mut index_initial_root_pages: hashbrown::HashMap<FileKey, u32> =
            hashbrown::HashMap::new();
        for key in index_rightmost_hints.keys() {
            if let Some(storage_arc) = storage_map.get(key) {
                let storage = storage_arc.read();
                let page = storage.page(0)?;
                let header = crate::storage::IndexFileHeader::from_bytes(page)?;
                let rp = header.root_page();
                index_root_pages.insert(key.clone(), rp);
                index_initial_root_pages.insert(key.clone(), rp);
            }
        }

        drop(file_manager_guard);

        for mut values in rows_to_insert.into_iter() {
            buffers.reset();

            if let Some(auto_col_idx) = auto_increment_col_idx {
                if values.get(auto_col_idx).is_none_or(|v| v.is_null()) {
                    auto_increment_current =
                        auto_increment_current.checked_add(1).ok_or_else(|| {
                            eyre::eyre!("auto_increment overflow: exceeded maximum value")
                        })?;
                    values[auto_col_idx] = OwnedValue::Int(auto_increment_current as i64);
                    if auto_increment_current > auto_increment_max {
                        auto_increment_max = auto_increment_current;
                    }
                } else if let Some(OwnedValue::Int(provided_val)) = values.get(auto_col_idx) {
                    if *provided_val < 0 {
                        bail!(
                            "auto_increment column cannot have negative value: {}",
                            provided_val
                        );
                    }
                    if (*provided_val as u64) > auto_increment_max {
                        auto_increment_max = *provided_val as u64;
                    }
                }
            }

            validator.validate_insert(&mut values)?;

            for (col_idx, col) in columns.iter().enumerate() {
                for constraint in col.constraints() {
                    if let Constraint::Check(expr_str) = constraint {
                        let col_value = values.get(col_idx);
                        if !Database::evaluate_check_expression(expr_str, col.name(), col_value) {
                            bail!(
                                "CHECK constraint violated on column '{}' in table '{}': {}",
                                col.name(),
                                table_name,
                                expr_str
                            );
                        }
                    }
                }
            }

            let fk_enabled = self.foreign_keys_enabled.load(Ordering::Acquire);
            if fk_enabled && !fk_constraints.is_empty() {
                let catalog_guard = self.shared.catalog.read();
                let catalog = catalog_guard.as_ref().unwrap();

                for (col_idx, fk_table, fk_column, fk_index_name) in &fk_constraints {
                    if let Some(value) = values.get(*col_idx) {
                        if value.is_null() {
                            continue;
                        }

                        let referenced_table = catalog.resolve_table(fk_table)?;
                        let ref_columns = referenced_table.columns();
                        let ref_col_idx = ref_columns
                            .iter()
                            .position(|c| c.name().eq_ignore_ascii_case(fk_column));

                        if ref_col_idx.is_none() {
                            bail!(
                                "FOREIGN KEY constraint: column '{}' not found in table '{}'",
                                fk_column,
                                fk_table
                            );
                        }

                        let found = if let Some(idx_name) = fk_index_name {
                            let fk_idx_key = crate::storage::FileManager::make_index_key(
                                schema_name,
                                fk_table,
                                idx_name,
                            );
                            if let Some(idx_storage_arc) = storage_map.get(&fk_idx_key) {
                                let mut idx_storage = idx_storage_arc.write();
                                let idx_btree = BTree::new(&mut *idx_storage, 1)?;

                                buffers.key_buffer.clear();
                                Self::encode_value_as_key(value, &mut buffers.key_buffer);
                                idx_btree.search(&buffers.key_buffer)?.is_some()
                            } else {
                                Self::fk_table_scan_check(
                                    schema_name,
                                    fk_table,
                                    ref_columns,
                                    ref_col_idx.unwrap(),
                                    value,
                                    &storage_map,
                                )?
                            }
                        } else {
                            Self::fk_table_scan_check(
                                schema_name,
                                fk_table,
                                ref_columns,
                                ref_col_idx.unwrap(),
                                value,
                                &storage_map,
                            )?
                        };

                        if !found {
                            bail!(
                                "FOREIGN KEY constraint violated on column '{}' in table '{}': referenced value not found in {}.{}",
                                columns[*col_idx].name(),
                                table_name,
                                fk_table,
                                fk_column
                            );
                        }
                    }
                }
            }

            let mut has_conflict = false;
            let mut conflicting_key: Option<Vec<u8>> = None;

            for (col_idx, index_key, is_pk, _) in &unique_column_keys {
                if let Some(value) = values.get(*col_idx) {
                    if value.is_null() {
                        continue;
                    }

                    let index_storage_arc = storage_map
                        .get(index_key)
                        .ok_or_else(|| eyre::eyre!("index storage not found"))?;
                    let mut index_storage = index_storage_arc.write();
                    let index_rp = *index_root_pages.get(index_key).expect("index root page should be initialized");
                    let index_btree = BTree::new(&mut *index_storage, index_rp)?;

                    buffers.key_buffer.clear();
                    Self::encode_value_as_key(value, &mut buffers.key_buffer);

                    if let Some(handle) = index_btree.search(&buffers.key_buffer)? {
                        if insert.on_conflict.is_some() {
                            has_conflict = true;
                            let row_key_bytes = index_btree.get_value(&handle)?;
                            conflicting_key = Some(row_key_bytes.to_vec());
                            break;
                        } else {
                            let constraint_type = if *is_pk { "PRIMARY KEY" } else { "UNIQUE" };
                            bail!(
                                "{} constraint violated on column '{}' in table '{}': value already exists",
                                constraint_type,
                                columns[*col_idx].name(),
                                table_name
                            );
                        }
                    }
                }
            }

            if !has_conflict {
                for (col_indices, key) in &unique_indexes {
                    let all_non_null = col_indices
                        .iter()
                        .all(|&idx| values.get(idx).map(|v| !v.is_null()).unwrap_or(false));

                    if all_non_null && storage_map.contains_key(key) {
                        let index_storage_arc = storage_map.get(key).unwrap();
                        let mut index_storage = index_storage_arc.write();
                        let index_rp = *index_root_pages.get(key).expect("index root page should be initialized");
                        let index_btree = BTree::new(&mut *index_storage, index_rp)?;

                        buffers.key_buffer.clear();
                        for &col_idx in col_indices {
                            if let Some(value) = values.get(col_idx) {
                                Self::encode_value_as_key(value, &mut buffers.key_buffer);
                            }
                        }

                        if let Some(handle) = index_btree.search(&buffers.key_buffer)? {
                            if insert.on_conflict.is_some() {
                                has_conflict = true;
                                let row_key_bytes = index_btree.get_value(&handle)?;
                                conflicting_key = Some(row_key_bytes.to_vec());
                                break;
                            } else {
                                let col_names: SmallVec<[&str; 8]> = col_indices
                                    .iter()
                                    .filter_map(|&idx| columns.get(idx).map(|c| c.name()))
                                    .collect();
                                let index_name = if let FileKey::Index { index_name, .. } = key {
                                    index_name.as_str()
                                } else {
                                    "unknown"
                                };
                                bail!(
                                    "UNIQUE constraint violated on index '{}' (columns: {}) in table '{}': value already exists",
                                    index_name,
                                    col_names.join(", "),
                                    table_name
                                );
                            }
                        }
                    }
                }
            }

            if has_conflict {
                use crate::sql::ast::OnConflictAction;

                if let Some(on_conflict) = insert.on_conflict {
                    match on_conflict.action {
                        OnConflictAction::DoNothing => {
                            continue;
                        }
                        OnConflictAction::DoUpdate(assignments) => {
                            if let Some(existing_key) = conflicting_key {
                                let table_storage_arc = storage_map
                                    .get(&table_file_key)
                                    .ok_or_else(|| eyre::eyre!("table storage not found"))?;
                                let mut table_storage = table_storage_arc.write();
                                let btree = BTree::new(&mut *table_storage, root_page)?;

                                if let Some(handle) = btree.search(&existing_key)? {
                                    let existing_value = btree.get_value(&handle)?;
                                    let user_data =
                                        crate::database::dml::mvcc_helpers::get_user_data(
                                            existing_value,
                                        );
                                    let record =
                                        crate::records::RecordView::new(user_data, &schema)?;
                                    let mut existing_values =
                                        OwnedValue::extract_row_from_record(&record, &columns)?;

                                    for assignment in assignments.iter() {
                                        if let Some(col_idx) = columns.iter().position(|c| {
                                            c.name().eq_ignore_ascii_case(assignment.column.column)
                                        }) {
                                            let new_value = Self::eval_literal(assignment.value)?;
                                            existing_values[col_idx] = new_value;
                                        }
                                    }

                                    let user_record = OwnedValue::build_record_from_values(
                                        &existing_values,
                                        &schema,
                                    )?;

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
                                    let updated_record = wrap_record_for_insert(
                                        txn_id,
                                        &user_record,
                                        in_transaction,
                                    );

                                    let mut btree_mut = BTree::new(&mut *table_storage, root_page)?;
                                    btree_mut.delete(&existing_key)?;
                                    btree_mut.insert(&existing_key, &updated_record)?;

                                    count += 1;

                                    if let Some(ref mut rows) = returned_rows {
                                        let returning_cols = insert.returning.unwrap();
                                        let row_values: Vec<OwnedValue> = returning_cols
                                            .iter()
                                            .flat_map(|col| match col {
                                                crate::sql::ast::SelectColumn::AllColumns => {
                                                    existing_values.clone()
                                                }
                                                crate::sql::ast::SelectColumn::TableAllColumns(
                                                    _,
                                                ) => existing_values.clone(),
                                                crate::sql::ast::SelectColumn::Expr {
                                                    expr,
                                                    ..
                                                } => {
                                                    if let crate::sql::ast::Expr::Column(col_ref) =
                                                        expr
                                                    {
                                                        columns
                                                            .iter()
                                                            .position(|c| {
                                                                c.name().eq_ignore_ascii_case(
                                                                    col_ref.column,
                                                                )
                                                            })
                                                            .and_then(|idx| {
                                                                existing_values.get(idx).cloned()
                                                            })
                                                            .map(|v| vec![v])
                                                            .unwrap_or_default()
                                                    } else {
                                                        vec![]
                                                    }
                                                }
                                            })
                                            .collect();
                                        rows.push(Row::new(row_values));
                                    }
                                }
                            }
                            continue;
                        }
                    }
                }
            }

            let row_id = self.shared.next_row_id.fetch_add(1, Ordering::Relaxed);
            let row_key = Self::generate_row_key(row_id);

            if !toastable_col_indices.is_empty() {
                use crate::storage::toast::{
                    make_chunk_key, needs_toast, ToastPointer, TOAST_CHUNK_SIZE,
                };

                for &col_idx in &toastable_col_indices {
                    let value = &values[col_idx];
                    let data = match value {
                        OwnedValue::Text(s) => Some(s.as_bytes()),
                        OwnedValue::Blob(b) => Some(b.as_slice()),
                        _ => None,
                    };

                    if let Some(data) = data {
                        if needs_toast(data) {
                            let toast_key = toast_file_key.as_ref().unwrap();
                            let toast_storage_arc = storage_map
                                .get(toast_key)
                                .ok_or_else(|| eyre::eyre!("toast storage not found"))?;
                            let mut toast_storage = toast_storage_arc.write();

                            let pointer =
                                ToastPointer::new(row_id, col_idx as u16, data.len() as u64);
                            let chunk_id = pointer.chunk_id;

                            let (new_hint, new_root) = if wal_enabled {
                                let mut wal_storage = WalStoragePerTable::new(
                                    &mut toast_storage,
                                    &self.shared.dirty_tracker,
                                    toast_table_id,
                                );
                                let mut btree = BTree::with_rightmost_hint(
                                    &mut wal_storage,
                                    toast_root_page,
                                    toast_rightmost_hint,
                                )?;
                                for (seq, chunk) in data.chunks(TOAST_CHUNK_SIZE).enumerate() {
                                    let chunk_key = make_chunk_key(chunk_id, seq as u32);
                                    btree.insert(&chunk_key, chunk)?;
                                }
                                (btree.rightmost_hint(), btree.root_page())
                            } else {
                                let mut btree = BTree::with_rightmost_hint(
                                    &mut *toast_storage,
                                    toast_root_page,
                                    toast_rightmost_hint,
                                )?;
                                for (seq, chunk) in data.chunks(TOAST_CHUNK_SIZE).enumerate() {
                                    let chunk_key = make_chunk_key(chunk_id, seq as u32);
                                    btree.insert(&chunk_key, chunk)?;
                                }
                                (btree.rightmost_hint(), btree.root_page())
                            };

                            toast_rightmost_hint = new_hint;
                            if new_root != toast_root_page {
                                toast_root_page = new_root;
                            }

                            values[col_idx] = OwnedValue::Blob(pointer.encode().to_vec());
                        }
                    }
                }
            }

            let table_storage_arc = storage_map
                .get(&table_file_key)
                .ok_or_else(|| eyre::eyre!("table storage not found in cache"))?;
            let mut table_storage = table_storage_arc.write();

            OwnedValue::build_record_into_buffer(
                &values,
                &mut record_builder,
                &mut buffers.record_buffer,
            )?;

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

            wrap_record_into_buffer(
                txn_id,
                &buffers.record_buffer,
                in_transaction,
                &mut buffers.mvcc_buffer,
            );

            if wal_enabled {
                let mut wal_storage = WalStoragePerTable::new(
                    &mut table_storage,
                    &self.shared.dirty_tracker,
                    table_id as u32,
                );
                let mut btree =
                    BTree::with_rightmost_hint(&mut wal_storage, root_page, rightmost_hint)?;
                btree.insert(&row_key, &buffers.mvcc_buffer)?;
                rightmost_hint = btree.rightmost_hint();
                root_page = btree.root_page();
            } else {
                let mut btree =
                    BTree::with_rightmost_hint(&mut *table_storage, root_page, rightmost_hint)?;
                btree.insert(&row_key, &buffers.mvcc_buffer)?;
                rightmost_hint = btree.rightmost_hint();
                root_page = btree.root_page();
            }

            {
                let mut active_txn = self.active_txn.lock();
                if let Some(ref mut txn) = *active_txn {
                    txn.add_write_entry(WriteEntry {
                        table_id: table_id as u32,
                        key: row_key.to_vec(),
                        page_id: 0,
                        offset: 0,
                        undo_page_id: None,
                        undo_offset: None,
                        is_insert: true,
                    });
                }
            }

            for (col_idx, index_key, _, _) in &unique_column_keys {
                if let Some(value) = values.get(*col_idx) {
                    if value.is_null() {
                        continue;
                    }

                    let index_storage_arc = storage_map
                        .get(index_key)
                        .ok_or_else(|| eyre::eyre!("index storage not found"))?;
                    let mut index_storage = index_storage_arc.write();

                    buffers.key_buffer.clear();
                    Self::encode_value_as_key(value, &mut buffers.key_buffer);

                    let row_id_bytes = row_id.to_be_bytes();

                    let hint = index_rightmost_hints.get(index_key).copied().flatten();
                    let index_rp = *index_root_pages.get(index_key).expect("index root page should be initialized");
                    let mut index_btree =
                        BTree::with_rightmost_hint(&mut *index_storage, index_rp, hint)?;
                    index_btree.insert(&buffers.key_buffer, &row_id_bytes)?;
                    if let Some(entry) = index_rightmost_hints.get_mut(index_key) {
                        *entry = index_btree.rightmost_hint();
                    }
                    let new_rp = index_btree.root_page();
                    if new_rp != index_rp {
                        index_root_pages.insert(index_key.clone(), new_rp);
                    }
                }
            }

            for (col_indices, key) in &unique_indexes {
                let all_non_null = col_indices
                    .iter()
                    .all(|&idx| values.get(idx).map(|v| !v.is_null()).unwrap_or(false));

                if all_non_null && storage_map.contains_key(key) {
                    let index_storage_arc = storage_map.get(key).unwrap();
                    let mut index_storage = index_storage_arc.write();

                    buffers.key_buffer.clear();
                    for &col_idx in col_indices {
                        if let Some(value) = values.get(col_idx) {
                            Self::encode_value_as_key(value, &mut buffers.key_buffer);
                        }
                    }

                    let row_id_bytes = row_id.to_be_bytes();

                    let hint = index_rightmost_hints.get(key).copied().flatten();
                    let index_rp = *index_root_pages.get(key).expect("index root page should be initialized");
                    let mut index_btree =
                        BTree::with_rightmost_hint(&mut *index_storage, index_rp, hint)?;
                    index_btree.insert(&buffers.key_buffer, &row_id_bytes)?;
                    if let Some(entry) = index_rightmost_hints.get_mut(key) {
                        *entry = index_btree.rightmost_hint();
                    }
                    let new_rp = index_btree.root_page();
                    if new_rp != index_rp {
                        index_root_pages.insert(key.clone(), new_rp);
                    }
                }
            }

            for (col_indices, key) in &secondary_indexes {
                if storage_map.contains_key(key) {
                    let index_storage_arc = storage_map.get(key).unwrap();
                    let mut index_storage = index_storage_arc.write();

                    buffers.key_buffer.clear();
                    for &col_idx in col_indices {
                        if let Some(value) = values.get(col_idx) {
                            Self::encode_value_as_key(value, &mut buffers.key_buffer);
                        }
                    }

                    buffers.key_buffer.extend_from_slice(&row_id.to_be_bytes());

                    let row_id_bytes = row_id.to_be_bytes();
                    let hint = index_rightmost_hints.get(key).copied().flatten();
                    let index_rp = *index_root_pages.get(key).expect("index root page should be initialized");
                    let mut index_btree =
                        BTree::with_rightmost_hint(&mut *index_storage, index_rp, hint)?;
                    index_btree.insert(&buffers.key_buffer, &row_id_bytes)?;
                    if let Some(entry) = index_rightmost_hints.get_mut(key) {
                        *entry = index_btree.rightmost_hint();
                    }
                    let new_rp = index_btree.root_page();
                    if new_rp != index_rp {
                        index_root_pages.insert(key.clone(), new_rp);
                    }
                }
            }

            for (index_name, col_idx) in &hnsw_indexes {
                if let Some(OwnedValue::Vector(vec)) = values.get(*col_idx) {
                    let hnsw = self.get_or_create_hnsw_index(schema_name, table_name, index_name)?;
                    let mut hnsw_guard = hnsw.write();
                    let random = Self::generate_random_for_hnsw(row_id);
                    hnsw_guard.insert(row_id, vec, random)?;
                }
            }

            count += 1;

            if let Some(ref mut rows) = returned_rows {
                let returning_cols = insert.returning.unwrap();
                let row_values: Vec<OwnedValue> = returning_cols
                    .iter()
                    .flat_map(|col| match col {
                        crate::sql::ast::SelectColumn::AllColumns => values.clone(),
                        crate::sql::ast::SelectColumn::TableAllColumns(_) => values.clone(),
                        crate::sql::ast::SelectColumn::Expr { expr, .. } => {
                            if let crate::sql::ast::Expr::Column(col_ref) = expr {
                                columns
                                    .iter()
                                    .position(|c| c.name().eq_ignore_ascii_case(col_ref.column))
                                    .and_then(|idx| values.get(idx).cloned())
                                    .map(|v| vec![v])
                                    .unwrap_or_default()
                            } else {
                                vec![]
                            }
                        }
                    })
                    .collect();
                rows.push(Row::new(row_values));
            }
        }

        if auto_increment_col_idx.is_some() && auto_increment_max > 0 {
            if let Some(storage_arc) = storage_map.get(&table_file_key) {
                let mut storage = storage_arc.write();
                let page = storage.page_mut(0)?;
                let header = TableFileHeader::from_bytes_mut(page)?;
                if auto_increment_max > header.auto_increment() {
                    header.set_auto_increment(auto_increment_max);
                }
            }
        }

        if let Some(hint) = rightmost_hint {
            if let Some(storage_arc) = storage_map.get(&table_file_key) {
                let mut storage = storage_arc.write();
                let page = storage.page_mut(0)?;
                let header = TableFileHeader::from_bytes_mut(page)?;
                header.set_rightmost_hint(hint);
            }
        }

        if count > 0 {
            if let Some(storage_arc) = storage_map.get(&table_file_key) {
                let mut storage = storage_arc.write();
                let page = storage.page_mut(0)?;
                let header = TableFileHeader::from_bytes_mut(page)?;
                let new_row_count = header.row_count().saturating_add(count as u64);
                header.set_row_count(new_row_count);
            }
        }

        if root_page != initial_root_page {
            if let Some(storage_arc) = storage_map.get(&table_file_key) {
                let mut storage = storage_arc.write();
                let page = storage.page_mut(0)?;
                let header = TableFileHeader::from_bytes_mut(page)?;
                header.set_root_page(root_page);
            }
        }

        if has_toast && toast_root_page != toast_initial_root_page {
            if let Some(ref key) = toast_file_key {
                if let Some(toast_storage_arc) = storage_map.get(key) {
                    let mut toast_storage = toast_storage_arc.write();
                    let page0 = toast_storage.page_mut(0)?;
                    let header = crate::storage::TableFileHeader::from_bytes_mut(page0)?;
                    header.set_root_page(toast_root_page);
                }
            }
        }

        for (key, new_rp) in &index_root_pages {
            let initial_rp = index_initial_root_pages.get(key).copied().unwrap_or(*new_rp);
            if *new_rp != initial_rp {
                if let Some(storage_arc) = storage_map.get(key) {
                    let mut storage = storage_arc.write();
                    let page = storage.page_mut(0)?;
                    let header = crate::storage::IndexFileHeader::from_bytes_mut(page)?;
                    header.set_root_page(*new_rp);
                }
            }
        }

        self.ensure_file_manager()?;
        let mut file_manager_guard = self.shared.file_manager.write();
        let file_manager = file_manager_guard.as_mut().unwrap();

        self.flush_wal_if_autocommit(file_manager, schema_name, table_name, table_id as u32)?;

        if has_toast {
            let toast_table_name_owned = crate::storage::toast::toast_table_name(table_name);
            self.flush_wal_if_autocommit(
                file_manager,
                schema_name,
                &toast_table_name_owned,
                toast_table_id,
            )?;
        }

        Ok(ExecuteResult::Insert {
            rows_affected: count,
            returned: returned_rows,
        })
    }

    fn fk_table_scan_check(
        schema_name: &str,
        fk_table: &str,
        ref_columns: &[crate::schema::ColumnDef],
        ref_col_idx: usize,
        value: &OwnedValue,
        storage_map: &hashbrown::HashMap<
            crate::storage::FileKey,
            std::sync::Arc<parking_lot::RwLock<crate::storage::MmapStorage>>,
        >,
    ) -> Result<bool> {
        let fk_key = crate::storage::FileManager::make_table_key(schema_name, fk_table);
        let ref_storage_arc = storage_map
            .get(&fk_key)
            .ok_or_else(|| eyre::eyre!("referenced table storage not found"))?;
        let mut ref_storage = ref_storage_arc.write();
        let ref_btree = BTree::new(&mut *ref_storage, 1)?;
        let ref_schema = create_record_schema(ref_columns);
        let mut ref_cursor = ref_btree.cursor_first()?;

        while ref_cursor.valid() {
            let existing_value = ref_cursor.value()?;
            let user_data = crate::database::dml::mvcc_helpers::get_user_data(existing_value);
            let existing_record = crate::records::RecordView::new(user_data, &ref_schema)?;
            let existing_values =
                OwnedValue::extract_row_from_record(&existing_record, ref_columns)?;

            if let Some(ref_val) = existing_values.get(ref_col_idx) {
                if !ref_val.is_null() && ref_val == value {
                    return Ok(true);
                }
            }
            ref_cursor.advance()?;
        }
        Ok(false)
    }

    pub(crate) fn generate_random_for_hnsw(row_id: u64) -> f64 {
        use std::time::SystemTime;
        let nanos = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0) as u64;
        let mut state = nanos ^ row_id;
        state ^= state >> 12;
        state ^= state << 25;
        state ^= state >> 27;
        (state.wrapping_mul(0x2545F4914F6CDD1D) as f64) / (u64::MAX as f64)
    }
}
