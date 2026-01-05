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
use crate::database::dml::mvcc_helpers::wrap_record_for_insert;
use crate::database::row::Row;
use crate::database::{Database, ExecuteResult};
use crate::mvcc::WriteEntry;
use crate::schema::table::Constraint;
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

        // Build row from params based on cached column count
        if params.len() != cached.column_count {
            eyre::bail!(
                "parameter count mismatch: expected {} but got {}",
                cached.column_count,
                params.len()
            );
        }

        // Use optimized insert_cached path
        let count = self.insert_cached(cached, params)?;

        Ok(ExecuteResult::Insert {
            rows_affected: count,
            returned: None,
        })
    }

    pub(crate) fn execute_insert_with_params(
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

        let table_def = catalog.resolve_table(table_name)?;
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

        let unique_indexes: Vec<(Vec<usize>, String)> = table_def
            .indexes()
            .iter()
            .filter(|idx| idx.is_unique())
            .filter(|idx| !unique_column_index_names.contains(idx.name()))
            .filter_map(|idx| {
                let col_indices: Vec<usize> = idx
                    .columns()
                    .iter()
                    .filter_map(|col_name| {
                        columns
                            .iter()
                            .position(|c| c.name().eq_ignore_ascii_case(col_name))
                    })
                    .collect();
                if col_indices.is_empty() {
                    None
                } else {
                    Some((col_indices, idx.name().to_string()))
                }
            })
            .collect();

        let fk_constraints: Vec<(usize, String, String)> = columns
            .iter()
            .enumerate()
            .flat_map(|(idx, col)| {
                col.constraints().iter().filter_map(move |c| {
                    if let Constraint::ForeignKey { table, column } = c {
                        Some((idx, table.clone(), column.clone()))
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
                    columns.iter().position(|c| c.name().eq_ignore_ascii_case(col_name))
                })
                .collect()
        });

        let auto_increment_col_idx: Option<usize> = columns
            .iter()
            .position(|c| c.has_constraint(&Constraint::AutoIncrement));

        let mut param_idx = 0usize;

        let rows_to_insert: Vec<Vec<OwnedValue>> = match &insert.source {
            crate::sql::ast::InsertSource::Values(values) => {
                let mut result = Vec::with_capacity(values.len());
                for row_exprs in values.iter() {
                    let mut row = vec![OwnedValue::Null; columns.len()];

                    if let Some(ref col_indices) = insert_col_indices {
                        for (val_idx, &col_idx) in col_indices.iter().enumerate() {
                            if let Some(expr) = row_exprs.get(val_idx) {
                                let data_type = column_types.get(col_idx);
                                row[col_idx] = Self::eval_expr_with_params(expr, data_type, params, &mut param_idx)?;
                            }
                        }
                    } else {
                        for (idx, expr) in row_exprs.iter().enumerate() {
                            if idx < columns.len() {
                                let data_type = column_types.get(idx);
                                row[idx] = Self::eval_expr_with_params(expr, data_type, params, &mut param_idx)?;
                            }
                        }
                    }

                    result.push(row);
                }
                result
            }
            crate::sql::ast::InsertSource::Select(select_stmt) => {
                drop(catalog_guard);
                let select_rows = self.execute_select_internal(select_stmt)?;
                select_rows.into_iter().map(|row| row.values).collect()
            }
            crate::sql::ast::InsertSource::Default => {
                bail!("DEFAULT VALUES insert not supported")
            }
        };

        let catalog_guard = self.shared.catalog.read();
        let root_page = 1u32;
        let validator = ConstraintValidator::new(&table_def_for_validator);

        drop(catalog_guard);

        let mut file_manager_guard = self.shared.file_manager.write();
        let file_manager = file_manager_guard.as_mut().unwrap();

        let mut count = 0;
        let mut key_buf: SmallVec<[u8; 64]> = SmallVec::new();
        let mut returned_rows: Option<Vec<Row>> = insert.returning.map(|_| Vec::new());

        let mut auto_increment_current = if auto_increment_col_idx.is_some() {
            let storage_arc = file_manager.table_data_mut(schema_name, table_name)?;
            let storage = storage_arc.write();
            let page = storage.page(0)?;
            let header = TableFileHeader::from_bytes(page)?;
            header.auto_increment()
        } else {
            0
        };
        let mut auto_increment_max = auto_increment_current;

        let mut rightmost_hint: Option<u32> = {
            let storage_arc = file_manager.table_data_mut(schema_name, table_name)?;
            let storage = storage_arc.write();
            let page = storage.page(0)?;
            let header = TableFileHeader::from_bytes(page)?;
            let hint = header.rightmost_hint();
            if hint > 0 { Some(hint) } else { None }
        };

        let mut toast_rightmost_hints: SmallVec<[Option<u32>; 8]> = SmallVec::new();
        let toastable_col_indices: SmallVec<[usize; 8]> = if has_toast {
            toast_rightmost_hints.resize(columns.len(), None);
            columns.iter()
                .enumerate()
                .filter(|(_, col)| col.data_type().is_toastable())
                .map(|(idx, _)| idx)
                .collect()
        } else {
            SmallVec::new()
        };

        let table_file_key = crate::storage::FileManager::make_table_key(schema_name, table_name);
        let mut record_builder = crate::records::RecordBuilder::new(&schema);

        use crate::storage::FileKey;
        let toast_file_key: Option<FileKey>;
        let mut toast_table_id: u32 = 0;
        let mut toast_root_page: u32 = 1;
        let mut toast_initial_root_page: u32 = 1;
        let mut toast_rightmost_hint: Option<u32> = None;
        if has_toast {
            let toast_table_name_owned = crate::storage::toast::toast_table_name(table_name);
            toast_file_key = Some(crate::storage::FileManager::make_table_key(schema_name, &toast_table_name_owned));
            let toast_storage_arc = file_manager.table_data_mut(schema_name, &toast_table_name_owned)?;
            let toast_storage = toast_storage_arc.write();
            let page0 = toast_storage.page(0)?;
            let header = crate::storage::TableFileHeader::from_bytes(page0)?;
            toast_table_id = header.table_id() as u32;
            toast_root_page = header.root_page();
            toast_initial_root_page = toast_root_page;
            let hint = header.rightmost_hint();
            toast_rightmost_hint = if hint > 0 { Some(hint) } else { None };
        } else {
            toast_file_key = None;
        }

        let unique_column_keys: Vec<(usize, FileKey, bool, bool)> = unique_columns
            .iter()
            .filter(|(_, _, _, is_auto_increment)| !is_auto_increment)
            .filter_map(|(col_idx, index_name, is_pk, _is_auto_increment)| {
                if file_manager.index_exists(schema_name, table_name, index_name) {
                    let _ = file_manager.index_data_mut(schema_name, table_name, index_name);
                    let key = crate::storage::FileManager::make_index_key(schema_name, table_name, index_name);
                    Some((*col_idx, key, *is_pk, false))
                } else {
                    None
                }
            })
            .collect();

        for row_values in rows_to_insert.iter() {
            let mut values: Vec<OwnedValue> = row_values.clone();

            if let Some(auto_col_idx) = auto_increment_col_idx {
                if values.get(auto_col_idx).is_none_or(|v| v.is_null()) {
                    auto_increment_current = auto_increment_current.checked_add(1).ok_or_else(|| {
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

                for (col_idx, fk_table, fk_column) in &fk_constraints {
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

                        let ref_storage_arc = file_manager.table_data_mut(schema_name, fk_table)?;
                        let mut ref_storage = ref_storage_arc.write();
                        let ref_btree = BTree::new(&mut *ref_storage, 1)?;
                        let ref_schema = create_record_schema(ref_columns);
                        let mut ref_cursor = ref_btree.cursor_first()?;

                        let mut found = false;
                        while ref_cursor.valid() {
                            let existing_value = ref_cursor.value()?;
                            let user_data = crate::database::dml::mvcc_helpers::get_user_data(existing_value);
                            let existing_record =
                                crate::records::RecordView::new(user_data, &ref_schema)?;
                            let existing_values =
                                OwnedValue::extract_row_from_record(&existing_record, ref_columns)?;

                            if let Some(ref_val) = existing_values.get(ref_col_idx.unwrap()) {
                                if !ref_val.is_null() && ref_val == value {
                                    found = true;
                                    break;
                                }
                            }
                            ref_cursor.advance()?;
                        }

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

                    let index_storage_arc = file_manager.index_data_mut_with_key(index_key)
                        .ok_or_else(|| eyre::eyre!("index storage not found"))?;
                    let mut index_storage = index_storage_arc.write();
                    let index_btree = BTree::new(&mut *index_storage, root_page)?;

                    key_buf.clear();
                    Self::encode_value_as_key(value, &mut key_buf);

                    if let Some(handle) = index_btree.search(&key_buf)? {
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
                for (col_indices, index_name) in &unique_indexes {
                    let all_non_null = col_indices
                        .iter()
                        .all(|&idx| values.get(idx).map(|v| !v.is_null()).unwrap_or(false));

                    if all_non_null
                        && file_manager.index_exists(schema_name, table_name, index_name)
                    {
                        let index_storage_arc =
                            file_manager.index_data_mut(schema_name, table_name, index_name)?;
                        let mut index_storage = index_storage_arc.write();
                        let index_btree = BTree::new(&mut *index_storage, root_page)?;

                        key_buf.clear();
                        for &col_idx in col_indices {
                            if let Some(value) = values.get(col_idx) {
                                Self::encode_value_as_key(value, &mut key_buf);
                            }
                        }

                        if let Some(handle) = index_btree.search(&key_buf)? {
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
                                let table_storage_arc =
                                    file_manager.table_data_mut(schema_name, table_name)?;
                                let mut table_storage = table_storage_arc.write();
                                let btree = BTree::new(&mut *table_storage, root_page)?;

                                if let Some(handle) = btree.search(&existing_key)? {
                                    let existing_value = btree.get_value(&handle)?;
                                    let user_data = crate::database::dml::mvcc_helpers::get_user_data(existing_value);
                                    let record = crate::records::RecordView::new(user_data, &schema)?;
                                    let mut existing_values =
                                        OwnedValue::extract_row_from_record(&record, &columns)?;

                                    for assignment in assignments.iter() {
                                        if let Some(col_idx) = columns
                                            .iter()
                                            .position(|c| {
                                                c.name()
                                                    .eq_ignore_ascii_case(assignment.column.column)
                                            })
                                        {
                                            let new_value = Self::eval_literal(assignment.value)?;
                                            existing_values[col_idx] = new_value;
                                        }
                                    }

                                    let user_record =
                                        OwnedValue::build_record_from_values(&existing_values, &schema)?;
                                    
                                    let (txn_id, in_transaction) = {
                                        let active_txn = self.active_txn.lock();
                                        if let Some(ref txn) = *active_txn {
                                            (txn.txn_id, true)
                                        } else {
                                            (self.shared.txn_manager.global_ts.fetch_add(1, Ordering::SeqCst), false)
                                        }
                                    };
                                    let updated_record = wrap_record_for_insert(txn_id, &user_record, in_transaction);

                                    drop(btree);
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
                                                crate::sql::ast::SelectColumn::TableAllColumns(_) => {
                                                    existing_values.clone()
                                                }
                                                crate::sql::ast::SelectColumn::Expr { expr, .. } => {
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
                use crate::storage::toast::{needs_toast, make_chunk_key, ToastPointer, TOAST_CHUNK_SIZE};

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
                            let toast_storage_arc = file_manager.table_data_mut_with_key(toast_key)
                                .ok_or_else(|| eyre::eyre!("toast storage not found"))?;
                            let mut toast_storage = toast_storage_arc.write();

                            let pointer = ToastPointer::new(row_id, col_idx as u16, data.len() as u64);
                            let chunk_id = pointer.chunk_id;

                            let (new_hint, new_root) = if wal_enabled {
                                let mut wal_storage =
                                    WalStoragePerTable::new(&mut *toast_storage, &self.shared.dirty_tracker, toast_table_id);
                                let mut btree = BTree::with_rightmost_hint(&mut wal_storage, toast_root_page, toast_rightmost_hint)?;
                                for (seq, chunk) in data.chunks(TOAST_CHUNK_SIZE).enumerate() {
                                    let chunk_key = make_chunk_key(chunk_id, seq as u32);
                                    btree.insert(&chunk_key, chunk)?;
                                }
                                (btree.rightmost_hint(), btree.root_page())
                            } else {
                                let mut btree = BTree::with_rightmost_hint(&mut *toast_storage, toast_root_page, toast_rightmost_hint)?;
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

            let table_storage_arc = file_manager.table_data_mut_with_key(&table_file_key)
                .ok_or_else(|| eyre::eyre!("table storage not found in cache"))?;
            let mut table_storage = table_storage_arc.write();

            let user_record = OwnedValue::build_record_with_builder(&values, &mut record_builder)?;

            let (txn_id, in_transaction) = {
                let active_txn = self.active_txn.lock();
                if let Some(ref txn) = *active_txn {
                    (txn.txn_id, true)
                } else {
                    (self.shared.txn_manager.global_ts.fetch_add(1, Ordering::SeqCst), false)
                }
            };
            let record_data = wrap_record_for_insert(txn_id, &user_record, in_transaction);

            if wal_enabled {
                let mut wal_storage =
                    WalStoragePerTable::new(&mut *table_storage, &self.shared.dirty_tracker, table_id as u32);
                let mut btree = BTree::with_rightmost_hint(&mut wal_storage, root_page, rightmost_hint)?;
                btree.insert(&row_key, &record_data)?;
                rightmost_hint = btree.rightmost_hint();
            } else {
                let mut btree = BTree::with_rightmost_hint(&mut *table_storage, root_page, rightmost_hint)?;
                btree.insert(&row_key, &record_data)?;
                rightmost_hint = btree.rightmost_hint();
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

                    let index_storage_arc = file_manager.index_data_mut_with_key(index_key)
                        .ok_or_else(|| eyre::eyre!("index storage not found"))?;
                    let mut index_storage = index_storage_arc.write();

                    key_buf.clear();
                    Self::encode_value_as_key(value, &mut key_buf);

                    let row_id_bytes = row_id.to_be_bytes();

                    let mut index_btree = BTree::new(&mut *index_storage, root_page)?;
                    index_btree.insert(&key_buf, &row_id_bytes)?;
                }
            }

            for (col_indices, index_name) in &unique_indexes {
                let all_non_null = col_indices
                    .iter()
                    .all(|&idx| values.get(idx).map(|v| !v.is_null()).unwrap_or(false));

                if all_non_null && file_manager.index_exists(schema_name, table_name, index_name) {
                    let index_storage_arc =
                        file_manager.index_data_mut(schema_name, table_name, index_name)?;
                    let mut index_storage = index_storage_arc.write();

                    key_buf.clear();
                    for &col_idx in col_indices {
                        if let Some(value) = values.get(col_idx) {
                            Self::encode_value_as_key(value, &mut key_buf);
                        }
                    }

                    let row_id_bytes = row_id.to_be_bytes();

                    let mut index_btree = BTree::new(&mut *index_storage, root_page)?;
                    index_btree.insert(&key_buf, &row_id_bytes)?;
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
            let storage_arc = file_manager.table_data_mut(schema_name, table_name)?;
            let mut storage = storage_arc.write();
            let page = storage.page_mut(0)?;
            let header = TableFileHeader::from_bytes_mut(page)?;
            if auto_increment_max > header.auto_increment() {
                header.set_auto_increment(auto_increment_max);
            }
        }

        if let Some(hint) = rightmost_hint {
            let storage_arc = file_manager.table_data_mut(schema_name, table_name)?;
            let mut storage = storage_arc.write();
            let page = storage.page_mut(0)?;
            let header = TableFileHeader::from_bytes_mut(page)?;
            header.set_rightmost_hint(hint);
        }

        if count > 0 {
            let storage_arc = file_manager.table_data_mut(schema_name, table_name)?;
            let mut storage = storage_arc.write();
            let page = storage.page_mut(0)?;
            let header = TableFileHeader::from_bytes_mut(page)?;
            let new_row_count = header.row_count().saturating_add(count as u64);
            header.set_row_count(new_row_count);
        }

        if has_toast && toast_root_page != toast_initial_root_page {
            let toast_table_name_owned = crate::storage::toast::toast_table_name(table_name);
            let toast_storage_arc = file_manager.table_data_mut(schema_name, &toast_table_name_owned)?;
            let mut toast_storage = toast_storage_arc.write();
            let page0 = toast_storage.page_mut(0)?;
            let header = crate::storage::TableFileHeader::from_bytes_mut(page0)?;

            header.set_root_page(toast_root_page);
        }

        self.flush_wal_if_autocommit(file_manager, schema_name, table_name, table_id as u32)?;

        if has_toast {
            let toast_table_name_owned = crate::storage::toast::toast_table_name(table_name);
            self.flush_wal_if_autocommit(file_manager, schema_name, &toast_table_name_owned, toast_table_id)?;
        }

        Ok(ExecuteResult::Insert {
            rows_affected: count,
            returned: returned_rows,
        })
    }
}
