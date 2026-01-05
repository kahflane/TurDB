//! # DELETE Operation Module
//!
//! This module implements DELETE operations for TurDB, handling row deletion
//! with constraint validation and TOAST cleanup.
//!
//! ## Purpose
//!
//! DELETE operations remove rows from tables while:
//! - Checking FOREIGN KEY constraints (prevent deleting referenced rows)
//! - Cleaning up TOAST chunks for large values
//! - Recording undo data for transaction rollback
//! - Supporting RETURNING clause for deleted rows
//!
//! ## Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────────────┐
//! │                        DELETE Operation Flow                            │
//! ├─────────────────────────────────────────────────────────────────────────┤
//! │                                                                         │
//! │   DELETE FROM table WHERE condition                                     │
//! │       │                                                                 │
//! │       ▼                                                                 │
//! │   ┌─────────────────────────────────────────────────────────────────┐   │
//! │   │ 1. Scan table for matching rows                                 │   │
//! │   │    - Apply WHERE predicate                                      │   │
//! │   │    - Collect rows to delete                                     │   │
//! │   │    - Store values for FK check and RETURNING                    │   │
//! │   └─────────────────────────────────────────────────────────────────┘   │
//! │       │                                                                 │
//! │       ▼                                                                 │
//! │   ┌─────────────────────────────────────────────────────────────────┐   │
//! │   │ 2. Check FOREIGN KEY constraints                                │   │
//! │   │    - Find all child tables referencing this table               │   │
//! │   │    - Scan child tables for matching FK values                   │   │
//! │   │    - Reject delete if references exist                          │   │
//! │   └─────────────────────────────────────────────────────────────────┘   │
//! │       │                                                                 │
//! │       ▼                                                                 │
//! │   ┌─────────────────────────────────────────────────────────────────┐   │
//! │   │ 3. Handle RETURNING clause                                      │   │
//! │   │    - Build result rows from deleted values                      │   │
//! │   └─────────────────────────────────────────────────────────────────┘   │
//! │       │                                                                 │
//! │       ▼                                                                 │
//! │   ┌─────────────────────────────────────────────────────────────────┐   │
//! │   │ 4. Clean up TOAST chunks                                        │   │
//! │   │    - For each toasted column value                              │   │
//! │   │    - Delete all chunks from TOAST table                         │   │
//! │   └─────────────────────────────────────────────────────────────────┘   │
//! │       │                                                                 │
//! │       ▼                                                                 │
//! │   ┌─────────────────────────────────────────────────────────────────┐   │
//! │   │ 5. Delete rows from BTree                                       │   │
//! │   │    - Remove each row key                                        │   │
//! │   │    - WAL tracking if enabled                                    │   │
//! │   └─────────────────────────────────────────────────────────────────┘   │
//! │       │                                                                 │
//! │       ▼                                                                 │
//! │   ┌─────────────────────────────────────────────────────────────────┐   │
//! │   │ 6. Record transaction write entries with undo data              │   │
//! │   │    - Store old row value for potential reinsert                 │   │
//! │   └─────────────────────────────────────────────────────────────────┘   │
//! │       │                                                                 │
//! │       ▼                                                                 │
//! │   ┌─────────────────────────────────────────────────────────────────┐   │
//! │   │ 7. Update table row count in header                             │   │
//! │   └─────────────────────────────────────────────────────────────────┘   │
//! │                                                                         │
//! └─────────────────────────────────────────────────────────────────────────┘
//! ```
//!
//! ## Foreign Key Constraint Checking
//!
//! DELETE checks for referencing rows in child tables:
//!
//! 1. Scans catalog for all FK constraints pointing to this table
//! 2. For each matching row to delete, checks if any child row references it
//! 3. If reference exists, rejects the delete with descriptive error
//!
//! Note: ON DELETE CASCADE is not yet implemented.
//!
//! ## Performance Characteristics
//!
//! - DELETE without WHERE: O(n) scan + O(n * log n) for n deletions
//! - DELETE with WHERE: O(n) scan + O(m * log n) for m matching rows
//! - FK check: O(n * k) where k is number of referencing tables
//! - TOAST cleanup: O(chunks) per toasted value
//!
//! ## Thread Safety
//!
//! DELETE acquires write lock on file_manager. Transaction write entries
//! include undo data (old row value) for rollback support.

use crate::btree::BTree;
use crate::database::dml::mvcc_helpers::{get_user_data, wrap_record_for_delete};
use crate::database::macros::with_btree_storage;
use crate::database::row::Row;
use crate::database::{Database, ExecuteResult};
use crate::mvcc::WriteEntry;
use crate::records::RecordView;
use crate::schema::table::{Constraint, IndexType};
use crate::sql::predicate::CompiledPredicate;
use crate::storage::{IndexFileHeader, TableFileHeader, DEFAULT_SCHEMA};
use crate::types::{create_column_map, create_record_schema, owned_values_to_values, OwnedValue};
use bumpalo::Bump;
use eyre::{bail, Result};
use smallvec::SmallVec;
use std::sync::atomic::Ordering;

impl Database {
    pub(crate) fn execute_delete(
        &self,
        delete: &crate::sql::ast::DeleteStmt<'_>,
        params: &[OwnedValue],
        arena: &Bump,
    ) -> Result<ExecuteResult> {
        self.ensure_catalog()?;
        self.ensure_file_manager()?;

        let catalog_guard = self.shared.catalog.read();
        let catalog = catalog_guard.as_ref().unwrap();

        let schema_name = delete.table.schema.unwrap_or(DEFAULT_SCHEMA);
        let table_name = delete.table.name;

        let table_def = catalog.resolve_table(table_name)?;
        let table_id = table_def.id();
        let columns = table_def.columns().to_vec();
        let has_toast = table_def.has_toast();

        let secondary_indexes: Vec<(String, Vec<usize>)> = table_def
            .indexes()
            .iter()
            .filter(|idx| idx.index_type() == IndexType::BTree)
            .map(|idx| {
                let col_indices: Vec<usize> = idx
                    .columns()
                    .iter()
                    .filter_map(|col_name| columns.iter().position(|c| c.name() == col_name))
                    .collect();
                (idx.name().to_string(), col_indices)
            })
            .collect();

        let unique_columns: Vec<(usize, String, bool)> = columns
            .iter()
            .enumerate()
            .filter_map(|(idx, col)| {
                let is_pk = col.has_constraint(&Constraint::PrimaryKey);
                let is_unique = col.has_constraint(&Constraint::Unique);
                if is_pk || is_unique {
                    let index_name = if is_pk {
                        format!("{}_pkey", col.name())
                    } else {
                        format!("{}_key", col.name())
                    };
                    Some((idx, index_name, is_pk))
                } else {
                    None
                }
            })
            .collect();

        let mut fk_references: Vec<(String, String, String, usize)> = Vec::new();
        for (schema_key, schema_val) in catalog.schemas() {
            for (child_table_name, child_table_def) in schema_val.tables() {
                for col in child_table_def.columns().iter() {
                    for constraint in col.constraints() {
                        if let Constraint::ForeignKey { table, column } = constraint {
                            if table == table_name {
                                let ref_col_idx =
                                    columns.iter().position(|c| c.name() == column).unwrap_or(0);
                                fk_references.push((
                                    schema_key.clone(),
                                    child_table_name.clone(),
                                    col.name().to_string(),
                                    ref_col_idx,
                                ));
                            }
                        }
                    }
                }
            }
        }

        let child_table_schemas: Vec<(
            String,
            String,
            Vec<crate::schema::table::ColumnDef>,
            usize,
        )> = fk_references
            .iter()
            .map(|(schema_key, child_name, fk_col_name, _ref_col_idx)| {
                let child_def = catalog
                    .schemas()
                    .get(schema_key)
                    .unwrap()
                    .tables()
                    .get(child_name)
                    .unwrap();
                let fk_col_idx = child_def
                    .columns()
                    .iter()
                    .position(|c| c.name() == fk_col_name)
                    .unwrap_or(0);
                (
                    schema_key.clone(),
                    child_name.clone(),
                    child_def.columns().to_vec(),
                    fk_col_idx,
                )
            })
            .collect();

        drop(catalog_guard);

        let schema = create_record_schema(&columns);
        let column_map = create_column_map(&columns);

        let predicate = delete
            .where_clause
            .map(|expr| CompiledPredicate::new(expr, column_map));

        let mut file_manager_guard = self.shared.file_manager.write();
        let file_manager = file_manager_guard.as_mut().unwrap();
        let storage_arc = file_manager.table_data_mut(schema_name, table_name)?;
        let mut storage = storage_arc.write();

        let root_page = 1u32;
        let btree = BTree::new(&mut *storage, root_page)?;

        let mut pk_lookup_info: Option<(Vec<u8>, OwnedValue)> = 'pk_analysis: {
            if let Some(crate::sql::ast::Expr::BinaryOp { left, op: crate::sql::ast::BinaryOperator::Eq, right }) = delete.where_clause.as_ref() {
                if let Some(pk_idx) = columns.iter().position(|c| c.has_constraint(&Constraint::PrimaryKey)) {
                    let pk_col_name = columns[pk_idx].name();

                    let val_expr = match (&**left, &**right) {
                        (crate::sql::ast::Expr::Column(c), val) if c.column.eq_ignore_ascii_case(pk_col_name) => Some(val),
                        (val, crate::sql::ast::Expr::Column(c)) if c.column.eq_ignore_ascii_case(pk_col_name) => Some(val),
                        _ => None,
                    };

                    if let Some(expr) = val_expr {
                        let val_opt = if let crate::sql::ast::Expr::Parameter(param_ref) = expr {
                             match param_ref {
                                 crate::sql::ast::ParameterRef::Anonymous => params.first().cloned(),
                                 crate::sql::ast::ParameterRef::Positional(idx) => if *idx > 0 { params.get((*idx - 1) as usize).cloned() } else { None },
                                 _ => None,
                             }
                        } else {
                            Self::eval_literal(expr).ok()
                        };

                        if let Some(val) = val_opt {
                            let pk_index_name = format!("{}_pkey", pk_col_name);
                            
                            if file_manager.index_exists(schema_name, table_name, &pk_index_name) {
                                if let Ok(index_storage_arc) = file_manager.index_data_mut(schema_name, table_name, &pk_index_name) {
                                    let mut index_storage = index_storage_arc.write();

                                    let index_root_page = {
                                        use crate::storage::IndexFileHeader;
                                        let page0 = index_storage.page(0)?;
                                        let header = IndexFileHeader::from_bytes(page0)?;
                                        header.root_page()
                                    };

                                    let index_btree = BTree::new(&mut *index_storage, index_root_page)?;

                                    let mut index_key = Vec::new();
                                    Self::encode_value_as_key(&val, &mut index_key);

                                    if let Some(handle) = index_btree.search(&index_key)? {
                                        let row_key = index_btree.get_value(&handle)?.to_vec();
                                        break 'pk_analysis Some((row_key, val));
                                    }
                                }
                            }
                            break 'pk_analysis None;
                        }
                    }
                }
            }
            None
        };

        let mut rows_to_delete: Vec<(Vec<u8>, Vec<u8>, Vec<OwnedValue>)> = Vec::new();
        let mut values_to_check: Vec<(usize, OwnedValue)> = Vec::new();

        loop {
            let mut cursor = if let Some((ref key, _)) = pk_lookup_info {
                 btree.cursor_seek(key)?
            } else {
                 btree.cursor_first()?
            };

        while cursor.valid() {
            let key = cursor.key()?;

            if let Some((ref target_key, _)) = pk_lookup_info {
                if key != target_key.as_slice() {
                    break;
                }
            }

            let value = cursor.value()?;

            let user_data = get_user_data(value);
            let record = RecordView::new(user_data, &schema)?;
            let row_values = OwnedValue::extract_row_from_record(&record, &columns)?;

            let should_delete = if let Some((_, ref target_val)) = pk_lookup_info {
                if let Some(pk_idx) = columns.iter().position(|c| c.has_constraint(&Constraint::PrimaryKey)) {
                    &row_values[pk_idx] == target_val
                } else {
                    false
                }
            } else if let Some(ref pred) = predicate {
                use crate::sql::executor::ExecutorRow;

                let values = owned_values_to_values(&row_values);
                let values_slice = arena.alloc_slice_fill_iter(values.into_iter());
                let exec_row = ExecutorRow::new(values_slice);
                pred.evaluate(&exec_row)
            } else {
                true
            };

            if should_delete {
                if !fk_references.is_empty() {
                    for (_, _, _, ref_col_idx) in &fk_references {
                        if let Some(v) = row_values.get(*ref_col_idx) {
                            values_to_check.push((*ref_col_idx, v.clone()));
                        }
                    }
                }
                rows_to_delete.push((key.to_vec(), value.to_vec(), row_values));
            }

            cursor.advance()?;
        }

        if pk_lookup_info.is_some() && rows_to_delete.is_empty() {
            pk_lookup_info = None;
            continue;
        }

        break;
        }

        if !values_to_check.is_empty() {
            for (child_schema, child_name, child_columns, fk_col_idx) in &child_table_schemas {
                let child_storage_arc = file_manager.table_data_mut(child_schema, child_name)?;
                let mut child_storage = child_storage_arc.write();
                let child_btree = BTree::new(&mut *child_storage, root_page)?;
                let mut child_cursor = child_btree.cursor_first()?;
                let child_record_schema = create_record_schema(child_columns);

                while child_cursor.valid() {
                    let child_value = child_cursor.value()?;
                    let child_user_data = get_user_data(child_value);
                    let child_record = RecordView::new(child_user_data, &child_record_schema)?;
                    let child_row =
                        OwnedValue::extract_row_from_record(&child_record, child_columns)?;

                    if let Some(child_fk_val) = child_row.get(*fk_col_idx) {
                        for (ref_col_idx, del_val) in &values_to_check {
                            if let Some((_, _, _, matching_ref_idx)) =
                                fk_references.iter().find(|(s, n, _, r)| {
                                    s == child_schema && n == child_name && r == ref_col_idx
                                })
                            {
                                if matching_ref_idx == ref_col_idx && child_fk_val == del_val {
                                    bail!(
                                        "FOREIGN KEY constraint violated: row in '{}' is still referenced by '{}'",
                                        table_name,
                                        child_name
                                    );
                                }
                            }
                        }
                    }

                    child_cursor.advance()?;
                }
            }
        }

        let rows_affected = rows_to_delete.len();

        let returned_rows: Option<Vec<Row>> = if let Some(returning_cols) = delete.returning {
            let mut rows = Vec::with_capacity(rows_to_delete.len());
            for (_key, _old_value, deleted_values) in &rows_to_delete {
                let row_values: Vec<OwnedValue> = returning_cols
                    .iter()
                    .flat_map(|col| match col {
                        crate::sql::ast::SelectColumn::AllColumns => deleted_values.clone(),
                        crate::sql::ast::SelectColumn::TableAllColumns(_) => deleted_values.clone(),
                        crate::sql::ast::SelectColumn::Expr { expr, .. } => {
                            if let crate::sql::ast::Expr::Column(col_ref) = expr {
                                columns
                                    .iter()
                                    .position(|c| c.name().eq_ignore_ascii_case(col_ref.column))
                                    .and_then(|idx| deleted_values.get(idx).cloned())
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
            Some(rows)
        } else {
            None
        };

        if has_toast {
            use crate::storage::toast::ToastPointer;
            for (_key, _old_value, row_values) in &rows_to_delete {
                for val in row_values.iter() {
                    if let OwnedValue::ToastPointer(ptr) = val {
                        if let Ok(pointer) = ToastPointer::decode(ptr) {
                            let _ = self.delete_toast_chunks(
                                file_manager,
                                schema_name,
                                table_name,
                                pointer.row_id(),
                                pointer.column_index(),
                                pointer.total_size,
                            );
                        }
                    }
                }
            }
        }

        let mut key_buf: SmallVec<[u8; 64]> = SmallVec::new();

        for (col_idx, index_name, _is_pk) in &unique_columns {
            if file_manager.index_exists(schema_name, table_name, index_name) {
                let index_storage_arc =
                    file_manager.index_data_mut(schema_name, table_name, index_name)?;
                let mut index_storage = index_storage_arc.write();

                let index_root_page = {
                    let page0 = index_storage.page(0)?;
                    let header = IndexFileHeader::from_bytes(page0)?;
                    header.root_page()
                };

                let mut index_btree = BTree::new(&mut *index_storage, index_root_page)?;

                for (_row_key, _old_value, row_values) in &rows_to_delete {
                    if let Some(value) = row_values.get(*col_idx) {
                        if !value.is_null() {
                            key_buf.clear();
                            Self::encode_value_as_key(value, &mut key_buf);
                            let _ = index_btree.delete(&key_buf);
                        }
                    }
                }
            }
        }

        for (index_name, col_indices) in &secondary_indexes {
            if col_indices.is_empty() {
                continue;
            }
            if file_manager.index_exists(schema_name, table_name, index_name) {
                let index_storage_arc =
                    file_manager.index_data_mut(schema_name, table_name, index_name)?;
                let mut index_storage = index_storage_arc.write();

                let index_root_page = {
                    let page0 = index_storage.page(0)?;
                    let header = IndexFileHeader::from_bytes(page0)?;
                    header.root_page()
                };

                let mut index_btree = BTree::new(&mut *index_storage, index_root_page)?;

                for (_row_key, _old_value, row_values) in &rows_to_delete {
                    let all_non_null = col_indices
                        .iter()
                        .all(|&idx| row_values.get(idx).is_some_and(|v| !v.is_null()));

                    if all_non_null {
                        key_buf.clear();
                        for &col_idx in col_indices {
                            if let Some(value) = row_values.get(col_idx) {
                                Self::encode_value_as_key(value, &mut key_buf);
                            }
                        }
                        let _ = index_btree.delete(&key_buf);
                    }
                }
            }
        }


        drop(storage);

        let wal_enabled = self.shared.wal_enabled.load(Ordering::Acquire);
        if wal_enabled {
            self.ensure_wal()?;
        }

        let (txn_id, in_transaction) = {
            let active_txn = self.active_txn.lock();
            if let Some(ref txn) = *active_txn {
                (txn.txn_id, true)
            } else {
                (self.shared.txn_manager.global_ts.fetch_add(1, Ordering::SeqCst), false)
            }
        };

        let storage_arc = file_manager.table_data_mut(schema_name, table_name)?;
        let mut storage = storage_arc.write();

        with_btree_storage!(wal_enabled, &mut *storage, &self.shared.dirty_tracker, table_id as u32, root_page, |btree_mut: &mut crate::btree::BTree<_>| {
            for (key, old_value, _row_values) in &rows_to_delete {
                let tombstone = wrap_record_for_delete(txn_id, old_value, in_transaction)?;
                if !btree_mut.update(key, &tombstone)? {
                    btree_mut.delete(key)?;
                    btree_mut.insert(key, &tombstone)?;
                }
            }
            Ok::<_, eyre::Report>(())
        });

        drop(storage);

        self.flush_wal_if_autocommit(file_manager, schema_name, table_name, table_id as u32)?;

        drop(file_manager_guard);

        {
            let mut active_txn = self.active_txn.lock();
            if let Some(ref mut txn) = *active_txn {
                for (key, old_value, _row_values) in rows_to_delete {
                    txn.add_write_entry_with_undo(
                        WriteEntry {
                            table_id: table_id as u32,
                            key,
                            page_id: 0,
                            offset: 0,
                            undo_page_id: None,
                            undo_offset: None,
                            is_insert: false,
                        },
                        old_value,
                    );
                }
            }
        }

        if rows_affected > 0 {
            let mut file_manager_guard = self.shared.file_manager.write();
            let file_manager = file_manager_guard.as_mut().unwrap();
            let storage_arc = file_manager.table_data_mut(schema_name, table_name)?;
            let mut storage = storage_arc.write();
            let page = storage.page_mut(0)?;
            let header = TableFileHeader::from_bytes_mut(page)?;
            let new_row_count = header.row_count().saturating_sub(rows_affected as u64);
            header.set_row_count(new_row_count);
        }

        Ok(ExecuteResult::Delete {
            rows_affected,
            returned: returned_rows,
        })
    }
}
