//! # UPDATE Operation Module
//!
//! This module implements UPDATE operations for TurDB, handling both simple
//! UPDATE statements and UPDATE...FROM syntax for joins.
//!
//! ## Purpose
//!
//! UPDATE operations modify existing rows while:
//! - Validating constraints on new values
//! - Processing TOAST for large updated values
//! - Supporting UPDATE...FROM for join-based updates
//! - Recording undo data for transaction rollback
//!
//! ## Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────────────┐
//! │                        UPDATE Operation Flow                            │
//! ├─────────────────────────────────────────────────────────────────────────┤
//! │                                                                         │
//! │   UPDATE table SET col=val WHERE condition                              │
//! │       │                                                                 │
//! │       ▼                                                                 │
//! │   ┌─────────────────────────────────────────────────────────────────┐   │
//! │   │ 1. Scan table for matching rows                                 │   │
//! │   │    - Apply WHERE predicate                                      │   │
//! │   │    - Collect rows to update                                     │   │
//! │   └─────────────────────────────────────────────────────────────────┘   │
//! │       │                                                                 │
//! │       ▼                                                                 │
//! │   ┌─────────────────────────────────────────────────────────────────┐   │
//! │   │ 2. For each matching row:                                       │   │
//! │   │    a. Store old value for undo                                  │   │
//! │   │    b. Apply SET assignments                                     │   │
//! │   │    c. Validate constraints on new values                        │   │
//! │   │    d. Check for UNIQUE violations                               │   │
//! │   └─────────────────────────────────────────────────────────────────┘   │
//! │       │                                                                 │
//! │       ▼                                                                 │
//! │   ┌─────────────────────────────────────────────────────────────────┐   │
//! │   │ 3. Handle TOAST                                                 │   │
//! │   │    - Delete old TOAST chunks if column was toasted              │   │
//! │   │    - Toast new value if exceeds threshold                       │   │
//! │   └─────────────────────────────────────────────────────────────────┘   │
//! │       │                                                                 │
//! │       ▼                                                                 │
//! │   ┌─────────────────────────────────────────────────────────────────┐   │
//! │   │ 4. Apply updates to BTree                                       │   │
//! │   │    - Delete old record                                          │   │
//! │   │    - Insert updated record                                      │   │
//! │   │    - WAL tracking if enabled                                    │   │
//! │   └─────────────────────────────────────────────────────────────────┘   │
//! │       │                                                                 │
//! │       ▼                                                                 │
//! │   ┌─────────────────────────────────────────────────────────────────┐   │
//! │   │ 5. Record transaction write entries with undo data              │   │
//! │   └─────────────────────────────────────────────────────────────────┘   │
//! │                                                                         │
//! └─────────────────────────────────────────────────────────────────────────┘
//! ```
//!
//! ## UPDATE...FROM Syntax
//!
//! Supports PostgreSQL-style UPDATE with FROM clause:
//! ```sql
//! UPDATE target SET col = source.val
//! FROM source
//! WHERE target.id = source.id
//! ```
//!
//! This performs a join between target and source tables, updating target
//! rows based on matching source rows.
//!
//! ## Performance Characteristics
//!
//! - Simple UPDATE: O(n) scan + O(m * log n) for m matching rows
//! - UPDATE...FROM: O(n * k) for cartesian product, filtered by predicate
//! - UNIQUE check: O(n) scan of existing rows (could be optimized)
//! - TOAST cleanup: O(chunks) per toasted column
//!
//! ## Thread Safety
//!
//! UPDATE acquires write lock on file_manager. Transaction write entries
//! include undo data (old row value) for rollback support.

use crate::btree::BTree;
use crate::database::dml::mvcc_helpers::{get_user_data, wrap_record_for_update};
use crate::database::macros::with_btree_storage;
use crate::database::row::Row;
use crate::database::{Database, ExecuteResult};
use crate::mvcc::WriteEntry;
use crate::records::RecordView;
use crate::schema::table::{Constraint, IndexType};
use crate::sql::decoder::RecordDecoder;
use crate::sql::executor::ExecutorRow;
use crate::sql::predicate::CompiledPredicate;
use crate::storage::{IndexFileHeader, DEFAULT_SCHEMA};
use crate::types::{create_record_schema, owned_values_to_values, OwnedValue, Value};
use bumpalo::Bump;
use eyre::{bail, Result, WrapErr};
use hashbrown::HashSet;
use smallvec::SmallVec;
use std::borrow::Cow;
use std::sync::atomic::Ordering;

fn count_params_in_expr(expr: &crate::sql::ast::Expr) -> usize {
    use crate::sql::ast::Expr;
    match expr {
        Expr::Parameter(_) => 1,
        Expr::BinaryOp { left, right, .. } => count_params_in_expr(left) + count_params_in_expr(right),
        Expr::UnaryOp { expr, .. } => count_params_in_expr(expr),
        Expr::Cast { expr, .. } => count_params_in_expr(expr),
        _ => 0,
    }
}

impl Database {
    pub(crate) fn execute_update(
        &self,
        update: &crate::sql::ast::UpdateStmt<'_>,
        params: &[OwnedValue],
        arena: &Bump,
    ) -> Result<ExecuteResult> {
        self.ensure_catalog()?;
        self.ensure_file_manager()?;

        let catalog_guard = self.shared.catalog.read();
        let catalog = catalog_guard.as_ref().unwrap();

        let schema_name = update.table.schema.unwrap_or(DEFAULT_SCHEMA);
        let table_name = update.table.name;
        let table_alias = update.table.alias;

        let table_def = catalog.resolve_table(table_name)?.clone();
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

        #[allow(clippy::type_complexity)]
        let from_tables_data: Option<Vec<(
            String,
            String,
            Option<&str>,
            Vec<crate::schema::table::ColumnDef>,
        )>> = if let Some(from_clause) = update.from {
            let mut tables = Vec::new();
            Self::extract_tables_from_clause(*from_clause, catalog, &mut tables)?;
            Some(tables)
        } else {
            None
        };

        drop(catalog_guard);

        let schema = create_record_schema(&columns);

        if let Some(from_tables) = from_tables_data {
            return self.execute_update_with_from(
                update,
                arena,
                schema_name,
                table_name,
                table_alias,
                &table_def,
                table_id as usize,
                &columns,
                &schema,
                from_tables,
            );
        }

        let assignment_indices: Vec<(usize, &crate::sql::ast::Expr<'_>)> = update
            .assignments
            .iter()
            .filter_map(|a| {
                columns
                    .iter()
                    .position(|c| c.name().eq_ignore_ascii_case(a.column.column))
                    .map(|idx| (idx, a.value))
            })
            .collect();

        let mut file_manager_guard = self.shared.file_manager.write();
        let file_manager = file_manager_guard.as_mut().unwrap();
        let storage_arc = file_manager.table_data_mut(schema_name, table_name)?;
        let mut storage = storage_arc.write();

        let column_types: Vec<crate::records::types::DataType> =
            columns.iter().map(|c| c.data_type()).collect();
        let decoder = crate::sql::decoder::SimpleDecoder::new(column_types);

        let root_page = 1u32;
        let btree = BTree::new(&mut *storage, root_page)?;

        let mut pk_lookup_info: Option<(Vec<u8>, OwnedValue)> = 'pk_analysis: {
            if let Some(ref w) = update.where_clause {
                if let crate::sql::ast::Expr::BinaryOp { left, op: crate::sql::ast::BinaryOperator::Eq, right } = w {
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
                                     crate::sql::ast::ParameterRef::Anonymous => {
                                         let mut param_offset = 0;
                                         for assign in update.assignments {
                                             param_offset += count_params_in_expr(&assign.value);
                                         }
                                         params.get(param_offset).cloned()
                                     },
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
            }
            None
        };

        #[allow(clippy::type_complexity)]
        let mut rows_to_update: Vec<(Vec<u8>, Vec<u8>, Vec<OwnedValue>, Vec<OwnedValue>, Vec<(usize, OwnedValue)>)> = Vec::new();

        let mut precomputed_assignments: Vec<(usize, OwnedValue)> = Vec::new();
        let set_param_count: usize;
        {
            let mut param_idx = 0;
            for (col_idx, value_expr) in &assignment_indices {
                let val = Self::eval_expr_with_params(value_expr, None, Some(params), &mut param_idx)?;
                precomputed_assignments.push((*col_idx, val));
            }
            set_param_count = param_idx;
        }

        let predicate: Option<crate::sql::predicate::CompiledPredicate> = if let Some(ref w) = update.where_clause {
             let col_map = columns.iter().enumerate().map(|(i, c)| (c.name().to_string(), i)).collect();
             Some(crate::sql::predicate::CompiledPredicate::with_params(w, col_map, params, set_param_count))
        } else {
             None
        };

        let modified_col_indices: HashSet<usize> = assignment_indices
            .iter()
            .map(|(idx, _)| *idx)
            .collect();

        let unique_col_indices: Vec<usize> = columns
            .iter()
            .enumerate()
            .filter(|(idx, col)| {
                (col.has_constraint(&Constraint::Unique) || col.has_constraint(&Constraint::PrimaryKey))
                && modified_col_indices.contains(idx)
            })
            .map(|(idx, _)| idx)
            .collect();

        let can_onepass = pk_lookup_info.is_some()
            && unique_col_indices.is_empty()
            && !has_toast;

        if can_onepass {
            if let Some((ref target_key, ref target_val)) = pk_lookup_info {
                
                let cursor = btree.cursor_seek(target_key)?;

                if cursor.valid() && cursor.key()? == target_key.as_slice() {
                    let key = cursor.key()?;
                    let value = cursor.value()?;
                    let user_data = get_user_data(value);
                    let values = decoder.decode(key, user_data)?;
                    let mut row_values: Vec<OwnedValue> = values.into_iter().map(OwnedValue::from).collect();
                    let pk_idx = columns.iter()
                        .position(|c| c.has_constraint(&Constraint::PrimaryKey))
                        .unwrap();

                    if &row_values[pk_idx] == target_val {
                        let old_value = value.to_vec();

                        for (col_idx, val) in &precomputed_assignments {
                            row_values[*col_idx] = val.clone();
                        }

                        let validator = crate::constraints::ConstraintValidator::new(&table_def);
                        validator.validate_update(&row_values)?;

                        for (col_idx, col) in columns.iter().enumerate() {
                            for constraint in col.constraints() {
                                if let Constraint::Check(expr_str) = constraint {
                                    let col_value = row_values.get(col_idx);
                                    if !Self::evaluate_check_expression(expr_str, col.name(), col_value) {
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

                        let user_record = OwnedValue::build_record_from_values(&row_values, &schema)?;
                        
                        let (txn_id, in_transaction) = {
                            let active_txn = self.active_txn.lock();
                            if let Some(ref txn) = *active_txn {
                                (txn.txn_id, true)
                            } else {
                                (self.shared.txn_manager.global_ts.fetch_add(1, Ordering::SeqCst), false)
                            }
                        };
                        let record_data = wrap_record_for_update(txn_id, &user_record, 0, 0, in_transaction);

                        drop(cursor);
                        drop(btree);
                        drop(storage);

                        let wal_enabled = self.shared.wal_enabled.load(Ordering::Acquire);
                        if wal_enabled {
                            self.ensure_wal()?;
                        }

                        let storage_arc = file_manager.table_data_mut(schema_name, table_name)?;
                        let mut storage_inner = storage_arc.write();

                        with_btree_storage!(wal_enabled, &mut *storage_inner, &self.shared.dirty_tracker, table_id as u32, root_page, |btree_mut: &mut crate::btree::BTree<_>| {
                            if !btree_mut.update(target_key, &record_data)? {
                                btree_mut.delete(target_key)?;
                                btree_mut.insert(target_key, &record_data)?;
                            }
                            Ok::<_, eyre::Report>(())
                        });
                        drop(storage_inner);

                        self.flush_wal_if_autocommit(file_manager, schema_name, table_name, table_id as u32)?;

                        {
                            let mut active_txn = self.active_txn.lock();
                            if let Some(ref mut txn) = *active_txn {
                                txn.add_write_entry_with_undo(
                                    WriteEntry {
                                        table_id: table_id as u32,
                                        key: target_key.clone(),
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

                        return Ok(ExecuteResult::Update {
                            rows_affected: 1,
                            returned: None,
                        });
                    }
                }

                // Key not found or didn't match - return 0 rows affected
                return Ok(ExecuteResult::Update {
                    rows_affected: 0,
                    returned: None,
                });
            }
        }

        // MULTIPASS: Fallback for complex queries
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
                let values = decoder.decode(key, user_data)?;
                let mut row_values: Vec<OwnedValue> = values.into_iter().map(OwnedValue::from).collect();

                let should_update = if let Some((_, ref target_val)) = pk_lookup_info {
                    if let Some(pk_idx) = columns.iter().position(|c| c.has_constraint(&Constraint::PrimaryKey)) {
                        &row_values[pk_idx] == target_val
                    } else {
                        false
                    }
                } else if let Some(ref pred) = predicate {
                    let values = owned_values_to_values(&row_values);
                    let values_slice = arena.alloc_slice_fill_iter(values.into_iter());
                    let exec_row = ExecutorRow::new(values_slice);
                    pred.evaluate(&exec_row)
                } else {
                    true
                };

                if should_update {
                    let old_value = value.to_vec();
                    let old_row_values = row_values.clone();

                    let mut old_toast_values: Vec<(usize, OwnedValue)> = Vec::new();

                    for (col_idx, val) in &precomputed_assignments {
                        if let OwnedValue::ToastPointer(_) = &row_values[*col_idx] {
                            old_toast_values.push((*col_idx, row_values[*col_idx].clone()));
                        }
                        row_values[*col_idx] = val.clone();
                    }

                    let validator = crate::constraints::ConstraintValidator::new(&table_def);
                    validator.validate_update(&row_values)?;

                    for (col_idx, col) in columns.iter().enumerate() {
                        for constraint in col.constraints() {
                            if let Constraint::Check(expr_str) = constraint {
                                let col_value = row_values.get(col_idx);
                                if !Self::evaluate_check_expression(expr_str, col.name(), col_value) {
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

                    rows_to_update.push((key.to_vec(), old_value, row_values, old_row_values, old_toast_values));
                }

                cursor.advance()?;
            }

            if pk_lookup_info.is_some() && rows_to_update.is_empty() {
                pk_lookup_info = None;
                continue;
            }

            break;
        }
        
        drop(btree);
        drop(storage);

        if !unique_col_indices.is_empty() {
            
            for (update_key, _old_value, updated_values, _old_row_values, _old_toast) in &rows_to_update {
                for &col_idx in &unique_col_indices {
                    let new_val = &updated_values[col_idx];
                    if new_val.is_null() {
                        continue;
                    }

                    let col_name = columns[col_idx].name();
                    let index_name = if columns[col_idx].has_constraint(&Constraint::PrimaryKey) {
                        format!("{}_pkey", col_name)
                    } else {
                        format!("{}_key", col_name)
                    };

                    if let Ok(index_storage_arc) =
                        file_manager.index_data_mut(schema_name, table_name, &index_name)
                    {
                        let mut index_storage = index_storage_arc.write();
                        let index_root_page = {
                            use crate::storage::IndexFileHeader;
                            let page0 = index_storage.page(0)?;
                            let header = IndexFileHeader::from_bytes(page0)?;
                            header.root_page()
                        };
                        let index_btree = BTree::new(&mut *index_storage, index_root_page)?;

                        let mut key_buf = Vec::new();
                        Self::encode_value_as_key(new_val, &mut key_buf);

                        // O(log n) index lookup
                        if let Some(handle) = index_btree.search(&key_buf)? {
                            let existing_row_key = index_btree.get_value(&handle)?;
                            if existing_row_key != update_key.as_slice() {
                                // Different row has this value - UNIQUE violation
                                bail!(
                                    "UNIQUE constraint violated on column '{}' in table '{}': value already exists",
                                    col_name,
                                    table_name
                                );
                            }
                        }
                    }
                }
            }
        }

        let rows_affected = rows_to_update.len();

        let returned_rows: Option<Vec<Row>> = update.returning.map(|returning_cols| {
            rows_to_update
                .iter()
                .map(|(_key, _old_value, updated_values, _old_row_values, _old_toast)| {
                    let row_values: Vec<OwnedValue> = returning_cols
                        .iter()
                        .flat_map(|col| match col {
                            crate::sql::ast::SelectColumn::AllColumns => updated_values.clone(),
                            crate::sql::ast::SelectColumn::TableAllColumns(_) => {
                                updated_values.clone()
                            }
                            crate::sql::ast::SelectColumn::Expr { expr, .. } => {
                                if let crate::sql::ast::Expr::Column(col_ref) = expr {
                                    columns
                                        .iter()
                                        .position(|c| c.name().eq_ignore_ascii_case(col_ref.column))
                                        .and_then(|idx| updated_values.get(idx).cloned())
                                        .map(|v| vec![v])
                                        .unwrap_or_default()
                                } else {
                                    vec![]
                                }
                            }
                        })
                        .collect();
                    Row::new(row_values)
                })
                .collect()
        });

        let mut processed_rows: Vec<(Vec<u8>, Vec<OwnedValue>)> = Vec::with_capacity(rows_to_update.len());

        let wal_enabled = self.shared.wal_enabled.load(std::sync::atomic::Ordering::Acquire);
        if wal_enabled {
            self.ensure_wal()?;
        }

        if has_toast {
            use crate::storage::toast::ToastPointer;
            for row_tuple in &mut rows_to_update {
                let (_key, _old_value, updated_values, _old_row_values, old_toast_values) = row_tuple;

                for (_col_idx, old_val) in old_toast_values.iter() {
                    if let OwnedValue::ToastPointer(ptr) = old_val {
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

                let pk_value = if let Some(pk_idx) = columns.iter().position(|c| c.has_constraint(&Constraint::PrimaryKey)) {
                    if let OwnedValue::Int(id) = &updated_values[pk_idx] {
                        *id as u64
                    } else {
                        0
                    }
                } else {
                    0
                };

                for (col_idx, val) in updated_values.iter_mut().enumerate() {
                    if columns[col_idx].data_type().is_toastable() {
                        let needs_toast = match val {
                            OwnedValue::Text(s) => crate::storage::toast::needs_toast(s.as_bytes()),
                            OwnedValue::Blob(b) => crate::storage::toast::needs_toast(b),
                            _ => false,
                        };
                        if needs_toast {
                            let data = match val {
                                OwnedValue::Text(s) => s.as_bytes().to_vec(),
                                OwnedValue::Blob(b) => b.clone(),
                                _ => continue,
                            };
                            let (pointer, _) = self.toast_value(
                                file_manager,
                                schema_name,
                                table_name,
                                pk_value,
                                col_idx as u16,
                                &data,
                                wal_enabled,
                                None,
                            )?;
                            *val = OwnedValue::ToastPointer(pointer);
                        }
                    }
                }
            }
        }

        let mut key_buf: SmallVec<[u8; 64]> = SmallVec::new();

        for (col_idx, index_name, _is_pk) in &unique_columns {
            if !modified_col_indices.contains(col_idx) {
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

                for (_row_key, _old_value, new_row_values, old_row_values, _old_toast) in &rows_to_update {
                    if let Some(old_value) = old_row_values.get(*col_idx) {
                        if !old_value.is_null() {
                            key_buf.clear();
                            Self::encode_value_as_key(old_value, &mut key_buf);
                            let _ = index_btree.delete(&key_buf);
                        }
                    }

                    if let Some(new_value) = new_row_values.get(*col_idx) {
                        if !new_value.is_null() {
                            key_buf.clear();
                            Self::encode_value_as_key(new_value, &mut key_buf);
                            if let Some(pk_idx) = columns.iter().position(|c| c.has_constraint(&Constraint::PrimaryKey)) {
                                if let Some(OwnedValue::Int(pk_val)) = new_row_values.get(pk_idx) {
                                    let row_id_bytes = (*pk_val as u64).to_be_bytes();
                                    let _ = index_btree.insert(&key_buf, &row_id_bytes);
                                }
                            }
                        }
                    }
                }
            }
        }

        for (index_name, col_indices) in &secondary_indexes {
            if col_indices.is_empty() {
                continue;
            }
            let any_modified = col_indices.iter().any(|idx| modified_col_indices.contains(idx));
            if !any_modified {
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

                for (_row_key, _old_value, new_row_values, old_row_values, _old_toast) in &rows_to_update {
                    let old_all_non_null = col_indices
                        .iter()
                        .all(|&idx| old_row_values.get(idx).is_some_and(|v| !v.is_null()));

                    if old_all_non_null {
                        key_buf.clear();
                        for &col_idx in col_indices {
                            if let Some(value) = old_row_values.get(col_idx) {
                                Self::encode_value_as_key(value, &mut key_buf);
                            }
                        }
                        let _ = index_btree.delete(&key_buf);
                    }

                    let new_all_non_null = col_indices
                        .iter()
                        .all(|&idx| new_row_values.get(idx).is_some_and(|v| !v.is_null()));

                    if new_all_non_null {
                        key_buf.clear();
                        for &col_idx in col_indices {
                            if let Some(value) = new_row_values.get(col_idx) {
                                Self::encode_value_as_key(value, &mut key_buf);
                            }
                        }
                        if let Some(pk_idx) = columns.iter().position(|c| c.has_constraint(&Constraint::PrimaryKey)) {
                            if let Some(OwnedValue::Int(pk_val)) = new_row_values.get(pk_idx) {
                                let row_id_bytes = (*pk_val as u64).to_be_bytes();
                                let _ = index_btree.insert(&key_buf, &row_id_bytes);
                            }
                        }
                    }
                }
            }
        }

        for (key, _old_value, updated_values, _old_row_values, _old_toast) in &rows_to_update {
            processed_rows.push((key.clone(), updated_values.clone()));
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
            for (key, updated_values) in &processed_rows {
                let user_record = OwnedValue::build_record_from_values(updated_values, &schema)?;
                let record_data = wrap_record_for_update(txn_id, &user_record, 0, 0, in_transaction);

                if !btree_mut.update(key, &record_data)? {
                    btree_mut.delete(key)?;
                    btree_mut.insert(key, &record_data)?;
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
                for (key, old_value, _updated_values, _old_row_values, _old_toast) in rows_to_update {
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

        Ok(ExecuteResult::Update {
            rows_affected,
            returned: returned_rows,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn execute_update_with_from(
        &self,
        update: &crate::sql::ast::UpdateStmt<'_>,
        arena: &Bump,
        schema_name: &str,
        table_name: &str,
        table_alias: Option<&str>,
        table_def: &crate::schema::table::TableDef,
        table_id: usize,
        columns: &[crate::schema::table::ColumnDef],
        schema: &crate::records::Schema,
        from_tables: Vec<(
            String,
            String,
            Option<&str>,
            Vec<crate::schema::table::ColumnDef>,
        )>,
    ) -> Result<ExecuteResult> {
        let mut combined_column_map: Vec<(String, usize)> = Vec::new();
        for (idx, col) in columns.iter().enumerate() {
            combined_column_map.push((col.name().to_string(), idx));
            combined_column_map.push((
                format!("{}.{}", table_name, col.name()),
                idx,
            ));
            if let Some(alias) = table_alias {
                combined_column_map.push((
                    format!("{}.{}", alias, col.name()),
                    idx,
                ));
            }
        }

        let mut current_col_offset = columns.len();
        let mut from_schemas: Vec<crate::records::Schema> = Vec::new();
        for (_, from_table_name, from_alias, from_columns) in &from_tables {
            for (idx, col) in from_columns.iter().enumerate() {
                combined_column_map.push((col.name().to_string(), current_col_offset + idx));
                combined_column_map.push((
                    format!("{}.{}", from_table_name, col.name()),
                    current_col_offset + idx,
                ));
                if let Some(alias) = from_alias {
                    combined_column_map.push((
                        format!("{}.{}", alias, col.name()),
                        current_col_offset + idx,
                    ));
                }
            }
            current_col_offset += from_columns.len();
            from_schemas.push(create_record_schema(from_columns));
        }

        let predicate = update
            .where_clause
            .map(|expr| CompiledPredicate::new(expr, combined_column_map.clone()));

        let assignment_indices: Vec<(usize, &crate::sql::ast::Expr<'_>)> = update
            .assignments
            .iter()
            .filter_map(|a| {
                columns
                    .iter()
                    .position(|c| c.name().eq_ignore_ascii_case(a.column.column))
                    .map(|idx| (idx, a.value))
            })
            .collect();

        let mut file_manager_guard = self.shared.file_manager.write();
        let file_manager = file_manager_guard.as_mut().unwrap();

        let mut all_from_rows: Vec<Vec<Vec<OwnedValue>>> = Vec::new();
        for (i, (from_schema_name, from_table_name, _, from_columns)) in from_tables.iter().enumerate() {
            let from_storage_arc = file_manager.table_data_mut(from_schema_name, from_table_name)?;
            let mut from_storage = from_storage_arc.write();
            let from_btree = BTree::new(&mut *from_storage, 1u32)?;
            let mut from_cursor = from_btree.cursor_first()?;

            let mut table_rows: Vec<Vec<OwnedValue>> = Vec::new();
            while from_cursor.valid() {
                let value = from_cursor.value()?;
                let user_data = get_user_data(value);
                let record = RecordView::new(user_data, &from_schemas[i])?;
                let row_values = OwnedValue::extract_row_from_record(&record, from_columns)?;
                table_rows.push(row_values);
                from_cursor.advance()?;
            }
            all_from_rows.push(table_rows);
        }

        let combined_from_rows = Self::cartesian_product(&all_from_rows);

        let storage_arc = file_manager.table_data_mut(schema_name, table_name)?;
        let mut storage = storage_arc.write();
        let root_page = 1u32;
        let btree = BTree::new(&mut *storage, root_page)?;
        let mut cursor = btree.cursor_first()?;

        let column_types: Vec<crate::records::types::DataType> =
            columns.iter().map(|c| c.data_type()).collect();
        let decoder = crate::sql::decoder::SimpleDecoder::new(column_types);

        let mut rows_to_update: Vec<(Vec<u8>, Vec<u8>, Vec<OwnedValue>)> = Vec::new();
        let mut updated_keys: HashSet<Vec<u8>> = HashSet::new();

        while cursor.valid() {
            let key = cursor.key()?;
            let value = cursor.value()?;

            let user_data = get_user_data(value);
            let values = decoder.decode(key, user_data)?;
            let target_row_values: Vec<OwnedValue> =
                values.into_iter().map(OwnedValue::from).collect();

            for from_row in &combined_from_rows {
                let mut combined_values: Vec<Value<'_>> = Vec::with_capacity(
                    target_row_values.len() + from_row.len(),
                );

                for val in &target_row_values {
                    combined_values.push(match val {
                        OwnedValue::Null => Value::Null,
                        OwnedValue::Bool(b) => Value::Int(if *b { 1 } else { 0 }),
                        OwnedValue::Int(i) => Value::Int(*i),
                        OwnedValue::Float(f) => Value::Float(*f),
                        OwnedValue::Text(s) => Value::Text(Cow::Borrowed(s.as_str())),
                        OwnedValue::Blob(b) => Value::Blob(Cow::Borrowed(b.as_slice())),
                        _ => Value::Null,
                    });
                }

                for val in from_row {
                    combined_values.push(match val {
                        OwnedValue::Null => Value::Null,
                        OwnedValue::Bool(b) => Value::Int(if *b { 1 } else { 0 }),
                        OwnedValue::Int(i) => Value::Int(*i),
                        OwnedValue::Float(f) => Value::Float(*f),
                        OwnedValue::Text(s) => Value::Text(Cow::Borrowed(s.as_str())),
                        OwnedValue::Blob(b) => Value::Blob(Cow::Borrowed(b.as_slice())),
                        _ => Value::Null,
                    });
                }

                let values_slice = arena.alloc_slice_fill_iter(combined_values.into_iter());
                let exec_row = ExecutorRow::new(values_slice);

                let should_update = if let Some(ref pred) = predicate {
                    pred.evaluate(&exec_row)
                } else {
                    true
                };

                if should_update && !updated_keys.contains(&key.to_vec()) {
                    let old_value = value.to_vec();
                    let mut row_values = target_row_values.clone();

                    for (col_idx, value_expr) in &assignment_indices {
                        let new_value = self.eval_expr_with_row(
                            value_expr,
                            &exec_row,
                            &combined_column_map,
                        )?;
                        row_values[*col_idx] = new_value;
                    }

                    let validator = crate::constraints::ConstraintValidator::new(table_def);
                    validator.validate_update(&row_values)?;

                    for (col_idx, col) in columns.iter().enumerate() {
                        for constraint in col.constraints() {
                            if let Constraint::Check(expr_str) = constraint {
                                let col_value = row_values.get(col_idx);
                                if !Self::evaluate_check_expression(
                                    expr_str,
                                    col.name(),
                                    col_value,
                                ) {
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

                    updated_keys.insert(key.to_vec());
                    rows_to_update.push((key.to_vec(), old_value, row_values));
                }
            }

            cursor.advance()?;
        }

        drop(cursor);
        drop(btree);
        drop(storage);

        let unique_col_indices: Vec<usize> = columns
            .iter()
            .enumerate()
            .filter(|(_, col)| {
                col.has_constraint(&Constraint::Unique)
                    || col.has_constraint(&Constraint::PrimaryKey)
            })
            .map(|(idx, _)| idx)
            .collect();

        if !unique_col_indices.is_empty() {
            let storage_for_check_arc = file_manager.table_data_mut(schema_name, table_name)?;
            let mut storage_for_check = storage_for_check_arc.write();
            let btree_for_check = BTree::new(&mut *storage_for_check, root_page)?;
            let mut check_cursor = btree_for_check.cursor_first()?;

            for (update_key, _old_value, updated_values) in &rows_to_update {
                while check_cursor.valid() {
                    let existing_key = check_cursor.key()?;

                    if existing_key != update_key.as_slice() {
                        let existing_value = check_cursor.value()?;
                        let existing_user_data = get_user_data(existing_value);
                        let existing_record = RecordView::new(existing_user_data, schema)?;
                        let existing_values =
                            OwnedValue::extract_row_from_record(&existing_record, columns)?;

                        for &col_idx in &unique_col_indices {
                            let new_val = updated_values.get(col_idx);
                            let existing_val = existing_values.get(col_idx);

                            if let (Some(new_v), Some(existing_v)) = (new_val, existing_val) {
                                if !new_v.is_null() && !existing_v.is_null() && new_v == existing_v
                                {
                                    let col_name = &columns[col_idx].name();
                                    bail!(
                                        "UNIQUE constraint violated on column '{}' in table '{}': value already exists",
                                        col_name,
                                        table_name
                                    );
                                }
                            }
                        }
                    }
                    check_cursor.advance()?;
                }
                check_cursor = btree_for_check.cursor_first()?;
            }
        }

        let rows_affected = rows_to_update.len();

        let returned_rows: Option<Vec<Row>> = update.returning.map(|returning_cols| {
            rows_to_update
                .iter()
                .map(|(_key, _old_value, updated_values)| {
                    let row_values: Vec<OwnedValue> = returning_cols
                        .iter()
                        .flat_map(|col| match col {
                            crate::sql::ast::SelectColumn::AllColumns => updated_values.clone(),
                            crate::sql::ast::SelectColumn::TableAllColumns(_) => {
                                updated_values.clone()
                            }
                            crate::sql::ast::SelectColumn::Expr { expr, .. } => {
                                if let crate::sql::ast::Expr::Column(col_ref) = expr {
                                    columns
                                        .iter()
                                        .position(|c| c.name().eq_ignore_ascii_case(col_ref.column))
                                        .and_then(|idx| updated_values.get(idx).cloned())
                                        .map(|v| vec![v])
                                        .unwrap_or_default()
                                } else {
                                    vec![]
                                }
                            }
                        })
                        .collect();
                    Row::new(row_values)
                })
                .collect()
        });

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
            for (key, _old_value, updated_values) in &rows_to_update {
                let user_record = OwnedValue::build_record_from_values(updated_values, schema)?;
                let record_data = wrap_record_for_update(txn_id, &user_record, 0, 0, in_transaction);

                if !btree_mut.update(key, &record_data)? {
                    btree_mut.delete(key)?;
                    btree_mut.insert(key, &record_data)?;
                }
            }
            Ok::<_, eyre::Report>(())
        });

        self.flush_wal_if_autocommit(file_manager, schema_name, table_name, table_id as u32)?;

        drop(file_manager_guard);
        {
            let mut active_txn = self.active_txn.lock();
            if let Some(ref mut txn) = *active_txn {
                for (key, old_value, _updated_values) in rows_to_update {
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

        Ok(ExecuteResult::Update {
            rows_affected,
            returned: returned_rows,
        })
    }

    pub(crate) fn cartesian_product(tables: &[Vec<Vec<OwnedValue>>]) -> Vec<Vec<OwnedValue>> {
        if tables.is_empty() {
            return vec![vec![]];
        }

        let mut result: Vec<Vec<OwnedValue>> = vec![vec![]];

        for table_rows in tables {
            let mut new_result: Vec<Vec<OwnedValue>> = Vec::new();
            for existing in &result {
                for row in table_rows {
                    let mut combined = existing.clone();
                    combined.extend(row.clone());
                    new_result.push(combined);
                }
            }
            result = new_result;
        }

        result
    }

    pub(crate) fn eval_expr_with_row(
        &self,
        expr: &crate::sql::ast::Expr<'_>,
        row: &ExecutorRow<'_>,
        column_map: &[(String, usize)],
    ) -> Result<OwnedValue> {
        use crate::sql::ast::{BinaryOperator, Expr, UnaryOperator};

        match expr {
            Expr::Literal(_) => Self::eval_literal(expr),
            Expr::Column(col_ref) => {
                let col_name = if let Some(table) = col_ref.table {
                    format!("{}.{}", table, col_ref.column)
                } else {
                    col_ref.column.to_string()
                };

                let col_idx = column_map
                    .iter()
                    .find(|(name, _)| name.eq_ignore_ascii_case(&col_name))
                    .map(|(_, idx)| *idx)
                    .or_else(|| {
                        column_map
                            .iter()
                            .find(|(name, _)| name.eq_ignore_ascii_case(col_ref.column))
                            .map(|(_, idx)| *idx)
                    });

                if let Some(idx) = col_idx {
                    if let Some(val) = row.get(idx) {
                        Ok(OwnedValue::from(val))
                    } else {
                        Ok(OwnedValue::Null)
                    }
                } else {
                    bail!(
                        "column '{}' not found in UPDATE...FROM context",
                        col_name
                    )
                }
            }
            Expr::BinaryOp { left, op, right } => {
                let left_val = self.eval_expr_with_row(left, row, column_map)?;
                let right_val = self.eval_expr_with_row(right, row, column_map)?;

                match op {
                    BinaryOperator::Plus => match (&left_val, &right_val) {
                        (OwnedValue::Int(a), OwnedValue::Int(b)) => Ok(OwnedValue::Int(a + b)),
                        (OwnedValue::Float(a), OwnedValue::Float(b)) => {
                            Ok(OwnedValue::Float(a + b))
                        }
                        (OwnedValue::Int(a), OwnedValue::Float(b)) => {
                            Ok(OwnedValue::Float(*a as f64 + b))
                        }
                        (OwnedValue::Float(a), OwnedValue::Int(b)) => {
                            Ok(OwnedValue::Float(a + *b as f64))
                        }
                        _ => bail!("unsupported types for addition"),
                    },
                    BinaryOperator::Minus => match (&left_val, &right_val) {
                        (OwnedValue::Int(a), OwnedValue::Int(b)) => Ok(OwnedValue::Int(a - b)),
                        (OwnedValue::Float(a), OwnedValue::Float(b)) => {
                            Ok(OwnedValue::Float(a - b))
                        }
                        (OwnedValue::Int(a), OwnedValue::Float(b)) => {
                            Ok(OwnedValue::Float(*a as f64 - b))
                        }
                        (OwnedValue::Float(a), OwnedValue::Int(b)) => {
                            Ok(OwnedValue::Float(a - *b as f64))
                        }
                        _ => bail!("unsupported types for subtraction"),
                    },
                    BinaryOperator::Multiply => match (&left_val, &right_val) {
                        (OwnedValue::Int(a), OwnedValue::Int(b)) => Ok(OwnedValue::Int(a * b)),
                        (OwnedValue::Float(a), OwnedValue::Float(b)) => {
                            Ok(OwnedValue::Float(a * b))
                        }
                        (OwnedValue::Int(a), OwnedValue::Float(b)) => {
                            Ok(OwnedValue::Float(*a as f64 * b))
                        }
                        (OwnedValue::Float(a), OwnedValue::Int(b)) => {
                            Ok(OwnedValue::Float(a * *b as f64))
                        }
                        _ => bail!("unsupported types for multiplication"),
                    },
                    BinaryOperator::Divide => match (&left_val, &right_val) {
                        (OwnedValue::Int(a), OwnedValue::Int(b)) if *b != 0 => {
                            Ok(OwnedValue::Int(a / b))
                        }
                        (OwnedValue::Float(a), OwnedValue::Float(b)) if *b != 0.0 => {
                            Ok(OwnedValue::Float(a / b))
                        }
                        (OwnedValue::Int(a), OwnedValue::Float(b)) if *b != 0.0 => {
                            Ok(OwnedValue::Float(*a as f64 / b))
                        }
                        (OwnedValue::Float(a), OwnedValue::Int(b)) if *b != 0 => {
                            Ok(OwnedValue::Float(a / *b as f64))
                        }
                        _ => bail!("division by zero or unsupported types"),
                    },
                    BinaryOperator::Concat => match (&left_val, &right_val) {
                        (OwnedValue::Text(a), OwnedValue::Text(b)) => {
                            Ok(OwnedValue::Text(format!("{}{}", a, b)))
                        }
                        _ => bail!("unsupported types for concatenation"),
                    },
                    _ => bail!("unsupported binary operator in UPDATE...FROM SET expression"),
                }
            }
            Expr::UnaryOp { op, expr: inner } => {
                let inner_val = self.eval_expr_with_row(inner, row, column_map)?;
                match (op, inner_val) {
                    (UnaryOperator::Minus, OwnedValue::Int(i)) => Ok(OwnedValue::Int(-i)),
                    (UnaryOperator::Minus, OwnedValue::Float(f)) => Ok(OwnedValue::Float(-f)),
                    (UnaryOperator::Plus, val) => Ok(val),
                    (UnaryOperator::Not, OwnedValue::Bool(b)) => Ok(OwnedValue::Bool(!b)),
                    _ => bail!("unsupported unary operation"),
                }
            }
            _ => Self::eval_literal(expr),
        }
    }

    pub(crate) fn execute_update_cached(
        &self,
        cached: &crate::database::prepared::CachedUpdatePlan,
        params: &[OwnedValue],
    ) -> Result<ExecuteResult> {
        self.ensure_file_manager()?;

        let wal_enabled = self.shared.wal_enabled.load(std::sync::atomic::Ordering::Acquire);
        if wal_enabled {
            self.ensure_wal()?;
        }

        if cached.is_simple_pk_update && cached.assignment_indices.len() + 1 == params.len() {
            return self.execute_update_param_only(cached, params, wal_enabled);
        }

        let arena = bumpalo::Bump::new();
        let mut parser = crate::sql::parser::Parser::new(&cached.original_sql, &arena);

        let stmt = parser.parse_statement()
            .wrap_err("failed to re-parse cached UPDATE statement")?;

        if let crate::sql::ast::Statement::Update(update) = stmt {
            self.execute_update(&update, params, &arena)
        } else {
            bail!("cached plan produced non-UPDATE statement")
        }
    }

    fn execute_update_param_only(
        &self,
        cached: &crate::database::prepared::CachedUpdatePlan,
        params: &[OwnedValue],
        wal_enabled: bool,
    ) -> Result<ExecuteResult> {
        let storage_arc = {
            let storage_weak = cached.storage.borrow();
            storage_weak.as_ref()
                .and_then(|weak| weak.upgrade())
                .ok_or_else(|| eyre::eyre!("cached plan storage no longer valid"))?
        };

        let root_page = cached.root_page.get();
        if root_page == 0 {
            let storage = storage_arc.write();
            let new_root = {
                use crate::storage::TableFileHeader;
                let page0 = storage.page(0)?;
                let header = TableFileHeader::from_bytes(page0)?;
                header.root_page()
            };
            drop(storage);
            cached.root_page.set(new_root);
        }
        let root_page = cached.root_page.get();

        let where_param_value = params.last().ok_or_else(|| eyre::eyre!("missing WHERE parameter"))?;

        let pk_col_idx = cached.pk_column_index.ok_or_else(|| eyre::eyre!("PK column index not cached"))?;

        self.ensure_catalog()?;
        let catalog_guard = self.shared.catalog.read();
        let catalog = catalog_guard.as_ref().unwrap();
        let table_def = catalog.resolve_table(&cached.table_name)?;
        let pk_col_name = table_def.columns()[pk_col_idx].name();
        let pk_index_name = format!("{}_pkey", pk_col_name);
        drop(catalog_guard);

        let mut file_manager_guard = self.shared.file_manager.write();
        let file_manager = file_manager_guard.as_mut().unwrap();

        let target_key = if file_manager.index_exists(&cached.schema_name, &cached.table_name, &pk_index_name) {
            if let Ok(index_storage_arc) = file_manager.index_data_mut(&cached.schema_name, &cached.table_name, &pk_index_name) {
                let mut index_storage = index_storage_arc.write();
                let index_root_page = {
                    use crate::storage::IndexFileHeader;
                    let page0 = index_storage.page(0)?;
                    let header = IndexFileHeader::from_bytes(page0)?;
                    header.root_page()
                };
                let index_btree = BTree::new(&mut *index_storage, index_root_page)?;
                let mut index_key = Vec::new();
                Self::encode_value_as_key(where_param_value, &mut index_key);
                if let Some(handle) = index_btree.search(&index_key)? {
                    index_btree.get_value(&handle)?.to_vec()
                } else {
                    return Ok(ExecuteResult::Update { rows_affected: 0, returned: None });
                }
            } else {
                bail!("failed to get index storage")
            }
        } else {
            bail!("PK index not found for fast path")
        };

        drop(file_manager_guard);

        let mut storage = storage_arc.write();
        let decoder = crate::sql::decoder::SimpleDecoder::new(cached.column_types.clone());
        let btree = BTree::new(&mut *storage, root_page)?;

        let cursor = btree.cursor_seek(&target_key)?;

        if !cursor.valid() || cursor.key()? != target_key.as_slice() {
            return Ok(ExecuteResult::Update { rows_affected: 0, returned: None });
        }

        let key = cursor.key()?;
        let value = cursor.value()?;
        let values = decoder.decode(key, value)?;
        let mut row_values: Vec<OwnedValue> = values.iter().map(OwnedValue::from).collect();

        for (i, (col_idx, _)) in cached.assignment_indices.iter().enumerate() {
            row_values[*col_idx] = params[i].clone();
        }

        drop(cursor);
        drop(btree);
        drop(storage);

        let mut storage = storage_arc.write();

        with_btree_storage!(wal_enabled, &mut *storage, &self.shared.dirty_tracker, cached.table_id as u32, root_page, |btree_mut: &mut crate::btree::BTree<_>| {
            let record_data = OwnedValue::build_record_from_values(&row_values, &cached.record_schema)?;

            if !btree_mut.update(&target_key, &record_data)? {
                btree_mut.delete(&target_key)?;
                btree_mut.insert(&target_key, &record_data)?;
            }
            Ok::<_, eyre::Report>(())
        });

        Ok(ExecuteResult::Update {
            rows_affected: 1,
            returned: None,
        })
    }

    pub(crate) fn extract_tables_from_clause<'a>(
        from_clause: crate::sql::ast::FromClause<'a>,
        catalog: &crate::schema::Catalog,
        tables: &mut Vec<(
            String,
            String,
            Option<&'a str>,
            Vec<crate::schema::table::ColumnDef>,
        )>,
    ) -> Result<()> {
        use crate::sql::ast::FromClause;

        match from_clause {
            FromClause::Table(table_ref) => {
                let schema = table_ref.schema.unwrap_or("root");
                let table_name = table_ref.name;
                let alias = table_ref.alias;
                let table_def = catalog.resolve_table(table_name)?;
                let columns = table_def.columns().to_vec();
                tables.push((schema.to_string(), table_name.to_string(), alias, columns));
            }
            FromClause::Join(join_clause) => {
                Self::extract_tables_from_clause(*join_clause.left, catalog, tables)?;
                Self::extract_tables_from_clause(*join_clause.right, catalog, tables)?;
            }
            FromClause::Subquery { .. } => {
                bail!("UPDATE...FROM does not support subqueries in FROM clause")
            }
            FromClause::Lateral { .. } => {
                bail!("UPDATE...FROM does not support LATERAL in FROM clause")
            }
        }
        Ok(())
    }
}
