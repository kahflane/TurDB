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
use crate::database::macros::with_btree_storage;
use crate::database::row::Row;
use crate::database::{Database, ExecuteResult};
use crate::mvcc::WriteEntry;
use crate::records::RecordView;
use crate::schema::table::Constraint;
use crate::sql::predicate::CompiledPredicate;
use crate::storage::TableFileHeader;
use crate::types::{create_column_map, create_record_schema, owned_values_to_values, OwnedValue};
use bumpalo::Bump;
use eyre::{bail, Result};

impl Database {
    pub(crate) fn execute_delete(
        &self,
        delete: &crate::sql::ast::DeleteStmt<'_>,
        arena: &Bump,
    ) -> Result<ExecuteResult> {
        self.ensure_catalog()?;
        self.ensure_file_manager()?;

        let catalog_guard = self.catalog.read();
        let catalog = catalog_guard.as_ref().unwrap();

        let schema_name = delete.table.schema.unwrap_or("root");
        let table_name = delete.table.name;

        let table_def = catalog.resolve_table(table_name)?;
        let table_id = table_def.id();
        let columns = table_def.columns().to_vec();
        let has_toast = table_def.has_toast();

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

        let mut file_manager_guard = self.file_manager.write();
        let file_manager = file_manager_guard.as_mut().unwrap();
        let storage = file_manager.table_data_mut(schema_name, table_name)?;

        let root_page = 1u32;
        let btree = BTree::new(storage, root_page)?;
        let mut cursor = btree.cursor_first()?;

        let mut rows_to_delete: Vec<(Vec<u8>, Vec<u8>, Vec<OwnedValue>)> = Vec::new();
        let mut values_to_check: Vec<(usize, OwnedValue)> = Vec::new();

        while cursor.valid() {
            let key = cursor.key()?;
            let value = cursor.value()?;

            let record = RecordView::new(value, &schema)?;
            let row_values = OwnedValue::extract_row_from_record(&record, &columns)?;

            let should_delete = if let Some(ref pred) = predicate {
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

        if !values_to_check.is_empty() {
            for (child_schema, child_name, child_columns, fk_col_idx) in &child_table_schemas {
                let child_storage = file_manager.table_data_mut(child_schema, child_name)?;
                let child_btree = BTree::new(child_storage, root_page)?;
                let mut child_cursor = child_btree.cursor_first()?;
                let child_record_schema = create_record_schema(child_columns);

                while child_cursor.valid() {
                    let child_value = child_cursor.value()?;
                    let child_record = RecordView::new(child_value, &child_record_schema)?;
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

        let wal_enabled = self.wal_enabled.load(std::sync::atomic::Ordering::Acquire);
        if wal_enabled {
            self.ensure_wal()?;
        }

        let storage = file_manager.table_data_mut(schema_name, table_name)?;

        with_btree_storage!(wal_enabled, storage, &self.dirty_tracker, table_id as u32, root_page, |btree_mut: &mut crate::btree::BTree<_>| {
            for (key, _old_value, _row_values) in &rows_to_delete {
                btree_mut.delete(key)?;
            }
            Ok::<_, eyre::Report>(())
        });

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
            let mut file_manager_guard = self.file_manager.write();
            let file_manager = file_manager_guard.as_mut().unwrap();
            let storage = file_manager.table_data_mut(schema_name, table_name)?;
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
