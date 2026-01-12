//! # Set Operations Execution
//!
//! This module handles execution of SQL set operations: UNION, INTERSECT, and EXCEPT.
//! These operations combine results from two or more SELECT queries.
//!
//! ## Set Operation Types
//!
//! - **UNION**: Combines results from multiple queries, removing duplicates by default
//! - **UNION ALL**: Combines results without removing duplicates
//! - **INTERSECT**: Returns only rows that appear in all queries
//! - **EXCEPT**: Returns rows from first query that don't appear in subsequent queries
//!
//! ## Implementation Details
//!
//! Set operations are implemented using hash-based deduplication for efficiency.
//! Each row is converted to a hash key for fast comparison. For UNION ALL,
//! no deduplication is performed.
//!
//! The recursive `execute_branch_for_set_op` function handles nested set operations
//! by traversing the physical plan tree.

use crate::database::row::Row;
use crate::database::Database;
use crate::sql::builder::ExecutorBuilder;
use crate::sql::context::ExecutionContext;
use crate::sql::executor::{Executor, StreamingBTreeSource};
use crate::sql::planner::{PhysicalOperator, SetOpKind};
use crate::storage::{FileManager, DEFAULT_SCHEMA};
use crate::types::OwnedValue;
use bumpalo::Bump;
use eyre::{Result, WrapErr};

impl Database {
    pub(crate) fn execute_physical_plan_recursive<'a>(
        &self,
        op: &'a PhysicalOperator<'a>,
        _arena: &'a Bump,
    ) -> Result<Vec<Row>> {
        let catalog_guard = self.shared.catalog.read();
        let catalog = catalog_guard.as_ref().unwrap();

        let mut file_manager_guard = self.shared.file_manager.write();
        let file_manager = file_manager_guard.as_mut().unwrap();

        execute_branch_for_set_op(&self.shared.path, op, catalog, file_manager)
    }
}

fn find_table_scan_for_set<'a>(
    op: &'a PhysicalOperator<'a>,
) -> Option<&'a crate::sql::planner::PhysicalTableScan<'a>> {
    match op {
        PhysicalOperator::TableScan(scan) => Some(scan),
        PhysicalOperator::FilterExec(f) => find_table_scan_for_set(f.input),
        PhysicalOperator::ProjectExec(p) => find_table_scan_for_set(p.input),
        PhysicalOperator::LimitExec(l) => find_table_scan_for_set(l.input),
        PhysicalOperator::SortExec(s) => find_table_scan_for_set(s.input),
        PhysicalOperator::SubqueryExec(sub) => find_table_scan_for_set(sub.child_plan),
        PhysicalOperator::SetOpExec(set) => find_table_scan_for_set(set.left),
        PhysicalOperator::WindowExec(w) => find_table_scan_for_set(w.input),
        _ => None,
    }
}

fn row_to_key(row: &Row) -> Vec<u64> {
    use std::hash::{Hash, Hasher};
    row.values
        .iter()
        .map(|v| {
            let mut hasher = std::collections::hash_map::DefaultHasher::new();
            format!("{:?}", v).hash(&mut hasher);
            hasher.finish()
        })
        .collect()
}

fn execute_branch_for_set_op<'a>(
    _db_path: &std::path::Path,
    op: &'a PhysicalOperator<'a>,
    catalog: &crate::schema::catalog::Catalog,
    file_manager: &mut FileManager,
) -> Result<Vec<Row>> {
    match op {
        PhysicalOperator::SortExec(sort) => {
            let mut rows = execute_branch_for_set_op(_db_path, sort.input, catalog, file_manager)?;
            if !sort.order_by.is_empty() {
                let first_key = &sort.order_by[0];
                let ascending = first_key.ascending;
                rows.sort_by(|a, b| {
                    let a_val = a.values.first();
                    let b_val = b.values.first();
                    let cmp = match (a_val, b_val) {
                        (Some(OwnedValue::Int(a_i)), Some(OwnedValue::Int(b_i))) => a_i.cmp(b_i),
                        (Some(OwnedValue::Text(a_t)), Some(OwnedValue::Text(b_t))) => a_t.cmp(b_t),
                        (Some(OwnedValue::Float(a_f)), Some(OwnedValue::Float(b_f))) => {
                            a_f.partial_cmp(b_f).unwrap_or(std::cmp::Ordering::Equal)
                        }
                        _ => std::cmp::Ordering::Equal,
                    };
                    if ascending {
                        cmp
                    } else {
                        cmp.reverse()
                    }
                });
            }
            Ok(rows)
        }
        PhysicalOperator::LimitExec(limit) => {
            let rows = execute_branch_for_set_op(_db_path, limit.input, catalog, file_manager)?;
            let offset = limit.offset.unwrap_or(0) as usize;
            let count = limit.limit.unwrap_or(usize::MAX as u64) as usize;
            Ok(rows.into_iter().skip(offset).take(count).collect())
        }
        PhysicalOperator::SetOpExec(set_op) => {
            let left_rows =
                execute_branch_for_set_op(_db_path, set_op.left, catalog, file_manager)?;
            let right_rows =
                execute_branch_for_set_op(_db_path, set_op.right, catalog, file_manager)?;

            let result = match set_op.kind {
                SetOpKind::Union => {
                    if set_op.all {
                        let mut all = left_rows;
                        all.extend(right_rows);
                        all
                    } else {
                        let mut seen: std::collections::HashSet<Vec<u64>> =
                            std::collections::HashSet::new();
                        let mut result = Vec::new();
                        for row in left_rows.into_iter().chain(right_rows.into_iter()) {
                            let key = row_to_key(&row);
                            if seen.insert(key) {
                                result.push(row);
                            }
                        }
                        result
                    }
                }
                SetOpKind::Intersect => {
                    let right_keys: std::collections::HashSet<Vec<u64>> =
                        right_rows.iter().map(row_to_key).collect();
                    if set_op.all {
                        left_rows
                            .into_iter()
                            .filter(|row| right_keys.contains(&row_to_key(row)))
                            .collect()
                    } else {
                        let mut seen: std::collections::HashSet<Vec<u64>> =
                            std::collections::HashSet::new();
                        left_rows
                            .into_iter()
                            .filter(|row| {
                                let key = row_to_key(row);
                                right_keys.contains(&key) && seen.insert(key)
                            })
                            .collect()
                    }
                }
                SetOpKind::Except => {
                    let right_keys: std::collections::HashSet<Vec<u64>> =
                        right_rows.iter().map(row_to_key).collect();
                    if set_op.all {
                        left_rows
                            .into_iter()
                            .filter(|row| !right_keys.contains(&row_to_key(row)))
                            .collect()
                    } else {
                        let mut seen: std::collections::HashSet<Vec<u64>> =
                            std::collections::HashSet::new();
                        left_rows
                            .into_iter()
                            .filter(|row| {
                                let key = row_to_key(row);
                                !right_keys.contains(&key) && seen.insert(key)
                            })
                            .collect()
                    }
                }
            };
            Ok(result)
        }
        _ => {
            let scan = find_table_scan_for_set(op)
                .ok_or_else(|| eyre::eyre!("set operation branch must have a table scan"))?;

            let schema_name = scan.schema.unwrap_or(DEFAULT_SCHEMA);
            let table_name = scan.table;

            let table_def = catalog
                .resolve_table_in_schema(scan.schema, table_name)
                .wrap_err_with(|| format!("table '{}' not found", table_name))?;

            let column_types: Vec<_> = table_def.columns().iter().map(|c| c.data_type()).collect();

            let storage_arc = file_manager
                .table_data(schema_name, table_name)
                .wrap_err_with(|| {
                    format!(
                        "failed to open table storage for {}.{}",
                        schema_name, table_name
                    )
                })?;
            let storage = storage_arc.read();

            let root_page = {
                use crate::storage::TableFileHeader;
                let page = storage.page(0)?;
                TableFileHeader::from_bytes(page)?.root_page()
            };
            let source = StreamingBTreeSource::from_btree_scan_with_projections(
                &storage,
                root_page,
                column_types.clone(),
                None,
            )
            .wrap_err("failed to create table scan")?;

            let branch_arena = Bump::new();
            let output_schema = crate::sql::planner::OutputSchema {
                columns: branch_arena.alloc_slice_fill_iter(table_def.columns().iter().map(
                    |col| crate::sql::planner::OutputColumn {
                        name: branch_arena.alloc_str(col.name()),
                        data_type: col.data_type(),
                        nullable: col.is_nullable(),
                    },
                )),
            };

            let branch_plan = crate::sql::planner::PhysicalPlan {
                root: op,
                output_schema,
            };

            let column_map: Vec<(String, usize)> = table_def
                .columns()
                .iter()
                .enumerate()
                .map(|(idx, col)| (col.name().to_lowercase(), idx))
                .collect();

            let ctx = ExecutionContext::new(&branch_arena);
            let builder = ExecutorBuilder::new(&ctx);
            let mut executor = builder
                .build_with_source_and_column_map(&branch_plan, source, &column_map)
                .wrap_err("failed to build executor")?;

            let mut rows = Vec::new();
            executor.open()?;
            while let Some(row) = executor.next()? {
                let owned: Vec<OwnedValue> = row.values.iter().map(OwnedValue::from).collect();
                rows.push(Row::new(owned));
            }
            executor.close()?;
            Ok(rows)
        }
    }
}
