//! # DML Statement Planning
//!
//! This module handles the logical planning of Data Manipulation Language (DML)
//! statements: INSERT, UPDATE, and DELETE.
//!
//! ## INSERT Planning
//!
//! - Validates target table exists
//! - Validates column names if specified
//! - Checks value count matches column count
//! - Supports VALUES, SELECT, and DEFAULT sources
//!
//! ## UPDATE Planning
//!
//! - Validates target table exists
//! - Validates assignment column names
//! - Validates expression columns in assignments and WHERE clause
//!
//! ## DELETE Planning
//!
//! - Validates target table exists
//! - Handles optional WHERE clause
//!
//! ## Design
//!
//! DML operators are converted to logical operators but cannot be directly
//! converted to physical plans - they are handled separately by the execution layer.

use super::logical::{
    InsertSource, LogicalDelete, LogicalInsert, LogicalOperator, LogicalPlan,
    LogicalUpdate, UpdateAssignment,
};
use super::schema::TableSource;
use super::Planner;
use eyre::{bail, Result};

impl<'a> Planner<'a> {
    pub(crate) fn plan_insert(&self, insert: &crate::sql::ast::InsertStmt<'a>) -> Result<LogicalPlan<'a>> {
        self.validate_table_exists(insert.table.schema, insert.table.name)?;

        let table_name = if let Some(schema) = insert.table.schema {
            self.arena
                .alloc_str(&format!("{}.{}", schema, insert.table.name))
        } else {
            insert.table.name
        };
        let table_def = self.catalog.resolve_table(table_name)?;

        let expected_column_count = if let Some(columns) = insert.columns {
            for col_name in columns.iter() {
                let col_exists = table_def
                    .columns()
                    .iter()
                    .any(|c| c.name().eq_ignore_ascii_case(col_name));
                if !col_exists {
                    bail!(
                        "column '{}' not found in table '{}'",
                        col_name,
                        insert.table.name
                    );
                }
            }
            columns.len()
        } else {
            table_def.columns().len()
        };

        if let crate::sql::ast::InsertSource::Values(rows) = insert.source {
            for (row_idx, row) in rows.iter().enumerate() {
                if row.len() != expected_column_count {
                    bail!(
                        "INSERT has {} values but {} columns were expected (row {})",
                        row.len(),
                        expected_column_count,
                        row_idx + 1
                    );
                }
            }
        }

        let source = match insert.source {
            crate::sql::ast::InsertSource::Values(rows) => InsertSource::Values(rows),
            crate::sql::ast::InsertSource::Select(select) => {
                let select_plan = self.plan_select(select)?;
                InsertSource::Select(select_plan.root)
            }
            crate::sql::ast::InsertSource::Default => InsertSource::Default,
        };

        let insert_op = self.arena.alloc(LogicalOperator::Insert(LogicalInsert {
            schema: insert.table.schema,
            table: insert.table.name,
            columns: insert.columns,
            source,
        }));

        Ok(LogicalPlan { root: insert_op })
    }

    pub(crate) fn plan_update(&self, update: &crate::sql::ast::UpdateStmt<'a>) -> Result<LogicalPlan<'a>> {
        self.validate_table_exists(update.table.schema, update.table.name)?;

        let table_name = if let Some(schema) = update.table.schema {
            self.arena
                .alloc_str(&format!("{}.{}", schema, update.table.name))
        } else {
            update.table.name
        };
        let table_def = self.catalog.resolve_table(table_name)?;

        let tables_in_scope: Vec<TableSource<'a>> = vec![TableSource::Table {
            schema: update.table.schema,
            name: update.table.name,
            alias: update.table.alias,
            def: table_def,
        }];

        for assign in update.assignments {
            let col_exists = table_def
                .columns()
                .iter()
                .any(|c| c.name().eq_ignore_ascii_case(assign.column.column));
            if !col_exists {
                bail!(
                    "column '{}' not found in table '{}'",
                    assign.column.column,
                    update.table.name
                );
            }

            self.validate_expr_columns(assign.value, &tables_in_scope)?;
        }

        if let Some(predicate) = update.where_clause {
            self.validate_expr_columns(predicate, &tables_in_scope)?;
        }

        let mut assignments = bumpalo::collections::Vec::new_in(self.arena);
        for assign in update.assignments {
            assignments.push(UpdateAssignment {
                column: assign.column.column,
                value: assign.value,
            });
        }

        let update_op = self.arena.alloc(LogicalOperator::Update(LogicalUpdate {
            schema: update.table.schema,
            table: update.table.name,
            assignments: assignments.into_bump_slice(),
            filter: update.where_clause,
        }));

        Ok(LogicalPlan { root: update_op })
    }

    pub(crate) fn plan_delete(&self, delete: &crate::sql::ast::DeleteStmt<'a>) -> Result<LogicalPlan<'a>> {
        self.validate_table_exists(delete.table.schema, delete.table.name)?;

        let delete_op = self.arena.alloc(LogicalOperator::Delete(LogicalDelete {
            schema: delete.table.schema,
            table: delete.table.name,
            filter: delete.where_clause,
        }));

        Ok(LogicalPlan { root: delete_op })
    }
}
