//! # DDL Operations Module
//!
//! This module implements Data Definition Language (DDL) operations for TurDB.
//! DDL statements modify the database schema rather than data.
//!
//! ## Supported Operations
//!
//! ### CREATE Operations
//! - `CREATE TABLE` - Creates a new table with columns, constraints, and indexes
//! - `CREATE SCHEMA` - Creates a new schema namespace
//! - `CREATE INDEX` - Creates a B-tree index on table columns or expressions
//!
//! ### DROP Operations
//! - `DROP TABLE` - Removes a table and all its data
//! - `DROP SCHEMA` - Removes a schema namespace
//! - `DROP INDEX` - Removes an index
//!
//! ### ALTER Operations
//! - `ALTER TABLE RENAME` - Renames a table
//! - `ALTER TABLE RENAME COLUMN` - Renames a column
//! - `ALTER TABLE ADD COLUMN` - Adds a new column
//! - `ALTER TABLE DROP COLUMN` - Removes a column (with data migration)
//!
//! ### TRUNCATE
//! - `TRUNCATE TABLE` - Removes all rows from a table without dropping it
//!
//! ## Architecture
//!
//! DDL operations typically involve:
//! 1. Validating the operation against the catalog
//! 2. Updating the catalog metadata
//! 3. Creating/modifying/removing physical storage files
//! 4. Persisting catalog changes to disk
//!
//! ## Transaction Behavior
//!
//! DDL operations are auto-committed and cannot be rolled back within a transaction.
//! This follows PostgreSQL semantics where DDL statements implicitly commit.
//!
//! ## Index Population
//!
//! When creating an index on an existing table with data, the index is populated
//! by scanning the table and inserting index entries for each row. Expression
//! indexes are not auto-populated (requires explicit REINDEX).
//!
//! ## Column Drop Migration
//!
//! Dropping a column requires migrating all existing rows to remove the column data.
//! This is done in batches (10,000 rows) to avoid holding locks too long.
//! Related indexes are automatically dropped before the column is removed.

use crate::database::{Database, ExecuteResult};
use crate::memory::Pool;
use crate::schema::ColumnDef as SchemaColumnDef;
use crate::storage::{IndexFileHeader, TableFileHeader, DEFAULT_SCHEMA};
use crate::types::{create_record_schema, OwnedValue};
use bumpalo::Bump;
use eyre::{bail, Result, WrapErr};

impl Database {
    pub(crate) fn execute_create_table(
        &self,
        create: &crate::sql::ast::CreateTableStmt<'_>,
        _arena: &Bump,
    ) -> Result<ExecuteResult> {
        self.ensure_catalog()?;
        self.ensure_file_manager()?;

        let mut catalog_guard = self.shared.catalog.write();
        let catalog = catalog_guard.as_mut().unwrap();

        let schema_name = create.schema.unwrap_or(DEFAULT_SCHEMA);
        let table_name = create.name;

        if catalog
            .get_schema(schema_name)
            .is_some_and(|s| s.table_exists(table_name))
        {
            if create.if_not_exists {
                return Ok(ExecuteResult::CreateTable { created: false });
            }
            bail!(
                "table '{}' already exists in schema '{}'",
                table_name,
                schema_name
            );
        }

        let mut unique_columns: Vec<(String, bool)> = Vec::new();

        let columns: Vec<SchemaColumnDef> = create
            .columns
            .iter()
            .map(|col| {
                use crate::schema::table::Constraint as SchemaConstraint;
                use crate::sql::ast::ColumnConstraint;
                use crate::sql::ast::DataType as SqlDataType;

                let data_type = Self::convert_data_type(&col.data_type);
                let mut column = SchemaColumnDef::new(col.name.to_string(), data_type);

                if let Some(max_len) = Self::extract_type_length(&col.data_type) {
                    column = column.with_max_length(max_len);
                }

                for constraint in col.constraints {
                    match constraint {
                        ColumnConstraint::NotNull => {
                            column = column.with_constraint(SchemaConstraint::NotNull);
                        }
                        ColumnConstraint::Unique => {
                            column = column.with_constraint(SchemaConstraint::Unique);
                            unique_columns.push((col.name.to_string(), false));
                        }
                        ColumnConstraint::PrimaryKey => {
                            column = column.with_constraint(SchemaConstraint::PrimaryKey);
                            column = column.with_constraint(SchemaConstraint::NotNull);
                            unique_columns.push((col.name.to_string(), true));
                        }
                        ColumnConstraint::Default(expr) => {
                            if let Some(default_str) = Self::expr_to_default_string(expr) {
                                column = column.with_default(default_str);
                            }
                        }
                        ColumnConstraint::Check(expr) => {
                            if let Some(check_str) = Self::expr_to_string(expr) {
                                column = column.with_constraint(SchemaConstraint::Check(check_str));
                            }
                        }
                        ColumnConstraint::References {
                            table,
                            column: ref_col,
                            on_delete,
                            on_update,
                        } => {
                            let fk_column = ref_col.unwrap_or(col.name);
                            column = column.with_constraint(SchemaConstraint::ForeignKey {
                                table: table.to_string(),
                                column: fk_column.to_string(),
                                on_delete: Self::convert_referential_action(*on_delete),
                                on_update: Self::convert_referential_action(*on_update),
                            });
                        }
                        ColumnConstraint::AutoIncrement => {
                            column = column.with_constraint(SchemaConstraint::AutoIncrement);
                        }
                        ColumnConstraint::Null | ColumnConstraint::Generated { .. } => {}
                    }
                }

                if matches!(
                    col.data_type,
                    SqlDataType::Serial | SqlDataType::BigSerial | SqlDataType::SmallSerial
                ) {
                    column = column.with_constraint(SchemaConstraint::AutoIncrement);
                    column = column.with_constraint(SchemaConstraint::NotNull);
                }

                column
            })
            .collect();

        let column_count = columns.len() as u32;
        let table_id = self.allocate_table_id();
        catalog.create_table_with_id(schema_name, table_name, columns, table_id)?;

        let estimated_table_memory = 1024 + (column_count as usize * 128);
        let _ = self
            .shared
            .memory_budget
            .allocate(Pool::Schema, estimated_table_memory);

        drop(catalog_guard);

        self.shared.table_id_lookup.write().insert(
            table_id as u32,
            (schema_name.to_string(), table_name.to_string()),
        );

        let mut file_manager_guard = self.shared.file_manager.write();
        let file_manager = file_manager_guard.as_mut().unwrap();
        file_manager.create_table(schema_name, table_name, table_id, column_count)?;

        let storage_arc = file_manager.table_data_mut(schema_name, table_name)?;
        let mut storage = storage_arc.write();
        storage.grow(2)?;
        crate::btree::BTree::create(&mut *storage, 1)?;

        let needs_toast = {
            let catalog_guard = self.shared.catalog.read();
            let catalog = catalog_guard.as_ref().unwrap();
            catalog
                .get_schema(schema_name)
                .and_then(|s| s.get_table(table_name))
                .map(|t| t.columns().iter().any(|c| c.data_type().is_toastable()))
                .unwrap_or(false)
        };

        if needs_toast {
            let toast_table_name = crate::storage::toast::toast_table_name(table_name);
            let toast_id = self.allocate_table_id();
            file_manager.create_table(schema_name, &toast_table_name, toast_id, 3)?;

            let toast_storage_arc = file_manager.table_data_mut(schema_name, &toast_table_name)?;
            let mut toast_storage = toast_storage_arc.write();
            toast_storage.grow(2)?;
            crate::btree::BTree::create(&mut *toast_storage, 1)?;

            self.shared.table_id_lookup.write().insert(
                toast_id as u32,
                (schema_name.to_string(), toast_table_name.clone()),
            );

            let mut catalog_guard = self.shared.catalog.write();
            let catalog = catalog_guard.as_mut().unwrap();
            if let Some(schema) = catalog.get_schema_mut(schema_name) {
                if let Some(table) = schema.get_table(table_name) {
                    let mut table_clone = table.clone();
                    table_clone.set_toast_id(Some(toast_id));
                    schema.remove_table(table_name);
                    schema.add_table(table_clone);
                }
            }
        }

        for (col_name, is_primary_key) in &unique_columns {
            let index_name = if *is_primary_key {
                format!("{}_pkey", col_name)
            } else {
                format!("{}_key", col_name)
            };

            let index_id = self.allocate_index_id();
            file_manager.create_index(
                schema_name,
                table_name,
                &index_name,
                index_id,
                table_id,
                1,
                true,
            )?;

            let index_storage_arc =
                file_manager.index_data_mut(schema_name, table_name, &index_name)?;
            let mut index_storage = index_storage_arc.write();
            index_storage.grow(2)?;
            crate::btree::BTree::create(&mut *index_storage, 1)?;

            let index_def = crate::schema::table::IndexDef::new(
                index_name.clone(),
                vec![col_name.clone()],
                true,
                crate::schema::table::IndexType::BTree,
            );

            let mut catalog_guard = self.shared.catalog.write();
            let catalog = catalog_guard.as_mut().unwrap();
            if let Some(schema) = catalog.get_schema_mut(schema_name) {
                if let Some(table) = schema.get_table(table_name) {
                    let table_with_index = table.clone().with_index(index_def);
                    schema.remove_table(table_name);
                    schema.add_table(table_with_index);
                }
            }
        }

        self.save_catalog()?;
        self.save_meta()?;

        Ok(ExecuteResult::CreateTable { created: true })
    }

    pub(crate) fn execute_create_schema(
        &self,
        create: &crate::sql::ast::CreateSchemaStmt<'_>,
    ) -> Result<ExecuteResult> {
        self.ensure_catalog()?;

        let mut catalog_guard = self.shared.catalog.write();
        let catalog = catalog_guard.as_mut().unwrap();

        if catalog.schema_exists(create.name) {
            if create.if_not_exists {
                return Ok(ExecuteResult::CreateSchema { created: false });
            }
            bail!("schema '{}' already exists", create.name);
        }

        catalog.create_schema(create.name)?;

        drop(catalog_guard);

        let schema_dir = self.path().join(create.name);
        std::fs::create_dir_all(&schema_dir)?;

        self.save_catalog()?;

        Ok(ExecuteResult::CreateSchema { created: true })
    }

    fn format_expr(expr: &crate::sql::ast::Expr<'_>) -> String {
        use crate::sql::ast::{Expr, FunctionArgs, Literal};

        match expr {
            Expr::Column(col) => {
                if let Some(table) = col.table {
                    format!("{}.{}", table, col.column)
                } else {
                    col.column.to_string()
                }
            }
            Expr::Literal(lit) => match lit {
                Literal::Null => "NULL".to_string(),
                Literal::Boolean(b) => {
                    if *b {
                        "TRUE".to_string()
                    } else {
                        "FALSE".to_string()
                    }
                }
                Literal::Integer(i) => i.to_string(),
                Literal::Float(f) => f.to_string(),
                Literal::String(s) => format!("'{}'", s),
                _ => format!("{:?}", lit),
            },
            Expr::Function(func) => {
                let name = func.name.name;
                let args = match func.args {
                    FunctionArgs::None => String::new(),
                    FunctionArgs::Star => "*".to_string(),
                    FunctionArgs::Args(args) => args
                        .iter()
                        .map(|arg| Self::format_expr(arg.value))
                        .collect::<Vec<_>>()
                        .join(", "),
                };
                format!("{}({})", name, args)
            }
            Expr::BinaryOp { left, op, right } => {
                format!(
                    "({} {:?} {})",
                    Self::format_expr(left),
                    op,
                    Self::format_expr(right)
                )
            }
            Expr::UnaryOp { op, expr } => {
                format!("{:?} {}", op, Self::format_expr(expr))
            }
            _ => format!("{:?}", expr),
        }
    }

    pub(crate) fn execute_create_index(
        &self,
        create: &crate::sql::ast::CreateIndexStmt<'_>,
        _arena: &Bump,
    ) -> Result<ExecuteResult> {
        use crate::btree::BTree;
        use crate::records::RecordView;
        use crate::schema::{IndexColumnDef, SortDirection};
        use crate::sql::ast::{Expr, OrderDirection};

        self.ensure_catalog()?;
        self.ensure_file_manager()?;

        let schema_name = create.table.schema.unwrap_or(DEFAULT_SCHEMA);
        let table_name = create.table.name;
        let index_name = create.name;

        let column_defs: Vec<IndexColumnDef> = create
            .columns
            .iter()
            .map(|c| {
                let direction = match c.direction {
                    Some(OrderDirection::Desc) => SortDirection::Desc,
                    Some(OrderDirection::Asc) | None => SortDirection::Asc,
                };
                let base = if let Expr::Column(ref col) = c.expr {
                    IndexColumnDef::column(col.column.to_string())
                } else {
                    IndexColumnDef::expression(Self::format_expr(c.expr))
                };
                base.with_direction(direction)
            })
            .collect();
        let key_column_count = column_defs.len() as u32;

        let has_expressions = column_defs.iter().any(|cd| cd.is_expression());

        let mut index_def = crate::schema::table::IndexDef::new_expression(
            index_name.to_string(),
            column_defs.clone(),
            create.unique,
            crate::schema::table::IndexType::BTree,
        );

        if let Some(where_clause) = create.where_clause {
            index_def = index_def.with_where_clause(Self::format_expr(where_clause));
        }

        let mut catalog_guard = self.shared.catalog.write();
        let catalog = catalog_guard.as_mut().unwrap();

        let table_def = catalog.resolve_table(table_name)?;
        let columns = table_def.columns().to_vec();
        let schema = create_record_schema(&columns);

        let index_col_indices: Vec<usize> = column_defs
            .iter()
            .filter_map(|cd| {
                cd.as_column().and_then(|col_name| {
                    columns
                        .iter()
                        .position(|c| c.name().eq_ignore_ascii_case(col_name))
                })
            })
            .collect();

        let can_populate = !has_expressions && index_col_indices.len() == column_defs.len();

        if let Some(schema_obj) = catalog.get_schema_mut(schema_name) {
            if let Some(table) = schema_obj.get_table(table_name) {
                let table_with_index = table.clone().with_index(index_def);
                schema_obj.remove_table(table_name);
                schema_obj.add_table(table_with_index);
            }
        }

        let table_id = catalog.resolve_table(table_name)?.id();

        drop(catalog_guard);

        let index_id = self.allocate_index_id();

        let mut file_manager_guard = self.shared.file_manager.write();
        let file_manager = file_manager_guard.as_mut().unwrap();
        file_manager.create_index(
            schema_name,
            table_name,
            index_name,
            index_id,
            table_id,
            key_column_count,
            create.unique,
        )?;

        {
            let index_storage_arc =
                file_manager.index_data_mut(schema_name, table_name, index_name)?;
            let mut index_storage = index_storage_arc.write();
            index_storage.grow(2)?;
            BTree::create(&mut *index_storage, 1)?;
        }

        if can_populate {
            let index_root_page = 1u32;
            let is_unique = create.unique;

            let table_storage_arc = file_manager.table_data_mut(schema_name, table_name)?;
            let index_storage_arc =
                file_manager.index_data_mut(schema_name, table_name, index_name)?;

            {
                let table_storage = table_storage_arc.read();
                let mut index_storage = index_storage_arc.write();

                let table_root_page = {
                    let page = table_storage.page(0)?;
                    let header = TableFileHeader::from_bytes(page)?;
                    header.root_page()
                };

                let table_reader =
                    crate::btree::BTreeReader::new(&table_storage, table_root_page)?;

                let mut key_buffer: Vec<u8> = Vec::new();
                let mut entries: Vec<(u32, u16, [u8; 8])> = Vec::new();
                let mut cursor = table_reader.cursor_first()?;

                if cursor.valid() {
                    loop {
                        let row_key = cursor.key()?;
                        let row_data = cursor.value()?;
                        let user_data = crate::database::dml::mvcc_helpers::get_user_data(row_data);

                        let row_id = u64::from_be_bytes(
                            row_key
                                .try_into()
                                .wrap_err("Invalid row key length in table")?,
                        );

                        let record = RecordView::new(user_data, &schema)?;

                        let key_start = key_buffer.len() as u32;
                        let mut all_non_null = true;
                        for &col_idx in &index_col_indices {
                            let col_def = &columns[col_idx];
                            let value = OwnedValue::from_record_column(
                                &record,
                                col_idx,
                                col_def.data_type(),
                            )?;
                            if value.is_null() {
                                all_non_null = false;
                                key_buffer.truncate(key_start as usize);
                                break;
                            }
                            Self::encode_value_as_key(&value, &mut key_buffer);
                        }

                        if all_non_null {
                            let row_id_bytes = row_id.to_be_bytes();
                            if !is_unique {
                                key_buffer.extend_from_slice(&row_id_bytes);
                            }
                            let key_len = (key_buffer.len() as u32 - key_start) as u16;
                            entries.push((key_start, key_len, row_id_bytes));
                        }

                        if !cursor.advance()? {
                            break;
                        }
                    }
                }

                drop(table_storage);

                entries.sort_unstable_by(|a, b| {
                    let key_a = &key_buffer[a.0 as usize..(a.0 as usize + a.1 as usize)];
                    let key_b = &key_buffer[b.0 as usize..(b.0 as usize + b.1 as usize)];
                    key_a.cmp(key_b)
                });

                let mut index_btree = BTree::new(&mut *index_storage, index_root_page)?;

                for (offset, len, row_id_bytes) in &entries {
                    let key = &key_buffer[*offset as usize..(*offset as usize + *len as usize)];
                    index_btree.insert_append(key, row_id_bytes)?;
                }

                let final_root = index_btree.root_page();
                if final_root != index_root_page {
                    let page = index_storage.page_mut(0)?;
                    let header = IndexFileHeader::from_bytes_mut(page)?;
                    header.set_root_page(final_root);
                }

                index_storage.sync()?;
            }
        }

        self.save_catalog()?;
        self.save_meta()?;

        Ok(ExecuteResult::CreateIndex { created: true })
    }

    pub(crate) fn execute_truncate(
        &self,
        truncate: &crate::sql::ast::TruncateStmt<'_>,
    ) -> Result<ExecuteResult> {
        use crate::btree::BTree;

        self.ensure_catalog()?;
        self.ensure_file_manager()?;

        let tables_info: Vec<(String, String)> = {
            let catalog_guard = self.shared.catalog.read();
            let catalog = catalog_guard.as_ref().unwrap();

            let mut info = Vec::new();
            for table_ref in truncate.tables {
                let schema_name = table_ref.schema.unwrap_or(DEFAULT_SCHEMA);
                let table_name = table_ref.name;

                catalog.resolve_table(table_name)?;
                info.push((schema_name.to_string(), table_name.to_string()));
            }
            info
        };

        let mut total_rows_affected: usize = 0;

        for (schema_name, table_name) in &tables_info {
            let mut file_manager_guard = self.shared.file_manager.write();
            let file_manager = file_manager_guard.as_mut().unwrap();
            let storage_arc = file_manager.table_data_mut(schema_name, table_name)?;
            let mut storage = storage_arc.write();

            let root_page = {
                let page = storage.page(0)?;
                TableFileHeader::from_bytes(page)?.root_page()
            };
            let btree = BTree::new(&mut *storage, root_page)?;
            let mut cursor = btree.cursor_first()?;

            let mut keys_to_delete: Vec<Vec<u8>> = Vec::new();
            while cursor.valid() {
                keys_to_delete.push(cursor.key()?.to_vec());
                cursor.advance()?;
            }

            let rows_affected = keys_to_delete.len();
            total_rows_affected += rows_affected;

            let mut btree_mut = BTree::new(&mut *storage, root_page)?;
            for key in &keys_to_delete {
                btree_mut.delete(key)?;
            }

            let page = storage.page_mut(0)?;
            let header = TableFileHeader::from_bytes_mut(page)?;

            header.set_row_count(0);

            if truncate.restart_identity {
                header.set_auto_increment(0);
            }

            storage.sync()?;

            let catalog_guard = self.shared.catalog.read();
            let catalog = catalog_guard.as_ref().unwrap();
            let table_def = catalog.resolve_table(table_name)?;
            let indexes: Vec<String> = table_def
                .indexes()
                .iter()
                .map(|i| i.name().to_string())
                .collect();
            drop(catalog_guard);

            for index_name in indexes {
                if file_manager.index_exists(schema_name, table_name, &index_name) {
                    let index_storage_arc =
                        file_manager.index_data_mut(schema_name, table_name, &index_name)?;
                    let mut index_storage = index_storage_arc.write();
                    let index_btree = BTree::new(&mut *index_storage, root_page)?;
                    let mut index_cursor = index_btree.cursor_first()?;

                    let mut index_keys_to_delete: Vec<Vec<u8>> = Vec::new();
                    while index_cursor.valid() {
                        index_keys_to_delete.push(index_cursor.key()?.to_vec());
                        index_cursor.advance()?;
                    }

                    let mut index_btree_mut = BTree::new(&mut *index_storage, root_page)?;
                    for key in &index_keys_to_delete {
                        index_btree_mut.delete(key)?;
                    }

                    index_storage.sync()?;
                }
            }
        }

        Ok(ExecuteResult::Truncate {
            rows_affected: total_rows_affected,
        })
    }

    pub(crate) fn execute_alter_table(
        &self,
        alter: &crate::sql::ast::AlterTableStmt<'_>,
    ) -> Result<ExecuteResult> {
        use crate::sql::ast::AlterTableAction;

        self.ensure_catalog()?;
        self.ensure_file_manager()?;

        let schema_name = alter.table.schema.unwrap_or(DEFAULT_SCHEMA);
        let table_name = alter.table.name;

        let action_desc = match &alter.action {
            AlterTableAction::RenameTable(new_name) => {
                {
                    let mut file_manager_guard = self.shared.file_manager.write();
                    let file_manager = file_manager_guard.as_mut().unwrap();
                    file_manager.rename_table(schema_name, table_name, new_name)?;
                }

                {
                    let mut catalog_guard = self.shared.catalog.write();
                    let catalog = catalog_guard.as_mut().unwrap();
                    let schema = catalog
                        .get_schema_mut(schema_name)
                        .ok_or_else(|| eyre::eyre!("schema '{}' not found", schema_name))?;
                    if !schema.table_exists(table_name) {
                        bail!(
                            "table '{}' not found in schema '{}'",
                            table_name,
                            schema_name
                        );
                    }
                    schema.rename_table(table_name, new_name);
                }

                format!("renamed table to '{}'", new_name)
            }
            AlterTableAction::RenameColumn { old_name, new_name } => {
                let mut catalog_guard = self.shared.catalog.write();
                let catalog = catalog_guard.as_mut().unwrap();
                let schema = catalog
                    .get_schema_mut(schema_name)
                    .ok_or_else(|| eyre::eyre!("schema '{}' not found", schema_name))?;
                if !schema.table_exists(table_name) {
                    bail!(
                        "table '{}' not found in schema '{}'",
                        table_name,
                        schema_name
                    );
                }
                let table = schema.get_table_mut(table_name).unwrap();
                if !table.rename_column(old_name, new_name) {
                    bail!("column '{}' not found in table '{}'", old_name, table_name);
                }
                format!("renamed column '{}' to '{}'", old_name, new_name)
            }
            AlterTableAction::AddColumn(col_def) => {
                let mut catalog_guard = self.shared.catalog.write();
                let catalog = catalog_guard.as_mut().unwrap();
                let schema = catalog
                    .get_schema_mut(schema_name)
                    .ok_or_else(|| eyre::eyre!("schema '{}' not found", schema_name))?;
                if !schema.table_exists(table_name) {
                    bail!(
                        "table '{}' not found in schema '{}'",
                        table_name,
                        schema_name
                    );
                }
                let table = schema.get_table_mut(table_name).unwrap();
                let column = Self::ast_column_to_schema_column(col_def)?;
                let col_name = column.name().to_string();
                table.add_column(column);
                format!("added column '{}'", col_name)
            }
            AlterTableAction::DropColumn {
                name, if_exists, ..
            } => self.migrate_table_drop_column(schema_name, table_name, name, *if_exists)?,
            _ => bail!("ALTER TABLE action not yet supported"),
        };

        self.save_catalog()?;

        Ok(ExecuteResult::AlterTable {
            action: action_desc,
        })
    }

    fn migrate_table_drop_column(
        &self,
        schema_name: &str,
        table_name: &str,
        column_name: &str,
        if_exists: bool,
    ) -> Result<String> {
        use crate::btree::BTree;
        use crate::sql::decoder::{RecordDecoder, SimpleDecoder};

        const BATCH_SIZE: usize = 10_000;

        let (old_columns, drop_idx, indexes_to_drop) = {
            let catalog_guard = self.shared.catalog.read();
            let catalog = catalog_guard.as_ref().unwrap();
            let schema = catalog
                .get_schema(schema_name)
                .ok_or_else(|| eyre::eyre!("schema '{}' not found", schema_name))?;
            let table = schema
                .get_table(table_name)
                .ok_or_else(|| eyre::eyre!("table '{}' not found", table_name))?;

            let old_columns: Vec<crate::schema::ColumnDef> = table.columns().to_vec();
            let drop_idx = old_columns
                .iter()
                .position(|c| c.name().eq_ignore_ascii_case(column_name));

            if drop_idx.is_none() && !if_exists {
                bail!(
                    "column '{}' not found in table '{}'",
                    column_name,
                    table_name
                );
            }

            let indexes_to_drop: Vec<String> = table
                .indexes()
                .iter()
                .filter(|idx| {
                    idx.columns()
                        .any(|c| c.eq_ignore_ascii_case(column_name))
                })
                .map(|idx| idx.name().to_string())
                .collect();

            (old_columns, drop_idx, indexes_to_drop)
        };

        let Some(drop_idx) = drop_idx else {
            return Ok(format!("column '{}' does not exist (skipped)", column_name));
        };

        for index_name in &indexes_to_drop {
            let mut file_manager_guard = self.shared.file_manager.write();
            let file_manager = file_manager_guard.as_mut().unwrap();

            if file_manager.index_exists(schema_name, table_name, index_name) {
                let index_storage_arc =
                    file_manager.index_data_mut(schema_name, table_name, index_name)?;
                let mut index_storage = index_storage_arc.write();
                let root_page = {
                    use crate::storage::IndexFileHeader;
                    let page = index_storage.page(0)?;
                    IndexFileHeader::from_bytes(page)?.root_page()
                };
                let index_btree = BTree::new(&mut *index_storage, root_page)?;
                let mut index_cursor = index_btree.cursor_first()?;

                let mut keys_to_delete: Vec<Vec<u8>> = Vec::new();
                while index_cursor.valid() {
                    keys_to_delete.push(index_cursor.key()?.to_vec());
                    index_cursor.advance()?;
                }

                let mut index_btree_mut = BTree::new(&mut *index_storage, root_page)?;
                for key in &keys_to_delete {
                    index_btree_mut.delete(key)?;
                }
                index_storage.sync()?;
            }
        }

        let old_column_types: Vec<crate::records::types::DataType> =
            old_columns.iter().map(|c| c.data_type()).collect();
        let decoder = SimpleDecoder::new(old_column_types);

        let mut new_columns = old_columns.clone();
        new_columns.remove(drop_idx);
        let new_schema = create_record_schema(&new_columns);

        {
            let mut file_manager_guard = self.shared.file_manager.write();
            let file_manager = file_manager_guard.as_mut().unwrap();
            let storage_arc = file_manager.table_data_mut(schema_name, table_name)?;
            let mut storage = storage_arc.write();
            let root_page = {
                let page = storage.page(0)?;
                TableFileHeader::from_bytes(page)?.root_page()
            };

            let all_keys: Vec<Vec<u8>> = {
                let btree = BTree::new(&mut *storage, root_page)?;
                let mut cursor = btree.cursor_first()?;
                let mut keys = Vec::new();
                while cursor.valid() {
                    keys.push(cursor.key()?.to_vec());
                    cursor.advance()?;
                }
                keys
            };

            for chunk in all_keys.chunks(BATCH_SIZE) {
                let mut batch: Vec<(Vec<u8>, Vec<u8>)> = Vec::with_capacity(chunk.len());

                {
                    let btree = BTree::new(&mut *storage, root_page)?;
                    for key in chunk {
                        if let Some(handle) = btree.search(key)? {
                            let value = btree.get_value(&handle)?;
                            let values = decoder.decode(key, value)?;
                            let mut owned_values: Vec<OwnedValue> =
                                values.into_iter().map(OwnedValue::from).collect();
                            owned_values.remove(drop_idx);

                            let new_record =
                                OwnedValue::build_record_from_values(&owned_values, &new_schema)?;
                            batch.push((key.clone(), new_record));
                        }
                    }
                }

                let mut btree_mut = BTree::new(&mut *storage, root_page)?;
                for (key, _) in &batch {
                    btree_mut.delete(key)?;
                }
                for (key, new_value) in &batch {
                    btree_mut.insert(key, new_value)?;
                }
            }

            storage.sync()?;
        }

        {
            let mut catalog_guard = self.shared.catalog.write();
            let catalog = catalog_guard.as_mut().unwrap();
            let schema = catalog.get_schema_mut(schema_name).unwrap();
            let table = schema.get_table_mut(table_name).unwrap();

            for index_name in &indexes_to_drop {
                table.remove_index(index_name);
            }

            table.drop_column(column_name);
        }

        Ok(format!("dropped column '{}'", column_name))
    }

    fn ast_column_to_schema_column(
        col_def: &crate::sql::ast::ColumnDef<'_>,
    ) -> Result<crate::schema::table::ColumnDef> {
        use crate::schema::table::{ColumnDef as SchemaColumnDef, Constraint as SchemaConstraint};

        let data_type = Self::convert_data_type(&col_def.data_type);
        let mut column = SchemaColumnDef::new(col_def.name, data_type);

        for constraint in col_def.constraints.iter() {
            match constraint {
                crate::sql::ast::ColumnConstraint::NotNull => {
                    column = column.with_constraint(SchemaConstraint::NotNull);
                }
                crate::sql::ast::ColumnConstraint::PrimaryKey => {
                    column = column.with_constraint(SchemaConstraint::PrimaryKey);
                    column = column.with_constraint(SchemaConstraint::NotNull);
                }
                crate::sql::ast::ColumnConstraint::Unique => {
                    column = column.with_constraint(SchemaConstraint::Unique);
                }
                crate::sql::ast::ColumnConstraint::AutoIncrement => {
                    column = column.with_constraint(SchemaConstraint::AutoIncrement);
                }
                crate::sql::ast::ColumnConstraint::Default(expr) => {
                    if let Some(default_str) = Self::expr_to_default_string(expr) {
                        column = column.with_default(default_str);
                    }
                }
                _ => {}
            }
        }

        Ok(column)
    }

    pub(crate) fn execute_drop_table(
        &self,
        drop_stmt: &crate::sql::ast::DropStmt<'_>,
    ) -> Result<ExecuteResult> {
        self.ensure_catalog()?;
        self.ensure_file_manager()?;

        let mut catalog_guard = self.shared.catalog.write();
        let catalog = catalog_guard.as_mut().unwrap();

        let mut actually_dropped = false;

        for table_ref in drop_stmt.names.iter() {
            let schema_name = table_ref.schema.unwrap_or(DEFAULT_SCHEMA);
            let table_name = table_ref.name;

            if let Some(schema) = catalog.get_schema_mut(schema_name) {
                if let Some(table_def) = schema.get_table(table_name) {
                    let table_id = table_def.id() as u32;
                    schema.remove_table(table_name);
                    self.shared.table_id_lookup.write().remove(&table_id);
                    actually_dropped = true;
                } else if !drop_stmt.if_exists {
                    bail!(
                        "table '{}' not found in schema '{}'",
                        table_name,
                        schema_name
                    );
                }
            } else if !drop_stmt.if_exists {
                bail!("schema '{}' not found", schema_name);
            }

            if actually_dropped {
                let mut file_manager_guard = self.shared.file_manager.write();
                let file_manager = file_manager_guard.as_mut().unwrap();
                let _ = file_manager.drop_table(schema_name, table_name);
            }
        }

        drop(catalog_guard);
        if actually_dropped {
            self.save_catalog()?;
        }

        Ok(ExecuteResult::DropTable {
            dropped: actually_dropped,
        })
    }

    pub(crate) fn execute_drop_index(
        &self,
        drop_stmt: &crate::sql::ast::DropStmt<'_>,
    ) -> Result<ExecuteResult> {
        self.ensure_catalog()?;
        self.ensure_file_manager()?;

        let mut indexes_to_drop: Vec<(String, String, String)> = Vec::new();

        {
            let catalog_guard = self.shared.catalog.read();
            let catalog = catalog_guard.as_ref().unwrap();

            for index_ref in drop_stmt.names.iter() {
                let index_name = index_ref.name;

                if let Some((schema_name, table_name, _)) = catalog.find_index(index_name) {
                    indexes_to_drop.push((
                        schema_name.to_string(),
                        table_name.to_string(),
                        index_name.to_string(),
                    ));
                } else if !drop_stmt.if_exists {
                    bail!("index '{}' not found", index_name);
                }
            }
        }

        if indexes_to_drop.is_empty() {
            return Ok(ExecuteResult::DropIndex { dropped: false });
        }

        {
            let mut catalog_guard = self.shared.catalog.write();
            let catalog = catalog_guard.as_mut().unwrap();

            for (_, _, index_name) in &indexes_to_drop {
                catalog.remove_index(index_name)?;
            }
        }

        {
            let mut file_manager_guard = self.shared.file_manager.write();
            let file_manager = file_manager_guard.as_mut().unwrap();

            for (schema_name, table_name, index_name) in &indexes_to_drop {
                file_manager.drop_index(schema_name, table_name, index_name)?;
            }
        }

        self.save_catalog()?;

        Ok(ExecuteResult::DropIndex { dropped: true })
    }

    pub(crate) fn execute_drop_schema_stmt(
        &self,
        drop_stmt: &crate::sql::ast::DropStmt<'_>,
    ) -> Result<ExecuteResult> {
        self.ensure_catalog()?;
        self.ensure_file_manager()?;

        let mut catalog_guard = self.shared.catalog.write();
        let catalog = catalog_guard.as_mut().unwrap();

        for schema_ref in drop_stmt.names.iter() {
            let schema_name = schema_ref.name;

            if catalog.schema_exists(schema_name) {
                catalog.drop_schema(schema_name)?;
            } else if !drop_stmt.if_exists {
                bail!("schema '{}' not found", schema_name);
            }
        }

        drop(catalog_guard);
        self.save_catalog()?;

        Ok(ExecuteResult::DropSchema { dropped: true })
    }
}
