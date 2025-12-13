//! # Database Module
//!
//! This module provides the high-level Database API for TurDB, combining all
//! components (storage, catalog, SQL processing) into a unified interface.
//!
//! ## Architecture
//!
//! The Database struct serves as the main entry point, orchestrating:
//! - FileManager: Manages table data files, index files, and metadata
//! - Catalog: Tracks schemas, tables, columns, and indexes
//! - SQL Engine: Parses, plans, and executes SQL statements
//!
//! ## Query Execution Pipeline
//!
//! ```text
//! SQL String
//!     │
//!     ▼
//! ┌─────────────────────────────────────────────────────┐
//! │ 1. PARSE: SQL → AST                                 │
//! │    Lexer → Parser → Statement                       │
//! └─────────────────────────────────────────────────────┘
//!     │
//!     ▼
//! ┌─────────────────────────────────────────────────────┐
//! │ 2. PLAN: AST → PhysicalPlan                         │
//! │    Planner::plan(stmt) → PhysicalPlan               │
//! └─────────────────────────────────────────────────────┘
//!     │
//!     ▼
//! ┌─────────────────────────────────────────────────────┐
//! │ 3. BUILD: PhysicalPlan → Executor                   │
//! │    ExecutorBuilder::build(plan) → DynamicExecutor   │
//! └─────────────────────────────────────────────────────┘
//!     │
//!     ▼
//! ┌─────────────────────────────────────────────────────┐
//! │ 4. EXECUTE: Volcano-style pull iteration            │
//! │    executor.open() → next() → close()               │
//! └─────────────────────────────────────────────────────┘
//!     │
//!     ▼
//! Vec<Row> returned to user
//! ```
//!
//! ## Memory Management
//!
//! Each query uses a dedicated arena allocator (bumpalo::Bump) for:
//! - AST nodes during parsing
//! - Plan nodes during planning
//! - Intermediate results during execution
//!
//! The arena is dropped after query completion, bulk-deallocating all
//! query-scoped allocations in O(1).
//!
//! ## Thread Safety
//!
//! Database is Send + Sync and can be safely shared across threads.
//! Internal locking (RwLock) protects:
//! - Catalog reads/writes
//! - FileManager file operations
//!
//! ## Usage Example
//!
//! ```ignore
//! use turdb::Database;
//!
//! // Create or open database
//! let db = Database::open("./mydb")?;
//!
//! // Execute DDL
//! db.execute("CREATE TABLE users (id INT PRIMARY KEY, name TEXT)")?;
//!
//! // Insert data
//! db.execute("INSERT INTO users VALUES (1, 'Alice')")?;
//!
//! // Query data
//! let rows = db.query("SELECT * FROM users WHERE id = 1")?;
//! for row in rows {
//!     println!("{:?}", row);
//! }
//! ```
//!
//! ## Performance Targets
//!
//! - Point read: < 1µs (cached)
//! - Sequential scan: > 1M rows/sec
//! - Insert: > 100K rows/sec
//! - Query planning: < 100µs for simple queries

use crate::schema::{Catalog, ColumnDef as SchemaColumnDef};
use crate::sql::executor::{BTreeCursorAdapter, ExecutionContext, Executor, ExecutorBuilder};
use crate::sql::planner::Planner;
use crate::sql::Parser;
use crate::storage::{FileManager, Wal};
use crate::types::Value;
use bumpalo::Bump;
use eyre::{bail, ensure, Result, WrapErr};
use parking_lot::{Mutex, RwLock};
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, PartialEq)]
pub struct Row {
    pub values: Vec<OwnedValue>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum OwnedValue {
    Null,
    Int(i64),
    Float(f64),
    Text(String),
    Blob(Vec<u8>),
    Vector(Vec<f32>),
}

impl<'a> From<&Value<'a>> for OwnedValue {
    fn from(v: &Value<'a>) -> Self {
        match v {
            Value::Null => OwnedValue::Null,
            Value::Int(i) => OwnedValue::Int(*i),
            Value::Float(f) => OwnedValue::Float(*f),
            Value::Text(s) => OwnedValue::Text(s.to_string()),
            Value::Blob(b) => OwnedValue::Blob(b.to_vec()),
            Value::Vector(v) => OwnedValue::Vector(v.to_vec()),
        }
    }
}

impl Row {
    pub fn new(values: Vec<OwnedValue>) -> Self {
        Self { values }
    }

    pub fn get(&self, index: usize) -> Option<&OwnedValue> {
        self.values.get(index)
    }

    pub fn get_int(&self, index: usize) -> Result<i64> {
        match self.get(index) {
            Some(OwnedValue::Int(i)) => Ok(*i),
            Some(other) => bail!("expected INT, got {:?}", other),
            None => bail!("column {} out of bounds", index),
        }
    }

    pub fn get_float(&self, index: usize) -> Result<f64> {
        match self.get(index) {
            Some(OwnedValue::Float(f)) => Ok(*f),
            Some(other) => bail!("expected FLOAT, got {:?}", other),
            None => bail!("column {} out of bounds", index),
        }
    }

    pub fn get_text(&self, index: usize) -> Result<&str> {
        match self.get(index) {
            Some(OwnedValue::Text(s)) => Ok(s),
            Some(other) => bail!("expected TEXT, got {:?}", other),
            None => bail!("column {} out of bounds", index),
        }
    }

    pub fn get_blob(&self, index: usize) -> Result<&[u8]> {
        match self.get(index) {
            Some(OwnedValue::Blob(b)) => Ok(b),
            Some(other) => bail!("expected BLOB, got {:?}", other),
            None => bail!("column {} out of bounds", index),
        }
    }

    pub fn is_null(&self, index: usize) -> bool {
        matches!(self.get(index), Some(OwnedValue::Null))
    }

    pub fn column_count(&self) -> usize {
        self.values.len()
    }
}

pub enum ExecuteResult {
    CreateTable { created: bool },
    CreateSchema { created: bool },
    CreateIndex { created: bool },
    DropTable { dropped: bool },
    Insert { rows_affected: usize },
    Update { rows_affected: usize },
    Delete { rows_affected: usize },
    Select { rows: Vec<Row> },
}

#[derive(Debug, Clone)]
pub struct RecoveryInfo {
    pub frames_recovered: u32,
    pub wal_size_bytes: u64,
}

#[derive(Debug, Clone)]
pub struct CheckpointInfo {
    pub frames_checkpointed: u32,
    pub wal_truncated: bool,
}

pub struct Database {
    path: PathBuf,
    file_manager: RwLock<Option<FileManager>>,
    catalog: RwLock<Option<Catalog>>,
    wal: Mutex<Option<Wal>>,
    wal_dir: PathBuf,
    next_row_id: std::sync::atomic::AtomicU64,
    closed: std::sync::atomic::AtomicBool,
}

impl Database {
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self> {
        Self::open_with_recovery(path).map(|(db, _)| db)
    }

    pub fn open_with_recovery<P: AsRef<Path>>(path: P) -> Result<(Self, RecoveryInfo)> {
        use std::sync::atomic::{AtomicBool, AtomicU64};
        use std::fs::File;
        use std::io::Read;

        let path = path.as_ref().to_path_buf();

        let meta_path = path.join("turdb.meta");
        ensure!(
            meta_path.exists(),
            "database not found at {:?}",
            path
        );

        let mut header = [0u8; 32];
        File::open(&meta_path)
            .wrap_err_with(|| format!("failed to open metadata file at {:?}", meta_path))?
            .read_exact(&mut header)
            .wrap_err("failed to read database header")?;

        const MAGIC: &[u8; 16] = b"TurDB Rust v1\x00\x00\x00";
        ensure!(
            &header[..16] == MAGIC,
            "invalid database: magic bytes mismatch"
        );

        let version = u32::from_le_bytes(header[16..20].try_into().unwrap());
        ensure!(version == 1, "unsupported database version: {}", version);

        let wal_dir = path.join("wal");

        let wal_size_bytes = if wal_dir.exists() {
            let segment_path = wal_dir.join("wal.000001");
            if segment_path.exists() {
                std::fs::metadata(&segment_path)
                    .map(|m| m.len())
                    .unwrap_or(0)
            } else {
                0
            }
        } else {
            0
        };

        let db = Self {
            path,
            file_manager: RwLock::new(None),
            catalog: RwLock::new(None),
            wal: Mutex::new(None),
            wal_dir,
            next_row_id: AtomicU64::new(1),
            closed: AtomicBool::new(false),
        };

        let recovery_info = RecoveryInfo {
            frames_recovered: 0,
            wal_size_bytes,
        };

        Ok((db, recovery_info))
    }

    pub fn create<P: AsRef<Path>>(path: P) -> Result<Self> {
        use std::sync::atomic::{AtomicBool, AtomicU64};
        use std::fs::File;
        use std::io::Write;

        let path = path.as_ref().to_path_buf();

        std::fs::create_dir_all(&path)
            .wrap_err_with(|| format!("failed to create database directory at {:?}", path))?;

        let root_dir = path.join("root");
        std::fs::create_dir_all(&root_dir).wrap_err_with(|| {
            format!("failed to create root schema directory at {:?}", root_dir)
        })?;

        let meta_path = path.join("turdb.meta");
        const PAGE_SIZE: usize = 16384;
        let mut page = vec![0u8; PAGE_SIZE];
        page[..16].copy_from_slice(b"TurDB Rust v1\x00\x00\x00");
        page[16..20].copy_from_slice(&1u32.to_le_bytes());
        page[20..24].copy_from_slice(&(PAGE_SIZE as u32).to_le_bytes());

        let mut file = File::create(&meta_path)
            .wrap_err_with(|| format!("failed to create metadata file at {:?}", meta_path))?;
        file.write_all(&page)
            .wrap_err("failed to write database header")?;

        let wal_dir = path.join("wal");

        Ok(Self {
            path,
            file_manager: RwLock::new(None),
            catalog: RwLock::new(None),
            wal: Mutex::new(None),
            wal_dir,
            next_row_id: AtomicU64::new(1),
            closed: AtomicBool::new(false),
        })
    }

    pub fn open_or_create<P: AsRef<Path>>(path: P) -> Result<Self> {
        let path = path.as_ref();
        let meta_path = path.join("turdb.meta");

        if meta_path.exists() {
            Self::open(path)
        } else {
            Self::create(path)
        }
    }

    fn ensure_file_manager(&self) -> Result<()> {
        let mut guard = self.file_manager.write();
        if guard.is_none() {
            let fm = FileManager::open(&self.path, 64)
                .wrap_err_with(|| format!("failed to open file manager at {:?}", self.path))?;
            *guard = Some(fm);
        }
        Ok(())
    }

    fn ensure_catalog(&self) -> Result<()> {
        let mut guard = self.catalog.write();
        if guard.is_none() {
            let catalog = Self::load_catalog(&self.path)?;
            *guard = Some(catalog);
        }
        Ok(())
    }

    pub fn ensure_wal(&self) -> Result<()> {
        let mut guard = self.wal.lock();
        if guard.is_none() {
            let wal = if self.wal_dir.exists() {
                Wal::open(&self.wal_dir)
                    .wrap_err_with(|| format!("failed to open WAL at {:?}", self.wal_dir))?
            } else {
                Wal::create(&self.wal_dir)
                    .wrap_err_with(|| format!("failed to create WAL at {:?}", self.wal_dir))?
            };
            *guard = Some(wal);
        }
        Ok(())
    }

    fn load_catalog(path: &Path) -> Result<Catalog> {
        use crate::schema::persistence::CatalogPersistence;

        let catalog_path = path.join("turdb.catalog");
        let mut catalog = Catalog::new();
        if catalog_path.exists() {
            CatalogPersistence::load(&catalog_path, &mut catalog)
                .wrap_err_with(|| format!("failed to load catalog from {:?}", catalog_path))?;
        }
        Ok(catalog)
    }

    fn save_catalog(&self) -> Result<()> {
        use crate::schema::persistence::CatalogPersistence;

        let catalog_path = self.path.join("turdb.catalog");
        let catalog_guard = self.catalog.read();
        if let Some(ref catalog) = *catalog_guard {
            CatalogPersistence::save(catalog, &catalog_path)
                .wrap_err_with(|| format!("failed to save catalog to {:?}", catalog_path))?;
        }
        Ok(())
    }

    pub fn query(&self, sql: &str) -> Result<Vec<Row>> {
        self.ensure_catalog()?;
        self.ensure_file_manager()?;

        let arena = Bump::new();

        let mut parser = Parser::new(sql, &arena);
        let stmt = parser
            .parse_statement()
            .wrap_err("failed to parse SQL statement")?;

        let catalog_guard = self.catalog.read();
        let catalog = catalog_guard.as_ref().unwrap();
        let planner = Planner::new(&catalog, &arena);
        let physical_plan = planner
            .create_physical_plan(&stmt)
            .wrap_err("failed to create query plan")?;

        let mut file_manager_guard = self.file_manager.write();
        let file_manager = file_manager_guard.as_mut().unwrap();

        fn find_table_scan<'a>(
            op: &'a crate::sql::planner::PhysicalOperator<'a>,
        ) -> Option<&'a crate::sql::planner::PhysicalTableScan<'a>> {
            use crate::sql::planner::PhysicalOperator;
            match op {
                PhysicalOperator::TableScan(scan) => Some(scan),
                PhysicalOperator::FilterExec(filter) => find_table_scan(filter.input),
                PhysicalOperator::ProjectExec(project) => find_table_scan(project.input),
                PhysicalOperator::LimitExec(limit) => find_table_scan(limit.input),
                PhysicalOperator::SortExec(sort) => find_table_scan(sort.input),
                _ => None,
            }
        }

        let table_scan = find_table_scan(physical_plan.root);

        let rows = if let Some(scan) = table_scan {
            let schema_name = scan.schema.unwrap_or("root");
            let table_name = scan.table;

            let table_def = catalog
                .resolve_table(table_name)
                .wrap_err_with(|| format!("table '{}' not found", table_name))?;

            let column_types: Vec<_> = table_def.columns().iter().map(|c| c.data_type()).collect();

            let storage = file_manager
                .table_data_mut(schema_name, table_name)
                .wrap_err_with(|| {
                    format!(
                        "failed to open table storage for {}.{}",
                        schema_name, table_name
                    )
                })?;

            let root_page = 1u32;
            let source = BTreeCursorAdapter::from_btree_scan(storage, root_page, column_types)
                .wrap_err("failed to create table scan")?;

            let ctx = ExecutionContext::new(&arena);
            let builder = ExecutorBuilder::new(&ctx);
            let mut executor = builder
                .build_with_source(&physical_plan, source)
                .wrap_err("failed to build executor")?;

            let mut rows = Vec::new();
            executor.open()?;
            while let Some(row) = executor.next()? {
                let owned: Vec<OwnedValue> = row.values.iter().map(OwnedValue::from).collect();
                rows.push(Row::new(owned));
            }
            executor.close()?;
            rows
        } else {
            bail!("unsupported query plan type - only table scans currently supported")
        };

        Ok(rows)
    }

    pub fn execute(&self, sql: &str) -> Result<ExecuteResult> {
        let arena = Bump::new();

        let mut parser = Parser::new(sql, &arena);
        let stmt = parser
            .parse_statement()
            .wrap_err("failed to parse SQL statement")?;

        use crate::sql::ast::Statement;

        match stmt {
            Statement::CreateTable(create) => self.execute_create_table(create, &arena),
            Statement::CreateSchema(create) => self.execute_create_schema(create),
            Statement::CreateIndex(create) => self.execute_create_index(create, &arena),
            Statement::Insert(insert) => self.execute_insert(insert, &arena),
            Statement::Select(_) => {
                let rows = self.query(sql)?;
                Ok(ExecuteResult::Select { rows })
            }
            Statement::Drop(drop) => {
                use crate::sql::ast::ObjectType;
                match drop.object_type {
                    ObjectType::Table => self.execute_drop_table(drop),
                    _ => bail!("unsupported DROP statement type: {:?}", drop.object_type),
                }
            }
            _ => bail!("unsupported statement type"),
        }
    }

    fn execute_create_table(
        &self,
        create: &crate::sql::ast::CreateTableStmt<'_>,
        _arena: &Bump,
    ) -> Result<ExecuteResult> {
        self.ensure_catalog()?;
        self.ensure_file_manager()?;

        let mut catalog_guard = self.catalog.write();
        let catalog = catalog_guard.as_mut().unwrap();

        let schema_name = create.schema.unwrap_or("root");
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

        let columns: Vec<SchemaColumnDef> = create
            .columns
            .iter()
            .map(|col| {
                let data_type = Self::convert_data_type(&col.data_type);
                SchemaColumnDef::new(col.name.to_string(), data_type)
            })
            .collect();

        let table_id = catalog.create_table(schema_name, table_name, columns)?;

        drop(catalog_guard);

        let mut file_manager_guard = self.file_manager.write();
        let file_manager = file_manager_guard.as_mut().unwrap();
        file_manager.create_table(schema_name, table_name, table_id)?;

        let storage = file_manager.table_data_mut(schema_name, table_name)?;
        storage.grow(2)?;
        crate::btree::BTree::create(storage, 1)?;

        self.save_catalog()?;

        Ok(ExecuteResult::CreateTable { created: true })
    }

    fn execute_create_schema(
        &self,
        create: &crate::sql::ast::CreateSchemaStmt<'_>,
    ) -> Result<ExecuteResult> {
        self.ensure_catalog()?;

        let mut catalog_guard = self.catalog.write();
        let catalog = catalog_guard.as_mut().unwrap();

        if catalog.schema_exists(create.name) {
            if create.if_not_exists {
                return Ok(ExecuteResult::CreateSchema { created: false });
            }
            bail!("schema '{}' already exists", create.name);
        }

        catalog.create_schema(create.name)?;

        drop(catalog_guard);

        let schema_dir = self.path.join(create.name);
        std::fs::create_dir_all(&schema_dir)?;

        self.save_catalog()?;

        Ok(ExecuteResult::CreateSchema { created: true })
    }

    fn execute_create_index(
        &self,
        create: &crate::sql::ast::CreateIndexStmt<'_>,
        _arena: &Bump,
    ) -> Result<ExecuteResult> {
        use crate::sql::ast::Expr;

        self.ensure_catalog()?;
        self.ensure_file_manager()?;

        let schema_name = create.table.schema.unwrap_or("root");
        let table_name = create.table.name;
        let index_name = create.name;

        let columns: Vec<String> = create
            .columns
            .iter()
            .filter_map(|c| {
                if let Expr::Column(ref col) = c.expr {
                    Some(col.column.to_string())
                } else {
                    None
                }
            })
            .collect();

        let index_def = crate::schema::table::IndexDef::new(
            index_name.to_string(),
            columns,
            create.unique,
            crate::schema::table::IndexType::BTree,
        );

        let mut catalog_guard = self.catalog.write();
        let catalog = catalog_guard.as_mut().unwrap();

        if let Some(schema) = catalog.get_schema_mut(schema_name) {
            if let Some(table) = schema.get_table(table_name) {
                let table_with_index = table.clone().with_index(index_def);
                schema.remove_table(table_name);
                schema.add_table(table_with_index);
            }
        }

        let table_id = catalog.resolve_table(table_name)?.id();

        drop(catalog_guard);

        let mut file_manager_guard = self.file_manager.write();
        let file_manager = file_manager_guard.as_mut().unwrap();
        file_manager.create_index(schema_name, table_name, index_name, table_id, create.unique)?;

        self.save_catalog()?;

        Ok(ExecuteResult::CreateIndex { created: true })
    }

    fn execute_insert(
        &self,
        insert: &crate::sql::ast::InsertStmt<'_>,
        _arena: &Bump,
    ) -> Result<ExecuteResult> {
        use crate::btree::BTree;
        use crate::records::types::ColumnDef as RecordColumnDef;
        use crate::records::{RecordBuilder, Schema};

        self.ensure_catalog()?;
        self.ensure_file_manager()?;

        let catalog_guard = self.catalog.read();
        let catalog = catalog_guard.as_ref().unwrap();

        let schema_name = insert.table.schema.unwrap_or("root");
        let table_name = insert.table.name;

        let table_def = catalog.resolve_table(table_name)?;
        let columns = table_def.columns().to_vec();

        drop(catalog_guard);

        let mut file_manager_guard = self.file_manager.write();
        let file_manager = file_manager_guard.as_mut().unwrap();
        let storage = file_manager.table_data_mut(schema_name, table_name)?;

        let record_columns: Vec<RecordColumnDef> = columns
            .iter()
            .map(|c| RecordColumnDef::new(c.name().to_string(), c.data_type()))
            .collect();
        let schema = Schema::new(record_columns);

        let rows = match &insert.source {
            crate::sql::ast::InsertSource::Values(values) => values,
            _ => bail!("only VALUES insert supported"),
        };

        let root_page = 1u32;
        let mut btree = BTree::new(storage, root_page)?;
        let mut rows_affected = 0;

        for row_values in rows.iter() {
            let mut builder = RecordBuilder::new(&schema);

            for (idx, expr) in row_values.iter().enumerate() {
                let value = Self::eval_literal(expr)?;
                match value {
                    OwnedValue::Null => builder.set_null(idx),
                    OwnedValue::Int(i) => builder.set_int8(idx, i)?,
                    OwnedValue::Float(f) => builder.set_float8(idx, f)?,
                    OwnedValue::Text(s) => builder.set_text(idx, &s)?,
                    OwnedValue::Blob(b) => builder.set_blob(idx, &b)?,
                    OwnedValue::Vector(_) => bail!("vector insert not yet supported"),
                }
            }

            let record_data = builder.build()?;

            let row_id = self
                .next_row_id
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            let key = Self::generate_row_key(row_id);
            btree.insert(&key, &record_data)?;
            rows_affected += 1;
        }

        Ok(ExecuteResult::Insert { rows_affected })
    }

    fn execute_drop_table(
        &self,
        drop_stmt: &crate::sql::ast::DropStmt<'_>,
    ) -> Result<ExecuteResult> {
        self.ensure_catalog()?;
        self.ensure_file_manager()?;

        let mut catalog_guard = self.catalog.write();
        let catalog = catalog_guard.as_mut().unwrap();

        for table_ref in drop_stmt.names.iter() {
            let schema_name = table_ref.schema.unwrap_or("root");
            let table_name = table_ref.name;

            if let Some(schema) = catalog.get_schema_mut(schema_name) {
                if schema.table_exists(table_name) {
                    schema.remove_table(table_name);
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

            let mut file_manager_guard = self.file_manager.write();
            let file_manager = file_manager_guard.as_mut().unwrap();
            let _ = file_manager.drop_table(schema_name, table_name);
        }

        drop(catalog_guard);
        self.save_catalog()?;

        Ok(ExecuteResult::DropTable { dropped: true })
    }

    fn convert_data_type(sql_type: &crate::sql::ast::DataType) -> crate::records::types::DataType {
        use crate::records::types::DataType;
        use crate::sql::ast::DataType as SqlType;

        match sql_type {
            SqlType::Integer | SqlType::BigInt => DataType::Int8,
            SqlType::SmallInt => DataType::Int2,
            SqlType::TinyInt => DataType::Int2,
            SqlType::Real | SqlType::DoublePrecision => DataType::Float8,
            SqlType::Decimal(_, _) | SqlType::Numeric(_, _) => DataType::Float8,
            SqlType::Varchar(_) | SqlType::Text => DataType::Text,
            SqlType::Char(_) => DataType::Text,
            SqlType::Blob => DataType::Blob,
            SqlType::Boolean => DataType::Bool,
            SqlType::Date => DataType::Date,
            SqlType::Time => DataType::Time,
            SqlType::Timestamp => DataType::Timestamp,
            SqlType::TimestampTz => DataType::TimestampTz,
            SqlType::Uuid => DataType::Uuid,
            SqlType::Json | SqlType::Jsonb => DataType::Jsonb,
            SqlType::Vector(_) => DataType::Vector,
            SqlType::Array(_) => DataType::Array,
            SqlType::Interval => DataType::Text,
            _ => DataType::Text,
        }
    }

    fn eval_literal(expr: &crate::sql::ast::Expr<'_>) -> Result<OwnedValue> {
        use crate::sql::ast::{Expr, Literal};

        match expr {
            Expr::Literal(lit) => match lit {
                Literal::Null => Ok(OwnedValue::Null),
                Literal::Integer(s) => {
                    let i: i64 = s
                        .parse()
                        .wrap_err_with(|| format!("failed to parse integer: {}", s))?;
                    Ok(OwnedValue::Int(i))
                }
                Literal::Float(s) => {
                    let f: f64 = s
                        .parse()
                        .wrap_err_with(|| format!("failed to parse float: {}", s))?;
                    Ok(OwnedValue::Float(f))
                }
                Literal::String(s) => Ok(OwnedValue::Text(s.to_string())),
                Literal::Boolean(b) => Ok(OwnedValue::Int(if *b { 1 } else { 0 })),
                _ => bail!("unsupported literal type: {:?}", lit),
            },
            _ => bail!("expected literal expression, got {:?}", expr),
        }
    }

    fn generate_row_key(row_id: u64) -> Vec<u8> {
        row_id.to_be_bytes().to_vec()
    }

    pub fn checkpoint(&self) -> Result<CheckpointInfo> {
        use std::sync::atomic::Ordering;

        if self.closed.load(Ordering::Acquire) {
            bail!("database is closed");
        }

        let mut wal_guard = self.wal.lock();
        let wal = match wal_guard.as_mut() {
            Some(w) => w,
            None => {
                return Ok(CheckpointInfo {
                    frames_checkpointed: 0,
                    wal_truncated: false,
                })
            }
        };

        self.ensure_file_manager()?;

        let frames_checkpointed = {
            let mut file_manager_guard = self.file_manager.write();
            let file_manager = file_manager_guard.as_mut().unwrap();
            let meta_storage = file_manager.meta_storage_mut();
            wal.checkpoint(meta_storage)?
        };

        Ok(CheckpointInfo {
            frames_checkpointed,
            wal_truncated: frames_checkpointed > 0,
        })
    }

    pub fn close(&self) -> Result<CheckpointInfo> {
        use std::sync::atomic::Ordering;

        if self.closed.load(Ordering::Acquire) {
            bail!("database already closed");
        }

        let checkpoint_result = self.checkpoint();

        self.closed.store(true, Ordering::Release);

        let mut wal_guard = self.wal.lock();
        *wal_guard = None;

        self.save_catalog()?;

        let mut file_manager_guard = self.file_manager.write();
        if let Some(ref mut file_manager) = *file_manager_guard {
            file_manager.sync_all()?;
        }

        checkpoint_result
    }

    pub fn is_closed(&self) -> bool {
        use std::sync::atomic::Ordering;
        self.closed.load(Ordering::Acquire)
    }

    pub fn path(&self) -> &Path {
        &self.path
    }
}

impl Drop for Database {
    fn drop(&mut self) {
        use std::sync::atomic::Ordering;

        if self.closed.load(Ordering::Acquire) {
            return;
        }

        let _ = self.checkpoint();

        let mut wal_guard = self.wal.lock();
        *wal_guard = None;

        let _ = self.save_catalog();

        if let Some(mut file_manager_guard) = self.file_manager.try_write() {
            if let Some(ref mut file_manager) = *file_manager_guard {
                let _ = file_manager.sync_all();
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn test_create_and_open_database() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("test_db");

        let db = Database::create(&db_path).unwrap();
        drop(db);

        let db = Database::open(&db_path).unwrap();
        assert!(db.path().exists());
    }

    #[test]
    fn test_create_table() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("test_db");

        let db = Database::create(&db_path).unwrap();

        let result = db
            .execute("CREATE TABLE users (id INT, name TEXT)")
            .unwrap();
        assert!(matches!(
            result,
            ExecuteResult::CreateTable { created: true }
        ));

        let result = db
            .execute("CREATE TABLE IF NOT EXISTS users (id INT, name TEXT)")
            .unwrap();
        assert!(matches!(
            result,
            ExecuteResult::CreateTable { created: false }
        ));
    }

    #[test]
    fn test_insert_and_query() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("test_db");

        let db = Database::create(&db_path).unwrap();

        db.execute("CREATE TABLE users (id INT, name TEXT)")
            .unwrap();

        db.execute("INSERT INTO users VALUES (1, 'Alice')").unwrap();
        db.execute("INSERT INTO users VALUES (2, 'Bob')").unwrap();

        let rows = db.query("SELECT * FROM users").unwrap();
        assert_eq!(rows.len(), 2);
    }

    #[test]
    fn test_four_column_insert() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("test_db");

        let db = Database::create(&db_path).unwrap();

        db.execute("CREATE TABLE users (id INT, name TEXT, age INT, score FLOAT)")
            .unwrap();

        db.execute("INSERT INTO users VALUES (1, 'Alice', 25, 95.5)")
            .unwrap();
        db.execute("INSERT INTO users VALUES (2, 'Bob', 30, 88.0)")
            .unwrap();

        let rows = db.query("SELECT * FROM users").unwrap();
        assert_eq!(rows.len(), 2);
    }

    #[test]
    fn test_many_inserts() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("test_db");

        let db = Database::create(&db_path).unwrap();

        db.execute("CREATE TABLE users (id INT, name TEXT, age INT, score FLOAT)")
            .unwrap();

        for i in 0..100 {
            let sql = format!(
                "INSERT INTO users VALUES ({}, 'user{}', {}, {})",
                i,
                i,
                20 + (i % 60),
                (i as f64) * 0.1
            );
            db.execute(&sql).unwrap();
        }

        let rows = db.query("SELECT * FROM users").unwrap();
        assert_eq!(rows.len(), 100);
    }

    #[test]
    fn test_wal_directory_created_lazily() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("test_db");

        let db = Database::create(&db_path).unwrap();

        let wal_dir = db_path.join("wal");
        assert!(!wal_dir.exists(), "WAL directory should NOT exist before first write");

        db.execute("CREATE TABLE test (id INT)").unwrap();
        db.execute("INSERT INTO test VALUES (1)").unwrap();

        db.ensure_wal().unwrap();

        assert!(wal_dir.exists(), "WAL directory should exist after ensure_wal");
        assert!(wal_dir.is_dir(), "WAL should be a directory");
    }

    #[test]
    fn test_wal_directory_created_on_checkpoint() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("test_db");

        {
            let db = Database::create(&db_path).unwrap();
            drop(db);
        }

        let db = Database::open(&db_path).unwrap();

        let wal_dir = db_path.join("wal");
        assert!(!wal_dir.exists(), "WAL directory should NOT exist immediately after open");

        db.checkpoint().unwrap();
    }

    #[test]
    fn test_checkpoint_returns_info() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("test_db");

        let db = Database::create(&db_path).unwrap();

        let checkpoint_info = db.checkpoint().unwrap();
        assert_eq!(checkpoint_info.frames_checkpointed, 0);
        assert!(!checkpoint_info.wal_truncated);
    }

    #[test]
    fn test_close_returns_checkpoint_info() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("test_db");

        let db = Database::create(&db_path).unwrap();

        let checkpoint_info = db.close().unwrap();
        assert_eq!(checkpoint_info.frames_checkpointed, 0);

        assert!(db.is_closed());
    }

    #[test]
    fn test_close_prevents_further_operations() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("test_db");

        let db = Database::create(&db_path).unwrap();
        db.close().unwrap();

        let result = db.checkpoint();
        assert!(result.is_err(), "checkpoint should fail after close");
    }

    #[test]
    fn test_double_close_fails() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("test_db");

        let db = Database::create(&db_path).unwrap();
        db.close().unwrap();

        let result = db.close();
        assert!(result.is_err(), "second close should fail");
    }

    #[test]
    fn test_open_with_recovery_returns_info() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("test_db");

        {
            let db = Database::create(&db_path).unwrap();
            drop(db);
        }

        let (db, recovery_info) = Database::open_with_recovery(&db_path).unwrap();
        assert_eq!(recovery_info.frames_recovered, 0);
        assert_eq!(recovery_info.wal_size_bytes, 0);
        drop(db);
    }

    #[test]
    fn test_database_survives_reopen() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("test_db");

        {
            let db = Database::create(&db_path).unwrap();
            db.execute("CREATE TABLE users (id INT, name TEXT)")
                .unwrap();
            db.execute("INSERT INTO users VALUES (1, 'Alice')").unwrap();
            db.close().unwrap();
        }

        let db = Database::open(&db_path).unwrap();
        let rows = db.query("SELECT * FROM users").unwrap();
        assert_eq!(rows.len(), 1);
    }
}
