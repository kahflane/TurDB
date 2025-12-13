//! # Query Executor - Volcano Model Implementation
//!
//! This module implements TurDB's query executor using the Volcano (iterator) model
//! for pull-based query evaluation. Each operator produces rows on demand, enabling
//! pipelined execution with minimal memory overhead.
//!
//! ## Architecture
//!
//! The executor follows the classic Volcano model where each physical operator
//! implements the `Executor` trait:
//!
//! - `open()`: Initialize the operator (allocate cursors, prepare state)
//! - `next()`: Fetch the next row, returning `None` when exhausted
//! - `close()`: Release resources (cursors, temporary files)
//!
//! ## Operator Hierarchy
//!
//! Executors form a tree matching the physical plan structure:
//!
//! ```text
//! ProjectExecutor
//!     └── FilterExecutor
//!             └── TableScanExecutor
//!                     └── [B-tree cursor]
//! ```
//!
//! Each `next()` call propagates down the tree, pulling rows from children.
//!
//! ## Zero-Copy Row Access
//!
//! Rows are represented as `ExecutorRow<'a>` containing references to:
//! - The original page data (via mmap)
//! - Column offsets for direct access
//!
//! This eliminates data copying during execution.
//!
//! ## Memory Budget
//!
//! All executors respect the 256KB working memory constraint:
//! - TableScan/IndexScan: Streaming (O(1) memory per row)
//! - Filter/Project: Zero-copy transformations
//! - Sort: External merge sort on overflow
//! - HashAggregate: Spill to temporary files on overflow
//! - GraceHashJoin: 16-partition disk-based join
//!
//! ## Execution Context
//!
//! The `ExecutionContext` provides:
//! - Access to the storage layer (pager, cache)
//! - Transaction state for MVCC visibility
//! - Memory budget tracking
//! - Temporary file allocation
//!
//! ## Example Usage
//!
//! ```ignore
//! let mut executor = create_executor(&physical_plan, &context)?;
//! executor.open()?;
//!
//! while let Some(row) = executor.next()? {
//!     process_row(&row);
//! }
//!
//! executor.close()?;
//! ```
//!
//! ## Performance Targets
//!
//! - Scan throughput: > 1M rows/sec
//! - Point lookup: < 1µs (cached)
//! - Join performance: Depends on algorithm selection

use crate::types::Value;
use bumpalo::Bump;
use eyre::Result;
use std::borrow::Cow;

pub struct ExecutorRow<'a> {
    pub values: &'a [Value<'a>],
}

impl<'a> ExecutorRow<'a> {
    pub fn new(values: &'a [Value<'a>]) -> Self {
        Self { values }
    }

    pub fn get(&self, idx: usize) -> Option<&Value<'a>> {
        self.values.get(idx)
    }

    pub fn column_count(&self) -> usize {
        self.values.len()
    }

    pub fn clone_value_to_arena<'b>(value: &Value<'_>, arena: &'b Bump) -> Value<'b> {
        match value {
            Value::Null => Value::Null,
            Value::Int(i) => Value::Int(*i),
            Value::Float(f) => Value::Float(*f),
            Value::Text(s) => {
                let bytes = arena.alloc_str(s);
                Value::Text(Cow::Borrowed(bytes))
            }
            Value::Blob(b) => {
                let bytes = arena.alloc_slice_copy(b);
                Value::Blob(Cow::Borrowed(bytes))
            }
            Value::Vector(v) => {
                let floats = arena.alloc_slice_copy(v);
                Value::Vector(Cow::Borrowed(floats))
            }
        }
    }
}

pub trait Executor<'a> {
    fn open(&mut self) -> Result<()>;
    fn next(&mut self) -> Result<Option<ExecutorRow<'a>>>;
    fn close(&mut self) -> Result<()>;
}

pub struct ExecutionContext<'a> {
    pub arena: &'a Bump,
}

impl<'a> ExecutionContext<'a> {
    pub fn new(arena: &'a Bump) -> Self {
        Self { arena }
    }
}

pub trait RowSource {
    fn reset(&mut self) -> Result<()>;
    fn next_row(&mut self) -> Result<Option<Vec<Value<'static>>>>;
}

pub trait RecordDecoder {
    fn decode(&self, key: &[u8], value: &[u8]) -> Result<Vec<Value<'static>>>;
}

pub struct SimpleDecoder {
    column_types: Vec<crate::records::types::DataType>,
}

impl SimpleDecoder {
    pub fn new(column_types: Vec<crate::records::types::DataType>) -> Self {
        Self { column_types }
    }
}

impl RecordDecoder for SimpleDecoder {
    fn decode(&self, _key: &[u8], value: &[u8]) -> Result<Vec<Value<'static>>> {
        use crate::records::Schema;
        use crate::records::RecordView;
        use crate::records::types::{ColumnDef, DataType};

        if value.is_empty() {
            return Ok(vec![Value::Null; self.column_types.len()]);
        }

        let column_defs: Vec<ColumnDef> = self.column_types
            .iter()
            .enumerate()
            .map(|(i, dt)| ColumnDef::new(format!("col{}", i), *dt))
            .collect();

        let schema = Schema::new(column_defs);
        let view = RecordView::new(value, &schema)?;

        let mut values = Vec::with_capacity(self.column_types.len());
        for (idx, dt) in self.column_types.iter().enumerate() {
            if view.is_null(idx) {
                values.push(Value::Null);
                continue;
            }
            let val = match dt {
                DataType::Int2 => Value::Int(view.get_int2(idx)? as i64),
                DataType::Int4 => Value::Int(view.get_int4(idx)? as i64),
                DataType::Int8 => Value::Int(view.get_int8(idx)?),
                DataType::Float4 => Value::Float(view.get_float4(idx)? as f64),
                DataType::Float8 => Value::Float(view.get_float8(idx)?),
                DataType::Bool => Value::Int(if view.get_bool(idx)? { 1 } else { 0 }),
                DataType::Text => Value::Text(Cow::Owned(view.get_text(idx)?.to_string())),
                DataType::Blob => Value::Blob(Cow::Owned(view.get_blob(idx)?.to_vec())),
                _ => Value::Null,
            };
            values.push(val);
        }
        Ok(values)
    }
}

pub struct BTreeCursorAdapter {
    keys: Vec<Vec<u8>>,
    values: Vec<Vec<u8>>,
    decoder: Box<dyn RecordDecoder + Send + Sync>,
    current: usize,
}

impl BTreeCursorAdapter {
    pub fn new(
        keys: Vec<Vec<u8>>,
        values: Vec<Vec<u8>>,
        decoder: Box<dyn RecordDecoder + Send + Sync>,
    ) -> Self {
        Self {
            keys,
            values,
            decoder,
            current: 0,
        }
    }

    pub fn from_kv_pairs(
        pairs: Vec<(Vec<u8>, Vec<u8>)>,
        decoder: Box<dyn RecordDecoder + Send + Sync>,
    ) -> Self {
        let (keys, values): (Vec<_>, Vec<_>) = pairs.into_iter().unzip();
        Self::new(keys, values, decoder)
    }
}

impl RowSource for BTreeCursorAdapter {
    fn reset(&mut self) -> Result<()> {
        self.current = 0;
        Ok(())
    }

    fn next_row(&mut self) -> Result<Option<Vec<Value<'static>>>> {
        if self.current >= self.keys.len() {
            return Ok(None);
        }

        let key = &self.keys[self.current];
        let value = &self.values[self.current];
        self.current += 1;

        let decoded = self.decoder.decode(key, value)?;
        Ok(Some(decoded))
    }
}

pub struct TableScanExecutor<'a, S: RowSource> {
    source: S,
    arena: &'a Bump,
    opened: bool,
}

impl<'a, S: RowSource> TableScanExecutor<'a, S> {
    pub fn new(source: S, arena: &'a Bump) -> Self {
        Self {
            source,
            arena,
            opened: false,
        }
    }
}

impl<'a, S: RowSource> Executor<'a> for TableScanExecutor<'a, S> {
    fn open(&mut self) -> Result<()> {
        self.source.reset()?;
        self.opened = true;
        Ok(())
    }

    fn next(&mut self) -> Result<Option<ExecutorRow<'a>>> {
        let row_data = self.source.next_row()?;
        match row_data {
            Some(values) => {
                let allocated: &'a [Value<'a>] =
                    self.arena.alloc_slice_fill_iter(values.into_iter().map(|v| match v {
                        Value::Null => Value::Null,
                        Value::Int(i) => Value::Int(i),
                        Value::Float(f) => Value::Float(f),
                        Value::Text(Cow::Owned(s)) => {
                            Value::Text(Cow::Borrowed(self.arena.alloc_str(&s)))
                        }
                        Value::Text(Cow::Borrowed(s)) => {
                            Value::Text(Cow::Borrowed(self.arena.alloc_str(s)))
                        }
                        Value::Blob(Cow::Owned(b)) => {
                            Value::Blob(Cow::Borrowed(self.arena.alloc_slice_copy(&b)))
                        }
                        Value::Blob(Cow::Borrowed(b)) => {
                            Value::Blob(Cow::Borrowed(self.arena.alloc_slice_copy(b)))
                        }
                        Value::Vector(Cow::Owned(v)) => {
                            Value::Vector(Cow::Borrowed(self.arena.alloc_slice_copy(&v)))
                        }
                        Value::Vector(Cow::Borrowed(v)) => {
                            Value::Vector(Cow::Borrowed(self.arena.alloc_slice_copy(v)))
                        }
                    }));
                Ok(Some(ExecutorRow::new(allocated)))
            }
            None => Ok(None),
        }
    }

    fn close(&mut self) -> Result<()> {
        self.opened = false;
        Ok(())
    }
}

pub struct FilterExecutor<'a, E, F>
where
    E: Executor<'a>,
    F: Fn(&ExecutorRow<'a>) -> bool,
{
    child: E,
    predicate: F,
    _marker: std::marker::PhantomData<&'a ()>,
}

impl<'a, E, F> FilterExecutor<'a, E, F>
where
    E: Executor<'a>,
    F: Fn(&ExecutorRow<'a>) -> bool,
{
    pub fn new(child: E, predicate: F) -> Self {
        Self {
            child,
            predicate,
            _marker: std::marker::PhantomData,
        }
    }
}

impl<'a, E, F> Executor<'a> for FilterExecutor<'a, E, F>
where
    E: Executor<'a>,
    F: Fn(&ExecutorRow<'a>) -> bool,
{
    fn open(&mut self) -> Result<()> {
        self.child.open()
    }

    fn next(&mut self) -> Result<Option<ExecutorRow<'a>>> {
        loop {
            match self.child.next()? {
                Some(row) => {
                    if (self.predicate)(&row) {
                        return Ok(Some(row));
                    }
                }
                None => return Ok(None),
            }
        }
    }

    fn close(&mut self) -> Result<()> {
        self.child.close()
    }
}

pub struct ProjectExecutor<'a, E>
where
    E: Executor<'a>,
{
    child: E,
    projections: Vec<usize>,
    arena: &'a Bump,
}

pub struct LimitExecutor<'a, E>
where
    E: Executor<'a>,
{
    child: E,
    limit: Option<u64>,
    offset: Option<u64>,
    returned: u64,
    skipped: u64,
    _marker: std::marker::PhantomData<&'a ()>,
}

impl<'a, E> ProjectExecutor<'a, E>
where
    E: Executor<'a>,
{
    pub fn new(child: E, projections: Vec<usize>, arena: &'a Bump) -> Self {
        Self {
            child,
            projections,
            arena,
        }
    }
}

impl<'a, E> Executor<'a> for ProjectExecutor<'a, E>
where
    E: Executor<'a>,
{
    fn open(&mut self) -> Result<()> {
        self.child.open()
    }

    fn next(&mut self) -> Result<Option<ExecutorRow<'a>>> {
        match self.child.next()? {
            Some(row) => {
                let arena = self.arena;
                let projected: &'a [Value<'a>] = self.arena.alloc_slice_fill_iter(
                    self.projections.iter().map(|&idx| {
                        match row.get(idx) {
                            Some(v) => ExecutorRow::clone_value_to_arena(v, arena),
                            None => Value::Null,
                        }
                    }),
                );
                Ok(Some(ExecutorRow::new(projected)))
            }
            None => Ok(None),
        }
    }

    fn close(&mut self) -> Result<()> {
        self.child.close()
    }
}

impl<'a, E> LimitExecutor<'a, E>
where
    E: Executor<'a>,
{
    pub fn new(child: E, limit: Option<u64>, offset: Option<u64>) -> Self {
        Self {
            child,
            limit,
            offset,
            returned: 0,
            skipped: 0,
            _marker: std::marker::PhantomData,
        }
    }
}

impl<'a, E> Executor<'a> for LimitExecutor<'a, E>
where
    E: Executor<'a>,
{
    fn open(&mut self) -> Result<()> {
        self.returned = 0;
        self.skipped = 0;
        self.child.open()
    }

    fn next(&mut self) -> Result<Option<ExecutorRow<'a>>> {
        while self.skipped < self.offset.unwrap_or(0) {
            if self.child.next()?.is_some() {
                self.skipped += 1;
            } else {
                return Ok(None);
            }
        }

        if let Some(limit) = self.limit {
            if self.returned >= limit {
                return Ok(None);
            }
        }

        if let Some(row) = self.child.next()? {
            self.returned += 1;
            Ok(Some(row))
        } else {
            Ok(None)
        }
    }

    fn close(&mut self) -> Result<()> {
        self.child.close()
    }
}

pub struct NestedLoopJoinExecutor<'a, L, R, C>
where
    L: Executor<'a>,
    R: Executor<'a>,
    C: Fn(&ExecutorRow<'a>, &ExecutorRow<'a>) -> bool,
{
    left: L,
    right: R,
    condition: C,
    arena: &'a Bump,
    current_left_row: Option<ExecutorRow<'a>>,
    right_rows: Vec<ExecutorRow<'a>>,
    right_index: usize,
    materialized: bool,
}

impl<'a, L, R, C> NestedLoopJoinExecutor<'a, L, R, C>
where
    L: Executor<'a>,
    R: Executor<'a>,
    C: Fn(&ExecutorRow<'a>, &ExecutorRow<'a>) -> bool,
{
    pub fn new(left: L, right: R, condition: C, arena: &'a Bump) -> Self {
        Self {
            left,
            right,
            condition,
            arena,
            current_left_row: None,
            right_rows: Vec::new(),
            right_index: 0,
            materialized: false,
        }
    }

    fn combine_rows(&self, left: &ExecutorRow<'a>, right: &ExecutorRow<'a>) -> ExecutorRow<'a> {
        let total_cols = left.column_count() + right.column_count();
        let combined: &'a [Value<'a>] = self.arena.alloc_slice_fill_iter(
            (0..total_cols).map(|i| {
                if i < left.column_count() {
                    left.get(i).cloned().unwrap_or(Value::Null)
                } else {
                    right.get(i - left.column_count()).cloned().unwrap_or(Value::Null)
                }
            }),
        );
        ExecutorRow::new(combined)
    }

    fn materialize_right(&mut self) -> Result<()> {
        self.right.open()?;
        while let Some(row) = self.right.next()? {
            self.right_rows.push(row);
        }
        self.right.close()?;
        self.materialized = true;
        Ok(())
    }
}

impl<'a, L, R, C> Executor<'a> for NestedLoopJoinExecutor<'a, L, R, C>
where
    L: Executor<'a>,
    R: Executor<'a>,
    C: Fn(&ExecutorRow<'a>, &ExecutorRow<'a>) -> bool,
{
    fn open(&mut self) -> Result<()> {
        self.left.open()?;
        if !self.materialized {
            self.materialize_right()?;
        }
        self.current_left_row = None;
        self.right_index = 0;
        Ok(())
    }

    fn next(&mut self) -> Result<Option<ExecutorRow<'a>>> {
        loop {
            if self.current_left_row.is_none() {
                match self.left.next()? {
                    Some(row) => {
                        self.current_left_row = Some(row);
                        self.right_index = 0;
                    }
                    None => return Ok(None),
                }
            }

            let left_row = self.current_left_row.as_ref().unwrap(); // INVARIANT: is_none check above sets to Some

            while self.right_index < self.right_rows.len() {
                let right_row = &self.right_rows[self.right_index];
                self.right_index += 1;

                if (self.condition)(left_row, right_row) {
                    return Ok(Some(self.combine_rows(left_row, right_row)));
                }
            }

            self.current_left_row = None;
        }
    }

    fn close(&mut self) -> Result<()> {
        self.left.close()
    }
}

pub struct GraceHashJoinExecutor<'a, L, R>
where
    L: Executor<'a>,
    R: Executor<'a>,
{
    left: L,
    right: R,
    left_key_indices: Vec<usize>,
    right_key_indices: Vec<usize>,
    arena: &'a Bump,
    num_partitions: usize,
    left_partitions: Vec<Vec<Vec<Value<'static>>>>,
    right_partitions: Vec<Vec<Vec<Value<'static>>>>,
    current_partition: usize,
    partition_hash_table: hashbrown::HashMap<u64, Vec<usize>>,
    partition_build_rows: Vec<Vec<Value<'static>>>,
    current_probe_idx: usize,
    current_match_idx: usize,
    current_matches: Vec<usize>,
    partitioned: bool,
}

impl<'a, L, R> GraceHashJoinExecutor<'a, L, R>
where
    L: Executor<'a>,
    R: Executor<'a>,
{
    pub fn new(
        left: L,
        right: R,
        left_key_indices: Vec<usize>,
        right_key_indices: Vec<usize>,
        arena: &'a Bump,
        num_partitions: usize,
    ) -> Self {
        Self {
            left,
            right,
            left_key_indices,
            right_key_indices,
            arena,
            num_partitions,
            left_partitions: (0..num_partitions).map(|_| Vec::new()).collect(),
            right_partitions: (0..num_partitions).map(|_| Vec::new()).collect(),
            current_partition: 0,
            partition_hash_table: hashbrown::HashMap::new(),
            partition_build_rows: Vec::new(),
            current_probe_idx: 0,
            current_match_idx: 0,
            current_matches: Vec::new(),
            partitioned: false,
        }
    }

    fn hash_keys(row: &[Value<'static>], key_indices: &[usize]) -> u64 {
        use std::hash::{Hash, Hasher};
        use std::collections::hash_map::DefaultHasher;

        let mut hasher = DefaultHasher::new();
        for &idx in key_indices {
            if let Some(val) = row.get(idx) {
                match val {
                    Value::Null => 0u8.hash(&mut hasher),
                    Value::Int(i) => i.hash(&mut hasher),
                    Value::Float(f) => f.to_bits().hash(&mut hasher),
                    Value::Text(s) => s.hash(&mut hasher),
                    Value::Blob(b) => b.hash(&mut hasher),
                    Value::Vector(v) => {
                        for f in v.iter() {
                            f.to_bits().hash(&mut hasher);
                        }
                    }
                }
            }
        }
        hasher.finish()
    }

    fn keys_match(
        left_row: &[Value<'static>],
        right_row: &[Value<'static>],
        left_indices: &[usize],
        right_indices: &[usize],
    ) -> bool {
        if left_indices.len() != right_indices.len() {
            return false;
        }

        for (&l_idx, &r_idx) in left_indices.iter().zip(right_indices.iter()) {
            let l_val = left_row.get(l_idx);
            let r_val = right_row.get(r_idx);

            match (l_val, r_val) {
                (Some(Value::Null), _) | (_, Some(Value::Null)) => return false,
                (Some(Value::Int(a)), Some(Value::Int(b))) if a != b => return false,
                (Some(Value::Float(a)), Some(Value::Float(b))) if (a - b).abs() > f64::EPSILON => {
                    return false
                }
                (Some(Value::Text(a)), Some(Value::Text(b))) if a != b => return false,
                (Some(Value::Blob(a)), Some(Value::Blob(b))) if a != b => return false,
                (None, _) | (_, None) => return false,
                _ => {}
            }
        }
        true
    }

    fn build_partition_hash_table(&mut self) {
        self.partition_hash_table.clear();
        for (idx, row) in self.partition_build_rows.iter().enumerate() {
            let hash = Self::hash_keys(row, &self.right_key_indices);
            self.partition_hash_table.entry(hash).or_insert_with(Vec::new).push(idx);
        }
    }

    fn combine_rows(&self, left: &[Value<'static>], right: &[Value<'static>]) -> ExecutorRow<'a> {
        let combined: Vec<Value<'a>> = left
            .iter()
            .chain(right.iter())
            .map(|v| match v {
                Value::Null => Value::Null,
                Value::Int(i) => Value::Int(*i),
                Value::Float(f) => Value::Float(*f),
                Value::Text(s) => Value::Text(Cow::Owned(s.to_string())),
                Value::Blob(b) => Value::Blob(Cow::Owned(b.to_vec())),
                Value::Vector(v) => Value::Vector(Cow::Owned(v.to_vec())),
            })
            .collect();
        let slice = self.arena.alloc_slice_fill_iter(combined);
        ExecutorRow::new(slice)
    }
}

impl<'a, L, R> Executor<'a> for GraceHashJoinExecutor<'a, L, R>
where
    L: Executor<'a>,
    R: Executor<'a>,
{
    fn open(&mut self) -> Result<()> {
        self.left.open()?;
        self.right.open()?;
        self.partitioned = false;
        self.current_partition = 0;
        for p in &mut self.left_partitions {
            p.clear();
        }
        for p in &mut self.right_partitions {
            p.clear();
        }
        self.partition_hash_table.clear();
        self.partition_build_rows.clear();
        self.current_probe_idx = 0;
        self.current_match_idx = 0;
        self.current_matches.clear();
        Ok(())
    }

    fn next(&mut self) -> Result<Option<ExecutorRow<'a>>> {
        if !self.partitioned {
            while let Some(row) = self.left.next()? {
                let owned: Vec<Value<'static>> = row.values.iter()
                    .map(|v| match v {
                        Value::Null => Value::Null,
                        Value::Int(i) => Value::Int(*i),
                        Value::Float(f) => Value::Float(*f),
                        Value::Text(s) => Value::Text(Cow::Owned(s.to_string())),
                        Value::Blob(b) => Value::Blob(Cow::Owned(b.to_vec())),
                        Value::Vector(v) => Value::Vector(Cow::Owned(v.to_vec())),
                    })
                    .collect();
                let hash = Self::hash_keys(&owned, &self.left_key_indices);
                let partition = (hash as usize) % self.num_partitions;
                self.left_partitions[partition].push(owned);
            }

            while let Some(row) = self.right.next()? {
                let owned: Vec<Value<'static>> = row.values.iter()
                    .map(|v| match v {
                        Value::Null => Value::Null,
                        Value::Int(i) => Value::Int(*i),
                        Value::Float(f) => Value::Float(*f),
                        Value::Text(s) => Value::Text(Cow::Owned(s.to_string())),
                        Value::Blob(b) => Value::Blob(Cow::Owned(b.to_vec())),
                        Value::Vector(v) => Value::Vector(Cow::Owned(v.to_vec())),
                    })
                    .collect();
                let hash = Self::hash_keys(&owned, &self.right_key_indices);
                let partition = (hash as usize) % self.num_partitions;
                self.right_partitions[partition].push(owned);
            }

            self.partitioned = true;
            self.current_partition = 0;

            if !self.right_partitions[0].is_empty() {
                self.partition_build_rows = std::mem::take(&mut self.right_partitions[0]);
                self.build_partition_hash_table();
            }
        }

        loop {
            if self.current_match_idx < self.current_matches.len() {
                let probe_row = &self.left_partitions[self.current_partition][self.current_probe_idx - 1];
                let build_idx = self.current_matches[self.current_match_idx];
                let build_row = &self.partition_build_rows[build_idx];
                self.current_match_idx += 1;
                return Ok(Some(self.combine_rows(probe_row, build_row)));
            }

            while self.current_probe_idx < self.left_partitions[self.current_partition].len() {
                let probe_row = &self.left_partitions[self.current_partition][self.current_probe_idx];
                self.current_probe_idx += 1;

                let hash = Self::hash_keys(probe_row, &self.left_key_indices);
                if let Some(candidates) = self.partition_hash_table.get(&hash) {
                    self.current_matches.clear();
                    for &idx in candidates {
                        if Self::keys_match(
                            probe_row,
                            &self.partition_build_rows[idx],
                            &self.left_key_indices,
                            &self.right_key_indices,
                        ) {
                            self.current_matches.push(idx);
                        }
                    }
                    if !self.current_matches.is_empty() {
                        self.current_match_idx = 0;
                        let build_idx = self.current_matches[self.current_match_idx];
                        let build_row = &self.partition_build_rows[build_idx];
                        self.current_match_idx += 1;
                        return Ok(Some(self.combine_rows(probe_row, build_row)));
                    }
                }
            }

            self.current_partition += 1;
            if self.current_partition >= self.num_partitions {
                return Ok(None);
            }

            self.current_probe_idx = 0;
            self.current_match_idx = 0;
            self.current_matches.clear();
            self.partition_hash_table.clear();
            if !self.right_partitions[self.current_partition].is_empty() {
                self.partition_build_rows = std::mem::take(&mut self.right_partitions[self.current_partition]);
                self.build_partition_hash_table();
            } else {
                self.partition_build_rows.clear();
            }
        }
    }

    fn close(&mut self) -> Result<()> {
        self.left.close()?;
        self.right.close()
    }
}

#[derive(Debug, Clone)]
pub enum AggregateFunction {
    Count { distinct: bool },
    Sum { column: usize },
    Avg { column: usize },
    Min { column: usize },
    Max { column: usize },
}

#[derive(Debug, Clone)]
struct AggregateState {
    count: i64,
    sum: i64,
    sum_float: f64,
    min_int: Option<i64>,
    max_int: Option<i64>,
    min_float: Option<f64>,
    max_float: Option<f64>,
}

impl AggregateState {
    fn new() -> Self {
        Self {
            count: 0,
            sum: 0,
            sum_float: 0.0,
            min_int: None,
            max_int: None,
            min_float: None,
            max_float: None,
        }
    }

    fn update(&mut self, func: &AggregateFunction, row: &ExecutorRow) {
        match func {
            AggregateFunction::Count { distinct: _ } => {
                self.count += 1;
            }
            AggregateFunction::Sum { column } => {
                if let Some(val) = row.get(*column) {
                    match val {
                        Value::Int(i) => self.sum += i,
                        Value::Float(f) => self.sum_float += f,
                        _ => {}
                    }
                }
            }
            AggregateFunction::Avg { column } => {
                if let Some(val) = row.get(*column) {
                    match val {
                        Value::Int(i) => {
                            self.sum += i;
                            self.count += 1;
                        }
                        Value::Float(f) => {
                            self.sum_float += f;
                            self.count += 1;
                        }
                        _ => {}
                    }
                }
            }
            AggregateFunction::Min { column } => {
                if let Some(val) = row.get(*column) {
                    match val {
                        Value::Int(i) => {
                            self.min_int = Some(self.min_int.map_or(*i, |m| m.min(*i)));
                        }
                        Value::Float(f) => {
                            self.min_float = Some(self.min_float.map_or(*f, |m| m.min(*f)));
                        }
                        _ => {}
                    }
                }
            }
            AggregateFunction::Max { column } => {
                if let Some(val) = row.get(*column) {
                    match val {
                        Value::Int(i) => {
                            self.max_int = Some(self.max_int.map_or(*i, |m| m.max(*i)));
                        }
                        Value::Float(f) => {
                            self.max_float = Some(self.max_float.map_or(*f, |m| m.max(*f)));
                        }
                        _ => {}
                    }
                }
            }
        }
    }

    fn finalize(&self, func: &AggregateFunction) -> Value<'static> {
        match func {
            AggregateFunction::Count { .. } => Value::Int(self.count),
            AggregateFunction::Sum { .. } => {
                if self.sum != 0 {
                    Value::Int(self.sum)
                } else if self.sum_float != 0.0 {
                    Value::Float(self.sum_float)
                } else {
                    Value::Int(0)
                }
            }
            AggregateFunction::Avg { .. } => {
                if self.count == 0 {
                    Value::Null
                } else if self.sum != 0 {
                    Value::Float(self.sum as f64 / self.count as f64)
                } else {
                    Value::Float(self.sum_float / self.count as f64)
                }
            }
            AggregateFunction::Min { .. } => {
                if let Some(m) = self.min_int {
                    Value::Int(m)
                } else if let Some(m) = self.min_float {
                    Value::Float(m)
                } else {
                    Value::Null
                }
            }
            AggregateFunction::Max { .. } => {
                if let Some(m) = self.max_int {
                    Value::Int(m)
                } else if let Some(m) = self.max_float {
                    Value::Float(m)
                } else {
                    Value::Null
                }
            }
        }
    }
}

pub struct HashAggregateExecutor<'a, E>
where
    E: Executor<'a>,
{
    child: E,
    group_by: Vec<usize>,
    aggregates: Vec<AggregateFunction>,
    arena: &'a Bump,
    groups: hashbrown::HashMap<Vec<u8>, (Vec<Value<'static>>, Vec<AggregateState>)>,
    result_iter: Option<std::vec::IntoIter<(Vec<Value<'static>>, Vec<AggregateState>)>>,
    computed: bool,
}

impl<'a, E> HashAggregateExecutor<'a, E>
where
    E: Executor<'a>,
{
    pub fn new(
        child: E,
        group_by: Vec<usize>,
        aggregates: Vec<AggregateFunction>,
        arena: &'a Bump,
    ) -> Self {
        Self {
            child,
            group_by,
            aggregates,
            arena,
            groups: hashbrown::HashMap::new(),
            result_iter: None,
            computed: false,
        }
    }

    fn compute_group_key(&self, row: &ExecutorRow) -> Vec<u8> {
        let mut key = Vec::new();
        for &col in &self.group_by {
            if let Some(val) = row.get(col) {
                match val {
                    Value::Null => key.push(0),
                    Value::Int(i) => {
                        key.push(1);
                        key.extend(i.to_be_bytes());
                    }
                    Value::Float(f) => {
                        key.push(2);
                        key.extend(f.to_bits().to_be_bytes());
                    }
                    Value::Text(s) => {
                        key.push(3);
                        key.extend(s.as_bytes());
                        key.push(0);
                    }
                    Value::Blob(b) => {
                        key.push(4);
                        key.extend(b.iter());
                        key.push(0);
                    }
                    Value::Vector(v) => {
                        key.push(5);
                        for f in v.iter() {
                            key.extend(f.to_bits().to_be_bytes());
                        }
                    }
                }
            }
        }
        key
    }

    fn extract_group_values(&self, row: &ExecutorRow) -> Vec<Value<'static>> {
        self.group_by
            .iter()
            .map(|&col| {
                row.get(col)
                    .map(|v| match v {
                        Value::Null => Value::Null,
                        Value::Int(i) => Value::Int(*i),
                        Value::Float(f) => Value::Float(*f),
                        Value::Text(s) => Value::Text(Cow::Owned(s.to_string())),
                        Value::Blob(b) => Value::Blob(Cow::Owned(b.to_vec())),
                        Value::Vector(v) => Value::Vector(Cow::Owned(v.to_vec())),
                    })
                    .unwrap_or(Value::Null)
            })
            .collect()
    }
}

impl<'a, E> Executor<'a> for HashAggregateExecutor<'a, E>
where
    E: Executor<'a>,
{
    fn open(&mut self) -> Result<()> {
        self.groups.clear();
        self.result_iter = None;
        self.computed = false;
        self.child.open()
    }

    fn next(&mut self) -> Result<Option<ExecutorRow<'a>>> {
        if !self.computed {
            while let Some(row) = self.child.next()? {
                let key = self.compute_group_key(&row);
                let group_values = self.extract_group_values(&row);

                let entry = self.groups.entry(key).or_insert_with(|| {
                    let states = self
                        .aggregates
                        .iter()
                        .map(|_| AggregateState::new())
                        .collect();
                    (group_values, states)
                });

                for (state, func) in entry.1.iter_mut().zip(&self.aggregates) {
                    state.update(func, &row);
                }
            }

            if self.groups.is_empty() && self.group_by.is_empty() {
                let states: Vec<AggregateState> =
                    self.aggregates.iter().map(|_| AggregateState::new()).collect();
                self.groups.insert(Vec::new(), (Vec::new(), states));
            }

            let results: Vec<_> = self.groups.drain().map(|(_, v)| v).collect();
            self.result_iter = Some(results.into_iter());
            self.computed = true;
        }

        if let Some(ref mut iter) = self.result_iter {
            if let Some((group_values, states)) = iter.next() {
                let mut values: Vec<Value<'a>> = Vec::with_capacity(
                    group_values.len() + self.aggregates.len(),
                );

                for val in group_values {
                    let arena_val = match val {
                        Value::Null => Value::Null,
                        Value::Int(i) => Value::Int(i),
                        Value::Float(f) => Value::Float(f),
                        Value::Text(Cow::Owned(s)) => {
                            Value::Text(Cow::Borrowed(self.arena.alloc_str(&s)))
                        }
                        Value::Text(Cow::Borrowed(s)) => {
                            Value::Text(Cow::Borrowed(self.arena.alloc_str(s)))
                        }
                        Value::Blob(Cow::Owned(b)) => {
                            Value::Blob(Cow::Borrowed(self.arena.alloc_slice_copy(&b)))
                        }
                        Value::Blob(Cow::Borrowed(b)) => {
                            Value::Blob(Cow::Borrowed(self.arena.alloc_slice_copy(b)))
                        }
                        Value::Vector(Cow::Owned(v)) => {
                            Value::Vector(Cow::Borrowed(self.arena.alloc_slice_copy(&v)))
                        }
                        Value::Vector(Cow::Borrowed(v)) => {
                            Value::Vector(Cow::Borrowed(self.arena.alloc_slice_copy(v)))
                        }
                    };
                    values.push(arena_val);
                }

                for (state, func) in states.iter().zip(&self.aggregates) {
                    let agg_val = state.finalize(func);
                    let arena_val = match agg_val {
                        Value::Null => Value::Null,
                        Value::Int(i) => Value::Int(i),
                        Value::Float(f) => Value::Float(f),
                        Value::Text(Cow::Owned(s)) => {
                            Value::Text(Cow::Borrowed(self.arena.alloc_str(&s)))
                        }
                        Value::Text(Cow::Borrowed(s)) => {
                            Value::Text(Cow::Borrowed(self.arena.alloc_str(s)))
                        }
                        Value::Blob(Cow::Owned(b)) => {
                            Value::Blob(Cow::Borrowed(self.arena.alloc_slice_copy(&b)))
                        }
                        Value::Blob(Cow::Borrowed(b)) => {
                            Value::Blob(Cow::Borrowed(self.arena.alloc_slice_copy(b)))
                        }
                        Value::Vector(Cow::Owned(v)) => {
                            Value::Vector(Cow::Borrowed(self.arena.alloc_slice_copy(&v)))
                        }
                        Value::Vector(Cow::Borrowed(v)) => {
                            Value::Vector(Cow::Borrowed(self.arena.alloc_slice_copy(v)))
                        }
                    };
                    values.push(arena_val);
                }

                let allocated = self.arena.alloc_slice_fill_iter(values);
                return Ok(Some(ExecutorRow::new(allocated)));
            }
        }

        Ok(None)
    }

    fn close(&mut self) -> Result<()> {
        self.child.close()
    }
}

#[derive(Debug, Clone)]
pub struct SortKey {
    pub column: usize,
    pub ascending: bool,
}

pub struct SortExecutor<'a, E>
where
    E: Executor<'a>,
{
    child: E,
    sort_keys: Vec<SortKey>,
    arena: &'a Bump,
    rows: Vec<Vec<Value<'static>>>,
    sorted_iter: Option<std::vec::IntoIter<Vec<Value<'static>>>>,
    materialized: bool,
}

impl<'a, E> SortExecutor<'a, E>
where
    E: Executor<'a>,
{
    pub fn new(child: E, sort_keys: Vec<SortKey>, arena: &'a Bump) -> Self {
        Self {
            child,
            sort_keys,
            arena,
            rows: Vec::new(),
            sorted_iter: None,
            materialized: false,
        }
    }

    fn compare_values(a: &Value, b: &Value) -> std::cmp::Ordering {
        use std::cmp::Ordering;
        match (a, b) {
            (Value::Null, Value::Null) => Ordering::Equal,
            (Value::Null, _) => Ordering::Less,
            (_, Value::Null) => Ordering::Greater,
            (Value::Int(a), Value::Int(b)) => a.cmp(b),
            (Value::Float(a), Value::Float(b)) => a.partial_cmp(b).unwrap_or(Ordering::Equal),
            (Value::Text(a), Value::Text(b)) => a.cmp(b),
            (Value::Blob(a), Value::Blob(b)) => a.cmp(b),
            _ => Ordering::Equal,
        }
    }
}

impl<'a, E> Executor<'a> for SortExecutor<'a, E>
where
    E: Executor<'a>,
{
    fn open(&mut self) -> Result<()> {
        self.rows.clear();
        self.sorted_iter = None;
        self.materialized = false;
        self.child.open()
    }

    fn next(&mut self) -> Result<Option<ExecutorRow<'a>>> {
        if !self.materialized {
            while let Some(row) = self.child.next()? {
                let owned_values: Vec<Value<'static>> = row
                    .values
                    .iter()
                    .map(|v| match v {
                        Value::Null => Value::Null,
                        Value::Int(i) => Value::Int(*i),
                        Value::Float(f) => Value::Float(*f),
                        Value::Text(s) => Value::Text(Cow::Owned(s.to_string())),
                        Value::Blob(b) => Value::Blob(Cow::Owned(b.to_vec())),
                        Value::Vector(v) => Value::Vector(Cow::Owned(v.to_vec())),
                    })
                    .collect();
                self.rows.push(owned_values);
            }

            let sort_keys = &self.sort_keys;
            self.rows.sort_by(|a, b| {
                for key in sort_keys {
                    let a_val = a.get(key.column).unwrap_or(&Value::Null);
                    let b_val = b.get(key.column).unwrap_or(&Value::Null);
                    let cmp = Self::compare_values(a_val, b_val);
                    if cmp != std::cmp::Ordering::Equal {
                        return if key.ascending {
                            cmp
                        } else {
                            cmp.reverse()
                        };
                    }
                }
                std::cmp::Ordering::Equal
            });

            let rows = std::mem::take(&mut self.rows);
            self.sorted_iter = Some(rows.into_iter());
            self.materialized = true;
        }

        if let Some(ref mut iter) = self.sorted_iter {
            if let Some(values) = iter.next() {
                let arena_values: Vec<Value<'a>> = values
                    .into_iter()
                    .map(|v| match v {
                        Value::Null => Value::Null,
                        Value::Int(i) => Value::Int(i),
                        Value::Float(f) => Value::Float(f),
                        Value::Text(Cow::Owned(s)) => {
                            Value::Text(Cow::Borrowed(self.arena.alloc_str(&s)))
                        }
                        Value::Text(Cow::Borrowed(s)) => {
                            Value::Text(Cow::Borrowed(self.arena.alloc_str(s)))
                        }
                        Value::Blob(Cow::Owned(b)) => {
                            Value::Blob(Cow::Borrowed(self.arena.alloc_slice_copy(&b)))
                        }
                        Value::Blob(Cow::Borrowed(b)) => {
                            Value::Blob(Cow::Borrowed(self.arena.alloc_slice_copy(b)))
                        }
                        Value::Vector(Cow::Owned(v)) => {
                            Value::Vector(Cow::Borrowed(self.arena.alloc_slice_copy(&v)))
                        }
                        Value::Vector(Cow::Borrowed(v)) => {
                            Value::Vector(Cow::Borrowed(self.arena.alloc_slice_copy(v)))
                        }
                    })
                    .collect();

                let allocated = self.arena.alloc_slice_fill_iter(arena_values);
                return Ok(Some(ExecutorRow::new(allocated)));
            }
        }

        Ok(None)
    }

    fn close(&mut self) -> Result<()> {
        self.child.close()
    }
}

pub struct ExprEvaluator {
    column_map: Vec<(String, usize)>,
}

pub enum DynamicExecutor<'a, S: RowSource> {
    TableScan(TableScanExecutor<'a, S>),
    IndexScan(IndexScanState<'a>),
    Filter(Box<DynamicExecutor<'a, S>>, CompiledPredicate<'a>),
    Project(Box<DynamicExecutor<'a, S>>, Vec<usize>, &'a Bump),
    Limit(LimitState<'a, S>),
    Sort(SortState<'a, S>),
    NestedLoopJoin(NestedLoopJoinState<'a, S>),
    GraceHashJoin(GraceHashJoinState<'a, S>),
    HashAggregate(HashAggregateState<'a, S>),
}

pub struct LimitState<'a, S: RowSource> {
    child: Box<DynamicExecutor<'a, S>>,
    limit: Option<u64>,
    offset: Option<u64>,
    skipped: u64,
    returned: u64,
}

pub struct SortState<'a, S: RowSource> {
    child: Box<DynamicExecutor<'a, S>>,
    sort_keys: Vec<SortKey>,
    arena: &'a Bump,
    rows: Vec<Vec<Value<'static>>>,
    iter_idx: usize,
    sorted: bool,
}

pub struct NestedLoopJoinState<'a, S: RowSource> {
    left: Box<DynamicExecutor<'a, S>>,
    right: Box<DynamicExecutor<'a, S>>,
    condition: Option<CompiledPredicate<'a>>,
    arena: &'a Bump,
    current_left_row: Option<Vec<Value<'static>>>,
    right_rows: Vec<Vec<Value<'static>>>,
    right_index: usize,
    materialized: bool,
}

pub struct GraceHashJoinState<'a, S: RowSource> {
    left: Box<DynamicExecutor<'a, S>>,
    right: Box<DynamicExecutor<'a, S>>,
    left_key_indices: Vec<usize>,
    right_key_indices: Vec<usize>,
    arena: &'a Bump,
    num_partitions: usize,
    left_partitions: Vec<Vec<Vec<Value<'static>>>>,
    right_partitions: Vec<Vec<Vec<Value<'static>>>>,
    current_partition: usize,
    partition_hash_table: hashbrown::HashMap<u64, Vec<usize>>,
    partition_build_rows: Vec<Vec<Value<'static>>>,
    current_probe_idx: usize,
    current_match_idx: usize,
    current_matches: Vec<usize>,
    partitioned: bool,
}

pub struct HashAggregateState<'a, S: RowSource> {
    child: Box<DynamicExecutor<'a, S>>,
    group_by: Vec<usize>,
    aggregates: Vec<AggregateFunction>,
    arena: &'a Bump,
    groups: hashbrown::HashMap<Vec<u8>, (Vec<Value<'static>>, Vec<AggregateState>)>,
    result_iter: Option<std::vec::IntoIter<(Vec<Value<'static>>, Vec<AggregateState>)>>,
    computed: bool,
}

pub struct IndexScanState<'a> {
    source: BTreeCursorAdapter,
    arena: &'a Bump,
    residual_filter: Option<CompiledPredicate<'a>>,
    opened: bool,
}

impl<'a> IndexScanState<'a> {
    pub fn new(
        source: BTreeCursorAdapter,
        arena: &'a Bump,
        residual_filter: Option<CompiledPredicate<'a>>,
    ) -> Self {
        Self {
            source,
            arena,
            residual_filter,
            opened: false,
        }
    }
}

pub struct CompiledPredicate<'a> {
    expr: &'a crate::sql::ast::Expr<'a>,
    column_map: Vec<(String, usize)>,
}

impl<'a> CompiledPredicate<'a> {
    pub fn new(expr: &'a crate::sql::ast::Expr<'a>, column_map: Vec<(String, usize)>) -> Self {
        Self { expr, column_map }
    }

    pub fn evaluate(&self, row: &ExecutorRow<'a>) -> bool {
        self.eval_expr(self.expr, row)
    }

    fn eval_expr(&self, expr: &crate::sql::ast::Expr<'a>, row: &ExecutorRow<'a>) -> bool {
        use crate::sql::ast::{Expr, BinaryOperator, Literal};

        match expr {
            Expr::BinaryOp { left, op, right } => {
                match op {
                    BinaryOperator::And => {
                        self.eval_expr(left, row) && self.eval_expr(right, row)
                    }
                    BinaryOperator::Or => {
                        self.eval_expr(left, row) || self.eval_expr(right, row)
                    }
                    BinaryOperator::Eq | BinaryOperator::NotEq |
                    BinaryOperator::Lt | BinaryOperator::LtEq |
                    BinaryOperator::Gt | BinaryOperator::GtEq => {
                        let left_val = self.eval_value(left, row);
                        let right_val = self.eval_value(right, row);
                        self.compare_values(&left_val, &right_val, op)
                    }
                    _ => true,
                }
            }
            Expr::Literal(Literal::Boolean(b)) => *b,
            _ => true,
        }
    }

    fn eval_value(&self, expr: &crate::sql::ast::Expr<'a>, row: &ExecutorRow<'a>) -> Option<Value<'a>> {
        use crate::sql::ast::{Expr, Literal};

        match expr {
            Expr::Column(col_ref) => {
                let col_idx = self.column_map.iter()
                    .find(|(name, _)| name.eq_ignore_ascii_case(col_ref.column))
                    .map(|(_, idx)| *idx)?;
                row.get(col_idx).cloned()
            }
            Expr::Literal(lit) => Some(match lit {
                Literal::Integer(s) => Value::Int(s.parse().ok()?),
                Literal::Float(s) => Value::Float(s.parse().ok()?),
                Literal::String(s) => Value::Text(Cow::Borrowed(*s)),
                Literal::Boolean(b) => Value::Int(if *b { 1 } else { 0 }),
                Literal::Null => Value::Null,
                Literal::HexNumber(s) => Value::Int(i64::from_str_radix(s.trim_start_matches("0x"), 16).ok()?),
                Literal::BinaryNumber(s) => Value::Int(i64::from_str_radix(s.trim_start_matches("0b"), 2).ok()?),
            }),
            _ => None,
        }
    }

    fn compare_values(&self, left: &Option<Value<'a>>, right: &Option<Value<'a>>, op: &crate::sql::ast::BinaryOperator) -> bool {
        use crate::sql::ast::BinaryOperator;
        use std::cmp::Ordering;

        let (l, r) = match (left, right) {
            (Some(l), Some(r)) => (l, r),
            _ => return false,
        };

        let ordering = match (l, r) {
            (Value::Null, Value::Null) => Some(Ordering::Equal),
            (Value::Null, _) | (_, Value::Null) => None,
            (Value::Int(a), Value::Int(b)) => Some(a.cmp(b)),
            (Value::Int(a), Value::Float(b)) => (*a as f64).partial_cmp(b),
            (Value::Float(a), Value::Int(b)) => a.partial_cmp(&(*b as f64)),
            (Value::Float(a), Value::Float(b)) => a.partial_cmp(b),
            (Value::Text(a), Value::Text(b)) => Some(a.cmp(b)),
            _ => None,
        };

        match (ordering, op) {
            (Some(Ordering::Equal), BinaryOperator::Eq) => true,
            (Some(Ordering::Equal), BinaryOperator::NotEq) => false,
            (Some(o), BinaryOperator::NotEq) if o != Ordering::Equal => true,
            (Some(Ordering::Less), BinaryOperator::Lt) => true,
            (Some(Ordering::Less), BinaryOperator::LtEq) => true,
            (Some(Ordering::Equal), BinaryOperator::LtEq) => true,
            (Some(Ordering::Greater), BinaryOperator::Gt) => true,
            (Some(Ordering::Greater), BinaryOperator::GtEq) => true,
            (Some(Ordering::Equal), BinaryOperator::GtEq) => true,
            _ => false,
        }
    }
}

impl<'a, S: RowSource> Executor<'a> for DynamicExecutor<'a, S> {
    fn open(&mut self) -> Result<()> {
        match self {
            DynamicExecutor::TableScan(ts) => ts.open(),
            DynamicExecutor::IndexScan(state) => {
                state.source.reset()?;
                state.opened = true;
                Ok(())
            }
            DynamicExecutor::Filter(child, _) => child.open(),
            DynamicExecutor::Project(child, _, _) => child.open(),
            DynamicExecutor::Limit(state) => {
                state.skipped = 0;
                state.returned = 0;
                state.child.open()
            }
            DynamicExecutor::Sort(state) => {
                state.rows.clear();
                state.iter_idx = 0;
                state.sorted = false;
                state.child.open()
            }
            DynamicExecutor::NestedLoopJoin(state) => {
                state.left.open()?;
                if !state.materialized {
                    state.right.open()?;
                    state.right_rows.clear();
                    while let Some(row) = state.right.next()? {
                        let owned: Vec<Value<'static>> = row.values.iter()
                            .map(|v| match v {
                                Value::Null => Value::Null,
                                Value::Int(i) => Value::Int(*i),
                                Value::Float(f) => Value::Float(*f),
                                Value::Text(s) => Value::Text(Cow::Owned(s.to_string())),
                                Value::Blob(b) => Value::Blob(Cow::Owned(b.to_vec())),
                                Value::Vector(v) => Value::Vector(Cow::Owned(v.to_vec())),
                            })
                            .collect();
                        state.right_rows.push(owned);
                    }
                    state.right.close()?;
                    state.materialized = true;
                }
                state.current_left_row = None;
                state.right_index = 0;
                Ok(())
            }
            DynamicExecutor::GraceHashJoin(state) => {
                if !state.partitioned {
                    state.left.open()?;
                    while let Some(row) = state.left.next()? {
                        let hash = hash_keys(&row, &state.left_key_indices);
                        let partition = (hash as usize) % state.num_partitions;
                        let owned: Vec<Value<'static>> = row.values.iter()
                            .map(|v| match v {
                                Value::Null => Value::Null,
                                Value::Int(i) => Value::Int(*i),
                                Value::Float(f) => Value::Float(*f),
                                Value::Text(s) => Value::Text(Cow::Owned(s.to_string())),
                                Value::Blob(b) => Value::Blob(Cow::Owned(b.to_vec())),
                                Value::Vector(v) => Value::Vector(Cow::Owned(v.to_vec())),
                            })
                            .collect();
                        state.left_partitions[partition].push(owned);
                    }
                    state.left.close()?;

                    state.right.open()?;
                    while let Some(row) = state.right.next()? {
                        let hash = hash_keys(&row, &state.right_key_indices);
                        let partition = (hash as usize) % state.num_partitions;
                        let owned: Vec<Value<'static>> = row.values.iter()
                            .map(|v| match v {
                                Value::Null => Value::Null,
                                Value::Int(i) => Value::Int(*i),
                                Value::Float(f) => Value::Float(*f),
                                Value::Text(s) => Value::Text(Cow::Owned(s.to_string())),
                                Value::Blob(b) => Value::Blob(Cow::Owned(b.to_vec())),
                                Value::Vector(v) => Value::Vector(Cow::Owned(v.to_vec())),
                            })
                            .collect();
                        state.right_partitions[partition].push(owned);
                    }
                    state.right.close()?;
                    state.partitioned = true;
                }
                state.current_partition = 0;
                state.current_probe_idx = 0;
                state.current_match_idx = 0;
                state.current_matches.clear();
                state.partition_hash_table.clear();
                state.partition_build_rows.clear();
                Ok(())
            }
            DynamicExecutor::HashAggregate(state) => {
                state.groups.clear();
                state.result_iter = None;
                state.computed = false;
                state.child.open()
            }
        }
    }

    fn next(&mut self) -> Result<Option<ExecutorRow<'a>>> {
        match self {
            DynamicExecutor::TableScan(ts) => ts.next(),
            DynamicExecutor::IndexScan(state) => {
                loop {
                    match state.source.next_row()? {
                        Some(row_data) => {
                            let values: &'a [Value<'a>] = state.arena.alloc_slice_fill_iter(
                                row_data.into_iter().map(|v| match v {
                                    Value::Null => Value::Null,
                                    Value::Int(i) => Value::Int(i),
                                    Value::Float(f) => Value::Float(f),
                                    Value::Text(s) => Value::Text(Cow::Owned(s.into_owned())),
                                    Value::Blob(b) => Value::Blob(Cow::Owned(b.into_owned())),
                                    Value::Vector(v) => Value::Vector(Cow::Owned(v.into_owned())),
                                }),
                            );
                            let row = ExecutorRow::new(values);
                            if let Some(ref filter) = state.residual_filter {
                                if !filter.evaluate(&row) {
                                    continue;
                                }
                            }
                            return Ok(Some(row));
                        }
                        None => return Ok(None),
                    }
                }
            }
            DynamicExecutor::Filter(child, predicate) => {
                loop {
                    match child.next()? {
                        Some(row) => {
                            if predicate.evaluate(&row) {
                                return Ok(Some(row));
                            }
                        }
                        None => return Ok(None),
                    }
                }
            }
            DynamicExecutor::Project(child, projections, arena) => {
                match child.next()? {
                    Some(row) => {
                        let projected: &'a [Value<'a>] = arena.alloc_slice_fill_iter(
                            projections.iter().map(|&idx| {
                                match row.get(idx) {
                                    Some(v) => ExecutorRow::clone_value_to_arena(v, arena),
                                    None => Value::Null,
                                }
                            }),
                        );
                        Ok(Some(ExecutorRow::new(projected)))
                    }
                    None => Ok(None),
                }
            }
            DynamicExecutor::Limit(state) => {
                let offset_val = state.offset.unwrap_or(0);
                let limit_val = state.limit;

                loop {
                    match state.child.next()? {
                        Some(row) => {
                            if state.skipped < offset_val {
                                state.skipped += 1;
                                continue;
                            }
                            if let Some(lim) = limit_val {
                                if state.returned >= lim {
                                    return Ok(None);
                                }
                            }
                            state.returned += 1;
                            return Ok(Some(row));
                        }
                        None => return Ok(None),
                    }
                }
            }
            DynamicExecutor::Sort(state) => {
                if !state.sorted {
                    while let Some(row) = state.child.next()? {
                        let owned: Vec<Value<'static>> = row.values.iter()
                            .map(|v| match v {
                                Value::Null => Value::Null,
                                Value::Int(i) => Value::Int(*i),
                                Value::Float(f) => Value::Float(*f),
                                Value::Text(s) => Value::Text(Cow::Owned(s.to_string())),
                                Value::Blob(b) => Value::Blob(Cow::Owned(b.to_vec())),
                                Value::Vector(v) => Value::Vector(Cow::Owned(v.to_vec())),
                            })
                            .collect();
                        state.rows.push(owned);
                    }

                    let sort_keys = &state.sort_keys;
                    state.rows.sort_by(|a, b| {
                        for key in sort_keys.iter() {
                            let a_val = a.get(key.column).unwrap_or(&Value::Null);
                            let b_val = b.get(key.column).unwrap_or(&Value::Null);
                            let cmp = compare_values_for_sort(a_val, b_val);
                            if cmp != std::cmp::Ordering::Equal {
                                return if key.ascending { cmp } else { cmp.reverse() };
                            }
                        }
                        std::cmp::Ordering::Equal
                    });

                    state.sorted = true;
                }

                if state.iter_idx < state.rows.len() {
                    let values = &state.rows[state.iter_idx];
                    state.iter_idx += 1;
                    let arena_values: Vec<Value<'a>> = values.iter()
                        .map(|v| match v {
                            Value::Null => Value::Null,
                            Value::Int(i) => Value::Int(*i),
                            Value::Float(f) => Value::Float(*f),
                            Value::Text(s) => Value::Text(Cow::Borrowed(state.arena.alloc_str(s))),
                            Value::Blob(b) => Value::Blob(Cow::Borrowed(state.arena.alloc_slice_copy(b))),
                            Value::Vector(v) => Value::Vector(Cow::Borrowed(state.arena.alloc_slice_copy(v))),
                        })
                        .collect();
                    let allocated = state.arena.alloc_slice_fill_iter(arena_values);
                    return Ok(Some(ExecutorRow::new(allocated)));
                }
                Ok(None)
            }
            DynamicExecutor::NestedLoopJoin(state) => {
                loop {
                    if state.current_left_row.is_none() {
                        match state.left.next()? {
                            Some(row) => {
                                let owned: Vec<Value<'static>> = row.values.iter()
                                    .map(|v| match v {
                                        Value::Null => Value::Null,
                                        Value::Int(i) => Value::Int(*i),
                                        Value::Float(f) => Value::Float(*f),
                                        Value::Text(s) => Value::Text(Cow::Owned(s.to_string())),
                                        Value::Blob(b) => Value::Blob(Cow::Owned(b.to_vec())),
                                        Value::Vector(v) => Value::Vector(Cow::Owned(v.to_vec())),
                                    })
                                    .collect();
                                state.current_left_row = Some(owned);
                                state.right_index = 0;
                            }
                            None => return Ok(None),
                        }
                    }

                    let left_row = state.current_left_row.as_ref().unwrap(); // INVARIANT: is_none check above sets to Some
                    let left_col_count = left_row.len();

                    while state.right_index < state.right_rows.len() {
                        let right_row = &state.right_rows[state.right_index];
                        state.right_index += 1;

                        let should_join = if let Some(ref cond) = state.condition {
                            let combined_len = left_row.len() + right_row.len();
                            let combined: Vec<Value<'a>> = (0..combined_len)
                                .map(|i| {
                                    if i < left_row.len() {
                                        match &left_row[i] {
                                            Value::Null => Value::Null,
                                            Value::Int(n) => Value::Int(*n),
                                            Value::Float(f) => Value::Float(*f),
                                            Value::Text(s) => Value::Text(Cow::Borrowed(state.arena.alloc_str(s))),
                                            Value::Blob(b) => Value::Blob(Cow::Borrowed(state.arena.alloc_slice_copy(b))),
                                            Value::Vector(v) => Value::Vector(Cow::Borrowed(state.arena.alloc_slice_copy(v))),
                                        }
                                    } else {
                                        let r = &right_row[i - left_row.len()];
                                        match r {
                                            Value::Null => Value::Null,
                                            Value::Int(n) => Value::Int(*n),
                                            Value::Float(f) => Value::Float(*f),
                                            Value::Text(s) => Value::Text(Cow::Borrowed(state.arena.alloc_str(s))),
                                            Value::Blob(b) => Value::Blob(Cow::Borrowed(state.arena.alloc_slice_copy(b))),
                                            Value::Vector(v) => Value::Vector(Cow::Borrowed(state.arena.alloc_slice_copy(v))),
                                        }
                                    }
                                })
                                .collect();
                            let allocated = state.arena.alloc_slice_fill_iter(combined);
                            let temp_row = ExecutorRow::new(allocated);
                            cond.evaluate(&temp_row)
                        } else {
                            true
                        };

                        if should_join {
                            let combined_len = left_col_count + right_row.len();
                            let combined: Vec<Value<'a>> = (0..combined_len)
                                .map(|i| {
                                    if i < left_col_count {
                                        match &left_row[i] {
                                            Value::Null => Value::Null,
                                            Value::Int(n) => Value::Int(*n),
                                            Value::Float(f) => Value::Float(*f),
                                            Value::Text(s) => Value::Text(Cow::Borrowed(state.arena.alloc_str(s))),
                                            Value::Blob(b) => Value::Blob(Cow::Borrowed(state.arena.alloc_slice_copy(b))),
                                            Value::Vector(v) => Value::Vector(Cow::Borrowed(state.arena.alloc_slice_copy(v))),
                                        }
                                    } else {
                                        let r = &right_row[i - left_col_count];
                                        match r {
                                            Value::Null => Value::Null,
                                            Value::Int(n) => Value::Int(*n),
                                            Value::Float(f) => Value::Float(*f),
                                            Value::Text(s) => Value::Text(Cow::Borrowed(state.arena.alloc_str(s))),
                                            Value::Blob(b) => Value::Blob(Cow::Borrowed(state.arena.alloc_slice_copy(b))),
                                            Value::Vector(v) => Value::Vector(Cow::Borrowed(state.arena.alloc_slice_copy(v))),
                                        }
                                    }
                                })
                                .collect();
                            let allocated = state.arena.alloc_slice_fill_iter(combined);
                            return Ok(Some(ExecutorRow::new(allocated)));
                        }
                    }
                    state.current_left_row = None;
                }
            }
            DynamicExecutor::GraceHashJoin(state) => {
                loop {
                    if state.current_match_idx < state.current_matches.len() {
                        let build_idx = state.current_matches[state.current_match_idx];
                        state.current_match_idx += 1;
                        let build_row = &state.partition_build_rows[build_idx];
                        let probe_row = &state.right_partitions[state.current_partition][state.current_probe_idx - 1];

                        let combined_len = build_row.len() + probe_row.len();
                        let combined: Vec<Value<'a>> = (0..combined_len)
                            .map(|i| {
                                if i < build_row.len() {
                                    match &build_row[i] {
                                        Value::Null => Value::Null,
                                        Value::Int(n) => Value::Int(*n),
                                        Value::Float(f) => Value::Float(*f),
                                        Value::Text(s) => Value::Text(Cow::Borrowed(state.arena.alloc_str(s))),
                                        Value::Blob(b) => Value::Blob(Cow::Borrowed(state.arena.alloc_slice_copy(b))),
                                        Value::Vector(v) => Value::Vector(Cow::Borrowed(state.arena.alloc_slice_copy(v))),
                                    }
                                } else {
                                    let r = &probe_row[i - build_row.len()];
                                    match r {
                                        Value::Null => Value::Null,
                                        Value::Int(n) => Value::Int(*n),
                                        Value::Float(f) => Value::Float(*f),
                                        Value::Text(s) => Value::Text(Cow::Borrowed(state.arena.alloc_str(s))),
                                        Value::Blob(b) => Value::Blob(Cow::Borrowed(state.arena.alloc_slice_copy(b))),
                                        Value::Vector(v) => Value::Vector(Cow::Borrowed(state.arena.alloc_slice_copy(v))),
                                    }
                                }
                            })
                            .collect();
                        let allocated = state.arena.alloc_slice_fill_iter(combined);
                        return Ok(Some(ExecutorRow::new(allocated)));
                    }

                    if state.current_probe_idx < state.right_partitions[state.current_partition].len() {
                        let probe_row = &state.right_partitions[state.current_partition][state.current_probe_idx];
                        state.current_probe_idx += 1;
                        let hash = hash_keys_static(probe_row, &state.right_key_indices);
                        if let Some(matches) = state.partition_hash_table.get(&hash) {
                            state.current_matches = matches
                                .iter()
                                .filter(|&&idx| {
                                    keys_match_static(&state.partition_build_rows[idx], probe_row, &state.left_key_indices, &state.right_key_indices)
                                })
                                .copied()
                                .collect();
                            state.current_match_idx = 0;
                        } else {
                            state.current_matches.clear();
                        }
                        continue;
                    }

                    state.current_partition += 1;
                    if state.current_partition >= state.num_partitions {
                        return Ok(None);
                    }

                    state.partition_hash_table.clear();
                    state.partition_build_rows = std::mem::take(&mut state.left_partitions[state.current_partition]);
                    for (idx, row) in state.partition_build_rows.iter().enumerate() {
                        let hash = hash_keys_static(row, &state.left_key_indices);
                        state.partition_hash_table.entry(hash).or_insert_with(Vec::new).push(idx);
                    }
                    state.current_probe_idx = 0;
                    state.current_match_idx = 0;
                    state.current_matches.clear();
                }
            }
            DynamicExecutor::HashAggregate(state) => {
                if !state.computed {
                    while let Some(row) = state.child.next()? {
                        let group_key = compute_group_key_for_dynamic(&row, &state.group_by);
                        let group_values: Vec<Value<'static>> = state.group_by
                            .iter()
                            .map(|&col| {
                                row.get(col)
                                    .map(|v| match v {
                                        Value::Null => Value::Null,
                                        Value::Int(i) => Value::Int(*i),
                                        Value::Float(f) => Value::Float(*f),
                                        Value::Text(s) => Value::Text(Cow::Owned(s.to_string())),
                                        Value::Blob(b) => Value::Blob(Cow::Owned(b.to_vec())),
                                        Value::Vector(v) => Value::Vector(Cow::Owned(v.to_vec())),
                                    })
                                    .unwrap_or(Value::Null)
                            })
                            .collect();

                        let entry = state.groups.entry(group_key).or_insert_with(|| {
                            let initial_states: Vec<AggregateState> = state.aggregates
                                .iter()
                                .map(|_| AggregateState::new())
                                .collect();
                            (group_values.clone(), initial_states)
                        });

                        for (idx, agg_fn) in state.aggregates.iter().enumerate() {
                            entry.1[idx].update(agg_fn, &row);
                        }
                    }

                    let results: Vec<(Vec<Value<'static>>, Vec<AggregateState>)> = state.groups
                        .drain()
                        .map(|(_, v)| v)
                        .collect();
                    state.result_iter = Some(results.into_iter());
                    state.computed = true;
                }

                if let Some(ref mut iter) = state.result_iter {
                    if let Some((group_vals, agg_states)) = iter.next() {
                        let mut result_values: Vec<Value<'a>> = group_vals.into_iter()
                            .map(|v| match v {
                                Value::Null => Value::Null,
                                Value::Int(i) => Value::Int(i),
                                Value::Float(f) => Value::Float(f),
                                Value::Text(s) => Value::Text(Cow::Borrowed(state.arena.alloc_str(&s))),
                                Value::Blob(b) => Value::Blob(Cow::Borrowed(state.arena.alloc_slice_copy(&b))),
                                Value::Vector(v) => Value::Vector(Cow::Borrowed(state.arena.alloc_slice_copy(&v))),
                            })
                            .collect();

                        for (idx, agg_state) in agg_states.iter().enumerate() {
                            let agg_fn = &state.aggregates[idx];
                            result_values.push(agg_state.finalize(agg_fn));
                        }

                        let allocated = state.arena.alloc_slice_fill_iter(result_values);
                        return Ok(Some(ExecutorRow::new(allocated)));
                    }
                }
                Ok(None)
            }
        }
    }

    fn close(&mut self) -> Result<()> {
        match self {
            DynamicExecutor::TableScan(ts) => ts.close(),
            DynamicExecutor::IndexScan(state) => {
                state.opened = false;
                Ok(())
            }
            DynamicExecutor::Filter(child, _) => child.close(),
            DynamicExecutor::Project(child, _, _) => child.close(),
            DynamicExecutor::Limit(state) => state.child.close(),
            DynamicExecutor::Sort(state) => state.child.close(),
            DynamicExecutor::NestedLoopJoin(state) => state.left.close(),
            DynamicExecutor::GraceHashJoin(_) => Ok(()),
            DynamicExecutor::HashAggregate(state) => state.child.close(),
        }
    }
}

fn compare_values_for_sort(a: &Value, b: &Value) -> std::cmp::Ordering {
    use std::cmp::Ordering;
    match (a, b) {
        (Value::Null, Value::Null) => Ordering::Equal,
        (Value::Null, _) => Ordering::Less,
        (_, Value::Null) => Ordering::Greater,
        (Value::Int(a), Value::Int(b)) => a.cmp(b),
        (Value::Float(a), Value::Float(b)) => a.partial_cmp(b).unwrap_or(Ordering::Equal),
        (Value::Text(a), Value::Text(b)) => a.cmp(b),
        (Value::Blob(a), Value::Blob(b)) => a.cmp(b),
        _ => Ordering::Equal,
    }
}

fn hash_keys<'a>(row: &ExecutorRow<'a>, key_indices: &[usize]) -> u64 {
    use std::hash::{Hash, Hasher};
    use std::collections::hash_map::DefaultHasher;

    let mut hasher = DefaultHasher::new();
    for &idx in key_indices {
        if let Some(val) = row.get(idx) {
            match val {
                Value::Null => 0u8.hash(&mut hasher),
                Value::Int(i) => i.hash(&mut hasher),
                Value::Float(f) => f.to_bits().hash(&mut hasher),
                Value::Text(s) => s.hash(&mut hasher),
                Value::Blob(b) => b.hash(&mut hasher),
                Value::Vector(v) => {
                    for f in v.iter() {
                        f.to_bits().hash(&mut hasher);
                    }
                }
            }
        }
    }
    hasher.finish()
}

fn hash_keys_static(row: &[Value<'static>], key_indices: &[usize]) -> u64 {
    use std::hash::{Hash, Hasher};
    use std::collections::hash_map::DefaultHasher;

    let mut hasher = DefaultHasher::new();
    for &idx in key_indices {
        if let Some(val) = row.get(idx) {
            match val {
                Value::Null => 0u8.hash(&mut hasher),
                Value::Int(i) => i.hash(&mut hasher),
                Value::Float(f) => f.to_bits().hash(&mut hasher),
                Value::Text(s) => s.hash(&mut hasher),
                Value::Blob(b) => b.hash(&mut hasher),
                Value::Vector(v) => {
                    for f in v.iter() {
                        f.to_bits().hash(&mut hasher);
                    }
                }
            }
        }
    }
    hasher.finish()
}

fn keys_match_static(
    left: &[Value<'static>],
    right: &[Value<'static>],
    left_key_indices: &[usize],
    right_key_indices: &[usize],
) -> bool {
    if left_key_indices.len() != right_key_indices.len() {
        return false;
    }
    for (&li, &ri) in left_key_indices.iter().zip(right_key_indices.iter()) {
        let lv = left.get(li);
        let rv = right.get(ri);
        match (lv, rv) {
            (Some(Value::Null), _) | (_, Some(Value::Null)) => return false,
            (Some(Value::Int(a)), Some(Value::Int(b))) if a != b => return false,
            (Some(Value::Float(a)), Some(Value::Float(b))) if (a - b).abs() > f64::EPSILON => return false,
            (Some(Value::Text(a)), Some(Value::Text(b))) if a != b => return false,
            (Some(Value::Blob(a)), Some(Value::Blob(b))) if a != b => return false,
            (Some(_), Some(_)) => {}
            _ => return false,
        }
    }
    true
}

fn compute_group_key_for_dynamic(row: &ExecutorRow, group_by: &[usize]) -> Vec<u8> {
    let mut key = Vec::new();
    for &col in group_by {
        if let Some(val) = row.get(col) {
            match val {
                Value::Null => key.push(0),
                Value::Int(i) => {
                    key.push(1);
                    key.extend(i.to_be_bytes());
                }
                Value::Float(f) => {
                    key.push(2);
                    key.extend(f.to_bits().to_be_bytes());
                }
                Value::Text(s) => {
                    key.push(3);
                    key.extend(s.as_bytes());
                    key.push(0);
                }
                Value::Blob(b) => {
                    key.push(4);
                    key.extend(b.iter());
                    key.push(0);
                }
                Value::Vector(v) => {
                    key.push(5);
                    for f in v.iter() {
                        key.extend(f.to_bits().to_be_bytes());
                    }
                }
            }
        }
    }
    key
}

pub struct ExecutorBuilder<'a> {
    ctx: &'a ExecutionContext<'a>,
}

impl<'a> ExecutorBuilder<'a> {
    pub fn new(ctx: &'a ExecutionContext<'a>) -> Self {
        Self { ctx }
    }

    pub fn build_with_source<S: RowSource>(
        &self,
        plan: &crate::sql::planner::PhysicalPlan<'a>,
        source: S,
    ) -> Result<DynamicExecutor<'a, S>> {
        let column_map: Vec<(String, usize)> = plan.output_schema.columns
            .iter()
            .enumerate()
            .map(|(idx, col)| (col.name.to_string(), idx))
            .collect();

        self.build_operator(plan.root, source, &column_map)
    }

    fn build_operator<S: RowSource>(
        &self,
        op: &'a crate::sql::planner::PhysicalOperator<'a>,
        source: S,
        column_map: &[(String, usize)],
    ) -> Result<DynamicExecutor<'a, S>> {
        use crate::sql::planner::PhysicalOperator;

        match op {
            PhysicalOperator::TableScan(_) => {
                Ok(DynamicExecutor::TableScan(TableScanExecutor::new(source, self.ctx.arena)))
            }
            PhysicalOperator::FilterExec(filter) => {
                let child = self.build_operator(filter.input, source, column_map)?;
                let predicate = CompiledPredicate::new(filter.predicate, column_map.to_vec());
                Ok(DynamicExecutor::Filter(Box::new(child), predicate))
            }
            PhysicalOperator::ProjectExec(project) => {
                let child = self.build_operator(project.input, source, column_map)?;
                let projections: Vec<usize> = (0..project.expressions.len()).collect();
                Ok(DynamicExecutor::Project(Box::new(child), projections, self.ctx.arena))
            }
            PhysicalOperator::LimitExec(limit) => {
                let child = self.build_operator(limit.input, source, column_map)?;
                Ok(DynamicExecutor::Limit(LimitState {
                    child: Box::new(child),
                    limit: limit.limit,
                    offset: limit.offset,
                    skipped: 0,
                    returned: 0,
                }))
            }
            PhysicalOperator::SortExec(sort) => {
                let child = self.build_operator(sort.input, source, column_map)?;
                let sort_keys: Vec<SortKey> = sort.order_by.iter()
                    .enumerate()
                    .map(|(idx, key)| SortKey { column: idx, ascending: key.ascending })
                    .collect();
                Ok(DynamicExecutor::Sort(SortState {
                    child: Box::new(child),
                    sort_keys,
                    arena: self.ctx.arena,
                    rows: Vec::new(),
                    iter_idx: 0,
                    sorted: false,
                }))
            }
            PhysicalOperator::IndexScan(_) => {
                eyre::bail!("IndexScan requires explicit BTreeCursorAdapter - use build_index_scan instead")
            }
            PhysicalOperator::HashAggregate(agg) => {
                let child = self.build_operator(agg.input, source, column_map)?;
                let group_by_indices: Vec<usize> = agg.group_by
                    .iter()
                    .filter_map(|expr| {
                        if let crate::sql::ast::Expr::Column(col) = expr {
                            column_map.iter().find(|(n, _)| n.eq_ignore_ascii_case(col.column)).map(|(_, i)| *i)
                        } else {
                            None
                        }
                    })
                    .collect();
                let agg_funcs: Vec<AggregateFunction> = agg.aggregates
                    .iter()
                    .map(|agg_expr| {
                        let column_idx = agg_expr.argument.and_then(|arg| {
                            if let crate::sql::ast::Expr::Column(col) = arg {
                                column_map.iter().find(|(n, _)| n.eq_ignore_ascii_case(col.column)).map(|(_, i)| *i)
                            } else {
                                None
                            }
                        }).unwrap_or(0);
                        match agg_expr.function {
                            crate::sql::planner::AggregateFunction::Count => AggregateFunction::Count { distinct: agg_expr.distinct },
                            crate::sql::planner::AggregateFunction::Sum => AggregateFunction::Sum { column: column_idx },
                            crate::sql::planner::AggregateFunction::Avg => AggregateFunction::Avg { column: column_idx },
                            crate::sql::planner::AggregateFunction::Min => AggregateFunction::Min { column: column_idx },
                            crate::sql::planner::AggregateFunction::Max => AggregateFunction::Max { column: column_idx },
                        }
                    })
                    .collect();
                Ok(DynamicExecutor::HashAggregate(HashAggregateState {
                    child: Box::new(child),
                    group_by: group_by_indices,
                    aggregates: agg_funcs,
                    arena: self.ctx.arena,
                    groups: hashbrown::HashMap::new(),
                    result_iter: None,
                    computed: false,
                }))
            }
            PhysicalOperator::SortedAggregate(agg) => {
                let child = self.build_operator(agg.input, source, column_map)?;
                let group_by_indices: Vec<usize> = agg.group_by
                    .iter()
                    .filter_map(|expr| {
                        if let crate::sql::ast::Expr::Column(col) = expr {
                            column_map.iter().find(|(n, _)| n.eq_ignore_ascii_case(col.column)).map(|(_, i)| *i)
                        } else {
                            None
                        }
                    })
                    .collect();
                let agg_funcs: Vec<AggregateFunction> = agg.aggregates
                    .iter()
                    .map(|agg_expr| {
                        let column_idx = agg_expr.argument.and_then(|arg| {
                            if let crate::sql::ast::Expr::Column(col) = arg {
                                column_map.iter().find(|(n, _)| n.eq_ignore_ascii_case(col.column)).map(|(_, i)| *i)
                            } else {
                                None
                            }
                        }).unwrap_or(0);
                        match agg_expr.function {
                            crate::sql::planner::AggregateFunction::Count => AggregateFunction::Count { distinct: agg_expr.distinct },
                            crate::sql::planner::AggregateFunction::Sum => AggregateFunction::Sum { column: column_idx },
                            crate::sql::planner::AggregateFunction::Avg => AggregateFunction::Avg { column: column_idx },
                            crate::sql::planner::AggregateFunction::Min => AggregateFunction::Min { column: column_idx },
                            crate::sql::planner::AggregateFunction::Max => AggregateFunction::Max { column: column_idx },
                        }
                    })
                    .collect();
                Ok(DynamicExecutor::HashAggregate(HashAggregateState {
                    child: Box::new(child),
                    group_by: group_by_indices,
                    aggregates: agg_funcs,
                    arena: self.ctx.arena,
                    groups: hashbrown::HashMap::new(),
                    result_iter: None,
                    computed: false,
                }))
            }
            PhysicalOperator::NestedLoopJoin(_) => {
                eyre::bail!("NestedLoopJoin requires two sources - use build_nested_loop_join instead")
            }
            PhysicalOperator::GraceHashJoin(_) => {
                eyre::bail!("GraceHashJoin requires two sources - use build_grace_hash_join instead")
            }
        }
    }

    pub fn build_index_scan(
        &self,
        index_scan: &'a crate::sql::planner::PhysicalIndexScan<'a>,
        adapter: BTreeCursorAdapter,
        column_map: &[(String, usize)],
    ) -> Result<IndexScanState<'a>> {
        let residual_filter = index_scan.residual_filter.map(|expr| {
            CompiledPredicate::new(expr, column_map.to_vec())
        });

        Ok(IndexScanState::new(adapter, self.ctx.arena, residual_filter))
    }

    pub fn build_nested_loop_join<S: RowSource>(
        &self,
        left: DynamicExecutor<'a, S>,
        right: DynamicExecutor<'a, S>,
        condition: Option<&'a crate::sql::ast::Expr<'a>>,
        column_map: &[(String, usize)],
    ) -> NestedLoopJoinState<'a, S> {
        let compiled_condition = condition.map(|expr| {
            CompiledPredicate::new(expr, column_map.to_vec())
        });
        NestedLoopJoinState {
            left: Box::new(left),
            right: Box::new(right),
            condition: compiled_condition,
            arena: self.ctx.arena,
            current_left_row: None,
            right_rows: Vec::new(),
            right_index: 0,
            materialized: false,
        }
    }

    pub fn build_grace_hash_join<S: RowSource>(
        &self,
        left: DynamicExecutor<'a, S>,
        right: DynamicExecutor<'a, S>,
        left_key_indices: Vec<usize>,
        right_key_indices: Vec<usize>,
        num_partitions: usize,
    ) -> GraceHashJoinState<'a, S> {
        GraceHashJoinState {
            left: Box::new(left),
            right: Box::new(right),
            left_key_indices,
            right_key_indices,
            arena: self.ctx.arena,
            num_partitions,
            left_partitions: (0..num_partitions).map(|_| Vec::new()).collect(),
            right_partitions: (0..num_partitions).map(|_| Vec::new()).collect(),
            current_partition: 0,
            partition_hash_table: hashbrown::HashMap::new(),
            partition_build_rows: Vec::new(),
            current_probe_idx: 0,
            current_match_idx: 0,
            current_matches: Vec::new(),
            partitioned: false,
        }
    }

    pub fn build_hash_aggregate<S: RowSource>(
        &self,
        child: DynamicExecutor<'a, S>,
        group_by: Vec<usize>,
        aggregates: Vec<AggregateFunction>,
    ) -> HashAggregateState<'a, S> {
        HashAggregateState {
            child: Box::new(child),
            group_by,
            aggregates,
            arena: self.ctx.arena,
            groups: hashbrown::HashMap::new(),
            result_iter: None,
            computed: false,
        }
    }
}

impl ExprEvaluator {
    pub fn new(column_map: &[(String, usize)]) -> Self {
        Self {
            column_map: column_map.to_vec(),
        }
    }

    pub fn eval_column<'a>(&self, row: &ExecutorRow<'a>, column_idx: usize) -> Option<Value<'a>> {
        row.get(column_idx).cloned()
    }

    pub fn resolve_column(&self, name: &str) -> Option<usize> {
        self.column_map
            .iter()
            .find(|(n, _)| n.eq_ignore_ascii_case(name))
            .map(|(_, idx)| *idx)
    }

    fn compare_values<'a>(a: &Value<'a>, b: &Value<'a>) -> Option<std::cmp::Ordering> {
        use std::cmp::Ordering;
        match (a, b) {
            (Value::Null, Value::Null) => Some(Ordering::Equal),
            (Value::Null, _) | (_, Value::Null) => None,
            (Value::Int(a), Value::Int(b)) => Some(a.cmp(b)),
            (Value::Int(a), Value::Float(b)) => (*a as f64).partial_cmp(b),
            (Value::Float(a), Value::Int(b)) => a.partial_cmp(&(*b as f64)),
            (Value::Float(a), Value::Float(b)) => a.partial_cmp(b),
            (Value::Text(a), Value::Text(b)) => Some(a.cmp(b)),
            (Value::Blob(a), Value::Blob(b)) => Some(a.cmp(b)),
            _ => None,
        }
    }

    pub fn eval_eq<'a>(&self, row: &ExecutorRow<'a>, column_idx: usize, value: &Value<'a>) -> bool {
        match row.get(column_idx) {
            Some(col_val) => Self::compare_values(col_val, value) == Some(std::cmp::Ordering::Equal),
            None => false,
        }
    }

    pub fn eval_neq<'a>(
        &self,
        row: &ExecutorRow<'a>,
        column_idx: usize,
        value: &Value<'a>,
    ) -> bool {
        match row.get(column_idx) {
            Some(col_val) => Self::compare_values(col_val, value) != Some(std::cmp::Ordering::Equal),
            None => true,
        }
    }

    pub fn eval_gt<'a>(&self, row: &ExecutorRow<'a>, column_idx: usize, value: &Value<'a>) -> bool {
        match row.get(column_idx) {
            Some(col_val) => {
                Self::compare_values(col_val, value) == Some(std::cmp::Ordering::Greater)
            }
            None => false,
        }
    }

    pub fn eval_lt<'a>(&self, row: &ExecutorRow<'a>, column_idx: usize, value: &Value<'a>) -> bool {
        match row.get(column_idx) {
            Some(col_val) => Self::compare_values(col_val, value) == Some(std::cmp::Ordering::Less),
            None => false,
        }
    }

    pub fn eval_gte<'a>(
        &self,
        row: &ExecutorRow<'a>,
        column_idx: usize,
        value: &Value<'a>,
    ) -> bool {
        match row.get(column_idx) {
            Some(col_val) => matches!(
                Self::compare_values(col_val, value),
                Some(std::cmp::Ordering::Greater) | Some(std::cmp::Ordering::Equal)
            ),
            None => false,
        }
    }

    pub fn eval_lte<'a>(
        &self,
        row: &ExecutorRow<'a>,
        column_idx: usize,
        value: &Value<'a>,
    ) -> bool {
        match row.get(column_idx) {
            Some(col_val) => matches!(
                Self::compare_values(col_val, value),
                Some(std::cmp::Ordering::Less) | Some(std::cmp::Ordering::Equal)
            ),
            None => false,
        }
    }

    pub fn eval_is_null<'a>(&self, row: &ExecutorRow<'a>, column_idx: usize) -> bool {
        matches!(row.get(column_idx), Some(Value::Null) | None)
    }

    pub fn eval_is_not_null<'a>(&self, row: &ExecutorRow<'a>, column_idx: usize) -> bool {
        matches!(row.get(column_idx), Some(v) if !matches!(v, Value::Null))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::borrow::Cow;

    struct MockRowSource {
        rows: Vec<Vec<Value<'static>>>,
        current: usize,
    }

    impl MockRowSource {
        fn new(rows: Vec<Vec<Value<'static>>>) -> Self {
            Self { rows, current: 0 }
        }
    }

    impl RowSource for MockRowSource {
        fn reset(&mut self) -> Result<()> {
            self.current = 0;
            Ok(())
        }

        fn next_row(&mut self) -> Result<Option<Vec<Value<'static>>>> {
            if self.current < self.rows.len() {
                let row = self.rows[self.current].clone();
                self.current += 1;
                Ok(Some(row))
            } else {
                Ok(None)
            }
        }
    }

    struct MockExecutor {
        opened: bool,
        closed: bool,
        row_count: usize,
        current: usize,
    }

    impl MockExecutor {
        fn new(row_count: usize) -> Self {
            Self {
                opened: false,
                closed: false,
                row_count,
                current: 0,
            }
        }
    }

    impl<'a> Executor<'a> for MockExecutor {
        fn open(&mut self) -> Result<()> {
            self.opened = true;
            self.current = 0;
            Ok(())
        }

        fn next(&mut self) -> Result<Option<ExecutorRow<'a>>> {
            if self.current < self.row_count {
                self.current += 1;
                Ok(Some(ExecutorRow { values: &[] }))
            } else {
                Ok(None)
            }
        }

        fn close(&mut self) -> Result<()> {
            self.closed = true;
            Ok(())
        }
    }

    #[test]
    fn executor_trait_lifecycle() {
        let mut exec = MockExecutor::new(3);
        assert!(!exec.opened);
        assert!(!exec.closed);

        exec.open().unwrap();
        assert!(exec.opened);

        let mut count = 0;
        while exec.next().unwrap().is_some() {
            count += 1;
        }
        assert_eq!(count, 3);

        exec.close().unwrap();
        assert!(exec.closed);
    }

    #[test]
    fn executor_row_provides_value_access() {
        let arena = Bump::new();
        let values: &[Value] = arena.alloc_slice_fill_iter([
            Value::Int(42),
            Value::Text(Cow::Borrowed("hello")),
            Value::Null,
        ]);
        let row = ExecutorRow::new(values);

        assert_eq!(row.column_count(), 3);
        assert!(matches!(row.get(0), Some(Value::Int(42))));
        assert!(matches!(row.get(1), Some(Value::Text(_))));
        assert!(matches!(row.get(2), Some(Value::Null)));
        assert!(row.get(3).is_none());
    }

    #[test]
    fn execution_context_holds_arena() {
        let arena = Bump::new();
        let ctx = ExecutionContext::new(&arena);
        let _allocated: &str = ctx.arena.alloc_str("test");
    }

    #[test]
    fn table_scan_executor_iterates_rows() {
        let arena = Bump::new();

        let rows_data: Vec<Vec<Value>> = vec![
            vec![Value::Int(1), Value::Text(Cow::Owned("Alice".to_string()))],
            vec![Value::Int(2), Value::Text(Cow::Owned("Bob".to_string()))],
            vec![Value::Int(3), Value::Text(Cow::Owned("Carol".to_string()))],
        ];

        let source = MockRowSource::new(rows_data);
        let mut executor = TableScanExecutor::new(source, &arena);

        executor.open().unwrap();

        let mut count = 0;
        while let Some(row) = executor.next().unwrap() {
            count += 1;
            assert_eq!(row.column_count(), 2);
        }

        assert_eq!(count, 3);
        executor.close().unwrap();
    }

    #[test]
    fn table_scan_executor_returns_correct_values() {
        let arena = Bump::new();

        let rows_data: Vec<Vec<Value>> = vec![
            vec![Value::Int(42), Value::Text(Cow::Owned("test".to_string()))],
        ];

        let source = MockRowSource::new(rows_data);
        let mut executor = TableScanExecutor::new(source, &arena);

        executor.open().unwrap();

        let row = executor.next().unwrap().unwrap();
        assert!(matches!(row.get(0), Some(Value::Int(42))));
        assert!(matches!(row.get(1), Some(Value::Text(text)) if text == "test"));

        assert!(executor.next().unwrap().is_none());
        executor.close().unwrap();
    }

    #[test]
    fn filter_executor_filters_rows() {
        let arena = Bump::new();

        let rows_data: Vec<Vec<Value>> = vec![
            vec![Value::Int(1), Value::Text(Cow::Owned("Alice".to_string()))],
            vec![Value::Int(2), Value::Text(Cow::Owned("Bob".to_string()))],
            vec![Value::Int(3), Value::Text(Cow::Owned("Carol".to_string()))],
            vec![Value::Int(4), Value::Text(Cow::Owned("Dave".to_string()))],
        ];

        let source = MockRowSource::new(rows_data);
        let scan = TableScanExecutor::new(source, &arena);
        let predicate = |row: &ExecutorRow| -> bool {
            matches!(row.get(0), Some(Value::Int(i)) if *i > 2)
        };
        let mut filter = FilterExecutor::new(scan, predicate);

        filter.open().unwrap();

        let mut results = Vec::new();
        while let Some(row) = filter.next().unwrap() {
            if let Some(&Value::Int(id)) = row.get(0) {
                results.push(id);
            }
        }

        assert_eq!(results, vec![3, 4]);
        filter.close().unwrap();
    }

    #[test]
    fn filter_executor_handles_empty_result() {
        let arena = Bump::new();

        let rows_data: Vec<Vec<Value>> = vec![
            vec![Value::Int(1)],
            vec![Value::Int(2)],
        ];

        let source = MockRowSource::new(rows_data);
        let scan = TableScanExecutor::new(source, &arena);
        let predicate = |row: &ExecutorRow| -> bool {
            matches!(row.get(0), Some(Value::Int(i)) if *i > 100)
        };
        let mut filter = FilterExecutor::new(scan, predicate);

        filter.open().unwrap();

        assert!(filter.next().unwrap().is_none());
        filter.close().unwrap();
    }

    #[test]
    fn project_executor_selects_columns() {
        let arena = Bump::new();

        let rows_data: Vec<Vec<Value>> = vec![
            vec![
                Value::Int(1),
                Value::Text(Cow::Owned("Alice".to_string())),
                Value::Int(25),
            ],
            vec![
                Value::Int(2),
                Value::Text(Cow::Owned("Bob".to_string())),
                Value::Int(30),
            ],
        ];

        let source = MockRowSource::new(rows_data);
        let scan = TableScanExecutor::new(source, &arena);
        let projections = vec![0, 2];
        let mut project = ProjectExecutor::new(scan, projections, &arena);

        project.open().unwrap();

        let row1 = project.next().unwrap().unwrap();
        assert_eq!(row1.column_count(), 2);
        assert!(matches!(row1.get(0), Some(&Value::Int(1))));
        assert!(matches!(row1.get(1), Some(&Value::Int(25))));

        let row2 = project.next().unwrap().unwrap();
        assert!(matches!(row2.get(0), Some(&Value::Int(2))));
        assert!(matches!(row2.get(1), Some(&Value::Int(30))));

        assert!(project.next().unwrap().is_none());
        project.close().unwrap();
    }

    #[test]
    fn project_executor_reorders_columns() {
        let arena = Bump::new();

        let rows_data: Vec<Vec<Value>> = vec![vec![
            Value::Int(1),
            Value::Text(Cow::Owned("test".to_string())),
        ]];

        let source = MockRowSource::new(rows_data);
        let scan = TableScanExecutor::new(source, &arena);
        let projections = vec![1, 0];
        let mut project = ProjectExecutor::new(scan, projections, &arena);

        project.open().unwrap();

        let row = project.next().unwrap().unwrap();
        assert_eq!(row.column_count(), 2);
        assert!(matches!(row.get(0), Some(Value::Text(_))));
        assert!(matches!(row.get(1), Some(&Value::Int(1))));

        project.close().unwrap();
    }

    #[test]
    fn limit_executor_limits_rows() {
        let arena = Bump::new();

        let rows_data: Vec<Vec<Value>> = vec![
            vec![Value::Int(1)],
            vec![Value::Int(2)],
            vec![Value::Int(3)],
            vec![Value::Int(4)],
            vec![Value::Int(5)],
        ];

        let source = MockRowSource::new(rows_data);
        let scan = TableScanExecutor::new(source, &arena);
        let mut limit = LimitExecutor::new(scan, Some(3), None);

        limit.open().unwrap();

        let mut results = Vec::new();
        while let Some(row) = limit.next().unwrap() {
            if let Some(&Value::Int(id)) = row.get(0) {
                results.push(id);
            }
        }

        assert_eq!(results, vec![1, 2, 3]);
        limit.close().unwrap();
    }

    #[test]
    fn limit_executor_handles_offset() {
        let arena = Bump::new();

        let rows_data: Vec<Vec<Value>> = vec![
            vec![Value::Int(1)],
            vec![Value::Int(2)],
            vec![Value::Int(3)],
            vec![Value::Int(4)],
            vec![Value::Int(5)],
        ];

        let source = MockRowSource::new(rows_data);
        let scan = TableScanExecutor::new(source, &arena);
        let mut limit = LimitExecutor::new(scan, Some(2), Some(2));

        limit.open().unwrap();

        let mut results = Vec::new();
        while let Some(row) = limit.next().unwrap() {
            if let Some(&Value::Int(id)) = row.get(0) {
                results.push(id);
            }
        }

        assert_eq!(results, vec![3, 4]);
        limit.close().unwrap();
    }

    #[test]
    fn limit_executor_with_offset_only() {
        let arena = Bump::new();

        let rows_data: Vec<Vec<Value>> = vec![
            vec![Value::Int(1)],
            vec![Value::Int(2)],
            vec![Value::Int(3)],
        ];

        let source = MockRowSource::new(rows_data);
        let scan = TableScanExecutor::new(source, &arena);
        let mut limit = LimitExecutor::new(scan, None, Some(1));

        limit.open().unwrap();

        let mut results = Vec::new();
        while let Some(row) = limit.next().unwrap() {
            if let Some(&Value::Int(id)) = row.get(0) {
                results.push(id);
            }
        }

        assert_eq!(results, vec![2, 3]);
        limit.close().unwrap();
    }

    #[test]
    fn limit_executor_exhausts_early_if_less_than_limit() {
        let arena = Bump::new();

        let rows_data: Vec<Vec<Value>> = vec![vec![Value::Int(1)], vec![Value::Int(2)]];

        let source = MockRowSource::new(rows_data);
        let scan = TableScanExecutor::new(source, &arena);
        let mut limit = LimitExecutor::new(scan, Some(10), None);

        limit.open().unwrap();

        let mut count = 0;
        while limit.next().unwrap().is_some() {
            count += 1;
        }

        assert_eq!(count, 2);
        limit.close().unwrap();
    }

    #[test]
    fn nested_loop_join_cross_join() {
        let arena = Bump::new();

        let left_rows: Vec<Vec<Value>> = vec![vec![Value::Int(1)], vec![Value::Int(2)]];

        let right_rows: Vec<Vec<Value>> = vec![
            vec![Value::Text(Cow::Owned("A".to_string()))],
            vec![Value::Text(Cow::Owned("B".to_string()))],
        ];

        let left = MockRowSource::new(left_rows);
        let right = MockRowSource::new(right_rows);
        let left_scan = TableScanExecutor::new(left, &arena);
        let right_scan = TableScanExecutor::new(right, &arena);

        let condition = |_left: &ExecutorRow, _right: &ExecutorRow| true;
        let mut join = NestedLoopJoinExecutor::new(left_scan, right_scan, condition, &arena);

        join.open().unwrap();

        let mut results = Vec::new();
        while let Some(row) = join.next().unwrap() {
            let id = match row.get(0) {
                Some(&Value::Int(i)) => i,
                _ => 0,
            };
            results.push(id);
        }

        assert_eq!(results.len(), 4);
        assert_eq!(results, vec![1, 1, 2, 2]);
        join.close().unwrap();
    }

    #[test]
    fn nested_loop_join_with_condition() {
        let arena = Bump::new();

        let left_rows: Vec<Vec<Value>> = vec![
            vec![Value::Int(1), Value::Int(10)],
            vec![Value::Int(2), Value::Int(20)],
            vec![Value::Int(3), Value::Int(30)],
        ];

        let right_rows: Vec<Vec<Value>> = vec![
            vec![Value::Int(1), Value::Text(Cow::Owned("Alice".to_string()))],
            vec![Value::Int(2), Value::Text(Cow::Owned("Bob".to_string()))],
        ];

        let left = MockRowSource::new(left_rows);
        let right = MockRowSource::new(right_rows);
        let left_scan = TableScanExecutor::new(left, &arena);
        let right_scan = TableScanExecutor::new(right, &arena);

        let condition = |left: &ExecutorRow, right: &ExecutorRow| {
            matches!((left.get(0), right.get(0)), (Some(&Value::Int(l)), Some(&Value::Int(r))) if l == r)
        };
        let mut join = NestedLoopJoinExecutor::new(left_scan, right_scan, condition, &arena);

        join.open().unwrap();

        let mut results = Vec::new();
        while let Some(row) = join.next().unwrap() {
            let id = match row.get(0) {
                Some(&Value::Int(i)) => i,
                _ => 0,
            };
            results.push(id);
        }

        assert_eq!(results, vec![1, 2]);
        join.close().unwrap();
    }

    #[test]
    fn nested_loop_join_combines_columns() {
        let arena = Bump::new();

        let left_rows: Vec<Vec<Value>> = vec![vec![Value::Int(1), Value::Int(10)]];

        let right_rows: Vec<Vec<Value>> = vec![vec![Value::Int(1), Value::Int(100)]];

        let left = MockRowSource::new(left_rows);
        let right = MockRowSource::new(right_rows);
        let left_scan = TableScanExecutor::new(left, &arena);
        let right_scan = TableScanExecutor::new(right, &arena);

        let condition = |_left: &ExecutorRow, _right: &ExecutorRow| true;
        let mut join = NestedLoopJoinExecutor::new(left_scan, right_scan, condition, &arena);

        join.open().unwrap();

        let row = join.next().unwrap().unwrap();
        assert_eq!(row.column_count(), 4);
        assert!(matches!(row.get(0), Some(&Value::Int(1))));
        assert!(matches!(row.get(1), Some(&Value::Int(10))));
        assert!(matches!(row.get(2), Some(&Value::Int(1))));
        assert!(matches!(row.get(3), Some(&Value::Int(100))));

        join.close().unwrap();
    }

    #[test]
    fn hash_aggregate_count_star() {
        let arena = Bump::new();

        let rows_data: Vec<Vec<Value>> = vec![
            vec![Value::Int(1)],
            vec![Value::Int(2)],
            vec![Value::Int(3)],
        ];

        let source = MockRowSource::new(rows_data);
        let scan = TableScanExecutor::new(source, &arena);
        let aggregates = vec![AggregateFunction::Count { distinct: false }];
        let mut agg = HashAggregateExecutor::new(scan, vec![], aggregates, &arena);

        agg.open().unwrap();

        let row = agg.next().unwrap().unwrap();
        assert_eq!(row.column_count(), 1);
        assert!(matches!(row.get(0), Some(&Value::Int(3))));

        assert!(agg.next().unwrap().is_none());
        agg.close().unwrap();
    }

    #[test]
    fn hash_aggregate_sum() {
        let arena = Bump::new();

        let rows_data: Vec<Vec<Value>> = vec![
            vec![Value::Int(10)],
            vec![Value::Int(20)],
            vec![Value::Int(30)],
        ];

        let source = MockRowSource::new(rows_data);
        let scan = TableScanExecutor::new(source, &arena);
        let aggregates = vec![AggregateFunction::Sum { column: 0 }];
        let mut agg = HashAggregateExecutor::new(scan, vec![], aggregates, &arena);

        agg.open().unwrap();

        let row = agg.next().unwrap().unwrap();
        assert!(matches!(row.get(0), Some(&Value::Int(60))));

        agg.close().unwrap();
    }

    #[test]
    fn hash_aggregate_with_group_by() {
        let arena = Bump::new();

        let rows_data: Vec<Vec<Value>> = vec![
            vec![Value::Text(Cow::Owned("A".to_string())), Value::Int(10)],
            vec![Value::Text(Cow::Owned("B".to_string())), Value::Int(20)],
            vec![Value::Text(Cow::Owned("A".to_string())), Value::Int(30)],
            vec![Value::Text(Cow::Owned("B".to_string())), Value::Int(40)],
        ];

        let source = MockRowSource::new(rows_data);
        let scan = TableScanExecutor::new(source, &arena);
        let group_by = vec![0];
        let aggregates = vec![AggregateFunction::Sum { column: 1 }];
        let mut agg = HashAggregateExecutor::new(scan, group_by, aggregates, &arena);

        agg.open().unwrap();

        let mut results: Vec<(String, i64)> = Vec::new();
        while let Some(row) = agg.next().unwrap() {
            let group = match row.get(0) {
                Some(Value::Text(s)) => s.to_string(),
                _ => "?".to_string(),
            };
            let sum = match row.get(1) {
                Some(&Value::Int(i)) => i,
                _ => 0,
            };
            results.push((group, sum));
        }
        results.sort();

        assert_eq!(results.len(), 2);
        assert_eq!(results[0], ("A".to_string(), 40));
        assert_eq!(results[1], ("B".to_string(), 60));

        agg.close().unwrap();
    }

    #[test]
    fn hash_aggregate_min_max() {
        let arena = Bump::new();

        let rows_data: Vec<Vec<Value>> = vec![
            vec![Value::Int(5)],
            vec![Value::Int(2)],
            vec![Value::Int(8)],
            vec![Value::Int(1)],
        ];

        let source = MockRowSource::new(rows_data);
        let scan = TableScanExecutor::new(source, &arena);
        let aggregates = vec![
            AggregateFunction::Min { column: 0 },
            AggregateFunction::Max { column: 0 },
        ];
        let mut agg = HashAggregateExecutor::new(scan, vec![], aggregates, &arena);

        agg.open().unwrap();

        let row = agg.next().unwrap().unwrap();
        assert_eq!(row.column_count(), 2);
        assert!(matches!(row.get(0), Some(&Value::Int(1))));
        assert!(matches!(row.get(1), Some(&Value::Int(8))));

        agg.close().unwrap();
    }

    #[test]
    fn sort_executor_ascending() {
        let arena = Bump::new();

        let rows_data: Vec<Vec<Value>> = vec![
            vec![Value::Int(3)],
            vec![Value::Int(1)],
            vec![Value::Int(4)],
            vec![Value::Int(2)],
        ];

        let source = MockRowSource::new(rows_data);
        let scan = TableScanExecutor::new(source, &arena);
        let sort_keys = vec![SortKey {
            column: 0,
            ascending: true,
        }];
        let mut sort = SortExecutor::new(scan, sort_keys, &arena);

        sort.open().unwrap();

        let mut results = Vec::new();
        while let Some(row) = sort.next().unwrap() {
            if let Some(&Value::Int(i)) = row.get(0) {
                results.push(i);
            }
        }

        assert_eq!(results, vec![1, 2, 3, 4]);
        sort.close().unwrap();
    }

    #[test]
    fn sort_executor_descending() {
        let arena = Bump::new();

        let rows_data: Vec<Vec<Value>> = vec![
            vec![Value::Int(3)],
            vec![Value::Int(1)],
            vec![Value::Int(4)],
            vec![Value::Int(2)],
        ];

        let source = MockRowSource::new(rows_data);
        let scan = TableScanExecutor::new(source, &arena);
        let sort_keys = vec![SortKey {
            column: 0,
            ascending: false,
        }];
        let mut sort = SortExecutor::new(scan, sort_keys, &arena);

        sort.open().unwrap();

        let mut results = Vec::new();
        while let Some(row) = sort.next().unwrap() {
            if let Some(&Value::Int(i)) = row.get(0) {
                results.push(i);
            }
        }

        assert_eq!(results, vec![4, 3, 2, 1]);
        sort.close().unwrap();
    }

    #[test]
    fn sort_executor_multiple_keys() {
        let arena = Bump::new();

        let rows_data: Vec<Vec<Value>> = vec![
            vec![Value::Int(1), Value::Int(2)],
            vec![Value::Int(1), Value::Int(1)],
            vec![Value::Int(2), Value::Int(1)],
            vec![Value::Int(1), Value::Int(3)],
        ];

        let source = MockRowSource::new(rows_data);
        let scan = TableScanExecutor::new(source, &arena);
        let sort_keys = vec![
            SortKey {
                column: 0,
                ascending: true,
            },
            SortKey {
                column: 1,
                ascending: true,
            },
        ];
        let mut sort = SortExecutor::new(scan, sort_keys, &arena);

        sort.open().unwrap();

        let mut results = Vec::new();
        while let Some(row) = sort.next().unwrap() {
            let c0 = match row.get(0) {
                Some(&Value::Int(i)) => i,
                _ => 0,
            };
            let c1 = match row.get(1) {
                Some(&Value::Int(i)) => i,
                _ => 0,
            };
            results.push((c0, c1));
        }

        assert_eq!(results, vec![(1, 1), (1, 2), (1, 3), (2, 1)]);
        sort.close().unwrap();
    }

    #[test]
    fn expr_evaluator_column_reference() {
        let arena = Bump::new();
        let values: &[Value] = arena.alloc_slice_fill_iter([Value::Int(42), Value::Int(100)]);
        let row = ExecutorRow::new(values);

        let column_map = vec![("id".to_string(), 0), ("age".to_string(), 1)];
        let evaluator = ExprEvaluator::new(&column_map);

        let result = evaluator.eval_column(&row, 0);
        assert!(matches!(result, Some(Value::Int(42))));
    }

    #[test]
    fn expr_evaluator_equality() {
        let arena = Bump::new();
        let values: &[Value] = arena.alloc_slice_fill_iter([Value::Int(42)]);
        let row = ExecutorRow::new(values);

        let column_map = vec![("id".to_string(), 0)];
        let evaluator = ExprEvaluator::new(&column_map);

        assert!(evaluator.eval_eq(&row, 0, &Value::Int(42)));
        assert!(!evaluator.eval_eq(&row, 0, &Value::Int(99)));
    }

    #[test]
    fn expr_evaluator_comparison_operators() {
        let arena = Bump::new();
        let values: &[Value] = arena.alloc_slice_fill_iter([Value::Int(50)]);
        let row = ExecutorRow::new(values);

        let column_map = vec![("age".to_string(), 0)];
        let evaluator = ExprEvaluator::new(&column_map);

        assert!(evaluator.eval_gt(&row, 0, &Value::Int(40)));
        assert!(!evaluator.eval_gt(&row, 0, &Value::Int(60)));

        assert!(evaluator.eval_lt(&row, 0, &Value::Int(60)));
        assert!(!evaluator.eval_lt(&row, 0, &Value::Int(40)));

        assert!(evaluator.eval_gte(&row, 0, &Value::Int(50)));
        assert!(evaluator.eval_gte(&row, 0, &Value::Int(40)));
        assert!(!evaluator.eval_gte(&row, 0, &Value::Int(60)));

        assert!(evaluator.eval_lte(&row, 0, &Value::Int(50)));
        assert!(evaluator.eval_lte(&row, 0, &Value::Int(60)));
        assert!(!evaluator.eval_lte(&row, 0, &Value::Int(40)));
    }

    #[test]
    fn expr_evaluator_logical_operators() {
        let arena = Bump::new();
        let values: &[Value] = arena.alloc_slice_fill_iter([Value::Int(50), Value::Int(100)]);
        let row = ExecutorRow::new(values);

        let column_map = vec![("age".to_string(), 0), ("score".to_string(), 1)];
        let evaluator = ExprEvaluator::new(&column_map);

        let cond_a = evaluator.eval_gt(&row, 0, &Value::Int(40));
        let cond_b = evaluator.eval_lt(&row, 1, &Value::Int(200));
        assert!(cond_a && cond_b);

        let cond_c = evaluator.eval_lt(&row, 0, &Value::Int(30));
        assert!(cond_a || cond_c);
        assert!(!cond_c);
    }

    #[test]
    fn executor_builder_creates_table_scan() {
        use crate::sql::planner::{PhysicalOperator, PhysicalPlan, PhysicalTableScan, OutputSchema, OutputColumn};
        use crate::records::types::DataType;

        let arena = Bump::new();

        let table_scan = PhysicalTableScan {
            schema: None,
            table: "users",
            alias: None,
            post_scan_filter: None,
            table_def: None,
        };
        let op = arena.alloc(PhysicalOperator::TableScan(table_scan));

        let plan = PhysicalPlan {
            root: op,
            output_schema: OutputSchema {
                columns: arena.alloc_slice_fill_iter([
                    OutputColumn { name: "id", data_type: DataType::Int8, nullable: false },
                    OutputColumn { name: "name", data_type: DataType::Text, nullable: false },
                ]),
            },
        };

        let rows_data: Vec<Vec<Value<'static>>> = vec![
            vec![Value::Int(1), Value::Text(Cow::Owned("Alice".to_string()))],
            vec![Value::Int(2), Value::Text(Cow::Owned("Bob".to_string()))],
        ];
        let source = MockRowSource::new(rows_data);

        let ctx = ExecutionContext::new(&arena);
        let builder = ExecutorBuilder::new(&ctx);
        let mut executor = builder.build_with_source(&plan, source).unwrap();

        executor.open().unwrap();

        let mut count = 0;
        while executor.next().unwrap().is_some() {
            count += 1;
        }

        assert_eq!(count, 2);
        executor.close().unwrap();
    }

    #[test]
    fn executor_builder_creates_filter() {
        use crate::sql::planner::{PhysicalOperator, PhysicalPlan, PhysicalTableScan, PhysicalFilterExec, OutputSchema, OutputColumn};
        use crate::sql::ast::{Expr, BinaryOperator, Literal, ColumnRef};
        use crate::records::types::DataType;

        let arena = Bump::new();

        let table_scan = PhysicalTableScan {
            schema: None,
            table: "users",
            alias: None,
            post_scan_filter: None,
            table_def: None,
        };
        let scan_op = arena.alloc(PhysicalOperator::TableScan(table_scan));

        let predicate = arena.alloc(Expr::BinaryOp {
            left: arena.alloc(Expr::Column(ColumnRef {
                schema: None,
                table: None,
                column: "id",
            })),
            op: BinaryOperator::Gt,
            right: arena.alloc(Expr::Literal(Literal::Integer("1"))),
        });

        let filter = PhysicalFilterExec {
            input: scan_op,
            predicate,
        };
        let filter_op = arena.alloc(PhysicalOperator::FilterExec(filter));

        let plan = PhysicalPlan {
            root: filter_op,
            output_schema: OutputSchema {
                columns: arena.alloc_slice_fill_iter([
                    OutputColumn { name: "id", data_type: DataType::Int8, nullable: false },
                    OutputColumn { name: "name", data_type: DataType::Text, nullable: false },
                ]),
            },
        };

        let rows_data: Vec<Vec<Value<'static>>> = vec![
            vec![Value::Int(1), Value::Text(Cow::Owned("Alice".to_string()))],
            vec![Value::Int(2), Value::Text(Cow::Owned("Bob".to_string()))],
            vec![Value::Int(3), Value::Text(Cow::Owned("Carol".to_string()))],
        ];
        let source = MockRowSource::new(rows_data);

        let ctx = ExecutionContext::new(&arena);
        let builder = ExecutorBuilder::new(&ctx);
        let mut executor = builder.build_with_source(&plan, source).unwrap();

        executor.open().unwrap();

        let mut results = Vec::new();
        while let Some(row) = executor.next().unwrap() {
            if let Some(&Value::Int(id)) = row.get(0) {
                results.push(id);
            }
        }

        assert_eq!(results, vec![2, 3]);
        executor.close().unwrap();
    }

    #[test]
    fn executor_builder_creates_limit() {
        use crate::sql::planner::{PhysicalOperator, PhysicalPlan, PhysicalTableScan, PhysicalLimitExec, OutputSchema, OutputColumn};
        use crate::records::types::DataType;

        let arena = Bump::new();

        let table_scan = PhysicalTableScan {
            schema: None,
            table: "users",
            alias: None,
            post_scan_filter: None,
            table_def: None,
        };
        let scan_op = arena.alloc(PhysicalOperator::TableScan(table_scan));

        let limit = PhysicalLimitExec {
            input: scan_op,
            limit: Some(2),
            offset: Some(1),
        };
        let limit_op = arena.alloc(PhysicalOperator::LimitExec(limit));

        let plan = PhysicalPlan {
            root: limit_op,
            output_schema: OutputSchema {
                columns: arena.alloc_slice_fill_iter([
                    OutputColumn { name: "id", data_type: DataType::Int8, nullable: false },
                ]),
            },
        };

        let rows_data: Vec<Vec<Value<'static>>> = vec![
            vec![Value::Int(1)],
            vec![Value::Int(2)],
            vec![Value::Int(3)],
            vec![Value::Int(4)],
            vec![Value::Int(5)],
        ];
        let source = MockRowSource::new(rows_data);

        let ctx = ExecutionContext::new(&arena);
        let builder = ExecutorBuilder::new(&ctx);
        let mut executor = builder.build_with_source(&plan, source).unwrap();

        executor.open().unwrap();

        let mut results = Vec::new();
        while let Some(row) = executor.next().unwrap() {
            if let Some(&Value::Int(id)) = row.get(0) {
                results.push(id);
            }
        }

        assert_eq!(results, vec![2, 3]);
        executor.close().unwrap();
    }

    #[test]
    fn btree_cursor_adapter_decodes_records() {
        use crate::records::types::{ColumnDef, DataType};
        use crate::records::{RecordBuilder, Schema};

        let column_defs = vec![
            ColumnDef::new("id", DataType::Int8),
            ColumnDef::new("score", DataType::Float8),
            ColumnDef::new("name", DataType::Text),
        ];
        let schema = Schema::new(column_defs);

        let mut builder = RecordBuilder::new(&schema);
        builder.set_int8(0, 100).unwrap();
        builder.set_float8(1, 3.15).unwrap();
        builder.set_text(2, "alice").unwrap();
        let record1 = builder.build().unwrap();

        builder.reset();
        builder.set_int8(0, 200).unwrap();
        builder.set_float8(1, 2.72).unwrap();
        builder.set_text(2, "bob").unwrap();
        let record2 = builder.build().unwrap();

        let keys = vec![vec![1u8], vec![2u8]];
        let values = vec![record1, record2];

        let decoder = SimpleDecoder::new(vec![DataType::Int8, DataType::Float8, DataType::Text]);
        let mut adapter = BTreeCursorAdapter::new(keys, values, Box::new(decoder));

        let row1 = adapter.next_row().unwrap().expect("expected row 1");
        assert_eq!(row1.len(), 3);
        if let Value::Int(id) = row1[0] {
            assert_eq!(id, 100);
        } else {
            panic!("expected Int for id");
        }
        if let Value::Float(score) = row1[1] {
            assert!((score - 3.15).abs() < 0.001);
        } else {
            panic!("expected Float for score");
        }
        if let Value::Text(ref name) = row1[2] {
            assert_eq!(name.as_ref(), "alice");
        } else {
            panic!("expected Text for name");
        }

        let row2 = adapter.next_row().unwrap().expect("expected row 2");
        if let Value::Int(id) = row2[0] {
            assert_eq!(id, 200);
        } else {
            panic!("expected Int for id");
        }

        let row3 = adapter.next_row().unwrap();
        assert!(row3.is_none(), "expected no more rows");

        adapter.reset().unwrap();
        let row1_again = adapter.next_row().unwrap().expect("expected row 1 after reset");
        if let Value::Int(id) = row1_again[0] {
            assert_eq!(id, 100);
        }
    }

    #[test]
    fn index_scan_executor_with_residual_filter() {
        use crate::records::types::{ColumnDef, DataType};
        use crate::records::{RecordBuilder, Schema};
        use crate::sql::ast::{BinaryOperator, ColumnRef, Expr, Literal};

        let arena = Bump::new();

        let column_defs = vec![
            ColumnDef::new("id", DataType::Int8),
            ColumnDef::new("age", DataType::Int4),
        ];
        let schema = Schema::new(column_defs);

        let mut builder = RecordBuilder::new(&schema);
        builder.set_int8(0, 1).unwrap();
        builder.set_int4(1, 25).unwrap();
        let record1 = builder.build().unwrap();

        builder.reset();
        builder.set_int8(0, 2).unwrap();
        builder.set_int4(1, 17).unwrap();
        let record2 = builder.build().unwrap();

        builder.reset();
        builder.set_int8(0, 3).unwrap();
        builder.set_int4(1, 30).unwrap();
        let record3 = builder.build().unwrap();

        let keys = vec![vec![1u8], vec![2u8], vec![3u8]];
        let values = vec![record1, record2, record3];

        let decoder = SimpleDecoder::new(vec![DataType::Int8, DataType::Int4]);
        let adapter = BTreeCursorAdapter::new(keys, values, Box::new(decoder));

        let col_ref = arena.alloc(ColumnRef {
            schema: None,
            table: None,
            column: "age",
        });
        let column_expr = arena.alloc(Expr::Column(*col_ref));
        let literal_expr = arena.alloc(Expr::Literal(Literal::Integer("18")));
        let filter_expr = arena.alloc(Expr::BinaryOp {
            left: column_expr,
            op: BinaryOperator::GtEq,
            right: literal_expr,
        });

        let column_map = vec![
            ("id".to_string(), 0),
            ("age".to_string(), 1),
        ];
        let residual_filter = CompiledPredicate::new(filter_expr, column_map);

        let index_scan_state = IndexScanState::new(adapter, &arena, Some(residual_filter));

        let mut executor: DynamicExecutor<MockRowSource> =
            DynamicExecutor::IndexScan(index_scan_state);

        executor.open().unwrap();

        let mut results = Vec::new();
        while let Some(row) = executor.next().unwrap() {
            if let Some(&Value::Int(id)) = row.get(0) {
                results.push(id);
            }
        }

        assert_eq!(results, vec![1, 3]);

        executor.close().unwrap();
    }

    #[test]
    fn grace_hash_join_basic() {
        let arena = Bump::new();

        let left_rows: Vec<Vec<Value<'static>>> = vec![
            vec![Value::Int(1), Value::Text(Cow::Owned("alice".to_string()))],
            vec![Value::Int(2), Value::Text(Cow::Owned("bob".to_string()))],
            vec![Value::Int(3), Value::Text(Cow::Owned("charlie".to_string()))],
        ];
        let right_rows: Vec<Vec<Value<'static>>> = vec![
            vec![Value::Int(1), Value::Int(100)],
            vec![Value::Int(2), Value::Int(200)],
            vec![Value::Int(4), Value::Int(400)],
        ];

        let left_source = MockRowSource::new(left_rows);
        let right_source = MockRowSource::new(right_rows);

        let left_exec = TableScanExecutor::new(left_source, &arena);
        let right_exec = TableScanExecutor::new(right_source, &arena);

        let mut join = GraceHashJoinExecutor::new(
            left_exec,
            right_exec,
            vec![0],
            vec![0],
            &arena,
            4,
        );

        join.open().unwrap();

        let mut results = Vec::new();
        while let Some(row) = join.next().unwrap() {
            let id = match row.get(0) {
                Some(&Value::Int(i)) => i,
                _ => panic!("expected int"),
            };
            let name = match row.get(1) {
                Some(Value::Text(s)) => s.to_string(),
                _ => panic!("expected text"),
            };
            let score = match row.get(3) {
                Some(&Value::Int(i)) => i,
                _ => panic!("expected int for score"),
            };
            results.push((id, name, score));
        }

        join.close().unwrap();

        results.sort_by_key(|r| r.0);

        assert_eq!(results.len(), 2);
        assert_eq!(results[0], (1, "alice".to_string(), 100));
        assert_eq!(results[1], (2, "bob".to_string(), 200));
    }

    #[test]
    fn grace_hash_join_multiple_partitions() {
        let arena = Bump::new();

        let left_rows: Vec<Vec<Value<'static>>> = (0..100)
            .map(|i| vec![Value::Int(i), Value::Text(Cow::Owned(format!("left_{}", i)))])
            .collect();

        let right_rows: Vec<Vec<Value<'static>>> = (0..100)
            .filter(|&i| i % 2 == 0)
            .map(|i| vec![Value::Int(i), Value::Int(i * 10)])
            .collect();

        let left_source = MockRowSource::new(left_rows);
        let right_source = MockRowSource::new(right_rows);

        let left_exec = TableScanExecutor::new(left_source, &arena);
        let right_exec = TableScanExecutor::new(right_source, &arena);

        let mut join = GraceHashJoinExecutor::new(
            left_exec,
            right_exec,
            vec![0],
            vec![0],
            &arena,
            16,
        );

        join.open().unwrap();

        let mut count = 0;
        while let Some(_row) = join.next().unwrap() {
            count += 1;
        }

        join.close().unwrap();

        assert_eq!(count, 50);
    }

    #[test]
    fn dynamic_executor_nested_loop_join() {
        let arena = Bump::new();

        let left_rows: Vec<Vec<Value<'static>>> = vec![
            vec![Value::Int(1), Value::Text(Cow::Owned("alice".to_string()))],
            vec![Value::Int(2), Value::Text(Cow::Owned("bob".to_string()))],
        ];
        let right_rows: Vec<Vec<Value<'static>>> = vec![
            vec![Value::Int(1), Value::Int(100)],
            vec![Value::Int(2), Value::Int(200)],
            vec![Value::Int(3), Value::Int(300)],
        ];

        let left_source = MockRowSource::new(left_rows);
        let right_source = MockRowSource::new(right_rows);

        let left_executor: DynamicExecutor<MockRowSource> = DynamicExecutor::TableScan(
            TableScanExecutor::new(left_source, &arena)
        );
        let right_executor: DynamicExecutor<MockRowSource> = DynamicExecutor::TableScan(
            TableScanExecutor::new(right_source, &arena)
        );

        let mut join_executor = DynamicExecutor::NestedLoopJoin(NestedLoopJoinState {
            left: Box::new(left_executor),
            right: Box::new(right_executor),
            condition: None,
            arena: &arena,
            current_left_row: None,
            right_rows: Vec::new(),
            right_index: 0,
            materialized: false,
        });

        join_executor.open().unwrap();

        let mut count = 0;
        while let Some(_row) = join_executor.next().unwrap() {
            count += 1;
        }

        join_executor.close().unwrap();

        assert_eq!(count, 6);
    }

    #[test]
    fn dynamic_executor_grace_hash_join() {
        let arena = Bump::new();

        let left_rows: Vec<Vec<Value<'static>>> = vec![
            vec![Value::Int(1), Value::Text(Cow::Owned("alice".to_string()))],
            vec![Value::Int(2), Value::Text(Cow::Owned("bob".to_string()))],
        ];
        let right_rows: Vec<Vec<Value<'static>>> = vec![
            vec![Value::Int(1), Value::Int(100)],
            vec![Value::Int(2), Value::Int(200)],
        ];

        let left_source = MockRowSource::new(left_rows);
        let right_source = MockRowSource::new(right_rows);

        let left_executor: DynamicExecutor<MockRowSource> = DynamicExecutor::TableScan(
            TableScanExecutor::new(left_source, &arena)
        );
        let right_executor: DynamicExecutor<MockRowSource> = DynamicExecutor::TableScan(
            TableScanExecutor::new(right_source, &arena)
        );

        let mut join_executor = DynamicExecutor::GraceHashJoin(GraceHashJoinState {
            left: Box::new(left_executor),
            right: Box::new(right_executor),
            left_key_indices: vec![0],
            right_key_indices: vec![0],
            arena: &arena,
            num_partitions: 4,
            left_partitions: (0..4).map(|_| Vec::new()).collect(),
            right_partitions: (0..4).map(|_| Vec::new()).collect(),
            current_partition: 0,
            partition_hash_table: hashbrown::HashMap::new(),
            partition_build_rows: Vec::new(),
            current_probe_idx: 0,
            current_match_idx: 0,
            current_matches: Vec::new(),
            partitioned: false,
        });

        join_executor.open().unwrap();

        let mut results = Vec::new();
        while let Some(row) = join_executor.next().unwrap() {
            let id = match row.get(0) {
                Some(&Value::Int(i)) => i,
                _ => panic!("expected int"),
            };
            results.push(id);
        }

        join_executor.close().unwrap();

        results.sort();
        assert_eq!(results, vec![1, 2]);
    }

    #[test]
    fn dynamic_executor_hash_aggregate() {
        let arena = Bump::new();

        let rows: Vec<Vec<Value<'static>>> = vec![
            vec![Value::Text(Cow::Owned("a".to_string())), Value::Int(10)],
            vec![Value::Text(Cow::Owned("a".to_string())), Value::Int(20)],
            vec![Value::Text(Cow::Owned("b".to_string())), Value::Int(30)],
            vec![Value::Text(Cow::Owned("b".to_string())), Value::Int(40)],
            vec![Value::Text(Cow::Owned("b".to_string())), Value::Int(50)],
        ];

        let source = MockRowSource::new(rows);

        let child_executor: DynamicExecutor<MockRowSource> = DynamicExecutor::TableScan(
            TableScanExecutor::new(source, &arena)
        );

        let mut agg_executor = DynamicExecutor::HashAggregate(HashAggregateState {
            child: Box::new(child_executor),
            group_by: vec![0],
            aggregates: vec![
                AggregateFunction::Count { distinct: false },
                AggregateFunction::Sum { column: 1 },
            ],
            arena: &arena,
            groups: hashbrown::HashMap::new(),
            result_iter: None,
            computed: false,
        });

        agg_executor.open().unwrap();

        let mut results = Vec::new();
        while let Some(row) = agg_executor.next().unwrap() {
            let group = match row.get(0) {
                Some(Value::Text(s)) => s.to_string(),
                _ => panic!("expected text"),
            };
            let count = match row.get(1) {
                Some(&Value::Int(i)) => i,
                _ => panic!("expected int for count"),
            };
            let sum = match row.get(2) {
                Some(&Value::Int(i)) => i,
                _ => panic!("expected int for sum"),
            };
            results.push((group, count, sum));
        }

        agg_executor.close().unwrap();

        results.sort_by(|a, b| a.0.cmp(&b.0));
        assert_eq!(results.len(), 2);
        assert_eq!(results[0], ("a".to_string(), 2, 30));
        assert_eq!(results[1], ("b".to_string(), 3, 120));
    }

    #[test]
    fn executor_builder_nested_loop_join() {
        let arena = Bump::new();
        let ctx = ExecutionContext::new(&arena);
        let builder = ExecutorBuilder::new(&ctx);

        let left_rows: Vec<Vec<Value<'static>>> = vec![
            vec![Value::Int(1), Value::Text(Cow::Owned("alice".to_string()))],
            vec![Value::Int(2), Value::Text(Cow::Owned("bob".to_string()))],
        ];
        let right_rows: Vec<Vec<Value<'static>>> = vec![
            vec![Value::Int(1), Value::Int(100)],
            vec![Value::Int(2), Value::Int(200)],
        ];

        let left_source = MockRowSource::new(left_rows);
        let right_source = MockRowSource::new(right_rows);

        let left_executor: DynamicExecutor<MockRowSource> = DynamicExecutor::TableScan(
            TableScanExecutor::new(left_source, &arena)
        );
        let right_executor: DynamicExecutor<MockRowSource> = DynamicExecutor::TableScan(
            TableScanExecutor::new(right_source, &arena)
        );

        let column_map = vec![
            ("id".to_string(), 0),
            ("name".to_string(), 1),
            ("user_id".to_string(), 2),
            ("score".to_string(), 3),
        ];

        let join_state = builder.build_nested_loop_join(
            left_executor,
            right_executor,
            None,
            &column_map,
        );

        let mut join_executor = DynamicExecutor::NestedLoopJoin(join_state);
        join_executor.open().unwrap();

        let mut count = 0;
        while let Some(_row) = join_executor.next().unwrap() {
            count += 1;
        }

        join_executor.close().unwrap();
        assert_eq!(count, 4);
    }

    #[test]
    fn executor_builder_grace_hash_join() {
        let arena = Bump::new();
        let ctx = ExecutionContext::new(&arena);
        let builder = ExecutorBuilder::new(&ctx);

        let left_rows: Vec<Vec<Value<'static>>> = vec![
            vec![Value::Int(1), Value::Text(Cow::Owned("alice".to_string()))],
            vec![Value::Int(2), Value::Text(Cow::Owned("bob".to_string()))],
        ];
        let right_rows: Vec<Vec<Value<'static>>> = vec![
            vec![Value::Int(1), Value::Int(100)],
            vec![Value::Int(2), Value::Int(200)],
        ];

        let left_source = MockRowSource::new(left_rows);
        let right_source = MockRowSource::new(right_rows);

        let left_executor: DynamicExecutor<MockRowSource> = DynamicExecutor::TableScan(
            TableScanExecutor::new(left_source, &arena)
        );
        let right_executor: DynamicExecutor<MockRowSource> = DynamicExecutor::TableScan(
            TableScanExecutor::new(right_source, &arena)
        );

        let join_state = builder.build_grace_hash_join(
            left_executor,
            right_executor,
            vec![0],
            vec![0],
            4,
        );

        let mut join_executor = DynamicExecutor::GraceHashJoin(join_state);
        join_executor.open().unwrap();

        let mut count = 0;
        while let Some(_row) = join_executor.next().unwrap() {
            count += 1;
        }

        join_executor.close().unwrap();
        assert_eq!(count, 2);
    }

    #[test]
    fn executor_builder_hash_aggregate() {
        let arena = Bump::new();
        let ctx = ExecutionContext::new(&arena);
        let builder = ExecutorBuilder::new(&ctx);

        let rows: Vec<Vec<Value<'static>>> = vec![
            vec![Value::Text(Cow::Owned("a".to_string())), Value::Int(10)],
            vec![Value::Text(Cow::Owned("a".to_string())), Value::Int(20)],
            vec![Value::Text(Cow::Owned("b".to_string())), Value::Int(30)],
        ];

        let source = MockRowSource::new(rows);
        let child_executor: DynamicExecutor<MockRowSource> = DynamicExecutor::TableScan(
            TableScanExecutor::new(source, &arena)
        );

        let agg_state = builder.build_hash_aggregate(
            child_executor,
            vec![0],
            vec![AggregateFunction::Sum { column: 1 }],
        );

        let mut agg_executor = DynamicExecutor::HashAggregate(agg_state);
        agg_executor.open().unwrap();

        let mut results = Vec::new();
        while let Some(row) = agg_executor.next().unwrap() {
            let group = match row.get(0) {
                Some(Value::Text(s)) => s.to_string(),
                _ => panic!("expected text"),
            };
            let sum = match row.get(1) {
                Some(&Value::Int(i)) => i,
                _ => panic!("expected int"),
            };
            results.push((group, sum));
        }

        agg_executor.close().unwrap();

        results.sort_by(|a, b| a.0.cmp(&b.0));
        assert_eq!(results.len(), 2);
        assert_eq!(results[0], ("a".to_string(), 30));
        assert_eq!(results[1], ("b".to_string(), 30));
    }
}
