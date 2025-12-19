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

use crate::sql::adapter::BTreeCursorAdapter;
use crate::sql::ast::JoinType;
use crate::sql::decoder::SimpleDecoder;
use crate::sql::predicate::CompiledPredicate;
use crate::sql::state::{
    AggregateState, GraceHashJoinState, HashAggregateState, IndexScanState, LimitState,
    NestedLoopJoinState, SortState, WindowState,
};
use crate::sql::util::{
    allocate_value_to_arena, clone_value_owned, clone_value_ref_to_arena, compare_values_for_sort,
    compute_group_key_for_dynamic, encode_value_to_key, hash_keys, hash_keys_static, hash_value,
    keys_match_static,
};
use crate::types::Value;
use bumpalo::Bump;
use eyre::Result;
use smallvec::SmallVec;
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
            Value::Uuid(u) => Value::Uuid(*u),
            Value::MacAddr(m) => Value::MacAddr(*m),
            Value::Inet4(ip) => Value::Inet4(*ip),
            Value::Inet6(ip) => Value::Inet6(*ip),
            Value::Jsonb(b) => {
                let bytes = arena.alloc_slice_copy(b);
                Value::Jsonb(Cow::Borrowed(bytes))
            }
            Value::TimestampTz {
                micros,
                offset_secs,
            } => Value::TimestampTz {
                micros: *micros,
                offset_secs: *offset_secs,
            },
            Value::Interval {
                micros,
                days,
                months,
            } => Value::Interval {
                micros: *micros,
                days: *days,
                months: *months,
            },
            Value::Point { x, y } => Value::Point { x: *x, y: *y },
            Value::GeoBox { low, high } => Value::GeoBox {
                low: *low,
                high: *high,
            },
            Value::Circle { center, radius } => Value::Circle {
                center: *center,
                radius: *radius,
            },
            Value::Enum { type_id, ordinal } => Value::Enum {
                type_id: *type_id,
                ordinal: *ordinal,
            },
            Value::Decimal { digits, scale } => Value::Decimal {
                digits: *digits,
                scale: *scale,
            },
            Value::ToastPointer(b) => {
                let bytes = arena.alloc_slice_copy(b);
                Value::ToastPointer(Cow::Borrowed(bytes))
            }
        }
    }
}

pub trait Executor<'a> {
    fn open(&mut self) -> Result<()>;
    fn next(&mut self) -> Result<Option<ExecutorRow<'a>>>;
    fn close(&mut self) -> Result<()>;
}

pub trait RowSource {
    fn reset(&mut self) -> Result<()>;
    fn next_row(&mut self) -> Result<Option<Vec<Value<'static>>>>;
}

pub struct MaterializedRowSource {
    rows: Vec<Vec<crate::types::OwnedValue>>,
    current: usize,
}

impl MaterializedRowSource {
    pub fn new(rows: Vec<Vec<crate::types::OwnedValue>>) -> Self {
        Self { rows, current: 0 }
    }

    fn owned_to_static_value(owned: &crate::types::OwnedValue) -> Value<'static> {
        use std::borrow::Cow;
        match owned {
            crate::types::OwnedValue::Null => Value::Null,
            crate::types::OwnedValue::Bool(b) => Value::Int(if *b { 1 } else { 0 }),
            crate::types::OwnedValue::Int(i) => Value::Int(*i),
            crate::types::OwnedValue::Float(f) => Value::Float(*f),
            crate::types::OwnedValue::Text(s) => Value::Text(Cow::Owned(s.clone())),
            crate::types::OwnedValue::Blob(b) => Value::Blob(Cow::Owned(b.clone())),
            crate::types::OwnedValue::Vector(v) => Value::Vector(Cow::Owned(v.clone())),
            crate::types::OwnedValue::Uuid(u) => Value::Uuid(*u),
            crate::types::OwnedValue::MacAddr(m) => Value::MacAddr(*m),
            crate::types::OwnedValue::Inet4(ip) => Value::Inet4(*ip),
            crate::types::OwnedValue::Inet6(ip) => Value::Inet6(*ip),
            crate::types::OwnedValue::Jsonb(b) => Value::Jsonb(Cow::Owned(b.clone())),
            crate::types::OwnedValue::Date(d) => Value::Int(*d as i64),
            crate::types::OwnedValue::Time(t) => Value::Int(*t),
            crate::types::OwnedValue::Timestamp(ts) => Value::Int(*ts),
            crate::types::OwnedValue::TimestampTz(micros, offset_secs) => Value::TimestampTz {
                micros: *micros,
                offset_secs: *offset_secs,
            },
            crate::types::OwnedValue::Interval(micros, days, months) => Value::Interval {
                micros: *micros,
                days: *days,
                months: *months,
            },
            crate::types::OwnedValue::Point(x, y) => Value::Point { x: *x, y: *y },
            crate::types::OwnedValue::Box(low, high) => Value::GeoBox {
                low: *low,
                high: *high,
            },
            crate::types::OwnedValue::Circle(center, radius) => Value::Circle {
                center: *center,
                radius: *radius,
            },
            crate::types::OwnedValue::Enum(type_id, ordinal) => Value::Enum {
                type_id: *type_id,
                ordinal: *ordinal,
            },
            crate::types::OwnedValue::Decimal(digits, scale) => Value::Decimal {
                digits: *digits,
                scale: *scale,
            },
            crate::types::OwnedValue::ToastPointer(b) => Value::ToastPointer(Cow::Owned(b.clone())),
        }
    }
}

impl RowSource for MaterializedRowSource {
    fn reset(&mut self) -> Result<()> {
        self.current = 0;
        Ok(())
    }

    fn next_row(&mut self) -> Result<Option<Vec<Value<'static>>>> {
        if self.current >= self.rows.len() {
            return Ok(None);
        }
        let row: Vec<Value<'static>> = self.rows[self.current]
            .iter()
            .map(Self::owned_to_static_value)
            .collect();
        self.current += 1;
        Ok(Some(row))
    }
}

#[derive(Default)]
pub struct DualSource {
    exhausted: bool,
}

impl RowSource for DualSource {
    fn reset(&mut self) -> Result<()> {
        self.exhausted = false;
        Ok(())
    }

    fn next_row(&mut self) -> Result<Option<Vec<Value<'static>>>> {
        if self.exhausted {
            return Ok(None);
        }
        self.exhausted = true;
        Ok(Some(Vec::new()))
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

pub struct StreamingBTreeSource<'storage> {
    cursor: crate::btree::Cursor<'storage, crate::storage::MmapStorage>,
    decoder: SimpleDecoder,
    started: bool,
    row_buffer: Vec<Value<'static>>,
}

impl<'storage> StreamingBTreeSource<'storage> {
    pub fn new(
        cursor: crate::btree::Cursor<'storage, crate::storage::MmapStorage>,
        decoder: SimpleDecoder,
        column_count: usize,
    ) -> Self {
        Self {
            cursor,
            decoder,
            started: false,
            row_buffer: Vec::with_capacity(column_count),
        }
    }

    pub fn from_btree_scan(
        storage: &'storage crate::storage::MmapStorage,
        root_page: u32,
        column_types: Vec<crate::records::types::DataType>,
    ) -> Result<Self> {
        Self::from_btree_scan_with_projections(storage, root_page, column_types, None)
    }

    pub fn from_btree_scan_with_projections(
        storage: &'storage crate::storage::MmapStorage,
        root_page: u32,
        column_types: Vec<crate::records::types::DataType>,
        projections: Option<Vec<usize>>,
    ) -> Result<Self> {
        Self::from_btree_scan_with_detoaster(storage, root_page, column_types, projections, None)
    }

    pub fn from_btree_scan_with_detoaster(
        storage: &'storage crate::storage::MmapStorage,
        root_page: u32,
        column_types: Vec<crate::records::types::DataType>,
        projections: Option<Vec<usize>>,
        detoaster: Option<std::sync::Arc<dyn crate::storage::toast::Detoaster + Send + Sync>>,
    ) -> Result<Self> {
        use crate::btree::BTreeReader;

        let reader = BTreeReader::new(storage, root_page)?;
        let cursor = reader.cursor_first()?;

        let output_count = projections
            .as_ref()
            .map(|p| p.len())
            .unwrap_or(column_types.len());
        let mut decoder = match projections {
            Some(proj) => SimpleDecoder::with_projections(column_types, proj),
            None => SimpleDecoder::new(column_types),
        };

        if let Some(d) = detoaster {
            decoder.set_detoaster(d);
        }

        Ok(Self::new(cursor, decoder, output_count))
    }
}

impl<'storage> RowSource for StreamingBTreeSource<'storage> {
    fn reset(&mut self) -> Result<()> {
        self.started = false;
        Ok(())
    }

    fn next_row(&mut self) -> Result<Option<Vec<Value<'static>>>> {
        if !self.started {
            self.started = true;
            if !self.cursor.valid() {
                return Ok(None);
            }
        } else if !self.cursor.advance()? {
            return Ok(None);
        }

        let key = self.cursor.key()?;
        let value = self.cursor.value()?;
        self.row_buffer.clear();
        self.decoder.decode_into(key, value, &mut self.row_buffer)?;
        Ok(Some(std::mem::take(&mut self.row_buffer)))
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
                let allocated: &'a [Value<'a>] = self.arena.alloc_slice_fill_iter(
                    values
                        .into_iter()
                        .map(|v| allocate_value_to_arena(v, self.arena)),
                );
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
                let projected: &'a [Value<'a>] =
                    self.arena
                        .alloc_slice_fill_iter(self.projections.iter().map(
                            |&idx| match row.get(idx) {
                                Some(v) => ExecutorRow::clone_value_to_arena(v, arena),
                                None => Value::Null,
                            },
                        ));
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
        let combined: &'a [Value<'a>] =
            self.arena.alloc_slice_fill_iter((0..total_cols).map(|i| {
                if i < left.column_count() {
                    left.get(i).cloned().unwrap_or(Value::Null)
                } else {
                    right
                        .get(i - left.column_count())
                        .cloned()
                        .unwrap_or(Value::Null)
                }
            }));
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

            let left_row = self.current_left_row.as_ref().unwrap();

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
    left_key_indices: SmallVec<[usize; 4]>,
    right_key_indices: SmallVec<[usize; 4]>,
    arena: &'a Bump,
    num_partitions: usize,
    left_partitions: Vec<Vec<SmallVec<[Value<'static>; 16]>>>,
    right_partitions: Vec<Vec<SmallVec<[Value<'static>; 16]>>>,
    current_partition: usize,
    partition_hash_table: hashbrown::HashMap<u64, SmallVec<[usize; 8]>>,
    partition_build_rows: Vec<SmallVec<[Value<'static>; 16]>>,
    current_probe_idx: usize,
    current_match_idx: usize,
    current_matches: SmallVec<[usize; 16]>,
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
        left_key_indices: SmallVec<[usize; 4]>,
        right_key_indices: SmallVec<[usize; 4]>,
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
            current_matches: SmallVec::new(),
            partitioned: false,
        }
    }

    fn hash_keys(row: &[Value<'static>], key_indices: &[usize]) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::Hasher;

        let mut hasher = DefaultHasher::new();
        for &idx in key_indices {
            if let Some(val) = row.get(idx) {
                hash_value(val, &mut hasher);
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
            self.partition_hash_table
                .entry(hash)
                .or_insert_with(SmallVec::new)
                .push(idx);
        }
    }

    fn combine_rows(&self, left: &[Value<'static>], right: &[Value<'static>]) -> ExecutorRow<'a> {
        let combined: Vec<Value<'a>> = left
            .iter()
            .chain(right.iter())
            .map(clone_value_owned)
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
                let owned: SmallVec<[Value<'static>; 16]> =
                    row.values.iter().map(clone_value_owned).collect();
                let hash = Self::hash_keys(&owned, &self.left_key_indices);
                let partition = (hash as usize) % self.num_partitions;
                self.left_partitions[partition].push(owned);
            }

            while let Some(row) = self.right.next()? {
                let owned: SmallVec<[Value<'static>; 16]> =
                    row.values.iter().map(clone_value_owned).collect();
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
                let probe_row =
                    &self.left_partitions[self.current_partition][self.current_probe_idx - 1];
                let build_idx = self.current_matches[self.current_match_idx];
                let build_row = &self.partition_build_rows[build_idx];
                self.current_match_idx += 1;
                return Ok(Some(self.combine_rows(probe_row, build_row)));
            }

            while self.current_probe_idx < self.left_partitions[self.current_partition].len() {
                let probe_row =
                    &self.left_partitions[self.current_partition][self.current_probe_idx];
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
                self.partition_build_rows =
                    std::mem::take(&mut self.right_partitions[self.current_partition]);
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

type GroupValue = SmallVec<[Value<'static>; 8]>;
type GroupAggStates = SmallVec<[AggregateState; 4]>;

#[allow(clippy::type_complexity)]
pub struct HashAggregateExecutor<'a, E>
where
    E: Executor<'a>,
{
    child: E,
    group_by: SmallVec<[usize; 4]>,
    aggregates: SmallVec<[AggregateFunction; 4]>,
    arena: &'a Bump,
    groups: hashbrown::HashMap<Vec<u8>, (GroupValue, GroupAggStates)>,
    result_iter: Option<std::vec::IntoIter<(GroupValue, GroupAggStates)>>,
    computed: bool,
}

impl<'a, E> HashAggregateExecutor<'a, E>
where
    E: Executor<'a>,
{
    pub fn new(
        child: E,
        group_by: SmallVec<[usize; 4]>,
        aggregates: SmallVec<[AggregateFunction; 4]>,
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
                encode_value_to_key(val, &mut key);
            }
        }
        key
    }

    fn extract_group_values(&self, row: &ExecutorRow) -> SmallVec<[Value<'static>; 8]> {
        self.group_by
            .iter()
            .map(|&col| row.get(col).map(clone_value_owned).unwrap_or(Value::Null))
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
                let states: SmallVec<[AggregateState; 4]> = self
                    .aggregates
                    .iter()
                    .map(|_| AggregateState::new())
                    .collect();
                self.groups.insert(Vec::new(), (SmallVec::new(), states));
            }

            let results: Vec<_> = self.groups.drain().map(|(_, v)| v).collect();
            self.result_iter = Some(results.into_iter());
            self.computed = true;
        }

        if let Some(ref mut iter) = self.result_iter {
            if let Some((group_values, states)) = iter.next() {
                let mut values: Vec<Value<'a>> =
                    Vec::with_capacity(group_values.len() + self.aggregates.len());

                for val in group_values {
                    values.push(allocate_value_to_arena(val, self.arena));
                }

                for (state, func) in states.iter().zip(&self.aggregates) {
                    let agg_val = state.finalize(func);
                    values.push(allocate_value_to_arena(agg_val, self.arena));
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
                let owned_values: Vec<Value<'static>> =
                    row.values.iter().map(clone_value_owned).collect();
                self.rows.push(owned_values);
            }

            let sort_keys = &self.sort_keys;
            self.rows.sort_by(|a, b| {
                for key in sort_keys {
                    let a_val = a.get(key.column).unwrap_or(&Value::Null);
                    let b_val = b.get(key.column).unwrap_or(&Value::Null);
                    let cmp = Self::compare_values(a_val, b_val);
                    if cmp != std::cmp::Ordering::Equal {
                        return if key.ascending { cmp } else { cmp.reverse() };
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
                    .map(|v| allocate_value_to_arena(v, self.arena))
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

pub enum DynamicExecutor<'a, S: RowSource> {
    TableScan(TableScanExecutor<'a, S>),
    IndexScan(IndexScanState<'a>),
    Filter(Box<DynamicExecutor<'a, S>>, CompiledPredicate<'a>),
    Project(Box<DynamicExecutor<'a, S>>, Vec<usize>, &'a Bump),
    ProjectExpr(
        Box<DynamicExecutor<'a, S>>,
        crate::sql::predicate::CompiledProjection<'a>,
        &'a Bump,
    ),
    Limit(LimitState<'a, S>),
    Sort(SortState<'a, S>),
    NestedLoopJoin(NestedLoopJoinState<'a, S>),
    GraceHashJoin(GraceHashJoinState<'a, S>),
    HashAggregate(HashAggregateState<'a, S>),
    Window(WindowState<'a, S>),
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
            DynamicExecutor::ProjectExpr(child, _, _) => child.open(),
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
                        let owned: Vec<Value<'static>> =
                            row.values.iter().map(clone_value_owned).collect();
                        state.right_rows.push(owned);
                    }
                    state.right.close()?;
                    state.materialized = true;
                }
                state.current_left_row = None;
                state.right_index = 0;
                state.left_matched = false;
                state.right_matched = vec![false; state.right_rows.len()];
                state.emitting_unmatched_right = false;
                state.unmatched_right_idx = 0;
                Ok(())
            }
            DynamicExecutor::GraceHashJoin(state) => {
                if !state.partitioned {
                    state.left.open()?;
                    while let Some(row) = state.left.next()? {
                        let hash = hash_keys(&row, &state.left_key_indices);
                        let partition = (hash as usize) % state.num_partitions;
                        let owned: Vec<Value<'static>> =
                            row.values.iter().map(clone_value_owned).collect();
                        state.left_partitions[partition].push(owned);
                    }
                    state.left.close()?;

                    state.right.open()?;
                    while let Some(row) = state.right.next()? {
                        let hash = hash_keys(&row, &state.right_key_indices);
                        let partition = (hash as usize) % state.num_partitions;
                        let owned: Vec<Value<'static>> =
                            row.values.iter().map(clone_value_owned).collect();
                        state.right_partitions[partition].push(owned);
                    }
                    state.right.close()?;
                    state.partitioned = true;

                    for p in 0..state.num_partitions {
                        state.build_matched[p] = vec![false; state.left_partitions[p].len()];
                    }
                }
                state.current_partition = 0;
                state.current_probe_idx = 0;
                state.current_match_idx = 0;
                state.current_matches.clear();
                state.partition_hash_table.clear();
                state.partition_build_rows = std::mem::take(&mut state.left_partitions[0]);
                for (idx, row) in state.partition_build_rows.iter().enumerate() {
                    let hash = hash_keys_static(row, &state.left_key_indices);
                    state
                        .partition_hash_table
                        .entry(hash)
                        .or_insert_with(Vec::new)
                        .push(idx);
                }
                state.probe_row_matched = false;
                state.emitting_unmatched_build = false;
                state.unmatched_build_partition = 0;
                state.unmatched_build_idx = 0;
                Ok(())
            }
            DynamicExecutor::HashAggregate(state) => {
                state.groups.clear();
                state.result_iter = None;
                state.computed = false;
                state.child.open()
            }
            DynamicExecutor::Window(state) => {
                state.rows.clear();
                state.window_results.clear();
                state.iter_idx = 0;
                state.computed = false;
                state.child.open()
            }
        }
    }

    fn next(&mut self) -> Result<Option<ExecutorRow<'a>>> {
        match self {
            DynamicExecutor::TableScan(ts) => ts.next(),
            DynamicExecutor::IndexScan(state) => loop {
                match state.source.next_row()? {
                    Some(row_data) => {
                        let values: &'a [Value<'a>] = state.arena.alloc_slice_fill_iter(
                            row_data
                                .into_iter()
                                .map(|v| allocate_value_to_arena(v, state.arena)),
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
            },
            DynamicExecutor::Filter(child, predicate) => loop {
                match child.next()? {
                    Some(row) => {
                        if predicate.evaluate(&row) {
                            return Ok(Some(row));
                        }
                    }
                    None => return Ok(None),
                }
            },
            DynamicExecutor::Project(child, projections, arena) => match child.next()? {
                Some(row) => {
                    let projected: &'a [Value<'a>] = arena.alloc_slice_fill_iter(
                        projections.iter().map(|&idx| match row.get(idx) {
                            Some(v) => ExecutorRow::clone_value_to_arena(v, arena),
                            None => Value::Null,
                        }),
                    );
                    Ok(Some(ExecutorRow::new(projected)))
                }
                None => Ok(None),
            },
            DynamicExecutor::ProjectExpr(child, projection, arena) => match child.next()? {
                Some(row) => {
                    let values = projection.evaluate(&row);
                    let projected: &'a [Value<'a>] = arena.alloc_slice_fill_iter(
                        values.into_iter().map(|opt_val| match opt_val {
                            Some(v) => ExecutorRow::clone_value_to_arena(&v, arena),
                            None => Value::Null,
                        }),
                    );
                    Ok(Some(ExecutorRow::new(projected)))
                }
                None => Ok(None),
            },
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
                        let owned: Vec<Value<'static>> =
                            row.values.iter().map(clone_value_owned).collect();
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
                    let arena_values: Vec<Value<'a>> = values
                        .iter()
                        .map(|v| clone_value_ref_to_arena(v, state.arena))
                        .collect();
                    let allocated = state.arena.alloc_slice_fill_iter(arena_values);
                    return Ok(Some(ExecutorRow::new(allocated)));
                }
                Ok(None)
            }
            DynamicExecutor::NestedLoopJoin(state) => loop {
                if state.emitting_unmatched_right {
                    while state.unmatched_right_idx < state.right_rows.len() {
                        let idx = state.unmatched_right_idx;
                        state.unmatched_right_idx += 1;
                        if !state.right_matched[idx] {
                            let right_row = &state.right_rows[idx];
                            let mut combined: Vec<Value<'a>> =
                                (0..state.left_col_count).map(|_| Value::Null).collect();
                            combined.extend(
                                right_row
                                    .iter()
                                    .map(|v| clone_value_ref_to_arena(v, state.arena)),
                            );
                            let allocated = state.arena.alloc_slice_fill_iter(combined);
                            return Ok(Some(ExecutorRow::new(allocated)));
                        }
                    }
                    return Ok(None);
                }

                if state.current_left_row.is_none() {
                    match state.left.next()? {
                        Some(row) => {
                            let owned: Vec<Value<'static>> =
                                row.values.iter().map(clone_value_owned).collect();
                            state.current_left_row = Some(owned);
                            state.right_index = 0;
                            state.left_matched = false;
                        }
                        None => {
                            if state.join_type == JoinType::Right
                                || state.join_type == JoinType::Full
                            {
                                state.emitting_unmatched_right = true;
                                continue;
                            }
                            return Ok(None);
                        }
                    }
                }

                let left_row = state.current_left_row.as_ref().unwrap();

                while state.right_index < state.right_rows.len() {
                    let idx = state.right_index;
                    let right_row = &state.right_rows[idx];
                    state.right_index += 1;

                    let should_join = if let Some(ref cond) = state.condition {
                        let combined: Vec<Value<'a>> = left_row
                            .iter()
                            .chain(right_row.iter())
                            .map(|v| clone_value_ref_to_arena(v, state.arena))
                            .collect();
                        let allocated = state.arena.alloc_slice_fill_iter(combined);
                        let temp_row = ExecutorRow::new(allocated);
                        cond.evaluate(&temp_row)
                    } else {
                        true
                    };

                    if should_join {
                        state.left_matched = true;
                        state.right_matched[idx] = true;
                        let combined: Vec<Value<'a>> = left_row
                            .iter()
                            .chain(right_row.iter())
                            .map(|v| clone_value_ref_to_arena(v, state.arena))
                            .collect();
                        let allocated = state.arena.alloc_slice_fill_iter(combined);
                        return Ok(Some(ExecutorRow::new(allocated)));
                    }
                }

                if !state.left_matched
                    && (state.join_type == JoinType::Left || state.join_type == JoinType::Full)
                {
                    let left_row = state.current_left_row.take().unwrap();
                    let mut combined: Vec<Value<'a>> = left_row
                        .iter()
                        .map(|v| clone_value_ref_to_arena(v, state.arena))
                        .collect();
                    combined.extend((0..state.right_col_count).map(|_| Value::Null));
                    let allocated = state.arena.alloc_slice_fill_iter(combined);
                    return Ok(Some(ExecutorRow::new(allocated)));
                }

                state.current_left_row = None;
            },
            DynamicExecutor::GraceHashJoin(state) => loop {
                if state.emitting_unmatched_build {
                    while state.unmatched_build_idx < state.partition_build_rows.len() {
                        let idx = state.unmatched_build_idx;
                        state.unmatched_build_idx += 1;
                        if !state.build_matched[state.unmatched_build_partition][idx] {
                            let build_row = &state.partition_build_rows[idx];
                            let mut combined: Vec<Value<'a>> = build_row
                                .iter()
                                .map(|v| clone_value_ref_to_arena(v, state.arena))
                                .collect();
                            for _ in 0..state.right_col_count {
                                combined.push(Value::Null);
                            }
                            let allocated = state.arena.alloc_slice_fill_iter(combined);
                            return Ok(Some(ExecutorRow::new(allocated)));
                        }
                    }
                    state.emitting_unmatched_build = false;
                    state.unmatched_build_partition += 1;

                    if state.unmatched_build_partition >= state.num_partitions {
                        return Ok(None);
                    }

                    state.partition_hash_table.clear();
                    state.partition_build_rows =
                        std::mem::take(&mut state.left_partitions[state.unmatched_build_partition]);
                    for (idx, row) in state.partition_build_rows.iter().enumerate() {
                        let hash = hash_keys_static(row, &state.left_key_indices);
                        state
                            .partition_hash_table
                            .entry(hash)
                            .or_insert_with(Vec::new)
                            .push(idx);
                    }
                    state.current_partition = state.unmatched_build_partition;
                    state.current_probe_idx = 0;
                    state.current_match_idx = 0;
                    state.current_matches.clear();
                    state.probe_row_matched = false;
                    continue;
                }

                if state.current_match_idx < state.current_matches.len() {
                    let build_idx = state.current_matches[state.current_match_idx];
                    state.current_match_idx += 1;
                    state.probe_row_matched = true;
                    if state.join_type == JoinType::Left || state.join_type == JoinType::Full {
                        state.build_matched[state.current_partition][build_idx] = true;
                    }
                    let build_row = &state.partition_build_rows[build_idx];
                    let probe_row = &state.right_partitions[state.current_partition]
                        [state.current_probe_idx - 1];

                    let combined: Vec<Value<'a>> = build_row
                        .iter()
                        .chain(probe_row.iter())
                        .map(|v| clone_value_ref_to_arena(v, state.arena))
                        .collect();
                    let allocated = state.arena.alloc_slice_fill_iter(combined);
                    return Ok(Some(ExecutorRow::new(allocated)));
                }

                if state.current_probe_idx > 0
                    && !state.probe_row_matched
                    && (state.join_type == JoinType::Right || state.join_type == JoinType::Full)
                {
                    let probe_row = &state.right_partitions[state.current_partition]
                        [state.current_probe_idx - 1];
                    let mut combined: Vec<Value<'a>> =
                        (0..state.left_col_count).map(|_| Value::Null).collect();
                    combined.extend(
                        probe_row
                            .iter()
                            .map(|v| clone_value_ref_to_arena(v, state.arena)),
                    );
                    let allocated = state.arena.alloc_slice_fill_iter(combined);
                    state.probe_row_matched = true;
                    return Ok(Some(ExecutorRow::new(allocated)));
                }

                if state.current_probe_idx < state.right_partitions[state.current_partition].len() {
                    let probe_row =
                        &state.right_partitions[state.current_partition][state.current_probe_idx];
                    state.current_probe_idx += 1;
                    state.probe_row_matched = false;
                    let hash = hash_keys_static(probe_row, &state.right_key_indices);
                    if let Some(matches) = state.partition_hash_table.get(&hash) {
                        state.current_matches = matches
                            .iter()
                            .filter(|&&idx| {
                                keys_match_static(
                                    &state.partition_build_rows[idx],
                                    probe_row,
                                    &state.left_key_indices,
                                    &state.right_key_indices,
                                )
                            })
                            .copied()
                            .collect();
                        state.current_match_idx = 0;
                    } else {
                        state.current_matches.clear();
                    }
                    continue;
                }

                if state.current_probe_idx > 0
                    && !state.probe_row_matched
                    && (state.join_type == JoinType::Right || state.join_type == JoinType::Full)
                {
                    let probe_row = &state.right_partitions[state.current_partition]
                        [state.current_probe_idx - 1];
                    let mut combined: Vec<Value<'a>> =
                        (0..state.left_col_count).map(|_| Value::Null).collect();
                    combined.extend(
                        probe_row
                            .iter()
                            .map(|v| clone_value_ref_to_arena(v, state.arena)),
                    );
                    let allocated = state.arena.alloc_slice_fill_iter(combined);
                    state.probe_row_matched = true;
                    return Ok(Some(ExecutorRow::new(allocated)));
                }

                if state.join_type == JoinType::Left || state.join_type == JoinType::Full {
                    state.emitting_unmatched_build = true;
                    state.unmatched_build_idx = 0;
                    continue;
                }

                state.current_partition += 1;
                if state.current_partition >= state.num_partitions {
                    return Ok(None);
                }

                state.partition_hash_table.clear();
                state.partition_build_rows =
                    std::mem::take(&mut state.left_partitions[state.current_partition]);
                for (idx, row) in state.partition_build_rows.iter().enumerate() {
                    let hash = hash_keys_static(row, &state.left_key_indices);
                    state
                        .partition_hash_table
                        .entry(hash)
                        .or_insert_with(Vec::new)
                        .push(idx);
                }
                state.current_probe_idx = 0;
                state.current_match_idx = 0;
                state.current_matches.clear();
                state.probe_row_matched = false;
            },
            DynamicExecutor::HashAggregate(state) => {
                if !state.computed {
                    while let Some(row) = state.child.next()? {
                        let group_key = compute_group_key_for_dynamic(&row, &state.group_by);
                        let group_values: Vec<Value<'static>> = state
                            .group_by
                            .iter()
                            .map(|&col| row.get(col).map(clone_value_owned).unwrap_or(Value::Null))
                            .collect();

                        let entry = state.groups.entry(group_key).or_insert_with(|| {
                            let initial_states: Vec<AggregateState> = state
                                .aggregates
                                .iter()
                                .map(|_| AggregateState::new())
                                .collect();
                            (group_values.clone(), initial_states)
                        });

                        for (idx, agg_fn) in state.aggregates.iter().enumerate() {
                            entry.1[idx].update(agg_fn, &row);
                        }
                    }

                    let results: Vec<(Vec<Value<'static>>, Vec<AggregateState>)> =
                        state.groups.drain().map(|(_, v)| v).collect();
                    state.result_iter = Some(results.into_iter());
                    state.computed = true;
                }

                if let Some(ref mut iter) = state.result_iter {
                    if let Some((group_vals, agg_states)) = iter.next() {
                        let mut result_values: Vec<Value<'a>> = group_vals
                            .into_iter()
                            .map(|v| allocate_value_to_arena(v, state.arena))
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
            DynamicExecutor::Window(state) => {
                if !state.computed {
                    while let Some(row) = state.child.next()? {
                        let owned: Vec<Value<'static>> =
                            row.values.iter().map(clone_value_owned).collect();
                        state.rows.push(owned);
                    }
                    state.compute_window_functions();
                    state.computed = true;
                }

                if state.iter_idx < state.rows.len() {
                    let values = &state.rows[state.iter_idx];
                    let window_vals = &state.window_results[state.iter_idx];
                    state.iter_idx += 1;

                    let mut result_values: Vec<Value<'a>> = values
                        .iter()
                        .map(|v| clone_value_ref_to_arena(v, state.arena))
                        .collect();

                    for &wval in window_vals.iter() {
                        result_values.push(Value::Int(wval));
                    }

                    let allocated = state.arena.alloc_slice_fill_iter(result_values);
                    return Ok(Some(ExecutorRow::new(allocated)));
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
            DynamicExecutor::ProjectExpr(child, _, _) => child.close(),
            DynamicExecutor::Limit(state) => state.child.close(),
            DynamicExecutor::Sort(state) => state.child.close(),
            DynamicExecutor::NestedLoopJoin(state) => state.left.close(),
            DynamicExecutor::GraceHashJoin(_) => Ok(()),
            DynamicExecutor::HashAggregate(state) => state.child.close(),
            DynamicExecutor::Window(state) => state.child.close(),
        }
    }
}
