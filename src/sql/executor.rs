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

use crate::mvcc::RecordHeader;
use crate::sql::adapter::BTreeCursorAdapter;
use crate::sql::ast::JoinType;
use crate::sql::decoder::SimpleDecoder;
use crate::sql::predicate::CompiledPredicate;
use crate::sql::partition_spiller::PartitionSpiller;
use crate::sql::state::{
    AggregateState, GraceHashJoinState, HashAggregateState, HashAntiJoinState, HashSemiJoinState,
    IndexScanState, LimitState, NestedLoopJoinState, SortState, StreamingHashJoinState, TopKState,
    WindowState,
};
use crate::sql::util::{
    allocate_value_to_arena, clone_value_owned, clone_value_ref_to_arena, compare_values_for_sort,
    compute_group_key_for_dynamic, compute_group_key_from_exprs, encode_value_to_key,
    evaluate_group_by_exprs, hash_keys, hash_keys_static, hash_value, keys_match_static,
};
use crate::types::Value;
use bumpalo::Bump;
use eyre::Result;
use smallvec::SmallVec;
use std::borrow::Cow;

fn get_sort_value_for_key<'a>(row: &[Value<'static>], key: &SortKey<'a>) -> Value<'static> {
    match &key.key_type {
        SortKeyType::Column(idx) => row.get(*idx).cloned().unwrap_or(Value::Null),
        SortKeyType::Expression { expr, column_map } => {
            eval_sort_expr_standalone(expr, row, column_map)
        }
    }
}

fn eval_sort_expr_standalone(
    expr: &crate::sql::ast::Expr<'_>,
    row: &[Value<'static>],
    column_map: &[(String, usize)],
) -> Value<'static> {
    use crate::sql::ast::{Expr, Literal};

    match expr {
        Expr::Column(col_ref) => {
            let lookup_name = if let Some(table) = col_ref.table {
                format!("{}.{}", table, col_ref.column)
            } else {
                col_ref.column.to_string()
            };
            let col_idx = column_map
                .iter()
                .find(|(name, _)| name.eq_ignore_ascii_case(&lookup_name))
                .or_else(|| {
                    column_map
                        .iter()
                        .find(|(name, _)| name.eq_ignore_ascii_case(col_ref.column))
                })
                .map(|(_, idx)| *idx);
            col_idx
                .and_then(|idx| row.get(idx).cloned())
                .unwrap_or(Value::Null)
        }
        Expr::Literal(lit) => match lit {
            Literal::Integer(s) => s.parse::<i64>().map(Value::Int).unwrap_or(Value::Null),
            Literal::Float(s) => s.parse::<f64>().map(Value::Float).unwrap_or(Value::Null),
            Literal::String(s) => Value::Text(Cow::Owned(s.to_string())),
            Literal::Null => Value::Null,
            Literal::Boolean(b) => Value::Int(if *b { 1 } else { 0 }),
            Literal::HexNumber(s) => i64::from_str_radix(s.trim_start_matches("0x").trim_start_matches("0X"), 16)
                .map(Value::Int)
                .unwrap_or(Value::Null),
            Literal::BinaryNumber(s) => i64::from_str_radix(s.trim_start_matches("0b").trim_start_matches("0B"), 2)
                .map(Value::Int)
                .unwrap_or(Value::Null),
        },
        Expr::BinaryOp { left, op, right } => {
            let left_val = eval_sort_expr_standalone(left, row, column_map);
            let right_val = eval_sort_expr_standalone(right, row, column_map);
            eval_binary_op_standalone(&left_val, op, &right_val)
        }
        Expr::Array(elements) => {
            let vals: Vec<f32> = elements
                .iter()
                .filter_map(|e| {
                    let v = eval_sort_expr_standalone(e, row, column_map);
                    match v {
                        Value::Float(f) => Some(f as f32),
                        Value::Int(i) => Some(i as f32),
                        _ => None,
                    }
                })
                .collect();
            Value::Vector(Cow::Owned(vals))
        }
        _ => Value::Null,
    }
}

fn eval_binary_op_standalone(
    left: &Value<'static>,
    op: &crate::sql::ast::BinaryOperator,
    right: &Value<'static>,
) -> Value<'static> {
    use crate::sql::ast::BinaryOperator;

    match op {
        BinaryOperator::VectorL2Distance => {
            let left_vec = value_to_vec_standalone(left);
            let right_vec = value_to_vec_standalone(right);
            if let (Some(l), Some(r)) = (left_vec, right_vec) {
                if l.len() == r.len() {
                    let dist: f64 = l
                        .iter()
                        .zip(r.iter())
                        .map(|(a, b)| ((a - b) as f64).powi(2))
                        .sum::<f64>()
                        .sqrt();
                    return Value::Float(dist);
                }
            }
            Value::Null
        }
        BinaryOperator::VectorCosineDistance => {
            let left_vec = value_to_vec_standalone(left);
            let right_vec = value_to_vec_standalone(right);
            if let (Some(l), Some(r)) = (left_vec, right_vec) {
                if l.len() == r.len() {
                    let dot: f64 = l
                        .iter()
                        .zip(r.iter())
                        .map(|(a, b)| (*a as f64) * (*b as f64))
                        .sum();
                    let mag_l: f64 = l.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
                    let mag_r: f64 = r.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
                    if mag_l > 0.0 && mag_r > 0.0 {
                        let cosine = dot / (mag_l * mag_r);
                        return Value::Float(1.0 - cosine);
                    }
                }
            }
            Value::Null
        }
        BinaryOperator::Plus => match (left, right) {
            (Value::Int(a), Value::Int(b)) => Value::Int(a + b),
            (Value::Float(a), Value::Float(b)) => Value::Float(a + b),
            (Value::Int(a), Value::Float(b)) => Value::Float(*a as f64 + b),
            (Value::Float(a), Value::Int(b)) => Value::Float(a + *b as f64),
            _ => Value::Null,
        },
        BinaryOperator::Minus => match (left, right) {
            (Value::Int(a), Value::Int(b)) => Value::Int(a - b),
            (Value::Float(a), Value::Float(b)) => Value::Float(a - b),
            (Value::Int(a), Value::Float(b)) => Value::Float(*a as f64 - b),
            (Value::Float(a), Value::Int(b)) => Value::Float(a - *b as f64),
            _ => Value::Null,
        },
        BinaryOperator::Multiply => match (left, right) {
            (Value::Int(a), Value::Int(b)) => Value::Int(a * b),
            (Value::Float(a), Value::Float(b)) => Value::Float(a * b),
            (Value::Int(a), Value::Float(b)) => Value::Float(*a as f64 * b),
            (Value::Float(a), Value::Int(b)) => Value::Float(a * *b as f64),
            _ => Value::Null,
        },
        BinaryOperator::Divide => match (left, right) {
            (Value::Int(a), Value::Int(b)) if *b != 0 => Value::Int(a / b),
            (Value::Float(a), Value::Float(b)) if *b != 0.0 => Value::Float(a / b),
            (Value::Int(a), Value::Float(b)) if *b != 0.0 => Value::Float(*a as f64 / b),
            (Value::Float(a), Value::Int(b)) if *b != 0 => Value::Float(a / *b as f64),
            _ => Value::Null,
        },
        _ => Value::Null,
    }
}

fn value_to_vec_standalone(val: &Value<'static>) -> Option<Vec<f32>> {
    match val {
        Value::Vector(v) => Some(v.to_vec()),
        Value::Text(s) => {
            let trimmed = s.trim();
            if trimmed.starts_with('[') && trimmed.ends_with(']') {
                let inner = &trimmed[1..trimmed.len() - 1];
                let parsed: Result<Vec<f32>, _> =
                    inner.split(',').map(|x| x.trim().parse::<f32>()).collect();
                parsed.ok()
            } else {
                None
            }
        }
        _ => None,
    }
}

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
    end_key: Option<Vec<u8>>,
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
            end_key: None,
        }
    }

    pub fn with_end_key(
        cursor: crate::btree::Cursor<'storage, crate::storage::MmapStorage>,
        decoder: SimpleDecoder,
        column_count: usize,
        end_key: Option<Vec<u8>>,
    ) -> Self {
        Self {
            cursor,
            decoder,
            started: false,
            row_buffer: Vec::with_capacity(column_count),
            end_key,
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

    pub fn from_btree_range_scan(
        storage: &'storage crate::storage::MmapStorage,
        root_page: u32,
        start_key: Option<&[u8]>,
        end_key: Option<&[u8]>,
        column_types: Vec<crate::records::types::DataType>,
    ) -> Result<Self> {
        Self::from_btree_range_scan_with_projections(
            storage,
            root_page,
            start_key,
            end_key,
            column_types,
            None,
        )
    }

    pub fn from_btree_range_scan_with_projections(
        storage: &'storage crate::storage::MmapStorage,
        root_page: u32,
        start_key: Option<&[u8]>,
        end_key: Option<&[u8]>,
        column_types: Vec<crate::records::types::DataType>,
        projections: Option<Vec<usize>>,
    ) -> Result<Self> {
        use crate::btree::BTreeReader;

        let reader = BTreeReader::new(storage, root_page)?;
        let cursor = if let Some(start) = start_key {
            reader.cursor_seek(start)?
        } else {
            reader.cursor_first()?
        };

        let output_count = projections
            .as_ref()
            .map(|p| p.len())
            .unwrap_or(column_types.len());
        let decoder = match projections {
            Some(proj) => SimpleDecoder::with_projections(column_types, proj),
            None => SimpleDecoder::new(column_types),
        };

        Ok(Self::with_end_key(
            cursor,
            decoder,
            output_count,
            end_key.map(|k| k.to_vec()),
        ))
    }
}

impl<'storage> RowSource for StreamingBTreeSource<'storage> {
    fn reset(&mut self) -> Result<()> {
        self.started = false;
        Ok(())
    }

    fn next_row(&mut self) -> Result<Option<Vec<Value<'static>>>> {
        loop {
            if !self.started {
                self.started = true;
                if !self.cursor.valid() {
                    return Ok(None);
                }
            } else if !self.cursor.advance()? {
                return Ok(None);
            }

            let key = self.cursor.key()?;

            if let Some(ref end) = self.end_key {
                if key >= end.as_slice() {
                    return Ok(None);
                }
            }

            let raw_value = self.cursor.value()?;

            if raw_value.len() >= RecordHeader::SIZE {
                let header = RecordHeader::from_bytes(raw_value);
                if header.is_deleted() {
                    continue;
                }
            }

            let user_data = if raw_value.len() > RecordHeader::SIZE {
                &raw_value[RecordHeader::SIZE..]
            } else {
                raw_value
            };

            self.row_buffer.clear();
            self.decoder
                .decode_into(key, user_data, &mut self.row_buffer)?;
            return Ok(Some(std::mem::take(&mut self.row_buffer)));
        }
    }
}

pub struct ReverseBTreeSource<'storage> {
    cursor: crate::btree::Cursor<'storage, crate::storage::MmapStorage>,
    decoder: SimpleDecoder,
    started: bool,
    row_buffer: Vec<Value<'static>>,
}

impl<'storage> ReverseBTreeSource<'storage> {
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

    pub fn from_btree_scan_reverse(
        storage: &'storage crate::storage::MmapStorage,
        root_page: u32,
        column_types: Vec<crate::records::types::DataType>,
    ) -> Result<Self> {
        Self::from_btree_scan_reverse_with_projections(storage, root_page, column_types, None)
    }

    pub fn from_btree_scan_reverse_with_projections(
        storage: &'storage crate::storage::MmapStorage,
        root_page: u32,
        column_types: Vec<crate::records::types::DataType>,
        projections: Option<Vec<usize>>,
    ) -> Result<Self> {
        use crate::btree::BTreeReader;

        let reader = BTreeReader::new(storage, root_page)?;
        let cursor = reader.cursor_last()?;

        let output_count = projections
            .as_ref()
            .map(|p| p.len())
            .unwrap_or(column_types.len());
        let decoder = match projections {
            Some(proj) => SimpleDecoder::with_projections(column_types, proj),
            None => SimpleDecoder::new(column_types),
        };

        Ok(Self::new(cursor, decoder, output_count))
    }
}

impl<'storage> RowSource for ReverseBTreeSource<'storage> {
    fn reset(&mut self) -> Result<()> {
        self.started = false;
        Ok(())
    }

    fn next_row(&mut self) -> Result<Option<Vec<Value<'static>>>> {
        loop {
            if !self.started {
                self.started = true;
                if !self.cursor.valid() {
                    return Ok(None);
                }
            } else if !self.cursor.prev()? {
                return Ok(None);
            }

            let key = self.cursor.key()?;
            let raw_value = self.cursor.value()?;

            if raw_value.len() >= RecordHeader::SIZE {
                let header = RecordHeader::from_bytes(raw_value);
                if header.is_deleted() {
                    continue;
                }
            }

            let user_data = if raw_value.len() > RecordHeader::SIZE {
                &raw_value[RecordHeader::SIZE..]
            } else {
                raw_value
            };

            self.row_buffer.clear();
            self.decoder
                .decode_into(key, user_data, &mut self.row_buffer)?;
            return Ok(Some(std::mem::take(&mut self.row_buffer)));
        }
    }
}

pub enum BTreeSource<'storage> {
    Forward(StreamingBTreeSource<'storage>),
    Reverse(ReverseBTreeSource<'storage>),
}

impl<'storage> RowSource for BTreeSource<'storage> {
    fn reset(&mut self) -> Result<()> {
        match self {
            BTreeSource::Forward(s) => s.reset(),
            BTreeSource::Reverse(s) => s.reset(),
        }
    }

    fn next_row(&mut self) -> Result<Option<Vec<Value<'static>>>> {
        match self {
            BTreeSource::Forward(s) => s.next_row(),
            BTreeSource::Reverse(s) => s.next_row(),
        }
    }
}

pub struct DynamicExecutorSource<'a, S: RowSource> {
    executor: DynamicExecutor<'a, S>,
    opened: bool,
}

impl<'a, S: RowSource> DynamicExecutorSource<'a, S> {
    pub fn new(executor: DynamicExecutor<'a, S>) -> Self {
        Self {
            executor,
            opened: false,
        }
    }
}

impl<'a, S: RowSource> RowSource for DynamicExecutorSource<'a, S> {
    fn reset(&mut self) -> Result<()> {
        if self.opened {
            self.executor.close()?;
        }
        self.executor.open()?;
        self.opened = true;
        Ok(())
    }

    fn next_row(&mut self) -> Result<Option<Vec<Value<'static>>>> {
        if !self.opened {
            self.executor.open()?;
            self.opened = true;
        }
        match self.executor.next()? {
            Some(row) => {
                let values: Vec<Value<'static>> = row
                    .values
                    .iter()
                    .map(|v| clone_value_owned(v))
                    .collect();
                Ok(Some(values))
            }
            None => Ok(None),
        }
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
pub enum SortKeyType<'a> {
    Column(usize),
    Expression {
        expr: &'a crate::sql::ast::Expr<'a>,
        column_map: Vec<(String, usize)>,
    },
}

#[derive(Debug, Clone)]
pub struct SortKey<'a> {
    pub key_type: SortKeyType<'a>,
    pub ascending: bool,
}

impl<'a> SortKey<'a> {
    pub fn column(idx: usize, ascending: bool) -> Self {
        Self {
            key_type: SortKeyType::Column(idx),
            ascending,
        }
    }

    pub fn expression(
        expr: &'a crate::sql::ast::Expr<'a>,
        column_map: Vec<(String, usize)>,
        ascending: bool,
    ) -> Self {
        Self {
            key_type: SortKeyType::Expression { expr, column_map },
            ascending,
        }
    }
}

pub struct SortExecutor<'a, E>
where
    E: Executor<'a>,
{
    child: E,
    sort_keys: Vec<SortKey<'a>>,
    arena: &'a Bump,
    rows: Vec<Vec<Value<'static>>>,
    sorted_iter: Option<std::vec::IntoIter<Vec<Value<'static>>>>,
    materialized: bool,
}

impl<'a, E> SortExecutor<'a, E>
where
    E: Executor<'a>,
{
    pub fn new(child: E, sort_keys: Vec<SortKey<'a>>, arena: &'a Bump) -> Self {
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

    fn get_sort_value(row: &[Value<'static>], key: &SortKey<'a>) -> Value<'static> {
        match &key.key_type {
            SortKeyType::Column(idx) => row.get(*idx).cloned().unwrap_or(Value::Null),
            SortKeyType::Expression { expr, column_map } => {
                Self::eval_sort_expr(expr, row, column_map)
            }
        }
    }

    fn eval_sort_expr(
        expr: &crate::sql::ast::Expr<'_>,
        row: &[Value<'static>],
        column_map: &[(String, usize)],
    ) -> Value<'static> {
        use crate::sql::ast::{Expr, Literal};

        match expr {
            Expr::Column(col_ref) => {
                let lookup_name = if let Some(table) = col_ref.table {
                    format!("{}.{}", table, col_ref.column)
                } else {
                    col_ref.column.to_string()
                };
                let col_idx = column_map
                    .iter()
                    .find(|(name, _)| name.eq_ignore_ascii_case(&lookup_name))
                    .or_else(|| {
                        column_map
                            .iter()
                            .find(|(name, _)| name.eq_ignore_ascii_case(col_ref.column))
                    })
                    .map(|(_, idx)| *idx);
                col_idx
                    .and_then(|idx| row.get(idx).cloned())
                    .unwrap_or(Value::Null)
            }
            Expr::Literal(lit) => match lit {
                Literal::Integer(s) => s
                    .parse::<i64>()
                    .map(Value::Int)
                    .unwrap_or(Value::Null),
                Literal::Float(s) => s
                    .parse::<f64>()
                    .map(Value::Float)
                    .unwrap_or(Value::Null),
                Literal::String(s) => Value::Text(Cow::Owned(s.to_string())),
                Literal::Null => Value::Null,
                Literal::Boolean(b) => Value::Int(if *b { 1 } else { 0 }),
                Literal::HexNumber(s) => i64::from_str_radix(s.trim_start_matches("0x").trim_start_matches("0X"), 16)
                    .map(Value::Int)
                    .unwrap_or(Value::Null),
                Literal::BinaryNumber(s) => i64::from_str_radix(s.trim_start_matches("0b").trim_start_matches("0B"), 2)
                    .map(Value::Int)
                    .unwrap_or(Value::Null),
            },
            Expr::BinaryOp { left, op, right } => {
                let left_val = Self::eval_sort_expr(left, row, column_map);
                let right_val = Self::eval_sort_expr(right, row, column_map);
                Self::eval_binary_op(&left_val, op, &right_val)
            }
            Expr::Array(elements) => {
                let vals: Vec<f32> = elements
                    .iter()
                    .filter_map(|e| {
                        let v = Self::eval_sort_expr(e, row, column_map);
                        match v {
                            Value::Float(f) => Some(f as f32),
                            Value::Int(i) => Some(i as f32),
                            _ => None,
                        }
                    })
                    .collect();
                Value::Vector(Cow::Owned(vals))
            }
            _ => Value::Null,
        }
    }

    fn eval_binary_op(left: &Value<'static>, op: &crate::sql::ast::BinaryOperator, right: &Value<'static>) -> Value<'static> {
        use crate::sql::ast::BinaryOperator;

        match op {
            BinaryOperator::VectorL2Distance => {
                let left_vec = Self::value_to_vec(left);
                let right_vec = Self::value_to_vec(right);
                if let (Some(l), Some(r)) = (left_vec, right_vec) {
                    if l.len() == r.len() {
                        let dist: f64 = l
                            .iter()
                            .zip(r.iter())
                            .map(|(a, b)| ((a - b) as f64).powi(2))
                            .sum::<f64>()
                            .sqrt();
                        return Value::Float(dist);
                    }
                }
                Value::Null
            }
            BinaryOperator::VectorCosineDistance => {
                let left_vec = Self::value_to_vec(left);
                let right_vec = Self::value_to_vec(right);
                if let (Some(l), Some(r)) = (left_vec, right_vec) {
                    if l.len() == r.len() {
                        let dot: f64 = l.iter().zip(r.iter()).map(|(a, b)| (*a as f64) * (*b as f64)).sum();
                        let mag_l: f64 = l.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
                        let mag_r: f64 = r.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
                        if mag_l > 0.0 && mag_r > 0.0 {
                            let cosine = dot / (mag_l * mag_r);
                            return Value::Float(1.0 - cosine);
                        }
                    }
                }
                Value::Null
            }
            BinaryOperator::Plus => match (left, right) {
                (Value::Int(a), Value::Int(b)) => Value::Int(a + b),
                (Value::Float(a), Value::Float(b)) => Value::Float(a + b),
                (Value::Int(a), Value::Float(b)) => Value::Float(*a as f64 + b),
                (Value::Float(a), Value::Int(b)) => Value::Float(a + *b as f64),
                _ => Value::Null,
            },
            BinaryOperator::Minus => match (left, right) {
                (Value::Int(a), Value::Int(b)) => Value::Int(a - b),
                (Value::Float(a), Value::Float(b)) => Value::Float(a - b),
                (Value::Int(a), Value::Float(b)) => Value::Float(*a as f64 - b),
                (Value::Float(a), Value::Int(b)) => Value::Float(a - *b as f64),
                _ => Value::Null,
            },
            BinaryOperator::Multiply => match (left, right) {
                (Value::Int(a), Value::Int(b)) => Value::Int(a * b),
                (Value::Float(a), Value::Float(b)) => Value::Float(a * b),
                (Value::Int(a), Value::Float(b)) => Value::Float(*a as f64 * b),
                (Value::Float(a), Value::Int(b)) => Value::Float(a * *b as f64),
                _ => Value::Null,
            },
            BinaryOperator::Divide => match (left, right) {
                (Value::Int(a), Value::Int(b)) if *b != 0 => Value::Int(a / b),
                (Value::Float(a), Value::Float(b)) if *b != 0.0 => Value::Float(a / b),
                (Value::Int(a), Value::Float(b)) if *b != 0.0 => Value::Float(*a as f64 / b),
                (Value::Float(a), Value::Int(b)) if *b != 0 => Value::Float(a / *b as f64),
                _ => Value::Null,
            },
            _ => Value::Null,
        }
    }

    fn value_to_vec(val: &Value<'static>) -> Option<Vec<f32>> {
        match val {
            Value::Vector(v) => Some(v.to_vec()),
            Value::Text(s) => {
                let trimmed = s.trim();
                if trimmed.starts_with('[') && trimmed.ends_with(']') {
                    let inner = &trimmed[1..trimmed.len() - 1];
                    let parsed: Result<Vec<f32>, _> = inner
                        .split(',')
                        .map(|x| x.trim().parse::<f32>())
                        .collect();
                    parsed.ok()
                } else {
                    None
                }
            }
            _ => None,
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
                    let a_val = Self::get_sort_value(a, key);
                    let b_val = Self::get_sort_value(b, key);
                    let cmp = Self::compare_values(&a_val, &b_val);
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

/// Unified executor enum for dynamic dispatch of query operators.
///
/// ## Boxing Strategy
///
/// Certain variants use `Box` to reduce the overall enum size:
/// - `GraceHashJoinState` is boxed because it contains large partition buffers
/// - Recursive variants (`Filter`, `Project`, `ProjectExpr`) are boxed for the child
///
/// These are **one-time allocations per operator** during query planning, not per-row
/// allocations during execution, so they comply with the zero-allocation-per-row goal.
#[allow(clippy::large_enum_variant)]
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
    TopK(TopKState<'a, S>),
    NestedLoopJoin(NestedLoopJoinState<'a, S>),
    /// Boxed to reduce enum size - GraceHashJoinState contains large partition buffers
    GraceHashJoin(Box<GraceHashJoinState<'a, S>>),
    StreamingHashJoin(StreamingHashJoinState<'a, S>),
    HashSemiJoin(HashSemiJoinState<'a, S>),
    HashAntiJoin(HashAntiJoinState<'a, S>),
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
            DynamicExecutor::TopK(state) => {
                state.heap.clear();
                state.result.clear();
                state.iter_idx = 0;
                state.computed = false;
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

                        if let Some(budget) = state.memory_budget {
                            let estimated_size = state.right_rows.len() * 128;
                            if estimated_size > state.last_reported_bytes + 64 * 1024 {
                                let delta = estimated_size - state.last_reported_bytes;
                                budget.allocate(crate::memory::Pool::Query, delta)?;
                                state.last_reported_bytes = estimated_size;
                            }
                        }
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
                    if state.use_spill {
                        let spill_dir = state.spill_dir.clone().unwrap();
                        std::fs::create_dir_all(&spill_dir).ok();
                        state.left_spiller = Some(PartitionSpiller::new(
                            spill_dir.clone(),
                            state.num_partitions,
                            state.memory_budget,
                            state.query_id,
                            'L',
                        )?);
                        state.right_spiller = Some(PartitionSpiller::new(
                            spill_dir,
                            state.num_partitions,
                            state.memory_budget,
                            state.query_id,
                            'R',
                        )?);

                        state.left.open()?;
                        while let Some(row) = state.left.next()? {
                            let hash = hash_keys(&row, &state.left_key_indices);
                            let partition = (hash as usize) % state.num_partitions;
                            let owned: SmallVec<[Value<'static>; 16]> =
                                row.values.iter().map(clone_value_owned).collect();
                            state
                                .left_spiller
                                .as_mut()
                                .unwrap()
                                .write_row(partition, owned)?;
                        }
                        state.left.close()?;

                        state.right.open()?;
                        while let Some(row) = state.right.next()? {
                            let hash = hash_keys(&row, &state.right_key_indices);
                            let partition = (hash as usize) % state.num_partitions;
                            let owned: SmallVec<[Value<'static>; 16]> =
                                row.values.iter().map(clone_value_owned).collect();
                            state
                                .right_spiller
                                .as_mut()
                                .unwrap()
                                .write_row(partition, owned)?;
                        }
                        state.right.close()?;
                    } else {
                        let mut total_rows = 0usize;
                        state.left.open()?;
                        while let Some(row) = state.left.next()? {
                            let hash = hash_keys(&row, &state.left_key_indices);
                            let partition = (hash as usize) % state.num_partitions;
                            let owned: Vec<Value<'static>> =
                                row.values.iter().map(clone_value_owned).collect();
                            state.left_partitions[partition].push(owned);
                            total_rows += 1;

                            if let Some(budget) = state.memory_budget_ref {
                                let estimated_size = total_rows * 128;
                                if estimated_size > state.last_reported_bytes + 64 * 1024 {
                                    let delta = estimated_size - state.last_reported_bytes;
                                    budget.allocate(crate::memory::Pool::Query, delta)?;
                                    state.last_reported_bytes = estimated_size;
                                }
                            }
                        }
                        state.left.close()?;

                        state.right.open()?;
                        while let Some(row) = state.right.next()? {
                            let hash = hash_keys(&row, &state.right_key_indices);
                            let partition = (hash as usize) % state.num_partitions;
                            let owned: Vec<Value<'static>> =
                                row.values.iter().map(clone_value_owned).collect();
                            state.right_partitions[partition].push(owned);
                            total_rows += 1;

                            if let Some(budget) = state.memory_budget_ref {
                                let estimated_size = total_rows * 128;
                                if estimated_size > state.last_reported_bytes + 64 * 1024 {
                                    let delta = estimated_size - state.last_reported_bytes;
                                    budget.allocate(crate::memory::Pool::Query, delta)?;
                                    state.last_reported_bytes = estimated_size;
                                }
                            }
                        }
                        state.right.close()?;

                        for p in 0..state.num_partitions {
                            state.build_matched[p] = vec![false; state.left_partitions[p].len()];
                        }
                    }
                    state.partitioned = true;
                }
                state.current_partition = 0;
                state.current_probe_idx = 0;
                state.current_match_idx = 0;
                state.current_matches.clear();
                state.partition_hash_table.clear();

                if state.use_spill {
                    state.left_spiller.as_mut().unwrap().start_read(0)?;
                    while let Some(row) = state.left_spiller.as_mut().unwrap().read_next()? {
                        let owned: Vec<Value<'static>> = row.to_vec();
                        let hash = hash_keys_static(&owned, &state.left_key_indices);
                        let idx = state.partition_build_rows.len();
                        state
                            .partition_hash_table
                            .entry(hash)
                            .or_insert_with(Vec::new)
                            .push(idx);
                        state.partition_build_rows.push(owned);
                    }
                    state.build_matched[0] = vec![false; state.partition_build_rows.len()];
                    state.right_spiller.as_mut().unwrap().start_read(0)?;
                } else {
                    state.partition_build_rows = std::mem::take(&mut state.left_partitions[0]);
                    for (idx, row) in state.partition_build_rows.iter().enumerate() {
                        let hash = hash_keys_static(row, &state.left_key_indices);
                        state
                            .partition_hash_table
                            .entry(hash)
                            .or_insert_with(Vec::new)
                            .push(idx);
                    }
                }
                state.probe_row_matched = false;
                state.emitting_unmatched_build = false;
                state.unmatched_build_partition = 0;
                state.unmatched_build_idx = 0;
                Ok(())
            }
            DynamicExecutor::StreamingHashJoin(state) => {
                if !state.built {
                    state.hash_table.clear();
                    state.build_rows.clear();

                    state.build.open()?;
                    while let Some(row) = state.build.next()? {
                        let hash = hash_keys(&row, &state.build_key_indices);
                        let owned: Vec<Value<'static>> =
                            row.values.iter().map(clone_value_owned).collect();
                        let idx = state.build_rows.len();
                        state
                            .hash_table
                            .entry(hash)
                            .or_insert_with(SmallVec::new)
                            .push(idx);
                        state.build_rows.push(owned);

                        if let Some(budget) = state.memory_budget_ref {
                            let estimated_size = state.build_rows.len() * 128;
                            if estimated_size > state.last_reported_bytes + 64 * 1024 {
                                let delta = estimated_size - state.last_reported_bytes;
                                budget.allocate(crate::memory::Pool::Query, delta)?;
                                state.last_reported_bytes = estimated_size;
                            }
                        }
                    }
                    state.build.close()?;

                    state.build_matched = vec![false; state.build_rows.len()];
                    state.built = true;
                }

                state.current_probe_row = None;
                state.current_matches.clear();
                state.current_match_idx = 0;
                state.probe_row_matched = false;
                state.emitting_unmatched_build = false;
                state.unmatched_build_idx = 0;

                state.probe.open()
            }
            DynamicExecutor::HashSemiJoin(state) => {
                if !state.built {
                    state.right.open()?;
                    state.hash_table.clear();
                    while let Some(row) = state.right.next()? {
                        let hash = hash_keys(&row, &state.right_key_indices);
                        state.hash_table.insert(hash);
                    }
                    state.right.close()?;
                    state.built = true;
                }
                state.left.open()
            }
            DynamicExecutor::HashAntiJoin(state) => {
                if !state.built {
                    state.right.open()?;
                    state.hash_table.clear();
                    while let Some(row) = state.right.next()? {
                        let hash = hash_keys(&row, &state.right_key_indices);
                        state.hash_table.insert(hash);
                    }
                    state.right.close()?;
                    state.built = true;
                }
                state.left.open()
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

                        if let Some(budget) = state.memory_budget {
                            let estimated_size = state.rows.len() * 128;
                            if estimated_size > state.last_reported_bytes + 64 * 1024 {
                                let delta = estimated_size - state.last_reported_bytes;
                                budget.allocate(crate::memory::Pool::Query, delta)?;
                                state.last_reported_bytes = estimated_size;
                            }
                        }
                    }

                    let sort_keys = &state.sort_keys;
                    state.rows.sort_by(|a, b| {
                        for key in sort_keys.iter() {
                            let a_val = get_sort_value_for_key(a, key);
                            let b_val = get_sort_value_for_key(b, key);
                            let cmp = compare_values_for_sort(&a_val, &b_val);
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
            DynamicExecutor::TopK(state) => {
                if !state.computed {
                    let heap_size = (state.limit + state.offset) as usize;
                    let sort_keys = &state.sort_keys;

                    while let Some(row) = state.child.next()? {
                        let owned: Vec<Value<'static>> =
                            row.values.iter().map(clone_value_owned).collect();

                        if state.heap.len() < heap_size {
                            state.heap.push(owned);

                            if let Some(budget) = state.memory_budget {
                                let estimated_size = state.heap.len() * 128;
                                if estimated_size > state.last_reported_bytes + 64 * 1024 {
                                    let delta = estimated_size - state.last_reported_bytes;
                                    budget.allocate(crate::memory::Pool::Query, delta)?;
                                    state.last_reported_bytes = estimated_size;
                                }
                            }

                            if state.heap.len() == heap_size {
                                state.heap.sort_by(|a, b| {
                                    for key in sort_keys.iter() {
                                        let a_val = get_sort_value_for_key(a, key);
                                        let b_val = get_sort_value_for_key(b, key);
                                        let cmp = compare_values_for_sort(&a_val, &b_val);
                                        if cmp != std::cmp::Ordering::Equal {
                                            return if key.ascending {
                                                cmp.reverse()
                                            } else {
                                                cmp
                                            };
                                        }
                                    }
                                    std::cmp::Ordering::Equal
                                });
                            }
                        } else {
                            let boundary = &state.heap[0];
                            let should_replace = {
                                let mut result = std::cmp::Ordering::Equal;
                                for key in sort_keys.iter() {
                                    let new_val = get_sort_value_for_key(&owned, key);
                                    let bound_val = get_sort_value_for_key(boundary, key);
                                    let cmp = compare_values_for_sort(&new_val, &bound_val);
                                    if cmp != std::cmp::Ordering::Equal {
                                        result = if key.ascending { cmp } else { cmp.reverse() };
                                        break;
                                    }
                                }
                                result == std::cmp::Ordering::Less
                            };

                            if should_replace {
                                state.heap[0] = owned;
                                let heap_len = state.heap.len();
                                let mut i = 0;
                                loop {
                                    let left = 2 * i + 1;
                                    let right = 2 * i + 2;
                                    let mut largest = i;

                                    if left < heap_len {
                                        let cmp = {
                                            let mut result = std::cmp::Ordering::Equal;
                                            for key in sort_keys.iter() {
                                                let l_val =
                                                    get_sort_value_for_key(&state.heap[left], key);
                                                let lg_val =
                                                    get_sort_value_for_key(&state.heap[largest], key);
                                                let c = compare_values_for_sort(&l_val, &lg_val);
                                                if c != std::cmp::Ordering::Equal {
                                                    result = if key.ascending {
                                                        c
                                                    } else {
                                                        c.reverse()
                                                    };
                                                    break;
                                                }
                                            }
                                            result
                                        };
                                        if cmp == std::cmp::Ordering::Greater {
                                            largest = left;
                                        }
                                    }

                                    if right < heap_len {
                                        let cmp = {
                                            let mut result = std::cmp::Ordering::Equal;
                                            for key in sort_keys.iter() {
                                                let r_val =
                                                    get_sort_value_for_key(&state.heap[right], key);
                                                let lg_val =
                                                    get_sort_value_for_key(&state.heap[largest], key);
                                                let c = compare_values_for_sort(&r_val, &lg_val);
                                                if c != std::cmp::Ordering::Equal {
                                                    result = if key.ascending {
                                                        c
                                                    } else {
                                                        c.reverse()
                                                    };
                                                    break;
                                                }
                                            }
                                            result
                                        };
                                        if cmp == std::cmp::Ordering::Greater {
                                            largest = right;
                                        }
                                    }

                                    if largest == i {
                                        break;
                                    }
                                    state.heap.swap(i, largest);
                                    i = largest;
                                }
                            }
                        }
                    }

                    state.heap.sort_by(|a, b| {
                        for key in sort_keys.iter() {
                            let a_val = get_sort_value_for_key(a, key);
                            let b_val = get_sort_value_for_key(b, key);
                            let cmp = compare_values_for_sort(&a_val, &b_val);
                            if cmp != std::cmp::Ordering::Equal {
                                return if key.ascending { cmp } else { cmp.reverse() };
                            }
                        }
                        std::cmp::Ordering::Equal
                    });

                    let offset = state.offset as usize;
                    let limit = state.limit as usize;
                    let start = offset.min(state.heap.len());
                    let end = (offset + limit).min(state.heap.len());
                    state.result = state.heap.drain(start..end).collect();
                    state.computed = true;
                }

                if state.iter_idx < state.result.len() {
                    let values = &state.result[state.iter_idx];
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
                    state.partition_build_rows.clear();
                    if state.use_spill {
                        let partition = state.unmatched_build_partition;
                        state.left_spiller.as_mut().unwrap().start_read(partition)?;
                        while let Some(row) =
                            state.left_spiller.as_mut().unwrap().read_next()?
                        {
                            let owned: Vec<Value<'static>> = row.to_vec();
                            state.partition_build_rows.push(owned);
                        }
                        state.build_matched[partition] =
                            vec![false; state.partition_build_rows.len()];
                        state
                            .right_spiller
                            .as_mut()
                            .unwrap()
                            .start_read(partition)?;
                    } else {
                        state.partition_build_rows = std::mem::take(
                            &mut state.left_partitions[state.unmatched_build_partition],
                        );
                    }
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
                    let probe_row: &[Value<'static>] = if state.use_spill {
                        &state.probe_row_buf
                    } else {
                        &state.right_partitions[state.current_partition]
                            [state.current_probe_idx - 1]
                    };

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
                    let probe_row: &[Value<'static>] = if state.use_spill {
                        &state.probe_row_buf
                    } else {
                        &state.right_partitions[state.current_partition]
                            [state.current_probe_idx - 1]
                    };
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

                let has_more_probe = if state.use_spill {
                    match state.right_spiller.as_mut().unwrap().read_next()? {
                        Some(row) => {
                            state.probe_row_buf.clear();
                            state.probe_row_buf.extend(row.iter().cloned());
                            true
                        }
                        None => false,
                    }
                } else {
                    state.current_probe_idx
                        < state.right_partitions[state.current_partition].len()
                };

                if has_more_probe {
                    let probe_row: &[Value<'static>] = if state.use_spill {
                        &state.probe_row_buf
                    } else {
                        let row = &state.right_partitions[state.current_partition]
                            [state.current_probe_idx];
                        row.as_slice()
                    };
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
                    let probe_row: &[Value<'static>] = if state.use_spill {
                        &state.probe_row_buf
                    } else {
                        &state.right_partitions[state.current_partition]
                            [state.current_probe_idx - 1]
                    };
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
                state.partition_build_rows.clear();
                if state.use_spill {
                    let partition = state.current_partition;
                    state.left_spiller.as_mut().unwrap().start_read(partition)?;
                    while let Some(row) = state.left_spiller.as_mut().unwrap().read_next()? {
                        let owned: Vec<Value<'static>> = row.to_vec();
                        state.partition_build_rows.push(owned);
                    }
                    state.build_matched[partition] =
                        vec![false; state.partition_build_rows.len()];
                    state.right_spiller.as_mut().unwrap().start_read(partition)?;
                } else {
                    state.partition_build_rows =
                        std::mem::take(&mut state.left_partitions[state.current_partition]);
                }
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
            DynamicExecutor::StreamingHashJoin(state) => loop {
                if state.emitting_unmatched_build {
                    while state.unmatched_build_idx < state.build_rows.len() {
                        let idx = state.unmatched_build_idx;
                        state.unmatched_build_idx += 1;
                        if !state.build_matched[idx] {
                            let build_row = &state.build_rows[idx];
                            let combined: Vec<Value<'a>> = if state.swapped {
                                let mut v: Vec<Value<'a>> =
                                    (0..state.probe_col_count).map(|_| Value::Null).collect();
                                v.extend(
                                    build_row
                                        .iter()
                                        .map(|val| clone_value_ref_to_arena(val, state.arena)),
                                );
                                v
                            } else {
                                let mut v: Vec<Value<'a>> = build_row
                                    .iter()
                                    .map(|val| clone_value_ref_to_arena(val, state.arena))
                                    .collect();
                                v.extend((0..state.probe_col_count).map(|_| Value::Null));
                                v
                            };
                            let allocated = state.arena.alloc_slice_fill_iter(combined);
                            return Ok(Some(ExecutorRow::new(allocated)));
                        }
                    }
                    return Ok(None);
                }

                if state.current_match_idx < state.current_matches.len() {
                    let build_idx = state.current_matches[state.current_match_idx];
                    state.current_match_idx += 1;
                    state.probe_row_matched = true;

                    if matches!(state.join_type, JoinType::Left | JoinType::Full) {
                        state.build_matched[build_idx] = true;
                    }

                    let build_row = &state.build_rows[build_idx];
                    let probe_row = state.current_probe_row.as_ref().unwrap();

                    let combined: Vec<Value<'a>> = if state.swapped {
                        probe_row
                            .iter()
                            .chain(build_row.iter())
                            .map(|v| clone_value_ref_to_arena(v, state.arena))
                            .collect()
                    } else {
                        build_row
                            .iter()
                            .chain(probe_row.iter())
                            .map(|v| clone_value_ref_to_arena(v, state.arena))
                            .collect()
                    };
                    let allocated = state.arena.alloc_slice_fill_iter(combined);
                    return Ok(Some(ExecutorRow::new(allocated)));
                }

                if state.current_probe_row.is_some()
                    && !state.probe_row_matched
                    && matches!(state.join_type, JoinType::Right | JoinType::Full)
                {
                    let probe_row = state.current_probe_row.as_ref().unwrap();
                    let combined: Vec<Value<'a>> = if state.swapped {
                        let mut v: Vec<Value<'a>> = probe_row
                            .iter()
                            .map(|val| clone_value_ref_to_arena(val, state.arena))
                            .collect();
                        v.extend((0..state.build_col_count).map(|_| Value::Null));
                        v
                    } else {
                        let mut v: Vec<Value<'a>> =
                            (0..state.build_col_count).map(|_| Value::Null).collect();
                        v.extend(
                            probe_row
                                .iter()
                                .map(|val| clone_value_ref_to_arena(val, state.arena)),
                        );
                        v
                    };
                    let allocated = state.arena.alloc_slice_fill_iter(combined);
                    state.current_probe_row = None;
                    return Ok(Some(ExecutorRow::new(allocated)));
                }

                match state.probe.next()? {
                    Some(row) => {
                        let owned: SmallVec<[Value<'static>; 16]> =
                            row.values.iter().map(clone_value_owned).collect();
                        let hash = hash_keys_static(&owned, &state.probe_key_indices);

                        state.current_matches.clear();
                        if let Some(matches) = state.hash_table.get(&hash) {
                            for &idx in matches.iter() {
                                if keys_match_static(
                                    &state.build_rows[idx],
                                    &owned,
                                    &state.build_key_indices,
                                    &state.probe_key_indices,
                                ) {
                                    state.current_matches.push(idx);
                                }
                            }
                        }
                        state.current_probe_row = Some(owned);
                        state.current_match_idx = 0;
                        state.probe_row_matched = false;
                    }
                    None => {
                        if matches!(state.join_type, JoinType::Left | JoinType::Full) {
                            state.emitting_unmatched_build = true;
                            continue;
                        }
                        return Ok(None);
                    }
                }
            },
            DynamicExecutor::HashSemiJoin(state) => loop {
                match state.left.next()? {
                    Some(row) => {
                        let hash = hash_keys(&row, &state.left_key_indices);
                        if state.hash_table.contains(&hash) {
                            let left_values: Vec<Value<'a>> = row
                                .values
                                .iter()
                                .take(state.left_col_count)
                                .map(|v| clone_value_ref_to_arena(v, state.arena))
                                .collect();
                            let allocated = state.arena.alloc_slice_fill_iter(left_values);
                            return Ok(Some(ExecutorRow::new(allocated)));
                        }
                    }
                    None => return Ok(None),
                }
            },
            DynamicExecutor::HashAntiJoin(state) => loop {
                match state.left.next()? {
                    Some(row) => {
                        let hash = hash_keys(&row, &state.left_key_indices);
                        if !state.hash_table.contains(&hash) {
                            let left_values: Vec<Value<'a>> = row
                                .values
                                .iter()
                                .take(state.left_col_count)
                                .map(|v| clone_value_ref_to_arena(v, state.arena))
                                .collect();
                            let allocated = state.arena.alloc_slice_fill_iter(left_values);
                            return Ok(Some(ExecutorRow::new(allocated)));
                        }
                    }
                    None => return Ok(None),
                }
            },
            DynamicExecutor::HashAggregate(state) => {
                if !state.computed {
                    while let Some(row) = state.child.next()? {
                        let (group_key, group_values) =
                            if let Some(ref group_by_exprs) = state.group_by_exprs {
                                let key = compute_group_key_from_exprs(&row, group_by_exprs);
                                let values = evaluate_group_by_exprs(&row, group_by_exprs);
                                (key, values)
                            } else {
                                let key = compute_group_key_for_dynamic(&row, &state.group_by);
                                let values: Vec<Value<'static>> = state
                                    .group_by
                                    .iter()
                                    .map(|&col| {
                                        row.get(col).map(clone_value_owned).unwrap_or(Value::Null)
                                    })
                                    .collect();
                                (key, values)
                            };

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

                        if let Some(budget) = state.memory_budget {
                            let estimated_size = state.groups.len() * 256;
                            if estimated_size > state.last_reported_bytes + 64 * 1024 {
                                let delta = estimated_size - state.last_reported_bytes;
                                budget.allocate(crate::memory::Pool::Query, delta)?;
                                state.last_reported_bytes = estimated_size;
                            }
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

                        if let Some(budget) = state.memory_budget {
                            let estimated_size = state.rows.len() * 128;
                            if estimated_size > state.last_reported_bytes + 64 * 1024 {
                                let delta = estimated_size - state.last_reported_bytes;
                                budget.allocate(crate::memory::Pool::Query, delta)?;
                                state.last_reported_bytes = estimated_size;
                            }
                        }
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

                    // 2^53 - maximum exactly representable integer in f64
                    const MAX_EXACT_INT: f64 = 9007199254740992.0;

                    for (idx, &wval) in window_vals.iter().enumerate() {
                        if wval.is_nan() {
                            result_values.push(Value::Null);
                            continue;
                        }

                        let returns_integer = state
                            .window_functions
                            .get(idx)
                            .map(|f| f.function_type.returns_integer())
                            .unwrap_or(false);

                        if returns_integer || (wval.fract() == 0.0 && wval.abs() <= MAX_EXACT_INT) {
                            result_values.push(Value::Int(wval.trunc() as i64));
                        } else {
                            result_values.push(Value::Float(wval));
                        }
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
            DynamicExecutor::TopK(state) => state.child.close(),
            DynamicExecutor::NestedLoopJoin(state) => state.left.close(),
            DynamicExecutor::GraceHashJoin(state) => {
                if let Some(ref mut spiller) = state.left_spiller {
                    spiller.cleanup()?;
                }
                if let Some(ref mut spiller) = state.right_spiller {
                    spiller.cleanup()?;
                }
                Ok(())
            }
            DynamicExecutor::StreamingHashJoin(state) => state.probe.close(),
            DynamicExecutor::HashSemiJoin(state) => state.left.close(),
            DynamicExecutor::HashAntiJoin(state) => state.left.close(),
            DynamicExecutor::HashAggregate(state) => state.child.close(),
            DynamicExecutor::Window(state) => state.child.close(),
        }
    }
}
