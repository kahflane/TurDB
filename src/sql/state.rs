use crate::memory::MemoryBudget;
use crate::sql::adapter::BTreeCursorAdapter;
use crate::sql::ast::JoinType;
use crate::sql::executor::{AggregateFunction, DynamicExecutor, ExecutorRow, RowSource, SortKey};
use crate::sql::partition_spiller::PartitionSpiller;
use crate::sql::predicate::CompiledPredicate;
use crate::types::Value;
use bumpalo::Bump;
use smallvec::SmallVec;
use std::path::PathBuf;
use std::sync::Arc;

pub struct LimitState<'a, S: RowSource> {
    pub child: Box<DynamicExecutor<'a, S>>,
    pub limit: Option<u64>,
    pub offset: Option<u64>,
    pub skipped: u64,
    pub returned: u64,
}

pub struct SortState<'a, S: RowSource> {
    pub child: Box<DynamicExecutor<'a, S>>,
    pub sort_keys: Vec<SortKey<'a>>,
    pub arena: &'a Bump,
    pub rows: Vec<Vec<Value<'static>>>,
    pub iter_idx: usize,
    pub sorted: bool,
    pub memory_budget: Option<&'a Arc<MemoryBudget>>,
    pub last_reported_bytes: usize,
}

pub struct TopKState<'a, S: RowSource> {
    pub child: Box<DynamicExecutor<'a, S>>,
    pub sort_keys: Vec<SortKey<'a>>,
    pub arena: &'a Bump,
    pub limit: u64,
    pub offset: u64,
    pub heap: Vec<Vec<Value<'static>>>,
    pub result: Vec<Vec<Value<'static>>>,
    pub iter_idx: usize,
    pub computed: bool,
}

pub struct HashAggregateState<'a, S: RowSource> {
    pub child: Box<DynamicExecutor<'a, S>>,
    pub group_by: Vec<usize>,
    pub group_by_exprs: Option<Vec<CompiledPredicate<'a>>>,
    pub aggregates: Vec<AggregateFunction>,
    pub arena: &'a Bump,
    pub groups: hashbrown::HashMap<Vec<u8>, (Vec<Value<'static>>, Vec<AggregateState>)>,
    pub result_iter: Option<std::vec::IntoIter<(Vec<Value<'static>>, Vec<AggregateState>)>>,
    pub computed: bool,
    pub memory_budget: Option<&'a Arc<MemoryBudget>>,
    pub last_reported_bytes: usize,
}

#[derive(Debug, Clone)]
pub struct AggregateState {
    pub count: i64,
    pub sum: i64,
    pub sum_float: f64,
    pub min_int: Option<i64>,
    pub max_int: Option<i64>,
    pub min_float: Option<f64>,
    pub max_float: Option<f64>,
}

impl AggregateState {
    pub(crate) fn new() -> Self {
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

    pub(crate) fn update(&mut self, func: &AggregateFunction, row: &ExecutorRow) {
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

    pub(crate) fn finalize(&self, func: &AggregateFunction) -> Value<'static> {
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

pub struct NestedLoopJoinState<'a, S: RowSource> {
    pub left: Box<DynamicExecutor<'a, S>>,
    pub right: Box<DynamicExecutor<'a, S>>,
    pub condition: Option<CompiledPredicate<'a>>,
    pub arena: &'a Bump,
    pub current_left_row: Option<Vec<Value<'static>>>,
    pub right_rows: Vec<Vec<Value<'static>>>,
    pub right_index: usize,
    pub materialized: bool,
    pub join_type: JoinType,
    pub left_matched: bool,
    pub right_matched: Vec<bool>,
    pub emitting_unmatched_right: bool,
    pub unmatched_right_idx: usize,
    pub left_col_count: usize,
    pub right_col_count: usize,
}

pub struct GraceHashJoinState<'a, S: RowSource> {
    pub left: Box<DynamicExecutor<'a, S>>,
    pub right: Box<DynamicExecutor<'a, S>>,
    pub left_key_indices: Vec<usize>,
    pub right_key_indices: Vec<usize>,
    pub arena: &'a Bump,
    pub num_partitions: usize,
    pub left_partitions: Vec<Vec<Vec<Value<'static>>>>,
    pub right_partitions: Vec<Vec<Vec<Value<'static>>>>,
    pub current_partition: usize,
    pub partition_hash_table: hashbrown::HashMap<u64, Vec<usize>>,
    pub partition_build_rows: Vec<Vec<Value<'static>>>,
    pub current_probe_idx: usize,
    pub current_match_idx: usize,
    pub current_matches: Vec<usize>,
    pub partitioned: bool,
    pub join_type: JoinType,
    pub build_matched: Vec<Vec<bool>>,
    pub probe_row_matched: bool,
    pub emitting_unmatched_build: bool,
    pub unmatched_build_partition: usize,
    pub unmatched_build_idx: usize,
    pub left_col_count: usize,
    pub right_col_count: usize,
    pub use_spill: bool,
    pub left_spiller: Option<PartitionSpiller>,
    pub right_spiller: Option<PartitionSpiller>,
    pub spill_dir: Option<PathBuf>,
    pub memory_budget: usize,
    pub query_id: u64,
    pub probe_row_buf: SmallVec<[Value<'static>; 16]>,
    pub build_row_buf: SmallVec<[Value<'static>; 16]>,
}

pub struct StreamingHashJoinState<'a, S: RowSource> {
    pub build: Box<DynamicExecutor<'a, S>>,
    pub probe: Box<DynamicExecutor<'a, S>>,
    pub build_key_indices: SmallVec<[usize; 4]>,
    pub probe_key_indices: SmallVec<[usize; 4]>,
    pub arena: &'a Bump,
    pub hash_table: hashbrown::HashMap<u64, SmallVec<[usize; 8]>>,
    pub build_rows: Vec<Vec<Value<'static>>>,
    pub current_probe_row: Option<SmallVec<[Value<'static>; 16]>>,
    pub current_matches: SmallVec<[usize; 8]>,
    pub current_match_idx: usize,
    pub join_type: JoinType,
    pub probe_row_matched: bool,
    pub build_matched: Vec<bool>,
    pub emitting_unmatched_build: bool,
    pub unmatched_build_idx: usize,
    pub build_col_count: usize,
    pub probe_col_count: usize,
    pub built: bool,
    pub swapped: bool,
}

pub struct HashSemiJoinState<'a, S: RowSource> {
    pub left: Box<DynamicExecutor<'a, S>>,
    pub right: Box<DynamicExecutor<'a, S>>,
    pub left_key_indices: Vec<usize>,
    pub right_key_indices: Vec<usize>,
    pub arena: &'a Bump,
    pub hash_table: hashbrown::HashSet<u64>,
    pub built: bool,
    pub left_col_count: usize,
}

pub struct HashAntiJoinState<'a, S: RowSource> {
    pub left: Box<DynamicExecutor<'a, S>>,
    pub right: Box<DynamicExecutor<'a, S>>,
    pub left_key_indices: Vec<usize>,
    pub right_key_indices: Vec<usize>,
    pub arena: &'a Bump,
    pub hash_table: hashbrown::HashSet<u64>,
    pub built: bool,
    pub left_col_count: usize,
}

pub struct IndexScanState<'a> {
    pub source: BTreeCursorAdapter,
    pub arena: &'a Bump,
    pub residual_filter: Option<CompiledPredicate<'a>>,
    pub opened: bool,
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

use crate::sql::planner::WindowFunctionDef;

pub struct WindowState<'a, S: RowSource> {
    pub child: Box<DynamicExecutor<'a, S>>,
    pub window_functions: &'a [WindowFunctionDef<'a>],
    pub arena: &'a Bump,
    pub rows: Vec<Vec<Value<'static>>>,
    /// Window function results stored as f64 to preserve precision for AVG and other float operations.
    /// Integer results (ROW_NUMBER, RANK, COUNT) are stored as whole numbers.
    pub window_results: Vec<Vec<f64>>,
    pub iter_idx: usize,
    pub computed: bool,
    pub column_map: Vec<(String, usize)>,
}

impl<'a, S: RowSource> WindowState<'a, S> {
    pub fn new(
        child: Box<DynamicExecutor<'a, S>>,
        window_functions: &'a [WindowFunctionDef<'a>],
        arena: &'a Bump,
    ) -> Self {
        Self {
            child,
            window_functions,
            arena,
            rows: Vec::new(),
            window_results: Vec::new(),
            iter_idx: 0,
            computed: false,
            column_map: Vec::new(),
        }
    }

    pub fn new_with_column_map(
        child: Box<DynamicExecutor<'a, S>>,
        window_functions: &'a [WindowFunctionDef<'a>],
        arena: &'a Bump,
        column_map: Vec<(String, usize)>,
    ) -> Self {
        Self {
            child,
            window_functions,
            arena,
            rows: Vec::new(),
            window_results: Vec::new(),
            iter_idx: 0,
            computed: false,
            column_map,
        }
    }

    pub fn compute_window_functions(&mut self) {
        let num_rows = self.rows.len();
        let num_funcs = self.window_functions.len();

        self.window_results = vec![vec![0.0f64; num_funcs]; num_rows];

        for (func_idx, window_func) in self.window_functions.iter().enumerate() {
            let func_name = window_func.function_name.to_ascii_lowercase();

            let partitions = self.get_partitions(window_func);

            for partition_indices in partitions {
                let sorted_indices =
                    self.get_sorted_indices_for_partition(window_func, &partition_indices);

                match func_name.as_str() {
                    "row_number" => {
                        for (rank, &orig_idx) in sorted_indices.iter().enumerate() {
                            self.window_results[orig_idx][func_idx] = (rank + 1) as f64;
                        }
                    }
                    "rank" => {
                        let mut current_rank = 1.0f64;
                        for (sorted_pos, &orig_idx) in sorted_indices.iter().enumerate() {
                            if sorted_pos == 0 {
                                self.window_results[orig_idx][func_idx] = 1.0;
                            } else {
                                let prev_orig_idx = sorted_indices[sorted_pos - 1];
                                if self.compare_row_values(prev_orig_idx, orig_idx, window_func) {
                                    self.window_results[orig_idx][func_idx] = current_rank;
                                } else {
                                    current_rank = (sorted_pos + 1) as f64;
                                    self.window_results[orig_idx][func_idx] = current_rank;
                                }
                            }
                        }
                    }
                    "dense_rank" => {
                        let mut current_rank = 1.0f64;
                        for (sorted_pos, &orig_idx) in sorted_indices.iter().enumerate() {
                            if sorted_pos == 0 {
                                self.window_results[orig_idx][func_idx] = 1.0;
                            } else {
                                let prev_orig_idx = sorted_indices[sorted_pos - 1];
                                if self.compare_row_values(prev_orig_idx, orig_idx, window_func) {
                                    self.window_results[orig_idx][func_idx] = current_rank;
                                } else {
                                    current_rank += 1.0;
                                    self.window_results[orig_idx][func_idx] = current_rank;
                                }
                            }
                        }
                    }
                    "sum" => {
                        let values: Vec<f64> = partition_indices
                            .iter()
                            .filter_map(|&row_idx| self.get_arg_value(row_idx, window_func))
                            .collect();
                        let result = if values.is_empty() {
                            f64::NAN
                        } else {
                            values.iter().sum()
                        };
                        for &row_idx in &partition_indices {
                            self.window_results[row_idx][func_idx] = result;
                        }
                    }
                    "count" => {
                        let count = if window_func.args.is_empty() {
                            partition_indices.len() as f64
                        } else {
                            partition_indices
                                .iter()
                                .filter(|&&row_idx| {
                                    self.get_arg_value(row_idx, window_func).is_some()
                                })
                                .count() as f64
                        };
                        for &row_idx in &partition_indices {
                            self.window_results[row_idx][func_idx] = count;
                        }
                    }
                    "avg" => {
                        let values: Vec<f64> = partition_indices
                            .iter()
                            .filter_map(|&row_idx| self.get_arg_value(row_idx, window_func))
                            .collect();
                        let result = if values.is_empty() {
                            f64::NAN
                        } else {
                            values.iter().sum::<f64>() / values.len() as f64
                        };
                        for &row_idx in &partition_indices {
                            self.window_results[row_idx][func_idx] = result;
                        }
                    }
                    "min" => {
                        let values: Vec<f64> = partition_indices
                            .iter()
                            .filter_map(|&row_idx| self.get_arg_value(row_idx, window_func))
                            .collect();
                        let result = if values.is_empty() {
                            f64::NAN
                        } else {
                            values.iter().cloned().fold(f64::INFINITY, f64::min)
                        };
                        for &row_idx in &partition_indices {
                            self.window_results[row_idx][func_idx] = result;
                        }
                    }
                    "max" => {
                        let values: Vec<f64> = partition_indices
                            .iter()
                            .filter_map(|&row_idx| self.get_arg_value(row_idx, window_func))
                            .collect();
                        let result = if values.is_empty() {
                            f64::NAN
                        } else {
                            values.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
                        };
                        for &row_idx in &partition_indices {
                            self.window_results[row_idx][func_idx] = result;
                        }
                    }
                    _ => {
                        for &row_idx in &partition_indices {
                            self.window_results[row_idx][func_idx] = 0.0;
                        }
                    }
                }
            }
        }
    }

    /// Gets the numeric value of the first argument for aggregate window functions.
    /// Returns Some(f64) for numeric values, None for NULL values.
    /// SQL standard: NULLs are ignored in aggregate functions (SUM, AVG, MIN, MAX).
    fn get_arg_value(&self, row_idx: usize, window_func: &WindowFunctionDef<'a>) -> Option<f64> {
        let crate::sql::ast::Expr::Column(col_ref) = window_func.args.first()? else {
            return None;
        };
        let col_idx = self.find_column_index(col_ref.column)?;
        let val = self.rows.get(row_idx).and_then(|r| r.get(col_idx))?;
        match val {
            Value::Int(i) => Some(*i as f64),
            Value::Float(f) => Some(*f),
            Value::Null => None,
            _ => None,
        }
    }

    fn get_partitions(&self, window_func: &WindowFunctionDef<'a>) -> Vec<Vec<usize>> {
        let num_rows = self.rows.len();

        if window_func.partition_by.is_empty() {
            return vec![(0..num_rows).collect()];
        }

        let mut partition_map: hashbrown::HashMap<Vec<u8>, Vec<usize>> = hashbrown::HashMap::new();

        for row_idx in 0..num_rows {
            let key = self.get_partition_key(row_idx, window_func);
            partition_map.entry(key).or_default().push(row_idx);
        }

        partition_map.into_values().collect()
    }

    fn get_partition_key(&self, row_idx: usize, window_func: &WindowFunctionDef<'a>) -> Vec<u8> {
        let mut key = Vec::new();

        for partition_expr in window_func.partition_by.iter() {
            if let crate::sql::ast::Expr::Column(col_ref) = partition_expr {
                if let Some(col_idx) = self.find_column_index(col_ref.column) {
                    if let Some(val) = self.rows.get(row_idx).and_then(|r| r.get(col_idx)) {
                        match val {
                            Value::Int(i) => key.extend(i.to_be_bytes()),
                            Value::Float(f) => key.extend(f.to_be_bytes()),
                            Value::Text(t) => {
                                key.extend(t.as_bytes());
                                key.push(0);
                            }
                            Value::Blob(b) => {
                                key.extend(b.as_ref());
                                key.push(0);
                            }
                            Value::Null => key.push(0xFF),
                            _ => {}
                        }
                    }
                }
            }
        }

        key
    }

    fn get_sorted_indices_for_partition(
        &self,
        window_func: &WindowFunctionDef<'a>,
        partition_indices: &[usize],
    ) -> Vec<usize> {
        let mut indices: Vec<usize> = partition_indices.to_vec();

        if window_func.order_by.is_empty() {
            return indices;
        }

        indices.sort_by(|&a, &b| {
            for sort_key in window_func.order_by.iter() {
                if let crate::sql::ast::Expr::Column(col_ref) = sort_key.expr {
                    if let Some(col_idx) = self.find_column_index(col_ref.column) {
                        let val_a = self.rows.get(a).and_then(|r| r.get(col_idx));
                        let val_b = self.rows.get(b).and_then(|r| r.get(col_idx));

                        let cmp = match (val_a, val_b) {
                            (Some(Value::Int(ia)), Some(Value::Int(ib))) => ia.cmp(ib),
                            (Some(Value::Float(fa)), Some(Value::Float(fb))) => {
                                fa.partial_cmp(fb).unwrap_or(std::cmp::Ordering::Equal)
                            }
                            (Some(Value::Text(ta)), Some(Value::Text(tb))) => ta.cmp(tb),
                            _ => std::cmp::Ordering::Equal,
                        };

                        let cmp = if sort_key.ascending {
                            cmp
                        } else {
                            cmp.reverse()
                        };

                        if cmp != std::cmp::Ordering::Equal {
                            return cmp;
                        }
                    }
                }
            }
            std::cmp::Ordering::Equal
        });

        indices
    }

    fn compare_row_values(
        &self,
        idx1: usize,
        idx2: usize,
        window_func: &WindowFunctionDef<'a>,
    ) -> bool {
        if window_func.order_by.is_empty() {
            return false;
        }

        for sort_key in window_func.order_by.iter() {
            if let crate::sql::ast::Expr::Column(col_ref) = sort_key.expr {
                if let Some(col_idx) = self.find_column_index(col_ref.column) {
                    let val1 = self.rows.get(idx1).and_then(|r| r.get(col_idx));
                    let val2 = self.rows.get(idx2).and_then(|r| r.get(col_idx));

                    match (val1, val2) {
                        (Some(Value::Int(a)), Some(Value::Int(b))) => {
                            if a != b {
                                return false;
                            }
                        }
                        (Some(Value::Float(a)), Some(Value::Float(b))) => {
                            if (a - b).abs() > f64::EPSILON {
                                return false;
                            }
                        }
                        (Some(Value::Text(a)), Some(Value::Text(b))) => {
                            if a != b {
                                return false;
                            }
                        }
                        _ => return false,
                    }
                }
            }
        }
        true
    }

    fn find_column_index(&self, col_name: &str) -> Option<usize> {
        self.column_map
            .iter()
            .find(|(name, _)| name.eq_ignore_ascii_case(col_name))
            .map(|(_, idx)| *idx)
    }
}
