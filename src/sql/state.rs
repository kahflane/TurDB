use crate::sql::adapter::BTreeCursorAdapter;
use crate::sql::executor::{AggregateFunction, DynamicExecutor, ExecutorRow, RowSource, SortKey};
use crate::sql::predicate::CompiledPredicate;
use crate::types::Value;
use bumpalo::Bump;

pub struct LimitState<'a, S: RowSource> {
    pub child: Box<DynamicExecutor<'a, S>>,
    pub limit: Option<u64>,
    pub offset: Option<u64>,
    pub skipped: u64,
    pub returned: u64,
}

pub struct SortState<'a, S: RowSource> {
    pub child: Box<DynamicExecutor<'a, S>>,
    pub sort_keys: Vec<SortKey>,
    pub arena: &'a Bump,
    pub rows: Vec<Vec<Value<'static>>>,
    pub iter_idx: usize,
    pub sorted: bool,
}

pub struct HashAggregateState<'a, S: RowSource> {
    pub child: Box<DynamicExecutor<'a, S>>,
    pub group_by: Vec<usize>,
    pub aggregates: Vec<AggregateFunction>,
    pub arena: &'a Bump,
    pub groups: hashbrown::HashMap<Vec<u8>, (Vec<Value<'static>>, Vec<AggregateState>)>,
    pub result_iter: Option<std::vec::IntoIter<(Vec<Value<'static>>, Vec<AggregateState>)>>,
    pub computed: bool,
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
