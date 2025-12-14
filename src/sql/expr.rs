use crate::sql::executor::{ExecutorRow};
use crate::types::Value;

pub struct ExprEvaluator {
    column_map: Vec<(String, usize)>,
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
            Some(col_val) => {
                Self::compare_values(col_val, value) == Some(std::cmp::Ordering::Equal)
            }
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
            Some(col_val) => {
                Self::compare_values(col_val, value) != Some(std::cmp::Ordering::Equal)
            }
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