use crate::sql::executor::ExecutorRow;
use crate::sql::state::GroupValues;
use crate::types::Value;
use bumpalo::Bump;
use std::cmp::Ordering;

/// Allocates a value to an arena, returning a value with the arena's lifetime.
#[inline]
pub fn allocate_value_to_arena<'a>(v: Value<'_>, arena: &'a Bump) -> Value<'a> {
    v.clone_to_arena(arena)
}

/// Clones a value reference to an arena, returning a value with the arena's lifetime.
#[inline]
pub fn clone_value_ref_to_arena<'a>(v: &Value<'_>, arena: &'a Bump) -> Value<'a> {
    v.clone_to_arena(arena)
}

/// Clones a value to a fully-owned static lifetime.
#[inline]
pub fn clone_value_owned(v: &Value<'_>) -> Value<'static> {
    v.to_owned_static()
}

/// Encodes a value to a byte-comparable key format.
#[inline]
pub fn encode_value_to_key(v: &Value<'_>, key: &mut Vec<u8>) {
    v.encode_to_key(key)
}

/// Hashes a value for use in hash tables.
#[inline]
pub fn hash_value<H: std::hash::Hasher>(v: &Value<'_>, hasher: &mut H) {
    v.hash_to(hasher)
}

/// Computes a group key for dynamic grouping.
pub fn compute_group_key_for_dynamic(row: &ExecutorRow, group_by: &[usize]) -> Vec<u8> {
    let mut key = Vec::new();
    for &col in group_by {
        if let Some(val) = row.get(col) {
            val.encode_to_key(&mut key);
        }
    }
    key
}

/// Computes a group key from expressions for dynamic grouping.
pub fn compute_group_key_from_exprs<'a>(
    row: &ExecutorRow<'a>,
    group_by_exprs: &[crate::sql::predicate::CompiledPredicate<'a>],
) -> Vec<u8> {
    let mut key = Vec::new();
    for expr in group_by_exprs {
        if let Some(val) = expr.evaluate_to_value(row) {
            val.encode_to_key(&mut key);
        }
    }
    key
}

/// Evaluates group by expressions and returns the values.
pub fn evaluate_group_by_exprs<'a>(
    row: &ExecutorRow<'a>,
    group_by_exprs: &[crate::sql::predicate::CompiledPredicate<'a>],
) -> GroupValues {
    group_by_exprs
        .iter()
        .map(|expr| {
            expr.evaluate_to_value(row)
                .map(|v| v.to_owned_static())
                .unwrap_or(Value::Null)
        })
        .collect()
}

/// Compares two values for sorting.
#[inline]
pub fn compare_values_for_sort(a: &Value, b: &Value) -> Ordering {
    a.compare_for_sort(b)
}

/// Hashes row keys for hash join operations.
pub fn hash_keys<'a>(row: &ExecutorRow<'a>, key_indices: &[usize]) -> u64 {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::Hasher;

    let mut hasher = DefaultHasher::new();
    for &idx in key_indices {
        if let Some(val) = row.get(idx) {
            val.hash_to(&mut hasher);
        }
    }
    hasher.finish()
}

/// Hashes static row keys for hash join operations.
pub fn hash_keys_static(row: &[Value<'static>], key_indices: &[usize]) -> u64 {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::Hasher;

    let mut hasher = DefaultHasher::new();
    for &idx in key_indices {
        if let Some(val) = row.get(idx) {
            val.hash_to(&mut hasher);
        }
    }
    hasher.finish()
}

/// Checks if keys match between two static value rows.
pub fn keys_match_static(
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
            (Some(l), Some(r)) => {
                if l.compare(r) != Some(Ordering::Equal) {
                    return false;
                }
            }
            _ => return false,
        }
    }
    true
}
