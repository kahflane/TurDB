use crate::sql::executor::ExecutorRow;
use crate::types::Value;
use bumpalo::Bump;
use std::borrow::Cow;

pub fn allocate_value_to_arena<'a>(v: Value<'_>, arena: &'a Bump) -> Value<'a> {
    match v {
        Value::Null => Value::Null,
        Value::Int(i) => Value::Int(i),
        Value::Float(f) => Value::Float(f),
        Value::Text(s) => Value::Text(Cow::Borrowed(arena.alloc_str(&s))),
        Value::Blob(b) => Value::Blob(Cow::Borrowed(arena.alloc_slice_copy(&b))),
        Value::Vector(v) => Value::Vector(Cow::Borrowed(arena.alloc_slice_copy(&v))),
        Value::Uuid(u) => Value::Uuid(u),
        Value::MacAddr(m) => Value::MacAddr(m),
        Value::Inet4(ip) => Value::Inet4(ip),
        Value::Inet6(ip) => Value::Inet6(ip),
        Value::Jsonb(b) => Value::Jsonb(Cow::Borrowed(arena.alloc_slice_copy(&b))),
        Value::TimestampTz {
            micros,
            offset_secs,
        } => Value::TimestampTz {
            micros,
            offset_secs,
        },
        Value::Interval {
            micros,
            days,
            months,
        } => Value::Interval {
            micros,
            days,
            months,
        },
        Value::Point { x, y } => Value::Point { x, y },
        Value::GeoBox { low, high } => Value::GeoBox { low, high },
        Value::Circle { center, radius } => Value::Circle { center, radius },
        Value::Enum { type_id, ordinal } => Value::Enum { type_id, ordinal },
        Value::Decimal { digits, scale } => Value::Decimal { digits, scale },
    }
}

pub fn clone_value_ref_to_arena<'a>(v: &Value<'_>, arena: &'a Bump) -> Value<'a> {
    match v {
        Value::Null => Value::Null,
        Value::Int(i) => Value::Int(*i),
        Value::Float(f) => Value::Float(*f),
        Value::Text(s) => Value::Text(Cow::Borrowed(arena.alloc_str(s))),
        Value::Blob(b) => Value::Blob(Cow::Borrowed(arena.alloc_slice_copy(b))),
        Value::Vector(v) => Value::Vector(Cow::Borrowed(arena.alloc_slice_copy(v))),
        Value::Uuid(u) => Value::Uuid(*u),
        Value::MacAddr(m) => Value::MacAddr(*m),
        Value::Inet4(ip) => Value::Inet4(*ip),
        Value::Inet6(ip) => Value::Inet6(*ip),
        Value::Jsonb(b) => Value::Jsonb(Cow::Borrowed(arena.alloc_slice_copy(b))),
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
    }
}

pub fn clone_value_owned(v: &Value<'_>) -> Value<'static> {
    match v {
        Value::Null => Value::Null,
        Value::Int(i) => Value::Int(*i),
        Value::Float(f) => Value::Float(*f),
        Value::Text(s) => Value::Text(Cow::Owned(s.to_string())),
        Value::Blob(b) => Value::Blob(Cow::Owned(b.to_vec())),
        Value::Vector(v) => Value::Vector(Cow::Owned(v.to_vec())),
        Value::Uuid(u) => Value::Uuid(*u),
        Value::MacAddr(m) => Value::MacAddr(*m),
        Value::Inet4(ip) => Value::Inet4(*ip),
        Value::Inet6(ip) => Value::Inet6(*ip),
        Value::Jsonb(b) => Value::Jsonb(Cow::Owned(b.to_vec())),
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
    }
}

pub fn encode_value_to_key(v: &Value<'_>, key: &mut Vec<u8>) {
    match v {
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
        Value::Uuid(u) => {
            key.push(6);
            key.extend(u.iter());
        }
        Value::MacAddr(m) => {
            key.push(7);
            key.extend(m.iter());
        }
        Value::Inet4(ip) => {
            key.push(8);
            key.extend(ip.iter());
        }
        Value::Inet6(ip) => {
            key.push(9);
            key.extend(ip.iter());
        }
        Value::Jsonb(b) => {
            key.push(10);
            key.extend(b.iter());
            key.push(0);
        }
        Value::TimestampTz {
            micros,
            offset_secs,
        } => {
            key.push(11);
            key.extend(micros.to_be_bytes());
            key.extend(offset_secs.to_be_bytes());
        }
        Value::Interval {
            micros,
            days,
            months,
        } => {
            key.push(12);
            key.extend(micros.to_be_bytes());
            key.extend(days.to_be_bytes());
            key.extend(months.to_be_bytes());
        }
        Value::Point { x, y } => {
            key.push(13);
            key.extend(x.to_bits().to_be_bytes());
            key.extend(y.to_bits().to_be_bytes());
        }
        Value::GeoBox { low, high } => {
            key.push(14);
            key.extend(low.0.to_bits().to_be_bytes());
            key.extend(low.1.to_bits().to_be_bytes());
            key.extend(high.0.to_bits().to_be_bytes());
            key.extend(high.1.to_bits().to_be_bytes());
        }
        Value::Circle { center, radius } => {
            key.push(15);
            key.extend(center.0.to_bits().to_be_bytes());
            key.extend(center.1.to_bits().to_be_bytes());
            key.extend(radius.to_bits().to_be_bytes());
        }
        Value::Enum { type_id, ordinal } => {
            key.push(16);
            key.extend(type_id.to_be_bytes());
            key.extend(ordinal.to_be_bytes());
        }
        Value::Decimal { digits, scale } => {
            key.push(17);
            key.extend(digits.to_be_bytes());
            key.extend(scale.to_be_bytes());
        }
    }
}

pub fn hash_value<H: std::hash::Hasher>(v: &Value<'_>, hasher: &mut H) {
    use std::hash::Hash;
    match v {
        Value::Null => 0u8.hash(hasher),
        Value::Int(i) => i.hash(hasher),
        Value::Float(f) => f.to_bits().hash(hasher),
        Value::Text(s) => s.hash(hasher),
        Value::Blob(b) => b.hash(hasher),
        Value::Vector(v) => {
            for f in v.iter() {
                f.to_bits().hash(hasher);
            }
        }
        Value::Uuid(u) => u.hash(hasher),
        Value::MacAddr(m) => m.hash(hasher),
        Value::Inet4(ip) => ip.hash(hasher),
        Value::Inet6(ip) => ip.hash(hasher),
        Value::Jsonb(b) => b.hash(hasher),
        Value::TimestampTz {
            micros,
            offset_secs,
        } => {
            micros.hash(hasher);
            offset_secs.hash(hasher);
        }
        Value::Interval {
            micros,
            days,
            months,
        } => {
            micros.hash(hasher);
            days.hash(hasher);
            months.hash(hasher);
        }
        Value::Point { x, y } => {
            x.to_bits().hash(hasher);
            y.to_bits().hash(hasher);
        }
        Value::GeoBox { low, high } => {
            low.0.to_bits().hash(hasher);
            low.1.to_bits().hash(hasher);
            high.0.to_bits().hash(hasher);
            high.1.to_bits().hash(hasher);
        }
        Value::Circle { center, radius } => {
            center.0.to_bits().hash(hasher);
            center.1.to_bits().hash(hasher);
            radius.to_bits().hash(hasher);
        }
        Value::Enum { type_id, ordinal } => {
            type_id.hash(hasher);
            ordinal.hash(hasher);
        }
        Value::Decimal { digits, scale } => {
            digits.hash(hasher);
            scale.hash(hasher);
        }
    }
}

pub fn compute_group_key_for_dynamic(row: &ExecutorRow, group_by: &[usize]) -> Vec<u8> {
    let mut key = Vec::new();
    for &col in group_by {
        if let Some(val) = row.get(col) {
            encode_value_to_key(val, &mut key);
        }
    }
    key
}

pub fn compare_values_for_sort(a: &Value, b: &Value) -> std::cmp::Ordering {
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

pub fn hash_keys<'a>(row: &ExecutorRow<'a>, key_indices: &[usize]) -> u64 {
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

pub fn hash_keys_static(row: &[Value<'static>], key_indices: &[usize]) -> u64 {
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
            (Some(Value::Int(a)), Some(Value::Int(b))) if a != b => return false,
            (Some(Value::Float(a)), Some(Value::Float(b))) if (a - b).abs() > f64::EPSILON => {
                return false
            }
            (Some(Value::Text(a)), Some(Value::Text(b))) if a != b => return false,
            (Some(Value::Blob(a)), Some(Value::Blob(b))) if a != b => return false,
            (Some(_), Some(_)) => {}
            _ => return false,
        }
    }
    true
}
