//! # Runtime Value Representation
//!
//! This module provides `Value<'a>`, the runtime representation for SQL values.
//! Values use `Cow` for text/blob/vector types to enable zero-copy when reading
//! from mmap'd pages while supporting owned data when needed.
//!
//! ## Design Goals
//!
//! 1. **Zero-copy**: Borrow directly from page data when possible
//! 2. **Type safety**: Strongly typed variants prevent invalid operations
//! 3. **SQL semantics**: NULL comparison returns UNKNOWN (None)
//! 4. **Efficient comparison**: Direct value comparison without type coercion
//!
//! ## Value Variants
//!
//! | Variant | Rust Type | Description |
//! |---------|-----------|-------------|
//! | Null | - | SQL NULL |
//! | Int | i64 | 64-bit signed integer |
//! | Float | f64 | 64-bit floating point |
//! | Text | Cow<str> | UTF-8 string |
//! | Blob | Cow<[u8]> | Binary data |
//! | Vector | Cow<[f32]> | Float32 vector |
//! | Uuid | [u8; 16] | UUID bytes |
//! | MacAddr | [u8; 6] | MAC address |
//! | Inet4 | [u8; 4] | IPv4 address |
//! | Inet6 | [u8; 16] | IPv6 address |
//! | Jsonb | Cow<[u8]> | Binary JSON |
//! | TimestampTz | {micros, offset} | Timestamp with timezone |
//! | Interval | {micros, days, months} | Time interval |
//! | Point | {x, y} | 2D point |
//! | GeoBox | {low, high} | Bounding box |
//! | Circle | {center, radius} | Circle |
//! | Enum | {type_id, ordinal} | Enum value |
//! | Decimal | {digits, scale} | Arbitrary precision decimal |
//!
//! ## Comparison Semantics
//!
//! - NULL compared to anything returns None (SQL UNKNOWN)
//! - Int vs Float: Int promoted to Float for comparison
//! - Cross-type ordering: Int/Float < Text < Blob < Vector < special types
//!
//! ## Memory Layout
//!
//! ```text
//! Value enum (~48 bytes on 64-bit):
//! +----------------+
//! | discriminant   |
//! +----------------+
//! | data (varies)  |
//! +----------------+
//! ```

use super::TypeAffinity;
use eyre::{bail, Result};
use std::borrow::Cow;
use std::cmp::Ordering;

/// Runtime value representation for SQL values.
#[derive(Debug, Clone, PartialEq)]
pub enum Value<'a> {
    Null,
    Int(i64),
    Float(f64),
    Text(Cow<'a, str>),
    Blob(Cow<'a, [u8]>),
    Vector(Cow<'a, [f32]>),
    Uuid([u8; 16]),
    MacAddr([u8; 6]),
    Inet4([u8; 4]),
    Inet6([u8; 16]),
    Jsonb(Cow<'a, [u8]>),
    TimestampTz { micros: i64, offset_secs: i32 },
    Interval { micros: i64, days: i32, months: i32 },
    Point { x: f64, y: f64 },
    GeoBox { low: (f64, f64), high: (f64, f64) },
    Circle { center: (f64, f64), radius: f64 },
    Enum { type_id: u16, ordinal: u16 },
    Decimal { digits: i128, scale: i16 },
}

impl<'a> Value<'a> {
    /// Returns true if this value is NULL.
    pub fn is_null(&self) -> bool {
        matches!(self, Value::Null)
    }

    /// Coerces this value to the target type affinity.
    pub fn coerce_to_affinity(&self, target: TypeAffinity) -> Result<Value<'a>> {
        match self {
            Value::Null => Ok(Value::Null),
            Value::Int(i) => match target {
                TypeAffinity::Integer | TypeAffinity::Numeric => Ok(Value::Int(*i)),
                TypeAffinity::Real => Ok(Value::Float(*i as f64)),
                TypeAffinity::Text => Ok(Value::Text(Cow::Owned(i.to_string()))),
                TypeAffinity::Blob => Ok(Value::Blob(Cow::Owned(i.to_le_bytes().to_vec()))),
            },
            Value::Float(f) => match target {
                TypeAffinity::Real | TypeAffinity::Numeric => Ok(Value::Float(*f)),
                TypeAffinity::Integer => Ok(Value::Int(*f as i64)),
                TypeAffinity::Text => Ok(Value::Text(Cow::Owned(f.to_string()))),
                TypeAffinity::Blob => Ok(Value::Blob(Cow::Owned(f.to_le_bytes().to_vec()))),
            },
            Value::Text(s) => match target {
                TypeAffinity::Text => Ok(Value::Text(s.clone())),
                TypeAffinity::Integer => {
                    let parsed = s.trim().parse::<i64>().map_err(|e| {
                        eyre::eyre!("cannot coerce text '{}' to integer: {}", s, e)
                    })?;
                    Ok(Value::Int(parsed))
                }
                TypeAffinity::Real | TypeAffinity::Numeric => {
                    let parsed = s.trim().parse::<f64>().map_err(|e| {
                        eyre::eyre!("cannot coerce text '{}' to real: {}", s, e)
                    })?;
                    Ok(Value::Float(parsed))
                }
                TypeAffinity::Blob => Ok(Value::Blob(Cow::Owned(s.as_bytes().to_vec()))),
            },
            Value::Blob(b) => match target {
                TypeAffinity::Blob => Ok(Value::Blob(b.clone())),
                TypeAffinity::Text => {
                    let s = std::str::from_utf8(b)
                        .map_err(|e| eyre::eyre!("cannot coerce blob to text: {}", e))?;
                    Ok(Value::Text(Cow::Owned(s.to_string())))
                }
                _ => bail!("cannot coerce blob to affinity {:?}", target),
            },
            Value::Vector(v) => match target {
                TypeAffinity::Blob => Ok(Value::Vector(v.clone())),
                _ => bail!("cannot coerce vector to affinity {:?}", target),
            },
            Value::Uuid(u) => match target {
                TypeAffinity::Blob => Ok(Value::Uuid(*u)),
                TypeAffinity::Text => {
                    let hex = u
                        .iter()
                        .map(|b| format!("{:02x}", b))
                        .collect::<Vec<_>>()
                        .join("");
                    let formatted = format!(
                        "{}-{}-{}-{}-{}",
                        &hex[0..8],
                        &hex[8..12],
                        &hex[12..16],
                        &hex[16..20],
                        &hex[20..32]
                    );
                    Ok(Value::Text(Cow::Owned(formatted)))
                }
                _ => bail!("cannot coerce uuid to affinity {:?}", target),
            },
            Value::MacAddr(m) => match target {
                TypeAffinity::Blob => Ok(Value::MacAddr(*m)),
                TypeAffinity::Text => {
                    let formatted = m
                        .iter()
                        .map(|b| format!("{:02x}", b))
                        .collect::<Vec<_>>()
                        .join(":");
                    Ok(Value::Text(Cow::Owned(formatted)))
                }
                _ => bail!("cannot coerce macaddr to affinity {:?}", target),
            },
            Value::Inet4(ip) => match target {
                TypeAffinity::Blob => Ok(Value::Inet4(*ip)),
                TypeAffinity::Text => Ok(Value::Text(Cow::Owned(format!(
                    "{}.{}.{}.{}",
                    ip[0], ip[1], ip[2], ip[3]
                )))),
                _ => bail!("cannot coerce inet4 to affinity {:?}", target),
            },
            Value::Inet6(ip) => match target {
                TypeAffinity::Blob => Ok(Value::Inet6(*ip)),
                TypeAffinity::Text => {
                    let parts: Vec<String> = (0..8)
                        .map(|i| format!("{:04x}", u16::from_be_bytes([ip[i * 2], ip[i * 2 + 1]])))
                        .collect();
                    Ok(Value::Text(Cow::Owned(parts.join(":"))))
                }
                _ => bail!("cannot coerce inet6 to affinity {:?}", target),
            },
            Value::Jsonb(b) => match target {
                TypeAffinity::Blob => Ok(Value::Jsonb(b.clone())),
                TypeAffinity::Text => Ok(Value::Text(Cow::Owned(format!(
                    "<jsonb:{} bytes>",
                    b.len()
                )))),
                _ => bail!("cannot coerce jsonb to affinity {:?}", target),
            },
            Value::TimestampTz {
                micros,
                offset_secs,
            } => match target {
                TypeAffinity::Integer | TypeAffinity::Numeric => Ok(Value::Int(*micros)),
                TypeAffinity::Text => Ok(Value::Text(Cow::Owned(format!(
                    "{}+{}",
                    micros, offset_secs
                )))),
                _ => bail!("cannot coerce timestamptz to affinity {:?}", target),
            },
            Value::Interval {
                micros,
                days,
                months,
            } => match target {
                TypeAffinity::Integer | TypeAffinity::Numeric => Ok(Value::Int(*micros)),
                TypeAffinity::Text => Ok(Value::Text(Cow::Owned(format!(
                    "{} months {} days {} us",
                    months, days, micros
                )))),
                _ => bail!("cannot coerce interval to affinity {:?}", target),
            },
            Value::Point { x, y } => match target {
                TypeAffinity::Text => Ok(Value::Text(Cow::Owned(format!("({},{})", x, y)))),
                _ => bail!("cannot coerce point to affinity {:?}", target),
            },
            Value::GeoBox { low, high } => match target {
                TypeAffinity::Text => Ok(Value::Text(Cow::Owned(format!(
                    "(({},{}),({},{}))",
                    low.0, low.1, high.0, high.1
                )))),
                _ => bail!("cannot coerce box to affinity {:?}", target),
            },
            Value::Circle { center, radius } => match target {
                TypeAffinity::Text => Ok(Value::Text(Cow::Owned(format!(
                    "<({},{}),{}>",
                    center.0, center.1, radius
                )))),
                _ => bail!("cannot coerce circle to affinity {:?}", target),
            },
            Value::Enum { type_id, ordinal } => match target {
                TypeAffinity::Integer | TypeAffinity::Numeric => {
                    Ok(Value::Int(((*type_id as i64) << 16) | (*ordinal as i64)))
                }
                TypeAffinity::Text => Ok(Value::Text(Cow::Owned(format!(
                    "enum({}:{})",
                    type_id, ordinal
                )))),
                _ => bail!("cannot coerce enum to affinity {:?}", target),
            },
            Value::Decimal { digits, scale } => match target {
                TypeAffinity::Integer | TypeAffinity::Numeric => {
                    if *scale <= 0 {
                        Ok(Value::Int(*digits as i64))
                    } else {
                        let divisor = 10i128.pow(*scale as u32);
                        Ok(Value::Int((*digits / divisor) as i64))
                    }
                }
                TypeAffinity::Real => {
                    let divisor = 10f64.powi(*scale as i32);
                    Ok(Value::Float(*digits as f64 / divisor))
                }
                TypeAffinity::Text => {
                    if *scale <= 0 {
                        Ok(Value::Text(Cow::Owned(format!("{}", digits))))
                    } else {
                        let divisor = 10i128.pow(*scale as u32);
                        let int_part = *digits / divisor;
                        let frac_part = (*digits % divisor).abs();
                        Ok(Value::Text(Cow::Owned(format!(
                            "{}.{:0>width$}",
                            int_part,
                            frac_part,
                            width = *scale as usize
                        ))))
                    }
                }
                _ => bail!("cannot coerce decimal to affinity {:?}", target),
            },
        }
    }

    /// Compares two values with SQL NULL semantics.
    /// Returns None if either value is NULL (SQL UNKNOWN).
    pub fn compare(&self, other: &Value) -> Option<Ordering> {
        match (self, other) {
            (Value::Null, _) | (_, Value::Null) => None,

            (Value::Int(a), Value::Int(b)) => Some(a.cmp(b)),
            (Value::Float(a), Value::Float(b)) => {
                if a.is_nan() || b.is_nan() {
                    None
                } else {
                    a.partial_cmp(b)
                }
            }

            (Value::Int(i), Value::Float(f)) => {
                let i_as_float = *i as f64;
                if f.is_nan() {
                    None
                } else {
                    i_as_float.partial_cmp(f)
                }
            }

            (Value::Float(f), Value::Int(i)) => {
                let i_as_float = *i as f64;
                if f.is_nan() {
                    None
                } else {
                    f.partial_cmp(&i_as_float)
                }
            }

            (Value::Text(a), Value::Text(b)) => Some(a.cmp(b)),
            (Value::Blob(a), Value::Blob(b)) => Some(a.cmp(b)),
            (Value::Vector(a), Value::Vector(b)) => {
                for (x, y) in a.iter().zip(b.iter()) {
                    if x.is_nan() || y.is_nan() {
                        return None;
                    }
                    match x.partial_cmp(y) {
                        Some(Ordering::Equal) => continue,
                        other => return other,
                    }
                }
                Some(a.len().cmp(&b.len()))
            }

            (Value::Int(_) | Value::Float(_), Value::Text(_))
            | (Value::Int(_) | Value::Float(_), Value::Blob(_))
            | (Value::Int(_) | Value::Float(_), Value::Vector(_)) => Some(Ordering::Less),

            (Value::Text(_), Value::Int(_) | Value::Float(_)) => Some(Ordering::Greater),
            (Value::Text(_), Value::Blob(_) | Value::Vector(_)) => Some(Ordering::Less),

            (Value::Blob(_) | Value::Vector(_), Value::Int(_) | Value::Float(_)) => {
                Some(Ordering::Greater)
            }
            (Value::Blob(_), Value::Text(_)) => Some(Ordering::Greater),
            (Value::Vector(_), Value::Text(_)) => Some(Ordering::Greater),

            (Value::Blob(_), Value::Vector(_)) => Some(Ordering::Less),
            (Value::Vector(_), Value::Blob(_)) => Some(Ordering::Greater),

            (Value::Uuid(a), Value::Uuid(b)) => Some(a.cmp(b)),
            (Value::MacAddr(a), Value::MacAddr(b)) => Some(a.cmp(b)),
            (Value::Inet4(a), Value::Inet4(b)) => Some(a.cmp(b)),
            (Value::Inet6(a), Value::Inet6(b)) => Some(a.cmp(b)),
            (Value::Jsonb(a), Value::Jsonb(b)) => Some(a.cmp(b)),

            (Value::TimestampTz { micros: a, .. }, Value::TimestampTz { micros: b, .. }) => {
                Some(a.cmp(b))
            }
            (
                Value::Interval {
                    micros: a,
                    days: ad,
                    months: am,
                },
                Value::Interval {
                    micros: b,
                    days: bd,
                    months: bm,
                },
            ) => Some((am, ad, a).cmp(&(bm, bd, b))),
            (Value::Point { x: ax, y: ay }, Value::Point { x: bx, y: by }) => {
                ax.partial_cmp(bx).and_then(|o| {
                    if o == Ordering::Equal {
                        ay.partial_cmp(by)
                    } else {
                        Some(o)
                    }
                })
            }
            (Value::GeoBox { low: al, high: ah }, Value::GeoBox { low: bl, high: bh }) => {
                al.0.partial_cmp(&bl.0).and_then(|o| {
                    if o == Ordering::Equal {
                        al.1.partial_cmp(&bl.1).and_then(|o| {
                            if o == Ordering::Equal {
                                ah.0.partial_cmp(&bh.0).and_then(|o| {
                                    if o == Ordering::Equal {
                                        ah.1.partial_cmp(&bh.1)
                                    } else {
                                        Some(o)
                                    }
                                })
                            } else {
                                Some(o)
                            }
                        })
                    } else {
                        Some(o)
                    }
                })
            }
            (
                Value::Circle {
                    center: ac,
                    radius: ar,
                },
                Value::Circle {
                    center: bc,
                    radius: br,
                },
            ) => ac.0.partial_cmp(&bc.0).and_then(|o| {
                if o == Ordering::Equal {
                    ac.1.partial_cmp(&bc.1).and_then(|o| {
                        if o == Ordering::Equal {
                            ar.partial_cmp(br)
                        } else {
                            Some(o)
                        }
                    })
                } else {
                    Some(o)
                }
            }),
            (
                Value::Enum {
                    type_id: at,
                    ordinal: ao,
                },
                Value::Enum {
                    type_id: bt,
                    ordinal: bo,
                },
            ) => Some((at, ao).cmp(&(bt, bo))),
            (
                Value::Decimal {
                    digits: a,
                    scale: as_,
                },
                Value::Decimal {
                    digits: b,
                    scale: bs_,
                },
            ) => {
                if as_ == bs_ {
                    Some(a.cmp(b))
                } else {
                    let max_scale = (*as_).max(*bs_);
                    let a_scaled = if *as_ < max_scale {
                        *a * 10i128.pow((max_scale - *as_) as u32)
                    } else {
                        *a
                    };
                    let b_scaled = if *bs_ < max_scale {
                        *b * 10i128.pow((max_scale - *bs_) as u32)
                    } else {
                        *b
                    };
                    Some(a_scaled.cmp(&b_scaled))
                }
            }

            (Value::Uuid(_), _) => Some(Ordering::Greater),
            (_, Value::Uuid(_)) => Some(Ordering::Less),

            (Value::MacAddr(_), _) => Some(Ordering::Greater),
            (_, Value::MacAddr(_)) => Some(Ordering::Less),

            (Value::Inet4(_), _) => Some(Ordering::Greater),
            (_, Value::Inet4(_)) => Some(Ordering::Less),

            (Value::Inet6(_), _) => Some(Ordering::Greater),
            (_, Value::Inet6(_)) => Some(Ordering::Less),

            (Value::Jsonb(_), _) => Some(Ordering::Greater),
            (_, Value::Jsonb(_)) => Some(Ordering::Less),

            (Value::TimestampTz { .. }, _) => Some(Ordering::Greater),
            (_, Value::TimestampTz { .. }) => Some(Ordering::Less),

            (Value::Interval { .. }, _) => Some(Ordering::Greater),
            (_, Value::Interval { .. }) => Some(Ordering::Less),

            (Value::Point { .. }, _) => Some(Ordering::Greater),
            (_, Value::Point { .. }) => Some(Ordering::Less),

            (Value::GeoBox { .. }, _) => Some(Ordering::Greater),
            (_, Value::GeoBox { .. }) => Some(Ordering::Less),

            (Value::Circle { .. }, _) => Some(Ordering::Greater),
            (_, Value::Circle { .. }) => Some(Ordering::Less),

            (Value::Enum { .. }, _) => Some(Ordering::Greater),
            (_, Value::Enum { .. }) => Some(Ordering::Less),

            (Value::Decimal { .. }, _) => Some(Ordering::Greater),
            (_, Value::Decimal { .. }) => Some(Ordering::Less),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_value_null_comparison() {
        let null = Value::Null;
        let int_val = Value::Int(42);
        assert_eq!(null.compare(&int_val), None);
        assert_eq!(int_val.compare(&null), None);
        assert_eq!(null.compare(&null), None);
    }

    #[test]
    fn test_value_int_comparison() {
        let v1 = Value::Int(10);
        let v2 = Value::Int(20);
        let v3 = Value::Int(10);

        assert_eq!(v1.compare(&v2), Some(Ordering::Less));
        assert_eq!(v2.compare(&v1), Some(Ordering::Greater));
        assert_eq!(v1.compare(&v3), Some(Ordering::Equal));
    }

    #[test]
    fn test_value_float_comparison() {
        let v1 = Value::Float(1.5);
        let v2 = Value::Float(2.5);

        assert_eq!(v1.compare(&v2), Some(Ordering::Less));
        assert_eq!(v2.compare(&v1), Some(Ordering::Greater));
    }

    #[test]
    fn test_value_text_comparison() {
        let v1 = Value::Text(Cow::Borrowed("apple"));
        let v2 = Value::Text(Cow::Borrowed("banana"));

        assert_eq!(v1.compare(&v2), Some(Ordering::Less));
    }

    #[test]
    fn test_coerce_int_to_text() {
        let v = Value::Int(42);
        let result = v.coerce_to_affinity(TypeAffinity::Text).unwrap();
        assert_eq!(result, Value::Text(Cow::Owned("42".to_string())));
    }

    #[test]
    fn test_coerce_text_to_int() {
        let v = Value::Text(Cow::Borrowed("42"));
        let result = v.coerce_to_affinity(TypeAffinity::Integer).unwrap();
        assert_eq!(result, Value::Int(42));
    }
}
