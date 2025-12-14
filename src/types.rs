//! # Type-Safe Value Representation
//!
//! This module provides type-safe value representation without boxing for TurDB.
//! Values are stored inline using enum variants to avoid heap allocation and
//! enable efficient type checking and comparison.
//!
//! ## Design Goals
//!
//! 1. **Zero-boxing**: All small values (integers, floats, booleans) stored inline
//! 2. **Type safety**: Strong typing prevents invalid operations at compile time
//! 3. **Efficient comparison**: Direct value comparison without type coercion overhead
//! 4. **Memory efficient**: Enum uses tagged union, minimal memory overhead
//! 5. **SQLite compatibility**: Type affinity system matches SQLite's behavior
//!
//! ## Value Representation
//!
//! The `Value` enum represents all possible SQL values:
//!
//! - **Null**: SQL NULL, represents absence of value
//! - **Int**: 64-bit signed integer (i64)
//! - **Float**: 64-bit floating point (f64)
//! - **Text**: UTF-8 string reference (Cow for zero-copy or owned)
//! - **Blob**: Binary data reference (Cow for zero-copy or owned)
//! - **Vector**: Float32 vector for HNSW search (Cow for zero-copy or owned)
//!
//! ## Type Affinity System
//!
//! Following SQLite's type affinity rules, TurDB uses dynamic typing with hints:
//!
//! - **Integer**: Prefers integer storage (INT, INTEGER, BIGINT, etc.)
//! - **Real**: Prefers floating-point storage (REAL, DOUBLE, FLOAT)
//! - **Text**: Prefers text storage (TEXT, VARCHAR, CHAR, etc.)
//! - **Blob**: Prefers binary storage (BLOB, BYTEA)
//! - **Numeric**: Prefers numeric storage but accepts text (NUMERIC, DECIMAL)
//!
//! ## Column Data Types
//!
//! The `DataType` enum defines the schema-level column types:
//!
//! - Primitive types: Int2, Int4, Int8, Float4, Float8, Bool
//! - Text types: Text, Varchar(n), Char(n)
//! - Binary types: Blob, Bytea
//! - Date/Time: Date, Time, Timestamp, TimestampTz, Interval
//! - Special: Uuid, Inet, MacAddr, Json, Jsonb
//! - Arrays: Array(element_type)
//! - Vectors: Vector(dimension)
//!
//! ## Type Coercion Rules
//!
//! Type coercion follows SQL standard rules:
//!
//! 1. NULL coerces to any type
//! 2. Integer → Float (lossless for values within float precision)
//! 3. Integer/Float → Text (string representation)
//! 4. Text → Integer/Float (parse if valid)
//! 5. Numeric types preserve precision when possible
//!
//! ## Comparison Semantics
//!
//! Value comparison follows SQL NULL semantics:
//!
//! - NULL compared to anything (including NULL) is UNKNOWN (None)
//! - Different type comparisons follow type affinity rules
//! - Integer vs Float: Integer promoted to Float
//! - Text comparisons are lexicographic (UTF-8 byte order)
//! - Blob comparisons are lexicographic (byte order)
//!
//! ## Memory Layout
//!
//! ```text
//! Value enum (24 bytes on 64-bit):
//! +----------------+
//! | discriminant:1 |  (Null/Int/Float/Text/Blob/Vector)
//! +----------------+
//! | data: 16       |  (i64, f64, or Cow pointer + len)
//! +----------------+
//! | padding: 7     |
//! +----------------+
//! ```
//!
//! Using `Cow<'a, [u8]>` for Text/Blob/Vector enables zero-copy when
//! reading from mmap'd pages, while supporting owned data when needed.
//!
//! ## Usage Examples
//!
//! ```ignore
//! use turdb::types::{Value, DataType, TypeAffinity};
//!
//! // Creating values
//! let v1 = Value::Int(42);
//! let v2 = Value::Float(3.14);
//! let v3 = Value::Text("hello".into());
//! let v4 = Value::Null;
//!
//! // Type affinity
//! assert_eq!(DataType::Integer.affinity(), TypeAffinity::Integer);
//! assert_eq!(DataType::Real.affinity(), TypeAffinity::Real);
//!
//! // Comparison (NULL-aware)
//! assert_eq!(v1.compare(&v2), Some(Ordering::Greater)); // 42 > 3.14
//! assert_eq!(v4.compare(&v1), None); // NULL comparison is UNKNOWN
//!
//! // Coercion
//! let coerced = v1.coerce_to_affinity(TypeAffinity::Real)?;
//! assert!(matches!(coerced, Value::Float(42.0)));
//! ```
//!
//! ## Performance Characteristics
//!
//! - Value creation: O(1) for inline types, O(n) for copy types
//! - Value comparison: O(1) for inline types, O(n) for strings/blobs
//! - Type coercion: O(1) for compatible types, O(n) for parsing
//! - Memory: 24 bytes per value (stack-allocated)
//!
//! ## Thread Safety
//!
//! Value types are `Send + Sync` when they own their data (borrowed variants
//! are tied to page lifetime). This enables safe sharing across threads for
//! query results.

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

/// Schema-level column data types.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DataType {
    Int2,
    Int4,
    Int8,
    Float4,
    Float8,
    Bool,
    Text,
    Varchar(u32),
    Char(u32),
    Blob,
    Bytea,
    Date,
    Time,
    Timestamp,
    TimestampTz,
    Interval,
    Uuid,
    Inet,
    MacAddr,
    Json,
    Jsonb,
    Array(Box<DataType>),
    Vector(u32),
}

/// Type affinity for SQLite-compatible type system.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TypeAffinity {
    Integer,
    Real,
    Text,
    Blob,
    Numeric,
}

impl DataType {
    /// Returns the type affinity for this data type.
    pub fn affinity(&self) -> TypeAffinity {
        match self {
            DataType::Int2 | DataType::Int4 | DataType::Int8 | DataType::Bool => {
                TypeAffinity::Integer
            }
            DataType::Float4 | DataType::Float8 => TypeAffinity::Real,
            DataType::Text
            | DataType::Varchar(_)
            | DataType::Char(_)
            | DataType::Json
            | DataType::Jsonb => TypeAffinity::Text,
            DataType::Blob | DataType::Bytea | DataType::Vector(_) => TypeAffinity::Blob,
            DataType::Date
            | DataType::Time
            | DataType::Timestamp
            | DataType::TimestampTz
            | DataType::Interval
            | DataType::Uuid
            | DataType::Inet
            | DataType::MacAddr => TypeAffinity::Numeric,
            DataType::Array(_) => TypeAffinity::Blob,
        }
    }
}

impl<'a> Value<'a> {
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
            Value::Text(s) => {
                match target {
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
                }
            }
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
    fn test_value_variants_exist() {
        let _null = Value::Null;
        let _int = Value::Int(42);
        let _float = Value::Float(2.5);
        let _text = Value::Text(Cow::Borrowed("hello"));
        let _blob = Value::Blob(Cow::Borrowed(b"data"));
        let _vector = Value::Vector(Cow::Borrowed(&[1.0, 2.0, 3.0]));
    }

    #[test]
    fn test_value_sizes() {
        use std::mem::size_of;
        assert!(size_of::<Value>() <= 48, "Value should be compact");
    }

    #[test]
    fn test_datatype_primitive_variants() {
        let _int2 = DataType::Int2;
        let _int4 = DataType::Int4;
        let _int8 = DataType::Int8;
        let _float4 = DataType::Float4;
        let _float8 = DataType::Float8;
        let _bool = DataType::Bool;
    }

    #[test]
    fn test_datatype_text_variants() {
        let _text = DataType::Text;
        let _varchar = DataType::Varchar(255);
        let _char = DataType::Char(10);
    }

    #[test]
    fn test_datatype_datetime_variants() {
        let _date = DataType::Date;
        let _time = DataType::Time;
        let _timestamp = DataType::Timestamp;
        let _timestamptz = DataType::TimestampTz;
        let _interval = DataType::Interval;
    }

    #[test]
    fn test_datatype_special_variants() {
        let _uuid = DataType::Uuid;
        let _inet = DataType::Inet;
        let _macaddr = DataType::MacAddr;
        let _json = DataType::Json;
        let _jsonb = DataType::Jsonb;
    }

    #[test]
    fn test_datatype_composite_variants() {
        let _blob = DataType::Blob;
        let _bytea = DataType::Bytea;
        let _array = DataType::Array(Box::new(DataType::Int4));
        let _vector = DataType::Vector(128);
    }

    #[test]
    fn test_type_affinity_variants() {
        let _integer = TypeAffinity::Integer;
        let _real = TypeAffinity::Real;
        let _text = TypeAffinity::Text;
        let _blob = TypeAffinity::Blob;
        let _numeric = TypeAffinity::Numeric;
    }

    #[test]
    fn test_datatype_affinity_mapping() {
        assert_eq!(DataType::Int2.affinity(), TypeAffinity::Integer);
        assert_eq!(DataType::Int4.affinity(), TypeAffinity::Integer);
        assert_eq!(DataType::Int8.affinity(), TypeAffinity::Integer);
        assert_eq!(DataType::Bool.affinity(), TypeAffinity::Integer);

        assert_eq!(DataType::Float4.affinity(), TypeAffinity::Real);
        assert_eq!(DataType::Float8.affinity(), TypeAffinity::Real);

        assert_eq!(DataType::Text.affinity(), TypeAffinity::Text);
        assert_eq!(DataType::Varchar(100).affinity(), TypeAffinity::Text);
        assert_eq!(DataType::Char(10).affinity(), TypeAffinity::Text);
        assert_eq!(DataType::Json.affinity(), TypeAffinity::Text);
        assert_eq!(DataType::Jsonb.affinity(), TypeAffinity::Text);

        assert_eq!(DataType::Blob.affinity(), TypeAffinity::Blob);
        assert_eq!(DataType::Bytea.affinity(), TypeAffinity::Blob);
        assert_eq!(DataType::Vector(128).affinity(), TypeAffinity::Blob);
    }

    #[test]
    fn test_coerce_null_to_any_affinity() {
        let null = Value::Null;
        assert_eq!(
            null.coerce_to_affinity(TypeAffinity::Integer).unwrap(),
            Value::Null
        );
        assert_eq!(
            null.coerce_to_affinity(TypeAffinity::Real).unwrap(),
            Value::Null
        );
        assert_eq!(
            null.coerce_to_affinity(TypeAffinity::Text).unwrap(),
            Value::Null
        );
        assert_eq!(
            null.coerce_to_affinity(TypeAffinity::Blob).unwrap(),
            Value::Null
        );
    }

    #[test]
    fn test_coerce_int_to_real() {
        let int_val = Value::Int(42);
        let result = int_val.coerce_to_affinity(TypeAffinity::Real).unwrap();
        assert_eq!(result, Value::Float(42.0));
    }

    #[test]
    fn test_coerce_int_to_text() {
        let int_val = Value::Int(123);
        let result = int_val.coerce_to_affinity(TypeAffinity::Text).unwrap();
        assert_eq!(result, Value::Text(Cow::Owned("123".to_string())));
    }

    #[test]
    fn test_coerce_float_to_text() {
        let float_val = Value::Float(2.5);
        let result = float_val.coerce_to_affinity(TypeAffinity::Text).unwrap();
        match result {
            Value::Text(s) => assert!(s.starts_with("2.5")),
            _ => panic!("Expected Text variant"),
        }
    }

    #[test]
    fn test_coerce_text_to_int() {
        let text_val = Value::Text(Cow::Borrowed("42"));
        let result = text_val.coerce_to_affinity(TypeAffinity::Integer).unwrap();
        assert_eq!(result, Value::Int(42));
    }

    #[test]
    fn test_coerce_text_to_real() {
        let text_val = Value::Text(Cow::Borrowed("2.5"));
        let result = text_val.coerce_to_affinity(TypeAffinity::Real).unwrap();
        assert_eq!(result, Value::Float(2.5));
    }

    #[test]
    fn test_coerce_invalid_text_to_int_fails() {
        let text_val = Value::Text(Cow::Borrowed("not a number"));
        let result = text_val.coerce_to_affinity(TypeAffinity::Integer);
        assert!(result.is_err());
    }

    #[test]
    fn test_coerce_same_affinity_noop() {
        let int_val = Value::Int(42);
        let result = int_val.coerce_to_affinity(TypeAffinity::Integer).unwrap();
        assert_eq!(result, Value::Int(42));
    }

    #[test]
    fn test_compare_null_returns_none() {
        let null = Value::Null;
        let int_val = Value::Int(42);
        assert_eq!(null.compare(&int_val), None);
        assert_eq!(int_val.compare(&null), None);
        assert_eq!(null.compare(&null), None);
    }

    #[test]
    fn test_compare_int_values() {
        let v1 = Value::Int(10);
        let v2 = Value::Int(20);
        let v3 = Value::Int(10);

        assert_eq!(v1.compare(&v2), Some(Ordering::Less));
        assert_eq!(v2.compare(&v1), Some(Ordering::Greater));
        assert_eq!(v1.compare(&v3), Some(Ordering::Equal));
    }

    #[test]
    fn test_compare_float_values() {
        let v1 = Value::Float(1.5);
        let v2 = Value::Float(2.5);
        let v3 = Value::Float(1.5);

        assert_eq!(v1.compare(&v2), Some(Ordering::Less));
        assert_eq!(v2.compare(&v1), Some(Ordering::Greater));
        assert_eq!(v1.compare(&v3), Some(Ordering::Equal));
    }

    #[test]
    fn test_compare_int_and_float() {
        let int_val = Value::Int(42);
        let float_val = Value::Float(42.5);
        let float_equal = Value::Float(42.0);

        assert_eq!(int_val.compare(&float_val), Some(Ordering::Less));
        assert_eq!(float_val.compare(&int_val), Some(Ordering::Greater));
        assert_eq!(int_val.compare(&float_equal), Some(Ordering::Equal));
    }

    #[test]
    fn test_compare_text_values() {
        let v1 = Value::Text(Cow::Borrowed("apple"));
        let v2 = Value::Text(Cow::Borrowed("banana"));
        let v3 = Value::Text(Cow::Borrowed("apple"));

        assert_eq!(v1.compare(&v2), Some(Ordering::Less));
        assert_eq!(v2.compare(&v1), Some(Ordering::Greater));
        assert_eq!(v1.compare(&v3), Some(Ordering::Equal));
    }

    #[test]
    fn test_compare_blob_values() {
        let v1 = Value::Blob(Cow::Borrowed(b"abc"));
        let v2 = Value::Blob(Cow::Borrowed(b"def"));
        let v3 = Value::Blob(Cow::Borrowed(b"abc"));

        assert_eq!(v1.compare(&v2), Some(Ordering::Less));
        assert_eq!(v2.compare(&v1), Some(Ordering::Greater));
        assert_eq!(v1.compare(&v3), Some(Ordering::Equal));
    }

    #[test]
    fn test_compare_different_types() {
        let int_val = Value::Int(42);
        let text_val = Value::Text(Cow::Borrowed("hello"));

        assert_eq!(int_val.compare(&text_val), Some(Ordering::Less));
        assert_eq!(text_val.compare(&int_val), Some(Ordering::Greater));
    }
}
