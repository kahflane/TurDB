//! # Heap-Owned Value Representation
//!
//! This module provides `OwnedValue`, a fully-owned version of SQL values
//! used for INSERT/UPDATE operations where data must outlive any borrowed
//! references.
//!
//! ## Design
//!
//! While `Value<'a>` uses `Cow` for zero-copy from mmap'd pages, `OwnedValue`
//! owns all its data on the heap. This is necessary when:
//!
//! - Building values from parsed SQL literals
//! - Storing values in write buffers
//! - Returning values across API boundaries
//!
//! ## Conversion
//!
//! Bidirectional conversion between `Value<'a>` and `OwnedValue`:
//!
//! ```ignore
//! // Value -> OwnedValue (always works, may allocate)
//! let owned: OwnedValue = (&value).into();
//!
//! // OwnedValue -> Value (borrows from owned)
//! let borrowed: Value<'_> = owned.to_value();
//! ```

use super::{DataType, Value};
use std::borrow::Cow;

/// Fully-owned SQL value for INSERT/UPDATE operations.
#[derive(Debug, Clone, PartialEq)]
pub enum OwnedValue {
    Null,
    Bool(bool),
    Int(i64),
    Float(f64),
    Text(String),
    Blob(Vec<u8>),
    Vector(Vec<f32>),
    Date(i32),
    Time(i64),
    Timestamp(i64),
    TimestampTz(i64, i32),
    Uuid([u8; 16]),
    MacAddr([u8; 6]),
    Inet4([u8; 4]),
    Inet6([u8; 16]),
    Interval(i64, i32, i32),
    Point(f64, f64),
    Box((f64, f64), (f64, f64)),
    Circle((f64, f64), f64),
    Jsonb(Vec<u8>),
    Decimal(i128, i16),
    Enum(u16, u16),
}

impl<'a> From<&Value<'a>> for OwnedValue {
    fn from(v: &Value<'a>) -> Self {
        match v {
            Value::Null => OwnedValue::Null,
            Value::Int(i) => OwnedValue::Int(*i),
            Value::Float(f) => OwnedValue::Float(*f),
            Value::Text(s) => OwnedValue::Text(s.to_string()),
            Value::Blob(b) => OwnedValue::Blob(b.to_vec()),
            Value::Vector(v) => OwnedValue::Vector(v.to_vec()),
            Value::Uuid(u) => OwnedValue::Uuid(*u),
            Value::MacAddr(m) => OwnedValue::MacAddr(*m),
            Value::Inet4(ip) => OwnedValue::Inet4(*ip),
            Value::Inet6(ip) => OwnedValue::Inet6(*ip),
            Value::Jsonb(b) => OwnedValue::Jsonb(b.to_vec()),
            Value::TimestampTz {
                micros,
                offset_secs,
            } => OwnedValue::TimestampTz(*micros, *offset_secs),
            Value::Interval {
                micros,
                days,
                months,
            } => OwnedValue::Interval(*micros, *days, *months),
            Value::Point { x, y } => OwnedValue::Point(*x, *y),
            Value::GeoBox { low, high } => OwnedValue::Box(*low, *high),
            Value::Circle { center, radius } => OwnedValue::Circle(*center, *radius),
            Value::Enum { type_id, ordinal } => OwnedValue::Enum(*type_id, *ordinal),
            Value::Decimal { digits, scale } => OwnedValue::Decimal(*digits, *scale),
        }
    }
}

impl<'a> From<Value<'a>> for OwnedValue {
    fn from(v: Value<'a>) -> Self {
        OwnedValue::from(&v)
    }
}

fn format_decimal(digits: i128, scale: i16) -> String {
    if scale <= 0 {
        format!("{}", digits)
    } else {
        let divisor = 10i128.pow(scale as u32);
        let int_part = digits / divisor;
        let frac_part = (digits % divisor).abs();
        format!(
            "{}.{:0>width$}",
            int_part,
            frac_part,
            width = scale as usize
        )
    }
}

impl OwnedValue {
    /// Returns true if this value is NULL.
    pub fn is_null(&self) -> bool {
        matches!(self, OwnedValue::Null)
    }

    /// Converts to a borrowed Value.
    pub fn to_value(&self) -> Value<'_> {
        match self {
            OwnedValue::Null => Value::Null,
            OwnedValue::Bool(b) => Value::Int(if *b { 1 } else { 0 }),
            OwnedValue::Int(i) => Value::Int(*i),
            OwnedValue::Float(f) => Value::Float(*f),
            OwnedValue::Text(s) => Value::Text(Cow::Borrowed(s.as_str())),
            OwnedValue::Blob(b) => Value::Blob(Cow::Borrowed(b.as_slice())),
            OwnedValue::Vector(v) => Value::Vector(Cow::Borrowed(v.as_slice())),
            OwnedValue::Date(d) => Value::Int(*d as i64),
            OwnedValue::Time(t) => Value::Int(*t),
            OwnedValue::Timestamp(ts) => Value::Int(*ts),
            OwnedValue::TimestampTz(ts, tz) => Value::TimestampTz {
                micros: *ts,
                offset_secs: *tz,
            },
            OwnedValue::Uuid(u) => Value::Uuid(*u),
            OwnedValue::MacAddr(m) => Value::MacAddr(*m),
            OwnedValue::Inet4(ip) => Value::Inet4(*ip),
            OwnedValue::Inet6(ip) => Value::Inet6(*ip),
            OwnedValue::Interval(micros, days, months) => Value::Interval {
                micros: *micros,
                days: *days,
                months: *months,
            },
            OwnedValue::Point(x, y) => Value::Point { x: *x, y: *y },
            OwnedValue::Box(p1, p2) => Value::GeoBox {
                low: *p1,
                high: *p2,
            },
            OwnedValue::Circle(center, radius) => Value::Circle {
                center: *center,
                radius: *radius,
            },
            OwnedValue::Jsonb(data) => Value::Jsonb(Cow::Borrowed(data.as_slice())),
            OwnedValue::Decimal(digits, scale) => Value::Decimal {
                digits: *digits,
                scale: *scale,
            },
            OwnedValue::Enum(type_id, ordinal) => Value::Enum {
                type_id: *type_id,
                ordinal: *ordinal,
            },
        }
    }

    /// Returns the DataType for this value.
    pub fn data_type(&self) -> DataType {
        match self {
            OwnedValue::Null => DataType::Int8,
            OwnedValue::Bool(_) => DataType::Bool,
            OwnedValue::Int(_) => DataType::Int8,
            OwnedValue::Float(_) => DataType::Float8,
            OwnedValue::Text(_) => DataType::Text,
            OwnedValue::Blob(_) => DataType::Blob,
            OwnedValue::Vector(_) => DataType::Vector,
            OwnedValue::Date(_) => DataType::Date,
            OwnedValue::Time(_) => DataType::Time,
            OwnedValue::Timestamp(_) => DataType::Timestamp,
            OwnedValue::TimestampTz(_, _) => DataType::TimestampTz,
            OwnedValue::Uuid(_) => DataType::Uuid,
            OwnedValue::MacAddr(_) => DataType::MacAddr,
            OwnedValue::Inet4(_) => DataType::Inet4,
            OwnedValue::Inet6(_) => DataType::Inet6,
            OwnedValue::Interval(_, _, _) => DataType::Interval,
            OwnedValue::Point(_, _) => DataType::Point,
            OwnedValue::Box(_, _) => DataType::Box,
            OwnedValue::Circle(_, _) => DataType::Circle,
            OwnedValue::Jsonb(_) => DataType::Jsonb,
            OwnedValue::Decimal(_, _) => DataType::Decimal,
            OwnedValue::Enum(_, _) => DataType::Enum,
        }
    }

    /// Returns true if this is a JSONB value.
    pub fn is_jsonb(&self) -> bool {
        matches!(self, OwnedValue::Jsonb(_))
    }

    /// Formats the value as a display string.
    pub fn display_string(&self) -> String {
        match self {
            OwnedValue::Null => "NULL".to_string(),
            OwnedValue::Bool(b) => if *b { "true" } else { "false" }.to_string(),
            OwnedValue::Int(i) => i.to_string(),
            OwnedValue::Float(f) => f.to_string(),
            OwnedValue::Text(s) => s.clone(),
            OwnedValue::Blob(b) => format!("\\x{}", hex::encode(b)),
            OwnedValue::Vector(v) => format!(
                "[{}]",
                v.iter()
                    .map(|f| f.to_string())
                    .collect::<Vec<_>>()
                    .join(",")
            ),
            OwnedValue::Date(d) => format!("date:{}", d),
            OwnedValue::Time(t) => format!("time:{}", t),
            OwnedValue::Timestamp(ts) => format!("ts:{}", ts),
            OwnedValue::TimestampTz(ts, tz) => format!("tstz:{}+{}", ts, tz),
            OwnedValue::Uuid(u) => {
                let h: String = u.iter().map(|b| format!("{:02x}", b)).collect();
                format!(
                    "{}-{}-{}-{}-{}",
                    &h[0..8],
                    &h[8..12],
                    &h[12..16],
                    &h[16..20],
                    &h[20..32]
                )
            }
            OwnedValue::MacAddr(m) => m
                .iter()
                .map(|b| format!("{:02x}", b))
                .collect::<Vec<_>>()
                .join(":"),
            OwnedValue::Inet4(ip) => format!("{}.{}.{}.{}", ip[0], ip[1], ip[2], ip[3]),
            OwnedValue::Inet6(ip) => {
                let parts: Vec<String> = (0..8)
                    .map(|i| format!("{:04x}", u16::from_be_bytes([ip[i * 2], ip[i * 2 + 1]])))
                    .collect();
                parts.join(":")
            }
            OwnedValue::Interval(micros, days, months) => {
                format!("{} months {} days {} us", months, days, micros)
            }
            OwnedValue::Point(x, y) => format!("({},{})", x, y),
            OwnedValue::Box(p1, p2) => format!("(({},{}),({},{}))", p1.0, p1.1, p2.0, p2.1),
            OwnedValue::Circle(center, radius) => {
                format!("<({},{}),{}>", center.0, center.1, radius)
            }
            OwnedValue::Jsonb(data) => format!("<jsonb:{} bytes>", data.len()),
            OwnedValue::Decimal(digits, scale) => format_decimal(*digits, *scale),
            OwnedValue::Enum(type_id, ordinal) => format!("enum({}:{})", type_id, ordinal),
        }
    }
}

mod hex {
    pub fn encode(data: &[u8]) -> String {
        data.iter().map(|b| format!("{:02x}", b)).collect()
    }
}

/// Converts a slice of OwnedValue to a Vec of Value references.
pub fn owned_values_to_values(owned: &[OwnedValue]) -> Vec<Value<'_>> {
    owned.iter().map(|ov| ov.to_value()).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_owned_value_to_value_roundtrip() {
        let owned = OwnedValue::Int(42);
        let value = owned.to_value();
        let back: OwnedValue = (&value).into();
        assert_eq!(owned, back);
    }

    #[test]
    fn test_owned_value_text() {
        let owned = OwnedValue::Text("hello".to_string());
        let value = owned.to_value();
        assert!(matches!(value, Value::Text(Cow::Borrowed("hello"))));
    }

    #[test]
    fn test_owned_value_data_type() {
        assert_eq!(OwnedValue::Int(42).data_type(), DataType::Int8);
        assert_eq!(OwnedValue::Text("x".into()).data_type(), DataType::Text);
        assert_eq!(OwnedValue::Uuid([0; 16]).data_type(), DataType::Uuid);
    }

    #[test]
    fn test_value_to_owned() {
        let value = Value::Text(Cow::Borrowed("hello"));
        let owned: OwnedValue = (&value).into();
        assert_eq!(owned, OwnedValue::Text("hello".to_string()));
    }
}
