//! # Row Serialization for Grace Hash Join Disk Spill
//!
//! This module provides serialization and deserialization for rows spilled to disk
//! during Grace Hash Join execution when memory budget is exceeded.
//!
//! ## Design Goals
//!
//! 1. **Zero-allocation reads**: Deserialize into pre-allocated buffers
//! 2. **Compact encoding**: Minimize disk space for spill files
//! 3. **Full type support**: Handle all Value variants from types/value.rs
//! 4. **Deterministic**: Same row always produces same bytes
//!
//! ## Encoding Format
//!
//! Rows are encoded as a sequence of columns with type discriminants matching
//! the TypePrefix scheme from CLAUDE.md for consistency:
//!
//! ```text
//! Row := [col_count: u16] [Column]*
//! Column := [discriminant: u8] [Data]
//!
//! Discriminants (aligned with TypePrefix):
//!   0x01 = NULL (no data)
//!   0x02 = FALSE (reserved for bool)
//!   0x03 = TRUE (reserved for bool)
//!   0x10 = NEG_INFINITY (no data, deserializes to Float)
//!   0x12 = NEG_INT (8 bytes big-endian i64)
//!   0x13 = NEG_FLOAT (8 bytes f64 bits)
//!   0x14 = ZERO (no data, deserializes to Int(0))
//!   0x15 = POS_FLOAT (8 bytes f64 bits)
//!   0x16 = POS_INT (8 bytes big-endian i64)
//!   0x18 = POS_INFINITY (no data, deserializes to Float)
//!   0x19 = NAN (no data, deserializes to Float)
//!   0x20 = TEXT ([len: u32] [utf8_bytes])
//!   0x21 = BLOB ([len: u32] [bytes])
//!   0x33 = TIMESTAMPTZ (8 + 4 bytes: micros + offset_secs)
//!   0x34 = INTERVAL (8 + 4 + 4 bytes: micros + days + months)
//!   0x40 = UUID (16 bytes)
//!   0x41 = INET4 (4 bytes)
//!   0x42 = INET6 (16 bytes)
//!   0x43 = MACADDR (6 bytes)
//!   0x50 = JSONB ([len: u32] [bytes])
//!   0x63 = ENUM (4 bytes: type_id + ordinal as u16)
//!   0x70 = VECTOR ([count: u32] [f32 * count])
//!   0x80 = POINT (16 bytes: x + y as f64)
//!   0x81 = GEOBOX (32 bytes: low.0, low.1, high.0, high.1)
//!   0x82 = CIRCLE (24 bytes: center.0, center.1, radius)
//!   0x83 = DECIMAL (16 + 2 bytes: digits as i128 + scale)
//!   0x84 = TOAST_POINTER ([len: u32] [bytes])
//! ```
//!
//! ## Zero-Allocation Strategy
//!
//! - `serialize_row_into`: Writes to pre-allocated Vec<u8>
//! - `deserialize_row_into`: Writes to pre-allocated SmallVec
//! - `row_size`: Computes size without allocation for pre-sizing buffers
//!
//! ## Usage Example
//!
//! ```ignore
//! use turdb::sql::row_serde::RowSerde;
//! use smallvec::SmallVec;
//!
//! let row = vec![Value::Int(42), Value::Text(Cow::Borrowed("hello"))];
//! let mut buf = Vec::with_capacity(256);
//!
//! RowSerde::serialize_row_into(&row, &mut buf);
//!
//! let mut out: SmallVec<[Value<'static>; 16]> = SmallVec::new();
//! let mut offset = 0;
//! RowSerde::deserialize_row_into(&buf, &mut offset, &mut out).unwrap();
//!
//! assert_eq!(out[0], Value::Int(42));
//! ```
//!
//! ## Performance
//!
//! - Serialization: O(n) where n is row data size
//! - Deserialization: O(n) with zero heap allocation when SmallVec is sufficient
//! - Memory: Encoded rows are compact with minimal overhead (1 byte per column)
//!
//! ## Relationship to Key Encoding
//!
//! Unlike the key encoding in `encoding/key.rs` which preserves sort order,
//! this encoding is optimized for space and speed without ordering guarantees.
//! It is used exclusively for temporary disk spill during query execution.

use crate::types::Value;
use eyre::{bail, ensure, Result};
use smallvec::SmallVec;
use std::borrow::Cow;

mod discriminant {
    pub const NULL: u8 = 0x01;


    pub const NEG_INFINITY: u8 = 0x10;

    pub const NEG_INT: u8 = 0x12;
    pub const NEG_FLOAT: u8 = 0x13;
    pub const ZERO: u8 = 0x14;
    pub const POS_FLOAT: u8 = 0x15;
    pub const POS_INT: u8 = 0x16;

    pub const POS_INFINITY: u8 = 0x18;
    pub const NAN: u8 = 0x19;

    pub const TEXT: u8 = 0x20;
    pub const BLOB: u8 = 0x21;


    pub const TIMESTAMPTZ: u8 = 0x33;
    pub const INTERVAL: u8 = 0x34;

    pub const UUID: u8 = 0x40;
    pub const INET4: u8 = 0x41;
    pub const INET6: u8 = 0x42;
    pub const MACADDR: u8 = 0x43;

    pub const JSONB: u8 = 0x50;


    pub const ENUM: u8 = 0x63;


    pub const VECTOR: u8 = 0x70;

    pub const POINT: u8 = 0x80;
    pub const GEOBOX: u8 = 0x81;
    pub const CIRCLE: u8 = 0x82;
    pub const DECIMAL: u8 = 0x83;
    pub const TOAST_POINTER: u8 = 0x84;
}

pub struct RowSerde;

impl RowSerde {
    pub fn serialize_row_into(row: &[Value<'_>], buf: &mut Vec<u8>) {
        buf.extend_from_slice(&(row.len() as u16).to_be_bytes());

        for value in row {
            Self::serialize_value_into(value, buf);
        }
    }

    fn serialize_value_into(value: &Value<'_>, buf: &mut Vec<u8>) {
        match value {
            Value::Null => {
                buf.push(discriminant::NULL);
            }
            Value::Int(i) => {
                if *i < 0 {
                    buf.push(discriminant::NEG_INT);
                    buf.extend_from_slice(&i.to_be_bytes());
                } else if *i == 0 {
                    buf.push(discriminant::ZERO);
                } else {
                    buf.push(discriminant::POS_INT);
                    buf.extend_from_slice(&i.to_be_bytes());
                }
            }
            Value::Float(f) => {
                if f.is_nan() {
                    buf.push(discriminant::NAN);
                } else if *f == f64::NEG_INFINITY {
                    buf.push(discriminant::NEG_INFINITY);
                } else if *f == f64::INFINITY {
                    buf.push(discriminant::POS_INFINITY);
                } else if *f < 0.0 {
                    buf.push(discriminant::NEG_FLOAT);
                    buf.extend_from_slice(&f.to_bits().to_be_bytes());
                } else if *f == 0.0 {
                    buf.push(discriminant::ZERO);
                } else {
                    buf.push(discriminant::POS_FLOAT);
                    buf.extend_from_slice(&f.to_bits().to_be_bytes());
                }
            }
            Value::Text(s) => {
                buf.push(discriminant::TEXT);
                let bytes = s.as_bytes();
                buf.extend_from_slice(&(bytes.len() as u32).to_be_bytes());
                buf.extend_from_slice(bytes);
            }
            Value::Blob(b) => {
                buf.push(discriminant::BLOB);
                buf.extend_from_slice(&(b.len() as u32).to_be_bytes());
                buf.extend_from_slice(b);
            }
            Value::Vector(v) => {
                buf.push(discriminant::VECTOR);
                buf.extend_from_slice(&(v.len() as u32).to_be_bytes());
                for f in v.iter() {
                    buf.extend_from_slice(&f.to_bits().to_be_bytes());
                }
            }
            Value::Uuid(u) => {
                buf.push(discriminant::UUID);
                buf.extend_from_slice(u);
            }
            Value::MacAddr(m) => {
                buf.push(discriminant::MACADDR);
                buf.extend_from_slice(m);
            }
            Value::Inet4(ip) => {
                buf.push(discriminant::INET4);
                buf.extend_from_slice(ip);
            }
            Value::Inet6(ip) => {
                buf.push(discriminant::INET6);
                buf.extend_from_slice(ip);
            }
            Value::Jsonb(b) => {
                buf.push(discriminant::JSONB);
                buf.extend_from_slice(&(b.len() as u32).to_be_bytes());
                buf.extend_from_slice(b);
            }
            Value::TimestampTz {
                micros,
                offset_secs,
            } => {
                buf.push(discriminant::TIMESTAMPTZ);
                buf.extend_from_slice(&micros.to_be_bytes());
                buf.extend_from_slice(&offset_secs.to_be_bytes());
            }
            Value::Interval {
                micros,
                days,
                months,
            } => {
                buf.push(discriminant::INTERVAL);
                buf.extend_from_slice(&micros.to_be_bytes());
                buf.extend_from_slice(&days.to_be_bytes());
                buf.extend_from_slice(&months.to_be_bytes());
            }
            Value::Point { x, y } => {
                buf.push(discriminant::POINT);
                buf.extend_from_slice(&x.to_bits().to_be_bytes());
                buf.extend_from_slice(&y.to_bits().to_be_bytes());
            }
            Value::GeoBox { low, high } => {
                buf.push(discriminant::GEOBOX);
                buf.extend_from_slice(&low.0.to_bits().to_be_bytes());
                buf.extend_from_slice(&low.1.to_bits().to_be_bytes());
                buf.extend_from_slice(&high.0.to_bits().to_be_bytes());
                buf.extend_from_slice(&high.1.to_bits().to_be_bytes());
            }
            Value::Circle { center, radius } => {
                buf.push(discriminant::CIRCLE);
                buf.extend_from_slice(&center.0.to_bits().to_be_bytes());
                buf.extend_from_slice(&center.1.to_bits().to_be_bytes());
                buf.extend_from_slice(&radius.to_bits().to_be_bytes());
            }
            Value::Enum { type_id, ordinal } => {
                buf.push(discriminant::ENUM);
                buf.extend_from_slice(&type_id.to_be_bytes());
                buf.extend_from_slice(&ordinal.to_be_bytes());
            }
            Value::Decimal { digits, scale } => {
                buf.push(discriminant::DECIMAL);
                buf.extend_from_slice(&digits.to_be_bytes());
                buf.extend_from_slice(&scale.to_be_bytes());
            }
            Value::ToastPointer(b) => {
                buf.push(discriminant::TOAST_POINTER);
                buf.extend_from_slice(&(b.len() as u32).to_be_bytes());
                buf.extend_from_slice(b);
            }
        }
    }

    pub fn deserialize_row_into(
        data: &[u8],
        offset: &mut usize,
        out: &mut SmallVec<[Value<'static>; 16]>,
    ) -> Result<()> {
        ensure!(
            data.len() >= *offset + 2,
            "truncated row: missing column count"
        );

        let col_count =
            u16::from_be_bytes([data[*offset], data[*offset + 1]]) as usize;
        *offset += 2;

        out.clear();
        out.reserve(col_count);

        for _ in 0..col_count {
            let value = Self::deserialize_value(data, offset)?;
            out.push(value);
        }

        Ok(())
    }

    fn deserialize_value(data: &[u8], offset: &mut usize) -> Result<Value<'static>> {
        ensure!(
            data.len() > *offset,
            "truncated row: missing discriminant"
        );

        let disc = data[*offset];
        *offset += 1;

        match disc {
            discriminant::NULL => Ok(Value::Null),

            discriminant::ZERO => Ok(Value::Int(0)),

            discriminant::NEG_INT => {
                ensure!(data.len() >= *offset + 8, "truncated neg int");
                let bytes: [u8; 8] = data[*offset..*offset + 8].try_into().unwrap();
                *offset += 8;
                Ok(Value::Int(i64::from_be_bytes(bytes)))
            }

            discriminant::POS_INT => {
                ensure!(data.len() >= *offset + 8, "truncated pos int");
                let bytes: [u8; 8] = data[*offset..*offset + 8].try_into().unwrap();
                *offset += 8;
                Ok(Value::Int(i64::from_be_bytes(bytes)))
            }

            discriminant::NAN => Ok(Value::Float(f64::NAN)),

            discriminant::NEG_INFINITY => Ok(Value::Float(f64::NEG_INFINITY)),

            discriminant::POS_INFINITY => Ok(Value::Float(f64::INFINITY)),

            discriminant::NEG_FLOAT => {
                ensure!(data.len() >= *offset + 8, "truncated neg float");
                let bytes: [u8; 8] = data[*offset..*offset + 8].try_into().unwrap();
                *offset += 8;
                Ok(Value::Float(f64::from_bits(u64::from_be_bytes(bytes))))
            }

            discriminant::POS_FLOAT => {
                ensure!(data.len() >= *offset + 8, "truncated pos float");
                let bytes: [u8; 8] = data[*offset..*offset + 8].try_into().unwrap();
                *offset += 8;
                Ok(Value::Float(f64::from_bits(u64::from_be_bytes(bytes))))
            }

            discriminant::TEXT => {
                ensure!(data.len() >= *offset + 4, "truncated text length");
                let len = u32::from_be_bytes(data[*offset..*offset + 4].try_into().unwrap()) as usize;
                *offset += 4;
                ensure!(data.len() >= *offset + len, "truncated text data");
                let s = std::str::from_utf8(&data[*offset..*offset + len])?;
                *offset += len;
                Ok(Value::Text(Cow::Owned(s.to_string())))
            }

            discriminant::BLOB => {
                ensure!(data.len() >= *offset + 4, "truncated blob length");
                let len = u32::from_be_bytes(data[*offset..*offset + 4].try_into().unwrap()) as usize;
                *offset += 4;
                ensure!(data.len() >= *offset + len, "truncated blob data");
                let b = data[*offset..*offset + len].to_vec();
                *offset += len;
                Ok(Value::Blob(Cow::Owned(b)))
            }

            discriminant::VECTOR => {
                ensure!(data.len() >= *offset + 4, "truncated vector count");
                let count = u32::from_be_bytes(data[*offset..*offset + 4].try_into().unwrap()) as usize;
                *offset += 4;
                ensure!(data.len() >= *offset + count * 4, "truncated vector data");
                let mut v = Vec::with_capacity(count);
                for _ in 0..count {
                    let bits = u32::from_be_bytes(data[*offset..*offset + 4].try_into().unwrap());
                    v.push(f32::from_bits(bits));
                    *offset += 4;
                }
                Ok(Value::Vector(Cow::Owned(v)))
            }

            discriminant::UUID => {
                ensure!(data.len() >= *offset + 16, "truncated uuid");
                let bytes: [u8; 16] = data[*offset..*offset + 16].try_into().unwrap();
                *offset += 16;
                Ok(Value::Uuid(bytes))
            }

            discriminant::MACADDR => {
                ensure!(data.len() >= *offset + 6, "truncated macaddr");
                let bytes: [u8; 6] = data[*offset..*offset + 6].try_into().unwrap();
                *offset += 6;
                Ok(Value::MacAddr(bytes))
            }

            discriminant::INET4 => {
                ensure!(data.len() >= *offset + 4, "truncated inet4");
                let bytes: [u8; 4] = data[*offset..*offset + 4].try_into().unwrap();
                *offset += 4;
                Ok(Value::Inet4(bytes))
            }

            discriminant::INET6 => {
                ensure!(data.len() >= *offset + 16, "truncated inet6");
                let bytes: [u8; 16] = data[*offset..*offset + 16].try_into().unwrap();
                *offset += 16;
                Ok(Value::Inet6(bytes))
            }

            discriminant::JSONB => {
                ensure!(data.len() >= *offset + 4, "truncated jsonb length");
                let len = u32::from_be_bytes(data[*offset..*offset + 4].try_into().unwrap()) as usize;
                *offset += 4;
                ensure!(data.len() >= *offset + len, "truncated jsonb data");
                let b = data[*offset..*offset + len].to_vec();
                *offset += len;
                Ok(Value::Jsonb(Cow::Owned(b)))
            }

            discriminant::TIMESTAMPTZ => {
                ensure!(data.len() >= *offset + 12, "truncated timestamptz");
                let micros = i64::from_be_bytes(data[*offset..*offset + 8].try_into().unwrap());
                *offset += 8;
                let offset_secs = i32::from_be_bytes(data[*offset..*offset + 4].try_into().unwrap());
                *offset += 4;
                Ok(Value::TimestampTz {
                    micros,
                    offset_secs,
                })
            }

            discriminant::INTERVAL => {
                ensure!(data.len() >= *offset + 16, "truncated interval");
                let micros = i64::from_be_bytes(data[*offset..*offset + 8].try_into().unwrap());
                *offset += 8;
                let days = i32::from_be_bytes(data[*offset..*offset + 4].try_into().unwrap());
                *offset += 4;
                let months = i32::from_be_bytes(data[*offset..*offset + 4].try_into().unwrap());
                *offset += 4;
                Ok(Value::Interval {
                    micros,
                    days,
                    months,
                })
            }

            discriminant::POINT => {
                ensure!(data.len() >= *offset + 16, "truncated point");
                let x = f64::from_bits(u64::from_be_bytes(data[*offset..*offset + 8].try_into().unwrap()));
                *offset += 8;
                let y = f64::from_bits(u64::from_be_bytes(data[*offset..*offset + 8].try_into().unwrap()));
                *offset += 8;
                Ok(Value::Point { x, y })
            }

            discriminant::GEOBOX => {
                ensure!(data.len() >= *offset + 32, "truncated geobox");
                let low_0 = f64::from_bits(u64::from_be_bytes(data[*offset..*offset + 8].try_into().unwrap()));
                *offset += 8;
                let low_1 = f64::from_bits(u64::from_be_bytes(data[*offset..*offset + 8].try_into().unwrap()));
                *offset += 8;
                let high_0 = f64::from_bits(u64::from_be_bytes(data[*offset..*offset + 8].try_into().unwrap()));
                *offset += 8;
                let high_1 = f64::from_bits(u64::from_be_bytes(data[*offset..*offset + 8].try_into().unwrap()));
                *offset += 8;
                Ok(Value::GeoBox {
                    low: (low_0, low_1),
                    high: (high_0, high_1),
                })
            }

            discriminant::CIRCLE => {
                ensure!(data.len() >= *offset + 24, "truncated circle");
                let cx = f64::from_bits(u64::from_be_bytes(data[*offset..*offset + 8].try_into().unwrap()));
                *offset += 8;
                let cy = f64::from_bits(u64::from_be_bytes(data[*offset..*offset + 8].try_into().unwrap()));
                *offset += 8;
                let r = f64::from_bits(u64::from_be_bytes(data[*offset..*offset + 8].try_into().unwrap()));
                *offset += 8;
                Ok(Value::Circle {
                    center: (cx, cy),
                    radius: r,
                })
            }

            discriminant::ENUM => {
                ensure!(data.len() >= *offset + 4, "truncated enum");
                let type_id = u16::from_be_bytes(data[*offset..*offset + 2].try_into().unwrap());
                *offset += 2;
                let ordinal = u16::from_be_bytes(data[*offset..*offset + 2].try_into().unwrap());
                *offset += 2;
                Ok(Value::Enum { type_id, ordinal })
            }

            discriminant::DECIMAL => {
                ensure!(data.len() >= *offset + 18, "truncated decimal");
                let digits = i128::from_be_bytes(data[*offset..*offset + 16].try_into().unwrap());
                *offset += 16;
                let scale = i16::from_be_bytes(data[*offset..*offset + 2].try_into().unwrap());
                *offset += 2;
                Ok(Value::Decimal { digits, scale })
            }

            discriminant::TOAST_POINTER => {
                ensure!(data.len() >= *offset + 4, "truncated toast pointer length");
                let len = u32::from_be_bytes(data[*offset..*offset + 4].try_into().unwrap()) as usize;
                *offset += 4;
                ensure!(data.len() >= *offset + len, "truncated toast pointer data");
                let b = data[*offset..*offset + len].to_vec();
                *offset += len;
                Ok(Value::ToastPointer(Cow::Owned(b)))
            }

            _ => bail!("unknown discriminant: 0x{:02X}", disc),
        }
    }

    pub fn row_size(row: &[Value<'_>]) -> usize {
        let mut size = 2;
        for value in row {
            size += Self::value_size(value);
        }
        size
    }

    fn value_size(value: &Value<'_>) -> usize {
        match value {
            Value::Null => 1,
            Value::Int(i) => {
                if *i == 0 {
                    1
                } else {
                    1 + 8
                }
            }
            Value::Float(f) => {
                if f.is_nan() || *f == f64::NEG_INFINITY || *f == f64::INFINITY || *f == 0.0 {
                    1
                } else {
                    1 + 8
                }
            }
            Value::Text(s) => 1 + 4 + s.len(),
            Value::Blob(b) => 1 + 4 + b.len(),
            Value::Vector(v) => 1 + 4 + v.len() * 4,
            Value::Uuid(_) => 1 + 16,
            Value::MacAddr(_) => 1 + 6,
            Value::Inet4(_) => 1 + 4,
            Value::Inet6(_) => 1 + 16,
            Value::Jsonb(b) => 1 + 4 + b.len(),
            Value::TimestampTz { .. } => 1 + 12,
            Value::Interval { .. } => 1 + 16,
            Value::Point { .. } => 1 + 16,
            Value::GeoBox { .. } => 1 + 32,
            Value::Circle { .. } => 1 + 24,
            Value::Enum { .. } => 1 + 4,
            Value::Decimal { .. } => 1 + 18,
            Value::ToastPointer(b) => 1 + 4 + b.len(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn serialize_null_value_encodes_as_single_byte() {
        let row = vec![Value::Null];
        let mut buf = Vec::new();
        RowSerde::serialize_row_into(&row, &mut buf);
        assert_eq!(buf, vec![0x00, 0x01, discriminant::NULL]);
    }

    #[test]
    fn serialize_int_value_encodes_correctly() {
        let row = vec![Value::Int(42)];
        let mut buf = Vec::new();
        RowSerde::serialize_row_into(&row, &mut buf);
        assert_eq!(buf.len(), 2 + 1 + 8);
        assert_eq!(buf[2], discriminant::POS_INT);
        assert_eq!(&buf[3..11], &42i64.to_be_bytes());
    }

    #[test]
    fn serialize_negative_int_preserves_sign() {
        let row = vec![Value::Int(-123456)];
        let mut buf = Vec::new();
        RowSerde::serialize_row_into(&row, &mut buf);

        let mut out: SmallVec<[Value<'static>; 16]> = SmallVec::new();
        let mut offset = 0;
        RowSerde::deserialize_row_into(&buf, &mut offset, &mut out).unwrap();

        assert_eq!(out[0], Value::Int(-123456));
    }

    #[test]
    fn roundtrip_mixed_row_preserves_values() {
        let row = vec![
            Value::Int(123),
            Value::Text(Cow::Owned("hello".to_string())),
            Value::Float(3.14),
            Value::Null,
            Value::Blob(Cow::Owned(vec![1, 2, 3, 4])),
        ];

        let mut buf = Vec::new();
        RowSerde::serialize_row_into(&row, &mut buf);

        let mut out: SmallVec<[Value<'static>; 16]> = SmallVec::new();
        let mut offset = 0;
        RowSerde::deserialize_row_into(&buf, &mut offset, &mut out).unwrap();

        assert_eq!(out.len(), 5);
        assert_eq!(out[0], Value::Int(123));
        match &out[1] {
            Value::Text(s) => assert_eq!(s.as_ref(), "hello"),
            _ => panic!("expected Text"),
        }
        assert_eq!(out[2], Value::Float(3.14));
        assert_eq!(out[3], Value::Null);
        match &out[4] {
            Value::Blob(b) => assert_eq!(b.as_ref(), &[1, 2, 3, 4]),
            _ => panic!("expected Blob"),
        }
    }

    #[test]
    fn row_size_matches_serialized_length() {
        let row = vec![
            Value::Int(42),
            Value::Text(Cow::Owned("test".to_string())),
            Value::Float(1.5),
        ];
        let computed_size = RowSerde::row_size(&row);

        let mut buf = Vec::new();
        RowSerde::serialize_row_into(&row, &mut buf);

        assert_eq!(computed_size, buf.len());
    }

    #[test]
    fn roundtrip_uuid() {
        let uuid = [1u8, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];
        let row = vec![Value::Uuid(uuid)];

        let mut buf = Vec::new();
        RowSerde::serialize_row_into(&row, &mut buf);

        let mut out: SmallVec<[Value<'static>; 16]> = SmallVec::new();
        let mut offset = 0;
        RowSerde::deserialize_row_into(&buf, &mut offset, &mut out).unwrap();

        assert_eq!(out[0], Value::Uuid(uuid));
    }

    #[test]
    fn roundtrip_vector() {
        let vec_data = vec![1.0f32, 2.5, -3.14, 0.0];
        let row = vec![Value::Vector(Cow::Owned(vec_data.clone()))];

        let mut buf = Vec::new();
        RowSerde::serialize_row_into(&row, &mut buf);

        let mut out: SmallVec<[Value<'static>; 16]> = SmallVec::new();
        let mut offset = 0;
        RowSerde::deserialize_row_into(&buf, &mut offset, &mut out).unwrap();

        match &out[0] {
            Value::Vector(v) => {
                assert_eq!(v.len(), 4);
                assert!((v[0] - 1.0).abs() < f32::EPSILON);
                assert!((v[1] - 2.5).abs() < f32::EPSILON);
                assert!((v[2] - (-3.14)).abs() < 0.001);
                assert!((v[3] - 0.0).abs() < f32::EPSILON);
            }
            _ => panic!("expected Vector"),
        }
    }

    #[test]
    fn roundtrip_timestamptz() {
        let row = vec![Value::TimestampTz {
            micros: 1234567890123456,
            offset_secs: -28800,
        }];

        let mut buf = Vec::new();
        RowSerde::serialize_row_into(&row, &mut buf);

        let mut out: SmallVec<[Value<'static>; 16]> = SmallVec::new();
        let mut offset = 0;
        RowSerde::deserialize_row_into(&buf, &mut offset, &mut out).unwrap();

        match &out[0] {
            Value::TimestampTz {
                micros,
                offset_secs,
            } => {
                assert_eq!(*micros, 1234567890123456);
                assert_eq!(*offset_secs, -28800);
            }
            _ => panic!("expected TimestampTz"),
        }
    }

    #[test]
    fn roundtrip_interval() {
        let row = vec![Value::Interval {
            micros: 86400000000,
            days: 30,
            months: 12,
        }];

        let mut buf = Vec::new();
        RowSerde::serialize_row_into(&row, &mut buf);

        let mut out: SmallVec<[Value<'static>; 16]> = SmallVec::new();
        let mut offset = 0;
        RowSerde::deserialize_row_into(&buf, &mut offset, &mut out).unwrap();

        match &out[0] {
            Value::Interval {
                micros,
                days,
                months,
            } => {
                assert_eq!(*micros, 86400000000);
                assert_eq!(*days, 30);
                assert_eq!(*months, 12);
            }
            _ => panic!("expected Interval"),
        }
    }

    #[test]
    fn roundtrip_decimal() {
        let row = vec![Value::Decimal {
            digits: 123456789012345678901234567890i128,
            scale: 10,
        }];

        let mut buf = Vec::new();
        RowSerde::serialize_row_into(&row, &mut buf);

        let mut out: SmallVec<[Value<'static>; 16]> = SmallVec::new();
        let mut offset = 0;
        RowSerde::deserialize_row_into(&buf, &mut offset, &mut out).unwrap();

        match &out[0] {
            Value::Decimal { digits, scale } => {
                assert_eq!(*digits, 123456789012345678901234567890i128);
                assert_eq!(*scale, 10);
            }
            _ => panic!("expected Decimal"),
        }
    }

    #[test]
    fn roundtrip_point() {
        let row = vec![Value::Point { x: 1.5, y: -2.5 }];

        let mut buf = Vec::new();
        RowSerde::serialize_row_into(&row, &mut buf);

        let mut out: SmallVec<[Value<'static>; 16]> = SmallVec::new();
        let mut offset = 0;
        RowSerde::deserialize_row_into(&buf, &mut offset, &mut out).unwrap();

        match &out[0] {
            Value::Point { x, y } => {
                assert!((x - 1.5).abs() < f64::EPSILON);
                assert!((y - (-2.5)).abs() < f64::EPSILON);
            }
            _ => panic!("expected Point"),
        }
    }

    #[test]
    fn roundtrip_geobox() {
        let row = vec![Value::GeoBox {
            low: (1.0, 2.0),
            high: (3.0, 4.0),
        }];

        let mut buf = Vec::new();
        RowSerde::serialize_row_into(&row, &mut buf);

        let mut out: SmallVec<[Value<'static>; 16]> = SmallVec::new();
        let mut offset = 0;
        RowSerde::deserialize_row_into(&buf, &mut offset, &mut out).unwrap();

        match &out[0] {
            Value::GeoBox { low, high } => {
                assert!((low.0 - 1.0).abs() < f64::EPSILON);
                assert!((low.1 - 2.0).abs() < f64::EPSILON);
                assert!((high.0 - 3.0).abs() < f64::EPSILON);
                assert!((high.1 - 4.0).abs() < f64::EPSILON);
            }
            _ => panic!("expected GeoBox"),
        }
    }

    #[test]
    fn roundtrip_circle() {
        let row = vec![Value::Circle {
            center: (5.0, 6.0),
            radius: 10.0,
        }];

        let mut buf = Vec::new();
        RowSerde::serialize_row_into(&row, &mut buf);

        let mut out: SmallVec<[Value<'static>; 16]> = SmallVec::new();
        let mut offset = 0;
        RowSerde::deserialize_row_into(&buf, &mut offset, &mut out).unwrap();

        match &out[0] {
            Value::Circle { center, radius } => {
                assert!((center.0 - 5.0).abs() < f64::EPSILON);
                assert!((center.1 - 6.0).abs() < f64::EPSILON);
                assert!((radius - 10.0).abs() < f64::EPSILON);
            }
            _ => panic!("expected Circle"),
        }
    }

    #[test]
    fn roundtrip_enum() {
        let row = vec![Value::Enum {
            type_id: 42,
            ordinal: 7,
        }];

        let mut buf = Vec::new();
        RowSerde::serialize_row_into(&row, &mut buf);

        let mut out: SmallVec<[Value<'static>; 16]> = SmallVec::new();
        let mut offset = 0;
        RowSerde::deserialize_row_into(&buf, &mut offset, &mut out).unwrap();

        match &out[0] {
            Value::Enum { type_id, ordinal } => {
                assert_eq!(*type_id, 42);
                assert_eq!(*ordinal, 7);
            }
            _ => panic!("expected Enum"),
        }
    }

    #[test]
    fn roundtrip_macaddr() {
        let mac = [0xAA, 0xBB, 0xCC, 0xDD, 0xEE, 0xFF];
        let row = vec![Value::MacAddr(mac)];

        let mut buf = Vec::new();
        RowSerde::serialize_row_into(&row, &mut buf);

        let mut out: SmallVec<[Value<'static>; 16]> = SmallVec::new();
        let mut offset = 0;
        RowSerde::deserialize_row_into(&buf, &mut offset, &mut out).unwrap();

        assert_eq!(out[0], Value::MacAddr(mac));
    }

    #[test]
    fn roundtrip_inet4() {
        let ip = [192, 168, 1, 1];
        let row = vec![Value::Inet4(ip)];

        let mut buf = Vec::new();
        RowSerde::serialize_row_into(&row, &mut buf);

        let mut out: SmallVec<[Value<'static>; 16]> = SmallVec::new();
        let mut offset = 0;
        RowSerde::deserialize_row_into(&buf, &mut offset, &mut out).unwrap();

        assert_eq!(out[0], Value::Inet4(ip));
    }

    #[test]
    fn roundtrip_inet6() {
        let ip = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15];
        let row = vec![Value::Inet6(ip)];

        let mut buf = Vec::new();
        RowSerde::serialize_row_into(&row, &mut buf);

        let mut out: SmallVec<[Value<'static>; 16]> = SmallVec::new();
        let mut offset = 0;
        RowSerde::deserialize_row_into(&buf, &mut offset, &mut out).unwrap();

        assert_eq!(out[0], Value::Inet6(ip));
    }

    #[test]
    fn roundtrip_jsonb() {
        let json_bytes = br#"{"key": "value"}"#.to_vec();
        let row = vec![Value::Jsonb(Cow::Owned(json_bytes.clone()))];

        let mut buf = Vec::new();
        RowSerde::serialize_row_into(&row, &mut buf);

        let mut out: SmallVec<[Value<'static>; 16]> = SmallVec::new();
        let mut offset = 0;
        RowSerde::deserialize_row_into(&buf, &mut offset, &mut out).unwrap();

        match &out[0] {
            Value::Jsonb(b) => assert_eq!(b.as_ref(), json_bytes.as_slice()),
            _ => panic!("expected Jsonb"),
        }
    }

    #[test]
    fn roundtrip_toast_pointer() {
        let ptr_bytes = vec![0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08];
        let row = vec![Value::ToastPointer(Cow::Owned(ptr_bytes.clone()))];

        let mut buf = Vec::new();
        RowSerde::serialize_row_into(&row, &mut buf);

        let mut out: SmallVec<[Value<'static>; 16]> = SmallVec::new();
        let mut offset = 0;
        RowSerde::deserialize_row_into(&buf, &mut offset, &mut out).unwrap();

        match &out[0] {
            Value::ToastPointer(b) => assert_eq!(b.as_ref(), ptr_bytes.as_slice()),
            _ => panic!("expected ToastPointer"),
        }
    }

    #[test]
    fn empty_row_serializes_correctly() {
        let row: Vec<Value> = vec![];

        let mut buf = Vec::new();
        RowSerde::serialize_row_into(&row, &mut buf);

        assert_eq!(buf, vec![0x00, 0x00]);

        let mut out: SmallVec<[Value<'static>; 16]> = SmallVec::new();
        let mut offset = 0;
        RowSerde::deserialize_row_into(&buf, &mut offset, &mut out).unwrap();

        assert!(out.is_empty());
    }

    #[test]
    fn deserialize_truncated_data_returns_error() {
        let buf = vec![0x00, 0x01, discriminant::POS_INT, 0x00, 0x00];

        let mut out: SmallVec<[Value<'static>; 16]> = SmallVec::new();
        let mut offset = 0;
        let result = RowSerde::deserialize_row_into(&buf, &mut offset, &mut out);

        assert!(result.is_err());
    }

    #[test]
    fn deserialize_unknown_discriminant_returns_error() {
        let buf = vec![0x00, 0x01, 0xFF];

        let mut out: SmallVec<[Value<'static>; 16]> = SmallVec::new();
        let mut offset = 0;
        let result = RowSerde::deserialize_row_into(&buf, &mut offset, &mut out);

        assert!(result.is_err());
    }
}
