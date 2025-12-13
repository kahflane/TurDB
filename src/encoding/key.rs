//! # Big-Endian Key Encoding for B-Tree Indexes
//!
//! This module provides byte-comparable key encoding for TurDB's B-tree indexes.
//! All encoded keys can be compared using a single `memcmp` call, enabling
//! efficient key comparison without type-specific logic at comparison time.
//!
//! ## Design Goals
//!
//! 1. **Byte-comparable**: Encoded keys preserve sort order when compared lexicographically
//! 2. **Type-aware ordering**: NULL < booleans < numbers < strings < complex types
//! 3. **Multi-column support**: Composite keys encode correctly for compound indexes
//! 4. **Deterministic**: Same value always produces same encoding
//! 5. **Invertible**: All encodings can be decoded back to original values
//!
//! ## Type Prefix Scheme
//!
//! Each encoded value starts with a type prefix byte that determines sort order
//! between different types:
//!
//! ```text
//! 0x01       NULL
//! 0x02-0x03  Booleans (FALSE < TRUE)
//! 0x10-0x19  Numbers (NEG_INFINITY < negatives < ZERO < positives < POS_INFINITY < NAN)
//! 0x20-0x21  Strings (TEXT < BLOB)
//! 0x30-0x34  Date/Time types
//! 0x40-0x42  Special types (UUID, INET, MACADDR)
//! 0x50-0x56  JSON types
//! 0x60-0x65  Composite types (ARRAY, TUPLE, RANGE, ENUM, COMPOSITE, DOMAIN)
//! 0x70       VECTOR
//! 0x80-0xFE  Custom types (extension point)
//! 0xFF       MAX_KEY (sentinel for range queries)
//! ```
//!
//! ## Number Encoding Strategy
//!
//! Numbers use a sign-split encoding for correct ordering:
//!
//! - Negative integers: NEG_INT prefix (0x12) + two's complement big-endian
//! - Zero: ZERO prefix (0x14) only
//! - Positive integers: POS_INT prefix (0x16) + big-endian bytes
//!
//! This ensures: -∞ < -100 < -1 < 0 < 1 < 100 < +∞
//!
//! For floats, IEEE 754 bit manipulation preserves ordering:
//! - Negative floats: invert all bits (!bits)
//! - Positive floats: flip sign bit (bits ^ (1 << 63))
//!
//! ### Zero Canonicalization
//!
//! Both integer 0 and floating-point 0.0 encode to the same representation:
//! the single-byte `ZERO` prefix (0x14). This is intentional for sort order
//! consistency. However, decoding always returns `DecodedKey::Int(0)`:
//!
//! ```text
//! encode_int(0)   → [0x14]
//! encode_float(0.0) → [0x14]
//! decode([0x14])    → DecodedKey::Int(0)  // Type information lost
//! ```
//!
//! This means type information is lost for zero values during round-trip
//! encoding/decoding. Applications requiring precise type preservation
//! for zero should handle this case explicitly.
//!
//! ## Text Encoding Strategy
//!
//! Text values use escape encoding to handle embedded null bytes:
//!
//! ```text
//! 0x00 -> 0x00 0xFF  (escape null byte)
//! 0xFF -> 0xFF 0x00  (escape 0xFF byte)
//! Terminator: 0x00 0x00
//! ```
//!
//! This ensures:
//! - Embedded nulls don't terminate the string early
//! - Lexicographic order is preserved
//! - Empty strings sort before non-empty strings
//!
//! ## Composite Type Encoding
//!
//! Arrays and composites use recursive encoding with separators:
//!
//! ```text
//! ARRAY: [prefix][element1][0x01][element2][0x01]...[0x00]
//! COMPOSITE: [prefix][type_id:4][field1][0x01][field2][0x01]...[0x00]
//! ```
//!
//! ## Usage Example
//!
//! ```ignore
//! use turdb::encoding::key::{KeyEncoder, type_prefix};
//!
//! let mut encoder = KeyEncoder::new();
//!
//! // Encode a composite key (INT, TEXT)
//! encoder.encode_int(42);
//! encoder.encode_text("hello");
//!
//! let key1 = encoder.finish();
//! encoder.reset();
//!
//! encoder.encode_int(42);
//! encoder.encode_text("world");
//!
//! let key2 = encoder.finish();
//!
//! // key1 < key2 because "hello" < "world"
//! assert!(key1 < key2);
//! ```
//!
//! ## Performance Characteristics
//!
//! - Encoding: O(n) where n is the total size of values
//! - Comparison: Single memcmp, O(min(len1, len2))
//! - Memory: Encoded keys are typically 1-2 bytes larger than raw values
//!
//! ## Zero-Allocation Mode
//!
//! For CRUD operations, use `encode_*_to` methods with pre-allocated buffers:
//!
//! ```ignore
//! let mut buf = Vec::with_capacity(256);
//! encode_int_to(42, &mut buf);
//! encode_text_to("hello", &mut buf);
//! // buf now contains the encoded key, no allocation during encode
//! ```

use eyre::{bail, ensure, Result};

pub mod type_prefix {
    pub const NULL: u8 = 0x01;
    pub const FALSE: u8 = 0x02;
    pub const TRUE: u8 = 0x03;

    pub const NEG_INFINITY: u8 = 0x10;
    pub const NEG_BIG_INT: u8 = 0x11;
    pub const NEG_INT: u8 = 0x12;
    pub const NEG_FLOAT: u8 = 0x13;
    pub const ZERO: u8 = 0x14;
    pub const POS_FLOAT: u8 = 0x15;
    pub const POS_INT: u8 = 0x16;
    pub const POS_BIG_INT: u8 = 0x17;
    pub const POS_INFINITY: u8 = 0x18;
    pub const NAN: u8 = 0x19;

    pub const TEXT: u8 = 0x20;
    pub const BLOB: u8 = 0x21;

    pub const DATE: u8 = 0x30;
    pub const TIME: u8 = 0x31;
    pub const TIMESTAMP: u8 = 0x32;
    pub const TIMESTAMPTZ: u8 = 0x33;
    pub const INTERVAL: u8 = 0x34;

    pub const UUID: u8 = 0x40;
    pub const INET: u8 = 0x41;
    pub const MACADDR: u8 = 0x42;

    pub const JSON_NULL: u8 = 0x50;
    pub const JSON_FALSE: u8 = 0x51;
    pub const JSON_TRUE: u8 = 0x52;
    pub const JSON_NUMBER: u8 = 0x53;
    pub const JSON_STRING: u8 = 0x54;
    pub const JSON_ARRAY: u8 = 0x55;
    pub const JSON_OBJECT: u8 = 0x56;

    pub const ARRAY: u8 = 0x60;
    pub const TUPLE: u8 = 0x61;
    pub const RANGE: u8 = 0x62;
    pub const ENUM: u8 = 0x63;
    pub const COMPOSITE: u8 = 0x64;
    pub const DOMAIN: u8 = 0x65;

    pub const VECTOR: u8 = 0x70;

    pub const CUSTOM_START: u8 = 0x80;
    pub const MAX_KEY: u8 = 0xFF;
}

pub fn encode_null(buf: &mut Vec<u8>) {
    buf.push(type_prefix::NULL);
}

pub fn encode_bool(b: bool, buf: &mut Vec<u8>) {
    buf.push(if b {
        type_prefix::TRUE
    } else {
        type_prefix::FALSE
    });
}

pub fn encode_int(n: i64, buf: &mut Vec<u8>) {
    if n < 0 {
        buf.push(type_prefix::NEG_INT);
        buf.extend((n as u64).to_be_bytes());
    } else if n == 0 {
        buf.push(type_prefix::ZERO);
    } else {
        buf.push(type_prefix::POS_INT);
        buf.extend((n as u64).to_be_bytes());
    }
}

pub fn encode_float(f: f64, buf: &mut Vec<u8>) {
    if f.is_nan() {
        buf.push(type_prefix::NAN);
    } else if f == f64::NEG_INFINITY {
        buf.push(type_prefix::NEG_INFINITY);
    } else if f == f64::INFINITY {
        buf.push(type_prefix::POS_INFINITY);
    } else if f < 0.0 {
        buf.push(type_prefix::NEG_FLOAT);
        buf.extend((!f.to_bits()).to_be_bytes());
    } else if f == 0.0 {
        buf.push(type_prefix::ZERO);
    } else {
        buf.push(type_prefix::POS_FLOAT);
        buf.extend((f.to_bits() ^ (1u64 << 63)).to_be_bytes());
    }
}

pub fn encode_text(s: &str, buf: &mut Vec<u8>) {
    buf.push(type_prefix::TEXT);
    encode_escaped_bytes(s.as_bytes(), buf);
}

pub fn encode_blob(data: &[u8], buf: &mut Vec<u8>) {
    buf.push(type_prefix::BLOB);
    encode_escaped_bytes(data, buf);
}

pub fn encode_date(days: i32, buf: &mut Vec<u8>) {
    buf.push(type_prefix::DATE);
    buf.extend(((days as u32) ^ (1u32 << 31)).to_be_bytes());
}

pub fn encode_timestamp(micros: i64, buf: &mut Vec<u8>) {
    buf.push(type_prefix::TIMESTAMP);
    buf.extend(((micros as u64) ^ (1u64 << 63)).to_be_bytes());
}

pub fn encode_uuid(uuid: &[u8; 16], buf: &mut Vec<u8>) {
    buf.push(type_prefix::UUID);
    buf.extend(uuid);
}

pub fn encode_time(micros: i64, buf: &mut Vec<u8>) {
    buf.push(type_prefix::TIME);
    buf.extend(((micros as u64) ^ (1u64 << 63)).to_be_bytes());
}

pub fn encode_timestamptz(micros: i64, tz_offset_mins: i16, buf: &mut Vec<u8>) {
    buf.push(type_prefix::TIMESTAMPTZ);
    buf.extend(((micros as u64) ^ (1u64 << 63)).to_be_bytes());
    buf.extend(((tz_offset_mins as u16) ^ (1u16 << 15)).to_be_bytes());
}

pub fn encode_interval(months: i32, days: i32, micros: i64, buf: &mut Vec<u8>) {
    buf.push(type_prefix::INTERVAL);
    buf.extend(((months as u32) ^ (1u32 << 31)).to_be_bytes());
    buf.extend(((days as u32) ^ (1u32 << 31)).to_be_bytes());
    buf.extend(((micros as u64) ^ (1u64 << 63)).to_be_bytes());
}

pub fn encode_inet(is_ipv6: bool, addr: &[u8], prefix_len: u8, buf: &mut Vec<u8>) {
    buf.push(type_prefix::INET);
    buf.push(if is_ipv6 { 1 } else { 0 });
    buf.push(prefix_len);
    if is_ipv6 {
        buf.extend(&addr[..16]);
    } else {
        buf.extend(&addr[..4]);
    }
}

pub fn encode_macaddr(addr: &[u8; 6], buf: &mut Vec<u8>) {
    buf.push(type_prefix::MACADDR);
    buf.extend(addr);
}

pub fn encode_tuple<F>(elements: &[F], buf: &mut Vec<u8>, encode_elem: impl Fn(&F, &mut Vec<u8>)) {
    buf.push(type_prefix::TUPLE);
    for (i, elem) in elements.iter().enumerate() {
        if i > 0 {
            buf.push(0x01);
        }
        encode_elem(elem, buf);
    }
    buf.push(0x00);
}

pub fn encode_range<T>(
    lower: Option<&T>,
    upper: Option<&T>,
    lower_inclusive: bool,
    upper_inclusive: bool,
    buf: &mut Vec<u8>,
    encode_bound: impl Fn(&T, &mut Vec<u8>),
) {
    buf.push(type_prefix::RANGE);
    let flags: u8 = (if lower.is_none() { 0x01 } else { 0 })
        | (if upper.is_none() { 0x02 } else { 0 })
        | (if lower_inclusive { 0x04 } else { 0 })
        | (if upper_inclusive { 0x08 } else { 0 });
    buf.push(flags);
    if let Some(l) = lower {
        encode_bound(l, buf);
    }
    if let Some(u) = upper {
        encode_bound(u, buf);
    }
}

pub fn encode_domain<F>(
    type_id: u32,
    value: &F,
    buf: &mut Vec<u8>,
    encode_val: impl Fn(&F, &mut Vec<u8>),
) {
    buf.push(type_prefix::DOMAIN);
    buf.extend(type_id.to_be_bytes());
    encode_val(value, buf);
}

pub fn encode_vector(dimensions: &[f32], buf: &mut Vec<u8>) {
    buf.push(type_prefix::VECTOR);
    buf.extend((dimensions.len() as u32).to_be_bytes());
    for &dim in dimensions {
        let bits = dim.to_bits();
        let encoded = if dim < 0.0 {
            !bits
        } else {
            bits ^ (1u32 << 31)
        };
        buf.extend(encoded.to_be_bytes());
    }
}

pub fn encode_array<F>(elements: &[F], buf: &mut Vec<u8>, encode_elem: impl Fn(&F, &mut Vec<u8>)) {
    buf.push(type_prefix::ARRAY);
    for (i, elem) in elements.iter().enumerate() {
        if i > 0 {
            buf.push(0x01);
        }
        encode_elem(elem, buf);
    }
    buf.push(0x00);
}

pub fn encode_enum(type_id: u32, ordinal: u32, buf: &mut Vec<u8>) {
    buf.push(type_prefix::ENUM);
    buf.extend(type_id.to_be_bytes());
    buf.extend(ordinal.to_be_bytes());
}

pub fn encode_composite<F>(
    type_id: u32,
    fields: &[F],
    buf: &mut Vec<u8>,
    encode_field: impl Fn(&F, &mut Vec<u8>),
) {
    buf.push(type_prefix::COMPOSITE);
    buf.extend(type_id.to_be_bytes());
    for (i, field) in fields.iter().enumerate() {
        if i > 0 {
            buf.push(0x01);
        }
        encode_field(field, buf);
    }
    buf.push(0x00);
}

#[derive(Debug, Clone, PartialEq)]
pub enum JsonValue<'a> {
    Null,
    Bool(bool),
    Number(f64),
    String(&'a str),
    Array(&'a [JsonValue<'a>]),
    Object(&'a [(&'a str, JsonValue<'a>)]),
}

pub fn encode_json(json: &JsonValue, buf: &mut Vec<u8>) {
    match json {
        JsonValue::Null => buf.push(type_prefix::JSON_NULL),
        JsonValue::Bool(false) => buf.push(type_prefix::JSON_FALSE),
        JsonValue::Bool(true) => buf.push(type_prefix::JSON_TRUE),
        JsonValue::Number(n) => {
            buf.push(type_prefix::JSON_NUMBER);
            if *n < 0.0 {
                buf.extend((!n.to_bits()).to_be_bytes());
            } else {
                buf.extend((n.to_bits() ^ (1u64 << 63)).to_be_bytes());
            }
        }
        JsonValue::String(s) => {
            buf.push(type_prefix::JSON_STRING);
            encode_escaped_bytes(s.as_bytes(), buf);
        }
        JsonValue::Array(arr) => {
            buf.push(type_prefix::JSON_ARRAY);
            for (i, elem) in arr.iter().enumerate() {
                if i > 0 {
                    buf.push(0x01);
                }
                encode_json(elem, buf);
            }
            buf.push(0x00);
        }
        JsonValue::Object(obj) => {
            buf.push(type_prefix::JSON_OBJECT);
            for (i, (key, val)) in obj.iter().enumerate() {
                if i > 0 {
                    buf.push(0x01);
                }
                encode_escaped_bytes(key.as_bytes(), buf);
                encode_json(val, buf);
            }
            buf.push(0x00);
        }
    }
}

fn encode_escaped_bytes(data: &[u8], buf: &mut Vec<u8>) {
    for &byte in data {
        match byte {
            0x00 => {
                buf.push(0x00);
                buf.push(0xFF);
            }
            0xFF => {
                buf.push(0xFF);
                buf.push(0x00);
            }
            b => buf.push(b),
        }
    }
    buf.push(0x00);
    buf.push(0x00);
}

#[derive(Debug, Clone, PartialEq)]
pub enum Value<'a> {
    Null,
    Bool(bool),
    Int(i64),
    Float(f64),
    Text(&'a str),
    Blob(&'a [u8]),
    Date(i32),
    Timestamp(i64),
    Uuid(&'a [u8; 16]),
}

pub fn encode_value(value: &Value, buf: &mut Vec<u8>) {
    match value {
        Value::Null => encode_null(buf),
        Value::Bool(b) => encode_bool(*b, buf),
        Value::Int(n) => encode_int(*n, buf),
        Value::Float(f) => encode_float(*f, buf),
        Value::Text(s) => encode_text(s, buf),
        Value::Blob(b) => encode_blob(b, buf),
        Value::Date(d) => encode_date(*d, buf),
        Value::Timestamp(t) => encode_timestamp(*t, buf),
        Value::Uuid(u) => encode_uuid(u, buf),
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum DecodedJson {
    Null,
    Bool(bool),
    Number(f64),
    String(String),
    Array(Vec<DecodedJson>),
    Object(Vec<(String, DecodedJson)>),
}

#[derive(Debug, Clone, PartialEq)]
pub enum DecodedKey {
    Null,
    Bool(bool),
    Int(i64),
    Float(f64),
    NegInfinity,
    PosInfinity,
    Nan,
    Text(String),
    Blob(Vec<u8>),
    Date(i32),
    Time(i64),
    Timestamp(i64),
    TimestampTz {
        micros: i64,
        tz_offset_mins: i16,
    },
    Interval {
        months: i32,
        days: i32,
        micros: i64,
    },
    Uuid([u8; 16]),
    Inet {
        is_ipv6: bool,
        addr: Vec<u8>,
        prefix_len: u8,
    },
    MacAddr([u8; 6]),
    Array(Vec<DecodedKey>),
    Tuple(Vec<DecodedKey>),
    Range {
        lower: Option<Box<DecodedKey>>,
        upper: Option<Box<DecodedKey>>,
        lower_inclusive: bool,
        upper_inclusive: bool,
    },
    Enum {
        type_id: u32,
        ordinal: u32,
    },
    Composite {
        type_id: u32,
        fields: Vec<DecodedKey>,
    },
    Domain {
        type_id: u32,
        value: Box<DecodedKey>,
    },
    Vector(Vec<f32>),
    Json(DecodedJson),
}

pub fn decode_key(data: &[u8]) -> Result<(DecodedKey, usize)> {
    ensure!(!data.is_empty(), "cannot decode empty key");

    let prefix = data[0];
    match prefix {
        type_prefix::NULL => Ok((DecodedKey::Null, 1)),
        type_prefix::FALSE => Ok((DecodedKey::Bool(false), 1)),
        type_prefix::TRUE => Ok((DecodedKey::Bool(true), 1)),
        type_prefix::NEG_INFINITY => Ok((DecodedKey::NegInfinity, 1)),
        type_prefix::POS_INFINITY => Ok((DecodedKey::PosInfinity, 1)),
        type_prefix::NAN => Ok((DecodedKey::Nan, 1)),
        type_prefix::ZERO => Ok((DecodedKey::Int(0), 1)),
        type_prefix::NEG_INT => {
            ensure!(data.len() >= 9, "truncated negative integer");
            let bytes: [u8; 8] = data[1..9].try_into().unwrap();
            let val = i64::from_be_bytes(bytes);
            Ok((DecodedKey::Int(val), 9))
        }
        type_prefix::POS_INT => {
            ensure!(data.len() >= 9, "truncated positive integer");
            let bytes: [u8; 8] = data[1..9].try_into().unwrap();
            let val = u64::from_be_bytes(bytes) as i64;
            Ok((DecodedKey::Int(val), 9))
        }
        type_prefix::NEG_FLOAT => {
            ensure!(data.len() >= 9, "truncated negative float");
            let bytes: [u8; 8] = data[1..9].try_into().unwrap();
            let bits = !u64::from_be_bytes(bytes);
            let val = f64::from_bits(bits);
            Ok((DecodedKey::Float(val), 9))
        }
        type_prefix::POS_FLOAT => {
            ensure!(data.len() >= 9, "truncated positive float");
            let bytes: [u8; 8] = data[1..9].try_into().unwrap();
            let bits = u64::from_be_bytes(bytes) ^ (1u64 << 63);
            let val = f64::from_bits(bits);
            Ok((DecodedKey::Float(val), 9))
        }
        type_prefix::TEXT => {
            let (decoded_bytes, consumed) = decode_escaped_bytes(&data[1..])?;
            let text = String::from_utf8(decoded_bytes)
                .map_err(|e| eyre::eyre!("invalid UTF-8 in text key: {}", e))?;
            Ok((DecodedKey::Text(text), 1 + consumed))
        }
        type_prefix::BLOB => {
            let (decoded_bytes, consumed) = decode_escaped_bytes(&data[1..])?;
            Ok((DecodedKey::Blob(decoded_bytes), 1 + consumed))
        }
        type_prefix::DATE => {
            ensure!(data.len() >= 5, "truncated date");
            let bytes: [u8; 4] = data[1..5].try_into().unwrap();
            let encoded = u32::from_be_bytes(bytes);
            let days = (encoded ^ (1u32 << 31)) as i32;
            Ok((DecodedKey::Date(days), 5))
        }
        type_prefix::TIME => {
            ensure!(data.len() >= 9, "truncated time");
            let bytes: [u8; 8] = data[1..9].try_into().unwrap();
            let encoded = u64::from_be_bytes(bytes);
            let micros = (encoded ^ (1u64 << 63)) as i64;
            Ok((DecodedKey::Time(micros), 9))
        }
        type_prefix::TIMESTAMP => {
            ensure!(data.len() >= 9, "truncated timestamp");
            let bytes: [u8; 8] = data[1..9].try_into().unwrap();
            let encoded = u64::from_be_bytes(bytes);
            let micros = (encoded ^ (1u64 << 63)) as i64;
            Ok((DecodedKey::Timestamp(micros), 9))
        }
        type_prefix::TIMESTAMPTZ => {
            ensure!(data.len() >= 11, "truncated timestamptz");
            let ts_bytes: [u8; 8] = data[1..9].try_into().unwrap();
            let tz_bytes: [u8; 2] = data[9..11].try_into().unwrap();
            let encoded_ts = u64::from_be_bytes(ts_bytes);
            let encoded_tz = u16::from_be_bytes(tz_bytes);
            let micros = (encoded_ts ^ (1u64 << 63)) as i64;
            let tz_offset_mins = (encoded_tz ^ (1u16 << 15)) as i16;
            Ok((
                DecodedKey::TimestampTz {
                    micros,
                    tz_offset_mins,
                },
                11,
            ))
        }
        type_prefix::INTERVAL => {
            ensure!(data.len() >= 17, "truncated interval");
            let m_bytes: [u8; 4] = data[1..5].try_into().unwrap();
            let d_bytes: [u8; 4] = data[5..9].try_into().unwrap();
            let u_bytes: [u8; 8] = data[9..17].try_into().unwrap();
            let months = (u32::from_be_bytes(m_bytes) ^ (1u32 << 31)) as i32;
            let days = (u32::from_be_bytes(d_bytes) ^ (1u32 << 31)) as i32;
            let micros = (u64::from_be_bytes(u_bytes) ^ (1u64 << 63)) as i64;
            Ok((
                DecodedKey::Interval {
                    months,
                    days,
                    micros,
                },
                17,
            ))
        }
        type_prefix::UUID => {
            ensure!(data.len() >= 17, "truncated uuid");
            let bytes: [u8; 16] = data[1..17].try_into().unwrap();
            Ok((DecodedKey::Uuid(bytes), 17))
        }
        type_prefix::INET => {
            ensure!(data.len() >= 3, "truncated inet");
            let is_ipv6 = data[1] != 0;
            let prefix_len = data[2];
            let addr_len = if is_ipv6 { 16 } else { 4 };
            ensure!(data.len() >= 3 + addr_len, "truncated inet address");
            let addr = data[3..3 + addr_len].to_vec();
            Ok((
                DecodedKey::Inet {
                    is_ipv6,
                    addr,
                    prefix_len,
                },
                3 + addr_len,
            ))
        }
        type_prefix::MACADDR => {
            ensure!(data.len() >= 7, "truncated macaddr");
            let bytes: [u8; 6] = data[1..7].try_into().unwrap();
            Ok((DecodedKey::MacAddr(bytes), 7))
        }
        type_prefix::ARRAY => {
            let (elements, consumed) = decode_array_elements(&data[1..])?;
            Ok((DecodedKey::Array(elements), 1 + consumed))
        }
        type_prefix::TUPLE => {
            let (elements, consumed) = decode_array_elements(&data[1..])?;
            Ok((DecodedKey::Tuple(elements), 1 + consumed))
        }
        type_prefix::RANGE => {
            ensure!(data.len() >= 2, "truncated range");
            let flags = data[1];
            let lower_empty = (flags & 0x01) != 0;
            let upper_empty = (flags & 0x02) != 0;
            let lower_inclusive = (flags & 0x04) != 0;
            let upper_inclusive = (flags & 0x08) != 0;
            let mut offset = 2;
            let lower = if lower_empty {
                None
            } else {
                let (val, consumed) = decode_key(&data[offset..])?;
                offset += consumed;
                Some(Box::new(val))
            };
            let upper = if upper_empty {
                None
            } else {
                let (val, consumed) = decode_key(&data[offset..])?;
                offset += consumed;
                Some(Box::new(val))
            };
            Ok((
                DecodedKey::Range {
                    lower,
                    upper,
                    lower_inclusive,
                    upper_inclusive,
                },
                offset,
            ))
        }
        type_prefix::ENUM => {
            ensure!(data.len() >= 9, "truncated enum");
            let type_id = u32::from_be_bytes(data[1..5].try_into().unwrap());
            let ordinal = u32::from_be_bytes(data[5..9].try_into().unwrap());
            Ok((DecodedKey::Enum { type_id, ordinal }, 9))
        }
        type_prefix::COMPOSITE => {
            ensure!(data.len() >= 5, "truncated composite");
            let type_id = u32::from_be_bytes(data[1..5].try_into().unwrap());
            let (fields, consumed) = decode_composite_fields(&data[5..])?;
            Ok((DecodedKey::Composite { type_id, fields }, 5 + consumed))
        }
        type_prefix::DOMAIN => {
            ensure!(data.len() >= 5, "truncated domain");
            let type_id = u32::from_be_bytes(data[1..5].try_into().unwrap());
            let (value, consumed) = decode_key(&data[5..])?;
            Ok((
                DecodedKey::Domain {
                    type_id,
                    value: Box::new(value),
                },
                5 + consumed,
            ))
        }
        type_prefix::VECTOR => {
            ensure!(data.len() >= 5, "truncated vector");
            let dim_count = u32::from_be_bytes(data[1..5].try_into().unwrap()) as usize;
            ensure!(
                data.len() >= 5 + dim_count * 4,
                "truncated vector dimensions"
            );
            let mut dimensions = Vec::with_capacity(dim_count);
            for i in 0..dim_count {
                let start = 5 + i * 4;
                let encoded = u32::from_be_bytes(data[start..start + 4].try_into().unwrap());
                let bits = if encoded & (1u32 << 31) != 0 {
                    encoded ^ (1u32 << 31)
                } else {
                    !encoded
                };
                dimensions.push(f32::from_bits(bits));
            }
            Ok((DecodedKey::Vector(dimensions), 5 + dim_count * 4))
        }
        type_prefix::JSON_NULL
        | type_prefix::JSON_FALSE
        | type_prefix::JSON_TRUE
        | type_prefix::JSON_NUMBER
        | type_prefix::JSON_STRING
        | type_prefix::JSON_ARRAY
        | type_prefix::JSON_OBJECT => {
            let (json, consumed) = decode_json(data)?;
            Ok((DecodedKey::Json(json), consumed))
        }
        _ => bail!("unknown key prefix: 0x{:02X}", prefix),
    }
}

fn decode_escaped_bytes(data: &[u8]) -> Result<(Vec<u8>, usize)> {
    let mut result = Vec::new();
    let mut i = 0;

    while i < data.len() {
        let byte = data[i];
        if byte == 0x00 {
            if i + 1 >= data.len() {
                bail!("truncated escape sequence");
            }
            let next = data[i + 1];
            if next == 0x00 {
                return Ok((result, i + 2));
            } else if next == 0xFF {
                result.push(0x00);
                i += 2;
            } else {
                bail!("invalid escape sequence: 0x00 0x{:02X}", next);
            }
        } else if byte == 0xFF {
            if i + 1 >= data.len() {
                bail!("truncated escape sequence");
            }
            let next = data[i + 1];
            if next == 0x00 {
                result.push(0xFF);
                i += 2;
            } else {
                bail!("invalid escape sequence: 0xFF 0x{:02X}", next);
            }
        } else {
            result.push(byte);
            i += 1;
        }
    }

    bail!("missing terminator in escaped bytes");
}

fn decode_array_elements(data: &[u8]) -> Result<(Vec<DecodedKey>, usize)> {
    let mut elements = Vec::new();
    let mut i = 0;

    while i < data.len() {
        if data[i] == 0x00 {
            return Ok((elements, i + 1));
        }
        if !elements.is_empty() {
            ensure!(data[i] == 0x01, "expected element separator 0x01");
            i += 1;
        }
        let (elem, consumed) = decode_key(&data[i..])?;
        elements.push(elem);
        i += consumed;
    }

    bail!("missing array terminator");
}

fn decode_composite_fields(data: &[u8]) -> Result<(Vec<DecodedKey>, usize)> {
    let mut fields = Vec::new();
    let mut i = 0;

    while i < data.len() {
        if data[i] == 0x00 {
            return Ok((fields, i + 1));
        }
        if !fields.is_empty() {
            ensure!(data[i] == 0x01, "expected field separator 0x01");
            i += 1;
        }
        let (field, consumed) = decode_key(&data[i..])?;
        fields.push(field);
        i += consumed;
    }

    bail!("missing composite terminator");
}

fn decode_json(data: &[u8]) -> Result<(DecodedJson, usize)> {
    ensure!(!data.is_empty(), "cannot decode empty json");

    let prefix = data[0];
    match prefix {
        type_prefix::JSON_NULL => Ok((DecodedJson::Null, 1)),
        type_prefix::JSON_FALSE => Ok((DecodedJson::Bool(false), 1)),
        type_prefix::JSON_TRUE => Ok((DecodedJson::Bool(true), 1)),
        type_prefix::JSON_NUMBER => {
            ensure!(data.len() >= 9, "truncated json number");
            let bytes: [u8; 8] = data[1..9].try_into().unwrap();
            let encoded = u64::from_be_bytes(bytes);
            let bits = if encoded & (1u64 << 63) != 0 {
                encoded ^ (1u64 << 63)
            } else {
                !encoded
            };
            let val = f64::from_bits(bits);
            Ok((DecodedJson::Number(val), 9))
        }
        type_prefix::JSON_STRING => {
            let (decoded_bytes, consumed) = decode_escaped_bytes(&data[1..])?;
            let text = String::from_utf8(decoded_bytes)
                .map_err(|e| eyre::eyre!("invalid UTF-8 in json string: {}", e))?;
            Ok((DecodedJson::String(text), 1 + consumed))
        }
        type_prefix::JSON_ARRAY => {
            let (elements, consumed) = decode_json_array(&data[1..])?;
            Ok((DecodedJson::Array(elements), 1 + consumed))
        }
        type_prefix::JSON_OBJECT => {
            let (obj, consumed) = decode_json_object(&data[1..])?;
            Ok((DecodedJson::Object(obj), 1 + consumed))
        }
        _ => bail!("unknown json prefix: 0x{:02X}", prefix),
    }
}

fn decode_json_array(data: &[u8]) -> Result<(Vec<DecodedJson>, usize)> {
    let mut elements = Vec::new();
    let mut i = 0;

    while i < data.len() {
        if data[i] == 0x00 {
            return Ok((elements, i + 1));
        }
        if !elements.is_empty() {
            ensure!(data[i] == 0x01, "expected json element separator 0x01");
            i += 1;
        }
        let (elem, consumed) = decode_json(&data[i..])?;
        elements.push(elem);
        i += consumed;
    }

    bail!("missing json array terminator");
}

fn decode_json_object(data: &[u8]) -> Result<(Vec<(String, DecodedJson)>, usize)> {
    let mut entries = Vec::new();
    let mut i = 0;

    while i < data.len() {
        if data[i] == 0x00 {
            return Ok((entries, i + 1));
        }
        if !entries.is_empty() {
            ensure!(data[i] == 0x01, "expected json object separator 0x01");
            i += 1;
        }
        let (key_bytes, key_consumed) = decode_escaped_bytes(&data[i..])?;
        let key = String::from_utf8(key_bytes)
            .map_err(|e| eyre::eyre!("invalid UTF-8 in json object key: {}", e))?;
        i += key_consumed;
        let (val, val_consumed) = decode_json(&data[i..])?;
        entries.push((key, val));
        i += val_consumed;
    }

    bail!("missing json object terminator");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn encode_null_produces_single_byte_0x01() {
        let mut buf = Vec::new();
        encode_null(&mut buf);
        assert_eq!(buf, vec![type_prefix::NULL]);
    }

    #[test]
    fn encode_bool_false_produces_0x02() {
        let mut buf = Vec::new();
        encode_bool(false, &mut buf);
        assert_eq!(buf, vec![type_prefix::FALSE]);
    }

    #[test]
    fn encode_bool_true_produces_0x03() {
        let mut buf = Vec::new();
        encode_bool(true, &mut buf);
        assert_eq!(buf, vec![type_prefix::TRUE]);
    }

    #[test]
    fn encode_bool_ordering() {
        let mut false_buf = Vec::new();
        encode_bool(false, &mut false_buf);

        let mut true_buf = Vec::new();
        encode_bool(true, &mut true_buf);

        assert!(false_buf < true_buf, "FALSE should sort before TRUE");
    }

    #[test]
    fn decode_bool_roundtrip() {
        for &val in &[false, true] {
            let mut buf = Vec::new();
            encode_bool(val, &mut buf);
            let (decoded, consumed) = decode_key(&buf).unwrap();
            assert_eq!(decoded, DecodedKey::Bool(val));
            assert_eq!(consumed, 1);
        }
    }

    #[test]
    fn encode_date_produces_prefix_and_4_bytes() {
        let mut buf = Vec::new();
        encode_date(19000, &mut buf);
        assert_eq!(buf[0], type_prefix::DATE);
        assert_eq!(buf.len(), 5);
    }

    #[test]
    fn encode_date_preserves_ordering() {
        let dates = [0_i32, 1000, 19000, 30000];
        let mut encoded: Vec<Vec<u8>> = dates
            .iter()
            .map(|&d| {
                let mut buf = Vec::new();
                encode_date(d, &mut buf);
                buf
            })
            .collect();

        let original = encoded.clone();
        encoded.sort();
        assert_eq!(encoded, original, "encoded dates should already be sorted");
    }

    #[test]
    fn decode_date_roundtrip() {
        for &days in &[0_i32, 1000, 19000, -1000] {
            let mut buf = Vec::new();
            encode_date(days, &mut buf);
            let (decoded, consumed) = decode_key(&buf).unwrap();
            assert_eq!(decoded, DecodedKey::Date(days));
            assert_eq!(consumed, 5);
        }
    }

    #[test]
    fn encode_timestamp_produces_prefix_and_8_bytes() {
        let mut buf = Vec::new();
        encode_timestamp(1_000_000_000_i64, &mut buf);
        assert_eq!(buf[0], type_prefix::TIMESTAMP);
        assert_eq!(buf.len(), 9);
    }

    #[test]
    fn encode_timestamp_preserves_ordering() {
        let timestamps = [-1000_i64, 0, 1000, 1_000_000_000];
        let mut encoded: Vec<Vec<u8>> = timestamps
            .iter()
            .map(|&t| {
                let mut buf = Vec::new();
                encode_timestamp(t, &mut buf);
                buf
            })
            .collect();

        let original = encoded.clone();
        encoded.sort();
        assert_eq!(
            encoded, original,
            "encoded timestamps should already be sorted"
        );
    }

    #[test]
    fn decode_timestamp_roundtrip() {
        for &micros in &[0_i64, 1_000_000, -1_000_000, i64::MAX, i64::MIN] {
            let mut buf = Vec::new();
            encode_timestamp(micros, &mut buf);
            let (decoded, consumed) = decode_key(&buf).unwrap();
            assert_eq!(decoded, DecodedKey::Timestamp(micros));
            assert_eq!(consumed, 9);
        }
    }

    #[test]
    fn encode_uuid_produces_prefix_and_16_bytes() {
        let uuid: [u8; 16] = [
            0x12, 0x34, 0x56, 0x78, 0x9a, 0xbc, 0xde, 0xf0, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66,
            0x77, 0x88,
        ];
        let mut buf = Vec::new();
        encode_uuid(&uuid, &mut buf);
        assert_eq!(buf[0], type_prefix::UUID);
        assert_eq!(buf.len(), 17);
        assert_eq!(&buf[1..], &uuid);
    }

    #[test]
    fn decode_uuid_roundtrip() {
        let uuid: [u8; 16] = [
            0x12, 0x34, 0x56, 0x78, 0x9a, 0xbc, 0xde, 0xf0, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66,
            0x77, 0x88,
        ];
        let mut buf = Vec::new();
        encode_uuid(&uuid, &mut buf);
        let (decoded, consumed) = decode_key(&buf).unwrap();
        assert_eq!(decoded, DecodedKey::Uuid(uuid));
        assert_eq!(consumed, 17);
    }

    #[test]
    fn encode_array_empty() {
        let elements: [i64; 0] = [];
        let mut buf = Vec::new();
        encode_array(&elements, &mut buf, |n, b| encode_int(*n, b));
        assert_eq!(buf, vec![type_prefix::ARRAY, 0x00]);
    }

    #[test]
    fn encode_array_with_elements() {
        let elements = [1_i64, 2, 3];
        let mut buf = Vec::new();
        encode_array(&elements, &mut buf, |n, b| encode_int(*n, b));
        assert_eq!(buf[0], type_prefix::ARRAY);
        assert_eq!(*buf.last().unwrap(), 0x00);
    }

    #[test]
    fn decode_array_roundtrip() {
        let elements = [1_i64, 2, 3];
        let mut buf = Vec::new();
        encode_array(&elements, &mut buf, |n, b| encode_int(*n, b));
        let (decoded, _) = decode_key(&buf).unwrap();
        match decoded {
            DecodedKey::Array(elems) => {
                assert_eq!(elems.len(), 3);
                assert_eq!(elems[0], DecodedKey::Int(1));
                assert_eq!(elems[1], DecodedKey::Int(2));
                assert_eq!(elems[2], DecodedKey::Int(3));
            }
            _ => panic!("expected array"),
        }
    }

    #[test]
    fn encode_enum_produces_prefix_and_8_bytes() {
        let mut buf = Vec::new();
        encode_enum(100, 5, &mut buf);
        assert_eq!(buf[0], type_prefix::ENUM);
        assert_eq!(buf.len(), 9);
    }

    #[test]
    fn decode_enum_roundtrip() {
        let mut buf = Vec::new();
        encode_enum(100, 5, &mut buf);
        let (decoded, consumed) = decode_key(&buf).unwrap();
        assert_eq!(
            decoded,
            DecodedKey::Enum {
                type_id: 100,
                ordinal: 5
            }
        );
        assert_eq!(consumed, 9);
    }

    #[test]
    fn encode_composite_with_fields() {
        let fields = [1_i64, 2_i64];
        let mut buf = Vec::new();
        encode_composite(42, &fields, &mut buf, |n, b| encode_int(*n, b));
        assert_eq!(buf[0], type_prefix::COMPOSITE);
        assert_eq!(*buf.last().unwrap(), 0x00);
    }

    #[test]
    fn decode_composite_roundtrip() {
        let fields = [1_i64, 2_i64];
        let mut buf = Vec::new();
        encode_composite(42, &fields, &mut buf, |n, b| encode_int(*n, b));
        let (decoded, _) = decode_key(&buf).unwrap();
        match decoded {
            DecodedKey::Composite { type_id, fields } => {
                assert_eq!(type_id, 42);
                assert_eq!(fields.len(), 2);
                assert_eq!(fields[0], DecodedKey::Int(1));
                assert_eq!(fields[1], DecodedKey::Int(2));
            }
            _ => panic!("expected composite"),
        }
    }

    #[test]
    fn encode_time_roundtrip() {
        for &micros in &[0_i64, 3600_000_000, -3600_000_000] {
            let mut buf = Vec::new();
            encode_time(micros, &mut buf);
            assert_eq!(buf[0], type_prefix::TIME);
            let (decoded, consumed) = decode_key(&buf).unwrap();
            assert_eq!(decoded, DecodedKey::Time(micros));
            assert_eq!(consumed, 9);
        }
    }

    #[test]
    fn encode_timestamptz_roundtrip() {
        let mut buf = Vec::new();
        encode_timestamptz(1_000_000_000, 60, &mut buf);
        assert_eq!(buf[0], type_prefix::TIMESTAMPTZ);
        let (decoded, consumed) = decode_key(&buf).unwrap();
        assert_eq!(
            decoded,
            DecodedKey::TimestampTz {
                micros: 1_000_000_000,
                tz_offset_mins: 60
            }
        );
        assert_eq!(consumed, 11);
    }

    #[test]
    fn encode_interval_roundtrip() {
        let mut buf = Vec::new();
        encode_interval(12, 30, 3600_000_000, &mut buf);
        assert_eq!(buf[0], type_prefix::INTERVAL);
        let (decoded, consumed) = decode_key(&buf).unwrap();
        assert_eq!(
            decoded,
            DecodedKey::Interval {
                months: 12,
                days: 30,
                micros: 3600_000_000
            }
        );
        assert_eq!(consumed, 17);
    }

    #[test]
    fn encode_inet_ipv4_roundtrip() {
        let addr = [192, 168, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
        let mut buf = Vec::new();
        encode_inet(false, &addr, 24, &mut buf);
        assert_eq!(buf[0], type_prefix::INET);
        let (decoded, consumed) = decode_key(&buf).unwrap();
        match decoded {
            DecodedKey::Inet {
                is_ipv6,
                addr: decoded_addr,
                prefix_len,
            } => {
                assert!(!is_ipv6);
                assert_eq!(decoded_addr, &addr[..4]);
                assert_eq!(prefix_len, 24);
            }
            _ => panic!("expected inet"),
        }
        assert_eq!(consumed, 7);
    }

    #[test]
    fn encode_inet_ipv6_roundtrip() {
        let addr = [
            0x20, 0x01, 0x0d, 0xb8, 0x85, 0xa3, 0x00, 0x00, 0x00, 0x00, 0x8a, 0x2e, 0x03, 0x70,
            0x73, 0x34,
        ];
        let mut buf = Vec::new();
        encode_inet(true, &addr, 64, &mut buf);
        let (decoded, consumed) = decode_key(&buf).unwrap();
        match decoded {
            DecodedKey::Inet {
                is_ipv6,
                addr: decoded_addr,
                prefix_len,
            } => {
                assert!(is_ipv6);
                assert_eq!(decoded_addr, &addr[..16]);
                assert_eq!(prefix_len, 64);
            }
            _ => panic!("expected inet"),
        }
        assert_eq!(consumed, 19);
    }

    #[test]
    fn encode_macaddr_roundtrip() {
        let addr: [u8; 6] = [0xAA, 0xBB, 0xCC, 0xDD, 0xEE, 0xFF];
        let mut buf = Vec::new();
        encode_macaddr(&addr, &mut buf);
        assert_eq!(buf[0], type_prefix::MACADDR);
        let (decoded, consumed) = decode_key(&buf).unwrap();
        assert_eq!(decoded, DecodedKey::MacAddr(addr));
        assert_eq!(consumed, 7);
    }

    #[test]
    fn encode_tuple_roundtrip() {
        let elements = [1_i64, 2, 3];
        let mut buf = Vec::new();
        encode_tuple(&elements, &mut buf, |n, b| encode_int(*n, b));
        assert_eq!(buf[0], type_prefix::TUPLE);
        let (decoded, _) = decode_key(&buf).unwrap();
        match decoded {
            DecodedKey::Tuple(elems) => {
                assert_eq!(elems.len(), 3);
                assert_eq!(elems[0], DecodedKey::Int(1));
            }
            _ => panic!("expected tuple"),
        }
    }

    #[test]
    fn encode_range_roundtrip() {
        let lower = 10_i64;
        let upper = 20_i64;
        let mut buf = Vec::new();
        encode_range(Some(&lower), Some(&upper), true, false, &mut buf, |n, b| {
            encode_int(*n, b)
        });
        assert_eq!(buf[0], type_prefix::RANGE);
        let (decoded, _) = decode_key(&buf).unwrap();
        match decoded {
            DecodedKey::Range {
                lower,
                upper,
                lower_inclusive,
                upper_inclusive,
            } => {
                assert!(lower_inclusive);
                assert!(!upper_inclusive);
                assert_eq!(*lower.unwrap(), DecodedKey::Int(10));
                assert_eq!(*upper.unwrap(), DecodedKey::Int(20));
            }
            _ => panic!("expected range"),
        }
    }

    #[test]
    fn encode_domain_roundtrip() {
        let value = 42_i64;
        let mut buf = Vec::new();
        encode_domain(100, &value, &mut buf, |n, b| encode_int(*n, b));
        assert_eq!(buf[0], type_prefix::DOMAIN);
        let (decoded, _) = decode_key(&buf).unwrap();
        match decoded {
            DecodedKey::Domain { type_id, value } => {
                assert_eq!(type_id, 100);
                assert_eq!(*value, DecodedKey::Int(42));
            }
            _ => panic!("expected domain"),
        }
    }

    #[test]
    fn encode_vector_roundtrip() {
        let dims = [1.0_f32, 2.5, -3.0, 0.0];
        let mut buf = Vec::new();
        encode_vector(&dims, &mut buf);
        assert_eq!(buf[0], type_prefix::VECTOR);
        let (decoded, consumed) = decode_key(&buf).unwrap();
        match decoded {
            DecodedKey::Vector(decoded_dims) => {
                assert_eq!(decoded_dims.len(), 4);
                assert_eq!(decoded_dims[0], 1.0);
                assert_eq!(decoded_dims[1], 2.5);
                assert_eq!(decoded_dims[2], -3.0);
                assert_eq!(decoded_dims[3], 0.0);
            }
            _ => panic!("expected vector"),
        }
        assert_eq!(consumed, 5 + 4 * 4);
    }

    #[test]
    fn encode_json_null() {
        let mut buf = Vec::new();
        encode_json(&JsonValue::Null, &mut buf);
        assert_eq!(buf, vec![type_prefix::JSON_NULL]);
        let (decoded, _) = decode_key(&buf).unwrap();
        assert_eq!(decoded, DecodedKey::Json(DecodedJson::Null));
    }

    #[test]
    fn encode_json_bool() {
        let mut buf = Vec::new();
        encode_json(&JsonValue::Bool(true), &mut buf);
        assert_eq!(buf, vec![type_prefix::JSON_TRUE]);

        buf.clear();
        encode_json(&JsonValue::Bool(false), &mut buf);
        assert_eq!(buf, vec![type_prefix::JSON_FALSE]);
    }

    #[test]
    fn encode_json_number_roundtrip() {
        for &val in &[-1000.0_f64, -1.5, 0.0, 1.5, 1000.0] {
            let mut buf = Vec::new();
            encode_json(&JsonValue::Number(val), &mut buf);
            let (decoded, _) = decode_key(&buf).unwrap();
            match decoded {
                DecodedKey::Json(DecodedJson::Number(n)) => assert_eq!(n, val),
                _ => panic!("expected json number"),
            }
        }
    }

    #[test]
    fn encode_json_string_roundtrip() {
        let mut buf = Vec::new();
        encode_json(&JsonValue::String("hello"), &mut buf);
        let (decoded, _) = decode_key(&buf).unwrap();
        match decoded {
            DecodedKey::Json(DecodedJson::String(s)) => assert_eq!(s, "hello"),
            _ => panic!("expected json string"),
        }
    }

    #[test]
    fn encode_json_array_roundtrip() {
        let arr = [JsonValue::Number(1.0), JsonValue::Number(2.0)];
        let mut buf = Vec::new();
        encode_json(&JsonValue::Array(&arr), &mut buf);
        let (decoded, _) = decode_key(&buf).unwrap();
        match decoded {
            DecodedKey::Json(DecodedJson::Array(elems)) => {
                assert_eq!(elems.len(), 2);
                assert_eq!(elems[0], DecodedJson::Number(1.0));
                assert_eq!(elems[1], DecodedJson::Number(2.0));
            }
            _ => panic!("expected json array"),
        }
    }

    #[test]
    fn encode_json_object_roundtrip() {
        let obj = [("key", JsonValue::String("value"))];
        let mut buf = Vec::new();
        encode_json(&JsonValue::Object(&obj), &mut buf);
        let (decoded, _) = decode_key(&buf).unwrap();
        match decoded {
            DecodedKey::Json(DecodedJson::Object(entries)) => {
                assert_eq!(entries.len(), 1);
                assert_eq!(entries[0].0, "key");
                assert_eq!(entries[0].1, DecodedJson::String("value".to_string()));
            }
            _ => panic!("expected json object"),
        }
    }

    #[test]
    fn encode_value_dispatcher_null() {
        let mut buf = Vec::new();
        encode_value(&Value::Null, &mut buf);
        assert_eq!(buf, vec![type_prefix::NULL]);
    }

    #[test]
    fn encode_value_dispatcher_bool() {
        let mut buf = Vec::new();
        encode_value(&Value::Bool(true), &mut buf);
        assert_eq!(buf, vec![type_prefix::TRUE]);

        buf.clear();
        encode_value(&Value::Bool(false), &mut buf);
        assert_eq!(buf, vec![type_prefix::FALSE]);
    }

    #[test]
    fn encode_value_dispatcher_int() {
        let mut buf = Vec::new();
        encode_value(&Value::Int(42), &mut buf);
        let (decoded, _) = decode_key(&buf).unwrap();
        assert_eq!(decoded, DecodedKey::Int(42));
    }

    #[test]
    fn encode_value_dispatcher_float() {
        let mut buf = Vec::new();
        encode_value(&Value::Float(3.14), &mut buf);
        let (decoded, _) = decode_key(&buf).unwrap();
        assert_eq!(decoded, DecodedKey::Float(3.14));
    }

    #[test]
    fn encode_value_dispatcher_text() {
        let mut buf = Vec::new();
        encode_value(&Value::Text("hello"), &mut buf);
        let (decoded, _) = decode_key(&buf).unwrap();
        assert_eq!(decoded, DecodedKey::Text("hello".to_string()));
    }

    #[test]
    fn encode_value_dispatcher_blob() {
        let mut buf = Vec::new();
        encode_value(&Value::Blob(&[1, 2, 3]), &mut buf);
        let (decoded, _) = decode_key(&buf).unwrap();
        assert_eq!(decoded, DecodedKey::Blob(vec![1, 2, 3]));
    }

    #[test]
    fn encode_value_dispatcher_date() {
        let mut buf = Vec::new();
        encode_value(&Value::Date(19000), &mut buf);
        let (decoded, _) = decode_key(&buf).unwrap();
        assert_eq!(decoded, DecodedKey::Date(19000));
    }

    #[test]
    fn encode_value_dispatcher_timestamp() {
        let mut buf = Vec::new();
        encode_value(&Value::Timestamp(1_000_000_000), &mut buf);
        let (decoded, _) = decode_key(&buf).unwrap();
        assert_eq!(decoded, DecodedKey::Timestamp(1_000_000_000));
    }

    #[test]
    fn encode_value_dispatcher_uuid() {
        let uuid: [u8; 16] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];
        let mut buf = Vec::new();
        encode_value(&Value::Uuid(&uuid), &mut buf);
        let (decoded, _) = decode_key(&buf).unwrap();
        assert_eq!(decoded, DecodedKey::Uuid(uuid));
    }

    #[test]
    fn encode_int_zero_produces_single_byte() {
        let mut buf = Vec::new();
        encode_int(0, &mut buf);
        assert_eq!(buf, vec![type_prefix::ZERO]);
    }

    #[test]
    fn encode_int_positive_produces_prefix_and_bytes() {
        let mut buf = Vec::new();
        encode_int(1, &mut buf);
        assert_eq!(buf[0], type_prefix::POS_INT);
        assert_eq!(buf.len(), 9);
        assert_eq!(&buf[1..], 1_u64.to_be_bytes());
    }

    #[test]
    fn encode_int_negative_produces_prefix_and_twos_complement() {
        let mut buf = Vec::new();
        encode_int(-1, &mut buf);
        assert_eq!(buf[0], type_prefix::NEG_INT);
        assert_eq!(buf.len(), 9);
        assert_eq!(&buf[1..], (-1_i64 as u64).to_be_bytes());
    }

    #[test]
    fn encode_int_preserves_ordering() {
        let values = [-1000, -100, -1, 0, 1, 100, 1000];
        let mut encoded: Vec<Vec<u8>> = values
            .iter()
            .map(|&v| {
                let mut buf = Vec::new();
                encode_int(v, &mut buf);
                buf
            })
            .collect();

        let original = encoded.clone();
        encoded.sort();
        assert_eq!(encoded, original, "encoded keys should already be sorted");
    }

    #[test]
    fn encode_float_zero_produces_single_byte() {
        let mut buf = Vec::new();
        encode_float(0.0, &mut buf);
        assert_eq!(buf, vec![type_prefix::ZERO]);
    }

    #[test]
    fn encode_float_positive_uses_pos_float_prefix() {
        let mut buf = Vec::new();
        encode_float(1.5, &mut buf);
        assert_eq!(buf[0], type_prefix::POS_FLOAT);
        assert_eq!(buf.len(), 9);
    }

    #[test]
    fn encode_float_negative_uses_neg_float_prefix() {
        let mut buf = Vec::new();
        encode_float(-1.5, &mut buf);
        assert_eq!(buf[0], type_prefix::NEG_FLOAT);
        assert_eq!(buf.len(), 9);
    }

    #[test]
    fn encode_float_infinity() {
        let mut pos_inf = Vec::new();
        encode_float(f64::INFINITY, &mut pos_inf);
        assert_eq!(pos_inf, vec![type_prefix::POS_INFINITY]);

        let mut neg_inf = Vec::new();
        encode_float(f64::NEG_INFINITY, &mut neg_inf);
        assert_eq!(neg_inf, vec![type_prefix::NEG_INFINITY]);
    }

    #[test]
    fn encode_float_nan() {
        let mut buf = Vec::new();
        encode_float(f64::NAN, &mut buf);
        assert_eq!(buf, vec![type_prefix::NAN]);
    }

    #[test]
    fn encode_float_preserves_ordering() {
        let values = [
            f64::NEG_INFINITY,
            -1000.0,
            -1.5,
            -0.001,
            0.0,
            0.001,
            1.5,
            1000.0,
            f64::INFINITY,
        ];
        let mut encoded: Vec<Vec<u8>> = values
            .iter()
            .map(|&v| {
                let mut buf = Vec::new();
                encode_float(v, &mut buf);
                buf
            })
            .collect();

        let original = encoded.clone();
        encoded.sort();
        assert_eq!(encoded, original, "encoded floats should already be sorted");
    }

    #[test]
    fn encode_text_empty_string() {
        let mut buf = Vec::new();
        encode_text("", &mut buf);
        assert_eq!(buf, vec![type_prefix::TEXT, 0x00, 0x00]);
    }

    #[test]
    fn encode_text_simple_string() {
        let mut buf = Vec::new();
        encode_text("hello", &mut buf);
        assert_eq!(buf[0], type_prefix::TEXT);
        assert_eq!(&buf[1..6], b"hello");
        assert_eq!(&buf[6..], &[0x00, 0x00]);
    }

    #[test]
    fn encode_text_escapes_null_byte() {
        let mut buf = Vec::new();
        encode_text("a\x00b", &mut buf);
        assert_eq!(buf[0], type_prefix::TEXT);
        assert_eq!(buf[1], b'a');
        assert_eq!(&buf[2..4], &[0x00, 0xFF]);
        assert_eq!(buf[4], b'b');
        assert_eq!(&buf[5..], &[0x00, 0x00]);
    }

    #[test]
    fn encode_blob_empty() {
        let mut buf = Vec::new();
        encode_blob(&[], &mut buf);
        assert_eq!(buf, vec![type_prefix::BLOB, 0x00, 0x00]);
    }

    #[test]
    fn encode_blob_simple() {
        let mut buf = Vec::new();
        encode_blob(b"hello", &mut buf);
        assert_eq!(buf[0], type_prefix::BLOB);
        assert_eq!(&buf[1..6], b"hello");
        assert_eq!(&buf[6..], &[0x00, 0x00]);
    }

    #[test]
    fn encode_blob_escapes_null_byte() {
        let mut buf = Vec::new();
        encode_blob(&[b'a', 0x00, b'b'], &mut buf);
        assert_eq!(buf[0], type_prefix::BLOB);
        assert_eq!(buf[1], b'a');
        assert_eq!(&buf[2..4], &[0x00, 0xFF]);
        assert_eq!(buf[4], b'b');
        assert_eq!(&buf[5..], &[0x00, 0x00]);
    }

    #[test]
    fn encode_blob_escapes_0xff_byte() {
        let mut buf = Vec::new();
        encode_blob(&[b'a', 0xFF, b'b'], &mut buf);
        assert_eq!(buf[0], type_prefix::BLOB);
        assert_eq!(buf[1], b'a');
        assert_eq!(&buf[2..4], &[0xFF, 0x00]);
        assert_eq!(buf[4], b'b');
        assert_eq!(&buf[5..], &[0x00, 0x00]);
    }

    #[test]
    fn encode_text_preserves_ordering() {
        let values = ["", "a", "aa", "ab", "b", "hello", "world"];
        let mut encoded: Vec<Vec<u8>> = values
            .iter()
            .map(|&v| {
                let mut buf = Vec::new();
                encode_text(v, &mut buf);
                buf
            })
            .collect();

        let original = encoded.clone();
        encoded.sort();
        assert_eq!(encoded, original, "encoded texts should already be sorted");
    }

    #[test]
    fn decode_null_roundtrip() {
        let mut buf = Vec::new();
        encode_null(&mut buf);
        let (decoded, consumed) = decode_key(&buf).unwrap();
        assert_eq!(decoded, DecodedKey::Null);
        assert_eq!(consumed, 1);
    }

    #[test]
    fn decode_int_roundtrip() {
        for &val in &[-1000_i64, -1, 0, 1, 1000, i64::MIN, i64::MAX] {
            let mut buf = Vec::new();
            encode_int(val, &mut buf);
            let (decoded, _) = decode_key(&buf).unwrap();
            assert_eq!(
                decoded,
                DecodedKey::Int(val),
                "roundtrip failed for {}",
                val
            );
        }
    }

    #[test]
    fn decode_float_roundtrip() {
        for &val in &[-1000.0_f64, -1.5, 1.5, 1000.0] {
            let mut buf = Vec::new();
            encode_float(val, &mut buf);
            let (decoded, _) = decode_key(&buf).unwrap();
            assert_eq!(
                decoded,
                DecodedKey::Float(val),
                "roundtrip failed for {}",
                val
            );
        }
    }

    #[test]
    fn decode_zero_is_int() {
        let mut buf = Vec::new();
        encode_float(0.0, &mut buf);
        let (decoded, _) = decode_key(&buf).unwrap();
        assert_eq!(decoded, DecodedKey::Int(0));

        buf.clear();
        encode_int(0, &mut buf);
        let (decoded2, _) = decode_key(&buf).unwrap();
        assert_eq!(decoded2, DecodedKey::Int(0));
    }

    #[test]
    fn decode_float_special_values() {
        let mut buf = Vec::new();
        encode_float(f64::INFINITY, &mut buf);
        let (decoded, _) = decode_key(&buf).unwrap();
        assert_eq!(decoded, DecodedKey::PosInfinity);

        buf.clear();
        encode_float(f64::NEG_INFINITY, &mut buf);
        let (decoded, _) = decode_key(&buf).unwrap();
        assert_eq!(decoded, DecodedKey::NegInfinity);

        buf.clear();
        encode_float(f64::NAN, &mut buf);
        let (decoded, _) = decode_key(&buf).unwrap();
        assert!(matches!(decoded, DecodedKey::Nan));
    }

    #[test]
    fn decode_text_roundtrip() {
        for &val in &["", "hello", "world", "a\x00b"] {
            let mut buf = Vec::new();
            encode_text(val, &mut buf);
            let (decoded, _) = decode_key(&buf).unwrap();
            assert_eq!(
                decoded,
                DecodedKey::Text(val.to_string()),
                "roundtrip failed for {:?}",
                val
            );
        }
    }

    #[test]
    fn decode_blob_roundtrip() {
        let test_cases: &[&[u8]] = &[&[], b"hello", &[0x00, 0xFF, 0x01]];
        for &val in test_cases {
            let mut buf = Vec::new();
            encode_blob(val, &mut buf);
            let (decoded, _) = decode_key(&buf).unwrap();
            assert_eq!(
                decoded,
                DecodedKey::Blob(val.to_vec()),
                "roundtrip failed for {:?}",
                val
            );
        }
    }

    #[test]
    fn cross_type_ordering_null_first() {
        let mut null_buf = Vec::new();
        encode_null(&mut null_buf);

        let mut int_buf = Vec::new();
        encode_int(-1000, &mut int_buf);

        let mut text_buf = Vec::new();
        encode_text("", &mut text_buf);

        assert!(null_buf < int_buf, "NULL should sort before integers");
        assert!(int_buf < text_buf, "integers should sort before text");
    }

    #[test]
    fn cross_type_ordering_numbers_before_strings() {
        let mut neg_inf = Vec::new();
        encode_float(f64::NEG_INFINITY, &mut neg_inf);

        let mut neg_int = Vec::new();
        encode_int(-100, &mut neg_int);

        let mut zero = Vec::new();
        encode_int(0, &mut zero);

        let mut pos_int = Vec::new();
        encode_int(100, &mut pos_int);

        let mut pos_inf = Vec::new();
        encode_float(f64::INFINITY, &mut pos_inf);

        let mut nan = Vec::new();
        encode_float(f64::NAN, &mut nan);

        let mut text = Vec::new();
        encode_text("a", &mut text);

        assert!(neg_inf < neg_int);
        assert!(neg_int < zero);
        assert!(zero < pos_int);
        assert!(pos_int < pos_inf);
        assert!(pos_inf < nan);
        assert!(nan < text);
    }

    #[test]
    fn comprehensive_type_ordering() {
        #[derive(Debug)]
        enum TestValue {
            Null,
            Int(i64),
            Float(f64),
            Text(&'static str),
            Blob(&'static [u8]),
        }

        let values_in_order = [
            TestValue::Null,
            TestValue::Float(f64::NEG_INFINITY),
            TestValue::Int(-1000),
            TestValue::Int(-1),
            TestValue::Float(-0.5),
            TestValue::Int(0),
            TestValue::Float(0.5),
            TestValue::Int(1),
            TestValue::Int(1000),
            TestValue::Float(f64::INFINITY),
            TestValue::Float(f64::NAN),
            TestValue::Text(""),
            TestValue::Text("a"),
            TestValue::Text("ab"),
            TestValue::Text("b"),
            TestValue::Blob(b""),
            TestValue::Blob(b"a"),
            TestValue::Blob(b"ab"),
        ];

        let encoded: Vec<Vec<u8>> = values_in_order
            .iter()
            .map(|v| {
                let mut buf = Vec::new();
                match v {
                    TestValue::Null => encode_null(&mut buf),
                    TestValue::Int(n) => encode_int(*n, &mut buf),
                    TestValue::Float(f) => encode_float(*f, &mut buf),
                    TestValue::Text(s) => encode_text(s, &mut buf),
                    TestValue::Blob(b) => encode_blob(b, &mut buf),
                }
                buf
            })
            .collect();

        for i in 0..encoded.len() - 1 {
            assert!(
                encoded[i] < encoded[i + 1],
                "ordering failed: {:?} should be < {:?}",
                values_in_order[i],
                values_in_order[i + 1]
            );
        }
    }

    #[test]
    fn multi_column_key_ordering() {
        fn encode_pair(a: i64, b: &str) -> Vec<u8> {
            let mut buf = Vec::new();
            encode_int(a, &mut buf);
            encode_text(b, &mut buf);
            buf
        }

        let keys = [
            encode_pair(-1, "z"),
            encode_pair(0, "a"),
            encode_pair(0, "b"),
            encode_pair(1, "a"),
            encode_pair(1, "z"),
            encode_pair(100, ""),
        ];

        for i in 0..keys.len() - 1 {
            assert!(
                keys[i] < keys[i + 1],
                "multi-column ordering failed at index {}",
                i
            );
        }
    }
}
