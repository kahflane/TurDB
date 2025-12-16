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

pub trait KeyBuffer {
    fn push(&mut self, byte: u8);
    fn extend_from_slice(&mut self, bytes: &[u8]);
}

impl KeyBuffer for Vec<u8> {
    #[inline]
    fn push(&mut self, byte: u8) {
        Vec::push(self, byte);
    }

    #[inline]
    fn extend_from_slice(&mut self, bytes: &[u8]) {
        self.extend(bytes);
    }
}

impl<A: smallvec::Array<Item = u8>> KeyBuffer for smallvec::SmallVec<A> {
    #[inline]
    fn push(&mut self, byte: u8) {
        smallvec::SmallVec::push(self, byte);
    }

    #[inline]
    fn extend_from_slice(&mut self, bytes: &[u8]) {
        self.extend(bytes.iter().copied());
    }
}

pub fn encode_null<B: KeyBuffer>(buf: &mut B) {
    buf.push(type_prefix::NULL);
}

pub fn encode_bool<B: KeyBuffer>(b: bool, buf: &mut B) {
    buf.push(if b {
        type_prefix::TRUE
    } else {
        type_prefix::FALSE
    });
}

pub fn encode_int<B: KeyBuffer>(n: i64, buf: &mut B) {
    if n < 0 {
        buf.push(type_prefix::NEG_INT);
        buf.extend_from_slice(&(n as u64).to_be_bytes());
    } else if n == 0 {
        buf.push(type_prefix::ZERO);
    } else {
        buf.push(type_prefix::POS_INT);
        buf.extend_from_slice(&(n as u64).to_be_bytes());
    }
}

pub fn encode_float<B: KeyBuffer>(f: f64, buf: &mut B) {
    if f.is_nan() {
        buf.push(type_prefix::NAN);
    } else if f == f64::NEG_INFINITY {
        buf.push(type_prefix::NEG_INFINITY);
    } else if f == f64::INFINITY {
        buf.push(type_prefix::POS_INFINITY);
    } else if f < 0.0 {
        buf.push(type_prefix::NEG_FLOAT);
        buf.extend_from_slice(&(!f.to_bits()).to_be_bytes());
    } else if f == 0.0 {
        buf.push(type_prefix::ZERO);
    } else {
        buf.push(type_prefix::POS_FLOAT);
        buf.extend_from_slice(&(f.to_bits() ^ (1u64 << 63)).to_be_bytes());
    }
}

pub fn encode_text<B: KeyBuffer>(s: &str, buf: &mut B) {
    buf.push(type_prefix::TEXT);
    encode_escaped_bytes(s.as_bytes(), buf);
}

pub fn encode_blob<B: KeyBuffer>(data: &[u8], buf: &mut B) {
    buf.push(type_prefix::BLOB);
    encode_escaped_bytes(data, buf);
}

pub fn encode_date<B: KeyBuffer>(days: i32, buf: &mut B) {
    buf.push(type_prefix::DATE);
    buf.extend_from_slice(&((days as u32) ^ (1u32 << 31)).to_be_bytes());
}

pub fn encode_timestamp<B: KeyBuffer>(micros: i64, buf: &mut B) {
    buf.push(type_prefix::TIMESTAMP);
    buf.extend_from_slice(&((micros as u64) ^ (1u64 << 63)).to_be_bytes());
}

pub fn encode_uuid<B: KeyBuffer>(uuid: &[u8; 16], buf: &mut B) {
    buf.push(type_prefix::UUID);
    buf.extend_from_slice(uuid);
}

pub fn encode_time<B: KeyBuffer>(micros: i64, buf: &mut B) {
    buf.push(type_prefix::TIME);
    buf.extend_from_slice(&((micros as u64) ^ (1u64 << 63)).to_be_bytes());
}

pub fn encode_timestamptz<B: KeyBuffer>(micros: i64, tz_offset_mins: i16, buf: &mut B) {
    buf.push(type_prefix::TIMESTAMPTZ);
    buf.extend_from_slice(&((micros as u64) ^ (1u64 << 63)).to_be_bytes());
    buf.extend_from_slice(&((tz_offset_mins as u16) ^ (1u16 << 15)).to_be_bytes());
}

pub fn encode_interval<B: KeyBuffer>(months: i32, days: i32, micros: i64, buf: &mut B) {
    buf.push(type_prefix::INTERVAL);
    buf.extend_from_slice(&((months as u32) ^ (1u32 << 31)).to_be_bytes());
    buf.extend_from_slice(&((days as u32) ^ (1u32 << 31)).to_be_bytes());
    buf.extend_from_slice(&((micros as u64) ^ (1u64 << 63)).to_be_bytes());
}

pub fn encode_inet<B: KeyBuffer>(is_ipv6: bool, addr: &[u8], prefix_len: u8, buf: &mut B) {
    buf.push(type_prefix::INET);
    buf.push(if is_ipv6 { 1 } else { 0 });
    buf.push(prefix_len);
    if is_ipv6 {
        buf.extend_from_slice(&addr[..16]);
    } else {
        buf.extend_from_slice(&addr[..4]);
    }
}

pub fn encode_macaddr<B: KeyBuffer>(addr: &[u8; 6], buf: &mut B) {
    buf.push(type_prefix::MACADDR);
    buf.extend_from_slice(addr);
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

pub fn encode_vector<B: KeyBuffer>(dimensions: &[f32], buf: &mut B) {
    buf.push(type_prefix::VECTOR);
    buf.extend_from_slice(&(dimensions.len() as u32).to_be_bytes());
    for &dim in dimensions {
        let bits = dim.to_bits();
        let encoded = if dim < 0.0 {
            !bits
        } else {
            bits ^ (1u32 << 31)
        };
        buf.extend_from_slice(&encoded.to_be_bytes());
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

pub fn encode_enum<B: KeyBuffer>(type_id: u32, ordinal: u32, buf: &mut B) {
    buf.push(type_prefix::ENUM);
    buf.extend_from_slice(&type_id.to_be_bytes());
    buf.extend_from_slice(&ordinal.to_be_bytes());
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

pub fn encode_json<B: KeyBuffer>(json: &JsonValue, buf: &mut B) {
    match json {
        JsonValue::Null => buf.push(type_prefix::JSON_NULL),
        JsonValue::Bool(false) => buf.push(type_prefix::JSON_FALSE),
        JsonValue::Bool(true) => buf.push(type_prefix::JSON_TRUE),
        JsonValue::Number(n) => {
            buf.push(type_prefix::JSON_NUMBER);
            if *n < 0.0 {
                buf.extend_from_slice(&(!n.to_bits()).to_be_bytes());
            } else {
                buf.extend_from_slice(&(n.to_bits() ^ (1u64 << 63)).to_be_bytes());
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

fn encode_escaped_bytes<B: KeyBuffer>(data: &[u8], buf: &mut B) {
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

pub fn encode_value<B: KeyBuffer>(value: &Value, buf: &mut B) {
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
            let bytes: [u8; 8] = data[1..9].try_into().unwrap(); // SAFETY: length validated by ensure above
            let val = i64::from_be_bytes(bytes);
            Ok((DecodedKey::Int(val), 9))
        }
        type_prefix::POS_INT => {
            ensure!(data.len() >= 9, "truncated positive integer");
            let bytes: [u8; 8] = data[1..9].try_into().unwrap(); // SAFETY: length validated by ensure above
            let val = u64::from_be_bytes(bytes) as i64;
            Ok((DecodedKey::Int(val), 9))
        }
        type_prefix::NEG_FLOAT => {
            ensure!(data.len() >= 9, "truncated negative float");
            let bytes: [u8; 8] = data[1..9].try_into().unwrap(); // SAFETY: length validated by ensure above
            let bits = !u64::from_be_bytes(bytes);
            let val = f64::from_bits(bits);
            Ok((DecodedKey::Float(val), 9))
        }
        type_prefix::POS_FLOAT => {
            ensure!(data.len() >= 9, "truncated positive float");
            let bytes: [u8; 8] = data[1..9].try_into().unwrap(); // SAFETY: length validated by ensure above
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
            let bytes: [u8; 4] = data[1..5].try_into().unwrap(); // SAFETY: length validated by ensure above
            let encoded = u32::from_be_bytes(bytes);
            let days = (encoded ^ (1u32 << 31)) as i32;
            Ok((DecodedKey::Date(days), 5))
        }
        type_prefix::TIME => {
            ensure!(data.len() >= 9, "truncated time");
            let bytes: [u8; 8] = data[1..9].try_into().unwrap(); // SAFETY: length validated by ensure above
            let encoded = u64::from_be_bytes(bytes);
            let micros = (encoded ^ (1u64 << 63)) as i64;
            Ok((DecodedKey::Time(micros), 9))
        }
        type_prefix::TIMESTAMP => {
            ensure!(data.len() >= 9, "truncated timestamp");
            let bytes: [u8; 8] = data[1..9].try_into().unwrap(); // SAFETY: length validated by ensure above
            let encoded = u64::from_be_bytes(bytes);
            let micros = (encoded ^ (1u64 << 63)) as i64;
            Ok((DecodedKey::Timestamp(micros), 9))
        }
        type_prefix::TIMESTAMPTZ => {
            ensure!(data.len() >= 11, "truncated timestamptz");
            let ts_bytes: [u8; 8] = data[1..9].try_into().unwrap(); // SAFETY: length validated by ensure above
            let tz_bytes: [u8; 2] = data[9..11].try_into().unwrap(); // SAFETY: length validated by ensure above
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
            let m_bytes: [u8; 4] = data[1..5].try_into().unwrap(); // SAFETY: length validated by ensure above
            let d_bytes: [u8; 4] = data[5..9].try_into().unwrap(); // SAFETY: length validated by ensure above
            let u_bytes: [u8; 8] = data[9..17].try_into().unwrap(); // SAFETY: length validated by ensure above
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
            let bytes: [u8; 16] = data[1..17].try_into().unwrap(); // SAFETY: length validated by ensure above
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
            let bytes: [u8; 6] = data[1..7].try_into().unwrap(); // SAFETY: length validated by ensure above
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
            let type_id = u32::from_be_bytes(data[1..5].try_into().unwrap()); // SAFETY: length validated by ensure above
            let ordinal = u32::from_be_bytes(data[5..9].try_into().unwrap()); // SAFETY: length validated by ensure above
            Ok((DecodedKey::Enum { type_id, ordinal }, 9))
        }
        type_prefix::COMPOSITE => {
            ensure!(data.len() >= 5, "truncated composite");
            let type_id = u32::from_be_bytes(data[1..5].try_into().unwrap()); // SAFETY: length validated by ensure above
            let (fields, consumed) = decode_composite_fields(&data[5..])?;
            Ok((DecodedKey::Composite { type_id, fields }, 5 + consumed))
        }
        type_prefix::DOMAIN => {
            ensure!(data.len() >= 5, "truncated domain");
            let type_id = u32::from_be_bytes(data[1..5].try_into().unwrap()); // SAFETY: length validated by ensure above
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
            let dim_count = u32::from_be_bytes(data[1..5].try_into().unwrap()) as usize; // SAFETY: length validated by ensure above
            ensure!(
                data.len() >= 5 + dim_count * 4,
                "truncated vector dimensions"
            );
            let mut dimensions = Vec::with_capacity(dim_count);
            for i in 0..dim_count {
                let start = 5 + i * 4;
                let encoded = u32::from_be_bytes(data[start..start + 4].try_into().unwrap()); // SAFETY: length validated by ensure above
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
            let bytes: [u8; 8] = data[1..9].try_into().unwrap(); // SAFETY: length validated by ensure above
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
