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
}

use eyre::{bail, ensure, Result};

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
            assert_eq!(decoded, DecodedKey::Int(val), "roundtrip failed for {}", val);
        }
    }

    #[test]
    fn decode_float_roundtrip() {
        for &val in &[-1000.0_f64, -1.5, 1.5, 1000.0] {
            let mut buf = Vec::new();
            encode_float(val, &mut buf);
            let (decoded, _) = decode_key(&buf).unwrap();
            assert_eq!(decoded, DecodedKey::Float(val), "roundtrip failed for {}", val);
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
            assert_eq!(decoded, DecodedKey::Text(val.to_string()), "roundtrip failed for {:?}", val);
        }
    }

    #[test]
    fn decode_blob_roundtrip() {
        let test_cases: &[&[u8]] = &[
            &[],
            b"hello",
            &[0x00, 0xFF, 0x01],
        ];
        for &val in test_cases {
            let mut buf = Vec::new();
            encode_blob(val, &mut buf);
            let (decoded, _) = decode_key(&buf).unwrap();
            assert_eq!(decoded, DecodedKey::Blob(val.to_vec()), "roundtrip failed for {:?}", val);
        }
    }
}
