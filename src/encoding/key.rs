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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn encode_null_produces_single_byte_0x01() {
        let mut buf = Vec::new();
        encode_null(&mut buf);
        assert_eq!(buf, vec![type_prefix::NULL]);
    }
}
