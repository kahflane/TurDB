//! # Variable-Length Integer Encoding
//!
//! This module provides variable-length integer encoding for TurDB, used for
//! encoding record lengths, cell sizes, and other length values in a space-efficient
//! manner. This is NOT used for type codes (which use fixed-size type prefixes).
//!
//! ## Encoding Format
//!
//! The varint encoding uses a leading byte to indicate the number of following bytes:
//!
//! | Value Range              | Bytes | Format                          |
//! |--------------------------|-------|---------------------------------|
//! | 0 - 240                  | 1     | `[value]`                       |
//! | 241 - 2287               | 2     | `[241 + (v-240)>>8, (v-240)&FF]`|
//! | 2288 - 67823             | 3     | `[249, (v-2288)>>8, (v-2288)&FF]`|
//! | 67824 - 16777215         | 4     | `[250, v>>16, v>>8, v]`         |
//! | 16777216 - 4294967295    | 5     | `[251, v>>24, v>>16, v>>8, v]`  |
//! | 4294967296 - u64::MAX    | 9     | `[255, 8-byte big-endian]`      |
//!
//! ## Design Rationale
//!
//! This encoding scheme is optimized for the common case of small values:
//!
//! - Single-byte encoding for values 0-240 (covers most length fields)
//! - Two-byte encoding extends to 2287 (covers typical row sizes)
//! - Three-byte encoding extends to 67823 (covers large rows)
//! - Four-byte encoding extends to ~16MB (covers overflow thresholds)
//! - Five-byte encoding extends to ~4GB (covers u32::MAX)
//! - Nine-byte encoding covers full u64 range (rarely needed)
//!
//! ## Marker Byte Interpretation
//!
//! ```text
//! Marker 0-240:   Value is the marker itself (1-byte encoding)
//! Marker 241-248: 2-byte encoding, value = 240 + ((marker - 241) << 8) + next_byte
//! Marker 249:     3-byte encoding, value = 2288 + (next_byte << 8) + next_next_byte
//! Marker 250:     4-byte encoding, value in next 3 bytes (big-endian)
//! Marker 251:     5-byte encoding, value in next 4 bytes (big-endian)
//! Marker 252-254: Reserved for future use
//! Marker 255:     9-byte encoding, value in next 8 bytes (big-endian)
//! ```
//!
//! ## Boundary Values
//!
//! Key boundary values for testing:
//!
//! - 240: Maximum 1-byte value
//! - 241: Minimum 2-byte value
//! - 2287: Maximum 2-byte value
//! - 2288: Minimum 3-byte value
//! - 67823: Maximum 3-byte value
//! - 67824: Minimum 4-byte value
//! - 16777215 (0xFF_FFFF): Maximum 4-byte value
//! - 16777216 (0x100_0000): Minimum 5-byte value
//! - 4294967295 (u32::MAX): Maximum 5-byte value
//! - 4294967296 (0x1_0000_0000): Minimum 9-byte value
//! - u64::MAX: Maximum 9-byte value
//!
//! ## Usage Example
//!
//! ```rust
//! use turdb::encoding::varint::{encode_varint, decode_varint, varint_len};
//!
//! // Check encoded length without encoding
//! let len = varint_len(1000);
//! assert_eq!(len, 2);
//!
//! // Encode a value
//! let mut buf = [0u8; 9];
//! let written = encode_varint(1000, &mut buf);
//! assert_eq!(written, 2);
//!
//! // Decode a value
//! let (value, read) = decode_varint(&buf).unwrap();
//! assert_eq!(value, 1000);
//! assert_eq!(read, 2);
//! ```
//!
//! ## Zero-Copy Design
//!
//! All functions operate on byte slices directly:
//! - `encode_varint` writes to a mutable slice, returns bytes written
//! - `decode_varint` reads from a slice, returns (value, bytes_read)
//! - `varint_len` computes length without any I/O
//!
//! No heap allocations are performed by any function in this module.
//!
//! ## Thread Safety
//!
//! All functions are pure and stateless, making them inherently thread-safe.
//! They can be called concurrently without any synchronization.
//!
//! ## Error Handling
//!
//! `decode_varint` returns `eyre::Result` with descriptive error messages:
//! - Empty buffer: "empty buffer for varint decode"
//! - Truncated encoding: "truncated N-byte varint"
//! - Invalid marker: "invalid varint marker: X"

use eyre::{bail, ensure, Result};

pub fn varint_len(value: u64) -> usize {
    if value <= 240 {
        1
    } else if value <= 2287 {
        2
    } else if value <= 67823 {
        3
    } else if value <= 0xFF_FFFF {
        4
    } else if value <= 0xFFFF_FFFF {
        5
    } else {
        9
    }
}

pub fn encode_varint(value: u64, buf: &mut [u8]) -> usize {
    if value <= 240 {
        buf[0] = value as u8;
        1
    } else if value <= 2287 {
        let v = value - 240;
        buf[0] = ((v >> 8) + 241) as u8;
        buf[1] = (v & 0xFF) as u8;
        2
    } else if value <= 67823 {
        let v = value - 2288;
        buf[0] = 249;
        buf[1] = (v >> 8) as u8;
        buf[2] = (v & 0xFF) as u8;
        3
    } else if value <= 0xFF_FFFF {
        buf[0] = 250;
        buf[1] = (value >> 16) as u8;
        buf[2] = (value >> 8) as u8;
        buf[3] = value as u8;
        4
    } else if value <= 0xFFFF_FFFF {
        buf[0] = 251;
        buf[1] = (value >> 24) as u8;
        buf[2] = (value >> 16) as u8;
        buf[3] = (value >> 8) as u8;
        buf[4] = value as u8;
        5
    } else {
        buf[0] = 255;
        buf[1..9].copy_from_slice(&value.to_be_bytes());
        9
    }
}

pub fn decode_varint(buf: &[u8]) -> Result<(u64, usize)> {
    ensure!(!buf.is_empty(), "empty buffer for varint decode");

    let first = buf[0];

    if first <= 240 {
        Ok((first as u64, 1))
    } else if first <= 248 {
        ensure!(buf.len() >= 2, "truncated 2-byte varint");
        let value = 240 + ((first as u64 - 241) << 8) + buf[1] as u64;
        Ok((value, 2))
    } else if first == 249 {
        ensure!(buf.len() >= 3, "truncated 3-byte varint");
        let value = 2288 + ((buf[1] as u64) << 8) + buf[2] as u64;
        Ok((value, 3))
    } else if first == 250 {
        ensure!(buf.len() >= 4, "truncated 4-byte varint");
        let value = ((buf[1] as u64) << 16) + ((buf[2] as u64) << 8) + buf[3] as u64;
        Ok((value, 4))
    } else if first == 251 {
        ensure!(buf.len() >= 5, "truncated 5-byte varint");
        let value = ((buf[1] as u64) << 24)
            + ((buf[2] as u64) << 16)
            + ((buf[3] as u64) << 8)
            + buf[4] as u64;
        Ok((value, 5))
    } else if first == 255 {
        ensure!(buf.len() >= 9, "truncated 9-byte varint");
        let value = u64::from_be_bytes(buf[1..9].try_into().unwrap()); // INVARIANT: length validated by ensure above
        Ok((value, 9))
    } else {
        bail!("invalid varint marker: {}", first)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn varint_len_single_byte_values() {
        assert_eq!(varint_len(0), 1);
        assert_eq!(varint_len(1), 1);
        assert_eq!(varint_len(127), 1);
        assert_eq!(varint_len(240), 1);
    }

    #[test]
    fn varint_len_two_byte_values() {
        assert_eq!(varint_len(241), 2);
        assert_eq!(varint_len(1000), 2);
        assert_eq!(varint_len(2287), 2);
    }

    #[test]
    fn varint_len_three_byte_values() {
        assert_eq!(varint_len(2288), 3);
        assert_eq!(varint_len(50000), 3);
        assert_eq!(varint_len(67823), 3);
    }

    #[test]
    fn varint_len_four_byte_values() {
        assert_eq!(varint_len(67824), 4);
        assert_eq!(varint_len(1_000_000), 4);
        assert_eq!(varint_len(0xFF_FFFF), 4);
    }

    #[test]
    fn varint_len_five_byte_values() {
        assert_eq!(varint_len(0x100_0000), 5);
        assert_eq!(varint_len(0xFFFF_FFFF), 5);
    }

    #[test]
    fn varint_len_nine_byte_values() {
        assert_eq!(varint_len(0x1_0000_0000), 9);
        assert_eq!(varint_len(u64::MAX), 9);
    }

    #[test]
    fn encode_varint_single_byte() {
        let mut buf = [0u8; 9];

        assert_eq!(encode_varint(0, &mut buf), 1);
        assert_eq!(buf[0], 0);

        assert_eq!(encode_varint(240, &mut buf), 1);
        assert_eq!(buf[0], 240);
    }

    #[test]
    fn encode_varint_two_byte() {
        let mut buf = [0u8; 9];

        assert_eq!(encode_varint(241, &mut buf), 2);
        assert_eq!(buf[0], 241);
        assert_eq!(buf[1], 1);

        assert_eq!(encode_varint(2287, &mut buf), 2);
        assert_eq!(buf[0], 248);
        assert_eq!(buf[1], 255);
    }

    #[test]
    fn encode_varint_three_byte() {
        let mut buf = [0u8; 9];

        assert_eq!(encode_varint(2288, &mut buf), 3);
        assert_eq!(buf[0], 249);
        assert_eq!(buf[1], 0);
        assert_eq!(buf[2], 0);

        assert_eq!(encode_varint(67823, &mut buf), 3);
        assert_eq!(buf[0], 249);
        assert_eq!(buf[1], 0xFF);
        assert_eq!(buf[2], 0xFF);
    }

    #[test]
    fn encode_varint_four_byte() {
        let mut buf = [0u8; 9];

        assert_eq!(encode_varint(67824, &mut buf), 4);
        assert_eq!(buf[0], 250);

        assert_eq!(encode_varint(0xFF_FFFF, &mut buf), 4);
        assert_eq!(buf[0], 250);
        assert_eq!(buf[1], 0xFF);
        assert_eq!(buf[2], 0xFF);
        assert_eq!(buf[3], 0xFF);
    }

    #[test]
    fn encode_varint_five_byte() {
        let mut buf = [0u8; 9];

        assert_eq!(encode_varint(0x100_0000, &mut buf), 5);
        assert_eq!(buf[0], 251);

        assert_eq!(encode_varint(0xFFFF_FFFF, &mut buf), 5);
        assert_eq!(buf[0], 251);
        assert_eq!(buf[1], 0xFF);
        assert_eq!(buf[2], 0xFF);
        assert_eq!(buf[3], 0xFF);
        assert_eq!(buf[4], 0xFF);
    }

    #[test]
    fn encode_varint_nine_byte() {
        let mut buf = [0u8; 9];

        assert_eq!(encode_varint(0x1_0000_0000, &mut buf), 9);
        assert_eq!(buf[0], 255);

        assert_eq!(encode_varint(u64::MAX, &mut buf), 9);
        assert_eq!(buf[0], 255);
        assert_eq!(&buf[1..9], &u64::MAX.to_be_bytes());
    }

    #[test]
    fn decode_varint_single_byte() {
        let buf = [0u8];
        let (value, len) = decode_varint(&buf).unwrap();
        assert_eq!(value, 0);
        assert_eq!(len, 1);

        let buf = [240u8];
        let (value, len) = decode_varint(&buf).unwrap();
        assert_eq!(value, 240);
        assert_eq!(len, 1);
    }

    #[test]
    fn decode_varint_two_byte() {
        let buf = [241u8, 1];
        let (value, len) = decode_varint(&buf).unwrap();
        assert_eq!(value, 241);
        assert_eq!(len, 2);

        let buf = [248u8, 255];
        let (value, len) = decode_varint(&buf).unwrap();
        assert_eq!(value, 2287);
        assert_eq!(len, 2);
    }

    #[test]
    fn decode_varint_three_byte() {
        let buf = [249u8, 0, 0];
        let (value, len) = decode_varint(&buf).unwrap();
        assert_eq!(value, 2288);
        assert_eq!(len, 3);

        let buf = [249u8, 0xFF, 0xFF];
        let (value, len) = decode_varint(&buf).unwrap();
        assert_eq!(value, 67823);
        assert_eq!(len, 3);
    }

    #[test]
    fn decode_varint_four_byte() {
        let buf = [250u8, 0x01, 0x08, 0xF0];
        let (value, len) = decode_varint(&buf).unwrap();
        assert_eq!(value, 67824);
        assert_eq!(len, 4);

        let buf = [250u8, 0xFF, 0xFF, 0xFF];
        let (value, len) = decode_varint(&buf).unwrap();
        assert_eq!(value, 0xFF_FFFF);
        assert_eq!(len, 4);
    }

    #[test]
    fn decode_varint_five_byte() {
        let buf = [251u8, 0x01, 0x00, 0x00, 0x00];
        let (value, len) = decode_varint(&buf).unwrap();
        assert_eq!(value, 0x100_0000);
        assert_eq!(len, 5);

        let buf = [251u8, 0xFF, 0xFF, 0xFF, 0xFF];
        let (value, len) = decode_varint(&buf).unwrap();
        assert_eq!(value, 0xFFFF_FFFF);
        assert_eq!(len, 5);
    }

    #[test]
    fn decode_varint_nine_byte() {
        let mut buf = [255u8, 0, 0, 0, 1, 0, 0, 0, 0];
        let (value, len) = decode_varint(&buf).unwrap();
        assert_eq!(value, 0x1_0000_0000);
        assert_eq!(len, 9);

        buf[1..9].copy_from_slice(&u64::MAX.to_be_bytes());
        let (value, len) = decode_varint(&buf).unwrap();
        assert_eq!(value, u64::MAX);
        assert_eq!(len, 9);
    }

    #[test]
    fn decode_varint_empty_buffer_fails() {
        let result = decode_varint(&[]);
        assert!(result.is_err());
    }

    #[test]
    fn decode_varint_truncated_two_byte_fails() {
        let buf = [241u8];
        let result = decode_varint(&buf);
        assert!(result.is_err());
    }

    #[test]
    fn decode_varint_truncated_three_byte_fails() {
        let buf = [249u8, 0];
        let result = decode_varint(&buf);
        assert!(result.is_err());
    }

    #[test]
    fn decode_varint_truncated_four_byte_fails() {
        let buf = [250u8, 0, 0];
        let result = decode_varint(&buf);
        assert!(result.is_err());
    }

    #[test]
    fn decode_varint_truncated_five_byte_fails() {
        let buf = [251u8, 0, 0, 0];
        let result = decode_varint(&buf);
        assert!(result.is_err());
    }

    #[test]
    fn decode_varint_truncated_nine_byte_fails() {
        let buf = [255u8, 0, 0, 0, 0, 0, 0, 0];
        let result = decode_varint(&buf);
        assert!(result.is_err());
    }

    #[test]
    fn decode_varint_invalid_marker_fails() {
        let buf = [252u8, 0, 0, 0, 0];
        let result = decode_varint(&buf);
        assert!(result.is_err());

        let buf = [253u8, 0, 0, 0, 0];
        let result = decode_varint(&buf);
        assert!(result.is_err());

        let buf = [254u8, 0, 0, 0, 0];
        let result = decode_varint(&buf);
        assert!(result.is_err());
    }

    #[test]
    fn roundtrip_boundary_values() {
        let boundary_values = [
            0u64,
            1,
            240,
            241,
            2287,
            2288,
            67823,
            67824,
            0xFF_FFFF,
            0x100_0000,
            0xFFFF_FFFF,
            0x1_0000_0000,
            u64::MAX,
        ];

        for &value in &boundary_values {
            let mut buf = [0u8; 9];
            let encoded_len = encode_varint(value, &mut buf);
            let (decoded, decoded_len) = decode_varint(&buf).unwrap();

            assert_eq!(
                encoded_len, decoded_len,
                "length mismatch for value {}",
                value
            );
            assert_eq!(value, decoded, "value mismatch for value {}", value);
            assert_eq!(
                varint_len(value),
                encoded_len,
                "varint_len mismatch for value {}",
                value
            );
        }
    }

    #[test]
    fn roundtrip_random_values() {
        let test_values = [
            42u64,
            100,
            255,
            256,
            1000,
            10000,
            100000,
            1_000_000,
            10_000_000,
            100_000_000,
            1_000_000_000,
            10_000_000_000,
            0x7FFF_FFFF_FFFF_FFFF,
        ];

        for &value in &test_values {
            let mut buf = [0u8; 9];
            let encoded_len = encode_varint(value, &mut buf);
            let (decoded, decoded_len) = decode_varint(&buf).unwrap();

            assert_eq!(
                encoded_len, decoded_len,
                "length mismatch for value {}",
                value
            );
            assert_eq!(value, decoded, "value mismatch for value {}", value);
        }
    }
}
