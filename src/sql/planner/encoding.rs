//! # Key Encoding Utilities
//!
//! This module provides functions for encoding SQL literals into byte sequences
//! suitable for index key comparisons. The encoding preserves sort order to enable
//! efficient range scans on B-tree indexes.
//!
//! ## Encoding Scheme
//!
//! Values are encoded with a type prefix byte followed by the encoded value:
//!
//! - Integers: Sign-aware encoding using big-endian for sort preservation
//! - Floats: IEEE 754 with sign bit manipulation for ordering
//! - Text: Escaped encoding with null terminator
//! - Booleans: Single byte (0x02 for false, 0x03 for true)
//!
//! ## Usage
//!
//! These functions are used during physical plan construction to encode
//! index scan bounds from literal expressions.

use crate::sql::ast::{Expr, Literal};
use super::types::ScanRange;
use bumpalo::Bump;

pub(crate) fn encode_literal_to_bytes<'a>(
    arena: &'a Bump,
    expr: &Expr<'a>,
) -> Option<&'a [u8]> {
    match expr {
        Expr::Literal(lit) => {
            let mut buf = bumpalo::collections::Vec::with_capacity_in(32, arena);
            match lit {
                Literal::Null => return None,
                Literal::Boolean(b) => {
                    buf.push(if *b { 0x03 } else { 0x02 });
                }
                Literal::Integer(s) => {
                    if let Ok(n) = s.parse::<i64>() {
                        encode_int_to_arena(n, &mut buf);
                    } else {
                        return None;
                    }
                }
                Literal::Float(s) => {
                    if let Ok(f) = s.parse::<f64>() {
                        encode_float_to_arena(f, &mut buf);
                    } else {
                        return None;
                    }
                }
                Literal::String(s) => {
                    encode_text_to_arena(s, &mut buf);
                }
                Literal::HexNumber(s) => {
                    if let Ok(n) = i64::from_str_radix(s.trim_start_matches("0x"), 16) {
                        encode_int_to_arena(n, &mut buf);
                    } else {
                        return None;
                    }
                }
                Literal::BinaryNumber(s) => {
                    if let Ok(n) = i64::from_str_radix(s.trim_start_matches("0b"), 2) {
                        encode_int_to_arena(n, &mut buf);
                    } else {
                        return None;
                    }
                }
            }
            Some(buf.into_bump_slice())
        }
        _ => None,
    }
}

pub(crate) fn encode_int_to_arena(n: i64, buf: &mut bumpalo::collections::Vec<'_, u8>) {
    use crate::encoding::key::type_prefix;
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

pub(crate) fn encode_float_to_arena(f: f64, buf: &mut bumpalo::collections::Vec<'_, u8>) {
    use crate::encoding::key::type_prefix;
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

pub(crate) fn encode_text_to_arena(s: &str, buf: &mut bumpalo::collections::Vec<'_, u8>) {
    use crate::encoding::key::type_prefix;
    buf.push(type_prefix::TEXT);
    for &byte in s.as_bytes() {
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

#[allow(dead_code)]
pub(crate) fn encode_scan_bounds<'a>(
    arena: &'a Bump,
    bounds: &super::super::optimizer::bounds::ColumnScanBounds<'a>,
) -> ScanRange<'a> {
    if let Some(point) = bounds.point_value {
        if let Some(encoded) = encode_literal_to_bytes(arena, point.value) {
            return ScanRange::PrefixScan { prefix: encoded };
        }
    }

    if bounds.lower.is_some() || bounds.upper.is_some() {
        let start = bounds
            .lower
            .and_then(|b| encode_literal_to_bytes(arena, b.value));
        let end = bounds
            .upper
            .and_then(|b| encode_literal_to_bytes(arena, b.value));
        return ScanRange::RangeScan { start, end };
    }

    ScanRange::FullScan
}
