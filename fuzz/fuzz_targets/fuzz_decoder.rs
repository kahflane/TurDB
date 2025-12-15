//! Fuzz testing for the record decoder.
//!
//! This fuzz target tests the SimpleDecoder with arbitrary byte sequences
//! to ensure it handles malformed input gracefully without panicking or
//! causing undefined behavior.

#![no_main]

use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;

use turdb::records::types::DataType;
use turdb::sql::decoder::{RecordDecoder, SimpleDecoder};

#[derive(Debug, Arbitrary)]
struct DecoderInput {
    column_types: Vec<FuzzDataType>,
    data: Vec<u8>,
}

#[derive(Debug, Arbitrary, Clone, Copy)]
enum FuzzDataType {
    Bool,
    Int2,
    Int4,
    Int8,
    Float4,
    Float8,
    Text,
    Blob,
    Uuid,
    Vector,
    Jsonb,
    Date,
    Time,
    Timestamp,
    TimestampTz,
    MacAddr,
    Inet4,
    Inet6,
    Interval,
    Point,
    Box,
    Circle,
    Enum,
    Decimal,
    Array,
    Composite,
    Int4Range,
    Int8Range,
    DateRange,
    TimestampRange,
}

impl From<FuzzDataType> for DataType {
    fn from(fdt: FuzzDataType) -> Self {
        match fdt {
            FuzzDataType::Bool => DataType::Bool,
            FuzzDataType::Int2 => DataType::Int2,
            FuzzDataType::Int4 => DataType::Int4,
            FuzzDataType::Int8 => DataType::Int8,
            FuzzDataType::Float4 => DataType::Float4,
            FuzzDataType::Float8 => DataType::Float8,
            FuzzDataType::Text => DataType::Text,
            FuzzDataType::Blob => DataType::Blob,
            FuzzDataType::Uuid => DataType::Uuid,
            FuzzDataType::Vector => DataType::Vector,
            FuzzDataType::Jsonb => DataType::Jsonb,
            FuzzDataType::Date => DataType::Date,
            FuzzDataType::Time => DataType::Time,
            FuzzDataType::Timestamp => DataType::Timestamp,
            FuzzDataType::TimestampTz => DataType::TimestampTz,
            FuzzDataType::MacAddr => DataType::MacAddr,
            FuzzDataType::Inet4 => DataType::Inet4,
            FuzzDataType::Inet6 => DataType::Inet6,
            FuzzDataType::Interval => DataType::Interval,
            FuzzDataType::Point => DataType::Point,
            FuzzDataType::Box => DataType::Box,
            FuzzDataType::Circle => DataType::Circle,
            FuzzDataType::Enum => DataType::Enum,
            FuzzDataType::Decimal => DataType::Decimal,
            FuzzDataType::Array => DataType::Array,
            FuzzDataType::Composite => DataType::Composite,
            FuzzDataType::Int4Range => DataType::Int4Range,
            FuzzDataType::Int8Range => DataType::Int8Range,
            FuzzDataType::DateRange => DataType::DateRange,
            FuzzDataType::TimestampRange => DataType::TimestampRange,
        }
    }
}

fuzz_target!(|input: DecoderInput| {
    if input.column_types.is_empty() || input.column_types.len() > 64 {
        return;
    }

    let column_types: Vec<DataType> = input.column_types.into_iter().map(Into::into).collect();
    let decoder = SimpleDecoder::new(column_types);
    let _ = decoder.decode(&[], &input.data);
});
