//! Fuzz testing for the record builder and serialization.
//!
//! This fuzz target tests RecordBuilder operations to ensure
//! records can be safely built and read back.

#![no_main]

use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;

use turdb::records::types::{ColumnDef, DataType};
use turdb::records::{RecordBuilder, RecordView, Schema};

#[derive(Debug, Arbitrary)]
struct RecordBuilderInput {
    schema: Vec<FuzzColumnDef>,
    operations: Vec<SetOperation>,
}

#[derive(Debug, Arbitrary)]
struct FuzzColumnDef {
    data_type: FuzzDataType,
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
}

#[derive(Debug, Arbitrary)]
enum SetOperation {
    SetNull(u8),
    SetBool(u8, bool),
    SetInt2(u8, i16),
    SetInt4(u8, i32),
    SetInt8(u8, i64),
    SetFloat4(u8, f32),
    SetFloat8(u8, f64),
    SetText(u8, String),
    SetBlob(u8, Vec<u8>),
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
        }
    }
}

fuzz_target!(|input: RecordBuilderInput| {
    if input.schema.is_empty() || input.schema.len() > 32 {
        return;
    }

    let column_defs: Vec<ColumnDef> = input
        .schema
        .iter()
        .enumerate()
        .map(|(i, col)| ColumnDef::new(format!("col{}", i), col.data_type.into()))
        .collect();

    let schema = Schema::new(column_defs);
    let mut builder = RecordBuilder::new(&schema);

    for op in &input.operations {
        let col_count = input.schema.len();
        match op {
            SetOperation::SetNull(idx) => {
                let idx = (*idx as usize) % col_count;
                builder.set_null(idx);
            }
            SetOperation::SetBool(idx, val) => {
                let idx = (*idx as usize) % col_count;
                let _ = builder.set_bool(idx, *val);
            }
            SetOperation::SetInt2(idx, val) => {
                let idx = (*idx as usize) % col_count;
                let _ = builder.set_int2(idx, *val);
            }
            SetOperation::SetInt4(idx, val) => {
                let idx = (*idx as usize) % col_count;
                let _ = builder.set_int4(idx, *val);
            }
            SetOperation::SetInt8(idx, val) => {
                let idx = (*idx as usize) % col_count;
                let _ = builder.set_int8(idx, *val);
            }
            SetOperation::SetFloat4(idx, val) => {
                let idx = (*idx as usize) % col_count;
                let _ = builder.set_float4(idx, *val);
            }
            SetOperation::SetFloat8(idx, val) => {
                let idx = (*idx as usize) % col_count;
                let _ = builder.set_float8(idx, *val);
            }
            SetOperation::SetText(idx, val) => {
                let idx = (*idx as usize) % col_count;
                if val.len() <= 1024 {
                    let _ = builder.set_text(idx, val);
                }
            }
            SetOperation::SetBlob(idx, val) => {
                let idx = (*idx as usize) % col_count;
                if val.len() <= 1024 {
                    let _ = builder.set_blob(idx, val);
                }
            }
        }
    }

    if let Ok(data) = builder.build() {
        let _ = RecordView::new(&data, &schema);
    }
});
