//! Fuzz testing for array operations.
//!
//! This fuzz target tests ArrayBuilder and ArrayView to ensure arrays
//! can be safely built, serialized, and read back.

#![no_main]

use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;

use turdb::records::array::{ArrayBuilder, ArrayView};
use turdb::records::types::DataType;

#[derive(Debug, Arbitrary)]
struct ArrayInput {
    element_type: FuzzArrayElemType,
    operations: Vec<ArrayOperation>,
}

#[derive(Debug, Arbitrary, Clone, Copy)]
enum FuzzArrayElemType {
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
enum ArrayOperation {
    PushNull,
    PushBool(bool),
    PushInt2(i16),
    PushInt4(i32),
    PushInt8(i64),
    PushFloat4(f32),
    PushFloat8(f64),
    PushText(String),
    PushBlob(Vec<u8>),
}

impl From<FuzzArrayElemType> for DataType {
    fn from(fdt: FuzzArrayElemType) -> Self {
        match fdt {
            FuzzArrayElemType::Bool => DataType::Bool,
            FuzzArrayElemType::Int2 => DataType::Int2,
            FuzzArrayElemType::Int4 => DataType::Int4,
            FuzzArrayElemType::Int8 => DataType::Int8,
            FuzzArrayElemType::Float4 => DataType::Float4,
            FuzzArrayElemType::Float8 => DataType::Float8,
            FuzzArrayElemType::Text => DataType::Text,
            FuzzArrayElemType::Blob => DataType::Blob,
        }
    }
}

fuzz_target!(|input: ArrayInput| {
    if input.operations.len() > 1000 {
        return;
    }

    let elem_type: DataType = input.element_type.into();
    let mut builder = ArrayBuilder::new(elem_type);

    for op in &input.operations {
        match op {
            ArrayOperation::PushNull => builder.push_null(),
            ArrayOperation::PushBool(v) => {
                if elem_type == DataType::Bool {
                    builder.push_bool(*v);
                }
            }
            ArrayOperation::PushInt2(v) => {
                if elem_type == DataType::Int2 {
                    builder.push_int2(*v);
                }
            }
            ArrayOperation::PushInt4(v) => {
                if elem_type == DataType::Int4 {
                    builder.push_int4(*v);
                }
            }
            ArrayOperation::PushInt8(v) => {
                if elem_type == DataType::Int8 {
                    builder.push_int8(*v);
                }
            }
            ArrayOperation::PushFloat4(v) => {
                if elem_type == DataType::Float4 {
                    builder.push_float4(*v);
                }
            }
            ArrayOperation::PushFloat8(v) => {
                if elem_type == DataType::Float8 {
                    builder.push_float8(*v);
                }
            }
            ArrayOperation::PushText(s) => {
                if elem_type == DataType::Text && s.len() <= 1024 {
                    builder.push_text(s);
                }
            }
            ArrayOperation::PushBlob(b) => {
                if elem_type == DataType::Blob && b.len() <= 1024 {
                    builder.push_blob(b);
                }
            }
        }
    }

    let data = builder.build();
    if let Ok(view) = ArrayView::new(&data) {
        assert_eq!(view.elem_type(), elem_type);
        for i in 0..view.len() {
            let _ = view.is_null(i);
            if !view.is_null(i) {
                match elem_type {
                    DataType::Bool => {
                        let _ = view.get_bool(i);
                    }
                    DataType::Int2 => {
                        let _ = view.get_int2(i);
                    }
                    DataType::Int4 => {
                        let _ = view.get_int4(i);
                    }
                    DataType::Int8 => {
                        let _ = view.get_int8(i);
                    }
                    DataType::Float4 => {
                        let _ = view.get_float4(i);
                    }
                    DataType::Float8 => {
                        let _ = view.get_float8(i);
                    }
                    DataType::Text => {
                        let _ = view.get_text(i);
                    }
                    DataType::Blob => {
                        let _ = view.get_blob(i);
                    }
                    _ => {}
                }
            }
        }
    }
});
