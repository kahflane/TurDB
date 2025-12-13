//! Record/Value benchmarks for TurDB
//!
//! These benchmarks measure the performance of value creation, coercion,
//! and comparison operations which are fundamental to record processing.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use std::borrow::Cow;
use std::hint::black_box as hint_black_box;
use turdb::types::{DataType, TypeAffinity, Value};

fn bench_value_creation(c: &mut Criterion) {
    let mut group = c.benchmark_group("value_creation");

    group.bench_function("null", |b| {
        b.iter(|| hint_black_box(Value::Null));
    });

    group.bench_function("int", |b| {
        b.iter(|| hint_black_box(Value::Int(black_box(42))));
    });

    group.bench_function("float", |b| {
        b.iter(|| hint_black_box(Value::Float(black_box(std::f64::consts::PI))));
    });

    group.bench_function("text_borrowed", |b| {
        let text = "hello, world";
        b.iter(|| hint_black_box(Value::Text(Cow::Borrowed(black_box(text)))));
    });

    group.bench_function("text_owned", |b| {
        b.iter(|| hint_black_box(Value::Text(Cow::Owned(String::from("hello, world")))));
    });

    group.bench_function("blob_borrowed", |b| {
        let blob: &[u8] = b"binary data here";
        b.iter(|| hint_black_box(Value::Blob(Cow::Borrowed(black_box(blob)))));
    });

    group.bench_function("vector_borrowed", |b| {
        let vec: &[f32] = &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        b.iter(|| hint_black_box(Value::Vector(Cow::Borrowed(black_box(vec)))));
    });

    group.finish();
}

fn bench_value_coercion(c: &mut Criterion) {
    let mut group = c.benchmark_group("value_coercion");

    group.bench_function("int_to_float", |b| {
        let value = Value::Int(12345678);
        b.iter(|| {
            let result = value.coerce_to_affinity(black_box(TypeAffinity::Real));
            hint_black_box(result)
        });
    });

    group.bench_function("int_to_text", |b| {
        let value = Value::Int(12345678);
        b.iter(|| {
            let result = value.coerce_to_affinity(black_box(TypeAffinity::Text));
            hint_black_box(result)
        });
    });

    group.bench_function("float_to_int", |b| {
        let value = Value::Float(12345.678);
        b.iter(|| {
            let result = value.coerce_to_affinity(black_box(TypeAffinity::Integer));
            hint_black_box(result)
        });
    });

    group.bench_function("text_to_int_valid", |b| {
        let value = Value::Text(Cow::Borrowed("42"));
        b.iter(|| {
            let result = value.coerce_to_affinity(black_box(TypeAffinity::Integer));
            hint_black_box(result)
        });
    });

    group.bench_function("text_to_float_valid", |b| {
        let value = Value::Text(Cow::Borrowed("3.14159"));
        b.iter(|| {
            let result = value.coerce_to_affinity(black_box(TypeAffinity::Real));
            hint_black_box(result)
        });
    });

    group.bench_function("null_coercion", |b| {
        let value = Value::Null;
        b.iter(|| {
            let result = value.coerce_to_affinity(black_box(TypeAffinity::Integer));
            hint_black_box(result)
        });
    });

    group.finish();
}

fn bench_value_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("value_comparison");

    group.bench_function("int_vs_int", |b| {
        let v1 = Value::Int(42);
        let v2 = Value::Int(100);
        b.iter(|| {
            let result = black_box(&v1).compare(black_box(&v2));
            hint_black_box(result)
        });
    });

    group.bench_function("float_vs_float", |b| {
        let v1 = Value::Float(std::f64::consts::PI);
        let v2 = Value::Float(std::f64::consts::E);
        b.iter(|| {
            let result = black_box(&v1).compare(black_box(&v2));
            hint_black_box(result)
        });
    });

    group.bench_function("int_vs_float", |b| {
        let v1 = Value::Int(42);
        let v2 = Value::Float(42.5);
        b.iter(|| {
            let result = black_box(&v1).compare(black_box(&v2));
            hint_black_box(result)
        });
    });

    group.bench_function("text_short", |b| {
        let v1 = Value::Text(Cow::Borrowed("apple"));
        let v2 = Value::Text(Cow::Borrowed("banana"));
        b.iter(|| {
            let result = black_box(&v1).compare(black_box(&v2));
            hint_black_box(result)
        });
    });

    group.bench_function("text_medium", |b| {
        let v1 = Value::Text(Cow::Borrowed("The quick brown fox jumps over the lazy dog"));
        let v2 = Value::Text(Cow::Borrowed("The quick brown fox jumps over the lazy cat"));
        b.iter(|| {
            let result = black_box(&v1).compare(black_box(&v2));
            hint_black_box(result)
        });
    });

    group.bench_function("blob_16", |b| {
        let blob1: Vec<u8> = (0..16).collect();
        let blob2: Vec<u8> = (1..17).collect();
        let v1 = Value::Blob(Cow::Owned(blob1));
        let v2 = Value::Blob(Cow::Owned(blob2));
        b.iter(|| {
            let result = black_box(&v1).compare(black_box(&v2));
            hint_black_box(result)
        });
    });

    group.bench_function("null_comparison", |b| {
        let v1 = Value::Null;
        let v2 = Value::Int(42);
        b.iter(|| {
            let result = black_box(&v1).compare(black_box(&v2));
            hint_black_box(result)
        });
    });

    group.bench_function("vector_8", |b| {
        let vec1: Vec<f32> = (0..8).map(|i| i as f32).collect();
        let vec2: Vec<f32> = (1..9).map(|i| i as f32).collect();
        let v1 = Value::Vector(Cow::Owned(vec1));
        let v2 = Value::Vector(Cow::Owned(vec2));
        b.iter(|| {
            let result = black_box(&v1).compare(black_box(&v2));
            hint_black_box(result)
        });
    });

    group.finish();
}

fn bench_type_affinity(c: &mut Criterion) {
    let mut group = c.benchmark_group("type_affinity");

    let data_types = vec![
        (DataType::Int8, "int8"),
        (DataType::Float8, "float8"),
        (DataType::Text, "text"),
        (DataType::Varchar(255), "varchar"),
        (DataType::Blob, "blob"),
        (DataType::Timestamp, "timestamp"),
        (DataType::Vector(128), "vector"),
        (DataType::Array(Box::new(DataType::Int4)), "array"),
    ];

    for (dtype, name) in data_types {
        group.bench_with_input(BenchmarkId::new("lookup", name), &dtype, |b, dtype| {
            b.iter(|| {
                let result = black_box(dtype).affinity();
                hint_black_box(result)
            });
        });
    }

    group.finish();
}

fn bench_value_clone(c: &mut Criterion) {
    let mut group = c.benchmark_group("value_clone");

    group.bench_function("int", |b| {
        let v = Value::Int(42);
        b.iter(|| hint_black_box(black_box(&v).clone()));
    });

    group.bench_function("float", |b| {
        let v = Value::Float(std::f64::consts::PI);
        b.iter(|| hint_black_box(black_box(&v).clone()));
    });

    group.bench_function("text_borrowed", |b| {
        let v = Value::Text(Cow::Borrowed("hello, world"));
        b.iter(|| hint_black_box(black_box(&v).clone()));
    });

    group.bench_function("text_owned_short", |b| {
        let v = Value::Text(Cow::Owned(String::from("hello")));
        b.iter(|| hint_black_box(black_box(&v).clone()));
    });

    group.bench_function("text_owned_long", |b| {
        let v = Value::Text(Cow::Owned(
            "The quick brown fox jumps over the lazy dog".to_string(),
        ));
        b.iter(|| hint_black_box(black_box(&v).clone()));
    });

    group.bench_function("blob_owned_256", |b| {
        let blob: Vec<u8> = (0..=255).collect();
        let v = Value::Blob(Cow::Owned(blob));
        b.iter(|| hint_black_box(black_box(&v).clone()));
    });

    group.bench_function("vector_128", |b| {
        let vec: Vec<f32> = (0..128).map(|i| i as f32 * 0.01).collect();
        let v = Value::Vector(Cow::Owned(vec));
        b.iter(|| hint_black_box(black_box(&v).clone()));
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_value_creation,
    bench_value_coercion,
    bench_value_comparison,
    bench_type_affinity,
    bench_value_clone,
);
criterion_main!(benches);
