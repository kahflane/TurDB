//! Key encoding benchmarks for TurDB
//!
//! These benchmarks measure the performance of the key encoding system
//! which is critical for B-tree key comparison and storage efficiency.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use std::hint::black_box as hint_black_box;
use turdb::encoding::key::{
    decode_key, encode_blob, encode_float, encode_int, encode_null, encode_text, encode_value,
    Value,
};
use turdb::encoding::varint::{decode_varint, encode_varint};

fn bench_varint_encode(c: &mut Criterion) {
    let mut group = c.benchmark_group("varint_encode");

    let test_values: Vec<(u64, &str)> = vec![
        (0, "zero"),
        (127, "1_byte_max"),
        (16383, "2_byte_max"),
        (2097151, "3_byte_max"),
        (268435455, "4_byte_max"),
        (u64::MAX, "max_u64"),
    ];

    for (value, name) in test_values {
        group.bench_with_input(BenchmarkId::new("encode", name), &value, |b, &value| {
            let mut buf = [0u8; 9];
            b.iter(|| {
                let len = encode_varint(black_box(value), &mut buf);
                hint_black_box(len)
            });
        });
    }

    group.finish();
}

fn bench_varint_decode(c: &mut Criterion) {
    let mut group = c.benchmark_group("varint_decode");

    let test_values: Vec<(u64, &str)> = vec![
        (0, "zero"),
        (127, "1_byte_max"),
        (16383, "2_byte_max"),
        (2097151, "3_byte_max"),
        (268435455, "4_byte_max"),
        (u64::MAX, "max_u64"),
    ];

    for (value, name) in test_values {
        let mut buf = [0u8; 9];
        let len = encode_varint(value, &mut buf);

        group.bench_with_input(BenchmarkId::new("decode", name), &buf[..len], |b, data| {
            b.iter(|| {
                let result = decode_varint(black_box(data));
                hint_black_box(result)
            });
        });
    }

    group.finish();
}

fn bench_key_encode(c: &mut Criterion) {
    let mut group = c.benchmark_group("key_encode");

    group.bench_function("null", |b| {
        let mut buf = Vec::with_capacity(16);
        b.iter(|| {
            buf.clear();
            encode_null(&mut buf);
            hint_black_box(buf.len())
        });
    });

    group.bench_function("int_positive", |b| {
        let mut buf = Vec::with_capacity(16);
        b.iter(|| {
            buf.clear();
            encode_int(black_box(12345678), &mut buf);
            hint_black_box(buf.len())
        });
    });

    group.bench_function("int_negative", |b| {
        let mut buf = Vec::with_capacity(16);
        b.iter(|| {
            buf.clear();
            encode_int(black_box(-12345678), &mut buf);
            hint_black_box(buf.len())
        });
    });

    group.bench_function("int_zero", |b| {
        let mut buf = Vec::with_capacity(16);
        b.iter(|| {
            buf.clear();
            encode_int(black_box(0), &mut buf);
            hint_black_box(buf.len())
        });
    });

    group.bench_function("float_positive", |b| {
        let mut buf = Vec::with_capacity(16);
        b.iter(|| {
            buf.clear();
            encode_float(black_box(std::f64::consts::PI), &mut buf);
            hint_black_box(buf.len())
        });
    });

    group.bench_function("float_negative", |b| {
        let mut buf = Vec::with_capacity(16);
        b.iter(|| {
            buf.clear();
            encode_float(black_box(-std::f64::consts::PI), &mut buf);
            hint_black_box(buf.len())
        });
    });

    group.bench_function("text_short", |b| {
        let mut buf = Vec::with_capacity(32);
        let text = "hello";
        b.iter(|| {
            buf.clear();
            encode_text(black_box(text), &mut buf);
            hint_black_box(buf.len())
        });
    });

    group.bench_function("text_medium", |b| {
        let mut buf = Vec::with_capacity(64);
        let text = "The quick brown fox jumps over the lazy dog";
        b.iter(|| {
            buf.clear();
            encode_text(black_box(text), &mut buf);
            hint_black_box(buf.len())
        });
    });

    group.bench_function("blob_16", |b| {
        let mut buf = Vec::with_capacity(32);
        let blob: Vec<u8> = (0..16).collect();
        b.iter(|| {
            buf.clear();
            encode_blob(black_box(&blob), &mut buf);
            hint_black_box(buf.len())
        });
    });

    group.bench_function("blob_256", |b| {
        let mut buf = Vec::with_capacity(300);
        let blob: Vec<u8> = (0..=255).collect();
        b.iter(|| {
            buf.clear();
            encode_blob(black_box(&blob), &mut buf);
            hint_black_box(buf.len())
        });
    });

    group.finish();
}

fn bench_key_decode(c: &mut Criterion) {
    let mut group = c.benchmark_group("key_decode");

    type EncodeFn = Box<dyn Fn(&mut Vec<u8>)>;
    let test_cases: Vec<(&str, EncodeFn)> = vec![
        ("null", Box::new(|buf: &mut Vec<u8>| encode_null(buf))),
        (
            "int_positive",
            Box::new(|buf: &mut Vec<u8>| encode_int(12345678, buf)),
        ),
        (
            "int_negative",
            Box::new(|buf: &mut Vec<u8>| encode_int(-12345678, buf)),
        ),
        (
            "float",
            Box::new(|buf: &mut Vec<u8>| encode_float(std::f64::consts::PI, buf)),
        ),
        (
            "text_short",
            Box::new(|buf: &mut Vec<u8>| encode_text("hello", buf)),
        ),
        (
            "text_medium",
            Box::new(|buf: &mut Vec<u8>| {
                encode_text("The quick brown fox jumps over the lazy dog", buf)
            }),
        ),
    ];

    for (name, encode_fn) in test_cases {
        let mut buf = Vec::new();
        encode_fn(&mut buf);

        group.bench_with_input(BenchmarkId::new("decode", name), &buf, |b, data| {
            b.iter(|| {
                let result = decode_key(black_box(data));
                hint_black_box(result)
            });
        });
    }

    group.finish();
}

fn bench_encode_value(c: &mut Criterion) {
    let mut group = c.benchmark_group("encode_value");

    group.bench_function("null", |b| {
        let mut buf = Vec::with_capacity(16);
        let value = Value::Null;
        b.iter(|| {
            buf.clear();
            encode_value(black_box(&value), &mut buf);
            hint_black_box(buf.len())
        });
    });

    group.bench_function("bool_true", |b| {
        let mut buf = Vec::with_capacity(16);
        let value = Value::Bool(true);
        b.iter(|| {
            buf.clear();
            encode_value(black_box(&value), &mut buf);
            hint_black_box(buf.len())
        });
    });

    group.bench_function("int", |b| {
        let mut buf = Vec::with_capacity(16);
        let value = Value::Int(12345678);
        b.iter(|| {
            buf.clear();
            encode_value(black_box(&value), &mut buf);
            hint_black_box(buf.len())
        });
    });

    group.bench_function("float", |b| {
        let mut buf = Vec::with_capacity(16);
        let value = Value::Float(std::f64::consts::PI);
        b.iter(|| {
            buf.clear();
            encode_value(black_box(&value), &mut buf);
            hint_black_box(buf.len())
        });
    });

    group.bench_function("text", |b| {
        let mut buf = Vec::with_capacity(64);
        let value = Value::Text("hello world");
        b.iter(|| {
            buf.clear();
            encode_value(black_box(&value), &mut buf);
            hint_black_box(buf.len())
        });
    });

    group.bench_function("blob", |b| {
        let mut buf = Vec::with_capacity(64);
        let data: Vec<u8> = (0..32).collect();
        let value = Value::Blob(&data);
        b.iter(|| {
            buf.clear();
            encode_value(black_box(&value), &mut buf);
            hint_black_box(buf.len())
        });
    });

    group.finish();
}

fn bench_multi_column_key(c: &mut Criterion) {
    let mut group = c.benchmark_group("multi_column_key");

    group.bench_function("encode_3_columns", |b| {
        let mut buf = Vec::with_capacity(64);
        b.iter(|| {
            buf.clear();
            encode_int(black_box(42), &mut buf);
            encode_text(black_box("user_123"), &mut buf);
            encode_float(black_box(99.99), &mut buf);
            hint_black_box(buf.len())
        });
    });

    group.bench_function("encode_5_columns", |b| {
        let mut buf = Vec::with_capacity(128);
        b.iter(|| {
            buf.clear();
            encode_int(black_box(1), &mut buf);
            encode_text(black_box("category"), &mut buf);
            encode_int(black_box(2024), &mut buf);
            encode_text(black_box("product_abc"), &mut buf);
            encode_float(black_box(123.456), &mut buf);
            hint_black_box(buf.len())
        });
    });

    group.finish();
}

fn bench_key_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("key_comparison");

    let mut key1 = Vec::new();
    encode_int(100, &mut key1);
    encode_text("apple", &mut key1);

    let mut key2 = Vec::new();
    encode_int(100, &mut key2);
    encode_text("banana", &mut key2);

    let mut key3 = Vec::new();
    encode_int(100, &mut key3);
    encode_text("apple", &mut key3);

    group.bench_function("compare_different", |b| {
        b.iter(|| {
            let result = black_box(&key1).cmp(black_box(&key2));
            hint_black_box(result)
        });
    });

    group.bench_function("compare_equal", |b| {
        b.iter(|| {
            let result = black_box(&key1).cmp(black_box(&key3));
            hint_black_box(result)
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_varint_encode,
    bench_varint_decode,
    bench_key_encode,
    bench_key_decode,
    bench_encode_value,
    bench_multi_column_key,
    bench_key_comparison,
);
criterion_main!(benches);
