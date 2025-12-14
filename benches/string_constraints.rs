//! String constraint benchmarks comparing TurDB vs SQLite
//!
//! These benchmarks measure the performance of CHAR/VARCHAR constraint
//! checking between TurDB and SQLite to validate our implementation
//! is competitive with the industry standard.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use rusqlite::Connection;
use turdb::records::{builder::RecordBuilder, schema::Schema, types::ColumnDef, view::RecordView};

fn setup_turdb_char_schema(length: u32) -> Schema {
    Schema::new(vec![ColumnDef::new_char("code", length)])
}

fn setup_turdb_varchar_schema(length: u32) -> Schema {
    Schema::new(vec![ColumnDef::new_varchar("name", Some(length))])
}

fn setup_sqlite_char_table(conn: &Connection, length: u32) {
    conn.execute("DROP TABLE IF EXISTS test_char", []).unwrap();
    conn.execute(
        &format!("CREATE TABLE test_char (code CHAR({}))", length),
        [],
    )
    .unwrap();
}

fn setup_sqlite_varchar_table(conn: &Connection, length: u32) {
    conn.execute("DROP TABLE IF EXISTS test_varchar", [])
        .unwrap();
    conn.execute(
        &format!("CREATE TABLE test_varchar (name VARCHAR({}))", length),
        [],
    )
    .unwrap();
}

fn bench_char_constraint_check(c: &mut Criterion) {
    let mut group = c.benchmark_group("char_constraint_check");

    let lengths = [5, 10, 50, 100, 255];

    for &length in &lengths {
        let test_string = "A".repeat(length as usize);
        let short_string = "A".repeat((length / 2) as usize);

        group.bench_with_input(
            BenchmarkId::new("turdb_exact_length", length),
            &(length, &test_string),
            |b, (len, s)| {
                let schema = setup_turdb_char_schema(*len);
                let mut builder = RecordBuilder::new(&schema);
                b.iter(|| {
                    builder.reset();
                    builder.set_char(0, black_box(s)).unwrap();
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("turdb_short_with_padding", length),
            &(length, &short_string),
            |b, (len, s)| {
                let schema = setup_turdb_char_schema(*len);
                let mut builder = RecordBuilder::new(&schema);
                b.iter(|| {
                    builder.reset();
                    builder.set_char(0, black_box(s)).unwrap();
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("sqlite_exact_length", length),
            &(length, &test_string),
            |b, (len, s)| {
                let conn = Connection::open_in_memory().unwrap();
                setup_sqlite_char_table(&conn, *len);
                let mut stmt = conn.prepare("INSERT INTO test_char VALUES (?)").unwrap();
                b.iter(|| {
                    black_box(stmt.execute([black_box(*s)]).unwrap());
                    conn.execute("DELETE FROM test_char", []).unwrap();
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("sqlite_short_with_padding", length),
            &(length, &short_string),
            |b, (len, s)| {
                let conn = Connection::open_in_memory().unwrap();
                setup_sqlite_char_table(&conn, *len);
                let mut stmt = conn.prepare("INSERT INTO test_char VALUES (?)").unwrap();
                b.iter(|| {
                    black_box(stmt.execute([black_box(*s)]).unwrap());
                    conn.execute("DELETE FROM test_char", []).unwrap();
                });
            },
        );
    }

    group.finish();
}

fn bench_varchar_constraint_check(c: &mut Criterion) {
    let mut group = c.benchmark_group("varchar_constraint_check");

    let lengths = [10, 50, 100, 255, 1000];

    for &length in &lengths {
        let test_string = "B".repeat(length as usize);
        let short_string = "B".repeat((length / 2) as usize);

        group.bench_with_input(
            BenchmarkId::new("turdb_exact_length", length),
            &(length, &test_string),
            |b, (len, s)| {
                let schema = setup_turdb_varchar_schema(*len);
                let mut builder = RecordBuilder::new(&schema);
                b.iter(|| {
                    builder.reset();
                    builder.set_varchar(0, black_box(s)).unwrap();
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("turdb_short_no_padding", length),
            &(length, &short_string),
            |b, (len, s)| {
                let schema = setup_turdb_varchar_schema(*len);
                let mut builder = RecordBuilder::new(&schema);
                b.iter(|| {
                    builder.reset();
                    builder.set_varchar(0, black_box(s)).unwrap();
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("sqlite_exact_length", length),
            &(length, &test_string),
            |b, (len, s)| {
                let conn = Connection::open_in_memory().unwrap();
                setup_sqlite_varchar_table(&conn, *len);
                let mut stmt = conn.prepare("INSERT INTO test_varchar VALUES (?)").unwrap();
                b.iter(|| {
                    black_box(stmt.execute([black_box(*s)]).unwrap());
                    conn.execute("DELETE FROM test_varchar", []).unwrap();
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("sqlite_short_no_padding", length),
            &(length, &short_string),
            |b, (len, s)| {
                let conn = Connection::open_in_memory().unwrap();
                setup_sqlite_varchar_table(&conn, *len);
                let mut stmt = conn.prepare("INSERT INTO test_varchar VALUES (?)").unwrap();
                b.iter(|| {
                    black_box(stmt.execute([black_box(*s)]).unwrap());
                    conn.execute("DELETE FROM test_varchar", []).unwrap();
                });
            },
        );
    }

    group.finish();
}

fn bench_unicode_char_counting(c: &mut Criterion) {
    let mut group = c.benchmark_group("unicode_char_counting");

    let ascii_string = "Hello World!!!!!";
    let unicode_string = "日本語テスト🎉🚀";
    let mixed_string = "Hello日本🎉";

    group.bench_function("turdb_ascii_16chars", |b| {
        let schema = setup_turdb_char_schema(20);
        let mut builder = RecordBuilder::new(&schema);
        b.iter(|| {
            builder.reset();
            builder.set_char(0, black_box(ascii_string)).unwrap();
        });
    });

    group.bench_function("turdb_unicode_8chars", |b| {
        let schema = setup_turdb_char_schema(20);
        let mut builder = RecordBuilder::new(&schema);
        b.iter(|| {
            builder.reset();
            builder.set_char(0, black_box(unicode_string)).unwrap();
        });
    });

    group.bench_function("turdb_mixed_10chars", |b| {
        let schema = setup_turdb_char_schema(20);
        let mut builder = RecordBuilder::new(&schema);
        b.iter(|| {
            builder.reset();
            builder.set_char(0, black_box(mixed_string)).unwrap();
        });
    });

    group.bench_function("sqlite_ascii_16chars", |b| {
        let conn = Connection::open_in_memory().unwrap();
        setup_sqlite_char_table(&conn, 20);
        let mut stmt = conn.prepare("INSERT INTO test_char VALUES (?)").unwrap();
        b.iter(|| {
            black_box(stmt.execute([black_box(ascii_string)]).unwrap());
            conn.execute("DELETE FROM test_char", []).unwrap();
        });
    });

    group.bench_function("sqlite_unicode_8chars", |b| {
        let conn = Connection::open_in_memory().unwrap();
        setup_sqlite_char_table(&conn, 20);
        let mut stmt = conn.prepare("INSERT INTO test_char VALUES (?)").unwrap();
        b.iter(|| {
            black_box(stmt.execute([black_box(unicode_string)]).unwrap());
            conn.execute("DELETE FROM test_char", []).unwrap();
        });
    });

    group.bench_function("sqlite_mixed_10chars", |b| {
        let conn = Connection::open_in_memory().unwrap();
        setup_sqlite_char_table(&conn, 20);
        let mut stmt = conn.prepare("INSERT INTO test_char VALUES (?)").unwrap();
        b.iter(|| {
            black_box(stmt.execute([black_box(mixed_string)]).unwrap());
            conn.execute("DELETE FROM test_char", []).unwrap();
        });
    });

    group.finish();
}

fn bench_roundtrip(c: &mut Criterion) {
    let mut group = c.benchmark_group("string_roundtrip");

    let test_string = "Hello, World!";

    group.bench_function("turdb_char_write_read", |b| {
        let schema = setup_turdb_char_schema(20);
        let mut builder = RecordBuilder::new(&schema);
        b.iter(|| {
            builder.reset();
            builder.set_char(0, black_box(test_string)).unwrap();
            let data = builder.build().unwrap();
            let view = RecordView::new(black_box(&data), &schema).unwrap();
            black_box(view.get_char(0).unwrap());
        });
    });

    group.bench_function("turdb_varchar_write_read", |b| {
        let schema = setup_turdb_varchar_schema(20);
        let mut builder = RecordBuilder::new(&schema);
        b.iter(|| {
            builder.reset();
            builder.set_varchar(0, black_box(test_string)).unwrap();
            let data = builder.build().unwrap();
            let view = RecordView::new(black_box(&data), &schema).unwrap();
            black_box(view.get_varchar(0).unwrap());
        });
    });

    group.bench_function("sqlite_char_write_read", |b| {
        let conn = Connection::open_in_memory().unwrap();
        setup_sqlite_char_table(&conn, 20);
        b.iter(|| {
            conn.execute("INSERT INTO test_char VALUES (?)", [black_box(test_string)])
                .unwrap();
            let result: String = conn
                .query_row("SELECT code FROM test_char LIMIT 1", [], |row| row.get(0))
                .unwrap();
            conn.execute("DELETE FROM test_char", []).unwrap();
            black_box(result);
        });
    });

    group.bench_function("sqlite_varchar_write_read", |b| {
        let conn = Connection::open_in_memory().unwrap();
        setup_sqlite_varchar_table(&conn, 20);
        b.iter(|| {
            conn.execute(
                "INSERT INTO test_varchar VALUES (?)",
                [black_box(test_string)],
            )
            .unwrap();
            let result: String = conn
                .query_row("SELECT name FROM test_varchar LIMIT 1", [], |row| row.get(0))
                .unwrap();
            conn.execute("DELETE FROM test_varchar", []).unwrap();
            black_box(result);
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_char_constraint_check,
    bench_varchar_constraint_check,
    bench_unicode_char_counting,
    bench_roundtrip,
);
criterion_main!(benches);
