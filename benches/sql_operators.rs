//! SQL Operator Evaluation Benchmarks
//!
//! Benchmarks comparing TurDB SQL operator evaluation against SQLite for:
//! - JSON path extraction (#>, #>>)
//! - JSON containment (@>, <@)
//! - Array operations
//!
//! ## Running Benchmarks
//!
//! ```bash
//! cargo bench --bench sql_operators
//! ```

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use rusqlite::{params, Connection};
use std::borrow::Cow;
use tempfile::tempdir;
use turdb::sql::predicate::CompiledPredicate;
use turdb::types::Value;
use turdb::Database;

fn create_turdb_json_database(row_count: usize) -> (tempfile::TempDir, Database) {
    let dir = tempdir().unwrap();
    let db_path = dir.path().join("bench_db");
    let db = Database::create(&db_path).unwrap();

    db.execute("CREATE TABLE docs (id INT, data TEXT)")
        .unwrap();

    for i in 0..row_count {
        let json = format!(
            r#"{{"id": {}, "user": {{"name": "user{}", "age": {}, "profile": {{"score": {}}}}}, "tags": [1, 2, 3, {}]}}"#,
            i,
            i,
            20 + (i % 60),
            (i as f64) * 0.1,
            i % 10
        );
        let sql = format!("INSERT INTO docs VALUES ({}, '{}')", i, json);
        db.execute(&sql).unwrap();
    }

    (dir, db)
}

fn create_sqlite_json_database(row_count: usize) -> (tempfile::TempDir, Connection) {
    let dir = tempdir().unwrap();
    let db_path = dir.path().join("sqlite_bench.db");
    let conn = Connection::open(&db_path).unwrap();

    conn.execute_batch(
        "PRAGMA journal_mode=WAL;
         PRAGMA synchronous=OFF;
         PRAGMA mmap_size=268435456;",
    )
    .unwrap();

    conn.execute("CREATE TABLE docs (id INTEGER, data TEXT)", [])
        .unwrap();

    for i in 0..row_count {
        let json = format!(
            r#"{{"id": {}, "user": {{"name": "user{}", "age": {}, "profile": {{"score": {}}}}}, "tags": [1, 2, 3, {}]}}"#,
            i,
            i,
            20 + (i % 60),
            (i as f64) * 0.1,
            i % 10
        );
        conn.execute("INSERT INTO docs VALUES (?1, ?2)", params![i as i64, json])
            .unwrap();
    }

    (dir, conn)
}

fn bench_json_extract_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("json_extract");

    for count in [100, 1000].iter() {
        group.throughput(Throughput::Elements(*count as u64));

        let (_turdb_dir, turdb) = create_turdb_json_database(*count);
        let (_sqlite_dir, sqlite_conn) = create_sqlite_json_database(*count);

        group.bench_with_input(
            BenchmarkId::new("turdb_arrow", count),
            count,
            |b, _count| {
                b.iter(|| {
                    let rows = turdb
                        .query(black_box("SELECT data->>'id' FROM docs"))
                        .unwrap();
                    black_box(rows.len())
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("sqlite_json_extract", count),
            count,
            |b, _count| {
                b.iter(|| {
                    let mut stmt = sqlite_conn
                        .prepare_cached("SELECT json_extract(data, '$.id') FROM docs")
                        .unwrap();
                    let rows: Vec<_> = stmt
                        .query_map([], |row| Ok(row.get::<_, i64>(0).unwrap_or(0)))
                        .unwrap()
                        .collect();
                    black_box(rows.len())
                });
            },
        );
    }

    group.finish();
}

fn bench_json_nested_extract_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("json_nested_extract");

    for count in [100, 1000].iter() {
        group.throughput(Throughput::Elements(*count as u64));

        let (_turdb_dir, turdb) = create_turdb_json_database(*count);
        let (_sqlite_dir, sqlite_conn) = create_sqlite_json_database(*count);

        group.bench_with_input(
            BenchmarkId::new("turdb_nested", count),
            count,
            |b, _count| {
                b.iter(|| {
                    let rows = turdb
                        .query(black_box(
                            "SELECT data->'user'->'profile'->>'score' FROM docs",
                        ))
                        .unwrap();
                    black_box(rows.len())
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("sqlite_nested", count),
            count,
            |b, _count| {
                b.iter(|| {
                    let mut stmt = sqlite_conn
                        .prepare_cached("SELECT json_extract(data, '$.user.profile.score') FROM docs")
                        .unwrap();
                    let rows: Vec<_> = stmt
                        .query_map([], |row| Ok(row.get::<_, f64>(0).unwrap_or(0.0)))
                        .unwrap()
                        .collect();
                    black_box(rows.len())
                });
            },
        );
    }

    group.finish();
}

fn bench_predicate_json_path_extract(c: &mut Criterion) {
    let mut group = c.benchmark_group("predicate_json_path");
    group.throughput(Throughput::Elements(1000));

    let json_data: Vec<String> = (0..1000)
        .map(|i| {
            format!(
                r#"{{"user": {{"profile": {{"name": "user{}", "score": {}}}}}}}"#,
                i,
                i * 10
            )
        })
        .collect();

    group.bench_function("turdb_path_extract", |b| {
        static DUMMY_EXPR: turdb::sql::ast::Expr =
            turdb::sql::ast::Expr::Literal(turdb::sql::ast::Literal::Null);
        let predicate = CompiledPredicate::new(&DUMMY_EXPR, vec![]);

        b.iter(|| {
            let mut sum = 0i64;
            for json in &json_data {
                let json_val = Value::Jsonb(Cow::Owned(json.as_bytes().to_vec()));
                let path_val = Value::Text(Cow::Borrowed("{user,profile,score}"));
                if let Some(Value::Int(n)) =
                    predicate.eval_json_path_extract(&json_val, &path_val, false)
                {
                    sum += n;
                }
            }
            black_box(sum)
        });
    });

    group.finish();
}

fn bench_predicate_json_contains(c: &mut Criterion) {
    let mut group = c.benchmark_group("predicate_json_contains");
    group.throughput(Throughput::Elements(1000));

    let json_data: Vec<String> = (0..1000)
        .map(|i| {
            format!(
                r#"{{"a": {}, "b": {}, "c": {}, "nested": {{"x": {}}}}}"#,
                i,
                i * 2,
                i * 3,
                i
            )
        })
        .collect();

    group.bench_function("turdb_json_contains", |b| {
        static DUMMY_EXPR: turdb::sql::ast::Expr =
            turdb::sql::ast::Expr::Literal(turdb::sql::ast::Literal::Null);
        let predicate = CompiledPredicate::new(&DUMMY_EXPR, vec![]);

        b.iter(|| {
            let mut count = 0;
            for json in &json_data {
                let json_val = Value::Jsonb(Cow::Owned(json.as_bytes().to_vec()));
                let pattern = Value::Jsonb(Cow::Borrowed(br#"{"a": 0}"#));
                if predicate
                    .eval_json_contains(&json_val, &pattern)
                    .unwrap_or(false)
                {
                    count += 1;
                }
            }
            black_box(count)
        });
    });

    group.finish();
}

fn bench_predicate_array_contains(c: &mut Criterion) {
    let mut group = c.benchmark_group("predicate_array_contains");
    group.throughput(Throughput::Elements(1000));

    let array_data: Vec<String> = (0..1000)
        .map(|i| format!("[{}, {}, {}, {}, {}]", i, i + 1, i + 2, i + 3, i + 4))
        .collect();

    group.bench_function("turdb_array_contains", |b| {
        static DUMMY_EXPR: turdb::sql::ast::Expr =
            turdb::sql::ast::Expr::Literal(turdb::sql::ast::Literal::Null);
        let predicate = CompiledPredicate::new(&DUMMY_EXPR, vec![]);

        b.iter(|| {
            let mut count = 0;
            for arr in &array_data {
                let arr_val = Value::Jsonb(Cow::Owned(arr.as_bytes().to_vec()));
                let pattern = Value::Jsonb(Cow::Borrowed(b"[1, 2]"));
                if predicate
                    .eval_array_contains(&arr_val, &pattern)
                    .unwrap_or(false)
                {
                    count += 1;
                }
            }
            black_box(count)
        });
    });

    group.finish();
}

fn bench_predicate_array_overlaps(c: &mut Criterion) {
    let mut group = c.benchmark_group("predicate_array_overlaps");
    group.throughput(Throughput::Elements(1000));

    let array_data: Vec<String> = (0..1000)
        .map(|i| format!("[{}, {}, {}, {}, {}]", i, i + 1, i + 2, i + 3, i + 4))
        .collect();

    group.bench_function("turdb_array_overlaps", |b| {
        static DUMMY_EXPR: turdb::sql::ast::Expr =
            turdb::sql::ast::Expr::Literal(turdb::sql::ast::Literal::Null);
        let predicate = CompiledPredicate::new(&DUMMY_EXPR, vec![]);

        b.iter(|| {
            let mut count = 0;
            for arr in &array_data {
                let arr_val = Value::Jsonb(Cow::Owned(arr.as_bytes().to_vec()));
                let pattern = Value::Jsonb(Cow::Borrowed(b"[5, 10, 15]"));
                if predicate
                    .eval_array_overlaps(&arr_val, &pattern)
                    .unwrap_or(false)
                {
                    count += 1;
                }
            }
            black_box(count)
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_json_extract_comparison,
    bench_json_nested_extract_comparison,
    bench_predicate_json_path_extract,
    bench_predicate_json_contains,
    bench_predicate_array_contains,
    bench_predicate_array_overlaps,
);

criterion_main!(benches);
