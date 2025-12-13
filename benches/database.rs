//! Database Integration Benchmarks for TurDB
//!
//! These benchmarks measure end-to-end database performance through the public API.
//! They test the full SQL execution pipeline: Parse → Plan → Execute.
//!
//! ## Performance Targets (from CLAUDE.md)
//!
//! - Point read: < 1µs (cached), < 50µs (disk)
//! - Sequential scan: > 1M rows/sec
//! - Insert: > 100K rows/sec
//!
//! ## Benchmark Groups
//!
//! 1. **Insert Throughput**: Measures INSERT statement performance
//! 2. **Point Read Latency**: Measures SELECT with WHERE clause
//! 3. **Sequential Scan**: Measures full table scan throughput
//! 4. **Mixed Workload**: Simulates realistic read/write patterns
//!
//! ## Running Benchmarks
//!
//! ```bash
//! cargo bench --bench database
//! cargo bench --bench database -- --save-baseline main
//! cargo bench --bench database -- --baseline main
//! ```

use criterion::{
    black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput,
};
use tempfile::tempdir;
use turdb::Database;

fn create_test_database(row_count: usize) -> (tempfile::TempDir, Database) {
    let dir = tempdir().unwrap();
    let db_path = dir.path().join("bench_db");
    let db = Database::create(&db_path).unwrap();

    db.execute("CREATE TABLE users (id INT, name TEXT, age INT, score FLOAT)")
        .unwrap();

    for i in 0..row_count {
        let sql = format!(
            "INSERT INTO users VALUES ({}, 'user{}', {}, {})",
            i,
            i,
            20 + (i % 60),
            (i as f64) * 0.1
        );
        db.execute(&sql).unwrap();
    }

    (dir, db)
}

fn bench_insert_throughput(c: &mut Criterion) {
    let mut group = c.benchmark_group("database_insert");

    for count in [100, 1000, 10000].iter() {
        group.throughput(Throughput::Elements(*count as u64));
        group.bench_with_input(
            BenchmarkId::new("sequential", count),
            count,
            |b, &count| {
                b.iter_with_setup(
                    || {
                        let dir = tempdir().unwrap();
                        let db_path = dir.path().join("bench_db");
                        let db = Database::create(&db_path).unwrap();
                        db.execute("CREATE TABLE users (id INT, name TEXT, age INT)")
                            .unwrap();
                        (dir, db)
                    },
                    |(_dir, db)| {
                        for i in 0..count {
                            let sql = format!(
                                "INSERT INTO users VALUES ({}, 'user{}', {})",
                                i,
                                i,
                                20 + (i % 60)
                            );
                            db.execute(&sql).unwrap();
                        }
                        db
                    },
                );
            },
        );
    }

    group.finish();
}

fn bench_insert_batch(c: &mut Criterion) {
    let mut group = c.benchmark_group("database_insert_batch");

    for count in [100, 1000].iter() {
        group.throughput(Throughput::Elements(*count as u64));
        group.bench_with_input(
            BenchmarkId::new("batch_insert", count),
            count,
            |b, &count| {
                let statements: Vec<String> = (0..count)
                    .map(|i| {
                        format!(
                            "INSERT INTO users VALUES ({}, 'user{}', {})",
                            i,
                            i,
                            20 + (i % 60)
                        )
                    })
                    .collect();

                b.iter_with_setup(
                    || {
                        let dir = tempdir().unwrap();
                        let db_path = dir.path().join("bench_db");
                        let db = Database::create(&db_path).unwrap();
                        db.execute("CREATE TABLE users (id INT, name TEXT, age INT)")
                            .unwrap();
                        (dir, db)
                    },
                    |(_dir, db)| {
                        for sql in &statements {
                            db.execute(sql).unwrap();
                        }
                        db
                    },
                );
            },
        );
    }

    group.finish();
}

fn bench_select_scan(c: &mut Criterion) {
    let mut group = c.benchmark_group("database_select_scan");

    for count in [100, 1000, 10000].iter() {
        group.throughput(Throughput::Elements(*count as u64));

        let (_dir, db) = create_test_database(*count);

        group.bench_with_input(
            BenchmarkId::new("full_scan", count),
            count,
            |b, _count| {
                b.iter(|| {
                    let rows = db.query(black_box("SELECT * FROM users")).unwrap();
                    black_box(rows.len())
                });
            },
        );
    }

    group.finish();
}

fn bench_select_projection(c: &mut Criterion) {
    let mut group = c.benchmark_group("database_select_projection");

    for count in [100, 1000, 10000].iter() {
        group.throughput(Throughput::Elements(*count as u64));

        let (_dir, db) = create_test_database(*count);

        group.bench_with_input(
            BenchmarkId::new("single_column", count),
            count,
            |b, _count| {
                b.iter(|| {
                    let rows = db.query(black_box("SELECT id FROM users")).unwrap();
                    black_box(rows.len())
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("two_columns", count),
            count,
            |b, _count| {
                b.iter(|| {
                    let rows = db.query(black_box("SELECT id, name FROM users")).unwrap();
                    black_box(rows.len())
                });
            },
        );
    }

    group.finish();
}

fn bench_database_create(c: &mut Criterion) {
    let mut group = c.benchmark_group("database_lifecycle");

    group.bench_function("create_empty", |b| {
        b.iter_with_setup(
            || {
                let dir = tempdir().unwrap();
                let db_path = dir.path().join("bench_db");
                (dir, db_path)
            },
            |(dir, db_path)| {
                let db = Database::create(&db_path).unwrap();
                (dir, db)
            },
        );
    });

    group.bench_function("create_with_table", |b| {
        b.iter_with_setup(
            || {
                let dir = tempdir().unwrap();
                let db_path = dir.path().join("bench_db");
                (dir, db_path)
            },
            |(dir, db_path)| {
                let db = Database::create(&db_path).unwrap();
                db.execute("CREATE TABLE test (id INT, value TEXT)")
                    .unwrap();
                (dir, db)
            },
        );
    });

    group.bench_function("open_existing", |b| {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("bench_db");
        {
            let db = Database::create(&db_path).unwrap();
            db.execute("CREATE TABLE test (id INT, value TEXT)")
                .unwrap();
            for i in 0..100 {
                db.execute(&format!("INSERT INTO test VALUES ({}, 'val{}')", i, i))
                    .unwrap();
            }
        }

        b.iter(|| {
            let db = Database::open(black_box(&db_path)).unwrap();
            black_box(db)
        });
    });

    group.finish();
}

fn bench_sql_parsing(c: &mut Criterion) {
    let mut group = c.benchmark_group("sql_parsing");

    group.bench_function("simple_select", |b| {
        let (_dir, db) = create_test_database(10);
        b.iter(|| {
            let rows = db.query(black_box("SELECT * FROM users")).unwrap();
            black_box(rows.len())
        });
    });

    group.bench_function("simple_insert", |b| {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("bench_db");
        let db = Database::create(&db_path).unwrap();
        db.execute("CREATE TABLE test (id INT)").unwrap();

        let mut counter = 0;
        b.iter(|| {
            let sql = format!("INSERT INTO test VALUES ({})", counter);
            db.execute(black_box(&sql)).unwrap();
            counter += 1;
        });
    });

    group.finish();
}

fn bench_mixed_workload(c: &mut Criterion) {
    let mut group = c.benchmark_group("database_mixed_workload");

    group.bench_function("read_heavy_90_10", |b| {
        let (_dir, db) = create_test_database(1000);
        let mut op_counter = 0;

        b.iter(|| {
            if op_counter % 10 == 0 {
                let sql = format!(
                    "INSERT INTO users VALUES ({}, 'newuser{}', {}, {})",
                    10000 + op_counter,
                    op_counter,
                    25,
                    1.5
                );
                db.execute(&sql).unwrap();
            } else {
                let rows = db.query("SELECT * FROM users").unwrap();
                black_box(rows.len());
            }
            op_counter += 1;
        });
    });

    group.bench_function("write_heavy_50_50", |b| {
        let (_dir, db) = create_test_database(100);
        let mut op_counter = 0;

        b.iter(|| {
            if op_counter % 2 == 0 {
                let sql = format!(
                    "INSERT INTO users VALUES ({}, 'newuser{}', {}, {})",
                    10000 + op_counter,
                    op_counter,
                    25,
                    1.5
                );
                db.execute(&sql).unwrap();
            } else {
                let rows = db.query("SELECT * FROM users").unwrap();
                black_box(rows.len());
            }
            op_counter += 1;
        });
    });

    group.finish();
}

fn bench_large_rows(c: &mut Criterion) {
    let mut group = c.benchmark_group("database_large_rows");

    group.bench_function("insert_large_text", |b| {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("bench_db");
        let db = Database::create(&db_path).unwrap();
        db.execute("CREATE TABLE docs (id INT, content TEXT)")
            .unwrap();

        let large_text = "x".repeat(1000);
        let mut counter = 0;

        b.iter(|| {
            let sql = format!(
                "INSERT INTO docs VALUES ({}, '{}')",
                counter, large_text
            );
            db.execute(&sql).unwrap();
            counter += 1;
        });
    });

    group.bench_function("select_large_text", |b| {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("bench_db");
        let db = Database::create(&db_path).unwrap();
        db.execute("CREATE TABLE docs (id INT, content TEXT)")
            .unwrap();

        let large_text = "x".repeat(1000);
        for i in 0..100 {
            let sql = format!("INSERT INTO docs VALUES ({}, '{}')", i, large_text);
            db.execute(&sql).unwrap();
        }

        b.iter(|| {
            let rows = db.query("SELECT * FROM docs").unwrap();
            black_box(rows.len())
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_insert_throughput,
    bench_insert_batch,
    bench_select_scan,
    bench_select_projection,
    bench_database_create,
    bench_sql_parsing,
    bench_mixed_workload,
    bench_large_rows,
);

criterion_main!(benches);
