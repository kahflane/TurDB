//! Database Integration Benchmarks for TurDB vs SQLite
//!
//! These benchmarks compare TurDB against SQLite with optimized settings:
//! - journal_mode=WAL
//! - synchronous=OFF
//! - mmap_size=268435456 (256MB)
//!
//! ## Performance Targets (from CLAUDE.md)
//!
//! - Point read: < 1µs (cached), < 50µs (disk)
//! - Sequential scan: > 1M rows/sec
//! - Insert: > 100K rows/sec
//!
//! ## Running Benchmarks
//!
//! ```bash
//! cargo bench --bench database
//! cargo bench --bench database -- "insert"
//! cargo bench --bench database -- "scan"
//! ```

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use rusqlite::{params, Connection};
use tempfile::tempdir;
use turdb::Database;

fn create_turdb_test_database(row_count: usize) -> (tempfile::TempDir, Database) {
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

fn create_sqlite_connection(dir: &tempfile::TempDir) -> Connection {
    let db_path = dir.path().join("sqlite_bench.db");
    let conn = Connection::open(&db_path).unwrap();

    conn.execute_batch(
        "PRAGMA journal_mode=WAL;
         PRAGMA synchronous=OFF;
         PRAGMA mmap_size=268435456;
         PRAGMA cache_size=-65536;
         PRAGMA temp_store=MEMORY;",
    )
    .unwrap();

    conn
}

fn create_sqlite_test_database(row_count: usize) -> (tempfile::TempDir, Connection) {
    let dir = tempdir().unwrap();
    let conn = create_sqlite_connection(&dir);

    conn.execute(
        "CREATE TABLE users (id INTEGER, name TEXT, age INTEGER, score REAL)",
        [],
    )
    .unwrap();

    for i in 0..row_count {
        conn.execute(
            "INSERT INTO users VALUES (?1, ?2, ?3, ?4)",
            params![
                i as i64,
                format!("user{}", i),
                20 + (i % 60) as i64,
                (i as f64) * 0.1
            ],
        )
        .unwrap();
    }

    (dir, conn)
}

fn bench_insert_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("insert_comparison");

    for count in [100, 1000, 10000].iter() {
        group.throughput(Throughput::Elements(*count as u64));

        group.bench_with_input(BenchmarkId::new("turdb", count), count, |b, &count| {
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
        });

        group.bench_with_input(BenchmarkId::new("sqlite", count), count, |b, &count| {
            b.iter_with_setup(
                || {
                    let dir = tempdir().unwrap();
                    let conn = create_sqlite_connection(&dir);
                    conn.execute(
                        "CREATE TABLE users (id INTEGER, name TEXT, age INTEGER)",
                        [],
                    )
                    .unwrap();
                    (dir, conn)
                },
                |(_dir, conn)| {
                    for i in 0..count {
                        conn.execute(
                            "INSERT INTO users VALUES (?1, ?2, ?3)",
                            params![i as i64, format!("user{}", i), 20 + (i % 60) as i64],
                        )
                        .unwrap();
                    }
                    conn
                },
            );
        });
    }

    group.finish();
}

fn bench_insert_prepared_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("insert_prepared_comparison");

    for count in [1000, 10000].iter() {
        group.throughput(Throughput::Elements(*count as u64));

        group.bench_with_input(BenchmarkId::new("turdb", count), count, |b, &count| {
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
        });

        group.bench_with_input(
            BenchmarkId::new("sqlite_prepared", count),
            count,
            |b, &count| {
                b.iter_with_setup(
                    || {
                        let dir = tempdir().unwrap();
                        let conn = create_sqlite_connection(&dir);
                        conn.execute(
                            "CREATE TABLE users (id INTEGER, name TEXT, age INTEGER)",
                            [],
                        )
                        .unwrap();
                        (dir, conn)
                    },
                    |(_dir, conn)| {
                        {
                            let mut stmt = conn
                                .prepare_cached("INSERT INTO users VALUES (?1, ?2, ?3)")
                                .unwrap();
                            for i in 0..count {
                                stmt.execute(params![
                                    i as i64,
                                    format!("user{}", i),
                                    20 + (i % 60) as i64
                                ])
                                .unwrap();
                            }
                        }
                        conn
                    },
                );
            },
        );
    }

    group.finish();
}

fn bench_scan_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("scan_comparison");

    for count in [100, 1000, 10000].iter() {
        group.throughput(Throughput::Elements(*count as u64));

        let (_turdb_dir, turdb) = create_turdb_test_database(*count);
        let (_sqlite_dir, sqlite_conn) = create_sqlite_test_database(*count);

        group.bench_with_input(BenchmarkId::new("turdb", count), count, |b, _count| {
            b.iter(|| {
                let rows = turdb.query(black_box("SELECT * FROM users")).unwrap();
                black_box(rows.len())
            });
        });

        group.bench_with_input(BenchmarkId::new("sqlite", count), count, |b, _count| {
            b.iter(|| {
                let mut stmt = sqlite_conn.prepare_cached("SELECT * FROM users").unwrap();
                let rows: Vec<_> = stmt
                    .query_map([], |row| {
                        Ok((
                            row.get::<_, i64>(0).unwrap(),
                            row.get::<_, String>(1).unwrap(),
                            row.get::<_, i64>(2).unwrap(),
                            row.get::<_, f64>(3).unwrap(),
                        ))
                    })
                    .unwrap()
                    .collect();
                black_box(rows.len())
            });
        });
    }

    group.finish();
}

fn bench_projection_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("projection_comparison");

    for count in [1000, 10000].iter() {
        group.throughput(Throughput::Elements(*count as u64));

        let (_turdb_dir, turdb) = create_turdb_test_database(*count);
        let (_sqlite_dir, sqlite_conn) = create_sqlite_test_database(*count);

        group.bench_with_input(
            BenchmarkId::new("turdb_single_col", count),
            count,
            |b, _count| {
                b.iter(|| {
                    let rows = turdb.query(black_box("SELECT id FROM users")).unwrap();
                    black_box(rows.len())
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("sqlite_single_col", count),
            count,
            |b, _count| {
                b.iter(|| {
                    let mut stmt = sqlite_conn.prepare_cached("SELECT id FROM users").unwrap();
                    let rows: Vec<_> = stmt
                        .query_map([], |row| Ok(row.get::<_, i64>(0).unwrap()))
                        .unwrap()
                        .collect();
                    black_box(rows.len())
                });
            },
        );
    }

    group.finish();
}

fn bench_mixed_workload_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("mixed_workload_comparison");

    let (_turdb_dir, turdb) = create_turdb_test_database(1000);
    let (_sqlite_dir, sqlite_conn) = create_sqlite_test_database(1000);

    group.bench_function("turdb_read_heavy_90_10", |b| {
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
                turdb.execute(&sql).unwrap();
            } else {
                let rows = turdb.query("SELECT * FROM users").unwrap();
                black_box(rows.len());
            }
            op_counter += 1;
        });
    });

    group.bench_function("sqlite_read_heavy_90_10", |b| {
        let mut op_counter = 0;
        b.iter(|| {
            if op_counter % 10 == 0 {
                sqlite_conn
                    .execute(
                        "INSERT INTO users VALUES (?1, ?2, ?3, ?4)",
                        params![
                            10000 + op_counter,
                            format!("newuser{}", op_counter),
                            25i64,
                            1.5f64
                        ],
                    )
                    .unwrap();
            } else {
                let mut stmt = sqlite_conn.prepare_cached("SELECT * FROM users").unwrap();
                let rows: Vec<_> = stmt
                    .query_map([], |row| {
                        Ok((
                            row.get::<_, i64>(0).unwrap(),
                            row.get::<_, String>(1).unwrap(),
                            row.get::<_, i64>(2).unwrap(),
                            row.get::<_, f64>(3).unwrap(),
                        ))
                    })
                    .unwrap()
                    .collect();
                black_box(rows.len());
            }
            op_counter += 1;
        });
    });

    group.finish();
}

fn bench_large_text_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("large_text_comparison");

    let large_text = "x".repeat(1000);

    group.bench_function("turdb_insert_1kb", |b| {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("bench_db");
        let db = Database::create(&db_path).unwrap();
        db.execute("CREATE TABLE docs (id INT, content TEXT)")
            .unwrap();

        let mut counter = 0;
        b.iter(|| {
            let sql = format!("INSERT INTO docs VALUES ({}, '{}')", counter, large_text);
            db.execute(&sql).unwrap();
            counter += 1;
        });
    });

    group.bench_function("sqlite_insert_1kb", |b| {
        let dir = tempdir().unwrap();
        let conn = create_sqlite_connection(&dir);
        conn.execute("CREATE TABLE docs (id INTEGER, content TEXT)", [])
            .unwrap();

        let mut counter = 0;
        b.iter(|| {
            conn.execute(
                "INSERT INTO docs VALUES (?1, ?2)",
                params![counter as i64, &large_text],
            )
            .unwrap();
            counter += 1;
        });
    });

    group.bench_function("turdb_scan_1kb_rows", |b| {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("bench_db");
        let db = Database::create(&db_path).unwrap();
        db.execute("CREATE TABLE docs (id INT, content TEXT)")
            .unwrap();
        for i in 0..100 {
            let sql = format!("INSERT INTO docs VALUES ({}, '{}')", i, large_text);
            db.execute(&sql).unwrap();
        }

        b.iter(|| {
            let rows = db.query("SELECT * FROM docs").unwrap();
            black_box(rows.len())
        });
    });

    group.bench_function("sqlite_scan_1kb_rows", |b| {
        let dir = tempdir().unwrap();
        let conn = create_sqlite_connection(&dir);
        conn.execute("CREATE TABLE docs (id INTEGER, content TEXT)", [])
            .unwrap();
        for i in 0..100 {
            conn.execute(
                "INSERT INTO docs VALUES (?1, ?2)",
                params![i as i64, &large_text],
            )
            .unwrap();
        }

        b.iter(|| {
            let mut stmt = conn.prepare_cached("SELECT * FROM docs").unwrap();
            let rows: Vec<_> = stmt
                .query_map([], |row| {
                    Ok((
                        row.get::<_, i64>(0).unwrap(),
                        row.get::<_, String>(1).unwrap(),
                    ))
                })
                .unwrap()
                .collect();
            black_box(rows.len())
        });
    });

    group.finish();
}

fn bench_lifecycle_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("lifecycle_comparison");

    group.bench_function("turdb_create_empty", |b| {
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

    group.bench_function("sqlite_create_empty", |b| {
        b.iter_with_setup(
            || {
                let dir = tempdir().unwrap();
                let db_path = dir.path().join("sqlite.db");
                (dir, db_path)
            },
            |(dir, db_path)| {
                let conn = Connection::open(&db_path).unwrap();
                conn.execute_batch(
                    "PRAGMA journal_mode=WAL;
                     PRAGMA synchronous=OFF;
                     PRAGMA mmap_size=268435456;",
                )
                .unwrap();
                (dir, conn)
            },
        );
    });

    group.bench_function("turdb_open_existing", |b| {
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

    group.bench_function("sqlite_open_existing", |b| {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("sqlite.db");
        {
            let conn = Connection::open(&db_path).unwrap();
            conn.execute_batch(
                "PRAGMA journal_mode=WAL;
                 PRAGMA synchronous=OFF;
                 PRAGMA mmap_size=268435456;",
            )
            .unwrap();
            conn.execute("CREATE TABLE test (id INTEGER, value TEXT)", [])
                .unwrap();
            for i in 0..100 {
                conn.execute(
                    "INSERT INTO test VALUES (?1, ?2)",
                    params![i as i64, format!("val{}", i)],
                )
                .unwrap();
            }
        }

        b.iter(|| {
            let conn = Connection::open(black_box(&db_path)).unwrap();
            conn.execute_batch(
                "PRAGMA journal_mode=WAL;
                 PRAGMA synchronous=OFF;
                 PRAGMA mmap_size=268435456;",
            )
            .unwrap();
            black_box(conn)
        });
    });

    group.finish();
}

fn bench_insert_wal_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("insert_wal_comparison");

    for count in [100, 1000, 10000].iter() {
        group.throughput(Throughput::Elements(*count as u64));

        group.bench_with_input(BenchmarkId::new("turdb_wal", count), count, |b, &count| {
            b.iter_with_setup(
                || {
                    let dir = tempdir().unwrap();
                    let db_path = dir.path().join("bench_db");
                    let db = Database::create(&db_path).unwrap();
                    db.execute("PRAGMA WAL=ON").unwrap();
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
        });

        group.bench_with_input(BenchmarkId::new("sqlite_wal", count), count, |b, &count| {
            b.iter_with_setup(
                || {
                    let dir = tempdir().unwrap();
                    let db_path = dir.path().join("sqlite_bench.db");
                    let conn = Connection::open(&db_path).unwrap();
                    conn.execute_batch(
                        "PRAGMA journal_mode=WAL;
                             PRAGMA synchronous=NORMAL;
                             PRAGMA mmap_size=268435456;
                             PRAGMA cache_size=-65536;",
                    )
                    .unwrap();
                    conn.execute(
                        "CREATE TABLE users (id INTEGER, name TEXT, age INTEGER)",
                        [],
                    )
                    .unwrap();
                    (dir, conn)
                },
                |(_dir, conn)| {
                    for i in 0..count {
                        conn.execute(
                            "INSERT INTO users VALUES (?1, ?2, ?3)",
                            params![i as i64, format!("user{}", i), 20 + (i % 60) as i64],
                        )
                        .unwrap();
                    }
                    conn
                },
            );
        });
    }

    group.finish();
}

fn bench_unique_constraint_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("unique_constraint_comparison");

    for count in [100, 1000].iter() {
        group.throughput(Throughput::Elements(*count as u64));

        group.bench_with_input(
            BenchmarkId::new("turdb_unique", count),
            count,
            |b, &count| {
                b.iter_with_setup(
                    || {
                        let dir = tempdir().unwrap();
                        let db_path = dir.path().join("bench_db");
                        let db = Database::create(&db_path).unwrap();
                        db.execute(
                            "CREATE TABLE users (id INT PRIMARY KEY, name TEXT, email TEXT UNIQUE)",
                        )
                        .unwrap();
                        (dir, db)
                    },
                    |(_dir, db)| {
                        for i in 0..count {
                            let sql = format!(
                                "INSERT INTO users VALUES ({}, 'user{}', 'user{}@example.com')",
                                i, i, i
                            );
                            db.execute(&sql).unwrap();
                        }
                        db
                    },
                );
            },
        );

        group.bench_with_input(
            BenchmarkId::new("sqlite_unique", count),
            count,
            |b, &count| {
                b.iter_with_setup(
                    || {
                        let dir = tempdir().unwrap();
                        let conn = create_sqlite_connection(&dir);
                        conn.execute(
                            "CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT, email TEXT UNIQUE)",
                            [],
                        )
                            .unwrap();
                        (dir, conn)
                    },
                    |(_dir, conn)| {
                        for i in 0..count {
                            conn.execute(
                                "INSERT INTO users VALUES (?1, ?2, ?3)",
                                params![
                                    i as i64,
                                    format!("user{}", i),
                                    format!("user{}@example.com", i)
                                ],
                            )
                                .unwrap();
                        }
                        conn
                    },
                );
            },
        );

        group.bench_with_input(
            BenchmarkId::new("turdb_no_constraint", count),
            count,
            |b, &count| {
                b.iter_with_setup(
                    || {
                        let dir = tempdir().unwrap();
                        let db_path = dir.path().join("bench_db");
                        let db = Database::create(&db_path).unwrap();
                        db.execute("CREATE TABLE users (id INT, name TEXT, email TEXT)")
                            .unwrap();
                        (dir, db)
                    },
                    |(_dir, db)| {
                        for i in 0..count {
                            let sql = format!(
                                "INSERT INTO users VALUES ({}, 'user{}', 'user{}@example.com')",
                                i, i, i
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

fn bench_check_constraint_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("check_constraint_comparison");

    for count in [100, 1000].iter() {
        group.throughput(Throughput::Elements(*count as u64));

        group.bench_with_input(
            BenchmarkId::new("turdb_check", count),
            count,
            |b, &count| {
                b.iter_with_setup(
                    || {
                        let dir = tempdir().unwrap();
                        let db_path = dir.path().join("bench_db");
                        let db = Database::create(&db_path).unwrap();
                        db.execute(
                            "CREATE TABLE products (id INT, price INT CHECK (price >= 0), qty INT CHECK (qty >= 0))",
                        )
                            .unwrap();
                        (dir, db)
                    },
                    |(_dir, db)| {
                        for i in 0..count {
                            let sql = format!("INSERT INTO products VALUES ({}, {}, {})", i, i * 10, i);
                            db.execute(&sql).unwrap();
                        }
                        db
                    },
                );
            },
        );

        group.bench_with_input(
            BenchmarkId::new("sqlite_check", count),
            count,
            |b, &count| {
                b.iter_with_setup(
                    || {
                        let dir = tempdir().unwrap();
                        let conn = create_sqlite_connection(&dir);
                        conn.execute(
                            "CREATE TABLE products (id INTEGER, price INTEGER CHECK (price >= 0), qty INTEGER CHECK (qty >= 0))",
                            [],
                        )
                            .unwrap();
                        (dir, conn)
                    },
                    |(_dir, conn)| {
                        for i in 0..count {
                            conn.execute(
                                "INSERT INTO products VALUES (?1, ?2, ?3)",
                                params![i as i64, (i * 10) as i64, i as i64],
                            )
                                .unwrap();
                        }
                        conn
                    },
                );
            },
        );
    }

    group.finish();
}

fn bench_foreign_key_insert_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("foreign_key_insert_comparison");

    for count in [100, 500].iter() {
        group.throughput(Throughput::Elements(*count as u64));

        group.bench_with_input(
            BenchmarkId::new("turdb_fk", count),
            count,
            |b, &count| {
                b.iter_with_setup(
                    || {
                        let dir = tempdir().unwrap();
                        let db_path = dir.path().join("bench_db");
                        let db = Database::create(&db_path).unwrap();
                        db.execute("CREATE TABLE categories (id INT PRIMARY KEY, name TEXT)")
                            .unwrap();
                        db.execute(
                            "CREATE TABLE products (id INT, cat_id INT REFERENCES categories(id), name TEXT)",
                        )
                            .unwrap();
                        for i in 0..10 {
                            db.execute(&format!("INSERT INTO categories VALUES ({}, 'cat{}')", i, i))
                                .unwrap();
                        }
                        (dir, db)
                    },
                    |(_dir, db)| {
                        for i in 0..count {
                            let cat_id = i % 10;
                            let sql = format!(
                                "INSERT INTO products VALUES ({}, {}, 'product{}')",
                                i, cat_id, i
                            );
                            db.execute(&sql).unwrap();
                        }
                        db
                    },
                );
            },
        );

        group.bench_with_input(
            BenchmarkId::new("sqlite_fk", count),
            count,
            |b, &count| {
                b.iter_with_setup(
                    || {
                        let dir = tempdir().unwrap();
                        let conn = create_sqlite_connection(&dir);
                        conn.execute_batch("PRAGMA foreign_keys=ON;").unwrap();
                        conn.execute(
                            "CREATE TABLE categories (id INTEGER PRIMARY KEY, name TEXT)",
                            [],
                        )
                            .unwrap();
                        conn.execute(
                            "CREATE TABLE products (id INTEGER, cat_id INTEGER REFERENCES categories(id), name TEXT)",
                            [],
                        )
                            .unwrap();
                        for i in 0..10 {
                            conn.execute(
                                "INSERT INTO categories VALUES (?1, ?2)",
                                params![i as i64, format!("cat{}", i)],
                            )
                                .unwrap();
                        }
                        (dir, conn)
                    },
                    |(_dir, conn)| {
                        for i in 0..count {
                            let cat_id = i % 10;
                            conn.execute(
                                "INSERT INTO products VALUES (?1, ?2, ?3)",
                                params![i as i64, cat_id as i64, format!("product{}", i)],
                            )
                                .unwrap();
                        }
                        conn
                    },
                );
            },
        );
    }

    group.finish();
}

fn bench_foreign_key_delete_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("foreign_key_delete_comparison");

    group.bench_function("turdb_fk_delete_unreferenced", |b| {
        b.iter_with_setup(
            || {
                let dir = tempdir().unwrap();
                let db_path = dir.path().join("bench_db");
                let db = Database::create(&db_path).unwrap();
                db.execute("CREATE TABLE categories (id INT PRIMARY KEY, name TEXT)")
                    .unwrap();
                db.execute(
                    "CREATE TABLE products (id INT, cat_id INT REFERENCES categories(id), name TEXT)",
                )
                    .unwrap();
                for i in 0..100 {
                    db.execute(&format!("INSERT INTO categories VALUES ({}, 'cat{}')", i, i))
                        .unwrap();
                }
                for i in 0..50 {
                    db.execute(&format!(
                        "INSERT INTO products VALUES ({}, {}, 'product{}')",
                        i, i, i
                    ))
                        .unwrap();
                }
                (dir, db)
            },
            |(_dir, db)| {
                for del_id in 50..100 {
                    db.execute(&format!("DELETE FROM categories WHERE id = {}", del_id))
                        .unwrap();
                }
                db
            },
        );
    });

    group.bench_function("sqlite_fk_delete_unreferenced", |b| {
        b.iter_with_setup(
            || {
                let dir = tempdir().unwrap();
                let conn = create_sqlite_connection(&dir);
                conn.execute_batch("PRAGMA foreign_keys=ON;").unwrap();
                conn.execute(
                    "CREATE TABLE categories (id INTEGER PRIMARY KEY, name TEXT)",
                    [],
                )
                    .unwrap();
                conn.execute(
                    "CREATE TABLE products (id INTEGER, cat_id INTEGER REFERENCES categories(id), name TEXT)",
                    [],
                )
                    .unwrap();
                for i in 0..100 {
                    conn.execute(
                        "INSERT INTO categories VALUES (?1, ?2)",
                        params![i as i64, format!("cat{}", i)],
                    )
                        .unwrap();
                }
                for i in 0..50 {
                    conn.execute(
                        "INSERT INTO products VALUES (?1, ?2, ?3)",
                        params![i as i64, i as i64, format!("product{}", i)],
                    )
                        .unwrap();
                }
                (dir, conn)
            },
            |(_dir, conn)| {
                for del_id in 50i64..100 {
                    conn.execute("DELETE FROM categories WHERE id = ?1", params![del_id])
                        .unwrap();
                }
                conn
            },
        );
    });

    group.finish();
}

fn bench_subquery_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("subquery_comparison");

    for count in [100, 1000, 10000].iter() {
        group.throughput(Throughput::Elements(*count as u64));

        let (_turdb_dir, turdb) = create_turdb_test_database(*count);
        let (_sqlite_dir, sqlite_conn) = create_sqlite_test_database(*count);

        group.bench_with_input(
            BenchmarkId::new("turdb_subquery", count),
            count,
            |b, _count| {
                b.iter(|| {
                    let rows = turdb
                        .query(black_box(
                            "SELECT s.id, s.name FROM (SELECT id, name FROM users) AS s",
                        ))
                        .unwrap();
                    black_box(rows.len())
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("sqlite_subquery", count),
            count,
            |b, _count| {
                b.iter(|| {
                    let mut stmt = sqlite_conn
                        .prepare_cached("SELECT s.id, s.name FROM (SELECT id, name FROM users) AS s")
                        .unwrap();
                    let rows: Vec<_> = stmt
                        .query_map([], |row| {
                            Ok((
                                row.get::<_, i64>(0).unwrap(),
                                row.get::<_, String>(1).unwrap(),
                            ))
                        })
                        .unwrap()
                        .collect();
                    black_box(rows.len())
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("turdb_direct_scan", count),
            count,
            |b, _count| {
                b.iter(|| {
                    let rows = turdb
                        .query(black_box("SELECT id, name FROM users"))
                        .unwrap();
                    black_box(rows.len())
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("sqlite_direct_scan", count),
            count,
            |b, _count| {
                b.iter(|| {
                    let mut stmt = sqlite_conn
                        .prepare_cached("SELECT id, name FROM users")
                        .unwrap();
                    let rows: Vec<_> = stmt
                        .query_map([], |row| {
                            Ok((
                                row.get::<_, i64>(0).unwrap(),
                                row.get::<_, String>(1).unwrap(),
                            ))
                        })
                        .unwrap()
                        .collect();
                    black_box(rows.len())
                });
            },
        );
    }

    group.finish();
}

fn bench_nested_subquery_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("nested_subquery_comparison");

    for count in [100, 1000].iter() {
        group.throughput(Throughput::Elements(*count as u64));

        let (_turdb_dir, _turdb) = create_turdb_test_database(*count);
        let (_sqlite_dir, sqlite_conn) = create_sqlite_test_database(*count);

        group.bench_with_input(
            BenchmarkId::new("sqlite_nested_2_levels", count),
            count,
            |b, _count| {
                b.iter(|| {
                    let mut stmt = sqlite_conn
                        .prepare_cached(
                            "SELECT t.id FROM (SELECT s.id FROM (SELECT id FROM users) AS s) AS t",
                        )
                        .unwrap();
                    let rows: Vec<_> = stmt
                        .query_map([], |row| Ok(row.get::<_, i64>(0).unwrap()))
                        .unwrap()
                        .collect();
                    black_box(rows.len())
                });
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_insert_comparison,
    bench_insert_wal_comparison,
    bench_insert_prepared_comparison,
    bench_scan_comparison,
    bench_projection_comparison,
    bench_mixed_workload_comparison,
    bench_large_text_comparison,
    bench_lifecycle_comparison,
    bench_unique_constraint_comparison,
    bench_check_constraint_comparison,
    bench_foreign_key_insert_comparison,
    bench_foreign_key_delete_comparison,
    bench_subquery_comparison,
    bench_nested_subquery_comparison,
);

criterion_main!(benches);
