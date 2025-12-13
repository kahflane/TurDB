//! B-tree benchmarks for TurDB
//!
//! These benchmarks measure the core B-tree operations that determine
//! database performance. Target metrics from CLAUDE.md:
//!
//! - Point read: < 1µs (cached), < 50µs (disk)
//! - Sequential scan: > 1M rows/sec
//! - Insert: > 100K rows/sec

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use std::hint::black_box as hint_black_box;
use tempfile::tempdir;
use turdb::btree::{BTree, InteriorNode, InteriorNodeMut, LeafNode, LeafNodeMut};
use turdb::storage::{MmapStorage, PAGE_SIZE};

fn bench_insert(c: &mut Criterion) {
    let mut group = c.benchmark_group("btree_insert");

    for count in [100, 1000].iter() {
        group.throughput(Throughput::Elements(*count as u64));
        group.bench_with_input(BenchmarkId::new("sequential", count), count, |b, &count| {
            b.iter_with_setup(
                || {
                    let dir = tempdir().unwrap();
                    let path = dir.path().join("bench.tbd");
                    let storage = MmapStorage::create(&path, count as u32 / 10 + 100).unwrap();
                    (dir, storage)
                },
                |(dir, mut storage)| {
                    let mut btree = BTree::create(&mut storage, 0).unwrap();
                    for i in 0..count {
                        let key = format!("key{:08}", i);
                        let value = format!("value{:08}", i);
                        btree.insert(key.as_bytes(), value.as_bytes()).unwrap();
                    }
                    (dir, storage)
                },
            );
        });

        group.bench_with_input(BenchmarkId::new("random", count), count, |b, &count| {
            b.iter_with_setup(
                || {
                    let keys: Vec<usize> = {
                        let mut v: Vec<usize> = (0..count).collect();
                        for i in (1..v.len()).rev() {
                            let j = i % (i + 1);
                            v.swap(i, j);
                        }
                        v
                    };
                    let dir = tempdir().unwrap();
                    let path = dir.path().join("bench.tbd");
                    let storage = MmapStorage::create(&path, count as u32 / 10 + 100).unwrap();
                    (dir, storage, keys)
                },
                |(dir, mut storage, keys)| {
                    let mut btree = BTree::create(&mut storage, 0).unwrap();
                    for i in keys {
                        let key = format!("key{:08}", i);
                        let value = format!("value{:08}", i);
                        btree.insert(key.as_bytes(), value.as_bytes()).unwrap();
                    }
                    (dir, storage)
                },
            );
        });
    }

    group.finish();
}

fn bench_get(c: &mut Criterion) {
    let mut group = c.benchmark_group("btree_get");

    for count in [100, 1000].iter() {
        group.bench_with_input(
            BenchmarkId::new("existing_key", count),
            count,
            |b, &count| {
                let dir = tempdir().unwrap();
                let path = dir.path().join("bench.tbd");
                let mut storage = MmapStorage::create(&path, count as u32 / 10 + 100).unwrap();

                {
                    let mut btree = BTree::create(&mut storage, 0).unwrap();
                    for i in 0..count {
                        let key = format!("key{:08}", i);
                        let value = format!("value{:08}", i);
                        btree.insert(key.as_bytes(), value.as_bytes()).unwrap();
                    }
                }

                let key = format!("key{:08}", count / 2);
                b.iter(|| {
                    let btree = BTree::new(&mut storage, 0).unwrap();
                    let result = btree.get(black_box(key.as_bytes()));
                    hint_black_box(result.is_ok())
                });

                drop(dir);
            },
        );

        group.bench_with_input(
            BenchmarkId::new("nonexistent_key", count),
            count,
            |b, &count| {
                let dir = tempdir().unwrap();
                let path = dir.path().join("bench.tbd");
                let mut storage = MmapStorage::create(&path, count as u32 / 10 + 100).unwrap();

                {
                    let mut btree = BTree::create(&mut storage, 0).unwrap();
                    for i in 0..count {
                        let key = format!("key{:08}", i);
                        let value = format!("value{:08}", i);
                        btree.insert(key.as_bytes(), value.as_bytes()).unwrap();
                    }
                }

                let key = b"nonexistent_key_that_does_not_exist";
                b.iter(|| {
                    let btree = BTree::new(&mut storage, 0).unwrap();
                    let result = btree.get(black_box(key));
                    hint_black_box(result.is_ok())
                });

                drop(dir);
            },
        );
    }

    group.finish();
}

fn bench_cursor_scan(c: &mut Criterion) {
    let mut group = c.benchmark_group("btree_cursor_scan");

    for count in [100, 1000].iter() {
        group.throughput(Throughput::Elements(*count as u64));
        group.bench_with_input(BenchmarkId::new("forward", count), count, |b, &count| {
            let dir = tempdir().unwrap();
            let path = dir.path().join("bench.tbd");
            let mut storage = MmapStorage::create(&path, count as u32 / 10 + 100).unwrap();

            {
                let mut btree = BTree::create(&mut storage, 0).unwrap();
                for i in 0..count {
                    let key = format!("key{:08}", i);
                    let value = format!("value{:08}", i);
                    btree.insert(key.as_bytes(), value.as_bytes()).unwrap();
                }
            }

            b.iter(|| {
                let btree = BTree::new(&mut storage, 0).unwrap();
                let mut cursor = btree.cursor_first().unwrap();
                let mut scanned = 0;
                while cursor.valid() {
                    let _ = hint_black_box(cursor.key());
                    let _ = hint_black_box(cursor.value());
                    scanned += 1;
                    if !cursor.advance().unwrap() {
                        break;
                    }
                }
                scanned
            });

            drop(dir);
        });
    }

    group.finish();
}

fn bench_delete(c: &mut Criterion) {
    let mut group = c.benchmark_group("btree_delete");

    for count in [100, 500].iter() {
        group.throughput(Throughput::Elements(*count as u64));
        group.bench_with_input(BenchmarkId::new("sequential", count), count, |b, &count| {
            b.iter_with_setup(
                || {
                    let dir = tempdir().unwrap();
                    let path = dir.path().join("bench.tbd");
                    let mut storage = MmapStorage::create(&path, count as u32 / 10 + 100).unwrap();
                    {
                        let mut btree = BTree::create(&mut storage, 0).unwrap();
                        for i in 0..count {
                            let key = format!("key{:08}", i);
                            let value = format!("value{:08}", i);
                            btree.insert(key.as_bytes(), value.as_bytes()).unwrap();
                        }
                    }
                    (dir, storage)
                },
                |(dir, mut storage)| {
                    let mut btree = BTree::new(&mut storage, 0).unwrap();
                    for i in 0..count {
                        let key = format!("key{:08}", i);
                        btree.delete(key.as_bytes()).unwrap();
                    }
                    (dir, storage)
                },
            );
        });
    }

    group.finish();
}

fn bench_leaf_node_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("leaf_node");

    group.bench_function("insert_cell", |b| {
        b.iter_with_setup(
            || {
                let mut page = vec![0u8; PAGE_SIZE];
                LeafNodeMut::init(&mut page).unwrap();
                page
            },
            |mut page| {
                let mut node = LeafNodeMut::from_page(&mut page).unwrap();
                for i in 0..100 {
                    let key = format!("key{:04}", i);
                    let value = format!("val{:04}", i);
                    let _ = node.insert_cell(key.as_bytes(), value.as_bytes());
                }
                page
            },
        );
    });

    group.bench_function("find_key", |b| {
        let mut page = vec![0u8; PAGE_SIZE];
        {
            let mut node = LeafNodeMut::init(&mut page).unwrap();
            for i in 0..100 {
                let key = format!("key{:04}", i);
                let value = format!("val{:04}", i);
                node.insert_cell(key.as_bytes(), value.as_bytes()).unwrap();
            }
        }

        b.iter(|| {
            let node = LeafNode::from_page(&page).unwrap();
            for i in 0..100 {
                let key = format!("key{:04}", i);
                hint_black_box(node.find_key(black_box(key.as_bytes())));
            }
        });
    });

    group.finish();
}

fn bench_interior_node_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("interior_node");

    group.bench_function("insert_separator", |b| {
        b.iter_with_setup(
            || {
                let mut page = vec![0u8; PAGE_SIZE];
                InteriorNodeMut::init(&mut page, 999).unwrap();
                page
            },
            |mut page| {
                let mut node = InteriorNodeMut::from_page(&mut page).unwrap();
                for i in 0..100 {
                    let key = format!("sep{:04}", i);
                    let _ = node.insert_separator(key.as_bytes(), i as u32);
                }
                page
            },
        );
    });

    group.bench_function("find_child", |b| {
        let mut page = vec![0u8; PAGE_SIZE];
        {
            let mut node = InteriorNodeMut::init(&mut page, 999).unwrap();
            for i in 0..100 {
                let key = format!("sep{:04}", i);
                node.insert_separator(key.as_bytes(), i as u32).unwrap();
            }
        }

        b.iter(|| {
            let node = InteriorNode::from_page(&page).unwrap();
            for i in 0..100 {
                let key = format!("sep{:04}", i);
                let _ = hint_black_box(node.find_child(black_box(key.as_bytes())));
            }
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_insert,
    bench_get,
    bench_cursor_scan,
    bench_delete,
    bench_leaf_node_operations,
    bench_interior_node_operations,
);
criterion_main!(benches);
