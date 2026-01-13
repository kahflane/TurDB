# TurDB vs SQLite Insertion Benchmarks

**Date:** 2026-01-13
**System:** macOS Darwin 25.1.0
**Benchmark:** `cargo bench --bench insertion`

---

## Benchmark Results

### Single Raw SQL (1,000 rows - full SQL parsing each iteration)

| Database | WAL Mode | Time | Throughput | vs SQLite WAL ON |
|----------|----------|------|------------|------------------|
| SQLite | ON | 1.57 ms | 638 K/s | baseline |
| SQLite | OFF | 1.74 ms | 574 K/s | 0.90x |
| TurDB | ON | 20.5 ms | 49 K/s | 0.08x |
| TurDB | OFF | 19.5 ms | 51 K/s | 0.08x |

**Gap: SQLite is ~13x faster**

---

### Single Prepared (10,000 rows - parse once, execute many)

| Database | WAL Mode | Time | Throughput | vs SQLite WAL ON |
|----------|----------|------|------------|------------------|
| SQLite | ON | 4.90 ms | 2.04 M/s | baseline |
| SQLite | OFF | 4.87 ms | 2.06 M/s | 1.01x |
| TurDB | ON | 18.8 ms | 532 K/s | 0.26x |
| TurDB | OFF | 11.9 ms | 842 K/s | 0.41x |

**Gap: SQLite is ~4x faster**

---

### Batch Insert (100,000 rows - multi-row SQL, 1,000 per batch)

| Database | WAL Mode | Time | Throughput | vs SQLite WAL ON |
|----------|----------|------|------------|------------------|
| SQLite | ON | 103 ms | 967 K/s | baseline |
| SQLite | OFF | 117 ms | 855 K/s | 0.88x |
| TurDB | ON | 607 ms | 165 K/s | 0.17x |
| TurDB | OFF | 397 ms | 252 K/s | 0.26x |

**Gap: SQLite is ~4-6x faster**

---

### Batch Prepared (100,000 rows - prepared statements, 1,000 per transaction)

| Database | WAL Mode | Time | Throughput | vs SQLite WAL ON |
|----------|----------|------|------------|------------------|
| SQLite | ON | 51 ms | 1.96 M/s | baseline |
| SQLite | OFF | 68 ms | 1.48 M/s | 0.76x |
| TurDB | ON | 303 ms | 330 K/s | 0.17x |
| TurDB | OFF | 98 ms | 1.02 M/s | 0.52x |

**Gap: SQLite is ~2-6x faster**

---

## Current State Summary

| Metric | SQLite Best | TurDB Best | Gap |
|--------|-------------|------------|-----|
| Raw SQL parsing | 638 K/s | 51 K/s | 12.5x |
| Prepared statement | 2.06 M/s | 842 K/s | 2.4x |
| Batch SQL | 967 K/s | 252 K/s | 3.8x |
| Batch prepared | 1.96 M/s | 1.02 M/s | 1.9x |

**Best case gap: 1.9x (batch prepared, WAL OFF)**
**Worst case gap: 12.5x (raw SQL parsing)**

---

## Target: Beat SQLite

### Phase 1: Close the Gap (Target: 1.5x of SQLite)

#### 1.1 SQL Parser Optimization

**Current bottleneck:** TurDB re-parses SQL for every execution, even with "prepared" statements.

**Actions:**
- [ ] Profile parser with `perf` to identify hot spots
- [ ] Implement true prepared statement caching (parse once, reuse AST)
- [ ] Use arena allocation for parser temporaries (already using `bumpalo`)
- [ ] Consider generating a bytecode/IR from AST for faster execution
- [ ] Evaluate PHF (perfect hash function) for keyword lookup (already in deps)

**Expected gain:** 3-5x for raw SQL, 1.5-2x for prepared statements

#### 1.2 Record Building Optimization

**Current bottleneck:** Record serialization allocates and copies data.

**Actions:**
- [ ] Profile `get_batch_timing_stats()` to measure record build vs B-tree time
- [ ] Implement streaming record builder (write directly to page buffer)
- [ ] Pre-compute record size to avoid reallocation
- [ ] Use SIMD for bulk value encoding where applicable

**Expected gain:** 1.2-1.5x for insert throughput

#### 1.3 B-Tree Insert Path

**Current bottleneck:** Page splits and key comparisons.

**Actions:**
- [ ] Optimize fastpath hit rate (currently tracked via `get_fastpath_stats()`)
- [ ] Implement bulk loading mode that bypasses normal insert path
- [ ] Use SIMD for key prefix comparison
- [ ] Reduce lock contention in page cache during sequential inserts

**Expected gain:** 1.3-1.5x for sequential inserts

---

### Phase 2: Match SQLite (Target: 1.0x)

#### 2.1 WAL Optimization

**Current state:** TurDB WAL adds significant overhead vs WAL OFF.

**Actions:**
- [ ] Profile WAL write path
- [ ] Implement group commit (batch multiple transactions into single WAL write)
- [ ] Use `io_uring` on Linux for async WAL writes
- [ ] Implement WAL frame coalescing (combine multiple writes to same page)
- [ ] Consider shadow paging as alternative to WAL for some workloads

**Expected gain:** 1.5-2x for WAL ON mode

#### 2.2 Transaction Overhead

**Current state:** BEGIN/COMMIT have measurable overhead.

**Actions:**
- [ ] Profile transaction start/commit paths
- [ ] Implement lightweight "auto-commit" mode for single statements
- [ ] Reduce MVCC overhead for single-threaded workloads
- [ ] Cache transaction state to avoid repeated lookups

**Expected gain:** 1.1-1.2x

#### 2.3 Memory Layout

**Actions:**
- [ ] Ensure hot data structures are cache-line aligned
- [ ] Profile cache misses with `perf stat`
- [ ] Consider column-oriented storage for bulk inserts
- [ ] Optimize page header layout for common access patterns

**Expected gain:** 1.1-1.3x

---

### Phase 3: Beat SQLite (Target: 1.5x+ faster)

#### 3.1 Leverage Rust Advantages

**SQLite limitations we can exploit:**
- Single-threaded write model
- C's lack of zero-cost abstractions
- Global lock for write transactions

**Actions:**
- [ ] Implement concurrent B-tree writes with fine-grained locking
- [ ] Use lock-free data structures for hot paths
- [ ] Exploit Rust's ownership model for zero-copy operations
- [ ] SIMD-accelerated encoding/decoding

**Expected gain:** 1.5-2x for concurrent workloads

#### 3.2 Modern Storage Techniques

**Actions:**
- [ ] Implement LSM-tree mode for write-heavy workloads
- [ ] Add Bw-tree option for high-concurrency scenarios
- [ ] Support direct I/O with custom buffer management
- [ ] Implement transparent compression for large values

**Expected gain:** Variable, workload-dependent

#### 3.3 Hardware Acceleration

**Actions:**
- [ ] AVX-512 for bulk operations
- [ ] GPU offload for vector operations (already have HNSW)
- [ ] NVMe optimizations (multiple queues, polling)
- [ ] Persistent memory (PMEM) support

**Expected gain:** 2-10x for specific operations

---

## Immediate Action Items (Next Sprint)

### Priority 1: Parser Performance
1. Add parser timing to benchmarks
2. Profile with `cargo flamegraph`
3. Implement AST caching for prepared statements

### Priority 2: Prepared Statement Path
1. Verify prepared statements actually skip parsing
2. Optimize parameter binding (avoid cloning OwnedValue)
3. Cache execution plans

### Priority 3: Measurement Infrastructure
1. Add more granular timing breakdowns
2. Create micro-benchmarks for each component
3. Set up continuous benchmarking in CI

---

## Success Metrics

| Milestone | Target | Timeline |
|-----------|--------|----------|
| Close gap to 2x | Batch prepared: 1.0 M/s | Current |
| Close gap to 1.5x | Batch prepared: 1.3 M/s | - |
| Match SQLite | Batch prepared: 2.0 M/s | - |
| Beat SQLite | Batch prepared: 3.0 M/s | - |

---

## Notes

- SQLite has 20+ years of optimization; closing the gap requires systematic profiling
- TurDB's architecture (zero-copy, Rust) provides theoretical advantages not yet realized
- WAL implementation needs significant work; currently slower than no-WAL
- Prepared statement path shows most promise (smallest gap at 1.9x)
- Focus on batch prepared path first as it's closest to parity
