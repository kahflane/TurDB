# TurDB Rust Implementation - Development Guidelines

## Project Overview

TurDB is a high-performance embedded database combining SQLite-inspired row storage with native vector search (HNSW). This Rust implementation prioritizes zero-copy operations, zero allocation during CRUD, and extreme memory efficiency.

## Core Principles

### 1. Zero-Copy Architecture

All data access MUST use zero-copy patterns:
- Use `&[u8]` slices pointing directly into mmap'd regions
- Parse data in-place without intermediate buffers
- Return references to page data, never cloned copies
- Use `zerocopy` crate for safe transmutation of bytes to structs

```rust
// CORRECT: Zero-copy access
fn get_key(&self, page: &Page) -> &[u8] {
    &page.data()[self.key_offset..self.key_end]
}

// WRONG: Copies data
fn get_key(&self, page: &Page) -> Vec<u8> {
    page.data()[self.key_offset..self.key_end].to_vec()
}
```

### Mmap Safety via Borrow Checker

MmapStorage uses Rust's borrow checker to prevent use-after-unmap at **compile time** with **zero runtime overhead**:

```rust
pub struct MmapStorage {
    file: File,
    mmap: MmapMut,
    page_size: usize,
    page_count: u32,
}

impl MmapStorage {
    /// Returns page slice. Borrows &self, preventing grow() while borrowed.
    pub fn page(&self, page_no: u32) -> Result<&[u8]> {
        ensure!(page_no < self.page_count, "page out of bounds");
        let offset = page_no as usize * self.page_size;
        Ok(&self.mmap[offset..offset + self.page_size])
    }

    /// Requires &mut self - impossible if any page is borrowed.
    pub fn grow(&mut self, new_page_count: u32) -> Result<()> {
        let new_size = new_page_count as u64 * self.page_size as u64;
        self.file.set_len(new_size)?;
        self.mmap = unsafe { MmapMut::map_mut(&self.file)? };
        self.page_count = new_page_count;
        Ok(())
    }
}
```

**Compile-time enforcement:**
```rust
// This compiles - page reference dropped before grow
let val = {
    let page = storage.page(5)?;
    page[0]
};
storage.grow(100)?;  // ✅ OK

// This FAILS to compile - borrow checker prevents use-after-unmap
let page = storage.page(5)?;
storage.grow(100)?;  // ❌ ERROR: cannot borrow as mutable
let byte = page[0];  //    because it is also borrowed as immutable
```

**Benefits:**
- Zero runtime cost (no locks, guards, or epoch tracking)
- Zero-copy (returns `&[u8]` directly into mmap)
- Compile-time safety (borrow checker prevents dangling pointers)
- Idiomatic Rust (uses the language's core safety mechanism)

### 2. Zero Allocation During CRUD

CRUD operations MUST NOT allocate heap memory:
- Pre-allocate all buffers during database open
- Use arena allocators for temporary data
- Reuse cursor structs across operations
- Stack-allocate small fixed-size buffers

```rust
// CORRECT: Reuse pre-allocated cursor
fn scan(&self, cursor: &mut Cursor) -> Result<()>

// WRONG: Allocates new cursor each call
fn scan(&self) -> Result<Cursor>
```

### 3. Page Size

All implementations MUST use 16KB (16384 bytes) page size:
- **Page header**: 16 bytes (every page)
- **Usable space**: 16368 bytes per page
- **File header**: 128 bytes (page 0 of each file only)
- **Page 0 usable**: 16256 bytes (128-byte file header eats into it)

```
Regular Page (1+):          Page 0 (file header):
+------------------+        +------------------+
| Page Hdr (16B)   |        | File Hdr (128B)  |
+------------------+        +------------------+
| Usable (16368B)  |        | Usable (16256B)  |
+------------------+        +------------------+
```

### 4. Memory Budget: 1MB Minimum

The database MUST be able to start and perform basic operations with only 1MB RAM:
- Page cache: 32 pages minimum (512KB)
- Fixed overhead: Maximum 256KB
- Working memory: Maximum 256KB
- Lazy load all metadata
- Stream large results instead of buffering

### 5. File-Per-Table Architecture (MySQL-Style)

```
database_dir/
├── turdb.meta           # Global metadata and catalog
├── root/                # Default schema
│   ├── table_name.tbd   # Table data file
│   ├── table_name.idx   # B-tree indexes
│   └── table_name.hnsw  # HNSW vector indexes (if any)
├── custom_schema/       # User-created schema
│   └── ...
└── wal/
    └── wal.000001       # Write-ahead log segments
```

### 6. Multi-Schema Support

Support multiple schemas (like PostgreSQL):
- Default schema: `root`
- System schema: `turdb_catalog`
- User schemas: Created via `CREATE SCHEMA`
- Fully qualified names: `schema.table.column`

## Error Handling

### Use `eyre` Exclusively

All error handling MUST use `eyre`:

```rust
use eyre::{Result, WrapErr, bail, ensure};

fn open_page(&self, page_no: u32) -> Result<&Page> {
    let page = self.cache.get(page_no)
        .wrap_err_with(|| format!("failed to load page {}", page_no))?;
    Ok(page)
}
```

### Error Context Requirements

Every error MUST include:
1. What operation was being performed
2. What resource was involved (page number, table name, etc.)
3. Why it failed (if known)

```rust
// CORRECT: Rich context
.wrap_err_with(|| format!("failed to insert row into table '{}' at page {}", table, page_no))

// WRONG: No context
.wrap_err("insert failed")
```

### No Custom Error Types

Do NOT create custom error enums. Use `eyre::Report` with context:

```rust
// WRONG
enum DbError {
    PageNotFound(u32),
    TableNotFound(String),
}

// CORRECT
bail!("page {} not found in table '{}'", page_no, table_name)
```

## Code Documentation Style

### No Inline Comments

NEVER write inline comments in code:

```rust
// WRONG
fn insert(&mut self, key: &[u8], value: &[u8]) -> Result<()> {
    // Find the right leaf page
    let leaf = self.find_leaf(key)?;
    // Insert into the leaf
    leaf.insert(key, value)?;
    Ok(())
}
```

### Block Documentation at File Top

Each file MUST start with 80-100 lines of block documentation explaining:
1. Purpose of this module
2. Architecture and design decisions
3. Key data structures
4. Usage patterns
5. Performance characteristics
6. Safety considerations

```rust
//! # B-Tree Implementation
//!
//! This module implements the core B-tree structure for TurDB's row storage.
//!
//! ## Architecture
//!
//! The B-tree uses a page-based design where each node occupies exactly one
//! 16KB page. Interior nodes contain keys and child page pointers, while
//! leaf nodes contain keys and row data (or overflow pointers for large rows).
//!
//! ## Page Layout
//!
//! ```text
//! +------------------+
//! | Header (16 bytes)|
//! +------------------+
//! | Cell Pointers    |
//! | (2 bytes each)   |
//! +------------------+
//! | Free Space       |
//! +------------------+
//! | Cell Content     |
//! | (grows upward)   |
//! +------------------+
//! ```
//!
//! ... (continue to 80-100 lines)
```

### Doc Comments for Public API Only

Use `///` only for public functions, structs, and traits. Keep them concise (1-3 lines):

```rust
/// Opens a database at the specified path.
pub fn open(path: &Path) -> Result<Database>

/// Returns the number of rows in this table.
pub fn row_count(&self) -> u64
```

## Module Structure

### Crate Organization

```
src/
├── lib.rs              # Public API re-exports
├── database.rs         # Database handle and lifecycle
├── transaction.rs      # Transaction management
├── schema/
│   ├── mod.rs          # Schema types and catalog
│   ├── table.rs        # Table definitions
│   └── index.rs        # Index definitions
├── storage/
│   ├── mod.rs          # Storage traits
│   ├── pager.rs        # Page manager
│   ├── page.rs         # Page types and layouts
│   ├── mmap.rs         # Memory-mapped storage
│   └── wal.rs          # Write-ahead log
├── btree/
│   ├── mod.rs          # B-tree implementation
│   ├── node.rs         # Node operations
│   └── cursor.rs       # Cursor for iteration
├── hnsw/
│   ├── mod.rs          # HNSW index
│   ├── graph.rs        # Graph structure
│   └── search.rs       # k-NN search
├── sql/
│   ├── mod.rs          # SQL processing
│   ├── lexer.rs        # Tokenization
│   ├── parser.rs       # Parsing
│   ├── planner.rs      # Query planning
│   └── executor.rs     # Execution engine
├── record.rs           # Row serialization
├── types.rs            # Value types
└── encoding.rs         # Varint and binary encoding
```

### Visibility Rules

- Minimize pub usage
- Use `pub(crate)` for internal APIs
- Only `pub` for user-facing API
- Never `pub use` internal modules at crate root

## Memory Management

### Arena Allocators for Queries

Use `bumpalo` for per-query allocations:

```rust
fn execute_query<'a>(&self, arena: &'a Bump, sql: &str) -> Result<Rows<'a>>
```

### No Global Allocators

Never use global state or singleton patterns. All state flows through explicit parameters.

### Drop Order

Implement Drop carefully to ensure:
1. Dirty pages flushed before pager drops
2. WAL synced before file closes
3. Locks released in reverse acquisition order

## Concurrency Model

### Send + Sync Requirements

All public types that cross thread boundaries MUST be `Send + Sync`:
- `Database`: `Send + Sync` (can be shared across threads)
- `Transaction`: `Send + !Sync` (bound to one thread, movable)
- `Cursor`: `!Send + !Sync` (thread-local only)

### Locking Strategy - Lock Sharding

NEVER use a single global lock for the page cache. Use **lock sharding**:

```rust
const SHARD_COUNT: usize = 64;

struct PageCache {
    shards: [RwLock<CacheShard>; SHARD_COUNT],
}

impl PageCache {
    fn shard_for(&self, page_no: u32) -> &RwLock<CacheShard> {
        &self.shards[(page_no as usize) % SHARD_COUNT]
    }
}
```

Use `parking_lot` for all synchronization:
- Sharded `RwLock` for page cache (64 shards minimum)
- `RwLock` for catalog (single lock OK - read-heavy, rarely written)
- `Mutex` for WAL, dirty list

### Lock Ordering

Strict lock order to prevent deadlocks:
1. Database lock
2. Schema lock
3. Table lock (alphabetical order if multiple)
4. Page cache shard lock (by shard index)
5. WAL lock

### No Async Runtime

The core engine is synchronous. Avoid `tokio`/`async-std`:
- mmap page faults block threads regardless of async
- Embedded use case has no network I/O in hot path
- Async adds ~100-500ns latency per await point

For server mode, use a separate `turdb-server` crate with `spawn_blocking`.

## Performance Requirements

### Benchmarks

All CRUD operations MUST meet these targets:
- Point read: < 1µs (cached), < 50µs (disk)
- Sequential scan: > 1M rows/sec
- Insert: > 100K rows/sec
- k-NN search (1M vectors, k=10): < 10ms

### Profiling

Use `criterion` for benchmarks. Every PR touching performance-critical code must include benchmark results.

## Algorithmic Optimizations

### Cache Eviction: SIEVE (Not LRU)

Standard LRU is bad for database scans (sequential scan flushes entire cache). Use **SIEVE** algorithm:

```rust
struct SieveCache {
    entries: Vec<CacheEntry>,
    hand: usize,
}

struct CacheEntry {
    page_no: u32,
    visited: AtomicBool,
    data: *mut [u8; PAGE_SIZE],
}

impl SieveCache {
    fn evict(&mut self) -> u32 {
        loop {
            let entry = &self.entries[self.hand];
            if entry.visited.swap(false, Ordering::Relaxed) {
                self.hand = (self.hand + 1) % self.entries.len();
            } else {
                return entry.page_no;
            }
        }
    }

    fn access(&self, idx: usize) {
        self.entries[idx].visited.store(true, Ordering::Relaxed);
    }
}
```

### B+Tree: Slot Array with Prefix Hints

Store first 4 bytes of each key in a compact slot array for fast SIMD comparison:

```rust
struct LeafPage {
    header: PageHeader,
    slot_count: u16,
    slots: [Slot; MAX_SLOTS],
    data: [u8],
}

#[repr(C)]
struct Slot {
    prefix: [u8; 4],
    offset: u16,
    len: u16,
}

impl LeafPage {
    fn find_key(&self, key: &[u8]) -> SearchResult {
        let target_prefix = key.get(..4).map(|s| u32::from_be_bytes(s.try_into().unwrap()));

        for (i, slot) in self.slots[..self.slot_count as usize].iter().enumerate() {
            let slot_prefix = u32::from_be_bytes(slot.prefix);
            if slot_prefix > target_prefix.unwrap_or(0) {
                return SearchResult::NotFound(i);
            }
            if slot_prefix == target_prefix.unwrap_or(0) {
                let full_key = self.get_key(slot);
                match full_key.cmp(key) {
                    Ordering::Equal => return SearchResult::Found(i),
                    Ordering::Greater => return SearchResult::NotFound(i),
                    Ordering::Less => continue,
                }
            }
        }
        SearchResult::NotFound(self.slot_count as usize)
    }
}
```

### B+Tree: Suffix Truncation for Interior Nodes

Interior nodes only need enough key bytes to distinguish children:

```rust
fn compute_separator(left_max: &[u8], right_min: &[u8]) -> Vec<u8> {
    let mut sep = Vec::new();
    for (l, r) in left_max.iter().zip(right_min.iter()) {
        if l < r {
            sep.push(*l + 1);
            return sep;
        }
        sep.push(*l);
    }
    sep
}
```

### Key Encoding: Big-Endian for memcmp

All keys MUST be encoded in big-endian byte-comparable format:

```rust
fn encode_key(columns: &[Value]) -> Vec<u8> {
    let mut buf = Vec::new();
    for col in columns {
        match col {
            Value::Int(n) => {
                buf.push(0x05);
                buf.extend((n ^ i64::MIN).to_be_bytes());
            }
            Value::Text(s) => {
                buf.push(0x06);
                buf.extend(s.as_bytes());
                buf.push(0x00);
            }
            Value::Null => buf.push(0x01),
        }
    }
    buf
}
```

This allows multi-column keys to be compared with single `memcmp`.

### Range Scans: madvise Prefetching

When scanning leaf pages sequentially, prefetch ahead:

```rust
impl Cursor {
    fn next(&mut self) -> Result<bool> {
        if self.at_page_end() {
            let next_page = self.current_leaf.next_leaf;
            let prefetch_page = self.current_leaf.next_leaf + 2;

            unsafe {
                libc::madvise(
                    self.mmap.page_ptr(prefetch_page),
                    PAGE_SIZE * 2,
                    libc::MADV_WILLNEED,
                );
            }

            self.current_leaf = self.load_page(next_page)?;
        }
        self.advance_within_page()
    }
}
```

### Joins: Grace Hash Join for Memory-Constrained

With 256KB working memory, use partitioned hash join:

```rust
impl GraceHashJoin {
    fn execute(&mut self) -> Result<()> {
        let partition_count = 16;

        for row in self.build_side.scan()? {
            let hash = self.hash_key(&row);
            let partition = hash % partition_count;
            self.partitions[partition].write(&row)?;
        }

        for partition_id in 0..partition_count {
            let build_partition = self.partitions[partition_id].read_all()?;
            let hash_table = self.build_hash_table(&build_partition);

            for row in self.probe_side.scan_partition(partition_id)? {
                if let Some(match_row) = hash_table.probe(&row) {
                    self.emit_result(&row, match_row)?;
                }
            }
        }
        Ok(())
    }
}
```

### SIMD Distance Functions

Use AVX2/NEON for vector distance calculations:

```rust
#[cfg(target_arch = "x86_64")]
pub fn euclidean_avx2(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::x86_64::*;

    unsafe {
        let mut sum = _mm256_setzero_ps();
        for i in (0..a.len()).step_by(8) {
            let va = _mm256_loadu_ps(a.as_ptr().add(i));
            let vb = _mm256_loadu_ps(b.as_ptr().add(i));
            let diff = _mm256_sub_ps(va, vb);
            sum = _mm256_fmadd_ps(diff, diff, sum);
        }
        horizontal_sum_avx2(sum).sqrt()
    }
}
```

### Vector Quantization: SQ8

Compress float32 vectors to uint8 for 4x memory reduction:

```rust
struct SQ8Vector {
    min: f32,
    scale: f32,
    data: Vec<u8>,
}

impl SQ8Vector {
    fn from_f32(vec: &[f32]) -> Self {
        let min = vec.iter().cloned().fold(f32::INFINITY, f32::min);
        let max = vec.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let scale = (max - min) / 255.0;

        let data = vec.iter()
            .map(|&v| ((v - min) / scale) as u8)
            .collect();

        Self { min, scale, data }
    }

    fn distance_sq8(&self, other: &Self) -> f32 {
        let mut sum: u32 = 0;
        for (a, b) in self.data.iter().zip(&other.data) {
            let diff = (*a as i32) - (*b as i32);
            sum += (diff * diff) as u32;
        }
        (sum as f32) * self.scale * self.scale
    }
}

## Testing Requirements

### Unit Tests

- Test every public function
- Test edge cases (empty, max size, boundary conditions)
- Test error paths

### Integration Tests

- Full CRUD cycles
- Crash recovery scenarios
- Concurrent access patterns
- Memory-constrained operation (1MB limit)

### Fuzz Testing

Use `cargo-fuzz` for:
- SQL parser
- Record serialization
- B-tree operations
- HNSW graph operations

## Dependencies

### Allowed Dependencies

```toml
[dependencies]
eyre = "0.6"
parking_lot = "0.12"
memmap2 = "0.9"
zerocopy = "0.7"
bumpalo = "3.14"
smallvec = "1.11"
hashbrown = "0.14"
```

### Forbidden Dependencies

- `tokio`, `async-std`: No async runtime (sync-only design)
- `serde`: Use custom zero-copy serialization
- `regex`: Too heavy for simple parsing needs
- Any dependency > 50KB compiled size without justification

## Unsafe Code Policy

### When Unsafe is Allowed

1. mmap operations (inherently unsafe)
2. Zero-copy transmutation with `zerocopy`
3. SIMD intrinsics for vector distance calculations
4. Custom allocator implementations

### Unsafe Requirements

Every `unsafe` block MUST have:
1. A comment block explaining why it's safe
2. Tests that verify the safety invariants
3. Miri compatibility (no undefined behavior)

```rust
unsafe {
    // SAFETY: We verified page_no < page_count in the check above,
    // and the mmap region is valid for the entire file lifetime.
    // The returned slice's lifetime is tied to &self, ensuring
    // the mmap stays valid.
    std::slice::from_raw_parts(ptr, PAGE_SIZE)
}
```

## Git Workflow

### Branch Naming

- `feature/description` for new features
- `fix/description` for bug fixes
- `perf/description` for performance improvements
- `refactor/description` for refactoring

### Commit Messages

```
type(scope): description

- Bullet points for details
- Reference issue numbers
```

Types: `feat`, `fix`, `perf`, `refactor`, `test`, `docs`, `chore`

## File Format Specifications

### Global Metadata File (`turdb.meta`)

```
Offset  Size  Description
0       16    Magic: "TurDB Rust v1\x00\x00\x00"
16      4     Version: 1
20      4     Page size: 16384
24      8     Schema count
32      8     Default schema ID
40      8     Next table ID
48      8     Next index ID
56      8     Flags
64      64    Reserved
```

### Table Data File Header (`.tbd`)

```
Offset  Size  Description
0       16    Magic: "TurDB Table\x00\x00\x00\x00"
16      8     Table ID
24      8     Row count
32      4     Root page number
36      4     Column count
40      8     First free page
48      8     Auto-increment value
56      72    Reserved
```

### Index File Header (`.idx`)

```
Offset  Size  Description
0       16    Magic: "TurDB Index\x00\x00\x00\x00"
16      8     Index ID
24      8     Table ID
32      4     Root page number
36      4     Key column count
40      1     Is unique
41      1     Index type (0=btree)
42      86    Reserved
```

### HNSW File Header (`.hnsw`)

```
Offset  Size  Description
0       16    Magic: "TurDB HNSW\x00\x00\x00\x00\x00"
16      8     Index ID
24      8     Table ID
32      4     Dimension
36      4     M (max connections)
40      4     EfConstruction
44      4     Entry point node
48      8     Node count
56      8     Vector count
64      64    Reserved
```

## Build Configuration

### Release Profile

```toml
[profile.release]
opt-level = 3
lto = "fat"
codegen-units = 1
panic = "abort"
strip = true
```

### Dev Profile

```toml
[profile.dev]
opt-level = 1
debug = true
```

## API Design

### Builder Pattern for Options

```rust
let db = Database::builder()
    .path("./mydb")
    .page_cache_size(64)
    .read_only(false)
    .open()?;
```

### Fluent Query Interface

```rust
let rows = db.query("SELECT * FROM users WHERE age > ?")
    .bind(18)
    .fetch_all()?;
```

### Resource Management

All resources use RAII:
- `Database` closes files on drop
- `Transaction` rolls back if not committed
- `Cursor` releases page pins on drop

## Migration from Go

### Phase Priority

1. Storage layer (pager, mmap, page types)
2. B-tree implementation
3. Record serialization
4. Schema and catalog
5. SQL parser and planner
6. Query executor
7. HNSW index
8. MVCC and transactions
9. WAL and recovery
10. Public API and CLI

### Compatibility

- Binary format is NOT compatible with Go version
- SQL syntax should be compatible
- Migration tool will be provided for data export/import
