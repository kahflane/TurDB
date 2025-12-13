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

The database MUST be able to start and perform basic operations with only 1MB RAM but it can grow more if needed:
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

### Key Encoding: Comprehensive Type Prefix Scheme

All keys MUST be encoded in big-endian byte-comparable format with type prefixes.
This allows multi-column keys of any type to be compared with single `memcmp`.

#### Type Prefix Constants

```rust
pub mod TypePrefix {
    // Special values
    pub const NULL: u8 = 0x01;
    pub const FALSE: u8 = 0x02;
    pub const TRUE: u8 = 0x03;

    // Numbers (ordered: negative < zero < positive)
    pub const NEG_INFINITY: u8 = 0x10;
    pub const NEG_BIG_INT: u8 = 0x11;   // Arbitrary precision negative
    pub const NEG_INT: u8 = 0x12;       // i64 negative
    pub const NEG_FLOAT: u8 = 0x13;     // f64 negative
    pub const ZERO: u8 = 0x14;
    pub const POS_FLOAT: u8 = 0x15;     // f64 positive
    pub const POS_INT: u8 = 0x16;       // i64 positive
    pub const POS_BIG_INT: u8 = 0x17;   // Arbitrary precision positive
    pub const POS_INFINITY: u8 = 0x18;
    pub const NAN: u8 = 0x19;           // NaN sorts after all numbers

    // Strings/Binary
    pub const TEXT: u8 = 0x20;
    pub const BLOB: u8 = 0x21;

    // Date/Time
    pub const DATE: u8 = 0x30;
    pub const TIME: u8 = 0x31;
    pub const TIMESTAMP: u8 = 0x32;
    pub const TIMESTAMPTZ: u8 = 0x33;
    pub const INTERVAL: u8 = 0x34;

    // Special types
    pub const UUID: u8 = 0x40;
    pub const INET: u8 = 0x41;          // IP addresses
    pub const MACADDR: u8 = 0x42;

    // JSON types (RFC 7159 ordering)
    pub const JSON_NULL: u8 = 0x50;
    pub const JSON_FALSE: u8 = 0x51;
    pub const JSON_TRUE: u8 = 0x52;
    pub const JSON_NUMBER: u8 = 0x53;
    pub const JSON_STRING: u8 = 0x54;
    pub const JSON_ARRAY: u8 = 0x55;
    pub const JSON_OBJECT: u8 = 0x56;

    // Composite/Custom types
    pub const ARRAY: u8 = 0x60;         // SQL arrays
    pub const TUPLE: u8 = 0x61;         // Row/Record types
    pub const RANGE: u8 = 0x62;         // PostgreSQL range types
    pub const ENUM: u8 = 0x63;          // Enum types
    pub const COMPOSITE: u8 = 0x64;     // User-defined composite
    pub const DOMAIN: u8 = 0x65;        // Domain types

    // Vectors
    pub const VECTOR: u8 = 0x70;

    // Extension point
    pub const CUSTOM_START: u8 = 0x80;  // 0x80-0xFE for custom types
    pub const MAX_KEY: u8 = 0xFF;       // Sentinel for range queries
}
```

#### Integer Encoding (Sign-Split)

```rust
fn encode_int(n: i64, buf: &mut Vec<u8>) {
    if n < 0 {
        buf.push(TypePrefix::NEG_INT);
        buf.extend((n as u64).to_be_bytes()); // Two's complement preserves order
    } else if n == 0 {
        buf.push(TypePrefix::ZERO);
    } else {
        buf.push(TypePrefix::POS_INT);
        buf.extend((n as u64).to_be_bytes());
    }
}
```

#### Float Encoding (IEEE Bit Manipulation)

```rust
fn encode_float(f: f64, buf: &mut Vec<u8>) {
    if f.is_nan() {
        buf.push(TypePrefix::NAN);
    } else if f == f64::NEG_INFINITY {
        buf.push(TypePrefix::NEG_INFINITY);
    } else if f == f64::INFINITY {
        buf.push(TypePrefix::POS_INFINITY);
    } else if f < 0.0 {
        buf.push(TypePrefix::NEG_FLOAT);
        buf.extend((!f.to_bits()).to_be_bytes()); // Invert all bits
    } else if f == 0.0 {
        buf.push(TypePrefix::ZERO);
    } else {
        buf.push(TypePrefix::POS_FLOAT);
        buf.extend((f.to_bits() ^ (1u64 << 63)).to_be_bytes()); // Flip sign bit
    }
}
```

#### Text Encoding (Escaped + Null-Terminated)

```rust
fn encode_text(s: &str, buf: &mut Vec<u8>) {
    buf.push(TypePrefix::TEXT);
    for byte in s.as_bytes() {
        match byte {
            0x00 => { buf.push(0x00); buf.push(0xFF); }  // Escape null
            0xFF => { buf.push(0xFF); buf.push(0x00); }  // Escape 0xFF
            _ => buf.push(*byte),
        }
    }
    buf.push(0x00);
    buf.push(0x00); // Double-null terminator
}
```

#### JSON Encoding (Recursive)

```rust
fn encode_json(json: &JsonValue, buf: &mut Vec<u8>) {
    match json {
        JsonValue::Null => buf.push(TypePrefix::JSON_NULL),
        JsonValue::Bool(false) => buf.push(TypePrefix::JSON_FALSE),
        JsonValue::Bool(true) => buf.push(TypePrefix::JSON_TRUE),
        JsonValue::Number(n) => {
            buf.push(TypePrefix::JSON_NUMBER);
            encode_json_number(n, buf);
        }
        JsonValue::String(s) => {
            buf.push(TypePrefix::JSON_STRING);
            encode_text_body(s, buf);
        }
        JsonValue::Array(arr) => {
            buf.push(TypePrefix::JSON_ARRAY);
            for elem in arr { encode_json(elem, buf); buf.push(0x01); }
            buf.push(0x00);
        }
        JsonValue::Object(obj) => {
            buf.push(TypePrefix::JSON_OBJECT);
            let mut keys: Vec<_> = obj.keys().collect();
            keys.sort(); // Deterministic key order
            for key in keys {
                encode_text_body(key, buf);
                buf.push(0x02);
                encode_json(&obj[key], buf);
                buf.push(0x01);
            }
            buf.push(0x00);
        }
    }
}
```

#### Composite/Custom Types

```rust
fn encode_composite(fields: &[Value], type_id: u32, buf: &mut Vec<u8>) {
    buf.push(TypePrefix::COMPOSITE);
    buf.extend(type_id.to_be_bytes()); // Type OID
    for field in fields {
        encode_value(field, buf);
        buf.push(0x01); // Field separator
    }
    buf.push(0x00); // Terminator
}

fn encode_enum(variant_ordinal: u32, type_id: u32, buf: &mut Vec<u8>) {
    buf.push(TypePrefix::ENUM);
    buf.extend(type_id.to_be_bytes());
    buf.extend(variant_ordinal.to_be_bytes()); // Ordinal preserves declaration order
}

fn encode_array(elements: &[Value], buf: &mut Vec<u8>) {
    buf.push(TypePrefix::ARRAY);
    for elem in elements {
        encode_value(elem, buf);
        buf.push(0x01);
    }
    buf.push(0x00);
}
```

#### Date/Time Encoding

```rust
fn encode_timestamp(micros_since_epoch: i64, buf: &mut Vec<u8>) {
    buf.push(TypePrefix::TIMESTAMP);
    buf.extend((micros_since_epoch ^ i64::MIN).to_be_bytes()); // XOR flip for signed
}

fn encode_uuid(uuid: &[u8; 16], buf: &mut Vec<u8>) {
    buf.push(TypePrefix::UUID);
    buf.extend(uuid); // Already byte-comparable
}
```

#### Type Ordering Summary

```
NULL < FALSE < TRUE < -∞ < negative numbers < 0 < positive numbers < +∞ < NaN
< TEXT < BLOB < DATE < TIME < TIMESTAMP < UUID < JSON_* < ARRAY < COMPOSITE < CUSTOM
```

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
