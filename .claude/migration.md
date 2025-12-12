# TurDB: Go to Rust Migration Plan

## Executive Summary

This document outlines the migration strategy for TurDB from Go to Rust. The migration targets:
- **Zero-copy data access** via mmap and careful lifetime management
- **Zero allocation during CRUD** through pre-allocated buffers and arena allocators
- **16KB page size** (up from Go's 4KB default)
- **1MB minimum RAM** requirement for basic operation
- **MySQL-style file layout** (one file per table) replacing single-file design
- **Multi-schema support** for database isolation

## Current Go Architecture Overview

### Package Mapping

| Go Package | Responsibility | Rust Module |
|------------|---------------|-------------|
| `pkg/turdb` | Public API | `src/lib.rs`, `src/database.rs` |
| `pkg/pager` | Page I/O, cache | `src/storage/pager.rs` |
| `pkg/dbfile` | File format | `src/storage/format.rs` |
| `pkg/btree` | B-tree index | `src/btree/mod.rs` |
| `pkg/cowbtree` | COW B-tree | `src/btree/cow.rs` |
| `pkg/hnsw` | Vector index | `src/hnsw/mod.rs` |
| `pkg/mvcc` | Concurrency | `src/transaction/mvcc.rs` |
| `pkg/wal` | Write-ahead log | `src/storage/wal.rs` |
| `pkg/record` | Row encoding | `src/record.rs` |
| `pkg/schema` | Catalog | `src/schema/mod.rs` |
| `pkg/types` | Value types | `src/types.rs` |
| `pkg/vdbe` | Bytecode VM | `src/sql/executor.rs` |
| `pkg/sql/*` | SQL processing | `src/sql/mod.rs` |
| `pkg/encoding` | Varint | `src/encoding.rs` |
| `pkg/cache` | Memory mgmt | `src/storage/cache.rs` |

### Key Architectural Changes

1. **Single File → Multiple Files**: Each table gets its own `.tbd` file, indexes get `.idx` files
2. **4KB Pages → 16KB Pages**: Larger pages reduce tree depth and improve sequential access
3. **GC → Manual Memory**: Rust ownership replaces Go GC; arenas for query memory
4. **Interface{} → Generics**: Type-safe value representation without boxing
5. **Goroutines → OS Threads**: `std::thread` with `parking_lot` synchronization

## Phase 1: Storage Foundation (Weeks 1-3)

### 1.1 Memory-Mapped File Storage

**Go Source**: `pkg/pager/mmap_unix.go`, `pkg/pager/mmap_windows.go`

**Rust Implementation**:

```rust
// src/storage/mmap.rs

pub struct MmapStorage {
    file: File,
    mmap: MmapMut,
    page_size: usize,
    page_count: u32,
}

impl MmapStorage {
    pub fn page(&self, page_no: u32) -> Result<&[u8]> {
        ensure!(page_no < self.page_count, "page {} out of bounds", page_no);
        let offset = page_no as usize * self.page_size;
        Ok(&self.mmap[offset..offset + self.page_size])
    }

    pub fn page_mut(&mut self, page_no: u32) -> Result<&mut [u8]> {
        ensure!(page_no < self.page_count, "page {} out of bounds", page_no);
        let offset = page_no as usize * self.page_size;
        Ok(&mut self.mmap[offset..offset + self.page_size])
    }
}
```

**Key Differences from Go**:
- Go uses `syscall.Mmap` with manual pointer arithmetic
- Rust uses `memmap2` crate with safe slice access
- Lifetime annotations ensure pages don't outlive mmap

**Zero-Copy Strategy**:
- Return `&[u8]` slices directly into mmap region
- Use `zerocopy::FromBytes` for safe struct transmutation
- Never copy page data into intermediate buffers

### 1.2 Page Types and Layout

**Go Source**: `pkg/pager/page.go`

**Rust Implementation**:

```rust
// src/storage/page.rs

#[repr(u8)]
pub enum PageType {
    Unknown = 0x00,
    BTreeInterior = 0x01,
    BTreeLeaf = 0x02,
    HnswNode = 0x10,
    HnswMeta = 0x11,
    Overflow = 0x20,
    FreeList = 0x30,
}

pub const PAGE_SIZE: usize = 16384;
pub const PAGE_HEADER_SIZE: usize = 16;
pub const PAGE_USABLE_SIZE: usize = PAGE_SIZE - PAGE_HEADER_SIZE;  // 16368 bytes

pub const FILE_HEADER_SIZE: usize = 128;  // Page 0 only
pub const PAGE0_USABLE_SIZE: usize = PAGE_SIZE - FILE_HEADER_SIZE; // 16256 bytes

#[repr(C)]
#[derive(zerocopy::FromBytes, zerocopy::AsBytes)]
pub struct PageHeader {
    page_type: u8,
    flags: u8,
    cell_count: u16,
    free_start: u16,
    free_end: u16,
    frag_bytes: u8,
    reserved: [u8; 3],
    right_child: u32,
}
```

**Migration Notes**:
- Go's 12-byte header expands to 16 bytes (alignment + reserved space)
- All multi-byte fields use little-endian encoding
- `zerocopy` derives enable safe cast from `&[u8]`

### 1.3 Page Cache (SIEVE - Not LRU)

**Go Source**: `pkg/pager/pager.go` (LRU cache implementation)

**Critical Change**: Standard LRU is bad for databases - a sequential scan flushes the entire cache. Use **SIEVE** algorithm instead.

**Rust Implementation**:

```rust
// src/storage/cache.rs

const SHARD_COUNT: usize = 64;

pub struct PageCache {
    shards: [RwLock<CacheShard>; SHARD_COUNT],
}

struct CacheShard {
    entries: Vec<CacheEntry>,
    index: HashMap<PageKey, usize>,
    hand: usize,
}

struct CacheEntry {
    key: PageKey,
    visited: AtomicBool,
    dirty: bool,
    pin_count: u32,
    data: *mut [u8; PAGE_SIZE],
}

impl CacheShard {
    fn evict(&mut self) -> Option<PageKey> {
        loop {
            let entry = &self.entries[self.hand];
            if entry.pin_count > 0 {
                self.hand = (self.hand + 1) % self.entries.len();
                continue;
            }
            if entry.visited.swap(false, Ordering::Relaxed) {
                self.hand = (self.hand + 1) % self.entries.len();
            } else {
                return Some(entry.key);
            }
        }
    }

    fn access(&self, idx: usize) {
        self.entries[idx].visited.store(true, Ordering::Relaxed);
    }
}

impl PageCache {
    fn shard_for(&self, key: &PageKey) -> &RwLock<CacheShard> {
        let hash = (key.file_id as usize * 31 + key.page_no as usize) % SHARD_COUNT;
        &self.shards[hash]
    }
}
```

**Key Differences from Go**:
- **SIEVE** instead of LRU (scan-resistant)
- **Lock sharding** (64 shards) instead of single global lock
- Pre-allocate all page buffers at startup
- Atomic visited flag for lock-free access marking

### 1.4 Freelist Management

**Go Source**: `pkg/pager/freelist.go`

**Rust Implementation**:

```rust
// src/storage/freelist.rs

pub struct Freelist {
    head_page: u32,
    free_count: u32,
    pending: SmallVec<[u32; 64]>,
}

impl Freelist {
    pub fn allocate(&mut self) -> Result<u32>
    pub fn release(&mut self, page_no: u32)
    pub fn sync(&mut self, storage: &mut MmapStorage) -> Result<()>
}
```

**Trunk Page Format**:
```
Offset  Size  Description
0       4     Next trunk page (0 if last)
4       4     Number of entries in this trunk
8       N*4   Free page numbers
```

### 1.5 Multi-File Manager

**New Component** (not in Go):

```rust
// src/storage/file_manager.rs

pub struct FileManager {
    base_path: PathBuf,
    meta_file: MmapStorage,
    tables: HashMap<TableId, TableFiles>,
    open_count: usize,
    max_open: usize,
}

pub struct TableFiles {
    data: Option<MmapStorage>,
    indexes: HashMap<IndexId, MmapStorage>,
}

impl FileManager {
    pub fn create_table(&mut self, schema: &str, table: &str) -> Result<TableId>
    pub fn drop_table(&mut self, table_id: TableId) -> Result<()>
    pub fn table_data(&mut self, table_id: TableId) -> Result<&mut MmapStorage>
    pub fn table_index(&mut self, table_id: TableId, index_id: IndexId) -> Result<&mut MmapStorage>
}
```

**File Structure**:
```
database_dir/
├── turdb.meta                 # Global metadata
├── root/                      # Default schema
│   ├── users.tbd              # Table data
│   ├── users_pk.idx           # Primary key index
│   └── users_email.idx        # Secondary index
├── analytics/                 # Custom schema
│   └── events.tbd
└── wal/
    ├── wal.000001
    └── wal.000002
```

## Phase 2: B-Tree Implementation (Weeks 4-6)

### 2.1 Node Structure with Slot Array

**Go Source**: `pkg/btree/btree.go`, `pkg/btree/node.go`

**Critical Optimization**: Use a **slot array** with 4-byte prefix hints for fast key lookup without deserializing full keys.

**Rust Implementation**:

```rust
// src/btree/node.rs

pub const MAX_SLOTS: usize = 512;

#[repr(C)]
pub struct LeafPageHeader {
    page_type: u8,
    flags: u8,
    slot_count: u16,
    free_start: u16,
    free_end: u16,
    next_leaf: u32,
    prev_leaf: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct Slot {
    prefix: [u8; 4],
    offset: u16,
    key_len: u16,
}

pub struct LeafNode<'a> {
    page: &'a [u8],
    header: &'a LeafPageHeader,
    slots: &'a [Slot],
}

impl<'a> LeafNode<'a> {
    pub fn find_key(&self, key: &[u8]) -> SearchResult {
        let target_prefix = Self::extract_prefix(key);

        for (i, slot) in self.slots.iter().enumerate() {
            let slot_prefix = u32::from_be_bytes(slot.prefix);
            match slot_prefix.cmp(&target_prefix) {
                Ordering::Greater => return SearchResult::NotFound(i),
                Ordering::Equal => {
                    let full_key = self.key_at(slot);
                    match full_key.cmp(key) {
                        Ordering::Equal => return SearchResult::Found(i),
                        Ordering::Greater => return SearchResult::NotFound(i),
                        Ordering::Less => continue,
                    }
                }
                Ordering::Less => continue,
            }
        }
        SearchResult::NotFound(self.slots.len())
    }

    fn extract_prefix(key: &[u8]) -> u32 {
        let mut buf = [0u8; 4];
        let len = key.len().min(4);
        buf[..len].copy_from_slice(&key[..len]);
        u32::from_be_bytes(buf)
    }
}
```

**Zero-Copy Cell Access**:
- `LeafNode` borrows directly from page buffer
- Slot array enables prefix comparison without full key deserialization
- Full key comparison only when prefixes match

### 2.2 Comprehensive Type Prefix Key Encoding

**Critical**: All keys MUST be encoded in **big-endian byte-comparable** format for single `memcmp` comparison.
This encoding supports ALL SQL types including JSON, composite types, enums, arrays, and custom types.

#### Type Prefix Constants

```rust
// src/encoding/key.rs

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

#### Blob Encoding

```rust
fn encode_blob(b: &[u8], buf: &mut Vec<u8>) {
    buf.push(TypePrefix::BLOB);
    for byte in b {
        match byte {
            0x00 => { buf.push(0x00); buf.push(0xFF); }
            0xFF => { buf.push(0xFF); buf.push(0x00); }
            _ => buf.push(*byte),
        }
    }
    buf.push(0x00);
    buf.push(0x00);
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
            encode_text_body(s, buf); // Same as TEXT but without prefix
        }
        JsonValue::Array(arr) => {
            buf.push(TypePrefix::JSON_ARRAY);
            for elem in arr {
                encode_json(elem, buf);
                buf.push(0x01); // Element separator
            }
            buf.push(0x00); // Array terminator
        }
        JsonValue::Object(obj) => {
            buf.push(TypePrefix::JSON_OBJECT);
            let mut keys: Vec<_> = obj.keys().collect();
            keys.sort(); // Deterministic key order for comparison
            for key in keys {
                encode_text_body(key, buf);
                buf.push(0x02); // Key-value separator
                encode_json(&obj[key], buf);
                buf.push(0x01); // Entry separator
            }
            buf.push(0x00); // Object terminator
        }
    }
}
```

#### Composite/Custom Types

```rust
fn encode_composite(fields: &[Value], type_id: u32, buf: &mut Vec<u8>) {
    buf.push(TypePrefix::COMPOSITE);
    buf.extend(type_id.to_be_bytes()); // Type OID for disambiguation
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
        buf.push(0x01); // Element separator
    }
    buf.push(0x00); // Terminator
}
```

#### Date/Time Encoding

```rust
fn encode_date(days_since_epoch: i32, buf: &mut Vec<u8>) {
    buf.push(TypePrefix::DATE);
    buf.extend((days_since_epoch as u32 ^ 0x80000000).to_be_bytes()); // XOR flip for signed
}

fn encode_timestamp(micros_since_epoch: i64, buf: &mut Vec<u8>) {
    buf.push(TypePrefix::TIMESTAMP);
    buf.extend((micros_since_epoch as u64 ^ 0x8000000000000000).to_be_bytes());
}

fn encode_uuid(uuid: &[u8; 16], buf: &mut Vec<u8>) {
    buf.push(TypePrefix::UUID);
    buf.extend(uuid); // Already byte-comparable (big-endian by spec)
}
```

#### Type Ordering Summary

```
NULL < FALSE < TRUE
< -∞ < negative big int < negative int < negative float
< 0
< positive float < positive int < positive big int < +∞ < NaN
< TEXT < BLOB
< DATE < TIME < TIMESTAMP < TIMESTAMPTZ < INTERVAL
< UUID < INET < MACADDR
< JSON_NULL < JSON_FALSE < JSON_TRUE < JSON_NUMBER < JSON_STRING < JSON_ARRAY < JSON_OBJECT
< ARRAY < TUPLE < RANGE < ENUM < COMPOSITE < DOMAIN
< VECTOR
< CUSTOM (0x80-0xFE)
< MAX_KEY (0xFF)
```

**Page Layout**:
```
Leaf Page (16KB):
+------------------------+
| Header (16 bytes)      |
+------------------------+
| Slot Array             |
| (8 bytes × slot_count) |
+------------------------+
| Free Space             |
+------------------------+
| Cell Content           |
| (grows upward)         |
+------------------------+

Interior Page:
+------------------------+
| Header (16 bytes)      |
| Right Child (4 bytes)  |
+------------------------+
| Cell Pointers (2B each)|
+------------------------+
| Free Space             |
+------------------------+
| Separator Keys         |
| (suffix-truncated)     |
+------------------------+
```

### 2.3 Suffix Truncation for Interior Nodes

**Optimization**: Interior nodes only store enough key bytes to distinguish children.

```rust
impl InteriorNode {
    fn compute_separator(left_max: &[u8], right_min: &[u8]) -> Vec<u8> {
        let mut sep = Vec::with_capacity(8);
        for (l, r) in left_max.iter().zip(right_min.iter()) {
            if l < r {
                sep.push(*l + 1);
                return sep;
            }
            sep.push(*l);
        }
        if left_max.len() < right_min.len() {
            sep.push(right_min[left_max.len()]);
        }
        sep
    }
}
```

This reduces interior node size, allowing more keys per page and shallower trees.

### 2.4 Varint Encoding

**Varint Encoding** (SQLite-compatible):
- 1-9 bytes for values 0 to 2^64-1
- First byte determines length
- Zero-copy decode via lookup table

### 2.3 B-Tree Operations

**Insert**:
```rust
impl BTree {
    pub fn insert(&mut self, key: &[u8], value: &[u8]) -> Result<()> {
        let path = self.find_path(key)?;
        let leaf = path.leaf();

        if leaf.has_space(key.len() + value.len()) {
            leaf.insert_cell(key, value)?;
        } else {
            self.split_and_insert(path, key, value)?;
        }
        Ok(())
    }
}
```

**Search**:
```rust
impl BTree {
    pub fn get<'a>(&'a self, key: &[u8]) -> Result<Option<&'a [u8]>> {
        let node = self.root()?;
        loop {
            match node.search_key(key) {
                SearchResult::Found(cell) => return Ok(Some(cell.value())),
                SearchResult::NotFound => return Ok(None),
                SearchResult::Child(page_no) => node = self.node(page_no)?,
            }
        }
    }
}
```

### 2.7 Cursor Implementation with Prefetching

**Go Source**: `pkg/btree/cursor.go`

**Critical Optimization**: Use `madvise(MADV_WILLNEED)` to prefetch upcoming leaf pages during sequential scans.

**Rust Implementation**:

```rust
// src/btree/cursor.rs

pub struct Cursor<'a> {
    tree: &'a BTree,
    stack: SmallVec<[(u32, u16); 8]>,
    current_leaf: u32,
    current_slot: u16,
    prefetch_ahead: u32,
}

impl<'a> Cursor<'a> {
    pub fn seek(&mut self, key: &[u8]) -> Result<bool>
    pub fn seek_first(&mut self) -> Result<bool>
    pub fn seek_last(&mut self) -> Result<bool>

    pub fn next(&mut self) -> Result<bool> {
        self.current_slot += 1;

        let leaf = self.tree.leaf_node(self.current_leaf)?;
        if self.current_slot >= leaf.slot_count() {
            let next_leaf = leaf.next_leaf();
            if next_leaf == 0 {
                return Ok(false);
            }

            self.prefetch_pages(next_leaf)?;
            self.current_leaf = next_leaf;
            self.current_slot = 0;
        }
        Ok(true)
    }

    fn prefetch_pages(&self, start_page: u32) -> Result<()> {
        let pages_to_prefetch = [start_page + 1, start_page + 2];
        for &page_no in &pages_to_prefetch {
            if page_no < self.tree.page_count() {
                unsafe {
                    libc::madvise(
                        self.tree.page_ptr(page_no) as *mut libc::c_void,
                        PAGE_SIZE,
                        libc::MADV_WILLNEED,
                    );
                }
            }
        }
        Ok(())
    }

    pub fn key(&self) -> Option<&'a [u8]>
    pub fn value(&self) -> Option<&'a [u8]>
}
```

**Zero-Allocation Iteration**:
- `SmallVec` stores path on stack (up to 8 levels)
- Key/value are references, not copies
- Cursor reusable across multiple seeks
- **Prefetching** reduces mmap stalls during range scans

## Phase 3: Record Serialization (Week 7)

### 3.1 Record Format

**Go Source**: `pkg/record/record.go`

**Format** (SQLite-compatible):
```
[header_size: varint][type0: varint][type1: varint]...[data0][data1]...
```

**Serial Types**:
| Type | Description |
|------|-------------|
| 0 | NULL |
| 1 | 8-bit signed integer |
| 2 | 16-bit signed integer (BE) |
| 3 | 24-bit signed integer (BE) |
| 4 | 32-bit signed integer (BE) |
| 5 | 48-bit signed integer (BE) |
| 6 | 64-bit signed integer (BE) |
| 7 | 64-bit IEEE float (BE) |
| 8 | Integer 0 |
| 9 | Integer 1 |
| ≥12 even | BLOB of (N-12)/2 bytes |
| ≥13 odd | TEXT of (N-13)/2 bytes |

### 3.2 Zero-Copy Record Access

```rust
// src/record.rs

pub struct Record<'a> {
    data: &'a [u8],
    header_size: usize,
    column_types: SmallVec<[u32; 16]>,
    column_offsets: SmallVec<[usize; 16]>,
}

impl<'a> Record<'a> {
    pub fn parse(data: &'a [u8]) -> Result<Self>

    pub fn column_count(&self) -> usize

    pub fn is_null(&self, col: usize) -> bool

    pub fn get_int(&self, col: usize) -> Result<i64>

    pub fn get_float(&self, col: usize) -> Result<f64>

    pub fn get_text(&self, col: usize) -> Result<&'a str>

    pub fn get_blob(&self, col: usize) -> Result<&'a [u8]>
}
```

**Key Design**:
- `Record` borrows from page buffer
- Text/blob access returns slices, not copies
- Column offsets computed once on parse

### 3.3 Record Builder

```rust
pub struct RecordBuilder {
    buffer: Vec<u8>,
    header: Vec<u8>,
    data: Vec<u8>,
}

impl RecordBuilder {
    pub fn new() -> Self
    pub fn push_null(&mut self)
    pub fn push_int(&mut self, val: i64)
    pub fn push_float(&mut self, val: f64)
    pub fn push_text(&mut self, val: &str)
    pub fn push_blob(&mut self, val: &[u8])
    pub fn finish(&mut self) -> &[u8]
    pub fn reset(&mut self)
}
```

**Buffer Reuse**:
- `RecordBuilder` pre-allocates buffers
- `reset()` clears without deallocating
- Same builder used for all inserts in a transaction

## Phase 4: Schema and Catalog (Weeks 8-9)

### 4.1 Multi-Schema Support

**New Feature** (not in Go):

```rust
// src/schema/mod.rs

pub struct Catalog {
    schemas: HashMap<String, Schema>,
    default_schema: String,
}

pub struct Schema {
    id: SchemaId,
    name: String,
    tables: HashMap<String, TableDef>,
}

pub struct TableDef {
    id: TableId,
    schema_id: SchemaId,
    name: String,
    columns: Vec<ColumnDef>,
    primary_key: Option<PrimaryKey>,
    indexes: Vec<IndexDef>,
    constraints: Vec<Constraint>,
}
```

**Schema Resolution**:
```rust
impl Catalog {
    pub fn resolve_table(&self, name: &str) -> Result<&TableDef> {
        if let Some((schema, table)) = name.split_once('.') {
            self.schemas.get(schema)
                .and_then(|s| s.tables.get(table))
                .ok_or_else(|| eyre!("table '{}' not found", name))
        } else {
            // Default schema is "root"
            self.schemas.get(&self.default_schema)
                .and_then(|s| s.tables.get(name))
                .ok_or_else(|| eyre!("table '{}' not found in default schema", name))
        }
    }
}
```

### 4.2 Catalog Persistence

**Metadata File Format** (`turdb.meta`):
```
Page 0: File Header
  - Magic, version, page size, schema count

Page 1+: Catalog Pages
  - Schema definitions
  - Table definitions (serialized)
  - Index definitions
  - Statistics
```

**Lazy Loading**:
- Load only schema names at startup
- Load full table definition on first access
- Cache loaded definitions in memory

### 4.3 Column Types

```rust
// src/types.rs

pub enum TypeAffinity {
    Integer,
    Real,
    Text,
    Blob,
    Numeric,
}

pub enum DataType {
    Null,
    Boolean,
    TinyInt,
    SmallInt,
    Int,
    BigInt,
    Float,
    Double,
    Decimal { precision: u8, scale: u8 },
    Char(u16),
    Varchar(u16),
    Text,
    Blob,
    Date,
    Time,
    Timestamp,
    Vector(u32),
}
```

## Phase 5: SQL Processing (Weeks 10-13)

### 5.1 Lexer

**Go Source**: `pkg/sql/lexer/lexer.go`

```rust
// src/sql/lexer.rs

pub struct Lexer<'a> {
    input: &'a str,
    pos: usize,
}

pub enum Token<'a> {
    Ident(&'a str),
    Number(&'a str),
    String(&'a str),
    Keyword(Keyword),
    Symbol(Symbol),
    Eof,
}
```

**Zero-Copy Tokens**:
- Identifiers and strings are slices into input
- No string allocation during lexing
- Keywords matched via perfect hash

### 5.2 Parser

**Go Source**: `pkg/sql/parser/parser.go`

```rust
// src/sql/parser.rs

pub struct Parser<'a> {
    lexer: Lexer<'a>,
    arena: &'a Bump,
}

impl<'a> Parser<'a> {
    pub fn parse_statement(&mut self) -> Result<Statement<'a>>
}

pub enum Statement<'a> {
    Select(SelectStmt<'a>),
    Insert(InsertStmt<'a>),
    Update(UpdateStmt<'a>),
    Delete(DeleteStmt<'a>),
    CreateTable(CreateTableStmt<'a>),
    CreateIndex(CreateIndexStmt<'a>),
    CreateSchema(CreateSchemaStmt<'a>),
}
```

**Arena Allocation**:
- All AST nodes allocated in `bumpalo` arena
- Arena cleared after query completes
- No per-node heap allocation

### 5.3 Query Planner

**Go Source**: `pkg/sql/optimizer/optimizer.go`

```rust
// src/sql/planner.rs

pub struct Planner<'a> {
    catalog: &'a Catalog,
    arena: &'a Bump,
}

pub enum PlanNode<'a> {
    Scan(ScanPlan<'a>),
    IndexScan(IndexScanPlan<'a>),
    Filter(FilterPlan<'a>),
    Project(ProjectPlan<'a>),
    Join(JoinPlan<'a>),
    Sort(SortPlan<'a>),
    Limit(LimitPlan<'a>),
    Aggregate(AggregatePlan<'a>),
}
```

### 5.4 Executor

**Go Source**: `pkg/sql/executor/*.go`, `pkg/vdbe/*.go`

```rust
// src/sql/executor.rs

pub trait Executor {
    fn open(&mut self) -> Result<()>;
    fn next(&mut self) -> Result<Option<Row>>;
    fn close(&mut self) -> Result<()>;
}

pub struct ScanExecutor<'a> {
    cursor: Cursor<'a>,
    columns: &'a [usize],
}

pub struct FilterExecutor<'a> {
    input: Box<dyn Executor + 'a>,
    predicate: &'a Expr<'a>,
}
```

**Volcano Model**:
- Pull-based execution
- Each operator implements `Executor` trait
- Rows streamed, not buffered

### 5.5 Grace Hash Join for Memory-Constrained Joins

**Critical**: With 256KB working memory, use partitioned hash join instead of naive hash join.

```rust
// src/sql/executor/join.rs

pub struct GraceHashJoin<'a> {
    build_side: Box<dyn Executor + 'a>,
    probe_side: Box<dyn Executor + 'a>,
    join_key: &'a [usize],
    partitions: Vec<TempFile>,
    partition_count: usize,
    memory_budget: usize,
}

impl<'a> GraceHashJoin<'a> {
    pub fn execute(&mut self) -> Result<impl Iterator<Item = Row> + 'a> {
        self.partition_build_side()?;
        self.partition_probe_side()?;

        let mut results = Vec::new();
        for partition_id in 0..self.partition_count {
            let build_rows = self.partitions[partition_id].read_all()?;
            let hash_table = self.build_in_memory_hash_table(&build_rows);

            let probe_partition = &self.partitions[self.partition_count + partition_id];
            for probe_row in probe_partition.iter() {
                if let Some(build_row) = hash_table.probe(&probe_row) {
                    results.push(self.join_rows(&build_row, &probe_row));
                }
            }
        }
        Ok(results.into_iter())
    }

    fn partition_build_side(&mut self) -> Result<()> {
        while let Some(row) = self.build_side.next()? {
            let hash = self.hash_key(&row);
            let partition_id = hash as usize % self.partition_count;
            self.partitions[partition_id].write(&row)?;
        }
        Ok(())
    }
}
```

**Partition Count Selection**:
- `partition_count = max(16, build_size / memory_budget)`
- Each partition should fit in working memory

## Phase 6: HNSW Vector Index (Weeks 14-16)

### 6.1 Graph Structure

**Go Source**: `pkg/hnsw/hnsw.go`

**Low-RAM Consideration**: HNSW requires the graph in memory for speed. For datasets exceeding RAM budget, consider **DiskANN/Vamana** as an alternative (keeps vectors on disk, caches entry points).

```rust
// src/hnsw/graph.rs

pub struct HnswIndex {
    dimension: u32,
    m: u32,
    m_max0: u32,
    ef_construction: u32,
    ef_search: u32,
    entry_point: Option<NodeId>,
    max_level: u8,
    storage: MmapStorage,
    quantization: QuantizationType,
}

pub enum QuantizationType {
    None,
    SQ8,
    PQ { m: u8, nbits: u8 },
}

pub struct HnswNode<'a> {
    id: NodeId,
    row_id: u64,
    level: u8,
    vector: VectorRef<'a>,
    neighbors: SmallVec<[&'a [NodeId]; 8]>,
}

pub enum VectorRef<'a> {
    F32(&'a [f32]),
    SQ8(&'a SQ8Vector),
}
```

### 6.2 Distance Functions with SIMD

**Critical Optimization**: Use AVX2/NEON SIMD for distance calculations.

```rust
// src/hnsw/distance.rs

pub trait Distance: Send + Sync {
    fn compute(&self, a: &[f32], b: &[f32]) -> f32;
    fn compute_sq8(&self, a: &SQ8Vector, b: &SQ8Vector) -> f32;
}

pub struct Euclidean;
pub struct Cosine;
pub struct InnerProduct;

#[cfg(target_arch = "x86_64")]
pub fn euclidean_avx2(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::x86_64::*;

    unsafe {
        let mut sum = _mm256_setzero_ps();
        let chunks = a.len() / 8;

        for i in 0..chunks {
            let offset = i * 8;
            let va = _mm256_loadu_ps(a.as_ptr().add(offset));
            let vb = _mm256_loadu_ps(b.as_ptr().add(offset));
            let diff = _mm256_sub_ps(va, vb);
            sum = _mm256_fmadd_ps(diff, diff, sum);
        }

        let mut result = [0f32; 8];
        _mm256_storeu_ps(result.as_mut_ptr(), sum);
        result.iter().sum::<f32>().sqrt()
    }
}

#[cfg(target_arch = "aarch64")]
pub fn euclidean_neon(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::aarch64::*;

    unsafe {
        let mut sum = vdupq_n_f32(0.0);
        let chunks = a.len() / 4;

        for i in 0..chunks {
            let offset = i * 4;
            let va = vld1q_f32(a.as_ptr().add(offset));
            let vb = vld1q_f32(b.as_ptr().add(offset));
            let diff = vsubq_f32(va, vb);
            sum = vfmaq_f32(sum, diff, diff);
        }

        vaddvq_f32(sum).sqrt()
    }
}
```

### 6.3 Scalar Quantization (SQ8)

**4x Memory Reduction**: Compress float32 vectors to uint8.

```rust
// src/hnsw/quantize.rs

#[repr(C)]
pub struct SQ8Vector {
    min: f32,
    scale: f32,
    data: [u8],
}

impl SQ8Vector {
    pub fn encode(vec: &[f32], buf: &mut [u8]) -> (f32, f32) {
        let min = vec.iter().cloned().fold(f32::INFINITY, f32::min);
        let max = vec.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let scale = (max - min) / 255.0;

        for (i, &v) in vec.iter().enumerate() {
            buf[i] = ((v - min) / scale) as u8;
        }
        (min, scale)
    }

    pub fn distance_l2(&self, other: &Self) -> f32 {
        let mut sum: u32 = 0;
        for (a, b) in self.data.iter().zip(&other.data) {
            let diff = (*a as i32) - (*b as i32);
            sum += (diff * diff) as u32;
        }
        (sum as f32) * self.scale * other.scale
    }
}

#[cfg(target_arch = "x86_64")]
pub fn sq8_distance_avx2(a: &[u8], b: &[u8]) -> u32 {
    use std::arch::x86_64::*;

    unsafe {
        let mut sum = _mm256_setzero_si256();
        for i in (0..a.len()).step_by(32) {
            let va = _mm256_loadu_si256(a.as_ptr().add(i) as *const __m256i);
            let vb = _mm256_loadu_si256(b.as_ptr().add(i) as *const __m256i);
            let sad = _mm256_sad_epu8(va, vb);
            sum = _mm256_add_epi64(sum, sad);
        }
        _mm256_extract_epi64(sum, 0) as u32 + _mm256_extract_epi64(sum, 2) as u32
    }
}
```

### 6.4 Search Algorithm

```rust
// src/hnsw/search.rs

impl HnswIndex {
    pub fn search(&self, query: &[f32], k: usize, ef: usize) -> Result<Vec<(NodeId, f32)>> {
        let query_sq8 = self.quantize_query(query);
        let mut current = self.entry_point.ok_or_else(|| eyre!("empty index"))?;

        for level in (1..=self.max_level).rev() {
            current = self.greedy_search(&query_sq8, current, level)?;
        }

        let candidates = self.beam_search(&query_sq8, current, 0, ef)?;

        if self.quantization != QuantizationType::None {
            self.rerank_with_full_vectors(query, candidates, k)
        } else {
            Ok(candidates.into_iter().take(k).collect())
        }
    }
}
```

## Phase 7: Transactions and MVCC (Weeks 17-19)

### 7.1 Transaction Manager

**Go Source**: `pkg/mvcc/mvcc.go`

```rust
// src/transaction/manager.rs

pub struct TransactionManager {
    next_txn_id: AtomicU64,
    active: RwLock<HashMap<TxnId, TxnState>>,
    oldest_active: AtomicU64,
}

pub struct Transaction {
    id: TxnId,
    start_ts: u64,
    commit_ts: Option<u64>,
    state: TxnState,
    write_set: Vec<WriteEntry>,
    read_set: Vec<ReadEntry>,
}
```

### 7.2 Version Chains

```rust
// src/transaction/version.rs

pub struct VersionChain {
    latest: AtomicPtr<RowVersion>,
}

pub struct RowVersion {
    txn_id: TxnId,
    commit_ts: u64,
    data: Box<[u8]>,
    prev: Option<Box<RowVersion>>,
}

impl VersionChain {
    pub fn read(&self, snapshot_ts: u64) -> Option<&[u8]> {
        let mut version = unsafe { self.latest.load(Ordering::Acquire).as_ref() };
        while let Some(v) = version {
            if v.commit_ts <= snapshot_ts {
                return Some(&v.data);
            }
            version = v.prev.as_ref().map(|b| b.as_ref());
        }
        None
    }
}
```

### 7.3 Conflict Detection

```rust
impl Transaction {
    pub fn commit(&mut self, manager: &TransactionManager) -> Result<()> {
        let commit_ts = manager.allocate_timestamp();

        for write in &self.write_set {
            if write.chain.has_newer_version(self.start_ts) {
                self.rollback()?;
                bail!("write-write conflict detected");
            }
        }

        for write in &self.write_set {
            write.chain.install_version(self.id, commit_ts, &write.data);
        }

        self.commit_ts = Some(commit_ts);
        self.state = TxnState::Committed;
        Ok(())
    }
}
```

## Phase 8: Write-Ahead Log (Week 20)

### 8.1 WAL Format

**Go Source**: `pkg/wal/wal.go`

```rust
// src/storage/wal.rs

pub struct Wal {
    dir: PathBuf,
    current_segment: WalSegment,
    segment_size: u64,
}

pub struct WalSegment {
    file: File,
    sequence: u64,
    offset: u64,
}

#[repr(C)]
#[derive(zerocopy::FromBytes, zerocopy::AsBytes)]
pub struct WalFrameHeader {
    page_no: u32,
    db_size: u32,
    salt1: u32,
    salt2: u32,
    checksum: u64,
}
```

### 8.2 Checkpointing

```rust
impl Wal {
    pub fn checkpoint(&mut self, pager: &mut Pager) -> Result<()> {
        for frame in self.frames() {
            let page = pager.page_mut(frame.page_no)?;
            page.copy_from_slice(frame.data());
        }
        self.truncate()?;
        Ok(())
    }
}
```

## Phase 9: Public API (Week 21)

### 9.1 Database Handle

```rust
// src/lib.rs

pub struct Database {
    file_manager: FileManager,
    catalog: Catalog,
    txn_manager: TransactionManager,
    wal: Wal,
    options: Options,
}

impl Database {
    pub fn open(path: &Path) -> Result<Self>
    pub fn open_with_options(path: &Path, options: Options) -> Result<Self>
    pub fn close(self) -> Result<()>
    pub fn transaction(&self) -> Result<Transaction>
    pub fn query(&self, sql: &str) -> QueryBuilder
    pub fn execute(&self, sql: &str) -> Result<u64>
}
```

### 9.2 Query Builder

```rust
pub struct QueryBuilder<'db> {
    db: &'db Database,
    sql: String,
    params: Vec<Value>,
}

impl<'db> QueryBuilder<'db> {
    pub fn bind<T: Into<Value>>(mut self, value: T) -> Self
    pub fn fetch_one(self) -> Result<Row>
    pub fn fetch_all(self) -> Result<Vec<Row>>
    pub fn fetch_iter(self) -> Result<RowIter<'db>>
}
```

### 9.3 Row Access

```rust
pub struct Row {
    columns: Vec<Value>,
    column_names: Arc<[String]>,
}

impl Row {
    pub fn get<T: FromValue>(&self, index: usize) -> Result<T>
    pub fn get_by_name<T: FromValue>(&self, name: &str) -> Result<T>
    pub fn try_get<T: FromValue>(&self, index: usize) -> Option<T>
}
```

## Memory Budget Analysis

### 1MB Startup Budget

| Component | Memory | Notes |
|-----------|--------|-------|
| Page Cache | 512KB | 32 × 16KB pages |
| Catalog (lazy) | 64KB | Schema names only |
| File Manager | 32KB | File handles |
| Transaction Manager | 32KB | Active txn tracking |
| WAL Buffer | 64KB | 4 × 16KB frames |
| Working Memory | 64KB | Cursor stack, etc. |
| **Total** | **768KB** | Under 1MB |

### Per-Query Budget

- Arena: 256KB default, grows if needed
- Result buffer: 64KB, streams if larger
- Sort buffer: 128KB, spills to disk if larger

## Testing Strategy

### Unit Tests

```rust
#[cfg(test)]
mod tests {
    #[test]
    fn btree_insert_split() { }

    #[test]
    fn record_roundtrip() { }

    #[test]
    fn mvcc_snapshot_isolation() { }
}
```

### Integration Tests

```rust
#[test]
fn crud_cycle() {
    let db = Database::open(":memory:")?;
    db.execute("CREATE TABLE t (id INT PRIMARY KEY, name TEXT)")?;
    db.execute("INSERT INTO t VALUES (1, 'hello')")?;
    let row = db.query("SELECT * FROM t WHERE id = ?").bind(1).fetch_one()?;
    assert_eq!(row.get::<i64>(0)?, 1);
}
```

### Memory Tests

```rust
#[test]
fn operates_under_1mb() {
    let budget = MemoryBudget::new(1024 * 1024);
    let db = Database::builder()
        .memory_budget(budget)
        .open(":memory:")?;
    // Perform operations
    assert!(budget.current() < 1024 * 1024);
}
```

### Fuzz Tests

```rust
#![no_main]
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    if let Ok(sql) = std::str::from_utf8(data) {
        let _ = turdb::sql::parse(sql);
    }
});
```

## Migration Tools

### Data Export (Go)

```go
func ExportToSQL(db *turdb.DB, w io.Writer) error {
    // Export schema
    for _, table := range db.Tables() {
        fmt.Fprintf(w, "CREATE TABLE %s (...);", table.Name)
    }
    // Export data
    for _, table := range db.Tables() {
        rows, _ := db.Query("SELECT * FROM " + table.Name)
        for rows.Next() {
            // Write INSERT statements
        }
    }
}
```

### Data Import (Rust)

```rust
impl Database {
    pub fn import_sql(&mut self, reader: impl BufRead) -> Result<ImportStats> {
        let mut stats = ImportStats::default();
        for line in reader.lines() {
            let sql = line?;
            match self.execute(&sql) {
                Ok(rows) => stats.rows += rows,
                Err(e) => stats.errors.push(e),
            }
        }
        Ok(stats)
    }
}
```

## Risk Assessment

### High Risk

1. **Zero-copy lifetime management**: Complex borrowing across page boundaries
   - Mitigation: Extensive testing, careful API design

2. **SIMD distance functions**: Platform-specific, unsafe code
   - Mitigation: Scalar fallback, runtime detection

3. **Memory-mapped I/O edge cases**: Signal handling, file truncation
   - Mitigation: Guard pages, careful error handling

### Medium Risk

1. **Multi-file consistency**: Crash during multi-file transaction
   - Mitigation: WAL covers all files atomically

2. **Schema migration complexity**: Multiple schemas increase state
   - Mitigation: Thorough catalog persistence testing

### Low Risk

1. **SQL compatibility**: Syntax differences from Go version
   - Mitigation: Comprehensive SQL test suite

## Timeline Summary

| Phase | Duration | Deliverables |
|-------|----------|--------------|
| 1. Storage | 3 weeks | mmap, pager, cache, freelist, file manager |
| 2. B-Tree | 3 weeks | Node ops, cursor, insert/delete/search |
| 3. Records | 1 week | Serialization, zero-copy access |
| 4. Schema | 2 weeks | Catalog, multi-schema, persistence |
| 5. SQL | 4 weeks | Lexer, parser, planner, executor |
| 6. HNSW | 3 weeks | Graph, distance, search, persistence |
| 7. MVCC | 3 weeks | Transactions, versions, conflicts |
| 8. WAL | 1 week | Logging, checkpointing, recovery |
| 9. API | 1 week | Public interface, builders |

**Total: ~21 weeks**

## Algorithmic Optimizations Summary

| Component | Go Approach | Rust Optimization | Benefit |
|-----------|-------------|-------------------|---------|
| Page Cache | LRU | SIEVE + Lock Sharding | Scan-resistant, 64x less lock contention |
| B+Tree Search | Binary search | Slot array with 4-byte prefix | Fewer comparisons, cache-friendly |
| Interior Keys | Full keys | Suffix truncation | More keys per page, shallower tree |
| Key Encoding | Mixed endian | Big-endian byte-comparable | Single memcmp for multi-column keys |
| Range Scans | Sequential | madvise prefetching | Reduced mmap stalls |
| Joins | Simple hash | Grace Hash Join | Works within memory budget |
| Vector Distance | Scalar | AVX2/NEON SIMD | 4-8x faster distance computation |
| Vector Storage | float32 | SQ8 quantization | 4x memory reduction |

## Tokio Decision

**Recommendation: Avoid Tokio for core engine.**

| Factor | Analysis |
|--------|----------|
| mmap page faults | Synchronous regardless of async runtime |
| Embedded use case | No network I/O in hot path |
| Latency overhead | ~100-500ns per await point |
| Complexity | Lifetime issues with borrowed page data |

**Alternative Architecture:**
```
turdb (core, sync)
├── All storage, btree, hnsw, mvcc
└── Pure Rust, no runtime dependency

turdb-server (optional, async)
├── Network protocol (PostgreSQL wire)
├── Connection pooling
└── Uses spawn_blocking for DB operations
```

## Success Criteria

1. All Go tests pass equivalent Rust tests
2. Benchmarks show equal or better performance
3. Memory usage stays under configured budget
4. Zero-copy verified via profiling (no unexpected allocations)
5. Crash recovery works correctly
6. Multi-schema operations work correctly
7. File-per-table operations work correctly
8. **Point read < 1µs (cached)**
9. **Sequential scan > 1M rows/sec**
10. **k-NN search (1M vectors, k=10) < 10ms**
11. **Startup with 1MB RAM budget**
