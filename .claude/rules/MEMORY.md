# Memory Rules - TurDB

> **THIS FILE IS MANDATORY READING BEFORE ANY MEMORY/PAGE WORK**

## Core Principle: Zero-Copy, Zero-Allocation

TurDB is designed for embedded/IoT devices. Memory efficiency is not optional.

---

## FORBIDDEN Patterns

### 1. Returning Owned Data from Page Access
```rust
// FORBIDDEN - copies data
fn get_key(&self, page: &Page) -> Vec<u8> {
    page.data()[self.key_offset..self.key_end].to_vec()
}

// CORRECT - zero-copy reference
fn get_key(&self, page: &Page) -> &[u8] {
    &page.data()[self.key_offset..self.key_end]
}
```

### 2. Cloning Page Data
```rust
// FORBIDDEN
let data = page.data().to_vec();
let key = row.key.clone();

// CORRECT - use references
let data: &[u8] = page.data();
let key: &[u8] = row.key();
```

### 3. Allocating in CRUD Paths
```rust
// FORBIDDEN - allocates new cursor each call
fn scan(&self) -> Result<Cursor> {
    Ok(Cursor::new())
}

// CORRECT - reuse pre-allocated cursor
fn scan(&self, cursor: &mut Cursor) -> Result<()> {
    cursor.reset();
    // ...
}
```

### 4. Single Global Lock
```rust
// FORBIDDEN - contention disaster
struct PageCache {
    pages: RwLock<HashMap<u32, Page>>,
}

// CORRECT - lock sharding (64 minimum)
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

---

## Page Layout

### Constants
```rust
const PAGE_SIZE: usize = 16384;      // 16KB
const PAGE_HEADER_SIZE: usize = 16;  // Every page
const FILE_HEADER_SIZE: usize = 128; // Page 0 only
const USABLE_SPACE: usize = 16368;   // PAGE_SIZE - PAGE_HEADER_SIZE
const PAGE0_USABLE: usize = 16256;   // PAGE_SIZE - FILE_HEADER_SIZE
```

### Visual Layout
```
Regular Page (1+):          Page 0 (with file header):
+------------------+        +------------------+
| Page Hdr (16B)   |        | File Hdr (128B)  |
+------------------+        +------------------+
| Usable (16368B)  |        | Usable (16256B)  |
+------------------+        +------------------+
```

---

## Memory Budget System

### Hard Limits (Not Advisory)
```rust
let budget = MemoryBudget::auto_detect();  // 25% RAM, 4MB floor
let budget = MemoryBudget::with_limit(16 * 1024 * 1024);  // Explicit 16MB
```

### Pool Allocation
| Pool | Size | Purpose |
|------|------|---------|
| Cache | 512 KB | Page cache (guaranteed minimum) |
| Query | 256 KB | Sort buffers, hash tables |
| Recovery | 256 KB | WAL frame processing |
| Schema | 128 KB | Catalog metadata |
| Shared | Remainder | Dynamic allocation |

### API Usage
```rust
// Allocate - returns Err if exceeds budget
budget.allocate(Pool::Cache, PAGE_SIZE)?;

// Release when done
budget.release(Pool::Cache, PAGE_SIZE);

// Check without committing
if budget.can_allocate(Pool::Cache, bytes) {
    // Safe to proceed
}
```

### PRAGMA Commands
- `PRAGMA memory_budget` - Query total budget
- `PRAGMA memory_stats` - Per-pool usage

---

## Mmap Safety via Borrow Checker

Rust's borrow checker prevents use-after-unmap at compile time:

```rust
impl MmapStorage {
    /// Borrows &self - prevents grow() while borrowed
    pub fn page(&self, page_no: u32) -> Result<&[u8]> {
        ensure!(page_no < self.page_count, "page out of bounds");
        let offset = page_no as usize * self.page_size;
        Ok(&self.mmap[offset..offset + self.page_size])
    }

    /// Requires &mut self - impossible if any page borrowed
    pub fn grow(&mut self, new_page_count: u32) -> Result<()> {
        // Remaps, invalidating old references
        self.mmap = unsafe { MmapMut::map_mut(&self.file)? };
        self.page_count = new_page_count;
        Ok(())
    }
}
```

**Compile-time enforcement:**
```rust
// This compiles - reference dropped before grow
let val = {
    let page = storage.page(5)?;
    page[0]
};
storage.grow(100)?;  // OK

// This FAILS to compile
let page = storage.page(5)?;
storage.grow(100)?;  // ERROR: cannot borrow as mutable
let byte = page[0];  // because borrowed as immutable
```

---

## Cache Eviction: SIEVE Algorithm

LRU is bad for database scans (flushes entire cache). Use SIEVE:

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
                // Recently visited, give second chance
                self.hand = (self.hand + 1) % self.entries.len();
            } else {
                // Not visited, evict this one
                return entry.page_no;
            }
        }
    }

    fn access(&self, idx: usize) {
        self.entries[idx].visited.store(true, Ordering::Relaxed);
    }
}
```

---

## Arena Allocators for Queries

Use `bumpalo` for per-query allocations:

```rust
use bumpalo::Bump;

fn execute_query<'a>(&self, arena: &'a Bump, sql: &str) -> Result<Rows<'a>> {
    // All temporary allocations go to arena
    // Arena freed when query completes
}
```

---

## No Global State

Never use global state or singleton patterns. All state flows through explicit parameters.

---

## Drop Order

Implement `Drop` carefully:
1. Dirty pages flushed before pager drops
2. WAL synced before file closes
3. Locks released in reverse acquisition order

---

## Range Scan Prefetching

Prefetch ahead during sequential scans:

```rust
impl Cursor {
    fn next(&mut self) -> Result<bool> {
        if self.at_page_end() {
            let next_page = self.current_leaf.next_leaf;
            let prefetch_page = next_page + 2;

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

---

## Quick Checklist

Before any memory-related code:

- [ ] Returning `&[u8]` not `Vec<u8>`?
- [ ] No `.to_vec()` or `.clone()` on page data?
- [ ] Reusing buffers, not allocating new ones?
- [ ] Using lock sharding (64 shards)?
- [ ] Using SIEVE, not LRU?
- [ ] Respecting memory budget pools?
