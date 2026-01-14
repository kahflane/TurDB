# Code Style Rules - TurDB

> **THIS FILE IS MANDATORY READING BEFORE WRITING ANY CODE**

## File Structure

### Maximum Length
- **Hard limit: 1800 lines per file**
- If approaching limit, split into submodules

### Modularity
- Prefer many small modules over one large module
- **"One Struct Rule"**: Major structs get their own file
- Use `mod.rs` only for `pub mod` and `pub use` re-exports

### Test Placement
- Unit tests (`#[cfg(test)]`) at file bottom
- If tests exceed 200 lines, move to:
  - `tests/` directory (integration style), or
  - Separate `_tests.rs` submodule

---

## Documentation

### FORBIDDEN: Inline Comments

```rust
// FORBIDDEN - never do this
fn insert(&mut self, key: &[u8], value: &[u8]) -> Result<()> {
    // Find the right leaf page
    let leaf = self.find_leaf(key)?;
    // Insert into the leaf
    leaf.insert(key, value)?;
    Ok(())
}

// CORRECT - no inline comments, code is self-documenting
fn insert(&mut self, key: &[u8], value: &[u8]) -> Result<()> {
    let leaf = self.find_leaf(key)?;
    leaf.insert(key, value)
}
```

### REQUIRED: Block Documentation at File Top

Each file MUST start with 80-100 lines of module documentation:

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
//! ## Key Invariants
//!
//! 1. All keys in left subtree < separator key < all keys in right subtree
//! 2. Leaf pages are linked for sequential scan
//! 3. Page splits happen bottom-up
//!
//! ## Usage Example
//!
//! ```rust
//! let tree = BTree::open(&storage, root_page)?;
//! tree.insert(b"key", b"value")?;
//! let value = tree.get(b"key")?;
//! ```
//!
//! ## Performance Characteristics
//!
//! - Point lookup: O(log n) page accesses
//! - Sequential scan: O(n/page_capacity) page accesses
//! - Insert: O(log n) average, O(log n * page_capacity) worst case
//!
//! ## Safety Considerations
//!
//! - All page accesses go through the buffer pool
//! - Mmap safety enforced by borrow checker
//! - Concurrent access protected by page-level locks
//!
//! ... (continue to 80-100 lines)
```

### Doc Comments for Public API Only

Use `///` only for public functions, structs, and traits. Keep concise (1-3 lines):

```rust
/// Opens a database at the specified path.
pub fn open(path: &Path) -> Result<Database>

/// Returns the number of rows in this table.
pub fn row_count(&self) -> u64
```

---

## Visibility

### Rules
1. Minimize `pub` usage
2. Use `pub(crate)` for internal APIs
3. Only `pub` for user-facing API
4. Never `pub use` internal modules at crate root

```rust
// CORRECT
pub struct Database { ... }           // User-facing
pub(crate) struct PageCache { ... }   // Internal
struct CacheEntry { ... }             // Module-private

// FORBIDDEN
pub use internal::implementation::details::*;
```

---

## Naming Conventions

### Types
```rust
// Structs: PascalCase
struct PageHeader { ... }

// Traits: PascalCase, often adjective
trait Encodable { ... }

// Enums: PascalCase, variants PascalCase
enum NodeType {
    Leaf,
    Interior,
}
```

### Functions and Variables
```rust
// Functions: snake_case
fn find_leaf_page() { ... }

// Variables: snake_case
let page_count = 0;
let root_page_no = 1;
```

### Constants
```rust
// Constants: SCREAMING_SNAKE_CASE
const PAGE_SIZE: usize = 16384;
const MAX_KEY_SIZE: usize = 1024;
```

---

## Error Handling

See `.claude/rules/ERRORS.md` for full details.

Quick reference:
- Use `eyre` exclusively
- No custom error enums
- Rich context on every error

---

## Unsafe Code

### When Allowed
1. mmap operations (inherently unsafe)
2. Zero-copy transmutation with `zerocopy`
3. SIMD intrinsics for vector distance
4. Custom allocator implementations

### Requirements

Every `unsafe` block MUST have:
1. `// SAFETY:` comment explaining why it's safe
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

---

## Concurrency

### Send + Sync Requirements
- `Database`: `Send + Sync` (shared across threads)
- `Transaction`: `Send + !Sync` (bound to one thread, movable)
- `Cursor`: `!Send + !Sync` (thread-local only)

### Synchronization
Use `parking_lot` for all synchronization:
- Sharded `RwLock` for page cache (64 shards)
- `RwLock` for catalog
- `Mutex` for WAL, dirty list

---

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

### RAII for Resources
- `Database` closes files on drop
- `Transaction` rolls back if not committed
- `Cursor` releases page pins on drop

---

## Dependencies

### Allowed
```toml
eyre = "0.6"
parking_lot = "0.12"
memmap2 = "0.9"
zerocopy = "0.7"
bumpalo = "3.14"
smallvec = "1.11"
hashbrown = "0.14"
```

### Forbidden
- `tokio`, `async-std` - No async runtime
- `serde` - Use custom zero-copy serialization
- `regex` - Too heavy
- Any dep > 50KB without justification

---

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

---

## Checklist

Before any code:

- [ ] File under 1800 lines?
- [ ] No inline comments?
- [ ] 80-100 line module doc at top?
- [ ] Using `pub(crate)` not `pub` where possible?
- [ ] `// SAFETY:` comment on every unsafe block?
- [ ] Using `parking_lot`, not `std::sync`?
- [ ] No forbidden dependencies?
