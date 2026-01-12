# TurDB Development Guidelines

> **STOP. READ THIS ENTIRE FILE BEFORE ANY ACTION.**
> These rules are NON-NEGOTIABLE. Violations = rejected code.

## Pre-Action Checklist (MANDATORY)

Before writing ANY code, confirm:

- [ ] **Zero-copy?** Returning `&[u8]`, never `Vec<u8>` or `.to_vec()`
- [ ] **Zero-alloc?** Reusing buffers, not allocating in CRUD paths
- [ ] **eyre errors?** Using `bail!`/`ensure!` with rich context, no custom enums
- [ ] **TDD?** Test written FIRST, test FAILS before implementation exists
- [ ] **Expected values?** Independently computed, not from the function being tested
- [ ] **No inline comments?** Documentation at file top only (80-100 lines)

---

## FORBIDDEN - INSTANT REJECTION

These patterns are NEVER acceptable:

### Memory/Performance
1. `Vec<u8>` returns from page access - use `&[u8]`
2. `.to_vec()` or `.clone()` on page data
3. Allocating in CRUD hot paths
4. Single global lock for page cache - use sharding (64 shards)
5. Async runtime (`tokio`, `async-std`)

### Testing
6. `assert!(result.is_ok())` - must check actual contents
7. `assert_eq!(f(x), f(x))` - tautology, tests nothing
8. Tests derived from reading implementation
9. Tests without specific expected values
10. Vague test names like `test_insert` or `test_1`

### Code Style
11. Inline comments anywhere in code
12. Custom error enums - use `eyre` only
13. Error messages without context (operation, resource, reason)
14. `pub` when `pub(crate)` suffices
15. Files over 800 lines

### Dependencies
16. `serde` - use custom zero-copy serialization
17. `regex` - too heavy
18. Any dep > 50KB without justification

---

## Project Overview

TurDB is an embedded database combining SQLite-inspired row storage with native HNSW vector search. Priorities:

1. **Zero-copy** - `&[u8]` slices into mmap'd regions
2. **Zero-allocation** - No heap allocs during CRUD
3. **16KB pages** - Fixed size, 16-byte headers
4. **Hard memory budget** - 25% RAM, 4MB floor

---

## Quick Reference

| Aspect | Rule |
|--------|------|
| Page size | 16384 bytes (16KB) |
| Page header | 16 bytes |
| File header | 128 bytes (page 0 only) |
| Cache algorithm | SIEVE (not LRU) |
| Lock sharding | 64 shards minimum |
| Memory budget | 25% RAM, 4MB floor |
| Error handling | `eyre` exclusively |
| Sync primitives | `parking_lot` only |

---

## Detailed Rules (READ BEFORE RELEVANT TASKS)

**Before writing tests:** Read `.claude/rules/TESTING.md`

**Before memory/page work:** Read `.claude/rules/MEMORY.md`

**Before error handling:** Read `.claude/rules/ERRORS.md`

**Before key encoding:** Read `.claude/rules/ENCODING.md`

**Before any code:** Read `.claude/rules/STYLE.md`

**Before file format work:** Read `.claude/rules/FILE_FORMATS.md`

---

## AI Self-Check Protocol

Before EVERY implementation, answer these questions:

### For Tests:
1. "What REQUIREMENT does this test verify?" (not "what does the code do")
2. "What specific bug would this test catch?"
3. "What's the expected value and HOW did I compute it?"
4. "If I made a common mistake, would this fail?"

### For Code:
1. "Am I returning a reference or copying data?"
2. "Does this allocate on the heap?"
3. "Does my error message include operation + resource + reason?"
4. "Did I write the test FIRST?"

**If you cannot answer all questions, STOP and reconsider.**

---

## Lock Ordering (Prevent Deadlocks)

Always acquire in this order:
1. Database lock
2. Schema lock
3. Table lock (alphabetical if multiple)
4. Page cache shard lock (by shard index)
5. WAL lock

---

## Performance Targets

| Operation | Target |
|-----------|--------|
| Point read (cached) | < 1µs |
| Point read (disk) | < 50µs |
| Sequential scan | > 1M rows/sec |
| Insert | > 100K rows/sec |
| k-NN (1M vectors, k=10) | < 10ms |

---

## Git Workflow

**Branches:** `feature/`, `fix/`, `perf/`, `refactor/`

**Commits:**
```
type(scope): description

- Bullet points for details
```

Types: `feat`, `fix`, `perf`, `refactor`, `test`, `docs`, `chore`

**Pre-commit:**
1. `cargo test` - ALL pass
2. `cargo clippy` - No warnings
3. Review: "Would each test catch a bug?"

---

## Allowed Dependencies

```toml
eyre = "0.6"
parking_lot = "0.12"
memmap2 = "0.9"
zerocopy = "0.7"
bumpalo = "3.14"
smallvec = "1.11"
hashbrown = "0.14"
```

---

## When Unsure

1. Check the relevant `.claude/rules/*.md` file
2. Default to the more restrictive interpretation
3. Ask for clarification rather than guessing

**Remember: These rules exist because violations cause real problems. Zero-copy isn't a preference - it's a performance requirement. TDD isn't bureaucracy - it catches bugs AI tends to miss.**
