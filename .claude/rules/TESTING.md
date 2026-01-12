# Testing Rules - TurDB

> **THIS FILE IS MANDATORY READING BEFORE WRITING ANY TEST**

## The Core Problem

AI assistants write tests that "pass" without verifying correctness because:
1. AI optimizes for "green tests" not "correct code"
2. AI sees implementation while writing tests (circular reasoning)
3. AI writes tests describing what code does, not what it should do

---

## MANDATORY: Red-Green-Refactor

**This sequence is NON-NEGOTIABLE:**

### 1. RED - Write Test First
```bash
# Test MUST fail before implementation
cargo test test_name
# Expected: FAILED
```
- If test passes before implementation exists, the test is WORTHLESS
- A test that never fails proves nothing

### 2. GREEN - Minimal Implementation
- Write ONLY enough code to make test pass
- No extra features, no "while I'm here" additions

### 3. REFACTOR - Clean Up
- Improve code while tests stay green
- Run tests after every change

**VIOLATION = REJECTED CODE. No exceptions.**

---

## FORBIDDEN Test Patterns

### 1. Tautologies (Testing Function Against Itself)
```rust
// FORBIDDEN - comparing function to itself
let result = parse("input");
assert_eq!(result, parse("input")); // USELESS!

// FORBIDDEN - using function to compute expected
let expected = encode_int(42);
assert_eq!(encode_int(42), expected); // TAUTOLOGY!
```

### 2. Testing Implementation, Not Behavior
```rust
// FORBIDDEN - testing internal state
tree.insert(b"key", b"value");
assert!(tree.root.is_some()); // WRONG - internal detail

// CORRECT - testing observable behavior
tree.insert(b"key", b"value").unwrap();
assert_eq!(tree.get(b"key").unwrap(), b"value"); // Behavior!
```

### 3. Empty/Vague Assertions
```rust
// FORBIDDEN
assert!(true);
assert!(result.is_ok()); // What's in the Ok?
assert!(list.len() > 0); // How many exactly?
let _ = Widget::new(); // No assertion at all

// REQUIRED - specific values
assert_eq!(result.unwrap(), expected_value);
assert_eq!(list.len(), 3);
```

### 4. Vague Test Names
```rust
// FORBIDDEN
fn test_insert() { }
fn test_1() { }
fn test_stuff() { }

// REQUIRED - describe what, condition, expected outcome
fn insert_duplicate_key_overwrites_existing_value() { }
fn delete_nonexistent_key_returns_not_found_error() { }
fn split_preserves_all_keys_when_node_full() { }
```

---

## REQUIRED: Expected Values Must Be Independent

```rust
// CORRECT: Independently derived expected value
// 42 in big-endian with POS_INT prefix = [0x16, 0, 0, 0, 0, 0, 0, 0, 42]
let expected = vec![0x16, 0, 0, 0, 0, 0, 0, 0, 42];
assert_eq!(encode_int(42), expected);

// Document WHY the expected value is what it is:
// - 0x16 = POS_INT type prefix
// - Remaining 8 bytes = big-endian u64 representation of 42
```

---

## REQUIRED: Edge Cases (Every Function)

1. **Empty/Zero**: `""`, `[]`, `0`, `None`
2. **Single element**: Minimum valid input
3. **Boundary**: `MAX-1`, `MAX`, `MAX+1`
4. **Error conditions**: Invalid inputs that SHOULD fail
5. **Negative cases**: What should NOT happen

```rust
#[test]
fn get_empty_key_returns_error() {
    let tree = BTree::new();
    assert!(tree.get(b"").is_err());
}

#[test]
fn get_nonexistent_key_returns_none() {
    let tree = BTree::new();
    assert!(tree.get(b"missing").unwrap().is_none());
}

#[test]
fn insert_max_size_key_succeeds() {
    let mut tree = BTree::new();
    let max_key = vec![0u8; MAX_KEY_SIZE];
    assert!(tree.insert(&max_key, b"v").is_ok());
}

#[test]
fn insert_oversized_key_returns_error() {
    let mut tree = BTree::new();
    let big_key = vec![0u8; MAX_KEY_SIZE + 1];
    assert!(tree.insert(&big_key, b"v").is_err());
}
```

---

## REQUIRED: Bug Fix Tests

For EVERY bug fix:

1. Write test that REPRODUCES the bug FIRST
2. Run test - MUST FAIL (proves bug exists)
3. Fix the bug
4. Run test - MUST PASS
5. Test stays forever as regression protection

```rust
#[test]
fn test_issue_42_off_by_one_in_split() {
    // This specific sequence triggered the bug
    let mut tree = BTree::new();
    for i in (0..100).rev() {
        tree.insert(&[i], &[i]).unwrap();
    }
    // Bug caused key 50 to be lost after split
    assert_eq!(tree.get(&[50]).unwrap(), Some(&[50][..]));
}
```

---

## REQUIRED: Mutation Testing Mindset

For every test, identify at least ONE mutation that would break it:

```rust
#[test]
fn binary_search_finds_middle_element() {
    let arr = [1, 3, 5, 7, 9];
    assert_eq!(binary_search(&arr, 5), Some(2));
    // Mutation that would break this:
    // If we used `<` instead of `<=` in comparison,
    // this would fail to find 5
}
```

If you can't identify a breaking mutation, the test is too weak.

---

## AI Self-Check Before Every Test

Ask yourself:

1. **"What REQUIREMENT does this test verify?"**
   - Not "what does the code do" but "what SHOULD it do"

2. **"What specific bug would this test catch?"**
   - Name the bug: "off-by-one in boundary check"

3. **"What's the expected value and HOW did I compute it?"**
   - Must be computed INDEPENDENTLY of the function

4. **"If I made a common mistake in implementation, would this fail?"**
   - Think: wrong comparison operator, off-by-one, null handling

**If you cannot answer ALL FOUR questions, DO NOT write the test.**

---

## Anti-Patterns to Avoid

1. **NO mocking your own code** - Only mock: file systems, network, time, randomness

2. **NO test-only methods in production code**
   ```rust
   // FORBIDDEN
   #[cfg(test)]
   pub fn get_internal_state(&self) -> &Node { ... }
   ```

3. **NO changing implementation to pass tests** - Figure out which is wrong first

4. **NO snapshot tests for logic** - Only for UI/output format

5. **NO ignoring flaky tests** - Fix or delete

---

## Coverage Requirements

- **Minimum 80%** line coverage for all modules
- **100%** for: serialization, encoding, public API
- Coverage alone is NOT sufficient - tests must be meaningful

---

## Integration Tests

Required scenarios:
- Full CRUD cycles with real files (not mocks)
- Crash recovery (kill process mid-operation)
- Concurrent access (multiple threads/processes)

---

## Pre-Commit Checklist

Before ANY commit:
1. `cargo test` - ALL pass
2. `cargo clippy` - No warnings
3. Review each test: "Would this catch a bug?"
