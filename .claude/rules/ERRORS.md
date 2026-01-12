# Error Handling Rules - TurDB

> **THIS FILE IS MANDATORY READING BEFORE ANY ERROR HANDLING**

## Core Rule: `eyre` Only

All error handling MUST use `eyre`. No exceptions.

```rust
use eyre::{Result, WrapErr, bail, ensure};
```

---

## FORBIDDEN: Custom Error Types

```rust
// FORBIDDEN - never do this
enum DbError {
    PageNotFound(u32),
    TableNotFound(String),
    InvalidKey,
}

impl std::error::Error for DbError {}

// Also FORBIDDEN
#[derive(thiserror::Error)]
enum MyError { ... }
```

---

## REQUIRED: Rich Context on Every Error

Every error MUST include:
1. **What operation** was being performed
2. **What resource** was involved (page number, table name, key, etc.)
3. **Why it failed** (if known)

### Good Examples
```rust
// CORRECT: Full context
.wrap_err_with(|| format!(
    "failed to insert row into table '{}' at page {}",
    table_name, page_no
))

// CORRECT: Using bail! with context
bail!(
    "page {} not found in table '{}' while scanning index '{}'",
    page_no, table_name, index_name
)

// CORRECT: Using ensure! for preconditions
ensure!(
    key.len() <= MAX_KEY_SIZE,
    "key size {} exceeds maximum {} for table '{}'",
    key.len(), MAX_KEY_SIZE, table_name
)
```

### Bad Examples
```rust
// FORBIDDEN: No context
.wrap_err("insert failed")

// FORBIDDEN: Vague
bail!("invalid input")

// FORBIDDEN: Missing resource info
ensure!(page_no < max, "page out of bounds")
```

---

## Pattern: Wrap at Boundaries

Add context at module/function boundaries:

```rust
pub fn open_table(&self, name: &str) -> Result<Table> {
    let file = self.open_file(name)
        .wrap_err_with(|| format!("failed to open table file for '{}'", name))?;

    let header = self.read_header(&file)
        .wrap_err_with(|| format!("failed to read header for table '{}'", name))?;

    Ok(Table { file, header })
}
```

---

## Pattern: Early Returns with ensure!

Use `ensure!` for precondition checks:

```rust
pub fn insert(&mut self, key: &[u8], value: &[u8]) -> Result<()> {
    ensure!(!key.is_empty(), "key cannot be empty");
    ensure!(
        key.len() <= MAX_KEY_SIZE,
        "key size {} exceeds maximum {}",
        key.len(), MAX_KEY_SIZE
    );
    ensure!(
        value.len() <= MAX_VALUE_SIZE,
        "value size {} exceeds maximum {}",
        value.len(), MAX_VALUE_SIZE
    );

    // ... actual implementation
}
```

---

## Pattern: bail! for Unrecoverable Conditions

Use `bail!` when you need to return an error immediately:

```rust
fn find_leaf(&self, key: &[u8]) -> Result<&LeafPage> {
    let mut page = self.root_page()?;

    for _ in 0..MAX_TREE_DEPTH {
        match page.node_type() {
            NodeType::Leaf => return Ok(page.as_leaf()),
            NodeType::Interior => {
                let child = page.find_child(key)?;
                page = self.load_page(child)?;
            }
            NodeType::Unknown(t) => {
                bail!(
                    "corrupted page {}: unknown node type 0x{:02x}",
                    page.page_no(), t
                );
            }
        }
    }

    bail!(
        "tree depth exceeded {} while searching for key {:?}",
        MAX_TREE_DEPTH, key
    );
}
```

---

## Pattern: Chaining Context

Build up context as errors propagate:

```rust
fn execute_query(&self, sql: &str) -> Result<Rows> {
    let ast = parse(sql)
        .wrap_err("failed to parse SQL")?;

    let plan = self.plan(&ast)
        .wrap_err("failed to create query plan")?;

    let rows = self.execute_plan(&plan)
        .wrap_err("failed to execute query plan")?;

    Ok(rows)
}
```

When this fails, the error chain shows:
```
failed to execute query plan
  caused by: failed to read page 42
    caused by: I/O error: permission denied
```

---

## Return Types

### Functions that can fail
```rust
fn do_something() -> Result<T>
```

### Functions that cannot fail
```rust
fn do_something() -> T
```

### Never use Option for errors
```rust
// FORBIDDEN
fn find(&self, key: &[u8]) -> Option<&[u8]>

// CORRECT - return Result with Option inside for "not found"
fn find(&self, key: &[u8]) -> Result<Option<&[u8]>>
```

---

## Error Messages: Style Guide

1. **Lowercase start** (eyre formats them nicely)
2. **No trailing punctuation**
3. **Include numeric values** where relevant
4. **Use {:?} for keys/binary data**

```rust
// CORRECT style
bail!("page {} not found in cache", page_no)
bail!("key {:?} too large: {} bytes (max {})", key, key.len(), MAX)

// WRONG style
bail!("Page not found.")  // Uppercase, period
bail!("Key too large")    // No specifics
```

---

## Testing Error Conditions

Test that errors are returned (not panics):

```rust
#[test]
fn insert_oversized_key_returns_error() {
    let mut tree = BTree::new();
    let big_key = vec![0u8; MAX_KEY_SIZE + 1];

    let result = tree.insert(&big_key, b"value");

    assert!(result.is_err());
    let err = result.unwrap_err();
    assert!(err.to_string().contains("exceeds maximum"));
}
```

---

## Quick Reference

| Situation | Use |
|-----------|-----|
| Precondition check | `ensure!(condition, "message")` |
| Immediate error return | `bail!("message")` |
| Add context to Result | `.wrap_err("context")` |
| Add dynamic context | `.wrap_err_with(\|\| format!(...))` |

---

## Checklist

Before any error handling code:

- [ ] Using `eyre`, not custom error types?
- [ ] Error includes operation being performed?
- [ ] Error includes resource (page, table, key)?
- [ ] Error includes why it failed?
- [ ] Using `ensure!` for preconditions?
- [ ] Using `bail!` for unrecoverable conditions?
- [ ] Using `wrap_err` at boundaries?
