# MVCC Transaction Integration Design

## Overview

Integrate the existing MVCC module with SQL-level transaction statements (BEGIN, COMMIT, ROLLBACK, SAVEPOINT) to enable actual data rollback.

## Approach

Wrap MVCC `Transaction` inside `ActiveTransaction` at the Database level.

## Data Structures

### Database struct changes

```rust
pub struct Database {
    // ... existing fields ...
    txn_manager: TransactionManager,
    active_txn: Mutex<Option<ActiveTransaction>>,
}
```

### ActiveTransaction (replaces TransactionState)

```rust
pub struct ActiveTransaction {
    mvcc_txn: OwnedTransaction,  // Owned version of MVCC Transaction
    isolation_level: Option<IsolationLevel>,
    read_only: bool,
    savepoints: SmallVec<[Savepoint; 4]>,
}

pub struct Savepoint {
    pub name: String,
    pub write_entry_idx: usize,
}
```

### OwnedTransaction

Since MVCC `Transaction<'a>` borrows `TransactionManager`, we need an owned version that can be stored in the Database struct alongside the manager. Options:

1. Use `Arc<TransactionManager>` and store manager reference
2. Create `OwnedTransaction` that owns its slot index and interacts with manager via reference when needed
3. Use raw slot management

Going with option 2: `OwnedTransaction` stores the transaction state and releases slot on drop.

## Integration Points

### BEGIN
- Call `txn_manager.begin_txn()` to get MVCC transaction
- Wrap in `ActiveTransaction` with SQL metadata
- Store in `active_txn`

### INSERT/UPDATE/DELETE
- Use `MvccTable::prepare_*_value()` to create MVCC-aware values
- Add `WriteEntry` to transaction's write set
- Store old versions to undo pages for UPDATE/DELETE

### COMMIT
- Call `mvcc_txn.commit_with_finalize()`
- Callback finalizes each record header with commit timestamp

### ROLLBACK (full)
- Call `mvcc_txn.rollback_with_undo()`
- Callback restores old versions from undo pages
- For inserts: delete the row
- For updates/deletes: restore from undo

### ROLLBACK TO SAVEPOINT
- Truncate `write_entries` to savepoint's `write_entry_idx`
- Undo only the entries after the savepoint

### SAVEPOINT
- Record current `write_entries.len()` as `write_entry_idx`

### RELEASE SAVEPOINT
- Remove savepoint from list (entries stay committed)

## Implementation Order (TDD)

1. Add TransactionManager to Database
2. Create OwnedTransaction that can be stored
3. Replace TransactionState with ActiveTransaction
4. Wire BEGIN to create MVCC transaction
5. Wire COMMIT to finalize with callback
6. Wire ROLLBACK to undo with callback
7. Update INSERT to track writes
8. Update DELETE to track writes
9. Update UPDATE to track writes
10. Enable ignored rollback tests
