# OUTER JOIN Implementation Design

## Overview

This document describes the implementation of OUTER JOINs (LEFT, RIGHT, FULL) in TurDB's query executor. Currently, the executor only supports INNER JOIN semantics despite the parser and planner supporting all join types.

## Problem Statement

The `JoinType` enum exists with `Inner`, `Left`, `Right`, `Full`, `Cross` variants, and the physical plan correctly stores `join_type`. However, the executor states (`NestedLoopJoinState`, `GraceHashJoinState`) don't have a `join_type` field, so all joins behave as INNER JOINs.

## OUTER JOIN Semantics

- **LEFT JOIN**: Emit all left rows; if no right match, pad right columns with NULLs
- **RIGHT JOIN**: Emit all right rows; if no left match, pad left columns with NULLs
- **FULL JOIN**: Emit all rows from both sides; pad missing side with NULLs

## Design: NestedLoopJoinState Changes

### New Fields

```rust
pub struct NestedLoopJoinState<'a, S: RowSource> {
    // ... existing fields ...
    pub join_type: JoinType,
    pub left_matched: bool,
    pub right_matched: Vec<bool>,
    pub emitting_unmatched_right: bool,
    pub unmatched_right_idx: usize,
    pub left_col_count: usize,
    pub right_col_count: usize,
}
```

### Execution Logic

1. **During inner loop (checking right rows):**
   - When match found: set `left_matched = true`, mark `right_matched[idx] = true`

2. **After exhausting right rows for current left row:**
   - If `!left_matched && (join_type == Left || join_type == Full)`:
     - Emit left row + NULLs for right columns
   - Reset `left_matched = false`, move to next left row

3. **After exhausting all left rows:**
   - If `join_type == Right || join_type == Full`:
     - Enter `emitting_unmatched_right` phase
     - Emit unmatched right rows + NULLs for left columns

## Design: GraceHashJoinState Changes

### New Fields

```rust
pub struct GraceHashJoinState<'a, S: RowSource> {
    // ... existing fields ...
    pub join_type: JoinType,
    pub build_matched: Vec<Vec<bool>>,
    pub probe_row_matched: bool,
    pub emitting_unmatched_build: bool,
    pub emitting_unmatched_probe: bool,
    pub unmatched_build_partition: usize,
    pub unmatched_build_idx: usize,
    pub unmatched_probe_partition: usize,
    pub unmatched_probe_idx: usize,
    pub left_col_count: usize,
    pub right_col_count: usize,
}
```

### Execution Phases

**Phase 1: Normal join (existing logic)**
- Probe right rows against left hash table
- Mark `build_matched[partition][idx] = true` when match found
- Track `probe_row_matched` for current probe row
- After each probe row exhausts matches:
  - If `!probe_row_matched && (join_type == Right || join_type == Full)`:
    - Emit probe row + NULLs for build columns

**Phase 2: Emit unmatched build rows (for LEFT/FULL)**
- After all partitions probed
- Iterate through `build_matched`, emit rows where `!matched`
- Emit with NULLs for probe columns

### Column Ordering

- Build = left side, Probe = right side
- Output order: `left_cols || right_cols`

## Builder Changes

Update `build_nested_loop_join` and `build_grace_hash_join` to accept `join_type` parameter:

```rust
pub fn build_nested_loop_join<S: RowSource>(
    &self,
    left: DynamicExecutor<'a, S>,
    right: DynamicExecutor<'a, S>,
    condition: Option<&'a crate::sql::ast::Expr<'a>>,
    column_map: &[(String, usize)],
    join_type: JoinType,  // NEW
    left_col_count: usize,  // NEW
    right_col_count: usize, // NEW
) -> NestedLoopJoinState<'a, S>
```

## Files to Modify

1. `src/sql/state.rs` - Add new fields to state structs
2. `src/sql/executor.rs` - Modify DynamicExecutor::next() for both join types
3. `src/sql/builder.rs` - Update builder methods to accept join_type
4. Wherever joins are built from physical plans - pass join_type through

## Testing Strategy

- Test LEFT JOIN with matching rows
- Test LEFT JOIN with unmatched left rows
- Test RIGHT JOIN with matching rows
- Test RIGHT JOIN with unmatched right rows
- Test FULL JOIN with rows on both sides
- Test empty left/right tables
- Test with NULL join keys
