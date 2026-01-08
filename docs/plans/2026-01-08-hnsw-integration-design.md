# HNSW Integration Design

## Overview

Wire HNSW vector indexes to DML operations (INSERT, DELETE, UPDATE) following PostgreSQL pgvector patterns.

## Design Decisions

1. **Vector Storage**: Vectors stored in table rows, HNSW nodes store only `row_id` and connectivity
2. **Index Maintenance**: Synchronous within DML transactions
3. **row_id Mapping**: B-tree index for O(log n) row_id → node_id lookups
4. **Zero-Allocation**: Fixed-size arrays for neighbor lists (max 4 levels, 32 L0 neighbors, 16 higher-level neighbors)

## Architecture

```
INSERT INTO t (id, embedding) VALUES (1, '[0.1, 0.2, ...]')
    │
    ├─► 1. Insert row into table BTree (existing)
    │
    └─► 2. For each HNSW index on vector columns:
            ├─► Load PersistentHnswIndex
            ├─► Call index.insert(row_id, vector, random)
            └─► Update row_id → node_id mapping

DELETE FROM t WHERE id = 1
    │
    ├─► 1. Delete row from table BTree (existing)
    │
    └─► 2. For each HNSW index:
            ├─► Find node_id from row_id (B-tree lookup)
            ├─► Call index.delete(node_id)
            └─► Remove from row_id mapping

SELECT * FROM t ORDER BY embedding <-> '[query]' LIMIT 10
    │
    └─► Call index.search(query, k, ctx, |row_id| {
           fetch_vector_from_table(row_id)
        })
```

## Implementation Tasks

### Task 1: Zero-Allocation Node Structure

Replace `HnswNode` with inline fixed-size arrays:

```rust
const MAX_LEVELS: usize = 4;
const MAX_L0_NEIGHBORS: usize = 32;
const MAX_LEVEL_NEIGHBORS: usize = 16;

#[repr(C)]
pub struct HnswNodeInline {
    row_id: u64,
    max_level: u8,
    l0_count: u8,
    l0_neighbors: [NodeId; MAX_L0_NEIGHBORS],
    higher_counts: [u8; MAX_LEVELS],
    higher_neighbors: [[NodeId; MAX_LEVEL_NEIGHBORS]; MAX_LEVELS],
}
```

### Task 2: row_id → node_id Mapping

Add B-tree within HNSW file for reverse lookup:

```rust
impl PersistentHnswIndex {
    row_id_map_root: u32,  // Root page of mapping B-tree

    pub fn find_node_by_row_id(&self, row_id: u64) -> Option<NodeId>
    pub fn delete_by_row_id(&mut self, row_id: u64) -> Result<()>
}
```

### Task 3: DML Integration - INSERT

In `src/database/dml/insert.rs`:

1. Collect HNSW indexes on vector columns (like existing `secondary_indexes`)
2. After row insert, call `hnsw.insert(row_id, vector, random)` for each
3. Generate random value for level selection

### Task 4: DML Integration - DELETE

In `src/database/dml/delete.rs`:

1. Collect HNSW index names for table
2. Before/after row delete, call `hnsw.delete_by_row_id(row_id)`

### Task 5: DML Integration - UPDATE

In `src/database/dml/update.rs`:

1. If vector column is being updated:
   - Delete old node: `hnsw.delete_by_row_id(row_id)`
   - Insert new node: `hnsw.insert(row_id, new_vector, random)`

### Task 6: Zero-Copy Search Callback

Modify search to use caller-provided buffer for vector fetch:

```rust
pub fn search(
    &self,
    query: &[f32],
    k: usize,
    ctx: &mut HnswSearchContext,
    results: &mut [SearchResult],
    get_vector: impl Fn(u64, &mut [f32]) -> bool,
) -> Result<usize>
```

## Testing Requirements

Per CLAUDE.md TDD requirements:

1. **INSERT test**: Insert row with vector, verify HNSW node created with correct row_id
2. **DELETE test**: Delete row, verify HNSW node removed and neighbors updated
3. **UPDATE test**: Update vector column, verify old node removed and new node inserted
4. **Search test**: Insert multiple vectors, verify k-NN search returns correct row_ids
5. **Edge cases**: Empty index, single node, max capacity, duplicate row_ids

## Files to Modify

- `src/hnsw/mod.rs` - HnswNodeInline, row_id mapping, delete_by_row_id
- `src/database/dml/insert.rs` - HNSW index updates
- `src/database/dml/delete.rs` - HNSW node deletion
- `src/database/dml/update.rs` - Vector column updates
- `src/storage/file_manager.rs` - HNSW loading integration