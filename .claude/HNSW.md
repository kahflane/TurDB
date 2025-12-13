### 1. HNSW Page Layout (16KB Pages)

Do not allocate one page per node. With 16KB pages, you should use a **Slotted Page** architecture (similar to your B-Tree) to pack multiple graph nodes per page.

**The Strategy: Separation of Topology and Data**
For optimal cache locality during graph traversal, you want neighbor lists packed tightly. The raw f32 vectors are bulky.
1.  **Topology (Neighbors):** Store in `.hnsw` file (slotted pages).
2.  **Vector Data:** Store either:
    *   **Quantized (SQ8)** in the `.hnsw` node (highly recommended for performance).
    *   **Pointer** to the `.tbd` file (if zero-copy from main storage is strict, but this causes random I/O thrashing during search).

**Recommended Page Structure:**
*   **Header (64 bytes):** PageType, count, free_space_offset, etc.
*   **Slot Array (grows forward):** `u16` offsets to nodes.
*   **Node Data (grows backward):** The actual neighbor lists.

**Node Format (Byte-aligned packed struct):**
```rust
struct HnswNode {
    // 1. Metadata
    row_id: u64,       // Pointer to actual row in .tbd (for MVCC check)
    max_level: u8,     // Highest level this node exists in
    padding: u8,       // Alignment
    
    // 2. Quantized Vector (Optional but recommended)
    // If using SQ8, store 128-1536 bytes here + 2 f32 (min/scale). 
    // This allows distance calc without jumping to .tbd file.
    
    // 3. Level 0 Neighbors (Base layer)
    // Fixed size array or offset to dynamic list.
    // Recommended: Fixed size [NeighborId; M_max0] for zero-allocation.
    l0_count: u8,
    l0_neighbors: [NodeId; 32], // Assuming M_max0 = 32
    
    // 4. Higher Level Neighbors (Levels 1..max_level)
    // Variable length. Usually stored contiguously after L0.
    // [Level 1 Count, N1, N2...], [Level 2 Count, N1, N2...]
}

// NodeId is usually a struct of { page_id: u32, slot_index: u16 }
// This fits in 6 bytes, or 8 bytes (u64) for alignment.
```

**Handling Overflow (Large Vectors):**
If `1536 dims * 4 bytes` = 6KB, a node fits in a 16KB page. If you have `M=64`, the neighbor list is small (~512 bytes).
*   **Case 1 (Standard):** Everything fits in one page.
*   **Case 2 (Huge Embeddings):** If a vector + neighbors > 16KB (rare), implement **Overflow Pages** (linked list of pages), exactly like a B-Tree implementation handles large BLOBs.

---

### 2. HNSW Parameters for Embedded Use

For a 1MB memory budget and ~1M vectors, you must be conservative.

*   **M (Max connections):** **16**.
    *   Low memory footprint.
    *   Performance degradation is minimal compared to 32/64 for < 1M vectors.
    *   Level 0 max connections ($M_{0}$) should be $2 \times M = 32$.
*   **efConstruction:** **100**.
    *   Good balance of build speed vs. graph quality. 200-400 is overkill for an embedded DB unless write speed is irrelevant.
*   **efSearch:** **Default 32** (allow override at runtime).
    *   Keep it low to meet memory budgets for the visited set/priority queue scratchpads.
*   **Level Generation:** Use a probability of $1 / \ln(M)$.

---

### 3. File Format Structure (`.hnsw`)

Yes, separate the graph structure.

**Page 0: Superblock / Header**
```rust
#[repr(C)]
struct HnswHeader {
    magic: [u8; 8],        // "TurDBVec"
    version: u32,          // 1
    page_size: u32,        // 16384
    
    // Config
    dimensions: u16,
    m: u16,                // 16
    m0: u16,               // 32
    ef_construction: u16,
    dist_func: u8,         // 0=L2, 1=Cosine, 2=Dot
    
    // State
    entry_point: NodeId,   // The node ID of the top-layer entry
    max_level: u8,         // Current max level of graph
    node_count: u64,
    
    // Quantization info (if global SQ8 is used)
    global_min: f32,
    global_max: f32,
}
```

**Pages 1+: Slotted Data Pages**
Containing the `HnswNode` structures defined in section 1.

---

### 4. MVCC Integration (Critical)

**Recommendation: Option D (Visibility Filter / Post-Filtering)**

Do **not** version the graph edges. It is too expensive and complex to maintain a DAG history.
Instead, maintain the HNSW graph as a **superset** of all visible data.

**The Algorithm:**
1.  **Insert (Txn T1):**
    *   T1 inserts the row into `.tbd` (with `xmin=T1`).
    *   T1 inserts the node into `.hnsw`. The node in HNSW points to `RowId`.
    *   T1 acts as if the node exists.
2.  **Concurrent Search (Txn T2):**
    *   T2 traverses the HNSW graph.
    *   The graph contains nodes from T1.
    *   **Crucial Step:** When calculating distance or collecting results, T2 checks the MVCC header of the target `RowId` (in `.tbd` or a cached visibility map).
    *   If `RowId` is not visible to T2, **ignore it as a result result**.
    *   *Note:* You still use the "invisible" node for graph traversal (stepping stones). If you don't, the graph becomes disconnected for T2. This is safe and standard practice (Lucene does this).
3.  **Rollback (Txn T1):**
    *   Mark the row in `.tbd` as aborted.
    *   **Lazy Cleanup:** You cannot immediately remove the node from HNSW because other transactions might be holding pointers to it.
    *   Add the `NodeId` to a "Vacuum Queue". When the vacuum runs, it repairs the graph (removes the node and rewires neighbors).

**Trade-off:** Rollbacked inserts leave "ghost" nodes that degrade search performance slightly until Vacuum runs. This is acceptable for an embedded DB.

---

### 5. SIMD Distance Functions

**Answers:**
*   **Non-multiples of 8:** Yes, handle the tail with a scalar loop or a masked load.
*   **FMA:** Absolutely use `_mm256_fmadd_ps` if available. It's significantly faster for dot products.
*   **Normalization:** For Cosine Similarity, **always pre-normalize** vectors to unit length at insertion time. Then just use Dot Product. It saves a `sqrt` and division per distance calculation (huge win).

**Rust/AVX2 Example (Zero-Allocation):**

```rust
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[target_feature(enable = "avx2", enable = "fma")]
pub unsafe fn dot_product_avx2(a: &[f32], b: &[f32]) -> f32 {
    let mut sum = _mm256_setzero_ps();
    let n = a.len();
    let mut i = 0;

    // Main loop: 8 floats at a time
    while i + 8 <= n {
        let va = _mm256_loadu_ps(a.as_ptr().add(i));
        let vb = _mm256_loadu_ps(b.as_ptr().add(i));
        // fused multiply-add: sum += va * vb
        sum = _mm256_fmadd_ps(va, vb, sum);
        i += 8;
    }

    // Horizontal sum of the register
    let mut s: f32 = 0.0;
    // ... extract horizontal sum from 'sum' register ...
    
    // Handle remainder scalar
    while i < n {
        s += a[i] * b[i];
        i += 1;
    }
    s
}
```
*Note: Use the `multiversion` crate or standard `std::is_x86_feature_detected!` to enable runtime dispatch between AVX2, NEON, and Scalar fallback.*

---

### 6. SQ8 Quantization Strategy

For embedded usage, **Per-Vector Quantization** is best. Global requires a training phase (analyzing all data) which contradicts "incremental insert."

**Data Structure:**
```rust
struct Sq8Vector {
    min_val: f32,    // 4 bytes
    diff: f32,       // 4 bytes: (max - min) / 255.0
    data: [u8],      // N bytes
}
```
**Storage:** Store this compressed struct directly in the HNSW Node (in the `.hnsw` file).
**Recall:** With 1536 dims, SQ8 is surprisingly accurate (usually >0.98 recall).
**Search:** Dequantize on the fly? No.
*   SIMD optimization: Convert the *Query* vector into `u8` (if possible) or upgrade the stored `u8` to `f32` inside the AVX register to compute distance. Upgrading `u8` -> `f32` in AVX is very fast (`_mm256_cvtepu8_epi32` + `_mm256_cvtepi32_ps`).

---

### 7. Zero-Copy & Zero-Allocation Search API

To achieve zero allocation per query, you must require the caller to provide the context/scratchpad.

**The Context Struct:**
```rust
pub struct HnswSearchContext {
    // Reusable buffers
    pub visited: bitvec::BitVec, // Or a HashSet with a custom allocator
    pub candidates: BinaryHeap<DistNode>,
    pub nearest: Vec<DistNode>,
    // SIMD alignment buffers if needed
}

impl HnswSearchContext {
    pub fn reset(&mut self) {
        self.visited.clear();
        self.candidates.clear();
        self.nearest.clear();
    }
}
```

**The Search API:**
```rust
impl HnswIndex {
    pub fn search(
        &self, 
        query: &[f32], 
        k: usize, 
        ctx: &mut HnswSearchContext, // Caller owns memory
        txn: &Transaction // For visibility check
    ) -> Result<usize, Error> {
        ctx.reset();
        
        // ... perform search ...
        // ... push results into ctx.nearest ...
        
        Ok(ctx.nearest.len())
    }
}
```
In your database layer, keep a `ThreadLocal<RefCell<HnswSearchContext>>` or a pool of contexts to reuse them across SQL queries.

---

### 8. Building the Index

For your constraints: **Incremental Insertion only.**
Bulk loading (Vamana/HNSW-construction) often requires holding the whole dataset in memory or complex sorting passes.
Since you have a WAL and page cache, insert vectors one by one as rows are committed.

**Memory Tip:** When inserting, you only need to hold the *current* node and the *neighbor lists* of nodes you are updating. The Page Cache handles the memory budgeting. If you traverse a node not in memory, the Page Cache loads it, possibly evicting an old page. This fits your architecture perfectly.

---

### 9. SQL Integration

**Option A (Dedicated Type) is superior.**

```sql
CREATE TABLE items (
    id INT PRIMARY KEY,
    embedding VECTOR(1536) -- Known fixed size at compile/table-create time
);
```

*   **Why:** You need the dimension `1536` known to allocate the fixed-size slots in your HNSW pages.
*   **Validation:** The type system guarantees input length.
*   **Storage:** Internally, map `VECTOR(N)` to a binary serialization format, but the metadata knows it's a vector for index selection.

---

### 10. Testing Strategy

1.  **Property-Based Testing (Proptest/Quickcheck):**
    *   Generate random vectors.
    *   Insert into HNSW.
    *   Brute-force scan the list.
    *   Assert `hnsw_result` overlaps with `brute_force_result` by X% (Recall).
2.  **The "Orphan" Test:**
    *   After N inserts, traverse the graph from the entry point.
    *   Ensure every node is reachable (use a BFS/DFS).
3.  **MVCC Isolation Test:**
    *   Thread A: Insert unique vector $V_a$, don't commit.
    *   Thread B: Search near $V_a$.
    *   Assert B does *not* find $V_a$.
    *   Thread A: Commit.
    *   Thread B: Search. Assert B *does* find $V_a$.
4.  **Crash Recovery:**
    *   Insert 1000 vectors.
    *   Kill process (no clean shutdown).
    *   Restart (WAL replay).
    *   Verify Index is valid and searchable.

### Critical Pitfall to Avoid in Rust

**Self-Referential Structs in Mmap:**
You cannot easily store Rust references (`&'a [u8]`) inside your Node structs if they point to the mmap region, because the mmap might be remapped (grown).
*   **Solution:** Always work with **PageId + Offset**. Resolve pointers only strictly within the scope of a function call where you hold a lock/guard on the page cache frame.

**Concurrency Deadlocks:**
HNSW involves walking widely across the graph.
*   **Safe approach:** Use "Hand-over-hand" locking (crabbing) if using fine-grained locks.
*   **Simpler approach (Start here):** `RwLock` on the Index structure.
    *   `read()` for Search.
    *   `write()` for Insert.
    *   Since insertion is fast, this might be sufficient for 1MB embedded use cases. If you need concurrent writes, you need fine-grained page locking (which your storage layer likely already has). Be very careful of lock ordering (always lock lower PageID before higher, or use standard deadlock detection).