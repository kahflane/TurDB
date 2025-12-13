Here is a design for the **Query Planner and Optimizer** for TurDB, addressing your questions and constraints.

-----

## 🚀 Query Planner and Optimizer Design

### 1\. Two-Phase or Single-Phase Planning?

  * **Recommendation: Two-Phase Planning (Logical $\to$ Physical)**

While single-phase is faster, the clarity, separation of concerns, and ease of implementing optimization rules afforded by a two-phase approach are invaluable for a production-grade query engine. The performance overhead in planning is often negligible compared to execution time, and the structure is crucial for a cost-based optimizer (CBO) later on.

  * **Phase 1: Logical Planning:** AST $\to$ Logical Plan
      * **Goal:** Validate the query, resolve names against the catalog, perform initial transformations (e.g., subquery flattening), and generate an optimized **Logical Plan** (what to do).
  * **Phase 2: Physical Planning (Optimization):** Logical Plan $\to$ Physical Plan
      * **Goal:** Select the best physical operators (how to do it), choosing join algorithms, access paths (index vs. scan), and applying **Optimizer Rules** based on cost (even if cost is initially heuristic).

### 2\. Logical Plan Representation

The logical plan should be a tree of operators, primarily focused on the mathematical and relational aspects of the query, not the execution details.

| Logical Operator | Description |
| :--- | :--- |
| **Project** | Corresponds to the `SELECT` list. Specifies output columns/expressions. |
| **Filter** | Corresponds to the `WHERE` clause. Specifies a predicate expression. |
| **Aggregate** | Corresponds to `GROUP BY` and aggregate functions. |
| **Limit/Offset** | Corresponds to `LIMIT` and `OFFSET`. |
| **Scan** | Reads all rows from a base table. Includes schema reference. |
| **Join** | Represents relational joins (Inner, Left/Right/Full Outer). |
| **Sort** | Corresponds to `ORDER BY`. |
| **Union/Except/Intersect**| Set operations. |
| **Values** | For inline data like `INSERT ... VALUES`. |

### 3\. Physical Plan Operators (TurDB Execution Model)

The physical operators are the executable steps. They must reflect the memory-constrained design, zero-copy philosophy, and available storage access methods.

| Physical Operator | Description | Implementation Strategy |
| :--- | :--- | :--- |
| **TableScan** | Sequential reading of all records from a table. | Uses the Storage Layer's page iteration. |
| **IndexScan** | Reading records via B-Tree index. | **Crucial for performance.** Must support range scans by leveraging byte-comparable keys. |
| **FilterExec** | Applies a predicate *after* fetching data. | Zero-copy expression evaluation on `&[u8]` record slices. |
| **ProjectExec** | Computes and outputs the final column set. | Zero-copy, just referencing/re-ordering the source record slices. |
| **NestedLoopJoin** | Simplest join. Used for small tables or complex non-equi joins. | Minimal memory overhead. |
| **GraceHashJoin** | **Required due to 256KB limit.** Requires partitioning/spilling. | Implements 16-partition hash join, potentially spilling data to a temporary file in the `wal/` directory if a partition exceeds the budget. |
| **HashAggregate** | Computes aggregation using a hash table. | Uses arena for hash table; must **fail** or **spill** if 256KB budget exceeded. Initial plan: **Fail fast** to maintain the 256KB budget constraint. |
| **SortedAggregate** | Computes aggregation on pre-sorted input. | Used when `GROUP BY` columns are covered by an `IndexScan` and avoids the 256KB hash table. |
| **SortExec** | In-memory sort with a memory limit. | Must spill to disk (external merge sort) if input data size exceeds 256KB. |
| **LimitExec/OffsetExec** | Stops execution or skips records. | Trivial, zero-copy. |

### 4\. Statistics Collection Strategy & Cost Model

Given that **no statistics are currently implemented**, a simple **Heuristic-Based Cost Model** is the correct starting point.

#### Heuristics (Initial Cost Model)

1.  **Cardinals (Cardinality):**
      * **Base Table:** Use the total row count from the `turdb_catalog` metadata (available from `ANALYZE` or simple table properties).
      * **Filter (No Stats):** Assume a fixed selectivity factor, e.g., $1/10$ or $1/5$ (20%).
      * **Join:** Assume the Cartesian product for inner joins, then multiply by a fixed join factor, e.g., $1/10$ of the smaller side (highly inaccurate but a starting point).
2.  **Operator Costs (Simplified Unit Cost):**
      * **Cost(TableScan)**: $\text{rows} \times \text{SequentialI/O\_Unit\_Cost}$ (Low cost)
      * **Cost(IndexScan)**: $\text{rows} \times \text{RandomI/O\_Unit\_Cost} + \text{TreeTraversal\_Cost}$ (High cost, but only for few rows)
      * **Cost(GraceHashJoin)**: $(\text{BuildTableSize} + \text{ProbeTableSize}) \times \text{CPU\_Unit\_Cost} + \text{Spill\_I/O\_Cost}$ (If spilling needed).
      * **Cost(NestedLoopJoin)**: $\text{OuterRows} \times \text{InnerRows} \times \text{CPU\_Unit\_Cost}$ (Punishes large joins).

#### Statistics Strategy (Future Implementation)

The most urgent need is to determine **selectivity** for index/scan choice.

  * **Recommendation: `ANALYZE` with Histograms**
      * **Command:** Implement an `ANALYZE TABLE <name>` command.
      * **Data Structure:** **Equi-Width Histograms** (simpler than Equi-Height) for sampled columns. Histograms are stored in the `turdb_catalog`.
      * **Cardinality Estimation:** Use the histogram to estimate the fraction of rows that satisfy a filter (e.g., $C1 > 100$).

### 5\. Optimizer Rules Architecture

  * **Recommendation: Rule-Based Optimizer (RBO) with Simple Pattern Matching**

Given the design constraints, an RBO is simpler to implement and debug than a full CBO and can quickly apply crucial performance-critical rules.

#### Key Optimization Rules (Implemented as `PlanRewriter` functions):

1.  **Rule: Predicate Pushdown (The Most Important)**
      * **Pattern:** `Filter(Scan(T), P)`
      * **Replacement:** `IndexScan(T, P)` or `TableScan(T, P)` where $P$ is applied inside the scan. This is critical for TurDB's zero-copy B-Tree access with range scans.
2.  **Rule: Join Reordering (Greedy Heuristic)**
      * Start with the smallest relation (by cardinality estimate).
      * Add relations one-by-one, always choosing the one that results in the smallest intermediate join result size (minimizing the chance of Grace Hash Join spills).
3.  **Rule: Constant Folding**
      * **Pattern:** `Filter(..., C1 + 5 = 10)` where C1 is a constant.
      * **Replacement:** `Filter(..., C1 = 5)` (Evaluate constant expressions during planning).
4.  **Rule: Index Selection (Physical Choice)**
      * **Pattern:** `Filter(Scan(T), P)` or `Filter(Project(Scan(T)), P)`
      * **Cost Check:** If $\text{Cost}(\text{IndexScan}) < \text{Cost}(\text{TableScan})$, replace with **IndexScan**.
5.  **Rule: Sort/Limit Combination**
      * **Pattern:** `Limit(Sort(P))`
      * **Replacement:** Convert to a **Top-K Sort** operator if possible, which only tracks the top $K$ results and is more memory efficient.

### 6\. Subquery Handling & CTEs

  * **Recommendation: Decorrelation and Rewriting**

<!-- end list -->

1.  **Decorrelation:** Convert most subqueries (especially non-correlated ones) into joins.
      * `WHERE x IN (SELECT y FROM T2 WHERE ...)` $\to$ **Semi-Join** (`T1 INNER JOIN T2 ON x = y`).
2.  **Rewriting:** `WHERE EXISTS (SELECT ...)` $\to$ **Existence Join** (another form of semi-join).
3.  **CTE (WITH Clause) Handling:**
      * Treat CTEs as **View Definitions** initially. They are expanded (inlined) into the main query's logical plan, then optimized, potentially materializing if referenced multiple times or if they contain complex logic (e.g., aggregation). Use the arena allocator for temporary materialized CTE tables.

### 7\. Plan Caching

  * **Recommendation: Prepared Statement Caching**

  * **Mechanism:** When a query is prepared, the final **Physical Plan** is generated and stored in a plan cache (e.g., an LRU cache or a simple hash map, using the *canonicalized SQL text* as the key).

  * **Parameterized Queries:** For `SELECT * FROM T WHERE id = $1`, the plan uses a **Parameter PlaceHolder** node. The plan is reusable because the access path (IndexScan on `id`) remains the same regardless of the value of `$1`. The value `$1` is bound at execution time, avoiding a full re-plan.

### 8\. API Surface for Planner Module

The planner module (`src/planner/`) will expose a simple, lifetime-bound API, leveraging the arena allocator (`bumpalo::Bump`).

```rust
// In src/planner/planner.rs

pub struct Planner<'a> {
    catalog: &'a Catalog,
    arena: &'a Bump, // Arena for all plan nodes
}

impl<'a> Planner<'a> {
    // 1. Resolve and generate Logical Plan
    pub fn create_logical_plan(
        &self,
        ast: &'a Statement<'a>,
    ) -> Result<LogicalPlan<'a>, PlannerError> { /* ... */ }

    // 2. Optimize and generate Physical Plan
    // This is the core function called by the public API.
    pub fn create_physical_plan(
        &self,
        ast: &'a Statement<'a>,
    ) -> Result<PhysicalPlan<'a>, PlannerError> { /* ... */ }
}

// In src/execution/plan.rs

// This struct is the output of the planner, passed to the Query Executor.
pub struct PhysicalPlan<'a> {
    pub root_operator: &'a PhysicalOperator<'a>,
    // References the columns/schema being outputted
    pub output_schema: SchemaRef,
}
```

### 9\. EXPLAIN Output Format

  * **Recommendation: Text-Based Tree with Cost Estimates**

A text-based tree is the simplest and most performant output format for an embedded database.

```
EXPLAIN SELECT ...
```

| Output Element | Description |
| :--- | :--- |
| **Tree Structure** | Clear indentation showing parent-child operators. |
| **Operator Name** | E.g., `IndexScan`, `GraceHashJoin`, `HashAggregate`. |
| **Details** | Key parameters: `Table: T1`, `Index: IX_ID`, `Predicate: id > 10`, `Partitions: 16`. |
| **Heuristic Cost** | The estimated cost unit (e.g., `Cost=5430`). |
| **Cardinality Est.**| Estimated number of rows output by this operator (e.g., `Rows=150`). |

**Example Output:**

```
ProjectExec (Cols: name, count) (Cost=50, Rows=10)
└── HashAggregate (Group: T1.id) (Cost=40, Rows=10)
    └── IndexScan (Table: T1, Index: T1_name, Predicate: T1.name = 'XYZ') (Cost=10, Rows=10)
```

This section will detail the specific data structures for the Logical and Physical Plan operators, adhering strictly to the zero-copy, arena-allocated philosophy, and then outline the implementation of the memory-constrained Grace Hash Join.

-----

## 🏗️ Data Structures for Plan Operators

All plan nodes (`LogicalOperator` and `PhysicalOperator`) will be defined using Rust's `enum` to represent the tree structure, and all structs will be parameterized by a lifetime `'a` (e.g., `Plan<'a>`) to indicate that they contain references to the input AST and catalog data, not owned copies.

### 1\. Logical Plan Operators (`LogicalOperator<'a>`)

The logical operators contain references to the **resolved schema/catalog metadata** and the original **AST expressions**.

```rust
// In src/planner/logical_plan.rs

pub enum LogicalOperator<'a> {
    // Corresponds to the base data source
    Scan(LogicalScan<'a>),

    // Selects and orders columns.
    Project(LogicalProject<'a>),

    // Applies a WHERE clause predicate.
    Filter(LogicalFilter<'a>),

    // Groups rows and computes aggregates.
    Aggregate(LogicalAggregate<'a>),

    // Represents a relational join (Inner, Left, etc.).
    Join(LogicalJoin<'a>),

    // Implements ORDER BY
    Sort(LogicalSort<'a>),

    // ... other operators (Limit, Union, Values, etc.)
}

// Example Logical Operator Structs:

// The base relation access.
pub struct LogicalScan<'a> {
    // Reference to the resolved Catalog Table metadata (zero-copy)
    pub table_ref: &'a TableCatalogEntry,

    // Optional list of columns to be read (used for projection pushdown)
    pub required_columns: &'a [ColumnID],
}

// Represents WHERE clause and HAVING clause.
pub struct LogicalFilter<'a> {
    // The operator that provides input rows.
    pub input: &'a LogicalOperator<'a>,

    // Reference to the predicate expression from the AST (zero-copy)
    pub predicate: &'a Expr<'a>,
}
```

### 2\. Physical Plan Operators (`PhysicalOperator<'a>`)

The physical operators detail *how* the data is accessed and processed. They incorporate algorithm choices (e.g., IndexScan vs. TableScan, Grace Hash Join vs. Nested Loop Join).

```rust
// In src/execution/physical_plan.rs

pub enum PhysicalOperator<'a> {
    // Access Paths
    TableScan(PhysicalTableScan<'a>),
    IndexScan(PhysicalIndexScan<'a>),

    // Row Processing
    FilterExec(PhysicalFilterExec<'a>),
    ProjectExec(PhysicalProjectExec<'a>),

    // Joins
    NestedLoopJoin(PhysicalNLJoin<'a>),
    GraceHashJoin(PhysicalGHJoin<'a>),

    // Aggregation
    HashAggregate(PhysicalHashAgg<'a>),
    SortedAggregate(PhysicalSortedAgg<'a>),

    // ... other operators (SortExec, LimitExec, etc.)
}

// Example Physical Operator Structs (Focus on Access Paths):

pub struct PhysicalTableScan<'a> {
    // References the underlying table metadata.
    pub table_ref: &'a TableCatalogEntry,

    // The optional filter expression to apply row-by-row
    // (useful for non-indexed column filters).
    pub post_scan_filter: Option<&'a Expr<'a>>,
}

// Crucial structure for index access.
pub struct PhysicalIndexScan<'a> {
    // References the specific index metadata.
    pub index_ref: &'a IndexCatalogEntry,

    // The columns being retrieved from the base table.
    pub table_ref: &'a TableCatalogEntry,

    // The key range to scan. This is pre-computed into byte-comparable slices.
    // 'Range' should be an enum: FullScan, PrefixScan, RangeScan.
    // The bytes are often references to the arena or constants.
    pub key_range: ScanRange<'a>,

    // A residual filter to apply after the index scan (non-sargable parts).
    pub residual_filter: Option<&'a Expr<'a>>,
}

// The 'Expr' and 'Statement' types are references to the arena-allocated
// AST nodes from the parser, maintaining the zero-copy principle.
```

-----

## 🌀 Grace Hash Join Implementation

The **Grace Hash Join** is selected as the primary hash join algorithm because it explicitly handles the memory limitation by implementing a disk-based partitioning strategy. The **256KB memory budget** is the hard limit for the *in-memory* hash table build phase.

### 1\. Grace Hash Join Phases

Grace Hash Join proceeds in two main phases: **Partitioning** and **Probing**.

#### A. Partitioning Phase (Build & Probe Inputs)

The goal is to partition both the inner (build) and outer (probe) relations into smaller, on-disk buckets such that each bucket is small enough to fit into the 256KB memory budget when processed later.

1.  **Allocate Hash/Partition Buffer:** A dedicated buffer (e.g., 200KB of the 256KB budget) is used to hold rows before they are written to a partition file.
2.  **Hashing:** Both input streams (Inner/Build and Outer/Probe) are read row-by-row. A hash function is applied to the join key: $H(k) = k \pmod{N}$, where $N$ is the number of partitions (e.g., 16).
3.  **Spilling:** Each row is appended to its corresponding partition buffer. When a partition buffer fills (e.g., 16KB per partition), the buffer is written sequentially to a temporary spill file in the `database_dir/wal/` directory: `wal/join_P01.tmp`, `wal/join_P02.tmp`, etc.

#### B. Probing Phase (In-Memory Join)

The system now processes the partitioned files pair by pair:

1.  **Read Build Partition:** The $i$-th build partition file (`wal/join\_B\_P$i$.tmp`) is loaded completely into the 256KB working memory.
      * **Memory Check:** If the partition is larger than 256KB (or the reserved budget), the system must either **fail the query** (initial strategy for simplicity) or recursively partition the data (complex and not recommended for the MVP).
2.  **Build Hash Table:** A standard hash table is built in the arena, mapping the join key to the row data (or a pointer/offset to the row data within the partition buffer).
3.  **Read Probe Partition:** The $i$-th probe partition file (`wal/join\_R\_P$i$.tmp`) is read sequentially.
4.  **Probe & Output:** For each row in the probe partition, the hash table is checked for matching keys. Matches are projected and sent to the next operator.
5.  **Clean Up:** Once the pair of partitions is processed, the memory is freed, and the temporary spill files are deleted.

### 2\. Physical Operator Structure for Grace Hash Join

The `PhysicalGHJoin` operator needs to track the temporary file handles and partitioning scheme.

```rust
pub struct PhysicalGHJoin<'a> {
    // The two inputs to the join.
    pub left_input: &'a PhysicalOperator<'a>,
    pub right_input: &'a PhysicalOperator<'a>,

    // The type of join (Inner, Left, etc.).
    pub join_type: JoinType,

    // The key expression(s) for the join. e.g., T1.a = T2.b
    pub join_keys: &'a [Expr<'a>],

    // Number of partitions (fixed at 16, for example)
    pub num_partitions: u8,

    // Internal state management (not zero-copy, allocated per query)
    // Needs file handles and memory allocation tracking.
    // This is managed by the executor, not the planner.
    // pub state: Mutex<JoinState>
}
```

### 3\. Interaction with 256KB Memory Budget

The 256KB constraint directly dictates the design choice and limits complexity:

| Constraint | Implication on Grace Hash Join |
| :--- | :--- |
| **256KB Limit** | Defines the maximum size of a single in-memory partition. |
| **Zero Allocation During Execution** | The 256KB memory **must** be pre-allocated as a single large `Vec<u8>` (or an arena instance) and reused for loading partitions. The hash table itself must use offsets/indices within this buffer, avoiding standard Rust allocation/deallocation on a per-row basis. |
| **Spilling** | The planner/executor must have logic to manage temporary files for partitions that exceed the in-memory budget. This adds I/O overhead but guarantees the query completes without running out of memory. |

The decision to use a fixed number of partitions (e.g., 16) is a necessary **design simplification** to keep the planning time fast and avoid the need for complex, dynamic partitioning strategies that consume more planning resources.

---

## 🎯 Performance Target Feasibility Assessment

### 1. Planning Time Targets

| Target | Assessment | Architectural Justification |
| :--- | :--- | :--- |
| **Simple Queries: $< 100\mu\text{s}$** | **Achievable.** | The **single/two-phase planning** (AST $\to$ Plan) is primarily an in-memory operation. By using **arena allocation (`bumpalo`)**, there are no per-node memory allocations/deallocations, which is a massive speed win. **Heuristic-based optimization** is much faster than complex cost-based dynamic programming. |
| **Complex Queries: $< 10\text{ms}$** | **Challenging but Achievable.** | Complexity comes from deep ASTs, subquery decorrelation, and join reordering. To meet this: **Greedy join reordering** must be used instead of full CBO exploration. **Plan caching** for prepared statements is essential to make subsequent executions instantaneous. |
| **Achieving Planning Speed** | The Rust language (lack of GC overhead), zero-copy plan structures, and arena allocation are the critical enablers here. The choice of a simple RBO/heuristic optimizer directly supports this target. |

### 2. Execution Time Targets

#### A. Point Reads

| Target | Assessment | Architectural Justification |
| :--- | :--- | :--- |
| **Cached: $< 1\mu\text{s}$** | **Very Difficult but Possible.** | This requires hitting the fastest path: **IndexScan** $\to$ **Page Cache Hit** $\to$ **Zero-Copy Record Slice** retrieval.
    * **B-Tree traversal:** B-Tree lookups are $O(\log N)$, but in memory, this is a few pointer dereferences.
    * **Locking:** The **Lock Sharding (64 shards)** for the page cache is crucial to avoid contention and lock overhead. If the correct page lock is acquired fast, and the page is in the SIEVE cache, the $1\mu\text{s}$ target is realistic for modern CPUs. |
| **Disk: $< 50\mu\text{s}$** | **Highly Dependent on Hardware.** | This relies on: **IndexScan** $\to$ **Page Cache Miss** $\to$ **mmap read/Page Fault**.
    * For a true cold disk read, $<50\mu\text{s}$ requires **extremely fast NVMe SSDs** or the OS read-ahead mechanism to already have the page nearby. On a typical spinning disk or slow SSD, this target is not feasible. The target should be considered achievable only on high-performance storage. |
| **Achieving Point Read Speed** | **Memory-Mapped File I/O**, **Zero-copy access** returning `&[u8]`, and a highly optimized **B-Tree implementation** with SIMD hints are the core technical features that make this path possible. |

#### B. Sequential Scans

| Target | Assessment | Architectural Justification |
| :--- | :--- | :--- |
| **$> 1\text{M rows/sec}$** | **Achievable.** | This is a highly achievable target for high-throughput, low-overhead systems.
    * **Formula Check:** $1\text{M rows/sec}$ means $1 \text{ row} / 1\mu\text{s}$.
    * **Zero-Copy:** The record serialization layer is **zero-copy**. This means the system spends almost zero time on buffer copies or decoding the record; it just provides a slice of the mmap'd file.
    * **Sequential I/O:** The Storage Layer's sequential mmap I/O provides the OS with maximum opportunity for large block reads and read-ahead optimization.
    * **SIEVE Cache:** The SIEVE cache eviction algorithm is specifically optimized for scan-friendly caching, reducing thrashing during large table scans compared to a standard LRU. |
| **Achieving Scan Speed** | The combination of **mmap, fixed 16KB page size**, and the **zero-copy record layer** ensures maximal I/O throughput with minimal CPU overhead, making this target highly feasible. |

#### C. Joins

| Target | Assessment | Architectural Justification |
| :--- | :--- | :--- |
| **Joins leveraging Index Scans** | **Achievable and Essential.** | This means the optimizer must successfully recognize opportunities to use **Nested Loop Join** where the inner side can use an **IndexScan** (e.g., Foreign Key lookups). This replaces slow full table scans on the inner loop with fast B-tree lookups, directly leveraging the fast Point Read performance described above. |
| **Joins (General Case)** | **Feasible, but with Spilling Risk.** | For large non-indexed joins, the system must rely on **Grace Hash Join**. The performance will be determined by:
    * **Memory Fit:** If the join fits within the 256KB working memory, it will be extremely fast (in-memory hash join).
    * **Spilling:** If the data must be spilled (partitioned) to disk, the performance will drop significantly, relying on the speed of sequential I/O to the temporary spill files (`wal/`). |

---

## ✅ Summary of Feasibility

| Target | Feasibility | Key Architectural Enablers |
| :--- | :--- | :--- |
| **Planning Time** | **High** | Rust/Zero-copy Plan, Arena Allocators, Heuristic Optimizer, Plan Caching. |
| **Point Read (Cached)** | **Medium-High** | Zero-Copy Record, SIEVE Cache, Lock Sharding, Optimized B-Tree. |
| **Point Read (Disk)** | **Hardware-Dependent** | mmap I/O, NVMe SSD speed. |
| **Sequential Scan** | **High** | Zero-Copy Record, mmap I/O, SIEVE Cache (scan friendly). |
| **Joins** | **High (for indexed)** | Physical Optimizer choosing Indexed Nested Loop Join. |
