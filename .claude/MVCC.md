This is the critical "brain" of the database. For TurDB, where memory is tight (1MB total) and latency must be low, standard enterprise MVCC (like Postgres's separate tuple heap or MySQL's undo segments) is too heavy.

I recommend a **Single-Writer / Multi-Reader (SWMR)** model for the initial implementation, evolving into **Optimistic Row-Level Locking** if writer throughput becomes a bottleneck. However, given the "Conflict Detection" requirement in your roadmap, I will design for **Row-Level Locking** from the start, but implemented efficiently inside the data pages to avoid external lock managers.

### 1\. High-Level Design Decisions

| Question | Recommended Decision | Rationale for TurDB |
| :--- | :--- | :--- |
| **Isolation** | **Snapshot Isolation (SI)** | Best balance. Consistent reads without the massive memory overhead of Serializable (SSI). |
| **Timestamps** | **A) Single Atomic Counter (u64)** | Simple, fast, zero-alloc. Sufficient for single-node embedded DBs. |
| **Version Storage** | **Delta-based (Inline Newest)** | Store the *newest* version in the B-Tree leaf. Store *deltas* (undo logs) to reconstruct old versions. Keeps scans fast. |
| **Chaining** | **A) Newest-to-Oldest (N2O)** | The B-tree points to the head (newest). Readers walk back if `txn_id > read_ts`. Optimizes for the common case (reading latest). |
| **GC** | **A) Background + Watermark** | Predictable cleanup. Cooperative GC makes read latency erratic. |
| **Locking** | **A) Row-level (In-Header)** | Store the "Lock" (TxnID) directly in the record header. Zero memory overhead for a separate lock table. |
| **Write Set** | **B) SmallVec\<Key, 16\>** | 90% of embedded txns touch \<16 rows. Stack allocate first, spill to heap later. |

-----

### 2\. Core Structures (Rust)

We need to add a header to your `Record` format. This header manages visibility and locking without external structures.

```rust
use std::sync::atomic::{AtomicU64, Ordering};

// 8 bytes for TxnId: Enough for centuries.
pub type TxnId = u64;

/// The Global Transaction Manager
pub struct TransactionManager {
    // Monotonically increasing counter
    global_ts: AtomicU64,
    // Minimum active transaction ID (Watermark for GC)
    min_active_ts: AtomicU64,
    // Track active readers (Conceptually. In 1MB budget, maybe just a counter or small array)
    active_txns: RwLock<Vec<TxnId>>, 
}

/// The Record Header stored at the start of every B-Tree value payload.
/// Layout: [ flags(1) | txn_id(8) | prev_version_ptr(8) ] = 17 bytes overhead
#[repr(C, packed)]
pub struct RecordHeader {
    // Bit 0: Is Deleted?
    // Bit 1: Is Locked? (Write-in-progress)
    pub flags: u8, 
    
    // The Transaction ID that created this version.
    // If Locked bit is set, this is the Owner TxnID.
    pub txn_id: TxnId, 

    // Pointer to the previous version (Undo Log).
    // u64 = (PageID << 16) | Offset
    pub prev_version: u64, 
}

/// Transaction Context (Arena Allocated)
pub struct Transaction<'a> {
    pub id: TxnId,
    pub read_ts: TxnId, // Start timestamp
    pub state: TxnState,
    
    // Track modified keys for Conflict Detection & Commit/Rollback
    // SmallVec avoids allocation for small transactions
    pub write_set: SmallVec<[(TableId, Key); 16]>, 
}
```

-----

### 3\. Workflow Design

#### A. Reading (Visibility Check)

When a cursor lands on a key in the B-Tree:

1.  **Read Header:** Parse the `RecordHeader`.
2.  **Check Visibility:**
      * If `header.txn_id <= my_txn.read_ts`: The version is visible. Return data.
      * If `header.txn_id > my_txn.read_ts`: This version is "from the future".
      * If `header.flags & LOCKED`: This row is being modified by another transaction.
          * *Standard SI:* Ignore the lock, look at the `prev_version`.
3.  **Traverse Chain:** If not visible, follow `header.prev_version`. This points to an "Undo Page" where the old version (or delta) lives. Repeat check.

#### B. Writing (Insert/Update)

1.  **Navigate:** Find the leaf node.
2.  **Locking (Conflict Detection):**
      * Check `RecordHeader`.
      * If `flags & LOCKED`: **Abort** (Write-Write Conflict).
      * If `txn_id > my_txn.read_ts`: **Abort** (Write Skew / Concurrent Modification). The row changed after we started our snapshot.
3.  **Create New Version:**
      * Set `flags |= LOCKED`.
      * Set `txn_id = my_txn.id`.
      * Copy the *current* data to an "Undo Buffer" (new page type).
      * Update `prev_version` to point to that Undo Buffer.
      * Overwrite the data in the B-Tree leaf with the new data.
4.  **Track:** Add `(TableId, Key)` to `write_set`.

#### C. Commit

1.  **Finalize:** Acquire a global commit timestamp (`commit_ts`).
2.  **Unlock:** Iterate through `write_set`.
      * Go to the page.
      * Update `RecordHeader.txn_id` = `commit_ts`.
      * Clear `LOCKED` bit.
3.  **Durability:** Flush WAL (future task).

-----

### 4\. Addressing Specific Constraints

#### Memory Budget (1MB)

  * **The Problem:** Keeping a list of all active transactions and their read sets is expensive.
  * **The Solution:** Do **not** track read sets. This precludes Serializable isolation but fits SI perfectly.
  * **Active Txn List:** Limit concurrent transactions to a fixed number (e.g., 64). `64 * sizeof(u64)` is tiny. If the slot array is full, reject new transactions (`BUSY`).

#### Zero-Copy / Zero-Allocation

  * **Reads:** The `RecordHeader` is read directly from the memory-mapped slice. No decoding needed if you use `#[repr(C)]`.
  * **Writes:** The "Undo Log" needs to be written. This requires finding a free slot in a page. Use your existing `Bump` allocator or `Freelist` to grab space in a `UndoPage`.

#### File Layout

  * **Undo Storage:** Don't mix Undo records with B-Tree data nodes. Create a specific page type `PageType::Undo`.
  * These pages are transient. They can be aggressively reused/overwritten once the `min_active_ts` moves past them.

### 5\. Detailed Answers to Your Design Questions

**1. Isolation Level: Snapshot Isolation (SI)**

  * *Why:* It is the standard for modern engines (Postgres, Oracle, InnoDB default). "Read Committed" is too inconsistent for complex apps. "Serializable" requires read-locks which kill concurrency and blow up memory.

**2. Timestamp Strategy: Single Atomic Counter**

  * *Implementation:* `AtomicU64`.
  * *Gap:* Yes, there is a gap between start and commit, but that's fine.
      * `start_ts = global.load()`
      * `commit_ts = global.fetch_add(1)`

**3. Version Storage: Inline Newest + Undo Log**

  * *Layout:*
      * **Main B-Tree:** Contains the *latest uncommitted* or *latest committed* version. This optimizes for the most common query: `SELECT * FROM table` (scanning current data).
      * **Undo Pages:** Store the *previous* images.
  * *Trade-off:* Updates require writing to two places (Undo Page + Main Page), but Reads are O(1) for the latest version.

**4. Version Chain: Newest-to-Oldest**

  * *Why:* Matches the storage layout. We start at the B-Tree (Newest) and walk backwards if necessary.

**5. Garbage Collection: Background Watermark**

  * *Strategy:* Maintain `global_min_active_ts`.
  * Every N operations (or triggered by a timer), scan the "Undo Pages".
  * Any undo record with `txn_id < global_min_active_ts` is unreachable by any active transaction. Mark that slot in the Undo Page as free.

**6. Transaction ID Width: u64**

  * `u64` is effectively infinite. `u128` is waste (16 bytes per row overhead is huge for 16KB pages). `u32` wraps too fast (4 billion is reachable in days at 100k/sec).

**7. Write Set Tracking: SmallVec\<Key, 16\>**

  * Allocating a `HashSet` on the heap for every transaction violates "Zero Allocation".
  * Use `SmallVec`. If a transaction grows beyond 16 writes, fall back to a bump-allocated linked list or a sorted vector (binary search is fast for \<100 items).

**8. Conflict Detection: Row-level (Header Bit)**

  * By using the `LOCKED` bit in the `RecordHeader`, you effectively have **zero-memory locks**. The data page *is* the lock table.
  * *Downside:* You must load the page to check the lock. (Acceptable for embedded DB).

**9. Savepoints: No (Defer)**

  * Keep V1 simple. Rollback affects the whole transaction.

**10. WAL Integration: Unified**

  * The WAL should log the *physical* page changes.
  * Since MVCC changes data (setting lock bits, updating pointers), these are just page updates. The WAL doesn't strictly need to know about "Versions", it just logs "Page X offset Y changed to bytes Z".

-----

### 6\. Diagram: The Version Chain

*(Self-correction: I cannot generate the image, but here is the ASCII representation)*

```text
       [ B-Tree Leaf Page (Data.tbd) ]
       +-----------------------------+
       | Page Header                 |
       +-----------------------------+
       | Slot 0: Key="user:1"        |
       | Value:                      |
       |  [Header]                   |
       |    Flags: 0x00              |
       |    TxnId: 105               |
       |    Prev:  Page 4, Off 64  -----\
       |  [Body: "Alice", 30]        |     \
       +-----------------------------+      \
                                             \
                                              \   [ Undo Page 4 (Data.tbd or separate) ]
                                               \  +-----------------------------+
                                                ->| Offset 64:                  |
                                                  |  [Header]                   |
                                                  |    Flags: 0x00              |
                                                  |    TxnId: 100               |
                                                  |    Prev:  NULL              |
                                                  |  [Body: "Alice", 29]        |
                                                  +-----------------------------+
```

### 7\. Action Plan (Next Steps)

1.  **Modify `Record` Serialization:** Update `records/` to reserve space for the 17-byte `RecordHeader` at the start of every payload.
2.  **Define `TransactionManager`:** Create the struct in `src/concurrency/mod.rs` with the atomic counters.
3.  **Implement `UndoPage`:** A simple append-only page format to store old versions.
4.  **Update B-Tree Cursor:** Modify the cursor read logic to parse the header and respect `read_ts`.


Here are the detailed Rust struct definitions designed for **TurDB's** constraints: **Zero-Allocation**, **1MB Memory Budget**, and **Lock-Free Read Paths**.

### 1\. `RecordHeader` (The Storage Layout)

This header sits at the very beginning of the value payload in your B-Tree leaf. We use `#[repr(C)]` for predictable layout, but in practice, you will likely interpret these bytes directly from the raw memory-mapped slice to avoid copying.

```rust
use std::mem;

/// 8-byte Transaction ID.
/// 0 is reserved for "Always Visible" (bootstrapped data).
pub type TxnId = u64;

/// Constants for Bitmask Flags
pub mod flags {
    pub const LOCK_BIT: u8    = 0b0000_0001; // Row is currently locked by a writer
    pub const DELETE_BIT: u8  = 0b0000_0010; // Row is logically deleted
    pub const VACUUM_BIT: u8  = 0b0000_0100; // Hint: Row is safe to garbage collect
}

/// The 17-byte header prepended to every row's value.
/// Layout: [Flags (1B) | TxnID (8B) | PrevPtr (8B)]
/// 
/// We do NOT use packed structs directly for access to avoid unaligned load faults 
/// on some architectures (ARM/WASM), opting for byte-parsing helpers instead.
#[derive(Debug, Clone, Copy)]
pub struct RecordHeader {
    pub flags: u8,
    pub txn_id: TxnId,
    pub prev_version: u64, // (PageID << 16) | Offset
}

impl RecordHeader {
    pub const SIZE: usize = 1 + 8 + 8; // 17 bytes

    /// Zero-copy parser: Reads header directly from a raw byte slice.
    #[inline(always)]
    pub fn from_bytes(slice: &[u8]) -> Self {
        debug_assert!(slice.len() >= Self::SIZE);
        
        // Safety: We manually reconstruct u64s to handle potential unaligned memory
        // in the mmap buffer without crashing (common in packed formats).
        let flags = slice[0];
        
        let txn_id = u64::from_be_bytes(slice[1..9].try_into().unwrap());
        let prev_version = u64::from_be_bytes(slice[9..17].try_into().unwrap());

        Self {
            flags,
            txn_id,
            prev_version,
        }
    }

    /// Serializes the header into a mutable slice (for writing to WAL/Page).
    #[inline(always)]
    pub fn write_to(&self, slice: &mut [u8]) {
        debug_assert!(slice.len() >= Self::SIZE);
        
        slice[0] = self.flags;
        slice[1..9].copy_from_slice(&self.txn_id.to_be_bytes());
        slice[9..17].copy_from_slice(&self.prev_version.to_be_bytes());
    }

    /// Helper to encode PageID (u48) and Offset (u16) into the prev pointer.
    pub fn encode_ptr(page_id: u64, offset: u16) -> u64 {
        (page_id << 16) | (offset as u64)
    }

    pub fn decode_ptr(ptr: u64) -> (u64, u16) {
        (ptr >> 16, (ptr & 0xFFFF) as u16)
    }

    pub fn is_locked(&self) -> bool { self.flags & flags::LOCK_BIT != 0 }
    pub fn is_deleted(&self) -> bool { self.flags & flags::DELETE_BIT != 0 }
}
```

-----

### 2\. `TransactionManager` (The Concurrency Controller)

To meet the **1MB Memory Budget** and **Zero-Allocation** constraint, we cannot use a dynamic `Vec` or `HashMap` to track active transactions.

Instead, we use a **Fixed-Size Slot Array** of atomic counters.

  * **Max Concurrent Txns:** 64 (Hard limit).
  * **Memory Overhead:** `64 * 8 bytes = 512 bytes`. Negligible.
  * **Lock-Free Watermark Calculation:** We can find the global `min_active_ts` by simply iterating this array without a global mutex.

<!-- end list -->

```rust
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex; // Only for slot allocation, not hot path

const MAX_CONCURRENT_TXNS: usize = 64;

pub struct TransactionManager {
    /// The global clock. Increments on every commit.
    global_ts: AtomicU64,

    /// Fixed slots for active transaction timestamps.
    /// Value 0 = Slot Empty.
    /// Value > 0 = Start Timestamp of an active transaction.
    active_slots: [AtomicU64; MAX_CONCURRENT_TXNS],

    /// Mutex to protect finding/claiming a free slot.
    /// This is the only lock, and it's only held during begin()/commit().
    slot_lock: Mutex<()>,
}

impl TransactionManager {
    pub fn new() -> Self {
        // Initialize all slots to 0 (Empty)
        // We use a safe array initialization trick
        let active_slots = [0; MAX_CONCURRENT_TXNS].map(|_| AtomicU64::new(0));

        Self {
            global_ts: AtomicU64::new(1), // Start at 1, 0 is reserved
            active_slots,
            slot_lock: Mutex::new(()),
        }
    }

    /// Begin a new transaction.
    /// Returns: (TxnId, SlotIndex)
    pub fn begin_txn(&self) -> Result<(TxnId, usize), &'static str> {
        let _guard = self.slot_lock.lock().unwrap();

        // 1. Get current logical time (Snapshot Point)
        let start_ts = self.global_ts.load(Ordering::SeqCst);

        // 2. Find a free slot to register this transaction
        for (idx, slot) in self.active_slots.iter().enumerate() {
            if slot.load(Ordering::Relaxed) == 0 {
                // Claim slot
                slot.store(start_ts, Ordering::SeqCst);
                return Ok((start_ts, idx));
            }
        }

        Err("Too many concurrent transactions (Max 64)")
    }

    /// Commit transaction.
    /// Returns: The Commit Timestamp (new global_ts)
    pub fn commit_txn(&self, slot_idx: usize) -> TxnId {
        // 1. Generate new Commit TS
        let commit_ts = self.global_ts.fetch_add(1, Ordering::SeqCst) + 1;

        // 2. Release the slot (Mark as 0)
        // We don't need the lock to release, just atomic store.
        self.active_slots[slot_idx].store(0, Ordering::SeqCst);

        commit_ts
    }

    /// Rollback/Abort transaction.
    pub fn abort_txn(&self, slot_idx: usize) {
        // Just release the slot.
        self.active_slots[slot_idx].store(0, Ordering::SeqCst);
    }

    /// Calculates the "Watermark" (Global Minimum Active Timestamp).
    /// Any version with txn_id < watermark is safe to Garbage Collect.
    pub fn get_global_watermark(&self) -> TxnId {
        let mut min_ts = self.global_ts.load(Ordering::Relaxed);

        for slot in &self.active_slots {
            let ts = slot.load(Ordering::Relaxed);
            if ts != 0 && ts < min_ts {
                min_ts = ts;
            }
        }
        min_ts
    }
}
```

-----

### 3\. `Transaction` (The Context Object)

This struct lives in the **stack** (or arena) during execution. It holds the write set. For the **write set**, we use `SmallVec` to keep allocation off the heap for small transactions (common in embedded use cases).

```rust
use smallvec::SmallVec;

/// Represents a key modified by this transaction.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WriteKey {
    pub table_id: u32,
    // We store the Key as an owned Vec<u8> or a fixed size hash.
    // For Zero-Copy, if the key is long, we might just store the PageID + SlotID
    // where we placed the lock. Let's assume RowID/Key for now.
    pub key: Vec<u8>, 
}

pub struct Transaction<'a> {
    /// The unique ID for this transaction (also its Read Timestamp).
    pub id: TxnId,
    
    /// Which slot in the Manager belongs to us? Needed for commit/abort.
    pub slot_idx: usize,
    
    /// Reference to the global manager.
    manager: &'a TransactionManager,

    /// Track writes for Conflict Detection and Rollback.
    /// Inline capacity of 16 means no heap allocs for 90% of txns.
    pub write_set: SmallVec<[WriteKey; 16]>,
    
    /// State tracking
    pub committed: bool,
}

impl<'a> Transaction<'a> {
    pub fn new(manager: &'a TransactionManager) -> Result<Self, &'static str> {
        let (id, slot_idx) = manager.begin_txn()?;
        Ok(Self {
            id,
            slot_idx,
            manager,
            write_set: SmallVec::new(),
            committed: false,
        })
    }

    pub fn commit(mut self) {
        // 1. Apply Commit Timestamp to all locked records (WAL + Page Update)
        //    (This logic belongs in the Executor/Storage layer, traversing write_set)
        
        // 2. Release slot
        self.manager.commit_txn(self.slot_idx);
        self.committed = true;
    }
}

// Ensure slot is released if Transaction is dropped (Panics/Early Return)
impl<'a> Drop for Transaction<'a> {
    fn drop(&mut self) {
        if !self.committed {
            self.manager.abort_txn(self.slot_idx);
            // In a real impl, we would also trigger the "Undo" process here
            // using the write_set to revert changes.
        }
    }
}
```

### Key Design Benefits for Your Architecture:

1.  **Alignment Safety:** `RecordHeader::from_bytes` handles unaligned reads, which is crucial when mapping raw pages where a record might start at an odd offset.
2.  **No Dynamic Allocation:** `TransactionManager` uses a fixed array. It never calls `malloc`.
3.  **Fast Garbage Collection:** `get_global_watermark` is $O(64)$ — effectively constant time. You can call this frequently (e.g., every 100 inserts) to reclaim Undo pages without stalling the system.
4.  **Zero-Copy Reads:** The `RecordHeader` isn't "read" into a struct; it's just a view logic over the `&[u8]`. The data remains in the mmap cache.