//! # B+Tree Implementation
//!
//! This module implements the core B+tree data structure for TurDB's row storage.
//! The B+tree provides ordered key-value storage with O(log N) search, insert,
//! and delete operations.
//!
//! ## Architecture Overview
//!
//! The B+tree is a balanced tree structure where:
//! - All data (key-value pairs) is stored in leaf nodes
//! - Interior nodes contain separator keys and child page pointers
//! - All leaf nodes are at the same depth (balanced)
//! - Leaf nodes are linked for efficient range scans
//!
//! ## Page-Based Storage
//!
//! Each node occupies exactly one 16KB page. The tree stores page numbers
//! rather than memory pointers, enabling persistence and memory-mapped access.
//!
//! ```text
//!                    [Interior Page 1]
//!                    /       |        \
//!           [Leaf 2]    [Leaf 3]    [Leaf 4]
//!              |------------>|---------->|  (linked list)
//! ```
//!
//! ## Zero-Copy Design
//!
//! Search operations return `SearchHandle` containing page number and cell index.
//! The caller can then access the value via `get_value()` which returns `&[u8]`
//! pointing directly into mmap'd memory:
//!
//! ```text
//! if let Some(handle) = btree.search(key)? {
//!     let value: &[u8] = btree.get_value(&handle)?;  // Zero-copy!
//! }
//! ```
//!
//! ## Memory Efficiency
//!
//! - Path stack uses `SmallVec<[u32; 8]>` - stack-allocated for trees up to 8 levels
//! - No heap allocation during search operations
//! - Split operations use per-operation `bumpalo` arena allocators
//!
//! ### Arena-Based Split Operations
//!
//! Node splits use ephemeral `bumpalo::Bump` arenas for temporary key/value storage
//! during redistribution. This design:
//! - Avoids heap fragmentation from many small allocations
//! - Deallocates all temporary storage in O(1) when arena is dropped
//! - Keeps the memory locality of temporary data high (single allocation block)
//!
//! The arena is created at split start and dropped at function end. Only the
//! separator key (which must outlive the split) uses heap allocation.
//!
//! ## Free Page Management
//!
//! The tree integrates with a freelist for page reuse:
//! - New pages allocated from freelist before growing file
//! - Prevents file bloat during mixed insert/delete workloads
//!
//! ## Cursor for Range Scans
//!
//! The `Cursor` struct enables efficient ordered iteration:
//! - Walks leaf pages via next_leaf pointers
//! - Zero-copy key/value access
//! - Uses madvise(MADV_WILLNEED) to prefetch upcoming pages
//!
//! ### Iteration Performance
//!
//! | Operation   | Within Page | Page Boundary |
//! |-------------|-------------|---------------|
//! | `advance()` | O(1)        | O(1)          |
//! | `prev()`    | O(1)        | O(log N)      |
//!
//! Forward iteration (`advance()`) is O(1) because leaf nodes are singly-linked
//! via `next_leaf` pointers. Backward iteration (`prev()`) requires O(log N)
//! tree re-traversal when crossing page boundaries because there is no
//! `prev_leaf` pointer.
//!
//! Future optimization: Add `prev_leaf` pointer to leaf header for O(1)
//! backward traversal if backward scans become a bottleneck.
//!
//! ## Node Splitting
//!
//! When a leaf node becomes full during insertion:
//! 1. Allocate new leaf page (from freelist or grow)
//! 2. Move upper half of keys to new leaf
//! 3. Compute separator key (suffix truncated for efficiency)
//! 4. Insert separator into parent interior node
//! 5. If parent is full, split recursively up the tree
//!
//! ## Delete Algorithm
//!
//! Simple deletion without rebalancing:
//! 1. Search for key in leaf
//! 2. If not found: return false
//! 3. Delete cell from leaf
//!
//! ### Page Reclamation
//!
//! Deleted pages are NOT automatically returned to the freelist because:
//! - Would require updating parent interior nodes (complex)
//! - Would require tracking which pages became empty
//! - Most workloads don't benefit (inserts often follow deletes)
//!
//! For delete-heavy workloads, consider:
//! - Periodic offline compaction (rebuild tree from cursor scan)
//! - Using a separate tombstone mechanism with background cleanup
//! - Accepting some space overhead in exchange for simpler code
//!
//! ## Thread Safety
//!
//! BTree is not thread-safe. External synchronization required. For concurrent
//! access, wrap in RwLock or use the higher-level Table API.
//!
//! ## Capacity
//!
//! With 16KB pages:
//! - Leaf nodes: ~100-1000 entries depending on key/value sizes
//! - Interior nodes: ~800+ children with suffix truncation
//! - Tree depth for 1M rows: typically 2-3 levels (fits in SmallVec)

use bumpalo::collections::CollectIn;
use bumpalo::collections::Vec as BumpVec;
use bumpalo::Bump;
use eyre::{bail, ensure, Result};
use smallvec::SmallVec;

use super::interior::{InteriorNode, InteriorNodeMut, INTERIOR_SLOT_SIZE};
use super::leaf::{LeafNode, LeafNodeMut, SearchResult, LEAF_CONTENT_START, SLOT_SIZE};
use crate::encoding::varint::varint_len;
use crate::storage::{Freelist, MmapStorage, PageHeader, PageType, Storage, PAGE_SIZE};

pub const MAX_TREE_DEPTH: usize = 8;

type PathStack = SmallVec<[u32; MAX_TREE_DEPTH]>;

#[derive(Debug)]
pub struct BTree<'a, S: Storage> {
    storage: &'a mut S,
    root_page: u32,
    freelist: Option<&'a mut Freelist>,
    /// Hint for the rightmost leaf page (PostgreSQL-style fastpath optimization).
    /// When set, insert() will first try this page for sequential inserts,
    /// avoiding tree traversal for monotonically increasing keys.
    rightmost_hint: Option<u32>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SearchHandle {
    pub page_no: u32,
    pub cell_index: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum InsertResult {
    Ok,
    Split { separator: Vec<u8>, new_page: u32 },
}

pub struct Cursor<'a, S: Storage + ?Sized> {
    storage: &'a S,
    root_page: u32,
    current_page: u32,
    current_index: usize,
    exhausted: bool,
}

pub struct BTreeReader<'a> {
    storage: &'a MmapStorage,
    root_page: u32,
}

impl<'a> BTreeReader<'a> {
    pub fn new(storage: &'a MmapStorage, root_page: u32) -> Result<Self> {
        ensure!(
            root_page < storage.page_count(),
            "root page {} out of bounds (page_count={})",
            root_page,
            storage.page_count()
        );
        Ok(Self { storage, root_page })
    }

    pub fn cursor_first(&self) -> Result<Cursor<'a, MmapStorage>> {
        let mut current_page = self.root_page;

        loop {
            let page_data = self.storage.page(current_page)?;
            let header = PageHeader::from_bytes(page_data)?;

            match header.page_type() {
                PageType::BTreeLeaf => {
                    let leaf = LeafNode::from_page(page_data)?;
                    let exhausted = leaf.cell_count() == 0;
                    return Ok(Cursor {
                        storage: self.storage,
                        root_page: self.root_page,
                        current_page,
                        current_index: 0,
                        exhausted,
                    });
                }
                PageType::BTreeInterior => {
                    let interior = InteriorNode::from_page(page_data)?;
                    if interior.cell_count() == 0 {
                        current_page = interior.right_child();
                    } else {
                        current_page = interior.slot_at(0)?.child_page();
                    }
                }
                _ => bail!(
                    "unexpected page type {:?} during cursor_first at page {}",
                    header.page_type(),
                    current_page
                ),
            }
        }
    }

    pub fn cursor_last(&self) -> Result<Cursor<'a, MmapStorage>> {
        let mut current_page = self.root_page;

        loop {
            let page_data = self.storage.page(current_page)?;
            let header = PageHeader::from_bytes(page_data)?;

            match header.page_type() {
                PageType::BTreeLeaf => {
                    let leaf = LeafNode::from_page(page_data)?;
                    let cell_count = leaf.cell_count() as usize;
                    if cell_count == 0 {
                        return Ok(Cursor {
                            storage: self.storage,
                            root_page: self.root_page,
                            current_page,
                            current_index: 0,
                            exhausted: true,
                        });
                    }
                    return Ok(Cursor {
                        storage: self.storage,
                        root_page: self.root_page,
                        current_page,
                        current_index: cell_count - 1,
                        exhausted: false,
                    });
                }
                PageType::BTreeInterior => {
                    let interior = InteriorNode::from_page(page_data)?;
                    current_page = interior.right_child();
                }
                _ => bail!(
                    "unexpected page type {:?} during cursor_last at page {}",
                    header.page_type(),
                    current_page
                ),
            }
        }
    }
}

impl<'a, S: Storage> BTree<'a, S> {
    pub fn new(storage: &'a mut S, root_page: u32) -> Result<Self> {
        ensure!(
            root_page < storage.page_count(),
            "root page {} out of bounds (page_count={})",
            root_page,
            storage.page_count()
        );
        Ok(Self {
            storage,
            root_page,
            freelist: None,
            rightmost_hint: None,
        })
    }

    /// Creates a BTree with a hint for the rightmost leaf page.
    /// This enables fastpath optimization for bulk sequential inserts.
    pub fn with_rightmost_hint(storage: &'a mut S, root_page: u32, hint: Option<u32>) -> Result<Self> {
        ensure!(
            root_page < storage.page_count(),
            "root page {} out of bounds (page_count={})",
            root_page,
            storage.page_count()
        );
        Ok(Self {
            storage,
            root_page,
            freelist: None,
            rightmost_hint: hint,
        })
    }

    pub fn with_freelist(
        storage: &'a mut S,
        root_page: u32,
        freelist: &'a mut Freelist,
    ) -> Result<Self> {
        ensure!(
            root_page < storage.page_count(),
            "root page {} out of bounds (page_count={})",
            root_page,
            storage.page_count()
        );
        Ok(Self {
            storage,
            root_page,
            freelist: Some(freelist),
            rightmost_hint: None,
        })
    }

    pub fn create(storage: &'a mut S, root_page: u32) -> Result<Self> {
        ensure!(
            root_page < storage.page_count(),
            "root page {} out of bounds (page_count={})",
            root_page,
            storage.page_count()
        );

        let page = storage.page_mut(root_page)?;
        LeafNodeMut::init(page)?;

        Ok(Self {
            storage,
            root_page,
            freelist: None,
            rightmost_hint: None,
        })
    }

    pub fn root_page(&self) -> u32 {
        self.root_page
    }

    pub fn search(&self, key: &[u8]) -> Result<Option<SearchHandle>> {
        let mut current_page = self.root_page;

        loop {
            let page_data = self.storage.page(current_page)?;
            let header = PageHeader::from_bytes(page_data)?;

            match header.page_type() {
                PageType::BTreeLeaf => {
                    let leaf = LeafNode::from_page(page_data)?;
                    match leaf.find_key(key) {
                        SearchResult::Found(idx) => {
                            return Ok(Some(SearchHandle {
                                page_no: current_page,
                                cell_index: idx,
                            }));
                        }
                        SearchResult::NotFound(_) => {
                            return Ok(None);
                        }
                    }
                }
                PageType::BTreeInterior => {
                    let interior = InteriorNode::from_page(page_data)?;
                    let (child_page, _) = interior.find_child(key)?;
                    current_page = child_page;
                }
                _ => bail!(
                    "unexpected page type {:?} during search at page {}",
                    header.page_type(),
                    current_page
                ),
            }
        }
    }

    pub fn get_value(&self, handle: &SearchHandle) -> Result<&[u8]> {
        let page_data = self.storage.page(handle.page_no)?;
        let leaf = LeafNode::from_page(page_data)?;
        leaf.value_at(handle.cell_index)
    }

    pub fn get_key(&self, handle: &SearchHandle) -> Result<&[u8]> {
        let page_data = self.storage.page(handle.page_no)?;
        let leaf = LeafNode::from_page(page_data)?;
        leaf.key_at(handle.cell_index)
    }

    pub fn get(&self, key: &[u8]) -> Result<Option<&[u8]>> {
        match self.search(key)? {
            Some(handle) => Ok(Some(self.get_value(&handle)?)),
            None => Ok(None),
        }
    }

    pub fn insert(&mut self, key: &[u8], value: &[u8]) -> Result<()> {
        // PostgreSQL-style fastpath optimization: try the cached rightmost leaf first.
        // This avoids tree traversal for monotonically increasing keys.
        if let Some(hint_page) = self.rightmost_hint {
            if let Ok(true) = self.try_fastpath_insert(hint_page, key, value) {
                return Ok(());
            }
            // Fastpath failed, fall back to normal insert
        }

        let mut path: PathStack = SmallVec::new();
        let mut current_page = self.root_page;

        loop {
            let page_data = self.storage.page(current_page)?;
            let header = PageHeader::from_bytes(page_data)?;

            match header.page_type() {
                PageType::BTreeLeaf => break,
                PageType::BTreeInterior => {
                    let interior = InteriorNode::from_page(page_data)?;
                    let (child_page, _) = interior.find_child(key)?;
                    path.push(current_page);
                    current_page = child_page;
                }
                _ => bail!(
                    "unexpected page type {:?} during insert at page {}",
                    header.page_type(),
                    current_page
                ),
            }
        }

        let result = self.insert_into_leaf(current_page, key, value)?;

        // Update rightmost hint after successful insert
        self.update_rightmost_hint(current_page);

        if let InsertResult::Split {
            separator,
            new_page,
        } = result
        {
            self.propagate_split(path, &separator, current_page, new_page)?;
            // After split, the new page becomes the rightmost
            self.update_rightmost_hint(new_page);
        }

        Ok(())
    }

    /// Attempts a fastpath insert into the hinted rightmost leaf page.
    /// Returns Ok(true) if successful, Ok(false) if fastpath conditions not met.
    fn try_fastpath_insert(&mut self, hint_page: u32, key: &[u8], value: &[u8]) -> Result<bool> {
        // Verify hint page is valid and is a leaf
        if hint_page >= self.storage.page_count() {
            return Ok(false);
        }

        let page_data = self.storage.page(hint_page)?;
        let header = PageHeader::from_bytes(page_data)?;

        if header.page_type() != PageType::BTreeLeaf {
            return Ok(false);
        }

        let leaf = LeafNode::from_page(page_data)?;

        // Check if this is still the rightmost leaf (next_leaf == 0)
        if leaf.next_leaf() != 0 {
            return Ok(false);
        }

        // Check if key is greater than all existing keys (should go at the end)
        let cell_count = leaf.cell_count() as usize;
        if cell_count > 0 {
            let last_key = leaf.key_at(cell_count - 1)?;
            if key <= last_key {
                return Ok(false); // Key doesn't belong at the end
            }
        }

        // Check if there's enough space
        let value_len_size = varint_len(value.len() as u64);
        let cell_size = key.len() + value_len_size + value.len();
        let space_needed = cell_size + SLOT_SIZE;

        if (leaf.free_space() as usize) < space_needed {
            return Ok(false); // Not enough space, need split
        }

        // Fastpath conditions met! Insert directly.
        let page_data = self.storage.page_mut(hint_page)?;
        let mut leaf = LeafNodeMut::from_page(page_data)?;
        leaf.insert_cell(key, value)?;

        Ok(true)
    }

    /// Updates the rightmost hint if the given page is the rightmost leaf.
    fn update_rightmost_hint(&mut self, page_no: u32) {
        if let Ok(page_data) = self.storage.page(page_no) {
            if let Ok(leaf) = LeafNode::from_page(page_data) {
                if leaf.next_leaf() == 0 {
                    self.rightmost_hint = Some(page_no);
                }
            }
        }
    }

    /// Returns the current rightmost hint for use in subsequent BTree instances.
    pub fn rightmost_hint(&self) -> Option<u32> {
        self.rightmost_hint
    }

    fn insert_into_leaf(&mut self, page_no: u32, key: &[u8], value: &[u8]) -> Result<InsertResult> {
        let page_data = self.storage.page_mut(page_no)?;
        let mut leaf = LeafNodeMut::from_page(page_data)?;

        let value_len_size = varint_len(value.len() as u64);
        let cell_size = key.len() + value_len_size + value.len();
        let space_needed = cell_size + SLOT_SIZE;

        if leaf.free_space() as usize >= space_needed {
            leaf.insert_cell(key, value)?;
            return Ok(InsertResult::Ok);
        }

        self.split_leaf(page_no, key, value)
    }

    fn split_leaf(&mut self, page_no: u32, key: &[u8], value: &[u8]) -> Result<InsertResult> {
        let arena = Bump::new();
        let new_page_no = self.allocate_page()?;

        let mut all_keys: BumpVec<&[u8]> = BumpVec::new_in(&arena);
        let mut all_values: BumpVec<&[u8]> = BumpVec::new_in(&arena);

        {
            let page_data = self.storage.page(page_no)?;
            let leaf = LeafNode::from_page(page_data)?;
            let count = leaf.cell_count() as usize;

            for i in 0..count {
                all_keys.push(arena.alloc_slice_copy(leaf.key_at(i)?));
                all_values.push(arena.alloc_slice_copy(leaf.value_at(i)?));
            }
        }

        let insert_pos = all_keys
            .iter()
            .position(|k| *k > key)
            .unwrap_or(all_keys.len());
        all_keys.insert(insert_pos, arena.alloc_slice_copy(key));
        all_values.insert(insert_pos, arena.alloc_slice_copy(value));

        let old_next_leaf;
        {
            let page_data = self.storage.page(page_no)?;
            let leaf = LeafNode::from_page(page_data)?;
            old_next_leaf = leaf.next_leaf();
        }

        // Calculate cell sizes for proper split point selection
        let cell_sizes: BumpVec<usize> = all_keys
            .iter()
            .zip(all_values.iter())
            .map(|(k, v)| k.len() + varint_len(v.len() as u64) + v.len() + SLOT_SIZE)
            .collect_in(&arena);
        let page_capacity = PAGE_SIZE - LEAF_CONTENT_START;

        // PostgreSQL-style optimization: use 90/10 split for rightmost leaf.
        // This optimizes for monotonically increasing keys (e.g., auto-increment, timestamps)
        // by leaving more space in the new rightmost page for future inserts.
        let is_rightmost = old_next_leaf == 0;
        let mut mid = if is_rightmost {
            // Target: ~10% in old page, ~90% in new page
            // But must ensure both pages fit their data
            (all_keys.len() / 10).max(1)
        } else {
            // Normal 50/50 split for non-rightmost pages
            all_keys.len() / 2
        };

        // Ensure the right page can hold its data by moving split point left if needed
        loop {
            let right_size: usize = cell_sizes[mid..].iter().sum();
            if right_size <= page_capacity || mid >= all_keys.len() - 1 {
                break;
            }
            mid += 1;
        }

        // Ensure the left page can hold its data by moving split point right if needed  
        while mid > 1 {
            let left_size: usize = cell_sizes[..mid].iter().sum();
            if left_size <= page_capacity {
                break;
            }
            mid -= 1;
        }

        {
            let page_data = self.storage.page_mut(page_no)?;
            let mut leaf = LeafNodeMut::init(page_data)?;
            for i in 0..mid {
                leaf.insert_cell(all_keys[i], all_values[i])?;
            }
            leaf.set_next_leaf(new_page_no)?;
        }

        let separator_key;
        {
            let page_data = self.storage.page_mut(new_page_no)?;
            let mut new_leaf = LeafNodeMut::init(page_data)?;
            for i in mid..all_keys.len() {
                new_leaf.insert_cell(all_keys[i], all_values[i])?;
            }
            new_leaf.set_next_leaf(old_next_leaf)?;

            separator_key = all_keys[mid].to_vec();
        }

        Ok(InsertResult::Split {
            separator: separator_key,
            new_page: new_page_no,
        })
    }

    fn propagate_split(
        &mut self,
        mut path: PathStack,
        separator: &[u8],
        left_child: u32,
        right_child: u32,
    ) -> Result<()> {
        let mut current_separator = separator.to_vec();
        let mut current_left = left_child;
        let mut current_right = right_child;

        while let Some(parent_page) = path.pop() {
            let result = self.insert_into_interior(
                parent_page,
                &current_separator,
                current_left,
                current_right,
            )?;

            match result {
                InsertResult::Ok => return Ok(()),
                InsertResult::Split {
                    separator,
                    new_page,
                } => {
                    current_separator = separator;
                    current_left = parent_page;
                    current_right = new_page;
                }
            }
        }

        self.create_new_root(&current_separator, current_left, current_right)
    }

    fn insert_into_interior(
        &mut self,
        page_no: u32,
        separator: &[u8],
        _left_child: u32,
        right_child: u32,
    ) -> Result<InsertResult> {
        let page_data = self.storage.page_mut(page_no)?;
        let mut interior = InteriorNodeMut::from_page(page_data)?;

        let space_needed = separator.len() + INTERIOR_SLOT_SIZE;

        if interior.free_space() as usize >= space_needed {
            let old_right = interior.right_child();

            if separator
                >= interior
                    .key_at(interior.cell_count() as usize - 1)
                    .unwrap_or(separator)
            {
                interior.insert_separator(separator, old_right)?;
                interior.set_right_child(right_child)?;
            } else {
                interior.insert_separator(separator, right_child)?;
            }
            return Ok(InsertResult::Ok);
        }

        self.split_interior(page_no, separator, right_child)
    }

    fn split_interior(
        &mut self,
        page_no: u32,
        new_separator: &[u8],
        new_right_child: u32,
    ) -> Result<InsertResult> {
        let arena = Bump::new();
        let new_page_no = self.allocate_page()?;

        let mut all_separators: BumpVec<&[u8]> = BumpVec::new_in(&arena);
        let mut all_children: BumpVec<u32> = BumpVec::new_in(&arena);
        let old_right_child: u32;

        {
            let page_data = self.storage.page(page_no)?;
            let interior = InteriorNode::from_page(page_data)?;
            let count = interior.cell_count() as usize;

            for i in 0..count {
                all_separators.push(arena.alloc_slice_copy(interior.key_at(i)?));
                all_children.push(interior.slot_at(i)?.child_page());
            }
            old_right_child = interior.right_child();
        }

        let insert_pos = all_separators
            .iter()
            .position(|s| *s > new_separator)
            .unwrap_or(all_separators.len());

        all_separators.insert(insert_pos, arena.alloc_slice_copy(new_separator));

        if insert_pos == all_children.len() {
            all_children.push(old_right_child);
        } else {
            all_children.insert(insert_pos + 1, new_right_child);
        }

        let mid = all_separators.len() / 2;
        let promoted_separator = all_separators[mid].to_vec();

        {
            let page_data = self.storage.page_mut(page_no)?;
            let first_child = all_children[0];
            let mut interior = InteriorNodeMut::init(page_data, all_children[mid])?;

            for i in 0..mid {
                interior.insert_separator(
                    all_separators[i],
                    if i == 0 { first_child } else { all_children[i] },
                )?;
            }
        }

        {
            let page_data = self.storage.page_mut(new_page_no)?;
            let last_child = if insert_pos == all_children.len() - 1 {
                new_right_child
            } else {
                old_right_child
            };
            let mut new_interior = InteriorNodeMut::init(page_data, last_child)?;

            for i in (mid + 1)..all_separators.len() {
                new_interior.insert_separator(all_separators[i], all_children[i])?;
            }
        }

        Ok(InsertResult::Split {
            separator: promoted_separator,
            new_page: new_page_no,
        })
    }

    fn create_new_root(
        &mut self,
        separator: &[u8],
        left_child: u32,
        right_child: u32,
    ) -> Result<()> {
        let new_root_no = self.allocate_page()?;

        let page_data = self.storage.page_mut(new_root_no)?;
        let mut root = InteriorNodeMut::init(page_data, right_child)?;
        root.insert_separator(separator, left_child)?;

        self.root_page = new_root_no;

        Ok(())
    }

    pub fn delete(&mut self, key: &[u8]) -> Result<bool> {
        let mut current_page = self.root_page;
        let mut path: PathStack = SmallVec::new();

        loop {
            let page_data = self.storage.page(current_page)?;
            let header = PageHeader::from_bytes(page_data)?;

            match header.page_type() {
                PageType::BTreeLeaf => break,
                PageType::BTreeInterior => {
                    let interior = InteriorNode::from_page(page_data)?;
                    let (child_page, _) = interior.find_child(key)?;
                    path.push(current_page);
                    current_page = child_page;
                }
                _ => bail!(
                    "unexpected page type {:?} during delete at page {}",
                    header.page_type(),
                    current_page
                ),
            }
        }

        let page_data = self.storage.page_mut(current_page)?;
        let mut leaf = LeafNodeMut::from_page(page_data)?;

        match leaf.find_key(key) {
            SearchResult::Found(idx) => {
                leaf.delete_cell(idx)?;
                Ok(true)
            }
            SearchResult::NotFound(_) => Ok(false),
        }
    }

    fn allocate_page(&mut self) -> Result<u32> {
        if let Some(ref mut freelist) = self.freelist {
            if let Some(page_no) = freelist.allocate(self.storage)? {
                return Ok(page_no);
            }
        }

        let new_page_no = self.storage.page_count();
        self.storage.grow(new_page_no + 1)?;
        Ok(new_page_no)
    }

    pub fn cursor_first(&self) -> Result<Cursor<'_, S>> {
        let mut current_page = self.root_page;

        loop {
            let page_data = self.storage.page(current_page)?;
            let header = PageHeader::from_bytes(page_data)?;

            match header.page_type() {
                PageType::BTreeLeaf => {
                    let leaf = LeafNode::from_page(page_data)?;
                    let exhausted = leaf.cell_count() == 0;
                    return Ok(Cursor {
                        storage: self.storage,
                        root_page: self.root_page,
                        current_page,
                        current_index: 0,
                        exhausted,
                    });
                }
                PageType::BTreeInterior => {
                    let interior = InteriorNode::from_page(page_data)?;
                    if interior.cell_count() == 0 {
                        current_page = interior.right_child();
                    } else {
                        current_page = interior.slot_at(0)?.child_page();
                    }
                }
                _ => bail!(
                    "unexpected page type {:?} during cursor_first at page {}",
                    header.page_type(),
                    current_page
                ),
            }
        }
    }

    pub fn cursor_seek(&self, key: &[u8]) -> Result<Cursor<'_, S>> {
        let mut current_page = self.root_page;

        loop {
            let page_data = self.storage.page(current_page)?;
            let header = PageHeader::from_bytes(page_data)?;

            match header.page_type() {
                PageType::BTreeLeaf => {
                    let leaf = LeafNode::from_page(page_data)?;
                    let index = match leaf.find_key(key) {
                        SearchResult::Found(idx) => idx,
                        SearchResult::NotFound(idx) => idx,
                    };

                    let exhausted = index >= leaf.cell_count() as usize;
                    return Ok(Cursor {
                        storage: self.storage,
                        root_page: self.root_page,
                        current_page,
                        current_index: index,
                        exhausted,
                    });
                }
                PageType::BTreeInterior => {
                    let interior = InteriorNode::from_page(page_data)?;
                    let (child_page, _) = interior.find_child(key)?;
                    current_page = child_page;
                }
                _ => bail!(
                    "unexpected page type {:?} during cursor_seek at page {}",
                    header.page_type(),
                    current_page
                ),
            }
        }
    }

    pub fn cursor_last(&self) -> Result<Cursor<'_, S>> {
        let mut current_page = self.root_page;

        loop {
            let page_data = self.storage.page(current_page)?;
            let header = PageHeader::from_bytes(page_data)?;

            match header.page_type() {
                PageType::BTreeLeaf => {
                    let leaf = LeafNode::from_page(page_data)?;
                    let cell_count = leaf.cell_count() as usize;
                    if cell_count == 0 {
                        return Ok(Cursor {
                            storage: self.storage,
                            root_page: self.root_page,
                            current_page,
                            current_index: 0,
                            exhausted: true,
                        });
                    }
                    return Ok(Cursor {
                        storage: self.storage,
                        root_page: self.root_page,
                        current_page,
                        current_index: cell_count - 1,
                        exhausted: false,
                    });
                }
                PageType::BTreeInterior => {
                    let interior = InteriorNode::from_page(page_data)?;
                    current_page = interior.right_child();
                }
                _ => bail!(
                    "unexpected page type {:?} during cursor_last at page {}",
                    header.page_type(),
                    current_page
                ),
            }
        }
    }
}

impl<'a, S: Storage + ?Sized> Cursor<'a, S> {
    pub fn valid(&self) -> bool {
        !self.exhausted
    }

    pub fn key(&self) -> Result<&'a [u8]> {
        ensure!(!self.exhausted, "cursor is exhausted");
        let page_data = self.storage.page(self.current_page)?;
        let leaf = LeafNode::from_page(page_data)?;
        leaf.key_at(self.current_index)
    }

    pub fn value(&self) -> Result<&'a [u8]> {
        ensure!(!self.exhausted, "cursor is exhausted");
        let page_data = self.storage.page(self.current_page)?;
        let leaf = LeafNode::from_page(page_data)?;
        leaf.value_at(self.current_index)
    }

    pub fn advance(&mut self) -> Result<bool> {
        if self.exhausted {
            return Ok(false);
        }

        self.current_index += 1;

        let page_data = self.storage.page(self.current_page)?;
        let leaf = LeafNode::from_page(page_data)?;

        if self.current_index < leaf.cell_count() as usize {
            return Ok(true);
        }

        let next_page = leaf.next_leaf();
        if next_page == 0 {
            self.exhausted = true;
            return Ok(false);
        }

        let page_count = self.storage.page_count();
        if next_page >= page_count {
            bail!(
                "corrupt next_leaf pointer: page {} has next_leaf={} but page_count={}",
                self.current_page,
                next_page,
                page_count
            );
        }

        self.storage.prefetch_pages(next_page + 1, 2);

        self.current_page = next_page;
        self.current_index = 0;

        let next_page_data = self.storage.page(self.current_page)?;
        let next_leaf = LeafNode::from_page(next_page_data)?;
        if next_leaf.cell_count() == 0 {
            self.exhausted = true;
            return Ok(false);
        }

        Ok(true)
    }

    /// Moves the cursor to the previous key. Returns false if exhausted.
    ///
    /// **Performance:** O(1) within a page, O(log N) when crossing page boundaries
    /// (requires tree re-traversal). Prefer forward iteration for large scans.
    pub fn prev(&mut self) -> Result<bool> {
        if self.exhausted {
            return Ok(false);
        }

        if self.current_index > 0 {
            self.current_index -= 1;
            return Ok(true);
        }

        let prev_leaf = self.find_prev_leaf()?;
        match prev_leaf {
            Some((page_no, last_index)) => {
                self.current_page = page_no;
                self.current_index = last_index;
                Ok(true)
            }
            None => {
                self.exhausted = true;
                Ok(false)
            }
        }
    }

    fn find_prev_leaf(&self) -> Result<Option<(u32, usize)>> {
        let current_key = self.key()?;
        let mut current_page = self.root_page;
        let mut path: SmallVec<[(u32, usize); MAX_TREE_DEPTH]> = SmallVec::new();

        loop {
            let page_data = self.storage.page(current_page)?;
            let header = PageHeader::from_bytes(page_data)?;

            match header.page_type() {
                PageType::BTreeLeaf => break,
                PageType::BTreeInterior => {
                    let interior = InteriorNode::from_page(page_data)?;
                    let (child_page, slot_idx) = interior.find_child(current_key)?;
                    let idx = slot_idx.unwrap_or(interior.cell_count() as usize);
                    path.push((current_page, idx));
                    current_page = child_page;
                }
                _ => bail!(
                    "unexpected page type {:?} during find_prev_leaf at page {}",
                    header.page_type(),
                    current_page
                ),
            }
        }

        while let Some((parent_page, child_idx)) = path.pop() {
            if child_idx > 0 {
                let page_data = self.storage.page(parent_page)?;
                let interior = InteriorNode::from_page(page_data)?;

                let prev_child = if child_idx == 1 {
                    interior.slot_at(0)?.child_page()
                } else if child_idx > 1 {
                    let target_idx = child_idx - 1;
                    if target_idx < interior.cell_count() as usize {
                        interior.slot_at(target_idx)?.child_page()
                    } else {
                        interior.right_child()
                    }
                } else {
                    continue;
                };

                return self.find_rightmost_in_subtree(prev_child);
            }
        }

        Ok(None)
    }

    fn find_rightmost_in_subtree(&self, mut page_no: u32) -> Result<Option<(u32, usize)>> {
        loop {
            let page_data = self.storage.page(page_no)?;
            let header = PageHeader::from_bytes(page_data)?;

            match header.page_type() {
                PageType::BTreeLeaf => {
                    let leaf = LeafNode::from_page(page_data)?;
                    let count = leaf.cell_count() as usize;
                    if count == 0 {
                        return Ok(None);
                    }
                    return Ok(Some((page_no, count - 1)));
                }
                PageType::BTreeInterior => {
                    let interior = InteriorNode::from_page(page_data)?;
                    page_no = interior.right_child();
                }
                _ => bail!(
                    "unexpected page type {:?} during find_rightmost at page {}",
                    header.page_type(),
                    page_no
                ),
            }
        }
    }
}
