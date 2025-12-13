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

use bumpalo::collections::Vec as BumpVec;
use bumpalo::Bump;
use eyre::{bail, ensure, Result};
use smallvec::SmallVec;

use super::interior::{separator_len, InteriorNode, InteriorNodeMut, INTERIOR_SLOT_SIZE};
use super::leaf::{LeafNode, LeafNodeMut, SearchResult, SLOT_SIZE};
use crate::encoding::varint::varint_len;
use crate::storage::{Freelist, MmapStorage, PageHeader, PageType};

pub const MAX_TREE_DEPTH: usize = 8;

type PathStack = SmallVec<[u32; MAX_TREE_DEPTH]>;

#[derive(Debug)]
pub struct BTree<'a> {
    storage: &'a mut MmapStorage,
    root_page: u32,
    freelist: Option<&'a mut Freelist>,
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

pub struct Cursor<'a> {
    storage: &'a MmapStorage,
    root_page: u32,
    current_page: u32,
    current_index: usize,
    exhausted: bool,
}

impl<'a> BTree<'a> {
    pub fn new(storage: &'a mut MmapStorage, root_page: u32) -> Result<Self> {
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
        })
    }

    pub fn with_freelist(
        storage: &'a mut MmapStorage,
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
        })
    }

    pub fn create(storage: &'a mut MmapStorage, root_page: u32) -> Result<Self> {
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

        if let InsertResult::Split {
            separator,
            new_page,
        } = result
        {
            self.propagate_split(path, &separator, current_page, new_page)?;
        }

        Ok(())
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

        let mid = all_keys.len() / 2;

        let old_next_leaf;
        {
            let page_data = self.storage.page(page_no)?;
            let leaf = LeafNode::from_page(page_data)?;
            old_next_leaf = leaf.next_leaf();
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

            let left_max = all_keys[mid - 1];
            let right_min = all_keys[mid];
            let sep_len = separator_len(left_max, right_min);
            separator_key = right_min[..sep_len].to_vec();
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

    pub fn cursor_first(&self) -> Result<Cursor<'_>> {
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

    pub fn cursor_seek(&self, key: &[u8]) -> Result<Cursor<'_>> {
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

    pub fn cursor_last(&self) -> Result<Cursor<'_>> {
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

impl<'a> Cursor<'a> {
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

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    fn create_test_storage(pages: u32) -> (tempfile::TempDir, MmapStorage) {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.db");
        let storage = MmapStorage::create(&path, pages).unwrap();
        (dir, storage)
    }

    #[test]
    fn btree_new_validates_root_page() {
        let (_dir, mut storage) = create_test_storage(5);

        let result = BTree::new(&mut storage, 10);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("out of bounds"));
    }

    #[test]
    fn btree_create_initializes_empty_leaf_root() {
        let (_dir, mut storage) = create_test_storage(5);

        let btree = BTree::create(&mut storage, 0).unwrap();

        assert_eq!(btree.root_page(), 0);

        let page = storage.page(0).unwrap();
        let header = PageHeader::from_bytes(page).unwrap();
        assert_eq!(header.page_type(), PageType::BTreeLeaf);
        assert_eq!(header.cell_count(), 0);
    }

    #[test]
    fn btree_search_empty_tree_returns_none() {
        let (_dir, mut storage) = create_test_storage(5);
        let btree = BTree::create(&mut storage, 0).unwrap();

        let result = btree.search(b"key").unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn btree_insert_and_search_single_key() {
        let (_dir, mut storage) = create_test_storage(5);
        let mut btree = BTree::create(&mut storage, 0).unwrap();

        btree.insert(b"hello", b"world").unwrap();

        let handle = btree.search(b"hello").unwrap().unwrap();
        let value = btree.get_value(&handle).unwrap();
        assert_eq!(value, b"world");
    }

    #[test]
    fn btree_get_returns_zero_copy_reference() {
        let (_dir, mut storage) = create_test_storage(5);

        {
            let mut btree = BTree::create(&mut storage, 0).unwrap();
            btree.insert(b"key", b"value").unwrap();
        }

        let page_data = storage.page(0).unwrap();
        let leaf = LeafNode::from_page(page_data).unwrap();
        let value = leaf.value_at(0).unwrap();
        assert_eq!(value, b"value");

        let page_ptr = page_data.as_ptr();
        let value_ptr = value.as_ptr();
        assert!(value_ptr >= page_ptr);
    }

    #[test]
    fn btree_insert_and_search_multiple_keys() {
        let (_dir, mut storage) = create_test_storage(5);
        let mut btree = BTree::create(&mut storage, 0).unwrap();

        btree.insert(b"charlie", b"3").unwrap();
        btree.insert(b"alpha", b"1").unwrap();
        btree.insert(b"bravo", b"2").unwrap();

        assert_eq!(btree.get(b"alpha").unwrap(), Some(&b"1"[..]));
        assert_eq!(btree.get(b"bravo").unwrap(), Some(&b"2"[..]));
        assert_eq!(btree.get(b"charlie").unwrap(), Some(&b"3"[..]));
        assert!(btree.get(b"delta").unwrap().is_none());
    }

    #[test]
    fn btree_delete_existing_key() {
        let (_dir, mut storage) = create_test_storage(5);
        let mut btree = BTree::create(&mut storage, 0).unwrap();

        btree.insert(b"key1", b"value1").unwrap();
        btree.insert(b"key2", b"value2").unwrap();
        btree.insert(b"key3", b"value3").unwrap();

        let deleted = btree.delete(b"key2").unwrap();
        assert!(deleted);

        assert!(btree.get(b"key2").unwrap().is_none());
        assert_eq!(btree.get(b"key1").unwrap(), Some(&b"value1"[..]));
        assert_eq!(btree.get(b"key3").unwrap(), Some(&b"value3"[..]));
    }

    #[test]
    fn btree_delete_nonexistent_key_returns_false() {
        let (_dir, mut storage) = create_test_storage(5);
        let mut btree = BTree::create(&mut storage, 0).unwrap();

        btree.insert(b"key1", b"value1").unwrap();

        let deleted = btree.delete(b"nonexistent").unwrap();
        assert!(!deleted);
    }

    #[test]
    fn btree_split_leaf_on_overflow() {
        let (_dir, mut storage) = create_test_storage(20);
        let mut btree = BTree::create(&mut storage, 0).unwrap();

        for i in 0..500 {
            let key = format!("key{:05}", i);
            let value = format!("value{:05}", i);
            btree.insert(key.as_bytes(), value.as_bytes()).unwrap();
        }

        for i in 0..500 {
            let key = format!("key{:05}", i);
            let expected_value = format!("value{:05}", i);
            let value = btree.get(key.as_bytes()).unwrap();
            assert_eq!(
                value,
                Some(expected_value.as_bytes()),
                "key {} not found",
                key
            );
        }
    }

    #[test]
    fn btree_maintains_sorted_order_after_splits() {
        let (_dir, mut storage) = create_test_storage(20);
        let mut btree = BTree::create(&mut storage, 0).unwrap();

        for i in (0..200).rev() {
            let key = format!("key{:05}", i);
            let value = format!("val{:05}", i);
            btree.insert(key.as_bytes(), value.as_bytes()).unwrap();
        }

        for i in 0..200 {
            let key = format!("key{:05}", i);
            let expected_value = format!("val{:05}", i);
            let value = btree.get(key.as_bytes()).unwrap();
            assert_eq!(value, Some(expected_value.as_bytes()));
        }
    }

    #[test]
    fn btree_handles_large_values() {
        let (_dir, mut storage) = create_test_storage(10);
        let mut btree = BTree::create(&mut storage, 0).unwrap();

        let large_value = vec![0xAB; 1000];
        btree.insert(b"bigkey", &large_value).unwrap();

        let value = btree.get(b"bigkey").unwrap().unwrap();
        assert_eq!(value, &large_value[..]);
    }

    #[test]
    fn btree_delete_after_split() {
        let (_dir, mut storage) = create_test_storage(20);
        let mut btree = BTree::create(&mut storage, 0).unwrap();

        for i in 0..300 {
            let key = format!("key{:05}", i);
            let value = format!("value{:05}", i);
            btree.insert(key.as_bytes(), value.as_bytes()).unwrap();
        }

        for i in (0..300).step_by(2) {
            let key = format!("key{:05}", i);
            let deleted = btree.delete(key.as_bytes()).unwrap();
            assert!(deleted, "failed to delete {}", key);
        }

        for i in 0..300 {
            let key = format!("key{:05}", i);
            let result = btree.get(key.as_bytes()).unwrap();
            if i % 2 == 0 {
                assert!(result.is_none(), "key {} should be deleted", key);
            } else {
                let expected = format!("value{:05}", i);
                assert_eq!(
                    result,
                    Some(expected.as_bytes()),
                    "key {} should exist",
                    key
                );
            }
        }
    }

    #[test]
    fn btree_root_page_accessor() {
        let (_dir, mut storage) = create_test_storage(5);
        let btree = BTree::create(&mut storage, 2).unwrap();

        assert_eq!(btree.root_page(), 2);
    }

    #[test]
    fn cursor_iterates_all_keys_in_order() {
        let (_dir, mut storage) = create_test_storage(20);
        let mut btree = BTree::create(&mut storage, 0).unwrap();

        for i in (0..100).rev() {
            let key = format!("key{:03}", i);
            let value = format!("val{:03}", i);
            btree.insert(key.as_bytes(), value.as_bytes()).unwrap();
        }

        let mut cursor = btree.cursor_first().unwrap();
        let mut count = 0;
        let mut prev_key: Option<Vec<u8>> = None;

        while cursor.valid() {
            let key = cursor.key().unwrap();
            let value = cursor.value().unwrap();

            if let Some(ref pk) = prev_key {
                assert!(key > pk.as_slice(), "keys should be in order");
            }
            prev_key = Some(key.to_vec());

            let expected_key = format!("key{:03}", count);
            let expected_value = format!("val{:03}", count);
            assert_eq!(key, expected_key.as_bytes());
            assert_eq!(value, expected_value.as_bytes());

            count += 1;
            cursor.advance().unwrap();
        }

        assert_eq!(count, 100);
    }

    #[test]
    fn cursor_seek_positions_correctly() {
        let (_dir, mut storage) = create_test_storage(10);
        let mut btree = BTree::create(&mut storage, 0).unwrap();

        btree.insert(b"alpha", b"1").unwrap();
        btree.insert(b"bravo", b"2").unwrap();
        btree.insert(b"charlie", b"3").unwrap();
        btree.insert(b"delta", b"4").unwrap();

        let cursor = btree.cursor_seek(b"bravo").unwrap();
        assert!(cursor.valid());
        assert_eq!(cursor.key().unwrap(), b"bravo");

        let cursor = btree.cursor_seek(b"beta").unwrap();
        assert!(cursor.valid());
        assert_eq!(cursor.key().unwrap(), b"bravo");

        let cursor = btree.cursor_seek(b"echo").unwrap();
        assert!(!cursor.valid());
    }

    #[test]
    fn cursor_on_empty_tree() {
        let (_dir, mut storage) = create_test_storage(5);
        let btree = BTree::create(&mut storage, 0).unwrap();

        let cursor = btree.cursor_first().unwrap();
        assert!(!cursor.valid());
    }

    #[test]
    fn btree_with_freelist_reuses_pages() {
        let (_dir, mut storage) = create_test_storage(10);

        let mut freelist = Freelist::new();
        freelist.release(&mut storage, 5).unwrap();
        freelist.release(&mut storage, 7).unwrap();

        assert_eq!(freelist.head_page(), 5);
        assert_eq!(freelist.free_count(), 2);

        {
            let page = storage.page_mut(0).unwrap();
            LeafNodeMut::init(page).unwrap();
        }

        {
            let allocated = freelist.allocate(&mut storage).unwrap();
            assert!(allocated.is_some());
            let page_no = allocated.unwrap();
            assert!(page_no == 5 || page_no == 7);

            freelist.release(&mut storage, page_no).unwrap();
        }

        {
            let mut btree = BTree::with_freelist(&mut storage, 0, &mut freelist).unwrap();

            for i in 0..50 {
                let key = format!("key{:05}", i);
                let value = format!("value{:05}", i);
                btree.insert(key.as_bytes(), value.as_bytes()).unwrap();
            }

            for i in 0..50 {
                let key = format!("key{:05}", i);
                let value = btree.get(key.as_bytes()).unwrap();
                assert!(value.is_some(), "key {} not found", key);
            }
        }
    }

    #[test]
    fn search_handle_provides_zero_copy_access() {
        let (_dir, mut storage) = create_test_storage(5);

        {
            let mut btree = BTree::create(&mut storage, 0).unwrap();
            btree.insert(b"testkey", b"testvalue").unwrap();

            let handle = btree.search(b"testkey").unwrap().unwrap();
            assert_eq!(handle.page_no, 0);
            assert_eq!(handle.cell_index, 0);

            let key = btree.get_key(&handle).unwrap();
            let value = btree.get_value(&handle).unwrap();

            assert_eq!(key, b"testkey");
            assert_eq!(value, b"testvalue");
        }

        let page = storage.page(0).unwrap();
        let leaf = LeafNode::from_page(page).unwrap();
        let key = leaf.key_at(0).unwrap();
        let value = leaf.value_at(0).unwrap();

        assert!(key.as_ptr() >= page.as_ptr());
        assert!(value.as_ptr() >= page.as_ptr());
    }

    #[test]
    fn cursor_last_positions_at_end() {
        let (_dir, mut storage) = create_test_storage(10);
        let mut btree = BTree::create(&mut storage, 0).unwrap();

        btree.insert(b"alpha", b"1").unwrap();
        btree.insert(b"bravo", b"2").unwrap();
        btree.insert(b"charlie", b"3").unwrap();
        btree.insert(b"delta", b"4").unwrap();

        let cursor = btree.cursor_last().unwrap();
        assert!(cursor.valid());
        assert_eq!(cursor.key().unwrap(), b"delta");
        assert_eq!(cursor.value().unwrap(), b"4");
    }

    #[test]
    fn cursor_last_on_empty_tree() {
        let (_dir, mut storage) = create_test_storage(5);
        let btree = BTree::create(&mut storage, 0).unwrap();

        let cursor = btree.cursor_last().unwrap();
        assert!(!cursor.valid());
    }

    #[test]
    fn cursor_prev_iterates_backwards() {
        let (_dir, mut storage) = create_test_storage(10);
        let mut btree = BTree::create(&mut storage, 0).unwrap();

        btree.insert(b"alpha", b"1").unwrap();
        btree.insert(b"bravo", b"2").unwrap();
        btree.insert(b"charlie", b"3").unwrap();

        let mut cursor = btree.cursor_last().unwrap();
        assert!(cursor.valid());
        assert_eq!(cursor.key().unwrap(), b"charlie");

        assert!(cursor.prev().unwrap());
        assert_eq!(cursor.key().unwrap(), b"bravo");

        assert!(cursor.prev().unwrap());
        assert_eq!(cursor.key().unwrap(), b"alpha");

        assert!(!cursor.prev().unwrap());
        assert!(!cursor.valid());
    }

    #[test]
    fn cursor_prev_on_first_element() {
        let (_dir, mut storage) = create_test_storage(5);
        let mut btree = BTree::create(&mut storage, 0).unwrap();

        btree.insert(b"only", b"one").unwrap();

        let mut cursor = btree.cursor_first().unwrap();
        assert!(cursor.valid());
        assert_eq!(cursor.key().unwrap(), b"only");

        assert!(!cursor.prev().unwrap());
        assert!(!cursor.valid());
    }

    #[test]
    fn cursor_prev_across_multiple_pages() {
        let (_dir, mut storage) = create_test_storage(20);
        let mut btree = BTree::create(&mut storage, 0).unwrap();

        for i in (0..100).rev() {
            let key = format!("key{:03}", i);
            let value = format!("val{:03}", i);
            btree.insert(key.as_bytes(), value.as_bytes()).unwrap();
        }

        let mut cursor = btree.cursor_last().unwrap();
        let mut count = 99;

        while cursor.valid() {
            let expected_key = format!("key{:03}", count);
            assert_eq!(cursor.key().unwrap(), expected_key.as_bytes());

            if count == 0 {
                assert!(!cursor.prev().unwrap());
            } else {
                assert!(cursor.prev().unwrap());
                count -= 1;
            }
        }

        assert_eq!(count, 0);
    }

    #[test]
    fn cursor_prev_deep_tree_with_many_keys() {
        let (_dir, mut storage) = create_test_storage(100);
        let mut btree = BTree::create(&mut storage, 0).unwrap();

        for i in 0..1500 {
            let key = format!("key{:05}", i);
            let value = format!("val{:05}", i);
            btree.insert(key.as_bytes(), value.as_bytes()).unwrap();
        }

        let mut cursor = btree.cursor_last().unwrap();
        assert!(cursor.valid());
        assert_eq!(cursor.key().unwrap(), b"key01499");

        let mut count = 1499;
        while cursor.valid() {
            let expected_key = format!("key{:05}", count);
            assert_eq!(
                cursor.key().unwrap(),
                expected_key.as_bytes(),
                "mismatch at count {}",
                count
            );

            if count == 0 {
                assert!(!cursor.prev().unwrap());
            } else {
                assert!(cursor.prev().unwrap());
                count -= 1;
            }
        }

        assert_eq!(count, 0);
    }

    #[test]
    fn cursor_bidirectional_navigation() {
        let (_dir, mut storage) = create_test_storage(20);
        let mut btree = BTree::create(&mut storage, 0).unwrap();

        for i in 0..200 {
            let key = format!("key{:03}", i);
            let value = format!("val{:03}", i);
            btree.insert(key.as_bytes(), value.as_bytes()).unwrap();
        }

        let mut cursor = btree.cursor_seek(b"key100").unwrap();
        assert!(cursor.valid());
        assert_eq!(cursor.key().unwrap(), b"key100");

        assert!(cursor.advance().unwrap());
        assert_eq!(cursor.key().unwrap(), b"key101");

        assert!(cursor.advance().unwrap());
        assert_eq!(cursor.key().unwrap(), b"key102");

        assert!(cursor.prev().unwrap());
        assert_eq!(cursor.key().unwrap(), b"key101");

        assert!(cursor.prev().unwrap());
        assert_eq!(cursor.key().unwrap(), b"key100");

        assert!(cursor.prev().unwrap());
        assert_eq!(cursor.key().unwrap(), b"key099");
    }
}
