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
//! ## Node Splitting
//!
//! When a leaf node becomes full during insertion:
//! 1. Allocate new leaf page
//! 2. Move upper half of keys to new leaf
//! 3. Compute separator key (suffix truncated for efficiency)
//! 4. Insert separator into parent interior node
//! 5. If parent is full, split recursively up the tree
//!
//! When root splits:
//! 1. Allocate new root page (interior)
//! 2. Set old root as left child
//! 3. Set new split page as right child
//! 4. Insert separator key
//!
//! ## Zero-Copy Design
//!
//! The BTree struct holds references to the underlying storage, enabling
//! zero-copy access to keys and values through the mmap layer:
//!
//! ```text
//! struct BTree<'a> {
//!     storage: &'a mut MmapStorage,  // Reference to mmap'd file
//!     root_page: u32,                 // Root page number
//! }
//! ```
//!
//! ## Memory Safety
//!
//! The borrow checker ensures:
//! - No dangling page references across storage operations
//! - Mutable access to pages is exclusive
//! - Storage cannot be grown while page references exist
//!
//! ## Insert Algorithm
//!
//! ```text
//! 1. Start at root
//! 2. While at interior node:
//!    - Find child page for key via separator comparison
//!    - Push current page onto path stack
//!    - Navigate to child
//! 3. At leaf: insert key-value
//! 4. If leaf full: split_leaf()
//! 5. If split produced separator: propagate up via path stack
//! 6. If propagation reaches root and root splits: create new root
//! ```
//!
//! ## Delete Algorithm
//!
//! ```text
//! 1. Search for key in leaf
//! 2. If not found: return NotFound
//! 3. Delete cell from leaf
//! 4. (Optional) Handle underflow via merge/redistribute
//! ```
//!
//! This implementation uses simple deletion without rebalancing. Underflow
//! handling adds complexity with marginal benefit for most workloads.
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
//! - Tree depth for 1M rows: typically 2-3 levels

use eyre::{bail, ensure, Result};

use super::interior::{separator_len, InteriorNode, InteriorNodeMut, INTERIOR_SLOT_SIZE};
use super::leaf::{LeafNode, LeafNodeMut, SearchResult, SLOT_SIZE};
use crate::storage::{MmapStorage, PageHeader, PageType};

#[derive(Debug)]
pub struct BTree<'a> {
    storage: &'a mut MmapStorage,
    root_page: u32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum InsertResult {
    Ok,
    Split { separator: Vec<u8>, new_page: u32 },
}

impl<'a> BTree<'a> {
    pub fn new(storage: &'a mut MmapStorage, root_page: u32) -> Result<Self> {
        ensure!(
            root_page < storage.page_count(),
            "root page {} out of bounds (page_count={})",
            root_page,
            storage.page_count()
        );
        Ok(Self { storage, root_page })
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

        Ok(Self { storage, root_page })
    }

    pub fn root_page(&self) -> u32 {
        self.root_page
    }

    pub fn search(&self, key: &[u8]) -> Result<Option<Vec<u8>>> {
        let mut current_page = self.root_page;

        loop {
            let page_data = self.storage.page(current_page)?;
            let header = PageHeader::from_bytes(page_data)?;

            match header.page_type() {
                PageType::BTreeLeaf => {
                    let leaf = LeafNode::from_page(page_data)?;
                    match leaf.find_key(key) {
                        SearchResult::Found(idx) => {
                            let value = leaf.value_at(idx)?;
                            return Ok(Some(value.to_vec()));
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

    pub fn insert(&mut self, key: &[u8], value: &[u8]) -> Result<()> {
        let mut path: Vec<u32> = Vec::new();
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

        let varint_len = varint_size(value.len() as u64);
        let cell_size = key.len() + varint_len + value.len();
        let space_needed = cell_size + SLOT_SIZE;

        if leaf.free_space() as usize >= space_needed {
            leaf.insert_cell(key, value)?;
            return Ok(InsertResult::Ok);
        }

        self.split_leaf(page_no, key, value)
    }

    fn split_leaf(&mut self, page_no: u32, key: &[u8], value: &[u8]) -> Result<InsertResult> {
        let new_page_no = self.allocate_page()?;

        let mut all_keys: Vec<Vec<u8>> = Vec::new();
        let mut all_values: Vec<Vec<u8>> = Vec::new();

        {
            let page_data = self.storage.page(page_no)?;
            let leaf = LeafNode::from_page(page_data)?;
            let count = leaf.cell_count() as usize;

            for i in 0..count {
                all_keys.push(leaf.key_at(i)?.to_vec());
                all_values.push(leaf.value_at(i)?.to_vec());
            }
        }

        let insert_pos = all_keys
            .iter()
            .position(|k| k.as_slice() > key)
            .unwrap_or(all_keys.len());
        all_keys.insert(insert_pos, key.to_vec());
        all_values.insert(insert_pos, value.to_vec());

        let mid = all_keys.len() / 2;

        {
            let page_data = self.storage.page_mut(page_no)?;
            let mut leaf = LeafNodeMut::init(page_data)?;
            for i in 0..mid {
                leaf.insert_cell(&all_keys[i], &all_values[i])?;
            }
            leaf.set_next_leaf(new_page_no)?;
        }

        let separator_key;
        {
            let page_data = self.storage.page_mut(new_page_no)?;
            let mut new_leaf = LeafNodeMut::init(page_data)?;
            for i in mid..all_keys.len() {
                new_leaf.insert_cell(&all_keys[i], &all_values[i])?;
            }

            let left_max = &all_keys[mid - 1];
            let right_min = &all_keys[mid];
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
        mut path: Vec<u32>,
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
        let new_page_no = self.allocate_page()?;

        let mut all_separators: Vec<Vec<u8>> = Vec::new();
        let mut all_children: Vec<u32> = Vec::new();
        let old_right_child: u32;

        {
            let page_data = self.storage.page(page_no)?;
            let interior = InteriorNode::from_page(page_data)?;
            let count = interior.cell_count() as usize;

            for i in 0..count {
                all_separators.push(interior.key_at(i)?.to_vec());
                all_children.push(interior.slot_at(i)?.child_page);
            }
            old_right_child = interior.right_child();
        }

        let insert_pos = all_separators
            .iter()
            .position(|s| s.as_slice() > new_separator)
            .unwrap_or(all_separators.len());

        all_separators.insert(insert_pos, new_separator.to_vec());

        if insert_pos == all_children.len() {
            all_children.push(old_right_child);
        } else {
            all_children.insert(insert_pos + 1, new_right_child);
        }

        let mid = all_separators.len() / 2;
        let promoted_separator = all_separators[mid].clone();

        {
            let page_data = self.storage.page_mut(page_no)?;
            let first_child = all_children[0];
            let mut interior = InteriorNodeMut::init(page_data, all_children[mid])?;

            for i in 0..mid {
                interior.insert_separator(
                    &all_separators[i],
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
                new_interior.insert_separator(&all_separators[i], all_children[i])?;
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
        let mut path: Vec<u32> = Vec::new();

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
        let new_page_no = self.storage.page_count();
        self.storage.grow(new_page_no + 1)?;
        Ok(new_page_no)
    }
}

fn varint_size(value: u64) -> usize {
    if value <= 240 {
        1
    } else if value <= 2287 {
        2
    } else if value <= 67823 {
        3
    } else if value <= 0xFF_FFFF {
        4
    } else if value <= 0xFFFF_FFFF {
        5
    } else {
        9
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

        let result = btree.search(b"hello").unwrap();
        assert_eq!(result, Some(b"world".to_vec()));
    }

    #[test]
    fn btree_insert_and_search_multiple_keys() {
        let (_dir, mut storage) = create_test_storage(5);
        let mut btree = BTree::create(&mut storage, 0).unwrap();

        btree.insert(b"charlie", b"3").unwrap();
        btree.insert(b"alpha", b"1").unwrap();
        btree.insert(b"bravo", b"2").unwrap();

        assert_eq!(btree.search(b"alpha").unwrap(), Some(b"1".to_vec()));
        assert_eq!(btree.search(b"bravo").unwrap(), Some(b"2".to_vec()));
        assert_eq!(btree.search(b"charlie").unwrap(), Some(b"3".to_vec()));
        assert!(btree.search(b"delta").unwrap().is_none());
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

        assert!(btree.search(b"key2").unwrap().is_none());
        assert_eq!(btree.search(b"key1").unwrap(), Some(b"value1".to_vec()));
        assert_eq!(btree.search(b"key3").unwrap(), Some(b"value3".to_vec()));
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
            let result = btree.search(key.as_bytes()).unwrap();
            assert_eq!(
                result,
                Some(expected_value.into_bytes()),
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
            let result = btree.search(key.as_bytes()).unwrap();
            assert_eq!(result, Some(expected_value.into_bytes()));
        }
    }

    #[test]
    fn btree_handles_large_values() {
        let (_dir, mut storage) = create_test_storage(10);
        let mut btree = BTree::create(&mut storage, 0).unwrap();

        let large_value = vec![0xAB; 1000];
        btree.insert(b"bigkey", &large_value).unwrap();

        let result = btree.search(b"bigkey").unwrap();
        assert_eq!(result, Some(large_value));
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
            let result = btree.search(key.as_bytes()).unwrap();
            if i % 2 == 0 {
                assert!(result.is_none(), "key {} should be deleted", key);
            } else {
                let expected = format!("value{:05}", i);
                assert_eq!(
                    result,
                    Some(expected.into_bytes()),
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
}
