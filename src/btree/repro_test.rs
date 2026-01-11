use super::*;
use crate::storage::MmapStorage;
use tempfile::NamedTempFile;

#[test]
fn test_separator_conflict_reproduction() -> Result<()> {
    let temp_file = NamedTempFile::new()?;
    let mut storage = MmapStorage::create(temp_file.path(), 2)?;
    let mut btree = BTree::create(&mut storage, 0)?;

    // 1. Fill a leaf page to force a split
    // Page size 16384. Key size 1000. 14 keys = 14000 + overhead > split threshold?
    let key_size = 1000;
    let value_size = 10;
    
    // Insert 0, 10, 20...
    for i in 0..15 {
        let key = vec![(i * 10) as u8; key_size];
        let value = vec![1u8; value_size];
        btree.insert(&key, &value)?;
    }
    
    // This should have caused a split.
    // Root should be interior.
    let root_page = storage.page(btree.root_page())?;
    let header = PageHeader::from_bytes(root_page)?;
    assert_eq!(header.page_type(), PageType::BTreeInterior);

    // 2. Determine the split point/separator
    // The separator is likely one of the keys, e.g., 70 or 80.
    // Let's assume the tree structure is balanced.
    
    // 3. Try to insert a key that might clash with the separator
    // If we insert a key that effectively forces the RIGHT child to split again,
    // and the new separator is identical to the parent's separator.
    
    // To trigger "separator exists":
    // Parent: [Sep 100]. Right Child (>= 100).
    // Right Child splits. New Sep = 100?
    // This requires Right Child to have [100, 100] (blocked) or [100, 101] and choose 100?
    // If Right Child has [100, 101]. Mid=1 (101). Sep=101. 101 != 100.
    // Taking the hypothesis that 'mid' calculation might pick index 0.
    
    Ok(())
}
