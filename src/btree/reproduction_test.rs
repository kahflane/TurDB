
#[cfg(test)]
mod reproduction_test {
    use super::*;
    use crate::storage::{PageHeader, PageType, PAGE_SIZE};

    #[test]
    fn test_specific_keys_corruption() -> Result<()> {
        let mut page = vec![0u8; PAGE_SIZE];
        LeafNodeMut::init(&mut page)?;
        let mut leaf = LeafNodeMut::from_page(&mut page)?;

        let key_small = [0, 0, 0, 0, 0, 0, 8, 27]; // 2075
        let key_big = [0, 0, 0, 0, 0, 0, 8, 70];   // 2118

        println!("Inserting Small...");
        leaf.insert_cell(&key_small, b"val1")?;
        
        {
            let leaf_read = LeafNode::from_page(&page)?;
            assert_eq!(leaf_read.cell_count(), 1);
            assert_eq!(leaf_read.key_at(0)?, &key_small);
            println!("Found key at 0: {:?}", leaf_read.key_at(0)?);
        }

        println!("Inserting Big...");
        leaf.insert_cell(&key_big, b"val2")?;

        {
            let leaf_read = LeafNode::from_page(&page)?;
            assert_eq!(leaf_read.cell_count(), 2);
            let k0 = leaf_read.key_at(0)?;
            let k1 = leaf_read.key_at(1)?;
            println!("K0: {:?}", k0);
            println!("K1: {:?}", k1);
            
            assert_eq!(k0, &key_small);
            assert_eq!(k1, &key_big);
            assert!(k0 < k1);
        }

        Ok(())
    }

    #[test]
    fn test_reverse_insertion_corruption() -> Result<()> {
        let mut page = vec![0u8; PAGE_SIZE];
        LeafNodeMut::init(&mut page)?;
        let mut leaf = LeafNodeMut::from_page(&mut page)?;

        let key_small = [0, 0, 0, 0, 0, 0, 8, 27]; // 2075
        let key_big = [0, 0, 0, 0, 0, 0, 8, 70];   // 2118

        println!("Inserting Big...");
        leaf.insert_cell(&key_big, b"val2")?;
        
        println!("Inserting Small...");
        leaf.insert_cell(&key_small, b"val1")?;

        {
            let leaf_read = LeafNode::from_page(&page)?;
            assert_eq!(leaf_read.cell_count(), 2);
            let k0 = leaf_read.key_at(0)?;
            let k1 = leaf_read.key_at(1)?;
            println!("K0: {:?}", k0);
            println!("K1: {:?}", k1);
            
            assert_eq!(k0, &key_small);
            assert_eq!(k1, &key_big);
            assert!(k0 < k1);
        }

        Ok(())
    }
}
