use crate::sql::decoder::{RecordDecoder, SimpleDecoder};

pub struct BTreeCursorAdapter {
    pub keys: Vec<Vec<u8>>,
    pub values: Vec<Vec<u8>>,
    pub decoder: Box<dyn RecordDecoder + Send + Sync>,
    pub current: usize,
}

impl BTreeCursorAdapter {
    pub fn new(
        keys: Vec<Vec<u8>>,
        values: Vec<Vec<u8>>,
        decoder: Box<dyn RecordDecoder + Send + Sync>,
    ) -> Self {
        Self {
            keys,
            values,
            decoder,
            current: 0,
        }
    }

    pub fn from_kv_pairs(
        pairs: Vec<(Vec<u8>, Vec<u8>)>,
        decoder: Box<dyn RecordDecoder + Send + Sync>,
    ) -> Self {
        let (keys, values): (Vec<_>, Vec<_>) = pairs.into_iter().unzip();
        Self::new(keys, values, decoder)
    }

    pub fn from_btree_scan(
        storage: &mut crate::storage::MmapStorage,
        root_page: u32,
        column_types: Vec<crate::records::types::DataType>,
    ) -> eyre::Result<Self> {
        Self::from_btree_scan_with_projections(storage, root_page, column_types, None)
    }

    pub fn from_btree_scan_with_projections(
        storage: &mut crate::storage::MmapStorage,
        root_page: u32,
        column_types: Vec<crate::records::types::DataType>,
        projections: Option<Vec<usize>>,
    ) -> eyre::Result<Self> {
        use crate::btree::BTree;

        let btree = BTree::new(storage, root_page)?;
        let mut cursor = btree.cursor_first()?;
        let mut keys = Vec::new();
        let mut values = Vec::new();

        if cursor.valid() {
            loop {
                keys.push(cursor.key()?.to_vec());
                values.push(cursor.value()?.to_vec());
                if !cursor.advance()? {
                    break;
                }
            }
        }

        let decoder: Box<dyn RecordDecoder + Send + Sync> = match projections {
            Some(proj) => Box::new(SimpleDecoder::with_projections(column_types, proj)),
            None => Box::new(SimpleDecoder::new(column_types)),
        };
        Ok(Self::new(keys, values, decoder))
    }

    pub fn from_btree_range_scan(
        storage: &mut crate::storage::MmapStorage,
        root_page: u32,
        start_key: Option<&[u8]>,
        end_key: Option<&[u8]>,
        column_types: Vec<crate::records::types::DataType>,
    ) -> eyre::Result<Self> {
        Self::from_btree_range_scan_with_projections(
            storage,
            root_page,
            start_key,
            end_key,
            column_types,
            None,
        )
    }

    pub fn from_btree_range_scan_with_projections(
        storage: &mut crate::storage::MmapStorage,
        root_page: u32,
        start_key: Option<&[u8]>,
        end_key: Option<&[u8]>,
        column_types: Vec<crate::records::types::DataType>,
        projections: Option<Vec<usize>>,
    ) -> eyre::Result<Self> {
        use crate::btree::BTree;

        let btree = BTree::new(storage, root_page)?;
        let mut cursor = if let Some(start) = start_key {
            btree.cursor_seek(start)?
        } else {
            btree.cursor_first()?
        };

        let mut keys = Vec::new();
        let mut values = Vec::new();

        if cursor.valid() {
            loop {
                let current_key = cursor.key()?;
                if let Some(end) = end_key {
                    if current_key >= end {
                        break;
                    }
                }
                keys.push(current_key.to_vec());
                values.push(cursor.value()?.to_vec());
                if !cursor.advance()? {
                    break;
                }
            }
        }

        let decoder: Box<dyn RecordDecoder + Send + Sync> = match projections {
            Some(proj) => Box::new(SimpleDecoder::with_projections(column_types, proj)),
            None => Box::new(SimpleDecoder::new(column_types)),
        };
        Ok(Self::new(keys, values, decoder))
    }
}
