//! # Page Types and Header Layout
//!
//! This module defines the page structure for TurDB's storage layer. Every 16KB
//! page begins with a 16-byte header containing metadata about the page contents.
//!
//! ## Page Header Layout (16 bytes)
//!
//! ```text
//! Offset  Size  Field        Description
//! ------  ----  -----------  ----------------------------------------
//! 0       1     page_type    Type of page (BTreeLeaf, HnswNode, etc.)
//! 1       1     flags        Page flags (dirty, overflow, etc.)
//! 2       2     cell_count   Number of cells/entries in this page
//! 4       2     free_start   Offset where free space begins
//! 6       2     free_end     Offset where free space ends
//! 8       1     frag_bytes   Fragmented free bytes within cell area
//! 9       3     reserved     Reserved for future use
//! 12      4     right_child  Right child page (interior) / next leaf (leaf)
//! ```
//!
//! ## Page Types
//!
//! TurDB uses several page types for different storage needs:
//!
//! - **BTreeInterior** (0x01): B-tree internal node with keys and child pointers
//! - **BTreeLeaf** (0x02): B-tree leaf node with keys and row data
//! - **HnswNode** (0x10): HNSW graph node with vector and neighbor links
//! - **HnswMeta** (0x11): HNSW index metadata (entry point, parameters)
//! - **Overflow** (0x20): Overflow page for large values
//! - **FreeList** (0x30): Free page list trunk
//!
//! ## Zero-Copy Access
//!
//! The `PageHeader` struct uses `zerocopy` for safe transmutation from raw bytes.
//! This enables reading headers directly from mmap'd pages without copying:
//!
//! ```text
//! let header = PageHeader::from_bytes(&page_data[..16]);
//! ```
//!
//! ## File Header vs Page Header
//!
//! Page 0 of each file has an extended 128-byte file header that includes
//! file-specific metadata (magic, version, root page, etc.). The file header
//! is defined separately per file type (.tbd, .idx, .hnsw).
//!
//! Regular pages (1+) use only the 16-byte page header, leaving 16368 bytes
//! for actual data. Page 0 has 16256 bytes usable after the file header.
//!
//! ## Cell Layout
//!
//! Within a page, cells (key-value pairs) are stored as follows:
//!
//! ```text
//! +------------------+
//! | Header (16 bytes)|
//! +------------------+
//! | Cell Pointers    |  <- Grows downward from offset 16
//! | (2 bytes each)   |
//! +------------------+
//! | Free Space       |
//! +------------------+
//! | Cell Content     |  <- Grows upward from end of page
//! +------------------+
//! ```
//!
//! Cell pointers are 2-byte offsets into the page where each cell's content
//! is stored. This indirection allows efficient insertion and deletion without
//! moving cell content.
//!
//! ## Checksum
//!
//! Page integrity is verified using a CRC32 checksum stored in the flags field
//! combined with a checksum stored at the end of the page. This detects both
//! corruption and partial writes.
//!
//! ## Thread Safety
//!
//! `PageHeader` is a plain data structure with no synchronization. Thread safety
//! is provided by the page cache layer which controls access to page buffers.

use eyre::{ensure, Result};
use zerocopy::{FromBytes, Immutable, IntoBytes, KnownLayout};

#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PageType {
    Unknown = 0x00,
    BTreeInterior = 0x01,
    BTreeLeaf = 0x02,
    HnswNode = 0x10,
    HnswMeta = 0x11,
    Overflow = 0x20,
    FreeList = 0x30,
}

impl PageType {
    pub fn from_byte(b: u8) -> Self {
        match b {
            0x01 => PageType::BTreeInterior,
            0x02 => PageType::BTreeLeaf,
            0x10 => PageType::HnswNode,
            0x11 => PageType::HnswMeta,
            0x20 => PageType::Overflow,
            0x30 => PageType::FreeList,
            _ => PageType::Unknown,
        }
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy, FromBytes, IntoBytes, Immutable, KnownLayout)]
pub struct PageHeader {
    page_type: u8,
    flags: u8,
    cell_count: u16,
    free_start: u16,
    free_end: u16,
    frag_bytes: u8,
    reserved: [u8; 3],
    right_child: u32,
}

impl PageHeader {
    pub fn new(page_type: PageType) -> Self {
        Self {
            page_type: page_type as u8,
            flags: 0,
            cell_count: 0,
            free_start: super::PAGE_HEADER_SIZE as u16,
            free_end: super::PAGE_SIZE as u16,
            frag_bytes: 0,
            reserved: [0; 3],
            right_child: 0,
        }
    }

    pub fn from_bytes(data: &[u8]) -> Result<&Self> {
        ensure!(
            data.len() >= size_of::<Self>(),
            "buffer too small for PageHeader: {} < {}",
            data.len(),
            size_of::<Self>()
        );

        Self::ref_from_bytes(&data[..size_of::<Self>()])
            .map_err(|e| eyre::eyre!("failed to read PageHeader: {:?}", e))
    }

    pub fn from_bytes_mut(data: &mut [u8]) -> Result<&mut Self> {
        ensure!(
            data.len() >= size_of::<Self>(),
            "buffer too small for PageHeader: {} < {}",
            data.len(),
            size_of::<Self>()
        );

        Self::mut_from_bytes(&mut data[..size_of::<Self>()])
            .map_err(|e| eyre::eyre!("failed to read PageHeader: {:?}", e))
    }

    pub fn write_to(&self, data: &mut [u8]) -> Result<()> {
        ensure!(
            data.len() >= size_of::<Self>(),
            "buffer too small for PageHeader: {} < {}",
            data.len(),
            size_of::<Self>()
        );

        data[..size_of::<Self>()].copy_from_slice(self.as_bytes());
        Ok(())
    }

    pub fn page_type(&self) -> PageType {
        PageType::from_byte(self.page_type)
    }

    pub fn set_page_type(&mut self, page_type: PageType) {
        self.page_type = page_type as u8;
    }

    pub fn flags(&self) -> u8 {
        self.flags
    }

    pub fn set_flags(&mut self, flags: u8) {
        self.flags = flags;
    }

    pub fn cell_count(&self) -> u16 {
        self.cell_count
    }

    pub fn set_cell_count(&mut self, count: u16) {
        self.cell_count = count;
    }

    pub fn free_start(&self) -> u16 {
        self.free_start
    }

    pub fn set_free_start(&mut self, offset: u16) {
        self.free_start = offset;
    }

    pub fn free_end(&self) -> u16 {
        self.free_end
    }

    pub fn set_free_end(&mut self, offset: u16) {
        self.free_end = offset;
    }

    pub fn free_space(&self) -> u16 {
        self.free_end.saturating_sub(self.free_start)
    }

    pub fn frag_bytes(&self) -> u8 {
        self.frag_bytes
    }

    pub fn set_frag_bytes(&mut self, bytes: u8) {
        self.frag_bytes = bytes;
    }

    pub fn right_child(&self) -> u32 {
        self.right_child
    }

    pub fn set_right_child(&mut self, page_no: u32) {
        self.right_child = page_no;
    }

    pub fn next_leaf(&self) -> u32 {
        self.right_child
    }

    pub fn set_next_leaf(&mut self, page_no: u32) {
        self.right_child = page_no;
    }
}

pub fn validate_page(data: &[u8]) -> Result<()> {
    ensure!(
        data.len() == super::PAGE_SIZE,
        "invalid page size: {} != {}",
        data.len(),
        super::PAGE_SIZE
    );

    let header = PageHeader::from_bytes(data)?;

    let is_zeroed = header.page_type == 0
        && header.flags == 0
        && header.cell_count == 0
        && header.free_start == 0
        && header.free_end == 0;

    if is_zeroed {
        return Ok(());
    }

    ensure!(
        header.page_type() != PageType::Unknown,
        "invalid page type: {:02x}",
        header.page_type
    );

    ensure!(
        header.free_start() >= super::PAGE_HEADER_SIZE as u16,
        "free_start {} < PAGE_HEADER_SIZE {}",
        header.free_start(),
        super::PAGE_HEADER_SIZE
    );

    ensure!(
        header.free_end() <= super::PAGE_SIZE as u16,
        "free_end {} > PAGE_SIZE {}",
        header.free_end(),
        super::PAGE_SIZE
    );

    ensure!(
        header.free_start() <= header.free_end(),
        "free_start {} > free_end {}",
        header.free_start(),
        header.free_end()
    );

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn page_type_from_byte() {
        assert_eq!(PageType::from_byte(0x00), PageType::Unknown);
        assert_eq!(PageType::from_byte(0x01), PageType::BTreeInterior);
        assert_eq!(PageType::from_byte(0x02), PageType::BTreeLeaf);
        assert_eq!(PageType::from_byte(0x10), PageType::HnswNode);
        assert_eq!(PageType::from_byte(0x11), PageType::HnswMeta);
        assert_eq!(PageType::from_byte(0x20), PageType::Overflow);
        assert_eq!(PageType::from_byte(0x30), PageType::FreeList);
        assert_eq!(PageType::from_byte(0xFF), PageType::Unknown);
    }

    #[test]
    fn page_header_size_is_16_bytes() {
        assert_eq!(size_of::<PageHeader>(), 16);
    }

    #[test]
    fn page_header_new_initializes_correctly() {
        let header = PageHeader::new(PageType::BTreeLeaf);

        assert_eq!(header.page_type(), PageType::BTreeLeaf);
        assert_eq!(header.flags(), 0);
        assert_eq!(header.cell_count(), 0);
        assert_eq!(header.free_start(), super::super::PAGE_HEADER_SIZE as u16);
        assert_eq!(header.free_end(), super::super::PAGE_SIZE as u16);
        assert_eq!(header.frag_bytes(), 0);
        assert_eq!(header.right_child(), 0);
    }

    #[test]
    fn page_header_from_bytes_zero_copy() {
        let mut data = [0u8; 16];
        data[0] = 0x02;
        data[2] = 5;
        data[3] = 0;
        data[4] = 16;
        data[5] = 0;
        data[6] = 0x00;
        data[7] = 0x40;

        let header = PageHeader::from_bytes(&data).unwrap();

        assert_eq!(header.page_type(), PageType::BTreeLeaf);
        assert_eq!(header.cell_count(), 5);
        assert_eq!(header.free_start(), 16);
        assert_eq!(header.free_end(), 0x4000);
    }

    #[test]
    fn page_header_from_bytes_too_small() {
        let data = [0u8; 8];
        let result = PageHeader::from_bytes(&data);

        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("buffer too small"));
    }

    #[test]
    fn page_header_from_bytes_mut_modifies_in_place() {
        let mut data = [0u8; 16];

        {
            let header = PageHeader::from_bytes_mut(&mut data).unwrap();
            header.set_page_type(PageType::HnswNode);
            header.set_cell_count(42);
        }

        assert_eq!(data[0], 0x10);
        assert_eq!(data[2], 42);
    }

    #[test]
    fn page_header_write_to() {
        let header = PageHeader::new(PageType::FreeList);
        let mut data = [0xFFu8; 32];

        header.write_to(&mut data).unwrap();

        assert_eq!(data[0], 0x30);
        assert_eq!(data[1], 0);
    }

    #[test]
    fn page_header_write_to_too_small() {
        let header = PageHeader::new(PageType::BTreeLeaf);
        let mut data = [0u8; 8];

        let result = header.write_to(&mut data);

        assert!(result.is_err());
    }

    #[test]
    fn page_header_free_space_calculation() {
        let mut header = PageHeader::new(PageType::BTreeLeaf);

        assert_eq!(
            header.free_space(),
            super::super::PAGE_SIZE as u16 - super::super::PAGE_HEADER_SIZE as u16
        );

        header.set_free_start(100);
        header.set_free_end(1000);

        assert_eq!(header.free_space(), 900);
    }

    #[test]
    fn page_header_right_child_and_next_leaf_alias() {
        let mut header = PageHeader::new(PageType::BTreeInterior);

        header.set_right_child(12345);
        assert_eq!(header.next_leaf(), 12345);

        header.set_next_leaf(67890);
        assert_eq!(header.right_child(), 67890);
    }

    #[test]
    fn validate_page_correct_size() {
        let data = [0u8; 100];
        let result = validate_page(&data);

        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("invalid page size"));
    }

    #[test]
    fn validate_page_zeroed_is_valid() {
        let data = [0u8; super::super::PAGE_SIZE];
        let result = validate_page(&data);

        assert!(result.is_ok());
    }

    #[test]
    fn validate_page_valid_btree_leaf() {
        let mut data = [0u8; super::super::PAGE_SIZE];
        let header = PageHeader::new(PageType::BTreeLeaf);
        header.write_to(&mut data).unwrap();

        let result = validate_page(&data);

        assert!(result.is_ok());
    }

    #[test]
    fn validate_page_invalid_free_start() {
        let mut data = [0u8; super::super::PAGE_SIZE];
        data[0] = 0x02;
        data[4] = 8;
        data[5] = 0;
        data[6] = 0x00;
        data[7] = 0x40;

        let result = validate_page(&data);

        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("free_start"));
    }

    #[test]
    fn validate_page_free_start_greater_than_free_end() {
        let mut data = [0u8; super::super::PAGE_SIZE];
        data[0] = 0x02;
        data[4] = 0x00;
        data[5] = 0x10;
        data[6] = 0x00;
        data[7] = 0x08;

        let result = validate_page(&data);

        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("free_start"));
    }
}
