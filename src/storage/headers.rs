//! # File Header Definitions
//!
//! This module provides type-safe, zerocopy-based header structs for all TurDB file types.
//! Each file type has a 128-byte header at the beginning that contains magic bytes,
//! version information, and type-specific metadata.
//!
//! ## File Types
//!
//! 1. **turdb.meta** - Global database metadata (MetaFileHeader)
//!    - Tracks schema count, default schema, next table/index IDs
//!    - Written once on database creation, updated when IDs change
//!
//! 2. **.tbd** - Table data files (TableFileHeader)
//!    - Tracks row count, root page, column count, free list
//!    - Updated during inserts, deletes, and page splits
//!
//! 3. **.idx** - B-tree index files (IndexFileHeader)
//!    - Tracks index-table relationship, uniqueness, index type
//!    - Updated during index modifications
//!
//! ## Header Layout
//!
//! All headers are exactly 128 bytes and occupy the first 128 bytes of page 0.
//! The remaining 16256 bytes of page 0 are available for use.
//!
//! ```text
//! +------------------+
//! | Header (128B)    |  <- File-type specific header
//! +------------------+
//! | Page 0 Data      |  <- 16256 bytes available
//! +------------------+
//! | Page 1+ (16KB)   |  <- Full 16384-byte pages
//! +------------------+
//! ```
//!
//! ## Zerocopy Safety
//!
//! All header structs use zerocopy traits for safe, zero-copy serialization:
//! - `FromBytes`: Safe to read from arbitrary bytes
//! - `IntoBytes`: Safe to write as bytes
//! - `Immutable`: No interior mutability
//! - `KnownLayout`: Compile-time size verification
//! - `Unaligned`: Works with unaligned memory (mmap)
//!
//! ## Usage Example
//!
//! ```ignore
//! use zerocopy::IntoBytes;
//!
//! // Create a new table header
//! let header = TableFileHeader::new(1, 0, 1, 5, 0, 0);
//!
//! // Write to page 0
//! let page = storage.page_mut(0)?;
//! page[..128].copy_from_slice(header.as_bytes());
//!
//! // Read from page 0
//! let header = TableFileHeader::from_bytes(&page[..128])?;
//! let row_count = header.row_count();
//! ```
//!
//! ## Endianness
//!
//! All multi-byte fields use little-endian encoding for x86/ARM compatibility.
//! The zerocopy `U32<LittleEndian>` and `U64<LittleEndian>` types handle
//! conversion automatically.

use eyre::{ensure, Result};
use zerocopy::little_endian::{U32, U64};
use zerocopy::{FromBytes, Immutable, IntoBytes, KnownLayout, Unaligned};

use super::FILE_HEADER_SIZE;

pub const META_MAGIC: &[u8; 16] = b"TurDB Rust v1\x00\x00\x00";
pub const TABLE_MAGIC: &[u8; 16] = b"TurDB Table\x00\x00\x00\x00\x00";
pub const INDEX_MAGIC: &[u8; 16] = b"TurDB Index\x00\x00\x00\x00\x00";

pub const CURRENT_VERSION: u32 = 1;
pub const DEFAULT_PAGE_SIZE: u32 = 16384;

#[repr(C)]
#[derive(Debug, Clone, Copy, FromBytes, IntoBytes, Immutable, KnownLayout, Unaligned)]
pub struct MetaFileHeader {
    magic: [u8; 16],
    version: U32,
    page_size: U32,
    schema_count: U64,
    default_schema_id: U64,
    next_table_id: U64,
    next_index_id: U64,
    flags: U64,
    reserved: [u8; 64],
}

const _: () = assert!(std::mem::size_of::<MetaFileHeader>() == FILE_HEADER_SIZE);

impl MetaFileHeader {
    pub fn new() -> Self {
        Self {
            magic: *META_MAGIC,
            version: U32::new(CURRENT_VERSION),
            page_size: U32::new(DEFAULT_PAGE_SIZE),
            schema_count: U64::new(1),
            default_schema_id: U64::new(0),
            next_table_id: U64::new(1),
            next_index_id: U64::new(1),
            flags: U64::new(0),
            reserved: [0u8; 64],
        }
    }

    pub fn from_bytes(bytes: &[u8]) -> Result<&Self> {
        ensure!(
            bytes.len() >= FILE_HEADER_SIZE,
            "buffer too small for MetaFileHeader: {} < {}",
            bytes.len(),
            FILE_HEADER_SIZE
        );

        let header = Self::ref_from_bytes(&bytes[..FILE_HEADER_SIZE])
            .map_err(|e| eyre::eyre!("failed to parse MetaFileHeader: {:?}", e))?;

        ensure!(
            &header.magic == META_MAGIC,
            "invalid magic bytes in turdb.meta"
        );

        ensure!(
            header.version.get() == CURRENT_VERSION,
            "unsupported version: {} (expected {})",
            header.version.get(),
            CURRENT_VERSION
        );

        Ok(header)
    }

    pub fn version(&self) -> u32 {
        self.version.get()
    }

    pub fn page_size(&self) -> u32 {
        self.page_size.get()
    }

    pub fn schema_count(&self) -> u64 {
        self.schema_count.get()
    }

    pub fn set_schema_count(&mut self, count: u64) {
        self.schema_count = U64::new(count);
    }

    pub fn default_schema_id(&self) -> u64 {
        self.default_schema_id.get()
    }

    pub fn set_default_schema_id(&mut self, id: u64) {
        self.default_schema_id = U64::new(id);
    }

    pub fn next_table_id(&self) -> u64 {
        self.next_table_id.get()
    }

    pub fn set_next_table_id(&mut self, id: u64) {
        self.next_table_id = U64::new(id);
    }

    pub fn next_index_id(&self) -> u64 {
        self.next_index_id.get()
    }

    pub fn set_next_index_id(&mut self, id: u64) {
        self.next_index_id = U64::new(id);
    }

    pub fn flags(&self) -> u64 {
        self.flags.get()
    }

    pub fn set_flags(&mut self, flags: u64) {
        self.flags = U64::new(flags);
    }
}

impl Default for MetaFileHeader {
    fn default() -> Self {
        Self::new()
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy, FromBytes, IntoBytes, Immutable, KnownLayout, Unaligned)]
pub struct TableFileHeader {
    magic: [u8; 16],
    table_id: U64,
    row_count: U64,
    root_page: U32,
    column_count: U32,
    first_free_page: U64,
    auto_increment: U64,
    reserved: [u8; 72],
}

const _: () = assert!(std::mem::size_of::<TableFileHeader>() == FILE_HEADER_SIZE);

impl TableFileHeader {
    pub fn new(
        table_id: u64,
        row_count: u64,
        root_page: u32,
        column_count: u32,
        first_free_page: u64,
        auto_increment: u64,
    ) -> Self {
        Self {
            magic: *TABLE_MAGIC,
            table_id: U64::new(table_id),
            row_count: U64::new(row_count),
            root_page: U32::new(root_page),
            column_count: U32::new(column_count),
            first_free_page: U64::new(first_free_page),
            auto_increment: U64::new(auto_increment),
            reserved: [0u8; 72],
        }
    }

    pub fn from_bytes(bytes: &[u8]) -> Result<&Self> {
        ensure!(
            bytes.len() >= FILE_HEADER_SIZE,
            "buffer too small for TableFileHeader: {} < {}",
            bytes.len(),
            FILE_HEADER_SIZE
        );

        let header = Self::ref_from_bytes(&bytes[..FILE_HEADER_SIZE])
            .map_err(|e| eyre::eyre!("failed to parse TableFileHeader: {:?}", e))?;

        ensure!(
            &header.magic == TABLE_MAGIC,
            "invalid magic bytes in table file"
        );

        Ok(header)
    }

    pub fn from_bytes_mut(bytes: &mut [u8]) -> Result<&mut Self> {
        ensure!(
            bytes.len() >= FILE_HEADER_SIZE,
            "buffer too small for TableFileHeader: {} < {}",
            bytes.len(),
            FILE_HEADER_SIZE
        );

        let header = Self::mut_from_bytes(&mut bytes[..FILE_HEADER_SIZE])
            .map_err(|e| eyre::eyre!("failed to parse TableFileHeader: {:?}", e))?;

        ensure!(
            &header.magic == TABLE_MAGIC,
            "invalid magic bytes in table file"
        );

        Ok(header)
    }

    pub fn table_id(&self) -> u64 {
        self.table_id.get()
    }

    pub fn row_count(&self) -> u64 {
        self.row_count.get()
    }

    pub fn set_row_count(&mut self, count: u64) {
        self.row_count = U64::new(count);
    }

    pub fn increment_row_count(&mut self) {
        self.row_count = U64::new(self.row_count.get() + 1);
    }

    pub fn root_page(&self) -> u32 {
        self.root_page.get()
    }

    pub fn set_root_page(&mut self, page: u32) {
        self.root_page = U32::new(page);
    }

    pub fn column_count(&self) -> u32 {
        self.column_count.get()
    }

    pub fn first_free_page(&self) -> u64 {
        self.first_free_page.get()
    }

    pub fn set_first_free_page(&mut self, page: u64) {
        self.first_free_page = U64::new(page);
    }

    pub fn auto_increment(&self) -> u64 {
        self.auto_increment.get()
    }

    pub fn set_auto_increment(&mut self, value: u64) {
        self.auto_increment = U64::new(value);
    }

    pub fn next_auto_increment(&mut self) -> u64 {
        let current = self.auto_increment.get();
        self.auto_increment = U64::new(current + 1);
        current
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy, FromBytes, IntoBytes, Immutable, KnownLayout, Unaligned)]
pub struct IndexFileHeader {
    magic: [u8; 16],
    index_id: U64,
    table_id: U64,
    root_page: U32,
    key_column_count: U32,
    is_unique: u8,
    index_type: u8,
    reserved: [u8; 86],
}

const _: () = assert!(std::mem::size_of::<IndexFileHeader>() == FILE_HEADER_SIZE);

pub const INDEX_TYPE_BTREE: u8 = 0;
pub const INDEX_TYPE_HASH: u8 = 1;

impl IndexFileHeader {
    pub fn new(
        index_id: u64,
        table_id: u64,
        root_page: u32,
        key_column_count: u32,
        is_unique: bool,
        index_type: u8,
    ) -> Self {
        Self {
            magic: *INDEX_MAGIC,
            index_id: U64::new(index_id),
            table_id: U64::new(table_id),
            root_page: U32::new(root_page),
            key_column_count: U32::new(key_column_count),
            is_unique: if is_unique { 1 } else { 0 },
            index_type,
            reserved: [0u8; 86],
        }
    }

    pub fn from_bytes(bytes: &[u8]) -> Result<&Self> {
        ensure!(
            bytes.len() >= FILE_HEADER_SIZE,
            "buffer too small for IndexFileHeader: {} < {}",
            bytes.len(),
            FILE_HEADER_SIZE
        );

        let header = Self::ref_from_bytes(&bytes[..FILE_HEADER_SIZE])
            .map_err(|e| eyre::eyre!("failed to parse IndexFileHeader: {:?}", e))?;

        ensure!(
            &header.magic == INDEX_MAGIC,
            "invalid magic bytes in index file"
        );

        Ok(header)
    }

    pub fn from_bytes_mut(bytes: &mut [u8]) -> Result<&mut Self> {
        ensure!(
            bytes.len() >= FILE_HEADER_SIZE,
            "buffer too small for IndexFileHeader: {} < {}",
            bytes.len(),
            FILE_HEADER_SIZE
        );

        let header = Self::mut_from_bytes(&mut bytes[..FILE_HEADER_SIZE])
            .map_err(|e| eyre::eyre!("failed to parse IndexFileHeader: {:?}", e))?;

        ensure!(
            &header.magic == INDEX_MAGIC,
            "invalid magic bytes in index file"
        );

        Ok(header)
    }

    pub fn index_id(&self) -> u64 {
        self.index_id.get()
    }

    pub fn table_id(&self) -> u64 {
        self.table_id.get()
    }

    pub fn root_page(&self) -> u32 {
        self.root_page.get()
    }

    pub fn set_root_page(&mut self, page: u32) {
        self.root_page = U32::new(page);
    }

    pub fn key_column_count(&self) -> u32 {
        self.key_column_count.get()
    }

    pub fn is_unique(&self) -> bool {
        self.is_unique != 0
    }

    pub fn index_type(&self) -> u8 {
        self.index_type
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn meta_header_size_is_128() {
        assert_eq!(std::mem::size_of::<MetaFileHeader>(), 128);
    }

    #[test]
    fn table_header_size_is_128() {
        assert_eq!(std::mem::size_of::<TableFileHeader>(), 128);
    }

    #[test]
    fn index_header_size_is_128() {
        assert_eq!(std::mem::size_of::<IndexFileHeader>(), 128);
    }

    #[test]
    fn meta_header_roundtrip() {
        let mut header = MetaFileHeader::new();
        header.set_schema_count(5);
        header.set_next_table_id(42);
        header.set_next_index_id(17);

        let bytes = header.as_bytes();
        let parsed = MetaFileHeader::from_bytes(bytes).unwrap();

        assert_eq!(parsed.version(), CURRENT_VERSION);
        assert_eq!(parsed.page_size(), DEFAULT_PAGE_SIZE);
        assert_eq!(parsed.schema_count(), 5);
        assert_eq!(parsed.next_table_id(), 42);
        assert_eq!(parsed.next_index_id(), 17);
    }

    #[test]
    fn table_header_roundtrip() {
        let header = TableFileHeader::new(1, 100, 2, 5, 0, 50);

        let bytes = header.as_bytes();
        let parsed = TableFileHeader::from_bytes(bytes).unwrap();

        assert_eq!(parsed.table_id(), 1);
        assert_eq!(parsed.row_count(), 100);
        assert_eq!(parsed.root_page(), 2);
        assert_eq!(parsed.column_count(), 5);
        assert_eq!(parsed.auto_increment(), 50);
    }

    #[test]
    fn index_header_roundtrip() {
        let header = IndexFileHeader::new(1, 2, 3, 2, true, INDEX_TYPE_BTREE);

        let bytes = header.as_bytes();
        let parsed = IndexFileHeader::from_bytes(bytes).unwrap();

        assert_eq!(parsed.index_id(), 1);
        assert_eq!(parsed.table_id(), 2);
        assert_eq!(parsed.root_page(), 3);
        assert_eq!(parsed.key_column_count(), 2);
        assert!(parsed.is_unique());
        assert_eq!(parsed.index_type(), INDEX_TYPE_BTREE);
    }

    #[test]
    fn meta_header_rejects_invalid_magic() {
        let mut bytes = [0u8; 128];
        bytes[..16].copy_from_slice(b"Invalid Magic!!!");

        let result = MetaFileHeader::from_bytes(&bytes);
        assert!(result.is_err());
    }

    #[test]
    fn table_header_rejects_invalid_magic() {
        let mut bytes = [0u8; 128];
        bytes[..16].copy_from_slice(b"Invalid Magic!!!");

        let result = TableFileHeader::from_bytes(&bytes);
        assert!(result.is_err());
    }

    #[test]
    fn index_header_rejects_invalid_magic() {
        let mut bytes = [0u8; 128];
        bytes[..16].copy_from_slice(b"Invalid Magic!!!");

        let result = IndexFileHeader::from_bytes(&bytes);
        assert!(result.is_err());
    }

    #[test]
    fn table_header_mutable_access() {
        let mut bytes = [0u8; 128];
        let header = TableFileHeader::new(1, 0, 1, 5, 0, 0);
        bytes.copy_from_slice(header.as_bytes());

        let header_mut = TableFileHeader::from_bytes_mut(&mut bytes).unwrap();
        header_mut.set_row_count(100);
        header_mut.set_auto_increment(50);

        let header_ref = TableFileHeader::from_bytes(&bytes).unwrap();
        assert_eq!(header_ref.row_count(), 100);
        assert_eq!(header_ref.auto_increment(), 50);
    }

    #[test]
    fn auto_increment_works() {
        let mut header = TableFileHeader::new(1, 0, 1, 5, 0, 1);

        assert_eq!(header.next_auto_increment(), 1);
        assert_eq!(header.next_auto_increment(), 2);
        assert_eq!(header.next_auto_increment(), 3);
        assert_eq!(header.auto_increment(), 4);
    }
}
