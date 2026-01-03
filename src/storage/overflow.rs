//! # Overflow Page Management
//!
//! This module implements SQLite-style overflow pages for handling large values
//! that exceed the usable space on a single B-tree page.
//!
//! ## Design Overview
//!
//! When a value (BLOB, TEXT, Vector, etc.) exceeds the inline threshold, it is
//! split across multiple overflow pages. The B-tree cell stores:
//! 1. Inline portion (first N bytes for prefix comparison)
//! 2. Total size of the value
//! 3. Page number of the first overflow page
//!
//! ## Overflow Page Layout
//!
//! ```text
//! +------------------+
//! | Page Header      |  16 bytes (standard page header)
//! +------------------+
//! | Overflow Header  |  8 bytes
//! |  - next_page     |  4 bytes (page number of next overflow, 0 if last)
//! |  - data_size     |  4 bytes (bytes of data on this page)
//! +------------------+
//! | Payload Data     |  up to PAGE_SIZE - 24 bytes
//! +------------------+
//! ```
//!
//! ## Overflow Chain
//!
//! Large values span multiple overflow pages linked via next_page pointers:
//!
//! ```text
//! B-tree Cell:
//! [inline_prefix | total_size | first_overflow_page]
//!      |
//!      v
//! Overflow Page 1:
//! [next_page=5 | data_size=16360 | data...]
//!      |
//!      v
//! Overflow Page 5:
//! [next_page=9 | data_size=16360 | data...]
//!      |
//!      v
//! Overflow Page 9:
//! [next_page=0 | data_size=1234 | remaining_data...]
//! ```
//!
//! ## Inline Threshold
//!
//! Following SQLite's design, we store values inline if they fit within:
//! - The cell's usable space after key and overhead
//! - A minimum inline portion for prefix comparison
//!
//! Values exceeding this threshold use overflow pages.
//!
//! ## Performance Characteristics
//!
//! - Write: O(n/page_capacity) page allocations
//! - Read: O(n/page_capacity) sequential page reads
//! - Delete: O(n/page_capacity) pages freed
//!
//! ## Thread Safety
//!
//! Overflow operations require exclusive access to storage (`&mut Storage`).
//! Thread safety is provided by the caller (typically through page locks).

use eyre::{ensure, Result};
use zerocopy::{
    byteorder::{LittleEndian, U32},
    FromBytes, Immutable, IntoBytes, KnownLayout, Unaligned,
};

use super::{PageHeader, PageType, Storage, PAGE_HEADER_SIZE, PAGE_SIZE};

/// Size of the overflow page header (after the standard page header)
pub const OVERFLOW_HEADER_SIZE: usize = 8;

/// Usable data space per overflow page
pub const OVERFLOW_DATA_SIZE: usize = PAGE_SIZE - PAGE_HEADER_SIZE - OVERFLOW_HEADER_SIZE;

/// Minimum bytes to keep inline in B-tree cell for prefix comparison
pub const MIN_INLINE_PAYLOAD: usize = 64;

/// Maximum inline payload before requiring overflow (leaving room for key + overhead)
/// This is a reasonable default; actual threshold depends on available cell space
pub const MAX_INLINE_PAYLOAD: usize = 2048;

/// Overflow page header structure
#[repr(C)]
#[derive(Debug, Clone, Copy, FromBytes, IntoBytes, Immutable, KnownLayout, Unaligned)]
pub struct OverflowHeader {
    /// Page number of the next overflow page (0 if this is the last page)
    next_page: U32<LittleEndian>,
    /// Number of bytes of payload data stored on this page
    data_size: U32<LittleEndian>,
}

impl OverflowHeader {
    /// Creates a new overflow header
    pub fn new(next_page: u32, data_size: u32) -> Self {
        Self {
            next_page: U32::new(next_page),
            data_size: U32::new(data_size),
        }
    }

    /// Parses an overflow header from page data
    pub fn from_bytes(data: &[u8]) -> Result<&Self> {
        ensure!(
            data.len() >= PAGE_HEADER_SIZE + OVERFLOW_HEADER_SIZE,
            "buffer too small for overflow header"
        );
        Self::ref_from_bytes(&data[PAGE_HEADER_SIZE..PAGE_HEADER_SIZE + OVERFLOW_HEADER_SIZE])
            .map_err(|e| eyre::eyre!("failed to read overflow header: {:?}", e))
    }

    /// Parses a mutable overflow header from page data
    pub fn from_bytes_mut(data: &mut [u8]) -> Result<&mut Self> {
        ensure!(
            data.len() >= PAGE_HEADER_SIZE + OVERFLOW_HEADER_SIZE,
            "buffer too small for overflow header"
        );
        Self::mut_from_bytes(&mut data[PAGE_HEADER_SIZE..PAGE_HEADER_SIZE + OVERFLOW_HEADER_SIZE])
            .map_err(|e| eyre::eyre!("failed to read overflow header: {:?}", e))
    }

    /// Returns the next overflow page number (0 if last)
    pub fn next_page(&self) -> u32 {
        self.next_page.get()
    }

    /// Sets the next overflow page number
    pub fn set_next_page(&mut self, page_no: u32) {
        self.next_page.set(page_no);
    }

    /// Returns the number of data bytes on this page
    pub fn data_size(&self) -> u32 {
        self.data_size.get()
    }

    /// Sets the data size
    pub fn set_data_size(&mut self, size: u32) {
        self.data_size.set(size);
    }
}

/// Offset where payload data begins on an overflow page
pub const OVERFLOW_DATA_OFFSET: usize = PAGE_HEADER_SIZE + OVERFLOW_HEADER_SIZE;

/// Determines if a value needs overflow pages based on available cell space
#[inline]
pub fn needs_overflow(value_len: usize, available_cell_space: usize) -> bool {
    value_len > available_cell_space
}

/// Calculates how many overflow pages are needed for the given payload
#[inline]
pub fn overflow_pages_needed(overflow_bytes: usize) -> usize {
    if overflow_bytes == 0 {
        0
    } else {
        overflow_bytes.div_ceil(OVERFLOW_DATA_SIZE)
    }
}

/// Writes a large value across overflow pages
///
/// Returns the page number of the first overflow page and the list of
/// all allocated page numbers.
///
/// # Arguments
/// * `storage` - The storage to write to
/// * `data` - The data to write (overflow portion only)
/// * `allocate_page` - Callback to allocate a new page, returns page number
pub fn write_overflow<S: Storage, F: FnMut(&mut S) -> Result<u32>>(
    storage: &mut S,
    data: &[u8],
    mut allocate_page: F,
) -> Result<(u32, Vec<u32>)> {
    if data.is_empty() {
        return Ok((0, Vec::new()));
    }

    let pages_needed = overflow_pages_needed(data.len());
    let mut allocated_pages = Vec::with_capacity(pages_needed);

    // Allocate all pages first
    for _ in 0..pages_needed {
        allocated_pages.push(allocate_page(storage)?);
    }

    let first_page = allocated_pages[0];

    // Write data across pages
    let mut remaining = data;
    for (i, &page_no) in allocated_pages.iter().enumerate() {
        let chunk_size = remaining.len().min(OVERFLOW_DATA_SIZE);
        let next_page = if i + 1 < allocated_pages.len() {
            allocated_pages[i + 1]
        } else {
            0
        };

        let page = storage.page_mut(page_no)?;

        // Initialize page header
        let page_header = PageHeader::from_bytes_mut(page)?;
        page_header.set_page_type(PageType::Overflow);
        page_header.set_cell_count(0);
        page_header.set_free_start(OVERFLOW_DATA_OFFSET as u16);
        page_header.set_free_end(PAGE_SIZE as u16);

        // Write overflow header
        let overflow_header = OverflowHeader::from_bytes_mut(page)?;
        overflow_header.set_next_page(next_page);
        overflow_header.set_data_size(chunk_size as u32);

        // Write payload data
        page[OVERFLOW_DATA_OFFSET..OVERFLOW_DATA_OFFSET + chunk_size]
            .copy_from_slice(&remaining[..chunk_size]);

        remaining = &remaining[chunk_size..];
    }

    Ok((first_page, allocated_pages))
}

/// Reads a value from overflow pages into a buffer
///
/// # Arguments
/// * `storage` - The storage to read from
/// * `first_page` - Page number of the first overflow page
/// * `total_size` - Total size of the value to read
///
/// # Returns
/// The complete value data
pub fn read_overflow<S: Storage>(
    storage: &S,
    first_page: u32,
    total_size: usize,
) -> Result<Vec<u8>> {
    if first_page == 0 || total_size == 0 {
        return Ok(Vec::new());
    }

    let mut result = Vec::with_capacity(total_size);
    let mut current_page = first_page;

    while current_page != 0 && result.len() < total_size {
        let page = storage.page(current_page)?;

        // Validate page type
        let page_header = PageHeader::from_bytes(page)?;
        ensure!(
            page_header.page_type() == PageType::Overflow,
            "expected overflow page at {}, got {:?}",
            current_page,
            page_header.page_type()
        );

        // Read overflow header
        let overflow_header = OverflowHeader::from_bytes(page)?;
        let data_size = overflow_header.data_size() as usize;

        // Read payload data
        let bytes_to_read = data_size.min(total_size - result.len());
        result.extend_from_slice(&page[OVERFLOW_DATA_OFFSET..OVERFLOW_DATA_OFFSET + bytes_to_read]);

        current_page = overflow_header.next_page();
    }

    ensure!(
        result.len() == total_size,
        "overflow read incomplete: got {} bytes, expected {}",
        result.len(),
        total_size
    );

    Ok(result)
}

/// Reads overflow data into a pre-allocated buffer
///
/// This is more efficient than `read_overflow` when the caller already has
/// a buffer of the correct size.
pub fn read_overflow_into<S: Storage>(
    storage: &S,
    first_page: u32,
    buffer: &mut [u8],
) -> Result<()> {
    if first_page == 0 || buffer.is_empty() {
        return Ok(());
    }

    let total_size = buffer.len();
    let mut offset = 0;
    let mut current_page = first_page;

    while current_page != 0 && offset < total_size {
        let page = storage.page(current_page)?;

        // Validate page type
        let page_header = PageHeader::from_bytes(page)?;
        ensure!(
            page_header.page_type() == PageType::Overflow,
            "expected overflow page at {}, got {:?}",
            current_page,
            page_header.page_type()
        );

        // Read overflow header
        let overflow_header = OverflowHeader::from_bytes(page)?;
        let data_size = overflow_header.data_size() as usize;

        // Copy payload data
        let bytes_to_read = data_size.min(total_size - offset);
        buffer[offset..offset + bytes_to_read]
            .copy_from_slice(&page[OVERFLOW_DATA_OFFSET..OVERFLOW_DATA_OFFSET + bytes_to_read]);
        offset += bytes_to_read;

        current_page = overflow_header.next_page();
    }

    ensure!(
        offset == total_size,
        "overflow read incomplete: got {} bytes, expected {}",
        offset,
        total_size
    );

    Ok(())
}

/// Frees all overflow pages in a chain
///
/// # Arguments
/// * `storage` - The storage to read from
/// * `first_page` - Page number of the first overflow page
/// * `free_page` - Callback to free a page
///
/// # Returns
/// The list of freed page numbers
pub fn free_overflow_chain<S: Storage, F: FnMut(&mut S, u32) -> Result<()>>(
    storage: &mut S,
    first_page: u32,
    mut free_page: F,
) -> Result<Vec<u32>> {
    if first_page == 0 {
        return Ok(Vec::new());
    }

    let mut freed_pages = Vec::new();
    let mut current_page = first_page;

    // First, collect all page numbers in the chain
    while current_page != 0 {
        let page = storage.page(current_page)?;
        let page_header = PageHeader::from_bytes(page)?;

        if page_header.page_type() != PageType::Overflow {
            break;
        }

        let overflow_header = OverflowHeader::from_bytes(page)?;
        freed_pages.push(current_page);
        current_page = overflow_header.next_page();
    }

    // Then free all pages
    for &page_no in &freed_pages {
        free_page(storage, page_no)?;
    }

    Ok(freed_pages)
}

/// Incremental BLOB I/O handle for reading/writing portions of a large value
///
/// This allows reading or writing portions of a BLOB without loading the
/// entire thing into memory, similar to SQLite's `sqlite3_blob_open()`.
pub struct BlobHandle<'a, S: Storage> {
    storage: &'a S,
    first_page: u32,
    inline_data: &'a [u8],
    total_size: usize,
}

impl<'a, S: Storage> BlobHandle<'a, S> {
    /// Creates a new BLOB handle for incremental I/O
    pub fn new(
        storage: &'a S,
        first_page: u32,
        inline_data: &'a [u8],
        total_size: usize,
    ) -> Self {
        Self {
            storage,
            first_page,
            inline_data,
            total_size,
        }
    }

    /// Returns the total size of the BLOB
    pub fn size(&self) -> usize {
        self.total_size
    }

    /// Reads a portion of the BLOB into the provided buffer
    ///
    /// # Arguments
    /// * `offset` - Starting offset within the BLOB
    /// * `buffer` - Buffer to read into
    ///
    /// # Returns
    /// Number of bytes actually read
    pub fn read(&self, offset: usize, buffer: &mut [u8]) -> Result<usize> {
        if offset >= self.total_size || buffer.is_empty() {
            return Ok(0);
        }

        let bytes_available = self.total_size - offset;
        let bytes_to_read = buffer.len().min(bytes_available);
        let mut bytes_read = 0;

        // Read from inline portion if applicable
        let inline_len = self.inline_data.len();
        if offset < inline_len {
            let inline_bytes = (inline_len - offset).min(bytes_to_read);
            buffer[..inline_bytes].copy_from_slice(&self.inline_data[offset..offset + inline_bytes]);
            bytes_read = inline_bytes;

            if bytes_read == bytes_to_read {
                return Ok(bytes_read);
            }
        }

        // Read from overflow pages
        if self.first_page != 0 {
            let overflow_offset = offset.saturating_sub(inline_len);
            let overflow_bytes_to_read = bytes_to_read - bytes_read;

            self.read_overflow_portion(
                overflow_offset,
                &mut buffer[bytes_read..bytes_read + overflow_bytes_to_read],
            )?;
            bytes_read += overflow_bytes_to_read;
        }

        Ok(bytes_read)
    }

    /// Reads a portion from the overflow chain
    fn read_overflow_portion(&self, offset: usize, buffer: &mut [u8]) -> Result<()> {
        if buffer.is_empty() {
            return Ok(());
        }

        // Skip pages until we reach the offset
        let mut current_page = self.first_page;
        let mut bytes_skipped = 0;

        while current_page != 0 && bytes_skipped + OVERFLOW_DATA_SIZE <= offset {
            let page = self.storage.page(current_page)?;
            let overflow_header = OverflowHeader::from_bytes(page)?;
            bytes_skipped += overflow_header.data_size() as usize;
            current_page = overflow_header.next_page();
        }

        // Read data starting from the current page
        let mut buffer_offset = 0;
        let page_offset = offset - bytes_skipped;

        while current_page != 0 && buffer_offset < buffer.len() {
            let page = self.storage.page(current_page)?;
            let overflow_header = OverflowHeader::from_bytes(page)?;
            let data_size = overflow_header.data_size() as usize;

            let start = if buffer_offset == 0 { page_offset } else { 0 };
            let end = data_size.min(start + buffer.len() - buffer_offset);
            let bytes_to_copy = end - start;

            buffer[buffer_offset..buffer_offset + bytes_to_copy].copy_from_slice(
                &page[OVERFLOW_DATA_OFFSET + start..OVERFLOW_DATA_OFFSET + end],
            );

            buffer_offset += bytes_to_copy;
            current_page = overflow_header.next_page();
        }

        Ok(())
    }
}

/// Mutable BLOB handle for writing portions of a large value
pub struct BlobHandleMut<'a, S: Storage> {
    storage: &'a mut S,
    first_page: u32,
    total_size: usize,
}

impl<'a, S: Storage> BlobHandleMut<'a, S> {
    /// Creates a new mutable BLOB handle
    pub fn new(storage: &'a mut S, first_page: u32, total_size: usize) -> Self {
        Self {
            storage,
            first_page,
            total_size,
        }
    }

    /// Returns the total size of the BLOB
    pub fn size(&self) -> usize {
        self.total_size
    }

    /// Writes data to a portion of the BLOB
    ///
    /// # Arguments
    /// * `offset` - Starting offset within the overflow portion
    /// * `data` - Data to write
    ///
    /// # Returns
    /// Number of bytes actually written
    pub fn write(&mut self, offset: usize, data: &[u8]) -> Result<usize> {
        if offset >= self.total_size || data.is_empty() {
            return Ok(0);
        }

        let bytes_available = self.total_size - offset;
        let bytes_to_write = data.len().min(bytes_available);

        // Skip pages until we reach the offset
        let mut current_page = self.first_page;
        let mut bytes_skipped = 0;

        while current_page != 0 && bytes_skipped + OVERFLOW_DATA_SIZE <= offset {
            let page = self.storage.page(current_page)?;
            let overflow_header = OverflowHeader::from_bytes(page)?;
            bytes_skipped += overflow_header.data_size() as usize;
            current_page = overflow_header.next_page();
        }

        // Write data starting from the current page
        let mut data_offset = 0;
        let page_offset = offset - bytes_skipped;

        while current_page != 0 && data_offset < bytes_to_write {
            let page = self.storage.page_mut(current_page)?;
            let overflow_header = OverflowHeader::from_bytes(page)?;
            let data_size = overflow_header.data_size() as usize;
            let next_page = overflow_header.next_page();

            let start = if data_offset == 0 { page_offset } else { 0 };
            let end = data_size.min(start + bytes_to_write - data_offset);
            let bytes_to_copy = end - start;

            page[OVERFLOW_DATA_OFFSET + start..OVERFLOW_DATA_OFFSET + end]
                .copy_from_slice(&data[data_offset..data_offset + bytes_to_copy]);

            data_offset += bytes_to_copy;
            current_page = next_page;
        }

        Ok(data_offset)
    }
}

/// Overflow pointer stored inline in B-tree cells
///
/// When a value overflows, the cell stores this structure instead of the full value.
#[repr(C)]
#[derive(Debug, Clone, Copy, FromBytes, IntoBytes, Immutable, KnownLayout, Unaligned)]
pub struct OverflowPointer {
    /// Total size of the value (including inline portion)
    pub total_size: U32<LittleEndian>,
    /// Page number of the first overflow page
    pub first_page: U32<LittleEndian>,
}

/// Size of the overflow pointer structure
pub const OVERFLOW_POINTER_SIZE: usize = 8;

impl OverflowPointer {
    /// Creates a new overflow pointer
    pub fn new(total_size: u32, first_page: u32) -> Self {
        Self {
            total_size: U32::new(total_size),
            first_page: U32::new(first_page),
        }
    }

    /// Parses an overflow pointer from bytes
    pub fn from_bytes(data: &[u8]) -> Result<&Self> {
        ensure!(
            data.len() >= OVERFLOW_POINTER_SIZE,
            "buffer too small for overflow pointer"
        );
        Self::ref_from_bytes(&data[..OVERFLOW_POINTER_SIZE])
            .map_err(|e| eyre::eyre!("failed to read overflow pointer: {:?}", e))
    }

    /// Returns the total value size
    pub fn total_size(&self) -> u32 {
        self.total_size.get()
    }

    /// Returns the first overflow page number
    pub fn first_page(&self) -> u32 {
        self.first_page.get()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_overflow_header_size() {
        assert_eq!(std::mem::size_of::<OverflowHeader>(), OVERFLOW_HEADER_SIZE);
    }

    #[test]
    fn test_overflow_pointer_size() {
        assert_eq!(std::mem::size_of::<OverflowPointer>(), OVERFLOW_POINTER_SIZE);
    }

    #[test]
    fn test_overflow_pages_needed() {
        assert_eq!(overflow_pages_needed(0), 0);
        assert_eq!(overflow_pages_needed(1), 1);
        assert_eq!(overflow_pages_needed(OVERFLOW_DATA_SIZE), 1);
        assert_eq!(overflow_pages_needed(OVERFLOW_DATA_SIZE + 1), 2);
        assert_eq!(overflow_pages_needed(OVERFLOW_DATA_SIZE * 3), 3);
        assert_eq!(overflow_pages_needed(OVERFLOW_DATA_SIZE * 3 + 100), 4);
    }

    #[test]
    fn test_overflow_data_size() {
        // Verify that overflow data size leaves room for headers
        assert_eq!(
            OVERFLOW_DATA_SIZE,
            PAGE_SIZE - PAGE_HEADER_SIZE - OVERFLOW_HEADER_SIZE
        );
        // Should be 16384 - 16 - 8 = 16360 bytes
        assert_eq!(OVERFLOW_DATA_SIZE, 16360);
    }

    #[test]
    fn test_overflow_header_roundtrip() {
        let mut page = vec![0u8; PAGE_SIZE];

        // Initialize page header
        let page_header = PageHeader::from_bytes_mut(&mut page).unwrap();
        page_header.set_page_type(PageType::Overflow);

        // Write overflow header
        {
            let header = OverflowHeader::from_bytes_mut(&mut page).unwrap();
            header.set_next_page(42);
            header.set_data_size(1234);
        }

        // Read back
        let header = OverflowHeader::from_bytes(&page).unwrap();
        assert_eq!(header.next_page(), 42);
        assert_eq!(header.data_size(), 1234);
    }

    #[test]
    fn test_overflow_pointer_roundtrip() {
        let ptr = OverflowPointer::new(123456, 789);
        let bytes = ptr.as_bytes();

        let parsed = OverflowPointer::from_bytes(bytes).unwrap();
        assert_eq!(parsed.total_size(), 123456);
        assert_eq!(parsed.first_page(), 789);
    }
}
