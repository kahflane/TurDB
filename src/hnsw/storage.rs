//! # HNSW Storage Layer
//!
//! This module implements the persistent storage layer for HNSW vector indexes,
//! including file format, page layout, and integration with TurDB's storage
//! infrastructure.
//!
//! ## File Format (.hnsw)
//!
//! Each HNSW index is stored in a dedicated file with the following structure:
//!
//! ```text
//! +------------------+
//! | File Header      |  Page 0: 128-byte header + metadata
//! | (128 bytes)      |
//! +------------------+
//! | Node Pages       |  Pages 1+: HNSW graph nodes
//! | (slotted format) |
//! +------------------+
//! ```
//!
//! ## File Header Layout (128 bytes)
//!
//! ```text
//! Offset  Size  Field             Description
//! ------  ----  ---------------   ----------------------------------------
//! 0       16    magic             "TurDB HNSW\x00\x00\x00\x00\x00\x00"
//! 16      8     index_id          Unique index identifier
//! 24      8     table_id          Associated table ID
//! 32      2     dimensions        Vector dimensionality
//! 34      2     m                 Max neighbors per layer (except L0)
//! 36      2     m0                Max neighbors at layer 0 (= 2*M)
//! 38      2     ef_construction   Search width during construction
//! 40      2     ef_search         Default search width for queries
//! 42      1     distance_fn       0=L2, 1=Cosine, 2=InnerProduct
//! 43      1     quantization      0=None, 1=SQ8, 2=PQ
//! 44      4     entry_point       Entry point node page number
//! 48      2     entry_slot        Entry point slot index
//! 50      1     max_level         Maximum level in the graph
//! 51      1     reserved          Reserved
//! 52      8     node_count        Total number of nodes
//! 60      8     vector_count      Number of vectors (may differ if deleted)
//! 68      4     first_free_page   First page in free list
//! 72      56    reserved          Reserved for future use
//! ```
//!
//! ## Node Page Layout (16KB slotted page)
//!
//! ```text
//! +------------------+
//! | Page Header      |  16 bytes (standard TurDB header)
//! +------------------+
//! | HNSW Page Hdr    |  48 bytes (HNSW-specific metadata)
//! +------------------+
//! | Slot Directory   |  Variable (grows down from offset 64)
//! | (4 bytes/slot)   |
//! +------------------+
//! | Free Space       |
//! +------------------+
//! | Node Data        |  Variable (grows up from page end)
//! +------------------+
//! ```
//!
//! ## Slot Directory Entry (4 bytes)
//!
//! ```text
//! Bits    Field       Description
//! 0-12    offset      Offset to node data within page
//! 13-14   status      0=free, 1=active, 2=deleted
//! 15      reserved    Reserved bit
//! 16-31   size        Size of node data in bytes
//! ```
//!
//! ## Node Placement Strategy
//!
//! Nodes are allocated using first-fit within pages. When a page has
//! insufficient space, a new page is allocated. Deleted nodes leave
//! free space that can be reused.
//!
//! ## Overflow Handling
//!
//! For large vectors (>4KB), data is split across multiple pages using
//! overflow pages linked through the right_child field.

use eyre::{ensure, Result, WrapErr};
use zerocopy::{
    byteorder::{LittleEndian, U16, U32, U64},
    FromBytes, Immutable, IntoBytes, KnownLayout, Unaligned,
};

use crate::storage::{
    MmapStorage, PageHeader, PageType, FILE_HEADER_SIZE, HNSW_MAGIC, PAGE_SIZE,
};

use super::{DistanceFunction, NodeId, QuantizationType};

pub const HNSW_PAGE_HEADER_SIZE: usize = 64;
pub const HNSW_SLOT_SIZE: usize = 4;
pub const HNSW_MIN_FREE_SPACE: usize = 64;

#[repr(C)]
#[derive(Debug, Clone, Copy, FromBytes, IntoBytes, Immutable, KnownLayout, Unaligned)]
pub struct HnswFileHeader {
    magic: [u8; 16],
    index_id: U64<LittleEndian>,
    table_id: U64<LittleEndian>,
    dimensions: U16<LittleEndian>,
    m: U16<LittleEndian>,
    m0: U16<LittleEndian>,
    ef_construction: U16<LittleEndian>,
    ef_search: U16<LittleEndian>,
    distance_fn: u8,
    quantization: u8,
    entry_point_page: U32<LittleEndian>,
    entry_point_slot: U16<LittleEndian>,
    max_level: u8,
    reserved1: u8,
    node_count: U64<LittleEndian>,
    vector_count: U64<LittleEndian>,
    first_free_page: U32<LittleEndian>,
    reserved2: [u8; 56],
}

impl HnswFileHeader {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        index_id: u64,
        table_id: u64,
        dimensions: u16,
        m: u16,
        ef_construction: u16,
        ef_search: u16,
        distance_fn: DistanceFunction,
        quantization: QuantizationType,
    ) -> Self {
        let mut magic = [0u8; 16];
        magic.copy_from_slice(HNSW_MAGIC);

        Self {
            magic,
            index_id: U64::new(index_id),
            table_id: U64::new(table_id),
            dimensions: U16::new(dimensions),
            m: U16::new(m),
            m0: U16::new(m * 2),
            ef_construction: U16::new(ef_construction),
            ef_search: U16::new(ef_search),
            distance_fn: distance_fn as u8,
            quantization: quantization as u8,
            entry_point_page: U32::new(u32::MAX),
            entry_point_slot: U16::new(u16::MAX),
            max_level: 0,
            reserved1: 0,
            node_count: U64::new(0),
            vector_count: U64::new(0),
            first_free_page: U32::new(0),
            reserved2: [0; 56],
        }
    }

    pub fn from_bytes(data: &[u8]) -> Result<&Self> {
        ensure!(
            data.len() >= FILE_HEADER_SIZE,
            "buffer too small for HnswFileHeader: {} < {}",
            data.len(),
            FILE_HEADER_SIZE
        );

        ensure!(
            &data[..16] == HNSW_MAGIC,
            "invalid HNSW file: magic bytes mismatch"
        );

        Self::ref_from_bytes(&data[..size_of::<Self>()])
            .map_err(|e| eyre::eyre!("failed to read HnswFileHeader: {:?}", e))
    }

    pub fn from_bytes_mut(data: &mut [u8]) -> Result<&mut Self> {
        ensure!(
            data.len() >= FILE_HEADER_SIZE,
            "buffer too small for HnswFileHeader: {} < {}",
            data.len(),
            FILE_HEADER_SIZE
        );

        Self::mut_from_bytes(&mut data[..size_of::<Self>()])
            .map_err(|e| eyre::eyre!("failed to read HnswFileHeader: {:?}", e))
    }

    pub fn write_to(&self, data: &mut [u8]) -> Result<()> {
        ensure!(
            data.len() >= FILE_HEADER_SIZE,
            "buffer too small for HnswFileHeader: {} < {}",
            data.len(),
            FILE_HEADER_SIZE
        );

        data[..size_of::<Self>()].copy_from_slice(self.as_bytes());
        Ok(())
    }

    pub fn index_id(&self) -> u64 {
        self.index_id.get()
    }

    pub fn table_id(&self) -> u64 {
        self.table_id.get()
    }

    pub fn dimensions(&self) -> u16 {
        self.dimensions.get()
    }

    pub fn m(&self) -> u16 {
        self.m.get()
    }

    pub fn m0(&self) -> u16 {
        self.m0.get()
    }

    pub fn ef_construction(&self) -> u16 {
        self.ef_construction.get()
    }

    pub fn ef_search(&self) -> u16 {
        self.ef_search.get()
    }

    pub fn distance_fn(&self) -> DistanceFunction {
        match self.distance_fn {
            1 => DistanceFunction::Cosine,
            2 => DistanceFunction::InnerProduct,
            _ => DistanceFunction::L2,
        }
    }

    pub fn quantization(&self) -> QuantizationType {
        match self.quantization {
            1 => QuantizationType::SQ8,
            2 => QuantizationType::PQ,
            _ => QuantizationType::None,
        }
    }

    pub fn entry_point(&self) -> Option<NodeId> {
        if self.entry_point_page.get() == u32::MAX {
            None
        } else {
            Some(NodeId::new(
                self.entry_point_page.get(),
                self.entry_point_slot.get(),
            ))
        }
    }

    pub fn set_entry_point(&mut self, node_id: Option<NodeId>) {
        match node_id {
            Some(id) => {
                self.entry_point_page.set(id.page_no());
                self.entry_point_slot.set(id.slot_index());
            }
            None => {
                self.entry_point_page.set(u32::MAX);
                self.entry_point_slot.set(u16::MAX);
            }
        }
    }

    pub fn max_level(&self) -> u8 {
        self.max_level
    }

    pub fn set_max_level(&mut self, level: u8) {
        self.max_level = level;
    }

    pub fn node_count(&self) -> u64 {
        self.node_count.get()
    }

    pub fn set_node_count(&mut self, count: u64) {
        self.node_count.set(count);
    }

    pub fn increment_node_count(&mut self) {
        self.node_count.set(self.node_count.get() + 1);
    }

    pub fn vector_count(&self) -> u64 {
        self.vector_count.get()
    }

    pub fn set_vector_count(&mut self, count: u64) {
        self.vector_count.set(count);
    }

    pub fn first_free_page(&self) -> u32 {
        self.first_free_page.get()
    }

    pub fn set_first_free_page(&mut self, page_no: u32) {
        self.first_free_page.set(page_no);
    }
}

#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SlotStatus {
    Free = 0,
    Active = 1,
    Deleted = 2,
}

impl SlotStatus {
    pub fn from_byte(b: u8) -> Self {
        match b {
            1 => SlotStatus::Active,
            2 => SlotStatus::Deleted,
            _ => SlotStatus::Free,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct SlotEntry {
    pub offset: u16,
    pub status: SlotStatus,
    pub size: u16,
}

impl SlotEntry {
    pub fn new(offset: u16, size: u16, status: SlotStatus) -> Self {
        Self {
            offset,
            status,
            size,
        }
    }

    pub fn encode(&self) -> [u8; 4] {
        let mut bytes = [0u8; 4];
        let offset_and_status = (self.offset & 0x1FFF) | ((self.status as u16) << 13);
        bytes[0..2].copy_from_slice(&offset_and_status.to_le_bytes());
        bytes[2..4].copy_from_slice(&self.size.to_le_bytes());
        bytes
    }

    pub fn decode(bytes: &[u8]) -> Self {
        let offset_and_status = u16::from_le_bytes([bytes[0], bytes[1]]);
        let offset = offset_and_status & 0x1FFF;
        let status = SlotStatus::from_byte(((offset_and_status >> 13) & 0x3) as u8);
        let size = u16::from_le_bytes([bytes[2], bytes[3]]);
        Self {
            offset,
            status,
            size,
        }
    }

    pub fn is_free(&self) -> bool {
        self.status == SlotStatus::Free
    }

    pub fn is_active(&self) -> bool {
        self.status == SlotStatus::Active
    }

    pub fn is_deleted(&self) -> bool {
        self.status == SlotStatus::Deleted
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy, FromBytes, IntoBytes, Immutable, KnownLayout, Unaligned)]
pub struct HnswPageHeader {
    slot_count: U16<LittleEndian>,
    free_start: U16<LittleEndian>,
    free_end: U16<LittleEndian>,
    active_nodes: U16<LittleEndian>,
    deleted_nodes: U16<LittleEndian>,
    total_free_space: U16<LittleEndian>,
    next_page: U32<LittleEndian>,
    reserved: [u8; 36],
}

impl HnswPageHeader {
    pub fn new() -> Self {
        Self {
            slot_count: U16::new(0),
            free_start: U16::new(HNSW_PAGE_HEADER_SIZE as u16),
            free_end: U16::new(PAGE_SIZE as u16),
            active_nodes: U16::new(0),
            deleted_nodes: U16::new(0),
            total_free_space: U16::new((PAGE_SIZE - HNSW_PAGE_HEADER_SIZE) as u16),
            next_page: U32::new(0),
            reserved: [0; 36],
        }
    }

    pub fn from_bytes(data: &[u8]) -> Result<&Self> {
        ensure!(
            data.len() >= size_of::<Self>(),
            "buffer too small for HnswPageHeader"
        );

        Self::ref_from_bytes(&data[..size_of::<Self>()])
            .map_err(|e| eyre::eyre!("failed to read HnswPageHeader: {:?}", e))
    }

    pub fn from_bytes_mut(data: &mut [u8]) -> Result<&mut Self> {
        ensure!(
            data.len() >= size_of::<Self>(),
            "buffer too small for HnswPageHeader"
        );

        Self::mut_from_bytes(&mut data[..size_of::<Self>()])
            .map_err(|e| eyre::eyre!("failed to read HnswPageHeader: {:?}", e))
    }

    pub fn slot_count(&self) -> u16 {
        self.slot_count.get()
    }

    pub fn set_slot_count(&mut self, count: u16) {
        self.slot_count.set(count);
    }

    pub fn free_start(&self) -> u16 {
        self.free_start.get()
    }

    pub fn set_free_start(&mut self, offset: u16) {
        self.free_start.set(offset);
    }

    pub fn free_end(&self) -> u16 {
        self.free_end.get()
    }

    pub fn set_free_end(&mut self, offset: u16) {
        self.free_end.set(offset);
    }

    pub fn free_space(&self) -> u16 {
        self.free_end.get().saturating_sub(self.free_start.get())
    }

    pub fn active_nodes(&self) -> u16 {
        self.active_nodes.get()
    }

    pub fn set_active_nodes(&mut self, count: u16) {
        self.active_nodes.set(count);
    }

    pub fn deleted_nodes(&self) -> u16 {
        self.deleted_nodes.get()
    }

    pub fn set_deleted_nodes(&mut self, count: u16) {
        self.deleted_nodes.set(count);
    }

    pub fn total_free_space(&self) -> u16 {
        self.total_free_space.get()
    }

    pub fn set_total_free_space(&mut self, space: u16) {
        self.total_free_space.set(space);
    }

    pub fn next_page(&self) -> u32 {
        self.next_page.get()
    }

    pub fn set_next_page(&mut self, page_no: u32) {
        self.next_page.set(page_no);
    }
}

impl Default for HnswPageHeader {
    fn default() -> Self {
        Self::new()
    }
}

pub struct HnswPageRef<'a> {
    data: &'a [u8],
}

impl<'a> HnswPageRef<'a> {
    pub fn from_bytes(data: &'a [u8]) -> Result<Self> {
        ensure!(data.len() == PAGE_SIZE, "invalid page size");

        let page_header = PageHeader::from_bytes(data)?;
        ensure!(
            page_header.page_type() == PageType::HnswNode,
            "not an HNSW node page"
        );

        Ok(Self { data })
    }

    fn hnsw_header(&self) -> &HnswPageHeader {
        let offset = size_of::<PageHeader>();
        HnswPageHeader::ref_from_bytes(&self.data[offset..offset + size_of::<HnswPageHeader>()])
            .unwrap()
    }

    pub fn slot_count(&self) -> u16 {
        self.hnsw_header().slot_count()
    }

    pub fn free_space(&self) -> u16 {
        self.hnsw_header().free_space()
    }

    pub fn can_fit(&self, data_size: usize) -> bool {
        let needed = data_size + HNSW_SLOT_SIZE;
        (self.free_space() as usize) >= needed + HNSW_MIN_FREE_SPACE
    }

    fn slot_offset(&self, slot_index: u16) -> usize {
        HNSW_PAGE_HEADER_SIZE + (slot_index as usize) * HNSW_SLOT_SIZE
    }

    pub fn get_slot(&self, slot_index: u16) -> Option<SlotEntry> {
        if slot_index >= self.slot_count() {
            return None;
        }

        let offset = self.slot_offset(slot_index);
        Some(SlotEntry::decode(&self.data[offset..offset + HNSW_SLOT_SIZE]))
    }

    pub fn read_node_data(&self, slot_index: u16) -> Result<&[u8]> {
        let slot = self
            .get_slot(slot_index)
            .ok_or_else(|| eyre::eyre!("invalid slot index"))?;

        ensure!(slot.is_active(), "slot is not active");

        let offset = slot.offset as usize;
        Ok(&self.data[offset..offset + slot.size as usize])
    }
}

pub struct HnswPage<'a> {
    data: &'a mut [u8],
}

impl<'a> HnswPage<'a> {
    pub fn init(data: &'a mut [u8]) -> Result<Self> {
        ensure!(data.len() == PAGE_SIZE, "invalid page size");

        let page_header = PageHeader::new(PageType::HnswNode);
        page_header.write_to(data)?;

        let hnsw_header = HnswPageHeader::new();
        let header_offset = size_of::<PageHeader>();
        data[header_offset..header_offset + size_of::<HnswPageHeader>()]
            .copy_from_slice(hnsw_header.as_bytes());

        Ok(Self { data })
    }

    pub fn from_bytes(data: &'a mut [u8]) -> Result<Self> {
        ensure!(data.len() == PAGE_SIZE, "invalid page size");

        let page_header = PageHeader::from_bytes(data)?;
        ensure!(
            page_header.page_type() == PageType::HnswNode,
            "not an HNSW node page"
        );

        Ok(Self { data })
    }

    pub fn from_bytes_readonly(data: &'a [u8]) -> Result<HnswPageRef<'a>> {
        HnswPageRef::from_bytes(data)
    }

    fn hnsw_header(&self) -> &HnswPageHeader {
        let offset = size_of::<PageHeader>();
        HnswPageHeader::ref_from_bytes(&self.data[offset..offset + size_of::<HnswPageHeader>()])
            .unwrap()
    }

    fn hnsw_header_mut(&mut self) -> &mut HnswPageHeader {
        let offset = size_of::<PageHeader>();
        HnswPageHeader::mut_from_bytes(
            &mut self.data[offset..offset + size_of::<HnswPageHeader>()],
        )
        .unwrap()
    }

    pub fn slot_count(&self) -> u16 {
        self.hnsw_header().slot_count()
    }

    pub fn free_space(&self) -> u16 {
        self.hnsw_header().free_space()
    }

    pub fn active_nodes(&self) -> u16 {
        self.hnsw_header().active_nodes()
    }

    fn slot_offset(&self, slot_index: u16) -> usize {
        HNSW_PAGE_HEADER_SIZE + (slot_index as usize) * HNSW_SLOT_SIZE
    }

    pub fn get_slot(&self, slot_index: u16) -> Option<SlotEntry> {
        if slot_index >= self.slot_count() {
            return None;
        }

        let offset = self.slot_offset(slot_index);
        Some(SlotEntry::decode(&self.data[offset..offset + HNSW_SLOT_SIZE]))
    }

    pub fn can_fit(&self, data_size: usize) -> bool {
        let needed = data_size + HNSW_SLOT_SIZE;
        (self.free_space() as usize) >= needed + HNSW_MIN_FREE_SPACE
    }

    pub fn allocate_slot(&mut self, data_size: u16) -> Result<u16> {
        ensure!(self.can_fit(data_size as usize), "insufficient space in page");

        let header = self.hnsw_header_mut();
        let slot_index = header.slot_count();
        let new_free_start = header.free_start() + HNSW_SLOT_SIZE as u16;
        let new_free_end = header.free_end() - data_size;

        header.set_slot_count(slot_index + 1);
        header.set_free_start(new_free_start);
        header.set_free_end(new_free_end);
        header.set_active_nodes(header.active_nodes() + 1);
        header.set_total_free_space(header.total_free_space() - data_size - HNSW_SLOT_SIZE as u16);

        let slot = SlotEntry::new(new_free_end, data_size, SlotStatus::Active);
        let slot_offset = self.slot_offset(slot_index);
        self.data[slot_offset..slot_offset + HNSW_SLOT_SIZE].copy_from_slice(&slot.encode());

        Ok(slot_index)
    }

    pub fn write_node_data(&mut self, slot_index: u16, data: &[u8]) -> Result<()> {
        let slot = self
            .get_slot(slot_index)
            .ok_or_else(|| eyre::eyre!("invalid slot index"))?;

        ensure!(
            data.len() <= slot.size as usize,
            "data size {} exceeds slot size {}",
            data.len(),
            slot.size
        );

        let offset = slot.offset as usize;
        self.data[offset..offset + data.len()].copy_from_slice(data);

        Ok(())
    }

    pub fn read_node_data(&self, slot_index: u16) -> Result<&[u8]> {
        let slot = self
            .get_slot(slot_index)
            .ok_or_else(|| eyre::eyre!("invalid slot index"))?;

        ensure!(slot.is_active(), "slot is not active");

        let offset = slot.offset as usize;
        Ok(&self.data[offset..offset + slot.size as usize])
    }

    pub fn mark_deleted(&mut self, slot_index: u16) -> Result<()> {
        let slot = self
            .get_slot(slot_index)
            .ok_or_else(|| eyre::eyre!("invalid slot index"))?;

        ensure!(slot.is_active(), "slot is not active");

        let new_slot = SlotEntry::new(slot.offset, slot.size, SlotStatus::Deleted);
        let slot_offset = self.slot_offset(slot_index);
        self.data[slot_offset..slot_offset + HNSW_SLOT_SIZE].copy_from_slice(&new_slot.encode());

        let header = self.hnsw_header_mut();
        header.set_active_nodes(header.active_nodes() - 1);
        header.set_deleted_nodes(header.deleted_nodes() + 1);

        Ok(())
    }
}

pub struct HnswStorage {
    storage: MmapStorage,
}

impl HnswStorage {
    #[allow(clippy::too_many_arguments)]
    pub fn create(
        path: &std::path::Path,
        index_id: u64,
        table_id: u64,
        dimensions: u16,
        m: u16,
        ef_construction: u16,
        ef_search: u16,
        distance_fn: DistanceFunction,
        quantization: QuantizationType,
    ) -> Result<Self> {
        let mut storage = MmapStorage::create(path, 1)?;

        let header = HnswFileHeader::new(
            index_id,
            table_id,
            dimensions,
            m,
            ef_construction,
            ef_search,
            distance_fn,
            quantization,
        );

        let page = storage.page_mut(0)?;
        header.write_to(page)?;
        storage.sync()?;

        Ok(Self { storage })
    }

    pub fn open(path: &std::path::Path) -> Result<Self> {
        let storage = MmapStorage::open(path)?;

        let page = storage.page(0)?;
        let _ = HnswFileHeader::from_bytes(page)?;

        Ok(Self { storage })
    }

    pub fn header(&self) -> Result<&HnswFileHeader> {
        let page = self.storage.page(0)?;
        HnswFileHeader::from_bytes(page)
    }

    pub fn header_mut(&mut self) -> Result<&mut HnswFileHeader> {
        let page = self.storage.page_mut(0)?;
        HnswFileHeader::from_bytes_mut(page)
    }

    pub fn allocate_page(&mut self) -> Result<u32> {
        let current_page_count = self.storage.page_count();
        self.storage
            .grow(current_page_count + 1)
            .wrap_err("failed to grow HNSW storage")?;

        let page_no = current_page_count;
        let page_data = self.storage.page_mut(page_no)?;
        HnswPage::init(page_data)?;

        Ok(page_no)
    }

    pub fn get_page(&self, page_no: u32) -> Result<&[u8]> {
        self.storage.page(page_no)
    }

    pub fn get_page_mut(&mut self, page_no: u32) -> Result<&mut [u8]> {
        self.storage.page_mut(page_no)
    }

    pub fn sync(&self) -> Result<()> {
        self.storage.sync()
    }

    pub fn page_count(&self) -> u32 {
        self.storage.page_count()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn hnsw_file_header_size() {
        assert_eq!(size_of::<HnswFileHeader>(), FILE_HEADER_SIZE);
    }

    #[test]
    fn hnsw_file_header_new() {
        let header = HnswFileHeader::new(
            1,
            2,
            128,
            16,
            100,
            32,
            DistanceFunction::L2,
            QuantizationType::None,
        );

        assert_eq!(header.index_id(), 1);
        assert_eq!(header.table_id(), 2);
        assert_eq!(header.dimensions(), 128);
        assert_eq!(header.m(), 16);
        assert_eq!(header.m0(), 32);
        assert_eq!(header.ef_construction(), 100);
        assert_eq!(header.ef_search(), 32);
        assert_eq!(header.distance_fn(), DistanceFunction::L2);
        assert_eq!(header.quantization(), QuantizationType::None);
        assert!(header.entry_point().is_none());
        assert_eq!(header.max_level(), 0);
        assert_eq!(header.node_count(), 0);
    }

    #[test]
    fn hnsw_file_header_entry_point() {
        let mut header = HnswFileHeader::new(
            1,
            2,
            128,
            16,
            100,
            32,
            DistanceFunction::L2,
            QuantizationType::None,
        );

        assert!(header.entry_point().is_none());

        let node_id = NodeId::new(5, 10);
        header.set_entry_point(Some(node_id));

        let entry = header.entry_point().unwrap();
        assert_eq!(entry.page_no(), 5);
        assert_eq!(entry.slot_index(), 10);

        header.set_entry_point(None);
        assert!(header.entry_point().is_none());
    }

    #[test]
    fn hnsw_file_header_serialization() {
        let header = HnswFileHeader::new(
            42,
            99,
            256,
            32,
            200,
            64,
            DistanceFunction::Cosine,
            QuantizationType::SQ8,
        );

        let mut buf = [0u8; FILE_HEADER_SIZE];
        header.write_to(&mut buf).unwrap();

        let read_header = HnswFileHeader::from_bytes(&buf).unwrap();

        assert_eq!(read_header.index_id(), 42);
        assert_eq!(read_header.table_id(), 99);
        assert_eq!(read_header.dimensions(), 256);
        assert_eq!(read_header.m(), 32);
        assert_eq!(read_header.distance_fn(), DistanceFunction::Cosine);
        assert_eq!(read_header.quantization(), QuantizationType::SQ8);
    }

    #[test]
    fn slot_entry_encode_decode() {
        let slot = SlotEntry::new(1000, 256, SlotStatus::Active);
        let encoded = slot.encode();
        let decoded = SlotEntry::decode(&encoded);

        assert_eq!(decoded.offset, 1000);
        assert_eq!(decoded.size, 256);
        assert_eq!(decoded.status, SlotStatus::Active);
    }

    #[test]
    fn slot_entry_status_bits() {
        let free = SlotEntry::new(100, 50, SlotStatus::Free);
        let active = SlotEntry::new(100, 50, SlotStatus::Active);
        let deleted = SlotEntry::new(100, 50, SlotStatus::Deleted);

        assert!(free.is_free());
        assert!(!free.is_active());

        assert!(active.is_active());
        assert!(!active.is_deleted());

        assert!(deleted.is_deleted());
        assert!(!deleted.is_active());
    }

    #[test]
    fn hnsw_page_init() {
        let mut data = vec![0u8; PAGE_SIZE];
        let page = HnswPage::init(&mut data).unwrap();

        assert_eq!(page.slot_count(), 0);
        assert_eq!(page.active_nodes(), 0);
        assert!(page.free_space() > 0);
    }

    #[test]
    fn hnsw_page_allocate_and_write() {
        let mut data = vec![0u8; PAGE_SIZE];
        let mut page = HnswPage::init(&mut data).unwrap();

        let node_data = vec![1u8, 2, 3, 4, 5, 6, 7, 8];
        let slot_index = page.allocate_slot(node_data.len() as u16).unwrap();

        assert_eq!(slot_index, 0);
        assert_eq!(page.slot_count(), 1);
        assert_eq!(page.active_nodes(), 1);

        page.write_node_data(slot_index, &node_data).unwrap();

        let read_data = page.read_node_data(slot_index).unwrap();
        assert_eq!(read_data, &node_data[..]);
    }

    #[test]
    fn hnsw_page_multiple_nodes() {
        let mut data = vec![0u8; PAGE_SIZE];
        let mut page = HnswPage::init(&mut data).unwrap();

        for i in 0..10 {
            let node_data = vec![i as u8; 100];
            let slot_index = page.allocate_slot(100).unwrap();
            page.write_node_data(slot_index, &node_data).unwrap();
        }

        assert_eq!(page.slot_count(), 10);
        assert_eq!(page.active_nodes(), 10);

        for i in 0..10 {
            let read_data = page.read_node_data(i).unwrap();
            assert_eq!(read_data[0], i as u8);
        }
    }

    #[test]
    fn hnsw_page_mark_deleted() {
        let mut data = vec![0u8; PAGE_SIZE];
        let mut page = HnswPage::init(&mut data).unwrap();

        let slot_index = page.allocate_slot(50).unwrap();
        page.write_node_data(slot_index, &[0u8; 50]).unwrap();

        assert_eq!(page.active_nodes(), 1);

        page.mark_deleted(slot_index).unwrap();

        assert_eq!(page.active_nodes(), 0);

        let slot = page.get_slot(slot_index).unwrap();
        assert!(slot.is_deleted());
    }

    #[test]
    fn hnsw_page_can_fit() {
        let mut data = vec![0u8; PAGE_SIZE];
        let page = HnswPage::init(&mut data).unwrap();

        assert!(page.can_fit(1000));
        assert!(page.can_fit(10000));
        assert!(!page.can_fit(PAGE_SIZE));
    }

    #[test]
    fn hnsw_storage_create_and_open() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.hnsw");

        {
            let storage = HnswStorage::create(
                &path,
                1,
                2,
                128,
                16,
                100,
                32,
                DistanceFunction::L2,
                QuantizationType::None,
            )
            .unwrap();

            let header = storage.header().unwrap();
            assert_eq!(header.dimensions(), 128);
        }

        {
            let storage = HnswStorage::open(&path).unwrap();
            let header = storage.header().unwrap();
            assert_eq!(header.dimensions(), 128);
            assert_eq!(header.m(), 16);
        }
    }

    #[test]
    fn hnsw_storage_allocate_page() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.hnsw");

        let mut storage = HnswStorage::create(
            &path,
            1,
            2,
            128,
            16,
            100,
            32,
            DistanceFunction::L2,
            QuantizationType::None,
        )
        .unwrap();

        assert_eq!(storage.page_count(), 1);

        let page_no = storage.allocate_page().unwrap();
        assert_eq!(page_no, 1);
        assert_eq!(storage.page_count(), 2);

        let page_data = storage.get_page_mut(page_no).unwrap();
        let page = HnswPage::from_bytes(page_data).unwrap();
        assert_eq!(page.slot_count(), 0);
    }

    #[test]
    fn hnsw_storage_modify_header() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.hnsw");

        {
            let mut storage = HnswStorage::create(
                &path,
                1,
                2,
                128,
                16,
                100,
                32,
                DistanceFunction::L2,
                QuantizationType::None,
            )
            .unwrap();

            let header = storage.header_mut().unwrap();
            header.set_node_count(100);
            header.set_max_level(3);
            header.set_entry_point(Some(NodeId::new(1, 5)));

            storage.sync().unwrap();
        }

        {
            let storage = HnswStorage::open(&path).unwrap();
            let header = storage.header().unwrap();

            assert_eq!(header.node_count(), 100);
            assert_eq!(header.max_level(), 3);

            let entry = header.entry_point().unwrap();
            assert_eq!(entry.page_no(), 1);
            assert_eq!(entry.slot_index(), 5);
        }
    }
}
