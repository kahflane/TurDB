//! # HNSW Vector Index Implementation
//!
//! This module implements the Hierarchical Navigable Small World (HNSW) graph-based
//! index for approximate nearest neighbor search. HNSW provides logarithmic search
//! complexity O(log N) with high recall, making it ideal for vector similarity search.
//!
//! ## Architecture Overview
//!
//! HNSW constructs a multi-layer proximity graph where each layer contains a subset
//! of nodes from the layer below. The top layers contain few nodes (for fast navigation)
//! while the bottom layer (level 0) contains all nodes (for accurate search).
//!
//! ```text
//! Level 3:     [A]-------------[B]           (few nodes, long edges)
//!               |               |
//! Level 2:     [A]----[C]------[B]----[D]    (more nodes)
//!               |      |        |      |
//! Level 1:     [A]-[E]-[C]-[F]-[B]-[G]-[D]   (even more nodes)
//!               |   |   |   |   |   |   |
//! Level 0:     [A]-[E]-[C]-[F]-[B]-[G]-[D]-[H]-[I]  (all nodes)
//! ```
//!
//! ## Search Algorithm
//!
//! 1. **Entry Phase**: Start from entry point at top layer
//! 2. **Greedy Descent**: At each layer, greedily move to closest neighbor
//! 3. **Beam Search**: At layer 0, perform beam search with ef_search candidates
//! 4. **Return**: Return k nearest neighbors from candidates
//!
//! ## Parameters
//!
//! - **M**: Maximum neighbors per node (except level 0). Default: 16
//! - **M0**: Maximum neighbors at level 0. Default: 2*M = 32
//! - **ef_construction**: Search width during insertion. Default: 100
//! - **ef_search**: Search width during queries. Default: 32
//!
//! ## Page Layout (16KB)
//!
//! HNSW uses slotted pages to store multiple nodes efficiently:
//!
//! ```text
//! +------------------+
//! | Header (64 bytes)|  PageType, count, free_space_offset
//! +------------------+
//! | Slot Array       |  u16 offsets to nodes (grows forward)
//! +------------------+
//! | Free Space       |
//! +------------------+
//! | Node Data        |  Actual neighbor lists (grows backward)
//! +------------------+
//! ```
//!
//! ## Node Format
//!
//! Each HNSW node contains:
//! - row_id (u64): Reference to row in .tbd file
//! - max_level (u8): Highest level this node exists in
//! - SQ8 vector (optional): Quantized vector for fast distance
//! - L0 neighbors: Fixed size array [NodeId; M0]
//! - Higher level neighbors: Variable length per level
//!
//! ## MVCC Integration
//!
//! The HNSW graph acts as a superset of all visible data. During search:
//! - Graph contains nodes from all transactions
//! - Visibility check happens when collecting results
//! - Invisible nodes still used for graph traversal (stepping stones)
//! - Vacuum removes nodes from aborted transactions
//!
//! ## Distance Functions
//!
//! Supported distance metrics with SIMD acceleration:
//! - L2 (Euclidean): sqrt(sum((a-b)^2))
//! - Cosine: 1 - dot(a,b) (vectors must be normalized)
//! - Inner Product: -dot(a,b) (for MIPS)
//!
//! ## Quantization
//!
//! SQ8 (Scalar Quantization) compresses f32 vectors to u8:
//! - 4x memory reduction
//! - Per-vector min/max scaling
//! - Recall >0.98 for typical embeddings
//! - AVX2 acceleration for distance computation
//!
//! ## File Format (.hnsw)
//!
//! Page 0 contains the index header:
//! ```text
//! Offset  Size  Description
//! 0       8     Magic: "TurDBVec"
//! 8       4     Version: 1
//! 12      4     Page size: 16384
//! 16      2     Dimensions
//! 18      2     M parameter
//! 20      2     M0 parameter
//! 22      2     ef_construction
//! 24      1     Distance function (0=L2, 1=Cosine, 2=IP)
//! 25      3     Reserved
//! 28      4     Entry point node ID
//! 32      1     Max level
//! 33      7     Reserved
//! 40      8     Node count
//! 48      8     Vector count
//! 56      72    Reserved
//! ```
//!
//! ## Zero-Copy Search API
//!
//! Searches use caller-provided context to avoid allocation:
//! ```text
//! pub fn search(
//!     &self,
//!     query: &[f32],
//!     k: usize,
//!     ctx: &mut HnswSearchContext,
//! ) -> Result<usize>
//! ```

pub mod distance;
pub mod operations;
pub mod quantization;
pub mod search;
pub mod storage;

pub const MAX_LEVELS: usize = 4;
pub const MAX_L0_NEIGHBORS: usize = 32;
pub const MAX_LEVEL_NEIGHBORS: usize = 16;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[repr(u8)]
pub enum DistanceFunction {
    #[default]
    L2 = 0,
    Cosine = 1,
    InnerProduct = 2,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[repr(u8)]
pub enum QuantizationType {
    #[default]
    None = 0,
    SQ8 = 1,
    PQ = 2,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NodeId {
    page_no: u32,
    slot_index: u16,
}

impl NodeId {
    pub const fn new(page_no: u32, slot_index: u16) -> Self {
        Self {
            page_no,
            slot_index,
        }
    }

    pub const fn none() -> Self {
        Self {
            page_no: u32::MAX,
            slot_index: u16::MAX,
        }
    }

    pub const fn page_no(&self) -> u32 {
        self.page_no
    }

    pub const fn slot_index(&self) -> u16 {
        self.slot_index
    }

    pub const fn is_none(&self) -> bool {
        self.page_no == u32::MAX && self.slot_index == u16::MAX
    }

    pub fn write_to(&self, buf: &mut [u8]) {
        buf[0..4].copy_from_slice(&self.page_no.to_le_bytes());
        buf[4..6].copy_from_slice(&self.slot_index.to_le_bytes());
    }

    pub fn read_from(buf: &[u8]) -> Self {
        let page_no = u32::from_le_bytes([buf[0], buf[1], buf[2], buf[3]]);
        let slot_index = u16::from_le_bytes([buf[4], buf[5]]);
        Self {
            page_no,
            slot_index,
        }
    }
}

impl Default for NodeId {
    fn default() -> Self {
        Self::none()
    }
}

#[derive(Debug, Clone, Copy)]
pub struct SearchResult {
    pub node_id: NodeId,
    pub row_id: u64,
    pub distance: f32,
}

#[derive(Debug, Clone, Copy)]
pub enum VectorRef<'a> {
    F32(&'a [f32]),
    SQ8 {
        min: f32,
        scale: f32,
        data: &'a [u8],
    },
}

impl<'a> VectorRef<'a> {
    pub fn dimension(&self) -> usize {
        match self {
            VectorRef::F32(data) => data.len(),
            VectorRef::SQ8 { data, .. } => data.len(),
        }
    }
}


pub struct HnswNode {
    row_id: u64,
    max_level: u8,
    l0_count: u8,
    l0_neighbors: [NodeId; MAX_L0_NEIGHBORS],
    higher_levels: Vec<Vec<NodeId>>,
}

impl HnswNode {
    pub fn new(row_id: u64, max_level: u8) -> Self {
        let mut higher_levels = Vec::with_capacity(max_level as usize);
        for _ in 0..max_level {
            higher_levels.push(Vec::with_capacity(MAX_LEVEL_NEIGHBORS));
        }

        Self {
            row_id,
            max_level,
            l0_count: 0,
            l0_neighbors: [NodeId::none(); MAX_L0_NEIGHBORS],
            higher_levels,
        }
    }

    pub fn max_serialized_size(max_level: u8) -> usize {
        let mut size = 8 + 1 + 1;
        size += MAX_L0_NEIGHBORS * 6;
        size += (max_level as usize) * (1 + MAX_LEVEL_NEIGHBORS * 6);
        size
    }

    pub fn row_id(&self) -> u64 {
        self.row_id
    }

    pub fn max_level(&self) -> u8 {
        self.max_level
    }

    pub fn level0_neighbor_count(&self) -> u8 {
        self.l0_count
    }

    pub fn level0_neighbors(&self) -> &[NodeId] {
        &self.l0_neighbors[..self.l0_count as usize]
    }

    pub fn add_level0_neighbor(&mut self, neighbor: NodeId) {
        if (self.l0_count as usize) < MAX_L0_NEIGHBORS {
            self.l0_neighbors[self.l0_count as usize] = neighbor;
            self.l0_count += 1;
        }
    }

    pub fn neighbors_at_level(&self, level: u8) -> &[NodeId] {
        if level == 0 {
            self.level0_neighbors()
        } else if (level as usize) <= self.higher_levels.len() {
            &self.higher_levels[(level - 1) as usize]
        } else {
            &[]
        }
    }

    pub fn add_neighbor_at_level(&mut self, level: u8, neighbor: NodeId) {
        if level == 0 {
            self.add_level0_neighbor(neighbor);
        } else if (level as usize) <= self.higher_levels.len() {
            let level_vec = &mut self.higher_levels[(level - 1) as usize];
            if level_vec.len() < MAX_LEVEL_NEIGHBORS {
                level_vec.push(neighbor);
            }
        }
    }

    pub fn remove_neighbor_at_level(&mut self, level: u8, neighbor: NodeId) {
        if level == 0 {
            if let Some(pos) = self.l0_neighbors[..self.l0_count as usize]
                .iter()
                .position(|&n| n == neighbor)
            {
                for i in pos..((self.l0_count as usize) - 1) {
                    self.l0_neighbors[i] = self.l0_neighbors[i + 1];
                }
                self.l0_count -= 1;
                self.l0_neighbors[self.l0_count as usize] = NodeId::none();
            }
        } else if (level as usize) <= self.higher_levels.len() {
            let level_vec = &mut self.higher_levels[(level - 1) as usize];
            if let Some(pos) = level_vec.iter().position(|&n| n == neighbor) {
                level_vec.remove(pos);
            }
        }
    }

    pub fn serialized_size(&self) -> usize {
        let mut size = 8 + 1 + 1;
        size += (self.l0_count as usize) * 6;
        for level_neighbors in &self.higher_levels {
            size += 1;
            size += level_neighbors.len() * 6;
        }
        size
    }

    pub fn write_to(&self, buf: &mut [u8]) -> usize {
        let mut offset = 0;
        buf[offset..offset + 8].copy_from_slice(&self.row_id.to_le_bytes());
        offset += 8;
        buf[offset] = self.max_level;
        offset += 1;
        buf[offset] = self.l0_count;
        offset += 1;

        for i in 0..self.l0_count as usize {
            self.l0_neighbors[i].write_to(&mut buf[offset..offset + 6]);
            offset += 6;
        }

        for level_neighbors in &self.higher_levels {
            buf[offset] = level_neighbors.len() as u8;
            offset += 1;
            for neighbor in level_neighbors {
                neighbor.write_to(&mut buf[offset..offset + 6]);
                offset += 6;
            }
        }

        offset
    }

    pub fn read_from(buf: &[u8]) -> eyre::Result<Self> {
        use eyre::ensure;

        ensure!(buf.len() >= 10, "buffer too small for HnswNode header");

        let row_id = u64::from_le_bytes([
            buf[0], buf[1], buf[2], buf[3], buf[4], buf[5], buf[6], buf[7],
        ]);
        let max_level = buf[8];
        let l0_count = buf[9];

        ensure!(
            l0_count as usize <= MAX_L0_NEIGHBORS,
            "l0_count {} exceeds maximum {}",
            l0_count,
            MAX_L0_NEIGHBORS
        );

        let mut offset = 10;
        let mut l0_neighbors = [NodeId::none(); MAX_L0_NEIGHBORS];
        for (i, neighbor) in l0_neighbors.iter_mut().enumerate().take(l0_count as usize) {
            ensure!(
                offset + 6 <= buf.len(),
                "buffer too small for L0 neighbor {}",
                i
            );
            *neighbor = NodeId::read_from(&buf[offset..offset + 6]);
            offset += 6;
        }

        let mut higher_levels = Vec::with_capacity(max_level as usize);
        for level in 0..max_level {
            ensure!(
                offset < buf.len(),
                "buffer too small for level {} count",
                level + 1
            );
            let level_count = buf[offset] as usize;
            offset += 1;

            let mut level_neighbors = Vec::with_capacity(level_count);
            for i in 0..level_count {
                ensure!(
                    offset + 6 <= buf.len(),
                    "buffer too small for level {} neighbor {}",
                    level + 1,
                    i
                );
                level_neighbors.push(NodeId::read_from(&buf[offset..offset + 6]));
                offset += 6;
            }
            higher_levels.push(level_neighbors);
        }

        Ok(Self {
            row_id,
            max_level,
            l0_count,
            l0_neighbors,
            higher_levels,
        })
    }
}

#[derive(Clone, Copy)]
#[repr(C)]
pub struct HnswNodeInline {
    row_id: u64,
    max_level: u8,
    l0_count: u8,
    l0_neighbors: [NodeId; MAX_L0_NEIGHBORS],
    higher_counts: [u8; MAX_LEVELS],
    higher_neighbors: [[NodeId; MAX_LEVEL_NEIGHBORS]; MAX_LEVELS],
}

impl HnswNodeInline {
    pub fn new(row_id: u64, max_level: u8) -> Self {
        let capped_level = max_level.min(MAX_LEVELS as u8);
        Self {
            row_id,
            max_level: capped_level,
            l0_count: 0,
            l0_neighbors: [NodeId::none(); MAX_L0_NEIGHBORS],
            higher_counts: [0u8; MAX_LEVELS],
            higher_neighbors: [[NodeId::none(); MAX_LEVEL_NEIGHBORS]; MAX_LEVELS],
        }
    }

    pub const fn serialized_size() -> usize {
        8 + 1 + 1 + (MAX_L0_NEIGHBORS * 6) + MAX_LEVELS + (MAX_LEVELS * MAX_LEVEL_NEIGHBORS * 6)
    }

    pub fn row_id(&self) -> u64 {
        self.row_id
    }

    pub fn max_level(&self) -> u8 {
        self.max_level
    }

    pub fn level0_neighbor_count(&self) -> u8 {
        self.l0_count
    }

    pub fn level0_neighbors(&self) -> &[NodeId] {
        &self.l0_neighbors[..self.l0_count as usize]
    }

    pub fn add_level0_neighbor(&mut self, neighbor: NodeId) {
        if (self.l0_count as usize) < MAX_L0_NEIGHBORS {
            self.l0_neighbors[self.l0_count as usize] = neighbor;
            self.l0_count += 1;
        }
    }

    pub fn neighbors_at_level(&self, level: u8) -> &[NodeId] {
        if level == 0 {
            self.level0_neighbors()
        } else if (level as usize) <= MAX_LEVELS && level <= self.max_level {
            let idx = (level - 1) as usize;
            &self.higher_neighbors[idx][..self.higher_counts[idx] as usize]
        } else {
            &[]
        }
    }

    pub fn add_neighbor_at_level(&mut self, level: u8, neighbor: NodeId) {
        if level == 0 {
            self.add_level0_neighbor(neighbor);
        } else if (level as usize) <= MAX_LEVELS && level <= self.max_level {
            let idx = (level - 1) as usize;
            if (self.higher_counts[idx] as usize) < MAX_LEVEL_NEIGHBORS {
                self.higher_neighbors[idx][self.higher_counts[idx] as usize] = neighbor;
                self.higher_counts[idx] += 1;
            }
        }
    }

    pub fn remove_neighbor_at_level(&mut self, level: u8, neighbor: NodeId) {
        if level == 0 {
            if let Some(pos) = self.l0_neighbors[..self.l0_count as usize]
                .iter()
                .position(|&n| n == neighbor)
            {
                for i in pos..((self.l0_count as usize) - 1) {
                    self.l0_neighbors[i] = self.l0_neighbors[i + 1];
                }
                self.l0_count -= 1;
                self.l0_neighbors[self.l0_count as usize] = NodeId::none();
            }
        } else if (level as usize) <= MAX_LEVELS && level <= self.max_level {
            let idx = (level - 1) as usize;
            let count = self.higher_counts[idx] as usize;
            if let Some(pos) = self.higher_neighbors[idx][..count]
                .iter()
                .position(|&n| n == neighbor)
            {
                for i in pos..(count - 1) {
                    self.higher_neighbors[idx][i] = self.higher_neighbors[idx][i + 1];
                }
                self.higher_counts[idx] -= 1;
                self.higher_neighbors[idx][self.higher_counts[idx] as usize] = NodeId::none();
            }
        }
    }

    pub fn write_to(&self, buf: &mut [u8]) -> usize {
        let mut offset = 0;
        buf[offset..offset + 8].copy_from_slice(&self.row_id.to_le_bytes());
        offset += 8;
        buf[offset] = self.max_level;
        offset += 1;
        buf[offset] = self.l0_count;
        offset += 1;

        for i in 0..self.l0_count as usize {
            self.l0_neighbors[i].write_to(&mut buf[offset..offset + 6]);
            offset += 6;
        }

        for level in 0..self.max_level as usize {
            if level >= MAX_LEVELS {
                break;
            }
            buf[offset] = self.higher_counts[level];
            offset += 1;
            for i in 0..self.higher_counts[level] as usize {
                self.higher_neighbors[level][i].write_to(&mut buf[offset..offset + 6]);
                offset += 6;
            }
        }

        offset
    }

    pub fn read_from(buf: &[u8]) -> eyre::Result<Self> {
        use eyre::ensure;

        ensure!(buf.len() >= 10, "buffer too small for HnswNodeInline header");

        let row_id = u64::from_le_bytes([
            buf[0], buf[1], buf[2], buf[3], buf[4], buf[5], buf[6], buf[7],
        ]);
        let max_level = buf[8].min(MAX_LEVELS as u8);
        let l0_count = buf[9].min(MAX_L0_NEIGHBORS as u8);

        let mut offset = 10;
        let mut l0_neighbors = [NodeId::none(); MAX_L0_NEIGHBORS];
        for neighbor in l0_neighbors.iter_mut().take(l0_count as usize) {
            ensure!(
                offset + 6 <= buf.len(),
                "buffer too small for L0 neighbor"
            );
            *neighbor = NodeId::read_from(&buf[offset..offset + 6]);
            offset += 6;
        }

        let mut higher_counts = [0u8; MAX_LEVELS];
        let mut higher_neighbors = [[NodeId::none(); MAX_LEVEL_NEIGHBORS]; MAX_LEVELS];

        for level in 0..max_level as usize {
            if level >= MAX_LEVELS {
                break;
            }
            ensure!(
                offset < buf.len(),
                "buffer too small for level {} count",
                level + 1
            );
            let count = buf[offset].min(MAX_LEVEL_NEIGHBORS as u8);
            higher_counts[level] = count;
            offset += 1;

            for neighbor in higher_neighbors[level].iter_mut().take(count as usize) {
                ensure!(
                    offset + 6 <= buf.len(),
                    "buffer too small for level {} neighbor",
                    level + 1
                );
                *neighbor = NodeId::read_from(&buf[offset..offset + 6]);
                offset += 6;
            }
        }

        Ok(Self {
            row_id,
            max_level,
            l0_count,
            l0_neighbors,
            higher_counts,
            higher_neighbors,
        })
    }
}

pub struct HnswIndex {
    dimensions: u16,
    m: u16,
    m0: u16,
    ef_construction: u16,
    ef_search: u16,
    distance_fn: DistanceFunction,
    quantization: QuantizationType,
    entry_point: Option<NodeId>,
    max_level: u8,
    node_count: u64,
}

impl HnswIndex {
    pub fn new(dimensions: u16, m: u16, ef_construction: u16, ef_search: u16) -> Self {
        Self {
            dimensions,
            m,
            m0: m * 2,
            ef_construction,
            ef_search,
            distance_fn: DistanceFunction::default(),
            quantization: QuantizationType::default(),
            entry_point: None,
            max_level: 0,
            node_count: 0,
        }
    }

    pub fn with_defaults(dimensions: u16) -> Self {
        Self::new(dimensions, 16, 100, 32)
    }

    pub fn from_header(header: &storage::HnswFileHeader) -> Self {
        Self {
            dimensions: header.dimensions(),
            m: header.m(),
            m0: header.m0(),
            ef_construction: header.ef_construction(),
            ef_search: header.ef_search(),
            distance_fn: header.distance_fn(),
            quantization: header.quantization(),
            entry_point: header.entry_point(),
            max_level: header.max_level(),
            node_count: header.node_count(),
        }
    }

    pub fn sync_to_header(&self, header: &mut storage::HnswFileHeader) {
        header.set_entry_point(self.entry_point);
        header.set_max_level(self.max_level);
        header.set_node_count(self.node_count);
    }

    pub fn dimensions(&self) -> u16 {
        self.dimensions
    }

    pub fn m(&self) -> u16 {
        self.m
    }

    pub fn m0(&self) -> u16 {
        self.m0
    }

    pub fn ef_construction(&self) -> u16 {
        self.ef_construction
    }

    pub fn ef_search(&self) -> u16 {
        self.ef_search
    }

    pub fn distance_fn(&self) -> DistanceFunction {
        self.distance_fn
    }

    pub fn quantization(&self) -> QuantizationType {
        self.quantization
    }

    pub fn entry_point(&self) -> Option<NodeId> {
        self.entry_point
    }

    pub fn set_entry_point(&mut self, entry: NodeId, level: u8) {
        self.entry_point = Some(entry);
        if level > self.max_level {
            self.max_level = level;
        }
    }

    pub fn max_level(&self) -> u8 {
        self.max_level
    }

    pub fn node_count(&self) -> u64 {
        self.node_count
    }

    pub fn increment_node_count(&mut self) {
        self.node_count += 1;
    }

    pub fn decrement_node_count(&mut self) {
        if self.node_count > 0 {
            self.node_count -= 1;
        }
    }
}

pub struct VacuumQueue {
    pending: Vec<NodeId>,
    max_batch_size: usize,
}

impl VacuumQueue {
    pub fn new(max_batch_size: usize) -> Self {
        Self {
            pending: Vec::new(),
            max_batch_size,
        }
    }

    pub fn enqueue(&mut self, node_id: NodeId) {
        self.pending.push(node_id);
    }

    pub fn is_empty(&self) -> bool {
        self.pending.is_empty()
    }

    pub fn len(&self) -> usize {
        self.pending.len()
    }

    pub fn needs_vacuum(&self) -> bool {
        self.pending.len() >= self.max_batch_size
    }

    pub fn take_batch(&mut self, max_count: usize) -> Vec<NodeId> {
        let count = max_count.min(self.pending.len());
        self.pending.drain(..count).collect()
    }
}

impl Default for VacuumQueue {
    fn default() -> Self {
        Self::new(1000)
    }
}

pub struct PersistentHnswIndex {
    index: HnswIndex,
    storage: storage::HnswStorage,
    current_page: u32,
    vacuum_queue: VacuumQueue,
    row_id_map: hashbrown::HashMap<u64, NodeId>,
}

impl PersistentHnswIndex {
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
    ) -> eyre::Result<Self> {
        let storage = storage::HnswStorage::create(
            path,
            index_id,
            table_id,
            dimensions,
            m,
            ef_construction,
            ef_search,
            distance_fn,
            quantization,
        )?;

        let header = storage.header()?;
        let index = HnswIndex::from_header(header);

        Ok(Self {
            index,
            storage,
            current_page: 0,
            vacuum_queue: VacuumQueue::default(),
            row_id_map: hashbrown::HashMap::new(),
        })
    }

    pub fn open(path: &std::path::Path) -> eyre::Result<Self> {
        let storage = storage::HnswStorage::open(path)?;
        let header = storage.header()?;
        let index = HnswIndex::from_header(header);

        let current_page = if storage.page_count() > 1 {
            storage.page_count() - 1
        } else {
            0
        };

        let mut result = Self {
            index,
            storage,
            current_page,
            vacuum_queue: VacuumQueue::default(),
            row_id_map: hashbrown::HashMap::new(),
        };

        result.rebuild_row_id_map()?;

        Ok(result)
    }

    fn rebuild_row_id_map(&mut self) -> eyre::Result<()> {
        self.row_id_map.clear();

        for page_no in 1..=self.current_page {
            if let Ok(page_data) = self.storage.get_page(page_no) {
                if let Ok(page) = storage::HnswPageRef::from_bytes(page_data) {
                    for slot_idx in 0..page.slot_count() {
                        if let Some(slot) = page.get_slot(slot_idx) {
                            if slot.is_active() {
                                if let Ok(data) = page.read_node_data(slot_idx) {
                                    if let Ok(node) = HnswNode::read_from(data) {
                                        let node_id = NodeId::new(page_no, slot_idx);
                                        self.row_id_map.insert(node.row_id(), node_id);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        Ok(())
    }

    pub fn find_node_by_row_id(&self, row_id: u64) -> Option<NodeId> {
        self.row_id_map.get(&row_id).copied()
    }

    pub fn delete_by_row_id(&mut self, row_id: u64) -> eyre::Result<()> {
        let node_id = self.row_id_map.remove(&row_id);
        if let Some(nid) = node_id {
            self.delete(nid)?;
        }
        Ok(())
    }

    pub fn index(&self) -> &HnswIndex {
        &self.index
    }

    pub fn sync(&mut self) -> eyre::Result<()> {
        let header = self.storage.header_mut()?;
        self.index.sync_to_header(header);
        self.storage.sync()
    }

    pub fn allocate_node(&mut self, node: &HnswNode) -> eyre::Result<NodeId> {
        let max_data_size = HnswNode::max_serialized_size(node.max_level());

        if self.current_page == 0 || !self.page_has_space(self.current_page, max_data_size)? {
            self.current_page = self.storage.allocate_page()?;
        }

        let page_data = self.storage.get_page_mut(self.current_page)?;
        let mut page = storage::HnswPage::from_bytes(page_data)?;

        let slot_index = page.allocate_slot(max_data_size as u16)?;

        let actual_size = node.serialized_size();
        let mut buf = vec![0u8; actual_size];
        node.write_to(&mut buf);
        page.write_node_data(slot_index, &buf)?;

        let node_id = NodeId::new(self.current_page, slot_index);
        self.index.increment_node_count();

        Ok(node_id)
    }

    pub fn read_node(&self, node_id: NodeId) -> eyre::Result<HnswNode> {
        let page_data = self.storage.get_page(node_id.page_no())?;
        let page = storage::HnswPage::from_bytes_readonly(page_data)?;
        let data = page.read_node_data(node_id.slot_index())?;
        HnswNode::read_from(data)
    }

    pub fn update_node(&mut self, node_id: NodeId, node: &HnswNode) -> eyre::Result<()> {
        let page_data = self.storage.get_page_mut(node_id.page_no())?;
        let mut page = storage::HnswPage::from_bytes(page_data)?;

        let slot = page
            .get_slot(node_id.slot_index())
            .ok_or_else(|| eyre::eyre!("invalid slot index"))?;

        let max_data_size = HnswNode::max_serialized_size(node.max_level());
        eyre::ensure!(
            max_data_size <= slot.size as usize,
            "node max size {} exceeds slot size {}",
            max_data_size,
            slot.size
        );

        let data_size = node.serialized_size();
        let mut buf = vec![0u8; data_size];
        node.write_to(&mut buf);
        page.write_node_data(node_id.slot_index(), &buf)?;

        Ok(())
    }

    pub fn mark_deleted(&mut self, node_id: NodeId) -> eyre::Result<()> {
        let page_data = self.storage.get_page_mut(node_id.page_no())?;
        let mut page = storage::HnswPage::from_bytes(page_data)?;
        page.mark_deleted(node_id.slot_index())
    }

    pub fn delete(&mut self, node_id: NodeId) -> eyre::Result<()> {
        self.mark_deleted(node_id)?;
        self.vacuum_queue.enqueue(node_id);
        self.index.decrement_node_count();
        Ok(())
    }

    pub fn vacuum_queue(&self) -> &VacuumQueue {
        &self.vacuum_queue
    }

    pub fn vacuum_batch(&mut self, max_nodes: usize) -> eyre::Result<usize> {
        let batch = self.vacuum_queue.take_batch(max_nodes);
        let count = batch.len();

        for deleted_node_id in batch {
            let deleted_node = match self.read_node(deleted_node_id) {
                Ok(n) => n,
                Err(_) => continue,
            };

            for level in 0..=deleted_node.max_level() {
                for &neighbor_id in deleted_node.neighbors_at_level(level) {
                    if neighbor_id.is_none() {
                        continue;
                    }

                    if let Ok(mut neighbor) = self.read_node(neighbor_id) {
                        neighbor.remove_neighbor_at_level(level, deleted_node_id);
                        let _ = self.update_node(neighbor_id, &neighbor);
                    }
                }
            }

            if self.index.entry_point() == Some(deleted_node_id) {
                self.find_new_entry_point()?;
            }
        }

        Ok(count)
    }

    fn find_new_entry_point(&mut self) -> eyre::Result<()> {
        self.index.set_entry_point(NodeId::none(), 0);
        Ok(())
    }

    pub fn insert(
        &mut self,
        row_id: u64,
        vector: &[f32],
        random_value: f64,
    ) -> eyre::Result<NodeId> {
        self.insert_with_callback(row_id, vector, random_value, |_| None)
    }

    pub fn insert_with_callback(
        &mut self,
        row_id: u64,
        vector: &[f32],
        random_value: f64,
        get_existing_vector: impl Fn(u64) -> Option<Vec<f32>>,
    ) -> eyre::Result<NodeId> {
        eyre::ensure!(
            vector.len() == self.index.dimensions() as usize,
            "vector dimension {} does not match index dimension {}",
            vector.len(),
            self.index.dimensions()
        );

        let ml = operations::calculate_ml(self.index.m());
        let target_level = operations::select_level(random_value, ml);

        let node = HnswNode::new(row_id, target_level);
        let node_id = self.allocate_node(&node)?;
        self.row_id_map.insert(row_id, node_id);

        if self.index.entry_point().is_none() {
            self.index.set_entry_point(node_id, target_level);
            return Ok(node_id);
        }

        let entry_point = self.index.entry_point().unwrap();
        let entry_row_id = self.read_node(entry_point)?.row_id();
        let entry_vector = get_existing_vector(entry_row_id).unwrap_or_default();
        let entry_distance = if entry_vector.is_empty() {
            f32::INFINITY
        } else {
            distance::euclidean_squared(vector, &entry_vector)
        };

        let mut insert_ctx = operations::InsertContext::new(target_level);
        insert_ctx.set_entry(entry_point, entry_distance, self.index.max_level());

        let get_neighbors = |n: NodeId, level: u8| {
            self.read_node(n)
                .map(|node| node.neighbors_at_level(level).to_vec())
                .unwrap_or_default()
        };

        let compute_distance = |n: NodeId| {
            let node_row_id = self.read_node(n).map(|node| node.row_id()).unwrap_or(0);
            get_existing_vector(node_row_id)
                .map(|v| distance::euclidean_squared(vector, &v))
                .unwrap_or(f32::INFINITY)
        };

        operations::insert_descent_phase(&mut insert_ctx, get_neighbors, compute_distance);

        let max_nodes_estimate = (self.index.node_count() as usize).max(1000);
        let mut search_ctx = search::HnswSearchContext::new(
            self.index.ef_construction() as usize,
            max_nodes_estimate,
        );

        operations::insert_connection_phase(
            &mut insert_ctx,
            &mut search_ctx,
            get_neighbors,
            compute_distance,
            self.index.m() as usize,
            self.index.m0() as usize,
        );

        for (level, neighbors) in insert_ctx.neighbors_to_add {
            let mut current_node = self.read_node(node_id)?;
            for &neighbor_id in &neighbors {
                current_node.add_neighbor_at_level(level, neighbor_id);

                let mut neighbor = self.read_node(neighbor_id)?;
                neighbor.add_neighbor_at_level(level, node_id);
                self.update_node(neighbor_id, &neighbor)?;
            }
            self.update_node(node_id, &current_node)?;
        }

        if target_level > self.index.max_level() {
            self.index.set_entry_point(node_id, target_level);
        }

        Ok(node_id)
    }

    fn page_has_space(&self, page_no: u32, data_size: usize) -> eyre::Result<bool> {
        let page_data = self.storage.get_page(page_no)?;
        let page = storage::HnswPage::from_bytes_readonly(page_data)?;
        Ok(page.can_fit(data_size))
    }

    pub fn search(
        &self,
        query: &[f32],
        k: usize,
        ctx: &mut search::HnswSearchContext,
        get_vector: impl Fn(u64) -> Option<Vec<f32>>,
    ) -> eyre::Result<Vec<SearchResult>> {
        eyre::ensure!(
            query.len() == self.index.dimensions() as usize,
            "query dimension {} does not match index dimension {}",
            query.len(),
            self.index.dimensions()
        );

        let entry_point = match self.index.entry_point() {
            Some(ep) => ep,
            None => return Ok(Vec::new()),
        };

        let compute_distance = |node_id: NodeId| -> f32 {
            let node = match self.read_node(node_id) {
                Ok(n) => n,
                Err(_) => return f32::INFINITY,
            };
            let row_id = node.row_id();
            match get_vector(row_id) {
                Some(v) => distance::euclidean_squared(query, &v),
                None => f32::INFINITY,
            }
        };

        let get_neighbors = |node_id: NodeId, level: u8| -> Vec<NodeId> {
            self.read_node(node_id)
                .map(|n| n.neighbors_at_level(level).to_vec())
                .unwrap_or_default()
        };

        let entry_distance = compute_distance(entry_point);

        let mut current = entry_point;
        let mut current_distance = entry_distance;

        for level in (1..=self.index.max_level()).rev() {
            let get_neighbors_at_level = |n: NodeId| get_neighbors(n, level);
            let (new_node, new_distance) = search::greedy_search(
                current,
                current_distance,
                get_neighbors_at_level,
                compute_distance,
                1000,
            );
            current = new_node;
            current_distance = new_distance;
        }

        let entry_candidate = search::Candidate::new(current, current_distance);
        let get_neighbors_at_level_0 = |n: NodeId| get_neighbors(n, 0);

        search::beam_search(
            ctx,
            &[entry_candidate],
            get_neighbors_at_level_0,
            compute_distance,
        );

        ctx.finalize_results(k);

        let results: Vec<SearchResult> = ctx
            .results()
            .iter()
            .map(|c| {
                let node = self.read_node(c.node_id).ok();
                let row_id = node.map(|n| n.row_id()).unwrap_or(0);
                SearchResult {
                    node_id: c.node_id,
                    row_id,
                    distance: c.distance,
                }
            })
            .collect();

        Ok(results)
    }

    pub fn search_filtered<F>(
        &self,
        query: &[f32],
        k: usize,
        ctx: &mut search::HnswSearchContext,
        get_vector: impl Fn(u64) -> Option<Vec<f32>>,
        is_visible: F,
    ) -> eyre::Result<Vec<SearchResult>>
    where
        F: Fn(u64) -> bool,
    {
        eyre::ensure!(
            query.len() == self.index.dimensions() as usize,
            "query dimension {} does not match index dimension {}",
            query.len(),
            self.index.dimensions()
        );

        let entry_point = match self.index.entry_point() {
            Some(ep) => ep,
            None => return Ok(Vec::new()),
        };

        let compute_distance = |node_id: NodeId| -> f32 {
            let node = match self.read_node(node_id) {
                Ok(n) => n,
                Err(_) => return f32::INFINITY,
            };
            let row_id = node.row_id();
            match get_vector(row_id) {
                Some(v) => distance::euclidean_squared(query, &v),
                None => f32::INFINITY,
            }
        };

        let get_neighbors = |node_id: NodeId, level: u8| -> Vec<NodeId> {
            self.read_node(node_id)
                .map(|n| n.neighbors_at_level(level).to_vec())
                .unwrap_or_default()
        };

        let node_is_visible = |node_id: NodeId| -> bool {
            self.read_node(node_id)
                .map(|n| is_visible(n.row_id()))
                .unwrap_or(false)
        };

        let entry_distance = compute_distance(entry_point);

        let mut current = entry_point;
        let mut current_distance = entry_distance;

        for level in (1..=self.index.max_level()).rev() {
            let get_neighbors_at_level = |n: NodeId| get_neighbors(n, level);
            let (new_node, new_distance) = search::greedy_search(
                current,
                current_distance,
                get_neighbors_at_level,
                compute_distance,
                1000,
            );
            current = new_node;
            current_distance = new_distance;
        }

        let entry_candidate = search::Candidate::new(current, current_distance);
        let get_neighbors_at_level_0 = |n: NodeId| get_neighbors(n, 0);

        search::beam_search_filtered(
            ctx,
            &[entry_candidate],
            get_neighbors_at_level_0,
            compute_distance,
            node_is_visible,
        );

        ctx.finalize_results(k);

        let results: Vec<SearchResult> = ctx
            .results()
            .iter()
            .filter_map(|c| {
                let node = self.read_node(c.node_id).ok()?;
                let row_id = node.row_id();
                if is_visible(row_id) {
                    Some(SearchResult {
                        node_id: c.node_id,
                        row_id,
                        distance: c.distance,
                    })
                } else {
                    None
                }
            })
            .collect();

        Ok(results)
    }
}
