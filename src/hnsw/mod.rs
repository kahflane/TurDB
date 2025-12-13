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
}

impl Default for NodeId {
    fn default() -> Self {
        Self::none()
    }
}

const MAX_LEVEL0_NEIGHBORS: usize = 32;
const MAX_LEVEL_NEIGHBORS: usize = 16;

pub struct HnswNode {
    row_id: u64,
    max_level: u8,
    l0_count: u8,
    l0_neighbors: [NodeId; MAX_LEVEL0_NEIGHBORS],
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
            l0_neighbors: [NodeId::none(); MAX_LEVEL0_NEIGHBORS],
            higher_levels,
        }
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
        if (self.l0_count as usize) < MAX_LEVEL0_NEIGHBORS {
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
}

pub struct HnswIndex {
    dimensions: u16,
    m: u16,
    m0: u16,
    ef_construction: u16,
    ef_search: u16,
    distance_fn: DistanceFunction,
    quantization: QuantizationType,
    entry_point: Option<u32>,
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

    pub fn entry_point(&self) -> Option<u32> {
        self.entry_point
    }

    pub fn max_level(&self) -> u8 {
        self.max_level
    }

    pub fn node_count(&self) -> u64 {
        self.node_count
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hnsw_index_has_required_parameters() {
        let index = HnswIndex::new(128, 16, 100, 32);

        assert_eq!(index.dimensions(), 128);
        assert_eq!(index.m(), 16);
        assert_eq!(index.m0(), 32);
        assert_eq!(index.ef_construction(), 100);
        assert_eq!(index.ef_search(), 32);
    }

    #[test]
    fn hnsw_index_default_parameters() {
        let index = HnswIndex::with_defaults(384);

        assert_eq!(index.dimensions(), 384);
        assert_eq!(index.m(), 16);
        assert_eq!(index.m0(), 32);
        assert_eq!(index.ef_construction(), 100);
        assert_eq!(index.ef_search(), 32);
    }

    #[test]
    fn hnsw_index_m0_is_double_m() {
        let index = HnswIndex::new(768, 24, 200, 64);

        assert_eq!(index.m(), 24);
        assert_eq!(index.m0(), 48);
    }

    #[test]
    fn hnsw_index_initially_empty() {
        let index = HnswIndex::with_defaults(128);

        assert_eq!(index.node_count(), 0);
        assert_eq!(index.max_level(), 0);
        assert!(index.entry_point().is_none());
    }

    #[test]
    fn quantization_type_variants() {
        let none = QuantizationType::None;
        let sq8 = QuantizationType::SQ8;
        let pq = QuantizationType::PQ;

        assert_eq!(none as u8, 0);
        assert_eq!(sq8 as u8, 1);
        assert_eq!(pq as u8, 2);
    }

    #[test]
    fn quantization_type_default_is_none() {
        let default = QuantizationType::default();
        assert_eq!(default, QuantizationType::None);
    }

    #[test]
    fn hnsw_index_has_quantization_type() {
        let index = HnswIndex::with_defaults(128);
        assert_eq!(index.quantization(), QuantizationType::None);
    }

    #[test]
    fn node_id_stores_page_and_slot() {
        let node_id = NodeId::new(42, 5);

        assert_eq!(node_id.page_no(), 42);
        assert_eq!(node_id.slot_index(), 5);
    }

    #[test]
    fn node_id_none_represents_null() {
        let null_id = NodeId::none();

        assert!(null_id.is_none());
        assert_eq!(null_id.page_no(), u32::MAX);
    }

    #[test]
    fn hnsw_node_has_row_id_and_level() {
        let node = HnswNode::new(12345, 3);

        assert_eq!(node.row_id(), 12345);
        assert_eq!(node.max_level(), 3);
    }

    #[test]
    fn hnsw_node_starts_with_no_neighbors() {
        let node = HnswNode::new(100, 0);

        assert_eq!(node.level0_neighbor_count(), 0);
        assert!(node.level0_neighbors().is_empty());
    }

    #[test]
    fn hnsw_node_can_add_level0_neighbors() {
        let mut node = HnswNode::new(100, 0);
        let neighbor = NodeId::new(1, 0);

        node.add_level0_neighbor(neighbor);

        assert_eq!(node.level0_neighbor_count(), 1);
        assert_eq!(node.level0_neighbors()[0], neighbor);
    }

    #[test]
    fn hnsw_node_stores_higher_level_neighbors() {
        let mut node = HnswNode::new(100, 2);
        let neighbor1 = NodeId::new(1, 0);
        let neighbor2 = NodeId::new(2, 0);

        node.add_neighbor_at_level(1, neighbor1);
        node.add_neighbor_at_level(2, neighbor2);

        assert_eq!(node.neighbors_at_level(1).len(), 1);
        assert_eq!(node.neighbors_at_level(2).len(), 1);
        assert_eq!(node.neighbors_at_level(1)[0], neighbor1);
        assert_eq!(node.neighbors_at_level(2)[0], neighbor2);
    }
}
