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
}
