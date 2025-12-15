//! # HNSW Graph Operations
//!
//! This module implements insertion and deletion operations for the HNSW index.
//! These operations maintain the graph structure while preserving the navigability
//! properties required for efficient search.
//!
//! ## Level Selection
//!
//! When inserting a new node, its maximum level is selected randomly using
//! an exponential distribution:
//!
//! ```text
//! level = floor(-ln(uniform(0, 1)) * ml)
//! where ml = 1 / ln(M)
//! ```
//!
//! This creates a hierarchy where:
//! - ~63% of nodes exist only at level 0
//! - ~23% of nodes reach level 1
//! - ~8.5% of nodes reach level 2
//! - Higher levels become exponentially rarer
//!
//! ## Insertion Algorithm
//!
//! 1. **Level Selection**: Randomly choose the maximum level for the new node
//!
//! 2. **Descent Phase**: Starting from the entry point at the top level,
//!    greedily descend to the level of the new node
//!
//! 3. **Search & Connect Phase**: For each level from the new node's level
//!    down to level 0:
//!    - Find ef_construction nearest neighbors using beam search
//!    - Select M (or M0 for level 0) best neighbors to connect
//!    - Add bidirectional edges between the new node and selected neighbors
//!    - Prune neighbors that exceed the maximum connection count
//!
//! 4. **Entry Point Update**: If the new node's level exceeds the current
//!    maximum level, update the entry point
//!
//! ## Neighbor Selection Heuristics
//!
//! HNSW uses heuristic neighbor selection to maintain graph diversity:
//!
//! - **Simple selection**: Choose M nearest neighbors
//! - **Heuristic selection**: Prefer neighbors that provide diverse paths
//!
//! The heuristic approach checks if adding a neighbor would create a
//! "shortcut" to already-selected neighbors, preferring edges that reach
//! new regions of the graph.
//!
//! ## Deletion Strategy
//!
//! Soft deletion is used for MVCC compatibility:
//!
//! - Deleted nodes are marked but not removed from the graph
//! - Marked nodes still participate in graph traversal (as stepping stones)
//! - Marked nodes are filtered from search results
//! - Vacuum process can later compact the graph
//!
//! ## Thread Safety
//!
//! Insert and delete operations require exclusive access to affected pages.
//! The HNSW index uses page-level locking to allow concurrent operations
//! on different parts of the graph.
//!
//! ## Page Layout Considerations
//!
//! - Nodes on the same page can be updated together efficiently
//! - Cross-page neighbor links require multiple page accesses
//! - Node placement strategy affects locality and performance

use super::search::{beam_search, greedy_search, Candidate, HnswSearchContext};
use super::NodeId;

pub fn select_level(random_value: f64, ml: f64) -> u8 {
    let level = (-random_value.ln() * ml).floor() as u8;
    level.min(15)
}

pub fn calculate_ml(m: u16) -> f64 {
    1.0 / (m as f64).ln()
}

pub struct InsertContext {
    pub entry_point: NodeId,
    pub entry_distance: f32,
    pub current_level: u8,
    pub target_level: u8,
    pub neighbors_to_add: Vec<(u8, Vec<NodeId>)>,
}

impl InsertContext {
    pub fn new(target_level: u8) -> Self {
        Self {
            entry_point: NodeId::none(),
            entry_distance: f32::INFINITY,
            current_level: 0,
            target_level,
            neighbors_to_add: Vec::with_capacity((target_level + 1) as usize),
        }
    }

    pub fn set_entry(&mut self, entry: NodeId, distance: f32, max_level: u8) {
        self.entry_point = entry;
        self.entry_distance = distance;
        self.current_level = max_level;
    }
}

pub fn insert_descent_phase<F>(
    ctx: &mut InsertContext,
    get_neighbors: F,
    compute_distance: impl Fn(NodeId) -> f32,
) where
    F: Fn(NodeId, u8) -> Vec<NodeId>,
{
    while ctx.current_level > ctx.target_level {
        let neighbors_at_level = |n: NodeId| get_neighbors(n, ctx.current_level);

        let (new_entry, new_distance) = greedy_search(
            ctx.entry_point,
            ctx.entry_distance,
            neighbors_at_level,
            &compute_distance,
            1000,
        );

        ctx.entry_point = new_entry;
        ctx.entry_distance = new_distance;
        ctx.current_level -= 1;
    }
}

pub fn insert_connection_phase<F>(
    ctx: &mut InsertContext,
    search_ctx: &mut HnswSearchContext,
    get_neighbors: F,
    compute_distance: impl Fn(NodeId) -> f32,
    m: usize,
    m0: usize,
) where
    F: Fn(NodeId, u8) -> Vec<NodeId>,
{
    let mut current_entry = Candidate::new(ctx.entry_point, ctx.entry_distance);

    for level in (0..=ctx.target_level).rev() {
        let neighbors_at_level = |n: NodeId| get_neighbors(n, level);

        beam_search(
            search_ctx,
            &[current_entry],
            neighbors_at_level,
            &compute_distance,
        );

        let max_neighbors = if level == 0 { m0 } else { m };

        search_ctx.finalize_results(max_neighbors);
        let selected: Vec<NodeId> = search_ctx.results().iter().map(|c| c.node_id).collect();

        ctx.neighbors_to_add.push((level, selected));

        if level > 0 {
            search_ctx.finalize_results(1);
            if let Some(best) = search_ctx.results().first() {
                current_entry = *best;
            }
        }
    }
}

pub fn select_neighbors_simple(candidates: &[Candidate], max_neighbors: usize) -> Vec<NodeId> {
    candidates
        .iter()
        .take(max_neighbors)
        .map(|c| c.node_id)
        .collect()
}

pub fn select_neighbors_heuristic<F>(
    candidates: &[Candidate],
    max_neighbors: usize,
    compute_distance: F,
) -> Vec<NodeId>
where
    F: Fn(NodeId, NodeId) -> f32,
{
    if candidates.is_empty() {
        return Vec::new();
    }

    let mut selected = Vec::with_capacity(max_neighbors);
    let mut remaining: Vec<&Candidate> = candidates.iter().collect();

    remaining.sort_by(|a, b| {
        a.distance
            .partial_cmp(&b.distance)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    for candidate in remaining {
        if selected.len() >= max_neighbors {
            break;
        }

        let mut is_closer_to_selected = false;
        for &existing in &selected {
            let dist_to_existing = compute_distance(candidate.node_id, existing);
            if dist_to_existing < candidate.distance {
                is_closer_to_selected = true;
                break;
            }
        }

        if !is_closer_to_selected {
            selected.push(candidate.node_id);
        }
    }

    if selected.len() < max_neighbors {
        for candidate in candidates {
            if selected.len() >= max_neighbors {
                break;
            }
            if !selected.contains(&candidate.node_id) {
                selected.push(candidate.node_id);
            }
        }
    }

    selected
}

pub fn prune_neighbors(
    current_neighbors: &[NodeId],
    max_neighbors: usize,
    compute_distance: impl Fn(NodeId) -> f32,
) -> Vec<NodeId> {
    if current_neighbors.len() <= max_neighbors {
        return current_neighbors.to_vec();
    }

    let mut with_distances: Vec<(NodeId, f32)> = current_neighbors
        .iter()
        .map(|&n| (n, compute_distance(n)))
        .collect();

    with_distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

    with_distances
        .into_iter()
        .take(max_neighbors)
        .map(|(n, _)| n)
        .collect()
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NodeStatus {
    Active,
    Deleted,
}

pub struct NodeMetadata {
    pub node_id: NodeId,
    pub row_id: u64,
    pub status: NodeStatus,
    pub level: u8,
}

impl NodeMetadata {
    pub fn new(node_id: NodeId, row_id: u64, level: u8) -> Self {
        Self {
            node_id,
            row_id,
            status: NodeStatus::Active,
            level,
        }
    }

    pub fn is_deleted(&self) -> bool {
        self.status == NodeStatus::Deleted
    }

    pub fn mark_deleted(&mut self) {
        self.status = NodeStatus::Deleted;
    }
}
