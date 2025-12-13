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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn level_selection_mostly_level_0() {
        let ml = calculate_ml(16);

        let mut level_counts = [0u32; 16];
        let samples = 10000;

        for i in 0..samples {
            let random_value = (i as f64 + 0.5) / samples as f64;
            let level = select_level(random_value, ml) as usize;
            if level < 16 {
                level_counts[level] += 1;
            }
        }

        let level_0_ratio = level_counts[0] as f64 / samples as f64;
        assert!(
            level_0_ratio > 0.5,
            "Expected >50% at level 0, got {}%",
            level_0_ratio * 100.0
        );
    }

    #[test]
    fn level_selection_capped_at_15() {
        let ml = calculate_ml(16);

        let level = select_level(0.0001, ml);
        assert!(level <= 15);

        let level = select_level(f64::MIN_POSITIVE, ml);
        assert!(level <= 15);
    }

    #[test]
    fn calculate_ml_for_m_16() {
        let ml = calculate_ml(16);

        assert!((ml - 0.3607).abs() < 0.01);
    }

    #[test]
    fn select_neighbors_simple_takes_first_m() {
        let candidates = vec![
            Candidate::new(NodeId::new(0, 0), 0.1),
            Candidate::new(NodeId::new(1, 0), 0.2),
            Candidate::new(NodeId::new(2, 0), 0.3),
            Candidate::new(NodeId::new(3, 0), 0.4),
        ];

        let selected = select_neighbors_simple(&candidates, 2);

        assert_eq!(selected.len(), 2);
        assert_eq!(selected[0], NodeId::new(0, 0));
        assert_eq!(selected[1], NodeId::new(1, 0));
    }

    #[test]
    fn select_neighbors_heuristic_prefers_diverse() {
        let candidates = vec![
            Candidate::new(NodeId::new(0, 0), 0.1),
            Candidate::new(NodeId::new(1, 0), 0.15),
            Candidate::new(NodeId::new(2, 0), 0.3),
        ];

        let compute_distance = |a: NodeId, b: NodeId| {
            if (a.page_no() == 0 && b.page_no() == 1) || (a.page_no() == 1 && b.page_no() == 0) {
                0.05
            } else {
                1.0
            }
        };

        let selected = select_neighbors_heuristic(&candidates, 2, compute_distance);

        assert_eq!(selected.len(), 2);
        assert_eq!(selected[0], NodeId::new(0, 0));
        assert!(selected.contains(&NodeId::new(2, 0)));
    }

    #[test]
    fn prune_neighbors_keeps_closest() {
        let neighbors = vec![
            NodeId::new(0, 0),
            NodeId::new(1, 0),
            NodeId::new(2, 0),
            NodeId::new(3, 0),
        ];

        let distances = [0.3, 0.1, 0.4, 0.2];
        let compute_distance = |n: NodeId| distances[n.page_no() as usize];

        let pruned = prune_neighbors(&neighbors, 2, compute_distance);

        assert_eq!(pruned.len(), 2);
        assert!(pruned.contains(&NodeId::new(1, 0)));
        assert!(pruned.contains(&NodeId::new(3, 0)));
    }

    #[test]
    fn prune_neighbors_no_change_if_under_limit() {
        let neighbors = vec![NodeId::new(0, 0), NodeId::new(1, 0)];

        let compute_distance = |n: NodeId| n.page_no() as f32;

        let pruned = prune_neighbors(&neighbors, 5, compute_distance);

        assert_eq!(pruned.len(), 2);
    }

    #[test]
    fn node_metadata_starts_active() {
        let meta = NodeMetadata::new(NodeId::new(1, 0), 100, 2);

        assert_eq!(meta.status, NodeStatus::Active);
        assert!(!meta.is_deleted());
    }

    #[test]
    fn node_metadata_can_be_marked_deleted() {
        let mut meta = NodeMetadata::new(NodeId::new(1, 0), 100, 2);

        meta.mark_deleted();

        assert_eq!(meta.status, NodeStatus::Deleted);
        assert!(meta.is_deleted());
    }

    #[test]
    fn insert_context_initialization() {
        let ctx = InsertContext::new(3);

        assert!(ctx.entry_point.is_none());
        assert_eq!(ctx.entry_distance, f32::INFINITY);
        assert_eq!(ctx.target_level, 3);
        assert!(ctx.neighbors_to_add.is_empty());
    }

    #[test]
    fn insert_context_set_entry() {
        let mut ctx = InsertContext::new(2);
        ctx.set_entry(NodeId::new(5, 0), 0.5, 4);

        assert_eq!(ctx.entry_point, NodeId::new(5, 0));
        assert_eq!(ctx.entry_distance, 0.5);
        assert_eq!(ctx.current_level, 4);
    }

    #[test]
    fn insert_descent_phase_descends_to_target() {
        let mut ctx = InsertContext::new(1);
        ctx.set_entry(NodeId::new(0, 0), 1.0, 3);

        let graph: Vec<Vec<NodeId>> = vec![
            vec![NodeId::new(1, 0)],
            vec![NodeId::new(0, 0), NodeId::new(2, 0)],
            vec![NodeId::new(1, 0)],
        ];

        let distances = [1.0, 0.5, 0.3];

        let get_neighbors = |n: NodeId, _level: u8| {
            if (n.page_no() as usize) < graph.len() {
                graph[n.page_no() as usize].clone()
            } else {
                vec![]
            }
        };

        let compute_distance = |n: NodeId| {
            if (n.page_no() as usize) < distances.len() {
                distances[n.page_no() as usize]
            } else {
                f32::INFINITY
            }
        };

        insert_descent_phase(&mut ctx, get_neighbors, compute_distance);

        assert_eq!(ctx.current_level, 1);
        assert_eq!(ctx.entry_point.page_no(), 2);
        assert_eq!(ctx.entry_distance, 0.3);
    }

    #[test]
    fn level_distribution_follows_exponential() {
        let ml = calculate_ml(16);

        let mut level_counts = [0u32; 8];
        let samples = 100000;

        for i in 0..samples {
            let random_value = (i as f64 + 0.5) / samples as f64;
            let level = select_level(random_value, ml) as usize;
            if level < level_counts.len() {
                level_counts[level] += 1;
            }
        }

        let total: u32 = level_counts.iter().sum();
        let level_0_fraction = level_counts[0] as f64 / total as f64;

        assert!(
            level_0_fraction > 0.9,
            "Expected >90% at level 0 for M=16, got {}%",
            level_0_fraction * 100.0
        );

        let expected_l1_fraction = 1.0 / 16.0;
        let actual_l1_fraction = level_counts[1] as f64 / total as f64;
        assert!(
            (actual_l1_fraction - expected_l1_fraction).abs() < 0.02,
            "Level 1 fraction {} differs from expected {}",
            actual_l1_fraction,
            expected_l1_fraction
        );
    }
}
