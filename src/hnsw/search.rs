//! # HNSW Search Algorithms
//!
//! This module implements the core search algorithms for the HNSW index:
//! greedy descent for upper layers and beam search for the base layer.
//!
//! ## Algorithm Overview
//!
//! HNSW search proceeds in two phases:
//!
//! 1. **Greedy Descent (layers > 0)**: Starting from the entry point at the
//!    topmost layer, greedily move to the closest neighbor until reaching
//!    a local minimum. Then descend to the next layer and repeat.
//!
//! 2. **Beam Search (layer 0)**: At the base layer, maintain a set of
//!    candidates (beam width = ef_search) and explore neighbors of all
//!    candidates to find the k nearest neighbors.
//!
//! ```text
//! Layer 2:  [Entry] → Greedy descent → [Local min]
//!              ↓
//! Layer 1:  [Start] → Greedy descent → [Local min]
//!              ↓
//! Layer 0:  [Start] → Beam search (ef=32) → [k-NN results]
//! ```
//!
//! ## Candidate Management
//!
//! The search maintains two priority queues:
//!
//! - **candidates**: Min-heap of unexplored nodes, ordered by distance (closest first)
//! - **results**: Max-heap of k best results, ordered by distance (furthest first)
//!
//! The furthest result distance serves as the search bound - we stop exploring
//! candidates further than our worst result.
//!
//! ## Visited Set
//!
//! A bitset tracks visited nodes to avoid re-exploration. This is critical
//! for performance as the same node may be referenced by multiple neighbors.
//!
//! ## Zero-Allocation Design
//!
//! The search context (HnswSearchContext) owns all buffers and is reused
//! across searches. The caller must ensure the context is properly sized
//! for their use case.
//!
//! ## Memory Layout
//!
//! ```text
//! HnswSearchContext:
//! +------------------+
//! | candidates heap  |  BinaryHeap<Candidate>
//! +------------------+
//! | results heap     |  BinaryHeap<Candidate>
//! +------------------+
//! | visited bitset   |  Vec<u64> (1 bit per node)
//! +------------------+
//! | query buffer     |  [f32; max_dim] for query copy
//! +------------------+
//! ```
//!
//! ## Distance Computation
//!
//! During search, distances are computed incrementally as neighbors are
//! explored. The distance function is selected at index creation time
//! and stored in the index metadata.
//!
//! ## Performance Characteristics
//!
//! - Time complexity: O(log N) for greedy descent, O(ef * M * log(ef)) for beam search
//! - Space complexity: O(ef) for candidates + O(N/64) bits for visited set
//! - Cache efficiency: Neighbor lists are read sequentially, hot nodes stay cached
//!
//! ## Thread Safety
//!
//! Search operations are read-only and can run concurrently. Each thread
//! should have its own HnswSearchContext to avoid contention.

use std::cmp::Ordering;
use std::collections::BinaryHeap;

#[derive(Debug, Clone, Copy)]
pub struct Candidate {
    pub node_id: super::NodeId,
    pub distance: f32,
}

impl Candidate {
    pub fn new(node_id: super::NodeId, distance: f32) -> Self {
        Self { node_id, distance }
    }
}

impl PartialEq for Candidate {
    fn eq(&self, other: &Self) -> bool {
        self.distance == other.distance
    }
}

impl Eq for Candidate {}

impl PartialOrd for Candidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Candidate {
    fn cmp(&self, other: &Self) -> Ordering {
        other
            .distance
            .partial_cmp(&self.distance)
            .unwrap_or(Ordering::Equal)
    }
}

#[derive(Debug, Clone, Copy)]
pub struct ReverseCandidate(pub Candidate);

impl PartialEq for ReverseCandidate {
    fn eq(&self, other: &Self) -> bool {
        self.0.distance == other.0.distance
    }
}

impl Eq for ReverseCandidate {}

impl PartialOrd for ReverseCandidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for ReverseCandidate {
    fn cmp(&self, other: &Self) -> Ordering {
        self.0
            .distance
            .partial_cmp(&other.0.distance)
            .unwrap_or(Ordering::Equal)
    }
}

pub struct VisitedSet {
    generation: u64,
    node_generations: Vec<u64>,
}

impl VisitedSet {
    pub fn new(capacity: usize) -> Self {
        Self {
            generation: 1,
            node_generations: vec![0; capacity],
        }
    }

    pub fn clear(&mut self) {
        self.generation += 1;
        if self.generation == 0 {
            self.node_generations.fill(0);
            self.generation = 1;
        }
    }

    pub fn contains(&self, index: usize) -> bool {
        if index < self.node_generations.len() {
            self.node_generations[index] == self.generation
        } else {
            false
        }
    }

    pub fn insert(&mut self, index: usize) -> bool {
        if index >= self.node_generations.len() {
            let new_size = (index + 1).next_power_of_two();
            self.node_generations.resize(new_size, 0);
        }
        if self.node_generations[index] == self.generation {
            false
        } else {
            self.node_generations[index] = self.generation;
            true
        }
    }

    #[allow(dead_code)]
    fn resize(&mut self, new_capacity: usize) {
        if new_capacity > self.node_generations.len() {
            self.node_generations.resize(new_capacity, 0);
        }
    }
}

pub struct HnswSearchContext {
    pub candidates: BinaryHeap<Candidate>,
    pub results: BinaryHeap<ReverseCandidate>,
    pub visited: VisitedSet,
    pub output: Vec<Candidate>,
    ef_search: usize,
}

impl HnswSearchContext {
    pub fn new(ef_search: usize, max_nodes: usize) -> Self {
        Self {
            candidates: BinaryHeap::with_capacity(ef_search * 2),
            results: BinaryHeap::with_capacity(ef_search + 1),
            visited: VisitedSet::new(max_nodes),
            output: Vec::with_capacity(ef_search),
            ef_search,
        }
    }

    pub fn reset(&mut self) {
        self.candidates.clear();
        self.results.clear();
        self.visited.clear();
        self.output.clear();
    }

    pub fn ef_search(&self) -> usize {
        self.ef_search
    }

    pub fn set_ef_search(&mut self, ef: usize) {
        self.ef_search = ef;
    }

    pub fn add_candidate(&mut self, candidate: Candidate) {
        self.candidates.push(candidate);
    }

    pub fn add_result(&mut self, candidate: Candidate) {
        self.results.push(ReverseCandidate(candidate));
        if self.results.len() > self.ef_search {
            self.results.pop();
        }
    }

    pub fn worst_result_distance(&self) -> f32 {
        self.results
            .peek()
            .map(|c| c.0.distance)
            .unwrap_or(f32::INFINITY)
    }

    pub fn finalize_results(&mut self, k: usize) {
        self.output.clear();
        while let Some(ReverseCandidate(c)) = self.results.pop() {
            self.output.push(c);
        }
        self.output.reverse();
        self.output.truncate(k);
    }

    pub fn results(&self) -> &[Candidate] {
        &self.output
    }
}

pub fn greedy_search_step<F>(
    current: super::NodeId,
    current_distance: f32,
    get_neighbors: F,
    compute_distance: impl Fn(super::NodeId) -> f32,
) -> (super::NodeId, f32)
where
    F: Fn(super::NodeId) -> Vec<super::NodeId>,
{
    let neighbors = get_neighbors(current);
    let mut best_node = current;
    let mut best_distance = current_distance;

    for neighbor in neighbors {
        let dist = compute_distance(neighbor);
        if dist < best_distance {
            best_distance = dist;
            best_node = neighbor;
        }
    }

    (best_node, best_distance)
}

pub fn greedy_search<F>(
    entry_point: super::NodeId,
    entry_distance: f32,
    get_neighbors: F,
    compute_distance: impl Fn(super::NodeId) -> f32,
    max_iterations: usize,
) -> (super::NodeId, f32)
where
    F: Fn(super::NodeId) -> Vec<super::NodeId>,
{
    let mut current = entry_point;
    let mut current_distance = entry_distance;

    for _ in 0..max_iterations {
        let (new_node, new_distance) =
            greedy_search_step(current, current_distance, &get_neighbors, &compute_distance);

        if new_node == current {
            break;
        }

        current = new_node;
        current_distance = new_distance;
    }

    (current, current_distance)
}

pub fn beam_search<F>(
    ctx: &mut HnswSearchContext,
    entry_points: &[Candidate],
    get_neighbors: F,
    compute_distance: impl Fn(super::NodeId) -> f32,
) where
    F: Fn(super::NodeId) -> Vec<super::NodeId>,
{
    ctx.reset();

    for &entry in entry_points {
        let index = node_to_index(entry.node_id);
        if ctx.visited.insert(index) {
            ctx.add_candidate(entry);
            ctx.add_result(entry);
        }
    }

    while let Some(current) = ctx.candidates.pop() {
        if current.distance > ctx.worst_result_distance() {
            break;
        }

        let neighbors = get_neighbors(current.node_id);

        for neighbor in neighbors {
            let index = node_to_index(neighbor);
            if !ctx.visited.insert(index) {
                continue;
            }

            let dist = compute_distance(neighbor);

            if dist < ctx.worst_result_distance() || ctx.results.len() < ctx.ef_search {
                ctx.add_candidate(Candidate::new(neighbor, dist));
                ctx.add_result(Candidate::new(neighbor, dist));
            }
        }
    }
}

pub fn beam_search_filtered<F, V>(
    ctx: &mut HnswSearchContext,
    entry_points: &[Candidate],
    get_neighbors: F,
    compute_distance: impl Fn(super::NodeId) -> f32,
    is_visible: V,
) where
    F: Fn(super::NodeId) -> Vec<super::NodeId>,
    V: Fn(super::NodeId) -> bool,
{
    ctx.reset();

    for &entry in entry_points {
        let index = node_to_index(entry.node_id);
        if ctx.visited.insert(index) {
            ctx.add_candidate(entry);
            if is_visible(entry.node_id) {
                ctx.add_result(entry);
            }
        }
    }

    while let Some(current) = ctx.candidates.pop() {
        if current.distance > ctx.worst_result_distance() {
            break;
        }

        let neighbors = get_neighbors(current.node_id);

        for neighbor in neighbors {
            let index = node_to_index(neighbor);
            if !ctx.visited.insert(index) {
                continue;
            }

            let dist = compute_distance(neighbor);

            ctx.add_candidate(Candidate::new(neighbor, dist));

            if is_visible(neighbor)
                && (dist < ctx.worst_result_distance() || ctx.results.len() < ctx.ef_search)
            {
                ctx.add_result(Candidate::new(neighbor, dist));
            }
        }
    }
}

fn node_to_index(node_id: super::NodeId) -> usize {
    ((node_id.page_no() as usize) << 16) | (node_id.slot_index() as usize)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hnsw::NodeId;

    #[test]
    fn candidate_ordering_min_heap() {
        let mut heap: BinaryHeap<Candidate> = BinaryHeap::new();

        heap.push(Candidate::new(NodeId::new(1, 0), 3.0));
        heap.push(Candidate::new(NodeId::new(2, 0), 1.0));
        heap.push(Candidate::new(NodeId::new(3, 0), 2.0));

        assert_eq!(heap.pop().unwrap().distance, 1.0);
        assert_eq!(heap.pop().unwrap().distance, 2.0);
        assert_eq!(heap.pop().unwrap().distance, 3.0);
    }

    #[test]
    fn reverse_candidate_ordering_max_heap() {
        let mut heap: BinaryHeap<ReverseCandidate> = BinaryHeap::new();

        heap.push(ReverseCandidate(Candidate::new(NodeId::new(1, 0), 3.0)));
        heap.push(ReverseCandidate(Candidate::new(NodeId::new(2, 0), 1.0)));
        heap.push(ReverseCandidate(Candidate::new(NodeId::new(3, 0), 2.0)));

        assert_eq!(heap.pop().unwrap().0.distance, 3.0);
        assert_eq!(heap.pop().unwrap().0.distance, 2.0);
        assert_eq!(heap.pop().unwrap().0.distance, 1.0);
    }

    #[test]
    fn visited_set_tracks_visits() {
        let mut visited = VisitedSet::new(100);

        assert!(!visited.contains(5));
        assert!(visited.insert(5));
        assert!(visited.contains(5));
        assert!(!visited.insert(5));
    }

    #[test]
    fn visited_set_clear_resets() {
        let mut visited = VisitedSet::new(100);

        visited.insert(5);
        visited.insert(10);
        visited.clear();

        assert!(!visited.contains(5));
        assert!(!visited.contains(10));
    }

    #[test]
    fn visited_set_grows_automatically() {
        let mut visited = VisitedSet::new(10);

        assert!(visited.insert(100));
        assert!(visited.contains(100));
    }

    #[test]
    fn search_context_manages_candidates() {
        let mut ctx = HnswSearchContext::new(3, 100);

        ctx.add_candidate(Candidate::new(NodeId::new(1, 0), 1.0));
        ctx.add_candidate(Candidate::new(NodeId::new(2, 0), 2.0));

        let first = ctx.candidates.pop().unwrap();
        assert_eq!(first.distance, 1.0);
    }

    #[test]
    fn search_context_limits_results() {
        let mut ctx = HnswSearchContext::new(2, 100);

        ctx.add_result(Candidate::new(NodeId::new(1, 0), 1.0));
        ctx.add_result(Candidate::new(NodeId::new(2, 0), 2.0));
        ctx.add_result(Candidate::new(NodeId::new(3, 0), 0.5));

        assert_eq!(ctx.results.len(), 2);

        let worst = ctx.worst_result_distance();
        assert!((1.0..=2.0).contains(&worst));
    }

    #[test]
    fn search_context_finalize_results() {
        let mut ctx = HnswSearchContext::new(10, 100);

        ctx.add_result(Candidate::new(NodeId::new(1, 0), 3.0));
        ctx.add_result(Candidate::new(NodeId::new(2, 0), 1.0));
        ctx.add_result(Candidate::new(NodeId::new(3, 0), 2.0));

        ctx.finalize_results(2);

        assert_eq!(ctx.results().len(), 2);
        assert_eq!(ctx.results()[0].distance, 1.0);
        assert_eq!(ctx.results()[1].distance, 2.0);
    }

    #[test]
    fn greedy_search_step_finds_closer_neighbor() {
        let neighbors = vec![NodeId::new(1, 0), NodeId::new(2, 0), NodeId::new(3, 0)];

        let distances = [0.5, 0.3, 0.8];

        let get_neighbors = |_: NodeId| neighbors.clone();
        let compute_distance = |n: NodeId| distances[n.page_no() as usize - 1];

        let current = NodeId::new(0, 0);
        let (best, best_dist) = greedy_search_step(current, 1.0, get_neighbors, compute_distance);

        assert_eq!(best.page_no(), 2);
        assert_eq!(best_dist, 0.3);
    }

    #[test]
    fn greedy_search_step_stays_if_no_improvement() {
        let neighbors = vec![NodeId::new(1, 0), NodeId::new(2, 0)];

        let get_neighbors = |_: NodeId| neighbors.clone();
        let compute_distance = |_: NodeId| 2.0;

        let current = NodeId::new(0, 0);
        let (best, best_dist) = greedy_search_step(current, 1.0, get_neighbors, compute_distance);

        assert_eq!(best, current);
        assert_eq!(best_dist, 1.0);
    }

    #[test]
    fn greedy_search_converges() {
        let graph: Vec<Vec<NodeId>> = vec![
            vec![NodeId::new(1, 0), NodeId::new(2, 0)],
            vec![NodeId::new(0, 0), NodeId::new(3, 0)],
            vec![NodeId::new(0, 0), NodeId::new(3, 0)],
            vec![NodeId::new(1, 0), NodeId::new(2, 0)],
        ];

        let distances = [1.0, 0.5, 0.8, 0.2];

        let get_neighbors = |n: NodeId| graph[n.page_no() as usize].clone();
        let compute_distance = |n: NodeId| distances[n.page_no() as usize];

        let entry = NodeId::new(0, 0);
        let entry_dist = distances[0];

        let (result, result_dist) =
            greedy_search(entry, entry_dist, get_neighbors, compute_distance, 100);

        assert_eq!(result.page_no(), 3);
        assert_eq!(result_dist, 0.2);
    }

    #[test]
    fn beam_search_finds_k_nearest() {
        let mut ctx = HnswSearchContext::new(10, 100);

        let graph: Vec<Vec<NodeId>> = vec![
            vec![NodeId::new(1, 0), NodeId::new(2, 0)],
            vec![NodeId::new(0, 0), NodeId::new(3, 0), NodeId::new(4, 0)],
            vec![NodeId::new(0, 0), NodeId::new(4, 0)],
            vec![NodeId::new(1, 0)],
            vec![NodeId::new(1, 0), NodeId::new(2, 0)],
        ];

        let distances = [0.5, 0.3, 0.8, 0.1, 0.2];

        let get_neighbors = |n: NodeId| graph[n.page_no() as usize].clone();
        let compute_distance = |n: NodeId| distances[n.page_no() as usize];

        let entry = Candidate::new(NodeId::new(0, 0), 0.5);
        beam_search(&mut ctx, &[entry], get_neighbors, compute_distance);

        ctx.finalize_results(3);
        let results = ctx.results();

        assert!(results.len() >= 3);
        assert_eq!(results[0].distance, 0.1);
        assert_eq!(results[1].distance, 0.2);
        assert_eq!(results[2].distance, 0.3);
    }

    #[test]
    fn beam_search_respects_ef_bound() {
        let mut ctx = HnswSearchContext::new(2, 100);

        let graph: Vec<Vec<NodeId>> = vec![
            vec![NodeId::new(1, 0), NodeId::new(2, 0)],
            vec![NodeId::new(0, 0)],
            vec![NodeId::new(0, 0)],
        ];

        let distances = [0.5, 0.3, 0.8];

        let get_neighbors = |n: NodeId| graph[n.page_no() as usize].clone();
        let compute_distance = |n: NodeId| distances[n.page_no() as usize];

        let entry = Candidate::new(NodeId::new(0, 0), 0.5);
        beam_search(&mut ctx, &[entry], get_neighbors, compute_distance);

        assert!(ctx.results.len() <= 2);
    }

    #[test]
    fn search_context_reset_clears_all() {
        let mut ctx = HnswSearchContext::new(10, 100);

        ctx.add_candidate(Candidate::new(NodeId::new(1, 0), 1.0));
        ctx.add_result(Candidate::new(NodeId::new(2, 0), 2.0));
        ctx.visited.insert(5);

        ctx.reset();

        assert!(ctx.candidates.is_empty());
        assert!(ctx.results.is_empty());
        assert!(!ctx.visited.contains(5));
    }

    #[test]
    fn beam_search_filtered_excludes_invisible_from_results() {
        let mut ctx = HnswSearchContext::new(10, 100);

        let graph: Vec<Vec<NodeId>> = vec![
            vec![NodeId::new(1, 0), NodeId::new(2, 0)],
            vec![NodeId::new(0, 0), NodeId::new(3, 0), NodeId::new(4, 0)],
            vec![NodeId::new(0, 0), NodeId::new(4, 0)],
            vec![NodeId::new(1, 0)],
            vec![NodeId::new(1, 0), NodeId::new(2, 0)],
        ];

        let distances = [0.5, 0.3, 0.8, 0.1, 0.2];

        let invisible_nodes = [3];

        let get_neighbors = |n: NodeId| graph[n.page_no() as usize].clone();
        let compute_distance = |n: NodeId| distances[n.page_no() as usize];
        let is_visible = |n: NodeId| !invisible_nodes.contains(&(n.page_no() as i32));

        let entry = Candidate::new(NodeId::new(0, 0), 0.5);
        beam_search_filtered(
            &mut ctx,
            &[entry],
            get_neighbors,
            compute_distance,
            is_visible,
        );

        ctx.finalize_results(10);
        let results = ctx.results();

        for result in results {
            assert!(
                is_visible(result.node_id),
                "invisible node {:?} found in results",
                result.node_id
            );
        }

        let result_pages: Vec<u32> = results.iter().map(|c| c.node_id.page_no()).collect();
        assert!(!result_pages.contains(&3), "node 3 should be invisible");

        assert_eq!(results[0].distance, 0.2);
    }

    #[test]
    fn beam_search_filtered_traverses_through_invisible_nodes() {
        let mut ctx = HnswSearchContext::new(10, 100);

        let graph: Vec<Vec<NodeId>> = vec![
            vec![NodeId::new(1, 0)],
            vec![NodeId::new(0, 0), NodeId::new(2, 0)],
            vec![NodeId::new(1, 0)],
        ];

        let distances = [1.0, 0.5, 0.1];

        let invisible_nodes = [1];

        let get_neighbors = |n: NodeId| graph[n.page_no() as usize].clone();
        let compute_distance = |n: NodeId| distances[n.page_no() as usize];
        let is_visible = |n: NodeId| !invisible_nodes.contains(&(n.page_no() as i32));

        let entry = Candidate::new(NodeId::new(0, 0), 1.0);
        beam_search_filtered(
            &mut ctx,
            &[entry],
            get_neighbors,
            compute_distance,
            is_visible,
        );

        ctx.finalize_results(10);
        let results = ctx.results();

        assert!(
            results.len() >= 2,
            "should find node 0 and 2 through invisible node 1"
        );

        let result_pages: Vec<u32> = results.iter().map(|c| c.node_id.page_no()).collect();
        assert!(
            result_pages.contains(&2),
            "node 2 should be reachable through invisible node 1"
        );
        assert!(
            !result_pages.contains(&1),
            "node 1 should be invisible in results"
        );
    }
}
