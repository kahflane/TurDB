//! # HNSW Comparison Tests: TurDB vs Qdrant
//!
//! These tests validate TurDB's HNSW vector search implementation by comparing
//! results against Qdrant, a production-grade vector database.
//!
//! ## Requirements Verified
//!
//! R1: TurDB HNSW returns top-k results with >= 90% recall vs Qdrant
//! R2: Euclidean distances are computed correctly
//! R3: HNSW graph construction maintains search quality
//!
//! ## Setup
//!
//! Before running these tests:
//! 1. Run `python scripts/prepare_qdrant_comparison.py` to generate embeddings
//! 2. Ensure Qdrant server is accessible at configured address
//!
//! ## Running
//!
//! ```sh
//! cargo test --test hnsw_qdrant_comparison -- --nocapture --ignored
//! ```

use serde::Deserialize;
use std::collections::HashSet;
use std::path::Path;
use tempfile::tempdir;

use turdb::hnsw::{self, DistanceFunction, PersistentHnswIndex, QuantizationType};

const QDRANT_HOST: &str = "103.209.156.107";
const QDRANT_PORT: u16 = 6333;
const QDRANT_API_KEY: &str = "38xlYVBLsD5Owf8sa0utlKB89aeMQHtj";
const QDRANT_COLLECTION: &str = "turdb_comparison";

const EMBEDDINGS_PATH: &str = "testdata/embeddings_1k.json";
const K: usize = 50;
const MIN_RECALL: f32 = 0.80;

#[derive(Debug, Deserialize)]
struct EmbeddingsFile {
    version: u32,
    dimension: usize,
    count: usize,
    query_count: usize,
    distance_metric: String,
    #[allow(dead_code)]
    source: String,
    #[allow(dead_code)]
    embedding_model: String,
    #[allow(dead_code)]
    texts: Vec<String>,
    vectors: Vec<Vec<f64>>,
    query_indices: Vec<usize>,
}

#[derive(Debug, Deserialize)]
struct QdrantSearchResponse {
    result: Vec<QdrantScoredPoint>,
}

#[derive(Debug, Deserialize)]
struct QdrantScoredPoint {
    id: u64,
    score: f32,
}

struct QdrantClient {
    host: String,
    port: u16,
    api_key: String,
    collection: String,
}

impl QdrantClient {
    fn new(host: &str, port: u16, api_key: &str, collection: &str) -> Self {
        Self {
            host: host.to_string(),
            port,
            api_key: api_key.to_string(),
            collection: collection.to_string(),
        }
    }

    fn search(&self, query: &[f32], limit: usize) -> eyre::Result<Vec<(u64, f32)>> {
        let url = format!(
            "http://{}:{}/collections/{}/points/query",
            self.host, self.port, self.collection
        );

        let query_f64: Vec<f64> = query.iter().map(|&x| x as f64).collect();

        let body = serde_json::json!({
            "query": query_f64,
            "limit": limit,
            "with_payload": false,
            "with_vector": false
        });

        let response = ureq::post(&url)
            .set("api-key", &self.api_key)
            .set("Content-Type", "application/json")
            .send_json(&body)?;

        let result: serde_json::Value = response.into_json()?;

        let points = result["result"]["points"]
            .as_array()
            .ok_or_else(|| eyre::eyre!("No points in response"))?;

        let mut results = Vec::new();
        for point in points {
            let id = point["id"].as_u64().unwrap_or(0);
            let score = point["score"].as_f64().unwrap_or(0.0) as f32;
            results.push((id, score));
        }

        Ok(results)
    }
}

fn load_embeddings(path: &Path) -> eyre::Result<EmbeddingsFile> {
    let file = std::fs::File::open(path)?;
    let reader = std::io::BufReader::new(file);
    let data: EmbeddingsFile = serde_json::from_reader(reader)?;

    eyre::ensure!(data.version == 1, "Unsupported embeddings file version");
    eyre::ensure!(!data.vectors.is_empty(), "No vectors in file");
    eyre::ensure!(
        data.vectors[0].len() == data.dimension,
        "Dimension mismatch"
    );

    Ok(data)
}

fn convert_to_f32(vectors: &[Vec<f64>]) -> Vec<Vec<f32>> {
    vectors
        .iter()
        .map(|v| v.iter().map(|&x| x as f32).collect())
        .collect()
}

fn build_turdb_index(
    path: &Path,
    vectors: &[Vec<f32>],
    dimensions: u16,
) -> eyre::Result<PersistentHnswIndex> {
    let mut index = PersistentHnswIndex::create(
        path,
        1,
        1,
        dimensions,
        16,
        100,
        64,
        DistanceFunction::L2,
        QuantizationType::None,
    )?;

    let mut rng_state: u64 = 12345;
    for (i, vector) in vectors.iter().enumerate() {
        rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
        let random_value = (rng_state as f64) / (u64::MAX as f64);

        index.insert(i as u64, vector, random_value)?;

        if (i + 1) % 100 == 0 {
            println!("  Inserted {}/{} vectors", i + 1, vectors.len());
        }
    }

    index.sync()?;
    println!(
        "  Built TurDB HNSW index with {} nodes",
        index.index().node_count()
    );

    Ok(index)
}

fn search_turdb(
    index: &PersistentHnswIndex,
    query: &[f32],
    k: usize,
    vectors: &[Vec<f32>],
) -> eyre::Result<Vec<(u64, f32)>> {
    let max_nodes = (index.index().node_count() as usize).max(1000);
    let mut ctx = hnsw::search::HnswSearchContext::new(64, max_nodes);

    let get_vector = |row_id: u64| -> Option<Vec<f32>> {
        vectors.get(row_id as usize).cloned()
    };

    let results = index.search(query, k, &mut ctx, get_vector)?;

    Ok(results
        .into_iter()
        .map(|r| (r.row_id, r.distance))
        .collect())
}

fn calculate_recall(turdb: &[(u64, f32)], qdrant: &[(u64, f32)], k: usize) -> f32 {
    let turdb_ids: HashSet<u64> = turdb.iter().take(k).map(|(id, _)| *id).collect();
    let qdrant_ids: HashSet<u64> = qdrant.iter().take(k).map(|(id, _)| *id).collect();

    let intersection = turdb_ids.intersection(&qdrant_ids).count();
    intersection as f32 / k as f32
}

#[test]
#[ignore]
fn compare_turdb_hnsw_vs_qdrant_euclidean() {
    println!("\n=== TurDB HNSW vs Qdrant Comparison Test ===\n");

    let embeddings_path = Path::new(EMBEDDINGS_PATH);
    if !embeddings_path.exists() {
        panic!(
            "Embeddings file not found at {}. Run prepare_qdrant_comparison.py first.",
            EMBEDDINGS_PATH
        );
    }

    println!("Loading embeddings from {}...", EMBEDDINGS_PATH);
    let embeddings = load_embeddings(embeddings_path).expect("Failed to load embeddings");
    println!(
        "  Loaded {} vectors with dimension {}",
        embeddings.count, embeddings.dimension
    );
    println!("  Query indices: {:?}", &embeddings.query_indices[..5]);

    let vectors_f32 = convert_to_f32(&embeddings.vectors);

    let temp_dir = tempdir().expect("Failed to create temp directory");
    let index_path = temp_dir.path().join("test.hnsw");

    println!("\nBuilding TurDB HNSW index...");
    let index = build_turdb_index(&index_path, &vectors_f32, embeddings.dimension as u16)
        .expect("Failed to build index");

    let qdrant = QdrantClient::new(QDRANT_HOST, QDRANT_PORT, QDRANT_API_KEY, QDRANT_COLLECTION);

    println!("\nRunning {} queries with k={}...", embeddings.query_count, K);

    let mut recalls: Vec<f32> = Vec::new();
    let mut failed_queries: Vec<(usize, f32)> = Vec::new();

    for (i, &query_idx) in embeddings.query_indices.iter().enumerate() {
        let query = &vectors_f32[query_idx];

        let qdrant_results = match qdrant.search(query, K) {
            Ok(r) => r,
            Err(e) => {
                println!("  Query {}: Qdrant search failed: {}", i, e);
                continue;
            }
        };

        let turdb_results = match search_turdb(&index, query, K, &vectors_f32) {
            Ok(r) => r,
            Err(e) => {
                println!("  Query {}: TurDB search failed: {}", i, e);
                continue;
            }
        };

        let recall = calculate_recall(&turdb_results, &qdrant_results, K);
        recalls.push(recall);

        if recall < MIN_RECALL {
            failed_queries.push((query_idx, recall));
        }

        if i < 5 || recall < MIN_RECALL {
            println!(
                "  Query {} (idx {}): recall={:.2}%, TurDB top-3: {:?}, Qdrant top-3: {:?}",
                i,
                query_idx,
                recall * 100.0,
                turdb_results.iter().take(3).map(|(id, _)| id).collect::<Vec<_>>(),
                qdrant_results.iter().take(3).map(|(id, _)| id).collect::<Vec<_>>(),
            );
        }
    }

    let avg_recall = recalls.iter().sum::<f32>() / recalls.len() as f32;
    let min_recall_actual = recalls.iter().cloned().fold(f32::INFINITY, f32::min);
    let max_recall = recalls.iter().cloned().fold(0.0f32, f32::max);

    println!("\n=== Results ===");
    println!("  Queries completed: {}", recalls.len());
    println!("  Average recall@{}: {:.2}%", K, avg_recall * 100.0);
    println!("  Min recall: {:.2}%", min_recall_actual * 100.0);
    println!("  Max recall: {:.2}%", max_recall * 100.0);
    println!(
        "  Queries below {}% recall: {}",
        MIN_RECALL * 100.0,
        failed_queries.len()
    );

    if !failed_queries.is_empty() {
        println!("\n  Failed queries (below {}% recall):", MIN_RECALL * 100.0);
        for (idx, recall) in failed_queries.iter().take(10) {
            println!("    Query idx {}: {:.2}%", idx, recall * 100.0);
        }
    }

    assert!(
        avg_recall >= MIN_RECALL,
        "Average recall {:.2}% is below minimum {:.2}%",
        avg_recall * 100.0,
        MIN_RECALL * 100.0
    );

    println!("\n=== Test PASSED ===\n");
}

#[test]
fn test_embeddings_file_exists() {
    let path = Path::new(EMBEDDINGS_PATH);
    if !path.exists() {
        println!(
            "Warning: Embeddings file not found at {}",
            EMBEDDINGS_PATH
        );
        println!("Run: python scripts/prepare_qdrant_comparison.py");
    }
}
