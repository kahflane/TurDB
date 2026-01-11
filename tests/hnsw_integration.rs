//! # HNSW Integration Tests
//!
//! Tests for HNSW vector index integration with DML operations using SQL.
//! Following TDD: tests written BEFORE implementation code.
//!
//! ## Test Coverage
//!
//! 1. CREATE INDEX ... USING HNSW creates HNSW index
//! 2. INSERT with vector column updates HNSW index
//! 3. DELETE removes vectors from HNSW index
//! 4. UPDATE on vector column updates HNSW index
//! 5. SELECT with ORDER BY <-> returns k-NN results
//!
//! ## Usage
//!
//! ```sh
//! cargo test --test hnsw_integration --release -- --nocapture
//! ```

use tempfile::TempDir;
use turdb::{Database, OwnedValue};

fn create_test_db() -> (Database, TempDir) {
    let temp_dir = TempDir::new().expect("Failed to create temp directory");
    let db = Database::create(temp_dir.path()).expect("Failed to create database");
    (db, temp_dir)
}

fn setup_vector_table(db: &Database) {
    db.execute(
        "CREATE TABLE embeddings (
            id BIGINT PRIMARY KEY,
            name TEXT,
            vec VECTOR(4)
        )",
    )
    .expect("Failed to create table");
}

fn setup_vector_table_with_hnsw_index(db: &Database) {
    setup_vector_table(db);

    db.execute("CREATE INDEX idx_vec ON embeddings USING HNSW (vec)")
        .expect("Failed to create HNSW index");
}

fn insert_vector(db: &Database, id: i64, name: &str, vec: &[f32; 4]) {
    let sql = format!(
        "INSERT INTO embeddings (id, name, vec) VALUES ({}, '{}', '[{},{},{},{}]')",
        id, name, vec[0], vec[1], vec[2], vec[3]
    );
    db.execute(&sql).expect("Failed to insert");
}

mod hnsw_node_inline_tests {
    use turdb::hnsw::{HnswNodeInline, NodeId, MAX_L0_NEIGHBORS, MAX_LEVEL_NEIGHBORS, MAX_LEVELS};

    #[test]
    fn new_creates_with_correct_row_id_and_level() {
        let node = HnswNodeInline::new(42, 2);
        assert_eq!(node.row_id(), 42);
        assert_eq!(node.max_level(), 2);
        assert_eq!(node.level0_neighbor_count(), 0);
    }

    #[test]
    fn add_level0_neighbor_stores_correctly() {
        let mut node = HnswNodeInline::new(1, 0);
        let neighbor = NodeId::new(10, 5);

        node.add_level0_neighbor(neighbor);

        assert_eq!(node.level0_neighbor_count(), 1);
        let neighbors = node.level0_neighbors();
        assert_eq!(neighbors[0].page_no(), 10);
        assert_eq!(neighbors[0].slot_index(), 5);
    }

    #[test]
    fn add_level0_neighbor_respects_max_capacity() {
        let mut node = HnswNodeInline::new(1, 0);

        for i in 0..MAX_L0_NEIGHBORS + 5 {
            node.add_level0_neighbor(NodeId::new(i as u32, 0));
        }

        assert_eq!(node.level0_neighbor_count() as usize, MAX_L0_NEIGHBORS);
    }

    #[test]
    fn add_higher_level_neighbor_stores_correctly() {
        let mut node = HnswNodeInline::new(1, 2);
        let neighbor = NodeId::new(20, 3);

        node.add_neighbor_at_level(1, neighbor);

        let neighbors = node.neighbors_at_level(1);
        assert_eq!(neighbors.len(), 1);
        assert_eq!(neighbors[0].page_no(), 20);
    }

    #[test]
    fn higher_level_respects_max_capacity() {
        let mut node = HnswNodeInline::new(1, 1);

        for i in 0..MAX_LEVEL_NEIGHBORS + 5 {
            node.add_neighbor_at_level(1, NodeId::new(i as u32, 0));
        }

        let neighbors = node.neighbors_at_level(1);
        assert_eq!(neighbors.len(), MAX_LEVEL_NEIGHBORS);
    }

    #[test]
    fn max_level_capped_at_max_levels() {
        let node = HnswNodeInline::new(1, MAX_LEVELS as u8 + 2);
        assert!(node.max_level() <= MAX_LEVELS as u8);
    }

    #[test]
    fn serializes_and_deserializes_correctly() {
        let mut node = HnswNodeInline::new(12345, 2);
        node.add_level0_neighbor(NodeId::new(1, 2));
        node.add_level0_neighbor(NodeId::new(3, 4));
        node.add_neighbor_at_level(1, NodeId::new(5, 6));
        node.add_neighbor_at_level(2, NodeId::new(7, 8));

        let mut buf = [0u8; 1024];
        let written = node.write_to(&mut buf);

        let restored = HnswNodeInline::read_from(&buf[..written]).unwrap();

        assert_eq!(restored.row_id(), 12345);
        assert_eq!(restored.max_level(), 2);
        assert_eq!(restored.level0_neighbor_count(), 2);
        assert_eq!(restored.level0_neighbors()[0].page_no(), 1);
        assert_eq!(restored.level0_neighbors()[1].page_no(), 3);
        assert_eq!(restored.neighbors_at_level(1).len(), 1);
        assert_eq!(restored.neighbors_at_level(1)[0].page_no(), 5);
    }

    #[test]
    fn remove_level0_neighbor_shifts_remaining() {
        let mut node = HnswNodeInline::new(1, 0);
        node.add_level0_neighbor(NodeId::new(10, 0));
        node.add_level0_neighbor(NodeId::new(20, 0));
        node.add_level0_neighbor(NodeId::new(30, 0));

        node.remove_neighbor_at_level(0, NodeId::new(20, 0));

        assert_eq!(node.level0_neighbor_count(), 2);
        assert_eq!(node.level0_neighbors()[0].page_no(), 10);
        assert_eq!(node.level0_neighbors()[1].page_no(), 30);
    }

    #[test]
    fn serialized_size_is_fixed() {
        let size = HnswNodeInline::serialized_size();
        assert!(size > 0);
        assert_eq!(size, HnswNodeInline::serialized_size());
    }
}

mod hnsw_dml_integration_tests {
    use super::*;

    #[test]
    fn create_hnsw_index_succeeds() {
        let (db, _temp_dir) = create_test_db();
        setup_vector_table_with_hnsw_index(&db);
    }

    #[test]
    fn insert_with_hnsw_index_succeeds() {
        let (db, _temp_dir) = create_test_db();
        setup_vector_table_with_hnsw_index(&db);

        insert_vector(&db, 1, "vec1", &[0.1, 0.2, 0.3, 0.4]);
        insert_vector(&db, 2, "vec2", &[0.5, 0.6, 0.7, 0.8]);

        let rows = db.query("SELECT COUNT(*) FROM embeddings").unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].values[0], OwnedValue::Int(2));
    }

    #[test]
    fn delete_with_hnsw_index_removes_from_index() {
        let (db, _temp_dir) = create_test_db();
        setup_vector_table_with_hnsw_index(&db);

        insert_vector(&db, 1, "vec1", &[0.1, 0.2, 0.3, 0.4]);
        insert_vector(&db, 2, "vec2", &[0.5, 0.6, 0.7, 0.8]);

        db.execute("DELETE FROM embeddings WHERE id = 1")
            .expect("DELETE should succeed");

        let rows = db.query("SELECT COUNT(*) FROM embeddings").unwrap();
        assert_eq!(rows[0].values[0], OwnedValue::Int(1));
    }

    #[test]
    fn update_vector_column_updates_index() {
        let (db, _temp_dir) = create_test_db();
        setup_vector_table_with_hnsw_index(&db);

        insert_vector(&db, 1, "vec1", &[0.1, 0.2, 0.3, 0.4]);

        db.execute("UPDATE embeddings SET vec = '[0.9, 0.8, 0.7, 0.6]' WHERE id = 1")
            .expect("UPDATE should succeed");

        let rows = db.query("SELECT vec FROM embeddings WHERE id = 1").unwrap();
        assert_eq!(rows.len(), 1);
        if let OwnedValue::Vector(v) = &rows[0].values[0] {
            assert!((v[0] - 0.9).abs() < 0.001);
        } else {
            panic!("Expected vector value");
        }
    }

    #[test]
    fn knn_search_returns_nearest_neighbors() {
        let (db, _temp_dir) = create_test_db();
        setup_vector_table_with_hnsw_index(&db);

        insert_vector(&db, 1, "near", &[0.1, 0.1, 0.1, 0.1]);
        insert_vector(&db, 2, "mid", &[0.5, 0.5, 0.5, 0.5]);
        insert_vector(&db, 3, "far", &[0.9, 0.9, 0.9, 0.9]);

        let rows = db
            .query("SELECT id, name FROM embeddings ORDER BY vec <-> '[0.1, 0.1, 0.1, 0.1]' LIMIT 2")
            .expect("k-NN query should succeed");

        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0].values[0], OwnedValue::Int(1));
        assert_eq!(rows[1].values[0], OwnedValue::Int(2));
    }

    #[test]
    fn knn_search_after_delete_excludes_deleted() {
        let (db, _temp_dir) = create_test_db();
        setup_vector_table_with_hnsw_index(&db);

        insert_vector(&db, 1, "near", &[0.1, 0.1, 0.1, 0.1]);
        insert_vector(&db, 2, "mid", &[0.5, 0.5, 0.5, 0.5]);
        insert_vector(&db, 3, "far", &[0.9, 0.9, 0.9, 0.9]);

        db.execute("DELETE FROM embeddings WHERE id = 1").unwrap();

        let rows = db
            .query("SELECT id FROM embeddings ORDER BY vec <-> '[0.1, 0.1, 0.1, 0.1]' LIMIT 2")
            .expect("k-NN query should succeed");

        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0].values[0], OwnedValue::Int(2));
        assert_eq!(rows[1].values[0], OwnedValue::Int(3));
    }

    #[test]
    fn insert_many_vectors_and_search() {
        let (db, _temp_dir) = create_test_db();
        setup_vector_table_with_hnsw_index(&db);

        for i in 0..20 {
            let f = (i as f32) / 20.0;
            insert_vector(&db, i, &format!("vec{}", i), &[f, f, f, f]);
        }

        let rows = db
            .query("SELECT id FROM embeddings ORDER BY vec <-> '[0.5, 0.5, 0.5, 0.5]' LIMIT 3")
            .expect("k-NN query should succeed");

        assert_eq!(rows.len(), 3);
        if let OwnedValue::Int(id) = rows[0].values[0] {
            assert!(id >= 8 && id <= 12, "Nearest should be around id=10, got {}", id);
        }
    }
}
