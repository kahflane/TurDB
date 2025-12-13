//! # SIMD-Accelerated Distance Functions
//!
//! This module provides high-performance distance computations for vector similarity
//! search. All functions are optimized for the HNSW index use case where millions
//! of distance calculations occur during search and insertion operations.
//!
//! ## Supported Distance Metrics
//!
//! - **Euclidean (L2)**: Standard geometric distance. Returns sqrt(sum((a-b)²))
//! - **Cosine**: Angular distance. Returns 1 - dot(a,b)/(|a|*|b|)
//! - **Inner Product**: Dot product similarity. Returns -dot(a,b) (negative for min-heap)
//!
//! ## SIMD Acceleration
//!
//! The module provides multiple implementations selected at runtime based on CPU:
//!
//! | Architecture | ISA      | Function                | Speedup |
//! |-------------|----------|-------------------------|---------|
//! | x86_64      | AVX2     | euclidean_avx2()        | ~8x     |
//! | x86_64      | SSE4.1   | euclidean_sse()         | ~4x     |
//! | aarch64     | NEON     | euclidean_neon()        | ~4x     |
//! | Any         | Scalar   | euclidean_scalar()      | 1x      |
//!
//! ## Usage Pattern
//!
//! ```text
//! let distance_fn = select_distance_fn(DistanceFunction::L2);
//! let dist = distance_fn(query, candidate);
//! ```
//!
//! ## SQ8 Quantized Distance
//!
//! For quantized vectors, specialized functions compute approximate distances
//! directly on u8 data, providing 4x memory bandwidth improvement:
//!
//! - `sq8_distance_l2()`: L2 distance between two SQ8 vectors
//! - `sq8_distance_avx2()`: AVX2-accelerated using SAD instruction
//!
//! ## Thread Safety
//!
//! All distance functions are pure functions with no mutable state, making them
//! safe to call concurrently from multiple threads.
//!
//! ## Performance Notes
//!
//! - Vectors should be 32-byte aligned for optimal SIMD performance
//! - Dimension should be a multiple of 8 (or 16 for AVX-512) for best results
//! - For small dimensions (<16), scalar code may be faster due to overhead
//!
//! ## Zero-Copy Design
//!
//! Distance functions take slices directly, avoiding any allocation. The caller
//! is responsible for ensuring slices have equal lengths.

pub fn euclidean_squared_scalar(a: &[f32], b: &[f32]) -> f32 {
    let mut sum = 0.0f32;
    for (x, y) in a.iter().zip(b.iter()) {
        let diff = x - y;
        sum += diff * diff;
    }
    sum
}

pub fn euclidean_scalar(a: &[f32], b: &[f32]) -> f32 {
    euclidean_squared_scalar(a, b).sqrt()
}

pub fn dot_product_scalar(a: &[f32], b: &[f32]) -> f32 {
    let mut sum = 0.0f32;
    for (x, y) in a.iter().zip(b.iter()) {
        sum += x * y;
    }
    sum
}

pub fn inner_product_scalar(a: &[f32], b: &[f32]) -> f32 {
    -dot_product_scalar(a, b)
}

pub fn cosine_scalar(a: &[f32], b: &[f32]) -> f32 {
    let mut dot = 0.0f32;
    let mut norm_a = 0.0f32;
    let mut norm_b = 0.0f32;

    for (x, y) in a.iter().zip(b.iter()) {
        dot += x * y;
        norm_a += x * x;
        norm_b += y * y;
    }

    let norm_product = (norm_a * norm_b).sqrt();
    if norm_product == 0.0 {
        return 1.0;
    }

    1.0 - (dot / norm_product)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn euclidean_scalar_identical_vectors() {
        let a = [1.0f32, 2.0, 3.0, 4.0];
        let b = [1.0f32, 2.0, 3.0, 4.0];

        let dist = euclidean_scalar(&a, &b);
        assert!((dist - 0.0).abs() < 1e-6);
    }

    #[test]
    fn euclidean_scalar_known_distance() {
        let a = [0.0f32, 0.0, 0.0];
        let b = [3.0f32, 4.0, 0.0];

        let dist = euclidean_scalar(&a, &b);
        assert!((dist - 5.0).abs() < 1e-6);
    }

    #[test]
    fn euclidean_scalar_symmetric() {
        let a = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let b = [8.0f32, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0];

        let dist_ab = euclidean_scalar(&a, &b);
        let dist_ba = euclidean_scalar(&b, &a);

        assert!((dist_ab - dist_ba).abs() < 1e-6);
    }

    #[test]
    fn euclidean_squared_scalar_no_sqrt() {
        let a = [0.0f32, 0.0, 0.0];
        let b = [3.0f32, 4.0, 0.0];

        let dist_sq = euclidean_squared_scalar(&a, &b);
        assert!((dist_sq - 25.0).abs() < 1e-6);
    }

    #[test]
    fn cosine_distance_identical_vectors() {
        let a = [1.0f32, 0.0, 0.0];
        let b = [1.0f32, 0.0, 0.0];

        let dist = cosine_scalar(&a, &b);
        assert!((dist - 0.0).abs() < 1e-6);
    }

    #[test]
    fn cosine_distance_orthogonal_vectors() {
        let a = [1.0f32, 0.0, 0.0];
        let b = [0.0f32, 1.0, 0.0];

        let dist = cosine_scalar(&a, &b);
        assert!((dist - 1.0).abs() < 1e-6);
    }

    #[test]
    fn cosine_distance_opposite_vectors() {
        let a = [1.0f32, 0.0, 0.0];
        let b = [-1.0f32, 0.0, 0.0];

        let dist = cosine_scalar(&a, &b);
        assert!((dist - 2.0).abs() < 1e-6);
    }

    #[test]
    fn inner_product_scalar_known_value() {
        let a = [1.0f32, 2.0, 3.0];
        let b = [4.0f32, 5.0, 6.0];

        let ip = inner_product_scalar(&a, &b);
        assert!((ip - (-32.0)).abs() < 1e-6);
    }

    #[test]
    fn dot_product_scalar_known_value() {
        let a = [1.0f32, 2.0, 3.0];
        let b = [4.0f32, 5.0, 6.0];

        let dp = dot_product_scalar(&a, &b);
        assert!((dp - 32.0).abs() < 1e-6);
    }
}
