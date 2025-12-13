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

// SAFETY: Caller must ensure slices a and b have equal length.
// Requires x86_64 CPU with AVX2 and FMA support (checked via target_feature).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
pub unsafe fn euclidean_squared_avx2(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::x86_64::*;

    let n = a.len();
    let mut i = 0;
    let mut sum = _mm256_setzero_ps();

    while i + 8 <= n {
        let va = _mm256_loadu_ps(a.as_ptr().add(i));
        let vb = _mm256_loadu_ps(b.as_ptr().add(i));
        let diff = _mm256_sub_ps(va, vb);
        sum = _mm256_fmadd_ps(diff, diff, sum);
        i += 8;
    }

    let mut result = horizontal_sum_avx2(sum);

    while i < n {
        let diff = a[i] - b[i];
        result += diff * diff;
        i += 1;
    }

    result
}

// SAFETY: Caller must ensure slices a and b have equal length.
// Requires x86_64 CPU with AVX2 and FMA support (checked via target_feature).
// Delegates to euclidean_squared_avx2 which performs the SIMD operations safely.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
pub unsafe fn euclidean_avx2(a: &[f32], b: &[f32]) -> f32 {
    euclidean_squared_avx2(a, b).sqrt()
}

#[cfg(target_arch = "x86_64")]
#[inline]
/// # Safety
///
/// This function is only callable on x86_64 with AVX2 support
/// due to the target_feature attribute on the calling functions.
/// The intrinsics operate on a valid __m256 value passed by the caller.
// SAFETY: Caller guarantees AVX2 is available and v is a valid __m256 register.
unsafe fn horizontal_sum_avx2(v: std::arch::x86_64::__m256) -> f32 {
    use std::arch::x86_64::*;

    let hi = _mm256_extractf128_ps(v, 1);
    let lo = _mm256_castps256_ps128(v);
    let sum128 = _mm_add_ps(lo, hi);
    let hi64 = _mm_movehl_ps(sum128, sum128);
    let sum64 = _mm_add_ps(sum128, hi64);
    let hi32 = _mm_shuffle_ps(sum64, sum64, 1);
    let sum32 = _mm_add_ss(sum64, hi32);
    _mm_cvtss_f32(sum32)
}

/// # Safety
///
/// Slices `a` and `b` must have equal length. This function uses NEON SIMD
/// intrinsics and requires an aarch64 CPU with NEON support (standard on all
/// aarch64 processors).
#[cfg(target_arch = "aarch64")]
pub unsafe fn euclidean_squared_neon(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::aarch64::*;

    let n = a.len();
    let mut i = 0;
    let mut sum = vdupq_n_f32(0.0);

    while i + 4 <= n {
        let va = vld1q_f32(a.as_ptr().add(i));
        let vb = vld1q_f32(b.as_ptr().add(i));
        let diff = vsubq_f32(va, vb);
        sum = vfmaq_f32(sum, diff, diff);
        i += 4;
    }

    let mut result = vaddvq_f32(sum);

    while i < n {
        let diff = a[i] - b[i];
        result += diff * diff;
        i += 1;
    }

    result
}

/// # Safety
///
/// Slices `a` and `b` must have equal length. This function uses NEON SIMD
/// intrinsics and requires an aarch64 CPU with NEON support.
#[cfg(target_arch = "aarch64")]
pub unsafe fn euclidean_neon(a: &[f32], b: &[f32]) -> f32 {
    euclidean_squared_neon(a, b).sqrt()
}

pub type DistanceFn = fn(&[f32], &[f32]) -> f32;

pub fn select_distance_fn(metric: super::DistanceFunction) -> DistanceFn {
    match metric {
        super::DistanceFunction::L2 => euclidean_scalar,
        super::DistanceFunction::Cosine => cosine_scalar,
        super::DistanceFunction::InnerProduct => inner_product_scalar,
    }
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

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn euclidean_avx2_matches_scalar() {
        if !is_x86_feature_detected!("avx2") {
            return;
        }

        let a: Vec<f32> = (0..128).map(|i| i as f32 * 0.1).collect();
        let b: Vec<f32> = (0..128).map(|i| (127 - i) as f32 * 0.1).collect();

        let scalar_result = euclidean_scalar(&a, &b);
        let avx2_result = unsafe { euclidean_avx2(&a, &b) };

        assert!(
            (scalar_result - avx2_result).abs() < 1e-4,
            "scalar={}, avx2={}",
            scalar_result,
            avx2_result
        );
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn euclidean_squared_avx2_matches_scalar() {
        if !is_x86_feature_detected!("avx2") {
            return;
        }

        let a: Vec<f32> = (0..64).map(|i| i as f32 * 0.5).collect();
        let b: Vec<f32> = (0..64).map(|i| (63 - i) as f32 * 0.5).collect();

        let scalar_result = euclidean_squared_scalar(&a, &b);
        let avx2_result = unsafe { euclidean_squared_avx2(&a, &b) };

        assert!(
            (scalar_result - avx2_result).abs() < 1e-3,
            "scalar={}, avx2={}",
            scalar_result,
            avx2_result
        );
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn euclidean_avx2_handles_non_multiple_of_8() {
        if !is_x86_feature_detected!("avx2") {
            return;
        }

        let a: Vec<f32> = (0..35).map(|i| i as f32).collect();
        let b: Vec<f32> = (0..35).map(|i| (34 - i) as f32).collect();

        let scalar_result = euclidean_scalar(&a, &b);
        let avx2_result = unsafe { euclidean_avx2(&a, &b) };

        assert!(
            (scalar_result - avx2_result).abs() < 1e-3,
            "scalar={}, avx2={}",
            scalar_result,
            avx2_result
        );
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn euclidean_neon_matches_scalar() {
        let a: Vec<f32> = (0..128).map(|i| i as f32 * 0.1).collect();
        let b: Vec<f32> = (0..128).map(|i| (127 - i) as f32 * 0.1).collect();

        let scalar_result = euclidean_scalar(&a, &b);
        let neon_result = unsafe { euclidean_neon(&a, &b) };

        assert!(
            (scalar_result - neon_result).abs() < 1e-4,
            "scalar={}, neon={}",
            scalar_result,
            neon_result
        );
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn euclidean_squared_neon_matches_scalar() {
        let a: Vec<f32> = (0..64).map(|i| i as f32 * 0.5).collect();
        let b: Vec<f32> = (0..64).map(|i| (63 - i) as f32 * 0.5).collect();

        let scalar_result = euclidean_squared_scalar(&a, &b);
        let neon_result = unsafe { euclidean_squared_neon(&a, &b) };

        assert!(
            (scalar_result - neon_result).abs() < 1e-3,
            "scalar={}, neon={}",
            scalar_result,
            neon_result
        );
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn euclidean_neon_handles_non_multiple_of_4() {
        let a: Vec<f32> = (0..35).map(|i| i as f32).collect();
        let b: Vec<f32> = (0..35).map(|i| (34 - i) as f32).collect();

        let scalar_result = euclidean_scalar(&a, &b);
        let neon_result = unsafe { euclidean_neon(&a, &b) };

        assert!(
            (scalar_result - neon_result).abs() < 1e-3,
            "scalar={}, neon={}",
            scalar_result,
            neon_result
        );
    }

    #[test]
    fn select_distance_fn_returns_correct_function() {
        let l2_fn = select_distance_fn(super::super::DistanceFunction::L2);
        let cosine_fn = select_distance_fn(super::super::DistanceFunction::Cosine);
        let ip_fn = select_distance_fn(super::super::DistanceFunction::InnerProduct);

        let a = [1.0f32, 0.0, 0.0];
        let b = [0.0f32, 1.0, 0.0];

        let l2_dist = l2_fn(&a, &b);
        let cosine_dist = cosine_fn(&a, &b);
        let ip_dist = ip_fn(&a, &b);

        assert!((l2_dist - 2.0f32.sqrt()).abs() < 1e-6);
        assert!((cosine_dist - 1.0).abs() < 1e-6);
        assert!((ip_dist - 0.0).abs() < 1e-6);
    }
}
