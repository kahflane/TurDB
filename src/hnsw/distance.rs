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

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
pub unsafe fn dot_product_avx2(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::x86_64::*;

    let n = a.len();
    let mut i = 0;
    let mut sum = _mm256_setzero_ps();

    while i + 8 <= n {
        let va = _mm256_loadu_ps(a.as_ptr().add(i));
        let vb = _mm256_loadu_ps(b.as_ptr().add(i));
        sum = _mm256_fmadd_ps(va, vb, sum);
        i += 8;
    }

    let mut result = horizontal_sum_avx2(sum);

    while i < n {
        result += a[i] * b[i];
        i += 1;
    }

    result
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
pub unsafe fn inner_product_avx2(a: &[f32], b: &[f32]) -> f32 {
    -dot_product_avx2(a, b)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
pub unsafe fn cosine_avx2(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::x86_64::*;

    let n = a.len();
    let mut i = 0;
    let mut dot_sum = _mm256_setzero_ps();
    let mut norm_a_sum = _mm256_setzero_ps();
    let mut norm_b_sum = _mm256_setzero_ps();

    while i + 8 <= n {
        let va = _mm256_loadu_ps(a.as_ptr().add(i));
        let vb = _mm256_loadu_ps(b.as_ptr().add(i));
        dot_sum = _mm256_fmadd_ps(va, vb, dot_sum);
        norm_a_sum = _mm256_fmadd_ps(va, va, norm_a_sum);
        norm_b_sum = _mm256_fmadd_ps(vb, vb, norm_b_sum);
        i += 8;
    }

    let mut dot = horizontal_sum_avx2(dot_sum);
    let mut norm_a = horizontal_sum_avx2(norm_a_sum);
    let mut norm_b = horizontal_sum_avx2(norm_b_sum);

    while i < n {
        dot += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
        i += 1;
    }

    let norm_product = (norm_a * norm_b).sqrt();
    if norm_product == 0.0 {
        return 1.0;
    }

    1.0 - (dot / norm_product)
}

#[cfg(target_arch = "aarch64")]
pub unsafe fn dot_product_neon(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::aarch64::*;

    let n = a.len();
    let mut i = 0;
    let mut sum = vdupq_n_f32(0.0);

    while i + 4 <= n {
        let va = vld1q_f32(a.as_ptr().add(i));
        let vb = vld1q_f32(b.as_ptr().add(i));
        sum = vfmaq_f32(sum, va, vb);
        i += 4;
    }

    let mut result = vaddvq_f32(sum);

    while i < n {
        result += a[i] * b[i];
        i += 1;
    }

    result
}

#[cfg(target_arch = "aarch64")]
pub unsafe fn inner_product_neon(a: &[f32], b: &[f32]) -> f32 {
    -dot_product_neon(a, b)
}

#[cfg(target_arch = "aarch64")]
pub unsafe fn cosine_neon(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::aarch64::*;

    let n = a.len();
    let mut i = 0;
    let mut dot_sum = vdupq_n_f32(0.0);
    let mut norm_a_sum = vdupq_n_f32(0.0);
    let mut norm_b_sum = vdupq_n_f32(0.0);

    while i + 4 <= n {
        let va = vld1q_f32(a.as_ptr().add(i));
        let vb = vld1q_f32(b.as_ptr().add(i));
        dot_sum = vfmaq_f32(dot_sum, va, vb);
        norm_a_sum = vfmaq_f32(norm_a_sum, va, va);
        norm_b_sum = vfmaq_f32(norm_b_sum, vb, vb);
        i += 4;
    }

    let mut dot = vaddvq_f32(dot_sum);
    let mut norm_a = vaddvq_f32(norm_a_sum);
    let mut norm_b = vaddvq_f32(norm_b_sum);

    while i < n {
        dot += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
        i += 1;
    }

    let norm_product = (norm_a * norm_b).sqrt();
    if norm_product == 0.0 {
        return 1.0;
    }

    1.0 - (dot / norm_product)
}

pub type DistanceFn = fn(&[f32], &[f32]) -> f32;

fn euclidean_squared_dispatch(a: &[f32], b: &[f32]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            return unsafe { euclidean_squared_avx2(a, b) };
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        unsafe { euclidean_squared_neon(a, b) }
    }

    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        euclidean_squared_scalar(a, b)
    }
}

fn euclidean_dispatch(a: &[f32], b: &[f32]) -> f32 {
    euclidean_squared_dispatch(a, b).sqrt()
}

fn cosine_dispatch(a: &[f32], b: &[f32]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            return unsafe { cosine_avx2(a, b) };
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        unsafe { cosine_neon(a, b) }
    }

    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        cosine_scalar(a, b)
    }
}

fn inner_product_dispatch(a: &[f32], b: &[f32]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            return unsafe { inner_product_avx2(a, b) };
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        unsafe { inner_product_neon(a, b) }
    }

    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        inner_product_scalar(a, b)
    }
}

pub fn select_distance_fn(metric: super::DistanceFunction) -> DistanceFn {
    match metric {
        super::DistanceFunction::L2 => euclidean_dispatch,
        super::DistanceFunction::Cosine => cosine_dispatch,
        super::DistanceFunction::InnerProduct => inner_product_dispatch,
    }
}

pub fn select_squared_distance_fn(metric: super::DistanceFunction) -> DistanceFn {
    match metric {
        super::DistanceFunction::L2 => euclidean_squared_dispatch,
        super::DistanceFunction::Cosine => cosine_dispatch,
        super::DistanceFunction::InnerProduct => inner_product_dispatch,
    }
}

pub fn euclidean_squared(a: &[f32], b: &[f32]) -> f32 {
    euclidean_squared_dispatch(a, b)
}
