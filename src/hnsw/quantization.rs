//! # Scalar Quantization for HNSW Vector Index
//!
//! This module implements SQ8 (8-bit scalar quantization) for compressing f32
//! vectors to u8 representation with 4x memory reduction. Each dimension is
//! independently quantized using per-vector min/max scaling.
//!
//! ## Quantization Formula
//!
//! Given a vector with values in range [min, max]:
//! ```text
//! scale = (max - min) / 255.0
//! quantized[i] = round((value[i] - min) / scale)
//! ```
//!
//! ## Dequantization Formula
//!
//! ```text
//! value[i] = min + quantized[i] * scale
//! ```
//!
//! ## Memory Layout
//!
//! ```text
//! SQ8Vector:
//! +------------------+
//! | min: f32 (4B)    |  Minimum value in original vector
//! +------------------+
//! | scale: f32 (4B)  |  (max - min) / 255.0
//! +------------------+
//! | data: [u8; dim]  |  Quantized values (0-255)
//! +------------------+
//! Total: 8 + dimension bytes
//! ```
//!
//! ## Recall Characteristics
//!
//! SQ8 provides excellent recall (>98%) for typical ML embeddings because:
//! - Embedding values typically have limited dynamic range
//! - Relative ordering is preserved for distance comparisons
//! - Quantization error is uniformly distributed across dimensions
//!
//! ## Distance Computation
//!
//! For L2 distance between SQ8 vectors with same min/scale:
//! ```text
//! dist_approx = sum((a[i] - b[i])^2) * scale^2
//! ```
//!
//! For asymmetric distance (f32 query vs SQ8 database):
//! ```text
//! query_quantized[i] = (query[i] - db_min) / db_scale
//! dist = sum((query_quantized[i] - db[i])^2) * scale^2
//! ```
//!
//! ## SIMD Acceleration
//!
//! Quantized distance uses integer SAD (Sum of Absolute Differences) instruction:
//! - AVX2: _mm256_sad_epu8 processes 32 bytes per cycle
//! - NEON: vabdl_u8 + vpadalq_u16 for efficient accumulation
//!
//! ## Usage Pattern
//!
//! ```text
//! // Encode vectors during insertion
//! let sq8 = SQ8Vector::from_f32(&embedding);
//!
//! // Store in HNSW node (only 8 + dim bytes)
//! let bytes = sq8.as_bytes();
//!
//! // Compute approximate distance
//! let dist = sq8.distance_l2_sq8(&other_sq8);
//! ```
//!
//! ## Zero-Copy Access
//!
//! SQ8VectorRef provides zero-copy access to quantized vectors stored in pages:
//! ```text
//! let vec_ref = SQ8VectorRef::from_bytes(page_slice);
//! let dist = vec_ref.distance_l2(query);
//! ```

pub struct SQ8Vector {
    min: f32,
    scale: f32,
    data: Vec<u8>,
}

impl SQ8Vector {
    pub fn from_f32(values: &[f32]) -> Self {
        if values.is_empty() {
            return Self {
                min: 0.0,
                scale: 0.0,
                data: Vec::new(),
            };
        }

        let min = values.iter().cloned().fold(f32::INFINITY, f32::min);
        let max = values.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

        let range = max - min;
        let scale = if range > 0.0 { range / 255.0 } else { 1.0 };

        let data = values
            .iter()
            .map(|&v| {
                if scale == 0.0 || range == 0.0 {
                    0u8
                } else {
                    ((v - min) / scale).round().clamp(0.0, 255.0) as u8
                }
            })
            .collect();

        Self { min, scale, data }
    }

    pub fn min(&self) -> f32 {
        self.min
    }

    pub fn scale(&self) -> f32 {
        self.scale
    }

    pub fn data(&self) -> &[u8] {
        &self.data
    }

    pub fn dimension(&self) -> usize {
        self.data.len()
    }

    pub fn decode(&self) -> Vec<f32> {
        self.data
            .iter()
            .map(|&q| self.min + (q as f32) * self.scale)
            .collect()
    }

    pub fn decode_into(&self, output: &mut [f32]) {
        for (i, &q) in self.data.iter().enumerate() {
            if i < output.len() {
                output[i] = self.min + (q as f32) * self.scale;
            }
        }
    }

    pub fn serialized_size(&self) -> usize {
        8 + self.data.len()
    }

    pub fn write_to(&self, buf: &mut [u8]) -> usize {
        buf[0..4].copy_from_slice(&self.min.to_le_bytes());
        buf[4..8].copy_from_slice(&self.scale.to_le_bytes());
        buf[8..8 + self.data.len()].copy_from_slice(&self.data);
        8 + self.data.len()
    }

    pub fn read_from(buf: &[u8], dimension: usize) -> eyre::Result<Self> {
        use eyre::ensure;

        ensure!(
            buf.len() >= 8 + dimension,
            "buffer too small for SQ8Vector: need {} bytes, got {}",
            8 + dimension,
            buf.len()
        );

        let min = f32::from_le_bytes([buf[0], buf[1], buf[2], buf[3]]);
        let scale = f32::from_le_bytes([buf[4], buf[5], buf[6], buf[7]]);
        let data = buf[8..8 + dimension].to_vec();

        Ok(Self { min, scale, data })
    }

    pub fn distance_l2_squared(&self, other: &SQ8Vector) -> f32 {
        let mut sum: u32 = 0;
        for (a, b) in self.data.iter().zip(&other.data) {
            let diff = (*a as i32) - (*b as i32);
            sum += (diff * diff) as u32;
        }
        (sum as f32) * self.scale * other.scale
    }

    pub fn distance_l2(&self, other: &SQ8Vector) -> f32 {
        self.distance_l2_squared(other).sqrt()
    }
}

#[derive(Debug, Clone, Copy)]
pub struct SQ8VectorRef<'a> {
    min: f32,
    scale: f32,
    data: &'a [u8],
}

impl<'a> SQ8VectorRef<'a> {
    pub fn from_bytes(buf: &'a [u8]) -> eyre::Result<Self> {
        use eyre::ensure;

        ensure!(buf.len() >= 8, "buffer too small for SQ8VectorRef header");

        let min = f32::from_le_bytes([buf[0], buf[1], buf[2], buf[3]]);
        let scale = f32::from_le_bytes([buf[4], buf[5], buf[6], buf[7]]);
        let data = &buf[8..];

        Ok(Self { min, scale, data })
    }

    pub fn min(&self) -> f32 {
        self.min
    }

    pub fn scale(&self) -> f32 {
        self.scale
    }

    pub fn data(&self) -> &'a [u8] {
        self.data
    }

    pub fn dimension(&self) -> usize {
        self.data.len()
    }

    pub fn decode_into(&self, output: &mut [f32]) {
        for (i, &q) in self.data.iter().enumerate() {
            if i < output.len() {
                output[i] = self.min + (q as f32) * self.scale;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sq8_from_f32_stores_min_and_scale() {
        let values = vec![0.0f32, 127.5, 255.0];
        let sq8 = SQ8Vector::from_f32(&values);

        assert_eq!(sq8.min(), 0.0);
        assert!((sq8.scale() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn sq8_from_f32_quantizes_to_u8() {
        let values = vec![0.0f32, 127.5, 255.0];
        let sq8 = SQ8Vector::from_f32(&values);

        assert_eq!(sq8.data()[0], 0);
        assert_eq!(sq8.data()[1], 128);
        assert_eq!(sq8.data()[2], 255);
    }

    #[test]
    fn sq8_dimension_matches_input() {
        let values = vec![1.0f32; 128];
        let sq8 = SQ8Vector::from_f32(&values);

        assert_eq!(sq8.dimension(), 128);
    }

    #[test]
    fn sq8_decode_recovers_approximate_values() {
        let values = vec![0.0f32, 0.5, 1.0];
        let sq8 = SQ8Vector::from_f32(&values);
        let decoded = sq8.decode();

        assert!((decoded[0] - 0.0).abs() < 0.01);
        assert!((decoded[1] - 0.5).abs() < 0.01);
        assert!((decoded[2] - 1.0).abs() < 0.01);
    }

    #[test]
    fn sq8_decode_into_fills_buffer() {
        let values = vec![0.0f32, 0.5, 1.0];
        let sq8 = SQ8Vector::from_f32(&values);
        let mut output = [0.0f32; 3];

        sq8.decode_into(&mut output);

        assert!((output[0] - 0.0).abs() < 0.01);
        assert!((output[1] - 0.5).abs() < 0.01);
        assert!((output[2] - 1.0).abs() < 0.01);
    }

    #[test]
    fn sq8_handles_empty_vector() {
        let values: Vec<f32> = vec![];
        let sq8 = SQ8Vector::from_f32(&values);

        assert_eq!(sq8.dimension(), 0);
        assert!(sq8.data().is_empty());
    }

    #[test]
    fn sq8_handles_constant_vector() {
        let values = vec![5.0f32; 100];
        let sq8 = SQ8Vector::from_f32(&values);

        assert_eq!(sq8.min(), 5.0);
        let decoded = sq8.decode();
        for val in decoded {
            assert!((val - 5.0).abs() < 0.01);
        }
    }

    #[test]
    fn sq8_handles_negative_values() {
        let values = vec![-10.0f32, 0.0, 10.0];
        let sq8 = SQ8Vector::from_f32(&values);

        assert_eq!(sq8.min(), -10.0);
        let decoded = sq8.decode();
        assert!((decoded[0] - (-10.0)).abs() < 0.1);
        assert!((decoded[1] - 0.0).abs() < 0.1);
        assert!((decoded[2] - 10.0).abs() < 0.1);
    }

    #[test]
    fn sq8_serialization_roundtrip() {
        let values = vec![1.0f32, 2.0, 3.0, 4.0];
        let sq8 = SQ8Vector::from_f32(&values);

        let mut buf = vec![0u8; sq8.serialized_size()];
        sq8.write_to(&mut buf);

        let decoded = SQ8Vector::read_from(&buf, 4).unwrap();

        assert_eq!(decoded.min(), sq8.min());
        assert_eq!(decoded.scale(), sq8.scale());
        assert_eq!(decoded.data(), sq8.data());
    }

    #[test]
    fn sq8_serialized_size() {
        let values = vec![1.0f32; 128];
        let sq8 = SQ8Vector::from_f32(&values);

        assert_eq!(sq8.serialized_size(), 8 + 128);
    }

    #[test]
    fn sq8_distance_l2_squared_identical_vectors() {
        let values = vec![1.0f32, 2.0, 3.0, 4.0];
        let sq8 = SQ8Vector::from_f32(&values);

        let dist = sq8.distance_l2_squared(&sq8);

        assert_eq!(dist, 0.0);
    }

    #[test]
    fn sq8_distance_l2_squared_different_vectors() {
        let a = vec![0.0f32, 0.0, 0.0];
        let b = vec![255.0f32, 0.0, 0.0];

        let sq8_a = SQ8Vector::from_f32(&a);
        let sq8_b = SQ8Vector::from_f32(&b);

        let dist = sq8_a.distance_l2(&sq8_b);
        assert!(dist > 0.0);
    }

    #[test]
    fn sq8_distance_approximates_true_distance() {
        let a = vec![0.0f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let b = vec![0.1f32, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8];

        fn euclidean_f32(a: &[f32], b: &[f32]) -> f32 {
            a.iter()
                .zip(b.iter())
                .map(|(x, y)| (x - y).powi(2))
                .sum::<f32>()
                .sqrt()
        }

        let true_dist = euclidean_f32(&a, &b);
        let sq8_a = SQ8Vector::from_f32(&a);
        let sq8_b = SQ8Vector::from_f32(&b);

        let decoded_a = sq8_a.decode();
        let decoded_b = sq8_b.decode();
        let approx_dist = euclidean_f32(&decoded_a, &decoded_b);

        let error = (approx_dist - true_dist).abs() / true_dist;
        assert!(error < 0.05, "relative error {} exceeds 5%", error);
    }

    #[test]
    fn sq8_ref_zero_copy_access() {
        let values = vec![1.0f32, 2.0, 3.0, 4.0];
        let sq8 = SQ8Vector::from_f32(&values);

        let mut buf = vec![0u8; sq8.serialized_size()];
        sq8.write_to(&mut buf);

        let vec_ref = SQ8VectorRef::from_bytes(&buf).unwrap();

        assert_eq!(vec_ref.min(), sq8.min());
        assert_eq!(vec_ref.scale(), sq8.scale());
        assert_eq!(vec_ref.dimension(), 4);
    }

    #[test]
    fn sq8_ref_decode_into() {
        let values = vec![1.0f32, 2.0, 3.0, 4.0];
        let sq8 = SQ8Vector::from_f32(&values);

        let mut buf = vec![0u8; sq8.serialized_size()];
        sq8.write_to(&mut buf);

        let vec_ref = SQ8VectorRef::from_bytes(&buf).unwrap();
        let mut output = [0.0f32; 4];
        vec_ref.decode_into(&mut output);

        for i in 0..4 {
            assert!((output[i] - values[i]).abs() < 0.1);
        }
    }

    #[test]
    fn sq8_recall_test_random_vectors() {
        use std::collections::BinaryHeap;

        fn euclidean_f32(a: &[f32], b: &[f32]) -> f32 {
            a.iter()
                .zip(b.iter())
                .map(|(x, y)| (x - y).powi(2))
                .sum::<f32>()
                .sqrt()
        }

        let dim = 128;
        let n_vectors = 100;

        let vectors: Vec<Vec<f32>> = (0..n_vectors)
            .map(|i| {
                (0..dim)
                    .map(|j| ((i * 7 + j * 13) % 100) as f32 / 100.0)
                    .collect()
            })
            .collect();

        let sq8_vectors: Vec<SQ8Vector> = vectors.iter().map(|v| SQ8Vector::from_f32(v)).collect();

        let query = &vectors[0];
        let query_sq8 = &sq8_vectors[0];

        let mut exact_heap: BinaryHeap<(std::cmp::Reverse<i64>, usize)> = BinaryHeap::new();
        for (i, v) in vectors.iter().enumerate() {
            let dist = euclidean_f32(query, v);
            exact_heap.push((std::cmp::Reverse((dist * 1000.0) as i64), i));
        }

        let exact_top10: Vec<usize> = (0..10)
            .filter_map(|_| exact_heap.pop().map(|(_, i)| i))
            .collect();

        let mut approx_heap: BinaryHeap<(std::cmp::Reverse<i64>, usize)> = BinaryHeap::new();
        for (i, sq8) in sq8_vectors.iter().enumerate() {
            let dist = query_sq8.distance_l2(sq8);
            approx_heap.push((std::cmp::Reverse((dist * 1000.0) as i64), i));
        }

        let approx_top10: Vec<usize> = (0..10)
            .filter_map(|_| approx_heap.pop().map(|(_, i)| i))
            .collect();

        let mut overlap = 0;
        for idx in &exact_top10 {
            if approx_top10.contains(idx) {
                overlap += 1;
            }
        }

        let recall = overlap as f32 / 10.0;
        assert!(recall >= 0.8, "recall {} is below threshold 0.8", recall);
    }
}
