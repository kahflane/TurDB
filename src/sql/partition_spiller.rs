//! # Partition Spiller for Grace Hash Join
//!
//! This module manages disk-based partitions for Grace Hash Join execution when
//! memory budget is exceeded. It enables large joins to complete without OOM by
//! spilling partition data to temporary files in the partition directory.
//!
//! ## Design Goals
//!
//! 1. **Memory bounded**: Spill to disk when partition exceeds threshold
//! 2. **Zero-allocation reads**: Reuse buffers across partition reads
//! 3. **Automatic cleanup**: Remove spill files on close or Drop
//! 4. **Efficient I/O**: Sequential writes, mmap-based reads
//!
//! ## Memory Budget Model
//!
//! The default 256KB working memory is divided among 16 partitions, giving
//! each partition a 16KB threshold. When a partition exceeds this threshold,
//! its in-memory rows are flushed to disk.
//!
//! ```text
//! Total Budget: 256KB
//! Partitions: 16
//! Per-Partition Budget: 256KB / 16 = 16KB
//! ```
//!
//! ## File Format
//!
//! Spill files contain serialized rows in the format defined by row_serde:
//!
//! ```text
//! SpillFile := [Header] [Row]*
//! Header := [row_count: u64] [data_size: u64]
//! Row := [serialized row bytes from RowSerde]
//! ```
//!
//! ## File Naming
//!
//! Spill files are created in the partition directory with naming:
//!
//! ```text
//! {db_dir}/partition/{query_id}_{side}_{partition_id}.spill
//!
//! Examples:
//!   partition/12345_L_0.spill  (left side, partition 0)
//!   partition/12345_R_3.spill  (right side, partition 3)
//! ```
//!
//! ## Zero-Allocation Strategy
//!
//! The spiller maintains internal buffers that are reused across operations:
//!
//! - `serialize_buf`: Reused for row serialization during writes
//! - `deserialize_buf`: Reused for row deserialization during reads
//! - `read_buf`: Pre-allocated SmallVec for decoded values
//!
//! During partition reads, PartitionReader borrows these buffers from the
//! spiller, avoiding per-row allocations.
//!
//! ## Lifecycle
//!
//! 1. **Creation**: `PartitionSpiller::new()` initializes empty partitions
//! 2. **Writes**: `write_row()` accumulates rows, auto-spills if threshold exceeded
//! 3. **Reads**: `read_partition()` returns iterator over rows (memory or disk)
//! 4. **Cleanup**: `cleanup()` or `Drop` removes all spill files
//!
//! ## Error Handling
//!
//! - I/O errors during spill propagate as `eyre::Result`
//! - Cleanup errors in Drop are silently ignored (best-effort)
//! - Partial spills are cleaned up on error
//!
//! ## Thread Safety
//!
//! PartitionSpiller is NOT thread-safe. It is designed for single-threaded
//! use within a query executor.
//!
//! ## Usage Example
//!
//! ```ignore
//! use turdb::sql::partition_spiller::PartitionSpiller;
//!
//! let mut spiller = PartitionSpiller::new(
//!     db_dir.join("partition"),
//!     16,        // num_partitions
//!     262144,    // 256KB budget
//!     12345,     // query_id
//!     'L',       // side indicator
//! )?;
//!
//! // Write rows to partitions
//! for row in input_rows {
//!     let partition = hash(&row) % 16;
//!     spiller.write_row(partition, row)?;
//! }
//!
//! // Read back partition
//! while let Some(row) = spiller.read_next(partition_id)? {
//!     process(row);
//! }
//!
//! // Cleanup (also happens on drop)
//! spiller.cleanup()?;
//! ```

use crate::sql::row_serde::RowSerde;
use crate::types::Value;
use eyre::{bail, ensure, Result, WrapErr};
use memmap2::Mmap;
use smallvec::SmallVec;
use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::PathBuf;

const SPILL_HEADER_SIZE: usize = 16;

pub struct PartitionSpiller {
    spill_dir: PathBuf,
    partitions: Vec<SpilledPartition>,
    per_partition_budget: usize,
    serialize_buf: Vec<u8>,
    dir_created: bool,
    current_read_partition: Option<usize>,
    read_mmap: Option<Mmap>,
    read_offset: usize,
    read_row_count: usize,
    read_rows_read: usize,
    read_memory_idx: usize,
    deserialize_buf: SmallVec<[Value<'static>; 16]>,
}

struct SpilledPartition {
    file_path: PathBuf,
    row_count: usize,
    byte_size: usize,
    is_spilled: bool,
    in_memory: Vec<SmallVec<[Value<'static>; 16]>>,
}

impl PartitionSpiller {
    pub fn new(
        spill_dir: PathBuf,
        num_partitions: usize,
        memory_budget: usize,
        query_id: u64,
        side: char,
    ) -> Result<Self> {
        let per_partition_budget = memory_budget / num_partitions;

        let partitions = (0..num_partitions)
            .map(|i| {
                let file_path = spill_dir.join(format!("{}_{}.spill", query_id, side))
                    .with_file_name(format!("{}_{}_{}.spill", query_id, side, i));
                SpilledPartition {
                    file_path,
                    row_count: 0,
                    byte_size: 0,
                    is_spilled: false,
                    in_memory: Vec::new(),
                }
            })
            .collect();

        Ok(Self {
            spill_dir,
            partitions,
            per_partition_budget,
            serialize_buf: Vec::with_capacity(4096),
            dir_created: false,
            current_read_partition: None,
            read_mmap: None,
            read_offset: 0,
            read_row_count: 0,
            read_rows_read: 0,
            read_memory_idx: 0,
            deserialize_buf: SmallVec::new(),
        })
    }

    pub fn write_row(
        &mut self,
        partition_id: usize,
        row: SmallVec<[Value<'static>; 16]>,
    ) -> Result<()> {
        ensure!(
            partition_id < self.partitions.len(),
            "partition_id {} out of bounds (max {})",
            partition_id,
            self.partitions.len() - 1
        );

        let row_size = RowSerde::row_size(&row);
        let is_spilled = self.partitions[partition_id].is_spilled;

        if is_spilled {
            self.serialize_buf.clear();
            RowSerde::serialize_row_into(&row, &mut self.serialize_buf);
            self.append_to_spill_file(partition_id)?;
            let serialized_len = self.serialize_buf.len();
            let partition = &mut self.partitions[partition_id];
            partition.row_count += 1;
            partition.byte_size += serialized_len;
        } else {
            let partition = &mut self.partitions[partition_id];
            partition.byte_size += row_size;
            partition.row_count += 1;
            partition.in_memory.push(row);

            let should_spill = partition.byte_size > self.per_partition_budget;
            if should_spill {
                self.spill_partition(partition_id)?;
            }
        }

        Ok(())
    }

    pub fn should_spill(&self, partition_id: usize) -> bool {
        if partition_id >= self.partitions.len() {
            return false;
        }
        let partition = &self.partitions[partition_id];
        !partition.is_spilled && partition.byte_size > self.per_partition_budget
    }

    pub fn spill_partition(&mut self, partition_id: usize) -> Result<()> {
        ensure!(
            partition_id < self.partitions.len(),
            "partition_id {} out of bounds",
            partition_id
        );

        if self.partitions[partition_id].is_spilled {
            return Ok(());
        }

        self.ensure_spill_dir()?;

        let partition = &mut self.partitions[partition_id];
        let file_path = partition.file_path.clone();

        let file = File::create(&file_path)
            .wrap_err_with(|| format!("failed to create spill file: {:?}", file_path))?;
        let mut writer = BufWriter::new(file);

        let mut header = [0u8; SPILL_HEADER_SIZE];
        writer.write_all(&header)?;

        let mut total_data_size = 0usize;
        let mut row_count = 0usize;

        let rows = std::mem::take(&mut partition.in_memory);
        for row in &rows {
            self.serialize_buf.clear();
            RowSerde::serialize_row_into(row, &mut self.serialize_buf);
            writer.write_all(&self.serialize_buf)?;
            total_data_size += self.serialize_buf.len();
            row_count += 1;
        }

        writer.flush()?;

        let file = writer.into_inner()?;
        file.sync_all()?;

        {
            use std::io::{Seek, SeekFrom};
            let mut file = File::options()
                .write(true)
                .open(&file_path)?;
            file.seek(SeekFrom::Start(0))?;
            header[..8].copy_from_slice(&(row_count as u64).to_le_bytes());
            header[8..16].copy_from_slice(&(total_data_size as u64).to_le_bytes());
            file.write_all(&header)?;
            file.sync_all()?;
        }

        partition.is_spilled = true;
        partition.byte_size = total_data_size;
        partition.row_count = row_count;

        Ok(())
    }

    fn append_to_spill_file(&mut self, partition_id: usize) -> Result<()> {
        let partition = &self.partitions[partition_id];
        let file_path = &partition.file_path;

        let mut file = File::options()
            .append(true)
            .open(file_path)
            .wrap_err_with(|| format!("failed to open spill file for append: {:?}", file_path))?;

        file.write_all(&self.serialize_buf)?;

        {
            use std::io::{Seek, SeekFrom};
            let partition = &self.partitions[partition_id];
            let mut header_file = File::options()
                .write(true)
                .open(&partition.file_path)?;
            header_file.seek(SeekFrom::Start(0))?;
            let new_count = partition.row_count + 1;
            let new_size = partition.byte_size + self.serialize_buf.len();
            let mut header = [0u8; SPILL_HEADER_SIZE];
            header[..8].copy_from_slice(&(new_count as u64).to_le_bytes());
            header[8..16].copy_from_slice(&(new_size as u64).to_le_bytes());
            header_file.write_all(&header)?;
        }

        Ok(())
    }

    fn ensure_spill_dir(&mut self) -> Result<()> {
        if !self.dir_created {
            fs::create_dir_all(&self.spill_dir)
                .wrap_err_with(|| format!("failed to create spill directory: {:?}", self.spill_dir))?;
            self.dir_created = true;
        }
        Ok(())
    }

    pub fn start_read(&mut self, partition_id: usize) -> Result<()> {
        ensure!(
            partition_id < self.partitions.len(),
            "partition_id {} out of bounds",
            partition_id
        );

        let partition = &self.partitions[partition_id];

        if partition.is_spilled {
            let file = File::open(&partition.file_path)
                .wrap_err_with(|| format!("failed to open spill file: {:?}", partition.file_path))?;

            let mmap = unsafe { Mmap::map(&file) }
                .wrap_err("failed to mmap spill file")?;

            ensure!(mmap.len() >= SPILL_HEADER_SIZE, "spill file too small");

            let row_count = u64::from_le_bytes(mmap[..8].try_into().unwrap()) as usize;
            let _data_size = u64::from_le_bytes(mmap[8..16].try_into().unwrap()) as usize;

            self.read_mmap = Some(mmap);
            self.read_offset = SPILL_HEADER_SIZE;
            self.read_row_count = row_count;
            self.read_rows_read = 0;
        } else {
            self.read_memory_idx = 0;
            self.read_row_count = partition.row_count;
            self.read_rows_read = 0;
        }

        self.current_read_partition = Some(partition_id);
        Ok(())
    }

    pub fn read_next(&mut self) -> Result<Option<&[Value<'static>]>> {
        let partition_id = match self.current_read_partition {
            Some(id) => id,
            None => bail!("no partition read in progress, call start_read first"),
        };

        let partition = &self.partitions[partition_id];

        if partition.is_spilled {
            if self.read_rows_read >= self.read_row_count {
                return Ok(None);
            }

            let mmap = self.read_mmap.as_ref().unwrap();
            self.deserialize_buf.clear();

            RowSerde::deserialize_row_into(
                mmap,
                &mut self.read_offset,
                &mut self.deserialize_buf,
            )?;

            self.read_rows_read += 1;
            Ok(Some(&self.deserialize_buf))
        } else {
            if self.read_memory_idx >= partition.in_memory.len() {
                return Ok(None);
            }

            let row = &partition.in_memory[self.read_memory_idx];
            self.read_memory_idx += 1;
            self.read_rows_read += 1;

            self.deserialize_buf.clear();
            self.deserialize_buf.extend(row.iter().map(|v| v.to_owned_static()));
            Ok(Some(&self.deserialize_buf))
        }
    }

    pub fn end_read(&mut self) {
        self.current_read_partition = None;
        self.read_mmap = None;
        self.read_offset = 0;
        self.read_row_count = 0;
        self.read_rows_read = 0;
        self.read_memory_idx = 0;
    }

    pub fn partition_row_count(&self, partition_id: usize) -> usize {
        if partition_id >= self.partitions.len() {
            return 0;
        }
        self.partitions[partition_id].row_count
    }

    pub fn partition_is_spilled(&self, partition_id: usize) -> bool {
        if partition_id >= self.partitions.len() {
            return false;
        }
        self.partitions[partition_id].is_spilled
    }

    pub fn num_partitions(&self) -> usize {
        self.partitions.len()
    }

    pub fn cleanup(&mut self) -> Result<()> {
        for partition in &self.partitions {
            if partition.is_spilled && partition.file_path.exists() {
                fs::remove_file(&partition.file_path)
                    .wrap_err_with(|| {
                        format!("failed to remove spill file: {:?}", partition.file_path)
                    })?;
            }
        }

        if self.dir_created && self.spill_dir.exists() {
            let _ = fs::remove_dir(&self.spill_dir);
        }

        Ok(())
    }
}

impl Drop for PartitionSpiller {
    fn drop(&mut self) {
        let _ = self.cleanup();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::borrow::Cow;
    use tempfile::tempdir;

    #[test]
    fn new_creates_empty_partitions() {
        let dir = tempdir().unwrap();
        let spill_dir = dir.path().join("spill");

        let spiller = PartitionSpiller::new(spill_dir.clone(), 16, 262144, 1, 'L').unwrap();

        assert_eq!(spiller.num_partitions(), 16);
        assert!(!spill_dir.exists());
        for i in 0..16 {
            assert_eq!(spiller.partition_row_count(i), 0);
            assert!(!spiller.partition_is_spilled(i));
        }
    }

    #[test]
    fn write_row_accumulates_in_memory() {
        let dir = tempdir().unwrap();
        let spiller_result = PartitionSpiller::new(
            dir.path().join("spill"),
            16,
            262144,
            2,
            'L',
        );
        let mut spiller = spiller_result.unwrap();

        let row: SmallVec<[Value<'static>; 16]> = smallvec::smallvec![Value::Int(42)];
        spiller.write_row(0, row).unwrap();

        assert_eq!(spiller.partition_row_count(0), 1);
        assert!(!spiller.partition_is_spilled(0));
    }

    #[test]
    fn write_row_triggers_spill_when_threshold_exceeded() {
        let dir = tempdir().unwrap();
        let mut spiller = PartitionSpiller::new(
            dir.path().join("spill"),
            16,
            1024,
            3,
            'L',
        )
        .unwrap();

        for i in 0..20 {
            let row: SmallVec<[Value<'static>; 16]> = smallvec::smallvec![
                Value::Int(i),
                Value::Text(Cow::Owned("x".repeat(50)))
            ];
            spiller.write_row(0, row).unwrap();
        }

        assert!(spiller.partition_is_spilled(0));
        assert_eq!(spiller.partition_row_count(0), 20);
    }

    #[test]
    fn roundtrip_in_memory_partition() {
        let dir = tempdir().unwrap();
        let mut spiller = PartitionSpiller::new(
            dir.path().join("spill"),
            16,
            262144,
            4,
            'L',
        )
        .unwrap();

        let rows: Vec<SmallVec<[Value<'static>; 16]>> = (0..5)
            .map(|i| smallvec::smallvec![Value::Int(i), Value::Text(Cow::Owned(format!("row_{}", i)))])
            .collect();

        for row in &rows {
            spiller.write_row(0, row.clone()).unwrap();
        }

        assert!(!spiller.partition_is_spilled(0));

        spiller.start_read(0).unwrap();
        let mut read_count = 0;
        while let Some(row) = spiller.read_next().unwrap() {
            match &row[0] {
                Value::Int(i) => assert_eq!(*i, read_count as i64),
                _ => panic!("expected Int"),
            }
            read_count += 1;
        }
        spiller.end_read();

        assert_eq!(read_count, 5);
    }

    #[test]
    fn roundtrip_spilled_partition() {
        let dir = tempdir().unwrap();
        let mut spiller = PartitionSpiller::new(
            dir.path().join("spill"),
            16,
            512,
            5,
            'R',
        )
        .unwrap();

        let rows: Vec<SmallVec<[Value<'static>; 16]>> = (0..30)
            .map(|i| smallvec::smallvec![Value::Int(i), Value::Text(Cow::Owned(format!("row_{}", i)))])
            .collect();

        for row in &rows {
            spiller.write_row(0, row.clone()).unwrap();
        }

        assert!(spiller.partition_is_spilled(0));

        spiller.start_read(0).unwrap();
        let mut read_count = 0;
        while let Some(row) = spiller.read_next().unwrap() {
            match &row[0] {
                Value::Int(i) => assert_eq!(*i, read_count as i64),
                _ => panic!("expected Int"),
            }
            read_count += 1;
        }
        spiller.end_read();

        assert_eq!(read_count, 30);
    }

    #[test]
    fn cleanup_removes_spill_files() {
        let dir = tempdir().unwrap();
        let spill_dir = dir.path().join("spill");
        let mut spiller = PartitionSpiller::new(
            spill_dir.clone(),
            16,
            256,
            6,
            'L',
        )
        .unwrap();

        for i in 0..50 {
            let row: SmallVec<[Value<'static>; 16]> = smallvec::smallvec![
                Value::Int(i),
                Value::Text(Cow::Owned("x".repeat(20)))
            ];
            spiller.write_row(0, row).unwrap();
        }

        assert!(spiller.partition_is_spilled(0));
        let file_path = spiller.partitions[0].file_path.clone();
        assert!(file_path.exists());

        spiller.cleanup().unwrap();

        assert!(!file_path.exists());
    }

    #[test]
    fn multiple_partitions_work_independently() {
        let dir = tempdir().unwrap();
        let mut spiller = PartitionSpiller::new(
            dir.path().join("spill"),
            4,
            1024,
            7,
            'L',
        )
        .unwrap();

        for i in 0..40 {
            let partition = (i % 4) as usize;
            let row: SmallVec<[Value<'static>; 16]> = smallvec::smallvec![Value::Int(i)];
            spiller.write_row(partition, row).unwrap();
        }

        for p in 0..4 {
            assert_eq!(spiller.partition_row_count(p), 10);

            spiller.start_read(p).unwrap();
            let mut count = 0;
            while spiller.read_next().unwrap().is_some() {
                count += 1;
            }
            spiller.end_read();
            assert_eq!(count, 10);
        }
    }

    #[test]
    fn drop_cleans_up_files() {
        let dir = tempdir().unwrap();
        let spill_dir = dir.path().join("spill");
        let file_path;

        {
            let mut spiller = PartitionSpiller::new(
                spill_dir.clone(),
                16,
                256,
                8,
                'L',
            )
            .unwrap();

            for i in 0..50 {
                let row: SmallVec<[Value<'static>; 16]> = smallvec::smallvec![
                    Value::Int(i),
                    Value::Text(Cow::Owned("x".repeat(20)))
                ];
                spiller.write_row(0, row).unwrap();
            }

            file_path = spiller.partitions[0].file_path.clone();
            assert!(file_path.exists());
        }

        assert!(!file_path.exists());
    }

    #[test]
    fn empty_partition_reads_correctly() {
        let dir = tempdir().unwrap();
        let mut spiller = PartitionSpiller::new(
            dir.path().join("spill"),
            16,
            262144,
            9,
            'L',
        )
        .unwrap();

        spiller.start_read(5).unwrap();
        let result = spiller.read_next().unwrap();
        assert!(result.is_none());
        spiller.end_read();
    }

    #[test]
    fn all_value_types_roundtrip_through_spill() {
        let dir = tempdir().unwrap();
        let mut spiller = PartitionSpiller::new(
            dir.path().join("spill"),
            16,
            128,
            10,
            'L',
        )
        .unwrap();

        let test_rows: Vec<SmallVec<[Value<'static>; 16]>> = vec![
            smallvec::smallvec![Value::Null],
            smallvec::smallvec![Value::Int(-12345)],
            smallvec::smallvec![Value::Float(3.14159)],
            smallvec::smallvec![Value::Text(Cow::Owned("hello world".to_string()))],
            smallvec::smallvec![Value::Blob(Cow::Owned(vec![1, 2, 3, 4, 5]))],
            smallvec::smallvec![Value::Uuid([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16])],
            smallvec::smallvec![Value::MacAddr([0xAA, 0xBB, 0xCC, 0xDD, 0xEE, 0xFF])],
            smallvec::smallvec![Value::Inet4([192, 168, 1, 1])],
            smallvec::smallvec![Value::TimestampTz { micros: 1234567890, offset_secs: -28800 }],
            smallvec::smallvec![Value::Interval { micros: 86400000000, days: 30, months: 12 }],
            smallvec::smallvec![Value::Point { x: 1.5, y: -2.5 }],
            smallvec::smallvec![Value::Enum { type_id: 42, ordinal: 7 }],
            smallvec::smallvec![Value::Decimal { digits: 123456789, scale: 4 }],
        ];

        for row in &test_rows {
            spiller.write_row(0, row.clone()).unwrap();
        }

        assert!(spiller.partition_is_spilled(0));

        spiller.start_read(0).unwrap();
        let mut idx = 0;
        while let Some(row) = spiller.read_next().unwrap() {
            assert_eq!(row.len(), 1);
            match (&row[0], &test_rows[idx][0]) {
                (Value::Null, Value::Null) => {}
                (Value::Int(a), Value::Int(b)) => assert_eq!(a, b),
                (Value::Float(a), Value::Float(b)) => assert!((a - b).abs() < f64::EPSILON),
                (Value::Text(a), Value::Text(b)) => assert_eq!(a.as_ref(), b.as_ref()),
                (Value::Blob(a), Value::Blob(b)) => assert_eq!(a.as_ref(), b.as_ref()),
                (Value::Uuid(a), Value::Uuid(b)) => assert_eq!(a, b),
                (Value::MacAddr(a), Value::MacAddr(b)) => assert_eq!(a, b),
                (Value::Inet4(a), Value::Inet4(b)) => assert_eq!(a, b),
                (Value::TimestampTz { micros: am, offset_secs: ao }, Value::TimestampTz { micros: bm, offset_secs: bo }) => {
                    assert_eq!(am, bm);
                    assert_eq!(ao, bo);
                }
                (Value::Interval { micros: am, days: ad, months: amo }, Value::Interval { micros: bm, days: bd, months: bmo }) => {
                    assert_eq!(am, bm);
                    assert_eq!(ad, bd);
                    assert_eq!(amo, bmo);
                }
                (Value::Point { x: ax, y: ay }, Value::Point { x: bx, y: by }) => {
                    assert!((ax - bx).abs() < f64::EPSILON);
                    assert!((ay - by).abs() < f64::EPSILON);
                }
                (Value::Enum { type_id: at, ordinal: ao }, Value::Enum { type_id: bt, ordinal: bo }) => {
                    assert_eq!(at, bt);
                    assert_eq!(ao, bo);
                }
                (Value::Decimal { digits: ad, scale: as_ }, Value::Decimal { digits: bd, scale: bs_ }) => {
                    assert_eq!(ad, bd);
                    assert_eq!(as_, bs_);
                }
                _ => panic!("type mismatch at idx {}", idx),
            }
            idx += 1;
        }
        spiller.end_read();

        assert_eq!(idx, test_rows.len());
    }
}
