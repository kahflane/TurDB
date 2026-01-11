//! # Spillable Buffer for Subquery Results
//!
//! Provides memory-bounded storage for subquery result sets that can spill to
//! disk when the memory budget is exceeded.
//!
//! ## Memory Budget Integration
//!
//! SpillableBuffer respects the 256KB query pool budget:
//! - Tracks memory usage as rows are added
//! - Triggers spill when threshold exceeded
//! - Uses temp files in the database directory
//!
//! ## Spill Strategy
//!
//! When memory limit is hit:
//! 1. Create temp file in database directory
//! 2. Serialize all buffered rows to disk
//! 3. Continue accepting rows, writing directly to disk
//! 4. On iteration, read from disk with buffered I/O
//!
//! ## Row Serialization
//!
//! Rows are serialized using a simple length-prefixed format:
//! ```text
//! [4 bytes: row length][row data][4 bytes: row length][row data]...
//! ```
//!
//! ## Usage
//!
//! ```ignore
//! let mut buffer = SpillableBuffer::new(256 * 1024); // 256KB limit
//!
//! for row in subquery_results {
//!     buffer.push(row)?;
//! }
//!
//! for row in buffer.iter()? {
//!     let row = row?;
//!     // process row
//! }
//! ```

use crate::types::OwnedValue;
use eyre::{Result, WrapErr};
use std::fs::{File, OpenOptions};
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::PathBuf;

#[derive(Debug, Clone, PartialEq)]
pub struct MaterializedRow {
    pub values: Vec<OwnedValue>,
}

impl MaterializedRow {
    pub fn new(values: Vec<OwnedValue>) -> Self {
        Self { values }
    }

    fn estimated_size(&self) -> usize {
        let base_size = std::mem::size_of::<Self>();
        let values_size: usize = self
            .values
            .iter()
            .map(|v| match v {
                OwnedValue::Null => 1,
                OwnedValue::Bool(_) => 2,
                OwnedValue::Int(_) => 9,
                OwnedValue::Float(_) => 9,
                OwnedValue::Text(s) => s.len() + 5,
                OwnedValue::Blob(b) => b.len() + 5,
                OwnedValue::Vector(v) => v.len() * 4 + 5,
                OwnedValue::Date(_) => 5,
                OwnedValue::Time(_) => 9,
                OwnedValue::Timestamp(_) => 9,
                OwnedValue::TimestampTz(_, _) => 13,
                OwnedValue::Uuid(_) => 17,
                OwnedValue::MacAddr(_) => 7,
                OwnedValue::Inet4(_) => 5,
                OwnedValue::Inet6(_) => 17,
                OwnedValue::Interval(_, _, _) => 17,
                OwnedValue::Point(_, _) => 17,
                OwnedValue::Box(_, _) => 33,
                OwnedValue::Circle(_, _) => 25,
                OwnedValue::Jsonb(b) => b.len() + 5,
                OwnedValue::Decimal(_, _) => 19,
                OwnedValue::Enum(_, _) => 5,
                OwnedValue::ToastPointer(b) => b.len() + 5,
            })
            .sum();
        base_size + values_size
    }

    fn serialize(&self, writer: &mut impl Write) -> Result<()> {
        let value_count = self.values.len() as u32;
        writer.write_all(&value_count.to_le_bytes())?;

        for value in &self.values {
            Self::serialize_value(value, writer)?;
        }
        Ok(())
    }

    fn serialize_value(value: &OwnedValue, writer: &mut impl Write) -> Result<()> {
        match value {
            OwnedValue::Null => writer.write_all(&[0u8])?,
            OwnedValue::Bool(b) => {
                writer.write_all(&[1u8])?;
                writer.write_all(&[if *b { 1u8 } else { 0u8 }])?;
            }
            OwnedValue::Int(i) => {
                writer.write_all(&[2u8])?;
                writer.write_all(&i.to_le_bytes())?;
            }
            OwnedValue::Float(f) => {
                writer.write_all(&[3u8])?;
                writer.write_all(&f.to_le_bytes())?;
            }
            OwnedValue::Text(s) => {
                writer.write_all(&[4u8])?;
                let len = s.len() as u32;
                writer.write_all(&len.to_le_bytes())?;
                writer.write_all(s.as_bytes())?;
            }
            OwnedValue::Blob(b) => {
                writer.write_all(&[5u8])?;
                let len = b.len() as u32;
                writer.write_all(&len.to_le_bytes())?;
                writer.write_all(b)?;
            }
            OwnedValue::Vector(v) => {
                writer.write_all(&[6u8])?;
                let len = v.len() as u32;
                writer.write_all(&len.to_le_bytes())?;
                for f in v {
                    writer.write_all(&f.to_le_bytes())?;
                }
            }
            OwnedValue::Date(d) => {
                writer.write_all(&[7u8])?;
                writer.write_all(&d.to_le_bytes())?;
            }
            OwnedValue::Time(t) => {
                writer.write_all(&[8u8])?;
                writer.write_all(&t.to_le_bytes())?;
            }
            OwnedValue::Timestamp(ts) => {
                writer.write_all(&[9u8])?;
                writer.write_all(&ts.to_le_bytes())?;
            }
            OwnedValue::TimestampTz(micros, offset) => {
                writer.write_all(&[10u8])?;
                writer.write_all(&micros.to_le_bytes())?;
                writer.write_all(&offset.to_le_bytes())?;
            }
            OwnedValue::Uuid(u) => {
                writer.write_all(&[11u8])?;
                writer.write_all(u)?;
            }
            OwnedValue::MacAddr(m) => {
                writer.write_all(&[12u8])?;
                writer.write_all(m)?;
            }
            OwnedValue::Inet4(ip) => {
                writer.write_all(&[13u8])?;
                writer.write_all(ip)?;
            }
            OwnedValue::Inet6(ip) => {
                writer.write_all(&[14u8])?;
                writer.write_all(ip)?;
            }
            OwnedValue::Interval(micros, days, months) => {
                writer.write_all(&[15u8])?;
                writer.write_all(&micros.to_le_bytes())?;
                writer.write_all(&days.to_le_bytes())?;
                writer.write_all(&months.to_le_bytes())?;
            }
            OwnedValue::Point(x, y) => {
                writer.write_all(&[16u8])?;
                writer.write_all(&x.to_le_bytes())?;
                writer.write_all(&y.to_le_bytes())?;
            }
            OwnedValue::Box((lx, ly), (hx, hy)) => {
                writer.write_all(&[17u8])?;
                writer.write_all(&lx.to_le_bytes())?;
                writer.write_all(&ly.to_le_bytes())?;
                writer.write_all(&hx.to_le_bytes())?;
                writer.write_all(&hy.to_le_bytes())?;
            }
            OwnedValue::Circle((cx, cy), r) => {
                writer.write_all(&[18u8])?;
                writer.write_all(&cx.to_le_bytes())?;
                writer.write_all(&cy.to_le_bytes())?;
                writer.write_all(&r.to_le_bytes())?;
            }
            OwnedValue::Jsonb(b) => {
                writer.write_all(&[19u8])?;
                let len = b.len() as u32;
                writer.write_all(&len.to_le_bytes())?;
                writer.write_all(b)?;
            }
            OwnedValue::Decimal(digits, scale) => {
                writer.write_all(&[20u8])?;
                writer.write_all(&digits.to_le_bytes())?;
                writer.write_all(&scale.to_le_bytes())?;
            }
            OwnedValue::Enum(type_id, ordinal) => {
                writer.write_all(&[21u8])?;
                writer.write_all(&type_id.to_le_bytes())?;
                writer.write_all(&ordinal.to_le_bytes())?;
            }
            OwnedValue::ToastPointer(b) => {
                writer.write_all(&[22u8])?;
                let len = b.len() as u32;
                writer.write_all(&len.to_le_bytes())?;
                writer.write_all(b)?;
            }
        }
        Ok(())
    }

    fn deserialize(reader: &mut impl Read) -> Result<Self> {
        let mut count_buf = [0u8; 4];
        reader.read_exact(&mut count_buf)?;
        let value_count = u32::from_le_bytes(count_buf) as usize;

        let mut values = Vec::with_capacity(value_count);
        for _ in 0..value_count {
            values.push(Self::deserialize_value(reader)?);
        }

        Ok(Self { values })
    }

    fn deserialize_value(reader: &mut impl Read) -> Result<OwnedValue> {
        let mut type_buf = [0u8; 1];
        reader.read_exact(&mut type_buf)?;

        match type_buf[0] {
            0 => Ok(OwnedValue::Null),
            1 => {
                let mut bool_buf = [0u8; 1];
                reader.read_exact(&mut bool_buf)?;
                Ok(OwnedValue::Bool(bool_buf[0] != 0))
            }
            2 => {
                let mut int_buf = [0u8; 8];
                reader.read_exact(&mut int_buf)?;
                Ok(OwnedValue::Int(i64::from_le_bytes(int_buf)))
            }
            3 => {
                let mut float_buf = [0u8; 8];
                reader.read_exact(&mut float_buf)?;
                Ok(OwnedValue::Float(f64::from_le_bytes(float_buf)))
            }
            4 => {
                let mut len_buf = [0u8; 4];
                reader.read_exact(&mut len_buf)?;
                let len = u32::from_le_bytes(len_buf) as usize;
                let mut string_buf = vec![0u8; len];
                reader.read_exact(&mut string_buf)?;
                Ok(OwnedValue::Text(
                    String::from_utf8(string_buf).wrap_err("invalid UTF-8 in spilled text")?,
                ))
            }
            5 => {
                let mut len_buf = [0u8; 4];
                reader.read_exact(&mut len_buf)?;
                let len = u32::from_le_bytes(len_buf) as usize;
                let mut blob_buf = vec![0u8; len];
                reader.read_exact(&mut blob_buf)?;
                Ok(OwnedValue::Blob(blob_buf))
            }
            6 => {
                let mut len_buf = [0u8; 4];
                reader.read_exact(&mut len_buf)?;
                let len = u32::from_le_bytes(len_buf) as usize;
                let mut vector = Vec::with_capacity(len);
                for _ in 0..len {
                    let mut float_buf = [0u8; 4];
                    reader.read_exact(&mut float_buf)?;
                    vector.push(f32::from_le_bytes(float_buf));
                }
                Ok(OwnedValue::Vector(vector))
            }
            7 => {
                let mut buf = [0u8; 4];
                reader.read_exact(&mut buf)?;
                Ok(OwnedValue::Date(i32::from_le_bytes(buf)))
            }
            8 => {
                let mut buf = [0u8; 8];
                reader.read_exact(&mut buf)?;
                Ok(OwnedValue::Time(i64::from_le_bytes(buf)))
            }
            9 => {
                let mut buf = [0u8; 8];
                reader.read_exact(&mut buf)?;
                Ok(OwnedValue::Timestamp(i64::from_le_bytes(buf)))
            }
            10 => {
                let mut micros_buf = [0u8; 8];
                let mut offset_buf = [0u8; 4];
                reader.read_exact(&mut micros_buf)?;
                reader.read_exact(&mut offset_buf)?;
                Ok(OwnedValue::TimestampTz(
                    i64::from_le_bytes(micros_buf),
                    i32::from_le_bytes(offset_buf),
                ))
            }
            11 => {
                let mut buf = [0u8; 16];
                reader.read_exact(&mut buf)?;
                Ok(OwnedValue::Uuid(buf))
            }
            12 => {
                let mut buf = [0u8; 6];
                reader.read_exact(&mut buf)?;
                Ok(OwnedValue::MacAddr(buf))
            }
            13 => {
                let mut buf = [0u8; 4];
                reader.read_exact(&mut buf)?;
                Ok(OwnedValue::Inet4(buf))
            }
            14 => {
                let mut buf = [0u8; 16];
                reader.read_exact(&mut buf)?;
                Ok(OwnedValue::Inet6(buf))
            }
            15 => {
                let mut micros_buf = [0u8; 8];
                let mut days_buf = [0u8; 4];
                let mut months_buf = [0u8; 4];
                reader.read_exact(&mut micros_buf)?;
                reader.read_exact(&mut days_buf)?;
                reader.read_exact(&mut months_buf)?;
                Ok(OwnedValue::Interval(
                    i64::from_le_bytes(micros_buf),
                    i32::from_le_bytes(days_buf),
                    i32::from_le_bytes(months_buf),
                ))
            }
            16 => {
                let mut x_buf = [0u8; 8];
                let mut y_buf = [0u8; 8];
                reader.read_exact(&mut x_buf)?;
                reader.read_exact(&mut y_buf)?;
                Ok(OwnedValue::Point(
                    f64::from_le_bytes(x_buf),
                    f64::from_le_bytes(y_buf),
                ))
            }
            17 => {
                let mut bufs = [[0u8; 8]; 4];
                for buf in &mut bufs {
                    reader.read_exact(buf)?;
                }
                Ok(OwnedValue::Box(
                    (f64::from_le_bytes(bufs[0]), f64::from_le_bytes(bufs[1])),
                    (f64::from_le_bytes(bufs[2]), f64::from_le_bytes(bufs[3])),
                ))
            }
            18 => {
                let mut cx_buf = [0u8; 8];
                let mut cy_buf = [0u8; 8];
                let mut r_buf = [0u8; 8];
                reader.read_exact(&mut cx_buf)?;
                reader.read_exact(&mut cy_buf)?;
                reader.read_exact(&mut r_buf)?;
                Ok(OwnedValue::Circle(
                    (f64::from_le_bytes(cx_buf), f64::from_le_bytes(cy_buf)),
                    f64::from_le_bytes(r_buf),
                ))
            }
            19 => {
                let mut len_buf = [0u8; 4];
                reader.read_exact(&mut len_buf)?;
                let len = u32::from_le_bytes(len_buf) as usize;
                let mut buf = vec![0u8; len];
                reader.read_exact(&mut buf)?;
                Ok(OwnedValue::Jsonb(buf))
            }
            20 => {
                let mut digits_buf = [0u8; 16];
                let mut scale_buf = [0u8; 2];
                reader.read_exact(&mut digits_buf)?;
                reader.read_exact(&mut scale_buf)?;
                Ok(OwnedValue::Decimal(
                    i128::from_le_bytes(digits_buf),
                    i16::from_le_bytes(scale_buf),
                ))
            }
            21 => {
                let mut type_id_buf = [0u8; 2];
                let mut ordinal_buf = [0u8; 2];
                reader.read_exact(&mut type_id_buf)?;
                reader.read_exact(&mut ordinal_buf)?;
                Ok(OwnedValue::Enum(
                    u16::from_le_bytes(type_id_buf),
                    u16::from_le_bytes(ordinal_buf),
                ))
            }
            22 => {
                let mut len_buf = [0u8; 4];
                reader.read_exact(&mut len_buf)?;
                let len = u32::from_le_bytes(len_buf) as usize;
                let mut buf = vec![0u8; len];
                reader.read_exact(&mut buf)?;
                Ok(OwnedValue::ToastPointer(buf))
            }
            t => eyre::bail!("unknown value type tag: {}", t),
        }
    }
}

pub struct SpillableBuffer {
    memory_buffer: Vec<MaterializedRow>,
    spill_file: Option<SpillFile>,
    memory_limit: usize,
    current_memory: usize,
    spilled: bool,
    row_count: usize,
}

struct SpillFile {
    path: PathBuf,
    writer: Option<BufWriter<File>>,
}

impl SpillFile {
    fn new(base_dir: &std::path::Path) -> Result<Self> {
        let path = base_dir.join(format!("subquery_spill_{}.tmp", uuid_simple()));
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(true)
            .open(&path)
            .wrap_err_with(|| format!("failed to create spill file: {:?}", path))?;

        Ok(Self {
            path,
            writer: Some(BufWriter::new(file)),
        })
    }

    fn write_row(&mut self, row: &MaterializedRow) -> Result<()> {
        if let Some(writer) = &mut self.writer {
            row.serialize(writer)?;
        }
        Ok(())
    }

    fn flush(&mut self) -> Result<()> {
        if let Some(writer) = &mut self.writer {
            writer.flush()?;
        }
        Ok(())
    }

    fn into_reader(mut self) -> Result<BufReader<File>> {
        self.flush()?;
        self.writer = None;

        let file = File::open(&self.path)
            .wrap_err_with(|| format!("failed to open spill file for reading: {:?}", self.path))?;
        Ok(BufReader::new(file))
    }
}

impl Drop for SpillFile {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.path);
    }
}

fn uuid_simple() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let duration = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default();
    format!(
        "{:x}{:x}",
        duration.as_secs(),
        duration.subsec_nanos() ^ std::process::id()
    )
}

impl SpillableBuffer {
    pub fn new(memory_limit: usize) -> Self {
        Self {
            memory_buffer: Vec::new(),
            spill_file: None,
            memory_limit,
            current_memory: 0,
            spilled: false,
            row_count: 0,
        }
    }

    pub fn with_spill_dir(memory_limit: usize, _spill_dir: PathBuf) -> Self {
        Self::new(memory_limit)
    }

    pub fn push(&mut self, row: MaterializedRow) -> Result<()> {
        let row_size = row.estimated_size();
        self.row_count += 1;

        if self.spilled {
            if let Some(spill) = &mut self.spill_file {
                spill.write_row(&row)?;
            }
            return Ok(());
        }

        if self.current_memory + row_size > self.memory_limit {
            self.spill_to_disk()?;
            if let Some(spill) = &mut self.spill_file {
                spill.write_row(&row)?;
            }
        } else {
            self.current_memory += row_size;
            self.memory_buffer.push(row);
        }

        Ok(())
    }

    fn spill_to_disk(&mut self) -> Result<()> {
        let spill_dir = std::env::temp_dir();
        let mut spill_file = SpillFile::new(&spill_dir)?;

        for row in self.memory_buffer.drain(..) {
            spill_file.write_row(&row)?;
        }

        self.spill_file = Some(spill_file);
        self.spilled = true;
        self.current_memory = 0;

        Ok(())
    }

    pub fn is_spilled(&self) -> bool {
        self.spilled
    }

    pub fn row_count(&self) -> usize {
        self.row_count
    }

    pub fn memory_usage(&self) -> usize {
        self.current_memory
    }

    pub fn iter(&mut self) -> Result<SpillableBufferIter> {
        if self.spilled {
            if let Some(spill) = self.spill_file.take() {
                let reader = spill.into_reader()?;
                return Ok(SpillableBufferIter::Disk {
                    reader,
                    remaining: self.row_count,
                });
            }
        }

        Ok(SpillableBufferIter::Memory {
            rows: std::mem::take(&mut self.memory_buffer).into_iter(),
        })
    }

    pub fn into_vec(mut self) -> Result<Vec<MaterializedRow>> {
        if self.spilled {
            let mut rows = Vec::with_capacity(self.row_count);
            for row in self.iter()? {
                rows.push(row?);
            }
            Ok(rows)
        } else {
            Ok(self.memory_buffer)
        }
    }
}

pub enum SpillableBufferIter {
    Memory {
        rows: std::vec::IntoIter<MaterializedRow>,
    },
    Disk {
        reader: BufReader<File>,
        remaining: usize,
    },
}

impl Iterator for SpillableBufferIter {
    type Item = Result<MaterializedRow>;

    fn next(&mut self) -> Option<Self::Item> {
        match self {
            SpillableBufferIter::Memory { rows } => rows.next().map(Ok),
            SpillableBufferIter::Disk { reader, remaining } => {
                if *remaining == 0 {
                    return None;
                }
                *remaining -= 1;
                Some(MaterializedRow::deserialize(reader))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_spillable_buffer_in_memory() {
        let mut buffer = SpillableBuffer::new(1024 * 1024);

        buffer
            .push(MaterializedRow::new(vec![OwnedValue::Int(1)]))
            .unwrap();
        buffer
            .push(MaterializedRow::new(vec![OwnedValue::Int(2)]))
            .unwrap();
        buffer
            .push(MaterializedRow::new(vec![OwnedValue::Int(3)]))
            .unwrap();

        assert!(!buffer.is_spilled());
        assert_eq!(buffer.row_count(), 3);

        let rows: Vec<_> = buffer.iter().unwrap().map(|r| r.unwrap()).collect();
        assert_eq!(rows.len(), 3);
        assert_eq!(rows[0].values[0], OwnedValue::Int(1));
    }

    #[test]
    fn test_spillable_buffer_spills() {
        let mut buffer = SpillableBuffer::new(100);

        for i in 0..100 {
            buffer
                .push(MaterializedRow::new(vec![
                    OwnedValue::Int(i),
                    OwnedValue::Text(format!("row_{}", i)),
                ]))
                .unwrap();
        }

        assert!(buffer.is_spilled());
        assert_eq!(buffer.row_count(), 100);

        let rows: Vec<_> = buffer.iter().unwrap().map(|r| r.unwrap()).collect();
        assert_eq!(rows.len(), 100);
        assert_eq!(rows[0].values[0], OwnedValue::Int(0));
        assert_eq!(rows[99].values[0], OwnedValue::Int(99));
    }

    #[test]
    fn test_materialized_row_serialize_roundtrip() {
        let row = MaterializedRow::new(vec![
            OwnedValue::Null,
            OwnedValue::Bool(true),
            OwnedValue::Int(42),
            OwnedValue::Float(3.14),
            OwnedValue::Text("hello".to_string()),
            OwnedValue::Blob(vec![1, 2, 3]),
            OwnedValue::Vector(vec![1.0, 2.0, 3.0]),
        ]);

        let mut buf = Vec::new();
        row.serialize(&mut buf).unwrap();

        let mut cursor = std::io::Cursor::new(buf);
        let deserialized = MaterializedRow::deserialize(&mut cursor).unwrap();

        assert_eq!(row, deserialized);
    }
}
