use crate::types::OwnedValue;
use eyre::{bail, Result};

#[derive(Debug, Clone, PartialEq)]
pub struct Row {
    pub values: Vec<OwnedValue>,
}

impl Row {
    pub fn new(values: Vec<OwnedValue>) -> Self {
        Self { values }
    }

    pub fn get(&self, index: usize) -> Option<&OwnedValue> {
        self.values.get(index)
    }

    pub fn get_int(&self, index: usize) -> Result<i64> {
        match self.get(index) {
            Some(OwnedValue::Int(i)) => Ok(*i),
            Some(other) => bail!("expected INT, got {:?}", other),
            None => bail!("column {} out of bounds", index),
        }
    }

    pub fn get_float(&self, index: usize) -> Result<f64> {
        match self.get(index) {
            Some(OwnedValue::Float(f)) => Ok(*f),
            Some(other) => bail!("expected FLOAT, got {:?}", other),
            None => bail!("column {} out of bounds", index),
        }
    }

    pub fn get_text(&self, index: usize) -> Result<&str> {
        match self.get(index) {
            Some(OwnedValue::Text(s)) => Ok(s),
            Some(other) => bail!("expected TEXT, got {:?}", other),
            None => bail!("column {} out of bounds", index),
        }
    }

    pub fn get_blob(&self, index: usize) -> Result<&[u8]> {
        match self.get(index) {
            Some(OwnedValue::Blob(b)) => Ok(b),
            Some(other) => bail!("expected BLOB, got {:?}", other),
            None => bail!("column {} out of bounds", index),
        }
    }

    pub fn is_null(&self, index: usize) -> bool {
        matches!(self.get(index), Some(OwnedValue::Null))
    }

    pub fn column_count(&self) -> usize {
        self.values.len()
    }
}
