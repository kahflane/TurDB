//! # Schema Definition
//!
//! This module provides the `Schema` struct that defines the structure of a record.
//! The schema pre-computes offsets for efficient O(1) column access.
//!
//! ## Schema Internals
//!
//! - `columns`: Vector of column definitions
//! - `var_column_indices`: Indices of variable-length columns (for offset table)
//! - `fixed_offsets`: Pre-computed byte offsets for each column in fixed data section
//! - `total_fixed_size`: Total size of all fixed-width columns

use crate::records::types::ColumnDef;

#[derive(Debug, Clone)]
pub struct Schema {
    pub(crate) columns: Vec<ColumnDef>,
    pub(crate) var_column_indices: Vec<usize>,
    pub(crate) fixed_offsets: Vec<usize>,
    pub(crate) total_fixed_size: usize,
}

impl Schema {
    pub fn new(columns: Vec<ColumnDef>) -> Self {
        let mut var_column_indices = Vec::new();
        let mut fixed_offsets = Vec::new();
        let mut offset = 0;

        for (idx, col) in columns.iter().enumerate() {
            fixed_offsets.push(offset);
            if let Some(size) = col.data_type.fixed_size() {
                offset += size;
            } else {
                var_column_indices.push(idx);
            }
        }

        Self {
            columns,
            var_column_indices,
            fixed_offsets,
            total_fixed_size: offset,
        }
    }

    pub fn column_count(&self) -> usize {
        self.columns.len()
    }

    pub fn var_column_count(&self) -> usize {
        self.var_column_indices.len()
    }

    pub fn column(&self, idx: usize) -> Option<&ColumnDef> {
        self.columns.get(idx)
    }

    pub fn var_column_index(&self, col_idx: usize) -> Option<usize> {
        self.var_column_indices
            .iter()
            .position(|&idx| idx == col_idx)
    }

    pub fn fixed_offset(&self, col_idx: usize) -> usize {
        self.fixed_offsets[col_idx]
    }

    pub fn total_fixed_size(&self) -> usize {
        self.total_fixed_size
    }

    pub fn null_bitmap_size(column_count: usize) -> usize {
        column_count.div_ceil(8)
    }

    pub fn columns(&self) -> &[ColumnDef] {
        &self.columns
    }
}
