//! # Internal Macros
//!
//! This module provides internal macros for reducing boilerplate in TurDB.
//!
//! ## zerocopy_accessors!
//!
//! Generates getter and setter methods for zerocopy struct fields that use
//! little-endian wrapper types (U16, U32, U64).
//!
//! ### Usage
//!
//! ```ignore
//! use zerocopy::little_endian::{U32, U64};
//!
//! #[repr(C)]
//! struct Header {
//!     row_count: U64,
//!     page_size: U32,
//! }
//!
//! impl Header {
//!     zerocopy_accessors! {
//!         row_count: u64,
//!         page_size: u32,
//!     }
//! }
//!
//! // Generates:
//! // pub fn row_count(&self) -> u64 { self.row_count.get() }
//! // pub fn set_row_count(&mut self, val: u64) { self.row_count = U64::new(val); }
//! // pub fn page_size(&self) -> u32 { self.page_size.get() }
//! // pub fn set_page_size(&mut self, val: u32) { self.page_size = U32::new(val); }
//! ```

/// Generates getter and setter methods for zerocopy little-endian fields.
#[macro_export]
macro_rules! zerocopy_accessors {
    (@impl $field:ident, u16) => {
        ::paste::paste! {
            #[inline]
            pub fn $field(&self) -> u16 {
                self.$field.get()
            }

            #[inline]
            pub fn [<set_ $field>](&mut self, val: u16) {
                self.$field = ::zerocopy::little_endian::U16::new(val);
            }
        }
    };
    (@impl $field:ident, u32) => {
        ::paste::paste! {
            #[inline]
            pub fn $field(&self) -> u32 {
                self.$field.get()
            }

            #[inline]
            pub fn [<set_ $field>](&mut self, val: u32) {
                self.$field = ::zerocopy::little_endian::U32::new(val);
            }
        }
    };
    (@impl $field:ident, u64) => {
        ::paste::paste! {
            #[inline]
            pub fn $field(&self) -> u64 {
                self.$field.get()
            }

            #[inline]
            pub fn [<set_ $field>](&mut self, val: u64) {
                self.$field = ::zerocopy::little_endian::U64::new(val);
            }
        }
    };
    ($($field:ident : $ty:tt),* $(,)?) => {
        $(
            $crate::zerocopy_accessors!(@impl $field, $ty);
        )*
    };
}

/// Generates only getter methods for zerocopy little-endian fields (read-only).
#[macro_export]
macro_rules! zerocopy_getters {
    ($($field:ident : $native_ty:ty),* $(,)?) => {
        $(
            #[inline]
            pub fn $field(&self) -> $native_ty {
                self.$field.get()
            }
        )*
    };
}
