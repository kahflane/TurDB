//! # Record Header for MVCC Version Tracking
//!
//! This module defines the 17-byte header that is prepended to every row's
//! value in the B-tree. The header stores version metadata for MVCC:
//!
//! ## Binary Layout (17 bytes)
//!
//! ```text
//! +----------+----------+------------------+
//! | Flags    | TxnId    | PrevVersion      |
//! | (1 byte) | (8 bytes)| (8 bytes)        |
//! +----------+----------+------------------+
//! ```
//!
//! ## Flags Byte (bitmask)
//!
//! ```text
//! Bit 0: LOCK_BIT    - Row is currently locked by a writer
//! Bit 1: DELETE_BIT  - Row is logically deleted (tombstone)
//! Bit 2: VACUUM_BIT  - Hint for garbage collection
//! Bits 3-7: Reserved for future use
//! ```
//!
//! ## Transaction ID (TxnId)
//!
//! The 8-byte transaction ID identifies which transaction created this version:
//! - During write: The owning transaction's ID (with LOCK_BIT set)
//! - After commit: The commit timestamp
//! - Value 0: Reserved for "always visible" bootstrapped data
//!
//! ## Previous Version Pointer
//!
//! Points to the previous version in the undo log:
//! - Encoded as: (PageID << 16) | Offset
//! - PageID: 48 bits (up to 256 TB with 16KB pages)
//! - Offset: 16 bits (up to 64KB within page)
//! - Value 0: No previous version (first version)
//!
//! ## Zero-Copy Design
//!
//! The `from_bytes()` method reads directly from mmap'd page data without
//! copying. We use manual byte parsing instead of `#[repr(C, packed)]` to
//! handle potentially unaligned memory safely on all architectures.
//!
//! ## Visibility Rules
//!
//! A version V is visible to transaction T if:
//! 1. V.flags & LOCK_BIT == 0 (not locked by another transaction)
//! 2. V.txn_id <= T.read_ts (created before snapshot)
//! 3. V.txn_id is from a committed transaction
//!
//! ## Lock Protocol
//!
//! Writers must:
//! 1. Check current header's LOCK_BIT and txn_id
//! 2. If locked by another txn: abort (write-write conflict)
//! 3. If txn_id > my_read_ts: abort (concurrent modification)
//! 4. Set LOCK_BIT and my txn_id
//! 5. Copy old data to undo page, update prev_version
//! 6. On commit: clear LOCK_BIT, set commit_ts
//!
//! ## Memory Usage
//!
//! 17 bytes per row is the minimum overhead for MVCC. This is acceptable
//! because it enables lock-free reads and version chain traversal without
//! external data structures.

use super::TxnId;

pub mod flags {
    pub const LOCK_BIT: u8 = 0b0000_0001;
    pub const DELETE_BIT: u8 = 0b0000_0010;
    pub const VACUUM_BIT: u8 = 0b0000_0100;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RecordHeader {
    pub flags: u8,
    pub txn_id: TxnId,
    pub prev_version: u64,
}

impl RecordHeader {
    pub const SIZE: usize = 1 + 8 + 8;

    pub fn new(txn_id: TxnId) -> Self {
        Self {
            flags: 0,
            txn_id,
            prev_version: 0,
        }
    }

    #[inline(always)]
    pub fn from_bytes(slice: &[u8]) -> Self {
        debug_assert!(slice.len() >= Self::SIZE);
        let flags = slice[0];
        let txn_id = u64::from_be_bytes(slice[1..9].try_into().unwrap());
        let prev_version = u64::from_be_bytes(slice[9..17].try_into().unwrap());
        Self {
            flags,
            txn_id,
            prev_version,
        }
    }

    #[inline(always)]
    pub fn write_to(&self, slice: &mut [u8]) {
        debug_assert!(slice.len() >= Self::SIZE);
        slice[0] = self.flags;
        slice[1..9].copy_from_slice(&self.txn_id.to_be_bytes());
        slice[9..17].copy_from_slice(&self.prev_version.to_be_bytes());
    }

    pub fn encode_ptr(page_id: u64, offset: u16) -> u64 {
        (page_id << 16) | (offset as u64)
    }

    pub fn decode_ptr(ptr: u64) -> (u64, u16) {
        (ptr >> 16, (ptr & 0xFFFF) as u16)
    }

    pub fn is_locked(&self) -> bool {
        self.flags & flags::LOCK_BIT != 0
    }

    pub fn is_deleted(&self) -> bool {
        self.flags & flags::DELETE_BIT != 0
    }

    pub fn is_vacuum_hint(&self) -> bool {
        self.flags & flags::VACUUM_BIT != 0
    }

    pub fn set_locked(&mut self, locked: bool) {
        if locked {
            self.flags |= flags::LOCK_BIT;
        } else {
            self.flags &= !flags::LOCK_BIT;
        }
    }

    pub fn set_deleted(&mut self, deleted: bool) {
        if deleted {
            self.flags |= flags::DELETE_BIT;
        } else {
            self.flags &= !flags::DELETE_BIT;
        }
    }

    pub fn set_vacuum_hint(&mut self, hint: bool) {
        if hint {
            self.flags |= flags::VACUUM_BIT;
        } else {
            self.flags &= !flags::VACUUM_BIT;
        }
    }

    pub fn has_prev_version(&self) -> bool {
        self.prev_version != 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn record_header_size_is_17_bytes() {
        assert_eq!(RecordHeader::SIZE, 17);
    }

    #[test]
    fn record_header_new_creates_unlocked_undeleted() {
        let hdr = RecordHeader::new(42);
        assert_eq!(hdr.txn_id, 42);
        assert_eq!(hdr.flags, 0);
        assert_eq!(hdr.prev_version, 0);
        assert!(!hdr.is_locked());
        assert!(!hdr.is_deleted());
    }

    #[test]
    fn record_header_from_bytes_parses_correctly() {
        let mut buf = [0u8; 17];
        buf[0] = flags::LOCK_BIT | flags::DELETE_BIT;
        buf[1..9].copy_from_slice(&100u64.to_be_bytes());
        buf[9..17].copy_from_slice(&0x1234_5678_9ABC_DEF0u64.to_be_bytes());

        let hdr = RecordHeader::from_bytes(&buf);
        assert!(hdr.is_locked());
        assert!(hdr.is_deleted());
        assert_eq!(hdr.txn_id, 100);
        assert_eq!(hdr.prev_version, 0x1234_5678_9ABC_DEF0);
    }

    #[test]
    fn record_header_write_to_serializes_correctly() {
        let hdr = RecordHeader {
            flags: flags::LOCK_BIT,
            txn_id: 999,
            prev_version: 0xDEAD_BEEF_0000_CAFE,
        };
        let mut buf = [0u8; 17];
        hdr.write_to(&mut buf);

        assert_eq!(buf[0], flags::LOCK_BIT);
        assert_eq!(u64::from_be_bytes(buf[1..9].try_into().unwrap()), 999);
        assert_eq!(
            u64::from_be_bytes(buf[9..17].try_into().unwrap()),
            0xDEAD_BEEF_0000_CAFE
        );
    }

    #[test]
    fn record_header_roundtrip() {
        let original = RecordHeader {
            flags: flags::DELETE_BIT | flags::VACUUM_BIT,
            txn_id: 12345,
            prev_version: RecordHeader::encode_ptr(0x1234, 0xABCD),
        };
        let mut buf = [0u8; 17];
        original.write_to(&mut buf);
        let restored = RecordHeader::from_bytes(&buf);
        assert_eq!(original, restored);
    }

    #[test]
    fn encode_decode_ptr_roundtrip() {
        let page_id = 0x0000_1234_5678_9ABC;
        let offset = 0xDEF0;
        let encoded = RecordHeader::encode_ptr(page_id, offset);
        let (decoded_page, decoded_offset) = RecordHeader::decode_ptr(encoded);
        assert_eq!(decoded_page, page_id);
        assert_eq!(decoded_offset, offset);
    }

    #[test]
    fn flag_set_and_clear() {
        let mut hdr = RecordHeader::new(1);
        assert!(!hdr.is_locked());
        hdr.set_locked(true);
        assert!(hdr.is_locked());
        hdr.set_locked(false);
        assert!(!hdr.is_locked());

        assert!(!hdr.is_deleted());
        hdr.set_deleted(true);
        assert!(hdr.is_deleted());
        hdr.set_deleted(false);
        assert!(!hdr.is_deleted());

        assert!(!hdr.is_vacuum_hint());
        hdr.set_vacuum_hint(true);
        assert!(hdr.is_vacuum_hint());
    }

    #[test]
    fn has_prev_version_checks_zero() {
        let mut hdr = RecordHeader::new(1);
        assert!(!hdr.has_prev_version());
        hdr.prev_version = 1;
        assert!(hdr.has_prev_version());
    }
}
