# File Format Specifications - TurDB

> **THIS FILE IS MANDATORY READING BEFORE ANY FILE FORMAT WORK**

## Directory Structure

```
database_dir/
├── turdb.meta           # Global metadata and catalog
├── root/                # Default schema
│   ├── table_name.tbd   # Table data file
│   ├── table_name.idx   # B-tree indexes
│   └── table_name.hnsw  # HNSW vector indexes (if any)
├── custom_schema/       # User-created schema
│   └── ...
└── wal/
    └── wal.000001       # Write-ahead log segments
```

---

## Page Constants

```rust
const PAGE_SIZE: usize = 16384;       // 16KB - NEVER change this
const PAGE_HEADER_SIZE: usize = 16;   // Every page
const FILE_HEADER_SIZE: usize = 128;  // Page 0 only
const USABLE_SPACE: usize = 16368;    // PAGE_SIZE - PAGE_HEADER_SIZE
const PAGE0_USABLE: usize = 16256;    // PAGE_SIZE - FILE_HEADER_SIZE
```

---

## Global Metadata File (`turdb.meta`)

```
Offset  Size  Description
------  ----  -----------
0       16    Magic: "TurDB Rust v1\x00\x00\x00"
16      4     Version: 1
20      4     Page size: 16384
24      8     Schema count
32      8     Default schema ID
40      8     Next table ID
48      8     Next index ID
56      8     Flags
64      64    Reserved (zero-filled)
```

**Magic bytes (hex):**
```
54 75 72 44 42 20 52 75 73 74 20 76 31 00 00 00
T  u  r  D  B     R  u  s  t     v  1  \0 \0 \0
```

---

## Table Data File Header (`.tbd`)

```
Offset  Size  Description
------  ----  -----------
0       16    Magic: "TurDB Table\x00\x00\x00\x00"
16      8     Table ID
24      8     Row count
32      4     Root page number
36      4     Column count
40      8     First free page (freelist head)
48      8     Auto-increment value
56      72    Reserved (zero-filled)
```

**Magic bytes (hex):**
```
54 75 72 44 42 20 54 61 62 6C 65 00 00 00 00 00
T  u  r  D  B     T  a  b  l  e  \0 \0 \0 \0 \0
```

---

## Index File Header (`.idx`)

```
Offset  Size  Description
------  ----  -----------
0       16    Magic: "TurDB Index\x00\x00\x00\x00"
16      8     Index ID
24      8     Table ID
32      4     Root page number
36      4     Key column count
40      1     Is unique (0 or 1)
41      1     Index type (0=btree)
42      86    Reserved (zero-filled)
```

**Magic bytes (hex):**
```
54 75 72 44 42 20 49 6E 64 65 78 00 00 00 00 00
T  u  r  D  B     I  n  d  e  x  \0 \0 \0 \0 \0
```

---

## HNSW File Header (`.hnsw`)

```
Offset  Size  Description
------  ----  -----------
0       16    Magic: "TurDB HNSW\x00\x00\x00\x00\x00"
16      8     Index ID
24      8     Table ID
32      4     Dimension (vector size)
36      4     M (max connections per layer)
40      4     EfConstruction (build-time search width)
44      4     Entry point node ID
48      8     Node count
56      8     Vector count
64      64    Reserved (zero-filled)
```

**Magic bytes (hex):**
```
54 75 72 44 42 20 48 4E 53 57 00 00 00 00 00 00
T  u  r  D  B     H  N  S  W  \0 \0 \0 \0 \0 \0
```

---

## Page Header (16 bytes)

Every page (except page 0 which has file header) starts with:

```
Offset  Size  Description
------  ----  -----------
0       1     Page type
1       1     Flags
2       2     Cell count
4       2     First free byte offset
6       2     Fragmented free bytes
8       4     Right sibling page (or 0)
12      4     Checksum (CRC32)
```

**Page types:**
```rust
const PAGE_TYPE_LEAF: u8 = 0x0D;      // B-tree leaf
const PAGE_TYPE_INTERIOR: u8 = 0x05;  // B-tree interior
const PAGE_TYPE_OVERFLOW: u8 = 0x0A;  // Overflow data
const PAGE_TYPE_FREELIST: u8 = 0x00;  // Free page
```

---

## B-Tree Leaf Page Layout

```
+------------------------+ 0
| Page Header (16 bytes) |
+------------------------+ 16
| Cell Pointer Array     |
| (2 bytes per cell)     |
+------------------------+
| Unallocated Space      |
|                        |
+------------------------+
| Cell Content Area      |
| (grows upward)         |
+------------------------+ PAGE_SIZE
```

**Cell format (variable length):**
```
+------------------+
| Key length (2B)  |
+------------------+
| Value length (4B)|
+------------------+
| Key data         |
+------------------+
| Value data       |
+------------------+
```

---

## B-Tree Interior Page Layout

```
+------------------------+ 0
| Page Header (16 bytes) |
+------------------------+ 16
| Rightmost child (4B)   |
+------------------------+ 20
| Cell Pointer Array     |
+------------------------+
| Unallocated Space      |
+------------------------+
| Cell Content Area      |
+------------------------+ PAGE_SIZE
```

**Interior cell format:**
```
+------------------+
| Left child (4B)  |
+------------------+
| Key length (2B)  |
+------------------+
| Key data         |
+------------------+
```

---

## Multi-Schema Support

- Default schema: `root`
- System schema: `turdb_catalog`
- User schemas: Created via `CREATE SCHEMA`
- Fully qualified names: `schema.table.column`

Schema directories are subdirectories of the database directory.

---

## WAL Format

WAL files are named `wal.NNNNNN` (6-digit sequence number).

**WAL file header (32 bytes):**
```
Offset  Size  Description
------  ----  -----------
0       4     Magic: 0x57414C31 ("WAL1")
4       4     Version: 1
8       8     Database ID
16      4     Page size: 16384
20      4     Checkpoint sequence
24      8     Salt (random, for integrity)
```

**WAL frame header (24 bytes):**
```
Offset  Size  Description
------  ----  -----------
0       4     Page number
4       4     Database size (pages) after commit
8       8     Salt copy
16      4     Frame checksum
20      4     Data checksum
```

Each frame is followed by PAGE_SIZE bytes of page data.

---

## Validation

### Magic Number Checks
Always validate magic numbers when opening files:

```rust
fn validate_table_magic(header: &[u8]) -> Result<()> {
    const EXPECTED: &[u8] = b"TurDB Table\x00\x00\x00\x00";
    ensure!(
        header[..16] == *EXPECTED,
        "invalid table file magic: expected {:?}, got {:?}",
        EXPECTED, &header[..16]
    );
    Ok(())
}
```

### Checksum Verification
All pages include CRC32 checksums. Verify on read:

```rust
fn verify_page_checksum(page: &[u8]) -> Result<()> {
    let stored = u32::from_le_bytes(page[12..16].try_into().unwrap());
    let computed = crc32(&page[16..]); // Checksum excludes header checksum field
    ensure!(
        stored == computed,
        "page checksum mismatch: stored 0x{:08x}, computed 0x{:08x}",
        stored, computed
    );
    Ok(())
}
```

---

## Checklist

Before any file format work:

- [ ] Using correct magic bytes?
- [ ] All offsets and sizes match spec?
- [ ] Validating magic on file open?
- [ ] Computing/verifying checksums?
- [ ] Reserved fields zero-filled?
- [ ] Page types using correct constants?
- [ ] Cell pointers sorted by key?
