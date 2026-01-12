# Key Encoding Rules - TurDB

> **THIS FILE IS MANDATORY READING BEFORE ANY KEY ENCODING WORK**

## Core Principle

All keys MUST be encoded in **big-endian byte-comparable format** with type prefixes.
This allows multi-column keys of any type to be compared with a single `memcmp`.

---

## Type Prefix Constants

```rust
pub mod TypePrefix {
    // Special values
    pub const NULL: u8 = 0x01;
    pub const FALSE: u8 = 0x02;
    pub const TRUE: u8 = 0x03;

    // Numbers (ordered: negative < zero < positive)
    pub const NEG_INFINITY: u8 = 0x10;
    pub const NEG_BIG_INT: u8 = 0x11;   // Arbitrary precision negative
    pub const NEG_INT: u8 = 0x12;       // i64 negative
    pub const NEG_FLOAT: u8 = 0x13;     // f64 negative
    pub const ZERO: u8 = 0x14;
    pub const POS_FLOAT: u8 = 0x15;     // f64 positive
    pub const POS_INT: u8 = 0x16;       // i64 positive
    pub const POS_BIG_INT: u8 = 0x17;   // Arbitrary precision positive
    pub const POS_INFINITY: u8 = 0x18;
    pub const NAN: u8 = 0x19;           // NaN sorts after all numbers

    // Strings/Binary
    pub const TEXT: u8 = 0x20;
    pub const BLOB: u8 = 0x21;

    // Date/Time
    pub const DATE: u8 = 0x30;
    pub const TIME: u8 = 0x31;
    pub const TIMESTAMP: u8 = 0x32;
    pub const TIMESTAMPTZ: u8 = 0x33;
    pub const INTERVAL: u8 = 0x34;

    // Special types
    pub const UUID: u8 = 0x40;
    pub const INET: u8 = 0x41;          // IP addresses
    pub const MACADDR: u8 = 0x42;

    // JSON types (RFC 7159 ordering)
    pub const JSON_NULL: u8 = 0x50;
    pub const JSON_FALSE: u8 = 0x51;
    pub const JSON_TRUE: u8 = 0x52;
    pub const JSON_NUMBER: u8 = 0x53;
    pub const JSON_STRING: u8 = 0x54;
    pub const JSON_ARRAY: u8 = 0x55;
    pub const JSON_OBJECT: u8 = 0x56;

    // Composite/Custom types
    pub const ARRAY: u8 = 0x60;         // SQL arrays
    pub const TUPLE: u8 = 0x61;         // Row/Record types
    pub const RANGE: u8 = 0x62;         // PostgreSQL range types
    pub const ENUM: u8 = 0x63;          // Enum types
    pub const COMPOSITE: u8 = 0x64;     // User-defined composite
    pub const DOMAIN: u8 = 0x65;        // Domain types

    // Vectors
    pub const VECTOR: u8 = 0x70;

    // Extension point
    pub const CUSTOM_START: u8 = 0x80;  // 0x80-0xFE for custom types
    pub const MAX_KEY: u8 = 0xFF;       // Sentinel for range queries
}
```

---

## Type Ordering Summary

```
NULL < FALSE < TRUE < -∞ < negative numbers < 0 < positive numbers < +∞ < NaN
< TEXT < BLOB < DATE < TIME < TIMESTAMP < UUID < JSON_* < ARRAY < COMPOSITE < CUSTOM
```

---

## Integer Encoding (Sign-Split)

Negative and positive integers use different prefixes to maintain sort order:

```rust
fn encode_int(n: i64, buf: &mut Vec<u8>) {
    if n < 0 {
        buf.push(TypePrefix::NEG_INT);
        buf.extend((n as u64).to_be_bytes()); // Two's complement preserves order
    } else if n == 0 {
        buf.push(TypePrefix::ZERO);
    } else {
        buf.push(TypePrefix::POS_INT);
        buf.extend((n as u64).to_be_bytes());
    }
}
```

**Example encodings:**
| Value | Encoded |
|-------|---------|
| -100 | `[0x12, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0x9C]` |
| 0 | `[0x14]` |
| 42 | `[0x16, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x2A]` |

---

## Float Encoding (IEEE Bit Manipulation)

Floats require special handling for byte-comparable ordering:

```rust
fn encode_float(f: f64, buf: &mut Vec<u8>) {
    if f.is_nan() {
        buf.push(TypePrefix::NAN);
    } else if f == f64::NEG_INFINITY {
        buf.push(TypePrefix::NEG_INFINITY);
    } else if f == f64::INFINITY {
        buf.push(TypePrefix::POS_INFINITY);
    } else if f < 0.0 {
        buf.push(TypePrefix::NEG_FLOAT);
        buf.extend((!f.to_bits()).to_be_bytes()); // Invert all bits
    } else if f == 0.0 {
        buf.push(TypePrefix::ZERO);
    } else {
        buf.push(TypePrefix::POS_FLOAT);
        buf.extend((f.to_bits() ^ (1u64 << 63)).to_be_bytes()); // Flip sign bit
    }
}
```

**Why this works:**
- Negative floats: Inverting all bits reverses the order
- Positive floats: Flipping sign bit makes IEEE order match byte order

---

## Text Encoding (Escaped + Null-Terminated)

Strings must escape null bytes and 0xFF to allow concatenation:

```rust
fn encode_text(s: &str, buf: &mut Vec<u8>) {
    buf.push(TypePrefix::TEXT);
    for byte in s.as_bytes() {
        match byte {
            0x00 => { buf.push(0x00); buf.push(0xFF); }  // Escape null
            0xFF => { buf.push(0xFF); buf.push(0x00); }  // Escape 0xFF
            _ => buf.push(*byte),
        }
    }
    buf.push(0x00);
    buf.push(0x00); // Double-null terminator
}
```

**Example:**
| String | Encoded |
|--------|---------|
| "hello" | `[0x20, 'h', 'e', 'l', 'l', 'o', 0x00, 0x00]` |
| "a\x00b" | `[0x20, 'a', 0x00, 0xFF, 'b', 0x00, 0x00]` |

---

## Timestamp Encoding

Use XOR flip for signed values:

```rust
fn encode_timestamp(micros_since_epoch: i64, buf: &mut Vec<u8>) {
    buf.push(TypePrefix::TIMESTAMP);
    buf.extend((micros_since_epoch ^ i64::MIN).to_be_bytes());
}
```

---

## UUID Encoding

UUIDs are already byte-comparable:

```rust
fn encode_uuid(uuid: &[u8; 16], buf: &mut Vec<u8>) {
    buf.push(TypePrefix::UUID);
    buf.extend(uuid);
}
```

---

## Composite Types

### Arrays
```rust
fn encode_array(elements: &[Value], buf: &mut Vec<u8>) {
    buf.push(TypePrefix::ARRAY);
    for elem in elements {
        encode_value(elem, buf);
        buf.push(0x01); // Element separator
    }
    buf.push(0x00); // Terminator
}
```

### Enums
```rust
fn encode_enum(variant_ordinal: u32, type_id: u32, buf: &mut Vec<u8>) {
    buf.push(TypePrefix::ENUM);
    buf.extend(type_id.to_be_bytes());
    buf.extend(variant_ordinal.to_be_bytes()); // Preserves declaration order
}
```

---

## JSON Encoding

```rust
fn encode_json(json: &JsonValue, buf: &mut Vec<u8>) {
    match json {
        JsonValue::Null => buf.push(TypePrefix::JSON_NULL),
        JsonValue::Bool(false) => buf.push(TypePrefix::JSON_FALSE),
        JsonValue::Bool(true) => buf.push(TypePrefix::JSON_TRUE),
        JsonValue::Number(n) => {
            buf.push(TypePrefix::JSON_NUMBER);
            encode_json_number(n, buf);
        }
        JsonValue::String(s) => {
            buf.push(TypePrefix::JSON_STRING);
            encode_text_body(s, buf);
        }
        JsonValue::Array(arr) => {
            buf.push(TypePrefix::JSON_ARRAY);
            for elem in arr {
                encode_json(elem, buf);
                buf.push(0x01);
            }
            buf.push(0x00);
        }
        JsonValue::Object(obj) => {
            buf.push(TypePrefix::JSON_OBJECT);
            let mut keys: Vec<_> = obj.keys().collect();
            keys.sort(); // Deterministic key order
            for key in keys {
                encode_text_body(key, buf);
                buf.push(0x02);
                encode_json(&obj[key], buf);
                buf.push(0x01);
            }
            buf.push(0x00);
        }
    }
}
```

---

## Testing Encoding

**CRITICAL: Expected values must be independently computed:**

```rust
#[test]
fn encode_positive_integer_42() {
    // Expected: POS_INT prefix (0x16) + big-endian 42
    // 42 = 0x2A, padded to 8 bytes
    let expected = vec![0x16, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x2A];

    let mut buf = Vec::new();
    encode_int(42, &mut buf);

    assert_eq!(buf, expected);
}

#[test]
fn encoded_integers_maintain_sort_order() {
    let values = [-1000i64, -1, 0, 1, 1000];
    let mut encoded: Vec<Vec<u8>> = values.iter()
        .map(|&n| {
            let mut buf = Vec::new();
            encode_int(n, &mut buf);
            buf
        })
        .collect();

    let sorted = encoded.clone();
    encoded.sort();

    assert_eq!(encoded, sorted, "encoded order must match value order");
}
```

---

## Checklist

Before any encoding work:

- [ ] Using correct type prefix?
- [ ] Big-endian byte order?
- [ ] Handling negative numbers correctly (sign-split or XOR)?
- [ ] Escaping null bytes in strings?
- [ ] Double-null terminator for strings?
- [ ] Tests with independently computed expected values?
- [ ] Tests verifying sort order is preserved?
