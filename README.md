# TurDB

<p align="center">
  <img src="assets/mascot.jpeg" alt="TurDB Mascot" width="200"/>
</p>

<p align="center">
  <strong>High-performance embedded database with row storage and native vector search</strong>
</p>

---

TurDB is an embedded database written in Rust combining SQLite-inspired row storage with HNSW vector search. Built for performance with zero-copy data access, memory-mapped I/O, and MVCC transactions.

## Architecture

```
┌─────────────────────────────────────┐
│         Public API (Database)       │
├─────────────────────────────────────┤
│     SQL Layer (Parser/Executor)     │
├─────────────────────────────────────┤
│  Schema & Catalog │ MVCC Transaction│
├───────────────────┼─────────────────┤
│   B-Tree Index    │   HNSW Index    │
├─────────────────────────────────────┤
│     Record Serialization Layer      │
├─────────────────────────────────────┤
│      Storage Layer (Pager/Cache)    │
├─────────────────────────────────────┤
│    Memory-Mapped File I/O + WAL     │
└─────────────────────────────────────┘
```

## SQL Dialect

TurDB implements a SQL dialect inspired by SQLite and PostgreSQL, with extensions for vector operations.

### Data Definition Language

#### CREATE TABLE

```sql
CREATE TABLE users (
    id BIGINT PRIMARY KEY AUTO_INCREMENT,
    name VARCHAR(100) NOT NULL,
    email TEXT UNIQUE,
    age INT DEFAULT 0,
    metadata JSONB,
    created_at TIMESTAMP
);

CREATE TABLE IF NOT EXISTS products (
    sku VARCHAR(50) PRIMARY KEY,
    name TEXT NOT NULL,
    price DECIMAL,
    embedding VECTOR(384)
);
```

#### Column Constraints

| Constraint       | Description                              |
|------------------|------------------------------------------|
| `PRIMARY KEY`    | Unique identifier, creates index         |
| `AUTO_INCREMENT` | Automatically assign sequential values   |
| `NOT NULL`       | Disallow NULL values                     |
| `UNIQUE`         | Enforce uniqueness                       |
| `DEFAULT value`  | Default value when not specified         |
| `CHECK (expr)`   | Validate with expression                 |
| `FOREIGN KEY`    | Cross-table reference constraint         |

#### CREATE INDEX

```sql
CREATE INDEX idx_email ON users (email);
CREATE UNIQUE INDEX idx_sku ON products (sku);
CREATE INDEX idx_composite ON orders (customer_id, status);
```

#### DROP Statements

```sql
DROP TABLE users;
DROP TABLE IF EXISTS users;
DROP INDEX idx_email;
DROP INDEX IF EXISTS idx_email;
```

#### Schema Management

```sql
CREATE SCHEMA analytics;
DROP SCHEMA analytics;
```

Qualified table names: `schema_name.table_name`

#### ALTER TABLE

```sql
ALTER TABLE users ADD COLUMN phone VARCHAR(20);
ALTER TABLE users DROP COLUMN phone;
ALTER TABLE users RENAME COLUMN email TO email_address;
```

### Data Types

| Type             | Description                         | Storage   |
|------------------|-------------------------------------|-----------|
| `BOOLEAN`        | True/false                          | 1 byte    |
| `SMALLINT`       | Signed 16-bit integer               | 2 bytes   |
| `INT`            | Signed 32-bit integer               | 4 bytes   |
| `BIGINT`         | Signed 64-bit integer               | 8 bytes   |
| `REAL`           | 32-bit IEEE 754 float               | 4 bytes   |
| `DOUBLE`         | 64-bit IEEE 754 float               | 8 bytes   |
| `DECIMAL`        | Arbitrary precision decimal         | Variable  |
| `CHAR(n)`        | Fixed-length string                 | n bytes   |
| `VARCHAR(n)`     | Variable-length string, max n       | Variable  |
| `TEXT`           | Unlimited length string             | Variable  |
| `BLOB`           | Binary data                         | Variable  |
| `DATE`           | Calendar date                       | 4 bytes   |
| `TIME`           | Time of day                         | 8 bytes   |
| `TIMESTAMP`      | Date and time                       | 8 bytes   |
| `TIMESTAMPTZ`    | Timestamp with timezone             | 12 bytes  |
| `INTERVAL`       | Time interval                       | 16 bytes  |
| `UUID`           | 128-bit unique identifier           | 16 bytes  |
| `JSONB`          | Binary JSON                         | Variable  |
| `VECTOR(dim)`    | Float array for similarity search   | dim*4 bytes |
| `INET`           | IPv4 or IPv6 address                | 4-16 bytes |
| `MACADDR`        | MAC address                         | 6 bytes   |
| `POINT`          | 2D point (x, y)                     | 16 bytes  |
| `BOX`            | 2D bounding box                     | 32 bytes  |
| `CIRCLE`         | Circle (center, radius)             | 24 bytes  |
| `INT4RANGE`      | Integer range                       | 9 bytes   |
| `INT8RANGE`      | Bigint range                        | 17 bytes  |
| `DATERANGE`      | Date range                          | 9 bytes   |
| `TSRANGE`        | Timestamp range                     | 17 bytes  |
| `ENUM`           | Enumerated type                     | 4 bytes   |
| `ARRAY`          | Array type                          | Variable  |
| `COMPOSITE`      | User-defined composite type         | Variable  |

#### Type Aliases

| Alias              | Canonical Type |
|--------------------|----------------|
| `INTEGER`          | `INT`          |
| `FLOAT`            | `REAL`         |
| `DOUBLE PRECISION` | `DOUBLE`       |
| `BOOL`             | `BOOLEAN`      |
| `INT2`             | `SMALLINT`     |
| `INT4`             | `INT`          |
| `INT8`             | `BIGINT`       |
| `FLOAT4`           | `REAL`         |
| `FLOAT8`           | `DOUBLE`       |

### Data Manipulation Language

#### INSERT

```sql
INSERT INTO users (name, email, age) VALUES ('Alice', 'alice@example.com', 30);

INSERT INTO users VALUES (1, 'Bob', 'bob@example.com', 25, NULL, CURRENT_TIMESTAMP);

INSERT INTO users (name, email) VALUES
    ('Carol', 'carol@example.com'),
    ('Dave', 'dave@example.com'),
    ('Eve', 'eve@example.com');

INSERT INTO users (name, email) VALUES ('Frank', 'frank@example.com') RETURNING id;
```

#### UPDATE

```sql
UPDATE users SET age = 31 WHERE name = 'Alice';

UPDATE products SET price = price * 1.1 WHERE category = 'electronics';

UPDATE users SET email = 'updated@example.com', age = age + 1 WHERE id = 1 RETURNING *;
```

#### DELETE

```sql
DELETE FROM users WHERE id = 1;

DELETE FROM sessions WHERE expires_at < CURRENT_TIMESTAMP;

DELETE FROM users WHERE id = 5 RETURNING name, email;
```

#### TRUNCATE

```sql
TRUNCATE TABLE logs;
```

### Query Language

#### SELECT

```sql
SELECT * FROM users;
SELECT id, name, email FROM users;

-- Expressions
SELECT id, name, age * 12 AS age_months FROM users;

-- Filtering
SELECT * FROM users WHERE age >= 21 AND status = 'active';
SELECT * FROM products WHERE price BETWEEN 10 AND 100;
SELECT * FROM users WHERE email LIKE '%@example.com';
SELECT * FROM users WHERE name IN ('Alice', 'Bob', 'Carol');
SELECT * FROM users WHERE metadata IS NOT NULL;

-- Ordering
SELECT * FROM users ORDER BY created_at DESC;
SELECT * FROM products ORDER BY category ASC, price DESC;

-- Limiting
SELECT * FROM users LIMIT 10;
SELECT * FROM users LIMIT 10 OFFSET 20;

-- Distinct
SELECT DISTINCT status FROM orders;
```

#### Operators

| Category    | Operators                                      |
|-------------|------------------------------------------------|
| Comparison  | `=`, `<>`, `!=`, `<`, `>`, `<=`, `>=`          |
| Logical     | `AND`, `OR`, `NOT`                             |
| Arithmetic  | `+`, `-`, `*`, `/`, `%`                        |
| Pattern     | `LIKE`, `NOT LIKE`                             |
| Range       | `BETWEEN`, `NOT BETWEEN`                       |
| List        | `IN`, `NOT IN`                                 |
| Null        | `IS NULL`, `IS NOT NULL`                       |
| Vector      | `<->` (L2 distance), `<=>` (cosine distance)   |

#### JOINs

```sql
-- Inner join
SELECT u.name, o.total
FROM users u
JOIN orders o ON u.id = o.customer_id;

-- Left outer join
SELECT u.name, COUNT(o.id) AS order_count
FROM users u
LEFT JOIN orders o ON u.id = o.customer_id
GROUP BY u.id, u.name;

-- Cross join
SELECT * FROM colors CROSS JOIN sizes;

-- Multiple joins (3-way and beyond)
SELECT u.name, p.name AS product, oi.quantity
FROM users u
JOIN orders o ON u.id = o.customer_id
JOIN order_items oi ON o.id = oi.order_id
JOIN products p ON oi.product_id = p.id;
```

Supported join types: `INNER JOIN`, `LEFT JOIN`, `RIGHT JOIN`, `FULL OUTER JOIN`, `CROSS JOIN`

Join algorithms: Hash joins (default for equi-joins), nested loop joins

#### Aggregate Functions

| Function   | Description                      |
|------------|----------------------------------|
| `COUNT(*)` | Number of rows                   |
| `COUNT(x)` | Number of non-NULL values        |
| `SUM(x)`   | Sum of values                    |
| `AVG(x)`   | Average of values                |
| `MIN(x)`   | Minimum value                    |
| `MAX(x)`   | Maximum value                    |

```sql
SELECT
    status,
    COUNT(*) AS count,
    SUM(total) AS revenue,
    AVG(total) AS avg_order
FROM orders
GROUP BY status
HAVING COUNT(*) > 10;
```

#### Subqueries

```sql
-- IN subquery
SELECT * FROM users WHERE id IN (
    SELECT customer_id FROM orders WHERE total > 1000
);

-- Scalar subquery
SELECT u.*, (
    SELECT COUNT(*) FROM orders WHERE customer_id = u.id
) AS order_count
FROM users u;

-- EXISTS subquery
SELECT * FROM users u WHERE EXISTS (
    SELECT 1 FROM orders WHERE customer_id = u.id
);
```

#### Set Operations

```sql
SELECT id, name FROM employees
UNION
SELECT id, name FROM contractors;

SELECT id FROM active_users
INTERSECT
SELECT id FROM premium_users;

SELECT id FROM all_users
EXCEPT
SELECT id FROM banned_users;

-- With ALL (preserve duplicates)
SELECT name FROM table1 UNION ALL SELECT name FROM table2;
```

#### Window Functions

```sql
SELECT
    name,
    department,
    salary,
    ROW_NUMBER() OVER (PARTITION BY department ORDER BY salary DESC) as rank,
    SUM(salary) OVER (PARTITION BY department) as dept_total,
    AVG(salary) OVER () as company_avg
FROM employees;
```

### Transactions

```sql
BEGIN;
UPDATE accounts SET balance = balance - 100 WHERE id = 1;
UPDATE accounts SET balance = balance + 100 WHERE id = 2;
COMMIT;

-- Rollback on error
BEGIN;
INSERT INTO audit_log (action) VALUES ('transfer');
UPDATE accounts SET balance = balance - 100 WHERE id = 1;
ROLLBACK;

-- Savepoints
BEGIN;
INSERT INTO logs (message) VALUES ('step 1');
SAVEPOINT sp1;
INSERT INTO logs (message) VALUES ('step 2');
ROLLBACK TO sp1;
INSERT INTO logs (message) VALUES ('step 2 retry');
RELEASE sp1;
COMMIT;
```

### EXPLAIN

```sql
EXPLAIN SELECT * FROM users WHERE id = 1;
```

### SQL Functions

#### String Functions

| Function | Description |
|----------|-------------|
| `UPPER(str)`, `UCASE(str)` | Convert to uppercase |
| `LOWER(str)`, `LCASE(str)` | Convert to lowercase |
| `LENGTH(str)`, `LEN(str)` | Byte length |
| `CHAR_LENGTH(str)` | Character count |
| `SUBSTR(str, pos, len)` | Substring extraction |
| `LEFT(str, len)` | Left substring |
| `RIGHT(str, len)` | Right substring |
| `CONCAT(s1, s2, ...)` | Concatenate strings |
| `CONCAT_WS(sep, s1, s2, ...)` | Concatenate with separator |
| `TRIM(str)` | Trim whitespace |
| `LTRIM(str)`, `RTRIM(str)` | Trim left/right |
| `LPAD(str, len, pad)` | Left pad |
| `RPAD(str, len, pad)` | Right pad |
| `REPLACE(str, from, to)` | Replace occurrences |
| `REVERSE(str)` | Reverse string |
| `REPEAT(str, n)` | Repeat n times |
| `INSTR(str, substr)` | Find position |
| `LOCATE(substr, str)` | Find position |
| `ASCII(str)` | ASCII code of first char |
| `STRCMP(s1, s2)` | Compare strings |

#### Numeric Functions

| Function | Description |
|----------|-------------|
| `ABS(n)` | Absolute value |
| `ROUND(n, d)` | Round to d decimals |
| `CEIL(n)`, `CEILING(n)` | Round up |
| `FLOOR(n)` | Round down |
| `TRUNCATE(n, d)` | Truncate to d decimals |
| `MOD(n, m)` | Modulo |
| `SQRT(n)` | Square root |
| `POW(x, y)`, `POWER(x, y)` | x raised to y |
| `EXP(n)` | e raised to n |
| `LOG(n)`, `LN(n)` | Natural log |
| `LOG10(n)` | Log base 10 |
| `LOG2(n)` | Log base 2 |
| `SIN(n)`, `COS(n)`, `TAN(n)` | Trig functions |
| `ASIN(n)`, `ACOS(n)`, `ATAN(n)` | Inverse trig |
| `DEGREES(n)` | Radians to degrees |
| `RADIANS(n)` | Degrees to radians |
| `PI()` | Value of pi |
| `RAND()` | Random number [0, 1) |
| `SIGN(n)` | Sign (-1, 0, 1) |
| `GREATEST(a, b, ...)` | Maximum value |
| `LEAST(a, b, ...)` | Minimum value |

#### Date/Time Functions

| Function | Description |
|----------|-------------|
| `NOW()`, `CURRENT_TIMESTAMP` | Current datetime |
| `CURDATE()`, `CURRENT_DATE` | Current date |
| `CURTIME()`, `CURRENT_TIME` | Current time |
| `DATE(datetime)` | Extract date |
| `TIME(datetime)` | Extract time |
| `YEAR(date)` | Extract year |
| `MONTH(date)` | Extract month |
| `DAY(date)` | Extract day |
| `HOUR(time)` | Extract hour |
| `MINUTE(time)` | Extract minute |
| `SECOND(time)` | Extract second |
| `DAYNAME(date)` | Day name |
| `MONTHNAME(date)` | Month name |
| `DAYOFWEEK(date)` | Day of week (1-7) |
| `DAYOFYEAR(date)` | Day of year |
| `QUARTER(date)` | Quarter (1-4) |
| `WEEK(date)` | Week number |
| `DATE_ADD(date, days)` | Add days |
| `DATE_SUB(date, days)` | Subtract days |
| `DATEDIFF(d1, d2)` | Days between dates |
| `LAST_DAY(date)` | Last day of month |
| `DATE_FORMAT(date, fmt)` | Format date |

#### Control Flow Functions

| Function | Description |
|----------|-------------|
| `IF(cond, then, else)` | Conditional |
| `IFNULL(expr, alt)` | NULL replacement |
| `NULLIF(a, b)` | Return NULL if equal |
| `COALESCE(a, b, ...)` | First non-NULL |
| `CASE WHEN ... THEN ... END` | Case expression |

#### System Functions

| Function | Description |
|----------|-------------|
| `VERSION()` | Database version |
| `DATABASE()` | Current database |
| `TYPEOF(expr)` | Type of expression |
| `CAST(expr AS type)` | Type conversion |

### Pragmas and Configuration

#### Write-Ahead Logging

```sql
PRAGMA wal = ON;              -- Enable WAL
PRAGMA wal = OFF;             -- Disable WAL
PRAGMA wal_autoflush = OFF;   -- Defer WAL writes for batches
PRAGMA wal_checkpoint;        -- Force checkpoint
PRAGMA wal_frame_count;       -- Current frame count
PRAGMA wal_size;              -- WAL size in bytes
```

#### Synchronous Mode

```sql
PRAGMA synchronous = OFF;     -- No fsync, fastest, risk of corruption
PRAGMA synchronous = NORMAL;  -- fsync at critical moments
PRAGMA synchronous = FULL;    -- fsync after every transaction
```

#### Memory Configuration

```sql
PRAGMA memory_budget;              -- Query total memory budget
PRAGMA memory_stats;               -- Per-pool memory usage
PRAGMA join_memory_budget = 65536; -- Set join memory limit
```

#### Database State

```sql
PRAGMA database_mode;   -- Query mode (read_write or read_only_degraded)
PRAGMA recover_wal;     -- Trigger streaming WAL recovery
```

### Vector Operations

TurDB supports vector columns for similarity search using HNSW indexes.

```sql
-- Create table with vector column
CREATE TABLE documents (
    id BIGINT PRIMARY KEY AUTO_INCREMENT,
    content TEXT,
    embedding VECTOR(384)
);

-- Insert vectors
INSERT INTO documents (content, embedding)
VALUES ('Hello world', '[0.1, 0.2, 0.3, ...]');

-- Similarity search (L2 distance)
SELECT id, content
FROM documents
ORDER BY embedding <-> '[0.15, 0.25, 0.35, ...]'
LIMIT 10;

-- Cosine similarity
SELECT id, content
FROM documents
ORDER BY embedding <=> '[0.15, 0.25, 0.35, ...]'
LIMIT 10;
```

## CLI Reference

The TurDB CLI provides an interactive shell for database operations.

```
turdb ./mydb              # Open or create database
turdb --create ./newdb    # Create new database
turdb --version           # Show version
turdb --help              # Show help
```

### Interactive Mode

```
TurDB - High-performance embedded database

turdb> SELECT * FROM users;
+----+-------+-----+
| id | name  | age |
+----+-------+-----+
|  1 | Alice |  30 |
|  2 | Bob   |  25 |
+----+-------+-----+
2 rows in set
```

### Dot Commands

| Command            | Description                              |
|--------------------|------------------------------------------|
| `.quit`, `.exit`, `.q` | Exit the CLI                         |
| `.tables`          | List all tables                          |
| `.schema [TABLE]`  | Show CREATE statement for table(s)       |
| `.indexes [TABLE]` | List indexes                             |
| `.help`, `.h`, `.?` | Show available commands                 |

Multi-line statements are supported. Press Enter to continue on the next line. Terminate statements with `;`. Use Ctrl+C to cancel or Ctrl+D to exit.

## Storage Architecture

### File Layout

```
database_dir/
├── turdb.meta           # Catalog and metadata
├── root/                # Default schema
│   ├── table.tbd        # Table data (B+tree)
│   ├── table.idx        # Secondary indexes
│   └── table.hnsw       # Vector indexes
├── custom_schema/       # User-created schemas
│   └── ...
└── wal/
    └── wal.000001       # Write-ahead log
```

### Page Format

All files use 16KB pages with the following structure:

```
+------------------+ 0
| Page Header (16B)|
+------------------+ 16
| Cell Pointers    |
| (2 bytes each)   |
+------------------+
| Free Space       |
+------------------+
| Cell Content     |
| (grows upward)   |
+------------------+ 16384
```

### B-Tree Features

- B+tree indexes with slot arrays for fast prefix filtering
- 4-byte prefix hints for optimized binary search
- Overflow pages for large values (TOAST)
- Leaf page linking for sequential scans

### HNSW Vector Index

- Hierarchical Navigable Small World graph
- O(log N) approximate nearest neighbor search
- Configurable M (connections) and efConstruction
- Distance metrics: L2 (Euclidean), Cosine, Inner Product
- SQ8 quantization for 4x memory reduction
- SIMD acceleration (AVX2)

## MVCC Transactions

TurDB uses Multi-Version Concurrency Control with Snapshot Isolation:

- Readers never block writers
- Writers never block readers
- Row-level versioning with undo pages
- Lock-free timestamp allocation
- Automatic rollback on transaction drop

```rust
// Transactions are automatically managed
db.execute("BEGIN")?;
db.execute("UPDATE accounts SET balance = balance - 100 WHERE id = 1")?;
db.execute("COMMIT")?;
```

## Programmatic API

### Opening a Database

```rust
use turdb::Database;

// Create new database
let db = Database::create("./mydb")?;

// Open existing database
let db = Database::open("./mydb")?;

// Open with recovery info
let (db, recovery) = Database::open_with_recovery("./mydb")?;
```

### Executing SQL

```rust
use turdb::{Database, ExecuteResult, OwnedValue};

// Execute DDL/DML
db.execute("CREATE TABLE users (id INT, name TEXT)")?;
db.execute("INSERT INTO users VALUES (1, 'Alice')")?;

// Execute with parameters
db.execute_with_params(
    "INSERT INTO users VALUES (?, ?)",
    &[OwnedValue::Int(2), OwnedValue::Text("Bob".into())]
)?;

// Query with results
match db.execute("SELECT * FROM users")? {
    ExecuteResult::Select { columns, rows } => {
        for row in rows {
            // Process row
        }
    }
    _ => {}
}

// Simple query
let rows = db.query("SELECT * FROM users")?;
```

### Prepared Statements

```rust
let stmt = db.prepare("SELECT * FROM users WHERE id = ?")?;
let result = stmt.bind(&[OwnedValue::Int(1)]).execute(&db)?;
```

### Value Types

```rust
use turdb::OwnedValue;

let values = vec![
    OwnedValue::Int(42),
    OwnedValue::Float(3.14),
    OwnedValue::Text("hello".into()),
    OwnedValue::Blob(vec![0x01, 0x02, 0x03]),
    OwnedValue::Null,
];
```

## Performance

### Design Principles

- **Zero-copy**: Direct mmap slices, no intermediate buffers
- **Zero-allocation**: Pre-allocated buffers in CRUD paths
- **Memory-mapped I/O**: Kernel-managed page caching
- **SIEVE eviction**: Better than LRU for database scans
- **Lock sharding**: 64+ shards for concurrent access

### Targets

| Operation | Target |
|-----------|--------|
| Point read (cached) | < 1 microsecond |
| Point read (disk) | < 50 microseconds |
| Sequential scan | > 1M rows/sec |
| Insert | > 100K rows/sec |
| k-NN search (1M vectors, k=10) | < 10ms |

### Memory Budget

- Hard limit: 25% of system RAM (4MB minimum)
- Pool allocation: Cache, Query, Recovery, Schema
- Query via `PRAGMA memory_stats`

## Dependencies

```toml
eyre = "0.6"           # Error handling
parking_lot = "0.12"   # Synchronization primitives
memmap2 = "0.9"        # Memory-mapped I/O
zerocopy = "0.8"       # Zero-copy serialization
bumpalo = "3.14"       # Arena allocator
smallvec = "1.11"      # Small vector optimization
hashbrown = "0.14"     # Fast hash maps
roaring = "0.10"       # Bitmap indexes
crc = "3.0"            # Page checksums
```

## License

MIT
