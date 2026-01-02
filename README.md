# TurDB

<p align="center">
  <img src="book/mascot.jpeg" alt="TurDB Mascot" width="200"/>
</p>

<p align="center">
  <strong>Embedded database with row storage and native vector search</strong>
</p>

---

TurDB is an embedded database written in Rust that combines row storage with HNSW vector search. It uses memory-mapped I/O, zero-copy data access, and a file-per-table architecture.

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
| `INET`           | IPv4 or IPv6 address                | Variable  |
| `MACADDR`        | MAC address                         | 6 bytes   |

#### Type Aliases

| Alias              | Canonical Type |
|--------------------|----------------|
| `INTEGER`          | `INT`          |
| `FLOAT`            | `REAL`         |
| `DOUBLE PRECISION` | `DOUBLE`       |
| `BOOL`             | `BOOLEAN`      |

### Data Manipulation Language

#### INSERT

```sql
-- Named columns
INSERT INTO users (name, email, age) VALUES ('Alice', 'alice@example.com', 30);

-- All columns
INSERT INTO users VALUES (1, 'Bob', 'bob@example.com', 25, NULL, CURRENT_TIMESTAMP);

-- Multiple rows
INSERT INTO users (name, email) VALUES
    ('Carol', 'carol@example.com'),
    ('Dave', 'dave@example.com'),
    ('Eve', 'eve@example.com');

-- With RETURNING clause
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
-- Basic query
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

-- Multiple joins
SELECT u.name, p.name AS product, oi.quantity
FROM users u
JOIN orders o ON u.id = o.customer_id
JOIN order_items oi ON o.id = oi.order_id
JOIN products p ON oi.product_id = p.id;
```

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
SELECT * FROM users WHERE id IN (
    SELECT customer_id FROM orders WHERE total > 1000
);

SELECT u.*, (
    SELECT COUNT(*) FROM orders WHERE customer_id = u.id
) AS order_count
FROM users u;
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

### Pragmas and Configuration

#### Write-Ahead Logging

```sql
PRAGMA WAL=ON;
PRAGMA WAL=OFF;
```

#### Synchronous Mode

```sql
PRAGMA synchronous=OFF;     -- No fsync, fastest, risk of corruption
PRAGMA synchronous=NORMAL;  -- fsync at critical moments
PRAGMA synchronous=FULL;    -- fsync after every transaction
```

#### Session Variables

```sql
SET cache_size = 1024;
SET foreign_keys = OFF;
SET foreign_keys = ON;
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

-- Similarity search (nearest neighbors)
SELECT id, content
FROM documents
ORDER BY embedding <-> '[0.15, 0.25, 0.35, ...]'
LIMIT 10;
```

## CLI Reference

The TurDB CLI provides an interactive shell for database operations.

```
turdb ./mydb           # Open existing database
turdb --create ./newdb # Create new database
turdb --version        # Show version
```

### Interactive Mode

```
TurDB version 0.1.0
Enter ".help" for usage hints.
Connected to: ./mydb

turdb> SELECT * FROM users;
+----+-------+-----+
| id | name  | age |
+----+-------+-----+
|  1 | Alice |  30 |
|  2 | Bob   |  25 |
+----+-------+-----+
2 rows in set (0.001 sec)
```

### Dot Commands

| Command            | Description                              |
|--------------------|------------------------------------------|
| `.quit`, `.exit`   | Exit the CLI                             |
| `.tables`          | List all tables                          |
| `.schema [TABLE]`  | Show CREATE statement for table(s)       |
| `.indexes [TABLE]` | List indexes                             |
| `.help`            | Show available commands                  |

Multi-line statements are supported. The prompt changes from `turdb>` to `    ->` when a statement continues across lines. Terminate statements with `;`.

## Storage Architecture

### File Layout

```
database_dir/
├── turdb.meta           # Catalog and metadata
├── root/                # Default schema
│   ├── table.tbd        # Table data (B+tree)
│   ├── table.idx        # Secondary indexes
│   └── table.hnsw       # Vector indexes
└── wal/
    └── wal.000001       # Write-ahead log
```

### Page Format

All files use 16KB pages with the following structure:

```
+------------------+
| Page Header (16B)|
+------------------+
| Cell Pointers    |
+------------------+
| Free Space       |
+------------------+
| Cell Content     |
+------------------+
```

## Programmatic API

### Opening a Database

```rust
use turdb::Database;

// Create new database
let db = Database::create("./mydb")?;

// Open existing database
let db = Database::open("./mydb")?;
```

### Executing SQL

```rust
// Execute DDL/DML
db.execute("CREATE TABLE users (id INT, name TEXT)")?;
db.execute("INSERT INTO users VALUES (1, 'Alice')")?;

// Query with results
match db.execute("SELECT * FROM users")? {
    ExecuteResult::Select { columns, rows } => {
        for row in rows {
            // Process row
        }
    }
    _ => {}
}
```

### Batch Operations

```rust
use turdb::{Database, OwnedValue};

let rows: Vec<Vec<OwnedValue>> = vec![
    vec![OwnedValue::Int(1), OwnedValue::Text("Alice".into())],
    vec![OwnedValue::Int(2), OwnedValue::Text("Bob".into())],
];

db.insert_batch("users", &rows)?;
```

## License

MIT
