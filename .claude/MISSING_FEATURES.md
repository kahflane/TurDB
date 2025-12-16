# TurDB Missing Features & Implementation Gaps

This document identifies all features that are parsed/defined but not fully implemented, discovered during comprehensive testing.

## Executive Summary

The database has a well-designed SQL parser and schema system, but several features are **only partially implemented**:
- Types are parsed but not all can be inserted/queried
- Constraints are stored but not enforced
- Operators are lexed/parsed but not evaluated
- Some column types decode as NULL

---

## 1. Data Type Insertion Issues

### 1.1 Types That Cannot Be Inserted via SQL Literals

| Type | Issue | Root Cause |
|------|-------|------------|
| **UUID** | Cannot insert string literal | `RecordBuilder::set_text()` called on fixed-size UUID column (16 bytes) |
| **JSONB** | Cannot insert JSON literal | No JSON literal parsing in `eval_literal()` |
| **JSON** | Cannot insert JSON literal | Same as JSONB |
| **VECTOR** | Cannot insert vector literal | No array literal parsing for vector values |
| **BLOB** | Only string literals work | No hex literal (x'AABB') to blob conversion |

**Location**: `src/database/database.rs:910-933` (`eval_literal` function)

**Current `eval_literal` only handles**:
- `Literal::Null`
- `Literal::Integer`
- `Literal::Float`
- `Literal::String`
- `Literal::Boolean`

**Missing literal types**:
- `Literal::HexNumber` → should convert to Blob
- Array literals `[1.0, 2.0, 3.0]` → should convert to Vector
- JSON literals `'{"key": "value"}'` → should parse and convert to JSONB

---

## 2. Data Type Query/Decode Issues

### 2.1 Types That Return NULL on Query

The executor's `decode_columns_into` only decodes 8 types. All others return `Value::Null`.

**Location**: `src/sql/executor.rs:265-274`

| Type | Status | Fix Required |
|------|--------|--------------|
| Int2 | ✅ Works | - |
| Int4 | ✅ Works | - |
| Int8 | ✅ Works | - |
| Float4 | ✅ Works | - |
| Float8 | ✅ Works | - |
| Bool | ✅ Works | - |
| Text | ✅ Works | - |
| Blob | ✅ Works | - |
| **Date** | ❌ Returns NULL | Add `view.get_date()` decoding |
| **Time** | ❌ Returns NULL | Add `view.get_time()` decoding |
| **Timestamp** | ❌ Returns NULL | Add `view.get_timestamp()` decoding |
| **TimestampTz** | ❌ Returns NULL | Add `view.get_timestamptz()` decoding |
| **UUID** | ❌ Returns NULL | Add `view.get_uuid()` decoding |
| **Jsonb** | ❌ Returns NULL | Add `view.get_jsonb()` decoding |
| **Vector** | ❌ Returns NULL | Add `view.get_vector()` decoding |
| **Interval** | ❌ Returns NULL | Add `view.get_interval()` decoding |
| **Decimal** | ❌ Returns NULL | Add `view.get_decimal()` decoding |
| **Point** | ❌ Returns NULL | Add `view.get_point()` decoding |
| **Box** | ❌ Returns NULL | Add `view.get_box()` decoding |
| **Circle** | ❌ Returns NULL | Add `view.get_circle()` decoding |
| **MacAddr** | ❌ Returns NULL | Add `view.get_macaddr()` decoding |
| **Inet4** | ❌ Returns NULL | Add `view.get_inet4()` decoding |
| **Inet6** | ❌ Returns NULL | Add `view.get_inet6()` decoding |
| **Enum** | ❌ Returns NULL | Add enum decoding |
| **Array** | ❌ Returns NULL | Add array decoding |
| **Composite** | ❌ Returns NULL | Add composite decoding |
| **Ranges** | ❌ Returns NULL | Add range decoding |

---

## 3. Constraint Enforcement

### 3.1 Constraints Defined but Not Enforced

Constraints are stored in schema metadata but **never validated** during INSERT/UPDATE/DELETE.

**Location**: `src/schema/table.rs:78-84`

| Constraint | Parsed | Stored | Enforced |
|------------|--------|--------|----------|
| NOT NULL | ✅ | ✅ | ❌ |
| PRIMARY KEY | ✅ | ✅ | ❌ |
| UNIQUE | ✅ | ✅ | ❌ |
| FOREIGN KEY | ✅ | ✅ | ❌ |
| CHECK | ✅ | ✅ | ❌ |
| DEFAULT | ✅ | ✅ | ❌ |

**Required Implementation**:
1. Add constraint validation in `execute_insert` (database.rs)
2. Add constraint validation in `execute_update` (database.rs)
3. Add FK validation in `execute_delete` (database.rs)
4. Implement unique index lookup for UNIQUE/PK constraints
5. Parse and evaluate CHECK expressions

---

## 4. SQL Operators Not Evaluated

### 4.1 Binary Operators Parsed but Not Executed

**Location**: `src/sql/executor.rs:1720-1808` (`CompiledPredicate::eval_expr`)

| Operator Category | Parsed | Evaluated |
|-------------------|--------|-----------|
| Comparison (=, <>, <, >, <=, >=) | ✅ | ✅ |
| Logical (AND, OR) | ✅ | ✅ |
| Arithmetic (+, -, *, /, %) | ✅ | ✅ |
| JSON (->>, ->) | ✅ | ✅ |
| **JSON Path (#>, #>>)** | ✅ | ❌ |
| **JSON Contains (@>, <@)** | ✅ | ❌ |
| **Array Operators** | ✅ | ❌ |
| Vector Distance (<->, <#>, <=>) | ✅ | ✅ |
| Bitwise (&, \|, ^, <<, >>) | ✅ | ✅ |
| String Concat (\|\|) | ✅ | ✅ |
| Power (^) | ✅ | ✅ |

---

## 5. SQL Types Not Exposed in Parser

### 5.1 Internal Types Without SQL Syntax

These types exist in `records::types::DataType` but have no SQL keyword:

| Type | Internal Enum | SQL Keyword |
|------|---------------|-------------|
| Point | `DataType::Point` | ❌ None |
| Box | `DataType::Box` | ❌ None |
| Circle | `DataType::Circle` | ❌ None |
| MacAddr | `DataType::MacAddr` | ❌ None |
| Inet4 | `DataType::Inet4` | ❌ None |
| Inet6 | `DataType::Inet6` | ❌ None |
| Int4Range | `DataType::Int4Range` | ❌ None |
| Int8Range | `DataType::Int8Range` | ❌ None |
| DateRange | `DataType::DateRange` | ❌ None |
| TimestampRange | `DataType::TimestampRange` | ❌ None |
| Enum | `DataType::Enum` | ❌ None |
| Composite | `DataType::Composite` | ❌ None |

**Required**: Add keywords to `src/sql/lexer.rs` and parsing in `src/sql/parser.rs:4917-5028`

---

## 6. Expression Evaluation Gaps

### 6.1 Missing Expression Types in WHERE Clauses

**Location**: `src/sql/executor.rs:1744-1768` (`eval_value`)

| Expression Type        | Supported |
|------------------------|-----------|
| Column reference       | ✅ |
| Integer literal        | ✅ |
| Float literal          | ✅ |
| String literal         | ✅ |
| Boolean literal        | ✅ |
| NULL literal           | ✅ |
| Hex literal            | ✅ |
| Binary literal         | ✅ |
| **Function calls**     | ❌ |
| **CASE expressions**   | ❌ |
| **Subqueries**         | ❌ |
| **Aggregate in WHERE** | ❌ |
| **Type casts**         | ❌ |
| **Array subscript**    | ❌ |
| **JSON Extraction**    | ❌ |

---

## 7. Missing SQL Features

### 7.1 DDL Features - Execution Status

**Implemented:**
| Feature | Parsed | Executed |
|---------|--------|----------|
| CREATE TABLE | ✅ | ✅ |
| CREATE INDEX | ✅ | ✅ |
| CREATE SCHEMA | ✅ | ✅ |
| DROP TABLE | ✅ | ✅ |

**Parsed but NOT Executed:**
| Feature | Parsed | Executed | Error |
|---------|--------|----------|-------|
| DROP INDEX | ✅ | ❌ | `"unsupported DROP type: Index"` |
| DROP SCHEMA | ✅ | ❌ | `"unsupported DROP type: Schema"` |
| DROP VIEW | ✅ | ❌ | `"unsupported DROP type: View"` |
| ALTER TABLE (all actions) | ✅ | ❌ | `"unsupported statement type"` |
| USE SCHEMA / SET search_path | ❌ | ❌ | No schema switching |

**ALTER TABLE Actions (Parsed only):**
- ADD COLUMN, DROP COLUMN
- ALTER COLUMN SET/DROP NOT NULL
- ALTER COLUMN SET/DROP DEFAULT
- ALTER COLUMN SET DATA TYPE
- ADD/DROP CONSTRAINT
- RENAME COLUMN, RENAME TO

**Not Parsed:**
| Feature | Status |
|---------|--------|
| ALTER INDEX | ❌ Not parsed |
| ALTER SCHEMA | ❌ Not parsed |
| CREATE TYPE (enum) | ❌ Not parsed |
| CREATE DOMAIN | ❌ Not parsed |

### 7.2 DML Features

| Feature | Status |
|---------|--------|
| INSERT ... SELECT | ❌ Not implemented |
| INSERT ... ON CONFLICT | ❌ Not implemented |
| UPDATE ... FROM | ❌ Not implemented |
| MERGE statement | ❌ Not implemented |
| RETURNING clause | ❌ Not implemented |

### 7.3 Query Features

| Feature | Status |
|---------|--------|
| Subqueries | ❌ Not implemented |
| CTEs (WITH clause) | ❌ Not implemented |
| Window functions | ❌ Not implemented |
| DISTINCT | ❌ Not implemented |
| GROUP BY ROLLUP/CUBE | ❌ Not implemented |
| HAVING clause | ❌ Not implemented |
| UNION/INTERSECT/EXCEPT | ❌ Not implemented |
| NATURAL JOIN | ❌ Not implemented |
| OUTER JOIN (LEFT/RIGHT/FULL) | ❌ Not implemented |
| LATERAL join | ❌ Not implemented |

---

## 8. Index Features

### 8.1 Index Usage in Queries

| Feature | Status |
|---------|--------|
| B-tree index creation | ✅ Works |
| HNSW index creation | ❌ Schema only |
| Index scan optimization | ⚠️ Partial |
| Index-only scans | ❌ Not implemented |
| Multi-column indexes | ⚠️ Partial |
| Expression indexes | ❌ Not implemented |
| Partial indexes | ❌ Not implemented |

---

## 9. Transaction Features

### 9.1 ACID Compliance

| Feature | Status |
|---------|--------|
| BEGIN/COMMIT | ⚠️ Parsed, not implemented |
| ROLLBACK | ⚠️ Parsed, not implemented |
| SAVEPOINT | ⚠️ Parsed, not implemented |
| Isolation levels | ⚠️ Parsed, not implemented |
| WAL for durability | ✅ Works |
| Crash recovery | ✅ Works |

---

## 10. Semantic SQL Checking

### 10.1 What EXISTS ✅

| Check | Location | Behavior |
|-------|----------|----------|
| Table existence | `planner.rs:640-648` | `validate_table_exists()` calls `catalog.resolve_table()` |
| Table in FROM clause | `planner.rs:508` | Validates table before creating scan plan |
| Table in INSERT | `planner.rs:651` | Validates table exists before insert |
| Table in UPDATE | `planner.rs:673` | Validates table exists before update |
| Table in DELETE | `planner.rs:694` | Validates table exists before delete |

**Example error**: `"table 'users' not found"` or `"does not exist"`

### 10.2 What's MISSING ❌

#### Column Existence Validation

| Check | Current Behavior | Expected Behavior |
|-------|------------------|-------------------|
| **SELECT column exists** | Silently returns `None`, query gives empty/wrong results | Error: "column 'xyz' not found in table 'abc'" |
| **INSERT column exists** | No validation | Error: "column 'xyz' does not exist in table 'abc'" |
| **UPDATE column exists** | No validation | Error: "column 'xyz' does not exist" |
| **WHERE column exists** | Returns `None` via `?` | Clear error message |
| **GROUP BY column exists** | No validation | Error: "column 'xyz' not in table" |
| **ORDER BY column exists** | No validation | Error: "column 'xyz' not in table" |
| **JOIN ON column exists** | No validation | Error: "column 'xyz' not in table" |

**Location of silent failures**: `executor.rs:1749-1753`
```rust
.find(|(name, _)| name.eq_ignore_ascii_case(col_ref.column))
.map(|(_, idx)| *idx)?;  // Returns None silently if column not found
```

#### Ambiguous Column Detection (JOINs)

| Check | Current Behavior | Expected Behavior |
|-------|------------------|-------------------|
| **Unqualified column in JOIN** | Uses first match or wrong table | Error: "column 'id' is ambiguous" |
| **Same column name in multiple tables** | No detection | Require qualified name: `table.column` |

**Example that should fail:**
```sql
SELECT id FROM users JOIN orders ON users.id = orders.user_id
-- Error: "column 'id' is ambiguous, exists in: users, orders"
```

**Required fix:** When resolving unqualified column references in JOINs, check if column exists in multiple tables and require qualification.

#### Other Semantic Validations

| Check | Current Behavior | Expected Behavior |
|-------|------------------|-------------------|
| **INSERT value count** | No validation | Error: "INSERT has 3 values but table has 5 columns" |
| **Type checking** | No validation | Error: "cannot insert TEXT into INT column" |
| **Function argument count** | No validation | Error: "function 'substr' expects 2-3 arguments, got 1" |
| **Aggregate in non-aggregate context** | No validation | Error: "aggregate functions not allowed in WHERE" |

### 10.3 Required Implementation

1. **Add column validation in planner** (~50 lines)
   - `validate_column_exists(table: &str, column: &str) -> Result<()>`
   - Call during plan_select, plan_update, plan_delete

2. **Add INSERT validation** (~20 lines)
   - Check `values.len() == columns.len()`
   - Validate each specified column exists

3. **Add type validation** (~100 lines)
   - Compare expression type with column type
   - Error on incompatible types

---

## 11. Priority Implementation Roadmap

### High Priority (Core Functionality)
1. **Decode all stored types** - Users can't see data they inserted
2. **Enforce NOT NULL** - Data integrity
3. **Enforce UNIQUE/PRIMARY KEY** - Data integrity
4. ~~**Arithmetic operators**~~ ✅ Implemented

### Medium Priority (Common Features)
5. ~~**JSON operators**~~ ✅ Implemented (basic -> and ->>)
6. **Date/Time literals** - Temporal data entry
7. ~~**Vector distance operators**~~ ✅ Implemented
8. **Subqueries** - Complex queries

### Lower Priority (Advanced Features)
9. **ALTER TABLE** - Schema evolution
10. **Transactions** - Multi-statement atomicity
11. **Window functions** - Analytics
12. **OUTER JOINs** - Query flexibility

---

## 12. Quick Wins

These can be fixed with minimal code changes:

1. **Add Date/Time/Timestamp decoding** (~20 lines in executor.rs)
2. **Add UUID decoding** (~5 lines)
3. ~~**Add arithmetic evaluation**~~ ✅ Implemented (~300 lines including tests)
4. **Add NOT NULL validation** (~15 lines in database.rs)

---

*Generated: 2024-12-14*
*Test: `test_comprehensive_all_column_types_and_operations`*
