//! # Constraint Enforcement Module
//!
//! This module provides constraint validation for TurDB's DML operations.
//! Constraints are defined in the schema but enforced during INSERT, UPDATE,
//! and DELETE operations to maintain data integrity.
//!
//! ## Supported Constraints
//!
//! | Constraint   | INSERT | UPDATE | DELETE | Description                          |
//! |--------------|--------|--------|--------|--------------------------------------|
//! | NOT NULL     | ✓      | ✓      | -      | Column cannot contain NULL values    |
//! | PRIMARY KEY  | ✓      | ✓      | -      | Unique, non-null row identifier      |
//! | UNIQUE       | ✓      | ✓      | -      | Column values must be unique         |
//! | FOREIGN KEY  | ✓      | ✓      | ✓      | References column in another table   |
//! | CHECK        | ✓      | ✓      | -      | Expression must evaluate to true     |
//! | DEFAULT      | ✓      | -      | -      | Default value when not specified     |
//!
//! ## Architecture
//!
//! Constraint validation follows a layered approach:
//!
//! ```text
//! ┌─────────────────────────────────────────┐
//! │          ConstraintValidator            │
//! │  (orchestrates all constraint checks)   │
//! ├─────────────────────────────────────────┤
//! │ ┌─────────┐ ┌────────┐ ┌─────────────┐ │
//! │ │NOT NULL │ │ UNIQUE │ │ PRIMARY KEY │ │
//! │ └─────────┘ └────────┘ └─────────────┘ │
//! │ ┌─────────────┐ ┌───────┐ ┌─────────┐  │
//! │ │ FOREIGN KEY │ │ CHECK │ │ DEFAULT │  │
//! │ └─────────────┘ └───────┘ └─────────┘  │
//! └─────────────────────────────────────────┘
//! ```
//!
//! ## Validation Order
//!
//! 1. Apply DEFAULT values for missing columns
//! 2. Validate NOT NULL constraints
//! 3. Validate CHECK constraints (expression evaluation)
//! 4. Validate UNIQUE constraints (requires index lookup)
//! 5. Validate PRIMARY KEY constraints (unique + not null)
//! 6. Validate FOREIGN KEY constraints (requires cross-table lookup)
//!
//! ## Error Handling
//!
//! All validation errors use `eyre` with rich context:
//! - Column name involved
//! - Constraint type violated
//! - Table name
//! - Offending value (when safe to display)
//!
//! Example error messages:
//! - "NOT NULL constraint violated on column 'email' in table 'users'"
//! - "UNIQUE constraint violated on column 'username' in table 'users': value 'alice' already exists"
//! - "CHECK constraint violated on column 'age' in table 'users': age must be >= 0"
//!
//! ## Performance Considerations
//!
//! - NOT NULL and CHECK: O(1) per column, no allocation
//! - UNIQUE/PRIMARY KEY: O(log n) index lookup per constraint
//! - FOREIGN KEY: O(log n) cross-table index lookup
//! - DEFAULT: O(1) value substitution
//!
//! Constraint validation is designed to:
//! - Minimize allocations (reuse buffers where possible)
//! - Fail fast (check NOT NULL before expensive index lookups)
//! - Provide clear error messages for debugging
//!
//! ## Usage
//!
//! ```rust,ignore
//! use turdb::constraints::ConstraintValidator;
//! use turdb::schema::TableDef;
//! use turdb::types::Value;
//!
//! let validator = ConstraintValidator::new(&table_def);
//!
//! // Validate a row before INSERT
//! let values = vec![Value::Int(1), Value::Text("alice".into())];
//! validator.validate_insert(&values)?;
//!
//! // Validate changes before UPDATE
//! validator.validate_update(&old_values, &new_values)?;
//!
//! // Check foreign key constraints before DELETE
//! validator.validate_delete(&values, &catalog)?;
//! ```
//!
//! ## Integration Points
//!
//! This module integrates with:
//! - `database::execute_insert` - validates before row insertion
//! - `database::execute_update` - validates after applying assignments
//! - `database::execute_delete` - validates FK references before deletion
//! - `btree::BTree` - used for UNIQUE/PK index lookups
//! - `schema::Catalog` - used for FK cross-table validation

use crate::database::OwnedValue;
use crate::schema::table::{Constraint, TableDef};
use eyre::{bail, Result};

pub struct ConstraintValidator<'a> {
    table: &'a TableDef,
}

impl<'a> ConstraintValidator<'a> {
    pub fn new(table: &'a TableDef) -> Self {
        Self { table }
    }

    pub fn validate_not_null(&self, values: &[OwnedValue]) -> Result<()> {
        for (idx, column) in self.table.columns().iter().enumerate() {
            if column.has_constraint(&Constraint::NotNull) {
                if let Some(value) = values.get(idx) {
                    if value.is_null() {
                        bail!(
                            "NOT NULL constraint violated on column '{}' in table '{}'",
                            column.name(),
                            self.table.name()
                        );
                    }
                } else {
                    bail!(
                        "NOT NULL constraint violated on column '{}' in table '{}': value not provided",
                        column.name(),
                        self.table.name()
                    );
                }
            }
        }
        Ok(())
    }

    pub fn apply_defaults(&self, values: &mut Vec<OwnedValue>) {
        let columns = self.table.columns();
        while values.len() < columns.len() {
            values.push(OwnedValue::Null);
        }

        for (idx, column) in columns.iter().enumerate() {
            if let Some(default_str) = column.default_value() {
                let value = &values[idx];
                if value.is_null() {
                    values[idx] = self.parse_default(default_str, column.data_type());
                }
            }
        }
    }

    fn parse_default(
        &self,
        default_str: &str,
        data_type: crate::records::types::DataType,
    ) -> OwnedValue {
        use crate::records::types::DataType;

        match data_type {
            DataType::Bool => match default_str.to_lowercase().as_str() {
                "true" | "1" => OwnedValue::Bool(true),
                _ => OwnedValue::Bool(false),
            },
            DataType::Int2 | DataType::Int4 | DataType::Int8 => default_str
                .parse::<i64>()
                .map(OwnedValue::Int)
                .unwrap_or(OwnedValue::Null),
            DataType::Float4 | DataType::Float8 => default_str
                .parse::<f64>()
                .map(OwnedValue::Float)
                .unwrap_or(OwnedValue::Null),
            DataType::Text | DataType::Varchar | DataType::Char => {
                OwnedValue::Text(default_str.to_string())
            }
            DataType::Uuid => Self::parse_uuid_default(default_str),
            DataType::Date => Self::parse_date_default(default_str),
            DataType::Time => Self::parse_time_default(default_str),
            DataType::Timestamp => Self::parse_timestamp_default(default_str),
            DataType::TimestampTz => Self::parse_timestamptz_default(default_str),
            DataType::Interval => Self::parse_interval_default(default_str),
            DataType::Point => Self::parse_point_default(default_str),
            DataType::Box => Self::parse_box_default(default_str),
            DataType::Circle => Self::parse_circle_default(default_str),
            DataType::Vector => Self::parse_vector_default(default_str),
            DataType::Jsonb => Self::parse_jsonb_default(default_str),
            DataType::MacAddr => Self::parse_macaddr_default(default_str),
            DataType::Inet4 => Self::parse_inet4_default(default_str),
            DataType::Inet6 => Self::parse_inet6_default(default_str),
            DataType::Decimal => Self::parse_decimal_default(default_str),
            DataType::Enum
            | DataType::Composite
            | DataType::Array
            | DataType::Int4Range
            | DataType::Int8Range
            | DataType::DateRange
            | DataType::TimestampRange
            | DataType::Blob => OwnedValue::Text(default_str.to_string()),
        }
    }

    fn parse_uuid_default(s: &str) -> OwnedValue {
        let hex_str: String = s.chars().filter(|c| c.is_ascii_hexdigit()).collect();
        if hex_str.len() != 32 {
            return OwnedValue::Null;
        }
        let mut bytes = [0u8; 16];
        for i in 0..16 {
            let byte_str = &hex_str[i * 2..i * 2 + 2];
            bytes[i] = match u8::from_str_radix(byte_str, 16) {
                Ok(b) => b,
                Err(_) => return OwnedValue::Null,
            };
        }
        OwnedValue::Uuid(bytes)
    }

    fn parse_date_default(s: &str) -> OwnedValue {
        let parts: Vec<&str> = s.split('-').collect();
        if parts.len() != 3 {
            return OwnedValue::Null;
        }
        let year: i32 = match parts[0].parse() {
            Ok(y) => y,
            Err(_) => return OwnedValue::Null,
        };
        let month: u32 = match parts[1].parse() {
            Ok(m) => m,
            Err(_) => return OwnedValue::Null,
        };
        let day: u32 = match parts[2].parse() {
            Ok(d) => d,
            Err(_) => return OwnedValue::Null,
        };
        let days = Self::days_from_ymd(year, month, day);
        OwnedValue::Date(days)
    }

    fn days_from_ymd(year: i32, month: u32, day: u32) -> i32 {
        let a = (14 - month as i32) / 12;
        let y = year + 4800 - a;
        let m = month as i32 + 12 * a - 3;
        let jdn = day as i32 + (153 * m + 2) / 5 + 365 * y + y / 4 - y / 100 + y / 400 - 32045;
        jdn - 2440588
    }

    fn parse_time_default(s: &str) -> OwnedValue {
        let parts: Vec<&str> = s.split(':').collect();
        if parts.len() < 2 {
            return OwnedValue::Null;
        }
        let hour: i64 = match parts[0].parse() {
            Ok(h) => h,
            Err(_) => return OwnedValue::Null,
        };
        let minute: i64 = match parts[1].parse() {
            Ok(m) => m,
            Err(_) => return OwnedValue::Null,
        };
        let (second, micros_frac) = if parts.len() > 2 {
            let sec_parts: Vec<&str> = parts[2].split('.').collect();
            let sec: i64 = sec_parts[0].parse().unwrap_or(0);
            let frac = if sec_parts.len() > 1 {
                let frac_str = sec_parts[1];
                let padded = format!("{:0<6}", frac_str);
                padded[..6].parse::<i64>().unwrap_or(0)
            } else {
                0
            };
            (sec, frac)
        } else {
            (0, 0)
        };
        let micros = hour * 3_600_000_000 + minute * 60_000_000 + second * 1_000_000 + micros_frac;
        OwnedValue::Time(micros)
    }

    fn parse_timestamp_default(s: &str) -> OwnedValue {
        let datetime_parts: Vec<&str> = s.split(&[' ', 'T'][..]).collect();
        if datetime_parts.is_empty() {
            return OwnedValue::Null;
        }
        let date_val = Self::parse_date_default(datetime_parts[0]);
        let days = match date_val {
            OwnedValue::Date(d) => d,
            _ => return OwnedValue::Null,
        };
        let time_micros = if datetime_parts.len() > 1 {
            match Self::parse_time_default(datetime_parts[1]) {
                OwnedValue::Time(t) => t,
                _ => 0,
            }
        } else {
            0
        };
        let epoch_micros = (days as i64) * 86_400_000_000 + time_micros;
        OwnedValue::Timestamp(epoch_micros)
    }

    fn parse_timestamptz_default(s: &str) -> OwnedValue {
        let ts_val = Self::parse_timestamp_default(s);
        match ts_val {
            OwnedValue::Timestamp(micros) => OwnedValue::TimestampTz(micros, 0),
            _ => OwnedValue::Null,
        }
    }

    fn parse_interval_default(s: &str) -> OwnedValue {
        let lower = s.to_lowercase();
        let mut months = 0i32;
        let mut days = 0i32;
        let mut micros = 0i64;
        let tokens: Vec<&str> = lower.split_whitespace().collect();
        let mut i = 0;
        while i < tokens.len() {
            if let Ok(num) = tokens[i].parse::<i64>() {
                if i + 1 < tokens.len() {
                    let unit = tokens[i + 1];
                    if unit.starts_with("year") {
                        months += (num * 12) as i32;
                    } else if unit.starts_with("month") {
                        months += num as i32;
                    } else if unit.starts_with("day") {
                        days += num as i32;
                    } else if unit.starts_with("hour") {
                        micros += num * 3_600_000_000;
                    } else if unit.starts_with("minute") || unit.starts_with("min") {
                        micros += num * 60_000_000;
                    } else if unit.starts_with("second") || unit.starts_with("sec") {
                        micros += num * 1_000_000;
                    }
                    i += 2;
                } else {
                    i += 1;
                }
            } else {
                i += 1;
            }
        }
        OwnedValue::Interval(micros, days, months)
    }

    fn parse_point_default(s: &str) -> OwnedValue {
        let stripped = s.trim().trim_matches(|c| c == '(' || c == ')');
        let parts: Vec<&str> = stripped.split(',').collect();
        if parts.len() != 2 {
            return OwnedValue::Null;
        }
        let x: f64 = match parts[0].trim().parse() {
            Ok(v) => v,
            Err(_) => return OwnedValue::Null,
        };
        let y: f64 = match parts[1].trim().parse() {
            Ok(v) => v,
            Err(_) => return OwnedValue::Null,
        };
        OwnedValue::Point(x, y)
    }

    fn parse_box_default(s: &str) -> OwnedValue {
        let stripped = s.replace(&['(', ')', ' '][..], "");
        let coords: Vec<f64> = stripped.split(',').filter_map(|p| p.parse().ok()).collect();
        if coords.len() != 4 {
            return OwnedValue::Null;
        }
        OwnedValue::Box((coords[0], coords[1]), (coords[2], coords[3]))
    }

    fn parse_circle_default(s: &str) -> OwnedValue {
        let stripped = s
            .trim()
            .trim_matches(|c| c == '<' || c == '>')
            .replace(&['(', ')'][..], "");
        let parts: Vec<f64> = stripped.split(',').filter_map(|p| p.trim().parse().ok()).collect();
        if parts.len() != 3 {
            return OwnedValue::Null;
        }
        OwnedValue::Circle((parts[0], parts[1]), parts[2])
    }

    fn parse_vector_default(s: &str) -> OwnedValue {
        let stripped = s.trim().trim_matches(|c| c == '[' || c == ']');
        let floats: Vec<f32> = stripped
            .split(',')
            .filter_map(|p| p.trim().parse().ok())
            .collect();
        if floats.is_empty() {
            return OwnedValue::Null;
        }
        OwnedValue::Vector(floats)
    }

    fn parse_jsonb_default(s: &str) -> OwnedValue {
        use crate::records::jsonb::JsonbBuilder;
        let trimmed = s.trim();
        if trimmed == "null" {
            return OwnedValue::Jsonb(JsonbBuilder::new_null().build());
        }
        if trimmed == "true" {
            return OwnedValue::Jsonb(JsonbBuilder::new_bool(true).build());
        }
        if trimmed == "false" {
            return OwnedValue::Jsonb(JsonbBuilder::new_bool(false).build());
        }
        if let Ok(n) = trimmed.parse::<f64>() {
            return OwnedValue::Jsonb(JsonbBuilder::new_number(n).build());
        }
        if trimmed.starts_with('"') && trimmed.ends_with('"') && trimmed.len() >= 2 {
            let inner = &trimmed[1..trimmed.len() - 1];
            return OwnedValue::Jsonb(JsonbBuilder::new_string(inner).build());
        }
        OwnedValue::Text(s.to_string())
    }

    fn parse_macaddr_default(s: &str) -> OwnedValue {
        let hex_parts: Vec<&str> = s.split(':').collect();
        if hex_parts.len() != 6 {
            let hex_parts: Vec<&str> = s.split('-').collect();
            if hex_parts.len() != 6 {
                return OwnedValue::Null;
            }
            return Self::parse_macaddr_parts(&hex_parts);
        }
        Self::parse_macaddr_parts(&hex_parts)
    }

    fn parse_macaddr_parts(parts: &[&str]) -> OwnedValue {
        let mut bytes = [0u8; 6];
        for (i, part) in parts.iter().enumerate() {
            bytes[i] = match u8::from_str_radix(part, 16) {
                Ok(b) => b,
                Err(_) => return OwnedValue::Null,
            };
        }
        OwnedValue::MacAddr(bytes)
    }

    fn parse_inet4_default(s: &str) -> OwnedValue {
        let parts: Vec<&str> = s.split('/').next().unwrap_or(s).split('.').collect();
        if parts.len() != 4 {
            return OwnedValue::Null;
        }
        let mut bytes = [0u8; 4];
        for (i, part) in parts.iter().enumerate() {
            bytes[i] = match part.parse() {
                Ok(b) => b,
                Err(_) => return OwnedValue::Null,
            };
        }
        OwnedValue::Inet4(bytes)
    }

    fn parse_inet6_default(s: &str) -> OwnedValue {
        let addr_str = s.split('/').next().unwrap_or(s);
        if addr_str == "::1" {
            let mut bytes = [0u8; 16];
            bytes[15] = 1;
            return OwnedValue::Inet6(bytes);
        }
        if addr_str == "::" {
            return OwnedValue::Inet6([0u8; 16]);
        }
        let parts: Vec<&str> = addr_str.split("::").collect();
        let mut bytes = [0u8; 16];
        if parts.len() == 2 {
            let left_groups: Vec<u16> = parts[0]
                .split(':')
                .filter(|p| !p.is_empty())
                .filter_map(|p| u16::from_str_radix(p, 16).ok())
                .collect();
            let right_groups: Vec<u16> = parts[1]
                .split(':')
                .filter(|p| !p.is_empty())
                .filter_map(|p| u16::from_str_radix(p, 16).ok())
                .collect();
            for (i, &group) in left_groups.iter().enumerate() {
                let idx = i * 2;
                bytes[idx] = (group >> 8) as u8;
                bytes[idx + 1] = group as u8;
            }
            let start_right = 16 - right_groups.len() * 2;
            for (i, &group) in right_groups.iter().enumerate() {
                let idx = start_right + i * 2;
                bytes[idx] = (group >> 8) as u8;
                bytes[idx + 1] = group as u8;
            }
        } else {
            let groups: Vec<u16> = addr_str
                .split(':')
                .filter_map(|p| u16::from_str_radix(p, 16).ok())
                .collect();
            if groups.len() != 8 {
                return OwnedValue::Null;
            }
            for (i, &group) in groups.iter().enumerate() {
                let idx = i * 2;
                bytes[idx] = (group >> 8) as u8;
                bytes[idx + 1] = group as u8;
            }
        }
        OwnedValue::Inet6(bytes)
    }

    fn parse_decimal_default(s: &str) -> OwnedValue {
        let trimmed = s.trim();
        let parts: Vec<&str> = trimmed.split('.').collect();
        let (int_part, frac_part, scale) = if parts.len() == 2 {
            (parts[0], parts[1], parts[1].len() as i16)
        } else {
            (parts[0], "", 0)
        };
        let int_val: i128 = match int_part.parse() {
            Ok(v) => v,
            Err(_) => return OwnedValue::Null,
        };
        let frac_val: i128 = if !frac_part.is_empty() {
            match frac_part.parse() {
                Ok(v) => v,
                Err(_) => return OwnedValue::Null,
            }
        } else {
            0
        };
        let multiplier = 10i128.pow(scale as u32);
        let is_negative = int_val < 0;
        let digits = if is_negative {
            int_val * multiplier - frac_val
        } else {
            int_val * multiplier + frac_val
        };
        OwnedValue::Decimal(digits, scale)
    }

    pub fn validate_insert(&self, values: &mut Vec<OwnedValue>) -> Result<()> {
        self.apply_defaults(values);
        self.validate_not_null(values)?;
        self.validate_string_lengths(values)?;
        Ok(())
    }

    pub fn validate_update(&self, values: &[OwnedValue]) -> Result<()> {
        self.validate_not_null(values)?;
        self.validate_primary_key(values)?;
        self.validate_string_lengths(values)?;
        Ok(())
    }

    pub fn validate_string_lengths(&self, values: &[OwnedValue]) -> Result<()> {
        for (idx, column) in self.table.columns().iter().enumerate() {
            if let (Some(max_len), Some(OwnedValue::Text(text))) =
                (column.max_length(), values.get(idx))
            {
                if text.len() > max_len as usize {
                    bail!(
                        "value for column '{}' in table '{}' exceeds maximum length {} (actual: {})",
                        column.name(),
                        self.table.name(),
                        max_len,
                        text.len()
                    );
                }
            }
        }
        Ok(())
    }

    pub fn table(&self) -> &TableDef {
        self.table
    }

    pub fn validate_primary_key(&self, values: &[OwnedValue]) -> Result<()> {
        for (idx, column) in self.table.columns().iter().enumerate() {
            if column.has_constraint(&Constraint::PrimaryKey) {
                let is_null = values.get(idx).map(|v| v.is_null()).unwrap_or(true);

                if is_null {
                    bail!(
                        "PRIMARY KEY constraint violated on column '{}' in table '{}': value cannot be NULL",
                        column.name(),
                        self.table.name()
                    );
                }
            }
        }
        Ok(())
    }

    pub fn validate_unique<F>(&self, values: &[OwnedValue], exists_checker: F) -> Result<()>
    where
        F: Fn(usize, &OwnedValue) -> bool,
    {
        for (idx, column) in self.table.columns().iter().enumerate() {
            if column.has_constraint(&Constraint::Unique) {
                if let Some(value) = values.get(idx) {
                    if value.is_null() {
                        continue;
                    }

                    if exists_checker(idx, value) {
                        bail!(
                            "UNIQUE constraint violated on column '{}' in table '{}': value already exists",
                            column.name(),
                            self.table.name()
                        );
                    }
                }
            }
        }
        Ok(())
    }

    pub fn validate_check<F>(&self, values: &[OwnedValue], check_evaluator: F) -> Result<()>
    where
        F: Fn(&str, &[OwnedValue]) -> bool,
    {
        for column in self.table.columns() {
            for constraint in column.constraints() {
                if let Constraint::Check(expr) = constraint {
                    if !check_evaluator(expr, values) {
                        bail!(
                            "CHECK constraint violated on column '{}' in table '{}': {}",
                            column.name(),
                            self.table.name(),
                            expr
                        );
                    }
                }
            }
        }
        Ok(())
    }

    pub fn validate_foreign_key<F>(&self, values: &[OwnedValue], fk_checker: F) -> Result<()>
    where
        F: Fn(&str, &str, &OwnedValue) -> bool,
    {
        for (idx, column) in self.table.columns().iter().enumerate() {
            for constraint in column.constraints() {
                if let Constraint::ForeignKey {
                    table: fk_table,
                    column: fk_column,
                } = constraint
                {
                    if let Some(value) = values.get(idx) {
                        if value.is_null() {
                            continue;
                        }

                        if !fk_checker(fk_table, fk_column, value) {
                            bail!(
                                "FOREIGN KEY constraint violated on column '{}' in table '{}': referenced value not found in {}.{}",
                                column.name(),
                                self.table.name(),
                                fk_table,
                                fk_column
                            );
                        }
                    }
                }
            }
        }
        Ok(())
    }

    pub fn validate_delete<F>(&self, values: &[OwnedValue], is_referenced_checker: F) -> Result<()>
    where
        F: Fn(&str, &str, &OwnedValue) -> Option<String>,
    {
        for (idx, column) in self.table.columns().iter().enumerate() {
            if column.has_constraint(&Constraint::PrimaryKey) {
                if let Some(value) = values.get(idx) {
                    if let Some(referencing_table) =
                        is_referenced_checker(self.table.name(), column.name(), value)
                    {
                        bail!(
                            "cannot delete row from '{}': column '{}' is referenced by table '{}'",
                            self.table.name(),
                            column.name(),
                            referencing_table
                        );
                    }
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::records::types::DataType;
    use crate::schema::table::ColumnDef;

    #[test]
    fn test_not_null_constraint_rejects_null_value() {
        let columns = vec![
            ColumnDef::new("id", DataType::Int8).with_constraint(Constraint::NotNull),
            ColumnDef::new("name", DataType::Text).with_constraint(Constraint::NotNull),
        ];
        let table = TableDef::new(1, "users", columns);
        let validator = ConstraintValidator::new(&table);

        let values = vec![OwnedValue::Int(1), OwnedValue::Null];

        let result = validator.validate_not_null(&values);
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("NOT NULL constraint violated"));
        assert!(err.contains("name"));
    }

    #[test]
    fn test_not_null_constraint_accepts_non_null_values() {
        let columns = vec![
            ColumnDef::new("id", DataType::Int8).with_constraint(Constraint::NotNull),
            ColumnDef::new("name", DataType::Text).with_constraint(Constraint::NotNull),
        ];
        let table = TableDef::new(1, "users", columns);
        let validator = ConstraintValidator::new(&table);

        let values = vec![OwnedValue::Int(1), OwnedValue::Text("alice".to_string())];

        let result = validator.validate_not_null(&values);
        assert!(result.is_ok());
    }

    #[test]
    fn test_nullable_column_accepts_null() {
        let columns = vec![
            ColumnDef::new("id", DataType::Int8).with_constraint(Constraint::NotNull),
            ColumnDef::new("bio", DataType::Text),
        ];
        let table = TableDef::new(1, "users", columns);
        let validator = ConstraintValidator::new(&table);

        let values = vec![OwnedValue::Int(1), OwnedValue::Null];

        let result = validator.validate_not_null(&values);
        assert!(result.is_ok());
    }

    #[test]
    fn test_not_null_constraint_rejects_missing_value() {
        let columns = vec![
            ColumnDef::new("id", DataType::Int8).with_constraint(Constraint::NotNull),
            ColumnDef::new("name", DataType::Text).with_constraint(Constraint::NotNull),
        ];
        let table = TableDef::new(1, "users", columns);
        let validator = ConstraintValidator::new(&table);

        let values = vec![OwnedValue::Int(1)];

        let result = validator.validate_not_null(&values);
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("NOT NULL constraint violated"));
        assert!(err.contains("value not provided"));
    }

    #[test]
    fn test_apply_defaults_fills_missing_values() {
        let columns = vec![
            ColumnDef::new("id", DataType::Int8).with_constraint(Constraint::NotNull),
            ColumnDef::new("status", DataType::Text).with_default("active"),
        ];
        let table = TableDef::new(1, "users", columns);
        let validator = ConstraintValidator::new(&table);

        let mut values = vec![OwnedValue::Int(1)];
        validator.apply_defaults(&mut values);

        assert_eq!(values.len(), 2);
        assert_eq!(values[1], OwnedValue::Text("active".to_string()));
    }

    #[test]
    fn test_apply_defaults_replaces_null_with_default() {
        let columns = vec![
            ColumnDef::new("id", DataType::Int8).with_constraint(Constraint::NotNull),
            ColumnDef::new("count", DataType::Int4).with_default("0"),
        ];
        let table = TableDef::new(1, "counters", columns);
        let validator = ConstraintValidator::new(&table);

        let mut values = vec![OwnedValue::Int(1), OwnedValue::Null];
        validator.apply_defaults(&mut values);

        assert_eq!(values[1], OwnedValue::Int(0));
    }

    #[test]
    fn test_apply_defaults_does_not_overwrite_explicit_values() {
        let columns = vec![
            ColumnDef::new("id", DataType::Int8),
            ColumnDef::new("status", DataType::Text).with_default("active"),
        ];
        let table = TableDef::new(1, "users", columns);
        let validator = ConstraintValidator::new(&table);

        let mut values = vec![OwnedValue::Int(1), OwnedValue::Text("inactive".to_string())];
        validator.apply_defaults(&mut values);

        assert_eq!(values[1], OwnedValue::Text("inactive".to_string()));
    }

    #[test]
    fn test_validate_string_length_varchar_rejects_too_long() {
        let columns = vec![
            ColumnDef::new("id", DataType::Int8),
            ColumnDef::new("code", DataType::Varchar).with_max_length(5),
        ];
        let table = TableDef::new(1, "products", columns);
        let validator = ConstraintValidator::new(&table);

        let values = vec![OwnedValue::Int(1), OwnedValue::Text("ABCDEFG".to_string())];

        let result = validator.validate_string_lengths(&values);
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("exceeds maximum length"));
        assert!(err.contains("code"));
        assert!(err.contains("5"));
    }

    #[test]
    fn test_validate_string_length_varchar_accepts_valid() {
        let columns = vec![
            ColumnDef::new("id", DataType::Int8),
            ColumnDef::new("code", DataType::Varchar).with_max_length(10),
        ];
        let table = TableDef::new(1, "products", columns);
        let validator = ConstraintValidator::new(&table);

        let values = vec![OwnedValue::Int(1), OwnedValue::Text("ABCDE".to_string())];

        let result = validator.validate_string_lengths(&values);
        assert!(result.is_ok());
    }

    #[test]
    fn test_validate_string_length_char_rejects_too_long() {
        let columns = vec![
            ColumnDef::new("id", DataType::Int8),
            ColumnDef::new("status", DataType::Char).with_max_length(1),
        ];
        let table = TableDef::new(1, "orders", columns);
        let validator = ConstraintValidator::new(&table);

        let values = vec![OwnedValue::Int(1), OwnedValue::Text("AB".to_string())];

        let result = validator.validate_string_lengths(&values);
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("exceeds maximum length"));
    }

    #[test]
    fn test_validate_string_length_ignores_null() {
        let columns = vec![
            ColumnDef::new("id", DataType::Int8),
            ColumnDef::new("code", DataType::Varchar).with_max_length(5),
        ];
        let table = TableDef::new(1, "products", columns);
        let validator = ConstraintValidator::new(&table);

        let values = vec![OwnedValue::Int(1), OwnedValue::Null];

        let result = validator.validate_string_lengths(&values);
        assert!(result.is_ok());
    }

    #[test]
    fn test_validate_insert_applies_defaults_then_checks_constraints() {
        let columns = vec![
            ColumnDef::new("id", DataType::Int8).with_constraint(Constraint::NotNull),
            ColumnDef::new("status", DataType::Varchar)
                .with_max_length(10)
                .with_default("active"),
        ];
        let table = TableDef::new(1, "users", columns);
        let validator = ConstraintValidator::new(&table);

        let mut values = vec![OwnedValue::Int(1)];
        let result = validator.validate_insert(&mut values);

        assert!(result.is_ok());
        assert_eq!(values.len(), 2);
        assert_eq!(values[1], OwnedValue::Text("active".to_string()));
    }

    #[test]
    fn test_validate_insert_fails_on_not_null_after_defaults() {
        let columns = vec![
            ColumnDef::new("id", DataType::Int8).with_constraint(Constraint::NotNull),
            ColumnDef::new("name", DataType::Text).with_constraint(Constraint::NotNull),
        ];
        let table = TableDef::new(1, "users", columns);
        let validator = ConstraintValidator::new(&table);

        let mut values = vec![OwnedValue::Int(1)];
        let result = validator.validate_insert(&mut values);

        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("NOT NULL"));
    }

    #[test]
    fn test_validate_insert_fails_on_string_length() {
        let columns = vec![
            ColumnDef::new("id", DataType::Int8),
            ColumnDef::new("code", DataType::Varchar).with_max_length(3),
        ];
        let table = TableDef::new(1, "products", columns);
        let validator = ConstraintValidator::new(&table);

        let mut values = vec![OwnedValue::Int(1), OwnedValue::Text("ABCDEFG".to_string())];
        let result = validator.validate_insert(&mut values);

        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("exceeds maximum length"));
    }

    #[test]
    fn test_validate_primary_key_rejects_null() {
        let columns = vec![
            ColumnDef::new("id", DataType::Int8)
                .with_constraint(Constraint::PrimaryKey)
                .with_constraint(Constraint::NotNull),
            ColumnDef::new("name", DataType::Text),
        ];
        let table = TableDef::new(1, "users", columns);
        let validator = ConstraintValidator::new(&table);

        let values = vec![OwnedValue::Null, OwnedValue::Text("alice".to_string())];

        let result = validator.validate_primary_key(&values);
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("PRIMARY KEY"));
        assert!(err.contains("cannot be NULL"));
    }

    #[test]
    fn test_validate_primary_key_accepts_non_null() {
        let columns = vec![
            ColumnDef::new("id", DataType::Int8)
                .with_constraint(Constraint::PrimaryKey)
                .with_constraint(Constraint::NotNull),
            ColumnDef::new("name", DataType::Text),
        ];
        let table = TableDef::new(1, "users", columns);
        let validator = ConstraintValidator::new(&table);

        let values = vec![OwnedValue::Int(1), OwnedValue::Text("alice".to_string())];

        let result = validator.validate_primary_key(&values);
        assert!(result.is_ok());
    }

    #[test]
    fn test_validate_unique_with_lookup_rejects_duplicate() {
        let columns = vec![
            ColumnDef::new("id", DataType::Int8),
            ColumnDef::new("email", DataType::Text).with_constraint(Constraint::Unique),
        ];
        let table = TableDef::new(1, "users", columns);
        let validator = ConstraintValidator::new(&table);

        let values = vec![
            OwnedValue::Int(1),
            OwnedValue::Text("alice@test.com".to_string()),
        ];

        let exists_checker = |_col_idx: usize, value: &OwnedValue| -> bool {
            matches!(value, OwnedValue::Text(s) if s == "alice@test.com")
        };

        let result = validator.validate_unique(&values, exists_checker);
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("UNIQUE constraint violated"));
        assert!(err.contains("email"));
    }

    #[test]
    fn test_validate_unique_with_lookup_accepts_new_value() {
        let columns = vec![
            ColumnDef::new("id", DataType::Int8),
            ColumnDef::new("email", DataType::Text).with_constraint(Constraint::Unique),
        ];
        let table = TableDef::new(1, "users", columns);
        let validator = ConstraintValidator::new(&table);

        let values = vec![
            OwnedValue::Int(1),
            OwnedValue::Text("bob@test.com".to_string()),
        ];

        let exists_checker = |_col_idx: usize, value: &OwnedValue| -> bool {
            matches!(value, OwnedValue::Text(s) if s == "alice@test.com")
        };

        let result = validator.validate_unique(&values, exists_checker);
        assert!(result.is_ok());
    }

    #[test]
    fn test_validate_unique_ignores_null() {
        let columns = vec![
            ColumnDef::new("id", DataType::Int8),
            ColumnDef::new("email", DataType::Text).with_constraint(Constraint::Unique),
        ];
        let table = TableDef::new(1, "users", columns);
        let validator = ConstraintValidator::new(&table);

        let values = vec![OwnedValue::Int(1), OwnedValue::Null];

        let exists_checker = |_col_idx: usize, _value: &OwnedValue| -> bool { true };

        let result = validator.validate_unique(&values, exists_checker);
        assert!(result.is_ok());
    }

    #[test]
    fn test_validate_check_rejects_failing_expression() {
        let columns = vec![
            ColumnDef::new("id", DataType::Int8),
            ColumnDef::new("age", DataType::Int4)
                .with_constraint(Constraint::Check("age >= 0".to_string())),
        ];
        let table = TableDef::new(1, "users", columns);
        let validator = ConstraintValidator::new(&table);

        let values = vec![OwnedValue::Int(1), OwnedValue::Int(-5)];

        let check_evaluator = |_expr: &str, vals: &[OwnedValue]| -> bool {
            if let Some(OwnedValue::Int(age)) = vals.get(1) {
                *age >= 0
            } else {
                true
            }
        };

        let result = validator.validate_check(&values, check_evaluator);
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("CHECK constraint violated"));
        assert!(err.contains("age >= 0"));
    }

    #[test]
    fn test_validate_check_accepts_passing_expression() {
        let columns = vec![
            ColumnDef::new("id", DataType::Int8),
            ColumnDef::new("age", DataType::Int4)
                .with_constraint(Constraint::Check("age >= 0".to_string())),
        ];
        let table = TableDef::new(1, "users", columns);
        let validator = ConstraintValidator::new(&table);

        let values = vec![OwnedValue::Int(1), OwnedValue::Int(25)];

        let check_evaluator = |_expr: &str, vals: &[OwnedValue]| -> bool {
            if let Some(OwnedValue::Int(age)) = vals.get(1) {
                *age >= 0
            } else {
                true
            }
        };

        let result = validator.validate_check(&values, check_evaluator);
        assert!(result.is_ok());
    }

    #[test]
    fn test_validate_check_multiple_constraints() {
        let columns = vec![
            ColumnDef::new("id", DataType::Int8),
            ColumnDef::new("age", DataType::Int4)
                .with_constraint(Constraint::Check("age >= 0".to_string())),
            ColumnDef::new("status", DataType::Text).with_constraint(Constraint::Check(
                "status IN ('active', 'inactive')".to_string(),
            )),
        ];
        let table = TableDef::new(1, "users", columns);
        let validator = ConstraintValidator::new(&table);

        let values = vec![
            OwnedValue::Int(1),
            OwnedValue::Int(25),
            OwnedValue::Text("pending".to_string()),
        ];

        let check_evaluator = |expr: &str, vals: &[OwnedValue]| -> bool {
            if expr.contains("age >= 0") {
                if let Some(OwnedValue::Int(age)) = vals.get(1) {
                    return *age >= 0;
                }
            }
            if expr.contains("status IN") {
                if let Some(OwnedValue::Text(status)) = vals.get(2) {
                    return status == "active" || status == "inactive";
                }
            }
            true
        };

        let result = validator.validate_check(&values, check_evaluator);
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("CHECK constraint violated"));
        assert!(err.contains("status IN"));
    }

    #[test]
    fn test_validate_check_ignores_null() {
        let columns = vec![
            ColumnDef::new("id", DataType::Int8),
            ColumnDef::new("age", DataType::Int4)
                .with_constraint(Constraint::Check("age >= 0".to_string())),
        ];
        let table = TableDef::new(1, "users", columns);
        let validator = ConstraintValidator::new(&table);

        let values = vec![OwnedValue::Int(1), OwnedValue::Null];

        let check_evaluator = |_expr: &str, vals: &[OwnedValue]| -> bool {
            if let Some(OwnedValue::Int(age)) = vals.get(1) {
                *age >= 0
            } else {
                true
            }
        };

        let result = validator.validate_check(&values, check_evaluator);
        assert!(result.is_ok());
    }

    #[test]
    fn test_validate_foreign_key_rejects_missing_reference() {
        let columns = vec![
            ColumnDef::new("id", DataType::Int8),
            ColumnDef::new("user_id", DataType::Int8).with_constraint(Constraint::ForeignKey {
                table: "users".to_string(),
                column: "id".to_string(),
            }),
        ];
        let table = TableDef::new(1, "orders", columns);
        let validator = ConstraintValidator::new(&table);

        let values = vec![OwnedValue::Int(1), OwnedValue::Int(999)];

        let fk_checker = |_fk_table: &str, _fk_column: &str, value: &OwnedValue| -> bool {
            matches!(value, OwnedValue::Int(id) if *id == 1 || *id == 2)
        };

        let result = validator.validate_foreign_key(&values, fk_checker);
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("FOREIGN KEY constraint violated"));
        assert!(err.contains("user_id"));
        assert!(err.contains("users.id"));
    }

    #[test]
    fn test_validate_foreign_key_accepts_valid_reference() {
        let columns = vec![
            ColumnDef::new("id", DataType::Int8),
            ColumnDef::new("user_id", DataType::Int8).with_constraint(Constraint::ForeignKey {
                table: "users".to_string(),
                column: "id".to_string(),
            }),
        ];
        let table = TableDef::new(1, "orders", columns);
        let validator = ConstraintValidator::new(&table);

        let values = vec![OwnedValue::Int(1), OwnedValue::Int(2)];

        let fk_checker = |_fk_table: &str, _fk_column: &str, value: &OwnedValue| -> bool {
            matches!(value, OwnedValue::Int(id) if *id == 1 || *id == 2)
        };

        let result = validator.validate_foreign_key(&values, fk_checker);
        assert!(result.is_ok());
    }

    #[test]
    fn test_validate_foreign_key_ignores_null() {
        let columns = vec![
            ColumnDef::new("id", DataType::Int8),
            ColumnDef::new("user_id", DataType::Int8).with_constraint(Constraint::ForeignKey {
                table: "users".to_string(),
                column: "id".to_string(),
            }),
        ];
        let table = TableDef::new(1, "orders", columns);
        let validator = ConstraintValidator::new(&table);

        let values = vec![OwnedValue::Int(1), OwnedValue::Null];

        let fk_checker = |_fk_table: &str, _fk_column: &str, _value: &OwnedValue| -> bool { false };

        let result = validator.validate_foreign_key(&values, fk_checker);
        assert!(result.is_ok());
    }

    #[test]
    fn test_validate_foreign_key_multiple_fks() {
        let columns = vec![
            ColumnDef::new("id", DataType::Int8),
            ColumnDef::new("user_id", DataType::Int8).with_constraint(Constraint::ForeignKey {
                table: "users".to_string(),
                column: "id".to_string(),
            }),
            ColumnDef::new("product_id", DataType::Int8).with_constraint(Constraint::ForeignKey {
                table: "products".to_string(),
                column: "id".to_string(),
            }),
        ];
        let table = TableDef::new(1, "order_items", columns);
        let validator = ConstraintValidator::new(&table);

        let values = vec![OwnedValue::Int(1), OwnedValue::Int(1), OwnedValue::Int(999)];

        let fk_checker = |fk_table: &str, _fk_column: &str, value: &OwnedValue| -> bool {
            if fk_table == "users" {
                return matches!(value, OwnedValue::Int(id) if *id == 1);
            }
            if fk_table == "products" {
                return matches!(value, OwnedValue::Int(id) if *id == 100);
            }
            false
        };

        let result = validator.validate_foreign_key(&values, fk_checker);
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("FOREIGN KEY constraint violated"));
        assert!(err.contains("product_id"));
    }

    #[test]
    fn test_validate_delete_rejects_referenced_row() {
        let columns = vec![
            ColumnDef::new("id", DataType::Int8).with_constraint(Constraint::PrimaryKey),
            ColumnDef::new("name", DataType::Text),
        ];
        let table = TableDef::new(1, "users", columns);
        let validator = ConstraintValidator::new(&table);

        let values = vec![OwnedValue::Int(1), OwnedValue::Text("alice".to_string())];

        let is_referenced_checker =
            |table_name: &str, col_name: &str, value: &OwnedValue| -> Option<String> {
                if table_name == "users" && col_name == "id" && matches!(value, OwnedValue::Int(1))
                {
                    return Some("orders".to_string());
                }
                None
            };

        let result = validator.validate_delete(&values, is_referenced_checker);
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("cannot delete"));
        assert!(err.contains("referenced by"));
        assert!(err.contains("orders"));
    }

    #[test]
    fn test_validate_delete_accepts_unreferenced_row() {
        let columns = vec![
            ColumnDef::new("id", DataType::Int8).with_constraint(Constraint::PrimaryKey),
            ColumnDef::new("name", DataType::Text),
        ];
        let table = TableDef::new(1, "users", columns);
        let validator = ConstraintValidator::new(&table);

        let values = vec![OwnedValue::Int(999), OwnedValue::Text("bob".to_string())];

        let is_referenced_checker =
            |_table_name: &str, _col_name: &str, _value: &OwnedValue| -> Option<String> { None };

        let result = validator.validate_delete(&values, is_referenced_checker);
        assert!(result.is_ok());
    }

    #[test]
    fn test_validate_delete_checks_pk_columns_only() {
        let columns = vec![
            ColumnDef::new("id", DataType::Int8).with_constraint(Constraint::PrimaryKey),
            ColumnDef::new("name", DataType::Text),
        ];
        let table = TableDef::new(1, "users", columns);
        let validator = ConstraintValidator::new(&table);

        let values = vec![OwnedValue::Int(1), OwnedValue::Text("alice".to_string())];

        let is_referenced_checker =
            |table_name: &str, col_name: &str, _value: &OwnedValue| -> Option<String> {
                if col_name == "id" {
                    assert_eq!(table_name, "users");
                }
                None
            };

        let result = validator.validate_delete(&values, is_referenced_checker);
        assert!(result.is_ok());
    }

    #[test]
    fn test_parse_uuid_default() {
        let result = ConstraintValidator::parse_uuid_default("123e4567-e89b-12d3-a456-426614174000");
        assert!(matches!(result, OwnedValue::Uuid(_)));
        if let OwnedValue::Uuid(bytes) = result {
            assert_eq!(bytes[0], 0x12);
            assert_eq!(bytes[1], 0x3e);
        }
    }

    #[test]
    fn test_parse_uuid_default_invalid() {
        let result = ConstraintValidator::parse_uuid_default("invalid-uuid");
        assert!(matches!(result, OwnedValue::Null));
    }

    #[test]
    fn test_parse_date_default() {
        let result = ConstraintValidator::parse_date_default("2023-12-25");
        assert!(matches!(result, OwnedValue::Date(_)));
        if let OwnedValue::Date(days) = result {
            assert!(days > 0);
        }
    }

    #[test]
    fn test_parse_time_default() {
        let result = ConstraintValidator::parse_time_default("12:30:45");
        assert!(matches!(result, OwnedValue::Time(_)));
        if let OwnedValue::Time(micros) = result {
            assert_eq!(micros, 12 * 3_600_000_000 + 30 * 60_000_000 + 45 * 1_000_000);
        }
    }

    #[test]
    fn test_parse_timestamp_default() {
        let result = ConstraintValidator::parse_timestamp_default("2023-12-25 12:30:45");
        assert!(matches!(result, OwnedValue::Timestamp(_)));
    }

    #[test]
    fn test_parse_interval_default() {
        let result = ConstraintValidator::parse_interval_default("1 year 2 months 3 days");
        assert!(matches!(result, OwnedValue::Interval(_, _, _)));
        if let OwnedValue::Interval(micros, days, months) = result {
            assert_eq!(months, 14);
            assert_eq!(days, 3);
            assert_eq!(micros, 0);
        }
    }

    #[test]
    fn test_parse_point_default() {
        let result = ConstraintValidator::parse_point_default("(1.5, 2.5)");
        assert!(matches!(result, OwnedValue::Point(_, _)));
        if let OwnedValue::Point(x, y) = result {
            assert!((x - 1.5).abs() < f64::EPSILON);
            assert!((y - 2.5).abs() < f64::EPSILON);
        }
    }

    #[test]
    fn test_parse_vector_default() {
        let result = ConstraintValidator::parse_vector_default("[1.0, 2.0, 3.0]");
        assert!(matches!(result, OwnedValue::Vector(_)));
        if let OwnedValue::Vector(v) = result {
            assert_eq!(v.len(), 3);
            assert!((v[0] - 1.0).abs() < f32::EPSILON);
            assert!((v[1] - 2.0).abs() < f32::EPSILON);
            assert!((v[2] - 3.0).abs() < f32::EPSILON);
        }
    }

    #[test]
    fn test_parse_jsonb_default_null() {
        let result = ConstraintValidator::parse_jsonb_default("null");
        assert!(matches!(result, OwnedValue::Jsonb(_)));
    }

    #[test]
    fn test_parse_jsonb_default_bool() {
        let result = ConstraintValidator::parse_jsonb_default("true");
        assert!(matches!(result, OwnedValue::Jsonb(_)));
    }

    #[test]
    fn test_parse_jsonb_default_number() {
        let result = ConstraintValidator::parse_jsonb_default("42");
        assert!(matches!(result, OwnedValue::Jsonb(_)));
    }

    #[test]
    fn test_parse_macaddr_default() {
        let result = ConstraintValidator::parse_macaddr_default("00:11:22:33:44:55");
        assert!(matches!(result, OwnedValue::MacAddr(_)));
        if let OwnedValue::MacAddr(bytes) = result {
            assert_eq!(bytes, [0x00, 0x11, 0x22, 0x33, 0x44, 0x55]);
        }
    }

    #[test]
    fn test_parse_inet4_default() {
        let result = ConstraintValidator::parse_inet4_default("192.168.1.1");
        assert!(matches!(result, OwnedValue::Inet4(_)));
        if let OwnedValue::Inet4(bytes) = result {
            assert_eq!(bytes, [192, 168, 1, 1]);
        }
    }

    #[test]
    fn test_parse_inet6_default_loopback() {
        let result = ConstraintValidator::parse_inet6_default("::1");
        assert!(matches!(result, OwnedValue::Inet6(_)));
        if let OwnedValue::Inet6(bytes) = result {
            assert_eq!(bytes[15], 1);
            for &b in &bytes[0..15] {
                assert_eq!(b, 0);
            }
        }
    }

    #[test]
    fn test_parse_decimal_default() {
        let result = ConstraintValidator::parse_decimal_default("123.45");
        assert!(matches!(result, OwnedValue::Decimal(_, _)));
        if let OwnedValue::Decimal(digits, scale) = result {
            assert_eq!(digits, 12345);
            assert_eq!(scale, 2);
        }
    }

    #[test]
    fn test_parse_box_default() {
        let result = ConstraintValidator::parse_box_default("((1,2),(3,4))");
        assert!(matches!(result, OwnedValue::Box(_, _)));
        if let OwnedValue::Box(p1, p2) = result {
            assert!((p1.0 - 1.0).abs() < f64::EPSILON);
            assert!((p1.1 - 2.0).abs() < f64::EPSILON);
            assert!((p2.0 - 3.0).abs() < f64::EPSILON);
            assert!((p2.1 - 4.0).abs() < f64::EPSILON);
        }
    }

    #[test]
    fn test_parse_circle_default() {
        let result = ConstraintValidator::parse_circle_default("<(1,2),5>");
        assert!(matches!(result, OwnedValue::Circle(_, _)));
        if let OwnedValue::Circle(center, radius) = result {
            assert!((center.0 - 1.0).abs() < f64::EPSILON);
            assert!((center.1 - 2.0).abs() < f64::EPSILON);
            assert!((radius - 5.0).abs() < f64::EPSILON);
        }
    }
}
