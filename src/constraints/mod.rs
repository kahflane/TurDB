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

use crate::schema::table::{Constraint, TableDef};
use crate::types::OwnedValue;
use eyre::{bail, Result};
use smallvec::SmallVec;

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
        let parts: SmallVec<[&str; 4]> = s.split('-').collect();
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
        let parts: SmallVec<[&str; 4]> = s.split(':').collect();
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
            let sec_parts: SmallVec<[&str; 2]> = parts[2].split('.').collect();
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
        let datetime_parts: SmallVec<[&str; 2]> = s.split(&[' ', 'T'][..]).collect();
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
        let parts: SmallVec<[&str; 2]> = stripped.split(',').collect();
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
        let parts: Vec<f64> = stripped
            .split(',')
            .filter_map(|p| p.trim().parse().ok())
            .collect();
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
        let parts: SmallVec<[&str; 4]> = s.split('/').next().unwrap_or(s).split('.').collect();
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
        let parts: SmallVec<[&str; 2]> = addr_str.split("::").collect();
        let mut bytes = [0u8; 16];
        if parts.len() == 2 {
            let left_groups: SmallVec<[u16; 8]> = parts[0]
                .split(':')
                .filter(|p| !p.is_empty())
                .filter_map(|p| u16::from_str_radix(p, 16).ok())
                .collect();
            let right_groups: SmallVec<[u16; 8]> = parts[1]
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
            let groups: SmallVec<[u16; 8]> = addr_str
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
        let parts: SmallVec<[&str; 2]> = trimmed.split('.').collect();
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
