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
            _ => OwnedValue::Text(default_str.to_string()),
        }
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
                if table_name == "users"
                    && col_name == "id"
                    && matches!(value, OwnedValue::Int(1))
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
}
