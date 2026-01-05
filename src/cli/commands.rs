//! # Dot Command Handler
//!
//! Parses and executes SQLite-style dot commands for database introspection
//! and CLI control. Dot commands start with a period and are not SQL.
//!
//! ## Supported Commands
//!
//! | Command              | Description                              |
//! |----------------------|------------------------------------------|
//! | `.quit` / `.exit`    | Exit the CLI                             |
//! | `.tables`            | List all tables in current schema        |
//! | `.schema [table]`    | Show CREATE statement for table(s)       |
//! | `.indexes [table]`   | List indexes, optionally for a table     |
//! | `.help`              | Show available commands                  |
//!
//! ## Parsing
//!
//! Commands are case-insensitive. Arguments are whitespace-separated.
//! Unrecognized commands return an error message.
//!
//! ## Implementation
//!
//! Each command is implemented as a method that returns a CommandResult:
//! - Output: Text to display to the user
//! - Exit: Signal to terminate the REPL
//! - Error: Error message to display
//!
//! Commands that need database access receive a reference to the Database.

use crate::records::types::DataType;
use crate::schema::{Constraint, IndexDef, TableDef};
use crate::Database;

#[derive(Debug, PartialEq)]
pub enum CommandResult {
    Output(String),
    Exit,
    Continue,
    Error(String),
}

pub struct CommandHandler;

impl CommandHandler {
    pub fn is_command(input: &str) -> bool {
        input.trim().starts_with('.')
    }

    pub fn execute(input: &str, db: &Database) -> CommandResult {
        let input = input.trim();
        let parts: Vec<&str> = input.split_whitespace().collect();

        if parts.is_empty() {
            return CommandResult::Continue;
        }

        let cmd = parts[0].to_lowercase();
        let args = &parts[1..];

        match cmd.as_str() {
            ".quit" | ".exit" | ".q" => CommandResult::Exit,
            ".help" | ".h" | ".?" => CommandResult::Output(help_text()),
            ".tables" => list_tables(db),
            ".schema" => show_schema(db, args),
            ".indexes" => list_indexes(db, args),
            _ => CommandResult::Error(format!("Unknown command: {}. Type .help for available commands.", cmd)),
        }
    }
}

fn help_text() -> String {
    r#"TurDB CLI Commands:

  .quit, .exit, .q     Exit the CLI
  .help, .h, .?        Show this help message
  .tables              List all tables in the database
  .schema [TABLE]      Show CREATE statement for TABLE (or all tables)
  .indexes [TABLE]     List indexes (optionally for a specific table)

SQL queries should end with a semicolon (;).
Multi-line queries are supported - press Enter to continue on next line.
Use Ctrl+C to cancel a multi-line query.
Use Ctrl+D or .quit to exit."#
        .to_string()
}

fn list_tables(db: &Database) -> CommandResult {
    let catalog = db.shared.catalog.read();
    let catalog = match catalog.as_ref() {
        Some(c) => c,
        None => return CommandResult::Error("Database not initialized".to_string()),
    };

    let mut tables: Vec<&str> = Vec::new();
    for schema in catalog.schemas().values() {
        for table in schema.tables().values() {
            tables.push(table.name());
        }
    }

    if tables.is_empty() {
        CommandResult::Output("No tables found.".to_string())
    } else {
        tables.sort();
        CommandResult::Output(tables.join("\n"))
    }
}

fn show_schema(db: &Database, args: &[&str]) -> CommandResult {
    let catalog = db.shared.catalog.read();
    let catalog = match catalog.as_ref() {
        Some(c) => c,
        None => return CommandResult::Error("Database not initialized".to_string()),
    };

    if let Some(table_name) = args.first() {
        match catalog.resolve_table(table_name) {
            Ok(table) => CommandResult::Output(format_create_table(table)),
            Err(_) => CommandResult::Error(format!("Table '{}' not found.", table_name)),
        }
    } else {
        let mut schemas: Vec<String> = Vec::new();
        for schema in catalog.schemas().values() {
            for table in schema.tables().values() {
                schemas.push(format_create_table(table));
            }
        }

        if schemas.is_empty() {
            CommandResult::Output("No tables found.".to_string())
        } else {
            CommandResult::Output(schemas.join("\n\n"))
        }
    }
}

fn format_create_table(table: &TableDef) -> String {
    let mut sql = format!("CREATE TABLE {} (\n", table.name());

    let columns: Vec<String> = table
        .columns()
        .iter()
        .map(|col| {
            let type_str = format_data_type(col.data_type(), col.max_length());
            let mut col_def = format!("  {} {}", col.name(), type_str);
            if !col.is_nullable() {
                col_def.push_str(" NOT NULL");
            }
            if col.has_constraint(&Constraint::PrimaryKey) {
                col_def.push_str(" PRIMARY KEY");
            }
            if col.has_constraint(&Constraint::AutoIncrement) {
                col_def.push_str(" AUTO_INCREMENT");
            }
            if col.has_constraint(&Constraint::Unique)
                && !col.has_constraint(&Constraint::PrimaryKey)
            {
                col_def.push_str(" UNIQUE");
            }
            if let Some(default) = col.default_value() {
                col_def.push_str(&format!(" DEFAULT {}", default));
            }
            col_def
        })
        .collect();

    sql.push_str(&columns.join(",\n"));
    sql.push_str("\n);");
    sql
}

fn format_data_type(data_type: DataType, max_length: Option<u32>) -> String {
    use DataType::*;
    match data_type {
        Varchar => match max_length {
            Some(len) => format!("VARCHAR({})", len),
            None => "VARCHAR".to_string(),
        },
        Char => match max_length {
            Some(len) => format!("CHAR({})", len),
            None => "CHAR".to_string(),
        },
        Bool => "BOOLEAN".to_string(),
        Int2 => "SMALLINT".to_string(),
        Int4 => "INT".to_string(),
        Int8 => "BIGINT".to_string(),
        Float4 => "REAL".to_string(),
        Float8 => "DOUBLE PRECISION".to_string(),
        Text => "TEXT".to_string(),
        Blob => "BLOB".to_string(),
        Date => "DATE".to_string(),
        Time => "TIME".to_string(),
        Timestamp => "TIMESTAMP".to_string(),
        TimestampTz => "TIMESTAMPTZ".to_string(),
        Uuid => "UUID".to_string(),
        Jsonb => "JSONB".to_string(),
        Vector => "VECTOR".to_string(),
        Array => "ARRAY".to_string(),
        Interval => "INTERVAL".to_string(),
        Point => "POINT".to_string(),
        Box => "BOX".to_string(),
        Circle => "CIRCLE".to_string(),
        MacAddr => "MACADDR".to_string(),
        Inet4 | Inet6 => "INET".to_string(),
        Decimal => "DECIMAL".to_string(),
        Int4Range => "INT4RANGE".to_string(),
        Int8Range => "INT8RANGE".to_string(),
        DateRange => "DATERANGE".to_string(),
        TimestampRange => "TSRANGE".to_string(),
        Enum => "ENUM".to_string(),
        Composite => "COMPOSITE".to_string(),
    }
}

fn list_indexes(db: &Database, args: &[&str]) -> CommandResult {
    let catalog = db.shared.catalog.read();
    let catalog = match catalog.as_ref() {
        Some(c) => c,
        None => return CommandResult::Error("Database not initialized".to_string()),
    };

    let mut output = Vec::new();

    if let Some(table_name) = args.first() {
        match catalog.resolve_table(table_name) {
            Ok(table) => {
                for idx in table.indexes() {
                    output.push(format_index(table.name(), idx));
                }
            }
            Err(_) => return CommandResult::Error(format!("Table '{}' not found.", table_name)),
        }
    } else {
        for schema in catalog.schemas().values() {
            for table in schema.tables().values() {
                for idx in table.indexes() {
                    output.push(format_index(table.name(), idx));
                }
            }
        }
    }

    if output.is_empty() {
        CommandResult::Output("No indexes found.".to_string())
    } else {
        CommandResult::Output(output.join("\n"))
    }
}

fn format_index(table_name: &str, idx: &IndexDef) -> String {
    let unique = if idx.is_unique() { "UNIQUE " } else { "" };
    let columns = idx.columns();
    format!(
        "{}INDEX {} ON {} ({})",
        unique,
        idx.name(),
        table_name,
        columns.join(", ")
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn is_command_returns_true_for_dot_prefix() {
        assert!(CommandHandler::is_command(".quit"));
        assert!(CommandHandler::is_command(".tables"));
        assert!(CommandHandler::is_command("  .help"));
    }

    #[test]
    fn is_command_returns_false_for_sql() {
        assert!(!CommandHandler::is_command("SELECT * FROM users"));
        assert!(!CommandHandler::is_command("CREATE TABLE foo"));
        assert!(!CommandHandler::is_command(""));
    }

    #[test]
    fn quit_commands_return_exit() {
        use tempfile::tempdir;

        let dir = tempdir().unwrap();
        let db = Database::create(dir.path().join("test")).unwrap();

        assert_eq!(CommandHandler::execute(".quit", &db), CommandResult::Exit);
        assert_eq!(CommandHandler::execute(".exit", &db), CommandResult::Exit);
        assert_eq!(CommandHandler::execute(".q", &db), CommandResult::Exit);
    }

    #[test]
    fn help_returns_help_text() {
        use tempfile::tempdir;

        let dir = tempdir().unwrap();
        let db = Database::create(dir.path().join("test")).unwrap();

        let result = CommandHandler::execute(".help", &db);
        match result {
            CommandResult::Output(text) => {
                assert!(text.contains(".quit"));
                assert!(text.contains(".tables"));
                assert!(text.contains(".schema"));
            }
            _ => panic!("Expected Output, got {:?}", result),
        }
    }

    #[test]
    fn unknown_command_returns_error() {
        use tempfile::tempdir;

        let dir = tempdir().unwrap();
        let db = Database::create(dir.path().join("test")).unwrap();

        let result = CommandHandler::execute(".unknown", &db);
        match result {
            CommandResult::Error(msg) => {
                assert!(msg.contains("Unknown command"));
            }
            _ => panic!("Expected Error, got {:?}", result),
        }
    }

    #[test]
    fn tables_command_lists_tables() {
        use tempfile::tempdir;

        let dir = tempdir().unwrap();
        let db = Database::create(dir.path().join("test")).unwrap();
        db.execute("CREATE TABLE users (id INT)").unwrap();
        db.execute("CREATE TABLE orders (id INT)").unwrap();

        let result = CommandHandler::execute(".tables", &db);
        match result {
            CommandResult::Output(text) => {
                assert!(text.contains("users"));
                assert!(text.contains("orders"));
            }
            _ => panic!("Expected Output, got {:?}", result),
        }
    }

    #[test]
    fn schema_command_shows_create_statement() {
        use tempfile::tempdir;

        let dir = tempdir().unwrap();
        let db = Database::create(dir.path().join("test")).unwrap();
        db.execute("CREATE TABLE users (id INT, name TEXT)").unwrap();

        let result = CommandHandler::execute(".schema users", &db);
        match result {
            CommandResult::Output(text) => {
                assert!(text.contains("CREATE TABLE users"));
                assert!(text.contains("id"));
                assert!(text.contains("name"));
            }
            _ => panic!("Expected Output, got {:?}", result),
        }
    }

    #[test]
    fn schema_nonexistent_table_returns_error() {
        use tempfile::tempdir;

        let dir = tempdir().unwrap();
        let db = Database::create(dir.path().join("test")).unwrap();
        db.execute("CREATE TABLE dummy (id INT)").unwrap();

        let result = CommandHandler::execute(".schema nonexistent", &db);
        match result {
            CommandResult::Error(msg) => {
                assert!(
                    msg.contains("not found"),
                    "Expected 'not found' in message, got: {}",
                    msg
                );
            }
            _ => panic!("Expected Error, got {:?}", result),
        }
    }
}
