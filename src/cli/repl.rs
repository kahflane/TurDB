//! # REPL - Read-Eval-Print Loop
//!
//! The main interactive loop for the TurDB CLI. Handles:
//!
//! - Reading input with rustyline (history, line editing)
//! - Dispatching dot commands vs SQL statements
//! - Executing SQL and formatting results
//! - Multi-line statement handling
//!
//! ## Input Handling
//!
//! The REPL distinguishes between:
//! - Dot commands: Start with `.`, executed immediately
//! - SQL statements: Accumulated until `;` is encountered
//!
//! Multi-line SQL is supported. The prompt changes from `turdb>` to
//! `    ->` to indicate continuation mode.
//!
//! ## Execution Flow
//!
//! ```text
//! ┌──────────────────────────────────────────────────────────┐
//! │                     Read Line                             │
//! └──────────────────────────────────────────────────────────┘
//!                           │
//!                           ▼
//! ┌──────────────────────────────────────────────────────────┐
//! │              Starts with '.'?                             │
//! └──────────────────────────────────────────────────────────┘
//!           │ Yes                          │ No
//!           ▼                              ▼
//! ┌──────────────────┐          ┌──────────────────────────┐
//! │ Execute Command  │          │ Accumulate SQL           │
//! └──────────────────┘          └──────────────────────────┘
//!           │                              │
//!           │                              ▼
//!           │                   ┌──────────────────────────┐
//!           │                   │ Ends with ';'?           │
//!           │                   └──────────────────────────┘
//!           │                        │ Yes       │ No
//!           │                        ▼           │
//!           │               ┌────────────────┐   │
//!           │               │ Execute SQL    │   │
//!           │               └────────────────┘   │
//!           │                        │           │
//!           ▼                        ▼           ▼
//! ┌──────────────────────────────────────────────────────────┐
//! │                   Print Result                            │
//! └──────────────────────────────────────────────────────────┘
//!                           │
//!                           ▼
//!                       [Loop]
//! ```
//!
//! ## Error Handling
//!
//! SQL errors are displayed but do not terminate the REPL.
//! Use `.quit` or Ctrl+D to exit.

use crate::cli::commands::{CommandHandler, CommandResult};
use crate::cli::history::history_path;
use crate::cli::table::TableFormatter;
use crate::database::ExecuteResult;
use crate::Database;
use eyre::{Result, WrapErr};
use rustyline::error::ReadlineError;
use rustyline::DefaultEditor;
use std::time::Instant;

const PRIMARY_PROMPT: &str = "turdb> ";
const CONTINUATION_PROMPT: &str = "    -> ";

pub struct Repl {
    db: Database,
    editor: DefaultEditor,
    sql_buffer: String,
}

impl Repl {
    pub fn new(db: Database) -> Result<Self> {
        let mut editor = DefaultEditor::new().wrap_err("failed to initialize line editor")?;

        if let Some(history_file) = history_path() {
            let _ = editor.load_history(&history_file);
        }

        Ok(Self {
            db,
            editor,
            sql_buffer: String::new(),
        })
    }

    pub fn run(&mut self) -> Result<()> {
        self.print_welcome();

        loop {
            let prompt = if self.sql_buffer.is_empty() {
                PRIMARY_PROMPT
            } else {
                CONTINUATION_PROMPT
            };

            match self.editor.readline(prompt) {
                Ok(line) => {
                    if !self.handle_line(&line)? {
                        break;
                    }
                }
                Err(ReadlineError::Interrupted) => {
                    self.sql_buffer.clear();
                    println!("^C");
                }
                Err(ReadlineError::Eof) => {
                    println!("Bye");
                    break;
                }
                Err(err) => {
                    eprintln!("Error reading input: {}", err);
                    break;
                }
            }
        }

        self.save_history();
        Ok(())
    }

    fn handle_line(&mut self, line: &str) -> Result<bool> {
        let trimmed = line.trim();

        if trimmed.is_empty() {
            return Ok(true);
        }

        if self.sql_buffer.is_empty() && CommandHandler::is_command(trimmed) {
            self.editor.add_history_entry(trimmed).ok();
            return Ok(self.execute_command(trimmed));
        }

        if !self.sql_buffer.is_empty() {
            self.sql_buffer.push(' ');
        }
        self.sql_buffer.push_str(trimmed);

        if self.sql_buffer.trim_end().ends_with(';') {
            let sql = std::mem::take(&mut self.sql_buffer);
            self.editor.add_history_entry(&sql).ok();
            self.execute_sql(&sql);
        }

        Ok(true)
    }

    fn execute_command(&mut self, input: &str) -> bool {
        match CommandHandler::execute(input, &self.db) {
            CommandResult::Exit => false,
            CommandResult::Output(text) => {
                println!("{}", text);
                true
            }
            CommandResult::Continue => true,
            CommandResult::Error(msg) => {
                eprintln!("Error: {}", msg);
                true
            }
        }
    }

    fn execute_sql(&mut self, sql: &str) {
        let start = Instant::now();

        match self.db.execute(sql) {
            Ok(result) => {
                let elapsed = start.elapsed();
                self.print_result(result, elapsed);
            }
            Err(err) => {
                eprintln!("Error: {}", err);
            }
        }
    }

    fn print_result(&self, result: ExecuteResult, elapsed: std::time::Duration) {
        match result {
            ExecuteResult::Select { rows } => {
                if rows.is_empty() {
                    println!("Empty set ({:.3} sec)", elapsed.as_secs_f64());
                } else {
                    let headers = self.generate_headers(&rows);
                    let formatter = TableFormatter::new(headers, &rows);
                    println!("{}", formatter.render());
                    println!(
                        "{} row{} in set ({:.3} sec)",
                        formatter.row_count(),
                        if formatter.row_count() == 1 { "" } else { "s" },
                        elapsed.as_secs_f64()
                    );
                }
            }
            ExecuteResult::Insert { rows_affected, returned } => {
                if let Some(rows) = returned {
                    if !rows.is_empty() {
                        let headers = self.generate_headers(&rows);
                        let formatter = TableFormatter::new(headers, &rows);
                        println!("{}", formatter.render());
                    }
                }
                println!(
                    "Query OK, {} row{} affected ({:.3} sec)",
                    rows_affected,
                    if rows_affected == 1 { "" } else { "s" },
                    elapsed.as_secs_f64()
                );
            }
            ExecuteResult::Update { rows_affected, returned } => {
                if let Some(rows) = returned {
                    if !rows.is_empty() {
                        let headers = self.generate_headers(&rows);
                        let formatter = TableFormatter::new(headers, &rows);
                        println!("{}", formatter.render());
                    }
                }
                println!(
                    "Query OK, {} row{} affected ({:.3} sec)",
                    rows_affected,
                    if rows_affected == 1 { "" } else { "s" },
                    elapsed.as_secs_f64()
                );
            }
            ExecuteResult::Delete { rows_affected, returned } => {
                if let Some(rows) = returned {
                    if !rows.is_empty() {
                        let headers = self.generate_headers(&rows);
                        let formatter = TableFormatter::new(headers, &rows);
                        println!("{}", formatter.render());
                    }
                }
                println!(
                    "Query OK, {} row{} affected ({:.3} sec)",
                    rows_affected,
                    if rows_affected == 1 { "" } else { "s" },
                    elapsed.as_secs_f64()
                );
            }
            ExecuteResult::CreateTable { created } => {
                if created {
                    println!("Table created ({:.3} sec)", elapsed.as_secs_f64());
                } else {
                    println!("Table already exists ({:.3} sec)", elapsed.as_secs_f64());
                }
            }
            ExecuteResult::CreateSchema { created } => {
                if created {
                    println!("Schema created ({:.3} sec)", elapsed.as_secs_f64());
                } else {
                    println!("Schema already exists ({:.3} sec)", elapsed.as_secs_f64());
                }
            }
            ExecuteResult::CreateIndex { created } => {
                if created {
                    println!("Index created ({:.3} sec)", elapsed.as_secs_f64());
                } else {
                    println!("Index already exists ({:.3} sec)", elapsed.as_secs_f64());
                }
            }
            ExecuteResult::DropTable { dropped } => {
                if dropped {
                    println!("Table dropped ({:.3} sec)", elapsed.as_secs_f64());
                } else {
                    println!("Table does not exist ({:.3} sec)", elapsed.as_secs_f64());
                }
            }
            ExecuteResult::DropIndex { dropped } => {
                if dropped {
                    println!("Index dropped ({:.3} sec)", elapsed.as_secs_f64());
                } else {
                    println!("Index does not exist ({:.3} sec)", elapsed.as_secs_f64());
                }
            }
            ExecuteResult::DropSchema { dropped } => {
                if dropped {
                    println!("Schema dropped ({:.3} sec)", elapsed.as_secs_f64());
                } else {
                    println!("Schema does not exist ({:.3} sec)", elapsed.as_secs_f64());
                }
            }
            ExecuteResult::Pragma { name, value } => {
                if let Some(v) = value {
                    println!("{} = {}", name, v);
                } else {
                    println!("{}", name);
                }
            }
            ExecuteResult::Begin => {
                println!("Transaction started ({:.3} sec)", elapsed.as_secs_f64());
            }
            ExecuteResult::Commit => {
                println!("Transaction committed ({:.3} sec)", elapsed.as_secs_f64());
            }
            ExecuteResult::Rollback => {
                println!("Transaction rolled back ({:.3} sec)", elapsed.as_secs_f64());
            }
            ExecuteResult::Savepoint { name } => {
                println!("Savepoint '{}' created ({:.3} sec)", name, elapsed.as_secs_f64());
            }
            ExecuteResult::Release { name } => {
                println!("Savepoint '{}' released ({:.3} sec)", name, elapsed.as_secs_f64());
            }
        }
    }

    fn generate_headers(&self, rows: &[crate::Row]) -> Vec<String> {
        if let Some(first_row) = rows.first() {
            (0..first_row.column_count())
                .map(|i| format!("col{}", i + 1))
                .collect()
        } else {
            vec![]
        }
    }

    fn print_welcome(&self) {
        println!("TurDB version 0.1.0");
        println!("Enter \".help\" for usage hints.");
        println!("Connected to: {}", self.db.path().display());
        println!();
    }

    fn save_history(&mut self) {
        if let Some(history_file) = history_path() {
            if let Err(e) = self.editor.save_history(&history_file) {
                eprintln!("Warning: could not save history: {}", e);
            }
        }
    }
}
