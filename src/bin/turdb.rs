//! # TurDB CLI Entry Point
//!
//! Binary entry point for the TurDB command-line interface.
//!
//! ## Usage
//!
//! ```bash
//! # Open existing database
//! turdb ./mydb
//!
//! # Create new database
//! turdb --create ./newdb
//!
//! # Show version
//! turdb --version
//!
//! # Show help
//! turdb --help
//! ```

use eyre::{bail, Result, WrapErr};
use std::env;
use std::path::PathBuf;
use turdb::cli::Repl;
use turdb::Database;

fn main() {
    if let Err(e) = run() {
        eprintln!("Error: {}", e);
        std::process::exit(1);
    }
}

fn run() -> Result<()> {
    let args: Vec<String> = env::args().collect();

    if args.len() < 2 {
        print_usage();
        return Ok(());
    }

    let mut create_mode = false;
    let mut db_path: Option<PathBuf> = None;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--help" | "-h" => {
                print_usage();
                return Ok(());
            }
            "--version" | "-v" => {
                println!("turdb {}", env!("CARGO_PKG_VERSION"));
                return Ok(());
            }
            "--create" | "-c" => {
                create_mode = true;
            }
            arg if arg.starts_with('-') => {
                bail!("Unknown option: {}", arg);
            }
            path => {
                if db_path.is_some() {
                    bail!("Multiple database paths specified");
                }
                db_path = Some(PathBuf::from(path));
            }
        }
        i += 1;
    }

    let db_path = match db_path {
        Some(p) => p,
        None => {
            print_usage();
            return Ok(());
        }
    };

    let db = if create_mode {
        Database::create(&db_path)
            .wrap_err_with(|| format!("failed to create database at {:?}", db_path))?
    } else if db_path.exists() {
        Database::open(&db_path)
            .wrap_err_with(|| format!("failed to open database at {:?}", db_path))?
    } else {
        Database::create(&db_path)
            .wrap_err_with(|| format!("failed to create database at {:?}", db_path))?
    };

    let mut repl = Repl::new(db)?;
    repl.run()?;

    Ok(())
}

fn print_usage() {
    println!("TurDB - High-performance embedded database");
    println!();
    println!("USAGE:");
    println!("    turdb [OPTIONS] <DATABASE_PATH>");
    println!();
    println!("ARGS:");
    println!("    <DATABASE_PATH>    Path to the database directory");
    println!();
    println!("OPTIONS:");
    println!("    -c, --create       Create a new database (default if path doesn't exist)");
    println!("    -h, --help         Print help information");
    println!("    -v, --version      Print version information");
    println!();
    println!("EXAMPLES:");
    println!("    turdb ./mydb           Open or create database at ./mydb");
    println!("    turdb --create ./new   Create new database at ./new");
}
