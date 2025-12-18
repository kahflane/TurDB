//! # History File Management
//!
//! Manages the location and configuration of the CLI history file.
//! By default, history is stored in `~/.turdb_history`.
//!
//! ## Configuration
//!
//! The history file location can be overridden using the `TURDB_HISTORY`
//! environment variable:
//!
//! ```bash
//! export TURDB_HISTORY=/custom/path/history
//! turdb ./mydb
//! ```
//!
//! To disable history persistence, set `TURDB_HISTORY` to an empty string
//! or a path to `/dev/null`.
//!
//! ## Implementation
//!
//! The history path is resolved once at CLI startup and passed to rustyline.
//! rustyline handles the actual file I/O.

use std::env;
use std::path::PathBuf;

const DEFAULT_HISTORY_FILE: &str = ".turdb_history";
const HISTORY_ENV_VAR: &str = "TURDB_HISTORY";

pub fn history_path() -> Option<PathBuf> {
    if let Ok(custom_path) = env::var(HISTORY_ENV_VAR) {
        if custom_path.is_empty() {
            return None;
        }
        return Some(PathBuf::from(custom_path));
    }

    home_dir().map(|home| home.join(DEFAULT_HISTORY_FILE))
}

fn home_dir() -> Option<PathBuf> {
    env::var("HOME").ok().map(PathBuf::from)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_history_path_is_in_home() {
        env::remove_var(HISTORY_ENV_VAR);

        if let Some(path) = history_path() {
            assert!(path.to_string_lossy().contains(".turdb_history"));
        }
    }

    #[test]
    fn custom_history_path_from_env() {
        env::set_var(HISTORY_ENV_VAR, "/custom/path");
        let path = history_path();
        env::remove_var(HISTORY_ENV_VAR);

        assert_eq!(path, Some(PathBuf::from("/custom/path")));
    }

    #[test]
    fn empty_env_disables_history() {
        env::set_var(HISTORY_ENV_VAR, "");
        let path = history_path();
        env::remove_var(HISTORY_ENV_VAR);

        assert_eq!(path, None);
    }
}
