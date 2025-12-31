//! # SQLite Batch Import for TurDB
//!
//! This module imports all data from the Kaggle meta SQLite database into TurDB.
//! It handles 33 tables with a total of ~350 million rows using batch processing.
//!
//! ## Architecture
//!
//! The import uses a multi-threaded pipeline:
//! 1. Reader threads: Read from SQLite in parallel (one per table)
//! 2. Writer thread: Single thread writes to TurDB (required for consistency)
//! 3. Channel-based communication between readers and writer
//!
//! ## Performance Optimizations
//!
//! - Multi-threaded reading from SQLite
//! - Multi-row VALUES syntax: INSERT INTO t VALUES (1,'a'), (2,'b'), ...
//! - PRAGMA synchronous=OFF: Disables fsync for bulk loads
//! - Batch size of 10000 rows per INSERT statement
//! - Channel buffering to keep writer busy
//!
//! ## Usage
//!
//! ```sh
//! cargo test --test sqlite_import --release -- --nocapture
//! ```

use rusqlite::Connection;
use std::path::Path;
use std::sync::mpsc;
use std::thread;
use std::time::Instant;
use turdb::Database;

const SQLITE_DB_PATH: &str = "/Users/julfikar/Downloads/_meta-kaggle.db";
const TURDB_PATH: &str = "/Users/julfikar/Documents/PassionFruit.nosync/turdb/turdb-core/.worktrees/bismillah";
const BATCH_SIZE: i64 = 50000;
const INSERT_BATCH_SIZE: usize = 5000;
const CHANNEL_BUFFER: usize = 32;
const PROGRESS_INTERVAL: u64 = 100000;

fn sqlite_db_exists() -> bool {
    Path::new(SQLITE_DB_PATH).exists()
}

struct TableSchema {
    name: &'static str,
    turdb_ddl: &'static str,
    columns: &'static str,
}

fn camel_to_snake(name: &str) -> String {
    name.chars()
        .enumerate()
        .flat_map(|(i, c)| {
            if c.is_uppercase() && i > 0 {
                vec!['_', c.to_ascii_lowercase()]
            } else {
                vec![c.to_ascii_lowercase()]
            }
        })
        .collect()
}

const TABLES: &[TableSchema] = &[
    TableSchema {
        name: "Competitions",
        turdb_ddl: "CREATE TABLE competitions (
            id BIGINT primary key auto_increment, slug TEXT, title TEXT, subtitle TEXT, host_segment_title TEXT,
            forum_id BIGINT, organization_id FLOAT, enabled_date TEXT, deadline_date TEXT,
            prohibit_new_entrants_deadline_date TEXT, team_merger_deadline_date TEXT,
            team_model_deadline_date TEXT, model_submission_deadline_date TEXT,
            final_leaderboard_has_been_verified BIGINT, has_kernels BIGINT,
            only_allow_kernel_submissions BIGINT, has_leaderboard BIGINT,
            leaderboard_percentage BIGINT, leaderboard_display_format FLOAT,
            evaluation_algorithm_abbreviation TEXT, evaluation_algorithm_name TEXT,
            evaluation_algorithm_description TEXT, evaluation_algorithm_is_max BIGINT,
            max_daily_submissions BIGINT, num_scored_submissions BIGINT, max_team_size BIGINT,
            ban_team_mergers BIGINT, enable_team_models BIGINT, reward_type TEXT,
            reward_quantity FLOAT, num_prizes BIGINT, user_rank_multiplier FLOAT,
            can_qualify_tiers BIGINT, total_teams BIGINT, total_competitors BIGINT,
            total_submissions BIGINT, validation_set_name FLOAT, validation_set_value FLOAT,
            enable_submission_model_hashes BIGINT, enable_submission_model_attachments BIGINT,
            host_name FLOAT, competition_type_id INT
        )",
        columns: "Id, Slug, Title, Subtitle, HostSegmentTitle, ForumId, OrganizationId, EnabledDate, DeadlineDate, ProhibitNewEntrantsDeadlineDate, TeamMergerDeadlineDate, TeamModelDeadlineDate, ModelSubmissionDeadlineDate, FinalLeaderboardHasBeenVerified, HasKernels, OnlyAllowKernelSubmissions, HasLeaderboard, LeaderboardPercentage, LeaderboardDisplayFormat, EvaluationAlgorithmAbbreviation, EvaluationAlgorithmName, EvaluationAlgorithmDescription, EvaluationAlgorithmIsMax, MaxDailySubmissions, NumScoredSubmissions, MaxTeamSize, BanTeamMergers, EnableTeamModels, RewardType, RewardQuantity, NumPrizes, UserRankMultiplier, CanQualifyTiers, TotalTeams, TotalCompetitors, TotalSubmissions, ValidationSetName, ValidationSetValue, EnableSubmissionModelHashes, EnableSubmissionModelAttachments, HostName, CompetitionTypeId",
    },
    TableSchema {
        name: "CompetitionTags",
        turdb_ddl: "CREATE TABLE competition_tags (id BIGINT primary key auto_increment, competition_id BIGINT, tag_id BIGINT)",
        columns: "Id, CompetitionId, TagId",
    },
    TableSchema {
        name: "Datasets",
        turdb_ddl: "CREATE TABLE datasets (
            id BIGINT primary key auto_increment, creator_user_id BIGINT, owner_user_id FLOAT, owner_organization_id FLOAT,
            current_dataset_version_id FLOAT, current_datasource_version_id FLOAT,
            forum_id BIGINT, type BIGINT, creation_date TEXT, last_activity_date TEXT,
            total_views BIGINT, total_downloads BIGINT, total_votes BIGINT, total_kernels INT
        )",
        columns: "Id, CreatorUserId, OwnerUserId, OwnerOrganizationId, CurrentDatasetVersionId, CurrentDatasourceVersionId, ForumId, Type, CreationDate, LastActivityDate, TotalViews, TotalDownloads, TotalVotes, TotalKernels",
    },
    TableSchema {
        name: "DatasetTags",
        turdb_ddl: "CREATE TABLE dataset_tags (id BIGINT primary key auto_increment, dataset_id BIGINT, tag_id BIGINT)",
        columns: "Id, DatasetId, TagId",
    },
    TableSchema {
        name: "DatasetTasks",
        turdb_ddl: "CREATE TABLE dataset_tasks (
            id BIGINT primary key auto_increment, dataset_id BIGINT, owner_user_id BIGINT, creation_date TEXT,
            description TEXT, forum_id FLOAT, title TEXT, subtitle TEXT,
            deadline TEXT, total_votes INT
        )",
        columns: "Id, DatasetId, OwnerUserId, CreationDate, Description, ForumId, Title, Subtitle, Deadline, TotalVotes",
    },
    TableSchema {
        name: "DatasetTaskSubmissions",
        turdb_ddl: "CREATE TABLE dataset_task_submissions (
            id BIGINT primary key auto_increment, dataset_task_id BIGINT, submitted_user_id FLOAT, creation_date TEXT,
            kernel_id FLOAT, dataset_id FLOAT, accepted_date TEXT
        )",
        columns: "Id, DatasetTaskId, SubmittedUserId, CreationDate, KernelId, DatasetId, AcceptedDate",
    },
    TableSchema {
        name: "DatasetVersions",
        turdb_ddl: "CREATE TABLE dataset_versions (
            id BIGINT primary key auto_increment, dataset_id BIGINT, datasource_version_id BIGINT, creator_user_id BIGINT,
            license_name TEXT, creation_date TEXT, version_number FLOAT, title TEXT,
            slug TEXT, subtitle TEXT, description TEXT, version_notes TEXT,
            total_compressed_bytes FLOAT, total_uncompressed_bytes REAL
        )",
        columns: "Id, DatasetId, DatasourceVersionId, CreatorUserId, LicenseName, CreationDate, VersionNumber, Title, Slug, Subtitle, Description, VersionNotes, TotalCompressedBytes, TotalUncompressedBytes",
    },
    TableSchema {
        name: "DatasetVotes",
        turdb_ddl: "CREATE TABLE dataset_votes (id BIGINT primary key auto_increment, user_id BIGINT, dataset_version_id BIGINT, vote_date TEXT)",
        columns: "Id, UserId, DatasetVersionId, VoteDate",
    },
    TableSchema {
        name: "Datasources",
        turdb_ddl: "CREATE TABLE datasources (
            id BIGINT primary key auto_increment, creator_user_id BIGINT, creation_date TEXT, type BIGINT,
            current_datasource_version_id INT
        )",
        columns: "Id, CreatorUserId, CreationDate, Type, CurrentDatasourceVersionId",
    },
    TableSchema {
        name: "Episodes",
        turdb_ddl: "CREATE TABLE episodes (id BIGINT primary key auto_increment, type BIGINT, competition_id BIGINT, create_time TEXT, end_time TEXT)",
        columns: "Id, Type, CompetitionId, CreateTime, EndTime",
    },
    TableSchema {
        name: "ForumMessages",
        turdb_ddl: "CREATE TABLE forum_messages (
            id BIGINT primary key auto_increment, forum_topic_id BIGINT, post_user_id BIGINT, post_date TEXT,
            reply_to_forum_message_id FLOAT, message TEXT, medal FLOAT, medal_award_date TEXT
        )",
        columns: "Id, ForumTopicId, PostUserId, PostDate, ReplyToForumMessageId, Message, Medal, MedalAwardDate",
    },
    TableSchema {
        name: "ForumMessageVotes",
        turdb_ddl: "CREATE TABLE forum_message_votes (
            id BIGINT primary key auto_increment, forum_message_id BIGINT, from_user_id BIGINT, to_user_id BIGINT, vote_date TEXT
        )",
        columns: "Id, ForumMessageId, FromUserId, ToUserId, VoteDate",
    },
    TableSchema {
        name: "Forums",
        turdb_ddl: "CREATE TABLE forums (id BIGINT primary key auto_increment, parent_forum_id FLOAT, title TEXT)",
        columns: "Id, ParentForumId, Title",
    },
    TableSchema {
        name: "ForumTopics",
        turdb_ddl: "CREATE TABLE forum_topics (
            id BIGINT primary key auto_increment, forum_id BIGINT, kernel_id FLOAT, last_forum_message_id FLOAT,
            first_forum_message_id FLOAT, creation_date TEXT, last_comment_date TEXT,
            title TEXT, is_sticky BIGINT, total_views BIGINT, score BIGINT,
            total_messages BIGINT, total_replies INT
        )",
        columns: "Id, ForumId, KernelId, LastForumMessageId, FirstForumMessageId, CreationDate, LastCommentDate, Title, IsSticky, TotalViews, Score, TotalMessages, TotalReplies",
    },
    TableSchema {
        name: "KernelLanguages",
        turdb_ddl: "CREATE TABLE kernel_languages (id BIGINT primary key auto_increment, name TEXT, display_name TEXT, is_notebook BIGINT)",
        columns: "Id, Name, DisplayName, IsNotebook",
    },
    TableSchema {
        name: "Kernels",
        turdb_ddl: "CREATE TABLE kernels (
            id BIGINT primary key auto_increment, author_user_id BIGINT, current_kernel_version_id FLOAT,
            fork_parent_kernel_version_id FLOAT, forum_topic_id FLOAT,
            first_kernel_version_id FLOAT, creation_date TEXT, evaluation_date TEXT,
            made_public_date TEXT, is_project_language_template BIGINT, current_url_slug TEXT,
            medal FLOAT, medal_award_date TEXT, total_views BIGINT, total_comments BIGINT, total_votes INT
        )",
        columns: "Id, AuthorUserId, CurrentKernelVersionId, ForkParentKernelVersionId, ForumTopicId, FirstKernelVersionId, CreationDate, EvaluationDate, MadePublicDate, IsProjectLanguageTemplate, CurrentUrlSlug, Medal, MedalAwardDate, TotalViews, TotalComments, TotalVotes",
    },
    TableSchema {
        name: "KernelTags",
        turdb_ddl: "CREATE TABLE kernel_tags (id BIGINT primary key auto_increment, kernel_id BIGINT, tag_id BIGINT)",
        columns: "Id, KernelId, TagId",
    },
    TableSchema {
        name: "KernelVersionCompetitionSources",
        turdb_ddl: "CREATE TABLE kernel_version_competition_sources (id BIGINT primary key auto_increment, kernel_version_id BIGINT, source_competition_id BIGINT)",
        columns: "Id, KernelVersionId, SourceCompetitionId",
    },
    TableSchema {
        name: "KernelVersionDatasetSources",
        turdb_ddl: "CREATE TABLE kernel_version_dataset_sources (id BIGINT primary key auto_increment, kernel_version_id BIGINT, source_dataset_version_id BIGINT)",
        columns: "Id, KernelVersionId, SourceDatasetVersionId",
    },
    TableSchema {
        name: "KernelVersionKernelSources",
        turdb_ddl: "CREATE TABLE kernel_version_kernel_sources (id BIGINT primary key auto_increment, kernel_version_id BIGINT, source_kernel_version_id BIGINT)",
        columns: "Id, KernelVersionId, SourceKernelVersionId",
    },
    TableSchema {
        name: "KernelVersions",
        turdb_ddl: "CREATE TABLE kernel_versions (
            id BIGINT primary key auto_increment, script_id BIGINT, parent_script_version_id FLOAT, script_language_id BIGINT,
            author_user_id BIGINT, creation_date TEXT, version_number FLOAT, title TEXT,
            evaluation_date TEXT, is_change BIGINT, total_lines FLOAT,
            lines_inserted_from_previous FLOAT, lines_changed_from_previous FLOAT,
            lines_unchanged_from_previous FLOAT, lines_inserted_from_fork FLOAT,
            lines_deleted_from_fork FLOAT, lines_changed_from_fork FLOAT,
            lines_unchanged_from_fork FLOAT, total_votes INT
        )",
        columns: "Id, ScriptId, ParentScriptVersionId, ScriptLanguageId, AuthorUserId, CreationDate, VersionNumber, Title, EvaluationDate, IsChange, TotalLines, LinesInsertedFromPrevious, LinesChangedFromPrevious, LinesUnchangedFromPrevious, LinesInsertedFromFork, LinesDeletedFromFork, LinesChangedFromFork, LinesUnchangedFromFork, TotalVotes",
    },
    TableSchema {
        name: "KernelVotes",
        turdb_ddl: "CREATE TABLE kernel_votes (id BIGINT primary key auto_increment, user_id BIGINT, kernel_version_id BIGINT, vote_date TEXT)",
        columns: "Id, UserId, KernelVersionId, VoteDate",
    },
    TableSchema {
        name: "Organizations",
        turdb_ddl: "CREATE TABLE organizations (id BIGINT primary key auto_increment, name TEXT, slug TEXT, creation_date TEXT, description TEXT)",
        columns: "Id, Name, Slug, CreationDate, Description",
    },
    TableSchema {
        name: "Submissions",
        turdb_ddl: "CREATE TABLE submissions (
            id BIGINT primary key auto_increment, submitted_user_id BIGINT, team_id BIGINT, source_kernel_version_id BIGINT,
            submission_date TEXT, score_date TEXT, is_after_deadline BIGINT,
            public_score_leaderboard_display TEXT, public_score_full_precision FLOAT,
            private_score_leaderboard_display TEXT, private_score_full_precision REAL
        )",
        columns: "Id, SubmittedUserId, TeamId, SourceKernelVersionId, SubmissionDate, ScoreDate, IsAfterDeadline, PublicScoreLeaderboardDisplay, PublicScoreFullPrecision, PrivateScoreLeaderboardDisplay, PrivateScoreFullPrecision",
    },
    TableSchema {
        name: "Tags",
        turdb_ddl: "CREATE TABLE tags (
            id BIGINT primary key auto_increment, parent_tag_id FLOAT, name TEXT, slug TEXT, full_path TEXT,
            description TEXT, dataset_count BIGINT, competition_count BIGINT, kernel_count INT
        )",
        columns: "Id, ParentTagId, Name, Slug, FullPath, Description, DatasetCount, CompetitionCount, KernelCount",
    },
    TableSchema {
        name: "TeamMemberships",
        turdb_ddl: "CREATE TABLE team_memberships (id BIGINT primary key auto_increment, team_id BIGINT, user_id BIGINT, request_date TEXT)",
        columns: "Id, TeamId, UserId, RequestDate",
    },
    TableSchema {
        name: "Teams",
        turdb_ddl: "CREATE TABLE teams (
            id BIGINT primary key auto_increment, competition_id BIGINT, team_leader_id FLOAT, team_name TEXT,
            score_first_submitted_date FLOAT, last_submission_date TEXT,
            public_leaderboard_submission_id FLOAT, private_leaderboard_submission_id FLOAT,
            is_benchmark BIGINT, medal FLOAT, medal_award_date TEXT,
            public_leaderboard_rank FLOAT, private_leaderboard_rank FLOAT,
            write_up_forum_topic_id REAL
        )",
        columns: "Id, CompetitionId, TeamLeaderId, TeamName, ScoreFirstSubmittedDate, LastSubmissionDate, PublicLeaderboardSubmissionId, PrivateLeaderboardSubmissionId, IsBenchmark, Medal, MedalAwardDate, PublicLeaderboardRank, PrivateLeaderboardRank, WriteUpForumTopicId",
    },
    TableSchema {
        name: "UserAchievements",
        turdb_ddl: "CREATE TABLE user_achievements (
            id BIGINT primary key auto_increment, user_id BIGINT, achievement_type TEXT, tier BIGINT,
            tier_achievement_date TEXT, points BIGINT, current_ranking FLOAT,
            highest_ranking FLOAT, total_gold BIGINT, total_silver BIGINT, total_bronze INT
        )",
        columns: "Id, UserId, AchievementType, Tier, TierAchievementDate, Points, CurrentRanking, HighestRanking, TotalGold, TotalSilver, TotalBronze",
    },
    TableSchema {
        name: "UserFollowers",
        turdb_ddl: "CREATE TABLE user_followers (id BIGINT primary key auto_increment, user_id BIGINT, following_user_id BIGINT, creation_date TEXT)",
        columns: "Id, UserId, FollowingUserId, CreationDate",
    },
    TableSchema {
        name: "UserOrganizations",
        turdb_ddl: "CREATE TABLE user_organizations (id BIGINT primary key auto_increment, user_id BIGINT, organization_id BIGINT, join_date TEXT)",
        columns: "Id, UserId, OrganizationId, JoinDate",
    },
    TableSchema {
        name: "Users",
        turdb_ddl: "CREATE TABLE users (id BIGINT primary key auto_increment, user_name TEXT, display_name TEXT, register_date TEXT, performance_tier BIGINT)",
        columns: "Id, UserName, DisplayName, RegisterDate, PerformanceTier",
    },
    TableSchema {
        name: "EpisodeAgents",
        turdb_ddl: "CREATE TABLE episode_agents (
            id BIGINT primary key auto_increment, episode_id BIGINT, idx BIGINT, reward FLOAT, state BIGINT, submission_id BIGINT,
            initial_confidence FLOAT, initial_score FLOAT, updated_confidence FLOAT, updated_score REAL
        )",
        columns: "Id, EpisodeId, \"Index\", Reward, State, SubmissionId, InitialConfidence, InitialScore, UpdatedConfidence, UpdatedScore",
    },
];

enum ImportMessage {
    CreateTable {
        ddl: String,
    },
    InsertBatch {
        table_name: String,
        values: String,
    },
    TableDone {
        table_name: String,
        rows: u64,
        elapsed_secs: f64,
    },
}

fn escape_sql_value(value: rusqlite::types::Value) -> String {
    use rusqlite::types::Value;
    match value {
        Value::Null => "NULL".to_string(),
        Value::Integer(i) => i.to_string(),
        Value::Real(f) => {
            if f.is_nan() || f.is_infinite() {
                "NULL".to_string()
            } else {
                format!("{:.15e}", f)
            }
        }
        Value::Text(s) => format!("'{}'", s.replace('\'', "''")),
        Value::Blob(_) => "NULL".to_string(),
    }
}

fn read_table_parallel(
    table: &'static TableSchema,
    tx: mpsc::SyncSender<ImportMessage>,
) {
    let start = Instant::now();
    
    let sqlite_conn = match Connection::open(SQLITE_DB_PATH) {
        Ok(conn) => conn,
        Err(e) => {
            eprintln!("  {} - ERROR opening SQLite: {}", table.name, e);
            return;
        }
    };

    if tx.send(ImportMessage::CreateTable {
        ddl: table.turdb_ddl.to_string(),
    }).is_err() {
        return;
    }

    let count: i64 = match sqlite_conn.query_row(
        &format!("SELECT COUNT(*) FROM {}", table.name),
        [],
        |row| row.get(0),
    ) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("  {} - ERROR counting: {}", table.name, e);
            return;
        }
    };

    if count == 0 {
        let _ = tx.send(ImportMessage::TableDone {
            table_name: table.name.to_string(),
            rows: 0,
            elapsed_secs: start.elapsed().as_secs_f64(),
        });
        return;
    }

    let col_count = table.columns.split(',').count();
    let turdb_table = camel_to_snake(table.name);
    let mut total_inserted: u64 = 0;
    let mut offset: i64 = 0;

    loop {
        let query = format!(
            "SELECT {} FROM {} LIMIT {} OFFSET {}",
            table.columns, table.name, BATCH_SIZE, offset
        );

        let mut stmt = match sqlite_conn.prepare(&query) {
            Ok(s) => s,
            Err(e) => {
                eprintln!("  {} - ERROR preparing: {}", table.name, e);
                break;
            }
        };
        
        let mut rows = match stmt.query([]) {
            Ok(r) => r,
            Err(e) => {
                eprintln!("  {} - ERROR querying: {}", table.name, e);
                break;
            }
        };

        let mut batch_count = 0u64;
        let mut value_batches: Vec<String> = Vec::with_capacity(INSERT_BATCH_SIZE);

        while let Ok(Some(row)) = rows.next() {
            let mut values = Vec::with_capacity(col_count);
            for i in 0..col_count {
                if let Ok(val) = row.get_ref(i) {
                    values.push(escape_sql_value(val.into()));
                } else {
                    values.push("NULL".to_string());
                }
            }

            value_batches.push(format!("({})", values.join(", ")));
            batch_count += 1;
            total_inserted += 1;

            if value_batches.len() >= INSERT_BATCH_SIZE {
                let values_str = std::mem::take(&mut value_batches).join(", ");
                if tx.send(ImportMessage::InsertBatch {
                    table_name: turdb_table.clone(),
                    values: values_str,
                }).is_err() {
                    return;
                }
            }
        }

        if !value_batches.is_empty() {
            let values_str = value_batches.join(", ");
            if tx.send(ImportMessage::InsertBatch {
                table_name: turdb_table.clone(),
                values: values_str,
            }).is_err() {
                return;
            }
        }

        if batch_count == 0 || offset + BATCH_SIZE >= count {
            break;
        }

        offset += BATCH_SIZE;
    }

    let _ = tx.send(ImportMessage::TableDone {
        table_name: table.name.to_string(),
        rows: total_inserted,
        elapsed_secs: start.elapsed().as_secs_f64(),
    });
}

fn import_table(
    sqlite_conn: &Connection,
    turdb: &Database,
    table: &TableSchema,
) -> eyre::Result<u64> {
    turdb.execute(table.turdb_ddl)?;

    let count: i64 = sqlite_conn.query_row(
        &format!("SELECT COUNT(*) FROM {}", table.name),
        [],
        |row| row.get(0),
    )?;

    if count == 0 {
        println!("  {} - 0 rows (empty table)", table.name);
        return Ok(0);
    }

    let col_count = table.columns.split(',').count();
    let turdb_table = camel_to_snake(table.name);
    let mut total_inserted: u64 = 0;
    let mut offset: i64 = 0;

    let start = Instant::now();

    loop {
        let query = format!(
            "SELECT {} FROM {} LIMIT {} OFFSET {}",
            table.columns, table.name, BATCH_SIZE, offset
        );

        let mut stmt = sqlite_conn.prepare(&query)?;
        let mut rows = stmt.query([])?;

        let mut batch_count = 0u64;
        let mut value_batches: Vec<String> = Vec::with_capacity(INSERT_BATCH_SIZE);

        while let Some(row) = rows.next()? {
            let mut values = Vec::with_capacity(col_count);
            for i in 0..col_count {
                let val = row.get_ref(i)?.into();
                values.push(escape_sql_value(val));
            }

            value_batches.push(format!("({})", values.join(", ")));
            batch_count += 1;
            total_inserted += 1;

            if value_batches.len() >= INSERT_BATCH_SIZE {
                let insert_sql = format!(
                    "INSERT INTO {} VALUES {}",
                    turdb_table,
                    value_batches.join(", ")
                );
                turdb.execute(&insert_sql)?;
                value_batches.clear();
            }

            if total_inserted.is_multiple_of(PROGRESS_INTERVAL) {
                let elapsed = start.elapsed().as_secs_f64();
                let rate = total_inserted as f64 / elapsed;
                println!(
                    "  {} - {}/{} rows ({:.0} rows/sec)",
                    table.name, total_inserted, count, rate
                );
            }
        }

        if !value_batches.is_empty() {
            let insert_sql = format!(
                "INSERT INTO {} VALUES {}",
                turdb_table,
                value_batches.join(", ")
            );
            turdb.execute(&insert_sql)?;
        }

        if batch_count == 0 {
            break;
        }

        offset += BATCH_SIZE;

        if offset >= count {
            break;
        }
    }

    let elapsed = start.elapsed();
    let rate = total_inserted as f64 / elapsed.as_secs_f64();
    println!(
        "  {} - DONE: {} rows in {:.2}s ({:.0} rows/sec)",
        table.name,
        total_inserted,
        elapsed.as_secs_f64(),
        rate
    );

    Ok(total_inserted)
}

fn import_table_with_txn(
    sqlite_conn: &Connection,
    turdb: &Database,
    table: &TableSchema,
) -> eyre::Result<u64> {
    turdb.execute(table.turdb_ddl)?;

    let count: i64 = sqlite_conn.query_row(
        &format!("SELECT COUNT(*) FROM {}", table.name),
        [],
        |row| row.get(0),
    )?;

    if count == 0 {
        println!("  {} - 0 rows (empty table)", table.name);
        return Ok(0);
    }

    let col_count = table.columns.split(',').count();
    let turdb_table = camel_to_snake(table.name);
    let mut total_inserted: u64 = 0;
    let mut offset: i64 = 0;

    let start = Instant::now();

    turdb.execute("BEGIN")?;

    loop {
        let query = format!(
            "SELECT {} FROM {} LIMIT {} OFFSET {}",
            table.columns, table.name, BATCH_SIZE, offset
        );

        let mut stmt = sqlite_conn.prepare(&query)?;
        let mut rows = stmt.query([])?;

        let mut batch_count = 0u64;
        let mut value_batches: Vec<String> = Vec::with_capacity(INSERT_BATCH_SIZE);

        while let Some(row) = rows.next()? {
            let mut values = Vec::with_capacity(col_count);
            for i in 0..col_count {
                let val = row.get_ref(i)?.into();
                values.push(escape_sql_value(val));
            }

            value_batches.push(format!("({})", values.join(", ")));
            batch_count += 1;
            total_inserted += 1;

            if value_batches.len() >= INSERT_BATCH_SIZE {
                let insert_sql = format!(
                    "INSERT INTO {} VALUES {}",
                    turdb_table,
                    value_batches.join(", ")
                );
                turdb.execute(&insert_sql)?;
                value_batches.clear();
            }

            if total_inserted.is_multiple_of(PROGRESS_INTERVAL) {
                let elapsed = start.elapsed().as_secs_f64();
                let rate = total_inserted as f64 / elapsed;
                println!(
                    "  {} - {}/{} rows ({:.0} rows/sec)",
                    table.name, total_inserted, count, rate
                );
            }
        }

        if !value_batches.is_empty() {
            let insert_sql = format!(
                "INSERT INTO {} VALUES {}",
                turdb_table,
                value_batches.join(", ")
            );
            turdb.execute(&insert_sql)?;
        }

        if batch_count == 0 {
            break;
        }

        offset += BATCH_SIZE;

        if offset >= count {
            break;
        }
    }

    turdb.execute("COMMIT")?;

    let elapsed = start.elapsed();
    let rate = total_inserted as f64 / elapsed.as_secs_f64();
    println!(
        "  {} - DONE: {} rows in {:.2}s ({:.0} rows/sec)",
        table.name,
        total_inserted,
        elapsed.as_secs_f64(),
        rate
    );

    Ok(total_inserted)
}

#[test]
fn import_all_tables_parallel() {
    if !sqlite_db_exists() {
        eprintln!("Skipping test: SQLite database not found at {}", SQLITE_DB_PATH);
        return;
    }

    if Path::new(TURDB_PATH).exists() {
        std::fs::remove_dir_all(TURDB_PATH).expect("Failed to remove existing TurDB directory");
        println!("Removed existing TurDB directory at {}", TURDB_PATH);
    }

    let db = Database::create(TURDB_PATH).unwrap();
    
    db.execute("PRAGMA WAL=ON").expect("Failed to enable WAL");
    db.execute("PRAGMA synchronous=OFF").expect("Failed to set synchronous mode");
    db.execute("SET foreign_keys = OFF").expect("Failed to set foreign keys");
    
    println!("\n=== Starting Parallel SQLite to TurDB Import ===\n");
    println!("Reading from {} tables in parallel...\n", TABLES.len());

    let overall_start = Instant::now();
    
    let (tx, rx) = mpsc::sync_channel::<ImportMessage>(CHANNEL_BUFFER);

    let reader_handles: Vec<_> = TABLES
        .iter()
        .map(|table| {
            let tx = tx.clone();
            thread::spawn(move || {
                read_table_parallel(table, tx);
            })
        })
        .collect();

    drop(tx);

    let mut total_rows: u64 = 0;
    let mut tables_done = 0;
    let mut insert_count = 0u64;

    for msg in rx {
        match msg {
            ImportMessage::CreateTable { ddl } => {
                if let Err(e) = db.execute(&ddl) {
                    eprintln!("ERROR creating table: {}", e);
                }
            }
            ImportMessage::InsertBatch { table_name, values } => {
                let insert_sql = format!("INSERT INTO {} VALUES {}", table_name, values);
                if let Err(e) = db.execute(&insert_sql) {
                    eprintln!("ERROR inserting into {}: {}", table_name, e);
                }
                insert_count += 1;
                if insert_count.is_multiple_of(100) {
                    let elapsed = overall_start.elapsed().as_secs_f64();
                    print!("\r  Batches: {} | Rows: ~{} | Time: {:.1}s    ", 
                           insert_count, insert_count * INSERT_BATCH_SIZE as u64, elapsed);
                    use std::io::Write;
                    let _ = std::io::stdout().flush();
                }
            }
            ImportMessage::TableDone { table_name, rows, elapsed_secs } => {
                total_rows += rows;
                tables_done += 1;
                let rate = if elapsed_secs > 0.0 { rows as f64 / elapsed_secs } else { 0.0 };
                println!("\n  {} - {} rows in {:.2}s ({:.0} rows/sec) [{}/{}]",
                         table_name, rows, elapsed_secs, rate, tables_done, TABLES.len());
            }

        }
    }

    for handle in reader_handles {
        let _ = handle.join();
    }

    db.execute("PRAGMA synchronous=FULL").expect("Failed to restore synchronous mode");
    db.execute("SET foreign_keys = ON").expect("Failed to set foreign keys");
    
    let overall_elapsed = overall_start.elapsed();
    let overall_rate = total_rows as f64 / overall_elapsed.as_secs_f64();

    println!("\n=== Import Complete ===");
    println!("Tables imported: {}", tables_done);
    println!("Total rows: {}", total_rows);
    println!("Total time: {:.2}s", overall_elapsed.as_secs_f64());
    println!("Average rate: {:.0} rows/sec", overall_rate);
}

#[test]
fn import_all_tables_from_sqlite() {
    if !sqlite_db_exists() {
        eprintln!("Skipping test: SQLite database not found at {}", SQLITE_DB_PATH);
        return;
    }

    let sqlite_conn = Connection::open(SQLITE_DB_PATH).expect("Failed to open SQLite DB");
    let db = Database::create(TURDB_PATH).unwrap();
    
    db.execute("PRAGMA WAL=ON").expect("Failed to enable WAL");
    db.execute("PRAGMA synchronous=OFF").expect("Failed to set synchronous mode");
    db.execute("SET foreign_keys = OFF").expect("Failed to set foreign keys");
    db.execute("SET cache_size = 1024").expect("Failed to set cache size");
    
    println!("\n=== Starting SQLite to TurDB Import ===\n");

    let overall_start = Instant::now();
    let mut total_rows: u64 = 0;
    let mut tables_imported = 0;

    for table in TABLES {
        match import_table(&sqlite_conn, &db, table) {
            Ok(rows) => {
                total_rows += rows;
                tables_imported += 1;
            }
            Err(e) => {
                eprintln!("  {} - ERROR: {}", table.name, e);
            }
        }
    }
    
    db.execute("PRAGMA synchronous=FULL").expect("Failed to restore synchronous mode");
    db.execute("SET foreign_keys = ON").expect("Failed to set foreign keys");
    
    let overall_elapsed = overall_start.elapsed();
    let overall_rate = total_rows as f64 / overall_elapsed.as_secs_f64();

    println!("\n=== Import Complete ===");
    println!("Tables imported: {}/{}", tables_imported, TABLES.len());
    println!("Total rows: {}", total_rows);
    println!("Total time: {:.2}s", overall_elapsed.as_secs_f64());
    println!("Average rate: {:.0} rows/sec", overall_rate);
}

#[test]
fn import_small_tables_only() {
    if !sqlite_db_exists() {
        eprintln!("Skipping test: SQLite database not found at {}", SQLITE_DB_PATH);
        return;
    }

    let small_tables = [
        "Tags", "KernelLanguages", "Organizations", "CompetitionTags",
        "Competitions", "DatasetTasks", "DatasetTaskSubmissions", "UserOrganizations",
    ];

    let sqlite_conn = Connection::open(SQLITE_DB_PATH).expect("Failed to open SQLite DB");
    let db = Database::create(TURDB_PATH).unwrap();
    db.execute("PRAGMA WAL=ON").expect("Failed to enable WAL");
    db.execute("PRAGMA synchronous=OFF").expect("Failed to set synchronous mode");

    println!("\n=== Importing Small Tables Only ===\n");

    let mut total_rows: u64 = 0;

    for table in TABLES {
        if small_tables.contains(&table.name) {
            match import_table(&sqlite_conn, &db, table) {
                Ok(rows) => total_rows += rows,
                Err(e) => eprintln!("  {} - ERROR: {}", table.name, e),
            }
        }
    }

    db.execute("PRAGMA synchronous=FULL").expect("Failed to restore synchronous mode");

    println!("\nTotal rows imported: {}", total_rows);
    assert!(total_rows > 0, "Should have imported some rows");
}

#[test]
fn import_users_table_batch() {
    if !sqlite_db_exists() {
        eprintln!("Skipping test: SQLite database not found at {}", SQLITE_DB_PATH);
        return;
    }

    let sqlite_conn = Connection::open(SQLITE_DB_PATH).expect("Failed to open SQLite DB");
    let db = Database::create(TURDB_PATH).unwrap();
    
    db.execute("PRAGMA WAL=ON").expect("Failed to enable WAL");
    db.execute("PRAGMA synchronous=OFF").expect("Failed to set synchronous mode");

    let users_table = TABLES.iter().find(|t| t.name == "Users").unwrap();

    println!("\n=== Importing Users Table (16M rows) ===\n");

    match import_table(&sqlite_conn, &db, users_table) {
        Ok(rows) => {
            println!("Imported {} user rows", rows);
            assert!(rows > 15_000_000, "Should have imported 16M+ users");
        }
        Err(e) => panic!("Failed to import users: {}", e),
    }
    
    db.execute("PRAGMA synchronous=FULL").expect("Failed to restore synchronous mode");
}

#[test]
fn import_users_table_with_txn() {
    if !sqlite_db_exists() {
        eprintln!("Skipping test: SQLite database not found at {}", SQLITE_DB_PATH);
        return;
    }

    let sqlite_conn = Connection::open(SQLITE_DB_PATH).expect("Failed to open SQLite DB");
    let db = Database::create(TURDB_PATH).unwrap();
    
    db.execute("PRAGMA WAL=ON").expect("Failed to enable WAL");
    db.execute("PRAGMA synchronous=OFF").expect("Failed to set synchronous mode");

    let users_table = TABLES.iter().find(|t| t.name == "Users").unwrap();

    println!("\n=== Importing Users Table with Transaction (16M rows) ===\n");

    match import_table_with_txn(&sqlite_conn, &db, users_table) {
        Ok(rows) => {
            println!("Imported {} user rows", rows);
            assert!(rows > 15_000_000, "Should have imported 16M+ users");
        }
        Err(e) => panic!("Failed to import users: {}", e),
    }
    
    db.execute("PRAGMA synchronous=FULL").expect("Failed to restore synchronous mode");
}

#[test]
fn verify_batch_import_integrity() {
    if !sqlite_db_exists() {
        eprintln!("Skipping test: SQLite database not found at {}", SQLITE_DB_PATH);
        return;
    }

    let sqlite_conn = Connection::open(SQLITE_DB_PATH).expect("Failed to open SQLite DB");
    let db = Database::create(TURDB_PATH).unwrap();

    let tags_table = TABLES.iter().find(|t| t.name == "Tags").unwrap();
    import_table(&sqlite_conn, &db, tags_table).unwrap();

    let sqlite_count: i64 = sqlite_conn
        .query_row("SELECT COUNT(*) FROM Tags", [], |row| row.get(0))
        .unwrap();

    let turdb_rows = db.query("SELECT COUNT(*) FROM tags").unwrap();
    let turdb_count = match &turdb_rows[0].values[0] {
        turdb::OwnedValue::Int(n) => *n,
        _ => panic!("Expected INT count"),
    };

    assert_eq!(
        sqlite_count, turdb_count,
        "Row counts must match: SQLite={}, TurDB={}",
        sqlite_count, turdb_count
    );

    println!("Verified: {} rows in both databases", turdb_count);
}
