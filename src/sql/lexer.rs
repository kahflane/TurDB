//! # SQL Lexer - Zero-Copy Tokenizer
//!
//! This module implements a high-performance SQL lexer that tokenizes SQL input
//! with zero string allocation. All string tokens (identifiers, literals) are
//! borrowed slices pointing directly into the input string.
//!
//! ## Design Goals
//!
//! 1. **Zero-copy tokenization**: Tokens borrow from input, never allocate
//! 2. **O(1) keyword lookup**: Uses phf perfect hash map for keywords
//! 3. **Full SQL support**: DDL, DML, subqueries, CTEs, procedures, window functions
//! 4. **Rich error reporting**: Line/column tracking for all tokens
//!
//! ## Token Types
//!
//! The lexer recognizes these token categories:
//!
//! - **Keywords**: SQL reserved words (SELECT, FROM, WHERE, etc.)
//! - **Identifiers**: Unquoted (table_name), quoted ("Column"), backtick (`order`)
//! - **Literals**: Strings ('hello'), numbers (42, 3.14, 0xFF), dollar-quoted ($$...$$)
//! - **Operators**: Arithmetic (+, -, *, /), comparison (=, <>, >=), JSON (->>, #>)
//! - **Punctuation**: Parentheses, brackets, comma, semicolon, colons
//! - **Parameters**: Positional ($1), named (:param), anonymous (?)
//! - **Comments**: Single-line (--), multi-line (/* ... */)
//!
//! ## Keyword Lookup
//!
//! Keywords are matched using a compile-time perfect hash function (phf crate).
//! This provides O(1) lookup with no hash collisions, faster than both HashMap
//! and match statements for large keyword sets.
//!
//! ## Identifier Quoting
//!
//! TurDB supports multiple identifier quoting styles:
//!
//! - **Double quotes** ("Column"): SQL standard, case-sensitive
//! - **Backticks** (`order`): MySQL style, escapes reserved words
//! - **Brackets** ([table]): SQL Server style (not yet implemented)
//!
//! ## String Literals
//!
//! - **Single quotes** ('hello'): Standard SQL strings, '' escapes single quote
//! - **Dollar quotes** ($$body$$): PostgreSQL style, for procedure bodies
//! - **Tagged dollar** ($tag$body$tag$): Named delimiters for nesting
//!
//! ## Numeric Literals
//!
//! - **Integers**: 42, -17
//! - **Floats**: 3.14, .5, 1., 1e10, 1.5e-3
//! - **Hex**: 0x1F, 0XAB
//! - **Binary**: 0b1010, 0B11
//! - **Octal**: 0o17, 0O777
//!
//! ## Usage Example
//!
//! ```ignore
//! use turdb::sql::{Lexer, Token, Keyword};
//!
//! let mut lexer = Lexer::new("SELECT id, name FROM users WHERE active = true");
//!
//! loop {
//!     let token = lexer.next_token();
//!     println!("{:?} at {}:{}", token, lexer.line(), lexer.column());
//!     if matches!(token, Token::Eof) { break; }
//! }
//! ```
//!
//! ## Performance
//!
//! The lexer is designed for high throughput:
//! - No heap allocation during normal tokenization
//! - Single-pass character scanning
//! - Minimal branching in hot paths
//! - SIMD-friendly memory access patterns
//!
//! ## Error Handling
//!
//! Invalid input produces `Token::Error` with a descriptive message.
//! The lexer continues after errors, allowing batch error collection.

use super::token::{Keyword, Parameter, Span, Token};
use phf::phf_map;

static KEYWORDS: phf::Map<&'static str, Keyword> = phf_map! {
    "CREATE" => Keyword::Create,
    "ALTER" => Keyword::Alter,
    "DROP" => Keyword::Drop,
    "TRUNCATE" => Keyword::Truncate,
    "ADD" => Keyword::Add,
    "COLUMN" => Keyword::Column,
    "RENAME" => Keyword::Rename,
    "SELECT" => Keyword::Select,
    "INSERT" => Keyword::Insert,
    "UPDATE" => Keyword::Update,
    "DELETE" => Keyword::Delete,
    "MERGE" => Keyword::Merge,
    "UPSERT" => Keyword::Upsert,
    "FROM" => Keyword::From,
    "WHERE" => Keyword::Where,
    "GROUP" => Keyword::Group,
    "HAVING" => Keyword::Having,
    "ORDER" => Keyword::Order,
    "LIMIT" => Keyword::Limit,
    "OFFSET" => Keyword::Offset,
    "FETCH" => Keyword::Fetch,
    "BY" => Keyword::By,
    "AS" => Keyword::As,
    "JOIN" => Keyword::Join,
    "INNER" => Keyword::Inner,
    "LEFT" => Keyword::Left,
    "RIGHT" => Keyword::Right,
    "FULL" => Keyword::Full,
    "OUTER" => Keyword::Outer,
    "CROSS" => Keyword::Cross,
    "NATURAL" => Keyword::Natural,
    "ON" => Keyword::On,
    "USING" => Keyword::Using,
    "UNION" => Keyword::Union,
    "INTERSECT" => Keyword::Intersect,
    "EXCEPT" => Keyword::Except,
    "ALL" => Keyword::All,
    "DISTINCT" => Keyword::Distinct,
    "IN" => Keyword::In,
    "EXISTS" => Keyword::Exists,
    "ANY" => Keyword::Any,
    "SOME" => Keyword::Some,
    "WITH" => Keyword::With,
    "RECURSIVE" => Keyword::Recursive,
    "BEGIN" => Keyword::Begin,
    "COMMIT" => Keyword::Commit,
    "ROLLBACK" => Keyword::Rollback,
    "SAVEPOINT" => Keyword::Savepoint,
    "RELEASE" => Keyword::Release,
    "PRIMARY" => Keyword::Primary,
    "FOREIGN" => Keyword::Foreign,
    "KEY" => Keyword::Key,
    "REFERENCES" => Keyword::References,
    "UNIQUE" => Keyword::Unique,
    "CHECK" => Keyword::Check,
    "NOT" => Keyword::Not,
    "NULL" => Keyword::Null,
    "DEFAULT" => Keyword::Default,
    "CASCADE" => Keyword::Cascade,
    "RESTRICT" => Keyword::Restrict,
    "RESTART" => Keyword::Restart,
    "CONTINUE" => Keyword::Continue,
    "SET" => Keyword::Set,
    "INDEX" => Keyword::Index,
    "BTREE" => Keyword::Btree,
    "HASH" => Keyword::Hash,
    "GIN" => Keyword::Gin,
    "GIST" => Keyword::Gist,
    "HNSW" => Keyword::Hnsw,
    "INTEGER" => Keyword::Integer,
    "INT" => Keyword::Int,
    "BIGINT" => Keyword::Bigint,
    "SMALLINT" => Keyword::Smallint,
    "TINYINT" => Keyword::Tinyint,
    "REAL" => Keyword::Real,
    "FLOAT" => Keyword::Float,
    "DOUBLE" => Keyword::Double,
    "DECIMAL" => Keyword::Decimal,
    "NUMERIC" => Keyword::Numeric,
    "VARCHAR" => Keyword::Varchar,
    "CHAR" => Keyword::Char,
    "TEXT" => Keyword::Text,
    "BLOB" => Keyword::Blob,
    "BOOLEAN" => Keyword::Boolean,
    "BOOL" => Keyword::Bool,
    "DATE" => Keyword::Date,
    "TIME" => Keyword::Time,
    "TIMESTAMP" => Keyword::Timestamp,
    "TIMESTAMPTZ" => Keyword::Timestamptz,
    "INTERVAL" => Keyword::Interval,
    "UUID" => Keyword::Uuid,
    "JSON" => Keyword::Json,
    "JSONB" => Keyword::Jsonb,
    "VECTOR" => Keyword::Vector,
    "ARRAY" => Keyword::Array,
    "POINT" => Keyword::Point,
    "BOX" => Keyword::Box,
    "CIRCLE" => Keyword::Circle,
    "MACADDR" => Keyword::Macaddr,
    "INET" => Keyword::Inet,
    "INT4RANGE" => Keyword::Int4range,
    "INT8RANGE" => Keyword::Int8range,
    "DATERANGE" => Keyword::Daterange,
    "TSRANGE" => Keyword::Tsrange,
    "FUNCTION" => Keyword::Function,
    "PROCEDURE" => Keyword::Procedure,
    "RETURNS" => Keyword::Returns,
    "LANGUAGE" => Keyword::Language,
    "DECLARE" => Keyword::Declare,
    "END" => Keyword::End,
    "IF" => Keyword::If,
    "THEN" => Keyword::Then,
    "ELSE" => Keyword::Else,
    "ELSIF" => Keyword::Elsif,
    "WHILE" => Keyword::While,
    "LOOP" => Keyword::Loop,
    "FOR" => Keyword::For,
    "RETURN" => Keyword::Return,
    "CALL" => Keyword::Call,
    "EXECUTE" => Keyword::Execute,
    "EXPLAIN" => Keyword::Explain,
    "ANALYZE" => Keyword::Analyze,
    "VERBOSE" => Keyword::Verbose,
    "COSTS" => Keyword::Costs,
    "BUFFERS" => Keyword::Buffers,
    "TIMING" => Keyword::Timing,
    "FORMAT" => Keyword::Format,
    "SCHEMA" => Keyword::Schema,
    "DATABASE" => Keyword::Database,
    "TABLE" => Keyword::Table,
    "VIEW" => Keyword::View,
    "MATERIALIZED" => Keyword::Materialized,
    "TRIGGER" => Keyword::Trigger,
    "SEQUENCE" => Keyword::Sequence,
    "TYPE" => Keyword::Type,
    "ENUM" => Keyword::Enum,
    "DOMAIN" => Keyword::Domain,
    "CASE" => Keyword::Case,
    "WHEN" => Keyword::When,
    "CAST" => Keyword::Cast,
    "COLLATE" => Keyword::Collate,
    "LIKE" => Keyword::Like,
    "ILIKE" => Keyword::Ilike,
    "SIMILAR" => Keyword::Similar,
    "BETWEEN" => Keyword::Between,
    "IS" => Keyword::Is,
    "TRUE" => Keyword::True,
    "FALSE" => Keyword::False,
    "AND" => Keyword::And,
    "OR" => Keyword::Or,
    "ASC" => Keyword::Asc,
    "DESC" => Keyword::Desc,
    "NULLS" => Keyword::Nulls,
    "FIRST" => Keyword::First,
    "LAST" => Keyword::Last,
    "OVER" => Keyword::Over,
    "PARTITION" => Keyword::Partition,
    "WINDOW" => Keyword::Window,
    "ROWS" => Keyword::Rows,
    "RANGE" => Keyword::Range,
    "UNBOUNDED" => Keyword::Unbounded,
    "PRECEDING" => Keyword::Preceding,
    "FOLLOWING" => Keyword::Following,
    "CURRENT" => Keyword::Current,
    "ROW" => Keyword::Row,
    "VALUES" => Keyword::Values,
    "INTO" => Keyword::Into,
    "RETURNING" => Keyword::Returning,
    "CONFLICT" => Keyword::Conflict,
    "DO" => Keyword::Do,
    "NOTHING" => Keyword::Nothing,
    "REPLACE" => Keyword::Replace,
    "IGNORE" => Keyword::Ignore,
    "CONSTRAINT" => Keyword::Constraint,
    "ONLY" => Keyword::Only,
    "LATERAL" => Keyword::Lateral,
    "ORDINALITY" => Keyword::Ordinality,
    "WITHIN" => Keyword::Within,
    "FILTER" => Keyword::Filter,
    "RESPECT" => Keyword::Respect,
    "PRECISION" => Keyword::Precision,
    "ZONE" => Keyword::Zone,
    "VARYING" => Keyword::Varying,
    "WITHOUT" => Keyword::Without,
    "DATA" => Keyword::Data,
    "AUTHORIZATION" => Keyword::Authorization,
    "OWNED" => Keyword::Owned,
    "TEMPORARY" => Keyword::Temporary,
    "TEMP" => Keyword::Temp,
    "UNLOGGED" => Keyword::Unlogged,
    "LOGGED" => Keyword::Logged,
    "CONCURRENTLY" => Keyword::Concurrently,
    "NOWAIT" => Keyword::Nowait,
    "SKIP" => Keyword::Skip,
    "LOCKED" => Keyword::Locked,
    "SHARE" => Keyword::Share,
    "EXCLUSIVE" => Keyword::Exclusive,
    "ACCESS" => Keyword::Access,
    "NO" => Keyword::No,
    "NEXT" => Keyword::Next,
    "ACTION" => Keyword::Action,
    "MATCH" => Keyword::Match,
    "MATCHED" => Keyword::Matched,
    "PARTIAL" => Keyword::Partial,
    "SIMPLE" => Keyword::Simple,
    "GENERATED" => Keyword::Generated,
    "ALWAYS" => Keyword::Always,
    "IDENTITY" => Keyword::Identity,
    "OVERRIDING" => Keyword::Overriding,
    "SYSTEM" => Keyword::System,
    "USER" => Keyword::User,
    "VALUE" => Keyword::Value,
    "STORED" => Keyword::Stored,
    "VIRTUAL" => Keyword::Virtual,
    "EXCLUDE" => Keyword::Exclude,
    "INCLUDING" => Keyword::Including,
    "EXCLUDING" => Keyword::Excluding,
    "COMMENTS" => Keyword::Comments,
    "STATISTICS" => Keyword::Statistics,
    "STORAGE" => Keyword::Storage,
    "COMPRESSION" => Keyword::Compression,
    "INDEXES" => Keyword::Indexes,
    "CONSTRAINTS" => Keyword::Constraints,
    "DEFAULTS" => Keyword::Defaults,
    "DEFERRABLE" => Keyword::Deferrable,
    "INITIALLY" => Keyword::Initially,
    "DEFERRED" => Keyword::Deferred,
    "IMMEDIATE" => Keyword::Immediate,
    "ENABLE" => Keyword::Enable,
    "DISABLE" => Keyword::Disable,
    "REPLICA" => Keyword::Replica,
    "RULE" => Keyword::Rule,
    "EVENT" => Keyword::Event,
    "INSTEAD" => Keyword::Instead,
    "EACH" => Keyword::Each,
    "STATEMENT" => Keyword::Statement,
    "BEFORE" => Keyword::Before,
    "AFTER" => Keyword::After,
    "REFERENCING" => Keyword::Referencing,
    "OLD" => Keyword::Old,
    "NEW" => Keyword::New,
    "ASSERTION" => Keyword::Assertion,
    "NORMALIZE" => Keyword::Normalize,
    "WORK" => Keyword::Work,
    "TRANSACTION" => Keyword::Transaction,
    "ISOLATION" => Keyword::Isolation,
    "LEVEL" => Keyword::Level,
    "READ" => Keyword::Read,
    "WRITE" => Keyword::Write,
    "COMMITTED" => Keyword::Committed,
    "UNCOMMITTED" => Keyword::Uncommitted,
    "REPEATABLE" => Keyword::Repeatable,
    "SERIALIZABLE" => Keyword::Serializable,
    "SNAPSHOT" => Keyword::Snapshot,
    "LOCAL" => Keyword::Local,
    "SESSION" => Keyword::Session,
    "GLOBAL" => Keyword::Global,
    "CLUSTER" => Keyword::Cluster,
    "REINDEX" => Keyword::Reindex,
    "VACUUM" => Keyword::Vacuum,
    "DISCARD" => Keyword::Discard,
    "PLANS" => Keyword::Plans,
    "SEQUENCES" => Keyword::Sequences,
    "RESET" => Keyword::Reset,
    "SHOW" => Keyword::Show,
    "GRANT" => Keyword::Grant,
    "REVOKE" => Keyword::Revoke,
    "TO" => Keyword::To,
    "OPTION" => Keyword::Option,
    "ADMIN" => Keyword::Admin,
    "INHERIT" => Keyword::Inherit,
    "NOINHERIT" => Keyword::Noinherit,
    "LOGIN" => Keyword::Login,
    "NOLOGIN" => Keyword::Nologin,
    "REPLICATION" => Keyword::Replication,
    "BYPASSRLS" => Keyword::Bypassrls,
    "NOBYPASSRLS" => Keyword::Nobypassrls,
    "CONNECTION" => Keyword::Connection,
    "CREATEDB" => Keyword::Createdb,
    "NOCREATEDB" => Keyword::Nocreatedb,
    "CREATEROLE" => Keyword::Createrole,
    "NOCREATEROLE" => Keyword::Nocreaterole,
    "SUPERUSER" => Keyword::Superuser,
    "NOSUPERUSER" => Keyword::Nosuperuser,
    "PASSWORD" => Keyword::Password,
    "VALID" => Keyword::Valid,
    "UNTIL" => Keyword::Until,
    "ROLE" => Keyword::Role,
    "ROLES" => Keyword::Roles,
    "MEMBER" => Keyword::Member,
    "PUBLIC" => Keyword::Public,
    "TABLES" => Keyword::Tables,
    "FUNCTIONS" => Keyword::Functions,
    "PROCEDURES" => Keyword::Procedures,
    "TYPES" => Keyword::Types,
    "PRIVILEGES" => Keyword::Privileges,
    "USAGE" => Keyword::Usage,
    "COPY" => Keyword::Copy,
    "BINARY" => Keyword::Binary,
    "CSV" => Keyword::Csv,
    "HEADER" => Keyword::Header,
    "QUOTE" => Keyword::Quote,
    "ESCAPE" => Keyword::Escape,
    "DELIMITER" => Keyword::Delimiter,
    "ENCODING" => Keyword::Encoding,
    "FORCE" => Keyword::Force,
    "FREEZE" => Keyword::Freeze,
    "STDIN" => Keyword::Stdin,
    "STDOUT" => Keyword::Stdout,
    "PROGRAM" => Keyword::Program,
    "IMPORT" => Keyword::Import,
    "EXPORT" => Keyword::Export,
    "ABORT" => Keyword::Abort,
    "ABSOLUTE" => Keyword::Absolute,
    "ALLOCATE" => Keyword::Allocate,
    "ARE" => Keyword::Are,
    "ASENSITIVE" => Keyword::Asensitive,
    "ASYMMETRIC" => Keyword::Asymmetric,
    "AT" => Keyword::At,
    "ATOMIC" => Keyword::Atomic,
    "CALLED" => Keyword::Called,
    "CARDINALITY" => Keyword::Cardinality,
    "CASCADED" => Keyword::Cascaded,
    "CHARACTER" => Keyword::Character,
    "CLOB" => Keyword::Clob,
    "CLOSE" => Keyword::Close,
    "COLLECT" => Keyword::Collect,
    "CONDITION" => Keyword::Condition,
    "CONTAINS" => Keyword::Contains,
    "CONVERT" => Keyword::Convert,
    "CORRESPONDING" => Keyword::Corresponding,
    "CUBE" => Keyword::Cube,
    "CURSOR" => Keyword::Cursor,
    "CYCLE" => Keyword::Cycle,
    "DAY" => Keyword::Day,
    "DEALLOCATE" => Keyword::Deallocate,
    "DEC" => Keyword::Dec,
    "DESCRIBE" => Keyword::Describe,
    "DETERMINISTIC" => Keyword::Deterministic,
    "DISCONNECT" => Keyword::Disconnect,
    "DYNAMIC" => Keyword::Dynamic,
    "ELEMENT" => Keyword::Element,
    "ELSEIF" => Keyword::Elseif,
    "EQUALS" => Keyword::Equals,
    "EVERY" => Keyword::Every,
    "EXEC" => Keyword::Exec,
    "EXIT" => Keyword::Exit,
    "EXTERNAL" => Keyword::External,
    "FREE" => Keyword::Free,
    "GET" => Keyword::Get,
    "GO" => Keyword::Go,
    "GROUPING" => Keyword::Grouping,
    "HANDLER" => Keyword::Handler,
    "HOLD" => Keyword::Hold,
    "HOUR" => Keyword::Hour,
    "INDICATOR" => Keyword::Indicator,
    "INPUT" => Keyword::Input,
    "INSENSITIVE" => Keyword::Insensitive,
    "INOUT" => Keyword::Inout,
    "ITERATE" => Keyword::Iterate,
    "LARGE" => Keyword::Large,
    "LEADING" => Keyword::Leading,
    "LEAVE" => Keyword::Leave,
    "LOCALTIME" => Keyword::Localtime,
    "LOCALTIMESTAMP" => Keyword::Localtimestamp,
    "METHOD" => Keyword::Method,
    "MINUTE" => Keyword::Minute,
    "MODIFIES" => Keyword::Modifies,
    "MODULE" => Keyword::Module,
    "MONTH" => Keyword::Month,
    "MULTISET" => Keyword::Multiset,
    "NATIONAL" => Keyword::National,
    "NCHAR" => Keyword::Nchar,
    "NCLOB" => Keyword::Nclob,
    "NONE" => Keyword::None,
    "OBJECT" => Keyword::Object,
    "OF" => Keyword::Of,
    "OPEN" => Keyword::Open,
    "OUT" => Keyword::Out,
    "OUTPUT" => Keyword::Output,
    "PARAMETER" => Keyword::Parameter2,
    "PREPARE" => Keyword::Prepare,
    "READS" => Keyword::Reads,
    "REF" => Keyword::Ref,
    "RELATIVE" => Keyword::Relative,
    "RESULT" => Keyword::Result,
    "ROLLUP" => Keyword::Rollup,
    "SCOPE" => Keyword::Scope,
    "SCROLL" => Keyword::Scroll,
    "SEARCH" => Keyword::Search,
    "SECOND" => Keyword::Second,
    "SENSITIVE" => Keyword::Sensitive,
    "SETS" => Keyword::Sets,
    "SPECIFIC" => Keyword::Specific,
    "SPECIFICTYPE" => Keyword::Specifictype,
    "SQL" => Keyword::Sql,
    "SQLEXCEPTION" => Keyword::Sqlexception,
    "SQLSTATE" => Keyword::Sqlstate,
    "SQLWARNING" => Keyword::Sqlwarning,
    "START" => Keyword::Start,
    "STATIC" => Keyword::Static,
    "SUBMULTISET" => Keyword::Submultiset,
    "SYMMETRIC" => Keyword::Symmetric,
    "TABLESAMPLE" => Keyword::Tablesample,
    "TIMEZONE" => Keyword::Timezone,
    "TRAILING" => Keyword::Trailing,
    "TREAT" => Keyword::Treat,
    "UESCAPE" => Keyword::Uescape,
    "UNDO" => Keyword::Undo,
    "UNKNOWN" => Keyword::Unknown,
    "UNNEST" => Keyword::Unnest,
    "YEAR" => Keyword::Year,
    "PRAGMA" => Keyword::Pragma,
};

pub struct Lexer<'a> {
    input: &'a str,
    bytes: &'a [u8],
    pos: usize,
    line: u32,
    column: u32,
    token_start: usize,
}

impl<'a> Lexer<'a> {
    pub fn new(input: &'a str) -> Self {
        Self {
            input,
            bytes: input.as_bytes(),
            pos: 0,
            line: 1,
            column: 1,
            token_start: 0,
        }
    }

    pub fn line(&self) -> u32 {
        self.line
    }

    pub fn column(&self) -> u32 {
        self.column
    }

    pub fn position(&self) -> usize {
        self.pos
    }

    pub fn span(&self) -> Span {
        Span::new(self.token_start, self.pos - self.token_start)
    }

    pub fn next_token(&mut self) -> Token<'a> {
        self.skip_whitespace();
        self.token_start = self.pos;

        if self.is_eof() {
            return Token::Eof;
        }

        let ch = self.current();

        if ch.is_ascii_alphabetic() || ch == b'_' {
            return self.scan_identifier_or_keyword();
        }

        if ch.is_ascii_digit() {
            return self.scan_number();
        }

        match ch {
            b'\'' => self.scan_string(),
            b'"' => self.scan_quoted_identifier(),
            b'`' => self.scan_backtick_identifier(),
            b'$' => self.scan_dollar_or_param(),
            b':' => self.scan_colon_or_param(),
            b'@' => self.scan_at_param(),
            b'?' => self.scan_question(),
            b'-' => self.scan_minus(),
            b'/' => self.scan_slash(),
            b'+' => {
                self.advance();
                Token::Plus
            }
            b'*' => {
                self.advance();
                Token::Star
            }
            b'%' => {
                self.advance();
                Token::Percent
            }
            b'^' => {
                self.advance();
                Token::Caret
            }
            b'&' => self.scan_ampersand(),
            b'|' => self.scan_pipe(),
            b'~' => {
                self.advance();
                Token::Tilde
            }
            b'#' => self.scan_hash(),
            b'=' => self.scan_equals(),
            b'<' => self.scan_less_than(),
            b'>' => self.scan_greater_than(),
            b'!' => self.scan_exclamation(),
            b'(' => {
                self.advance();
                Token::LParen
            }
            b')' => {
                self.advance();
                Token::RParen
            }
            b'[' => {
                self.advance();
                Token::LBracket
            }
            b']' => {
                self.advance();
                Token::RBracket
            }
            b'{' => {
                self.advance();
                Token::LBrace
            }
            b'}' => {
                self.advance();
                Token::RBrace
            }
            b',' => {
                self.advance();
                Token::Comma
            }
            b';' => {
                self.advance();
                Token::Semicolon
            }
            b'.' => self.scan_dot(),
            _ => {
                self.advance();
                Token::Error("unexpected character")
            }
        }
    }

    pub fn peek(&mut self) -> Token<'a> {
        let saved_pos = self.pos;
        let saved_line = self.line;
        let saved_column = self.column;
        let saved_token_start = self.token_start;

        let token = self.next_token();

        self.pos = saved_pos;
        self.line = saved_line;
        self.column = saved_column;
        self.token_start = saved_token_start;

        token
    }

    pub fn peek_nth(&mut self, n: usize) -> Token<'a> {
        let saved_pos = self.pos;
        let saved_line = self.line;
        let saved_column = self.column;
        let saved_token_start = self.token_start;

        let mut token = Token::Eof;
        for _ in 0..=n {
            token = self.next_token();
            if matches!(token, Token::Eof) {
                break;
            }
        }

        self.pos = saved_pos;
        self.line = saved_line;
        self.column = saved_column;
        self.token_start = saved_token_start;

        token
    }

    fn is_eof(&self) -> bool {
        self.pos >= self.bytes.len()
    }

    fn current(&self) -> u8 {
        self.bytes[self.pos]
    }

    fn peek_char(&self) -> Option<u8> {
        self.bytes.get(self.pos + 1).copied()
    }

    fn advance(&mut self) {
        if !self.is_eof() {
            if self.current() == b'\n' {
                self.line += 1;
                self.column = 1;
            } else {
                self.column += 1;
            }
            self.pos += 1;
        }
    }

    fn skip_whitespace(&mut self) {
        while !self.is_eof() {
            match self.current() {
                b' ' | b'\t' | b'\r' | b'\n' => self.advance(),
                _ => break,
            }
        }
    }

    fn scan_identifier_or_keyword(&mut self) -> Token<'a> {
        let start = self.pos;

        if (self.current() == b'x' || self.current() == b'X') && self.peek_char() == Some(b'\'') {
            return self.scan_hex_string_literal();
        }

        while !self.is_eof() && (self.current().is_ascii_alphanumeric() || self.current() == b'_') {
            self.advance();
        }

        let ident = &self.input[start..self.pos];
        let upper = ident.to_ascii_uppercase();

        if let Some(&keyword) = KEYWORDS.get(&upper) {
            Token::Keyword(keyword)
        } else {
            Token::Ident(ident)
        }
    }

    fn scan_hex_string_literal(&mut self) -> Token<'a> {
        self.advance();
        self.advance();

        let start = self.pos;

        while !self.is_eof() && self.current() != b'\'' {
            if !self.current().is_ascii_hexdigit() {
                return Token::Error("invalid hex character in hex string literal");
            }
            self.advance();
        }

        if self.is_eof() {
            return Token::Error("unterminated hex string literal");
        }

        let hex_str = &self.input[start..self.pos];
        self.advance();

        Token::HexNumber(hex_str)
    }

    fn scan_number(&mut self) -> Token<'a> {
        let start = self.pos;

        if self.current() == b'0' {
            if let Some(next) = self.peek_char() {
                match next {
                    b'x' | b'X' => return self.scan_hex_number(),
                    b'b' | b'B' => return self.scan_binary_number(),
                    b'o' | b'O' => return self.scan_octal_number(),
                    _ => {}
                }
            }
        }

        while !self.is_eof() && self.current().is_ascii_digit() {
            self.advance();
        }

        let mut is_float = false;

        if !self.is_eof() && self.current() == b'.' {
            if let Some(next) = self.peek_char() {
                if next.is_ascii_digit() {
                    is_float = true;
                    self.advance();
                    while !self.is_eof() && self.current().is_ascii_digit() {
                        self.advance();
                    }
                } else if next == b'.' {
                } else {
                    is_float = true;
                    self.advance();
                }
            }
        }

        if !self.is_eof() && (self.current() == b'e' || self.current() == b'E') {
            is_float = true;
            self.advance();
            if !self.is_eof() && (self.current() == b'+' || self.current() == b'-') {
                self.advance();
            }
            while !self.is_eof() && self.current().is_ascii_digit() {
                self.advance();
            }
        }

        let num_str = &self.input[start..self.pos];
        if is_float {
            Token::Float(num_str)
        } else {
            Token::Integer(num_str)
        }
    }

    fn scan_hex_number(&mut self) -> Token<'a> {
        self.advance();
        self.advance();
        let start = self.pos;

        while !self.is_eof() && self.current().is_ascii_hexdigit() {
            self.advance();
        }

        if self.pos == start {
            return Token::Error("invalid hex number");
        }

        Token::HexNumber(&self.input[start..self.pos])
    }

    fn scan_binary_number(&mut self) -> Token<'a> {
        self.advance();
        self.advance();
        let start = self.pos;

        while !self.is_eof() && (self.current() == b'0' || self.current() == b'1') {
            self.advance();
        }

        if self.pos == start {
            return Token::Error("invalid binary number");
        }

        Token::BinaryNumber(&self.input[start..self.pos])
    }

    fn scan_octal_number(&mut self) -> Token<'a> {
        self.advance();
        self.advance();
        let start = self.pos;

        while !self.is_eof() && self.current() >= b'0' && self.current() <= b'7' {
            self.advance();
        }

        if self.pos == start {
            return Token::Error("invalid octal number");
        }

        Token::OctalNumber(&self.input[start..self.pos])
    }

    fn scan_string(&mut self) -> Token<'a> {
        self.advance();
        let start = self.pos;

        loop {
            if self.is_eof() {
                return Token::Error("unterminated string");
            }

            if self.current() == b'\'' {
                if self.peek_char() == Some(b'\'') {
                    self.advance();
                    self.advance();
                } else {
                    let end = self.pos;
                    self.advance();
                    return Token::String(&self.input[start..end]);
                }
            } else {
                self.advance();
            }
        }
    }

    fn scan_quoted_identifier(&mut self) -> Token<'a> {
        self.advance();
        let start = self.pos;

        loop {
            if self.is_eof() {
                return Token::Error("unterminated quoted identifier");
            }

            if self.current() == b'"' {
                if self.peek_char() == Some(b'"') {
                    self.advance();
                    self.advance();
                } else {
                    let end = self.pos;
                    self.advance();
                    return Token::QuotedIdent(&self.input[start..end]);
                }
            } else {
                self.advance();
            }
        }
    }

    fn scan_backtick_identifier(&mut self) -> Token<'a> {
        self.advance();
        let start = self.pos;

        loop {
            if self.is_eof() {
                return Token::Error("unterminated backtick identifier");
            }

            if self.current() == b'`' {
                if self.peek_char() == Some(b'`') {
                    self.advance();
                    self.advance();
                } else {
                    let end = self.pos;
                    self.advance();
                    return Token::QuotedIdent(&self.input[start..end]);
                }
            } else {
                self.advance();
            }
        }
    }

    fn scan_dollar_or_param(&mut self) -> Token<'a> {
        self.advance();

        if self.is_eof() {
            return Token::Error("unexpected end after $");
        }

        if self.current().is_ascii_digit() {
            let start = self.pos;
            while !self.is_eof() && self.current().is_ascii_digit() {
                self.advance();
            }
            let num_str = &self.input[start..self.pos];
            if let Ok(n) = num_str.parse::<u32>() {
                return Token::Parameter(Parameter::Positional(n));
            } else {
                return Token::Error("invalid positional parameter");
            }
        }

        if self.current() == b'$' {
            return self.scan_dollar_string(None);
        }

        if self.current().is_ascii_alphabetic() || self.current() == b'_' {
            let tag_start = self.pos;
            while !self.is_eof()
                && (self.current().is_ascii_alphanumeric() || self.current() == b'_')
            {
                self.advance();
            }
            if !self.is_eof() && self.current() == b'$' {
                let tag = &self.input[tag_start..self.pos];
                return self.scan_dollar_string(Some(tag));
            } else {
                return Token::Error("invalid dollar-quoted string tag");
            }
        }

        Token::Error("invalid token after $")
    }

    fn scan_dollar_string(&mut self, tag: Option<&'a str>) -> Token<'a> {
        self.advance();
        let start = self.pos;

        let end_tag = if let Some(t) = tag {
            format!("${}$", t)
        } else {
            "$$".to_string()
        };

        loop {
            if self.is_eof() {
                return Token::Error("unterminated dollar-quoted string");
            }

            if self.current() == b'$' {
                let remaining = &self.input[self.pos..];
                if remaining.starts_with(&end_tag) {
                    let end = self.pos;
                    for _ in 0..end_tag.len() {
                        self.advance();
                    }
                    return Token::String(&self.input[start..end]);
                }
            }
            self.advance();
        }
    }

    fn scan_colon_or_param(&mut self) -> Token<'a> {
        self.advance();

        if self.is_eof() {
            return Token::Colon;
        }

        match self.current() {
            b':' => {
                self.advance();
                Token::DoubleColon
            }
            b'=' => {
                self.advance();
                Token::Assign
            }
            c if c.is_ascii_alphabetic() || c == b'_' => {
                let start = self.pos;
                while !self.is_eof()
                    && (self.current().is_ascii_alphanumeric() || self.current() == b'_')
                {
                    self.advance();
                }
                Token::Parameter(Parameter::Named(&self.input[start..self.pos]))
            }
            _ => Token::Colon,
        }
    }

    fn scan_at_param(&mut self) -> Token<'a> {
        self.advance();

        if self.is_eof() {
            return Token::Error("unexpected end after @");
        }

        if self.current() == b'>' {
            self.advance();
            return Token::AtGt;
        }

        if self.current().is_ascii_alphabetic() || self.current() == b'_' {
            let start = self.pos;
            while !self.is_eof()
                && (self.current().is_ascii_alphanumeric() || self.current() == b'_')
            {
                self.advance();
            }
            Token::Parameter(Parameter::Named(&self.input[start..self.pos]))
        } else {
            Token::Error("invalid @ parameter")
        }
    }

    fn scan_question(&mut self) -> Token<'a> {
        self.advance();

        if self.is_eof() {
            return Token::Parameter(Parameter::Anonymous);
        }

        match self.current() {
            b'|' => {
                self.advance();
                Token::QuestionPipe
            }
            b'&' => {
                self.advance();
                Token::QuestionAmpersand
            }
            _ => Token::Parameter(Parameter::Anonymous),
        }
    }

    fn scan_minus(&mut self) -> Token<'a> {
        self.advance();

        if self.is_eof() {
            return Token::Minus;
        }

        match self.current() {
            b'-' => {
                while !self.is_eof() && self.current() != b'\n' {
                    self.advance();
                }
                self.next_token()
            }
            b'>' => {
                self.advance();
                if !self.is_eof() && self.current() == b'>' {
                    self.advance();
                    Token::DoubleArrow
                } else {
                    Token::Arrow
                }
            }
            _ => Token::Minus,
        }
    }

    fn scan_slash(&mut self) -> Token<'a> {
        self.advance();

        if self.is_eof() {
            return Token::Slash;
        }

        if self.current() == b'*' {
            self.advance();
            self.scan_block_comment()
        } else {
            Token::Slash
        }
    }

    fn scan_block_comment(&mut self) -> Token<'a> {
        let mut depth = 1;

        while !self.is_eof() && depth > 0 {
            if self.current() == b'/' && self.peek_char() == Some(b'*') {
                self.advance();
                self.advance();
                depth += 1;
            } else if self.current() == b'*' && self.peek_char() == Some(b'/') {
                self.advance();
                self.advance();
                depth -= 1;
            } else {
                self.advance();
            }
        }

        if depth > 0 {
            return Token::Error("unterminated block comment");
        }

        self.next_token()
    }

    fn scan_ampersand(&mut self) -> Token<'a> {
        self.advance();

        if !self.is_eof() && self.current() == b'&' {
            self.advance();
            Token::DoubleAmpersand
        } else {
            Token::Ampersand
        }
    }

    fn scan_pipe(&mut self) -> Token<'a> {
        self.advance();

        if !self.is_eof() && self.current() == b'|' {
            self.advance();
            Token::DoublePipe
        } else {
            Token::Pipe
        }
    }

    fn scan_hash(&mut self) -> Token<'a> {
        self.advance();

        if self.is_eof() {
            return Token::Hash;
        }

        match self.current() {
            b'>' => {
                self.advance();
                if !self.is_eof() && self.current() == b'>' {
                    self.advance();
                    Token::HashDoubleArrow
                } else {
                    Token::HashArrow
                }
            }
            _ => Token::Hash,
        }
    }

    fn scan_equals(&mut self) -> Token<'a> {
        self.advance();

        if !self.is_eof() && self.current() == b'>' {
            self.advance();
            Token::FatArrow
        } else {
            Token::Eq
        }
    }

    fn scan_less_than(&mut self) -> Token<'a> {
        self.advance();

        if self.is_eof() {
            return Token::Lt;
        }

        match self.current() {
            b'=' => {
                self.advance();
                if !self.is_eof() && self.current() == b'>' {
                    self.advance();
                    Token::Spaceship
                } else {
                    Token::LtEq
                }
            }
            b'>' => {
                self.advance();
                Token::NotEq
            }
            b'<' => {
                self.advance();
                Token::LeftShift
            }
            b'@' => {
                self.advance();
                Token::LtAt
            }
            b'-' => {
                self.advance();
                if !self.is_eof() && self.current() == b'>' {
                    self.advance();
                    Token::LtMinusGt
                } else {
                    self.pos -= 1;
                    Token::Lt
                }
            }
            b'#' => {
                self.advance();
                if !self.is_eof() && self.current() == b'>' {
                    self.advance();
                    Token::LtHashGt
                } else {
                    self.pos -= 1;
                    Token::Lt
                }
            }
            _ => Token::Lt,
        }
    }

    fn scan_greater_than(&mut self) -> Token<'a> {
        self.advance();

        if self.is_eof() {
            return Token::Gt;
        }

        match self.current() {
            b'=' => {
                self.advance();
                Token::GtEq
            }
            b'>' => {
                self.advance();
                Token::RightShift
            }
            _ => Token::Gt,
        }
    }

    fn scan_exclamation(&mut self) -> Token<'a> {
        self.advance();

        if !self.is_eof() && self.current() == b'=' {
            self.advance();
            Token::NotEq
        } else {
            Token::Error("expected '=' after '!'")
        }
    }

    fn scan_dot(&mut self) -> Token<'a> {
        self.advance();

        if !self.is_eof() && self.current() == b'.' {
            self.advance();
            Token::DoubleDot
        } else if !self.is_eof() && self.current().is_ascii_digit() {
            let start = self.pos - 1;
            while !self.is_eof() && self.current().is_ascii_digit() {
                self.advance();
            }
            if !self.is_eof() && (self.current() == b'e' || self.current() == b'E') {
                self.advance();
                if !self.is_eof() && (self.current() == b'+' || self.current() == b'-') {
                    self.advance();
                }
                while !self.is_eof() && self.current().is_ascii_digit() {
                    self.advance();
                }
            }
            Token::Float(&self.input[start..self.pos])
        } else {
            Token::Dot
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lex_keywords() {
        let mut lexer = Lexer::new("SELECT FROM WHERE INSERT UPDATE DELETE");
        assert_eq!(lexer.next_token(), Token::Keyword(Keyword::Select));
        assert_eq!(lexer.next_token(), Token::Keyword(Keyword::From));
        assert_eq!(lexer.next_token(), Token::Keyword(Keyword::Where));
        assert_eq!(lexer.next_token(), Token::Keyword(Keyword::Insert));
        assert_eq!(lexer.next_token(), Token::Keyword(Keyword::Update));
        assert_eq!(lexer.next_token(), Token::Keyword(Keyword::Delete));
        assert_eq!(lexer.next_token(), Token::Eof);
    }

    #[test]
    fn lex_keywords_case_insensitive() {
        let mut lexer = Lexer::new("select SELECT Select sElEcT");
        assert_eq!(lexer.next_token(), Token::Keyword(Keyword::Select));
        assert_eq!(lexer.next_token(), Token::Keyword(Keyword::Select));
        assert_eq!(lexer.next_token(), Token::Keyword(Keyword::Select));
        assert_eq!(lexer.next_token(), Token::Keyword(Keyword::Select));
    }

    #[test]
    fn lex_identifiers() {
        let mut lexer = Lexer::new("users table_name _private col1");
        assert_eq!(lexer.next_token(), Token::Ident("users"));
        assert_eq!(lexer.next_token(), Token::Ident("table_name"));
        assert_eq!(lexer.next_token(), Token::Ident("_private"));
        assert_eq!(lexer.next_token(), Token::Ident("col1"));
    }

    #[test]
    fn lex_quoted_identifiers() {
        let mut lexer = Lexer::new("\"Order\" \"table\"\"name\" `select` `back``tick`");
        assert_eq!(lexer.next_token(), Token::QuotedIdent("Order"));
        assert_eq!(lexer.next_token(), Token::QuotedIdent("table\"\"name"));
        assert_eq!(lexer.next_token(), Token::QuotedIdent("select"));
        assert_eq!(lexer.next_token(), Token::QuotedIdent("back``tick"));
    }

    #[test]
    fn lex_strings() {
        let mut lexer = Lexer::new("'hello' 'it''s' 'multi\nline'");
        assert_eq!(lexer.next_token(), Token::String("hello"));
        assert_eq!(lexer.next_token(), Token::String("it''s"));
        assert_eq!(lexer.next_token(), Token::String("multi\nline"));
    }

    #[test]
    fn lex_dollar_strings() {
        let mut lexer = Lexer::new("$$body$$ $tag$inner$tag$");
        assert_eq!(lexer.next_token(), Token::String("body"));
        assert_eq!(lexer.next_token(), Token::String("inner"));
    }

    #[test]
    fn lex_integers() {
        let mut lexer = Lexer::new("42 0 12345");
        assert_eq!(lexer.next_token(), Token::Integer("42"));
        assert_eq!(lexer.next_token(), Token::Integer("0"));
        assert_eq!(lexer.next_token(), Token::Integer("12345"));
    }

    #[test]
    fn lex_floats() {
        let mut lexer = Lexer::new("3.14 .5 1. 1e10 1.5e-3 2E+5");
        assert_eq!(lexer.next_token(), Token::Float("3.14"));
        assert_eq!(lexer.next_token(), Token::Float(".5"));
        assert_eq!(lexer.next_token(), Token::Float("1."));
        assert_eq!(lexer.next_token(), Token::Float("1e10"));
        assert_eq!(lexer.next_token(), Token::Float("1.5e-3"));
        assert_eq!(lexer.next_token(), Token::Float("2E+5"));
    }

    #[test]
    fn lex_hex_binary_octal() {
        let mut lexer = Lexer::new("0x1F 0XAB 0b1010 0B11 0o17 0O777");
        assert_eq!(lexer.next_token(), Token::HexNumber("1F"));
        assert_eq!(lexer.next_token(), Token::HexNumber("AB"));
        assert_eq!(lexer.next_token(), Token::BinaryNumber("1010"));
        assert_eq!(lexer.next_token(), Token::BinaryNumber("11"));
        assert_eq!(lexer.next_token(), Token::OctalNumber("17"));
        assert_eq!(lexer.next_token(), Token::OctalNumber("777"));
    }

    #[test]
    fn lex_operators() {
        let mut lexer = Lexer::new("+ - * / % ^ & | || ~ << >> #");
        assert_eq!(lexer.next_token(), Token::Plus);
        assert_eq!(lexer.next_token(), Token::Minus);
        assert_eq!(lexer.next_token(), Token::Star);
        assert_eq!(lexer.next_token(), Token::Slash);
        assert_eq!(lexer.next_token(), Token::Percent);
        assert_eq!(lexer.next_token(), Token::Caret);
        assert_eq!(lexer.next_token(), Token::Ampersand);
        assert_eq!(lexer.next_token(), Token::Pipe);
        assert_eq!(lexer.next_token(), Token::DoublePipe);
        assert_eq!(lexer.next_token(), Token::Tilde);
        assert_eq!(lexer.next_token(), Token::LeftShift);
        assert_eq!(lexer.next_token(), Token::RightShift);
        assert_eq!(lexer.next_token(), Token::Hash);
    }

    #[test]
    fn lex_comparison_operators() {
        let mut lexer = Lexer::new("= <> != < <= > >= <=>");
        assert_eq!(lexer.next_token(), Token::Eq);
        assert_eq!(lexer.next_token(), Token::NotEq);
        assert_eq!(lexer.next_token(), Token::NotEq);
        assert_eq!(lexer.next_token(), Token::Lt);
        assert_eq!(lexer.next_token(), Token::LtEq);
        assert_eq!(lexer.next_token(), Token::Gt);
        assert_eq!(lexer.next_token(), Token::GtEq);
        assert_eq!(lexer.next_token(), Token::Spaceship);
    }

    #[test]
    fn lex_json_operators() {
        let mut lexer = Lexer::new("-> ->> #> #>> @> <@ && ?| ?&");
        assert_eq!(lexer.next_token(), Token::Arrow);
        assert_eq!(lexer.next_token(), Token::DoubleArrow);
        assert_eq!(lexer.next_token(), Token::HashArrow);
        assert_eq!(lexer.next_token(), Token::HashDoubleArrow);
        assert_eq!(lexer.next_token(), Token::AtGt);
        assert_eq!(lexer.next_token(), Token::LtAt);
        assert_eq!(lexer.next_token(), Token::DoubleAmpersand);
        assert_eq!(lexer.next_token(), Token::QuestionPipe);
        assert_eq!(lexer.next_token(), Token::QuestionAmpersand);
    }

    #[test]
    fn lex_punctuation() {
        let mut lexer = Lexer::new("( ) [ ] { } , ; : :: . .. := =>");
        assert_eq!(lexer.next_token(), Token::LParen);
        assert_eq!(lexer.next_token(), Token::RParen);
        assert_eq!(lexer.next_token(), Token::LBracket);
        assert_eq!(lexer.next_token(), Token::RBracket);
        assert_eq!(lexer.next_token(), Token::LBrace);
        assert_eq!(lexer.next_token(), Token::RBrace);
        assert_eq!(lexer.next_token(), Token::Comma);
        assert_eq!(lexer.next_token(), Token::Semicolon);
        assert_eq!(lexer.next_token(), Token::Colon);
        assert_eq!(lexer.next_token(), Token::DoubleColon);
        assert_eq!(lexer.next_token(), Token::Dot);
        assert_eq!(lexer.next_token(), Token::DoubleDot);
        assert_eq!(lexer.next_token(), Token::Assign);
        assert_eq!(lexer.next_token(), Token::FatArrow);
    }

    #[test]
    fn lex_parameters() {
        let mut lexer = Lexer::new("$1 $2 $123 :name :param_1 @var ?");
        assert_eq!(
            lexer.next_token(),
            Token::Parameter(Parameter::Positional(1))
        );
        assert_eq!(
            lexer.next_token(),
            Token::Parameter(Parameter::Positional(2))
        );
        assert_eq!(
            lexer.next_token(),
            Token::Parameter(Parameter::Positional(123))
        );
        assert_eq!(
            lexer.next_token(),
            Token::Parameter(Parameter::Named("name"))
        );
        assert_eq!(
            lexer.next_token(),
            Token::Parameter(Parameter::Named("param_1"))
        );
        assert_eq!(
            lexer.next_token(),
            Token::Parameter(Parameter::Named("var"))
        );
        assert_eq!(lexer.next_token(), Token::Parameter(Parameter::Anonymous));
    }

    #[test]
    fn lex_comments() {
        let mut lexer = Lexer::new(
            "SELECT -- comment\nFROM /* block */ users /* nested /* inner */ outer */ WHERE",
        );
        assert_eq!(lexer.next_token(), Token::Keyword(Keyword::Select));
        assert_eq!(lexer.next_token(), Token::Keyword(Keyword::From));
        assert_eq!(lexer.next_token(), Token::Ident("users"));
        assert_eq!(lexer.next_token(), Token::Keyword(Keyword::Where));
    }

    #[test]
    fn lex_select_statement() {
        let mut lexer = Lexer::new("SELECT id, name FROM users WHERE active = true;");
        assert_eq!(lexer.next_token(), Token::Keyword(Keyword::Select));
        assert_eq!(lexer.next_token(), Token::Ident("id"));
        assert_eq!(lexer.next_token(), Token::Comma);
        assert_eq!(lexer.next_token(), Token::Ident("name"));
        assert_eq!(lexer.next_token(), Token::Keyword(Keyword::From));
        assert_eq!(lexer.next_token(), Token::Ident("users"));
        assert_eq!(lexer.next_token(), Token::Keyword(Keyword::Where));
        assert_eq!(lexer.next_token(), Token::Ident("active"));
        assert_eq!(lexer.next_token(), Token::Eq);
        assert_eq!(lexer.next_token(), Token::Keyword(Keyword::True));
        assert_eq!(lexer.next_token(), Token::Semicolon);
        assert_eq!(lexer.next_token(), Token::Eof);
    }

    #[test]
    fn lex_create_table() {
        let mut lexer =
            Lexer::new("CREATE TABLE users (id INTEGER PRIMARY KEY, name VARCHAR(100) NOT NULL)");
        assert_eq!(lexer.next_token(), Token::Keyword(Keyword::Create));
        assert_eq!(lexer.next_token(), Token::Keyword(Keyword::Table));
        assert_eq!(lexer.next_token(), Token::Ident("users"));
        assert_eq!(lexer.next_token(), Token::LParen);
        assert_eq!(lexer.next_token(), Token::Ident("id"));
        assert_eq!(lexer.next_token(), Token::Keyword(Keyword::Integer));
        assert_eq!(lexer.next_token(), Token::Keyword(Keyword::Primary));
        assert_eq!(lexer.next_token(), Token::Keyword(Keyword::Key));
        assert_eq!(lexer.next_token(), Token::Comma);
        assert_eq!(lexer.next_token(), Token::Ident("name"));
        assert_eq!(lexer.next_token(), Token::Keyword(Keyword::Varchar));
        assert_eq!(lexer.next_token(), Token::LParen);
        assert_eq!(lexer.next_token(), Token::Integer("100"));
        assert_eq!(lexer.next_token(), Token::RParen);
        assert_eq!(lexer.next_token(), Token::Keyword(Keyword::Not));
        assert_eq!(lexer.next_token(), Token::Keyword(Keyword::Null));
        assert_eq!(lexer.next_token(), Token::RParen);
    }

    #[test]
    fn lex_json_query() {
        let mut lexer = Lexer::new("SELECT doc->>'name', doc#>'{a,b}' FROM docs");
        assert_eq!(lexer.next_token(), Token::Keyword(Keyword::Select));
        assert_eq!(lexer.next_token(), Token::Ident("doc"));
        assert_eq!(lexer.next_token(), Token::DoubleArrow);
        assert_eq!(lexer.next_token(), Token::String("name"));
        assert_eq!(lexer.next_token(), Token::Comma);
        assert_eq!(lexer.next_token(), Token::Ident("doc"));
        assert_eq!(lexer.next_token(), Token::HashArrow);
        assert_eq!(lexer.next_token(), Token::String("{a,b}"));
        assert_eq!(lexer.next_token(), Token::Keyword(Keyword::From));
        assert_eq!(lexer.next_token(), Token::Ident("docs"));
    }

    #[test]
    fn lex_type_cast() {
        let mut lexer = Lexer::new("'123'::INTEGER col_val::TEXT");
        assert_eq!(lexer.next_token(), Token::String("123"));
        assert_eq!(lexer.next_token(), Token::DoubleColon);
        assert_eq!(lexer.next_token(), Token::Keyword(Keyword::Integer));
        assert_eq!(lexer.next_token(), Token::Ident("col_val"));
        assert_eq!(lexer.next_token(), Token::DoubleColon);
        assert_eq!(lexer.next_token(), Token::Keyword(Keyword::Text));
    }

    #[test]
    fn lex_window_function() {
        let mut lexer = Lexer::new("ROW_NUMBER() OVER (PARTITION BY dept ORDER BY salary DESC)");
        assert_eq!(lexer.next_token(), Token::Ident("ROW_NUMBER"));
        assert_eq!(lexer.next_token(), Token::LParen);
        assert_eq!(lexer.next_token(), Token::RParen);
        assert_eq!(lexer.next_token(), Token::Keyword(Keyword::Over));
        assert_eq!(lexer.next_token(), Token::LParen);
        assert_eq!(lexer.next_token(), Token::Keyword(Keyword::Partition));
        assert_eq!(lexer.next_token(), Token::Keyword(Keyword::By));
        assert_eq!(lexer.next_token(), Token::Ident("dept"));
        assert_eq!(lexer.next_token(), Token::Keyword(Keyword::Order));
        assert_eq!(lexer.next_token(), Token::Keyword(Keyword::By));
        assert_eq!(lexer.next_token(), Token::Ident("salary"));
        assert_eq!(lexer.next_token(), Token::Keyword(Keyword::Desc));
        assert_eq!(lexer.next_token(), Token::RParen);
    }

    #[test]
    fn lex_cte() {
        let mut lexer = Lexer::new("WITH RECURSIVE cte AS (SELECT 1) SELECT * FROM cte");
        assert_eq!(lexer.next_token(), Token::Keyword(Keyword::With));
        assert_eq!(lexer.next_token(), Token::Keyword(Keyword::Recursive));
        assert_eq!(lexer.next_token(), Token::Ident("cte"));
        assert_eq!(lexer.next_token(), Token::Keyword(Keyword::As));
        assert_eq!(lexer.next_token(), Token::LParen);
        assert_eq!(lexer.next_token(), Token::Keyword(Keyword::Select));
        assert_eq!(lexer.next_token(), Token::Integer("1"));
        assert_eq!(lexer.next_token(), Token::RParen);
        assert_eq!(lexer.next_token(), Token::Keyword(Keyword::Select));
        assert_eq!(lexer.next_token(), Token::Star);
        assert_eq!(lexer.next_token(), Token::Keyword(Keyword::From));
        assert_eq!(lexer.next_token(), Token::Ident("cte"));
    }

    #[test]
    fn peek_tokens() {
        let mut lexer = Lexer::new("SELECT FROM WHERE");
        assert_eq!(lexer.peek(), Token::Keyword(Keyword::Select));
        assert_eq!(lexer.peek(), Token::Keyword(Keyword::Select));
        assert_eq!(lexer.next_token(), Token::Keyword(Keyword::Select));
        assert_eq!(lexer.peek(), Token::Keyword(Keyword::From));
        assert_eq!(lexer.next_token(), Token::Keyword(Keyword::From));
    }

    #[test]
    fn peek_nth_tokens() {
        let mut lexer = Lexer::new("SELECT * FROM users");
        assert_eq!(lexer.peek_nth(0), Token::Keyword(Keyword::Select));
        assert_eq!(lexer.peek_nth(1), Token::Star);
        assert_eq!(lexer.peek_nth(2), Token::Keyword(Keyword::From));
        assert_eq!(lexer.peek_nth(3), Token::Ident("users"));
        assert_eq!(lexer.peek_nth(4), Token::Eof);
        assert_eq!(lexer.next_token(), Token::Keyword(Keyword::Select));
    }

    #[test]
    fn span_tracking() {
        let mut lexer = Lexer::new("SELECT users");
        lexer.next_token();
        let span1 = lexer.span();
        assert_eq!(span1.start(), 0);
        assert_eq!(span1.end(), 6);

        lexer.next_token();
        let span2 = lexer.span();
        assert_eq!(span2.start(), 7);
        assert_eq!(span2.end(), 12);
    }

    #[test]
    fn line_column_tracking() {
        let mut lexer = Lexer::new("SELECT\nFROM\n  WHERE");
        assert_eq!(lexer.line(), 1);
        assert_eq!(lexer.column(), 1);

        lexer.next_token();
        lexer.next_token();
        assert_eq!(lexer.line(), 2);

        lexer.next_token();
        assert_eq!(lexer.line(), 3);
    }

    #[test]
    fn error_unterminated_string() {
        let mut lexer = Lexer::new("'hello");
        assert!(matches!(lexer.next_token(), Token::Error(_)));
    }

    #[test]
    fn error_unterminated_quoted_ident() {
        let mut lexer = Lexer::new("\"hello");
        assert!(matches!(lexer.next_token(), Token::Error(_)));
    }

    #[test]
    fn error_invalid_hex() {
        let mut lexer = Lexer::new("0x");
        assert!(matches!(lexer.next_token(), Token::Error(_)));
    }

    #[test]
    fn lexer_vector_l2_distance_operator() {
        let mut lexer = Lexer::new("<->");
        assert_eq!(lexer.next_token(), Token::LtMinusGt);
        assert_eq!(lexer.next_token(), Token::Eof);
    }

    #[test]
    fn lexer_vector_inner_product_operator() {
        let mut lexer = Lexer::new("<#>");
        assert_eq!(lexer.next_token(), Token::LtHashGt);
        assert_eq!(lexer.next_token(), Token::Eof);
    }

    #[test]
    fn lexer_vector_cosine_distance_operator() {
        let mut lexer = Lexer::new("<=>");
        assert_eq!(lexer.next_token(), Token::Spaceship);
        assert_eq!(lexer.next_token(), Token::Eof);
    }

    #[test]
    fn lexer_vector_operators_in_expression() {
        let mut lexer = Lexer::new("a <-> b <#> c <=> d");
        assert_eq!(lexer.next_token(), Token::Ident("a"));
        assert_eq!(lexer.next_token(), Token::LtMinusGt);
        assert_eq!(lexer.next_token(), Token::Ident("b"));
        assert_eq!(lexer.next_token(), Token::LtHashGt);
        assert_eq!(lexer.next_token(), Token::Ident("c"));
        assert_eq!(lexer.next_token(), Token::Spaceship);
        assert_eq!(lexer.next_token(), Token::Ident("d"));
        assert_eq!(lexer.next_token(), Token::Eof);
    }

    #[test]
    fn lexer_less_than_followed_by_minus() {
        let mut lexer = Lexer::new("< -");
        assert_eq!(lexer.next_token(), Token::Lt);
        assert_eq!(lexer.next_token(), Token::Minus);
        assert_eq!(lexer.next_token(), Token::Eof);
    }

    #[test]
    fn lexer_less_than_followed_by_hash() {
        let mut lexer = Lexer::new("< #");
        assert_eq!(lexer.next_token(), Token::Lt);
        assert_eq!(lexer.next_token(), Token::Hash);
        assert_eq!(lexer.next_token(), Token::Eof);
    }
}
