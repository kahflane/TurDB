//! # SQL Abstract Syntax Tree
//!
//! This module defines the AST (Abstract Syntax Tree) types produced by the SQL
//! parser. All AST nodes are arena-allocated using bumpalo, with string slices
//! borrowing directly from the original input for zero-copy parsing.
//!
//! ## Design Philosophy
//!
//! The AST is designed for:
//!
//! 1. **Arena allocation**: All nodes allocated in a single arena for fast deallocation
//! 2. **Zero-copy strings**: Identifiers and literals borrow from input SQL
//! 3. **Type safety**: Strongly typed enums prevent invalid combinations
//! 4. **Visitor pattern ready**: Flat enum variants enable easy pattern matching
//!
//! ## Statement Types
//!
//! The top-level `Statement` enum covers all supported SQL statements:
//!
//! - **DML**: SELECT, INSERT, UPDATE, DELETE
//! - **DDL**: CREATE/ALTER/DROP TABLE, INDEX, SCHEMA
//! - **Transaction**: BEGIN, COMMIT, ROLLBACK, SAVEPOINT
//! - **Utility**: EXPLAIN, ANALYZE, SHOW
//!
//! ## Expression Types
//!
//! The `Expr` enum represents all expression types:
//!
//! - **Literals**: Numbers, strings, booleans, NULL
//! - **References**: Column references, qualified names
//! - **Operations**: Binary, unary, comparison operators
//! - **Functions**: Built-in and user-defined functions
//! - **Subqueries**: Scalar, IN, EXISTS subqueries
//! - **Complex**: CASE, CAST, array subscript, JSON path
//!
//! ## Memory Layout
//!
//! Arena-allocated types use `&'a T` for child nodes:
//!
//! ```text
//! Statement<'a>
//!     └── SelectStmt<'a>
//!             ├── columns: &'a [SelectColumn<'a>]
//!             ├── from: Option<&'a FromClause<'a>>
//!             ├── where_clause: Option<&'a Expr<'a>>
//!             └── ...
//! ```
//!
//! ## Usage Example
//!
//! ```ignore
//! use turdb::sql::ast::{Statement, Expr, Literal};
//!
//! match stmt {
//!     Statement::Select(select) => {
//!         for col in select.columns {
//!             println!("Column: {:?}", col);
//!         }
//!     }
//!     _ => {}
//! }
//! ```
//!
//! ## Operator Precedence
//!
//! Binary operators have associated precedence levels used by the parser:
//!
//! | Precedence | Operators |
//! |------------|-----------|
//! | 1 | OR |
//! | 2 | AND |
//! | 3 | NOT (prefix) |
//! | 4 | =, <>, <, >, <=, >=, IS, LIKE, ILIKE, IN, BETWEEN |
//! | 5 | || (concat) |
//! | 6 | +, - (binary) |
//! | 7 | *, /, % |
//! | 8 | ^ (power) |
//! | 9 | - (unary), ~ (bitwise not) |
//! | 10 | :: (cast) |
//! | 11 | . (member), [] (subscript) |

#[derive(Debug, Clone, PartialEq)]
pub enum Statement<'a> {
    Select(&'a SelectStmt<'a>),
    Insert(&'a InsertStmt<'a>),
    Update(&'a UpdateStmt<'a>),
    Delete(&'a DeleteStmt<'a>),
    CreateTable(&'a CreateTableStmt<'a>),
    CreateIndex(&'a CreateIndexStmt<'a>),
    CreateSchema(&'a CreateSchemaStmt<'a>),
    AlterTable(&'a AlterTableStmt<'a>),
    Drop(&'a DropStmt<'a>),
    Truncate(&'a TruncateStmt<'a>),
    CreateView(&'a CreateViewStmt<'a>),
    CreateFunction(&'a CreateFunctionStmt<'a>),
    CreateProcedure(&'a CreateProcedureStmt<'a>),
    CreateTrigger(&'a CreateTriggerStmt<'a>),
    CreateType(&'a CreateTypeStmt<'a>),
    Call(&'a CallStmt<'a>),
    Merge(&'a MergeStmt<'a>),
    Begin(&'a BeginStmt),
    Commit,
    Rollback(&'a RollbackStmt<'a>),
    Savepoint(&'a SavepointStmt<'a>),
    Release(&'a ReleaseStmt<'a>),
    Explain(&'a ExplainStmt<'a>),
    Set(&'a SetStmt<'a>),
    Show(&'a ShowStmt<'a>),
    Reset(&'a ResetStmt<'a>),
    Grant(&'a GrantStmt<'a>),
    Revoke(&'a RevokeStmt<'a>),
    Pragma(&'a PragmaStmt<'a>),
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SelectStmt<'a> {
    pub with: Option<&'a WithClause<'a>>,
    pub distinct: Distinct,
    pub columns: &'a [SelectColumn<'a>],
    pub from: Option<&'a FromClause<'a>>,
    pub where_clause: Option<&'a Expr<'a>>,
    pub group_by: &'a [&'a Expr<'a>],
    pub having: Option<&'a Expr<'a>>,
    pub order_by: &'a [OrderByItem<'a>],
    pub limit: Option<&'a Expr<'a>>,
    pub offset: Option<&'a Expr<'a>>,
    pub set_op: Option<&'a SetOperation<'a>>,
    pub for_clause: Option<&'a ForClause<'a>>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ForClause<'a> {
    pub lock_mode: LockMode,
    pub tables: Option<&'a [&'a str]>,
    pub wait_policy: WaitPolicy,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LockMode {
    Update,
    NoKeyUpdate,
    Share,
    KeyShare,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WaitPolicy {
    Wait,
    Nowait,
    SkipLocked,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Distinct {
    All,
    Distinct,
    DistinctOn,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SelectColumn<'a> {
    AllColumns,
    TableAllColumns(&'a str),
    Expr {
        expr: &'a Expr<'a>,
        alias: Option<&'a str>,
    },
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WithClause<'a> {
    pub recursive: bool,
    pub ctes: &'a [Cte<'a>],
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Cte<'a> {
    pub name: &'a str,
    pub columns: Option<&'a [&'a str]>,
    pub query: &'a SelectStmt<'a>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SetOperation<'a> {
    pub op: SetOperator,
    pub all: bool,
    pub right: &'a SelectStmt<'a>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SetOperator {
    Union,
    Intersect,
    Except,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum FromClause<'a> {
    Table(TableRef<'a>),
    Join(&'a JoinClause<'a>),
    Subquery {
        query: &'a SelectStmt<'a>,
        alias: &'a str,
    },
    Lateral {
        subquery: &'a SelectStmt<'a>,
        alias: &'a str,
    },
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TableRef<'a> {
    pub schema: Option<&'a str>,
    pub name: &'a str,
    pub alias: Option<&'a str>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct JoinClause<'a> {
    pub left: &'a FromClause<'a>,
    pub join_type: JoinType,
    pub right: &'a FromClause<'a>,
    pub condition: JoinCondition<'a>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JoinType {
    Inner,
    Left,
    Right,
    Full,
    Cross,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum JoinCondition<'a> {
    On(&'a Expr<'a>),
    Using(&'a [&'a str]),
    Natural,
    None,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct OrderByItem<'a> {
    pub expr: &'a Expr<'a>,
    pub direction: OrderDirection,
    pub nulls: NullsOrder,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OrderDirection {
    Asc,
    Desc,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NullsOrder {
    First,
    Last,
    Default,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct InsertStmt<'a> {
    pub table: TableRef<'a>,
    pub columns: Option<&'a [&'a str]>,
    pub source: InsertSource<'a>,
    pub on_conflict: Option<&'a OnConflict<'a>>,
    pub returning: Option<&'a [SelectColumn<'a>]>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum InsertSource<'a> {
    Values(&'a [&'a [&'a Expr<'a>]]),
    Select(&'a SelectStmt<'a>),
    Default,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct OnConflict<'a> {
    pub target: OnConflictTarget<'a>,
    pub action: OnConflictAction<'a>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum OnConflictTarget<'a> {
    Columns(&'a [&'a str]),
    Constraint(&'a str),
    None,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum OnConflictAction<'a> {
    DoNothing,
    DoUpdate(&'a [Assignment<'a>]),
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Assignment<'a> {
    pub column: ColumnRef<'a>,
    pub value: &'a Expr<'a>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct UpdateStmt<'a> {
    pub table: TableRef<'a>,
    pub assignments: &'a [Assignment<'a>],
    pub from: Option<&'a FromClause<'a>>,
    pub where_clause: Option<&'a Expr<'a>>,
    pub returning: Option<&'a [SelectColumn<'a>]>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DeleteStmt<'a> {
    pub table: TableRef<'a>,
    pub using: Option<&'a FromClause<'a>>,
    pub where_clause: Option<&'a Expr<'a>>,
    pub returning: Option<&'a [SelectColumn<'a>]>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CreateTableStmt<'a> {
    pub if_not_exists: bool,
    pub schema: Option<&'a str>,
    pub name: &'a str,
    pub columns: &'a [ColumnDef<'a>],
    pub constraints: &'a [TableConstraint<'a>],
    pub temporary: bool,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ColumnDef<'a> {
    pub name: &'a str,
    pub data_type: DataType<'a>,
    pub constraints: &'a [ColumnConstraint<'a>],
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DataType<'a> {
    Integer,
    BigInt,
    SmallInt,
    TinyInt,
    Real,
    DoublePrecision,
    Decimal(Option<u32>, Option<u32>),
    Numeric(Option<u32>, Option<u32>),
    Varchar(Option<u32>),
    Char(Option<u32>),
    Text,
    Blob,
    Boolean,
    Date,
    Time,
    Timestamp,
    TimestampTz,
    Interval,
    Uuid,
    Json,
    Jsonb,
    Vector(Option<u32>),
    Array(&'a DataType<'a>),
    Point,
    Box,
    Circle,
    MacAddr,
    Inet,
    Int4Range,
    Int8Range,
    DateRange,
    TsRange,
    Custom(&'a str),
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ColumnConstraint<'a> {
    NotNull,
    Null,
    Unique,
    PrimaryKey,
    Default(&'a Expr<'a>),
    Check(&'a Expr<'a>),
    References {
        table: &'a str,
        column: Option<&'a str>,
        on_delete: Option<ReferentialAction>,
        on_update: Option<ReferentialAction>,
    },
    Generated {
        expr: &'a Expr<'a>,
        stored: bool,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReferentialAction {
    Cascade,
    Restrict,
    NoAction,
    SetNull,
    SetDefault,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TableConstraint<'a> {
    PrimaryKey {
        name: Option<&'a str>,
        columns: &'a [&'a str],
    },
    Unique {
        name: Option<&'a str>,
        columns: &'a [&'a str],
    },
    ForeignKey {
        name: Option<&'a str>,
        columns: &'a [&'a str],
        ref_table: &'a str,
        ref_columns: &'a [&'a str],
        on_delete: Option<ReferentialAction>,
        on_update: Option<ReferentialAction>,
    },
    Check {
        name: Option<&'a str>,
        expr: &'a Expr<'a>,
    },
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CreateIndexStmt<'a> {
    pub if_not_exists: bool,
    pub unique: bool,
    pub name: &'a str,
    pub table: TableRef<'a>,
    pub columns: &'a [IndexColumn<'a>],
    pub index_type: Option<IndexType>,
    pub where_clause: Option<&'a Expr<'a>>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct IndexColumn<'a> {
    pub expr: &'a Expr<'a>,
    pub direction: Option<OrderDirection>,
    pub nulls: Option<NullsOrder>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IndexType {
    BTree,
    Hash,
    Gin,
    Gist,
    Hnsw,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CreateSchemaStmt<'a> {
    pub if_not_exists: bool,
    pub name: &'a str,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AlterTableStmt<'a> {
    pub table: TableRef<'a>,
    pub action: AlterTableAction<'a>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum AlterTableAction<'a> {
    AddColumn(ColumnDef<'a>),
    DropColumn {
        name: &'a str,
        if_exists: bool,
        cascade: bool,
    },
    AlterColumn {
        name: &'a str,
        action: AlterColumnAction<'a>,
    },
    AddConstraint(TableConstraint<'a>),
    DropConstraint {
        name: &'a str,
        if_exists: bool,
        cascade: bool,
    },
    RenameColumn {
        old_name: &'a str,
        new_name: &'a str,
    },
    RenameTable(&'a str),
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum AlterColumnAction<'a> {
    SetDataType(DataType<'a>),
    SetDefault(&'a Expr<'a>),
    DropDefault,
    SetNotNull,
    DropNotNull,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DropStmt<'a> {
    pub object_type: ObjectType,
    pub if_exists: bool,
    pub names: &'a [ObjectName<'a>],
    pub cascade: bool,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ObjectName<'a> {
    pub schema: Option<&'a str>,
    pub name: &'a str,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ObjectType {
    Table,
    Index,
    Schema,
    View,
    Sequence,
    Function,
    Procedure,
    Trigger,
    Database,
    Type,
    Domain,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TruncateStmt<'a> {
    pub tables: &'a [TableRef<'a>],
    pub restart_identity: bool,
    pub cascade: bool,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CreateViewStmt<'a> {
    pub or_replace: bool,
    pub materialized: bool,
    pub schema: Option<&'a str>,
    pub name: &'a str,
    pub columns: Option<&'a [&'a str]>,
    pub query: &'a SelectStmt<'a>,
    pub with_check_option: bool,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CreateFunctionStmt<'a> {
    pub or_replace: bool,
    pub schema: Option<&'a str>,
    pub name: &'a str,
    pub params: &'a [FunctionParam<'a>],
    pub return_type: DataType<'a>,
    pub body: &'a str,
    pub language: &'a str,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FunctionParam<'a> {
    pub name: &'a str,
    pub data_type: DataType<'a>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CreateProcedureStmt<'a> {
    pub or_replace: bool,
    pub schema: Option<&'a str>,
    pub name: &'a str,
    pub params: &'a [FunctionParam<'a>],
    pub body: &'a str,
    pub language: &'a str,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CreateTriggerStmt<'a> {
    pub or_replace: bool,
    pub name: &'a str,
    pub timing: TriggerTiming,
    pub events: &'a [TriggerEvent],
    pub table: &'a str,
    pub for_each_row: bool,
    pub function_name: &'a str,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TriggerTiming {
    Before,
    After,
    InsteadOf,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TriggerEvent {
    Insert,
    Update,
    Delete,
    Truncate,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CreateTypeStmt<'a> {
    pub schema: Option<&'a str>,
    pub name: &'a str,
    pub definition: TypeDefinition<'a>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TypeDefinition<'a> {
    Enum(&'a [&'a str]),
    Composite(&'a [TypeField<'a>]),
    Domain(DataType<'a>),
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TypeField<'a> {
    pub name: &'a str,
    pub data_type: DataType<'a>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CallStmt<'a> {
    pub schema: Option<&'a str>,
    pub name: &'a str,
    pub args: &'a [&'a Expr<'a>],
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MergeStmt<'a> {
    pub target_table: &'a str,
    pub target_alias: Option<&'a str>,
    pub source: MergeSource<'a>,
    pub on_condition: &'a Expr<'a>,
    pub clauses: &'a [MergeClause<'a>],
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum MergeSource<'a> {
    Table {
        name: &'a str,
        alias: Option<&'a str>,
    },
    Subquery {
        query: &'a SelectStmt<'a>,
        alias: &'a str,
    },
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum MergeClause<'a> {
    MatchedUpdate(&'a [Assignment<'a>]),
    MatchedDelete,
    NotMatchedInsert {
        columns: Option<&'a [&'a str]>,
        values: &'a [&'a Expr<'a>],
    },
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BeginStmt {
    pub isolation_level: Option<IsolationLevel>,
    pub read_only: Option<bool>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IsolationLevel {
    ReadUncommitted,
    ReadCommitted,
    RepeatableRead,
    Serializable,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RollbackStmt<'a> {
    pub savepoint: Option<&'a str>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SavepointStmt<'a> {
    pub name: &'a str,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ReleaseStmt<'a> {
    pub name: &'a str,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ExplainStmt<'a> {
    pub analyze: bool,
    pub verbose: bool,
    pub format: ExplainFormat,
    pub statement: &'a Statement<'a>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExplainFormat {
    Text,
    Json,
    Xml,
    Yaml,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SetStmt<'a> {
    pub scope: SetScope,
    pub name: &'a str,
    pub value: &'a [&'a Expr<'a>],
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SetScope {
    Session,
    Local,
    Global,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PragmaStmt<'a> {
    pub name: &'a str,
    pub value: Option<&'a str>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ShowStmt<'a> {
    pub name: Option<&'a str>,
    pub all: bool,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ResetStmt<'a> {
    pub name: Option<&'a str>,
    pub all: bool,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GrantStmt<'a> {
    pub privileges: &'a [Privilege],
    pub object_type: Option<ObjectType>,
    pub object_name: Option<TableRef<'a>>,
    pub grantees: &'a [&'a str],
    pub with_grant_option: bool,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RevokeStmt<'a> {
    pub privileges: &'a [Privilege],
    pub object_type: Option<ObjectType>,
    pub object_name: Option<TableRef<'a>>,
    pub grantees: &'a [&'a str],
    pub cascade: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Privilege {
    Select,
    Insert,
    Update,
    Delete,
    Truncate,
    References,
    Trigger,
    Create,
    Connect,
    Temporary,
    Execute,
    Usage,
    All,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Expr<'a> {
    Literal(Literal<'a>),
    Column(ColumnRef<'a>),
    Parameter(ParameterRef<'a>),
    BinaryOp {
        left: &'a Expr<'a>,
        op: BinaryOperator,
        right: &'a Expr<'a>,
    },
    UnaryOp {
        op: UnaryOperator,
        expr: &'a Expr<'a>,
    },
    Between {
        expr: &'a Expr<'a>,
        negated: bool,
        low: &'a Expr<'a>,
        high: &'a Expr<'a>,
    },
    Like {
        expr: &'a Expr<'a>,
        negated: bool,
        pattern: &'a Expr<'a>,
        escape: Option<&'a Expr<'a>>,
        case_insensitive: bool,
    },
    InList {
        expr: &'a Expr<'a>,
        negated: bool,
        list: &'a [&'a Expr<'a>],
    },
    InSubquery {
        expr: &'a Expr<'a>,
        negated: bool,
        subquery: &'a SelectStmt<'a>,
    },
    IsNull {
        expr: &'a Expr<'a>,
        negated: bool,
    },
    IsDistinctFrom {
        left: &'a Expr<'a>,
        right: &'a Expr<'a>,
        negated: bool,
    },
    Function(FunctionCall<'a>),
    Case {
        operand: Option<&'a Expr<'a>>,
        conditions: &'a [WhenClause<'a>],
        else_result: Option<&'a Expr<'a>>,
    },
    Cast {
        expr: &'a Expr<'a>,
        data_type: DataType<'a>,
    },
    Subquery(&'a SelectStmt<'a>),
    Exists {
        subquery: &'a SelectStmt<'a>,
        negated: bool,
    },
    ArraySubscript {
        array: &'a Expr<'a>,
        index: &'a Expr<'a>,
    },
    ArraySlice {
        array: &'a Expr<'a>,
        lower: Option<&'a Expr<'a>>,
        upper: Option<&'a Expr<'a>>,
    },
    Row(&'a [&'a Expr<'a>]),
    Array(&'a [&'a Expr<'a>]),
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Literal<'a> {
    Null,
    Boolean(bool),
    Integer(&'a str),
    Float(&'a str),
    String(&'a str),
    HexNumber(&'a str),
    BinaryNumber(&'a str),
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ColumnRef<'a> {
    pub schema: Option<&'a str>,
    pub table: Option<&'a str>,
    pub column: &'a str,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ParameterRef<'a> {
    Positional(u32),
    Named(&'a str),
    Anonymous,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinaryOperator {
    Plus,
    Minus,
    Multiply,
    Divide,
    Modulo,
    Power,
    Concat,
    Eq,
    NotEq,
    Lt,
    LtEq,
    Gt,
    GtEq,
    And,
    Or,
    BitwiseAnd,
    BitwiseOr,
    BitwiseXor,
    LeftShift,
    RightShift,
    JsonExtract,
    JsonExtractText,
    JsonPathExtract,
    JsonPathExtractText,
    JsonContains,
    JsonContainedBy,
    ArrayContains,
    ArrayContainedBy,
    ArrayOverlaps,
    VectorL2Distance,
    VectorInnerProduct,
    VectorCosineDistance,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnaryOperator {
    Not,
    Minus,
    Plus,
    BitwiseNot,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FunctionCall<'a> {
    pub name: FunctionName<'a>,
    pub args: FunctionArgs<'a>,
    pub distinct: bool,
    pub filter: Option<&'a Expr<'a>>,
    pub over: Option<WindowSpec<'a>>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FunctionName<'a> {
    pub schema: Option<&'a str>,
    pub name: &'a str,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum FunctionArgs<'a> {
    None,
    Star,
    Args(&'a [FunctionArg<'a>]),
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FunctionArg<'a> {
    pub name: Option<&'a str>,
    pub value: &'a Expr<'a>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WindowSpec<'a> {
    pub partition_by: &'a [&'a Expr<'a>],
    pub order_by: &'a [OrderByItem<'a>],
    pub frame: Option<WindowFrame>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WindowFrame {
    pub mode: WindowFrameMode,
    pub start: WindowFrameBound,
    pub end: Option<WindowFrameBound>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WindowFrameMode {
    Rows,
    Range,
    Groups,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WindowFrameBound {
    CurrentRow,
    UnboundedPreceding,
    UnboundedFollowing,
    Preceding(u64),
    Following(u64),
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WhenClause<'a> {
    pub condition: &'a Expr<'a>,
    pub result: &'a Expr<'a>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use bumpalo::Bump;

    #[test]
    fn statement_select_variant() {
        let arena = Bump::new();
        let select = arena.alloc(SelectStmt {
            with: None,
            distinct: Distinct::All,
            columns: &[],
            from: None,
            where_clause: None,
            group_by: &[],
            having: None,
            order_by: &[],
            limit: None,
            offset: None,
            set_op: None,
            for_clause: None,
        });
        let stmt = Statement::Select(select);
        assert!(matches!(stmt, Statement::Select(_)));
    }

    #[test]
    fn expr_literal_variants() {
        let expr_null = Expr::Literal(Literal::Null);
        let expr_bool = Expr::Literal(Literal::Boolean(true));
        let expr_int = Expr::Literal(Literal::Integer("42"));
        let expr_float = Expr::Literal(Literal::Float("3.14"));
        let expr_str = Expr::Literal(Literal::String("hello"));

        assert!(matches!(expr_null, Expr::Literal(Literal::Null)));
        assert!(matches!(expr_bool, Expr::Literal(Literal::Boolean(true))));
        assert!(matches!(expr_int, Expr::Literal(Literal::Integer("42"))));
        assert!(matches!(expr_float, Expr::Literal(Literal::Float("3.14"))));
        assert!(matches!(expr_str, Expr::Literal(Literal::String("hello"))));
    }

    #[test]
    fn expr_column_ref() {
        let col = Expr::Column(ColumnRef {
            schema: Some("public"),
            table: Some("users"),
            column: "id",
        });
        if let Expr::Column(c) = col {
            assert_eq!(c.schema, Some("public"));
            assert_eq!(c.table, Some("users"));
            assert_eq!(c.column, "id");
        } else {
            panic!("Expected Column");
        }
    }

    #[test]
    fn expr_binary_op() {
        let arena = Bump::new();
        let left = arena.alloc(Expr::Literal(Literal::Integer("1")));
        let right = arena.alloc(Expr::Literal(Literal::Integer("2")));
        let expr = Expr::BinaryOp {
            left,
            op: BinaryOperator::Plus,
            right,
        };
        assert!(matches!(
            expr,
            Expr::BinaryOp {
                op: BinaryOperator::Plus,
                ..
            }
        ));
    }

    #[test]
    fn data_type_variants() {
        let int = DataType::Integer;
        let varchar = DataType::Varchar(Some(255));
        let decimal = DataType::Decimal(Some(10), Some(2));
        let vector = DataType::Vector(Some(128));

        assert!(matches!(int, DataType::Integer));
        assert!(matches!(varchar, DataType::Varchar(Some(255))));
        assert!(matches!(decimal, DataType::Decimal(Some(10), Some(2))));
        assert!(matches!(vector, DataType::Vector(Some(128))));
    }

    #[test]
    fn table_ref_with_alias() {
        let table = TableRef {
            schema: Some("public"),
            name: "users",
            alias: Some("u"),
        };
        assert_eq!(table.schema, Some("public"));
        assert_eq!(table.name, "users");
        assert_eq!(table.alias, Some("u"));
    }

    #[test]
    fn join_types() {
        assert_ne!(JoinType::Inner, JoinType::Left);
        assert_ne!(JoinType::Left, JoinType::Right);
        assert_ne!(JoinType::Right, JoinType::Full);
        assert_ne!(JoinType::Full, JoinType::Cross);
    }

    #[test]
    fn binary_operators_equality() {
        assert_eq!(BinaryOperator::Plus, BinaryOperator::Plus);
        assert_ne!(BinaryOperator::Plus, BinaryOperator::Minus);
    }

    #[test]
    fn column_constraint_variants() {
        let not_null = ColumnConstraint::NotNull;
        let unique = ColumnConstraint::Unique;
        let pk = ColumnConstraint::PrimaryKey;

        assert!(matches!(not_null, ColumnConstraint::NotNull));
        assert!(matches!(unique, ColumnConstraint::Unique));
        assert!(matches!(pk, ColumnConstraint::PrimaryKey));
    }

    #[test]
    fn expr_size_monitor() {
        let expr_size = std::mem::size_of::<Expr>();
        let statement_size = std::mem::size_of::<Statement>();

        assert!(
            expr_size <= 160,
            "Expr size {} bytes exceeds 160-byte target (may indicate large variants growing)",
            expr_size
        );
        assert!(
            statement_size <= 24,
            "Statement size {} bytes exceeds 24-byte target",
            statement_size
        );
    }

    #[test]
    fn expr_size_report() {
        let expr_size = std::mem::size_of::<Expr>();
        let statement_size = std::mem::size_of::<Statement>();
        let literal_size = std::mem::size_of::<Literal>();
        let column_ref_size = std::mem::size_of::<ColumnRef>();
        let binary_op_size = std::mem::size_of::<BinaryOperator>();
        let function_call_size = std::mem::size_of::<FunctionCall>();

        eprintln!("AST type sizes:");
        eprintln!("  Expr: {} bytes", expr_size);
        eprintln!("  Statement: {} bytes", statement_size);
        eprintln!("  Literal: {} bytes", literal_size);
        eprintln!("  ColumnRef: {} bytes", column_ref_size);
        eprintln!("  BinaryOperator: {} bytes", binary_op_size);
        eprintln!("  FunctionCall: {} bytes", function_call_size);
    }
}
