//! # SQL Parser - Arena-Allocated AST Builder
//!
//! This module implements a recursive descent SQL parser that produces an
//! arena-allocated AST. The parser supports a PostgreSQL-compatible SQL dialect
//! including SELECT, INSERT, UPDATE, DELETE, DDL statements, subqueries, CTEs,
//! window functions, and more.
//!
//! ## Design Goals
//!
//! 1. **Arena allocation**: All AST nodes allocated in bumpalo arena
//! 2. **Zero-copy identifiers**: String slices borrow from input
//! 3. **Rich error reporting**: Line/column tracking with error recovery
//! 4. **Pratt parsing**: Operator precedence via Pratt parser for expressions
//!
//! ## Parser Architecture
//!
//! The parser uses recursive descent for statement-level parsing and Pratt
//! parsing for expression precedence. Each parse function consumes tokens
//! from the lexer and returns an AST node allocated in the arena.
//!
//! ```text
//! Input SQL → Lexer → Parser → AST (arena-allocated)
//! ```
//!
//! ## Supported Statements
//!
//! - **DML**: SELECT, INSERT, UPDATE, DELETE
//! - **DDL**: CREATE/ALTER/DROP TABLE, INDEX, SCHEMA
//! - **Queries**: Subqueries, CTEs, window functions, set operations
//! - **Transactions**: BEGIN, COMMIT, ROLLBACK
//! - **Analysis**: EXPLAIN, ANALYZE
//!
//! ## Expression Precedence (Pratt Parser)
//!
//! The expression parser uses binding power to handle precedence:
//!
//! | Precedence | Operators |
//! |------------|-----------|
//! | 1 (lowest) | OR |
//! | 2 | AND |
//! | 3 | NOT |
//! | 4 | =, <>, <, >, <=, >=, IS, LIKE, IN, BETWEEN |
//! | 5 | + , - (binary) |
//! | 6 | *, /, % |
//! | 7 | ^ |
//! | 8 | - (unary), NOT |
//! | 9 | :: (cast) |
//! | 10 | . (member access) |
//! | 11 (highest) | () function call, [] subscript |
//!
//! ## Error Handling
//!
//! The parser accumulates errors and attempts recovery to report multiple
//! issues in a single parse. Use `Parser::errors()` to retrieve all errors
//! after parsing completes.
//!
//! ## Usage Example
//!
//! ```ignore
//! use turdb::sql::{Parser, Statement};
//! use bumpalo::Bump;
//!
//! let arena = Bump::new();
//! let mut parser = Parser::new("SELECT id, name FROM users WHERE active = true", &arena);
//!
//! match parser.parse_statement() {
//!     Ok(stmt) => println!("{:?}", stmt),
//!     Err(e) => eprintln!("Parse error: {}", e),
//! }
//! ```
//!
//! ## Memory Management
//!
//! All AST nodes are allocated in the provided arena. The arena should be
//! cleared after query execution completes. This eliminates per-node heap
//! allocation and enables fast bulk deallocation.
//!
//! ## Performance
//!
//! - Single-pass parsing with minimal lookahead
//! - No heap allocation for identifiers (borrowed slices)
//! - Arena allocation for AST nodes
//! - O(1) operator precedence lookup

use super::ast::*;
use super::lexer::Lexer;
use super::token::{Keyword, Parameter, Span, Token};
use bumpalo::Bump;
use eyre::{bail, Result};

pub struct Parser<'a> {
    lexer: Lexer<'a>,
    arena: &'a Bump,
    current: Token<'a>,
    errors: Vec<ParseError<'a>>,
}

#[derive(Debug, Clone)]
pub struct ParseError<'a> {
    pub message: &'a str,
    pub span: Span,
    pub line: u32,
    pub column: u32,
}

#[derive(Debug)]
pub struct ParseResult<'a> {
    pub statements: Vec<Statement<'a>>,
    pub errors: Vec<ParseError<'a>>,
}

impl<'a> Parser<'a> {
    pub fn new(input: &'a str, arena: &'a Bump) -> Self {
        let mut lexer = Lexer::new(input);
        let current = lexer.next_token();
        Self {
            lexer,
            arena,
            current,
            errors: Vec::new(),
        }
    }

    pub fn is_at_end(&self) -> bool {
        matches!(self.current, Token::Eof)
    }

    pub fn peek(&self) -> &Token<'a> {
        &self.current
    }

    pub fn advance(&mut self) -> Token<'a> {
        let prev = std::mem::replace(&mut self.current, self.lexer.next_token());
        prev
    }

    pub fn check_keyword(&self, keyword: Keyword) -> bool {
        matches!(&self.current, Token::Keyword(k) if *k == keyword)
    }

    pub fn consume_keyword(&mut self, keyword: Keyword) -> bool {
        if self.check_keyword(keyword) {
            self.advance();
            true
        } else {
            false
        }
    }

    pub fn expect_keyword(&mut self, keyword: Keyword) -> Result<()> {
        if self.check_keyword(keyword) {
            self.advance();
            Ok(())
        } else {
            bail!(
                "expected keyword {:?}, found {:?} at line {} column {}",
                keyword,
                self.current,
                self.lexer.line(),
                self.lexer.column()
            )
        }
    }

    pub fn check_token(&self, expected: &Token<'_>) -> bool {
        std::mem::discriminant(&self.current) == std::mem::discriminant(expected)
    }

    pub fn consume_token(&mut self, expected: &Token<'_>) -> bool {
        if self.check_token(expected) {
            self.advance();
            true
        } else {
            false
        }
    }

    pub fn expect_token(&mut self, expected: &Token<'_>) -> Result<()> {
        if self.check_token(expected) {
            self.advance();
            Ok(())
        } else {
            bail!(
                "expected {:?}, found {:?} at line {} column {}",
                expected,
                self.current,
                self.lexer.line(),
                self.lexer.column()
            )
        }
    }

    pub fn errors(&self) -> &[ParseError<'a>] {
        &self.errors
    }

    #[allow(dead_code)]
    fn add_error(&mut self, message: &'a str) {
        self.errors.push(ParseError {
            message,
            span: self.lexer.span(),
            line: self.lexer.line(),
            column: self.lexer.column(),
        });
    }

    fn add_error_from_report(&mut self, err: eyre::Report) {
        let message = self.arena.alloc_str(&err.to_string());
        self.errors.push(ParseError {
            message,
            span: self.lexer.span(),
            line: self.lexer.line(),
            column: self.lexer.column(),
        });
    }

    fn synchronize(&mut self) {
        while !self.is_at_end() {
            if matches!(self.current, Token::Semicolon) {
                self.advance();
                return;
            }
            match self.current {
                Token::Keyword(Keyword::Select)
                | Token::Keyword(Keyword::Insert)
                | Token::Keyword(Keyword::Update)
                | Token::Keyword(Keyword::Delete)
                | Token::Keyword(Keyword::Create)
                | Token::Keyword(Keyword::Drop)
                | Token::Keyword(Keyword::Alter)
                | Token::Keyword(Keyword::Truncate)
                | Token::Keyword(Keyword::Begin)
                | Token::Keyword(Keyword::Commit)
                | Token::Keyword(Keyword::Rollback)
                | Token::Keyword(Keyword::Explain)
                | Token::Keyword(Keyword::Call)
                | Token::Keyword(Keyword::Merge)
                | Token::Keyword(Keyword::With) => return,
                _ => {
                    self.advance();
                }
            }
        }
    }

    #[allow(dead_code)]
    fn is_statement_start(&self) -> bool {
        matches!(
            self.current,
            Token::Keyword(Keyword::Select)
                | Token::Keyword(Keyword::Insert)
                | Token::Keyword(Keyword::Update)
                | Token::Keyword(Keyword::Delete)
                | Token::Keyword(Keyword::Create)
                | Token::Keyword(Keyword::Drop)
                | Token::Keyword(Keyword::Alter)
                | Token::Keyword(Keyword::Truncate)
                | Token::Keyword(Keyword::Begin)
                | Token::Keyword(Keyword::Commit)
                | Token::Keyword(Keyword::Rollback)
                | Token::Keyword(Keyword::Explain)
                | Token::Keyword(Keyword::Call)
                | Token::Keyword(Keyword::Merge)
                | Token::Keyword(Keyword::With)
        )
    }

    pub fn parse_statements(&mut self) -> ParseResult<'a> {
        let mut statements = Vec::new();

        while !self.is_at_end() {
            while self.consume_token(&Token::Semicolon) {}
            if self.is_at_end() {
                break;
            }

            match self.parse_statement() {
                Ok(stmt) => {
                    statements.push(stmt);
                }
                Err(err) => {
                    self.add_error_from_report(err);
                    self.synchronize();
                }
            }
        }

        ParseResult {
            statements,
            errors: std::mem::take(&mut self.errors),
        }
    }

    pub fn parse_statement(&mut self) -> Result<Statement<'a>> {
        match self.peek() {
            Token::Keyword(Keyword::Select) | Token::Keyword(Keyword::With) => {
                let select = self.parse_select()?;
                Ok(Statement::Select(self.arena.alloc(select)))
            }
            Token::Keyword(Keyword::Insert) => {
                let insert = self.parse_insert()?;
                Ok(Statement::Insert(self.arena.alloc(insert)))
            }
            Token::Keyword(Keyword::Update) => {
                let update = self.parse_update()?;
                Ok(Statement::Update(self.arena.alloc(update)))
            }
            Token::Keyword(Keyword::Delete) => {
                let delete = self.parse_delete()?;
                Ok(Statement::Delete(self.arena.alloc(delete)))
            }
            Token::Keyword(Keyword::Create) => self.parse_create(),
            Token::Keyword(Keyword::Drop) => {
                let drop = self.parse_drop()?;
                Ok(Statement::Drop(self.arena.alloc(drop)))
            }
            Token::Keyword(Keyword::Truncate) => {
                let truncate = self.parse_truncate()?;
                Ok(Statement::Truncate(self.arena.alloc(truncate)))
            }
            Token::Keyword(Keyword::Alter) => self.parse_alter(),
            Token::Keyword(Keyword::Begin) => {
                let begin = self.parse_begin()?;
                Ok(Statement::Begin(self.arena.alloc(begin)))
            }
            Token::Keyword(Keyword::Commit) => {
                self.advance();
                Ok(Statement::Commit)
            }
            Token::Keyword(Keyword::Rollback) => {
                let rollback = self.parse_rollback()?;
                Ok(Statement::Rollback(self.arena.alloc(rollback)))
            }
            Token::Keyword(Keyword::Savepoint) => {
                let savepoint = self.parse_savepoint()?;
                Ok(Statement::Savepoint(self.arena.alloc(savepoint)))
            }
            Token::Keyword(Keyword::Release) => {
                let release = self.parse_release()?;
                Ok(Statement::Release(self.arena.alloc(release)))
            }
            Token::Keyword(Keyword::Explain) => {
                let explain = self.parse_explain()?;
                Ok(Statement::Explain(self.arena.alloc(explain)))
            }
            Token::Keyword(Keyword::Call) => {
                let call = self.parse_call()?;
                Ok(Statement::Call(self.arena.alloc(call)))
            }
            Token::Keyword(Keyword::Merge) => {
                let merge = self.parse_merge()?;
                Ok(Statement::Merge(self.arena.alloc(merge)))
            }
            Token::Keyword(Keyword::Set) => {
                let set = self.parse_set()?;
                Ok(Statement::Set(self.arena.alloc(set)))
            }
            Token::Keyword(Keyword::Show) => {
                let show = self.parse_show()?;
                Ok(Statement::Show(self.arena.alloc(show)))
            }
            Token::Keyword(Keyword::Reset) => {
                let reset = self.parse_reset()?;
                Ok(Statement::Reset(self.arena.alloc(reset)))
            }
            Token::Keyword(Keyword::Grant) => {
                let grant = self.parse_grant()?;
                Ok(Statement::Grant(self.arena.alloc(grant)))
            }
            Token::Keyword(Keyword::Revoke) => {
                let revoke = self.parse_revoke_stmt()?;
                Ok(Statement::Revoke(self.arena.alloc(revoke)))
            }
            Token::Keyword(Keyword::Pragma) => {
                let pragma = self.parse_pragma()?;
                Ok(Statement::Pragma(self.arena.alloc(pragma)))
            }
            _ => bail!("unexpected token {:?} at start of statement", self.current),
        }
    }

    fn parse_select(&mut self) -> Result<SelectStmt<'a>> {
        let with = if self.check_keyword(Keyword::With) {
            Some(self.parse_with_clause()?)
        } else {
            None
        };

        self.expect_keyword(Keyword::Select)?;

        let distinct = if self.consume_keyword(Keyword::Distinct) {
            Distinct::Distinct
        } else {
            self.consume_keyword(Keyword::All);
            Distinct::All
        };

        let columns = self.parse_select_columns()?;

        let from = if self.consume_keyword(Keyword::From) {
            Some(self.parse_from_clause()?)
        } else {
            None
        };

        let where_clause: Option<&Expr<'a>> = if self.consume_keyword(Keyword::Where) {
            Some(self.arena.alloc(self.parse_expr(0)?))
        } else {
            None
        };

        let group_by = if self.consume_keyword(Keyword::Group) {
            self.expect_keyword(Keyword::By)?;
            self.parse_expr_list()?
        } else {
            &[]
        };

        let having: Option<&Expr<'a>> = if self.consume_keyword(Keyword::Having) {
            Some(self.arena.alloc(self.parse_expr(0)?))
        } else {
            None
        };

        let order_by = if self.consume_keyword(Keyword::Order) {
            self.expect_keyword(Keyword::By)?;
            self.parse_order_by_list()?
        } else {
            &[]
        };

        let mut limit: Option<&Expr<'a>> = if self.consume_keyword(Keyword::Limit) {
            Some(self.arena.alloc(self.parse_expr(0)?))
        } else {
            None
        };

        let offset: Option<&Expr<'a>> = if self.consume_keyword(Keyword::Offset) {
            let expr = self.arena.alloc(self.parse_expr(0)?);
            self.consume_keyword(Keyword::Rows);
            self.consume_keyword(Keyword::Row);
            Some(expr)
        } else {
            None
        };

        if self.consume_keyword(Keyword::Fetch) {
            self.consume_keyword(Keyword::First);
            self.consume_keyword(Keyword::Next);
            let count = self.arena.alloc(self.parse_expr(0)?);
            self.consume_keyword(Keyword::Rows);
            self.consume_keyword(Keyword::Row);
            self.expect_keyword(Keyword::Only)?;
            limit = Some(count);
        }

        let set_op = self.parse_set_operation()?;

        let for_clause = self.parse_for_clause()?;

        Ok(SelectStmt {
            with,
            distinct,
            columns,
            from,
            where_clause,
            group_by,
            having,
            order_by,
            limit,
            offset,
            set_op,
            for_clause,
        })
    }

    fn parse_for_clause(&mut self) -> Result<Option<&'a ForClause<'a>>> {
        if !self.consume_keyword(Keyword::For) {
            return Ok(None);
        }

        let lock_mode = if self.consume_keyword(Keyword::Update) {
            LockMode::Update
        } else if self.consume_keyword(Keyword::Share) {
            LockMode::Share
        } else if self.consume_keyword(Keyword::No) {
            self.expect_keyword(Keyword::Key)?;
            self.expect_keyword(Keyword::Update)?;
            LockMode::NoKeyUpdate
        } else if self.consume_keyword(Keyword::Key) {
            self.expect_keyword(Keyword::Share)?;
            LockMode::KeyShare
        } else {
            bail!(
                "expected UPDATE, SHARE, NO KEY UPDATE, or KEY SHARE after FOR at line {} column {}",
                self.lexer.line(),
                self.lexer.column()
            )
        };

        let tables = if self.consume_keyword(Keyword::Of) {
            let mut table_list = Vec::new();
            loop {
                let table_name = self.expect_ident()?;
                table_list.push(table_name);
                if !self.consume_token(&Token::Comma) {
                    break;
                }
            }
            Some(self.arena.alloc_slice_copy(&table_list) as &[&str])
        } else {
            None
        };

        let wait_policy = if self.consume_keyword(Keyword::Nowait) {
            WaitPolicy::Nowait
        } else if self.consume_keyword(Keyword::Skip) {
            self.expect_keyword(Keyword::Locked)?;
            WaitPolicy::SkipLocked
        } else {
            WaitPolicy::Wait
        };

        Ok(Some(self.arena.alloc(ForClause {
            lock_mode,
            tables,
            wait_policy,
        })))
    }

    fn parse_with_clause(&mut self) -> Result<&'a WithClause<'a>> {
        self.expect_keyword(Keyword::With)?;
        let recursive = self.consume_keyword(Keyword::Recursive);

        let mut ctes = Vec::new();
        loop {
            let name = self.expect_ident()?;
            let columns = if self.consume_token(&Token::LParen) {
                let cols = self.parse_ident_list()?;
                self.expect_token(&Token::RParen)?;
                Some(cols)
            } else {
                None
            };
            self.expect_keyword(Keyword::As)?;
            self.expect_token(&Token::LParen)?;
            let query = self.parse_select()?;
            self.expect_token(&Token::RParen)?;

            ctes.push(Cte {
                name,
                columns,
                query: self.arena.alloc(query),
            });

            if !self.consume_token(&Token::Comma) {
                break;
            }
        }

        let ctes_slice = self.arena.alloc_slice_copy(&ctes);
        Ok(self.arena.alloc(WithClause {
            recursive,
            ctes: ctes_slice,
        }))
    }

    fn parse_select_columns(&mut self) -> Result<&'a [SelectColumn<'a>]> {
        let mut columns = Vec::new();
        loop {
            if self.consume_token(&Token::Star) {
                columns.push(SelectColumn::AllColumns);
            } else {
                let expr = self.parse_expr(0)?;

                if let Expr::Column(ref col) = expr {
                    if self.consume_token(&Token::Dot) {
                        if self.consume_token(&Token::Star) {
                            columns.push(SelectColumn::TableAllColumns(col.column));
                            if !self.consume_token(&Token::Comma) {
                                break;
                            }
                            continue;
                        } else {
                            let col2 = self.expect_ident()?;
                            let new_expr = self.arena.alloc(Expr::Column(ColumnRef {
                                schema: None,
                                table: Some(col.column),
                                column: col2,
                            }));
                            let alias = self.parse_optional_alias()?;
                            columns.push(SelectColumn::Expr {
                                expr: new_expr,
                                alias,
                            });
                        }
                    } else {
                        let alias = self.parse_optional_alias()?;
                        columns.push(SelectColumn::Expr {
                            expr: self.arena.alloc(expr),
                            alias,
                        });
                    }
                } else {
                    let alias = self.parse_optional_alias()?;
                    columns.push(SelectColumn::Expr {
                        expr: self.arena.alloc(expr),
                        alias,
                    });
                }
            }

            if !self.consume_token(&Token::Comma) {
                break;
            }
        }
        Ok(self.arena.alloc_slice_copy(&columns))
    }

    fn parse_optional_alias(&mut self) -> Result<Option<&'a str>> {
        if self.consume_keyword(Keyword::As) {
            Ok(Some(self.expect_ident()?))
        } else if let Token::Ident(_) = self.peek() {
            if !self.is_keyword_at_current() {
                Ok(Some(self.expect_ident()?))
            } else {
                Ok(None)
            }
        } else {
            Ok(None)
        }
    }

    fn is_keyword_at_current(&self) -> bool {
        matches!(
            self.peek(),
            Token::Keyword(
                Keyword::From
                    | Keyword::Where
                    | Keyword::Group
                    | Keyword::Having
                    | Keyword::Order
                    | Keyword::Limit
                    | Keyword::Offset
                    | Keyword::Union
                    | Keyword::Intersect
                    | Keyword::Except
                    | Keyword::Inner
                    | Keyword::Left
                    | Keyword::Right
                    | Keyword::Full
                    | Keyword::Cross
                    | Keyword::Join
                    | Keyword::On
                    | Keyword::And
                    | Keyword::Or
            )
        )
    }

    fn parse_from_clause(&mut self) -> Result<&'a FromClause<'a>> {
        let mut left = self.parse_table_ref()?;

        loop {
            if let Some(join_type) = self.parse_join_type() {
                let right = self.parse_table_ref()?;
                let condition = self.parse_join_condition()?;

                let join = self.arena.alloc(JoinClause {
                    left,
                    join_type,
                    right,
                    condition,
                });
                left = self.arena.alloc(FromClause::Join(join));
            } else if self.consume_token(&Token::Comma) {
                let right = self.parse_table_ref()?;

                let join = self.arena.alloc(JoinClause {
                    left,
                    join_type: JoinType::Cross,
                    right,
                    condition: JoinCondition::None,
                });
                left = self.arena.alloc(FromClause::Join(join));
            } else {
                break;
            }
        }

        Ok(left)
    }

    fn parse_table_ref(&mut self) -> Result<&'a FromClause<'a>> {
        if self.consume_token(&Token::LParen) {
            if self.check_keyword(Keyword::Select) || self.check_keyword(Keyword::With) {
                let query = self.parse_select()?;
                self.expect_token(&Token::RParen)?;
                let alias = self.parse_optional_alias()?.unwrap_or("subquery");
                return Ok(self.arena.alloc(FromClause::Subquery {
                    query: self.arena.alloc(query),
                    alias,
                }));
            }
            let from = self.parse_from_clause()?;
            self.expect_token(&Token::RParen)?;
            return Ok(from);
        }

        let lateral = self.consume_keyword(Keyword::Lateral);
        if lateral {
            self.expect_token(&Token::LParen)?;
            let query = self.parse_select()?;
            self.expect_token(&Token::RParen)?;
            let alias = self.parse_optional_alias()?.unwrap_or("lateral");
            return Ok(self.arena.alloc(FromClause::Lateral {
                subquery: self.arena.alloc(query),
                alias,
            }));
        }

        let name = self.expect_ident()?;
        let (schema, table_name) = if self.consume_token(&Token::Dot) {
            let table_name = self.expect_ident()?;
            (Some(name), table_name)
        } else {
            (None, name)
        };

        let alias = self.parse_optional_alias()?;

        Ok(self.arena.alloc(FromClause::Table(TableRef {
            schema,
            name: table_name,
            alias,
        })))
    }

    fn parse_join_type(&mut self) -> Option<JoinType> {
        if self.consume_keyword(Keyword::Inner) {
            self.consume_keyword(Keyword::Join);
            Some(JoinType::Inner)
        } else if self.consume_keyword(Keyword::Left) {
            self.consume_keyword(Keyword::Outer);
            self.consume_keyword(Keyword::Join);
            Some(JoinType::Left)
        } else if self.consume_keyword(Keyword::Right) {
            self.consume_keyword(Keyword::Outer);
            self.consume_keyword(Keyword::Join);
            Some(JoinType::Right)
        } else if self.consume_keyword(Keyword::Full) {
            self.consume_keyword(Keyword::Outer);
            self.consume_keyword(Keyword::Join);
            Some(JoinType::Full)
        } else if self.consume_keyword(Keyword::Cross) {
            self.consume_keyword(Keyword::Join);
            Some(JoinType::Cross)
        } else if self.consume_keyword(Keyword::Join) {
            Some(JoinType::Inner)
        } else if self.consume_keyword(Keyword::Natural) {
            let has_direction = self.consume_keyword(Keyword::Left)
                || self.consume_keyword(Keyword::Right)
                || self.consume_keyword(Keyword::Full);
            if has_direction {
                self.consume_keyword(Keyword::Outer);
            }
            self.consume_keyword(Keyword::Join);
            Some(JoinType::Inner)
        } else {
            None
        }
    }

    fn parse_join_condition(&mut self) -> Result<JoinCondition<'a>> {
        if self.consume_keyword(Keyword::On) {
            let expr = self.parse_expr(0)?;
            Ok(JoinCondition::On(self.arena.alloc(expr)))
        } else if self.consume_keyword(Keyword::Using) {
            self.expect_token(&Token::LParen)?;
            let columns = self.parse_ident_list()?;
            self.expect_token(&Token::RParen)?;
            Ok(JoinCondition::Using(columns))
        } else {
            Ok(JoinCondition::None)
        }
    }

    fn parse_order_by_list(&mut self) -> Result<&'a [OrderByItem<'a>]> {
        let mut items = Vec::new();
        loop {
            let expr = self.parse_expr(0)?;
            let direction = if self.consume_keyword(Keyword::Desc) {
                OrderDirection::Desc
            } else {
                self.consume_keyword(Keyword::Asc);
                OrderDirection::Asc
            };
            let nulls = if self.consume_keyword(Keyword::Nulls) {
                if self.consume_keyword(Keyword::First) {
                    NullsOrder::First
                } else if self.consume_keyword(Keyword::Last) {
                    NullsOrder::Last
                } else {
                    NullsOrder::Default
                }
            } else {
                NullsOrder::Default
            };
            items.push(OrderByItem {
                expr: self.arena.alloc(expr),
                direction,
                nulls,
            });
            if !self.consume_token(&Token::Comma) {
                break;
            }
        }
        Ok(self.arena.alloc_slice_copy(&items))
    }

    fn parse_set_operation(&mut self) -> Result<Option<&'a SetOperation<'a>>> {
        let op = if self.consume_keyword(Keyword::Union) {
            SetOperator::Union
        } else if self.consume_keyword(Keyword::Intersect) {
            SetOperator::Intersect
        } else if self.consume_keyword(Keyword::Except) {
            SetOperator::Except
        } else {
            return Ok(None);
        };

        let all = self.consume_keyword(Keyword::All);
        self.consume_keyword(Keyword::Distinct);

        let right = self.parse_select()?;
        Ok(Some(self.arena.alloc(SetOperation {
            op,
            all,
            right: self.arena.alloc(right),
        })))
    }

    fn parse_expr(&mut self, min_bp: u8) -> Result<Expr<'a>> {
        let mut lhs = self.parse_prefix()?;

        loop {
            let op = match self.peek() {
                Token::Plus => Some((BinaryOperator::Plus, 10, 11)),
                Token::Minus => Some((BinaryOperator::Minus, 10, 11)),
                Token::Star => Some((BinaryOperator::Multiply, 12, 13)),
                Token::Slash => Some((BinaryOperator::Divide, 12, 13)),
                Token::Percent => Some((BinaryOperator::Modulo, 12, 13)),
                Token::Caret => Some((BinaryOperator::Power, 15, 14)),
                Token::DoublePipe => Some((BinaryOperator::Concat, 8, 9)),
                Token::Eq => Some((BinaryOperator::Eq, 6, 7)),
                Token::NotEq => Some((BinaryOperator::NotEq, 6, 7)),
                Token::Lt => Some((BinaryOperator::Lt, 6, 7)),
                Token::LtEq => Some((BinaryOperator::LtEq, 6, 7)),
                Token::Gt => Some((BinaryOperator::Gt, 6, 7)),
                Token::GtEq => Some((BinaryOperator::GtEq, 6, 7)),
                Token::Keyword(Keyword::And) => Some((BinaryOperator::And, 4, 5)),
                Token::Keyword(Keyword::Or) => Some((BinaryOperator::Or, 2, 3)),
                Token::Ampersand => Some((BinaryOperator::BitwiseAnd, 8, 9)),
                Token::Pipe => Some((BinaryOperator::BitwiseOr, 8, 9)),
                Token::LeftShift => Some((BinaryOperator::LeftShift, 8, 9)),
                Token::RightShift => Some((BinaryOperator::RightShift, 8, 9)),
                Token::Arrow => Some((BinaryOperator::JsonExtract, 16, 17)),
                Token::DoubleArrow => Some((BinaryOperator::JsonExtractText, 16, 17)),
                Token::HashArrow => Some((BinaryOperator::JsonPathExtract, 16, 17)),
                Token::HashDoubleArrow => Some((BinaryOperator::JsonPathExtractText, 16, 17)),
                Token::AtGt => Some((BinaryOperator::JsonContains, 6, 7)),
                Token::LtAt => Some((BinaryOperator::JsonContainedBy, 6, 7)),
                Token::DoubleAmpersand => Some((BinaryOperator::ArrayOverlaps, 6, 7)),
                Token::LtMinusGt => Some((BinaryOperator::VectorL2Distance, 6, 7)),
                Token::LtHashGt => Some((BinaryOperator::VectorInnerProduct, 6, 7)),
                Token::Spaceship => Some((BinaryOperator::VectorCosineDistance, 6, 7)),
                _ => None,
            };

            if let Some((op, l_bp, r_bp)) = op {
                if l_bp < min_bp {
                    break;
                }
                self.advance();
                let rhs = self.parse_expr(r_bp)?;
                lhs = Expr::BinaryOp {
                    left: self.arena.alloc(lhs),
                    op,
                    right: self.arena.alloc(rhs),
                };
                continue;
            }

            if self.check_keyword(Keyword::Is) {
                if 6 < min_bp {
                    break;
                }
                self.advance();
                let negated = self.consume_keyword(Keyword::Not);
                if self.consume_keyword(Keyword::Null) {
                    lhs = Expr::IsNull {
                        expr: self.arena.alloc(lhs),
                        negated,
                    };
                    continue;
                } else if self.consume_keyword(Keyword::Distinct) {
                    self.expect_keyword(Keyword::From)?;
                    let right = self.parse_expr(7)?;
                    lhs = Expr::IsDistinctFrom {
                        left: self.arena.alloc(lhs),
                        right: self.arena.alloc(right),
                        negated,
                    };
                    continue;
                }
            }

            let negated = self.consume_keyword(Keyword::Not);

            if self.check_keyword(Keyword::Between) {
                if 6 < min_bp {
                    break;
                }
                self.advance();
                let low = self.parse_expr(7)?;
                self.expect_keyword(Keyword::And)?;
                let high = self.parse_expr(7)?;
                lhs = Expr::Between {
                    expr: self.arena.alloc(lhs),
                    negated,
                    low: self.arena.alloc(low),
                    high: self.arena.alloc(high),
                };
                continue;
            }

            if self.check_keyword(Keyword::In) {
                if 6 < min_bp {
                    break;
                }
                self.advance();
                self.expect_token(&Token::LParen)?;
                if self.check_keyword(Keyword::Select) || self.check_keyword(Keyword::With) {
                    let subquery = self.parse_select()?;
                    self.expect_token(&Token::RParen)?;
                    lhs = Expr::InSubquery {
                        expr: self.arena.alloc(lhs),
                        negated,
                        subquery: self.arena.alloc(subquery),
                    };
                } else {
                    let list = self.parse_expr_list()?;
                    self.expect_token(&Token::RParen)?;
                    lhs = Expr::InList {
                        expr: self.arena.alloc(lhs),
                        negated,
                        list,
                    };
                }
                continue;
            }

            if self.check_keyword(Keyword::Like) || self.check_keyword(Keyword::Ilike) {
                if 6 < min_bp {
                    break;
                }
                let case_insensitive = self.check_keyword(Keyword::Ilike);
                self.advance();
                let pattern = self.parse_expr(7)?;
                let escape: Option<&Expr<'a>> = if self.consume_keyword(Keyword::Escape) {
                    Some(self.arena.alloc(self.parse_expr(7)?))
                } else {
                    None
                };
                lhs = Expr::Like {
                    expr: self.arena.alloc(lhs),
                    negated,
                    pattern: self.arena.alloc(pattern),
                    escape,
                    case_insensitive,
                };
                continue;
            }

            if negated {
                bail!("unexpected NOT without IN, BETWEEN, or LIKE");
            }

            if self.check_token(&Token::DoubleColon) {
                if 18 < min_bp {
                    break;
                }
                self.advance();
                let data_type = self.parse_data_type()?;
                lhs = Expr::Cast {
                    expr: self.arena.alloc(lhs),
                    data_type,
                };
                continue;
            }

            if self.check_token(&Token::LBracket) {
                if 20 < min_bp {
                    break;
                }
                self.advance();
                let index = self.parse_expr(0)?;
                if self.consume_token(&Token::Colon) {
                    let upper: Option<&Expr<'a>> = if self.check_token(&Token::RBracket) {
                        None
                    } else {
                        Some(self.arena.alloc(self.parse_expr(0)?))
                    };
                    self.expect_token(&Token::RBracket)?;
                    lhs = Expr::ArraySlice {
                        array: self.arena.alloc(lhs),
                        lower: Some(self.arena.alloc(index)),
                        upper,
                    };
                } else {
                    self.expect_token(&Token::RBracket)?;
                    lhs = Expr::ArraySubscript {
                        array: self.arena.alloc(lhs),
                        index: self.arena.alloc(index),
                    };
                }
                continue;
            }

            break;
        }

        Ok(lhs)
    }

    fn parse_prefix(&mut self) -> Result<Expr<'a>> {
        match self.peek().clone() {
            Token::Keyword(Keyword::Not) => {
                self.advance();
                if self.check_keyword(Keyword::Exists) {
                    self.advance();
                    self.expect_token(&Token::LParen)?;
                    let subquery = self.parse_select()?;
                    self.expect_token(&Token::RParen)?;
                    Ok(Expr::Exists {
                        subquery: self.arena.alloc(subquery),
                        negated: true,
                    })
                } else {
                    let expr = self.parse_expr(14)?;
                    Ok(Expr::UnaryOp {
                        op: UnaryOperator::Not,
                        expr: self.arena.alloc(expr),
                    })
                }
            }
            Token::Minus => {
                self.advance();
                let expr = self.parse_expr(14)?;
                Ok(Expr::UnaryOp {
                    op: UnaryOperator::Minus,
                    expr: self.arena.alloc(expr),
                })
            }
            Token::Plus => {
                self.advance();
                let expr = self.parse_expr(14)?;
                Ok(Expr::UnaryOp {
                    op: UnaryOperator::Plus,
                    expr: self.arena.alloc(expr),
                })
            }
            Token::Tilde => {
                self.advance();
                let expr = self.parse_expr(14)?;
                Ok(Expr::UnaryOp {
                    op: UnaryOperator::BitwiseNot,
                    expr: self.arena.alloc(expr),
                })
            }
            Token::Integer(s) => {
                self.advance();
                Ok(Expr::Literal(Literal::Integer(s)))
            }
            Token::Float(s) => {
                self.advance();
                Ok(Expr::Literal(Literal::Float(s)))
            }
            Token::String(s) => {
                self.advance();
                let unescaped = if s.contains("''") {
                    self.arena.alloc_str(&s.replace("''", "'"))
                } else {
                    s
                };
                Ok(Expr::Literal(Literal::String(unescaped)))
            }
            Token::HexNumber(s) => {
                self.advance();
                Ok(Expr::Literal(Literal::HexNumber(s)))
            }
            Token::BinaryNumber(s) => {
                self.advance();
                Ok(Expr::Literal(Literal::BinaryNumber(s)))
            }
            Token::Keyword(Keyword::True) => {
                self.advance();
                Ok(Expr::Literal(Literal::Boolean(true)))
            }
            Token::Keyword(Keyword::False) => {
                self.advance();
                Ok(Expr::Literal(Literal::Boolean(false)))
            }
            Token::Keyword(Keyword::Null) => {
                self.advance();
                Ok(Expr::Literal(Literal::Null))
            }
            Token::Parameter(p) => {
                self.advance();
                let param = match p {
                    Parameter::Positional(n) => ParameterRef::Positional(n),
                    Parameter::Named(s) => ParameterRef::Named(s),
                    Parameter::Anonymous => ParameterRef::Anonymous,
                };
                Ok(Expr::Parameter(param))
            }
            Token::LParen => {
                self.advance();
                if self.check_keyword(Keyword::Select) || self.check_keyword(Keyword::With) {
                    let subquery = self.parse_select()?;
                    self.expect_token(&Token::RParen)?;
                    Ok(Expr::Subquery(self.arena.alloc(subquery)))
                } else {
                    let expr = self.parse_expr(0)?;
                    if self.consume_token(&Token::Comma) {
                        let mut exprs: Vec<&Expr<'a>> = vec![self.arena.alloc(expr)];
                        loop {
                            exprs.push(self.arena.alloc(self.parse_expr(0)?));
                            if !self.consume_token(&Token::Comma) {
                                break;
                            }
                        }
                        self.expect_token(&Token::RParen)?;
                        Ok(Expr::Row(self.arena.alloc_slice_copy(&exprs)))
                    } else {
                        self.expect_token(&Token::RParen)?;
                        Ok(expr)
                    }
                }
            }
            Token::LBracket => {
                self.advance();
                if self.check_token(&Token::RBracket) {
                    self.advance();
                    Ok(Expr::Array(&[]))
                } else {
                    let exprs = self.parse_expr_list()?;
                    self.expect_token(&Token::RBracket)?;
                    Ok(Expr::Array(exprs))
                }
            }
            Token::Keyword(Keyword::Case) => self.parse_case(),
            Token::Keyword(Keyword::Cast) => self.parse_cast(),
            Token::Keyword(Keyword::Exists) => {
                self.advance();
                self.expect_token(&Token::LParen)?;
                let subquery = self.parse_select()?;
                self.expect_token(&Token::RParen)?;
                Ok(Expr::Exists {
                    subquery: self.arena.alloc(subquery),
                    negated: false,
                })
            }
            Token::Keyword(Keyword::Array) => {
                self.advance();
                self.expect_token(&Token::LBracket)?;
                if self.check_token(&Token::RBracket) {
                    self.advance();
                    Ok(Expr::Array(&[]))
                } else {
                    let exprs = self.parse_expr_list()?;
                    self.expect_token(&Token::RBracket)?;
                    Ok(Expr::Array(exprs))
                }
            }
            Token::Ident(name) => {
                self.advance();
                if self.check_token(&Token::LParen) {
                    self.parse_function_call(name)
                } else if self.check_token(&Token::Dot) {
                    if self.lexer.peek() == Token::Star {
                        Ok(Expr::Column(ColumnRef {
                            schema: None,
                            table: None,
                            column: name,
                        }))
                    } else {
                        self.advance();
                        let col = self.expect_ident()?;
                        if self.consume_token(&Token::Dot) {
                            let col2 = self.expect_ident()?;
                            Ok(Expr::Column(ColumnRef {
                                schema: Some(name),
                                table: Some(col),
                                column: col2,
                            }))
                        } else {
                            Ok(Expr::Column(ColumnRef {
                                schema: None,
                                table: Some(name),
                                column: col,
                            }))
                        }
                    }
                } else {
                    Ok(Expr::Column(ColumnRef {
                        schema: None,
                        table: None,
                        column: name,
                    }))
                }
            }
            Token::QuotedIdent(name) => {
                self.advance();
                if self.check_token(&Token::LParen) {
                    self.parse_function_call(name)
                } else {
                    Ok(Expr::Column(ColumnRef {
                        schema: None,
                        table: None,
                        column: name,
                    }))
                }
            }
            Token::Keyword(kw) => {
                let name = kw.as_str();
                self.advance();
                if self.check_token(&Token::LParen) {
                    self.parse_function_call(name)
                } else {
                    Ok(Expr::Column(ColumnRef {
                        schema: None,
                        table: None,
                        column: name,
                    }))
                }
            }
            _ => bail!("unexpected token {:?} in expression", self.current),
        }
    }

    fn parse_case(&mut self) -> Result<Expr<'a>> {
        self.expect_keyword(Keyword::Case)?;

        let operand: Option<&Expr<'a>> = if !self.check_keyword(Keyword::When) {
            Some(self.arena.alloc(self.parse_expr(0)?))
        } else {
            None
        };

        let mut conditions = Vec::new();
        while self.consume_keyword(Keyword::When) {
            let condition = self.parse_expr(0)?;
            self.expect_keyword(Keyword::Then)?;
            let result = self.parse_expr(0)?;
            conditions.push(WhenClause {
                condition: self.arena.alloc(condition),
                result: self.arena.alloc(result),
            });
        }

        let else_result: Option<&Expr<'a>> = if self.consume_keyword(Keyword::Else) {
            Some(self.arena.alloc(self.parse_expr(0)?))
        } else {
            None
        };

        self.expect_keyword(Keyword::End)?;

        Ok(Expr::Case {
            operand,
            conditions: self.arena.alloc_slice_copy(&conditions),
            else_result,
        })
    }

    fn parse_cast(&mut self) -> Result<Expr<'a>> {
        self.expect_keyword(Keyword::Cast)?;
        self.expect_token(&Token::LParen)?;
        let expr = self.parse_expr(0)?;
        self.expect_keyword(Keyword::As)?;
        let data_type = self.parse_data_type()?;
        self.expect_token(&Token::RParen)?;
        Ok(Expr::Cast {
            expr: self.arena.alloc(expr),
            data_type,
        })
    }

    fn parse_function_call(&mut self, name: &'a str) -> Result<Expr<'a>> {
        self.expect_token(&Token::LParen)?;

        let distinct = self.consume_keyword(Keyword::Distinct);

        let args = if self.consume_token(&Token::Star) {
            FunctionArgs::Star
        } else if self.check_token(&Token::RParen) {
            FunctionArgs::None
        } else {
            let mut args = Vec::new();
            loop {
                let value = self.parse_expr(0)?;
                args.push(FunctionArg {
                    name: None,
                    value: self.arena.alloc(value),
                });
                if !self.consume_token(&Token::Comma) {
                    break;
                }
            }
            FunctionArgs::Args(self.arena.alloc_slice_copy(&args))
        };

        self.expect_token(&Token::RParen)?;

        let filter: Option<&Expr<'a>> = if self.consume_keyword(Keyword::Filter) {
            self.expect_token(&Token::LParen)?;
            self.expect_keyword(Keyword::Where)?;
            let expr = self.parse_expr(0)?;
            self.expect_token(&Token::RParen)?;
            Some(self.arena.alloc(expr))
        } else {
            None
        };

        let over = if self.consume_keyword(Keyword::Over) {
            Some(self.parse_window_spec()?)
        } else {
            None
        };

        Ok(Expr::Function(FunctionCall {
            name: FunctionName { schema: None, name },
            args,
            distinct,
            filter,
            over,
        }))
    }

    fn parse_window_spec(&mut self) -> Result<WindowSpec<'a>> {
        self.expect_token(&Token::LParen)?;

        let partition_by = if self.consume_keyword(Keyword::Partition) {
            self.expect_keyword(Keyword::By)?;
            self.parse_expr_list()?
        } else {
            &[]
        };

        let order_by = if self.consume_keyword(Keyword::Order) {
            self.expect_keyword(Keyword::By)?;
            self.parse_order_by_list()?
        } else {
            &[]
        };

        let frame = if self.check_keyword(Keyword::Rows)
            || self.check_keyword(Keyword::Range)
            || self.consume_keyword(Keyword::Current)
        {
            Some(self.parse_window_frame()?)
        } else {
            None
        };

        self.expect_token(&Token::RParen)?;

        Ok(WindowSpec {
            partition_by,
            order_by,
            frame,
        })
    }

    fn parse_window_frame(&mut self) -> Result<WindowFrame> {
        let mode = if self.consume_keyword(Keyword::Rows) {
            WindowFrameMode::Rows
        } else if self.consume_keyword(Keyword::Range) {
            WindowFrameMode::Range
        } else {
            WindowFrameMode::Rows
        };

        let start = self.parse_window_frame_bound()?;

        let end = if self.consume_keyword(Keyword::And) {
            Some(self.parse_window_frame_bound()?)
        } else {
            None
        };

        Ok(WindowFrame { mode, start, end })
    }

    fn parse_window_frame_bound(&mut self) -> Result<WindowFrameBound> {
        if self.consume_keyword(Keyword::Current) {
            self.expect_keyword(Keyword::Row)?;
            Ok(WindowFrameBound::CurrentRow)
        } else if self.consume_keyword(Keyword::Unbounded) {
            if self.consume_keyword(Keyword::Preceding) {
                Ok(WindowFrameBound::UnboundedPreceding)
            } else {
                self.expect_keyword(Keyword::Following)?;
                Ok(WindowFrameBound::UnboundedFollowing)
            }
        } else if let Token::Integer(n) = self.peek().clone() {
            self.advance();
            let n: u64 = n.parse().unwrap_or(0);
            if self.consume_keyword(Keyword::Preceding) {
                Ok(WindowFrameBound::Preceding(n))
            } else {
                self.expect_keyword(Keyword::Following)?;
                Ok(WindowFrameBound::Following(n))
            }
        } else {
            bail!("expected window frame bound");
        }
    }

    fn parse_expr_list(&mut self) -> Result<&'a [&'a Expr<'a>]> {
        let mut exprs: Vec<&Expr<'a>> = Vec::new();
        loop {
            exprs.push(self.arena.alloc(self.parse_expr(0)?));
            if !self.consume_token(&Token::Comma) {
                break;
            }
        }
        Ok(self.arena.alloc_slice_copy(&exprs))
    }

    fn parse_ident_list(&mut self) -> Result<&'a [&'a str]> {
        let mut idents = Vec::new();
        loop {
            idents.push(self.expect_ident()?);
            if !self.consume_token(&Token::Comma) {
                break;
            }
        }
        Ok(self.arena.alloc_slice_copy(&idents))
    }

    fn expect_ident(&mut self) -> Result<&'a str> {
        match self.advance() {
            Token::Ident(s) => Ok(s),
            Token::QuotedIdent(s) => Ok(s),
            Token::Keyword(kw) => Ok(kw.as_str()),
            other => bail!("expected identifier, found {:?}", other),
        }
    }

    fn parse_insert(&mut self) -> Result<InsertStmt<'a>> {
        self.expect_keyword(Keyword::Insert)?;
        self.expect_keyword(Keyword::Into)?;

        let table = self.parse_table_name()?;

        let columns = if self.consume_token(&Token::LParen) {
            let cols = self.parse_ident_list()?;
            self.expect_token(&Token::RParen)?;
            Some(cols)
        } else {
            None
        };

        let source = if self.consume_keyword(Keyword::Values) {
            let mut rows = Vec::new();
            loop {
                self.expect_token(&Token::LParen)?;
                let exprs = self.parse_expr_list()?;
                self.expect_token(&Token::RParen)?;
                rows.push(exprs);
                if !self.consume_token(&Token::Comma) {
                    break;
                }
            }
            InsertSource::Values(self.arena.alloc_slice_copy(&rows))
        } else if self.consume_keyword(Keyword::Default) {
            self.expect_keyword(Keyword::Values)?;
            InsertSource::Default
        } else if self.check_keyword(Keyword::Select) || self.check_keyword(Keyword::With) {
            let select = self.parse_select()?;
            InsertSource::Select(self.arena.alloc(select))
        } else {
            bail!("expected VALUES, DEFAULT VALUES, or SELECT");
        };

        let on_conflict = if self.consume_keyword(Keyword::On) {
            self.expect_keyword(Keyword::Conflict)?;
            Some(self.parse_on_conflict()?)
        } else {
            None
        };

        let returning = if self.consume_keyword(Keyword::Returning) {
            Some(self.parse_select_columns()?)
        } else {
            None
        };

        Ok(InsertStmt {
            table,
            columns,
            source,
            on_conflict,
            returning,
        })
    }

    fn parse_on_conflict(&mut self) -> Result<&'a OnConflict<'a>> {
        let target = if self.consume_token(&Token::LParen) {
            let cols = self.parse_ident_list()?;
            self.expect_token(&Token::RParen)?;
            OnConflictTarget::Columns(cols)
        } else if self.consume_keyword(Keyword::On) {
            self.expect_keyword(Keyword::Constraint)?;
            let name = self.expect_ident()?;
            OnConflictTarget::Constraint(name)
        } else {
            OnConflictTarget::None
        };

        self.expect_keyword(Keyword::Do)?;

        let action = if self.consume_keyword(Keyword::Nothing) {
            OnConflictAction::DoNothing
        } else {
            self.expect_keyword(Keyword::Update)?;
            self.expect_keyword(Keyword::Set)?;
            let assignments = self.parse_assignments()?;
            OnConflictAction::DoUpdate(assignments)
        };

        Ok(self.arena.alloc(OnConflict { target, action }))
    }

    fn parse_update(&mut self) -> Result<UpdateStmt<'a>> {
        self.expect_keyword(Keyword::Update)?;
        let table = self.parse_table_name()?;
        self.expect_keyword(Keyword::Set)?;
        let assignments = self.parse_assignments()?;

        let from = if self.consume_keyword(Keyword::From) {
            Some(self.parse_from_clause()?)
        } else {
            None
        };

        let where_clause: Option<&Expr<'a>> = if self.consume_keyword(Keyword::Where) {
            Some(self.arena.alloc(self.parse_expr(0)?))
        } else {
            None
        };

        let returning = if self.consume_keyword(Keyword::Returning) {
            Some(self.parse_select_columns()?)
        } else {
            None
        };

        Ok(UpdateStmt {
            table,
            assignments,
            from,
            where_clause,
            returning,
        })
    }

    fn parse_assignments(&mut self) -> Result<&'a [Assignment<'a>]> {
        let mut assignments = Vec::new();
        loop {
            let first = self.expect_ident()?;
            let column = if self.consume_token(&Token::Dot) {
                let second = self.expect_ident()?;
                if self.consume_token(&Token::Dot) {
                    let third = self.expect_ident()?;
                    ColumnRef {
                        schema: Some(first),
                        table: Some(second),
                        column: third,
                    }
                } else {
                    ColumnRef {
                        schema: None,
                        table: Some(first),
                        column: second,
                    }
                }
            } else {
                ColumnRef {
                    schema: None,
                    table: None,
                    column: first,
                }
            };
            self.expect_token(&Token::Eq)?;
            let value = self.parse_expr(0)?;
            assignments.push(Assignment {
                column,
                value: self.arena.alloc(value),
            });
            if !self.consume_token(&Token::Comma) {
                break;
            }
        }
        Ok(self.arena.alloc_slice_copy(&assignments))
    }

    fn parse_delete(&mut self) -> Result<DeleteStmt<'a>> {
        self.expect_keyword(Keyword::Delete)?;
        self.expect_keyword(Keyword::From)?;
        let table = self.parse_table_name()?;

        let using = if self.consume_keyword(Keyword::Using) {
            Some(self.parse_from_clause()?)
        } else {
            None
        };

        let where_clause: Option<&Expr<'a>> = if self.consume_keyword(Keyword::Where) {
            Some(self.arena.alloc(self.parse_expr(0)?))
        } else {
            None
        };

        let returning = if self.consume_keyword(Keyword::Returning) {
            Some(self.parse_select_columns()?)
        } else {
            None
        };

        Ok(DeleteStmt {
            table,
            using,
            where_clause,
            returning,
        })
    }

    fn parse_table_name(&mut self) -> Result<TableRef<'a>> {
        let name = self.expect_ident()?;
        if self.consume_token(&Token::Dot) {
            let table_name = self.expect_ident()?;
            let alias = self.parse_optional_alias()?;
            Ok(TableRef {
                schema: Some(name),
                name: table_name,
                alias,
            })
        } else {
            let alias = self.parse_optional_alias()?;
            Ok(TableRef {
                schema: None,
                name,
                alias,
            })
        }
    }

    fn parse_create(&mut self) -> Result<Statement<'a>> {
        self.expect_keyword(Keyword::Create)?;

        let or_replace = if self.consume_keyword(Keyword::Or) {
            self.expect_keyword(Keyword::Replace)?;
            true
        } else {
            false
        };

        if self.consume_keyword(Keyword::Table) {
            let stmt = self.parse_create_table()?;
            Ok(Statement::CreateTable(self.arena.alloc(stmt)))
        } else if self.consume_keyword(Keyword::Index) || self.check_keyword(Keyword::Unique) {
            let unique = self.consume_keyword(Keyword::Unique);
            if unique {
                self.expect_keyword(Keyword::Index)?;
            }
            let stmt = self.parse_create_index(unique)?;
            Ok(Statement::CreateIndex(self.arena.alloc(stmt)))
        } else if self.consume_keyword(Keyword::Schema) {
            let stmt = self.parse_create_schema()?;
            Ok(Statement::CreateSchema(self.arena.alloc(stmt)))
        } else if self.consume_keyword(Keyword::View) {
            let stmt = self.parse_create_view(or_replace, false)?;
            Ok(Statement::CreateView(self.arena.alloc(stmt)))
        } else if self.consume_keyword(Keyword::Materialized) {
            self.expect_keyword(Keyword::View)?;
            let stmt = self.parse_create_view(or_replace, true)?;
            Ok(Statement::CreateView(self.arena.alloc(stmt)))
        } else if self.consume_keyword(Keyword::Function) {
            let stmt = self.parse_create_function(or_replace)?;
            Ok(Statement::CreateFunction(self.arena.alloc(stmt)))
        } else if self.consume_keyword(Keyword::Procedure) {
            let stmt = self.parse_create_procedure(or_replace)?;
            Ok(Statement::CreateProcedure(self.arena.alloc(stmt)))
        } else if self.consume_keyword(Keyword::Trigger) {
            let stmt = self.parse_create_trigger(or_replace)?;
            Ok(Statement::CreateTrigger(self.arena.alloc(stmt)))
        } else if self.consume_keyword(Keyword::Type) {
            let stmt = self.parse_create_type()?;
            Ok(Statement::CreateType(self.arena.alloc(stmt)))
        } else if self.consume_keyword(Keyword::Domain) {
            let stmt = self.parse_create_domain()?;
            Ok(Statement::CreateType(self.arena.alloc(stmt)))
        } else {
            bail!("expected TABLE, INDEX, SCHEMA, VIEW, MATERIALIZED, FUNCTION, PROCEDURE, TRIGGER, TYPE, or DOMAIN after CREATE");
        }
    }

    fn parse_create_table(&mut self) -> Result<CreateTableStmt<'a>> {
        let if_not_exists = if self.consume_keyword(Keyword::If) {
            self.expect_keyword(Keyword::Not)?;
            self.expect_keyword(Keyword::Exists)?;
            true
        } else {
            false
        };

        let name = self.expect_ident()?;
        let (schema, table_name) = if self.consume_token(&Token::Dot) {
            (Some(name), self.expect_ident()?)
        } else {
            (None, name)
        };

        self.expect_token(&Token::LParen)?;

        let mut columns = Vec::new();
        let mut constraints = Vec::new();

        loop {
            if self.check_keyword(Keyword::Primary)
                || self.check_keyword(Keyword::Unique)
                || self.check_keyword(Keyword::Foreign)
                || self.check_keyword(Keyword::Check)
                || self.check_keyword(Keyword::Constraint)
            {
                constraints.push(self.parse_table_constraint()?);
            } else {
                columns.push(self.parse_column_def()?);
            }

            if !self.consume_token(&Token::Comma) {
                break;
            }
        }

        self.expect_token(&Token::RParen)?;

        Ok(CreateTableStmt {
            if_not_exists,
            schema,
            name: table_name,
            columns: self.arena.alloc_slice_copy(&columns),
            constraints: self.arena.alloc_slice_copy(&constraints),
            temporary: false,
        })
    }

    fn parse_column_def(&mut self) -> Result<ColumnDef<'a>> {
        let name = self.expect_ident()?;
        let data_type = self.parse_data_type()?;

        let mut constraints = Vec::new();
        loop {
            if self.consume_keyword(Keyword::Not) {
                self.expect_keyword(Keyword::Null)?;
                constraints.push(ColumnConstraint::NotNull);
            } else if self.consume_keyword(Keyword::Null) {
                constraints.push(ColumnConstraint::Null);
            } else if self.consume_keyword(Keyword::Primary) {
                self.expect_keyword(Keyword::Key)?;
                constraints.push(ColumnConstraint::PrimaryKey);
            } else if self.consume_keyword(Keyword::Unique) {
                constraints.push(ColumnConstraint::Unique);
            } else if self.consume_keyword(Keyword::Default) {
                let expr = self.parse_expr(0)?;
                constraints.push(ColumnConstraint::Default(self.arena.alloc(expr)));
            } else if self.consume_keyword(Keyword::Check) {
                self.expect_token(&Token::LParen)?;
                let expr = self.parse_expr(0)?;
                self.expect_token(&Token::RParen)?;
                constraints.push(ColumnConstraint::Check(self.arena.alloc(expr)));
            } else if self.consume_keyword(Keyword::References) {
                let table = self.expect_ident()?;
                let column = if self.consume_token(&Token::LParen) {
                    let col = self.expect_ident()?;
                    self.expect_token(&Token::RParen)?;
                    Some(col)
                } else {
                    None
                };
                let on_delete = self.parse_referential_action(Keyword::Delete)?;
                let on_update = self.parse_referential_action(Keyword::Update)?;
                constraints.push(ColumnConstraint::References {
                    table,
                    column,
                    on_delete,
                    on_update,
                });
            } else if self.consume_keyword(Keyword::Generated) {
                self.expect_keyword(Keyword::Always)?;
                self.expect_keyword(Keyword::As)?;
                self.expect_token(&Token::LParen)?;
                let expr = self.parse_expr(0)?;
                self.expect_token(&Token::RParen)?;
                let stored = self.consume_keyword(Keyword::Stored);
                constraints.push(ColumnConstraint::Generated {
                    expr: self.arena.alloc(expr),
                    stored,
                });
            } else {
                break;
            }
        }

        Ok(ColumnDef {
            name,
            data_type,
            constraints: self.arena.alloc_slice_copy(&constraints),
        })
    }

    fn parse_referential_action(&mut self, keyword: Keyword) -> Result<Option<ReferentialAction>> {
        if self.consume_keyword(Keyword::On) {
            self.expect_keyword(keyword)?;
            if self.consume_keyword(Keyword::Cascade) {
                Ok(Some(ReferentialAction::Cascade))
            } else if self.consume_keyword(Keyword::Restrict) {
                Ok(Some(ReferentialAction::Restrict))
            } else if self.consume_keyword(Keyword::Set) {
                if self.consume_keyword(Keyword::Null) {
                    Ok(Some(ReferentialAction::SetNull))
                } else {
                    self.expect_keyword(Keyword::Default)?;
                    Ok(Some(ReferentialAction::SetDefault))
                }
            } else if self.consume_keyword(Keyword::No) {
                self.expect_keyword(Keyword::Action)?;
                Ok(Some(ReferentialAction::NoAction))
            } else {
                Ok(None)
            }
        } else {
            Ok(None)
        }
    }

    fn parse_table_constraint(&mut self) -> Result<TableConstraint<'a>> {
        let name = if self.consume_keyword(Keyword::Constraint) {
            Some(self.expect_ident()?)
        } else {
            None
        };

        if self.consume_keyword(Keyword::Primary) {
            self.expect_keyword(Keyword::Key)?;
            self.expect_token(&Token::LParen)?;
            let columns = self.parse_ident_list()?;
            self.expect_token(&Token::RParen)?;
            Ok(TableConstraint::PrimaryKey { name, columns })
        } else if self.consume_keyword(Keyword::Unique) {
            self.expect_token(&Token::LParen)?;
            let columns = self.parse_ident_list()?;
            self.expect_token(&Token::RParen)?;
            Ok(TableConstraint::Unique { name, columns })
        } else if self.consume_keyword(Keyword::Foreign) {
            self.expect_keyword(Keyword::Key)?;
            self.expect_token(&Token::LParen)?;
            let columns = self.parse_ident_list()?;
            self.expect_token(&Token::RParen)?;
            self.expect_keyword(Keyword::References)?;
            let ref_table = self.expect_ident()?;
            self.expect_token(&Token::LParen)?;
            let ref_columns = self.parse_ident_list()?;
            self.expect_token(&Token::RParen)?;
            let on_delete = self.parse_referential_action(Keyword::Delete)?;
            let on_update = self.parse_referential_action(Keyword::Update)?;
            Ok(TableConstraint::ForeignKey {
                name,
                columns,
                ref_table,
                ref_columns,
                on_delete,
                on_update,
            })
        } else if self.consume_keyword(Keyword::Check) {
            self.expect_token(&Token::LParen)?;
            let expr = self.parse_expr(0)?;
            self.expect_token(&Token::RParen)?;
            Ok(TableConstraint::Check {
                name,
                expr: self.arena.alloc(expr),
            })
        } else {
            bail!("expected PRIMARY KEY, UNIQUE, FOREIGN KEY, or CHECK");
        }
    }

    fn parse_create_index(&mut self, unique: bool) -> Result<CreateIndexStmt<'a>> {
        let if_not_exists = if self.consume_keyword(Keyword::If) {
            self.expect_keyword(Keyword::Not)?;
            self.expect_keyword(Keyword::Exists)?;
            true
        } else {
            false
        };

        let name = self.expect_ident()?;
        self.expect_keyword(Keyword::On)?;
        let table = self.parse_table_name()?;

        let index_type = if self.consume_keyword(Keyword::Using) {
            if self.consume_keyword(Keyword::Btree) {
                Some(IndexType::BTree)
            } else if self.consume_keyword(Keyword::Hash) {
                Some(IndexType::Hash)
            } else if self.consume_keyword(Keyword::Gin) {
                Some(IndexType::Gin)
            } else if self.consume_keyword(Keyword::Gist) {
                Some(IndexType::Gist)
            } else if self.consume_keyword(Keyword::Hnsw) {
                Some(IndexType::Hnsw)
            } else {
                None
            }
        } else {
            None
        };

        self.expect_token(&Token::LParen)?;
        let mut columns = Vec::new();
        loop {
            let expr = self.parse_expr(0)?;
            let direction = if self.consume_keyword(Keyword::Desc) {
                Some(OrderDirection::Desc)
            } else if self.consume_keyword(Keyword::Asc) {
                Some(OrderDirection::Asc)
            } else {
                None
            };
            let nulls = if self.consume_keyword(Keyword::Nulls) {
                if self.consume_keyword(Keyword::First) {
                    Some(NullsOrder::First)
                } else if self.consume_keyword(Keyword::Last) {
                    Some(NullsOrder::Last)
                } else {
                    None
                }
            } else {
                None
            };
            columns.push(IndexColumn {
                expr: self.arena.alloc(expr),
                direction,
                nulls,
            });
            if !self.consume_token(&Token::Comma) {
                break;
            }
        }
        self.expect_token(&Token::RParen)?;

        let where_clause: Option<&Expr<'a>> = if self.consume_keyword(Keyword::Where) {
            Some(self.arena.alloc(self.parse_expr(0)?))
        } else {
            None
        };

        Ok(CreateIndexStmt {
            if_not_exists,
            unique,
            name,
            table,
            columns: self.arena.alloc_slice_copy(&columns),
            index_type,
            where_clause,
        })
    }

    fn parse_create_schema(&mut self) -> Result<CreateSchemaStmt<'a>> {
        let if_not_exists = if self.consume_keyword(Keyword::If) {
            self.expect_keyword(Keyword::Not)?;
            self.expect_keyword(Keyword::Exists)?;
            true
        } else {
            false
        };

        let name = self.expect_ident()?;

        Ok(CreateSchemaStmt {
            if_not_exists,
            name,
        })
    }

    fn parse_create_view(
        &mut self,
        or_replace: bool,
        materialized: bool,
    ) -> Result<CreateViewStmt<'a>> {
        let name = self.expect_ident()?;
        let (schema, view_name) = if self.consume_token(&Token::Dot) {
            (Some(name), self.expect_ident()?)
        } else {
            (None, name)
        };

        let columns = if self.consume_token(&Token::LParen) {
            let cols = self.parse_ident_list()?;
            self.expect_token(&Token::RParen)?;
            Some(cols)
        } else {
            None
        };

        self.expect_keyword(Keyword::As)?;
        let query = self.parse_select()?;

        let with_check_option = if self.consume_keyword(Keyword::With) {
            self.consume_keyword(Keyword::Cascaded);
            self.consume_keyword(Keyword::Local);
            self.expect_keyword(Keyword::Check)?;
            self.expect_keyword(Keyword::Option)?;
            true
        } else {
            false
        };

        Ok(CreateViewStmt {
            or_replace,
            materialized,
            schema,
            name: view_name,
            columns,
            query: self.arena.alloc(query),
            with_check_option,
        })
    }

    fn parse_create_function(&mut self, or_replace: bool) -> Result<CreateFunctionStmt<'a>> {
        let name = self.expect_ident()?;
        let (schema, func_name) = if self.consume_token(&Token::Dot) {
            (Some(name), self.expect_ident()?)
        } else {
            (None, name)
        };

        self.expect_token(&Token::LParen)?;
        let mut params = Vec::new();
        if !self.check_token(&Token::RParen) {
            loop {
                let param_name = self.expect_ident()?;
                let data_type = self.parse_data_type()?;
                params.push(FunctionParam {
                    name: param_name,
                    data_type,
                });
                if !self.consume_token(&Token::Comma) {
                    break;
                }
            }
        }
        self.expect_token(&Token::RParen)?;

        self.expect_keyword(Keyword::Returns)?;
        let return_type = self.parse_data_type()?;

        self.expect_keyword(Keyword::As)?;
        let body = self.parse_function_body()?;

        self.expect_keyword(Keyword::Language)?;
        let language = self.expect_ident()?;

        Ok(CreateFunctionStmt {
            or_replace,
            schema,
            name: func_name,
            params: self.arena.alloc_slice_copy(&params),
            return_type,
            body,
            language,
        })
    }

    fn parse_function_body(&mut self) -> Result<&'a str> {
        match &self.current {
            Token::String(s) => {
                let body = *s;
                self.advance();
                Ok(body)
            }
            _ => bail!("expected function body (string or $$ ... $$)"),
        }
    }

    fn parse_create_procedure(&mut self, or_replace: bool) -> Result<CreateProcedureStmt<'a>> {
        let name = self.expect_ident()?;
        let (schema, proc_name) = if self.consume_token(&Token::Dot) {
            (Some(name), self.expect_ident()?)
        } else {
            (None, name)
        };

        self.expect_token(&Token::LParen)?;
        let mut params = Vec::new();
        if !self.check_token(&Token::RParen) {
            loop {
                let param_name = self.expect_ident()?;
                let data_type = self.parse_data_type()?;
                params.push(FunctionParam {
                    name: param_name,
                    data_type,
                });
                if !self.consume_token(&Token::Comma) {
                    break;
                }
            }
        }
        self.expect_token(&Token::RParen)?;

        self.expect_keyword(Keyword::As)?;
        let body = self.parse_function_body()?;

        self.expect_keyword(Keyword::Language)?;
        let language = self.expect_ident()?;

        Ok(CreateProcedureStmt {
            or_replace,
            schema,
            name: proc_name,
            params: self.arena.alloc_slice_copy(&params),
            body,
            language,
        })
    }

    fn parse_create_trigger(&mut self, or_replace: bool) -> Result<CreateTriggerStmt<'a>> {
        let name = self.expect_ident()?;

        let timing = if self.consume_keyword(Keyword::Before) {
            TriggerTiming::Before
        } else if self.consume_keyword(Keyword::After) {
            TriggerTiming::After
        } else if self.consume_keyword(Keyword::Instead) {
            self.expect_keyword(Keyword::Of)?;
            TriggerTiming::InsteadOf
        } else {
            bail!("expected BEFORE, AFTER, or INSTEAD OF");
        };

        let mut events = Vec::new();
        loop {
            if self.consume_keyword(Keyword::Insert) {
                events.push(TriggerEvent::Insert);
            } else if self.consume_keyword(Keyword::Update) {
                events.push(TriggerEvent::Update);
            } else if self.consume_keyword(Keyword::Delete) {
                events.push(TriggerEvent::Delete);
            } else if self.consume_keyword(Keyword::Truncate) {
                events.push(TriggerEvent::Truncate);
            } else {
                bail!("expected INSERT, UPDATE, DELETE, or TRUNCATE");
            }
            if !self.consume_keyword(Keyword::Or) {
                break;
            }
        }

        self.expect_keyword(Keyword::On)?;
        let table = self.expect_ident()?;

        let for_each_row = if self.consume_keyword(Keyword::For) {
            self.expect_keyword(Keyword::Each)?;
            self.consume_keyword(Keyword::Row);
            self.consume_keyword(Keyword::Statement);
            true
        } else {
            false
        };

        self.expect_keyword(Keyword::Execute)?;
        self.consume_keyword(Keyword::Function);
        self.consume_keyword(Keyword::Procedure);
        let function_name = self.expect_ident()?;
        self.expect_token(&Token::LParen)?;
        self.expect_token(&Token::RParen)?;

        Ok(CreateTriggerStmt {
            or_replace,
            name,
            timing,
            events: self.arena.alloc_slice_copy(&events),
            table,
            for_each_row,
            function_name,
        })
    }

    fn parse_create_type(&mut self) -> Result<CreateTypeStmt<'a>> {
        let name = self.expect_ident()?;
        let (schema, type_name) = if self.consume_token(&Token::Dot) {
            (Some(name), self.expect_ident()?)
        } else {
            (None, name)
        };

        self.expect_keyword(Keyword::As)?;

        let definition = if self.consume_keyword(Keyword::Enum) {
            self.expect_token(&Token::LParen)?;
            let mut values = Vec::new();
            loop {
                if let Token::String(s) = &self.current {
                    values.push(*s);
                    self.advance();
                } else {
                    bail!("expected string literal for enum value");
                }
                if !self.consume_token(&Token::Comma) {
                    break;
                }
            }
            self.expect_token(&Token::RParen)?;
            TypeDefinition::Enum(self.arena.alloc_slice_copy(&values))
        } else {
            self.expect_token(&Token::LParen)?;
            let mut fields = Vec::new();
            loop {
                let field_name = self.expect_ident()?;
                let data_type = self.parse_data_type()?;
                fields.push(TypeField {
                    name: field_name,
                    data_type,
                });
                if !self.consume_token(&Token::Comma) {
                    break;
                }
            }
            self.expect_token(&Token::RParen)?;
            TypeDefinition::Composite(self.arena.alloc_slice_copy(&fields))
        };

        Ok(CreateTypeStmt {
            schema,
            name: type_name,
            definition,
        })
    }

    fn parse_create_domain(&mut self) -> Result<CreateTypeStmt<'a>> {
        let name = self.expect_ident()?;
        let (schema, type_name) = if self.consume_token(&Token::Dot) {
            (Some(name), self.expect_ident()?)
        } else {
            (None, name)
        };

        self.expect_keyword(Keyword::As)?;
        let base_type = self.parse_data_type()?;

        Ok(CreateTypeStmt {
            schema,
            name: type_name,
            definition: TypeDefinition::Domain(base_type),
        })
    }

    fn parse_call(&mut self) -> Result<CallStmt<'a>> {
        self.expect_keyword(Keyword::Call)?;
        let name = self.expect_ident()?;
        let (schema, proc_name) = if self.consume_token(&Token::Dot) {
            (Some(name), self.expect_ident()?)
        } else {
            (None, name)
        };

        self.expect_token(&Token::LParen)?;
        let mut args: Vec<&'a Expr<'a>> = Vec::new();
        if !self.check_token(&Token::RParen) {
            loop {
                let expr = self.parse_expr(0)?;
                let expr_ref: &'a Expr<'a> = self.arena.alloc(expr);
                args.push(expr_ref);
                if !self.consume_token(&Token::Comma) {
                    break;
                }
            }
        }
        self.expect_token(&Token::RParen)?;

        Ok(CallStmt {
            schema,
            name: proc_name,
            args: self.arena.alloc_slice_copy(&args),
        })
    }

    fn parse_merge(&mut self) -> Result<MergeStmt<'a>> {
        self.expect_keyword(Keyword::Merge)?;
        self.expect_keyword(Keyword::Into)?;

        let target_table = self.expect_ident()?;
        let target_alias = if !self.check_keyword(Keyword::Using) {
            self.consume_keyword(Keyword::As);
            if let Token::Ident(alias) = &self.current {
                let a = *alias;
                self.advance();
                Some(a)
            } else {
                None
            }
        } else {
            None
        };

        self.expect_keyword(Keyword::Using)?;
        let source_name = self.expect_ident()?;
        let source_alias = if !self.check_keyword(Keyword::On) {
            self.consume_keyword(Keyword::As);
            if let Token::Ident(alias) = &self.current {
                let a = *alias;
                self.advance();
                Some(a)
            } else {
                None
            }
        } else {
            None
        };
        let source = MergeSource::Table {
            name: source_name,
            alias: source_alias,
        };

        self.expect_keyword(Keyword::On)?;
        let on_condition = self.parse_expr(0)?;
        let on_condition_ref: &'a Expr<'a> = self.arena.alloc(on_condition);

        let mut clauses = Vec::new();
        while self.consume_keyword(Keyword::When) {
            let is_not = self.consume_keyword(Keyword::Not);
            self.expect_keyword(Keyword::Matched)?;
            self.expect_keyword(Keyword::Then)?;

            if is_not {
                self.expect_keyword(Keyword::Insert)?;
                let columns = if self.consume_token(&Token::LParen) {
                    let cols = self.parse_ident_list()?;
                    self.expect_token(&Token::RParen)?;
                    Some(cols)
                } else {
                    None
                };
                self.expect_keyword(Keyword::Values)?;
                self.expect_token(&Token::LParen)?;
                let values = self.parse_expr_list()?;
                self.expect_token(&Token::RParen)?;
                let values_refs: Vec<&'a Expr<'a>> =
                    values.iter().map(|e| *e as &'a Expr<'a>).collect();
                clauses.push(MergeClause::NotMatchedInsert {
                    columns,
                    values: self.arena.alloc_slice_copy(&values_refs),
                });
            } else if self.consume_keyword(Keyword::Update) {
                self.expect_keyword(Keyword::Set)?;
                let assignments = self.parse_assignments()?;
                clauses.push(MergeClause::MatchedUpdate(assignments));
            } else if self.consume_keyword(Keyword::Delete) {
                clauses.push(MergeClause::MatchedDelete);
            } else {
                bail!("expected UPDATE, DELETE, or INSERT in MERGE clause");
            }
        }

        Ok(MergeStmt {
            target_table,
            target_alias,
            source,
            on_condition: on_condition_ref,
            clauses: self.arena.alloc_slice_copy(&clauses),
        })
    }

    fn parse_drop(&mut self) -> Result<DropStmt<'a>> {
        self.expect_keyword(Keyword::Drop)?;

        let object_type = if self.consume_keyword(Keyword::Table) {
            ObjectType::Table
        } else if self.consume_keyword(Keyword::Index) {
            ObjectType::Index
        } else if self.consume_keyword(Keyword::Schema) {
            ObjectType::Schema
        } else if self.consume_keyword(Keyword::View) {
            ObjectType::View
        } else if self.consume_keyword(Keyword::Sequence) {
            ObjectType::Sequence
        } else if self.consume_keyword(Keyword::Function) {
            ObjectType::Function
        } else if self.consume_keyword(Keyword::Procedure) {
            ObjectType::Procedure
        } else if self.consume_keyword(Keyword::Trigger) {
            ObjectType::Trigger
        } else {
            bail!("expected object type after DROP");
        };

        let if_exists = if self.consume_keyword(Keyword::If) {
            self.expect_keyword(Keyword::Exists)?;
            true
        } else {
            false
        };

        let mut names = Vec::new();
        loop {
            let name = self.expect_ident()?;
            let (schema, obj_name) = if self.consume_token(&Token::Dot) {
                (Some(name), self.expect_ident()?)
            } else {
                (None, name)
            };
            names.push(ObjectName {
                schema,
                name: obj_name,
            });
            if !self.consume_token(&Token::Comma) {
                break;
            }
        }

        let cascade = self.consume_keyword(Keyword::Cascade);
        self.consume_keyword(Keyword::Restrict);

        Ok(DropStmt {
            object_type,
            if_exists,
            names: self.arena.alloc_slice_copy(&names),
            cascade,
        })
    }

    fn parse_truncate(&mut self) -> Result<TruncateStmt<'a>> {
        self.expect_keyword(Keyword::Truncate)?;
        self.consume_keyword(Keyword::Table);

        let mut tables = Vec::new();
        loop {
            let table = self.parse_table_name()?;
            tables.push(table);
            if !self.consume_token(&Token::Comma) {
                break;
            }
        }

        let restart_identity = if self.consume_keyword(Keyword::Restart) {
            self.expect_keyword(Keyword::Identity)?;
            true
        } else {
            self.consume_keyword(Keyword::Continue);
            self.consume_keyword(Keyword::Identity);
            false
        };

        let cascade = self.consume_keyword(Keyword::Cascade);
        self.consume_keyword(Keyword::Restrict);

        Ok(TruncateStmt {
            tables: self.arena.alloc_slice_copy(&tables),
            restart_identity,
            cascade,
        })
    }

    fn parse_alter(&mut self) -> Result<Statement<'a>> {
        self.expect_keyword(Keyword::Alter)?;
        self.expect_keyword(Keyword::Table)?;

        let table = self.parse_table_name()?;
        let action = self.parse_alter_action()?;

        Ok(Statement::AlterTable(
            self.arena.alloc(AlterTableStmt { table, action }),
        ))
    }

    fn parse_alter_action(&mut self) -> Result<AlterTableAction<'a>> {
        if self.consume_keyword(Keyword::Add) {
            if self.consume_keyword(Keyword::Constraint) {
                let constraint = self.parse_table_constraint()?;
                Ok(AlterTableAction::AddConstraint(constraint))
            } else {
                self.consume_keyword(Keyword::Column);
                let column = self.parse_column_def()?;
                Ok(AlterTableAction::AddColumn(column))
            }
        } else if self.consume_keyword(Keyword::Drop) {
            if self.consume_keyword(Keyword::Constraint) {
                let if_exists = if self.consume_keyword(Keyword::If) {
                    self.expect_keyword(Keyword::Exists)?;
                    true
                } else {
                    false
                };
                let name = self.expect_ident()?;
                let cascade = self.consume_keyword(Keyword::Cascade);
                Ok(AlterTableAction::DropConstraint {
                    name,
                    if_exists,
                    cascade,
                })
            } else {
                self.consume_keyword(Keyword::Column);
                let if_exists = if self.consume_keyword(Keyword::If) {
                    self.expect_keyword(Keyword::Exists)?;
                    true
                } else {
                    false
                };
                let name = self.expect_ident()?;
                let cascade = self.consume_keyword(Keyword::Cascade);
                Ok(AlterTableAction::DropColumn {
                    name,
                    if_exists,
                    cascade,
                })
            }
        } else if self.consume_keyword(Keyword::Alter) {
            self.consume_keyword(Keyword::Column);
            let name = self.expect_ident()?;
            let action = if self.consume_keyword(Keyword::Set) {
                if self.consume_keyword(Keyword::Not) {
                    self.expect_keyword(Keyword::Null)?;
                    AlterColumnAction::SetNotNull
                } else if self.consume_keyword(Keyword::Default) {
                    let expr = self.parse_expr(0)?;
                    AlterColumnAction::SetDefault(self.arena.alloc(expr))
                } else if self.consume_keyword(Keyword::Data) {
                    self.expect_keyword(Keyword::Type)?;
                    let data_type = self.parse_data_type()?;
                    AlterColumnAction::SetDataType(data_type)
                } else {
                    bail!("expected NOT NULL, DEFAULT, or DATA TYPE");
                }
            } else if self.consume_keyword(Keyword::Drop) {
                if self.consume_keyword(Keyword::Not) {
                    self.expect_keyword(Keyword::Null)?;
                    AlterColumnAction::DropNotNull
                } else {
                    self.expect_keyword(Keyword::Default)?;
                    AlterColumnAction::DropDefault
                }
            } else if self.consume_keyword(Keyword::Type) {
                let data_type = self.parse_data_type()?;
                AlterColumnAction::SetDataType(data_type)
            } else {
                bail!("expected SET or DROP");
            };
            Ok(AlterTableAction::AlterColumn { name, action })
        } else if self.consume_keyword(Keyword::Rename) {
            if self.consume_keyword(Keyword::Column) {
                let old_name = self.expect_ident()?;
                self.expect_keyword(Keyword::To)?;
                let new_name = self.expect_ident()?;
                Ok(AlterTableAction::RenameColumn { old_name, new_name })
            } else {
                self.expect_keyword(Keyword::To)?;
                let new_name = self.expect_ident()?;
                Ok(AlterTableAction::RenameTable(new_name))
            }
        } else {
            bail!("expected ADD, DROP, ALTER, or RENAME");
        }
    }

    fn parse_data_type(&mut self) -> Result<DataType<'a>> {
        match self.peek().clone() {
            Token::Keyword(Keyword::Integer) | Token::Keyword(Keyword::Int) => {
                self.advance();
                Ok(DataType::Integer)
            }
            Token::Keyword(Keyword::Bigint) => {
                self.advance();
                Ok(DataType::BigInt)
            }
            Token::Keyword(Keyword::Smallint) => {
                self.advance();
                Ok(DataType::SmallInt)
            }
            Token::Keyword(Keyword::Tinyint) => {
                self.advance();
                Ok(DataType::TinyInt)
            }
            Token::Keyword(Keyword::Real) | Token::Keyword(Keyword::Float) => {
                self.advance();
                Ok(DataType::Real)
            }
            Token::Keyword(Keyword::Double) => {
                self.advance();
                self.consume_keyword(Keyword::Precision);
                Ok(DataType::DoublePrecision)
            }
            Token::Keyword(Keyword::Decimal) => {
                self.advance();
                let (p, s) = self.parse_precision_scale()?;
                Ok(DataType::Decimal(p, s))
            }
            Token::Keyword(Keyword::Numeric) => {
                self.advance();
                let (p, s) = self.parse_precision_scale()?;
                Ok(DataType::Numeric(p, s))
            }
            Token::Keyword(Keyword::Varchar) | Token::Keyword(Keyword::Character) => {
                self.advance();
                self.consume_keyword(Keyword::Varying);
                let len = self.parse_type_length()?;
                Ok(DataType::Varchar(len))
            }
            Token::Keyword(Keyword::Char) => {
                self.advance();
                let len = self.parse_type_length()?;
                Ok(DataType::Char(len))
            }
            Token::Keyword(Keyword::Text) => {
                self.advance();
                Ok(DataType::Text)
            }
            Token::Keyword(Keyword::Blob) => {
                self.advance();
                Ok(DataType::Blob)
            }
            Token::Keyword(Keyword::Boolean) | Token::Keyword(Keyword::Bool) => {
                self.advance();
                Ok(DataType::Boolean)
            }
            Token::Keyword(Keyword::Date) => {
                self.advance();
                Ok(DataType::Date)
            }
            Token::Keyword(Keyword::Time) => {
                self.advance();
                Ok(DataType::Time)
            }
            Token::Keyword(Keyword::Timestamp) => {
                self.advance();
                if self.consume_keyword(Keyword::With) {
                    self.expect_keyword(Keyword::Time)?;
                    self.expect_keyword(Keyword::Zone)?;
                    Ok(DataType::TimestampTz)
                } else {
                    self.consume_keyword(Keyword::Without);
                    self.consume_keyword(Keyword::Time);
                    self.consume_keyword(Keyword::Zone);
                    Ok(DataType::Timestamp)
                }
            }
            Token::Keyword(Keyword::Timestamptz) => {
                self.advance();
                Ok(DataType::TimestampTz)
            }
            Token::Keyword(Keyword::Interval) => {
                self.advance();
                Ok(DataType::Interval)
            }
            Token::Keyword(Keyword::Uuid) => {
                self.advance();
                Ok(DataType::Uuid)
            }
            Token::Keyword(Keyword::Json) => {
                self.advance();
                Ok(DataType::Json)
            }
            Token::Keyword(Keyword::Jsonb) => {
                self.advance();
                Ok(DataType::Jsonb)
            }
            Token::Keyword(Keyword::Vector) => {
                self.advance();
                let dim = self.parse_type_length()?;
                Ok(DataType::Vector(dim))
            }
            Token::Keyword(Keyword::Point) => {
                self.advance();
                Ok(DataType::Point)
            }
            Token::Keyword(Keyword::Box) => {
                self.advance();
                Ok(DataType::Box)
            }
            Token::Keyword(Keyword::Circle) => {
                self.advance();
                Ok(DataType::Circle)
            }
            Token::Keyword(Keyword::MacAddr) => {
                self.advance();
                Ok(DataType::MacAddr)
            }
            Token::Keyword(Keyword::Inet) => {
                self.advance();
                Ok(DataType::Inet)
            }
            Token::Keyword(Keyword::Int4Range) => {
                self.advance();
                Ok(DataType::Int4Range)
            }
            Token::Keyword(Keyword::Int8Range) => {
                self.advance();
                Ok(DataType::Int8Range)
            }
            Token::Keyword(Keyword::DateRange) => {
                self.advance();
                Ok(DataType::DateRange)
            }
            Token::Keyword(Keyword::TsRange) => {
                self.advance();
                Ok(DataType::TsRange)
            }
            Token::Ident(name) => {
                self.advance();
                Ok(DataType::Custom(name))
            }
            _ => bail!("expected data type, found {:?}", self.current),
        }
    }

    fn parse_precision_scale(&mut self) -> Result<(Option<u32>, Option<u32>)> {
        if self.consume_token(&Token::LParen) {
            let precision: u32 = match self.advance() {
                Token::Integer(s) => s.parse().unwrap_or(0),
                _ => bail!("expected precision"),
            };
            let scale = if self.consume_token(&Token::Comma) {
                match self.advance() {
                    Token::Integer(s) => Some(s.parse().unwrap_or(0)),
                    _ => bail!("expected scale"),
                }
            } else {
                None
            };
            self.expect_token(&Token::RParen)?;
            Ok((Some(precision), scale))
        } else {
            Ok((None, None))
        }
    }

    fn parse_type_length(&mut self) -> Result<Option<u32>> {
        if self.consume_token(&Token::LParen) {
            let len: u32 = match self.advance() {
                Token::Integer(s) => s.parse().unwrap_or(0),
                _ => bail!("expected length"),
            };
            self.expect_token(&Token::RParen)?;
            Ok(Some(len))
        } else {
            Ok(None)
        }
    }

    fn parse_begin(&mut self) -> Result<BeginStmt> {
        self.expect_keyword(Keyword::Begin)?;
        self.consume_keyword(Keyword::Work);
        self.consume_keyword(Keyword::Transaction);

        let mut isolation_level = None;
        let mut read_only = None;

        while self.check_keyword(Keyword::Isolation) || self.check_keyword(Keyword::Read) {
            if self.consume_keyword(Keyword::Isolation) {
                self.expect_keyword(Keyword::Level)?;
                isolation_level = Some(if self.consume_keyword(Keyword::Read) {
                    if self.consume_keyword(Keyword::Uncommitted) {
                        IsolationLevel::ReadUncommitted
                    } else {
                        self.expect_keyword(Keyword::Committed)?;
                        IsolationLevel::ReadCommitted
                    }
                } else if self.consume_keyword(Keyword::Repeatable) {
                    self.expect_keyword(Keyword::Read)?;
                    IsolationLevel::RepeatableRead
                } else {
                    self.expect_keyword(Keyword::Serializable)?;
                    IsolationLevel::Serializable
                });
            } else if self.consume_keyword(Keyword::Read) {
                if self.consume_keyword(Keyword::Only) {
                    read_only = Some(true);
                } else {
                    self.expect_keyword(Keyword::Write)?;
                    read_only = Some(false);
                }
            }
            self.consume_token(&Token::Comma);
        }

        Ok(BeginStmt {
            isolation_level,
            read_only,
        })
    }

    fn parse_rollback(&mut self) -> Result<RollbackStmt<'a>> {
        self.expect_keyword(Keyword::Rollback)?;
        self.consume_keyword(Keyword::Work);
        self.consume_keyword(Keyword::Transaction);

        let savepoint = if self.consume_keyword(Keyword::To) {
            self.consume_keyword(Keyword::Savepoint);
            Some(self.expect_ident()?)
        } else {
            None
        };

        Ok(RollbackStmt { savepoint })
    }

    fn parse_savepoint(&mut self) -> Result<SavepointStmt<'a>> {
        self.expect_keyword(Keyword::Savepoint)?;
        let name = self.expect_ident()?;
        Ok(SavepointStmt { name })
    }

    fn parse_release(&mut self) -> Result<ReleaseStmt<'a>> {
        self.expect_keyword(Keyword::Release)?;
        self.consume_keyword(Keyword::Savepoint);
        let name = self.expect_ident()?;
        Ok(ReleaseStmt { name })
    }

    fn parse_explain(&mut self) -> Result<ExplainStmt<'a>> {
        self.expect_keyword(Keyword::Explain)?;

        let mut analyze = false;
        let mut verbose = false;
        let mut format = ExplainFormat::Text;

        if self.consume_token(&Token::LParen) {
            loop {
                if self.consume_keyword(Keyword::Analyze) {
                    analyze = true;
                } else if self.consume_keyword(Keyword::Verbose) {
                    verbose = true;
                } else if self.consume_keyword(Keyword::Format) {
                    if self.consume_keyword(Keyword::Json) {
                        format = ExplainFormat::Json;
                    } else if self.consume_keyword(Keyword::Text) {
                        format = ExplainFormat::Text;
                    }
                } else {
                    break;
                }
                if !self.consume_token(&Token::Comma) {
                    break;
                }
            }
            self.expect_token(&Token::RParen)?;
        } else {
            analyze = self.consume_keyword(Keyword::Analyze);
            verbose = self.consume_keyword(Keyword::Verbose);
        }

        let statement = self.parse_statement()?;

        Ok(ExplainStmt {
            analyze,
            verbose,
            format,
            statement: self.arena.alloc(statement),
        })
    }

    fn parse_set(&mut self) -> Result<SetStmt<'a>> {
        self.expect_keyword(Keyword::Set)?;

        let scope = if self.consume_keyword(Keyword::Session) {
            SetScope::Session
        } else if self.consume_keyword(Keyword::Local) {
            SetScope::Local
        } else if self.consume_keyword(Keyword::Global) {
            SetScope::Global
        } else {
            SetScope::Session
        };

        let name = self.expect_ident()?;

        if !self.consume_token(&Token::Eq) {
            self.expect_keyword(Keyword::To)?;
        }

        let mut values = Vec::new();
        loop {
            let expr = self.parse_expr(0)?;
            values.push(self.arena.alloc(expr) as &Expr<'a>);
            if !self.consume_token(&Token::Comma) {
                break;
            }
        }

        let value = self.arena.alloc_slice_copy(&values);

        Ok(SetStmt { scope, name, value })
    }

    fn parse_show(&mut self) -> Result<ShowStmt<'a>> {
        self.expect_keyword(Keyword::Show)?;

        if self.consume_keyword(Keyword::All) {
            Ok(ShowStmt {
                name: None,
                all: true,
            })
        } else {
            let name = self.expect_ident()?;
            Ok(ShowStmt {
                name: Some(name),
                all: false,
            })
        }
    }

    fn parse_reset(&mut self) -> Result<ResetStmt<'a>> {
        self.expect_keyword(Keyword::Reset)?;

        if self.consume_keyword(Keyword::All) {
            Ok(ResetStmt {
                name: None,
                all: true,
            })
        } else {
            let name = self.expect_ident()?;
            Ok(ResetStmt {
                name: Some(name),
                all: false,
            })
        }
    }

    fn parse_pragma(&mut self) -> Result<PragmaStmt<'a>> {
        self.expect_keyword(Keyword::Pragma)?;

        let name = self.expect_ident()?;

        let value = if self.consume_token(&Token::Eq) {
            Some(self.expect_ident()?)
        } else {
            None
        };

        Ok(PragmaStmt { name, value })
    }

    fn parse_grant(&mut self) -> Result<GrantStmt<'a>> {
        self.expect_keyword(Keyword::Grant)?;

        let privileges = self.parse_privileges()?;

        self.expect_keyword(Keyword::On)?;

        let object_type = self.parse_object_type();

        let object_name = if !self.check_keyword(Keyword::To) {
            Some(self.parse_table_name()?)
        } else {
            None
        };

        self.expect_keyword(Keyword::To)?;

        let grantees = self.parse_grantee_list()?;

        let with_grant_option = if self.consume_keyword(Keyword::With) {
            self.expect_keyword(Keyword::Grant)?;
            self.expect_keyword(Keyword::Option)?;
            true
        } else {
            false
        };

        Ok(GrantStmt {
            privileges,
            object_type,
            object_name,
            grantees,
            with_grant_option,
        })
    }

    fn parse_revoke_stmt(&mut self) -> Result<RevokeStmt<'a>> {
        self.expect_keyword(Keyword::Revoke)?;

        let privileges = self.parse_privileges()?;

        self.expect_keyword(Keyword::On)?;

        let object_type = self.parse_object_type();

        let object_name = if !self.check_keyword(Keyword::From) {
            Some(self.parse_table_name()?)
        } else {
            None
        };

        self.expect_keyword(Keyword::From)?;

        let grantees = self.parse_grantee_list()?;

        let cascade = self.consume_keyword(Keyword::Cascade);
        self.consume_keyword(Keyword::Restrict);

        Ok(RevokeStmt {
            privileges,
            object_type,
            object_name,
            grantees,
            cascade,
        })
    }

    fn parse_privileges(&mut self) -> Result<&'a [Privilege]> {
        let mut privileges = Vec::new();

        if self.consume_keyword(Keyword::All) {
            self.consume_keyword(Keyword::Privileges);
            privileges.push(Privilege::All);
        } else {
            loop {
                let priv_kw = self.parse_privilege()?;
                privileges.push(priv_kw);
                if !self.consume_token(&Token::Comma) {
                    break;
                }
            }
        }

        Ok(self.arena.alloc_slice_copy(&privileges))
    }

    fn parse_privilege(&mut self) -> Result<Privilege> {
        if self.consume_keyword(Keyword::Select) {
            Ok(Privilege::Select)
        } else if self.consume_keyword(Keyword::Insert) {
            Ok(Privilege::Insert)
        } else if self.consume_keyword(Keyword::Update) {
            Ok(Privilege::Update)
        } else if self.consume_keyword(Keyword::Delete) {
            Ok(Privilege::Delete)
        } else if self.consume_keyword(Keyword::Truncate) {
            Ok(Privilege::Truncate)
        } else if self.consume_keyword(Keyword::References) {
            Ok(Privilege::References)
        } else if self.consume_keyword(Keyword::Trigger) {
            Ok(Privilege::Trigger)
        } else if self.consume_keyword(Keyword::Create) {
            Ok(Privilege::Create)
        } else if self.consume_keyword(Keyword::Execute) {
            Ok(Privilege::Execute)
        } else if self.consume_keyword(Keyword::Usage) {
            Ok(Privilege::Usage)
        } else if self.consume_keyword(Keyword::All) {
            self.consume_keyword(Keyword::Privileges);
            Ok(Privilege::All)
        } else {
            bail!("expected privilege keyword, found {:?}", self.current)
        }
    }

    fn parse_object_type(&mut self) -> Option<ObjectType> {
        if self.consume_keyword(Keyword::Table) {
            Some(ObjectType::Table)
        } else if self.consume_keyword(Keyword::Schema) {
            Some(ObjectType::Schema)
        } else if self.consume_keyword(Keyword::Database) {
            Some(ObjectType::Database)
        } else if self.consume_keyword(Keyword::Sequence) {
            Some(ObjectType::Sequence)
        } else if self.consume_keyword(Keyword::Function) {
            Some(ObjectType::Function)
        } else if self.consume_keyword(Keyword::Procedure) {
            Some(ObjectType::Procedure)
        } else if self.consume_keyword(Keyword::Type) {
            Some(ObjectType::Type)
        } else if self.consume_keyword(Keyword::Domain) {
            Some(ObjectType::Domain)
        } else if self.consume_keyword(Keyword::View) {
            Some(ObjectType::View)
        } else {
            None
        }
    }

    fn parse_grantee_list(&mut self) -> Result<&'a [&'a str]> {
        let mut grantees = Vec::new();

        loop {
            let grantee = self.expect_ident()?;
            grantees.push(grantee);
            if !self.consume_token(&Token::Comma) {
                break;
            }
        }

        Ok(self.arena.alloc_slice_copy(&grantees))
    }
}
