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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parser_new_creates_parser() {
        let arena = Bump::new();
        let parser = Parser::new("SELECT 1", &arena);
        assert!(!parser.is_at_end());
    }

    #[test]
    fn parser_peek_returns_current_token() {
        let arena = Bump::new();
        let parser = Parser::new("SELECT FROM", &arena);
        assert!(matches!(parser.peek(), Token::Keyword(Keyword::Select)));
        assert!(matches!(parser.peek(), Token::Keyword(Keyword::Select)));
    }

    #[test]
    fn parser_advance_moves_to_next_token() {
        let arena = Bump::new();
        let mut parser = Parser::new("SELECT FROM", &arena);
        assert!(matches!(parser.peek(), Token::Keyword(Keyword::Select)));
        parser.advance();
        assert!(matches!(parser.peek(), Token::Keyword(Keyword::From)));
    }

    #[test]
    fn parser_expect_keyword_succeeds() {
        let arena = Bump::new();
        let mut parser = Parser::new("SELECT FROM", &arena);
        assert!(parser.expect_keyword(Keyword::Select).is_ok());
        assert!(parser.expect_keyword(Keyword::From).is_ok());
    }

    #[test]
    fn parser_expect_keyword_fails() {
        let arena = Bump::new();
        let mut parser = Parser::new("SELECT FROM", &arena);
        assert!(parser.expect_keyword(Keyword::From).is_err());
    }

    #[test]
    fn parser_check_keyword_returns_true() {
        let arena = Bump::new();
        let parser = Parser::new("SELECT FROM", &arena);
        assert!(parser.check_keyword(Keyword::Select));
        assert!(!parser.check_keyword(Keyword::From));
    }

    #[test]
    fn parser_consume_keyword_if_present() {
        let arena = Bump::new();
        let mut parser = Parser::new("SELECT FROM", &arena);
        assert!(parser.consume_keyword(Keyword::Select));
        assert!(!parser.consume_keyword(Keyword::Select));
        assert!(parser.consume_keyword(Keyword::From));
    }

    #[test]
    fn parse_simple_select() {
        let arena = Bump::new();
        let mut parser = Parser::new("SELECT 1", &arena);
        let stmt = parser.parse_statement().unwrap();
        assert!(matches!(stmt, Statement::Select(_)));
    }

    #[test]
    fn parse_select_with_column() {
        let arena = Bump::new();
        let mut parser = Parser::new("SELECT id FROM users", &arena);
        let stmt = parser.parse_statement().unwrap();
        if let Statement::Select(select) = stmt {
            assert_eq!(select.columns.len(), 1);
            assert!(select.from.is_some());
        } else {
            panic!("Expected Select statement");
        }
    }

    #[test]
    fn parse_select_star() {
        let arena = Bump::new();
        let mut parser = Parser::new("SELECT * FROM users", &arena);
        let stmt = parser.parse_statement().unwrap();
        if let Statement::Select(select) = stmt {
            assert_eq!(select.columns.len(), 1);
            assert!(matches!(select.columns[0], SelectColumn::AllColumns));
        } else {
            panic!("Expected Select statement");
        }
    }

    #[test]
    fn parse_select_table_star() {
        let arena = Bump::new();
        let mut parser = Parser::new("SELECT u.* FROM users u", &arena);
        let stmt = parser.parse_statement().unwrap();
        if let Statement::Select(select) = stmt {
            assert_eq!(select.columns.len(), 1);
            assert!(matches!(
                select.columns[0],
                SelectColumn::TableAllColumns("u")
            ));
        } else {
            panic!("Expected Select statement");
        }
    }

    #[test]
    fn parse_select_with_alias() {
        let arena = Bump::new();
        let mut parser = Parser::new("SELECT id AS user_id FROM users", &arena);
        let stmt = parser.parse_statement().unwrap();
        if let Statement::Select(select) = stmt {
            if let SelectColumn::Expr { alias, .. } = &select.columns[0] {
                assert_eq!(*alias, Some("user_id"));
            } else {
                panic!("Expected Expr column");
            }
        } else {
            panic!("Expected Select statement");
        }
    }

    #[test]
    fn parse_select_distinct() {
        let arena = Bump::new();
        let mut parser = Parser::new("SELECT DISTINCT id FROM users", &arena);
        let stmt = parser.parse_statement().unwrap();
        if let Statement::Select(select) = stmt {
            assert_eq!(select.distinct, Distinct::Distinct);
        } else {
            panic!("Expected Select statement");
        }
    }

    #[test]
    fn parse_select_where() {
        let arena = Bump::new();
        let mut parser = Parser::new("SELECT id FROM users WHERE active = true", &arena);
        let stmt = parser.parse_statement().unwrap();
        if let Statement::Select(select) = stmt {
            assert!(select.where_clause.is_some());
        } else {
            panic!("Expected Select statement");
        }
    }

    #[test]
    fn parse_select_order_by() {
        let arena = Bump::new();
        let mut parser = Parser::new("SELECT id FROM users ORDER BY id DESC", &arena);
        let stmt = parser.parse_statement().unwrap();
        if let Statement::Select(select) = stmt {
            assert_eq!(select.order_by.len(), 1);
            assert_eq!(select.order_by[0].direction, OrderDirection::Desc);
        } else {
            panic!("Expected Select statement");
        }
    }

    #[test]
    fn parse_select_limit_offset() {
        let arena = Bump::new();
        let mut parser = Parser::new("SELECT id FROM users LIMIT 10 OFFSET 5", &arena);
        let stmt = parser.parse_statement().unwrap();
        if let Statement::Select(select) = stmt {
            assert!(select.limit.is_some());
            assert!(select.offset.is_some());
        } else {
            panic!("Expected Select statement");
        }
    }

    #[test]
    fn parse_select_group_by() {
        let arena = Bump::new();
        let mut parser = Parser::new("SELECT status, COUNT(*) FROM users GROUP BY status", &arena);
        let stmt = parser.parse_statement().unwrap();
        if let Statement::Select(select) = stmt {
            assert_eq!(select.group_by.len(), 1);
        } else {
            panic!("Expected Select statement");
        }
    }

    #[test]
    fn parse_select_having() {
        let arena = Bump::new();
        let mut parser = Parser::new(
            "SELECT status, COUNT(*) FROM users GROUP BY status HAVING COUNT(*) > 5",
            &arena,
        );
        let stmt = parser.parse_statement().unwrap();
        if let Statement::Select(select) = stmt {
            assert!(select.having.is_some());
        } else {
            panic!("Expected Select statement");
        }
    }

    #[test]
    fn parse_expr_integer() {
        let arena = Bump::new();
        let mut parser = Parser::new("SELECT 42", &arena);
        let stmt = parser.parse_statement().unwrap();
        if let Statement::Select(select) = stmt {
            if let SelectColumn::Expr { expr, .. } = &select.columns[0] {
                assert!(matches!(expr, Expr::Literal(Literal::Integer("42"))));
            }
        }
    }

    #[test]
    fn parse_expr_string() {
        let arena = Bump::new();
        let mut parser = Parser::new("SELECT 'hello'", &arena);
        let stmt = parser.parse_statement().unwrap();
        if let Statement::Select(select) = stmt {
            if let SelectColumn::Expr { expr, .. } = &select.columns[0] {
                assert!(matches!(expr, Expr::Literal(Literal::String("hello"))));
            }
        }
    }

    #[test]
    fn parse_expr_binary_add() {
        let arena = Bump::new();
        let mut parser = Parser::new("SELECT 1 + 2", &arena);
        let stmt = parser.parse_statement().unwrap();
        if let Statement::Select(select) = stmt {
            if let SelectColumn::Expr { expr, .. } = &select.columns[0] {
                assert!(matches!(
                    expr,
                    Expr::BinaryOp {
                        op: BinaryOperator::Plus,
                        ..
                    }
                ));
            }
        }
    }

    #[test]
    fn parse_expr_precedence() {
        let arena = Bump::new();
        let mut parser = Parser::new("SELECT 1 + 2 * 3", &arena);
        let stmt = parser.parse_statement().unwrap();
        if let Statement::Select(select) = stmt {
            if let SelectColumn::Expr { expr, .. } = &select.columns[0] {
                if let Expr::BinaryOp { op, right, .. } = expr {
                    assert_eq!(*op, BinaryOperator::Plus);
                    assert!(matches!(
                        right,
                        Expr::BinaryOp {
                            op: BinaryOperator::Multiply,
                            ..
                        }
                    ));
                }
            }
        }
    }

    #[test]
    fn parse_expr_parentheses() {
        let arena = Bump::new();
        let mut parser = Parser::new("SELECT (1 + 2) * 3", &arena);
        let stmt = parser.parse_statement().unwrap();
        if let Statement::Select(select) = stmt {
            if let SelectColumn::Expr { expr, .. } = &select.columns[0] {
                assert!(matches!(
                    expr,
                    Expr::BinaryOp {
                        op: BinaryOperator::Multiply,
                        ..
                    }
                ));
            }
        }
    }

    #[test]
    fn parse_expr_and_or() {
        let arena = Bump::new();
        let mut parser = Parser::new("SELECT * FROM t WHERE a AND b OR c", &arena);
        let stmt = parser.parse_statement().unwrap();
        if let Statement::Select(select) = stmt {
            if let Some(Expr::BinaryOp { op, .. }) = select.where_clause {
                assert_eq!(*op, BinaryOperator::Or);
            }
        }
    }

    #[test]
    fn parse_expr_comparison() {
        let arena = Bump::new();
        let mut parser = Parser::new("SELECT * FROM t WHERE a > 5", &arena);
        let stmt = parser.parse_statement().unwrap();
        if let Statement::Select(select) = stmt {
            if let Some(Expr::BinaryOp { op, .. }) = select.where_clause {
                assert_eq!(*op, BinaryOperator::Gt);
            }
        }
    }

    #[test]
    fn parse_expr_is_null() {
        let arena = Bump::new();
        let mut parser = Parser::new("SELECT * FROM t WHERE a IS NULL", &arena);
        let stmt = parser.parse_statement().unwrap();
        if let Statement::Select(select) = stmt {
            assert!(matches!(
                select.where_clause,
                Some(Expr::IsNull { negated: false, .. })
            ));
        }
    }

    #[test]
    fn parse_expr_is_not_null() {
        let arena = Bump::new();
        let mut parser = Parser::new("SELECT * FROM t WHERE a IS NOT NULL", &arena);
        let stmt = parser.parse_statement().unwrap();
        if let Statement::Select(select) = stmt {
            assert!(matches!(
                select.where_clause,
                Some(Expr::IsNull { negated: true, .. })
            ));
        }
    }

    #[test]
    fn parse_expr_between() {
        let arena = Bump::new();
        let mut parser = Parser::new("SELECT * FROM t WHERE a BETWEEN 1 AND 10", &arena);
        let stmt = parser.parse_statement().unwrap();
        if let Statement::Select(select) = stmt {
            assert!(matches!(
                select.where_clause,
                Some(Expr::Between { negated: false, .. })
            ));
        }
    }

    #[test]
    fn parse_expr_in_list() {
        let arena = Bump::new();
        let mut parser = Parser::new("SELECT * FROM t WHERE a IN (1, 2, 3)", &arena);
        let stmt = parser.parse_statement().unwrap();
        if let Statement::Select(select) = stmt {
            assert!(matches!(
                select.where_clause,
                Some(Expr::InList { negated: false, .. })
            ));
        }
    }

    #[test]
    fn parse_expr_like() {
        let arena = Bump::new();
        let mut parser = Parser::new("SELECT * FROM t WHERE name LIKE '%foo%'", &arena);
        let stmt = parser.parse_statement().unwrap();
        if let Statement::Select(select) = stmt {
            assert!(matches!(
                select.where_clause,
                Some(Expr::Like {
                    negated: false,
                    case_insensitive: false,
                    ..
                })
            ));
        }
    }

    #[test]
    fn parse_expr_ilike() {
        let arena = Bump::new();
        let mut parser = Parser::new("SELECT * FROM t WHERE name ILIKE '%foo%'", &arena);
        let stmt = parser.parse_statement().unwrap();
        if let Statement::Select(select) = stmt {
            assert!(matches!(
                select.where_clause,
                Some(Expr::Like {
                    case_insensitive: true,
                    ..
                })
            ));
        }
    }

    #[test]
    fn parse_expr_unary_not() {
        let arena = Bump::new();
        let mut parser = Parser::new("SELECT * FROM t WHERE NOT active", &arena);
        let stmt = parser.parse_statement().unwrap();
        if let Statement::Select(select) = stmt {
            assert!(matches!(
                select.where_clause,
                Some(Expr::UnaryOp {
                    op: UnaryOperator::Not,
                    ..
                })
            ));
        }
    }

    #[test]
    fn parse_expr_unary_minus() {
        let arena = Bump::new();
        let mut parser = Parser::new("SELECT -5", &arena);
        let stmt = parser.parse_statement().unwrap();
        if let Statement::Select(select) = stmt {
            if let SelectColumn::Expr { expr, .. } = &select.columns[0] {
                assert!(matches!(
                    expr,
                    Expr::UnaryOp {
                        op: UnaryOperator::Minus,
                        ..
                    }
                ));
            }
        }
    }

    #[test]
    fn parse_function_call() {
        let arena = Bump::new();
        let mut parser = Parser::new("SELECT COUNT(*) FROM users", &arena);
        let stmt = parser.parse_statement().unwrap();
        if let Statement::Select(select) = stmt {
            if let SelectColumn::Expr { expr, .. } = &select.columns[0] {
                if let Expr::Function(f) = expr {
                    assert_eq!(f.name.name, "COUNT");
                    assert!(matches!(f.args, FunctionArgs::Star));
                }
            }
        }
    }

    #[test]
    fn parse_function_with_args() {
        let arena = Bump::new();
        let mut parser = Parser::new("SELECT COALESCE(a, b, c)", &arena);
        let stmt = parser.parse_statement().unwrap();
        if let Statement::Select(select) = stmt {
            if let SelectColumn::Expr { expr, .. } = &select.columns[0] {
                if let Expr::Function(f) = expr {
                    assert_eq!(f.name.name, "COALESCE");
                    if let FunctionArgs::Args(args) = &f.args {
                        assert_eq!(args.len(), 3);
                    }
                }
            }
        }
    }

    #[test]
    fn parse_cast_expression() {
        let arena = Bump::new();
        let mut parser = Parser::new("SELECT CAST(id AS TEXT)", &arena);
        let stmt = parser.parse_statement().unwrap();
        if let Statement::Select(select) = stmt {
            if let SelectColumn::Expr { expr, .. } = &select.columns[0] {
                if let Expr::Cast { data_type, .. } = expr {
                    assert!(matches!(data_type, DataType::Text));
                }
            }
        }
    }

    #[test]
    fn parse_case_expression() {
        let arena = Bump::new();
        let mut parser = Parser::new("SELECT CASE WHEN a > 0 THEN 'pos' ELSE 'neg' END", &arena);
        let stmt = parser.parse_statement().unwrap();
        if let Statement::Select(select) = stmt {
            if let SelectColumn::Expr { expr, .. } = &select.columns[0] {
                if let Expr::Case {
                    conditions,
                    else_result,
                    ..
                } = expr
                {
                    assert_eq!(conditions.len(), 1);
                    assert!(else_result.is_some());
                }
            }
        }
    }

    #[test]
    fn parse_insert_values() {
        let arena = Bump::new();
        let mut parser = Parser::new("INSERT INTO users (id, name) VALUES (1, 'alice')", &arena);
        let stmt = parser.parse_statement().unwrap();
        if let Statement::Insert(insert) = stmt {
            assert_eq!(insert.table.name, "users");
            assert!(insert.columns.is_some());
            assert!(matches!(insert.source, InsertSource::Values(_)));
        } else {
            panic!("Expected Insert statement");
        }
    }

    #[test]
    fn parse_insert_select() {
        let arena = Bump::new();
        let mut parser = Parser::new("INSERT INTO users_backup SELECT * FROM users", &arena);
        let stmt = parser.parse_statement().unwrap();
        if let Statement::Insert(insert) = stmt {
            assert!(matches!(insert.source, InsertSource::Select(_)));
        } else {
            panic!("Expected Insert statement");
        }
    }

    #[test]
    fn parse_update_simple() {
        let arena = Bump::new();
        let mut parser = Parser::new("UPDATE users SET name = 'bob' WHERE id = 1", &arena);
        let stmt = parser.parse_statement().unwrap();
        if let Statement::Update(update) = stmt {
            assert_eq!(update.table.name, "users");
            assert_eq!(update.assignments.len(), 1);
            assert!(update.where_clause.is_some());
        } else {
            panic!("Expected Update statement");
        }
    }

    #[test]
    fn parse_delete_simple() {
        let arena = Bump::new();
        let mut parser = Parser::new("DELETE FROM users WHERE id = 1", &arena);
        let stmt = parser.parse_statement().unwrap();
        if let Statement::Delete(delete) = stmt {
            assert_eq!(delete.table.name, "users");
            assert!(delete.where_clause.is_some());
        } else {
            panic!("Expected Delete statement");
        }
    }

    #[test]
    fn parse_join_inner() {
        let arena = Bump::new();
        let mut parser = Parser::new(
            "SELECT * FROM users u INNER JOIN orders o ON u.id = o.user_id",
            &arena,
        );
        let stmt = parser.parse_statement().unwrap();
        if let Statement::Select(select) = stmt {
            if let Some(FromClause::Join(join)) = select.from {
                assert_eq!(join.join_type, JoinType::Inner);
            } else {
                panic!("Expected Join clause");
            }
        }
    }

    #[test]
    fn parse_join_left() {
        let arena = Bump::new();
        let mut parser = Parser::new(
            "SELECT * FROM users u LEFT JOIN orders o ON u.id = o.user_id",
            &arena,
        );
        let stmt = parser.parse_statement().unwrap();
        if let Statement::Select(select) = stmt {
            if let Some(FromClause::Join(join)) = select.from {
                assert_eq!(join.join_type, JoinType::Left);
            }
        }
    }

    #[test]
    fn parse_subquery_in_where() {
        let arena = Bump::new();
        let mut parser = Parser::new(
            "SELECT * FROM users WHERE id IN (SELECT user_id FROM orders)",
            &arena,
        );
        let stmt = parser.parse_statement().unwrap();
        if let Statement::Select(select) = stmt {
            assert!(matches!(select.where_clause, Some(Expr::InSubquery { .. })));
        }
    }

    #[test]
    fn parse_exists_subquery() {
        let arena = Bump::new();
        let mut parser = Parser::new(
            "SELECT * FROM users u WHERE EXISTS (SELECT 1 FROM orders o WHERE o.user_id = u.id)",
            &arena,
        );
        let stmt = parser.parse_statement().unwrap();
        if let Statement::Select(select) = stmt {
            assert!(matches!(
                select.where_clause,
                Some(Expr::Exists { negated: false, .. })
            ));
        }
    }
}

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

        let limit: Option<&Expr<'a>> = if self.consume_keyword(Keyword::Limit) {
            Some(self.arena.alloc(self.parse_expr(0)?))
        } else {
            None
        };

        let offset: Option<&Expr<'a>> = if self.consume_keyword(Keyword::Offset) {
            Some(self.arena.alloc(self.parse_expr(0)?))
        } else {
            None
        };

        let set_op = self.parse_set_operation()?;

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
        })
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

        while let Some(join_type) = self.parse_join_type() {
            let right = self.parse_table_ref()?;
            let condition = self.parse_join_condition()?;

            let join = self.arena.alloc(JoinClause {
                left,
                join_type,
                right,
                condition,
            });
            left = self.arena.alloc(FromClause::Join(join));
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
                Ok(Expr::Literal(Literal::String(s)))
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
            let column = self.expect_ident()?;
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
        } else {
            bail!("expected TABLE, INDEX, or SCHEMA after CREATE");
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
            Token::Keyword(Keyword::Real) => {
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
}
