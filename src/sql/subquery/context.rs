//! # Subquery Execution Context
//!
//! Manages the execution context for subqueries, particularly tracking outer
//! row values needed for correlated subquery evaluation.
//!
//! ## Context Hierarchy
//!
//! Subqueries can be nested arbitrarily deep, creating a stack of contexts:
//!
//! ```text
//! SELECT * FROM orders o WHERE EXISTS (
//!     SELECT 1 FROM items i WHERE EXISTS (
//!         SELECT 1 FROM inventory inv WHERE inv.item_id = i.id AND inv.order_id = o.id
//!     )
//! )
//! ```
//!
//! The innermost subquery needs access to both `i` and `o` from outer scopes.
//!
//! ## Binding Strategy
//!
//! For each outer row:
//! 1. Push current outer row values onto context
//! 2. Execute subquery (may access outer values via correlation refs)
//! 3. Pop context when done
//!
//! ## Memory Efficiency
//!
//! The context stores only references to the current outer row, not copies.
//! This is critical for correlated subqueries that execute millions of times.

use crate::types::OwnedValue;
use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct OuterRowBinding<'a> {
    pub table_alias: Option<&'a str>,
    pub column_name: &'a str,
    pub value: OwnedValue,
}

#[derive(Debug)]
pub struct SubqueryContext<'a> {
    bindings: Vec<HashMap<(&'a str, &'a str), OwnedValue>>,
    current_depth: usize,
}

impl<'a> SubqueryContext<'a> {
    pub fn new() -> Self {
        Self {
            bindings: Vec::new(),
            current_depth: 0,
        }
    }

    pub fn push_scope(&mut self) {
        self.bindings.push(HashMap::new());
        self.current_depth += 1;
    }

    pub fn pop_scope(&mut self) {
        if !self.bindings.is_empty() {
            self.bindings.pop();
            self.current_depth = self.current_depth.saturating_sub(1);
        }
    }

    pub fn bind(&mut self, table: &'a str, column: &'a str, value: OwnedValue) {
        if let Some(current_scope) = self.bindings.last_mut() {
            current_scope.insert((table, column), value);
        }
    }

    pub fn bind_row(&mut self, bindings: impl IntoIterator<Item = OuterRowBinding<'a>>) {
        if let Some(current_scope) = self.bindings.last_mut() {
            for binding in bindings {
                let table = binding.table_alias.unwrap_or("");
                current_scope.insert((table, binding.column_name), binding.value);
            }
        }
    }

    pub fn resolve(&self, table: Option<&str>, column: &str) -> Option<&OwnedValue> {
        for scope in self.bindings.iter().rev() {
            if let Some(t) = table {
                for ((tbl, col), value) in scope {
                    if *tbl == t && *col == column {
                        return Some(value);
                    }
                }
            } else {
                for ((_, col), value) in scope {
                    if *col == column {
                        return Some(value);
                    }
                }
            }
        }
        None
    }

    pub fn resolve_qualified(&self, table: &str, column: &str) -> Option<&OwnedValue> {
        self.resolve(Some(table), column)
    }

    pub fn current_depth(&self) -> usize {
        self.current_depth
    }

    pub fn has_bindings(&self) -> bool {
        self.bindings.iter().any(|scope| !scope.is_empty())
    }

    pub fn clear_current_scope(&mut self) {
        if let Some(current_scope) = self.bindings.last_mut() {
            current_scope.clear();
        }
    }
}

impl Default for SubqueryContext<'_> {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_context_new() {
        let ctx = SubqueryContext::new();
        assert_eq!(ctx.current_depth(), 0);
        assert!(!ctx.has_bindings());
    }

    #[test]
    fn test_push_pop_scope() {
        let mut ctx = SubqueryContext::new();
        ctx.push_scope();
        assert_eq!(ctx.current_depth(), 1);
        ctx.push_scope();
        assert_eq!(ctx.current_depth(), 2);
        ctx.pop_scope();
        assert_eq!(ctx.current_depth(), 1);
        ctx.pop_scope();
        assert_eq!(ctx.current_depth(), 0);
    }

    #[test]
    fn test_bind_and_resolve() {
        let mut ctx = SubqueryContext::new();
        ctx.push_scope();
        ctx.bind("orders", "id", OwnedValue::Int(42));

        let resolved = ctx.resolve_qualified("orders", "id");
        assert!(resolved.is_some());
        assert_eq!(*resolved.unwrap(), OwnedValue::Int(42));
    }

    #[test]
    fn test_nested_scope_resolution() {
        let mut ctx = SubqueryContext::new();

        ctx.push_scope();
        ctx.bind("orders", "id", OwnedValue::Int(1));

        ctx.push_scope();
        ctx.bind("items", "id", OwnedValue::Int(100));

        assert_eq!(
            *ctx.resolve_qualified("orders", "id").unwrap(),
            OwnedValue::Int(1)
        );
        assert_eq!(
            *ctx.resolve_qualified("items", "id").unwrap(),
            OwnedValue::Int(100)
        );

        ctx.pop_scope();
        assert!(ctx.resolve_qualified("items", "id").is_none());
        assert!(ctx.resolve_qualified("orders", "id").is_some());
    }

    #[test]
    fn test_resolve_without_table() {
        let mut ctx = SubqueryContext::new();
        ctx.push_scope();
        ctx.bind("orders", "total", OwnedValue::Float(99.99));

        let resolved = ctx.resolve(None, "total");
        assert!(resolved.is_some());
    }

    #[test]
    fn test_clear_current_scope() {
        let mut ctx = SubqueryContext::new();
        ctx.push_scope();
        ctx.bind("t", "col", OwnedValue::Int(1));
        assert!(ctx.has_bindings());

        ctx.clear_current_scope();
        assert!(!ctx.has_bindings());
    }
}
