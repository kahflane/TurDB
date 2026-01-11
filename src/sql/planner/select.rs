//! # SELECT Statement Planning
//!
//! This module handles the logical planning of SELECT statements, including:
//!
//! - FROM clause processing (tables, joins, subqueries)
//! - WHERE clause filtering
//! - GROUP BY and aggregate handling
//! - Window function detection and planning
//! - ORDER BY and LIMIT processing
//! - CTE (Common Table Expression) support
//! - Set operations (UNION, INTERSECT, EXCEPT)
//!
//! ## Planning Flow
//!
//! 1. Build CTE context from WITH clause
//! 2. Plan FROM clause to get base relations
//! 3. Apply WHERE clause as filter
//! 4. Handle GROUP BY and aggregates
//! 5. Add window function operators
//! 6. Apply projection for SELECT list
//! 7. Handle set operations if present
//! 8. Apply ORDER BY and LIMIT
//!
//! ## Design
//!
//! The planning methods are implemented as methods on Planner and take
//! mutable references to build the logical plan tree in the arena.

use crate::sql::ast::{Expr, NullsOrder, OrderDirection, SelectColumn, SelectStmt};
use super::logical::{
    LogicalAggregate, LogicalFilter, LogicalLimit, LogicalOperator,
    LogicalPlan, LogicalProject, LogicalSetOp, LogicalSort, LogicalSubquery,
    LogicalWindow, SetOpKind, SortKey, WindowFunctionDef, WindowFunctionType,
};
use super::schema::{CteContext, OutputColumn, OutputSchema, PlannedCte};
use super::Planner;
use eyre::{bail, Result};

impl<'a> Planner<'a> {
    pub(crate) fn plan_select(&self, select: &SelectStmt<'a>) -> Result<LogicalPlan<'a>> {
        let cte_context = self.build_cte_context(select.with)?;
        self.plan_select_with_ctes(select, &cte_context)
    }

    pub(crate) fn build_cte_context(
        &self,
        with_clause: Option<&crate::sql::ast::WithClause<'a>>,
    ) -> Result<CteContext<'a>> {
        let mut ctx = CteContext::new();

        if let Some(with) = with_clause {
            for cte in with.ctes {
                let cte_plan = self.plan_select_with_ctes(cte.query, &ctx)?;
                let mut output_schema = self.compute_logical_output_schema(cte_plan.root)?;

                if let Some(col_names) = cte.columns {
                    if col_names.len() != output_schema.columns.len() {
                        bail!(
                            "CTE '{}' has {} column names but query returns {} columns",
                            cte.name,
                            col_names.len(),
                            output_schema.columns.len()
                        );
                    }
                    let mut new_cols = bumpalo::collections::Vec::new_in(self.arena);
                    for (i, col) in output_schema.columns.iter().enumerate() {
                        new_cols.push(OutputColumn {
                            name: col_names[i],
                            data_type: col.data_type,
                            nullable: col.nullable,
                        });
                    }
                    output_schema = OutputSchema {
                        columns: new_cols.into_bump_slice(),
                    };
                }

                ctx.insert(
                    cte.name,
                    PlannedCte {
                        plan: cte_plan.root,
                        output_schema,
                        columns: cte.columns,
                    },
                );
            }
        }

        Ok(ctx)
    }

    pub(crate) fn plan_select_with_ctes(
        &self,
        select: &SelectStmt<'a>,
        cte_context: &CteContext<'a>,
    ) -> Result<LogicalPlan<'a>> {
        let mut current: &'a LogicalOperator<'a> = match select.from {
            Some(from) => self.plan_from_clause_with_ctes(from, cte_context)?,
            None => self.arena.alloc(LogicalOperator::DualScan),
        };

        let tables_in_scope = self.collect_tables_in_scope(current);
        if select.from.is_some() {
            self.validate_select_columns(select.columns, &tables_in_scope)?;
        }

        if let Some(predicate) = select.where_clause {
            self.validate_expr_columns(predicate, &tables_in_scope)?;
            let filter = self.arena.alloc(LogicalOperator::Filter(LogicalFilter {
                input: current,
                predicate,
            }));
            current = filter;
        }

        let has_aggregates = self.select_has_aggregates(select.columns);
        if !select.group_by.is_empty() || has_aggregates {
            for group_expr in select.group_by.iter() {
                self.validate_expr_columns(group_expr, &tables_in_scope)?;
            }

            let aggregates = self.extract_aggregates(select.columns);
            let agg = self
                .arena
                .alloc(LogicalOperator::Aggregate(LogicalAggregate {
                    input: current,
                    group_by: select.group_by,
                    aggregates,
                }));
            current = agg;

            if let Some(having) = select.having {
                self.validate_expr_columns(having, &tables_in_scope)?;
                let having_filter = self.arena.alloc(LogicalOperator::Filter(LogicalFilter {
                    input: current,
                    predicate: having,
                }));
                current = having_filter;
            }
        }

        let has_window_functions = self.select_has_window_functions(select.columns);

        if has_window_functions {
            let window_functions = self.extract_window_functions(select.columns)?;
            let window = self.arena.alloc(LogicalOperator::Window(LogicalWindow {
                input: current,
                window_functions,
            }));
            current = window;
        }

        let (exprs, aliases) = self.extract_select_expressions(select.columns);
        let project = self.arena.alloc(LogicalOperator::Project(LogicalProject {
            input: current,
            expressions: exprs,
            aliases,
        }));
        current = project;

        if let Some(set_op) = select.set_op {
            let right_select = set_op.right;
            let right_order_by = right_select.order_by;
            let right_limit = right_select.limit;
            let right_offset = right_select.offset;

            let right_plan = self.plan_select_core(right_select, cte_context)?;
            let kind = match set_op.op {
                crate::sql::ast::SetOperator::Union => SetOpKind::Union,
                crate::sql::ast::SetOperator::Intersect => SetOpKind::Intersect,
                crate::sql::ast::SetOperator::Except => SetOpKind::Except,
            };
            let set_op_node = self.arena.alloc(LogicalOperator::SetOp(LogicalSetOp {
                left: current,
                right: right_plan.root,
                kind,
                all: set_op.all,
            }));
            current = set_op_node;

            if !right_order_by.is_empty() {
                let sort_keys = self.convert_order_by(right_order_by);
                let sort = self.arena.alloc(LogicalOperator::Sort(LogicalSort {
                    input: current,
                    order_by: sort_keys,
                }));
                current = sort;
            }

            if right_limit.is_some() || right_offset.is_some() {
                let limit_val = right_limit.and_then(|e| self.eval_const_u64(e));
                let offset_val = right_offset.and_then(|e| self.eval_const_u64(e));
                let limit_op = self.arena.alloc(LogicalOperator::Limit(LogicalLimit {
                    input: current,
                    limit: limit_val,
                    offset: offset_val,
                }));
                current = limit_op;
            }
        } else {
            if !select.order_by.is_empty() {
                for order_item in select.order_by.iter() {
                    let is_alias = if let Expr::Column(col_ref) = order_item.expr {
                        col_ref.table.is_none()
                            && aliases.contains(&Some(col_ref.column))
                    } else {
                        false
                    };

                    if !is_alias {
                        self.validate_expr_columns(order_item.expr, &tables_in_scope)?;
                    }
                }
                let sort_keys = self.convert_order_by(select.order_by);
                let sort = self.arena.alloc(LogicalOperator::Sort(LogicalSort {
                    input: current,
                    order_by: sort_keys,
                }));
                current = sort;
            }

            if select.limit.is_some() || select.offset.is_some() {
                let limit_val = select.limit.and_then(|e| self.eval_const_u64(e));
                let offset_val = select.offset.and_then(|e| self.eval_const_u64(e));

                let limit_op = self.arena.alloc(LogicalOperator::Limit(LogicalLimit {
                    input: current,
                    limit: limit_val,
                    offset: offset_val,
                }));
                current = limit_op;
            }
        }

        Ok(LogicalPlan { root: current })
    }

    pub(crate) fn plan_select_core(
        &self,
        select: &SelectStmt<'a>,
        cte_context: &CteContext<'a>,
    ) -> Result<LogicalPlan<'a>> {
        let mut current: &'a LogicalOperator<'a> = match select.from {
            Some(from) => self.plan_from_clause_with_ctes(from, cte_context)?,
            None => self.arena.alloc(LogicalOperator::DualScan),
        };

        let tables_in_scope = self.collect_tables_in_scope(current);
        if select.from.is_some() {
            self.validate_select_columns(select.columns, &tables_in_scope)?;
        }

        if let Some(predicate) = select.where_clause {
            self.validate_expr_columns(predicate, &tables_in_scope)?;
            let filter = self.arena.alloc(LogicalOperator::Filter(LogicalFilter {
                input: current,
                predicate,
            }));
            current = filter;
        }

        let has_aggregates = self.select_has_aggregates(select.columns);
        if !select.group_by.is_empty() || has_aggregates {
            for group_expr in select.group_by.iter() {
                self.validate_expr_columns(group_expr, &tables_in_scope)?;
            }

            let aggregates = self.extract_aggregates(select.columns);
            let agg = self
                .arena
                .alloc(LogicalOperator::Aggregate(LogicalAggregate {
                    input: current,
                    group_by: select.group_by,
                    aggregates,
                }));
            current = agg;

            if let Some(having) = select.having {
                self.validate_expr_columns(having, &tables_in_scope)?;
                let having_filter = self.arena.alloc(LogicalOperator::Filter(LogicalFilter {
                    input: current,
                    predicate: having,
                }));
                current = having_filter;
            }
        }

        let (exprs, aliases) = self.extract_select_expressions(select.columns);
        let project = self.arena.alloc(LogicalOperator::Project(LogicalProject {
            input: current,
            expressions: exprs,
            aliases,
        }));
        current = project;

        if let Some(set_op) = select.set_op {
            let right_plan = self.plan_select_core(set_op.right, cte_context)?;
            let kind = match set_op.op {
                crate::sql::ast::SetOperator::Union => SetOpKind::Union,
                crate::sql::ast::SetOperator::Intersect => SetOpKind::Intersect,
                crate::sql::ast::SetOperator::Except => SetOpKind::Except,
            };
            let set_op_node = self.arena.alloc(LogicalOperator::SetOp(LogicalSetOp {
                left: current,
                right: right_plan.root,
                kind,
                all: set_op.all,
            }));
            current = set_op_node;
        }

        Ok(LogicalPlan { root: current })
    }

    pub(crate) fn plan_from_clause_with_ctes(
        &self,
        from: &'a crate::sql::ast::FromClause<'a>,
        cte_context: &CteContext<'a>,
    ) -> Result<&'a LogicalOperator<'a>> {
        use crate::sql::ast::FromClause;
        use super::logical::LogicalScan;

        match from {
            FromClause::Table(table_ref) => {
                if table_ref.schema.is_none() {
                    if let Some(cte) = cte_context.get(table_ref.name) {
                        let alias = table_ref.alias.unwrap_or(table_ref.name);
                        let subquery_op =
                            self.arena.alloc(LogicalOperator::Subquery(LogicalSubquery {
                                plan: cte.plan,
                                alias,
                                output_schema: cte.output_schema.clone(),
                            }));
                        return Ok(subquery_op);
                    }
                }

                self.validate_table_exists(table_ref.schema, table_ref.name)?;

                let scan = self.arena.alloc(LogicalOperator::Scan(LogicalScan {
                    schema: table_ref.schema,
                    table: table_ref.name,
                    alias: table_ref.alias,
                }));
                Ok(scan)
            }
            FromClause::Join(join) => self.plan_join_with_ctes(join, cte_context),
            FromClause::Subquery { query, alias } => {
                let subquery_plan = self.plan_select_with_ctes(query, cte_context)?;
                let output_schema = self.compute_logical_output_schema(subquery_plan.root)?;
                let subquery_op = self.arena.alloc(LogicalOperator::Subquery(LogicalSubquery {
                    plan: subquery_plan.root,
                    alias,
                    output_schema,
                }));
                Ok(subquery_op)
            }
            FromClause::Lateral {
                subquery: _,
                alias: _,
            } => {
                bail!("LATERAL subqueries not yet implemented")
            }
        }
    }

    pub(crate) fn plan_join_with_ctes(
        &self,
        join: &'a crate::sql::ast::JoinClause<'a>,
        cte_context: &CteContext<'a>,
    ) -> Result<&'a LogicalOperator<'a>> {
        use super::logical::LogicalJoin;

        let left = self.plan_from_clause_with_ctes(join.left, cte_context)?;
        let right = self.plan_from_clause_with_ctes(join.right, cte_context)?;

        let condition = match join.condition {
            crate::sql::ast::JoinCondition::On(expr) => Some(expr),
            crate::sql::ast::JoinCondition::Using(_) => {
                bail!("USING clause in joins not yet implemented")
            }
            crate::sql::ast::JoinCondition::Natural => {
                bail!("NATURAL joins not yet implemented")
            }
            crate::sql::ast::JoinCondition::None => None,
        };

        let join_op = self.arena.alloc(LogicalOperator::Join(LogicalJoin {
            left,
            right,
            join_type: join.join_type,
            condition,
        }));

        Ok(join_op)
    }

    pub(crate) fn extract_aggregates(
        &self,
        columns: &'a [SelectColumn<'a>],
    ) -> &'a [&'a Expr<'a>] {
        let mut aggregates = bumpalo::collections::Vec::new_in(self.arena);

        for col in columns {
            if let SelectColumn::Expr { expr, .. } = col {
                self.collect_aggregates_from_expr(expr, &mut aggregates);
            }
        }

        aggregates.into_bump_slice()
    }

    pub(crate) fn traverse_expr_for_aggregates<F>(&self, expr: &'a Expr<'a>, on_aggregate: &mut F) -> bool
    where
        F: FnMut(&'a Expr<'a>) -> bool,
    {
        use crate::sql::ast::{FunctionArgs, FunctionCall};

        match expr {
            Expr::Function(FunctionCall {
                name, over, args, ..
            }) => {
                if over.is_none() {
                    let func_name = name.name.to_ascii_lowercase();
                    if matches!(func_name.as_str(), "count" | "sum" | "avg" | "min" | "max") {
                        if on_aggregate(expr) {
                            return true;
                        }
                        return false;
                    }
                }
                if let FunctionArgs::Args(fn_args) = args {
                    for arg in *fn_args {
                        if self.traverse_expr_for_aggregates(arg.value, on_aggregate) {
                            return true;
                        }
                    }
                }
                false
            }
            Expr::BinaryOp { left, right, .. } => {
                self.traverse_expr_for_aggregates(left, on_aggregate)
                    || self.traverse_expr_for_aggregates(right, on_aggregate)
            }
            Expr::UnaryOp { expr: inner, .. } => {
                self.traverse_expr_for_aggregates(inner, on_aggregate)
            }
            Expr::Between {
                expr, low, high, ..
            } => {
                self.traverse_expr_for_aggregates(expr, on_aggregate)
                    || self.traverse_expr_for_aggregates(low, on_aggregate)
                    || self.traverse_expr_for_aggregates(high, on_aggregate)
            }
            Expr::Like {
                expr,
                pattern,
                escape,
                ..
            } => {
                self.traverse_expr_for_aggregates(expr, on_aggregate)
                    || self.traverse_expr_for_aggregates(pattern, on_aggregate)
                    || escape
                        .map(|e| self.traverse_expr_for_aggregates(e, on_aggregate))
                        .unwrap_or(false)
            }
            Expr::InList { expr, list, .. } => {
                self.traverse_expr_for_aggregates(expr, on_aggregate)
                    || list
                        .iter()
                        .any(|e| self.traverse_expr_for_aggregates(e, on_aggregate))
            }
            Expr::IsNull { expr, .. } => self.traverse_expr_for_aggregates(expr, on_aggregate),
            Expr::IsDistinctFrom { left, right, .. } => {
                self.traverse_expr_for_aggregates(left, on_aggregate)
                    || self.traverse_expr_for_aggregates(right, on_aggregate)
            }
            Expr::Case {
                operand,
                conditions,
                else_result,
            } => {
                operand
                    .map(|o| self.traverse_expr_for_aggregates(o, on_aggregate))
                    .unwrap_or(false)
                    || conditions.iter().any(|c| {
                        self.traverse_expr_for_aggregates(c.condition, on_aggregate)
                            || self.traverse_expr_for_aggregates(c.result, on_aggregate)
                    })
                    || else_result
                        .map(|e| self.traverse_expr_for_aggregates(e, on_aggregate))
                        .unwrap_or(false)
            }
            Expr::Cast { expr, .. } => self.traverse_expr_for_aggregates(expr, on_aggregate),
            Expr::ArraySubscript { array, index, .. } => {
                self.traverse_expr_for_aggregates(array, on_aggregate)
                    || self.traverse_expr_for_aggregates(index, on_aggregate)
            }
            Expr::ArraySlice {
                array,
                lower,
                upper,
            } => {
                self.traverse_expr_for_aggregates(array, on_aggregate)
                    || lower
                        .map(|e| self.traverse_expr_for_aggregates(e, on_aggregate))
                        .unwrap_or(false)
                    || upper
                        .map(|e| self.traverse_expr_for_aggregates(e, on_aggregate))
                        .unwrap_or(false)
            }
            Expr::Array(items) => items
                .iter()
                .any(|e| self.traverse_expr_for_aggregates(e, on_aggregate)),
            Expr::Row(items) => items
                .iter()
                .any(|e| self.traverse_expr_for_aggregates(e, on_aggregate)),
            Expr::Literal(_)
            | Expr::Column(_)
            | Expr::Parameter(_)
            | Expr::Subquery(_)
            | Expr::Exists { .. }
            | Expr::InSubquery { .. } => false,
        }
    }

    pub(crate) fn collect_aggregates_from_expr(
        &self,
        expr: &'a Expr<'a>,
        aggregates: &mut bumpalo::collections::Vec<&'a Expr<'a>>,
    ) {
        self.traverse_expr_for_aggregates(expr, &mut |agg| {
            aggregates.push(agg);
            false
        });
    }

    pub(crate) fn contains_aggregate(&self, expr: &Expr<'a>) -> bool {
        self.traverse_expr_for_aggregates(expr, &mut |_| true)
    }

    pub(crate) fn select_has_aggregates(&self, columns: &'a [SelectColumn<'a>]) -> bool {
        for col in columns {
            if let SelectColumn::Expr { expr, .. } = col {
                if self.contains_aggregate(expr) {
                    return true;
                }
            }
        }
        false
    }

    pub(crate) fn is_window_function(&self, expr: &Expr<'a>) -> bool {
        if let Expr::Function(func) = expr {
            func.over.is_some()
        } else {
            false
        }
    }

    pub(crate) fn select_has_window_functions(
        &self,
        columns: &'a [SelectColumn<'a>],
    ) -> bool {
        for col in columns {
            if let SelectColumn::Expr { expr, .. } = col {
                if self.is_window_function(expr) {
                    return true;
                }
            }
        }
        false
    }

    pub(crate) fn extract_window_functions(
        &self,
        columns: &'a [SelectColumn<'a>],
    ) -> Result<&'a [WindowFunctionDef<'a>]> {
        let mut window_funcs = bumpalo::collections::Vec::new_in(self.arena);

        for col in columns {
            if let SelectColumn::Expr {
                expr: Expr::Function(func),
                alias,
            } = col
            {
                if let Some(window_spec) = &func.over {
                    let order_by_keys: &[SortKey<'a>] = {
                        let mut keys = bumpalo::collections::Vec::new_in(self.arena);
                        for item in window_spec.order_by.iter() {
                            keys.push(SortKey {
                                expr: item.expr,
                                ascending: matches!(item.direction, OrderDirection::Asc),
                                nulls_first: matches!(item.nulls, NullsOrder::First),
                            });
                        }
                        keys.into_bump_slice()
                    };

                    let args: &[&Expr<'a>] = match &func.args {
                        crate::sql::ast::FunctionArgs::Args(func_args) => {
                            let mut arg_exprs = bumpalo::collections::Vec::new_in(self.arena);
                            for arg in func_args.iter() {
                                arg_exprs.push(arg.value);
                            }
                            arg_exprs.into_bump_slice()
                        }
                        crate::sql::ast::FunctionArgs::Star => &[],
                        crate::sql::ast::FunctionArgs::None => &[],
                    };

                    let function_type =
                        WindowFunctionType::from_name(func.name.name).ok_or_else(|| {
                            eyre::eyre!(
                                "unsupported window function: '{}'. Supported: ROW_NUMBER, RANK, \
                                 DENSE_RANK, COUNT, SUM, AVG, MIN, MAX",
                                func.name.name
                            )
                        })?;
                    window_funcs.push(WindowFunctionDef {
                        function_name: func.name.name,
                        function_type,
                        args,
                        partition_by: window_spec.partition_by,
                        order_by: order_by_keys,
                        alias: *alias,
                    });
                }
            }
        }

        Ok(window_funcs.into_bump_slice())
    }

    pub(crate) fn extract_select_expressions(
        &self,
        columns: &'a [SelectColumn<'a>],
    ) -> (&'a [&'a Expr<'a>], &'a [Option<&'a str>]) {
        let mut exprs = bumpalo::collections::Vec::new_in(self.arena);
        let mut aliases = bumpalo::collections::Vec::new_in(self.arena);

        for col in columns {
            match col {
                SelectColumn::Expr { expr, alias } => {
                    exprs.push(*expr);
                    aliases.push(*alias);
                }
                SelectColumn::AllColumns => {}
                SelectColumn::TableAllColumns(_) => {}
            }
        }

        (exprs.into_bump_slice(), aliases.into_bump_slice())
    }

    pub(crate) fn convert_order_by(
        &self,
        order_by: &'a [crate::sql::ast::OrderByItem<'a>],
    ) -> &'a [SortKey<'a>] {
        let mut keys = bumpalo::collections::Vec::new_in(self.arena);

        for item in order_by {
            keys.push(SortKey {
                expr: item.expr,
                ascending: matches!(item.direction, OrderDirection::Asc),
                nulls_first: matches!(item.nulls, NullsOrder::First),
            });
        }

        keys.into_bump_slice()
    }

    pub(crate) fn eval_const_u64(&self, expr: &Expr<'a>) -> Option<u64> {
        match expr {
            Expr::Literal(crate::sql::ast::Literal::Integer(n)) => n.parse().ok(),
            _ => None,
        }
    }
}
