//! # Join Reordering Rule
//!
//! Reorders multi-way joins by table cardinality for optimal performance.
//! Smaller tables are placed first in the join tree to minimize intermediate
//! result sizes.
//!
//! ## Problem
//!
//! When parsing `FROM a, b, c, d WHERE ...`, the parser creates a left-deep
//! join tree in left-to-right order:
//! ```text
//! ((a ⨝ b) ⨝ c) ⨝ d
//! ```
//!
//! This order may be suboptimal. For example, if `d` has 53M rows and `a` has
//! 100 rows, the final join processes 53M rows from `d`. Reordering to join
//! smaller tables first reduces intermediate result sizes.
//!
//! ## Transformation
//!
//! Given tables with estimated cardinalities:
//! - `a`: 100 rows
//! - `b`: 1000 rows
//! - `c`: 500 rows
//! - `d`: 53,000,000 rows
//!
//! ```text
//! Before: ((a ⨝ b) ⨝ c) ⨝ d
//! After:  ((a ⨝ c) ⨝ b) ⨝ d
//! ```
//!
//! Tables are sorted by cardinality (smallest first) and conditions are
//! remapped to the appropriate join levels.
//!
//! ## Algorithm
//!
//! 1. Detect multi-way inner join chains
//! 2. Extract all base tables from the join tree
//! 3. Estimate cardinality for each table
//! 4. Sort tables by cardinality (smallest first)
//! 5. Collect all join conditions
//! 6. Build left-deep tree with conditions applied where both tables present
//!
//! ## Constraints
//!
//! - Only reorders INNER joins (outer join semantics depend on order)
//! - Preserves join conditions correctly across reordering
//! - Uses CostEstimator for cardinality estimates
//!
//! ## Usage
//!
//! ```ignore
//! let optimizer = Optimizer::new(); // Includes JoinReorderingRule
//! let optimized = optimizer.optimize(plan, arena)?;
//! ```

use crate::schema::Catalog;
use crate::sql::ast::{BinaryOperator, Expr, JoinType};
use crate::sql::optimizer::{CostEstimator, OptimizationRule};
use crate::sql::planner::{LogicalJoin, LogicalOperator};
use bumpalo::Bump;
use eyre::Result;
use smallvec::SmallVec;
use std::collections::HashSet;

pub struct JoinReorderingRule<'c> {
    catalog: &'c Catalog,
}

impl<'c> JoinReorderingRule<'c> {
    pub fn new(catalog: &'c Catalog) -> Self {
        Self { catalog }
    }
}

impl<'c> OptimizationRule for JoinReorderingRule<'c> {
    fn name(&self) -> &'static str {
        "join_reordering"
    }

    fn apply<'a>(
        &self,
        plan: &'a LogicalOperator<'a>,
        arena: &'a Bump,
    ) -> Result<Option<&'a LogicalOperator<'a>>> {
        self.reorder_joins(plan, arena)
    }
}

impl<'c> JoinReorderingRule<'c> {
    fn reorder_joins<'a>(
        &self,
        plan: &'a LogicalOperator<'a>,
        arena: &'a Bump,
    ) -> Result<Option<&'a LogicalOperator<'a>>> {
        match plan {
            LogicalOperator::Join(join) => {
                if self.is_reorderable_join_chain(plan) {
                    let tables = self.collect_base_tables(plan, arena);
                    if tables.len() >= 2 {
                        let conditions = self.collect_all_conditions(plan, arena);
                        let cost_estimator = CostEstimator::new(self.catalog, arena);

                        let mut tables_with_card: SmallVec<[(&'a LogicalOperator<'a>, u64); 8]> =
                            SmallVec::new();
                        for table in tables.iter() {
                            let card = self.estimate_table_cardinality(*table, &cost_estimator);
                            tables_with_card.push((*table, card));
                        }

                        tables_with_card.sort_by_key(|(_, card)| *card);

                        let current_order: Vec<_> = tables.iter().copied().collect();
                        let new_order: Vec<_> =
                            tables_with_card.iter().map(|(t, _)| *t).collect();

                        if current_order != new_order {
                            let new_tree =
                                self.build_join_tree(&new_order, &conditions, arena);
                            return Ok(Some(new_tree));
                        }
                    }
                }

                let left_result = self.reorder_joins(join.left, arena)?;
                let right_result = self.reorder_joins(join.right, arena)?;

                if left_result.is_some() || right_result.is_some() {
                    let new_join = arena.alloc(LogicalOperator::Join(LogicalJoin {
                        left: left_result.unwrap_or(join.left),
                        right: right_result.unwrap_or(join.right),
                        join_type: join.join_type,
                        condition: join.condition,
                    }));
                    return Ok(Some(new_join));
                }
                Ok(None)
            }

            LogicalOperator::Filter(filter) => {
                let result = self.reorder_joins(filter.input, arena)?;
                if let Some(new_input) = result {
                    let new_filter =
                        arena.alloc(LogicalOperator::Filter(crate::sql::planner::LogicalFilter {
                            input: new_input,
                            predicate: filter.predicate,
                        }));
                    return Ok(Some(new_filter));
                }
                Ok(None)
            }

            LogicalOperator::Project(project) => {
                let result = self.reorder_joins(project.input, arena)?;
                if let Some(new_input) = result {
                    let new_project =
                        arena.alloc(LogicalOperator::Project(crate::sql::planner::LogicalProject {
                            input: new_input,
                            expressions: project.expressions,
                            aliases: project.aliases,
                        }));
                    return Ok(Some(new_project));
                }
                Ok(None)
            }

            LogicalOperator::Aggregate(agg) => {
                let result = self.reorder_joins(agg.input, arena)?;
                if let Some(new_input) = result {
                    let new_agg =
                        arena.alloc(LogicalOperator::Aggregate(crate::sql::planner::LogicalAggregate {
                            input: new_input,
                            group_by: agg.group_by,
                            aggregates: agg.aggregates,
                        }));
                    return Ok(Some(new_agg));
                }
                Ok(None)
            }

            LogicalOperator::Sort(sort) => {
                let result = self.reorder_joins(sort.input, arena)?;
                if let Some(new_input) = result {
                    let new_sort =
                        arena.alloc(LogicalOperator::Sort(crate::sql::planner::LogicalSort {
                            input: new_input,
                            order_by: sort.order_by,
                        }));
                    return Ok(Some(new_sort));
                }
                Ok(None)
            }

            LogicalOperator::Limit(limit) => {
                let result = self.reorder_joins(limit.input, arena)?;
                if let Some(new_input) = result {
                    let new_limit =
                        arena.alloc(LogicalOperator::Limit(crate::sql::planner::LogicalLimit {
                            input: new_input,
                            limit: limit.limit,
                            offset: limit.offset,
                        }));
                    return Ok(Some(new_limit));
                }
                Ok(None)
            }

            _ => Ok(None),
        }
    }

    fn is_reorderable_join_chain<'a>(&self, plan: &'a LogicalOperator<'a>) -> bool {
        match plan {
            LogicalOperator::Join(join) => {
                matches!(join.join_type, JoinType::Inner | JoinType::Cross)
                    && self.is_reorderable_join_chain(join.left)
            }
            LogicalOperator::Scan(_) => true,
            _ => false,
        }
    }

    fn collect_base_tables<'a>(
        &self,
        plan: &'a LogicalOperator<'a>,
        _arena: &'a Bump,
    ) -> SmallVec<[&'a LogicalOperator<'a>; 8]> {
        let mut tables = SmallVec::new();
        self.collect_tables_recursive(plan, &mut tables);
        tables
    }

    fn collect_tables_recursive<'a>(
        &self,
        plan: &'a LogicalOperator<'a>,
        tables: &mut SmallVec<[&'a LogicalOperator<'a>; 8]>,
    ) {
        match plan {
            LogicalOperator::Join(join) => {
                self.collect_tables_recursive(join.left, tables);
                self.collect_tables_recursive(join.right, tables);
            }
            LogicalOperator::Scan(_) => {
                tables.push(plan);
            }
            _ => {
                tables.push(plan);
            }
        }
    }

    fn collect_all_conditions<'a>(
        &self,
        plan: &'a LogicalOperator<'a>,
        _arena: &'a Bump,
    ) -> SmallVec<[&'a Expr<'a>; 8]> {
        let mut conditions = SmallVec::new();
        self.collect_conditions_recursive(plan, &mut conditions);
        conditions
    }

    fn collect_conditions_recursive<'a>(
        &self,
        plan: &'a LogicalOperator<'a>,
        conditions: &mut SmallVec<[&'a Expr<'a>; 8]>,
    ) {
        match plan {
            LogicalOperator::Join(join) => {
                if let Some(cond) = join.condition {
                    self.flatten_and(cond, conditions);
                }
                self.collect_conditions_recursive(join.left, conditions);
                self.collect_conditions_recursive(join.right, conditions);
            }
            _ => {}
        }
    }

    fn flatten_and<'a>(&self, expr: &'a Expr<'a>, out: &mut SmallVec<[&'a Expr<'a>; 8]>) {
        match expr {
            Expr::BinaryOp {
                left,
                op: BinaryOperator::And,
                right,
            } => {
                self.flatten_and(left, out);
                self.flatten_and(right, out);
            }
            _ => out.push(expr),
        }
    }

    fn estimate_table_cardinality<'a>(
        &self,
        table: &'a LogicalOperator<'a>,
        cost_estimator: &CostEstimator<'a>,
    ) -> u64 {
        match table {
            LogicalOperator::Scan(scan) => {
                if let Ok(table_def) = self.catalog.resolve_table_in_schema(scan.schema, scan.table)
                {
                    let row_count = table_def.row_count();
                    if row_count > 0 {
                        return row_count;
                    }
                }
                cost_estimator.estimate_cardinality(table)
            }
            _ => cost_estimator.estimate_cardinality(table),
        }
    }

    fn build_join_tree<'a>(
        &self,
        ordered_tables: &[&'a LogicalOperator<'a>],
        all_conditions: &[&'a Expr<'a>],
        arena: &'a Bump,
    ) -> &'a LogicalOperator<'a> {
        if ordered_tables.is_empty() {
            return arena.alloc(LogicalOperator::DualScan);
        }
        if ordered_tables.len() == 1 {
            return ordered_tables[0];
        }

        let mut accumulated_tables: HashSet<&str> = HashSet::new();
        let mut result = ordered_tables[0];
        self.add_table_names(result, &mut accumulated_tables);

        for i in 1..ordered_tables.len() {
            let right_table = ordered_tables[i];
            let mut right_tables: HashSet<&str> = HashSet::new();
            self.add_table_names(right_table, &mut right_tables);

            let applicable_conditions: SmallVec<[&'a Expr<'a>; 8]> = all_conditions
                .iter()
                .filter(|cond| self.condition_applies(&accumulated_tables, &right_tables, cond))
                .copied()
                .collect();

            let join_condition = if applicable_conditions.is_empty() {
                None
            } else {
                Some(self.combine_conditions(&applicable_conditions, arena))
            };

            let join_type = if join_condition.is_some() {
                JoinType::Inner
            } else {
                JoinType::Cross
            };

            result = arena.alloc(LogicalOperator::Join(LogicalJoin {
                left: result,
                right: right_table,
                join_type,
                condition: join_condition,
            }));

            for table in right_tables {
                accumulated_tables.insert(table);
            }
        }

        result
    }

    fn add_table_names<'a>(&self, plan: &'a LogicalOperator<'a>, tables: &mut HashSet<&'a str>) {
        match plan {
            LogicalOperator::Scan(scan) => {
                tables.insert(scan.alias.unwrap_or(scan.table));
            }
            LogicalOperator::Join(join) => {
                self.add_table_names(join.left, tables);
                self.add_table_names(join.right, tables);
            }
            LogicalOperator::Subquery(subq) => {
                tables.insert(subq.alias);
            }
            _ => {}
        }
    }

    fn condition_applies<'a>(
        &self,
        left_tables: &HashSet<&str>,
        right_tables: &HashSet<&str>,
        condition: &'a Expr<'a>,
    ) -> bool {
        let mut condition_tables: HashSet<&str> = HashSet::new();
        self.extract_table_refs(condition, &mut condition_tables);

        if condition_tables.is_empty() {
            return false;
        }

        let references_left = condition_tables.iter().any(|t| left_tables.contains(t));
        let references_right = condition_tables.iter().any(|t| right_tables.contains(t));

        references_left && references_right
    }

    fn extract_table_refs<'a>(&self, expr: &'a Expr<'a>, tables: &mut HashSet<&'a str>) {
        match expr {
            Expr::Column(col) => {
                if let Some(t) = col.table {
                    tables.insert(t);
                }
            }
            Expr::BinaryOp { left, right, .. } => {
                self.extract_table_refs(left, tables);
                self.extract_table_refs(right, tables);
            }
            _ => {}
        }
    }

    fn combine_conditions<'a>(
        &self,
        conditions: &[&'a Expr<'a>],
        arena: &'a Bump,
    ) -> &'a Expr<'a> {
        if conditions.len() == 1 {
            return conditions[0];
        }

        let mut result = conditions[0];
        for cond in &conditions[1..] {
            result = arena.alloc(Expr::BinaryOp {
                left: result,
                op: BinaryOperator::And,
                right: cond,
            });
        }
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schema::{Catalog, ColumnDef};
    use crate::sql::ast::ColumnRef;
    use crate::sql::planner::LogicalScan;
    use crate::records::types::DataType;

    fn create_test_catalog() -> Catalog {
        let mut catalog = Catalog::new();

        catalog
            .create_table(
                "",
                "small_table",
                vec![ColumnDef::new("id", DataType::Int8)],
            )
            .unwrap();
        if let Some(table) = catalog.get_table_mut("", "small_table") {
            table.set_row_count(100);
        }

        catalog
            .create_table(
                "",
                "medium_table",
                vec![
                    ColumnDef::new("id", DataType::Int8),
                    ColumnDef::new("small_id", DataType::Int8),
                ],
            )
            .unwrap();
        if let Some(table) = catalog.get_table_mut("", "medium_table") {
            table.set_row_count(1000);
        }

        catalog
            .create_table(
                "",
                "large_table",
                vec![
                    ColumnDef::new("id", DataType::Int8),
                    ColumnDef::new("medium_id", DataType::Int8),
                ],
            )
            .unwrap();
        if let Some(table) = catalog.get_table_mut("", "large_table") {
            table.set_row_count(1_000_000);
        }

        catalog
    }

    #[test]
    fn rule_name_is_join_reordering() {
        let catalog = create_test_catalog();
        let rule = JoinReorderingRule::new(&catalog);
        assert_eq!(rule.name(), "join_reordering");
    }

    #[test]
    fn single_table_query_not_modified() {
        let catalog = create_test_catalog();
        let rule = JoinReorderingRule::new(&catalog);
        let arena = Bump::new();

        let scan = arena.alloc(LogicalOperator::Scan(LogicalScan {
            schema: None,
            table: "small_table",
            alias: None,
        }));

        let result = rule.apply(scan, &arena).unwrap();
        assert!(result.is_none(), "single table should not be modified");
    }

    #[test]
    fn two_tables_reordered_smaller_first() {
        let catalog = create_test_catalog();
        let rule = JoinReorderingRule::new(&catalog);
        let arena = Bump::new();

        let large_scan = arena.alloc(LogicalOperator::Scan(LogicalScan {
            schema: None,
            table: "large_table",
            alias: Some("l"),
        }));
        let small_scan = arena.alloc(LogicalOperator::Scan(LogicalScan {
            schema: None,
            table: "small_table",
            alias: Some("s"),
        }));

        let l_id = arena.alloc(Expr::Column(ColumnRef {
            schema: None,
            table: Some("l"),
            column: "id",
        }));
        let s_id = arena.alloc(Expr::Column(ColumnRef {
            schema: None,
            table: Some("s"),
            column: "id",
        }));
        let condition = arena.alloc(Expr::BinaryOp {
            left: l_id,
            op: BinaryOperator::Eq,
            right: s_id,
        });

        let join = arena.alloc(LogicalOperator::Join(LogicalJoin {
            left: large_scan,
            right: small_scan,
            join_type: JoinType::Inner,
            condition: Some(condition),
        }));

        let result = rule.apply(join, &arena).unwrap();
        assert!(result.is_some(), "join should be reordered");

        let new_plan = result.unwrap();
        if let LogicalOperator::Join(new_join) = new_plan {
            if let LogicalOperator::Scan(left_scan) = new_join.left {
                assert_eq!(
                    left_scan.table, "small_table",
                    "smaller table should be on the left"
                );
            } else {
                panic!("left side should be a scan");
            }
        } else {
            panic!("result should be a join");
        }
    }

    #[test]
    fn three_way_join_reordered_by_cardinality() {
        let catalog = create_test_catalog();
        let rule = JoinReorderingRule::new(&catalog);
        let arena = Bump::new();

        let large = arena.alloc(LogicalOperator::Scan(LogicalScan {
            schema: None,
            table: "large_table",
            alias: Some("l"),
        }));
        let medium = arena.alloc(LogicalOperator::Scan(LogicalScan {
            schema: None,
            table: "medium_table",
            alias: Some("m"),
        }));
        let small = arena.alloc(LogicalOperator::Scan(LogicalScan {
            schema: None,
            table: "small_table",
            alias: Some("s"),
        }));

        let first_join = arena.alloc(LogicalOperator::Join(LogicalJoin {
            left: large,
            right: medium,
            join_type: JoinType::Inner,
            condition: None,
        }));

        let second_join = arena.alloc(LogicalOperator::Join(LogicalJoin {
            left: first_join,
            right: small,
            join_type: JoinType::Inner,
            condition: None,
        }));

        let result = rule.apply(second_join, &arena).unwrap();
        assert!(result.is_some(), "join should be reordered");

        fn get_leftmost_table<'a>(plan: &'a LogicalOperator<'a>) -> &'a str {
            match plan {
                LogicalOperator::Join(join) => get_leftmost_table(join.left),
                LogicalOperator::Scan(scan) => scan.table,
                _ => panic!("unexpected operator"),
            }
        }

        let new_plan = result.unwrap();
        let leftmost = get_leftmost_table(new_plan);
        assert_eq!(
            leftmost, "small_table",
            "smallest table should be leftmost after reordering"
        );
    }

    #[test]
    fn left_join_not_reordered() {
        let catalog = create_test_catalog();
        let rule = JoinReorderingRule::new(&catalog);
        let arena = Bump::new();

        let large = arena.alloc(LogicalOperator::Scan(LogicalScan {
            schema: None,
            table: "large_table",
            alias: None,
        }));
        let small = arena.alloc(LogicalOperator::Scan(LogicalScan {
            schema: None,
            table: "small_table",
            alias: None,
        }));

        let join = arena.alloc(LogicalOperator::Join(LogicalJoin {
            left: large,
            right: small,
            join_type: JoinType::Left,
            condition: None,
        }));

        let result = rule.apply(join, &arena).unwrap();
        assert!(
            result.is_none(),
            "LEFT JOIN should not be reordered"
        );
    }

    #[test]
    fn conditions_correctly_remapped_after_reorder() {
        let catalog = create_test_catalog();
        let rule = JoinReorderingRule::new(&catalog);
        let arena = Bump::new();

        let large = arena.alloc(LogicalOperator::Scan(LogicalScan {
            schema: None,
            table: "large_table",
            alias: Some("l"),
        }));
        let small = arena.alloc(LogicalOperator::Scan(LogicalScan {
            schema: None,
            table: "small_table",
            alias: Some("s"),
        }));

        let l_id = arena.alloc(Expr::Column(ColumnRef {
            schema: None,
            table: Some("l"),
            column: "id",
        }));
        let s_id = arena.alloc(Expr::Column(ColumnRef {
            schema: None,
            table: Some("s"),
            column: "id",
        }));
        let condition = arena.alloc(Expr::BinaryOp {
            left: l_id,
            op: BinaryOperator::Eq,
            right: s_id,
        });

        let join = arena.alloc(LogicalOperator::Join(LogicalJoin {
            left: large,
            right: small,
            join_type: JoinType::Inner,
            condition: Some(condition),
        }));

        let result = rule.apply(join, &arena).unwrap();
        assert!(result.is_some(), "join should be reordered");

        let new_plan = result.unwrap();
        if let LogicalOperator::Join(new_join) = new_plan {
            assert!(
                new_join.condition.is_some(),
                "condition should be preserved after reordering"
            );
        } else {
            panic!("result should be a join");
        }
    }
}
