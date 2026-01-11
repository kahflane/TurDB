//! Tests for Semi and Anti join operators

use bumpalo::Bump;
use turdb::sql::ast::JoinType;
use turdb::sql::planner::{LogicalJoin, LogicalOperator, LogicalScan};

mod semi_join_tests {
    use super::*;

    #[test]
    fn test_semi_join_type_exists() {
        assert_eq!(JoinType::Semi, JoinType::Semi);
    }

    #[test]
    fn test_anti_join_type_exists() {
        assert_eq!(JoinType::Anti, JoinType::Anti);
    }

    #[test]
    fn test_logical_join_with_semi_type() {
        let arena = Bump::new();

        let left = arena.alloc(LogicalOperator::Scan(LogicalScan {
            schema: None,
            table: "orders",
            alias: None,
        }));

        let right = arena.alloc(LogicalOperator::Scan(LogicalScan {
            schema: None,
            table: "items",
            alias: None,
        }));

        let join = LogicalJoin {
            left,
            right,
            join_type: JoinType::Semi,
            condition: None,
        };

        assert_eq!(join.join_type, JoinType::Semi);
    }

    #[test]
    fn test_logical_join_with_anti_type() {
        let arena = Bump::new();

        let left = arena.alloc(LogicalOperator::Scan(LogicalScan {
            schema: None,
            table: "orders",
            alias: None,
        }));

        let right = arena.alloc(LogicalOperator::Scan(LogicalScan {
            schema: None,
            table: "items",
            alias: None,
        }));

        let join = LogicalJoin {
            left,
            right,
            join_type: JoinType::Anti,
            condition: None,
        };

        assert_eq!(join.join_type, JoinType::Anti);
    }
}

mod decorrelation_tests {
    use turdb::sql::optimizer::rules::SubqueryDecorrelationRule;
    use turdb::sql::optimizer::OptimizationRule;

    #[test]
    fn test_decorrelation_rule_name() {
        let rule = SubqueryDecorrelationRule;
        assert_eq!(rule.name(), "subquery_decorrelation");
    }
}

mod physical_operator_tests {
    use bumpalo::Bump;
    use turdb::sql::planner::{
        PhysicalHashAntiJoin, PhysicalHashSemiJoin, PhysicalOperator, PhysicalTableScan,
    };

    #[test]
    fn test_physical_hash_semi_join_creation() {
        let arena = Bump::new();

        let left = arena.alloc(PhysicalOperator::TableScan(PhysicalTableScan {
            schema: None,
            table: "orders",
            alias: None,
            post_scan_filter: None,
            table_def: None,
            reverse: false,
        }));

        let right = arena.alloc(PhysicalOperator::TableScan(PhysicalTableScan {
            schema: None,
            table: "items",
            alias: None,
            post_scan_filter: None,
            table_def: None,
            reverse: false,
        }));

        let semi_join = PhysicalHashSemiJoin {
            left,
            right,
            join_keys: &[],
        };

        let _op = PhysicalOperator::HashSemiJoin(semi_join);
    }

    #[test]
    fn test_physical_hash_anti_join_creation() {
        let arena = Bump::new();

        let left = arena.alloc(PhysicalOperator::TableScan(PhysicalTableScan {
            schema: None,
            table: "orders",
            alias: None,
            post_scan_filter: None,
            table_def: None,
            reverse: false,
        }));

        let right = arena.alloc(PhysicalOperator::TableScan(PhysicalTableScan {
            schema: None,
            table: "items",
            alias: None,
            post_scan_filter: None,
            table_def: None,
            reverse: false,
        }));

        let anti_join = PhysicalHashAntiJoin {
            left,
            right,
            join_keys: &[],
        };

        let _op = PhysicalOperator::HashAntiJoin(anti_join);
    }
}

mod subquery_exec_tests {
    use bumpalo::Bump;
    use turdb::sql::planner::{
        PhysicalExistsSubqueryExec, PhysicalInListSubqueryExec, PhysicalOperator,
        PhysicalScalarSubqueryExec, PhysicalTableScan,
    };

    #[test]
    fn test_scalar_subquery_exec_creation() {
        let arena = Bump::new();

        let subquery = arena.alloc(PhysicalOperator::TableScan(PhysicalTableScan {
            schema: None,
            table: "products",
            alias: None,
            post_scan_filter: None,
            table_def: None,
            reverse: false,
        }));

        let scalar_subq = PhysicalScalarSubqueryExec {
            subquery,
            is_correlated: false,
        };

        let _op = PhysicalOperator::ScalarSubqueryExec(scalar_subq);
    }

    #[test]
    fn test_exists_subquery_exec_creation() {
        let arena = Bump::new();

        let subquery = arena.alloc(PhysicalOperator::TableScan(PhysicalTableScan {
            schema: None,
            table: "items",
            alias: None,
            post_scan_filter: None,
            table_def: None,
            reverse: false,
        }));

        let exists_subq = PhysicalExistsSubqueryExec {
            subquery,
            negated: false,
            is_correlated: true,
        };

        let _op = PhysicalOperator::ExistsSubqueryExec(exists_subq);
    }

    #[test]
    fn test_exists_subquery_negated() {
        let arena = Bump::new();

        let subquery = arena.alloc(PhysicalOperator::TableScan(PhysicalTableScan {
            schema: None,
            table: "banned_users",
            alias: None,
            post_scan_filter: None,
            table_def: None,
            reverse: false,
        }));

        let not_exists_subq = PhysicalExistsSubqueryExec {
            subquery,
            negated: true,
            is_correlated: true,
        };

        assert!(not_exists_subq.negated);
    }

    #[test]
    fn test_in_list_subquery_exec_creation() {
        let arena = Bump::new();
        use turdb::sql::ast::{ColumnRef, Expr};

        let expr = arena.alloc(Expr::Column(ColumnRef {
            schema: None,
            table: None,
            column: "id",
        }));

        let subquery = arena.alloc(PhysicalOperator::TableScan(PhysicalTableScan {
            schema: None,
            table: "valid_ids",
            alias: None,
            post_scan_filter: None,
            table_def: None,
            reverse: false,
        }));

        let in_subq = PhysicalInListSubqueryExec {
            expr,
            subquery,
            negated: false,
            is_correlated: false,
        };

        let _op = PhysicalOperator::InListSubqueryExec(in_subq);
    }
}
