//! Integration tests for subquery and optimizer functionality

use bumpalo::Bump;

mod optimizer_tests {
    use super::*;
    use turdb::sql::optimizer::Optimizer;
    use turdb::sql::planner::{LogicalOperator, LogicalScan};

    #[test]
    fn test_optimizer_returns_same_plan_when_no_changes() {
        let arena = Bump::new();
        let optimizer = Optimizer::new();

        let scan = arena.alloc(LogicalOperator::Scan(LogicalScan {
            schema: None,
            table: "users",
            alias: None,
        }));

        let result = optimizer.optimize(scan, &arena).unwrap();
        assert!(matches!(result, LogicalOperator::Scan(_)));
    }

    #[test]
    fn test_optimizer_handles_dual_scan() {
        let arena = Bump::new();
        let optimizer = Optimizer::new();

        let dual = arena.alloc(LogicalOperator::DualScan);
        let result = optimizer.optimize(dual, &arena).unwrap();
        assert!(matches!(result, LogicalOperator::DualScan));
    }
}

mod classifier_tests {
    use turdb::sql::subquery::SubqueryClassifier;

    #[test]
    fn test_classifier_creation() {
        let classifier = SubqueryClassifier::new();
        assert_eq!(classifier.current_depth(), 0);
    }

    #[test]
    fn test_classifier_with_outer_scope() {
        let tables = vec!["orders", "users"];
        let columns = vec![("orders", "id"), ("users", "name")];
        let classifier = SubqueryClassifier::with_outer_scope(tables, columns);
        assert_eq!(classifier.current_depth(), 0);
    }

    #[test]
    fn test_classifier_depth_tracking() {
        let mut classifier = SubqueryClassifier::new();
        assert_eq!(classifier.current_depth(), 0);
        classifier.increment_depth();
        assert_eq!(classifier.current_depth(), 1);
        classifier.increment_depth();
        assert_eq!(classifier.current_depth(), 2);
    }
}

mod context_tests {
    use turdb::sql::subquery::SubqueryContext;
    use turdb::types::OwnedValue;

    #[test]
    fn test_context_scope_management() {
        let mut ctx = SubqueryContext::new();
        assert_eq!(ctx.current_depth(), 0);
        assert!(!ctx.has_bindings());

        ctx.push_scope();
        assert_eq!(ctx.current_depth(), 1);

        ctx.bind("users", "id", OwnedValue::Int(42));
        assert!(ctx.has_bindings());

        ctx.pop_scope();
        assert_eq!(ctx.current_depth(), 0);
        assert!(!ctx.has_bindings());
    }

    #[test]
    fn test_context_nested_resolution() {
        let mut ctx = SubqueryContext::new();

        ctx.push_scope();
        ctx.bind("outer", "id", OwnedValue::Int(1));

        ctx.push_scope();
        ctx.bind("inner", "id", OwnedValue::Int(100));

        assert_eq!(
            *ctx.resolve_qualified("outer", "id").unwrap(),
            OwnedValue::Int(1)
        );
        assert_eq!(
            *ctx.resolve_qualified("inner", "id").unwrap(),
            OwnedValue::Int(100)
        );

        ctx.pop_scope();
        assert!(ctx.resolve_qualified("inner", "id").is_none());
        assert!(ctx.resolve_qualified("outer", "id").is_some());
    }

    #[test]
    fn test_context_unqualified_resolution() {
        let mut ctx = SubqueryContext::new();
        ctx.push_scope();
        ctx.bind("table", "value", OwnedValue::Float(3.14));

        let resolved = ctx.resolve(None, "value");
        assert!(resolved.is_some());
        assert_eq!(*resolved.unwrap(), OwnedValue::Float(3.14));
    }
}

mod spill_tests {
    use turdb::sql::subquery::{MaterializedRow, SpillableBuffer};
    use turdb::types::OwnedValue;

    #[test]
    fn test_buffer_in_memory() {
        let mut buffer = SpillableBuffer::new(1024 * 1024);

        for i in 0..10 {
            buffer
                .push(MaterializedRow::new(vec![OwnedValue::Int(i)]))
                .unwrap();
        }

        assert!(!buffer.is_spilled());
        assert_eq!(buffer.row_count(), 10);

        let rows: Vec<_> = buffer.iter().unwrap().map(|r| r.unwrap()).collect();
        assert_eq!(rows.len(), 10);

        for (i, row) in rows.iter().enumerate() {
            assert_eq!(row.values[0], OwnedValue::Int(i as i64));
        }
    }

    #[test]
    fn test_buffer_spills_on_limit() {
        let mut buffer = SpillableBuffer::new(100);

        for i in 0..50 {
            buffer
                .push(MaterializedRow::new(vec![
                    OwnedValue::Int(i),
                    OwnedValue::Text(format!("row_{}", i)),
                ]))
                .unwrap();
        }

        assert!(buffer.is_spilled());
        assert_eq!(buffer.row_count(), 50);

        let rows: Vec<_> = buffer.iter().unwrap().map(|r| r.unwrap()).collect();
        assert_eq!(rows.len(), 50);
        assert_eq!(rows[0].values[0], OwnedValue::Int(0));
        assert_eq!(rows[49].values[0], OwnedValue::Int(49));
    }

    #[test]
    fn test_buffer_into_vec() {
        let mut buffer = SpillableBuffer::new(1024 * 1024);

        buffer
            .push(MaterializedRow::new(vec![OwnedValue::Int(1)]))
            .unwrap();
        buffer
            .push(MaterializedRow::new(vec![OwnedValue::Int(2)]))
            .unwrap();
        buffer
            .push(MaterializedRow::new(vec![OwnedValue::Int(3)]))
            .unwrap();

        let vec = buffer.into_vec().unwrap();
        assert_eq!(vec.len(), 3);
    }
}

mod rule_tests {
    use super::*;
    use turdb::sql::optimizer::rules::{
        ConstantFoldingRule, PredicatePushdownRule, ProjectionPruningRule,
        SubqueryDecorrelationRule,
    };
    use turdb::sql::optimizer::OptimizationRule;
    use turdb::sql::planner::LogicalOperator;

    #[test]
    fn test_constant_folding_rule_name() {
        let rule = ConstantFoldingRule;
        assert_eq!(rule.name(), "constant_folding");
    }

    #[test]
    fn test_predicate_pushdown_rule_name() {
        let rule = PredicatePushdownRule;
        assert_eq!(rule.name(), "predicate_pushdown");
    }

    #[test]
    fn test_projection_pruning_rule_name() {
        let rule = ProjectionPruningRule;
        assert_eq!(rule.name(), "projection_pruning");
    }

    #[test]
    fn test_decorrelation_rule_name() {
        let rule = SubqueryDecorrelationRule;
        assert_eq!(rule.name(), "subquery_decorrelation");
    }

    #[test]
    fn test_constant_folding_no_change_on_scan() {
        let arena = Bump::new();
        let rule = ConstantFoldingRule;

        let scan = arena.alloc(LogicalOperator::Scan(turdb::sql::planner::LogicalScan {
            schema: None,
            table: "test",
            alias: None,
        }));

        let result = rule.apply(scan, &arena).unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn test_predicate_pushdown_no_change_on_scan() {
        let arena = Bump::new();
        let rule = PredicatePushdownRule;

        let scan = arena.alloc(LogicalOperator::Scan(turdb::sql::planner::LogicalScan {
            schema: None,
            table: "test",
            alias: None,
        }));

        let result = rule.apply(scan, &arena).unwrap();
        assert!(result.is_none());
    }
}
