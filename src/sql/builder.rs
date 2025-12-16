use crate::sql::adapter::BTreeCursorAdapter;
use crate::sql::ast::JoinType;
use crate::sql::context::ExecutionContext;
use crate::sql::executor::{
    AggregateFunction, DynamicExecutor, RowSource, SortKey, TableScanExecutor,
};
use crate::sql::predicate::CompiledPredicate;
use crate::sql::state::{
    GraceHashJoinState, HashAggregateState, IndexScanState, LimitState, NestedLoopJoinState,
    SortState, WindowState,
};

pub struct ExecutorBuilder<'a> {
    ctx: &'a ExecutionContext<'a>,
}

impl<'a> ExecutorBuilder<'a> {
    pub fn new(ctx: &'a ExecutionContext<'a>) -> Self {
        Self { ctx }
    }

    pub fn build_with_source<S: RowSource>(
        &self,
        plan: &crate::sql::planner::PhysicalPlan<'a>,
        source: S,
    ) -> eyre::Result<DynamicExecutor<'a, S>> {
        let column_map: Vec<(String, usize)> = plan
            .output_schema
            .columns
            .iter()
            .enumerate()
            .map(|(idx, col)| (col.name.to_string(), idx))
            .collect();

        self.build_operator(plan.root, source, &column_map)
    }

    pub fn build_with_source_and_column_map<S: RowSource>(
        &self,
        plan: &crate::sql::planner::PhysicalPlan<'a>,
        source: S,
        column_map: &[(String, usize)],
    ) -> eyre::Result<DynamicExecutor<'a, S>> {
        self.build_operator(plan.root, source, column_map)
    }

    fn build_operator<S: RowSource>(
        &self,
        op: &'a crate::sql::planner::PhysicalOperator<'a>,
        source: S,
        column_map: &[(String, usize)],
    ) -> eyre::Result<DynamicExecutor<'a, S>> {
        use crate::sql::planner::PhysicalOperator;

        match op {
            PhysicalOperator::TableScan(_) => Ok(DynamicExecutor::TableScan(
                TableScanExecutor::new(source, self.ctx.arena),
            )),
            PhysicalOperator::FilterExec(filter) => {
                let child = self.build_operator(filter.input, source, column_map)?;
                let predicate = CompiledPredicate::new(filter.predicate, column_map.to_vec());
                Ok(DynamicExecutor::Filter(Box::new(child), predicate))
            }
            PhysicalOperator::ProjectExec(project) => {
                use crate::sql::ast::{Expr, FunctionCall};
                use crate::sql::predicate::CompiledProjection;

                let window_functions = self.find_window_functions_in_input(project.input);
                let base_col_count =
                    self.count_base_columns_before_window(project.input, column_map.len());

                let child = self.build_operator(project.input, source, column_map)?;

                let has_complex_expressions = project.expressions.iter().any(|expr| {
                    !matches!(expr, Expr::Column(_)) && !matches!(expr, Expr::Function(_))
                });

                if has_complex_expressions && !project.expressions.is_empty() {
                    let expressions: Vec<&'a Expr<'a>> = project.expressions.to_vec();
                    let projection = CompiledProjection::new(expressions, column_map.to_vec());
                    Ok(DynamicExecutor::ProjectExpr(
                        Box::new(child),
                        projection,
                        self.ctx.arena,
                    ))
                } else {
                    let projections: Vec<usize> = if project.expressions.is_empty() {
                        (0..column_map.len()).collect()
                    } else {
                        project
                            .expressions
                            .iter()
                            .enumerate()
                            .map(|(default_idx, expr)| {
                                if let Expr::Column(col_ref) = expr {
                                    column_map
                                        .iter()
                                        .find(|(name, _)| name.eq_ignore_ascii_case(col_ref.column))
                                        .map(|(_, idx)| *idx)
                                        .unwrap_or(default_idx)
                                } else if let Expr::Function(FunctionCall {
                                    over: Some(_), ..
                                }) = expr
                                {
                                    if let Some(window_idx) =
                                        self.find_window_function_index(expr, window_functions)
                                    {
                                        base_col_count + window_idx
                                    } else {
                                        default_idx
                                    }
                                } else {
                                    default_idx
                                }
                            })
                            .collect()
                    };
                    Ok(DynamicExecutor::Project(
                        Box::new(child),
                        projections,
                        self.ctx.arena,
                    ))
                }
            }
            PhysicalOperator::LimitExec(limit) => {
                let child = self.build_operator(limit.input, source, column_map)?;
                Ok(DynamicExecutor::Limit(LimitState {
                    child: Box::new(child),
                    limit: limit.limit,
                    offset: limit.offset,
                    skipped: 0,
                    returned: 0,
                }))
            }
            PhysicalOperator::SortExec(sort) => {
                let child = self.build_operator(sort.input, source, column_map)?;
                let sort_keys: Vec<SortKey> = sort
                    .order_by
                    .iter()
                    .enumerate()
                    .map(|(idx, key)| SortKey {
                        column: idx,
                        ascending: key.ascending,
                    })
                    .collect();
                Ok(DynamicExecutor::Sort(SortState {
                    child: Box::new(child),
                    sort_keys,
                    arena: self.ctx.arena,
                    rows: Vec::new(),
                    iter_idx: 0,
                    sorted: false,
                }))
            }
            PhysicalOperator::IndexScan(_) => Ok(DynamicExecutor::TableScan(
                TableScanExecutor::new(source, self.ctx.arena),
            )),
            PhysicalOperator::HashAggregate(agg) => {
                let child = self.build_operator(agg.input, source, column_map)?;
                let group_by_indices: Vec<usize> = agg
                    .group_by
                    .iter()
                    .filter_map(|expr| {
                        if let crate::sql::ast::Expr::Column(col) = expr {
                            column_map
                                .iter()
                                .find(|(n, _)| n.eq_ignore_ascii_case(col.column))
                                .map(|(_, i)| *i)
                        } else {
                            None
                        }
                    })
                    .collect();
                let agg_funcs: Vec<AggregateFunction> = agg
                    .aggregates
                    .iter()
                    .map(|agg_expr| {
                        let column_idx = agg_expr
                            .argument
                            .and_then(|arg| {
                                if let crate::sql::ast::Expr::Column(col) = arg {
                                    column_map
                                        .iter()
                                        .find(|(n, _)| n.eq_ignore_ascii_case(col.column))
                                        .map(|(_, i)| *i)
                                } else {
                                    None
                                }
                            })
                            .unwrap_or(0);
                        match agg_expr.function {
                            crate::sql::planner::AggregateFunction::Count => {
                                AggregateFunction::Count {
                                    distinct: agg_expr.distinct,
                                }
                            }
                            crate::sql::planner::AggregateFunction::Sum => {
                                AggregateFunction::Sum { column: column_idx }
                            }
                            crate::sql::planner::AggregateFunction::Avg => {
                                AggregateFunction::Avg { column: column_idx }
                            }
                            crate::sql::planner::AggregateFunction::Min => {
                                AggregateFunction::Min { column: column_idx }
                            }
                            crate::sql::planner::AggregateFunction::Max => {
                                AggregateFunction::Max { column: column_idx }
                            }
                        }
                    })
                    .collect();
                Ok(DynamicExecutor::HashAggregate(HashAggregateState {
                    child: Box::new(child),
                    group_by: group_by_indices,
                    aggregates: agg_funcs,
                    arena: self.ctx.arena,
                    groups: hashbrown::HashMap::new(),
                    result_iter: None,
                    computed: false,
                }))
            }
            PhysicalOperator::SortedAggregate(agg) => {
                let child = self.build_operator(agg.input, source, column_map)?;
                let group_by_indices: Vec<usize> = agg
                    .group_by
                    .iter()
                    .filter_map(|expr| {
                        if let crate::sql::ast::Expr::Column(col) = expr {
                            column_map
                                .iter()
                                .find(|(n, _)| n.eq_ignore_ascii_case(col.column))
                                .map(|(_, i)| *i)
                        } else {
                            None
                        }
                    })
                    .collect();
                let agg_funcs: Vec<AggregateFunction> = agg
                    .aggregates
                    .iter()
                    .map(|agg_expr| {
                        let column_idx = agg_expr
                            .argument
                            .and_then(|arg| {
                                if let crate::sql::ast::Expr::Column(col) = arg {
                                    column_map
                                        .iter()
                                        .find(|(n, _)| n.eq_ignore_ascii_case(col.column))
                                        .map(|(_, i)| *i)
                                } else {
                                    None
                                }
                            })
                            .unwrap_or(0);
                        match agg_expr.function {
                            crate::sql::planner::AggregateFunction::Count => {
                                AggregateFunction::Count {
                                    distinct: agg_expr.distinct,
                                }
                            }
                            crate::sql::planner::AggregateFunction::Sum => {
                                AggregateFunction::Sum { column: column_idx }
                            }
                            crate::sql::planner::AggregateFunction::Avg => {
                                AggregateFunction::Avg { column: column_idx }
                            }
                            crate::sql::planner::AggregateFunction::Min => {
                                AggregateFunction::Min { column: column_idx }
                            }
                            crate::sql::planner::AggregateFunction::Max => {
                                AggregateFunction::Max { column: column_idx }
                            }
                        }
                    })
                    .collect();
                Ok(DynamicExecutor::HashAggregate(HashAggregateState {
                    child: Box::new(child),
                    group_by: group_by_indices,
                    aggregates: agg_funcs,
                    arena: self.ctx.arena,
                    groups: hashbrown::HashMap::new(),
                    result_iter: None,
                    computed: false,
                }))
            }
            PhysicalOperator::NestedLoopJoin(_) => {
                eyre::bail!(
                    "NestedLoopJoin requires two sources - use build_nested_loop_join instead"
                )
            }
            PhysicalOperator::GraceHashJoin(_) => {
                eyre::bail!(
                    "GraceHashJoin requires two sources - use build_grace_hash_join instead"
                )
            }
            PhysicalOperator::SubqueryExec(_) => Ok(DynamicExecutor::TableScan(
                TableScanExecutor::new(source, self.ctx.arena),
            )),
            PhysicalOperator::WindowExec(window) => {
                let child = self.build_operator(window.input, source, column_map)?;
                Ok(DynamicExecutor::Window(WindowState::new_with_column_map(
                    Box::new(child),
                    window.window_functions,
                    self.ctx.arena,
                    column_map.to_vec(),
                )))
            }
            PhysicalOperator::SetOpExec(_) => {
                eyre::bail!(
                    "SetOpExec requires special handling - use Database::query instead"
                )
            }
        }
    }

    pub fn build_index_scan(
        &self,
        index_scan: &'a crate::sql::planner::PhysicalIndexScan<'a>,
        adapter: BTreeCursorAdapter,
        column_map: &[(String, usize)],
    ) -> eyre::Result<IndexScanState<'a>> {
        let residual_filter = index_scan
            .residual_filter
            .map(|expr| CompiledPredicate::new(expr, column_map.to_vec()));

        Ok(IndexScanState::new(
            adapter,
            self.ctx.arena,
            residual_filter,
        ))
    }

    #[allow(clippy::too_many_arguments)]
    pub fn build_nested_loop_join<S: RowSource>(
        &self,
        left: DynamicExecutor<'a, S>,
        right: DynamicExecutor<'a, S>,
        condition: Option<&'a crate::sql::ast::Expr<'a>>,
        column_map: &[(String, usize)],
        join_type: JoinType,
        left_col_count: usize,
        right_col_count: usize,
    ) -> NestedLoopJoinState<'a, S> {
        let compiled_condition =
            condition.map(|expr| CompiledPredicate::new(expr, column_map.to_vec()));
        NestedLoopJoinState {
            left: Box::new(left),
            right: Box::new(right),
            condition: compiled_condition,
            arena: self.ctx.arena,
            current_left_row: None,
            right_rows: Vec::new(),
            right_index: 0,
            materialized: false,
            join_type,
            left_matched: false,
            right_matched: Vec::new(),
            emitting_unmatched_right: false,
            unmatched_right_idx: 0,
            left_col_count,
            right_col_count,
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn build_grace_hash_join<S: RowSource>(
        &self,
        left: DynamicExecutor<'a, S>,
        right: DynamicExecutor<'a, S>,
        left_key_indices: Vec<usize>,
        right_key_indices: Vec<usize>,
        num_partitions: usize,
        join_type: JoinType,
        left_col_count: usize,
        right_col_count: usize,
    ) -> GraceHashJoinState<'a, S> {
        GraceHashJoinState {
            left: Box::new(left),
            right: Box::new(right),
            left_key_indices,
            right_key_indices,
            arena: self.ctx.arena,
            num_partitions,
            left_partitions: (0..num_partitions).map(|_| Vec::new()).collect(),
            right_partitions: (0..num_partitions).map(|_| Vec::new()).collect(),
            current_partition: 0,
            partition_hash_table: hashbrown::HashMap::new(),
            partition_build_rows: Vec::new(),
            current_probe_idx: 0,
            current_match_idx: 0,
            current_matches: Vec::new(),
            partitioned: false,
            join_type,
            build_matched: (0..num_partitions).map(|_| Vec::new()).collect(),
            probe_row_matched: false,
            emitting_unmatched_build: false,
            unmatched_build_partition: 0,
            unmatched_build_idx: 0,
            left_col_count,
            right_col_count,
        }
    }

    pub fn build_hash_aggregate<S: RowSource>(
        &self,
        child: DynamicExecutor<'a, S>,
        group_by: Vec<usize>,
        aggregates: Vec<AggregateFunction>,
    ) -> HashAggregateState<'a, S> {
        HashAggregateState {
            child: Box::new(child),
            group_by,
            aggregates,
            arena: self.ctx.arena,
            groups: hashbrown::HashMap::new(),
            result_iter: None,
            computed: false,
        }
    }

    fn find_window_functions_in_input(
        &self,
        op: &'a crate::sql::planner::PhysicalOperator<'a>,
    ) -> &'a [crate::sql::planner::WindowFunctionDef<'a>] {
        use crate::sql::planner::PhysicalOperator;

        match op {
            PhysicalOperator::WindowExec(window) => window.window_functions,
            PhysicalOperator::FilterExec(filter) => {
                self.find_window_functions_in_input(filter.input)
            }
            PhysicalOperator::SortExec(sort) => self.find_window_functions_in_input(sort.input),
            PhysicalOperator::LimitExec(limit) => self.find_window_functions_in_input(limit.input),
            _ => &[],
        }
    }

    fn count_base_columns_before_window(
        &self,
        op: &'a crate::sql::planner::PhysicalOperator<'a>,
        default: usize,
    ) -> usize {
        use crate::sql::planner::PhysicalOperator;

        match op {
            PhysicalOperator::WindowExec(window) => {
                self.count_base_columns_before_window(window.input, default)
            }
            PhysicalOperator::TableScan(_) => default,
            PhysicalOperator::FilterExec(filter) => {
                self.count_base_columns_before_window(filter.input, default)
            }
            PhysicalOperator::SortExec(sort) => {
                self.count_base_columns_before_window(sort.input, default)
            }
            PhysicalOperator::LimitExec(limit) => {
                self.count_base_columns_before_window(limit.input, default)
            }
            _ => default,
        }
    }

    fn find_window_function_index(
        &self,
        expr: &crate::sql::ast::Expr<'a>,
        window_functions: &[crate::sql::planner::WindowFunctionDef<'a>],
    ) -> Option<usize> {
        use crate::sql::ast::{Expr, FunctionCall};

        if let Expr::Function(FunctionCall {
            name,
            over: Some(_),
            ..
        }) = expr
        {
            for (idx, wf) in window_functions.iter().enumerate() {
                if wf.function_name.eq_ignore_ascii_case(name.name) {
                    return Some(idx);
                }
            }
        }
        None
    }
}
