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
            PhysicalOperator::DualScan => Ok(DynamicExecutor::TableScan(TableScanExecutor::new(
                source,
                self.ctx.arena,
            ))),
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

                let agg_info = self.get_aggregate_info(project.input);

                let child = self.build_operator(project.input, source, column_map)?;

                let has_complex_expressions = project.expressions.iter().any(|expr| match expr {
                    Expr::Column(_) => false,
                    Expr::Function(func) => {
                        if func.over.is_some() {
                            return false;
                        }
                        let name = func.name.name.to_uppercase();
                        !matches!(name.as_str(), "COUNT" | "SUM" | "AVG" | "MIN" | "MAX")
                    }
                    _ => true,
                });

                if has_complex_expressions && !project.expressions.is_empty() {
                    let effective_column_map = if let Some((group_by, aggregates)) = &agg_info {
                        self.build_aggregate_column_map(group_by, aggregates, column_map)
                    } else {
                        self.compute_input_column_map(project.input)
                    };
                    let expressions: Vec<&'a Expr<'a>> = project.expressions.to_vec();
                    let projection = CompiledProjection::new(expressions, effective_column_map);
                    Ok(DynamicExecutor::ProjectExpr(
                        Box::new(child),
                        projection,
                        self.ctx.arena,
                    ))
                } else {
                    let projections: Vec<usize> = if project.expressions.is_empty() {
                        let output_len = if let Some((group_by, aggregates)) = &agg_info {
                            group_by.len() + aggregates.len()
                        } else {
                            column_map.len()
                        };
                        (0..output_len).collect()
                    } else {
                        project
                            .expressions
                            .iter()
                            .enumerate()
                            .map(|(default_idx, expr)| {
                                if let Some((group_by, aggregates)) = &agg_info {
                                    if let Expr::Column(col_ref) = expr {
                                        for (idx, group_expr) in group_by.iter().enumerate() {
                                            if let Expr::Column(group_col) = group_expr {
                                                if group_col
                                                    .column
                                                    .eq_ignore_ascii_case(col_ref.column)
                                                {
                                                    return idx;
                                                }
                                            }
                                        }
                                    }
                                    if let Expr::Function(FunctionCall { name, args, .. }) = expr {
                                        for (idx, agg) in aggregates.iter().enumerate() {
                                            let matches_func = match agg.function {
                                                crate::sql::planner::AggregateFunction::Count => {
                                                    name.name.eq_ignore_ascii_case("count")
                                                }
                                                crate::sql::planner::AggregateFunction::Sum => {
                                                    name.name.eq_ignore_ascii_case("sum")
                                                }
                                                crate::sql::planner::AggregateFunction::Avg => {
                                                    name.name.eq_ignore_ascii_case("avg")
                                                }
                                                crate::sql::planner::AggregateFunction::Min => {
                                                    name.name.eq_ignore_ascii_case("min")
                                                }
                                                crate::sql::planner::AggregateFunction::Max => {
                                                    name.name.eq_ignore_ascii_case("max")
                                                }
                                            };
                                            if matches_func {
                                                use crate::sql::ast::FunctionArgs;
                                                let first_arg = match args {
                                                    FunctionArgs::Args(arg_list) => {
                                                        arg_list.first().map(|a| a.value)
                                                    }
                                                    _ => None,
                                                };
                                                let args_match = match (agg.argument, first_arg) {
                                                    (
                                                        Some(Expr::Column(agg_col)),
                                                        Some(Expr::Column(arg_col)),
                                                    ) => agg_col
                                                        .column
                                                        .eq_ignore_ascii_case(arg_col.column),
                                                    (None, _) | (Some(Expr::Literal(_)), _) => true,
                                                    _ => false,
                                                };
                                                if args_match {
                                                    return group_by.len() + idx;
                                                }
                                            }
                                        }
                                    }
                                    default_idx
                                } else if let Expr::Column(col_ref) = expr {
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
                    .filter_map(|key| {
                        if let crate::sql::ast::Expr::Column(col) = key.expr {
                            column_map
                                .iter()
                                .find(|(n, _)| n.eq_ignore_ascii_case(col.column))
                                .map(|(_, idx)| SortKey {
                                    column: *idx,
                                    ascending: key.ascending,
                                })
                        } else {
                            None
                        }
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
            PhysicalOperator::SecondaryIndexScan(_) => Ok(DynamicExecutor::TableScan(
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
                eyre::bail!("SetOpExec requires special handling - use Database::query instead")
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
        spill_dir: Option<std::path::PathBuf>,
        memory_budget: usize,
        query_id: u64,
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
            use_spill: spill_dir.is_some(),
            left_spiller: None,
            right_spiller: None,
            spill_dir,
            memory_budget,
            query_id,
            probe_row_buf: smallvec::SmallVec::new(),
            build_row_buf: smallvec::SmallVec::new(),
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

    fn get_aggregate_info(
        &self,
        op: &'a crate::sql::planner::PhysicalOperator<'a>,
    ) -> Option<(
        &'a [&'a crate::sql::ast::Expr<'a>],
        &'a [crate::sql::planner::AggregateExpr<'a>],
    )> {
        use crate::sql::planner::PhysicalOperator;

        match op {
            PhysicalOperator::HashAggregate(agg) => Some((agg.group_by, agg.aggregates)),
            PhysicalOperator::SortedAggregate(agg) => Some((agg.group_by, agg.aggregates)),
            PhysicalOperator::FilterExec(filter) => self.get_aggregate_info(filter.input),
            PhysicalOperator::SortExec(sort) => self.get_aggregate_info(sort.input),
            PhysicalOperator::LimitExec(limit) => self.get_aggregate_info(limit.input),
            _ => None,
        }
    }

    fn build_aggregate_column_map(
        &self,
        group_by: &[&'a crate::sql::ast::Expr<'a>],
        aggregates: &[crate::sql::planner::AggregateExpr<'a>],
        original_column_map: &[(String, usize)],
    ) -> Vec<(String, usize)> {
        use crate::sql::ast::Expr;

        let mut result = Vec::new();

        for (idx, expr) in group_by.iter().enumerate() {
            if let Expr::Column(col) = expr {
                result.push((col.column.to_lowercase(), idx));
            }
        }

        let group_count = group_by.len();
        for (idx, agg) in aggregates.iter().enumerate() {
            if let Some(Expr::Column(col)) = agg.argument {
                let agg_name = match agg.function {
                    crate::sql::planner::AggregateFunction::Count => {
                        format!("count_{}", col.column)
                    }
                    crate::sql::planner::AggregateFunction::Sum => format!("sum_{}", col.column),
                    crate::sql::planner::AggregateFunction::Avg => format!("avg_{}", col.column),
                    crate::sql::planner::AggregateFunction::Min => format!("min_{}", col.column),
                    crate::sql::planner::AggregateFunction::Max => format!("max_{}", col.column),
                };
                result.push((agg_name.to_lowercase(), group_count + idx));
            } else {
                let agg_name = match agg.function {
                    crate::sql::planner::AggregateFunction::Count => "count".to_string(),
                    crate::sql::planner::AggregateFunction::Sum => "sum".to_string(),
                    crate::sql::planner::AggregateFunction::Avg => "avg".to_string(),
                    crate::sql::planner::AggregateFunction::Min => "min".to_string(),
                    crate::sql::planner::AggregateFunction::Max => "max".to_string(),
                };
                result.push((agg_name.to_lowercase(), group_count + idx));
            }
        }

        let _ = original_column_map;

        result
    }

    fn compute_input_column_map(
        &self,
        op: &'a crate::sql::planner::PhysicalOperator<'a>,
    ) -> Vec<(String, usize)> {
        use crate::sql::planner::PhysicalOperator;

        match op {
            PhysicalOperator::TableScan(scan) => {
                if let Some(table_def) = scan.table_def {
                    table_def
                        .columns()
                        .iter()
                        .enumerate()
                        .map(|(idx, col)| (col.name().to_string(), idx))
                        .collect()
                } else {
                    Vec::new()
                }
            }
            PhysicalOperator::DualScan => Vec::new(),
            PhysicalOperator::IndexScan(_) => Vec::new(),
            PhysicalOperator::FilterExec(filter) => self.compute_input_column_map(filter.input),
            PhysicalOperator::SortExec(sort) => self.compute_input_column_map(sort.input),
            PhysicalOperator::LimitExec(limit) => self.compute_input_column_map(limit.input),
            PhysicalOperator::ProjectExec(project) => self.compute_input_column_map(project.input),
            PhysicalOperator::WindowExec(window) => self.compute_input_column_map(window.input),
            PhysicalOperator::HashAggregate(agg) => {
                self.build_aggregate_column_map(agg.group_by, agg.aggregates, &[])
            }
            PhysicalOperator::SortedAggregate(agg) => {
                self.build_aggregate_column_map(agg.group_by, agg.aggregates, &[])
            }
            PhysicalOperator::SubqueryExec(subq) => subq
                .output_schema
                .columns
                .iter()
                .enumerate()
                .map(|(idx, col)| (col.name.to_string(), idx))
                .collect(),
            PhysicalOperator::NestedLoopJoin(join) => {
                let mut result = self.compute_input_column_map(join.left);
                let right_cols = self.compute_input_column_map(join.right);
                let offset = result.len();
                for (name, idx) in right_cols {
                    result.push((name, idx + offset));
                }
                result
            }
            PhysicalOperator::GraceHashJoin(join) => {
                let mut result = self.compute_input_column_map(join.left);
                let right_cols = self.compute_input_column_map(join.right);
                let offset = result.len();
                for (name, idx) in right_cols {
                    result.push((name, idx + offset));
                }
                result
            }
            PhysicalOperator::SetOpExec(_) => Vec::new(),
            PhysicalOperator::SecondaryIndexScan(scan) => {
                if let Some(table_def) = scan.table_def {
                    table_def
                        .columns()
                        .iter()
                        .enumerate()
                        .map(|(idx, col)| (col.name().to_lowercase(), idx))
                        .collect()
                } else {
                    Vec::new()
                }
            }
        }
    }
}
