//! # Type Inference for Query Plans
//!
//! This module handles type inference for expressions and output schema computation
//! for both logical and physical operators.
//!
//! ## Type Inference
//!
//! - **Expression Types**: Infer data types from column references, literals, operators
//! - **Aggregate Types**: Map aggregate functions to their output types
//! - **Nullable Analysis**: Track nullability through expressions
//!
//! ## Output Schema Computation
//!
//! The output schema defines the columns produced by each operator:
//!
//! - **Scans**: Columns from table definition
//! - **Projects**: Types from projected expressions
//! - **Joins**: Union of left and right schemas
//! - **Aggregates**: Group-by columns + aggregate result columns
//!
//! ## Usage
//!
//! ```ignore
//! let schema = planner.compute_output_schema(physical_op)?;
//! let (name, dtype, nullable) = planner.infer_expr_type(expr, &schema)?;
//! ```

use crate::records::types::DataType;
use crate::sql::ast::{Expr, Literal};
use super::logical::LogicalOperator;
use super::physical::{AggregateExpr, AggregateFunction, PhysicalOperator};
use super::schema::{OutputColumn, OutputSchema};
use super::Planner;
use eyre::Result;

impl<'a> Planner<'a> {
    pub(crate) fn compute_output_schema(
        &self,
        op: &'a PhysicalOperator<'a>,
    ) -> Result<OutputSchema<'a>> {
        match op {
            PhysicalOperator::TableScan(scan) => {
                let table_def = if let Some(td) = scan.table_def {
                    td
                } else {
                    let table_name = if let Some(schema) = scan.schema {
                        self.arena.alloc_str(&format!("{}.{}", schema, scan.table))
                    } else {
                        scan.table
                    };
                    self.catalog.resolve_table(table_name)?
                };

                let mut columns = bumpalo::collections::Vec::new_in(self.arena);

                for col in table_def.columns() {
                    columns.push(OutputColumn {
                        name: self.arena.alloc_str(col.name()),
                        data_type: col.data_type(),
                        nullable: col.is_nullable(),
                    });
                }

                Ok(OutputSchema {
                    columns: columns.into_bump_slice(),
                })
            }
            PhysicalOperator::DualScan => Ok(OutputSchema { columns: &[] }),
            PhysicalOperator::IndexScan(scan) => {
                let table_name = if let Some(schema) = scan.schema {
                    self.arena.alloc_str(&format!("{}.{}", schema, scan.table))
                } else {
                    scan.table
                };

                let table_def = self.catalog.resolve_table(table_name)?;
                let mut columns = bumpalo::collections::Vec::new_in(self.arena);

                for col in table_def.columns() {
                    columns.push(OutputColumn {
                        name: self.arena.alloc_str(col.name()),
                        data_type: col.data_type(),
                        nullable: col.is_nullable(),
                    });
                }

                Ok(OutputSchema {
                    columns: columns.into_bump_slice(),
                })
            }
            PhysicalOperator::SecondaryIndexScan(scan) => {
                let table_def = scan.table_def.ok_or_else(|| {
                    eyre::eyre!("SecondaryIndexScan missing table_def for {}", scan.table)
                })?;
                let mut columns = bumpalo::collections::Vec::new_in(self.arena);

                for col in table_def.columns() {
                    columns.push(OutputColumn {
                        name: self.arena.alloc_str(col.name()),
                        data_type: col.data_type(),
                        nullable: col.is_nullable(),
                    });
                }

                Ok(OutputSchema {
                    columns: columns.into_bump_slice(),
                })
            }
            PhysicalOperator::FilterExec(filter) => self.compute_output_schema(filter.input),
            PhysicalOperator::LimitExec(limit) => self.compute_output_schema(limit.input),
            PhysicalOperator::SortExec(sort) => self.compute_output_schema(sort.input),
            PhysicalOperator::TopKExec(topk) => self.compute_output_schema(topk.input),
            PhysicalOperator::ProjectExec(project) => {
                let input_schema = self.compute_output_schema(project.input)?;

                if project.expressions.is_empty() {
                    return Ok(input_schema);
                }

                let mut columns = bumpalo::collections::Vec::new_in(self.arena);

                for (i, expr) in project.expressions.iter().enumerate() {
                    let (name, data_type, nullable) = self.infer_expr_type(expr, &input_schema)?;
                    let col_name = if let Some(alias) = project.aliases.get(i).and_then(|a| *a) {
                        alias
                    } else {
                        name
                    };
                    columns.push(OutputColumn {
                        name: col_name,
                        data_type,
                        nullable,
                    });
                }

                Ok(OutputSchema {
                    columns: columns.into_bump_slice(),
                })
            }
            PhysicalOperator::NestedLoopJoin(join) => {
                let left_schema = self.compute_output_schema(join.left)?;
                let right_schema = self.compute_output_schema(join.right)?;
                let mut columns = bumpalo::collections::Vec::new_in(self.arena);

                for col in left_schema.columns {
                    columns.push(*col);
                }
                for col in right_schema.columns {
                    columns.push(*col);
                }

                Ok(OutputSchema {
                    columns: columns.into_bump_slice(),
                })
            }
            PhysicalOperator::GraceHashJoin(join) => {
                let left_schema = self.compute_output_schema(join.left)?;
                let right_schema = self.compute_output_schema(join.right)?;
                let mut columns = bumpalo::collections::Vec::new_in(self.arena);

                for col in left_schema.columns {
                    columns.push(*col);
                }
                for col in right_schema.columns {
                    columns.push(*col);
                }

                Ok(OutputSchema {
                    columns: columns.into_bump_slice(),
                })
            }
            PhysicalOperator::HashAggregate(agg) => {
                let mut columns = bumpalo::collections::Vec::new_in(self.arena);
                let input_schema = self.compute_output_schema(agg.input)?;

                for group_expr in agg.group_by.iter() {
                    let (name, data_type, nullable) =
                        self.infer_expr_type(group_expr, &input_schema)?;
                    columns.push(OutputColumn {
                        name,
                        data_type,
                        nullable,
                    });
                }

                for agg_expr in agg.aggregates.iter() {
                    let (name, data_type) = self.infer_aggregate_type(agg_expr, &input_schema)?;
                    columns.push(OutputColumn {
                        name,
                        data_type,
                        nullable: true,
                    });
                }

                Ok(OutputSchema {
                    columns: columns.into_bump_slice(),
                })
            }
            PhysicalOperator::SortedAggregate(agg) => {
                let mut columns = bumpalo::collections::Vec::new_in(self.arena);
                let input_schema = self.compute_output_schema(agg.input)?;

                for group_expr in agg.group_by.iter() {
                    let (name, data_type, nullable) =
                        self.infer_expr_type(group_expr, &input_schema)?;
                    columns.push(OutputColumn {
                        name,
                        data_type,
                        nullable,
                    });
                }

                for agg_expr in agg.aggregates.iter() {
                    let (name, data_type) = self.infer_aggregate_type(agg_expr, &input_schema)?;
                    columns.push(OutputColumn {
                        name,
                        data_type,
                        nullable: true,
                    });
                }

                Ok(OutputSchema {
                    columns: columns.into_bump_slice(),
                })
            }
            PhysicalOperator::SubqueryExec(subq) => Ok(subq.output_schema.clone()),
            PhysicalOperator::SetOpExec(set_op) => self.compute_output_schema(set_op.left),
            PhysicalOperator::WindowExec(window) => {
                let input_schema = self.compute_output_schema(window.input)?;
                let mut columns = bumpalo::collections::Vec::new_in(self.arena);

                for col in input_schema.columns {
                    columns.push(*col);
                }

                for window_func in window.window_functions.iter() {
                    let name = window_func.alias.unwrap_or(window_func.function_name);
                    columns.push(OutputColumn {
                        name,
                        data_type: crate::records::types::DataType::Int8,
                        nullable: false,
                    });
                }

                Ok(OutputSchema {
                    columns: columns.into_bump_slice(),
                })
            }
            PhysicalOperator::HashSemiJoin(join) => self.compute_output_schema(join.left),
            PhysicalOperator::HashAntiJoin(join) => self.compute_output_schema(join.left),
            PhysicalOperator::ScalarSubqueryExec(subq) => {
                let subq_schema = self.compute_output_schema(subq.subquery)?;
                if subq_schema.columns.is_empty() {
                    Ok(OutputSchema { columns: &[] })
                } else {
                    let mut columns = bumpalo::collections::Vec::new_in(self.arena);
                    columns.push(subq_schema.columns[0]);
                    Ok(OutputSchema {
                        columns: columns.into_bump_slice(),
                    })
                }
            }
            PhysicalOperator::ExistsSubqueryExec(_) => {
                let mut columns = bumpalo::collections::Vec::new_in(self.arena);
                columns.push(OutputColumn {
                    name: "exists_result",
                    data_type: crate::records::types::DataType::Bool,
                    nullable: false,
                });
                Ok(OutputSchema {
                    columns: columns.into_bump_slice(),
                })
            }
            PhysicalOperator::InListSubqueryExec(_) => {
                let mut columns = bumpalo::collections::Vec::new_in(self.arena);
                columns.push(OutputColumn {
                    name: "in_result",
                    data_type: crate::records::types::DataType::Bool,
                    nullable: false,
                });
                Ok(OutputSchema {
                    columns: columns.into_bump_slice(),
                })
            }
        }
    }

    pub(crate) fn compute_logical_output_schema(
        &self,
        op: &'a LogicalOperator<'a>,
    ) -> Result<OutputSchema<'a>> {
        match op {
            LogicalOperator::Scan(scan) => {
                let table_name = if let Some(schema) = scan.schema {
                    self.arena.alloc_str(&format!("{}.{}", schema, scan.table))
                } else {
                    scan.table
                };
                let table_def = self.catalog.resolve_table(table_name)?;
                let mut columns = bumpalo::collections::Vec::new_in(self.arena);

                for col in table_def.columns() {
                    columns.push(OutputColumn {
                        name: self.arena.alloc_str(col.name()),
                        data_type: col.data_type(),
                        nullable: col.is_nullable(),
                    });
                }

                Ok(OutputSchema {
                    columns: columns.into_bump_slice(),
                })
            }
            LogicalOperator::DualScan => Ok(OutputSchema { columns: &[] }),
            LogicalOperator::Project(project) => {
                let input_schema = self.compute_logical_output_schema(project.input)?;

                if project.expressions.is_empty() {
                    return Ok(input_schema);
                }

                let mut columns = bumpalo::collections::Vec::new_in(self.arena);

                for (i, expr) in project.expressions.iter().enumerate() {
                    let (name, data_type, nullable) = self.infer_expr_type(expr, &input_schema)?;
                    let col_name = if let Some(alias) = project.aliases.get(i).and_then(|a| *a) {
                        alias
                    } else {
                        name
                    };
                    columns.push(OutputColumn {
                        name: col_name,
                        data_type,
                        nullable,
                    });
                }

                Ok(OutputSchema {
                    columns: columns.into_bump_slice(),
                })
            }
            LogicalOperator::Filter(filter) => self.compute_logical_output_schema(filter.input),
            LogicalOperator::Sort(sort) => self.compute_logical_output_schema(sort.input),
            LogicalOperator::Limit(limit) => self.compute_logical_output_schema(limit.input),
            LogicalOperator::Join(join) => {
                let left_schema = self.compute_logical_output_schema(join.left)?;
                let right_schema = self.compute_logical_output_schema(join.right)?;
                let mut columns = bumpalo::collections::Vec::new_in(self.arena);

                for col in left_schema.columns {
                    columns.push(*col);
                }
                for col in right_schema.columns {
                    columns.push(*col);
                }

                Ok(OutputSchema {
                    columns: columns.into_bump_slice(),
                })
            }
            LogicalOperator::Aggregate(agg) => {
                let mut columns = bumpalo::collections::Vec::new_in(self.arena);
                let input_schema = self.compute_logical_output_schema(agg.input)?;

                for group_expr in agg.group_by.iter() {
                    let (name, data_type, nullable) =
                        self.infer_expr_type(group_expr, &input_schema)?;
                    columns.push(OutputColumn {
                        name,
                        data_type,
                        nullable,
                    });
                }

                for agg_expr in agg.aggregates.iter() {
                    if let Expr::Function(func) = agg_expr {
                        let name = self.arena.alloc_str(func.name.name);
                        let data_type = match func.name.name.to_lowercase().as_str() {
                            "count" => DataType::Int8,
                            "avg" => DataType::Float8,
                            "sum" | "min" | "max" => {
                                if let crate::sql::ast::FunctionArgs::Args(args) = func.args {
                                    if let Some(first_arg) = args.first() {
                                        let (_, dt, _) =
                                            self.infer_expr_type(first_arg.value, &input_schema)?;
                                        dt
                                    } else {
                                        DataType::Int8
                                    }
                                } else {
                                    DataType::Int8
                                }
                            }
                            _ => DataType::Text,
                        };
                        columns.push(OutputColumn {
                            name,
                            data_type,
                            nullable: true,
                        });
                    }
                }

                Ok(OutputSchema {
                    columns: columns.into_bump_slice(),
                })
            }
            LogicalOperator::Subquery(subq) => Ok(subq.output_schema.clone()),
            LogicalOperator::SetOp(set_op) => self.compute_logical_output_schema(set_op.left),
            LogicalOperator::Window(window) => {
                let input_schema = self.compute_logical_output_schema(window.input)?;
                let mut columns = bumpalo::collections::Vec::new_in(self.arena);

                for col in input_schema.columns {
                    columns.push(*col);
                }

                for window_func in window.window_functions.iter() {
                    let name = window_func.alias.unwrap_or(window_func.function_name);
                    columns.push(OutputColumn {
                        name,
                        data_type: DataType::Int8,
                        nullable: false,
                    });
                }

                Ok(OutputSchema {
                    columns: columns.into_bump_slice(),
                })
            }
            LogicalOperator::Values(_)
            | LogicalOperator::Insert(_)
            | LogicalOperator::Update(_)
            | LogicalOperator::Delete(_) => Ok(OutputSchema::empty()),
        }
    }

    pub(crate) fn infer_expr_type(
        &self,
        expr: &'a Expr<'a>,
        input_schema: &OutputSchema<'a>,
    ) -> Result<(&'a str, DataType, bool)> {
        match expr {
            Expr::Column(col_ref) => {
                if let Some(col) = input_schema.get_column(col_ref.column) {
                    Ok((col.name, col.data_type, col.nullable))
                } else {
                    Ok((col_ref.column, DataType::Text, true))
                }
            }
            Expr::Literal(lit) => {
                let (name, data_type) = match lit {
                    Literal::Null => ("?column?", DataType::Text),
                    Literal::Boolean(_) => ("?column?", DataType::Bool),
                    Literal::Integer(_) => ("?column?", DataType::Int8),
                    Literal::Float(_) => ("?column?", DataType::Float8),
                    Literal::String(_) => ("?column?", DataType::Text),
                    Literal::HexNumber(_) => ("?column?", DataType::Int8),
                    Literal::BinaryNumber(_) => ("?column?", DataType::Int8),
                };
                Ok((self.arena.alloc_str(name), data_type, true))
            }
            Expr::BinaryOp { left, op, right } => {
                use crate::sql::ast::BinaryOperator;
                let data_type = match op {
                    BinaryOperator::Plus
                    | BinaryOperator::Minus
                    | BinaryOperator::Multiply
                    | BinaryOperator::Divide
                    | BinaryOperator::Modulo
                    | BinaryOperator::Power
                    | BinaryOperator::BitwiseAnd
                    | BinaryOperator::BitwiseOr
                    | BinaryOperator::BitwiseXor
                    | BinaryOperator::LeftShift
                    | BinaryOperator::RightShift => {
                        let (_, left_type, _) = self.infer_expr_type(left, input_schema)?;
                        let (_, right_type, _) = self.infer_expr_type(right, input_schema)?;
                        if left_type == DataType::Float8 || right_type == DataType::Float8 {
                            DataType::Float8
                        } else {
                            left_type
                        }
                    }
                    BinaryOperator::Concat => DataType::Text,
                    BinaryOperator::VectorL2Distance
                    | BinaryOperator::VectorCosineDistance
                    | BinaryOperator::VectorInnerProduct => DataType::Float8,
                    _ => DataType::Bool,
                };
                Ok((self.arena.alloc_str("?column?"), data_type, true))
            }
            Expr::UnaryOp { expr, .. } => self.infer_expr_type(expr, input_schema),
            Expr::Function(func) => {
                let name = self.arena.alloc_str(func.name.name);
                Ok((name, DataType::Text, true))
            }
            _ => Ok((self.arena.alloc_str("?column?"), DataType::Text, true)),
        }
    }

    pub(crate) fn infer_aggregate_type(
        &self,
        agg: &AggregateExpr<'a>,
        input_schema: &OutputSchema<'a>,
    ) -> Result<(&'a str, DataType)> {
        let func_name = match agg.function {
            AggregateFunction::Count => "count",
            AggregateFunction::Sum => "sum",
            AggregateFunction::Avg => "avg",
            AggregateFunction::Min => "min",
            AggregateFunction::Max => "max",
        };

        let data_type = match agg.function {
            AggregateFunction::Count => DataType::Int8,
            AggregateFunction::Avg => DataType::Float8,
            AggregateFunction::Sum | AggregateFunction::Min | AggregateFunction::Max => {
                if let Some(arg) = agg.argument {
                    let (_, dt, _) = self.infer_expr_type(arg, input_schema)?;
                    dt
                } else {
                    DataType::Int8
                }
            }
        };

        Ok((self.arena.alloc_str(func_name), data_type))
    }
}
