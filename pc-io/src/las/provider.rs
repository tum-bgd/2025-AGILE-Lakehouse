use std::{any::Any, collections::HashMap, sync::Arc};

use datafusion::{
    arrow::datatypes::SchemaRef,
    catalog::Session,
    common::{ColumnStatistics, DFSchema, Result, Statistics, stats::Precision},
    datasource::TableProvider,
    execution::context::ExecutionProps,
    logical_expr::{Operator, TableProviderFilterPushDown, TableType, utils::conjunction},
    physical_expr::create_physical_expr,
    physical_optimizer::pruning::PruningPredicate,
    physical_plan::{ExecutionPlan, empty::EmptyExec},
    prelude::Expr,
    scalar::ScalarValue,
};

use super::{LasDataSource, exec::LasExec};

#[async_trait::async_trait]
impl TableProvider for LasDataSource {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn schema(&self) -> SchemaRef {
        self.schema()
    }

    fn table_type(&self) -> TableType {
        TableType::Base
    }

    async fn scan(
        &self,
        _state: &(dyn Session),
        projection: Option<&Vec<usize>>,
        filters: &[Expr],
        _limit: Option<usize>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        // println!("{:#?}", filters);

        // pruning predicate
        let pruning_predicate = conjunction(filters.to_vec()).map(|expr| {
            let df_schema = DFSchema::try_from(self.schema()).unwrap();
            let props = ExecutionProps::new();
            let physical_expr = create_physical_expr(&expr, &df_schema, &props).unwrap();
            PruningPredicate::try_new(physical_expr, self.schema()).unwrap()
        });

        // files pruning
        let files_filter = if let Some(pruning_predicate) = &pruning_predicate {
            let files_filter = pruning_predicate.prune(&self.files_statistics).unwrap();

            if files_filter.iter().filter(|b| **b).count() == 0 {
                return Ok(Arc::new(EmptyExec::new(self.schema())));
            }

            files_filter
        } else {
            vec![true; self.table_paths.len()]
        };

        // chunks pruning
        let chunks_filters = HashMap::from_iter(
            self.table_paths
                .clone()
                .into_iter()
                .zip(files_filter.clone())
                .filter_map(|(path, f)| {
                    if f {
                        if let Some(pruning_predicate) = &pruning_predicate {
                            if let Some(stats) = &self.chunk_statistics.get(&path) {
                                let chunk_filter = pruning_predicate.prune(*stats).unwrap();

                                return Some((path, chunk_filter));
                            }
                        }
                    }
                    None
                }),
        );

        Ok(Arc::new(LasExec::new(
            self.clone(),
            projection.cloned(),
            files_filter,
            chunks_filters,
        )))
    }

    fn supports_filters_pushdown(
        &self,
        filters: &[&Expr],
    ) -> Result<Vec<TableProviderFilterPushDown>> {
        filters
            .iter()
            .map(|f| match f {
                // support binary comparison >, >=, <, <= on x, y, z
                Expr::BinaryExpr(binary) => match &binary.op {
                    Operator::Gt | Operator::Lt | Operator::GtEq | Operator::LtEq => {
                        match (binary.left.as_ref(), binary.right.as_ref()) {
                            (Expr::Column(column), Expr::Literal(_))
                            | (Expr::Literal(_), Expr::Column(column)) => {
                                if ["x", "y", "z"].contains(&column.name.as_str()) {
                                    Ok(TableProviderFilterPushDown::Inexact)
                                } else {
                                    Ok(TableProviderFilterPushDown::Unsupported)
                                }
                            }
                            _ => Ok(TableProviderFilterPushDown::Unsupported),
                        }
                    }
                    _ => Ok(TableProviderFilterPushDown::Unsupported),
                },
                _ => Ok(TableProviderFilterPushDown::Unsupported),
            })
            .collect()
    }

    fn statistics(&self) -> Option<Statistics> {
        let num_rows: u64 = self
            .table_headers
            .iter()
            .map(|header| header.number_of_points())
            .sum();

        let column_statistics = if self.table_options.raw || num_rows == 0 {
            Statistics::unknown_column(&self.schema())
        } else {
            let bounds = self
                .table_headers
                .iter()
                .map(|header| header.bounds())
                .reduce(|mut acc, e| {
                    acc.min.x = acc.min.x.min(e.min.x);
                    acc.max.x = acc.max.x.max(e.max.x);
                    acc.min.y = acc.min.y.min(e.min.y);
                    acc.max.y = acc.max.y.max(e.max.y);
                    acc.min.z = acc.min.z.min(e.min.z);
                    acc.max.z = acc.max.z.max(e.max.z);
                    acc
                })
                .unwrap();

            self.schema()
                .fields()
                .iter()
                .map(|field| match field.name().as_str() {
                    "x" => ColumnStatistics {
                        null_count: Precision::Exact(0),
                        max_value: Precision::Exact(ScalarValue::Float64(Some(bounds.max.x))),
                        min_value: Precision::Exact(ScalarValue::Float64(Some(bounds.min.x))),
                        sum_value: Precision::Absent,
                        distinct_count: Precision::Inexact(num_rows as usize),
                    },
                    "y" => ColumnStatistics {
                        null_count: Precision::Exact(0),
                        max_value: Precision::Exact(ScalarValue::Float64(Some(bounds.max.y))),
                        min_value: Precision::Exact(ScalarValue::Float64(Some(bounds.min.y))),
                        sum_value: Precision::Absent,
                        distinct_count: Precision::Inexact(num_rows as usize),
                    },
                    "z" => ColumnStatistics {
                        null_count: Precision::Exact(0),
                        max_value: Precision::Exact(ScalarValue::Float64(Some(bounds.max.z))),
                        min_value: Precision::Exact(ScalarValue::Float64(Some(bounds.min.z))),
                        sum_value: Precision::Absent,
                        distinct_count: Precision::Inexact(num_rows as usize),
                    },
                    _ => ColumnStatistics::new_unknown(),
                })
                .collect()
        };

        let stats = Statistics {
            num_rows: Precision::Exact(num_rows as usize),
            total_byte_size: Precision::Absent,
            column_statistics,
        };

        Some(stats)
    }
}
