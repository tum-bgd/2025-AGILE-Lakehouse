use std::{any::Any, sync::Arc};

use datafusion::{
    arrow::datatypes::SchemaRef,
    catalog::{Session, TableProvider},
    common::{Column, ColumnStatistics, Statistics, stats::Precision},
    datasource::TableType,
    logical_expr::{BinaryExpr, Operator, TableProviderFilterPushDown},
    physical_plan::ExecutionPlan,
    prelude::Expr,
    scalar::ScalarValue,
};
use rstar::{Envelope, Point};

use crate::{AABB, PointCloud, PointCloudExec, PointXYZI};

// impl std::fmt::Debug for PointCloud {
//     fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
//         f.write_str("pointcloud_db")
//     }
// }

#[async_trait::async_trait]
impl TableProvider for PointCloud {
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
        _state: &dyn Session,
        projection: Option<&Vec<usize>>,
        filters: &[Expr],
        _limit: Option<usize>,
    ) -> datafusion::error::Result<Arc<dyn ExecutionPlan>> {
        // extract bounds from filters
        let mut lower = PointXYZI::generate(|_| f64::MIN);
        let mut upper = PointXYZI::generate(|_| f64::MAX);

        for filter in filters {
            match filter {
                Expr::BinaryExpr(BinaryExpr { left, op, right }) => {
                    match (left.as_ref(), op, right.as_ref()) {
                        (
                            Expr::Literal(value),
                            Operator::Lt | Operator::LtEq,
                            Expr::Column(Column {
                                relation: _,
                                name,
                                spans: _,
                            }),
                        )
                        | (
                            Expr::Column(Column {
                                relation: _,
                                name,
                                spans: _,
                            }),
                            Operator::Gt | Operator::GtEq,
                            Expr::Literal(value),
                        ) => {
                            let index = match name.as_str() {
                                "x" => 0,
                                "y" => 1,
                                "z" => 2,
                                "i" => 3,
                                _ => continue,
                            };
                            *upper.nth_mut(index) = match value {
                                ScalarValue::Float64(v) => v.unwrap(),
                                _ => continue,
                            }
                        }
                        (
                            Expr::Literal(value),
                            Operator::Gt | Operator::GtEq,
                            Expr::Column(Column {
                                relation: _,
                                name,
                                spans: _,
                            }),
                        )
                        | (
                            Expr::Column(Column {
                                relation: _,
                                name,
                                spans: _,
                            }),
                            Operator::Lt | Operator::LtEq,
                            Expr::Literal(value),
                        ) => {
                            let index = match name.as_str() {
                                "x" => 0,
                                "y" => 1,
                                "z" => 2,
                                "i" => 3,
                                _ => continue,
                            };
                            *lower.nth_mut(index) = match value {
                                ScalarValue::Float64(v) => v.unwrap(),
                                _ => continue,
                            }
                        }
                        _ => continue,
                    }
                }
                _ => continue,
            }
        }

        let bounds = AABB::from_corners(lower, upper);

        Ok(Arc::new(PointCloudExec::new(
            projection,
            self.clone(),
            bounds,
        )))
    }

    fn supports_filters_pushdown(
        &self,
        filters: &[&Expr],
    ) -> datafusion::error::Result<Vec<TableProviderFilterPushDown>> {
        filters
            .iter()
            .map(|f| match f {
                // support binary comparison >, >=, <, <= on x, y, z and i
                Expr::BinaryExpr(binary) => match &binary.op {
                    Operator::Gt | Operator::Lt | Operator::GtEq | Operator::LtEq => {
                        match (binary.left.as_ref(), binary.right.as_ref()) {
                            (Expr::Column(column), Expr::Literal(_))
                            | (Expr::Literal(_), Expr::Column(column)) => {
                                if ["x", "y", "z", "i"].contains(&column.name.as_str()) {
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
        let bounds = self
            .store
            .iter()
            .map(|entry| *entry.value().read().unwrap().bounds())
            .reduce(|acc, e| acc.merged(&e))
            .unwrap();

        let lower = bounds.lower();
        let upper = bounds.upper();

        let column_statistics = self
            .schema()
            .fields()
            .iter()
            .map(|field| match field.name().as_str() {
                "x" => ColumnStatistics {
                    null_count: Precision::Inexact(0),
                    max_value: Precision::Exact(ScalarValue::Float64(Some(upper.nth(0)))),
                    min_value: Precision::Exact(ScalarValue::Float64(Some(lower.nth(0)))),
                    sum_value: Precision::Absent,
                    distinct_count: Precision::Absent,
                },
                "y" => ColumnStatistics {
                    null_count: Precision::Inexact(0),
                    max_value: Precision::Exact(ScalarValue::Float64(Some(upper.nth(1)))),
                    min_value: Precision::Exact(ScalarValue::Float64(Some(lower.nth(1)))),
                    sum_value: Precision::Absent,
                    distinct_count: Precision::Absent,
                },
                "z" => ColumnStatistics {
                    null_count: Precision::Inexact(0),
                    max_value: Precision::Exact(ScalarValue::Float64(Some(upper.nth(2)))),
                    min_value: Precision::Exact(ScalarValue::Float64(Some(lower.nth(2)))),
                    sum_value: Precision::Absent,
                    distinct_count: Precision::Absent,
                },
                "i" => ColumnStatistics {
                    null_count: Precision::Inexact(0),
                    max_value: Precision::Exact(ScalarValue::Float64(Some(upper.nth(3)))),
                    min_value: Precision::Exact(ScalarValue::Float64(Some(lower.nth(3)))),
                    sum_value: Precision::Absent,
                    distinct_count: Precision::Absent,
                },
                _ => ColumnStatistics::new_unknown(),
            })
            .collect();

        let stats = Statistics {
            num_rows: Precision::Absent,
            total_byte_size: Precision::Absent,
            column_statistics,
        };

        Some(stats)
    }
}
