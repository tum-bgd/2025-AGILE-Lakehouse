use std::{collections::HashSet, sync::Arc};

use datafusion::{
    arrow::{
        array::{ArrayRef, BooleanArray, Float64Array, Int32Array, PrimitiveBuilder, RecordBatch},
        datatypes::{DataType, Field, Float64Type, Schema},
    },
    physical_optimizer::pruning::PruningStatistics,
    prelude::*,
    scalar::ScalarValue,
};

#[derive(Clone)]
pub(super) struct LasStatistics {
    pub(super) values: RecordBatch,
}

impl PruningStatistics for LasStatistics {
    fn min_values(&self, column: &Column) -> Option<ArrayRef> {
        match column.name.as_str() {
            "x" => self.values.column_by_name("x_min").cloned(),
            "y" => self.values.column_by_name("y_min").cloned(),
            "z" => self.values.column_by_name("z_min").cloned(),
            _ => None,
        }
    }

    fn max_values(&self, column: &Column) -> Option<ArrayRef> {
        match column.name.as_str() {
            "x" => self.values.column_by_name("x_max").cloned(),
            "y" => self.values.column_by_name("y_max").cloned(),
            "z" => self.values.column_by_name("z_max").cloned(),
            _ => None,
        }
    }

    fn num_containers(&self) -> usize {
        self.values.num_rows()
    }

    fn null_counts(&self, column: &Column) -> Option<ArrayRef> {
        match column.name.as_str() {
            "x" | "y" | "z" => Some(Arc::new(Int32Array::from_value(0, self.values.num_rows()))),
            _ => None,
        }
    }

    fn row_counts(&self, _column: &Column) -> Option<ArrayRef> {
        None
    }

    fn contained(&self, _column: &Column, _values: &HashSet<ScalarValue>) -> Option<BooleanArray> {
        None
    }
}

pub(super) struct LasStatisticsBuilder {
    x_min: PrimitiveBuilder<Float64Type>,
    x_max: PrimitiveBuilder<Float64Type>,
    y_min: PrimitiveBuilder<Float64Type>,
    y_max: PrimitiveBuilder<Float64Type>,
    z_min: PrimitiveBuilder<Float64Type>,
    z_max: PrimitiveBuilder<Float64Type>,
}

impl LasStatisticsBuilder {
    pub(super) fn new_with_capacity(capacity: usize) -> Self {
        LasStatisticsBuilder {
            x_min: Float64Array::builder(capacity),
            x_max: Float64Array::builder(capacity),
            y_min: Float64Array::builder(capacity),
            y_max: Float64Array::builder(capacity),
            z_min: Float64Array::builder(capacity),
            z_max: Float64Array::builder(capacity),
        }
    }

    pub(super) fn add_values(&mut self, values: &[f64; 6]) {
        self.x_min.append_value(values[0]);
        self.x_max.append_value(values[1]);
        self.y_min.append_value(values[2]);
        self.y_max.append_value(values[3]);
        self.z_min.append_value(values[4]);
        self.z_max.append_value(values[5]);
    }

    pub(super) fn finish(mut self) -> LasStatistics {
        let schema = Schema::new([
            Arc::new(Field::new("x_min", DataType::Float64, false)),
            Arc::new(Field::new("x_max", DataType::Float64, false)),
            Arc::new(Field::new("y_min", DataType::Float64, false)),
            Arc::new(Field::new("y_max", DataType::Float64, false)),
            Arc::new(Field::new("z_min", DataType::Float64, false)),
            Arc::new(Field::new("z_max", DataType::Float64, false)),
        ]);

        let batch = RecordBatch::try_new(
            Arc::new(schema),
            vec![
                Arc::new(self.x_min.finish()),
                Arc::new(self.x_max.finish()),
                Arc::new(self.y_min.finish()),
                Arc::new(self.y_max.finish()),
                Arc::new(self.z_min.finish()),
                Arc::new(self.z_max.finish()),
            ],
        )
        .unwrap();

        LasStatistics { values: batch }
    }
}
