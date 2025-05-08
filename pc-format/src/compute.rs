use std::sync::Arc;

use datafusion::{
    arrow::{
        array::{
            Array, AsArray, BooleanArray, Float16Array, Float32Array, Float64Array, Int32Array,
            Int64Array,
        },
        compute::{
            and, cast,
            kernels::cmp::{gt_eq, lt, lt_eq},
        },
        datatypes::{
            DataType, Float16Type, Float32Type, Float64Type, Int16Type, Int32Type, Int64Type,
            SchemaRef, UInt16Type, UInt32Type, UInt64Type,
        },
        error::ArrowError,
        record_batch::RecordBatch,
    },
    error::DataFusionError,
};
use num_traits::{NumCast, One};
use rand::{Rng, SeedableRng, rngs::SmallRng};
use rstar::Envelope;

use crate::{AABB, PointTrait};

/// add random importance
pub fn add_importance(batch: RecordBatch, schema: &SchemaRef) -> Result<RecordBatch, ArrowError> {
    match super::schema::importance(schema) {
        Some(i) => {
            // skip if existing
            if schema.contains(&batch.schema()) {
                return Ok(batch);
            }

            let mut rng = SmallRng::from_rng(&mut rand::rng());
            let mut columns = batch.columns().to_vec();

            match &schema.fields[i].data_type() {
                DataType::Float16 => {
                    let mut importance = vec![0.; batch.num_rows()];
                    rng.fill(&mut importance[..]);
                    columns.insert(
                        i,
                        cast(
                            &(Arc::new(Float32Array::from(importance)) as Arc<dyn Array>),
                            &DataType::Float16,
                        )?,
                    );
                }
                DataType::Float32 => {
                    let mut importance = vec![0.; batch.num_rows()];
                    rng.fill(&mut importance[..]);
                    columns.insert(i, Arc::new(Float32Array::from(importance)));
                }
                DataType::Float64 => {
                    let mut importance = vec![0.; batch.num_rows()];
                    rng.fill(&mut importance[..]);
                    columns.insert(i, Arc::new(Float64Array::from(importance)));
                }
                data_type => {
                    return Err(ArrowError::NotYetImplemented(format!(
                        "Unsuported data type for importance: `{data_type:?}`"
                    )));
                }
            }

            RecordBatch::try_new(schema.to_owned(), columns)
        }
        None => Err(ArrowError::SchemaError(
            "Schema does not contain `importance` field".to_string(),
        )),
    }
}

/// calculate bounds of record batch
pub fn record_batch_aabb<P>(rb: &RecordBatch) -> Result<AABB<P>, DataFusionError>
where
    P: PointTrait,
    <P as rstar::Point>::Scalar: num_traits::NumCast,
{
    let aabb: rstar::AABB<P> = rstar::AABB::new_empty();
    let (mut lower, mut upper) = (aabb.lower(), aabb.upper());

    for (i, d) in P::DIMS.names().iter().enumerate() {
        let column = rb.column_by_name(d.as_ref()).unwrap();

        match column.data_type() {
            DataType::Int16 => {
                let array = column.as_primitive::<Int16Type>();
                let min = datafusion::arrow::compute::min(array).unwrap();
                let max = datafusion::arrow::compute::max(array).unwrap();

                *lower.nth_mut(i) = num_traits::cast(min).unwrap();
                *upper.nth_mut(i) = num_traits::cast(max).unwrap();
            }
            DataType::Int32 => {
                let array = column.as_primitive::<Int32Type>();
                let min = datafusion::arrow::compute::min(array).unwrap();
                let max = datafusion::arrow::compute::max(array).unwrap();

                *lower.nth_mut(i) = num_traits::cast(min).unwrap();
                *upper.nth_mut(i) = num_traits::cast(max).unwrap();
            }
            DataType::Int64 => {
                let array = column.as_primitive::<Int64Type>();
                let min = datafusion::arrow::compute::min(array).unwrap();
                let max = datafusion::arrow::compute::max(array).unwrap();

                *lower.nth_mut(i) = num_traits::cast(min).unwrap();
                *upper.nth_mut(i) = num_traits::cast(max).unwrap();
            }
            DataType::UInt16 => {
                let array = column.as_primitive::<UInt16Type>();
                let min = datafusion::arrow::compute::min(array).unwrap();
                let max = datafusion::arrow::compute::max(array).unwrap();

                *lower.nth_mut(i) = num_traits::cast(min).unwrap();
                *upper.nth_mut(i) = num_traits::cast(max).unwrap();
            }
            DataType::UInt32 => {
                let array = column.as_primitive::<UInt32Type>();
                let min = datafusion::arrow::compute::min(array).unwrap();
                let max = datafusion::arrow::compute::max(array).unwrap();

                *lower.nth_mut(i) = num_traits::cast(min).unwrap();
                *upper.nth_mut(i) = num_traits::cast(max).unwrap();
            }
            DataType::UInt64 => {
                let array = column.as_primitive::<UInt64Type>();
                let min = datafusion::arrow::compute::min(array).unwrap();
                let max = datafusion::arrow::compute::max(array).unwrap();

                *lower.nth_mut(i) = num_traits::cast(min).unwrap();
                *upper.nth_mut(i) = num_traits::cast(max).unwrap();
            }
            DataType::Float16 => {
                let array = column.as_primitive::<Float16Type>();
                let min = datafusion::arrow::compute::min(array).unwrap();
                let max = datafusion::arrow::compute::max(array).unwrap();

                *lower.nth_mut(i) = num_traits::cast(min).unwrap();
                *upper.nth_mut(i) = num_traits::cast(max).unwrap();
            }
            DataType::Float32 => {
                let array = column.as_primitive::<Float32Type>();
                let min = datafusion::arrow::compute::min(array).unwrap();
                let max = datafusion::arrow::compute::max(array).unwrap();

                *lower.nth_mut(i) = num_traits::cast(min).unwrap();
                *upper.nth_mut(i) = num_traits::cast(max).unwrap();
            }
            DataType::Float64 => {
                let array = column.as_primitive::<Float64Type>();
                let min = datafusion::arrow::compute::min(array).unwrap();
                let max = datafusion::arrow::compute::max(array).unwrap();

                *lower.nth_mut(i) = num_traits::cast(min).unwrap();
                *upper.nth_mut(i) = num_traits::cast(max).unwrap();
            }
            dt => panic!("unsuported data type `{dt}`"),
        }
    }

    Ok(AABB::from_corners(lower, upper))
}

// filter by bounds
pub fn record_batch_aabb_filter<P>(rb: &RecordBatch, aabb: &AABB<P>) -> BooleanArray
where
    P: PointTrait,
    <P as rstar::Point>::Scalar: NumCast,
{
    let lower = aabb.lower();
    let upper = aabb.upper();

    P::DIMS
        .names()
        .iter()
        .enumerate()
        .map(|(index, name)| {
            let column = rb.column_by_name(name.as_ref()).unwrap();

            let min = lower.nth(index);
            let max = upper.nth(index);

            // account for floating cast instabilities for importance
            let right_open = !(*name == "i" && max >= P::Scalar::one());

            match column.data_type() {
                DataType::Float64 => {
                    let array = column.as_primitive::<Float64Type>();
                    let l = Float64Array::new_scalar(num_traits::cast(min).unwrap());
                    let h = Float64Array::new_scalar(num_traits::cast(max).unwrap());
                    let right = match right_open {
                        true => lt(array, &h).unwrap(),
                        false => lt_eq(array, &h).unwrap(),
                    };
                    and(&gt_eq(array, &l).unwrap(), &right).unwrap()
                }
                DataType::Float32 => {
                    let array = column.as_primitive::<Float32Type>();
                    let l = Float32Array::new_scalar(num_traits::cast(min).unwrap());
                    let h = Float32Array::new_scalar(num_traits::cast(max).unwrap());
                    let right = match right_open {
                        true => lt(array, &h).unwrap(),
                        false => lt_eq(array, &h).unwrap(),
                    };
                    and(&gt_eq(array, &l).unwrap(), &right).unwrap()
                }
                DataType::Float16 => {
                    let array = column.as_primitive::<Float16Type>();
                    let l = Float16Array::new_scalar(num_traits::cast(min).unwrap());
                    let h = Float16Array::new_scalar(num_traits::cast(max).unwrap());
                    let right = match right_open {
                        true => lt(array, &h).unwrap(),
                        false => lt_eq(array, &h).unwrap(),
                    };
                    and(&gt_eq(array, &l).unwrap(), &right).unwrap()
                }
                DataType::Int64 => {
                    let array = column.as_primitive::<Int64Type>();
                    let l = Int64Array::new_scalar(num_traits::cast(min).unwrap());
                    let h = Int64Array::new_scalar(num_traits::cast(max).unwrap());
                    let right = match right_open {
                        true => lt(array, &h).unwrap(),
                        false => lt_eq(array, &h).unwrap(),
                    };
                    and(&gt_eq(array, &l).unwrap(), &right).unwrap()
                }
                DataType::Int32 => {
                    let array = column.as_primitive::<Int32Type>();
                    let l = Int32Array::new_scalar(num_traits::cast(min).unwrap());
                    let h = Int32Array::new_scalar(num_traits::cast(max).unwrap());
                    let right = match right_open {
                        true => lt(array, &h).unwrap(),
                        false => lt_eq(array, &h).unwrap(),
                    };
                    and(&gt_eq(array, &l).unwrap(), &right).unwrap()
                }
                dt => unimplemented!("Column type to filter `{dt}`"),
            }
        })
        .reduce(|a, b| and(&a, &b).unwrap())
        .unwrap()
}
