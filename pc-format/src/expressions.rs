use datafusion::{
    arrow::{
        array::AsArray,
        datatypes::{
            DataType, Float16Type, Float32Type, Float64Type, Int16Type, Int32Type, Int64Type,
            UInt16Type, UInt32Type, UInt64Type,
        },
    },
    error::DataFusionError,
    functions_aggregate::min_max::{max, min},
    prelude::{DataFrame, and, col, lit},
};
use rstar::Envelope;

use crate::{AABB, PointTrait};

/// calculate bounds of dataframe
pub async fn df_aabb<P>(df: &DataFrame) -> Result<AABB<P>, DataFusionError>
where
    P: PointTrait,
    <P as rstar::Point>::Scalar: num_traits::NumCast,
{
    let aabb: rstar::AABB<P> = rstar::AABB::new_empty();
    let (mut lower, mut upper) = (aabb.lower(), aabb.upper());

    for (i, d) in P::DIMS.names().iter().enumerate() {
        let bounds = df
            .clone()
            .aggregate(vec![], vec![min(col(*d)), max(col(*d))])?;
        let bounds = bounds.collect().await?;
        assert_eq!(bounds.len(), 1);
        assert_eq!(bounds[0].num_rows(), 1);

        let bounds = &bounds[0];

        match bounds.schema().field(0).data_type() {
            DataType::Int16 => {
                let min = bounds.column(0).as_primitive::<Int16Type>().value(0);
                let max = bounds.column(1).as_primitive::<Int16Type>().value(0);

                *lower.nth_mut(i) = num_traits::cast(min).unwrap();
                *upper.nth_mut(i) = num_traits::cast(max).unwrap();
            }
            DataType::Int32 => {
                let min = bounds.column(0).as_primitive::<Int32Type>().value(0);
                let max = bounds.column(1).as_primitive::<Int32Type>().value(0);

                *lower.nth_mut(i) = num_traits::cast(min).unwrap();
                *upper.nth_mut(i) = num_traits::cast(max).unwrap();
            }
            DataType::Int64 => {
                let min = bounds.column(0).as_primitive::<Int64Type>().value(0);
                let max = bounds.column(1).as_primitive::<Int64Type>().value(0);

                *lower.nth_mut(i) = num_traits::cast(min).unwrap();
                *upper.nth_mut(i) = num_traits::cast(max).unwrap();
            }
            DataType::UInt16 => {
                let min = bounds.column(0).as_primitive::<UInt16Type>().value(0);
                let max = bounds.column(1).as_primitive::<UInt16Type>().value(0);

                *lower.nth_mut(i) = num_traits::cast(min).unwrap();
                *upper.nth_mut(i) = num_traits::cast(max).unwrap();
            }
            DataType::UInt32 => {
                let min = bounds.column(0).as_primitive::<UInt32Type>().value(0);
                let max = bounds.column(1).as_primitive::<UInt32Type>().value(0);

                *lower.nth_mut(i) = num_traits::cast(min).unwrap();
                *upper.nth_mut(i) = num_traits::cast(max).unwrap();
            }
            DataType::UInt64 => {
                let min = bounds.column(0).as_primitive::<UInt64Type>().value(0);
                let max = bounds.column(1).as_primitive::<UInt64Type>().value(0);

                *lower.nth_mut(i) = num_traits::cast(min).unwrap();
                *upper.nth_mut(i) = num_traits::cast(max).unwrap();
            }
            DataType::Float16 => {
                let min = bounds.column(0).as_primitive::<Float16Type>().value(0);
                let max = bounds.column(1).as_primitive::<Float16Type>().value(0);

                *lower.nth_mut(i) = num_traits::cast(min).unwrap();
                *upper.nth_mut(i) = num_traits::cast(max).unwrap();
            }
            DataType::Float32 => {
                let min = bounds.column(0).as_primitive::<Float32Type>().value(0);
                let max = bounds.column(1).as_primitive::<Float32Type>().value(0);

                *lower.nth_mut(i) = num_traits::cast(min).unwrap();
                *upper.nth_mut(i) = num_traits::cast(max).unwrap();
            }
            DataType::Float64 => {
                let min = bounds.column(0).as_primitive::<Float64Type>().value(0);
                let max = bounds.column(1).as_primitive::<Float64Type>().value(0);

                *lower.nth_mut(i) = num_traits::cast(min).unwrap();
                *upper.nth_mut(i) = num_traits::cast(max).unwrap();
            }
            dt => panic!("unsuported data type `{dt}`"),
        }
    }

    Ok(AABB::from_corners(lower, upper))
}

// filter by bounds
pub async fn filter_df_by_aabb<P>(
    df: DataFrame,
    aabb: &AABB<P>,
    right_open: bool,
) -> Result<DataFrame, DataFusionError>
where
    P: PointTrait,
    <P as rstar::Point>::Scalar: num_traits::NumCast,
{
    let lower = aabb.lower();
    let upper = aabb.upper();

    let filter = P::DIMS
        .names()
        .iter()
        .zip(lower.coords().zip(upper.coords()))
        .map(|(name, (low, high))| {
            let field = df.schema().field_with_unqualified_name(name).unwrap();
            let (min, max) = match field.data_type() {
                DataType::Float64 => {
                    let min = num_traits::cast::<P::Scalar, f64>(low).unwrap();
                    let max = num_traits::cast::<P::Scalar, f64>(high).unwrap();
                    (lit(min), lit(max))
                }
                DataType::Float32 => {
                    let min = num_traits::cast::<P::Scalar, f32>(low).unwrap();
                    let max = num_traits::cast::<P::Scalar, f32>(high).unwrap();
                    (lit(min), lit(max))
                }
                DataType::Float16 => {
                    let min = num_traits::cast::<P::Scalar, f32>(low).unwrap();
                    let max = num_traits::cast::<P::Scalar, f32>(high).unwrap();
                    (lit(min), lit(max))
                }
                DataType::Int64 => {
                    let min = num_traits::cast::<P::Scalar, i64>(low).unwrap();
                    let max = num_traits::cast::<P::Scalar, i64>(high).unwrap();
                    (lit(min), lit(max))
                }
                DataType::Int32 => {
                    let min = num_traits::cast::<P::Scalar, i32>(low).unwrap();
                    let max = num_traits::cast::<P::Scalar, i32>(high).unwrap();
                    (lit(min), lit(max))
                }
                dt => unimplemented!("Column type to filter `{dt}`"),
            };
            if right_open {
                col(*name).gt_eq(min).and(col(*name).lt(max))
            } else {
                col(*name).gt_eq(min).and(col(*name).lt_eq(max))
            }
        })
        .reduce(and)
        .unwrap();

    df.filter(filter)
}
