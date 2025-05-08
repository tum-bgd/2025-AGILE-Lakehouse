use std::{
    collections::HashMap,
    fmt::Debug,
    ops::{Add, Div, Mul, Sub},
};

use datafusion::{
    arrow::datatypes::{ArrowNativeType, DataType, Field, Fields, Schema, SchemaRef},
    logical_expr::Literal,
};
use num_traits::Zero;
use rstar::RTreeNum;

use crate::{
    Dims,
    schema::{PC_DIMENSION_KEY, PC_IMPORTANCE_KEY, PC_LOCATION_KEY},
};

/// Coordinate trait
pub trait Coord: ArrowNativeType + Literal + RTreeNum {
    const DATA_TYPE: DataType;
}

// Coordinate trait implementations
impl Coord for f64 {
    const DATA_TYPE: DataType = DataType::Float64;
}
impl Coord for f32 {
    const DATA_TYPE: DataType = DataType::Float32;
}
impl Coord for i64 {
    const DATA_TYPE: DataType = DataType::Int64;
}
impl Coord for i32 {
    const DATA_TYPE: DataType = DataType::Int32;
}

/// Point trait
pub trait PointTrait: rstar::Point + rstar::RTreeObject + Send {
    const DATA_TYPE: DataType;
    const DIMS: Dims;

    fn schema() -> SchemaRef;

    #[inline]
    fn from_slice(components: &[Self::Scalar]) -> Self {
        Self::generate(|i| {
            components
                .get(i)
                .cloned()
                .unwrap_or_else(Self::Scalar::zero)
        })
    }

    fn coords(&self) -> impl Iterator<Item = Self::Scalar> {
        (0..<Self as rstar::Point>::DIMENSIONS).map(|i| rstar::Point::nth(self, i))
    }

    fn add(&self, other: &Self) -> Self {
        Self::generate(|i| self.nth(i).add(other.nth(i)))
    }

    fn sub(&self, other: &Self) -> Self {
        Self::generate(|i| self.nth(i).sub(other.nth(i)))
    }

    fn mul(&self, other: &Self) -> Self {
        Self::generate(|i| self.nth(i).mul(other.nth(i)))
    }

    fn div(&self, other: &Self) -> Self {
        Self::generate(|i| self.nth(i).div(other.nth(i)))
    }
}

/// 2D Point
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PointXY<T> {
    x: T,
    y: T,
}

impl<T: Coord> PointTrait for PointXY<T> {
    const DATA_TYPE: DataType = T::DATA_TYPE;

    const DIMS: Dims = Dims::XY;

    fn schema() -> SchemaRef {
        Schema::new(Fields::from_iter([
            Field::new("x".to_string(), T::DATA_TYPE, false).with_metadata(HashMap::from([
                (PC_DIMENSION_KEY.to_owned(), "1".to_string()),
                (PC_LOCATION_KEY.to_owned(), "x".to_string()),
            ])),
            Field::new("y".to_string(), T::DATA_TYPE, false).with_metadata(HashMap::from([
                (PC_DIMENSION_KEY.to_owned(), "2".to_string()),
                (PC_LOCATION_KEY.to_owned(), "y".to_string()),
            ])),
        ]))
        .into()
    }
}

impl<T: Coord> rstar::Point for PointXY<T> {
    type Scalar = T;

    const DIMENSIONS: usize = 2;

    fn generate(mut generator: impl FnMut(usize) -> Self::Scalar) -> Self {
        Self {
            x: generator(0),
            y: generator(1),
        }
    }

    fn nth(&self, index: usize) -> Self::Scalar {
        match index {
            0 => self.x,
            1 => self.y,
            _ => unreachable!(),
        }
    }

    fn nth_mut(&mut self, index: usize) -> &mut Self::Scalar {
        match index {
            0 => &mut self.x,
            1 => &mut self.y,
            _ => unreachable!(),
        }
    }
}

/// 3D Point
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PointXYZ<T> {
    x: T,
    y: T,
    z: T,
}

impl<T: Coord> PointTrait for PointXYZ<T> {
    const DATA_TYPE: DataType = T::DATA_TYPE;

    const DIMS: Dims = Dims::XYZ;

    fn schema() -> SchemaRef {
        Schema::new(Fields::from_iter([
            Field::new("x".to_string(), T::DATA_TYPE, false).with_metadata(HashMap::from([
                (PC_DIMENSION_KEY.to_owned(), "1".to_string()),
                (PC_LOCATION_KEY.to_owned(), "x".to_string()),
            ])),
            Field::new("y".to_string(), T::DATA_TYPE, false).with_metadata(HashMap::from([
                (PC_DIMENSION_KEY.to_owned(), "2".to_string()),
                (PC_LOCATION_KEY.to_owned(), "y".to_string()),
            ])),
            Field::new("z".to_string(), T::DATA_TYPE, false).with_metadata(HashMap::from([
                (PC_DIMENSION_KEY.to_owned(), "3".to_string()),
                (PC_LOCATION_KEY.to_owned(), "z".to_string()),
            ])),
        ]))
        .into()
    }
}

impl<T: Coord> rstar::Point for PointXYZ<T> {
    type Scalar = T;

    const DIMENSIONS: usize = 3;

    fn generate(mut generator: impl FnMut(usize) -> Self::Scalar) -> Self {
        Self {
            x: generator(0),
            y: generator(1),
            z: generator(2),
        }
    }

    fn nth(&self, index: usize) -> Self::Scalar {
        match index {
            0 => self.x,
            1 => self.y,
            2 => self.z,
            _ => unreachable!(),
        }
    }

    fn nth_mut(&mut self, index: usize) -> &mut Self::Scalar {
        match index {
            0 => &mut self.x,
            1 => &mut self.y,
            2 => &mut self.z,
            _ => unreachable!(),
        }
    }
}

/// 2D Point with Importance
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PointXYI<T> {
    x: T,
    y: T,
    i: T,
}

impl<T: Coord> PointTrait for PointXYI<T> {
    const DATA_TYPE: DataType = T::DATA_TYPE;

    const DIMS: Dims = Dims::XYI;

    fn schema() -> SchemaRef {
        Schema::new(Fields::from_iter([
            Field::new("x".to_string(), T::DATA_TYPE, false).with_metadata(HashMap::from([
                (PC_DIMENSION_KEY.to_owned(), "1".to_string()),
                (PC_LOCATION_KEY.to_owned(), "x".to_string()),
            ])),
            Field::new("y".to_string(), T::DATA_TYPE, false).with_metadata(HashMap::from([
                (PC_DIMENSION_KEY.to_owned(), "2".to_string()),
                (PC_LOCATION_KEY.to_owned(), "y".to_string()),
            ])),
            Field::new("i".to_string(), T::DATA_TYPE, false).with_metadata(HashMap::from([
                (PC_DIMENSION_KEY.to_owned(), "3".to_string()),
                (PC_IMPORTANCE_KEY.to_owned(), "i".to_string()),
            ])),
        ]))
        .into()
    }
}

impl<T: Coord> rstar::Point for PointXYI<T> {
    type Scalar = T;

    const DIMENSIONS: usize = 3;

    fn generate(mut generator: impl FnMut(usize) -> Self::Scalar) -> Self {
        Self {
            x: generator(0),
            y: generator(1),
            i: generator(2),
        }
    }

    fn nth(&self, index: usize) -> Self::Scalar {
        match index {
            0 => self.x,
            1 => self.y,
            2 => self.i,
            _ => unreachable!(),
        }
    }

    fn nth_mut(&mut self, index: usize) -> &mut Self::Scalar {
        match index {
            0 => &mut self.x,
            1 => &mut self.y,
            2 => &mut self.i,
            _ => unreachable!(),
        }
    }
}

/// 3D Point with Importance
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PointXYZI<T> {
    x: T,
    y: T,
    z: T,
    i: T,
}

impl<T: Coord> PointTrait for PointXYZI<T> {
    const DATA_TYPE: DataType = T::DATA_TYPE;

    const DIMS: Dims = Dims::XYZ;

    fn schema() -> SchemaRef {
        Schema::new(Fields::from_iter([
            Field::new("x".to_string(), T::DATA_TYPE, false).with_metadata(HashMap::from([
                (PC_DIMENSION_KEY.to_owned(), "1".to_string()),
                (PC_LOCATION_KEY.to_owned(), "x".to_string()),
            ])),
            Field::new("y".to_string(), T::DATA_TYPE, false).with_metadata(HashMap::from([
                (PC_DIMENSION_KEY.to_owned(), "2".to_string()),
                (PC_LOCATION_KEY.to_owned(), "y".to_string()),
            ])),
            Field::new("z".to_string(), T::DATA_TYPE, false).with_metadata(HashMap::from([
                (PC_DIMENSION_KEY.to_owned(), "3".to_string()),
                (PC_LOCATION_KEY.to_owned(), "z".to_string()),
            ])),
            Field::new("i".to_string(), T::DATA_TYPE, false).with_metadata(HashMap::from([
                (PC_DIMENSION_KEY.to_owned(), "4".to_string()),
                (PC_IMPORTANCE_KEY.to_owned(), "i".to_string()),
            ])),
        ]))
        .into()
    }
}

impl<T: Coord> rstar::Point for PointXYZI<T> {
    type Scalar = T;

    const DIMENSIONS: usize = 4;

    fn generate(mut generator: impl FnMut(usize) -> Self::Scalar) -> Self {
        Self {
            x: generator(0),
            y: generator(1),
            z: generator(2),
            i: generator(3),
        }
    }

    fn nth(&self, index: usize) -> Self::Scalar {
        match index {
            0 => self.x,
            1 => self.y,
            2 => self.z,
            3 => self.i,
            _ => unreachable!(),
        }
    }

    fn nth_mut(&mut self, index: usize) -> &mut Self::Scalar {
        match index {
            0 => &mut self.x,
            1 => &mut self.y,
            2 => &mut self.z,
            3 => &mut self.i,
            _ => unreachable!(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn slice() {
        let p: PointXY<f64> = PointTrait::from_slice(&[1.; 3]);
        assert_eq!(p, PointTrait::from_slice(&[1.; 2]));

        let p: PointXYZI<f64> = PointTrait::from_slice(&[1.; 3]);
        assert_eq!(p, PointTrait::from_slice(&[1., 1., 1., 0.]));
    }

    #[test]
    fn ops() {
        let one: PointXYZ<i64> = PointTrait::from_slice(&[1; 3]);
        let two: PointXYZ<i64> = PointTrait::from_slice(&[2; 3]);
        let three: PointXYZ<i64> = PointTrait::from_slice(&[3; 3]);
        let four: PointXYZ<i64> = PointTrait::from_slice(&[4; 3]);

        assert_eq!(one.add(&one), two);
        assert_eq!(three.sub(&one), two);
        assert_eq!(two.mul(&two), four);
        assert_eq!(four.div(&two), two);
    }
}
