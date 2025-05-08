use std::fmt::Write;

use num_traits::NumCast;
use rstar::{Envelope, RTreeObject};
use serde::{Deserialize, Serialize};

use crate::{Dims, PointTrait};

/// Axis aligned bounding box, wraps [rstar::AABB]
#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct AABB<P: PointTrait> {
    bounds: rstar::AABB<P>,
}

unsafe impl<P> Send for AABB<P> where P: PointTrait {}
unsafe impl<P> Sync for AABB<P> where P: PointTrait {}

impl<P> AABB<P>
where
    P: PointTrait,
{
    /// Generate id based on dimensions and coordinates
    pub fn id(&self) -> String {
        let upper = self.upper();
        let lower = self.lower();

        let mut id = String::new();

        P::DIMS.names().iter().enumerate().for_each(|(i, name)| {
            write!(&mut id, "{}-{:?}-{:?}", name, lower.nth(i), upper.nth(i)).unwrap()
        });

        id
    }

    /// Create aabb from point.
    pub fn from_point(p: P) -> Self {
        AABB {
            bounds: rstar::AABB::from_point(p),
        }
    }

    /// Creates a new [AABB] encompassing two points.
    pub fn from_corners(p1: P, p2: P) -> Self {
        AABB {
            bounds: rstar::AABB::from_corners(p1, p2),
        }
    }

    /// Returns the AABB's lower corner.
    ///
    /// This is the point contained within the AABB with the smallest coordinate value in each
    /// dimension.
    pub fn lower(&self) -> P {
        self.bounds.lower()
    }

    /// Returns the AABB's upper corner.
    ///
    /// This is the point contained within the AABB with the largest coordinate value in each
    /// dimension.
    pub fn upper(&self) -> P {
        self.bounds.upper()
    }

    pub fn with_importance<Q: PointTrait>(&self, from: Q::Scalar, to: Q::Scalar) -> AABB<Q>
    where
        <P as rstar::Point>::Scalar: NumCast,
        <Q as rstar::Point>::Scalar: NumCast,
    {
        match P::DIMS {
            Dims::XY => {
                let p1 = Q::generate(|i| {
                    if i < 2 {
                        num_traits::cast(self.lower().nth(i)).unwrap()
                    } else {
                        from
                    }
                });
                let p2 = Q::generate(|i| {
                    if i < 2 {
                        num_traits::cast(self.upper().nth(i)).unwrap()
                    } else {
                        to
                    }
                });
                AABB::from_corners(p1, p2)
            }
            Dims::XYZ => {
                let p1 = Q::generate(|i| {
                    if i < 3 {
                        num_traits::cast(self.lower().nth(i)).unwrap()
                    } else {
                        from
                    }
                });
                let p2 = Q::generate(|i| {
                    if i < 3 {
                        num_traits::cast(self.upper().nth(i)).unwrap()
                    } else {
                        to
                    }
                });
                AABB::from_corners(p1, p2)
            }
            Dims::XYI => {
                let mut p1 = Q::generate(|i| num_traits::cast(self.lower().nth(i)).unwrap());
                let mut p2 = Q::generate(|i| num_traits::cast(self.upper().nth(i)).unwrap());
                *p1.nth_mut(2) = from;
                *p2.nth_mut(2) = to;
                AABB::from_corners(p1, p2)
            }
            Dims::XYZI => {
                let mut p1 = Q::generate(|i| num_traits::cast(self.lower().nth(i)).unwrap());
                let mut p2 = Q::generate(|i| num_traits::cast(self.upper().nth(i)).unwrap());
                *p1.nth_mut(3) = from;
                *p2.nth_mut(3) = to;
                AABB::from_corners(p1, p2)
            }
        }
    }
}

impl<P: PointTrait> RTreeObject for AABB<P> {
    type Envelope = rstar::AABB<P>;

    fn envelope(&self) -> Self::Envelope {
        self.bounds.to_owned()
    }
}

impl<P: PointTrait> Envelope for AABB<P> {
    type Point = P;

    fn new_empty() -> Self {
        AABB {
            bounds: rstar::AABB::new_empty(),
        }
    }

    fn contains_point(&self, point: &Self::Point) -> bool {
        self.bounds.contains_point(point)
    }

    fn contains_envelope(&self, aabb: &Self) -> bool {
        self.bounds.contains_envelope(&aabb.bounds)
    }

    fn merge(&mut self, other: &Self) {
        self.bounds.merge(&other.bounds)
    }

    fn merged(&self, other: &Self) -> Self {
        AABB {
            bounds: self.bounds.merged(&other.bounds),
        }
    }

    fn intersects(&self, other: &Self) -> bool {
        self.bounds.intersects(&other.bounds)
    }

    fn intersection_area(&self, other: &Self) -> <Self::Point as rstar::Point>::Scalar {
        self.bounds.intersection_area(&other.bounds)
    }

    fn area(&self) -> <Self::Point as rstar::Point>::Scalar {
        self.bounds.area()
    }

    fn distance_2(&self, point: &Self::Point) -> <Self::Point as rstar::Point>::Scalar {
        self.bounds.distance_2(point)
    }

    fn min_max_dist_2(&self, point: &Self::Point) -> <Self::Point as rstar::Point>::Scalar {
        self.bounds.min_max_dist_2(point)
    }

    fn center(&self) -> Self::Point {
        self.bounds.center()
    }

    fn perimeter_value(&self) -> <Self::Point as rstar::Point>::Scalar {
        self.bounds.perimeter_value()
    }

    fn sort_envelopes<T: RTreeObject<Envelope = Self>>(axis: usize, envelopes: &mut [T]) {
        envelopes.sort_by(|l, r| {
            l.envelope()
                .lower()
                .nth(axis)
                .partial_cmp(&r.envelope().lower().nth(axis))
                .unwrap()
        });
    }

    fn partition_envelopes<T: RTreeObject<Envelope = Self>>(
        axis: usize,
        envelopes: &mut [T],
        selection_size: usize,
    ) {
        envelopes.select_nth_unstable_by(selection_size, |l, r| {
            l.envelope()
                .lower()
                .nth(axis)
                .partial_cmp(&r.envelope().lower().nth(axis))
                .unwrap()
        });
    }
}
