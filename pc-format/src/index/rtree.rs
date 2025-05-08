use rstar::{RStarInsertionStrategy, RTree, RTreeParams, primitives::GeomWithData};

use crate::{AABB, PointTrait};

/// Custom R*-Tree parameters for point clouds
#[derive(Debug, Clone)]
pub struct RstarParams;

impl RTreeParams for RstarParams {
    const MIN_SIZE: usize = 4;
    const MAX_SIZE: usize = 16;
    const REINSERTION_COUNT: usize = 2;
    type DefaultInsertionStrategy = RStarInsertionStrategy;
}

/// Point cloud rtree index
#[derive(Debug, Clone)]
pub enum RtreeIndex<P: PointTrait>
where
    <P as rstar::Point>::Scalar: num_traits::NumCast,
{
    Point(Box<PointIndex<P>>),
    Batch(Box<BatchIndex<P>>),
    Multi(Box<MultiLevelIndex<P>>),
    None,
}

pub type PointIndex<P> = RTree<GeomWithData<P, usize>, RstarParams>;
pub type BatchIndex<P> = RTree<GeomWithData<AABB<P>, String>, RstarParams>;
pub type MultiLevelIndex<P> = RTree<GeomWithData<AABB<P>, (String, PointIndex<P>)>>;
