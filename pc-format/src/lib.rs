mod aabb;
pub use aabb::AABB;

mod chunk;
pub use chunk::PointChunk;

pub mod compute;

mod dims;
pub use dims::Dims;

mod exec;
pub use exec::PointCloudExec;

pub mod expressions;

pub mod framework;

pub mod helpers;

pub mod index;

mod point;
pub use point::{Coord, PointTrait, PointXY, PointXYI, PointXYZ, PointXYZI};

pub mod schema;

mod pointcloud;
pub use pointcloud::PointCloud;

mod table;

#[derive(thiserror::Error, Debug)]
pub enum PointCloudError {
    #[error("arrow error")]
    ArrowError(#[from] datafusion::arrow::error::ArrowError),
    #[error("datafusion error")]
    DataFusionError(#[from] datafusion::error::DataFusionError),
    #[error("cache error: {0}")]
    CacheError(String),
    #[error("schema validation error: {0}")]
    SchemaError(String),
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
}
