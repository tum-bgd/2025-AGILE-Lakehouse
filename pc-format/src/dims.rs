use serde::{Deserialize, Serialize};

use crate::PointCloudError;

#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Dims {
    XY,
    XYZ,
    XYI,
    XYZI,
}

impl Dims {
    pub const fn names(&self) -> &[&str] {
        match self {
            Dims::XY => &["x", "y"],
            Dims::XYZ => &["x", "y", "z"],
            Dims::XYI => &["x", "y", "i"],
            Dims::XYZI => &["x", "y", "z", "i"],
        }
    }
}

impl<T> TryFrom<&[T]> for Dims
where
    T: AsRef<str>,
{
    type Error = PointCloudError;

    fn try_from(value: &[T]) -> Result<Self, Self::Error> {
        let dimension: Vec<&str> = value.iter().map(|d| d.as_ref()).collect();
        match dimension.as_slice() {
            &["x", "y"] => Ok(Dims::XY),
            &["x", "y", "z"] => Ok(Dims::XYZ),
            &["x", "y", "i"] => Ok(Dims::XYI),
            &["x", "y", "z", "i"] => Ok(Dims::XYZI),
            d => Err(PointCloudError::SchemaError(format!(
                "Unknown dims `{d:?}`"
            ))),
        }
    }
}
