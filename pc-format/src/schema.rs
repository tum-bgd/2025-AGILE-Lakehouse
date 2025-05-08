use std::{borrow::BorrowMut, sync::Arc};

use datafusion::arrow::datatypes::{DataType, Field, Schema, SchemaRef};
use itertools::Itertools;

use crate::PointCloudError;

/// Indexable dimension like location, time or importance.
///
/// Integer values define order.
pub const PC_DIMENSION_KEY: &str = "PC:dimension";

pub const PC_LOCATION_KEY: &str = "PC:location";
pub const PC_IMPORTANCE_KEY: &str = "PC:importance";

pub const PC_OFFSET_KEY: &str = "PC:offset";
pub const PC_SCALE_KEY: &str = "PC:scale";

/// extract dimensions from schema
pub fn dimensions(schema: &SchemaRef) -> Vec<usize> {
    schema
        .fields()
        .iter()
        .enumerate()
        .filter(|(_, f)| f.metadata().contains_key(PC_DIMENSION_KEY))
        .sorted_by_key(|(_, f)| f.metadata().get(PC_DIMENSION_KEY))
        .map(|(i, _)| i)
        .collect_vec()
}

/// test whether schema has importance dimension
pub fn importance(schema: &SchemaRef) -> Option<usize> {
    schema
        .fields()
        .iter()
        .find_position(|f| {
            f.metadata().contains_key(PC_DIMENSION_KEY)
                && f.metadata().contains_key(PC_IMPORTANCE_KEY)
        })
        .map(|(i, _)| i)
}

/// add random importance
pub fn add_importance(
    schema: SchemaRef,
    name: impl Into<String>,
    data_type: DataType,
    index: usize,
) -> SchemaRef {
    let field = Field::new(name, data_type, false);

    // update definition if already exists
    if let Some(i) = importance(&schema) {
        *schema.field(i).borrow_mut() = &field;
        return schema;
    }

    // add field if missing
    let mut metadata = field.metadata().to_owned();
    if !metadata.contains_key(PC_DIMENSION_KEY) {
        metadata.insert(
            PC_DIMENSION_KEY.to_owned(),
            (dimensions(&schema).len() + 1).to_string(),
        );
    }
    if !metadata.contains_key(PC_IMPORTANCE_KEY) {
        metadata.insert(PC_IMPORTANCE_KEY.to_owned(), "random".to_string());
    }

    let mut fields = schema.fields().to_vec();
    fields.insert(index, Arc::new(field.to_owned().with_metadata(metadata)));
    Arc::new(Schema::new(fields))
}

/// check for point cloud schema validity
pub fn validate(schema: &SchemaRef) -> Result<(), PointCloudError> {
    let dimensions = dimensions(schema);

    // assert schema has at least 3 dimensions
    if dimensions.len() < 3 {
        return Err(PointCloudError::SchemaError(
            "schema has at least 3 dimensions".to_string(),
        ));
    }

    // assert all dimensions have a numeric data type
    if !dimensions
        .iter()
        .all(|i| schema.field(*i).data_type().is_numeric())
    {
        return Err(PointCloudError::SchemaError(
            "schema has non numeric dimensions specified".to_string(),
        ));
    }

    Ok(())
}
