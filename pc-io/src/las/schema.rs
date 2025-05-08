use std::{collections::HashMap, sync::Arc};

use datafusion::arrow::datatypes::{DataType, Field, Schema, SchemaRef};

use pc_format::schema::{PC_DIMENSION_KEY, PC_LOCATION_KEY, PC_OFFSET_KEY, PC_SCALE_KEY};

// Arrow schema for LAS points
pub(super) fn schema_from_header(header: &las::Header, raw: bool) -> SchemaRef {
    let mut fields = Vec::new();

    if raw {
        fields.extend([
            Field::new("x", DataType::Int32, false).with_metadata(HashMap::from([
                (PC_DIMENSION_KEY.to_owned(), "1".to_owned()),
                (PC_LOCATION_KEY.to_owned(), "x".to_string()),
            ])),
            Field::new("y", DataType::Int32, false).with_metadata(HashMap::from([
                (PC_DIMENSION_KEY.to_owned(), "2".to_owned()),
                (PC_LOCATION_KEY.to_owned(), "y".to_string()),
            ])),
            Field::new("z", DataType::Int32, false).with_metadata(HashMap::from([
                (PC_DIMENSION_KEY.to_owned(), "3".to_owned()),
                (PC_LOCATION_KEY.to_owned(), "z".to_string()),
            ])),
            Field::new("x_scale", DataType::Float64, false)
                .with_metadata(HashMap::from([(PC_SCALE_KEY.to_owned(), "x".to_string())])),
            Field::new("y_scale", DataType::Float64, false)
                .with_metadata(HashMap::from([(PC_SCALE_KEY.to_owned(), "y".to_string())])),
            Field::new("z_scale", DataType::Float64, false)
                .with_metadata(HashMap::from([(PC_SCALE_KEY.to_owned(), "z".to_string())])),
            Field::new("x_offset", DataType::Float64, false)
                .with_metadata(HashMap::from([(PC_OFFSET_KEY.to_owned(), "x".to_string())])),
            Field::new("y_offset", DataType::Float64, false)
                .with_metadata(HashMap::from([(PC_OFFSET_KEY.to_owned(), "y".to_string())])),
            Field::new("z_offset", DataType::Float64, false)
                .with_metadata(HashMap::from([(PC_OFFSET_KEY.to_owned(), "z".to_string())])),
        ])
    } else {
        fields.extend([
            Field::new("x", DataType::Float64, false).with_metadata(HashMap::from([
                (PC_DIMENSION_KEY.to_owned(), "1".to_owned()),
                (PC_LOCATION_KEY.to_owned(), "x".to_string()),
            ])),
            Field::new("y", DataType::Float64, false).with_metadata(HashMap::from([
                (PC_DIMENSION_KEY.to_owned(), "2".to_owned()),
                (PC_LOCATION_KEY.to_owned(), "y".to_string()),
            ])),
            Field::new("z", DataType::Float64, false).with_metadata(HashMap::from([
                (PC_DIMENSION_KEY.to_owned(), "3".to_owned()),
                (PC_LOCATION_KEY.to_owned(), "z".to_string()),
            ])),
        ])
    }

    fields.extend([
        Field::new("intensity", DataType::UInt16, true),
        Field::new("return_number", DataType::UInt8, false),
        Field::new("number_of_returns", DataType::UInt8, false),
        Field::new("is_synthetic", DataType::Boolean, false),
        Field::new("is_key_point", DataType::Boolean, false),
        Field::new("is_withheld", DataType::Boolean, false),
        Field::new("is_overlap", DataType::Boolean, false),
        Field::new("scanner_channel", DataType::UInt8, false),
        Field::new("scan_direction", DataType::UInt8, false),
        Field::new("is_edge_of_flight_line", DataType::Boolean, false),
        Field::new("classification", DataType::UInt8, false),
        Field::new("user_data", DataType::UInt8, false),
        Field::new("scan_angle", DataType::Float32, false),
        Field::new("point_source_id", DataType::UInt16, false),
    ]);
    if header.point_format().has_gps_time {
        fields.push(Field::new("gps_time", DataType::Float64, false));
    }
    if header.point_format().has_color {
        fields.extend([
            Field::new("red", DataType::UInt16, false),
            Field::new("green", DataType::UInt16, false),
            Field::new("blue", DataType::UInt16, false),
        ])
    }
    if header.point_format().has_nir {
        fields.push(Field::new("nir", DataType::UInt16, false));
    }
    Arc::new(Schema::new(fields))
}
