use std::{fmt::Debug, sync::Arc};

use datafusion::arrow::{
    array::{
        ArrayRef, BooleanArray, BooleanBuilder, Float32Array, Float32Builder, Float64Array,
        Float64Builder, Int32Array, Int32Builder, StructArray, UInt8Array, UInt8Builder,
        UInt16Array, UInt16Builder,
    },
    datatypes::SchemaRef,
};
use las::{Header, Point};

#[derive(Debug)]
pub(super) struct RowBuilder {
    raw: bool,
    x: Float64Builder,
    y: Float64Builder,
    z: Float64Builder,
    x_int: Int32Builder,
    y_int: Int32Builder,
    z_int: Int32Builder,
    x_scale: Float64Builder,
    y_scale: Float64Builder,
    z_scale: Float64Builder,
    x_offset: Float64Builder,
    y_offset: Float64Builder,
    z_offset: Float64Builder,
    intensity: UInt16Builder,
    return_number: UInt8Builder,
    number_of_returns: UInt8Builder,
    is_synthetic: BooleanBuilder,
    is_key_point: BooleanBuilder,
    is_withheld: BooleanBuilder,
    is_overlap: BooleanBuilder,
    scanner_channel: UInt8Builder,
    scan_direction: UInt8Builder,
    is_edge_of_flight_line: BooleanBuilder,
    classification: UInt8Builder,
    user_data: UInt8Builder,
    scan_angle: Float32Builder,
    point_source_id: UInt16Builder,
    gps_time: Float64Builder,
    red: UInt16Builder,
    green: UInt16Builder,
    blue: UInt16Builder,
    nir: UInt16Builder,
}

impl RowBuilder {
    pub(super) fn new(capacity: usize, raw: bool) -> Self {
        Self {
            raw,
            x: Float64Array::builder(capacity),
            y: Float64Array::builder(capacity),
            z: Float64Array::builder(capacity),
            x_int: Int32Array::builder(capacity),
            y_int: Int32Array::builder(capacity),
            z_int: Int32Array::builder(capacity),
            x_scale: Float64Array::builder(capacity),
            y_scale: Float64Array::builder(capacity),
            z_scale: Float64Array::builder(capacity),
            x_offset: Float64Array::builder(capacity),
            y_offset: Float64Array::builder(capacity),
            z_offset: Float64Array::builder(capacity),
            intensity: UInt16Array::builder(capacity),
            return_number: UInt8Array::builder(capacity),
            number_of_returns: UInt8Array::builder(capacity),
            is_synthetic: BooleanArray::builder(capacity),
            is_key_point: BooleanArray::builder(capacity),
            is_withheld: BooleanArray::builder(capacity),
            is_overlap: BooleanArray::builder(capacity),
            scanner_channel: UInt8Array::builder(capacity),
            scan_direction: UInt8Array::builder(capacity),
            is_edge_of_flight_line: BooleanArray::builder(capacity),
            classification: UInt8Array::builder(capacity),
            user_data: UInt8Array::builder(capacity),
            scan_angle: Float32Array::builder(capacity),
            point_source_id: UInt16Array::builder(capacity),
            gps_time: Float64Array::builder(capacity),
            red: UInt16Array::builder(capacity),
            green: UInt16Array::builder(capacity),
            blue: UInt16Array::builder(capacity),
            nir: UInt16Array::builder(capacity),
        }
    }

    pub(super) fn append(&mut self, p: Point, header: &Header) {
        if self.raw {
            let transforms = header.transforms();
            let raw = p.clone().into_raw(transforms).expect("transform into raw");
            self.x_int.append_value(raw.x);
            self.y_int.append_value(raw.y);
            self.z_int.append_value(raw.z);
            self.x_scale.append_value(transforms.x.scale);
            self.y_scale.append_value(transforms.y.scale);
            self.z_scale.append_value(transforms.z.scale);
            self.x_offset.append_value(transforms.x.offset);
            self.y_offset.append_value(transforms.y.offset);
            self.z_offset.append_value(transforms.z.offset);
        } else {
            self.x.append_value(p.x);
            self.y.append_value(p.y);
            self.z.append_value(p.z);
        }

        self.intensity.append_option(Some(p.intensity));
        self.return_number.append_value(p.return_number);
        self.number_of_returns.append_value(p.number_of_returns);
        self.is_synthetic.append_value(p.is_synthetic);
        self.is_key_point.append_value(p.is_key_point);
        self.is_withheld.append_value(p.is_withheld);
        self.is_overlap.append_value(p.is_overlap);
        self.scanner_channel.append_value(p.scanner_channel);
        self.scan_direction.append_value(p.scan_direction as u8);
        self.is_edge_of_flight_line
            .append_value(p.is_edge_of_flight_line);
        self.classification.append_value(u8::from(p.classification));
        self.user_data.append_value(p.user_data);
        self.scan_angle.append_value(p.scan_angle);
        self.point_source_id.append_value(p.point_source_id);
        if header.point_format().has_gps_time {
            self.gps_time.append_value(p.gps_time.unwrap());
        }
        if header.point_format().has_color {
            let color = p.color.unwrap();
            self.red.append_value(color.red);
            self.green.append_value(color.green);
            self.blue.append_value(color.blue);
        }
        if header.point_format().has_nir {
            self.nir.append_value(p.nir.unwrap());
        }
    }

    /// Note: returns StructArray to allow nesting within another array if desired
    pub(super) fn finish(&mut self, schema: &SchemaRef, header: &Header) -> StructArray {
        let mut columns = Vec::new();
        if self.raw {
            columns.extend([
                Arc::new(self.x_int.finish()) as ArrayRef,
                Arc::new(self.y_int.finish()) as ArrayRef,
                Arc::new(self.z_int.finish()) as ArrayRef,
                Arc::new(self.x_scale.finish()) as ArrayRef,
                Arc::new(self.y_scale.finish()) as ArrayRef,
                Arc::new(self.z_scale.finish()) as ArrayRef,
                Arc::new(self.x_offset.finish()) as ArrayRef,
                Arc::new(self.y_offset.finish()) as ArrayRef,
                Arc::new(self.z_offset.finish()) as ArrayRef,
            ])
        } else {
            columns.extend([
                Arc::new(self.x.finish()) as ArrayRef,
                Arc::new(self.y.finish()) as ArrayRef,
                Arc::new(self.z.finish()) as ArrayRef,
            ])
        }
        columns.extend([
            Arc::new(self.intensity.finish()) as ArrayRef,
            Arc::new(self.return_number.finish()) as ArrayRef,
            Arc::new(self.number_of_returns.finish()) as ArrayRef,
            Arc::new(self.is_synthetic.finish()) as ArrayRef,
            Arc::new(self.is_key_point.finish()) as ArrayRef,
            Arc::new(self.is_withheld.finish()) as ArrayRef,
            Arc::new(self.is_overlap.finish()) as ArrayRef,
            Arc::new(self.scanner_channel.finish()) as ArrayRef,
            Arc::new(self.scan_direction.finish()) as ArrayRef,
            Arc::new(self.is_edge_of_flight_line.finish()) as ArrayRef,
            Arc::new(self.classification.finish()) as ArrayRef,
            Arc::new(self.user_data.finish()) as ArrayRef,
            Arc::new(self.scan_angle.finish()) as ArrayRef,
            Arc::new(self.point_source_id.finish()) as ArrayRef,
        ]);
        if header.point_format().has_gps_time {
            columns.push(Arc::new(self.gps_time.finish()) as ArrayRef);
        }
        if header.point_format().has_color {
            columns.extend([
                Arc::new(self.red.finish()) as ArrayRef,
                Arc::new(self.green.finish()) as ArrayRef,
                Arc::new(self.blue.finish()) as ArrayRef,
            ]);
        }
        if header.point_format().has_nir {
            columns.push(Arc::new(self.nir.finish()) as ArrayRef);
        }
        StructArray::new(schema.fields.to_owned(), columns, None)
    }
}
