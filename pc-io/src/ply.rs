use std::{
    collections::HashMap,
    fs::File,
    io::{BufRead, BufReader, BufWriter, Write},
    path::Path,
    sync::Arc,
};

use datafusion::arrow::{
    array::{
        ArrayRef, AsArray, Float32Array, Float64Array, Int8Array, Int16Array, Int32Array,
        UInt8Array, UInt16Array, UInt32Array,
    },
    datatypes::{
        DataType, Field, Float32Type, Float64Type, Int16Type, Int32Type, Schema, SchemaRef,
        UInt8Type, UInt16Type, UInt32Type,
    },
    error::ArrowError,
    record_batch::{RecordBatch, RecordBatchReader},
};
use ply_rs::{
    ply::{
        Addable, DefaultElement, ElementDef, Header, Ply, Property, PropertyDef, PropertyType,
        ScalarType,
    },
    writer::Writer,
};

use pc_format::schema::{PC_DIMENSION_KEY, PC_LOCATION_KEY};

use crate::config::DEFAULT_BATCH_SIZE;

pub use ply_rs::ply::Encoding;

/// Default vertex element name
const VERTEX: &str = "vertex";

/// Ply point cloud reader
pub struct PlyReader {
    reader: Box<dyn BufRead + Send>,
    header: Header,
}

impl PlyReader {
    pub fn from_path<P: AsRef<std::path::Path>>(path: P) -> std::io::Result<Self> {
        let file = File::open(path).unwrap();
        let mut reader = BufReader::new(file);

        // create a parser
        let p = ply_rs::parser::Parser::<DefaultElement>::new();

        // read the header
        let header = p.read_header(&mut reader).expect("parse ply header");
        Ok(PlyReader {
            reader: Box::new(reader),
            header,
        })
    }

    pub fn record_batch_reader(&mut self) -> PlyRecordBatchReader {
        self.into_iter()
    }
}

impl IntoIterator for &mut PlyReader {
    type Item = Result<RecordBatch, ArrowError>;

    type IntoIter = PlyRecordBatchReader;

    fn into_iter(self) -> Self::IntoIter {
        // create a parser
        let p = ply_rs::parser::Parser::<DefaultElement>::new();

        // get vertex element
        let element = self.header.elements.get(VERTEX).expect("vertex element");

        // read payload
        let payload = p
            .read_payload_for_element(&mut self.reader, element, &self.header)
            .unwrap();

        // extract fields and columns
        let mut fields = Vec::new();
        let mut columns = Vec::new();

        for (name, property) in &element.properties {
            match &property.data_type {
                PropertyType::Scalar(s) => match s {
                    ScalarType::Char => {
                        fields.push(Field::new(name, DataType::Int8, false));

                        let column = Int8Array::from_iter(payload.iter().map(|r| match r[name] {
                            Property::Char(v) => v,
                            _ => panic!(),
                        }));
                        columns.push(Arc::new(column) as ArrayRef);
                    }
                    ScalarType::UChar => {
                        fields.push(Field::new(name, DataType::UInt8, false));

                        let column = UInt8Array::from_iter(payload.iter().map(|r| match r[name] {
                            Property::UChar(v) => v,
                            _ => panic!(),
                        }));
                        columns.push(Arc::new(column) as ArrayRef);
                    }
                    ScalarType::Short => {
                        fields.push(Field::new(name, DataType::Int16, false));

                        let column = Int16Array::from_iter(payload.iter().map(|r| match r[name] {
                            Property::Short(v) => v,
                            _ => panic!(),
                        }));
                        columns.push(Arc::new(column) as ArrayRef);
                    }
                    ScalarType::UShort => {
                        fields.push(Field::new(name, DataType::UInt16, false));

                        let column =
                            UInt16Array::from_iter(payload.iter().map(|r| match r[name] {
                                Property::UShort(v) => v,
                                _ => panic!(),
                            }));
                        columns.push(Arc::new(column) as ArrayRef);
                    }
                    ScalarType::Int => {
                        fields.push(Field::new(name.to_owned(), DataType::Int32, false));

                        let column = Int32Array::from_iter(payload.iter().map(|r| match r[name] {
                            Property::Int(v) => v,
                            _ => panic!(),
                        }));
                        columns.push(Arc::new(column) as ArrayRef);
                    }
                    ScalarType::UInt => {
                        fields.push(Field::new(name.to_owned(), DataType::UInt32, false));

                        let column =
                            UInt32Array::from_iter(payload.iter().map(|r| match r[name] {
                                Property::UInt(v) => v,
                                _ => panic!(),
                            }));
                        columns.push(Arc::new(column) as ArrayRef);
                    }
                    ScalarType::Float => {
                        fields.push(Field::new(name.to_owned(), DataType::Float32, false));

                        let column =
                            Float32Array::from_iter(payload.iter().map(|r| match r[name] {
                                Property::Float(v) => v,
                                _ => panic!(),
                            }));
                        columns.push(Arc::new(column) as ArrayRef);
                    }
                    ScalarType::Double => {
                        fields.push(Field::new(name.to_owned(), DataType::Float64, false));

                        let column =
                            Float64Array::from_iter(payload.iter().map(|r| match r[name] {
                                Property::Double(v) => v,
                                _ => panic!(),
                            }));
                        columns.push(Arc::new(column) as ArrayRef);
                    }
                },
                PropertyType::List(_, _) => unimplemented!(),
            };

            if let Some(metadata) = match name.as_str() {
                "x" => Some(HashMap::from([
                    (PC_DIMENSION_KEY.to_owned(), "1".to_owned()),
                    (PC_LOCATION_KEY.to_owned(), "x".to_string()),
                ])),
                "y" => Some(HashMap::from([
                    (PC_DIMENSION_KEY.to_owned(), "2".to_owned()),
                    (PC_LOCATION_KEY.to_owned(), "y".to_string()),
                ])),
                "z" => Some(HashMap::from([
                    (PC_DIMENSION_KEY.to_owned(), "3".to_owned()),
                    (PC_LOCATION_KEY.to_owned(), "z".to_string()),
                ])),
                _ => None,
            } {
                fields.last_mut().unwrap().set_metadata(metadata);
            }
        }

        let schema = Schema::new(fields);
        let batch = RecordBatch::try_new(schema.into(), columns).unwrap();

        PlyRecordBatchReader { batch, offset: 0 }
    }
}

pub struct PlyRecordBatchReader {
    batch: RecordBatch,
    offset: usize,
}

impl RecordBatchReader for PlyRecordBatchReader {
    fn schema(&self) -> SchemaRef {
        self.batch.schema()
    }
}

impl Iterator for PlyRecordBatchReader {
    type Item = Result<RecordBatch, ArrowError>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.offset < self.batch.num_rows() {
            let length = DEFAULT_BATCH_SIZE.min(self.batch.num_rows() - self.offset);
            let batch = self.batch.slice(self.offset, length);
            self.offset += length;

            Some(Ok(batch))
        } else {
            None
        }
    }
}

/// Ply point cloud writer
pub struct PlyWriter {
    dst: BufWriter<File>,
    w: Writer<DefaultElement>,
    ply: Ply<DefaultElement>,
    elements: Vec<DefaultElement>,
    header: bool,
}

impl PlyWriter {
    pub fn try_new<P: AsRef<Path>>(
        path: P,
        schema: SchemaRef,
        encoding: Encoding,
        count: Option<usize>,
    ) -> Result<Self, ArrowError> {
        // crete a ply object
        let mut ply = Ply::<DefaultElement>::new();
        ply.header.encoding = encoding;
        ply.header.elements.add(element_def_from_schema(&schema));

        // create writer
        let mut writer = PlyWriter {
            dst: BufWriter::new(File::create(path.as_ref())?),
            w: ply_rs::writer::Writer::new(),
            ply,
            elements: Vec::new(),
            header: false,
        };

        // write header if count is known
        if let Some(count) = count {
            writer.ply.header.elements.get_mut(VERTEX).unwrap().count = count;
            writer.w.write_header(&mut writer.dst, &writer.ply.header)?;
            writer.header = true;
        }

        Ok(writer)
    }

    pub fn write(&mut self, batch: &RecordBatch) -> Result<(), ArrowError> {
        let elements = (0..batch.num_rows()).map(|i| element_from_row(i, batch));
        if self.header {
            // stream
            let element_def = self.ply.header.elements.get(VERTEX).unwrap();
            match self.ply.header.encoding {
                Encoding::Ascii => {
                    for element in elements {
                        self.w
                            .write_ascii_element(&mut self.dst, &element, element_def)
                            .unwrap();
                    }
                }
                Encoding::BinaryBigEndian => {
                    for element in elements {
                        self.w
                            .write_big_endian_element(&mut self.dst, &element, element_def)
                            .unwrap();
                    }
                }
                Encoding::BinaryLittleEndian => {
                    for element in elements {
                        self.w
                            .write_little_endian_element(&mut self.dst, &element, element_def)
                            .unwrap();
                    }
                }
            }
        } else {
            // collect
            self.elements.extend(elements);
        }

        Ok(())
    }

    pub fn close(mut self) -> Result<(), ArrowError> {
        if self.elements.is_empty() {
            // write ply elements
            self.w.write_ply(&mut self.dst, &mut self.ply)?;
        }

        self.dst.flush()?;

        Ok(())
    }
}

fn element_def_from_schema(schema: &Schema) -> ElementDef {
    let mut element = ElementDef::new(VERTEX.to_string());

    schema.fields().iter().for_each(|f| {
        let property = property_definition_from_field(f);
        element.properties.add(property);
    });

    element
}

fn property_definition_from_field(field: &Field) -> PropertyDef {
    let data_type = match field.data_type() {
        DataType::Int8 => PropertyType::Scalar(ScalarType::Char),
        DataType::Int16 => PropertyType::Scalar(ScalarType::Short),
        DataType::Int32 => PropertyType::Scalar(ScalarType::Int),
        DataType::Int64 => todo!("try cast to i32 or use list"),
        DataType::UInt8 => PropertyType::Scalar(ScalarType::UChar),
        DataType::UInt16 => PropertyType::Scalar(ScalarType::UShort),
        DataType::UInt32 => PropertyType::Scalar(ScalarType::UInt),
        DataType::UInt64 => todo!("try cast to u32 or use list"),
        DataType::Float32 => PropertyType::Scalar(ScalarType::Float),
        DataType::Float64 => PropertyType::Scalar(ScalarType::Double),
        x => unimplemented!("{x}"),
    };
    PropertyDef::new(field.name().to_owned(), data_type)
}

fn element_from_row(i: usize, batch: &RecordBatch) -> DefaultElement {
    let mut element = DefaultElement::new();
    for f in batch.schema().fields().iter() {
        let column = batch.column_by_name(f.name()).unwrap();
        let value = match f.data_type() {
            DataType::Int8 => Property::Short(column.as_primitive::<Int16Type>().value(i)),
            DataType::Int16 => Property::UShort(column.as_primitive::<UInt16Type>().value(i)),
            DataType::Int32 => Property::Int(column.as_primitive::<Int32Type>().value(i)),
            DataType::Int64 => todo!(),
            DataType::UInt8 => Property::UChar(column.as_primitive::<UInt8Type>().value(i)),
            DataType::UInt16 => Property::UShort(column.as_primitive::<UInt16Type>().value(i)),
            DataType::UInt32 => Property::UInt(column.as_primitive::<UInt32Type>().value(i)),
            DataType::UInt64 => todo!(),
            DataType::Float32 => Property::Float(column.as_primitive::<Float32Type>().value(i)),
            DataType::Float64 => Property::Double(column.as_primitive::<Float64Type>().value(i)),
            x => unimplemented!("{x}"),
        };

        element.insert(f.name().to_owned(), value);
    }

    element
}

#[cfg(test)]
mod tests {
    use datafusion::prelude::SessionContext;
    use futures::StreamExt;
    use ply_rs::{parser::Parser, writer::Writer};
    use rstar::{Envelope, Point, RTreeObject};

    use pc_format::{AABB, PointCloud, PointTrait, PointXYZ};

    use crate::las::LasDataSource;

    use super::*;

    #[test]
    fn ascii_to_binary() {
        // read
        let mut f = std::fs::File::open("../assets/sofa.ply").unwrap();
        let p = Parser::<DefaultElement>::new();
        let mut ply = p.read_ply(&mut f).unwrap();
        // println!("Ply header: {:#?}", ply.header);

        // convert
        ply.header.encoding = Encoding::BinaryLittleEndian;

        // writer
        let mut out = std::fs::File::create("../data/sofa_bin.ply").unwrap();
        let w = Writer::new();
        w.write_ply(&mut out, &mut ply).unwrap();
    }

    #[test]
    fn read() {
        let mut reader = PlyReader::from_path("../assets/sofa.ply").unwrap();

        let pc = PointCloud::from(reader.record_batch_reader());

        assert_eq!(pc.num_points(), 10000);

        dbg!(pc.aabb::<PointXYZ<f64>>());
    }

    #[tokio::test]
    async fn transform() {
        let mut reader = PlyReader::from_path("../assets/sofa.ply").unwrap();

        let pc = PointCloud::from(reader.record_batch_reader());

        let aabb: AABB<PointXYZ<f64>> = pc.aabb();

        // translate
        let center: Vec<f64> = aabb.envelope().center().coords().collect();
        let translate =
            nalgebra::Translation3::from(nalgebra::Point::from_slice(&center)).inverse();

        // scale
        let scaler = aabb
            .upper()
            .sub(&aabb.lower())
            .coords()
            .max_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap();

        let scale = nalgebra::Scale3::from([2. / scaler; 3]);

        // tranform
        let pc = PointCloud::from_iter(pc.points().map(|p: PointXYZ<f64>| -> PointXYZ<f64> {
            let p: Vec<f64> = p.coords().collect();
            let coords = scale
                .transform_point(&translate.transform_point(&nalgebra::Point::from_slice(&p)))
                .coords;
            PointTrait::from_slice(coords.as_slice())
        }));

        // write
        let mut writer = PlyWriter::try_new(
            "../data/sofa_transformed.ply",
            pc.schema(),
            Encoding::Ascii,
            Some(pc.num_points()),
        )
        .unwrap();
        for entry in pc.store.iter() {
            for batch in entry.value().read().unwrap().read(None, None).unwrap() {
                writer.write(&batch.unwrap()).unwrap();
            }
        }
        writer.close().unwrap();

        let extent: AABB<PointXYZ<f64>> = pc.aabb();
        let bounds = AABB::from_corners(
            PointTrait::from_slice(&[-1.; 3]),
            PointTrait::from_slice(&[1.; 3]),
        );

        assert!(bounds.envelope().contains_envelope(&extent.envelope()));
    }

    #[ignore]
    #[tokio::test]
    async fn convert() {
        let ctx = SessionContext::new();

        let ds = LasDataSource::try_new(&[
            "../data/CampusOttobrunn/OTN_ULS_20241125.laz",
            "../data/CampusOttobrunn/OTN_TLS_20241029.laz",
            // "../data/Downtown/TUM_Downtown_Photogrammetry_20241217.laz",s
            // "../data/Downtown/TUM_Downtown_ULS_20241217_manual.laz",
            // "../data/Downtown/TUM_Downtown_ULS_20241217_manual_5mm_fillparts.laz",
            // "../data/Downtown/TUM_Downtown_ULS_20241217_nadir.laz",
        ])
        .unwrap();
        ctx.register_table("laz", Arc::new(ds)).unwrap();
        let df = ctx.table("laz").await.unwrap();

        let bounds: AABB<PointXYZ<f64>> = pc_format::expressions::df_aabb(&df).await.unwrap();
        let center = bounds.center();

        let df = ctx
            .sql(&format!(
                r#"
                select
                    arrow_cast(x - {}, 'Float32') as x, 
                    arrow_cast(y - {}, 'Float32') as y, 
                    arrow_cast(z - {}, 'Float32') as z,
                    arrow_cast(arrow_cast(red, 'Float64') / {m16} * {m8}, 'UInt8') as red, 
                    arrow_cast(arrow_cast(green, 'Float64') / {m16} * {m8}, 'UInt8') as green, 
                    arrow_cast(arrow_cast(blue, 'Float64') / {m16} * {m8}, 'UInt8') as blue 
                from laz
                where red > 0 and green > 0 and blue > 0
                "#,
                center.nth(0),
                center.nth(1),
                center.nth(2),
                m16 = u16::MAX as f64,
                m8 = u8::MAX as f64,
            ))
            .await
            .unwrap();

        // write
        df.clone()
            .write_parquet(
                "../data/CampusOttobrunn/otn.parquet",
                Default::default(),
                None,
            )
            .await
            .unwrap();

        let count = df.clone().count().await.unwrap();

        let mut writer = PlyWriter::try_new(
            "../data/CampusOttobrunn/otn.ply",
            // "../data/Downtown/photogrammetry.ply",
            // "../data/Downtown/uls.ply",
            Arc::new(df.schema().as_arrow().to_owned()),
            Encoding::BinaryLittleEndian,
            Some(count),
        )
        .unwrap();

        let mut stream = df.execute_stream().await.unwrap();

        while let Some(result) = stream.next().await {
            writer.write(&result.unwrap()).unwrap();
        }

        writer.close().unwrap();
    }
}
