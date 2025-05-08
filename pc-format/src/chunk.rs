use std::{fs::File, path::PathBuf};

use datafusion::{
    arrow::{
        array::{RecordBatch, RecordBatchIterator, RecordBatchReader},
        compute::filter_record_batch,
        datatypes::SchemaRef,
        error::ArrowError,
        ipc::{reader::FileReader, writer::FileWriter},
    },
    common::project_schema,
};
use rstar::{Envelope, primitives::GeomWithData};

use crate::{
    AABB, PointCloudError, PointXYZI,
    compute::record_batch_aabb,
    index::rtree::{PointIndex, RtreeIndex},
    pointcloud::points,
};

/// Point chunk
#[derive(Debug)]
pub struct PointChunk {
    id: String,
    schema: SchemaRef,
    dir: PathBuf,
    data: Vec<RecordBatch>,
    bounds: AABB<PointXYZI<f64>>,
    index: RtreeIndex<PointXYZI<f64>>,
}

impl PointChunk {
    pub fn new(id: String, dir: PathBuf, schema: SchemaRef) -> Self {
        Self {
            id,
            dir,
            schema,
            data: Vec::new(),
            bounds: AABB::new_empty(),
            index: RtreeIndex::None,
        }
    }

    pub fn bounds(&self) -> &AABB<PointXYZI<f64>> {
        &self.bounds
    }

    pub fn path(&self) -> PathBuf {
        self.dir.join(format!("{}.arrow", self.id))
    }

    pub fn push(&mut self, batch: RecordBatch) {
        assert!(batch.schema().contains(&self.schema));

        self.bounds.merge(&record_batch_aabb(&batch).unwrap());

        self.data.push(batch);
    }

    pub fn read(
        &self,
        projection: Option<Vec<usize>>,
        indices: Option<Vec<usize>>,
    ) -> Result<Box<dyn RecordBatchReader>, PointCloudError> {
        if self.data.is_empty() {
            let file = File::open(self.path())?;
            let reader = FileReader::try_new(file, projection)?;
            Ok(Box::new(reader))
        } else {
            let schema = project_schema(&self.schema, projection.as_ref())?;

            // projection
            let iter = self.data.clone().into_iter().map(move |b| {
                if let Some(indices) = &projection {
                    b.project(indices)
                } else {
                    Ok(b)
                }
            });

            // filter
            let mut offset = 0;
            let iter = iter.map(move |result| {
                if let Some(indices) = &indices {
                    let batch = result.unwrap();

                    let mut selection = vec![false; batch.num_rows()];
                    for i in indices {
                        if let Some(s) = selection.get_mut(i - offset) {
                            *s = true;
                        }
                    }
                    offset += batch.num_rows();

                    filter_record_batch(&batch, &selection.into())
                } else {
                    result
                }
            });

            Ok(Box::new(RecordBatchIterator::new(iter, schema)))
        }
    }

    pub fn spill(&mut self) -> Result<(), PointCloudError> {
        // read data from file if not in memory
        if self.data.is_empty() {
            let file = File::open(self.path())?;
            let reader = FileReader::try_new(file, None)?;
            self.data = reader.collect::<Result<Vec<RecordBatch>, ArrowError>>()?;
        }

        // create file
        let file = File::create(self.path())?;

        // create writer
        let mut writer = FileWriter::try_new(file, &self.schema)?;

        // splill
        for batch in self.data.drain(..) {
            writer.write(&batch)?;
        }

        writer.finish()?;

        Ok(())
    }

    pub fn index(&self) -> &RtreeIndex<PointXYZI<f64>> {
        &self.index
    }

    pub fn create_index(&mut self) -> Result<(), PointCloudError> {
        let objects = points(self.read(None, None)?)
            .enumerate()
            .map(|(i, p)| GeomWithData::new(p, i))
            .collect();

        let rtree = PointIndex::bulk_load_with_params(objects);

        self.index = RtreeIndex::Point(Box::new(rtree));

        Ok(())
    }

    pub fn drop_index(&mut self) {
        self.index = RtreeIndex::None;
    }
}
