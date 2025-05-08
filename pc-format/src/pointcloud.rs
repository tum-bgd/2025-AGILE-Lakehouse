use std::{
    path::PathBuf,
    sync::{Arc, RwLock},
};

use ahash::RandomState;
use dashmap::DashMap;
use datafusion::arrow::{
    array::{
        ArrayBuilder, ArrayRef, Float32Builder, Float64Builder, Int32Builder, Int64Builder,
        RecordBatchIterator, as_primitive_array,
    },
    compute::filter_record_batch,
    datatypes::{DataType, Float32Type, Float64Type, Int32Type, Int64Type, SchemaRef},
    record_batch::{RecordBatch, RecordBatchReader},
};
use moka::{notification::RemovalCause, sync::Cache};
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};
use rstar::Envelope;
use uuid::Uuid;

use crate::{
    AABB, Dims, PointChunk, PointCloudError, PointTrait,
    compute::{record_batch_aabb, record_batch_aabb_filter},
    framework::Grid,
    index::rtree::RtreeIndex,
    point::PointXYZI,
    schema::{dimensions, validate},
};

/// Point cloud
#[derive(Debug, Clone)]
pub struct PointCloud {
    schema: SchemaRef,
    dims: Dims,
    pub dir: PathBuf,
    framework: Grid<PointXYZI<f64>>,
    pub store: Arc<DashMap<String, Arc<RwLock<PointChunk>>, RandomState>>,
    cache: Cache<String, (), RandomState>,
    _index: RtreeIndex<PointXYZI<f64>>,
}

impl PointCloud {
    pub fn try_new(schema: SchemaRef) -> Result<Self, PointCloudError> {
        let dir = tempfile::tempdir().unwrap().into_path();

        PointCloud::try_new_with(schema, u64::MAX, dir)
    }
    pub fn try_new_with(
        schema: SchemaRef,
        capacity: u64,
        dir: PathBuf,
    ) -> Result<Self, PointCloudError> {
        validate(&schema)?;

        let dims: Vec<String> = dimensions(&schema)
            .into_iter()
            .map(|i| schema.field(i).name().to_owned())
            .collect();
        let dims = Dims::try_from(dims.as_slice())?;

        if !&dir.is_dir() {
            match std::fs::create_dir(dir.as_path()) {
                Ok(_) => (),
                Err(_) => {
                    return Err(PointCloudError::CacheError(format!(
                        "Failed to create store directory: {dir:?}"
                    )));
                }
            }
        }

        let store = Arc::new(DashMap::default());

        let eviction_store = store.clone();

        let cache = Cache::builder()
            .max_capacity(capacity)
            // .time_to_idle(Duration::from_secs(5))
            // .weigher(|_: &String, v: &Arc<RwLock<Vec<RecordBatch>>>| {
            //     v.read()
            //         .unwrap()
            //         .iter()
            //         .map(|batch| batch.num_rows())
            //         .sum::<usize>() as u32
            // })
            .eviction_listener(move |k: Arc<String>, _, cause| {
                if cause == RemovalCause::Replaced {
                    return;
                }

                eviction_store.alter(k.as_str(), |_, v: Arc<RwLock<PointChunk>>| {
                    v.write().unwrap().spill().unwrap();
                    v
                });
            })
            .build_with_hasher(RandomState::default());

        Ok(Self {
            schema,
            dims,
            dir,
            framework: Grid::new(),
            store,
            cache,
            _index: RtreeIndex::None,
        })
    }

    pub fn with_dims(mut self, dims: Dims) -> Self {
        self.dims = dims;
        self
    }

    pub fn dims(&self) -> Dims {
        self.dims
    }

    pub fn chunk(&self, key: &str) -> Arc<RwLock<PointChunk>> {
        match self.store.get(key) {
            Some(v) => v.value().clone(),
            None => panic!("no such key in store: {key}"),
        }
    }

    pub fn push(&self, id: String, batch: RecordBatch) -> Result<(), PointCloudError> {
        let batch = batch.with_schema(self.schema.clone())?;

        self.store
            .entry(id.clone())
            .or_insert_with(|| {
                Arc::new(RwLock::new(PointChunk::new(
                    id,
                    self.dir.to_owned(),
                    self.schema.clone(),
                )))
            })
            .value()
            .write()
            .unwrap()
            .push(batch);

        Ok(())
    }

    pub fn append(&mut self, batch: RecordBatch) -> Result<(), PointCloudError> {
        let batch = batch.with_schema(self.schema.clone())?;

        if self.framework.delta.is_some() {
            let aabb: AABB<PointXYZI<f64>> = record_batch_aabb(&batch)?;

            // create missing cells
            let cells = self.framework.create_cells(&aabb);

            for cell in cells {
                // get points for cell
                let filter = record_batch_aabb_filter(&batch, &cell);
                let partition = filter_record_batch(&batch, &filter)?;

                // insert records
                if partition.num_rows() != 0 {
                    let id = cell.id();
                    self.push(id, partition)?;
                }
            }
        } else {
            self.push(Uuid::new_v4().to_string(), batch)?;
        };

        Ok(())
    }

    pub fn flush(&self) {
        self.cache.invalidate_all();
        self.cache.run_pending_tasks();
    }
}

impl PointCloud {
    pub fn schema(&self) -> SchemaRef {
        self.schema.clone()
    }

    pub fn num_points(&self) -> usize {
        self.store
            .iter()
            .map(|entry| {
                entry
                    .value()
                    .read()
                    .unwrap()
                    .read(None, None)
                    .unwrap()
                    .map(|b| b.unwrap().num_rows())
                    .sum::<usize>()
            })
            .sum()
    }

    pub fn points<'a, P>(&'a self) -> Box<dyn Iterator<Item = P> + 'a>
    where
        P: PointTrait + 'a,
        <P as rstar::Point>::Scalar: num_traits::NumCast,
    {
        Box::new(points(RecordBatchIterator::new(
            self.store
                .iter()
                .flat_map(|entry| entry.value().read().unwrap().read(None, None).unwrap()),
            self.schema(),
        )))
    }

    pub fn aabb<P>(&self) -> AABB<P>
    where
        P: PointTrait,
        <P as rstar::Point>::Scalar: num_traits::NumCast,
    {
        self.store
            .par_iter()
            .map(|entry| {
                entry
                    .read()
                    .unwrap()
                    .read(None, None)
                    .unwrap()
                    .map(|batch| record_batch_aabb(&batch.unwrap()).unwrap())
                    .fold(AABB::new_empty(), |acc, e| acc.merged(&e))
            })
            .reduce(AABB::new_empty, |a, b| a.merged(&b))
    }
}

impl<P> FromIterator<P> for PointCloud
where
    P: PointTrait,
    <P as rstar::Point>::Scalar: num_traits::NumCast,
{
    fn from_iter<T: IntoIterator<Item = P>>(iter: T) -> Self {
        let schema = P::schema();

        let mut data_builders: Vec<Box<dyn ArrayBuilder>> = schema
            .fields()
            .iter()
            .map(|f| match f.data_type() {
                DataType::Int32 => Box::new(Int32Builder::new()) as Box<dyn ArrayBuilder>,
                DataType::Int64 => Box::new(Int64Builder::new()),
                DataType::Float32 => Box::new(Float32Builder::new()),
                DataType::Float64 => Box::new(Float64Builder::new()),
                x => unimplemented!("{x}"),
            })
            .collect();

        for p in iter.into_iter() {
            for (i, f) in schema.fields().iter().enumerate() {
                match f.data_type() {
                    DataType::Int32 => data_builders[i]
                        .as_any_mut()
                        .downcast_mut::<Int32Builder>()
                        .unwrap()
                        .append_value(num_traits::cast(p.nth(i)).unwrap()),
                    DataType::Int64 => data_builders[i]
                        .as_any_mut()
                        .downcast_mut::<Int64Builder>()
                        .unwrap()
                        .append_value(num_traits::cast(p.nth(i)).unwrap()),
                    DataType::Float32 => data_builders[i]
                        .as_any_mut()
                        .downcast_mut::<Float32Builder>()
                        .unwrap()
                        .append_value(num_traits::cast(p.nth(i)).unwrap()),
                    DataType::Float64 => data_builders[i]
                        .as_any_mut()
                        .downcast_mut::<Float64Builder>()
                        .unwrap()
                        .append_value(num_traits::cast(p.nth(i)).unwrap()),
                    _ => unimplemented!(),
                }
            }
        }

        let array_refs: Vec<ArrayRef> = data_builders
            .iter_mut()
            .map(|builder| builder.finish())
            .collect();
        let batch = RecordBatch::try_new(schema.clone(), array_refs).unwrap();

        let mut pc = Self::try_new(schema).unwrap();

        pc.append(batch).unwrap();

        pc
    }
}

impl<RBR> From<RBR> for PointCloud
where
    RBR: RecordBatchReader,
{
    fn from(value: RBR) -> Self {
        let schema = value.schema();

        let mut pc = Self::try_new(schema).unwrap();

        for batch in value {
            pc.append(batch.unwrap()).unwrap();
        }

        pc
    }
}

pub fn points<'a, P, R>(reader: R) -> impl Iterator<Item = P>
where
    R: RecordBatchReader + 'a,
    P: PointTrait + 'a,
    <P as rstar::Point>::Scalar: num_traits::NumCast,
{
    let dimensions = dimensions(&reader.schema());
    let d = dimensions.len().min(P::DIMENSIONS);

    reader.flat_map(move |result| {
        let batch = result.unwrap();
        match P::DATA_TYPE {
            DataType::Int32 => {
                let columns: Vec<_> = (0..d)
                    .map(|j| {
                        let column = batch.column(dimensions[j]);
                        let column =
                            datafusion::arrow::compute::cast(&column, &P::DATA_TYPE).unwrap();
                        as_primitive_array::<Int32Type>(&column).to_owned()
                    })
                    .collect();

                (0..batch.num_rows())
                    .map(|i| {
                        P::generate(|nth| {
                            columns
                                .get(nth)
                                .and_then(|column| num_traits::cast(column.value(i)))
                                .unwrap_or_else(num_traits::zero)
                        })
                    })
                    .collect::<Vec<P>>()
            }
            DataType::Int64 => {
                let columns: Vec<_> = (0..d)
                    .map(|j| {
                        let column = batch.column(dimensions[j]);
                        let column =
                            datafusion::arrow::compute::cast(&column, &P::DATA_TYPE).unwrap();
                        as_primitive_array::<Int64Type>(&column).to_owned()
                    })
                    .collect();

                (0..batch.num_rows())
                    .map(|i| {
                        P::generate(|nth| {
                            columns
                                .get(nth)
                                .and_then(|column| num_traits::cast(column.value(i)))
                                .unwrap_or_else(num_traits::zero)
                        })
                    })
                    .collect()
            }
            DataType::Float32 => {
                let columns: Vec<_> = (0..d)
                    .map(|j| {
                        let column = batch.column(dimensions[j]);
                        let column =
                            datafusion::arrow::compute::cast(&column, &P::DATA_TYPE).unwrap();
                        as_primitive_array::<Float32Type>(&column).to_owned()
                    })
                    .collect();

                (0..batch.num_rows())
                    .map(|i| {
                        P::generate(|nth| {
                            columns
                                .get(nth)
                                .and_then(|column| num_traits::cast(column.value(i)))
                                .unwrap_or_else(num_traits::zero)
                        })
                    })
                    .collect()
            }
            DataType::Float64 => {
                let columns: Vec<_> = (0..d)
                    .map(|j| {
                        let column = batch.column(dimensions[j]);
                        let column =
                            datafusion::arrow::compute::cast(&column, &P::DATA_TYPE).unwrap();
                        as_primitive_array::<Float64Type>(&column).to_owned()
                    })
                    .collect();

                (0..batch.num_rows())
                    .map(|i| {
                        P::generate(|nth| {
                            columns
                                .get(nth)
                                .and_then(|column| num_traits::cast(column.value(i)))
                                .unwrap_or_else(num_traits::zero)
                        })
                    })
                    .collect()
            }
            _ => unimplemented!(),
        }
    })
}
