use std::{any::Any, sync::Arc};

use datafusion::{
    arrow::datatypes::SchemaRef,
    common::project_schema,
    error::{DataFusionError, Result},
    execution::{SendableRecordBatchStream, TaskContext},
    physical_expr::EquivalenceProperties,
    physical_plan::{
        DisplayAs, DisplayFormatType, ExecutionPlan, Partitioning, PlanProperties,
        execution_plan::{Boundedness, EmissionType},
        stream::RecordBatchStreamAdapter,
    },
};
use rayon::iter::{IntoParallelIterator, IntoParallelRefIterator, ParallelIterator};
use rstar::{Envelope, RTreeObject};

use crate::{AABB, PointCloud, PointXYZI, index::rtree::RtreeIndex};

#[derive(Debug)]
pub struct PointCloudExec {
    pc: PointCloud,
    projection: Option<Vec<usize>>,
    projected_schema: SchemaRef,
    cache: PlanProperties,
    bounds: AABB<PointXYZI<f64>>,
}

impl PointCloudExec {
    pub fn new(
        projection: Option<&Vec<usize>>,
        pc: PointCloud,
        bounds: AABB<PointXYZI<f64>>,
    ) -> Self {
        let projected_schema = project_schema(&pc.schema(), projection).unwrap();
        let cache = Self::compute_properties(projected_schema.clone());
        Self {
            pc,
            projection: projection.cloned(),
            projected_schema,
            cache,
            bounds,
        }
    }

    /// This function creates the cache object that stores the plan properties such as schema, equivalence properties, ordering, partitioning, etc.
    fn compute_properties(schema: SchemaRef) -> PlanProperties {
        PlanProperties::new(
            EquivalenceProperties::new(schema),
            Partitioning::UnknownPartitioning(1),
            EmissionType::Incremental,
            Boundedness::Bounded,
        )
    }
}

impl DisplayAs for PointCloudExec {
    fn fmt_as(&self, _t: DisplayFormatType, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "PointCloudExec")
    }
}

impl ExecutionPlan for PointCloudExec {
    fn name(&self) -> &str {
        "PointCloudExec"
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn properties(&self) -> &PlanProperties {
        &self.cache
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        vec![]
    }

    fn with_new_children(
        self: Arc<Self>,
        _children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> datafusion::error::Result<Arc<dyn ExecutionPlan>> {
        Ok(self)
    }

    fn execute(
        &self,
        _partition: usize,
        _context: Arc<TaskContext>,
    ) -> Result<SendableRecordBatchStream> {
        let entries: Vec<_> = self
            .pc
            .store
            .par_iter()
            .filter_map(|entry| {
                if entry
                    .value()
                    .read()
                    .unwrap()
                    .bounds()
                    .intersects(&self.bounds)
                {
                    Some(entry.value().to_owned())
                } else {
                    None
                }
            })
            .collect();

        let projection = self.projection.clone();
        let aabb = self.bounds.envelope();

        let batches = entries
            .into_par_iter()
            .flat_map(move |entry| {
                let chunk = entry.read().unwrap();

                let indices = match chunk.index() {
                    RtreeIndex::Point(rtree) => Some(
                        rtree
                            .locate_in_envelope_intersecting(&aabb)
                            .map(|o| o.data)
                            .collect(),
                    ),
                    RtreeIndex::Batch(_rtree) => todo!(),
                    RtreeIndex::Multi(_rtree) => todo!(),
                    RtreeIndex::None => None,
                };

                chunk
                    .read(projection.clone(), indices)
                    .unwrap()
                    .collect::<Vec<_>>()
            })
            .map(|batch| batch.map_err(|e| DataFusionError::ArrowError(e, None)));

        let stream = RecordBatchStreamAdapter::new(
            self.projected_schema.clone(),
            futures::stream::iter(crate::helpers::into_iter(batches)),
        );

        Ok(Box::pin(stream))
    }
}
