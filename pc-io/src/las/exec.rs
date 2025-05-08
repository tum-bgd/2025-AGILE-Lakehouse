use std::{any::Any, collections::HashMap, fmt, sync::Arc};

use datafusion::{
    arrow::{array::RecordBatch, datatypes::SchemaRef},
    common::{Result, Statistics},
    datasource::TableProvider,
    error::DataFusionError,
    execution::TaskContext,
    physical_expr::EquivalenceProperties,
    physical_plan::{
        DisplayAs, DisplayFormatType, ExecutionPlan, Partitioning, PlanProperties,
        SendableRecordBatchStream,
        execution_plan::{Boundedness, EmissionType},
        project_schema,
        stream::RecordBatchStreamAdapter,
    },
};
use las::Reader;
use rayon::iter::{IndexedParallelIterator, IntoParallelIterator, ParallelIterator};

use super::{LasDataSource, builder::RowBuilder, chunk_table};

#[derive(Debug)]
pub(super) struct LasExec {
    source: LasDataSource,
    projection: Option<Vec<usize>>,
    projected_schema: SchemaRef,
    projected_statistics: Statistics,
    files_filter: Vec<bool>,
    chunks_filters: HashMap<String, Vec<bool>>,
    properties: PlanProperties,
}

impl LasExec {
    pub(super) fn new(
        source: LasDataSource,
        projection: Option<Vec<usize>>,
        files_filter: Vec<bool>,
        chunks_filters: HashMap<String, Vec<bool>>,
    ) -> Self {
        let projected_schema = project_schema(&source.schema(), projection.as_ref()).unwrap();

        let projected_statistics = if let Some(mut statistics) = source.statistics() {
            if let Some(projection) = &projection {
                statistics.column_statistics = projection
                    .iter()
                    .map(|i| statistics.column_statistics[*i].to_owned())
                    .collect();
            }
            statistics
        } else {
            Statistics::new_unknown(&projected_schema)
        };

        let properties = PlanProperties::new(
            EquivalenceProperties::new(projected_schema.clone()),
            Partitioning::UnknownPartitioning(1),
            EmissionType::Incremental,
            Boundedness::Bounded,
        );

        Self {
            source,
            projection,
            projected_schema,
            projected_statistics,
            files_filter,
            chunks_filters,
            properties,
        }
    }
}

impl DisplayAs for LasExec {
    fn fmt_as(&self, _t: DisplayFormatType, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "LasExec")
    }
}

impl ExecutionPlan for LasExec {
    fn name(&self) -> &str {
        "LasExec"
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn schema(&self) -> SchemaRef {
        self.projected_schema.clone()
    }

    fn properties(&self) -> &PlanProperties {
        &self.properties
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        vec![]
    }

    fn with_new_children(
        self: Arc<Self>,
        _: Vec<Arc<dyn ExecutionPlan>>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        Ok(self)
    }

    fn execute(
        &self,
        _partition: usize,
        _context: Arc<TaskContext>,
    ) -> Result<SendableRecordBatchStream> {
        let schema = self.source.schema().clone();
        let chunks_filters = self.chunks_filters.clone();
        let raw = self.source.table_options.raw;
        let projection = self.projection.clone();

        let table_paths = self
            .source
            .table_paths
            .clone()
            .into_par_iter()
            .zip(self.files_filter.clone())
            .filter_map(|(path, f)| if f { Some(path.to_owned()) } else { None });

        let batches = table_paths.flat_map(move |path| {
            let schema = schema.clone();
            let projection = projection.clone();

            let chunk_table = chunk_table(&path);

            let chunks_filter = chunks_filters
                .get(&path)
                .cloned()
                .unwrap_or_else(|| vec![true; chunk_table.len()]);

            chunk_table.into_par_iter().zip(chunks_filter).filter_map(
                move |((offset, point_count), f)| {
                    if f {
                        let mut reader = Reader::from_path(&path).expect("files should exist");
                        reader.seek(offset).expect("seek to offset");

                        let mut builder = RowBuilder::new(point_count as usize, raw);
                        for point in reader.read_points(point_count).expect("read n points") {
                            builder.append(point, reader.header());
                        }

                        let batch = RecordBatch::from(builder.finish(&schema, reader.header()));

                        Some(match &projection {
                            Some(indices) => batch
                                .project(indices)
                                .map_err(|e| DataFusionError::ArrowError(e, None)),
                            None => Ok(batch),
                        })
                    } else {
                        None
                    }
                },
            )
        });

        let stream = RecordBatchStreamAdapter::new(
            self.schema(),
            futures::stream::iter(pc_format::helpers::into_iter(batches)),
        );

        Ok(Box::pin(stream))
    }

    fn statistics(&self) -> Result<Statistics> {
        Ok(self.projected_statistics.clone())
    }
}
