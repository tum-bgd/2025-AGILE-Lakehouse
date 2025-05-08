use std::path::Path;

use clap::Args;
use datafusion::{
    arrow::datatypes::DataType, error::DataFusionError,
    physical_plan::stream::RecordBatchStreamAdapter, prelude::SessionContext,
};
use rayon::iter::ParallelIterator;

use pc_format::{PointCloudError, compute, schema};
use pc_io::las::LasDataSource;

#[derive(Args, Debug)]
pub struct Conversion {
    /// Input LAS/LAZ files
    #[arg(short, long)]
    pub input: Vec<String>,
    /// Output folder or file path
    #[arg(short, long)]
    pub output: Option<String>,
    /// Use raw las point format (grid rounded coordinates)
    #[arg(long, default_value = "false")]
    pub raw: bool,
    /// Create random importance
    #[arg(long, default_value = "false")]
    pub importance: bool,
    /// Compression: uncompressed, snappy, gzip(level), lzo, brotli(level), lz4, zstd(level), or lz4_raw
    #[arg(long)]
    pub compression: Option<String>,
    /// Statistics: none, chunk or page
    #[arg(long)]
    pub statistics: Option<String>,
    /// Merge input files and partitions
    #[arg(long, default_value = "false")]
    pub merge: bool,
}

impl Conversion {
    pub fn new<T: AsRef<str>>(input: &[T]) -> Self {
        let input = input.iter().map(|s| s.as_ref().to_owned()).collect();

        Conversion {
            input,
            output: None,
            raw: false,
            importance: true,
            compression: None,
            statistics: None,
            merge: false,
        }
    }

    pub async fn convert(&self) -> Result<(), PointCloudError> {
        // session context
        let mut config = pc_io::config::default_session_config();
        if let Some(statistics) = self.statistics.clone() {
            config.options_mut().execution.parquet.statistics_enabled = Some(statistics);
        }
        if let Some(compression) = self.compression.clone() {
            config.options_mut().execution.parquet.compression = Some(compression);
        }
        let ctx = SessionContext::new_with_config(config);

        // table paths
        let ds = LasDataSource::try_new(&self.input)?;
        let table_paths = if self.merge {
            vec![ds.table_paths().to_owned()]
        } else {
            ds.table_paths()
                .iter()
                .map(|p| vec![p.to_owned()])
                .collect()
        };

        for paths in &table_paths {
            // source
            let ds = LasDataSource::try_new(paths).unwrap();

            // destination
            let dst = if let Some(output) = self.output.as_ref() {
                let output_path = Path::new(output);

                if output_path.extension().is_some() {
                    if table_paths.len() == 1 {
                        output_path.to_owned()
                    } else {
                        let name = Path::new(&paths[0]).file_name().unwrap();
                        output_path.with_file_name(name).with_extension("parquet")
                    }
                } else {
                    // if path.is_dir()
                    let name = Path::new(&paths[0]).file_name().unwrap();
                    output_path.join(name).with_extension("parquet")
                }
            } else {
                let path = Path::new(&paths[0]);
                let stem = path.file_stem().unwrap();
                let name = format!("{}_convert.parquet", stem.to_str().unwrap());
                path.with_file_name(name)
            };

            if !dst.exists() {
                if dst.extension().is_some() {
                    let _ = std::fs::create_dir(dst.parent().unwrap());
                } else {
                    let _ = std::fs::create_dir(&dst);
                }
            }

            // schema
            let schema = if self.importance {
                schema::add_importance(
                    ds.schema(),
                    "i",
                    DataType::Float32,
                    ds.schema().fields().len(),
                )
            } else {
                ds.schema()
            };

            // stream
            let stream_schema = schema.clone();
            let importance = self.importance;

            let batches = ds.par_record_batch_iter().map(move |record_batch| {
                if importance {
                    compute::add_importance(record_batch, &stream_schema)
                        .map_err(|e| DataFusionError::ArrowError(e, None))
                } else {
                    Ok(record_batch)
                }
            });
            let stream = futures::stream::iter(pc_format::helpers::into_iter(batches));

            let data = RecordBatchStreamAdapter::new(schema, stream);

            // sink
            pc_io::helpers::sink(data, dst.to_str().unwrap(), &ctx).await;
        }

        Ok(())
    }
}
