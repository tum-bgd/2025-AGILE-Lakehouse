use std::{
    future::Future,
    path::Path,
    pin::Pin,
    sync::{Arc, RwLock},
};

use clap::Args;
use dashmap::DashMap;
use datafusion::{
    arrow::{
        array::RecordBatch,
        compute::{concat_batches, filter_record_batch},
        datatypes::DataType,
        error::ArrowError,
    },
    error::DataFusionError,
    execution::RecordBatchStream,
    logical_expr::Cast,
    physical_plan::stream::RecordBatchStreamAdapter,
    prelude::*,
};
use futures::StreamExt;
use rayon::iter::{
    IntoParallelIterator, IntoParallelRefIterator, ParallelBridge, ParallelExtend, ParallelIterator,
};
use rstar::{Envelope, Point as _, RTree, RTreeObject};

use pc_format::{
    AABB, PointCloudError, PointTrait, PointXY, PointXYI,
    compute::{record_batch_aabb, record_batch_aabb_filter},
    expressions::{df_aabb, filter_df_by_aabb},
    framework::{grid_coverage, quadtree_cells, split_aabb},
};
use pc_io::{
    config::{DEFAULT_BATCH_SIZE, default_session_config},
    helpers::{deglob, sink},
    las::{LasDataSource, LasDataSourceOptions},
};

#[derive(Args, Debug)]
pub struct Partitioning {
    /// Input files
    #[arg(short, long)]
    pub input: Vec<String>,
    /// Output folder or file path
    #[arg(short, long)]
    pub output: Option<String>,
    /// Partitioning method (grid(i) or quadtree)
    #[arg(long)]
    pub method: String,
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

impl Partitioning {
    pub fn new<T: AsRef<str>>(input: &[T], method: T) -> Self {
        let input = input
            .iter()
            .flat_map(|s| glob::glob(s.as_ref()).expect("read glob pattern"))
            .map(|p| p.unwrap().to_str().unwrap().to_string())
            .collect();

        Partitioning {
            input,
            output: None,
            method: method.as_ref().to_owned(),
            compression: None,
            statistics: None,
            merge: false,
        }
    }
    pub async fn partition(&self) -> Result<(), PointCloudError> {
        // session context
        let mut config = default_session_config();
        if let Some(statistics) = self.statistics.clone() {
            config.options_mut().execution.parquet.statistics_enabled = Some(statistics);
        }

        if let Some(compression) = self.compression.clone() {
            config.options_mut().execution.parquet.compression = Some(compression);
        }
        let ctx = SessionContext::new_with_config(config);

        // table paths
        let table_paths = if self.merge {
            vec![deglob(&self.input).collect()]
        } else {
            self.input
                .iter()
                .map(|i| deglob([i]).collect::<Vec<_>>())
                .collect()
        };

        for (i, paths) in table_paths.iter().enumerate() {
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
                let name = format!("{}_{}.parquet", stem.to_str().unwrap(), self.method);
                path.with_file_name(name)
            };

            if !dst.exists() {
                if dst.extension().is_some() {
                    let _ = std::fs::create_dir(dst.parent().unwrap());
                } else {
                    let _ = std::fs::create_dir(&dst);
                }
            }

            // source
            let table = i.to_string();
            if paths[0].ends_with("parquet") {
                ctx.register_parquet(&table, &paths[0], ParquetReadOptions::default())
                    .await
                    .unwrap();
            } else {
                let options = LasDataSourceOptions {
                    raw: false,
                    stats: true,
                };
                let ds = LasDataSource::try_new_with(paths, options).unwrap();

                ctx.register_table(&table, Arc::new(ds.clone())).unwrap();
            }

            let df = ctx.table(&table).await?;

            // importance
            let df = if df.schema().fields().find("i").is_some() {
                df
            } else {
                df.with_column(
                    "i",
                    Expr::Cast(Cast::new(Box::new(random()), DataType::Float32)),
                )
                .unwrap()
            };

            // partition
            let (cells, windows) = if self.method.to_lowercase().starts_with("grid") {
                let splits: f64 = self
                    .method
                    .split("(")
                    .nth(1)
                    .unwrap()
                    .trim_end_matches(")")
                    .parse()
                    .unwrap();

                // grid size
                let count = df.clone().count().await.unwrap();

                let bounds: AABB<PointXY<f64>> = df_aabb(&df).await.unwrap();

                let num_batches = count / DEFAULT_BATCH_SIZE;
                let size = (bounds.envelope().area() / (num_batches as f64 / splits))
                    .sqrt()
                    .round();
                println!("Grid size: {size}");

                // framework
                let acc = 8. / splits.sqrt(); // number of cells to accumulate on each dimension

                let delta = PointTrait::from_slice(&[acc * size, acc * size]);
                let windows: Vec<_> = grid_coverage(&bounds, delta).collect();

                let splits: PointXYI<f64> = PointTrait::from_slice(&[acc, acc, splits]);
                let cells = windows
                    .iter()
                    .flat_map(|window| split_aabb(&window.with_importance(0., 1.), &splits))
                    .collect();

                // partition
                (cells, windows)
            } else if self.method.to_lowercase().starts_with("quadtree") {
                // octree
                let count = df.clone().count().await.unwrap();
                let bounds: AABB<PointXY<f64>> = df_aabb(&df).await.unwrap();

                let delta = bounds.upper().sub(&bounds.lower());
                let size = delta.coords().max_by(|a, b| a.total_cmp(b)).unwrap() + 0.1;
                println!("Size: {size:.2}");

                let count_normalized =
                    (count as f64 * (size.powi(2) / bounds.envelope().area())) as usize;
                let num_batches = count_normalized / DEFAULT_BATCH_SIZE;
                println!("Batches: {num_batches}");
                let depth = (num_batches as f64).log(4.).round() as usize;
                println!("Depth: {depth}");

                // cells
                let center = bounds.envelope().center();
                let xmid = center.nth(0);
                let ymid = center.nth(1);
                let half = size / 2.;
                let aabb: AABB<PointXYI<f64>> = AABB::from_corners(
                    PointTrait::from_slice(&[xmid - half, ymid - half, 0.]),
                    PointTrait::from_slice(&[xmid + half, ymid + half, 1.]),
                );

                let cells = quadtree_cells(&aabb, depth).collect();

                // windows
                let split = 2_f64.powi(5); // number of leaves to cover on xy
                let leaf_size = 2_usize.pow(depth as u32) as f64;
                let delta = PointTrait::from_slice(&[split * leaf_size, split * leaf_size]);
                let windows = grid_coverage(&bounds, delta).collect();

                // partition
                (cells, windows)
            } else {
                panic!("Unhandled partitioning `{}`", self.method)
            };

            // partition
            let data = partition(&df, windows, cells).await;

            // sink
            sink(data, dst.to_str().unwrap(), &ctx).await;
        }
        Ok(())
    }
}

async fn partition(
    df: &DataFrame,
    windows: Vec<AABB<PointXY<f64>>>,
    cells: Vec<AABB<PointXYI<f64>>>,
) -> impl RecordBatchStream + use<> {
    const WINDOWS_BUFFER: usize = 1;

    let schema = Arc::new(df.schema().as_arrow().to_owned());

    let window_tree = Arc::new(RwLock::new(RTree::bulk_load(
        windows.iter().map(|w| w.with_importance(0., 1.)).collect(),
    )));
    let windows = RTree::bulk_load(windows);
    let cells = RTree::bulk_load(cells);

    let bounds: AABB<PointXY<f64>> = df_aabb(df).await.unwrap();

    // cache
    let global_cache: Arc<DashMap<String, Vec<RecordBatch>>> = Arc::new(DashMap::new());

    let futures: Vec<Pin<Box<dyn Future<Output = Vec<Result<RecordBatch, ArrowError>>> + Send>>> =
        windows
            .into_iter()
            .chain([bounds])
            .map(|window| {
                let schema = schema.clone();

                let global_cache = global_cache.clone();
                let window_tree = window_tree.clone();

                if window != bounds {
                    let df = df.clone();

                    let cells = RTree::bulk_load(
                        cells
                            .locate_in_envelope_intersecting(
                                &window.with_importance(0., 1.).envelope(),
                            )
                            .cloned()
                            .collect(),
                    );

                    let f = async move {
                        let partition = filter_df_by_aabb(df, &window, true).await.unwrap();

                        let results = partition.collect().await.unwrap();

                        // split
                        let local_cache: DashMap<String, Vec<RecordBatch>> = DashMap::new();

                        let window = window.with_importance(0., 1.);

                        results.into_par_iter().for_each(|batch| {
                            let aabb = record_batch_aabb(&batch).unwrap();

                            for cell in cells.locate_in_envelope_intersecting(&aabb.envelope()) {
                                // get points for cell
                                let filter = record_batch_aabb_filter(&batch, cell);
                                let filtered = filter_record_batch(&batch, &filter).unwrap();

                                // insert records
                                if filtered.num_rows() > 0 {
                                    if window.envelope().contains_envelope(&cell.envelope()) {
                                        local_cache.entry(cell.id()).or_default().push(filtered);
                                    } else {
                                        global_cache.entry(cell.id()).or_default().push(filtered);
                                    }
                                }
                            }
                        });

                        // evict completed
                        let mut evicted: Vec<_> = local_cache
                            .into_par_iter()
                            .map(|(_, record_batches)| concat_batches(&schema, &record_batches))
                            .collect();

                        window_tree.write().unwrap().remove(&window);

                        let distinct = cells
                            .into_iter()
                            .par_bridge()
                            .filter(move |cell| {
                                let f = window_tree
                                    .read()
                                    .unwrap()
                                    .locate_in_envelope_intersecting(&cell.envelope())
                                    .next()
                                    .is_none();
                                f
                            })
                            .filter_map(move |cell| {
                                global_cache.remove(&cell.id()).map(|(_, record_batches)| {
                                    concat_batches(&schema, &record_batches)
                                })
                            });

                        evicted.par_extend(distinct);

                        evicted
                    };
                    Box::pin(f) as Pin<Box<_>>
                } else {
                    Box::pin(async move {
                        global_cache
                            .par_iter()
                            .map(|item| concat_batches(&schema, item.value()))
                            .collect::<Vec<_>>()
                    }) as Pin<Box<_>>
                }
            })
            .collect();

    // stream
    let stream = futures::stream::iter(futures)
        .buffered(WINDOWS_BUFFER)
        .flat_map(|results: Vec<_>| {
            futures::stream::iter(
                results
                    .into_iter()
                    .map(|result| result.map_err(|e| DataFusionError::ArrowError(e, None))),
            )
        });

    RecordBatchStreamAdapter::new(schema, stream)
}
