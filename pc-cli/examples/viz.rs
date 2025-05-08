use std::{fs::File, io::Write, time::Instant};

use datafusion::{
    arrow::{array::RecordBatch, compute::filter_record_batch},
    parquet::arrow::{
        ArrowSchemaConverter, ProjectionMask, arrow_reader::ParquetRecordBatchReaderBuilder,
    },
    prelude::{ParquetReadOptions, SessionContext},
};
use rand::seq::SliceRandom;
use rayon::iter::{
    IntoParallelIterator, IntoParallelRefIterator, ParallelBridge, ParallelIterator,
};
use rstar::{Envelope, Point as _, RTree, RTreeObject, primitives::GeomWithData};

use pc_format::{
    AABB, Dims, PointTrait, PointXY, PointXYI,
    compute::record_batch_aabb_filter,
    expressions::{df_aabb, filter_df_by_aabb},
    framework::quadtree_cells,
};
use pc_io::{config::DEFAULT_BATCH_SIZE, helpers::bounds_from_parquet_files};

// const NAME: &str = "200M";
// const PATH_PARQUET: &str = "../data/AHN4/C_69AZ1_convert.parquet";
// const PATH_GRID: &str = "../data/AHN4/C_69AZ1_grid(1).parquet";
// const PATH_GRID_8: &str = "../data/AHN4/C_69AZ1_grid(8).parquet";
// const PATH_QUADTREE: &str = "../data/AHN4/C_69AZ1_quadtree.parquet";

const NAME: &str = "2B";
const PATH_PARQUET: &str = "./data/AHN3/C_37E*_convert.parquet";
const PATH_GRID: &str = "./data/AHN3/C_37E*_grid(1).parquet";
const PATH_GRID_8: &str = "./data/AHN3/C_37E*_grid(8).parquet";
const PATH_QUADTREE: &str = "./data/AHN3/C_37E*_quadtree.parquet";

#[tokio::main]
async fn main() {
    // session context
    let config = pc_io::config::default_session_config();
    let ctx = SessionContext::new_with_config(config);

    // data source
    ctx.register_parquet("test", PATH_QUADTREE, ParquetReadOptions::default())
        .await
        .unwrap();
    let df = ctx.table("test").await.unwrap();

    // attribute selection
    let _mask = ProjectionMask::all();
    let mask = ProjectionMask::leaves(
        &ArrowSchemaConverter::new()
            .convert(df.schema().as_arrow())
            .unwrap(),
        vec![0, 1, 2, 18],
    );

    let df = df.select_columns(&["x", "y", "z", "i"]).unwrap();

    // octree
    let count = df.clone().count().await.unwrap();
    let bounds: AABB<PointXY<f64>> = df_aabb(&df).await.unwrap();

    let delta = bounds.upper().sub(&bounds.lower());
    let size = delta.coords().max_by(|a, b| a.total_cmp(b)).unwrap() + 0.1;

    let count_normalized = (count as f64 * (size.powi(2) / bounds.envelope().area())) as usize;
    let num_batches = count_normalized / DEFAULT_BATCH_SIZE;
    println!("Batches: {num_batches}");
    let depth = (num_batches as f64).log(4.).round() as usize;
    println!("Depth: {depth}");

    let center = bounds.envelope().center();
    let xmid = center.nth(0);
    let ymid = center.nth(1);
    let half = size / 2.;
    let aabb: AABB<PointXYI<f64>> = AABB::from_corners(
        PointTrait::from_slice(&[xmid - half, ymid - half, 0.]),
        PointTrait::from_slice(&[xmid + half, ymid + half, 1.]),
    );
    println!("{aabb:#?}");

    let cells: Vec<_> = quadtree_cells(&aabb, depth).collect();
    println!("Quadtree with {} cells", cells.len());

    let mut levels_importance: Vec<f64> = cells
        .iter()
        .map(|cell| cell.envelope().center().nth(2))
        .collect();
    levels_importance.dedup();
    levels_importance.sort_by(|a, b| a.partial_cmp(b).unwrap());
    // dbg!(&levels_importance);

    // log
    let mut log = File::create(format!("./logs/rg_{NAME}.csv")).unwrap();
    log.write_all(b"file,level,mean,std\n").unwrap();

    for (id, path) in [
        ("convert", PATH_PARQUET),
        ("grid(1)", PATH_GRID),
        ("grid(8)", PATH_GRID_8),
        ("quadtree", PATH_QUADTREE),
    ] {
        println!("PATH: {path}");

        // row group tree
        let rtree = RTree::bulk_load(
            bounds_from_parquet_files(&[path], Dims::XYI)
                .into_iter()
                .flat_map(|(k, v)| {
                    v.into_iter()
                        .enumerate()
                        .map(move |(i, b)| GeomWithData::new(b, (k.to_owned(), i)))
                })
                .collect(),
        );

        println!("Create tree from {} row groups!", rtree.size());

        // STATS: row group / cell intersection
        let data: Vec<i64> = cells
            .par_iter()
            .map(|cell| {
                rtree
                    .locate_in_envelope_intersecting(&cell.envelope())
                    .count() as i64
            })
            .collect();

        println!(
            "Mean row groups intersections per cell: {:>5.1} ({:>5.1})",
            mean(&data).unwrap(),
            std_deviation(&data).unwrap()
        );

        for (i, level) in levels_importance.clone().iter().enumerate() {
            let level_data: Vec<i64> = cells
                .iter()
                .enumerate()
                .filter_map(|(i, cell)| {
                    if cell.envelope().center().nth(2) == *level {
                        Some(data[i])
                    } else {
                        None
                    }
                })
                .collect();

            let mean = mean(&level_data).unwrap();
            let stdev = std_deviation(&level_data).unwrap();

            println!("\t Level {i}: {:>5.1} ({:>5.1})", mean, stdev);

            let s = format!("{id},{i},{mean},{stdev}\n");
            log.write_all(s.as_bytes()).unwrap();
        }

        // STATS: viz query performance
        if [PATH_QUADTREE].contains(&path) {
            const M: usize = 16; // probes per level

            let mut selected: Vec<_> = cells
                .iter()
                .enumerate()
                .filter_map(|(i, cell)| {
                    let level = levels_importance
                        .binary_search_by(|a| {
                            a.partial_cmp(&cell.envelope().center().nth(2)).unwrap()
                        })
                        .unwrap();

                    let n = 4_usize.pow(level as u32);

                    if i % (n / M).max(1) == 0 {
                        Some([(level, cell)].repeat((M / n).max(1)))
                    } else {
                        None
                    }
                })
                .flatten()
                .collect();

            selected.shuffle(&mut rand::rng());
            println!("Running {} viz queries...", selected.len());

            let mut times: Vec<i64> = Vec::with_capacity(selected.len());
            let mut counts: Vec<i64> = Vec::with_capacity(selected.len());
            let mut times_indexed: Vec<i64> = Vec::with_capacity(selected.len());
            let mut counts_indexed: Vec<i64> = Vec::with_capacity(selected.len());

            for (_, cell) in selected.iter() {
                // datafusion
                let now = Instant::now();
                let results = filter_df_by_aabb(df.clone(), cell, true)
                    .await
                    .unwrap()
                    .collect()
                    .await
                    .unwrap();

                times.push(now.elapsed().as_millis() as i64);
                counts.push(results.iter().map(|b| b.num_rows() as i64).sum());

                // indexed
                let now = Instant::now();
                let results = query_rtree(&rtree, cell, mask.clone());

                times_indexed.push(now.elapsed().as_millis() as i64);
                counts_indexed.push(results.iter().map(|b| b.num_rows() as i64).sum())
            }

            // log
            println!(
                "Mean query time per cell [datafusion]: {:>5.0}ms ({:>5.0}ms) | {:>7.0} ({:>7.0})",
                mean(&times).unwrap(),
                std_deviation(&times).unwrap(),
                mean(&counts).unwrap(),
                std_deviation(&counts).unwrap()
            );

            let mut log = File::create(format!("./logs/viz_{NAME}.csv")).unwrap();
            log.write_all(b"level,count,time,time_indexed\n").unwrap();

            for i in 0..levels_importance.len() {
                let mut level_time: Vec<i64> = Vec::new();
                let mut level_count: Vec<i64> = Vec::new();
                for (ii, (l, _)) in selected.iter().enumerate() {
                    if i == *l {
                        level_time.push(times[ii]);
                        level_count.push(counts[ii]);

                        let s = format!("{l},{},{},{}\n", counts[ii], times[ii], times_indexed[ii]);
                        log.write_all(s.as_bytes()).unwrap();
                    }
                }

                println!(
                    "\t Level {i}: {:>5.0}ms ({:>5.0}ms) | {:>7.0} ({:>7.0})",
                    mean(&level_time).unwrap(),
                    std_deviation(&level_time).unwrap(),
                    mean(&level_count).unwrap(),
                    std_deviation(&level_count).unwrap()
                );
            }

            println!(
                "Mean query time per cell [indexed]:    {:>5.0}ms ({:>5.0}ms) | {:>7.0} ({:>7.0})",
                mean(&times_indexed).unwrap(),
                std_deviation(&times_indexed).unwrap(),
                mean(&counts_indexed).unwrap(),
                std_deviation(&counts_indexed).unwrap()
            );

            for i in 0..levels_importance.len() {
                let mut level_time: Vec<i64> = Vec::new();
                let mut level_count: Vec<i64> = Vec::new();
                for (ii, (l, _)) in selected.iter().enumerate() {
                    if i == *l {
                        level_time.push(times_indexed[ii]);
                        level_count.push(counts_indexed[ii]);
                    }
                }

                println!(
                    "\t Level {i}: {:>5.0}ms ({:>5.0}ms) | {:>7.0} ({:>7.0})",
                    mean(&level_time).unwrap(),
                    std_deviation(&level_time).unwrap(),
                    mean(&level_count).unwrap(),
                    std_deviation(&level_count).unwrap()
                );
            }
        }
    }
}

fn query_rtree<P: PointTrait>(
    rtree: &RTree<GeomWithData<AABB<P>, (String, usize)>>,
    aabb: &AABB<P>,
    mask: ProjectionMask,
) -> Vec<RecordBatch>
where
    <P as rstar::Point>::Scalar: num_traits::cast::NumCast,
{
    let candidates: Vec<_> = rtree
        .locate_in_envelope_intersecting(&aabb.envelope())
        .map(|object| object.data.to_owned())
        .collect();
    candidates
        .into_par_iter()
        .flat_map(|(path, index)| {
            ParquetRecordBatchReaderBuilder::try_new(File::open(path).unwrap())
                .unwrap()
                .with_row_groups(vec![index])
                .with_projection(mask.clone())
                .build()
                .unwrap()
                .par_bridge()
        })
        .map(|result| {
            let record_batch = result.unwrap();
            let predicate = record_batch_aabb_filter(&record_batch, aabb);
            filter_record_batch(&record_batch, &predicate).unwrap()
        })
        .collect()
}

fn mean(data: &[i64]) -> Option<f64> {
    let sum = data.iter().sum::<i64>() as f64;
    let count = data.len();

    match count {
        positive if positive > 0 => Some(sum / count as f64),
        _ => None,
    }
}

fn std_deviation(data: &[i64]) -> Option<f64> {
    match (mean(data), data.len()) {
        (Some(data_mean), count) if count > 0 => {
            let variance = data
                .iter()
                .map(|value| {
                    let diff = data_mean - (*value as f64);

                    diff * diff
                })
                .sum::<f64>()
                / count as f64;

            Some(variance.sqrt())
        }
        _ => None,
    }
}
