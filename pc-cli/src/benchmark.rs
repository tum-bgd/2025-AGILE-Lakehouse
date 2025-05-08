use std::{
    path::PathBuf,
    sync::Arc,
    time::{Duration, Instant},
};

use clap::Parser;
use datafusion::{
    common::Column,
    prelude::{Add, DataFrame, ParquetReadOptions, SessionContext, Sub, col, lit, power, random},
};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha20Rng;
use rstar::Point as _;

use pc_format::{AABB, PointXYZ, expressions::df_aabb};
use pc_io::{
    config::default_session_config,
    las::{LasDataSource, LasDataSourceOptions},
};

const TABLE: &str = "test";

#[derive(Parser, Debug)]
pub struct Benchmark {
    /// Input files
    pub input: PathBuf,
    /// Collect statistics
    #[arg(long)]
    pub collect_statistics: bool,
    /// Execute mpling query
    #[arg(long)]
    pub sampling: bool,
    /// Number of runs per query
    #[arg(long)]
    pub runs: usize,
}

impl Benchmark {
    pub fn new(input: &str, stats: bool, sampling: bool) -> Self {
        Benchmark {
            input: PathBuf::from(input),
            collect_statistics: stats,
            sampling,
            runs: 10,
        }
    }

    pub async fn benchmark(&mut self) {
        println!("Benchmarking {}", self.input.display());
        assert!(self.runs > 0);

        let ctx = SessionContext::new_with_config(default_session_config());

        // registration
        let now = Instant::now();

        match self.input.extension() {
            Some(ext) => match ext.to_str().unwrap() {
                "LAZ" | "laz" => {
                    if !self.collect_statistics {
                        self.runs = 1;
                    }

                    let options = LasDataSourceOptions {
                        raw: false,
                        stats: self.collect_statistics,
                    };
                    let ds = LasDataSource::try_new_with(&[self.input.to_str().unwrap()], options)
                        .unwrap();
                    ctx.register_table(TABLE, Arc::new(ds)).unwrap();
                }
                "parquet" => ctx
                    .register_parquet(
                        TABLE,
                        self.input.to_str().unwrap(),
                        ParquetReadOptions::default(),
                    )
                    .await
                    .unwrap(),
                ext => unimplemented!("Unhandled extension: {}", ext),
            },
            None => unimplemented!("Path without extension: {}", self.input.to_str().unwrap()),
        }

        println!("    Registered table in {:?}", now.elapsed());

        // benchmark
        let df = ctx.table(TABLE).await.unwrap();
        // let df = df.select_columns(&["x", "y", "z", "i"]).unwrap();
        benchmark_df(df, self.runs, self.sampling).await;
    }
}

pub async fn benchmark_df(df: DataFrame, runs: usize, sampling: bool) {
    // seed
    let mut rng = ChaCha20Rng::seed_from_u64(76);

    // count
    let now = Instant::now();

    let count = df.clone().count().await.unwrap();

    show("count (df)", &[now.elapsed()], count);

    // bounds
    let now = Instant::now();

    let bounds: AABB<PointXYZ<f64>> = df_aabb(&df).await.unwrap();

    show("bounds (df)", &[now.elapsed()], count);
    // println!("    {:?}", bounds);

    // ranges
    let lower = bounds.lower();
    let upper = bounds.upper();
    let xmin = lower.nth(0) + 1000.0;
    let xmax = upper.nth(0) - 400.0;
    let ymin = lower.nth(1) + 1000.0;
    let ymax = upper.nth(1) - 400.0;

    // xy (1 S_RECT / 2 M_RECT)
    for (name, edge) in [("[ 1] S_RECT", 70.0), ("[ 2] M_RECT", 220.0)] {
        let mut count = 0;
        let mut times = Vec::with_capacity(runs);

        for _ in 0..runs {
            let xorigin = rng.random_range(xmin..xmax);
            let yorigin = rng.random_range(ymin..ymax);

            let now = Instant::now();
            let expr = col("x")
                .gt_eq(lit(xorigin))
                .and(col("x").lt(lit(xorigin + edge)))
                .and(col("y").gt_eq(lit(yorigin)))
                .and(col("y").lt(lit(yorigin + edge)));
            let df = df.clone().filter(expr.clone()).unwrap();
            let results = df.collect().await.unwrap();
            count += results.iter().map(|batch| batch.num_rows()).sum::<usize>();
            times.push(now.elapsed());
        }

        show(name, &times, count / runs);
    }

    // importance
    if sampling {
        let total = df.clone().count().await.unwrap();

        for (name, amount) in [
            ("p (100000)", 100000.),
            ("p (500000)", 500000.),
            ("p (700000)", 700000.),
        ] {
            let mut count = 0;
            let mut times = Vec::with_capacity(runs);

            for _ in 0..runs {
                let now = Instant::now();

                let df2 = if df.schema().has_column(&Column::from_name("i")) {
                    df.clone()
                } else {
                    df.clone().with_column("i", random()).unwrap()
                };

                let results = df2
                    .filter(col("i").lt(lit((amount / total as f64) as f32)))
                    .unwrap()
                    .collect()
                    .await
                    .unwrap();

                count += results.iter().map(|batch| batch.num_rows()).sum::<usize>();
                times.push(now.elapsed());
            }

            show(name, &times, count / runs);
        }
    }

    // circle (3 S_CRC / 4 M_CRC)
    for (name, radius) in [("[ 3] S_CRC", 25.), ("[ 4] M_CRC", 100.)] {
        let mut count = 0;
        let mut times = Vec::with_capacity(runs);

        for _ in 0..runs {
            let xorigin = rng.random_range(xmin..xmax);
            let yorigin = rng.random_range(ymin..ymax);

            let now = Instant::now();
            let results = df
                .clone()
                .filter(
                    col("x")
                        .gt_eq(lit(xorigin - radius))
                        .and(col("x").lt(lit(xorigin + radius)))
                        .and(col("y").gt_eq(lit(yorigin - radius)))
                        .and(col("y").lt(lit(yorigin + radius))),
                )
                .unwrap()
                .cache()
                .await
                .unwrap()
                .filter(
                    power(col("x").sub(lit(xorigin)), lit(2))
                        .add(power(col("y").sub(lit(yorigin)), lit(2)))
                        .lt(power(lit(radius), lit(2))),
                )
                .unwrap()
                .collect()
                .await
                .unwrap();

            count += results.iter().map(|batch| batch.num_rows()).sum::<usize>();
            times.push(now.elapsed());
        }

        show(name, &times, count / runs);
    }

    // nn (18 NN_1000 / 19 NN_5000)
    for (name, k) in [("[18] NN_1000", 1000), ("[19] NN_5000", 5000)] {
        let mut count = 0;
        let mut times = Vec::with_capacity(runs);

        for _ in 0..runs {
            let radius = (k as f64 / 10.).sqrt();

            let xorigin = rng.random_range(xmin..xmax);
            let yorigin = rng.random_range(ymin..ymax);

            let now = Instant::now();

            let df_filtered = df
                .clone()
                .filter(
                    col("x")
                        .gt_eq(lit(xorigin - radius))
                        .and(col("x").lt(lit(xorigin + radius)))
                        .and(col("y").gt_eq(lit(yorigin - radius)))
                        .and(col("y").lt(lit(yorigin + radius))),
                )
                .unwrap()
                .cache()
                .await
                .unwrap();
            let df_sorted = df_filtered
                .with_column(
                    "d",
                    power(col("x").sub(lit(xorigin + radius / 2.)), lit(2))
                        .add(power(col("y").sub(lit(yorigin + radius / 2.)), lit(2))),
                )
                .unwrap()
                .sort(vec![col("d").sort(true, false)])
                .unwrap();
            let df_k = df_sorted.limit(0, Some(k)).unwrap();
            let results = df_k.collect().await.unwrap();

            count += results.iter().map(|batch| batch.num_rows()).sum::<usize>();
            times.push(now.elapsed());
        }
        show(name, &times, count / runs);
    }
}

fn show(name: &str, times: &[Duration], count: usize) {
    let total = times.iter().map(|d| d.as_secs_f64()).sum::<f64>();
    let mean = total / times.len() as f64;

    let var = (times
        .iter()
        .map(|time| ((mean - time.as_secs_f64()) * 1000.).powi(2))
        .sum::<f64>()
        / times.len() as f64)
        .sqrt();

    let s = mean.floor() as u64;
    let ms = ((mean - s as f64) * 1000.).floor() as u64;

    println!("\tQuery {name:18} {s:>5?}s {ms:03}ms (+/- {var:0.0}ms) [{count}]");
}
