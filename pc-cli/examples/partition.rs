#[allow(unused_imports)]
use std::time::Instant;

#[allow(unused_imports)]
use pc_cli::{benchmark::Benchmark, convert::Conversion, partition::Partitioning};

// 200 M
const PATH_LAZ_200M: &str = "./data/AHN4/C_69AZ1.LAZ";
const PATH_PARQUET_200M: &str = "./data/AHN4/C_69AZ1_convert.parquet";
const PATH_GRID_200M: &str = "./data/AHN4/C_69AZ1_grid(1).parquet";
const PATH_GRID_8_200M: &str = "./data/AHN4/C_69AZ1_grid(8).parquet";
const PATH_QUADTREE_200M: &str = "./data/AHN4/C_69AZ1_quadtree.parquet";

// 2 B Delft
const PATH_LAZ_2B: &str = "./data/AHN3/C_37E*.LAZ";
const PATH_PARQUET_2B: &str = "./data/AHN3/C_37E*_convert.parquet";
const PATH_GRID_2B: &str = "./data/AHN3/C_37E*_grid(1).parquet";
const PATH_GRID_8_2B: &str = "./data/AHN3/C_37E*_grid(8).parquet";
const PATH_QUADTREE_2B: &str = "./data/AHN3/C_37E*_quadtree.parquet";

#[tokio::main]
async fn main() {
    println!("Processing 200M point extract!");

    // LEVEL 0: benchmark
    Benchmark::new(PATH_LAZ_200M, false, true).benchmark().await;
    // with chunk index
    Benchmark::new(PATH_LAZ_200M, true, true).benchmark().await; // 11.8s | 90.6s

    // LEVEL 1: importance augmentation & conversion
    let now = Instant::now();

    Conversion::new(&[PATH_LAZ_200M]).convert().await.unwrap();

    println!("Converted in {:?}", now.elapsed()); // 22.1s | 222.5s

    Benchmark::new(PATH_PARQUET_200M, true, true)
        .benchmark()
        .await;

    // LEVEL 2a: partition (xy windowed)
    let now = Instant::now();

    Partitioning::new(&[PATH_LAZ_200M], "grid(1)")
        .partition()
        .await
        .unwrap();

    println!("Partitioned in {:?}", now.elapsed()); // 46.0s | 375.6s

    Benchmark::new(PATH_GRID_200M, true, true).benchmark().await;

    // LEVEL 2b: partition (xyi windowed)
    let now = Instant::now();

    Partitioning::new(&[PATH_LAZ_200M], "grid(8)")
        .partition()
        .await
        .unwrap();

    println!("Partitioned in {:?}", now.elapsed()); // 51.5s | 358.6s

    Benchmark::new(PATH_GRID_8_200M, true, true)
        .benchmark()
        .await;

    // LEVEL 3: build tree
    let now = Instant::now();

    Partitioning::new(&[PATH_LAZ_200M], "quadtree")
        .partition()
        .await
        .unwrap();

    println!("Built tree in {:?}", now.elapsed()); // 65.5s | 398.2s

    Benchmark::new(PATH_QUADTREE_200M, true, true)
        .benchmark()
        .await;

    println!("Processing 2B point extract!");

    // LEVEL 0: benchmark
    Benchmark::new(PATH_LAZ_2B, false, true).benchmark().await;
    // with chunk index
    Benchmark::new(PATH_LAZ_2B, true, true).benchmark().await; // 11.8s | 90.6s

    // LEVEL 1: importance augmentation & conversion
    let now = Instant::now();

    Conversion::new(&[PATH_LAZ_2B]).convert().await.unwrap();

    println!("Converted in {:?}", now.elapsed()); // 22.1s | 222.5s

    Benchmark::new(PATH_PARQUET_2B, true, true)
        .benchmark()
        .await;

    // LEVEL 2a: partition (xy windowed)
    let now = Instant::now();

    Partitioning::new(&[PATH_LAZ_2B], "grid(1)")
        .partition()
        .await
        .unwrap();

    println!("Partitioned in {:?}", now.elapsed()); // 46.0s | 375.6s

    Benchmark::new(PATH_GRID_2B, true, true).benchmark().await;

    // LEVEL 2b: partition (xyi windowed)
    let now = Instant::now();

    Partitioning::new(&[PATH_LAZ_2B], "grid(8)")
        .partition()
        .await
        .unwrap();

    println!("Partitioned in {:?}", now.elapsed()); // 51.5s | 358.6s

    Benchmark::new(PATH_GRID_8_2B, true, true).benchmark().await;

    // LEVEL 3: build tree
    let now = Instant::now();

    Partitioning::new(&[PATH_LAZ_2B], "quadtree")
        .partition()
        .await
        .unwrap();

    println!("Built tree in {:?}", now.elapsed()); // 65.5s | 398.2s

    Benchmark::new(PATH_QUADTREE_2B, true, true)
        .benchmark()
        .await;
}
