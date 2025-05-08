use std::time::Instant;

use clap::{Parser, Subcommand};

use pc_cli::{benchmark::Benchmark, convert::Conversion, merge::Merging};

#[derive(Parser)]
#[command(about = "A point cloud cli.")]
#[command(author, version, long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Option<Commands>,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// Benchmark
    Benchmark(Benchmark),
    /// Convert point cloud format
    Convert(Conversion),
    /// Merge point cloud files
    Merge(Merging),
}

#[tokio::main]
async fn main() {
    let cli = Cli::parse();

    let start = Instant::now();

    println!("{:#?}", cli.command);
    match cli.command {
        Some(Commands::Benchmark(mut args)) => args.benchmark().await,
        Some(Commands::Convert(args)) => args.convert().await.unwrap(),
        Some(Commands::Merge(args)) => args.merge().await.unwrap(),
        None => {}
    }

    println!("Finished command in {:?}", start.elapsed())
}
