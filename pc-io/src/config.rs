use datafusion::prelude::SessionConfig;

pub const DEFAULT_BATCH_SIZE: usize = 1_048_576 / 2; // 1024 * 1024 / 2
pub const DEFAULT_BATCH_SIZE_GDAL: usize = 65_536;

pub fn default_session_config() -> SessionConfig {
    let num_cpus = std::thread::available_parallelism().unwrap().get();

    let mut config = SessionConfig::new();
    config.options_mut().execution.collect_statistics = true;
    config.options_mut().execution.coalesce_batches = false;
    config.options_mut().execution.parquet.pushdown_filters = true;
    config.options_mut().execution.parquet.reorder_filters = true;
    // config.options_mut().execution.parquet.statistics_enabled = Some("chunk".to_string());
    config.options_mut().execution.parquet.compression = Some("uncompressed".to_string());
    config
        .options_mut()
        .execution
        .max_buffered_batches_per_output_file = num_cpus * 8;
    config
        .options_mut()
        .execution
        .parquet
        .maximum_parallel_row_group_writers = num_cpus;
    config
        .options_mut()
        .execution
        .parquet
        .maximum_buffered_record_batches_per_stream = num_cpus * 8;
    config.options_mut().execution.parquet.max_row_group_size = DEFAULT_BATCH_SIZE;

    config
}
