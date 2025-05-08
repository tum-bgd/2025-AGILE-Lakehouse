use std::{collections::HashMap, fs::File};

use datafusion::{
    config::TableParquetOptions,
    datasource::{
        file_format::parquet::ParquetSink, listing::ListingTableUrl, physical_plan::FileSinkConfig,
    },
    execution::{RecordBatchStream, object_store::ObjectStoreUrl},
    logical_expr::dml::InsertOp,
    parquet::{
        basic::Type,
        file::{
            metadata::RowGroupMetaData,
            reader::{FileReader, SerializedFileReader},
        },
    },
    physical_plan::insert::DataSink,
    prelude::SessionContext,
};

use pc_format::{AABB, Dims, PointTrait};

/// Resolve glob patterns in file paths.
pub fn deglob<I: IntoIterator<Item: AsRef<str>>>(patterns: I) -> impl Iterator<Item = String> {
    patterns
        .into_iter()
        .flat_map(|s: <I as IntoIterator>::Item| {
            glob::glob(s.as_ref())
                .unwrap()
                .map(|p| p.unwrap().to_str().unwrap().to_owned())
        })
}

/// Extract bounds from Parquet files.
pub fn bounds_from_parquet_files<P: PointTrait<Scalar = f64>>(
    input: &[impl AsRef<str>],
    dims: Dims,
) -> HashMap<String, Vec<AABB<P>>> {
    let mut bounds = HashMap::new();

    for path in deglob(input) {
        let reader = SerializedFileReader::new(File::open(&path).unwrap()).unwrap();

        let meta = reader.metadata();

        // column indices
        let mut column_indices = vec![0; dims.names().len()];
        for (column_index, column) in meta
            .file_metadata()
            .schema_descr()
            .columns()
            .iter()
            .enumerate()
        {
            if let Some(pos) = dims.names().iter().position(|d| *d == column.name()) {
                column_indices[pos] = column_index;
            }
        }

        // row groups bounds
        let row_groups_bounds = meta
            .row_groups()
            .iter()
            .map(|row_group_meta_data| aabb_from_row_group(row_group_meta_data, &column_indices))
            .collect();

        bounds.insert(path, row_groups_bounds);
    }

    bounds
}

/// Create [AABB] from Parquet [RowGroupMetaData].
pub fn aabb_from_row_group<P: PointTrait<Scalar = f64>>(
    row_group_meta_data: &RowGroupMetaData,
    column_indices: &[usize],
) -> AABB<P> {
    let mut min = vec![f64::MAX; column_indices.len()];
    let mut max = vec![f64::MIN; column_indices.len()];

    for (index, column_index) in column_indices.iter().enumerate() {
        let column = row_group_meta_data.column(*column_index);

        if let Some(stats) = column.statistics() {
            if stats.min_is_exact() && stats.max_is_exact() {
                match stats.physical_type() {
                    Type::FLOAT => {
                        let mb = stats.min_bytes_opt().unwrap();
                        min[index] = f32::from_le_bytes([mb[0], mb[1], mb[2], mb[3]]) as f64;
                        let mb = stats.max_bytes_opt().unwrap();
                        max[index] = f32::from_le_bytes([mb[0], mb[1], mb[2], mb[3]]) as f64;
                    }
                    Type::DOUBLE => {
                        let mb = stats.min_bytes_opt().unwrap();
                        min[index] = f64::from_le_bytes([
                            mb[0], mb[1], mb[2], mb[3], mb[4], mb[5], mb[6], mb[7],
                        ]);
                        let mb = stats.max_bytes_opt().unwrap();
                        max[index] = f64::from_le_bytes([
                            mb[0], mb[1], mb[2], mb[3], mb[4], mb[5], mb[6], mb[7],
                        ]);
                    }
                    Type::FIXED_LEN_BYTE_ARRAY => {
                        let mb = stats.min_bytes_opt().unwrap();
                        min[index] = f64::from(half::f16::from_le_bytes([mb[0], mb[1]]));
                        let mb = stats.max_bytes_opt().unwrap();
                        max[index] = f64::from(half::f16::from_le_bytes([mb[0], mb[1]]));
                    }
                    t => {
                        dbg!(stats);
                        unimplemented!("{t}");
                    }
                }
            } else {
                eprintln!("No exact statistics!");
                dbg!(stats);
            }
        } else {
            eprintln!("Missing statistics!");
            dbg!(column);
        }
    }

    AABB::from_corners(PointTrait::from_slice(&min), PointTrait::from_slice(&max))
}

pub async fn sink<S: RecordBatchStream + Send + 'static>(
    data: S,
    output: &str,
    ctx: &SessionContext,
) {
    let config = FileSinkConfig {
        object_store_url: ObjectStoreUrl::local_filesystem(),
        file_groups: vec![],
        table_paths: vec![ListingTableUrl::parse(output).unwrap()],
        output_schema: data.schema(),
        table_partition_cols: vec![],
        insert_op: InsertOp::Overwrite,
        keep_partition_by_columns: false,
        file_extension: output.split('.').last().unwrap().to_string(),
    };

    let parquet_options = TableParquetOptions {
        global: ctx.state().config().options().execution.parquet.clone(),
        column_specific_options: HashMap::new(),
        key_value_metadata: HashMap::new(),
    };

    let sink = ParquetSink::new(config, parquet_options);

    sink.write_all(Box::pin(data), &ctx.task_ctx())
        .await
        .unwrap();
}
