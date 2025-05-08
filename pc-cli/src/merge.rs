use std::{fs::File, path::Path, str::FromStr};

use clap::Args;
use datafusion::parquet::{
    arrow::{ArrowWriter, arrow_reader::ParquetRecordBatchReaderBuilder},
    basic::Compression,
    file::properties::{EnabledStatistics, WriterProperties},
};

use pc_format::PointCloudError;

#[derive(Args, Debug)]
pub struct Merging {
    /// Input file paths
    pub input: Vec<String>,
    /// Output folder or file path
    pub output: String,
    /// Create random importance (f16)
    #[arg(long, default_value = "false")]
    pub importance: bool,
    /// Compression: uncompressed, snappy, gzip(level), lzo, brotli(level), lz4, zstd(level), or lz4_raw
    #[arg(long, default_value = "zstd(3)")]
    pub compression: String,
    /// Statistics: none, chunk or page
    #[arg(long, default_value = "page")]
    pub statistics: String,
}

impl Merging {
    pub fn new<T: AsRef<str>>(input: &[T], output: T) -> Self {
        let input = input
            .iter()
            .flat_map(|s| glob::glob(s.as_ref()).expect("read glob pattern"))
            .map(|p| p.unwrap().to_str().unwrap().to_string())
            .collect();

        Merging {
            input,
            output: output.as_ref().to_owned(),
            importance: false,
            compression: "zstd(3)".to_string(),
            statistics: "page".to_string(),
        }
    }

    pub async fn merge(&self) -> Result<(), PointCloudError> {
        // schema
        let file = File::open(Path::new(&self.input[0])).unwrap();
        let reader = ParquetRecordBatchReaderBuilder::try_new(file).unwrap();

        // writer
        let file = File::create(Path::new(&self.output)).unwrap();

        let properties = WriterProperties::builder()
            .set_compression(Compression::from_str(&self.compression).unwrap())
            .set_statistics_enabled(EnabledStatistics::from_str(&self.statistics).unwrap())
            .build();
        let mut writer =
            ArrowWriter::try_new(file, reader.schema().clone(), Some(properties)).unwrap();

        // merge
        let mut parent = None;
        for path in &self.input {
            let path = Path::new(path);

            if parent.is_some() && parent != path.parent() {
                writer.flush().unwrap();
                parent = path.parent();
            } else {
                parent = path.parent();
            }

            let file = File::open(path).unwrap();
            let reader = ParquetRecordBatchReaderBuilder::try_new(file).unwrap();

            for batch in reader.build().unwrap() {
                writer.write(&batch.unwrap()).unwrap();
            }
        }

        writer.close().unwrap();

        Ok(())
    }
}
