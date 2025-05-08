use std::{
    collections::HashMap,
    fmt::{Debug, Formatter},
    fs::File,
    io::BufReader,
    path::Path,
};

use datafusion::arrow::{datatypes::SchemaRef, error::ArrowError, record_batch::RecordBatch};
use las::{Header, Reader};
use laz::{las::file::read_header_and_vlrs, laszip::ChunkTable};
use rayon::iter::{IntoParallelIterator, ParallelIterator};

use pc_format::PointCloudError;

use crate::config::DEFAULT_BATCH_SIZE;

use super::{
    builder::RowBuilder,
    pruning::{LasStatistics, LasStatisticsBuilder},
    schema::schema_from_header,
};

#[derive(Clone, Copy, Default)]
pub struct LasDataSourceOptions {
    /// keep grid rounded integer coordinates and transformation
    pub raw: bool,
    /// extract chunk statistics on load
    pub stats: bool,
}

/// A LAS/LAZ data source
#[derive(Clone)]
pub struct LasDataSource {
    pub(super) table_paths: Vec<String>,
    pub(super) table_options: LasDataSourceOptions,
    pub(super) table_headers: Vec<Header>,
    table_schema: SchemaRef,
    pub(super) files_statistics: LasStatistics,
    pub(super) chunk_statistics: HashMap<String, LasStatistics>,
}

impl Debug for LasDataSource {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.write_str("las_files")
    }
}

impl LasDataSource {
    pub fn try_new(table_paths: &[impl AsRef<str>]) -> Result<Self, PointCloudError> {
        LasDataSource::try_new_with(table_paths, LasDataSourceOptions::default())
    }

    pub fn try_new_with(
        table_paths: &[impl AsRef<str>],
        table_options: LasDataSourceOptions,
    ) -> Result<Self, PointCloudError> {
        assert!(!table_paths.is_empty());

        let table_paths: Vec<String> = crate::helpers::deglob(table_paths).collect();

        let mut table_headers = Vec::with_capacity(table_paths.len());

        let mut table_schema = None;

        for table_path in &table_paths {
            let reader = Reader::from_path(table_path).map_err(|e| {
                PointCloudError::ArrowError(ArrowError::from_external_error(Box::new(e)))
            })?;
            table_headers.push(reader.header().to_owned());
            if let Some(schema) = table_schema.as_ref() {
                assert_eq!(
                    schema,
                    &schema_from_header(reader.header(), table_options.raw)
                );
            } else {
                table_schema = Some(schema_from_header(reader.header(), table_options.raw));
            }
        }

        let files_statistics = files_statistics(&table_headers);

        let chunk_statistics = if table_options.stats {
            chunk_statistics(&table_paths)
        } else {
            HashMap::new()
        };

        Ok(Self {
            table_paths,
            table_options,
            table_headers,
            table_schema: table_schema.unwrap(),
            files_statistics,
            chunk_statistics,
        })
    }

    pub fn schema(&self) -> SchemaRef {
        self.table_schema.clone()
    }

    pub fn table_paths(&self) -> &[String] {
        &self.table_paths
    }

    pub fn par_record_batch_iter(&self) -> impl ParallelIterator<Item = RecordBatch> + use<> {
        let schema = self.schema();

        let raw = self.table_options.raw;

        self.table_paths
            .clone()
            .into_par_iter()
            .flat_map(move |path| {
                let schema = schema.clone();

                let chunk_table = chunk_table(&path);

                chunk_table
                    .into_par_iter()
                    .map(move |(offset, point_count)| {
                        let mut reader = Reader::from_path(&path).expect("files should exist");
                        reader.seek(offset).expect("seek to offset");

                        let mut builder = RowBuilder::new(point_count as usize, raw);
                        for point in reader.read_points(point_count).expect("read n points") {
                            builder.append(point, reader.header());
                        }

                        RecordBatch::from(builder.finish(&schema, reader.header()))
                    })
            })
    }

    pub fn record_batch_iter(&self) -> impl Iterator<Item = Result<RecordBatch, ArrowError>> {
        pc_format::helpers::into_iter(self.par_record_batch_iter()).map(Result::Ok)
    }
}

fn files_statistics(headers: &[Header]) -> LasStatistics {
    let mut builder = LasStatisticsBuilder::new_with_capacity(headers.len());

    for header in headers.iter() {
        let bounds = header.bounds();
        builder.add_values(&[
            bounds.min.x,
            bounds.max.x,
            bounds.min.y,
            bounds.max.y,
            bounds.min.z,
            bounds.max.z,
        ]);
    }

    builder.finish()
}

pub fn chunk_table(path: impl AsRef<Path>) -> Vec<(u64, u64)> {
    let path = path.as_ref();

    let mut offset = 0;

    match path
        .extension()
        .and_then(|s| s.to_str().map(|s| s.to_uppercase()))
        .as_deref()
    {
        Some("LAS") => {
            let mut total_points = Reader::from_path(path).unwrap().header().number_of_points();

            let chunck_size = DEFAULT_BATCH_SIZE as u64;

            (0..total_points.div_ceil(chunck_size))
                .map(|_| {
                    let point_count = total_points.min(chunck_size);

                    let chunk = (offset, point_count);
                    offset += chunck_size;
                    total_points -= chunck_size;
                    chunk
                })
                .collect()
        }
        Some("LAZ") => {
            let file = File::open(path).unwrap();
            let mut reader = BufReader::new(file);

            let (_, laz_vlr) = read_header_and_vlrs(&mut reader).unwrap();
            let laz_vlr = laz_vlr.expect("Expected a laszip VLR for laz file");
            let chunk_table = ChunkTable::read_from(reader, &laz_vlr).unwrap();

            chunk_table
                .as_ref()
                .iter()
                .map(|chunk| {
                    let chunk = (offset, chunk.point_count);
                    offset += chunk.1;
                    chunk
                })
                .collect()
        }
        Some(_) | None => {
            eprintln!("Unsupported extension `{:?}`", path);
            Vec::new()
        }
    }
}

fn chunk_statistics(paths: &[String]) -> HashMap<String, LasStatistics> {
    // TODO: maybe speed up with `pasture`
    HashMap::from_iter(paths.iter().map(|path| {
        let chunks = chunk_table(path);

        let mut builder = LasStatisticsBuilder::new_with_capacity(chunks.len());

        let stats: Vec<[f64; 6]> = chunks
            .into_par_iter()
            .map(|(offset, point_count)| {
                let mut reader = Reader::from_path(path).expect("files should exist");
                reader.seek(offset).expect("seek to offset");

                reader
                    .read_points(point_count)
                    .expect("read n points")
                    .iter()
                    .map(|p| [p.x, p.x, p.y, p.y, p.z, p.z])
                    .reduce(|acc, e| {
                        [
                            acc[0].min(e[0]),
                            acc[1].max(e[1]),
                            acc[2].min(e[2]),
                            acc[3].max(e[3]),
                            acc[4].min(e[4]),
                            acc[5].max(e[5]),
                        ]
                    })
                    .unwrap()
            })
            .collect();

        for values in stats.iter() {
            builder.add_values(values);
        }

        (path.to_owned(), builder.finish())
    }))
}
