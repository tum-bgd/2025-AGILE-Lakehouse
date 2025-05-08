mod builder;
mod exec;
mod provider;
mod pruning;
mod schema;
mod source;

pub use source::*;

#[cfg(test)]
mod tests {
    use std::{sync::Arc, time::Instant};

    use datafusion::execution::context::SessionContext;
    use las::Reader;
    use rayon::iter::ParallelIterator;

    use pc_format::{PointCloud, PointXYZI};

    use super::*;

    #[test]
    fn baseline() {
        let mut reader = Reader::from_path("../data/AHN3/C_69AZ1.LAZ").unwrap();

        // header
        let header = reader.header();
        println!("Version: {} - {}", header.version(), header.point_format());
        println!("Number of Points: {}", header.number_of_points());
        println!("{header:#?}");

        // points
        let start = Instant::now();
        let points: Vec<_> = reader
            .points()
            .map(|p| {
                let p = p.unwrap();
                [p.x, p.y, p.z]
            })
            .collect();
        println!(
            "Loaded {} points ({:.2} points/s)",
            points.len(),
            points.len() as f64 / start.elapsed().as_secs_f64()
        );
    }

    #[test]
    fn record_batch_reader() {
        let start = Instant::now();

        let ds = LasDataSource::try_new(&["../data/AHN3/C_69AZ1.LAZ"]).unwrap();

        let mut pc = PointCloud::try_new(ds.schema()).unwrap();

        for batch in ds.record_batch_iter() {
            pc.append(batch.unwrap()).unwrap();
        }

        let duration = start.elapsed();

        println!(
            "Load {} points ({:.2} points/s)",
            pc.num_points(),
            pc.num_points() as f64 / duration.as_secs_f64()
        );

        println!("{:#?}", pc.aabb::<PointXYZI<f64>>());
    }

    #[tokio::test]
    async fn par_record_batch_reader() {
        let start = Instant::now();

        let ds = LasDataSource::try_new(&["../data/AHN3/C_69AZ1.LAZ"]).unwrap();

        let num_points: usize = ds
            .par_record_batch_iter()
            .map(|result| result.num_rows())
            .sum();

        let duration = start.elapsed();

        println!(
            "Load {} points ({:.2} points/s)",
            num_points,
            num_points as f64 / duration.as_secs_f64()
        );
    }

    #[tokio::test]
    async fn sql() {
        let start = Instant::now();

        let ds = LasDataSource::try_new(&["../data/AHN3/C_69AZ1.LAZ"]).unwrap();

        let ctx = SessionContext::new();

        ctx.register_table("laz", Arc::new(ds)).unwrap();

        let results = ctx
            .sql("SELECT * FROM 'laz'")
            .await
            .unwrap()
            .collect()
            .await
            .unwrap();

        let duration = start.elapsed();

        let num_points: usize = results.iter().map(|b| b.num_rows()).sum();

        println!(
            "Load {} points ({:.2} points/s)",
            num_points,
            num_points as f64 / duration.as_secs_f64()
        );

        // aggregation
        ctx.sql("SELECT mean(x) FROM 'laz'")
            .await
            .unwrap()
            .show()
            .await
            .unwrap();
    }
}
