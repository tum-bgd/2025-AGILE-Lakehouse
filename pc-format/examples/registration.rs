use std::f64::consts::PI;

use nalgebra::{DMatrix, Matrix4, OPoint, RealField, Rotation3, SimdValue};
use num_traits::{Float, NumCast, Zero};
use simplers_optimization::Optimizer;

use pc_format::{PointTrait, PointXYZ};

fn main() {
    let a = vec![
        PointTrait::from_slice(&[0.0, 1.0, 2.0]),
        PointTrait::from_slice(&[2.0, 0.0, 1.0]),
        PointTrait::from_slice(&[1.0, 2.0, 0.0]),
    ];

    let angle = rand::random_range(0.0..2. * PI);
    println!("The objective angle is {angle:.3}");

    let rot = Rotation3::from_euler_angles(0., 0., angle);
    let x = transformed::<PointXYZ<f64>>(&a, &rot.to_homogeneous());

    let inferred = registration::<PointXYZ<f64>>(&a, &x, 10);
    println!("Inferred angle is {inferred:.3}");

    let d = inferred - angle;
    let error = d.min((2. * PI + d).abs()).min((-2. * PI + d).abs());
    println!("Final error is {error:.3}");
}

/// returns a transform pointcloud
pub fn transformed<P>(pc: &[P], transform: &Matrix4<<P as rstar::Point>::Scalar>) -> Vec<P>
where
    P: PointTrait,
    <P as rstar::Point>::Scalar: NumCast + RealField,
{
    pc.iter()
        .map(|p| {
            let coords: Vec<P::Scalar> = p.coords().collect();
            P::from_slice(
                transform
                    .transform_point(&OPoint::from_slice(&coords))
                    .coords
                    .as_slice(),
            )
        })
        .collect()
}

/// calculate discrete hausdorff distance
fn discrete_hausdorff_distance<P>(pc_x: &[P], pc_y: &[P]) -> f64
where
    P: PointTrait,
    <P as rstar::Point>::Scalar: NumCast + SimdValue + 'static,
{
    // get maximum of minimum shortes distance between point pair
    let dm = DMatrix::from_iterator(
        pc_x.len(),
        pc_x.len(),
        pc_x.iter().flat_map(|p_x| {
            pc_y.iter().map(move |p_y| {
                let d = p_y.sub(p_x);
                d.mul(&d)
                    .coords()
                    .fold(<P as rstar::Point>::Scalar::zero(), |acc, d| acc + d)
            })
        }),
    );

    let max_min_x = dm
        .row_iter()
        .map(|r| {
            r.iter()
                .min_by(|a, b| a.partial_cmp(b).unwrap())
                .unwrap()
                .to_owned()
        })
        .max_by(|a, b| a.partial_cmp(b).unwrap())
        .unwrap();

    let max_min_y = dm
        .column_iter()
        .map(|r| {
            r.iter()
                .min_by(|a, b| a.partial_cmp(b).unwrap())
                .unwrap()
                .to_owned()
        })
        .max_by(|a, b| a.partial_cmp(b).unwrap())
        .unwrap();

    if max_min_x > max_min_y {
        num_traits::cast::<_, f64>(max_min_x).unwrap().sqrt()
    } else {
        num_traits::cast::<_, f64>(max_min_y).unwrap().sqrt()
    }
}

fn registration<P>(pc_a: &[P], pc_x: &[P], nb_iterations: usize) -> f64
where
    P: PointTrait,
    <P as rstar::Point>::Scalar: NumCast + RealField + Float,
{
    let zero = <P as rstar::Point>::Scalar::zero();

    let f = |the_angle: &[<P as rstar::Point>::Scalar]| {
        let rot = Rotation3::from_euler_angles(zero, zero, the_angle[0]);
        let pc_y = transformed::<P>(pc_a, &rot.to_homogeneous());
        let ret = discrete_hausdorff_distance::<P>(pc_x, &pc_y);
        println!("Tried angle {:.3} with loss {ret:.3}", the_angle[0]);
        ret
    };

    let input_intervals = [(zero, num_traits::cast(2. * PI).unwrap())];

    let r = Optimizer::minimize(&f, &input_intervals, nb_iterations);

    num_traits::cast(r.1[0]).unwrap()
}
