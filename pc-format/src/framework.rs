use itertools::Itertools;
use num_traits::{NumCast, One};

use crate::{AABB, Dims, PointTrait};

/// Framework for spatial decomposition and partitioning
#[derive(Debug, Default, Clone, Copy)]
pub struct Grid<P: PointTrait> {
    pub delta: Option<P>,
    // pub cells: RTree<Cell<P>>,
}

impl<P: PointTrait> Grid<P> {
    pub fn new() -> Self {
        Grid { delta: None }
    }

    pub fn with_delta(mut self, delta: P) -> Self {
        self.delta = Some(delta);
        self
    }

    pub fn set_delta(&mut self, delta: P) {
        self.delta = Some(delta);
    }

    pub fn create_cells<'a>(
        &'a self,
        content: &'a AABB<P>,
    ) -> Box<dyn Iterator<Item = AABB<P>> + Send + 'a>
    where
        P: 'static,
        P::Scalar: NumCast,
    {
        if let Some(delta) = self.delta.to_owned() {
            Box::new(grid_coverage(content, delta)) as Box<dyn Iterator<Item = AABB<P>> + Send>
        } else {
            Box::new([content.to_owned()].into_iter()) as Box<dyn Iterator<Item = AABB<P>> + Send>
        }
    }
}

// fn aabb_cast<P: PointTrait, Q: PointTrait>(aabb: &AABB<P>, dims: Dims) -> AABB<Q>
// where
//     P::Scalar: NumCast,
//     Q::Scalar: NumCast,
// {
//     let lower = aabb.lower();
//     let lower = Q::generate(|i| {
//         if i < P::DIMENSIONS {
//             num_traits::cast(lower.nth(i)).unwrap()
//         } else {
//             Q::Scalar::min_value()
//         }
//     });
//     let upper = aabb.upper();
//     let upper = Q::generate(|i| {
//         if i < P::DIMENSIONS {
//             num_traits::cast(upper.nth(i)).unwrap()
//         } else {
//             Q::Scalar::max_value()
//         }
//     });

//     AABB::from_corners(lower, upper, dims)
// }

/// Split [AABB] on each dimension into quasi equal slices of the amount specfied by `splits`.
pub fn split_aabb<P>(aabb: &AABB<P>, splits: &P) -> impl Iterator<Item = AABB<P>> + use<P>
where
    P: PointTrait,
    <P as rstar::Point>::Scalar: NumCast,
{
    let lower = aabb.lower();
    let upper = aabb.upper();

    let delta = upper.sub(&lower).div(splits);
    // dbg!(&delta);

    let fractions = (0..P::DIMENSIONS).map(|index| {
        let lower_nth = lower.nth(index);
        let upper_nth = upper.nth(index);
        let delta_nth = delta.nth(index);

        let splits_nth = num_traits::cast(splits.nth(index)).unwrap();

        (0..splits_nth).map(move |split| {
            let start = if split == 0 {
                lower_nth
            } else {
                lower_nth + delta_nth * num_traits::cast(split).unwrap()
            };

            let end = if split == splits_nth - 1 {
                upper_nth
            } else {
                lower_nth + delta_nth * num_traits::cast(split + 1).unwrap()
            };

            (start, end)
        })
    });

    Itertools::multi_cartesian_product(fractions).map(move |item| {
        AABB::from_corners(
            P::generate(|index| item[index].0),
            P::generate(|index| item[index].1),
        )
    })
}

// pub fn split_aabb_at<P: PointTrait>(aabb: &AABB<P>, dim: usize, at: P::Scalar) -> [AABB<P>; 2]
// where
//     <P as rstar::Point>::Scalar: NumCast,
// {
//     let mut lower = aabb.lower();
//     let mut upper = aabb.upper();

//     *lower.nth_mut(dim) = at;
//     *upper.nth_mut(dim) = at;

//     [
//         AABB::from_corners(aabb.lower(), lower, aabb.dims()),
//         AABB::from_corners(upper, aabb.upper(), aabb.dims()),
//     ]
// }

pub fn grid_coverage<P: PointTrait>(content: &AABB<P>, delta: P) -> impl Iterator<Item = AABB<P>>
where
    <P as rstar::Point>::Scalar: NumCast,
{
    let lower = content.lower();
    let upper = content.upper();

    let grid_indices = (0..P::DIMENSIONS).map(|index| {
        let delta_nth = num_traits::cast::<P::Scalar, f64>(delta.nth(index)).unwrap();

        let from = num_traits::cast::<P::Scalar, f64>(lower.nth(index))
            .unwrap()
            .div_euclid(delta_nth) as i64;
        let to = num_traits::cast::<P::Scalar, f64>(upper.nth(index))
            .unwrap()
            .div_euclid(delta_nth) as i64;

        assert!(
            num_traits::cast::<f64, P::Scalar>(from as f64).unwrap() * delta.nth(index)
                <= lower.nth(index)
        );
        assert!(
            num_traits::cast::<f64, P::Scalar>((to + 1) as f64).unwrap() * delta.nth(index)
                > upper.nth(index)
        );

        from..=to
    });

    Itertools::multi_cartesian_product(grid_indices).map(move |c| {
        let lower = P::generate(|i| {
            num_traits::cast::<f64, P::Scalar>(c[i] as f64).unwrap() * delta.nth(i)
        });
        let upper = P::generate(|i| {
            num_traits::cast::<f64, P::Scalar>((c[i] + 1) as f64).unwrap() * delta.nth(i)
        });

        AABB::from_corners(lower, upper)
    })
}

pub fn quadtree_cells<P: PointTrait>(
    aabb: &AABB<P>,
    depth: usize,
) -> impl Iterator<Item = AABB<P>> + '_
where
    <P as rstar::Point>::Scalar: NumCast,
{
    assert_eq!(P::DIMS, Dims::XYI);

    let num_cells = (0..=depth).map(|d| 4_usize.pow(d as u32)).sum::<usize>();

    let i_fraction = 1. / num_cells as f64;

    let mut cell_count = 0;

    (0..=depth).flat_map(move |level| {
        print!("Level: {level}, ");

        // importance fraction
        let i_lower = if level == 0 {
            0.
        } else {
            cell_count as f64 * i_fraction
        };

        let level_cells = 4_usize.pow(level as u32);
        print!("Cells: {level_cells}, ");
        cell_count += level_cells;

        let i_upper = if level == depth {
            1.
        } else {
            cell_count as f64 * i_fraction
        };
        println!("Importance: [{i_lower}, {i_upper})");

        // level bounds
        let mut p1 = aabb.lower();
        let mut p2 = aabb.upper();

        *p1.nth_mut(2) = num_traits::cast(i_lower).unwrap();
        *p2.nth_mut(2) = num_traits::cast(i_upper).unwrap();

        let level_bounds = AABB::from_corners(p1, p2);

        // level cells
        let split = num_traits::cast((level_cells as f64).sqrt().round()).unwrap();
        let splits = P::from_slice(&[split, split, P::Scalar::one()]);
        split_aabb(&level_bounds, &splits)
    })
}

#[cfg(test)]
mod tests {
    use crate::point::PointXYZ;

    use super::*;

    #[test]
    fn framework() {
        let aabb: AABB<PointXYZ<f64>> = AABB::from_corners(
            PointTrait::from_slice(&[-0.9, -0.9, -0.9]),
            PointTrait::from_slice(&[0.9, 0.9, 0.9]),
        );

        // on cell per content
        let fw: Grid<PointXYZ<f64>> = Grid::new();

        let cells: Vec<_> = fw.create_cells(&aabb).collect();
        assert_eq!(cells.len(), 1);

        // fixed grid
        let fw = Grid::new().with_delta(PointTrait::from_slice(&[1., 1., 1.]));

        let cells: Vec<_> = fw.create_cells(&aabb).collect();
        assert_eq!(cells.len(), 8);
    }
}
