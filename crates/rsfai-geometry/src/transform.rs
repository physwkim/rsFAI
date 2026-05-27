//! The pixel → sample-frame coordinate transform, ported from
//! `pyFAI/ext/_geometry.pyx` (`f_t1`/`f_t2`/`f_t3`, `calc_pos_zyx`).
//!
//! Given detector pixel centres `(p1, p2[, p3])` in metres and the PONI
//! geometry, this returns the lab coordinates `(z, y, x)` = `(t3, t1, t2)`:
//!   * z (t3): along the incident beam,
//!   * y (t1): toward the top,
//!   * x (t2): toward the ring centre.
//!
//! The per-pixel math is pure f64 `+ - *` (IEEE-exact given identical inputs);
//! the only transcendentals are the six scalar `sin`/`cos` of the rotation
//! angles, computed once. With no FMA contraction this reproduces pyFAI's
//! `calc_pos_zyx` bit-for-bit (validated in the golden tests). The per-pixel
//! loop is parallelized (each output element is an independent pure function of
//! its index, no cross-pixel reduction), so the result is bit-identical to the
//! serial order regardless of thread count.

/// Lab-frame coordinates for a flat array of pixels (row-major), named to match
/// pyFAI's `position_array` last-axis order `[z, y, x]`.
#[derive(Debug, Clone)]
pub struct PosZyx {
    /// Along the beam (t3).
    pub z: Vec<f64>,
    /// Toward the top (t1).
    pub y: Vec<f64>,
    /// Toward the ring centre (t2).
    pub x: Vec<f64>,
}

/// `orient` factor for `f_t1`: -1 for orientation 1 or 2, else +1
/// (`_geometry.pyx:82`).
#[inline]
fn orient_t1(orientation: i32) -> f64 {
    if orientation == 1 || orientation == 2 {
        -1.0
    } else {
        1.0
    }
}

/// `orient` factor for `f_t2`: -1 for orientation 1 or 4, else +1
/// (`_geometry.pyx:102`).
#[inline]
fn orient_t2(orientation: i32) -> f64 {
    if orientation == 1 || orientation == 4 {
        -1.0
    } else {
        1.0
    }
}

/// Port of `_geometry.calc_pos_zyx`. `p1`/`p2` are pixel centres (m) along the
/// slow/fast axes; `p3` is the per-pixel altitude (m) for non-flat detectors
/// (`None` for flat, matching pyFAI's `pos3 is None` branch where `L3 = dist`).
#[allow(clippy::too_many_arguments)]
pub fn calc_pos_zyx(
    dist: f64,
    poni1: f64,
    poni2: f64,
    rot1: f64,
    rot2: f64,
    rot3: f64,
    p1: &[f64],
    p2: &[f64],
    p3: Option<&[f64]>,
    orientation: i32,
) -> PosZyx {
    use rayon::prelude::*;
    assert_eq!(p1.len(), p2.len(), "p1/p2 length mismatch");
    if let Some(p3) = p3 {
        assert_eq!(p1.len(), p3.len(), "p1/p3 length mismatch");
    }
    let n = p1.len();

    let sin_rot1 = rot1.sin();
    let cos_rot1 = rot1.cos();
    let sin_rot2 = rot2.sin();
    let cos_rot2 = rot2.cos();
    let sin_rot3 = rot3.sin();
    let cos_rot3 = rot3.cos();
    let orient1 = orient_t1(orientation);
    let orient2 = orient_t2(orientation);

    let mut z = vec![0.0f64; n];
    let mut y = vec![0.0f64; n];
    let mut x = vec![0.0f64; n];

    // Write z/y/x in place; element i depends only on (p1[i], p2[i], p3[i]) and
    // the hoisted rotation sines/cosines, so any thread order yields the same
    // bits. `enumerate` supplies the flat index i for the slice reads.
    z.par_iter_mut()
        .zip(y.par_iter_mut())
        .zip(x.par_iter_mut())
        .enumerate()
        .for_each(|(i, ((zi, yi), xi))| {
            let pp1 = p1[i] - poni1;
            let pp2 = p2[i] - poni2;
            let l3 = match p3 {
                Some(p3) => p3[i] + dist,
                None => dist,
            };

            // f_t1 (y): grouped exactly as the C expression (_geometry.pyx:83-85).
            let t1 = orient1
                * (pp1 * cos_rot2 * cos_rot3
                    + pp2 * (cos_rot3 * sin_rot1 * sin_rot2 - cos_rot1 * sin_rot3)
                    - l3 * (cos_rot1 * cos_rot3 * sin_rot2 + sin_rot1 * sin_rot3));

            // f_t2 (x): _geometry.pyx:103-105.
            let t2 = orient2
                * (pp1 * cos_rot2 * sin_rot3
                    + pp2 * (cos_rot1 * cos_rot3 + sin_rot1 * sin_rot2 * sin_rot3)
                    - l3 * (-(cos_rot3 * sin_rot1) + cos_rot1 * sin_rot2 * sin_rot3));

            // f_t3 (z): _geometry.pyx:123 (orientation has no effect).
            let t3 = pp1 * sin_rot2 - pp2 * cos_rot2 * sin_rot1 + l3 * cos_rot1 * cos_rot2;

            *zi = t3;
            *yi = t1;
            *xi = t2;
        });

    PosZyx { z, y, x }
}
