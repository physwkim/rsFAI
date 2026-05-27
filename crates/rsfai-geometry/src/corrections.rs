//! Geometry-level correction arrays: solid angle and polarization, ported from
//! `pyFAI/geometry/core.py` (`solidAngleArray`/`diffSolidAngle`/`cos_incidence`
//! + `_geometry.f_cosa`, and `polarization`).
//!
//! These are `Geometry` methods in pyFAI (they combine the detector's pixel
//! positions with the PONI/distance and, for polarization, the unit equations),
//! so they live here rather than in `rsfai-detectors`.
//!
//! ## dtype contract (critical for bit-exactness)
//!
//! * **Solid angle** is built with `numpy.fromfunction(..., dtype=float32)`, so
//!   the pixel positions are computed in **f32** (incl. the `- poni`), then
//!   `f_cosa` upcasts them to **f64** (`calc_cosa` does
//!   `ascontiguousarray(..., dtype=float64)`). The cosine and the `**order`
//!   power are f64; the returned array is f64.
//! * **Polarization** evaluates `tth`/`chi` (the f64 unit equations,
//!   `scale=False`) through a numexpr expression in **f64**, then casts the
//!   result to **f32**.

use rsfai_detectors::Detector;

use crate::units::equation;
use crate::Space;

/// Per-pixel solid-angle correction `cos(incidence)^order`, ported from
/// `solidAngleArray` (default `order = 3`, `absolute = False`).
///
/// Returns a flat row-major `f64` array of length `det.size()`.
///
/// The verified arithmetic (see module docs):
/// ```text
/// p1 = (pixel1 * (i + 0.5) - poni1)   in f32   (likewise p2 from j, poni2)
/// c1 = f64(p1)                                  (f_cosa upcasts to f64)
/// cosa = dist / sqrt(dist*dist + (c1*c1 + c2*c2))
/// dsa  = cosa.powf(order)
/// ```
pub fn solid_angle_array(
    det: &Detector,
    dist: f64,
    poni1: f64,
    poni2: f64,
    order: f64,
) -> Vec<f64> {
    let (p1, p2) = det.centers_f32(); // raw f32 pixel centres, before PONI
    let poni1 = poni1 as f32;
    let poni2 = poni2 as f32;
    p1.iter()
        .zip(&p2)
        .map(|(&a, &b)| {
            // PONI subtraction stays in f32 (numpy weak promotion), then f_cosa
            // upcasts to f64.
            let c1 = (a - poni1) as f64;
            let c2 = (b - poni2) as f64;
            let cosa = dist / (dist * dist + (c1 * c1 + c2 * c2)).sqrt();
            cosa.powf(order)
        })
        .collect()
}

/// Per-pixel polarization correction, ported from `Geometry.polarization`
/// (numexpr path):
/// ```text
/// 0.5 * (1 + cos(tth)^2 - factor * cos(2*(chi + axis_offset)) * (1 - cos(tth)^2))
/// ```
/// evaluated in f64 from the `scale=False` `2th_rad`/`chi_rad` arrays, then cast
/// to f32. `(x, y, z)` are the flat lab coordinates (pyFAI mapping `x=t2`,
/// `y=t1`, `z=t3`); `factor` is the polarization factor, `axis_offset` in rad.
pub fn polarization_array(
    x: &[f64],
    y: &[f64],
    z: &[f64],
    factor: f64,
    axis_offset: f64,
) -> Vec<f32> {
    assert_eq!(x.len(), y.len());
    assert_eq!(x.len(), z.len());
    (0..x.len())
        .map(|i| {
            // tth/chi are the scale=False unit equations (rad).
            let tth = equation(Space::TwoTheta, x[i], y[i], z[i], 1.0);
            let chi = equation(Space::Chi, x[i], y[i], z[i], 1.0);
            let cos2_tth = {
                let c = tth.cos();
                c * c
            };
            let pola = 0.5
                * (1.0 + cos2_tth - factor * (2.0 * (chi + axis_offset)).cos() * (1.0 - cos2_tth));
            pola as f32
        })
        .collect()
}
