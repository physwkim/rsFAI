//! Corner & delta geometry arrays, ported from pyFAI's `corner_array` and
//! `delta_array` (`geometry/core.py`) for a contiguous flat detector.
//!
//! `corner_array` takes the cython fast path `_geometry.calc_rad_azim`: for every
//! node of the `(s0+1)×(s1+1)` corner grid it evaluates the radial value (the
//! same f64 `equation` as `center_array`) and `chi = atan2(t1, t2)` — both in
//! f64, each **stored as f32** (`calc_rad_azim` writes a `float[:, ::1]`). The
//! four grid nodes around pixel `(i, j)` are then gathered in pyFAI's winding
//! `[0]=(i,j) [1]=(i+1,j) [2]=(i+1,j+1) [3]=(i,j+1)` (`geometry/core.py`:
//! `corners[:,:,0,:]=res[:-1,:-1]; [1]=res[1:,:-1]; [2]=res[1:,1:]; [3]=res[:-1,1:]`).
//!
//! `delta_array` for a radial space is `max_c |f64(corner_radial_f32[c]) −
//! center|`: the f32 corner widens to f64 before the subtraction (numpy
//! `f32 − f64 → f64`), then the max over the 4 corners. The chi-space variant
//! (`min(δ, 2π−δ)` wrap) is `delta_chi`.

use std::f64::consts::PI;

use rayon::prelude::*;

use crate::transform::PosZyx;
use crate::units::{equation, Space, Unit};

/// `2π` in f64 — pyFAI's `twopi = 2.0 * M_PI` (`_geometry.pyx:55`).
const TWO_PI: f64 = 2.0 * PI;

/// Per-corner `(radial, chi)` array — the f32 `(npix, 4, 2)` of pyFAI's
/// `corner_array(unit, scale=False)` for a contiguous flat detector, flattened
/// row-major: pixel `p`, corner `c`, component `k` at `((p*4)+c)*2 + k`
/// (`k=0` radial in `unit.space`, `k=1` chi in rad).
///
/// `grid` are the lab coords `(z,y,x)` of the `(s0+1)×(s1+1)` corner-grid nodes
/// (row-major; the output of `calc_pos_zyx` over `Detector::corner_positions_f64`).
/// `shape = (s0, s1)` is the pixel grid. `chi_disc_at_pi = true` keeps chi in
/// `[-π, π)` (`atan2`); `false` maps it to `[0, 2π)` — pyFAI's `chiDiscAtPi`.
///
/// Each node's `(radial, chi)` is computed **once** (then gathered into the 4
/// pixels that share it), matching pyFAI's single `res` array sliced four ways —
/// bit-identical to recomputing per pixel, and four times less transcendental work.
pub fn corner_array_f32(
    grid: &PosZyx,
    shape: (usize, usize),
    unit: Unit,
    wavelength: f64,
    chi_disc_at_pi: bool,
) -> Vec<f32> {
    let (s0, s1) = shape;
    let (c0, c1) = (s0 + 1, s1 + 1);
    assert_eq!(grid.x.len(), c0 * c1, "corner grid length mismatch");

    // res[node] = (radial_f32, chi_f32), the calc_rad_azim store dtype. Each
    // node is an independent pure function of its coords -> bit-exact in parallel.
    let res: Vec<(f32, f32)> = (0..c0 * c1)
        .into_par_iter()
        .map(|idx| {
            let (x, y, z) = (grid.x[idx], grid.y[idx], grid.z[idx]);
            let radial = equation(unit.space, x, y, z, wavelength);
            // chi = atan2(t1, t2) = atan2(y, x) = equation(Chi, ...).
            let mut chi = equation(Space::Chi, x, y, z, wavelength);
            if !chi_disc_at_pi {
                chi = (chi + TWO_PI) % TWO_PI;
            }
            (radial as f32, chi as f32)
        })
        .collect();

    let mut out = vec![0.0f32; s0 * s1 * 4 * 2];
    for i in 0..s0 {
        for j in 0..s1 {
            let p = i * s1 + j;
            // Winding (geometry/core.py): corner 0=(i,j) 1=(i+1,j) 2=(i+1,j+1) 3=(i,j+1).
            let nodes = [
                i * c1 + j,
                (i + 1) * c1 + j,
                (i + 1) * c1 + (j + 1),
                i * c1 + (j + 1),
            ];
            for (c, &nidx) in nodes.iter().enumerate() {
                let (rad, chi) = res[nidx];
                let base = (p * 4 + c) * 2;
                out[base] = rad;
                out[base + 1] = chi;
            }
        }
    }
    out
}

/// Radial delta (`dpos0`) — pyFAI `delta_array(unit)` for a radial space:
/// per pixel, `max_c |f64(corner_radial_f32[c]) − center[pixel]|`.
///
/// `corner` is the f32 `(npix,4,2)` from [`corner_array_f32`]; `center` is the
/// f64 `center_array(unit, scale=False)` (length npix). The f32 corner radial
/// widens to f64 before the subtraction, matching numpy's `f32 − f64 → f64`.
pub fn delta_radial(corner: &[f32], center: &[f64]) -> Vec<f64> {
    let npix = center.len();
    assert_eq!(
        corner.len(),
        npix * 8,
        "corner must be (npix,4,2) flattened"
    );
    (0..npix)
        .into_par_iter()
        .map(|p| {
            let c = center[p];
            let mut m = (corner[p * 8] as f64 - c).abs();
            for k in 1..4 {
                let cr = corner[(p * 4 + k) * 2] as f64; // [...,0] radial component
                m = m.max((cr - c).abs());
            }
            m
        })
        .collect()
}

/// Chi delta (`dpos1`) — pyFAI `delta_array("chi_rad")` (space `chi_delta`):
/// per pixel, `max_c min(δ, 2π−δ)` where `δ = |f64(corner_chi_f32[c]) − center|`.
///
/// `corner` is the f32 `(npix,4,2)` whose `[...,1]` component holds chi (rad);
/// `center` is the f64 `center_array(CHI_RAD, scale=False)`. The `min(δ, 2π−δ)`
/// folds the wrap-around so a corner straddling the chi discontinuity yields the
/// short arc (`geometry/core.py`: `numpy.minimum(delta, TWO_PI - delta)`).
pub fn delta_chi(corner: &[f32], center: &[f64]) -> Vec<f64> {
    let npix = center.len();
    assert_eq!(
        corner.len(),
        npix * 8,
        "corner must be (npix,4,2) flattened"
    );
    (0..npix)
        .into_par_iter()
        .map(|p| {
            let c = center[p];
            let wrap = |k: usize| -> f64 {
                let cc = corner[(p * 4 + k) * 2 + 1] as f64; // [...,1] chi component
                let d = (cc - c).abs();
                d.min(TWO_PI - d)
            };
            let mut m = wrap(0);
            for k in 1..4 {
                m = m.max(wrap(k));
            }
            m
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A flat 2×2-pixel grid (3×3 corner grid) on the beam axis: with no rotation
    /// and the PONI at the centre, `r`-space corners are the pixel-corner radii.
    #[test]
    fn delta_radial_is_max_corner_minus_center() {
        // center at r=0; corners at r in {0,1,2,3} -> delta = max = 3.
        let center = vec![0.0f64];
        // (npix=1, 4 corners, 2 comps) radial in slot 0.
        let corner = vec![
            0.0f32, 0.0, // c0 radial=0
            1.0, 0.0, // c1 radial=1
            2.0, 0.0, // c2 radial=2
            3.0, 0.0, // c3 radial=3
        ];
        assert_eq!(delta_radial(&corner, &center), vec![3.0]);
    }

    /// chi delta folds the wrap: a corner at +π−ε and center at −π+ε are 2ε apart
    /// the short way, not 2π−2ε.
    #[test]
    fn delta_chi_takes_the_short_arc() {
        let eps = 0.01f64;
        let center = vec![-PI + eps];
        let near = (PI - eps) as f32; // straddles the discontinuity
        let corner = vec![0.0f32, near, 0.0, near, 0.0, near, 0.0, near];
        let got = delta_chi(&corner, &center)[0];
        // short arc = 2π − (|chi − center|); base `expected` on the f32-rounded
        // corner chi (delta_chi widens the f32 corner to f64, so the f32 rounding
        // is part of the computed value, not error to tolerate).
        let raw = ((near as f64) - (-PI + eps)).abs();
        let expected = raw.min(TWO_PI - raw);
        assert!(
            (got - expected).abs() < 1e-12,
            "got {got}, expected {expected}"
        );
        assert!(got < 0.05, "short arc should be ~2eps, got {got}");
    }
}
