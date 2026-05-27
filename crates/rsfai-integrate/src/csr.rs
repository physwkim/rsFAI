//! Port of pyFAI's bbox→CSR 1D path: `calc_boundaries` + `SplitBBoxIntegrator.
//! calc_lut_1d` (`ext/splitBBox_common.pyx`) for the **build**, and
//! `CsrIntegrator.integrate_ng` (`ext/CSR_common.pxi`) for the **apply** — the
//! `("bbox", "csr", "cython")` integration path. This is the **Tier-A** gate for
//! the CSR engine: built `(data, indices, indptr)` and every applied output
//! field must be bit-exact vs golden, fed identical inputs.
//!
//! ## What the engine bins/builds on
//!
//! Both build and apply work in **unscaled** radial space: `pos0 =
//! center_array(unit, scale=False)`, `delta_pos0 = delta_array(unit,
//! scale=False)`. The reported position is `bin_centers * unit.scale` (a single
//! f64 multiply applied by the caller). For 2th_deg `unit.scale ≈ 57.296`.
//!
//! ## dtype contract (decisive for bit-exactness)
//!
//! - positions / boundaries / bin centers: f64 (`position_t`).
//! - CSR coefficients (`data`): f32 (`data_t`) — the split fractions are
//!   computed in f64 (`inv_area * delta_left`, …) then **downcast to f32** at
//!   insertion (`SparseBuilder.cinsert` takes a `float32_t` coef).
//! - per-bin CSR entry order = **pixel insertion order = ascending pixel index**
//!   (the builder appends; pixels are inserted in raster order). Verified
//!   against the golden `csr_indices`.
//! - apply accumulators are f64 (`acc_t`); every per-pixel value (`coef`, `sig`,
//!   `var`, `norm`, `count`) is promoted to f64 **before** the arithmetic. The
//!   `sum_*` outputs stay f64 (no downcast — unlike the histogram engine);
//!   `intensity`/`std`/`sem` are downcast to f32. `std`/`sem` use libc double
//!   `sqrt` on f64 operands, then downcast.

use rsfai_core::dtype::{calc_upper_bound, AccT, DataT, ErrorModel, IndexT, PositionT};

use crate::histogram::{numpy_linspace, Integrate2d};

/// A built CSR sparse matrix (bin-major), matching pyFAI's `(data, indices,
/// indptr)` LUT tuple. For output bin `b`, the entries `indptr[b]..indptr[b+1]`
/// give `(indices[k], data[k])` — contributing pixel index and overlap
/// coefficient.
#[derive(Debug, Clone, PartialEq)]
pub struct Csr {
    /// Overlap coefficients (`data_t`, f32).
    pub data: Vec<DataT>,
    /// Pixel indices (`index_t`, i32).
    pub indices: Vec<IndexT>,
    /// Bin pointers, length `nbin + 1` (`index_t`, i32).
    pub indptr: Vec<IndexT>,
}

/// The `Integrate1dtpl` fields produced by the CSR apply. The `sum_*` fields are
/// **f64** (the raw `acc_t` accumulators pyFAI exposes); `position` is f64
/// (unscaled — multiply by `unit.scale`); `intensity`/`sigma`/`std`/`sem` are
/// f32 (downcast in the reduction).
#[derive(Debug, Clone, PartialEq)]
pub struct CsrIntegrate1d {
    /// Unscaled radial bin centers (multiply by `unit.scale` for the report).
    pub position: Vec<PositionT>,
    /// Average intensity `signal / normalization` (f32), or `empty`.
    pub intensity: Vec<DataT>,
    /// Standard error on the mean (= `sem`; f32). `empty` unless an error model.
    pub sigma: Vec<DataT>,
    /// Sum of `coef * signal` (f64).
    pub sum_signal: Vec<AccT>,
    /// Sum of `coef² * variance` (f64).
    pub sum_variance: Vec<AccT>,
    /// Sum of `coef * norm` (f64).
    pub sum_normalization: Vec<AccT>,
    /// Sum of `coef * count` (f64).
    pub count: Vec<AccT>,
    /// Propagated std `sqrt(var / norm²)` (f32). `empty` unless an error model.
    pub std: Vec<DataT>,
    /// Standard error on the mean `sqrt(var) / norm` (f32). `empty` otherwise.
    pub sem: Vec<DataT>,
    /// Sum of `(coef * norm)²` (f64).
    pub sum_norm_sq: Vec<AccT>,
}

/// Radial boundaries in bbox mode (1D, no azimuth): the `(pos0_min, pos0_maxin)`
/// fold of `calc_boundaries`. `delta_pos0` is the per-pixel half-width (max
/// center-corner distance); `None` disables splitting. `allow_pos0_neg = false`
/// clamps both ends to `>= 0`. The caller applies [`calc_upper_bound`] to
/// `pos0_maxin` to get `pos0_max`.
///
/// pyFAI seeds the fold with the first unmasked *center*; since a center always
/// lies within `[c0-d0, c0+d0]`, the `±INF` seed used here yields the identical
/// fold result (verified against the golden boundaries).
fn calc_boundaries_1d(
    pos0: &[PositionT],
    delta_pos0: Option<&[PositionT]>,
    mask: Option<&[i8]>,
    allow_pos0_neg: bool,
) -> (PositionT, PositionT) {
    let mut pos0_min = PositionT::INFINITY;
    let mut pos0_max = PositionT::NEG_INFINITY;
    for idx in 0..pos0.len() {
        if let Some(m) = mask {
            if m[idx] != 0 {
                continue;
            }
        }
        let c0 = pos0[idx];
        let d0 = delta_pos0.map_or(0.0, |d| d[idx]);
        pos0_max = pos0_max.max(c0 + d0);
        pos0_min = pos0_min.min(c0 - d0);
    }
    if !allow_pos0_neg {
        pos0_min = pos0_min.max(0.0);
        pos0_max = pos0_max.max(0.0);
    }
    (pos0_min, pos0_max)
}

/// Build the bbox CSR matrix and the (unscaled) bin centers — port of
/// `SplitBBoxIntegrator.__init__` + `calc_lut_1d`. `pos0`/`delta_pos0` are the
/// unscaled radial center / half-width per pixel; masked pixels (`mask[i] != 0`)
/// are skipped. Coefficients are computed in f64 and downcast to f32.
pub fn build_bbox_csr_1d(
    pos0: &[PositionT],
    delta_pos0: Option<&[PositionT]>,
    mask: Option<&[i8]>,
    bins: usize,
    allow_pos0_neg: bool,
) -> (Csr, Vec<PositionT>) {
    assert!(bins >= 1, "bins must be >= 1");
    let size = pos0.len();
    if let Some(d) = delta_pos0 {
        assert_eq!(d.len(), size, "delta_pos0 length mismatch");
    }
    if let Some(m) = mask {
        assert_eq!(m.len(), size, "mask length mismatch");
    }

    let (pos0_min, pos0_maxin) = calc_boundaries_1d(pos0, delta_pos0, mask, allow_pos0_neg);
    let pos0_max = calc_upper_bound(pos0_maxin);
    let delta = (pos0_max - pos0_min) / (bins as PositionT);

    // Per-bin entry lists; pixels inserted in ascending index order, so each
    // bin's list stays ascending (matches the SparseBuilder CSR ordering).
    let mut bin_idx: Vec<Vec<IndexT>> = vec![Vec::new(); bins];
    let mut bin_coef: Vec<Vec<DataT>> = vec![Vec::new(); bins];
    let bins_i = bins as i64;

    for idx in 0..size {
        if let Some(m) = mask {
            if m[idx] != 0 {
                continue;
            }
        }
        let c0 = pos0[idx];
        let d0 = delta_pos0.map_or(0.0, |d| d[idx]);
        let min0 = c0 - d0;
        let max0 = c0 + d0;

        let fbin0_min = (min0 - pos0_min) / delta; // get_bin_number
        let fbin0_max = (max0 - pos0_min) / delta;
        let mut bin0_min = fbin0_min as i64; // <Py_ssize_t>: trunc toward zero
        let mut bin0_max = fbin0_max as i64;

        if bin0_max < 0 || bin0_min >= bins_i {
            continue;
        }
        bin0_max = bin0_max.min(bins_i - 1);
        bin0_min = bin0_min.max(0);

        let i = idx as IndexT;
        if bin0_min == bin0_max {
            let b = bin0_min as usize;
            bin_idx[b].push(i);
            bin_coef[b].push(1.0);
        } else {
            // Pixel splitting: weight each bin by its overlap fraction.
            let inv_area = 1.0 / (fbin0_max - fbin0_min);
            let delta_left = (bin0_min + 1) as PositionT - fbin0_min;
            let bmin = bin0_min as usize;
            bin_idx[bmin].push(i);
            bin_coef[bmin].push((inv_area * delta_left) as DataT);

            let delta_right = fbin0_max - bin0_max as PositionT;
            let bmax = bin0_max as usize;
            bin_idx[bmax].push(i);
            bin_coef[bmax].push((inv_area * delta_right) as DataT);

            for b in (bin0_min + 1)..bin0_max {
                let bu = b as usize;
                bin_idx[bu].push(i);
                bin_coef[bu].push(inv_area as DataT);
            }
        }
    }

    let bin_centers = numpy_linspace(pos0_min + 0.5 * delta, pos0_max - 0.5 * delta, bins);
    (flatten_csr(&bin_idx, &bin_coef), bin_centers)
}

/// Flatten per-bin `(index, coef)` lists into a bin-major CSR matrix. Within each
/// bin the entries keep insertion order — and since pixels are inserted in
/// ascending index order, that order is ascending pixel index, matching pyFAI's
/// `SparseBuilder.to_csr()` output.
fn flatten_csr(bin_idx: &[Vec<IndexT>], bin_coef: &[Vec<DataT>]) -> Csr {
    let bins = bin_idx.len();
    let mut indptr = vec![0 as IndexT; bins + 1];
    let nnz: usize = bin_idx.iter().map(|v| v.len()).sum();
    let mut indices = Vec::with_capacity(nnz);
    let mut data = Vec::with_capacity(nnz);
    for b in 0..bins {
        indices.extend_from_slice(&bin_idx[b]);
        data.extend_from_slice(&bin_coef[b]);
        indptr[b + 1] = indptr[b] + bin_idx[b].len() as IndexT;
    }
    Csr {
        data,
        indices,
        indptr,
    }
}

// ---------------------------------------------------------------------------
// BBox pixel splitting (2D): port of `SplitBBoxIntegrator.calc_lut_2d`
// (`ext/splitBBox_common.pyx`) for the build, applied by [`csr_integrate2d`].
// Each pixel's bounding box (centre ± delta in both radial and azimuthal) is
// clipped against the `(radial, azimuthal)` grid; the overlap fractions are the
// CSR coefficients. Output bin index is `bin0·bins1 + bin1` (radial-major).

/// Axis-bounding policy for the 2D bbox boundary fold — the `calc_boundaries`
/// knobs passed as one unit. `allow_pos0_neg = false` clamps the radial axis to
/// `>= 0`; when `pos1_period > 0` the azimuthal axis is clipped to `[-π, π]`
/// (`chi_disc_at_pi`) or `[0, 2π]`, using **f32 π** (`float pi = <float> M_PI`).
/// The 1D CSR setup (`common.py`) does not forward `chiDiscAtPi` to
/// `HistoBBox2d`, so it takes the constructor default `true`; `pos1_period` is
/// `azimuth_unit.period` (acts only as the clip flag — the range is radian ±π).
#[derive(Debug, Clone)]
pub struct Bbox2dBounds {
    /// Allow the radial axis below 0 (false clamps min/max to `>= 0`).
    pub allow_pos0_neg: bool,
    /// Azimuthal discontinuity at π (true) vs 0/2π (false) — sets the clip range.
    pub chi_disc_at_pi: bool,
    /// Azimuthal period; `> 0` turns on the `[-π, π]` clip.
    pub pos1_period: PositionT,
}

/// 2D bbox boundaries — port of `calc_boundaries` with `delta != None`
/// (`do_split = True`): folds each pixel's bounding box `c0±d0` (radial) and
/// `c1±d1` (azimuthal), clamps the radial axis, and clips the azimuthal axis
/// with f32 π (see [`Bbox2dBounds`]). Returns `(pos0_min, pos0_maxin, pos1_min,
/// pos1_maxin)`; the caller applies [`calc_upper_bound`] to the `*_maxin`
/// values. The `±INF` seed yields the same fold as pyFAI's "seed with the first
/// unmasked pixel's box".
fn calc_boundaries_2d(
    pos0: &[PositionT],
    delta_pos0: &[PositionT],
    pos1: &[PositionT],
    delta_pos1: &[PositionT],
    mask: Option<&[i8]>,
    bounds: &Bbox2dBounds,
) -> (PositionT, PositionT, PositionT, PositionT) {
    let mut pos0_min = PositionT::INFINITY;
    let mut pos0_max = PositionT::NEG_INFINITY;
    let mut pos1_min = PositionT::INFINITY;
    let mut pos1_max = PositionT::NEG_INFINITY;
    for idx in 0..pos0.len() {
        if let Some(m) = mask {
            if m[idx] != 0 {
                continue;
            }
        }
        let c0 = pos0[idx];
        let d0 = delta_pos0[idx];
        pos0_max = pos0_max.max(c0 + d0);
        pos0_min = pos0_min.min(c0 - d0);
        let c1 = pos1[idx];
        let d1 = delta_pos1[idx];
        pos1_max = pos1_max.max(c1 + d1);
        pos1_min = pos1_min.min(c1 - d1);
    }
    if !bounds.allow_pos0_neg {
        pos0_min = pos0_min.max(0.0);
        pos0_max = pos0_max.max(0.0);
    }
    if bounds.pos1_period > 0.0 {
        // pyFAI: (2 - chiDiscAtPi) * pi with `pi` an f32 and chiDiscAtPi an int;
        // the product is evaluated in f32 then widened to f64.
        let cd: i32 = if bounds.chi_disc_at_pi { 1 } else { 0 };
        let pi32 = std::f32::consts::PI;
        let max_bound = ((2 - cd) as f32 * pi32) as PositionT;
        let min_bound = (-(cd as f32) * pi32) as PositionT;
        pos1_max = pos1_max.min(max_bound);
        pos1_min = pos1_min.max(min_bound);
    }
    (pos0_min, pos0_max, pos1_min, pos1_max)
}

/// Build the 2D bbox CSR matrix and the (unscaled) radial / radian azimuthal bin
/// centers — port of `SplitBBoxIntegrator.calc_lut_2d`. `pos0`/`delta_pos0` are
/// the unscaled radial center / half-width; `pos1`/`delta_pos1` the azimuthal
/// (chi, radians) center / half-width per pixel. Masked pixels (`mask[i] != 0`)
/// are skipped. `bins` is `(radial, azimuthal)`. The output bin index is
/// `bin0·bins1 + bin1` (radial-major); a pixel confined to one cell gets coef
/// 1.0, otherwise its overlap fractions (computed in f64, downcast to f32) tile
/// the spanned cells. Returns `(csr, bin_centers0, bin_centers1)`.
pub fn build_bbox_csr_2d(
    pos0: &[PositionT],
    delta_pos0: &[PositionT],
    pos1: &[PositionT],
    delta_pos1: &[PositionT],
    mask: Option<&[i8]>,
    bins: (usize, usize),
    bounds: &Bbox2dBounds,
) -> (Csr, Vec<PositionT>, Vec<PositionT>) {
    let (bins0, bins1) = bins;
    assert!(bins0 >= 1 && bins1 >= 1, "bins must be >= 1 in each dim");
    let size = pos0.len();
    assert_eq!(delta_pos0.len(), size, "delta_pos0 length mismatch");
    assert_eq!(pos1.len(), size, "pos1 length mismatch");
    assert_eq!(delta_pos1.len(), size, "delta_pos1 length mismatch");
    if let Some(m) = mask {
        assert_eq!(m.len(), size, "mask length mismatch");
    }

    let (pos0_min, pos0_maxin, pos1_min, pos1_maxin) =
        calc_boundaries_2d(pos0, delta_pos0, pos1, delta_pos1, mask, bounds);
    let pos0_max = calc_upper_bound(pos0_maxin);
    let pos1_max = calc_upper_bound(pos1_maxin);
    let delta0 = (pos0_max - pos0_min) / (bins0 as PositionT);
    let delta1 = (pos1_max - pos1_min) / (bins1 as PositionT);

    let n_out = bins0 * bins1;
    let mut bin_idx: Vec<Vec<IndexT>> = vec![Vec::new(); n_out];
    let mut bin_coef: Vec<Vec<DataT>> = vec![Vec::new(); n_out];
    let b0i = bins0 as i64;
    let b1i = bins1 as i64;

    for idx in 0..size {
        if let Some(m) = mask {
            if m[idx] != 0 {
                continue;
            }
        }
        let c0 = pos0[idx];
        let d0 = delta_pos0[idx];
        let c1 = pos1[idx];
        let d1 = delta_pos1[idx];

        let fbin0_min = (c0 - d0 - pos0_min) / delta0; // get_bin_number
        let fbin0_max = (c0 + d0 - pos0_min) / delta0;
        let fbin1_min = (c1 - d1 - pos1_min) / delta1;
        let fbin1_max = (c1 + d1 - pos1_min) / delta1;

        let mut bin0_min = fbin0_min as i64; // <Py_ssize_t>: trunc toward zero
        let mut bin0_max = fbin0_max as i64;
        let mut bin1_min = fbin1_min as i64;
        let mut bin1_max = fbin1_max as i64;

        if bin0_max < 0 || bin0_min >= b0i || bin1_max < 0 || bin1_min >= b1i {
            continue;
        }
        bin0_max = bin0_max.min(b0i - 1);
        bin0_min = bin0_min.max(0);
        bin1_max = bin1_max.min(b1i - 1);
        bin1_min = bin1_min.max(0);

        let i = idx as IndexT;
        let mut insert = |bin0: i64, bin1: i64, coef: DataT| {
            let b = (bin0 * b1i + bin1) as usize;
            bin_idx[b].push(i);
            bin_coef[b].push(coef);
        };

        if bin0_min == bin0_max {
            if bin1_min == bin1_max {
                // All of the pixel falls in a single bin.
                insert(bin0_min, bin1_min, 1.0);
            } else {
                // Spread over >1 bin in dim 1 only.
                let delta_down = (bin1_min + 1) as PositionT - fbin1_min;
                let delta_up = fbin1_max - bin1_max as PositionT;
                let inv_area = 1.0 / (fbin1_max - fbin1_min);
                insert(bin0_min, bin1_min, (inv_area * delta_down) as DataT);
                insert(bin0_min, bin1_max, (inv_area * delta_up) as DataT);
                for j in (bin1_min + 1)..bin1_max {
                    insert(bin0_min, j, inv_area as DataT);
                }
            }
        } else if bin1_min == bin1_max {
            // Spread over >1 bin in dim 0 only.
            let inv_area = 1.0 / (fbin0_max - fbin0_min);
            let delta_left = (bin0_min + 1) as PositionT - fbin0_min;
            insert(bin0_min, bin1_min, (inv_area * delta_left) as DataT);
            let delta_right = fbin0_max - bin0_max as PositionT;
            insert(bin0_max, bin1_min, (inv_area * delta_right) as DataT);
            for ii in (bin0_min + 1)..bin0_max {
                insert(ii, bin1_min, inv_area as DataT);
            }
        } else {
            // Spread over n bins in dim 0 and m bins in dim 1.
            let delta_left = (bin0_min + 1) as PositionT - fbin0_min;
            let delta_right = fbin0_max - bin0_max as PositionT;
            let delta_down = (bin1_min + 1) as PositionT - fbin1_min;
            let delta_up = fbin1_max - bin1_max as PositionT;
            let inv_area = 1.0 / ((fbin0_max - fbin0_min) * (fbin1_max - fbin1_min));

            insert(
                bin0_min,
                bin1_min,
                (inv_area * delta_left * delta_down) as DataT,
            );
            insert(
                bin0_min,
                bin1_max,
                (inv_area * delta_left * delta_up) as DataT,
            );
            insert(
                bin0_max,
                bin1_min,
                (inv_area * delta_right * delta_down) as DataT,
            );
            insert(
                bin0_max,
                bin1_max,
                (inv_area * delta_right * delta_up) as DataT,
            );

            for ii in (bin0_min + 1)..bin0_max {
                insert(ii, bin1_min, (inv_area * delta_down) as DataT);
                for j in (bin1_min + 1)..bin1_max {
                    insert(ii, j, inv_area as DataT);
                }
                insert(ii, bin1_max, (inv_area * delta_up) as DataT);
            }
            for j in (bin1_min + 1)..bin1_max {
                insert(bin0_min, j, (inv_area * delta_left) as DataT);
                insert(bin0_max, j, (inv_area * delta_right) as DataT);
            }
        }
    }

    let bin_centers0 = numpy_linspace(pos0_min + 0.5 * delta0, pos0_max - 0.5 * delta0, bins0);
    let bin_centers1 = numpy_linspace(pos1_min + 0.5 * delta1, pos1_max - 0.5 * delta1, bins1);
    (flatten_csr(&bin_idx, &bin_coef), bin_centers0, bin_centers1)
}

// ---------------------------------------------------------------------------
// Full pixel splitting (1D): port of `FullSplitIntegrator.calc_lut_1d`
// (`ext/splitpixel_common.pyx`) + the `calc_boundaries`/`_recenter`/`_integrate1d`
// helpers in `ext/regrid_common.pxi`. Each pixel is a quadrilateral with 4
// corners (radial, azimuth); its overlap with each radial bin is the trapezoidal
// area swept by the 4 edges, normalized so the per-pixel coefficients sum to 1.
// The built CSR is applied by the same [`csr_integrate1d`] as the bbox path.

/// Minimum of four values, folded with `<` (matches Cython `min(a,b,c,d)` for
/// the non-NaN corner coordinates the engine sees).
fn min4(a: f64, b: f64, c: f64, d: f64) -> f64 {
    let mut m = a;
    if b < m {
        m = b;
    }
    if c < m {
        m = c;
    }
    if d < m {
        m = d;
    }
    m
}

/// Maximum of four values, folded with `>` (matches Cython `max(a,b,c,d)`).
fn max4(a: f64, b: f64, c: f64, d: f64) -> f64 {
    let mut m = a;
    if b > m {
        m = b;
    }
    if c > m {
        m = c;
    }
    if d > m {
        m = d;
    }
    m
}

/// Approximate signed area of quad ABCD, `0.5·(AC ⨯ BD)` — port of `area4p`.
/// The literal pyFAI precedence is `(0.5 * X) - Y` (the `0.5` scales only the
/// first cross-product term); reproduce it exactly. A positive area flags a
/// pixel straddling the azimuthal discontinuity.
fn area4p(p: &[[f64; 2]; 4]) -> f64 {
    let [[a0, a1], [b0, b1], [c0, c1], [d0, d1]] = *p;
    0.5 * ((c0 - a0) * (d1 - b1)) - ((c1 - a1) * (d0 - b0))
}

/// Shift one azimuth into the canonical period — port of `_recenter_helper`.
fn recenter_helper(azim: f64, period: f64, chi_disc_at_pi: bool) -> f64 {
    if (chi_disc_at_pi && azim < 0.0) || (!chi_disc_at_pi && azim < 0.5 * period) {
        azim + period
    } else {
        azim
    }
}

/// Recenter the azimuthal corner coordinates of one pixel **in place** when the
/// pixel straddles the chi discontinuity (`area4p > 0`) — port of `_recenter`.
/// The radial coordinates (dim 0) are never touched. The 1D LUT only consumes
/// the recentered corners (the signed-area return value is used solely by the
/// 2D path), so this returns nothing.
fn recenter(v8: &mut [[f64; 2]; 4], pos1_period: f64, chi_disc_at_pi: bool) {
    let area = area4p(v8);
    if pos1_period > 0.0 && area > 0.0 {
        let mut na1 = recenter_helper(v8[0][1], pos1_period, chi_disc_at_pi);
        let mut nb1 = recenter_helper(v8[1][1], pos1_period, chi_disc_at_pi);
        let mut nc1 = recenter_helper(v8[2][1], pos1_period, chi_disc_at_pi);
        let mut nd1 = recenter_helper(v8[3][1], pos1_period, chi_disc_at_pi);
        let center1 = 0.25 * (na1 + nb1 + nc1 + nd1);
        let hi = if chi_disc_at_pi {
            0.5 * pos1_period
        } else {
            pos1_period
        };
        if center1 > hi {
            na1 -= pos1_period;
            nb1 -= pos1_period;
            nc1 -= pos1_period;
            nd1 -= pos1_period;
        }
        v8[0][1] = na1;
        v8[1][1] = nb1;
        v8[2][1] = nc1;
        v8[3][1] = nd1;
    }
}

/// Area between `i1` and `i2` under a line of given slope & intercept — port of
/// `_calc_area`: `(i2 - i1)·(0.5·slope·(i2 + i1) + intercept)`.
fn calc_area(i1: f64, i2: f64, slope: f64, intercept: f64) -> f64 {
    (i2 - i1) * (0.5 * slope * (i2 + i1) + intercept)
}

/// Accumulate the trapezoidal area of the segment `(start0,start1)→(stop0,stop1)`
/// into the per-radial-bin `buffer` — port of `_integrate1d`. `dim0` (radial,
/// in bin units) drives the binning; `dim1` (azimuth) sets the line height. The
/// buffer is f32 storage but each contribution is computed in f64 and added as
/// `(old as f64 + area) as f32`, matching C's `float += double`.
fn integrate1d(buffer: &mut [DataT], start0: f64, start1: f64, stop0: f64, stop1: f64) {
    if stop0 == start0 {
        // slope is infinite, area is null: no change to the buffer.
        return;
    }
    let bs = buffer.len() as i64;
    let istart0 = start0.floor() as i64;
    let istop0 = stop0.floor() as i64;
    let slope = (stop1 - start1) / (stop0 - start0);
    let intercept = start1 - slope * start0;

    let mut acc = |i: i64, v: f64| {
        let b = &mut buffer[i as usize];
        *b = (*b as f64 + v) as DataT;
    };

    if bs > istop0 && istop0 == istart0 && istart0 >= 0 {
        acc(istart0, calc_area(start0, stop0, slope, intercept));
    } else if stop0 > start0 {
        if (0.0..bs as f64).contains(&start0) {
            acc(
                istart0,
                calc_area(start0, (start0 + 1.0).floor(), slope, intercept),
            );
        }
        for i in (istart0 + 1).max(0)..istop0.min(bs) {
            acc(i, calc_area(i as f64, (i + 1) as f64, slope, intercept));
        }
        if stop0 < bs as f64 && stop0 >= 0.0 {
            acc(istop0, calc_area(istop0 as f64, stop0, slope, intercept));
        }
    } else {
        if (0.0..bs as f64).contains(&start0) {
            acc(istart0, calc_area(start0, istart0 as f64, slope, intercept));
        }
        let bound = (stop0.floor() as i64).max(-1);
        let mut i = istart0.min(bs) - 1;
        while i > bound {
            acc(i, calc_area((i + 1) as f64, i as f64, slope, intercept));
            i -= 1;
        }
        if stop0 < bs as f64 && stop0 >= 0.0 {
            acc(
                istop0,
                calc_area((stop0 + 1.0).floor(), stop0, slope, intercept),
            );
        }
    }
}

/// Radial boundaries for the full-split path: the `(pos0_min, pos0_maxin)` fold
/// of `calc_boundaries` over the min/max corner radial of every unmasked pixel.
/// The azimuthal bounds are not computed here — the 1D LUT only checks them when
/// an explicit `pos1_range` is given, which this path does not support yet.
/// `allow_pos0_neg = false` clamps both ends to `>= 0`. The `±INF` seed yields
/// the same fold as pyFAI's "seed with the first unmasked corner".
fn calc_boundaries_full_1d(
    corners: &[PositionT],
    mask: Option<&[i8]>,
    allow_pos0_neg: bool,
) -> (PositionT, PositionT) {
    let size = corners.len() / 8;
    let mut pos0_min = PositionT::INFINITY;
    let mut pos0_max = PositionT::NEG_INFINITY;
    for idx in 0..size {
        if let Some(m) = mask {
            if m[idx] != 0 {
                continue;
            }
        }
        let base = idx * 8;
        let (r0, r1, r2, r3) = (
            corners[base],
            corners[base + 2],
            corners[base + 4],
            corners[base + 6],
        );
        let mn = min4(r0, r1, r2, r3);
        let mx = max4(r0, r1, r2, r3);
        if mx > pos0_max {
            pos0_max = mx;
        }
        if mn < pos0_min {
            pos0_min = mn;
        }
    }
    if !allow_pos0_neg {
        pos0_min = pos0_min.max(0.0);
        pos0_max = pos0_max.max(0.0);
    }
    (pos0_min, pos0_max)
}

/// Build the full-split CSR matrix and (unscaled) bin centers — port of
/// `FullSplitCSR_1d.__init__` + `FullSplitIntegrator.calc_lut_1d`. `corners` is
/// the `(npix, 4, 2)` corner array flattened C-order — dim 0 radial (unscaled),
/// dim 1 azimuth (chi, radians) — upcast to f64 before this call (pyFAI stores
/// it as `position_d`). Masked pixels (`mask[i] != 0`) are skipped. A pixel
/// confined to one bin gets coef 1.0; a split pixel's coefficients are its
/// per-bin trapezoidal overlap normalized to sum to 1 (computed in f64, downcast
/// to f32). For the standard radial units this path is invoked with
/// `chi_disc_at_pi = true`, `pos1_period = 2π`, `allow_pos0_neg = false`.
pub fn build_full_csr_1d(
    corners: &[PositionT],
    mask: Option<&[i8]>,
    bins: usize,
    allow_pos0_neg: bool,
    chi_disc_at_pi: bool,
    pos1_period: PositionT,
) -> (Csr, Vec<PositionT>) {
    assert!(bins >= 1, "bins must be >= 1");
    assert_eq!(
        corners.len() % 8,
        0,
        "corners must be (npix, 4, 2) flattened"
    );
    let size = corners.len() / 8;
    if let Some(m) = mask {
        assert_eq!(m.len(), size, "mask length mismatch");
    }

    let (pos0_min, pos0_maxin) = calc_boundaries_full_1d(corners, mask, allow_pos0_neg);
    let pos0_max = calc_upper_bound(pos0_maxin);
    let delta = (pos0_max - pos0_min) / (bins as PositionT);

    let mut bin_idx: Vec<Vec<IndexT>> = vec![Vec::new(); bins];
    let mut bin_coef: Vec<Vec<DataT>> = vec![Vec::new(); bins];
    let mut buffer = vec![0.0 as DataT; bins];
    let bins_i = bins as i64;

    for idx in 0..size {
        if let Some(m) = mask {
            if m[idx] != 0 {
                continue;
            }
        }
        let base = idx * 8;
        let mut v8 = [
            [corners[base], corners[base + 1]],
            [corners[base + 2], corners[base + 3]],
            [corners[base + 4], corners[base + 5]],
            [corners[base + 6], corners[base + 7]],
        ];
        recenter(&mut v8, pos1_period, chi_disc_at_pi);

        // To bin space (radial); azimuth carried through for the line heights.
        let a0 = (v8[0][0] - pos0_min) / delta;
        let a1 = v8[0][1];
        let b0 = (v8[1][0] - pos0_min) / delta;
        let b1 = v8[1][1];
        let c0 = (v8[2][0] - pos0_min) / delta;
        let c1 = v8[2][1];
        let d0 = (v8[3][0] - pos0_min) / delta;
        let d1 = v8[3][1];

        let min0 = min4(a0, b0, c0, d0);
        let max0 = max4(a0, b0, c0, d0);
        if max0 < 0.0 || min0 >= bins as f64 {
            continue;
        }
        // pos1_range is None for this path -> no azimuthal range rejection.

        let mut bin0_min = min0.floor() as i64;
        let mut bin0_max = max0.floor() as i64;

        let i = idx as IndexT;
        if bin0_min == bin0_max {
            let b = bin0_min as usize;
            bin_idx[b].push(i);
            bin_coef[b].push(1.0);
        } else {
            bin0_min = bin0_min.max(0);
            bin0_max = (bin0_max + 1).min(bins_i);

            integrate1d(&mut buffer, a0, a1, b0, b1); // A-B
            integrate1d(&mut buffer, b0, b1, c0, c1); // B-C
            integrate1d(&mut buffer, c0, c1, d0, d1); // C-D
            integrate1d(&mut buffer, d0, d1, a0, a1); // D-A

            let mut sum_area = 0.0f64;
            for b in bin0_min..bin0_max {
                sum_area += buffer[b as usize] as f64;
            }
            let inv_area = 1.0 / sum_area;
            for b in bin0_min..bin0_max {
                let bu = b as usize;
                bin_idx[bu].push(i);
                bin_coef[bu].push((buffer[bu] as f64 * inv_area) as DataT);
            }
            for b in bin0_min..bin0_max {
                buffer[b as usize] = 0.0;
            }
        }
    }

    let bin_centers = numpy_linspace(pos0_min + 0.5 * delta, pos0_max - 0.5 * delta, bins);
    (flatten_csr(&bin_idx, &bin_coef), bin_centers)
}

// ---------------------------------------------------------------------------
// Full pixel splitting (2D): port of `FullSplitIntegrator.calc_lut_2d`
// (`ext/splitpixel_common.pyx`) + the corner-based `calc_boundaries` and the
// `_integrate2d` box clipping in `ext/regrid_common.pxi`. Each pixel
// quadrilateral is clipped into a small `(w0+1)·(w1+1)` box of relative areas
// (the four edges swept by `_integrate2d`); the per-cell area normalized by the
// box total gives the CSR coefficients. Output bin index is `bin0·bins1 + bin1`
// (radial-major), applied by the same [`csr_integrate2d`] as the bbox path.

/// Limit `value` to `[min_val, max_val]` — port of `_clip`.
fn clip(value: f64, min_val: f64, max_val: f64) -> f64 {
    if value < min_val {
        min_val
    } else if value > max_val {
        max_val
    } else {
        value
    }
}

/// 2D full-split boundaries — port of the corner-based `calc_boundaries`
/// (`splitpixel_common.pyx`): folds the min/max corner of every unmasked pixel in
/// both radial (dim 0) and azimuthal (dim 1), clamps the radial axis to `>= 0`
/// when `!allow_pos0_neg`, and (when `pos1_period > 0`) clips the azimuthal axis
/// to `[-π, π]` (`chi_disc_at_pi`) or `[0, 2π]` with **f32 π** (`(2-chiDiscAtPi)·pi`
/// evaluated in f32 then widened). Returns `(pos0_min, pos0_maxin, pos1_min,
/// pos1_maxin)`; the caller applies [`calc_upper_bound`] to the `*_maxin` values.
/// Verified bit-exact against the engine's boundary attributes.
fn calc_boundaries_full_2d(
    corners: &[PositionT],
    mask: Option<&[i8]>,
    allow_pos0_neg: bool,
    chi_disc_at_pi: bool,
    pos1_period: PositionT,
) -> (PositionT, PositionT, PositionT, PositionT) {
    let size = corners.len() / 8;
    let mut pos0_min = PositionT::INFINITY;
    let mut pos0_max = PositionT::NEG_INFINITY;
    let mut pos1_min = PositionT::INFINITY;
    let mut pos1_max = PositionT::NEG_INFINITY;
    for idx in 0..size {
        if let Some(m) = mask {
            if m[idx] != 0 {
                continue;
            }
        }
        let base = idx * 8;
        let (r0, r1, r2, r3) = (
            corners[base],
            corners[base + 2],
            corners[base + 4],
            corners[base + 6],
        );
        let (z0, z1, z2, z3) = (
            corners[base + 1],
            corners[base + 3],
            corners[base + 5],
            corners[base + 7],
        );
        pos0_max = pos0_max.max(max4(r0, r1, r2, r3));
        pos0_min = pos0_min.min(min4(r0, r1, r2, r3));
        pos1_max = pos1_max.max(max4(z0, z1, z2, z3));
        pos1_min = pos1_min.min(min4(z0, z1, z2, z3));
    }
    if !allow_pos0_neg {
        pos0_min = pos0_min.max(0.0);
        pos0_max = pos0_max.max(0.0);
    }
    if pos1_period > 0.0 {
        let cd: i32 = if chi_disc_at_pi { 1 } else { 0 };
        let pi32 = std::f32::consts::PI;
        let max_bound = ((2 - cd) as f32 * pi32) as PositionT;
        let min_bound = (-(cd as f32) * pi32) as PositionT;
        pos1_max = pos1_max.min(max_bound);
        pos1_min = pos1_min.max(min_bound);
    }
    (pos0_min, pos0_max, pos1_min, pos1_max)
}

/// DOUBLE (`fuse_1`) specialization of `_calc_area`: all operands f64. Used for
/// the same-unit `(start0, stop0)` and subsection `((double)i, (double)(i±1))`
/// segments, whose operands are `floating` (f64).
fn calc_area_2d_double(i1: f64, i2: f64, slope: f32, intercept: f32) -> f32 {
    calc_area(i1, i2, slope as f64, intercept as f64) as f32
}

/// FLOAT (`fuse_0`) specialization of `_calc_area`, used for the segments bounded
/// by the f32 `P` local (the dP and Pn→B segments) — Cython resolves the fused
/// `_calc_area` per call site, and a call with one `float` operand selects the
/// float specialization. Narrow `i1`,`i2` to f32; `(I2-I1)` and `(I2+I1)` are
/// computed in f32 (both operands float), but `0.5·slope`, the products and
/// `+intercept` promote to f64 (the `0.5` C double literal), with a final narrow
/// to f32. Getting this fused resolution wrong is a ~110-ULP systematic error on
/// ~19% of coefficients (see the rsfai-port memory note).
fn calc_area_2d_float(i1: f64, i2: f64, slope: f32, intercept: f32) -> f32 {
    let i1f = i1 as f32;
    let i2f = i2 as f32;
    let sub = i2f - i1f;
    let s = i2f + i1f;
    let inner = 0.5 * slope as f64 * s as f64 + intercept as f64;
    (sub as f64 * inner) as f32
}

/// Accumulate the area of the segment `(start0,start1)→(stop0,stop1)` into the 2D
/// `box_buf` (row-major, `shape1` columns) — port of `_integrate2d`. The float
/// locals (`slope`, `intercept`, `P`, `dP`, the spread amounts) are f32 while the
/// inputs are f64; the segment area is f64 for same-unit/subsection segments and
/// f32 for segments bounded by the float `P` (see [`calc_area_2d_float`]). Each
/// contribution `box[i,h] += copysign(dA, seg)` promotes the cell to f64, adds the
/// libc-double `copysign` result, and narrows once — not a pure-f32 add.
fn integrate2d(
    box_buf: &mut [DataT],
    shape1: usize,
    start0: f64,
    start1: f64,
    stop0: f64,
    stop1: f64,
) {
    if start0 == stop0 {
        return;
    }
    let slope = ((stop1 - start1) / (stop0 - start0)) as f32;
    let intercept = (stop1 - slope as f64 * stop0) as f32;

    // Distribute |seg| across the azimuthal columns `h` of one box row, in chunks
    // of `da` (the first chunk capped at the remaining area).
    let mut spread = |row: i64, seg: f32, da0: f64| {
        if seg == 0.0 {
            return;
        }
        let mut abs_area = seg.abs();
        let mut da = da0 as f32;
        let mut h: usize = 0;
        while abs_area > 0.0 && h < shape1 {
            if da > abs_area {
                da = abs_area;
                abs_area = -1.0;
            }
            let cell = row as usize * shape1 + h;
            box_buf[cell] = (box_buf[cell] as f64 + (da as f64).copysign(seg as f64)) as DataT;
            abs_area -= da;
            h += 1;
        }
    };

    if start0 < stop0 {
        // Positive contribution.
        let p = start0.ceil() as f32;
        let dp = (p as f64 - start0) as f32;
        if p as f64 > stop0 {
            // start0 and stop0 in the same unit.
            spread(
                start0 as i64,
                calc_area_2d_double(start0, stop0, slope, intercept),
                stop0 - start0,
            );
        } else {
            if dp > 0.0 {
                spread(
                    p as i64 - 1,
                    calc_area_2d_float(start0, p as f64, slope, intercept),
                    dp as f64,
                );
            }
            // Subsection P1->Pn (whole-unit segments).
            let lo = (p as f64).floor() as i64;
            let hi = stop0.floor() as i64;
            for i in lo..hi {
                spread(
                    i,
                    calc_area_2d_double(i as f64, (i + 1) as f64, slope, intercept),
                    1.0,
                );
            }
            // Section Pn->B.
            let p2 = stop0.floor() as f32;
            let dp2 = (stop0 - p2 as f64) as f32;
            if dp2 > 0.0 {
                spread(
                    p2 as i64,
                    calc_area_2d_float(p2 as f64, stop0, slope, intercept),
                    (dp2 as f64).abs(),
                );
            }
        }
    } else {
        // Negative contribution (start0 > stop0; start0 == stop0 returned above).
        let p = start0.floor() as f32;
        if stop0 > p as f64 {
            spread(
                start0 as i64,
                calc_area_2d_double(start0, stop0, slope, intercept),
                start0 - stop0,
            );
        } else {
            let dp = (p as f64 - start0) as f32;
            if dp < 0.0 {
                spread(
                    p as i64,
                    calc_area_2d_float(start0, p as f64, slope, intercept),
                    (dp as f64).abs(),
                );
            }
            // Subsection P1->Pn, descending.
            let mut i = start0 as i64;
            let stop_excl = stop0.ceil() as i64;
            while i > stop_excl {
                spread(
                    i - 1,
                    calc_area_2d_double(i as f64, (i - 1) as f64, slope, intercept),
                    1.0,
                );
                i -= 1;
            }
            // Section Pn->B.
            let p2 = stop0.ceil() as f32;
            let dp2 = (stop0 - p2 as f64) as f32;
            if dp2 < 0.0 {
                spread(
                    stop0 as i64,
                    calc_area_2d_float(p2 as f64, stop0, slope, intercept),
                    (dp2 as f64).abs(),
                );
            }
        }
    }
}

/// Build the 2D full-split CSR matrix and the (unscaled) radial / radian
/// azimuthal bin centers — port of `FullSplitIntegrator.__init__` + `calc_lut_2d`
/// (`ext/splitpixel_common.pyx`). `corners` is the `(npix, 4, 2)` corner array
/// flattened C-order (dim 0 radial unscaled, dim 1 chi radians), upcast to f64.
/// Masked pixels (`mask[i] != 0`) are skipped. `bins` is `(radial, azimuthal)`.
/// Each pixel is clipped into a small box, swept by [`integrate2d`], normalized so
/// its coefficients sum to 1 (computed in f64, downcast to f32); the output bin
/// index is `bin0·bins1 + bin1` (radial-major). Unlike the bbox-2D and full-1D
/// paths, `common.py` **forwards** `chiDiscAtPi` (default `true`) and
/// `pos1_period = unit1.period` (360, degrees — applied to radian azimuths, a
/// pyFAI quirk replicated here) to this path. Returns `(csr, bin_centers0,
/// bin_centers1)`.
pub fn build_full_csr_2d(
    corners: &[PositionT],
    mask: Option<&[i8]>,
    bins: (usize, usize),
    allow_pos0_neg: bool,
    chi_disc_at_pi: bool,
    pos1_period: PositionT,
) -> (Csr, Vec<PositionT>, Vec<PositionT>) {
    let (bins0, bins1) = bins;
    assert!(bins0 >= 1 && bins1 >= 1, "bins must be >= 1 in each dim");
    assert_eq!(
        corners.len() % 8,
        0,
        "corners must be (npix, 4, 2) flattened"
    );
    let size = corners.len() / 8;
    if let Some(m) = mask {
        assert_eq!(m.len(), size, "mask length mismatch");
    }

    let (pos0_min, pos0_maxin, pos1_min, pos1_maxin) =
        calc_boundaries_full_2d(corners, mask, allow_pos0_neg, chi_disc_at_pi, pos1_period);
    let pos0_max = calc_upper_bound(pos0_maxin);
    let pos1_max = calc_upper_bound(pos1_maxin);
    let delta0 = (pos0_max - pos0_min) / (bins0 as PositionT);
    let delta1 = (pos1_max - pos1_min) / (bins1 as PositionT);

    let n_out = bins0 * bins1;
    let mut bin_idx: Vec<Vec<IndexT>> = vec![Vec::new(); n_out];
    let mut bin_coef: Vec<Vec<DataT>> = vec![Vec::new(); n_out];
    let b1i = bins1 as i64;

    for idx in 0..size {
        if let Some(m) = mask {
            if m[idx] != 0 {
                continue;
            }
        }
        let base = idx * 8;
        let mut v8 = [
            [corners[base], corners[base + 1]],
            [corners[base + 2], corners[base + 3]],
            [corners[base + 4], corners[base + 5]],
            [corners[base + 6], corners[base + 7]],
        ];
        recenter(&mut v8, pos1_period, chi_disc_at_pi);
        let (mut a0, mut a1) = (v8[0][0], v8[0][1]);
        let (mut b0, mut b1) = (v8[1][0], v8[1][1]);
        let (mut c0, mut c1) = (v8[2][0], v8[2][1]);
        let (mut d0, mut d1) = (v8[3][0], v8[3][1]);

        let min0 = min4(a0, b0, c0, d0);
        let max0 = max4(a0, b0, c0, d0);
        let min1 = min4(a1, b1, c1, d1);
        let max1 = max4(a1, b1, c1, d1);
        if max0 < pos0_min || min0 > pos0_maxin || max1 < pos1_min || min1 >= pos1_max {
            continue;
        }

        // Switch to bin space (radial dim 0, azimuthal dim 1), clipping into range.
        a0 = (clip(a0, pos0_min, pos0_maxin) - pos0_min) / delta0;
        a1 = (clip(a1, pos1_min, pos1_maxin) - pos1_min) / delta1;
        b0 = (clip(b0, pos0_min, pos0_maxin) - pos0_min) / delta0;
        b1 = (clip(b1, pos1_min, pos1_maxin) - pos1_min) / delta1;
        c0 = (clip(c0, pos0_min, pos0_maxin) - pos0_min) / delta0;
        c1 = (clip(c1, pos1_min, pos1_maxin) - pos1_min) / delta1;
        d0 = (clip(d0, pos0_min, pos0_maxin) - pos0_min) / delta0;
        d1 = (clip(d1, pos1_min, pos1_maxin) - pos1_min) / delta1;

        let min0 = min4(a0, b0, c0, d0);
        let max0 = max4(a0, b0, c0, d0);
        let min1 = min4(a1, b1, c1, d1);
        let max1 = max4(a1, b1, c1, d1);
        let foffset0 = min0.floor();
        let foffset1 = min1.floor();
        let ioffset0 = foffset0 as i64;
        let ioffset1 = foffset1 as i64;
        let w0 = (max0.ceil() - foffset0) as i64;
        let w1 = (max1.ceil() - foffset1) as i64;

        a0 -= foffset0;
        a1 -= foffset1;
        b0 -= foffset0;
        b1 -= foffset1;
        c0 -= foffset0;
        c1 -= foffset1;
        d0 -= foffset0;
        d1 -= foffset1;

        let shape1 = (w1 + 1) as usize;
        let mut box_buf = vec![0.0 as DataT; ((w0 + 1) * (w1 + 1)) as usize];
        // ABCD is anti-trigonometric order: feed the four edges in turn.
        integrate2d(&mut box_buf, shape1, a0, a1, b0, b1);
        integrate2d(&mut box_buf, shape1, b0, b1, c0, c1);
        integrate2d(&mut box_buf, shape1, c0, c1, d0, d1);
        integrate2d(&mut box_buf, shape1, d0, d1, a0, a1);

        let mut sum_area = 0.0f64; // position_t
        for i in 0..w0 {
            for j in 0..w1 {
                sum_area += box_buf[(i * (w1 + 1) + j) as usize] as f64;
            }
        }
        let inv_area = 1.0 / sum_area;
        let pix = idx as IndexT;
        for i in 0..w0 {
            for j in 0..w1 {
                let coef = (box_buf[(i * (w1 + 1) + j) as usize] as f64 * inv_area) as DataT;
                let b = ((ioffset0 + i) * b1i + ioffset1 + j) as usize;
                bin_idx[b].push(pix);
                bin_coef[b].push(coef);
            }
        }
    }

    let bin_centers0 = numpy_linspace(pos0_min + 0.5 * delta0, pos0_max - 0.5 * delta0, bins0);
    let bin_centers1 = numpy_linspace(pos1_min + 0.5 * delta1, pos1_max - 0.5 * delta1, bins1);
    (flatten_csr(&bin_idx, &bin_coef), bin_centers0, bin_centers1)
}

/// The per-bin reduction outputs shared by the 1D and 2D CSR apply — the body of
/// `CsrIntegrator.integrate_ng` before the dimension-specific packaging. All
/// vectors are flat, indexed by output bin (length `indptr.len() - 1`). `sum_*`
/// / `count` are f64 (`acc_t`); `intensity`/`std`/`sem` are f32.
struct CsrReduction {
    sum_signal: Vec<AccT>,
    sum_variance: Vec<AccT>,
    sum_normalization: Vec<AccT>,
    sum_norm_sq: Vec<AccT>,
    count: Vec<AccT>,
    intensity: Vec<DataT>,
    std: Vec<DataT>,
    sem: Vec<DataT>,
}

/// One output bin's reduction outputs, produced independently per bin and then
/// scattered into the flat [`CsrReduction`] arrays.
struct BinReduction {
    sum_signal: AccT,
    sum_variance: AccT,
    sum_normalization: AccT,
    sum_norm_sq: AccT,
    count: AccT,
    intensity: DataT,
    std: DataT,
    sem: DataT,
}

/// The per-output-bin weighted-mean reduction at the heart of
/// `CsrIntegrator.integrate_ng`, dimension-agnostic: it iterates the
/// `indptr.len() - 1` output bins (which is `bins` for 1D and `bins0·bins1` for
/// 2D) and produces flat per-bin arrays. The 1D and 2D entry points differ only
/// in how they package these (1D keeps them flat with a position axis; 2D
/// reshapes to `(bins0, bins1)` and transposes). Every per-pixel value is
/// promoted to f64 before the arithmetic; `sum_*` stay f64, `intensity`/`std`/
/// `sem` are downcast to f32 (`std`/`sem` via libc double `sqrt`). The
/// `acc_norm_sq > 0` guard mirrors pyFAI exactly.
///
/// Parallelized over output bins, and **bit-exact** while parallel: each bin
/// reads only its own CSR row (`indptr[i]..indptr[i+1]`) in ascending entry
/// order and writes only its own slot, so the per-bin accumulation order is
/// identical to the serial code. (pyFAI's `prange` likewise assigns each output
/// bin wholly to one thread — the same partition.)
fn csr_reduce(csr: &Csr, prep: &[DataT], error_model: ErrorModel, empty: DataT) -> CsrReduction {
    use rayon::prelude::*;

    let bins = csr.indptr.len() - 1;
    let do_variance = error_model != ErrorModel::No;

    let per_bin: Vec<BinReduction> = (0..bins)
        .into_par_iter()
        .map(|i| {
            let mut acc_sig: AccT = 0.0;
            let mut acc_var: AccT = 0.0;
            let mut acc_norm: AccT = 0.0;
            let mut acc_norm_sq: AccT = 0.0;
            let mut acc_count: AccT = 0.0;

            let lo = csr.indptr[i] as usize;
            let hi = csr.indptr[i + 1] as usize;
            for j in lo..hi {
                let coef = csr.data[j] as AccT; // data_t -> acc_t
                if coef == 0.0 {
                    continue;
                }
                let idx = csr.indices[j] as usize;
                let sig = prep[4 * idx] as AccT;
                let var = prep[4 * idx + 1] as AccT;
                let norm = prep[4 * idx + 2] as AccT;
                let cnt = prep[4 * idx + 3] as AccT;

                acc_count += coef * cnt;
                match error_model {
                    ErrorModel::Azimuthal => {
                        unimplemented!("azimuthal (Welford) CSR variance not yet ported")
                    }
                    _ => {
                        acc_sig += coef * sig;
                        if do_variance {
                            acc_var += coef * coef * var;
                        }
                        let w = coef * norm;
                        acc_norm += w;
                        acc_norm_sq += w * w;
                    }
                }
            }

            let (intensity, std, sem) = if acc_norm_sq > 0.0 {
                let intensity = (acc_sig / acc_norm) as DataT;
                if do_variance {
                    // libc double sqrt on f64 accumulators, then downcast to f32.
                    let std = (acc_var / acc_norm_sq).sqrt() as DataT;
                    let sem = (acc_var.sqrt() / acc_norm) as DataT;
                    (intensity, std, sem)
                } else {
                    (intensity, empty, empty)
                }
            } else {
                (empty, empty, empty)
            };

            BinReduction {
                sum_signal: acc_sig,
                sum_variance: acc_var,
                sum_normalization: acc_norm,
                sum_norm_sq: acc_norm_sq,
                count: acc_count,
                intensity,
                std,
                sem,
            }
        })
        .collect();

    // Scatter the per-bin results into the flat output arrays (serial, O(bins)).
    let mut sum_signal = Vec::with_capacity(bins);
    let mut sum_variance = Vec::with_capacity(bins);
    let mut sum_normalization = Vec::with_capacity(bins);
    let mut sum_norm_sq = Vec::with_capacity(bins);
    let mut count = Vec::with_capacity(bins);
    let mut intensity = Vec::with_capacity(bins);
    let mut std = Vec::with_capacity(bins);
    let mut sem = Vec::with_capacity(bins);
    for b in per_bin {
        sum_signal.push(b.sum_signal);
        sum_variance.push(b.sum_variance);
        sum_normalization.push(b.sum_normalization);
        sum_norm_sq.push(b.sum_norm_sq);
        count.push(b.count);
        intensity.push(b.intensity);
        std.push(b.std);
        sem.push(b.sem);
    }

    CsrReduction {
        sum_signal,
        sum_variance,
        sum_normalization,
        sum_norm_sq,
        count,
        intensity,
        std,
        sem,
    }
}

/// Apply a CSR matrix to preprocessed rows (1D) — port of `CsrIntegrator.
/// integrate_ng`'s 1D return. `prep` is the flat `[signal, variance, norm,
/// count]`-per-pixel f32 array (the `preproc(..., split_result=4)` output).
/// `bin_centers` are the unscaled centers from [`build_bbox_csr_1d`] /
/// [`build_full_csr_1d`]. `empty` fills bins with no normalization (pyFAI's
/// `self.empty`, default `0.0`).
pub fn csr_integrate1d(
    csr: &Csr,
    prep: &[DataT],
    bin_centers: Vec<PositionT>,
    error_model: ErrorModel,
    empty: DataT,
) -> CsrIntegrate1d {
    let bins = bin_centers.len();
    assert_eq!(csr.indptr.len(), bins + 1, "indptr length must be bins + 1");

    let r = csr_reduce(csr, prep, error_model, empty);
    CsrIntegrate1d {
        position: bin_centers,
        intensity: r.intensity,
        sigma: r.sem.clone(), // Integrate1dtpl position 3 (sigma) == position 9 (sem)
        sum_signal: r.sum_signal,
        sum_variance: r.sum_variance,
        sum_normalization: r.sum_normalization,
        count: r.count,
        std: r.std,
        sem: r.sem,
        sum_norm_sq: r.sum_norm_sq,
    }
}

/// Apply a 2D CSR matrix to preprocessed rows — port of `CsrIntegrator.
/// integrate_ng`'s 2D return. The reduction is identical to the 1D apply; the
/// flat per-bin arrays (output bin `i·bins1 + j`, radial-major) are reshaped to
/// `(bins0, bins1)` and transposed to **(azimuthal, radial)** (pyFAI's
/// `.reshape(self.bins).T`), so cell `(azimuthal j, radial i)` lands at flat
/// index `j·bins0 + i`. `bin_centers0`/`bin_centers1` are the unscaled radial /
/// radian azimuthal centers from [`build_bbox_csr_2d`]; the binned sums are
/// exposed at full f64 (`acc_t`).
pub fn csr_integrate2d(
    csr: &Csr,
    prep: &[DataT],
    bin_centers0: Vec<PositionT>,
    bin_centers1: Vec<PositionT>,
    error_model: ErrorModel,
    empty: DataT,
) -> Integrate2d {
    let bins0 = bin_centers0.len();
    let bins1 = bin_centers1.len();
    assert_eq!(
        csr.indptr.len(),
        bins0 * bins1 + 1,
        "indptr length must be bins0 * bins1 + 1"
    );

    let r = csr_reduce(csr, prep, error_model, empty);

    let n = bins0 * bins1;
    let mut signal = vec![0.0f64; n];
    let mut variance = vec![0.0f64; n];
    let mut normalization = vec![0.0f64; n];
    let mut count = vec![0.0f64; n];
    let mut norm_sq = vec![0.0f64; n];
    let mut intensity = vec![0.0f32; n];
    let mut std = vec![0.0f32; n];
    let mut sem = vec![0.0f32; n];
    for i in 0..bins0 {
        for j in 0..bins1 {
            let b = i * bins1 + j; // reduction index (radial-major)
            let t = j * bins0 + i; // transposed (azimuthal, radial)
            signal[t] = r.sum_signal[b];
            variance[t] = r.sum_variance[b];
            normalization[t] = r.sum_normalization[b];
            count[t] = r.count[b];
            norm_sq[t] = r.sum_norm_sq[b];
            intensity[t] = r.intensity[b];
            std[t] = r.std[b];
            sem[t] = r.sem[b];
        }
    }

    Integrate2d {
        radial: bin_centers0,
        azimuthal: bin_centers1,
        bins: (bins0, bins1),
        intensity,
        sigma: sem.clone(), // Integrate2dtpl position 4 (sigma) == position 10 (sem)
        signal,
        variance,
        normalization,
        count,
        std,
        sem,
        norm_sq,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn single_bin_no_split_has_unit_coef() {
        // Two pixels with no delta -> no splitting -> each lands in one bin with
        // coef 1.0.
        let pos0 = [1.0f64, 2.0];
        let (csr, centers) = build_bbox_csr_1d(&pos0, None, None, 4, false);
        assert_eq!(centers.len(), 4);
        assert_eq!(csr.indptr[0], 0);
        assert_eq!(*csr.indptr.last().unwrap(), 2); // two entries total
        assert!(csr.data.iter().all(|&c| c == 1.0));
    }

    #[test]
    fn split_pixel_coefs_sum_to_one() {
        // One pixel spanning multiple bins: its split coefficients sum to 1.
        let pos0 = [5.0f64];
        let delta = [3.0f64]; // wide -> spans several bins
        let (csr, _) = build_bbox_csr_1d(&pos0, Some(&delta), None, 10, false);
        let s: f32 = csr.data.iter().sum();
        assert!((s - 1.0).abs() < 1e-6, "split coefs sum to ~1, got {s}");
    }

    /// Two pixels as a flat `(2, 4, 2)` corner array: pixel 0 is radially tight
    /// (one bin); pixel 1 is a rectangle spanning radial `[0, 4]` (splits).
    fn two_pixel_corners() -> Vec<f64> {
        vec![
            // pixel 0: A,B,C,D all at radial 0.5 -> single bin.
            0.5, 0.0, 0.5, 0.1, 0.5, 0.1, 0.5, 0.0, //
            // pixel 1: rectangle radial in [0,4], chi in [0,0.5].
            0.0, 0.0, 4.0, 0.0, 4.0, 0.5, 0.0, 0.5,
        ]
    }

    #[test]
    fn full_split_single_bin_pixel_keeps_unit_coef() {
        let corners = two_pixel_corners();
        let (csr, _) = build_full_csr_1d(&corners, None, 8, false, true, std::f64::consts::TAU);
        // Pixel 0 stays within one bin -> exactly one entry, coef 1.0.
        let p0: Vec<f32> = csr
            .indices
            .iter()
            .zip(&csr.data)
            .filter(|(&i, _)| i == 0)
            .map(|(_, &c)| c)
            .collect();
        assert_eq!(p0, vec![1.0]);
    }

    #[test]
    fn full_split_coefs_sum_to_one() {
        let corners = two_pixel_corners();
        let (csr, _) = build_full_csr_1d(&corners, None, 8, false, true, std::f64::consts::TAU);
        // The split pixel's overlap coefficients are normalized to sum to 1.
        let s: f32 = csr
            .indices
            .iter()
            .zip(&csr.data)
            .filter(|(&i, _)| i == 1)
            .map(|(_, &c)| c)
            .sum();
        assert!((s - 1.0).abs() < 1e-5, "split coefs sum to ~1, got {s}");
    }

    #[test]
    fn apply_weighted_mean_and_f64_sums() {
        // Two pixels in bin 0 (no split): signal [10, 20], norm [2, 2].
        // sum_signal = 30, sum_norm = 4, intensity = 7.5.
        let pos0 = [1.0f64, 1.0];
        let (csr, centers) = build_bbox_csr_1d(&pos0, None, None, 3, false);
        let prep = [10.0f32, 0.0, 2.0, 1.0, 20.0, 0.0, 2.0, 1.0];
        let r = csr_integrate1d(&csr, &prep, centers, ErrorModel::No, 0.0);
        assert_eq!(r.sum_signal[0], 30.0);
        assert_eq!(r.sum_normalization[0], 4.0);
        assert_eq!(r.count[0], 2.0);
        assert_eq!(r.intensity[0], 7.5);
        // norm² = (1*2)² + (1*2)² = 8 in f64.
        assert_eq!(r.sum_norm_sq[0], 8.0);
    }
}
