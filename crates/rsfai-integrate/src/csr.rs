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

use crate::histogram::numpy_linspace;

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

/// Apply a CSR matrix to preprocessed rows — port of `CsrIntegrator.
/// integrate_ng`. `prep` is the flat `[signal, variance, norm, count]`-per-pixel
/// f32 array (the `preproc(..., split_result=4)` output). `bin_centers` are the
/// unscaled centers from [`build_bbox_csr_1d`]. `empty` fills bins with no
/// normalization (pyFAI's `self.empty`, default `0.0`).
///
/// Per-bin accumulation over the CSR row is bit-reproducible regardless of
/// threads: pyFAI's `prange` assigns each output bin wholly to one thread.
pub fn csr_integrate1d(
    csr: &Csr,
    prep: &[DataT],
    bin_centers: Vec<PositionT>,
    error_model: ErrorModel,
    empty: DataT,
) -> CsrIntegrate1d {
    let bins = bin_centers.len();
    assert_eq!(csr.indptr.len(), bins + 1, "indptr length must be bins + 1");

    let do_variance = error_model != ErrorModel::No;

    let mut sum_signal = vec![0.0f64; bins];
    let mut sum_variance = vec![0.0f64; bins];
    let mut sum_normalization = vec![0.0f64; bins];
    let mut sum_norm_sq = vec![0.0f64; bins];
    let mut count = vec![0.0f64; bins];
    let mut intensity = vec![0.0f32; bins];
    let mut std = vec![0.0f32; bins];
    let mut sem = vec![0.0f32; bins];

    for i in 0..bins {
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

        sum_signal[i] = acc_sig;
        sum_variance[i] = acc_var;
        sum_normalization[i] = acc_norm;
        sum_norm_sq[i] = acc_norm_sq;
        count[i] = acc_count;
        if acc_norm_sq > 0.0 {
            intensity[i] = (acc_sig / acc_norm) as DataT;
            if do_variance {
                // libc double sqrt on f64 accumulators, then downcast to f32.
                std[i] = (acc_var / acc_norm_sq).sqrt() as DataT;
                sem[i] = (acc_var.sqrt() / acc_norm) as DataT;
            } else {
                std[i] = empty;
                sem[i] = empty;
            }
        } else {
            intensity[i] = empty;
            std[i] = empty;
            sem[i] = empty;
        }
    }

    CsrIntegrate1d {
        position: bin_centers,
        intensity,
        sigma: sem.clone(), // Integrate1dtpl position 3 (sigma) == position 9 (sem)
        sum_signal,
        sum_variance,
        sum_normalization,
        count,
        std,
        sem,
        sum_norm_sq,
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
