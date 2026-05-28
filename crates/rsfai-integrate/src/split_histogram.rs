//! Direct-split histogram engines — the `(_, "histogram", "cython")` paths with
//! pixel splitting: the **bbox** split (`splitBBox.histoBBox1d_engine` /
//! `histoBBox2d_engine`), the **full** pixel split
//! (`splitPixel.fullSplit1D_engine` / `fullSplit2D_engine`), and the **pseudo**
//! split (`splitPixel.pseudoSplit2D_engine`, 2D only).
//!
//! All four reuse the boundary fold and per-pixel overlap of the matching CSR
//! build — bbox via [`calc_boundaries_1d`]/[`calc_boundaries_2d`] + the bbox
//! fractions, full via [`calc_boundaries_full_1d`]/[`calc_boundaries_full_2d`] +
//! the corner-clipping [`recenter`]/[`integrate1d`]/[`integrate2d`] machinery —
//! but accumulate the fractional contributions **directly into bins** (no sparse
//! matrix) via pyFAI's `update_1d/2d_accumulator`. Two consequences for the bbox
//! engines' bit-exactness vs the CSR path, both reproduced verbatim below:
//!
//!   * The split coefficient is computed in the engine's own arithmetic: the 1D
//!     engine casts `(bin + 1)` to **f32** (`<float>`) in `delta_left/right`,
//!     while the 2D engine casts to **f64** (`<position_t>`); and the coef stays
//!     `acc_t` (f64) into the accumulator — it is **not** rounded through f32 the
//!     way the CSR `data` array is.
//!   * `update_1d_accumulator` has **no error-model fork**: `sum_nrm2 += w²` with
//!     `w = weight·norm` always in f64 (the no-split `histogram.pyx` engine forks
//!     to an f32 `norm·norm` for `error_model == 0`). `update_2d_accumulator`
//!     keeps the f32 `norm·norm` but multiplies by `weight²` in f64.
//!
//! The full-split engines compute the same normalized overlap buffer as the
//! full-CSR build, then scatter `buffer[bin]·inv_area` (f64) into the accumulator
//! instead of storing an f32 CSR coef — and they carry the engine's own range
//! checks (the 2D engine rejects on `min1 > pos1_maxin`, NOT the CSR build's
//! `min1 >= pos1_max`), so the per-pixel orchestration is ported here rather than
//! reusing the build loop.
//!
//! The per-pixel scatter (bbox and full) runs **serially** in pixel-index order —
//! bit-for-bit the same sequence of f64 adds pyFAI performs single-threaded
//! (golden is generated with `OMP_NUM_THREADS=1`). The fractional split
//! coefficients make each bin's f64 sum order-dependent (the sums do not fit in
//! <53 bits, unlike the integer-valued no-split histogram sums), so a parallel
//! fold/reduce would reorder the adds and diverge by a few ULP; the serial scatter
//! is the bit-exact construction. The final reduction guards on **count** and
//! exposes the binned sums at f64 — it matches the CSR / 2D-histogram container,
//! not the f32 no-split 1D histogram.

use rsfai_core::dtype::{calc_upper_bound, AccT, DataT, ErrorModel, PositionT};

use crate::csr::{
    calc_boundaries_1d, calc_boundaries_2d, calc_boundaries_full_1d, calc_boundaries_full_2d, clip,
    integrate1d, integrate2d, max4, min4, recenter, Bbox2dBounds, CsrIntegrate1d,
};
use crate::histogram::{numpy_linspace, reduce_2d, Integrate2d};

/// Port of `update_1d_accumulator` (`regrid_common.pxi`): add the preproc tuple
/// weighted by `weight` into one bin row. `s/v/n/c` are f32 (`preproc_t =
/// data_t`); `weight` is f64.
///
/// Non-azimuthal branch — matches C promotion exactly: `sig += signal·weight`,
/// `var += variance·weight²`, `nrm += weight·norm`, `nrm2 += (weight·norm)²`,
/// `cnt += count·weight`, all accumulated in f64.
///
/// Azimuthal (error_model==3) branch — the weighted Welford update
/// ([`crate::azimuthal::azimuthal_step`]): `omega_b = weight·norm`,
/// `sig_inc = weight·signal`, and `b = value.signal/value.norm` is the **f32**
/// division promoted (pyFAI divides the `preproc_t`/`data_t` fields). A zero-norm
/// contribution that is not the bin's first is skipped (pyFAI's `if value.norm`).
/// The direct-split histograms accumulate serially, so this is bit-exact.
#[inline]
fn accumulate_1d(
    row: &mut [AccT; 5],
    s: DataT,
    v: DataT,
    n: DataT,
    c: DataT,
    weight: AccT,
    error_model: ErrorModel,
) {
    if error_model == ErrorModel::Azimuthal {
        if row[4] <= 0.0 || n != 0.0 {
            let omega_b = weight * n as AccT;
            let sig_inc = weight * s as AccT;
            let b = (s / n) as AccT; // f32 division, promoted (value.signal/value.norm)
            let [sum_sig, sum_var, sum_norm, _cnt, sum_norm_sq] = &mut *row;
            crate::azimuthal::azimuthal_step(
                sum_sig,
                sum_var,
                sum_norm,
                sum_norm_sq,
                omega_b,
                sig_inc,
                b,
            );
        }
        row[3] += (c as AccT) * weight;
        return;
    }
    let w = weight * n as AccT;
    let w2 = w * w;
    let weight2 = weight * weight;
    row[0] += (s as AccT) * weight;
    row[1] += (v as AccT) * weight2;
    row[2] += w;
    row[4] += w2;
    row[3] += (c as AccT) * weight;
}

/// Port of `update_2d_accumulator` (`regrid_common.pxi`): like [`accumulate_1d`]
/// but with no error-model fork and norm² kept as the **f32** product
/// `norm·norm` scaled by `weight²` (f64). `cell` is the flat `(radial, azimuthal)`
/// index `bin0·bins1 + bin1`.
#[inline]
fn accumulate_2d(
    acc: &mut [[AccT; 5]],
    cell: usize,
    s: DataT,
    v: DataT,
    n: DataT,
    c: DataT,
    weight: AccT,
) {
    let w2 = weight * weight;
    let row = &mut acc[cell];
    row[0] += (s as AccT) * weight;
    row[1] += (v as AccT) * w2;
    row[2] += (n as AccT) * weight;
    row[3] += (c as AccT) * weight;
    row[4] += ((n * n) as AccT) * w2; // norm·norm in f32, then · weight² in f64
}

/// Port of `splitBBox.histoBBox1d_engine`: 1D direct-split bbox histogram. Each
/// unmasked pixel's bounding box `[c0-d0, c0+d0]` is distributed across the radial
/// bins it overlaps (overlap fraction as the weight) and accumulated directly.
/// `pos0`/`delta_pos0` are the unscaled radial center / half-width; `prep` is the
/// flat `[signal, variance, norm, count]`-per-pixel f32 array (masked/invalid
/// pixels carry zeroed rows, so splitting them contributes nothing). Returns the
/// [`CsrIntegrate1d`] field set (f64 sums + f32 derived).
#[allow(clippy::too_many_arguments)]
pub fn histogram1d_bbox(
    pos0: &[PositionT],
    delta_pos0: &[PositionT],
    prep: &[DataT],
    mask: Option<&[i8]>,
    npt: usize,
    error_model: ErrorModel,
    empty: DataT,
    allow_pos0_neg: bool,
) -> CsrIntegrate1d {
    assert!(npt > 1, "bins must be > 1");
    let size = pos0.len();
    assert_eq!(delta_pos0.len(), size, "delta_pos0 length mismatch");
    assert_eq!(prep.len(), 4 * size, "prep length must be 4 * pos0.len()");
    if let Some(m) = mask {
        assert_eq!(m.len(), size, "mask length mismatch");
    }

    let (pos0_min, pos0_maxin) = calc_boundaries_1d(pos0, Some(delta_pos0), mask, allow_pos0_neg);
    let pos0_max = calc_upper_bound(pos0_maxin);
    let delta = (pos0_max - pos0_min) / (npt as PositionT);

    let accumulate = |acc: &mut [[AccT; 5]], idx: usize| {
        if let Some(m) = mask {
            if m[idx] != 0 {
                return;
            }
        }
        let s = prep[4 * idx];
        // Direct-split engines re-run preproc internally → hybrid variance 0
        // (see crate::internal_preproc_variance).
        let v = crate::internal_preproc_variance(error_model, prep[4 * idx + 1]);
        let n = prep[4 * idx + 2];
        let c = prep[4 * idx + 3];
        let c0 = pos0[idx];
        let d0 = delta_pos0[idx];
        let min0 = c0 - d0;
        let max0 = c0 + d0;
        // pyFAI's float range skip (histoBBox1d_engine), NOT the integer bin check
        // the CSR build / 2D engine use.
        if max0 < pos0_min || min0 > pos0_maxin {
            return;
        }
        let fbin0_min = (min0 - pos0_min) / delta; // get_bin_number
        let fbin0_max = (max0 - pos0_min) / delta;
        let bin0_max = (fbin0_max as i64).min(npt as i64 - 1);
        let bin0_min = (fbin0_min as i64).max(0);
        if bin0_min == bin0_max {
            accumulate_1d(&mut acc[bin0_min as usize], s, v, n, c, 1.0, error_model);
        } else {
            let inv_area = 1.0 / (fbin0_max - fbin0_min);
            // <float> cast of the bin index (engine-specific; the CSR build casts
            // to f64). fbin0_min/max stay f64, so the subtraction is in f64.
            let delta_left = (bin0_min + 1) as f32 as PositionT - fbin0_min;
            let delta_right = fbin0_max - bin0_max as f32 as PositionT;
            accumulate_1d(
                &mut acc[bin0_min as usize],
                s,
                v,
                n,
                c,
                inv_area * delta_left,
                error_model,
            );
            accumulate_1d(
                &mut acc[bin0_max as usize],
                s,
                v,
                n,
                c,
                inv_area * delta_right,
                error_model,
            );
            for b in (bin0_min + 1)..bin0_max {
                accumulate_1d(&mut acc[b as usize], s, v, n, c, inv_area, error_model);
            }
        }
    };

    // Serial scatter in pixel-index order, matching pyFAI's single-threaded
    // accumulation exactly: the fractional bbox coefficients make each bin's f64
    // sum order-dependent, so this is the bit-exact construction (a parallel
    // fold/reduce would reorder the adds and diverge by a few ULP).
    let mut res = vec![[0.0f64; 5]; npt];
    for idx in 0..size {
        accumulate(&mut res, idx);
    }

    let position = numpy_linspace(pos0_min + 0.5 * delta, pos0_max - 0.5 * delta, npt);
    reduce_1d(&res, position, error_model, empty)
}

/// Final per-bin 1D reduction shared by `histoBBox1d_engine` and
/// `fullSplit1D_engine`: guard on **count** (`cnt != 0`, = pyFAI's `if cnt:` for
/// the non-negative counts here), compute `intensity = sig/nrm`, `sem =
/// sqrt(var)/nrm`, `std = sqrt(var/nrm²)` in f64 (libc double sqrt) then downcast
/// to f32; the binned sums stay f64. `std`/`sem` are touched only when an error
/// model is set (both bounded by `if error_model:` in pyFAI). Returns the
/// [`CsrIntegrate1d`] container (f64 sums, NOT the f32 no-split histogram).
fn reduce_1d(
    res: &[[AccT; 5]],
    position: Vec<PositionT>,
    error_model: ErrorModel,
    empty: DataT,
) -> CsrIntegrate1d {
    let npt = res.len();
    let do_variance = error_model != ErrorModel::No;
    let mut sum_signal = vec![0.0f64; npt];
    let mut sum_variance = vec![0.0f64; npt];
    let mut sum_normalization = vec![0.0f64; npt];
    let mut count = vec![0.0f64; npt];
    let mut sum_norm_sq = vec![0.0f64; npt];
    let mut intensity = vec![0.0f32; npt];
    let mut std = vec![0.0f32; npt];
    let mut sem = vec![0.0f32; npt];
    for i in 0..npt {
        let (sig, var, nrm, cnt, nrm2) = (res[i][0], res[i][1], res[i][2], res[i][3], res[i][4]);
        sum_signal[i] = sig;
        sum_variance[i] = var;
        sum_normalization[i] = nrm;
        count[i] = cnt;
        sum_norm_sq[i] = nrm2;
        if cnt != 0.0 {
            intensity[i] = (sig / nrm) as DataT;
        } else {
            intensity[i] = empty;
        }
        if do_variance {
            if cnt != 0.0 {
                sem[i] = (var.sqrt() / nrm) as DataT;
                std[i] = (var / nrm2).sqrt() as DataT;
            } else {
                sem[i] = empty;
                std[i] = empty;
            }
        }
    }
    CsrIntegrate1d {
        position,
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

/// Port of `splitBBox.histoBBox2d_engine`: 2D direct-split bbox histogram. Each
/// unmasked pixel's bounding box (center ± delta in both radial and azimuthal) is
/// distributed across the `(radial, azimuthal)` cells it overlaps via the 4-branch
/// split (single cell; spread in one axis; n×m), accumulated directly. `bounds`
/// supplies the azimuthal clip / radial sign policy (same [`Bbox2dBounds`] as the
/// 2D bbox-CSR build). Returns the [`Integrate2d`] field set via the shared
/// [`reduce_2d`] (count guard, transpose to (azimuthal, radial), f64 sums).
#[allow(clippy::too_many_arguments)]
pub fn histogram2d_bbox(
    pos0: &[PositionT],
    delta_pos0: &[PositionT],
    pos1: &[PositionT],
    delta_pos1: &[PositionT],
    prep: &[DataT],
    mask: Option<&[i8]>,
    bins: (usize, usize),
    bounds: &Bbox2dBounds,
    error_model: ErrorModel,
    empty: DataT,
) -> Integrate2d {
    let (bins0, bins1) = bins;
    assert!(bins0 >= 1 && bins1 >= 1, "bins must be >= 1 in each dim");
    let size = pos0.len();
    assert_eq!(delta_pos0.len(), size, "delta_pos0 length mismatch");
    assert_eq!(pos1.len(), size, "pos1 length mismatch");
    assert_eq!(delta_pos1.len(), size, "delta_pos1 length mismatch");
    assert_eq!(prep.len(), 4 * size, "prep length must be 4 * pos0.len()");
    if let Some(m) = mask {
        assert_eq!(m.len(), size, "mask length mismatch");
    }

    let (pos0_min, pos0_maxin, pos1_min, pos1_maxin) =
        calc_boundaries_2d(pos0, Some(delta_pos0), pos1, Some(delta_pos1), mask, bounds);
    let pos0_max = calc_upper_bound(pos0_maxin);
    let pos1_max = calc_upper_bound(pos1_maxin);
    let delta0 = (pos0_max - pos0_min) / (bins0 as PositionT);
    let delta1 = (pos1_max - pos1_min) / (bins1 as PositionT);
    let b0i = bins0 as i64;
    let b1i = bins1 as i64;

    let accumulate = |acc: &mut [[AccT; 5]], idx: usize| {
        if let Some(m) = mask {
            if m[idx] != 0 {
                return;
            }
        }
        let s = prep[4 * idx];
        // Direct-split engines re-run preproc internally → hybrid variance 0
        // (see crate::internal_preproc_variance).
        let v = crate::internal_preproc_variance(error_model, prep[4 * idx + 1]);
        let n = prep[4 * idx + 2];
        let c = prep[4 * idx + 3];
        let c0 = pos0[idx];
        let c1 = pos1[idx];
        let d0 = delta_pos0[idx];
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
            return;
        }
        bin0_max = bin0_max.min(b0i - 1);
        bin0_min = bin0_min.max(0);
        bin1_max = bin1_max.min(b1i - 1);
        bin1_min = bin1_min.max(0);

        let cell = |bin0: i64, bin1: i64| -> usize { (bin0 * b1i + bin1) as usize };

        if bin0_min == bin0_max {
            if bin1_min == bin1_max {
                accumulate_2d(acc, cell(bin0_min, bin1_min), s, v, n, c, 1.0);
            } else {
                let delta_down = (bin1_min + 1) as PositionT - fbin1_min;
                let delta_up = fbin1_max - bin1_max as PositionT;
                let inv_area = 1.0 / (fbin1_max - fbin1_min);
                accumulate_2d(
                    acc,
                    cell(bin0_min, bin1_min),
                    s,
                    v,
                    n,
                    c,
                    inv_area * delta_down,
                );
                accumulate_2d(
                    acc,
                    cell(bin0_min, bin1_max),
                    s,
                    v,
                    n,
                    c,
                    inv_area * delta_up,
                );
                for j in (bin1_min + 1)..bin1_max {
                    accumulate_2d(acc, cell(bin0_min, j), s, v, n, c, inv_area);
                }
            }
        } else if bin1_min == bin1_max {
            let inv_area = 1.0 / (fbin0_max - fbin0_min);
            let delta_left = (bin0_min + 1) as PositionT - fbin0_min;
            accumulate_2d(
                acc,
                cell(bin0_min, bin1_min),
                s,
                v,
                n,
                c,
                inv_area * delta_left,
            );
            let delta_right = fbin0_max - bin0_max as PositionT;
            accumulate_2d(
                acc,
                cell(bin0_max, bin1_min),
                s,
                v,
                n,
                c,
                inv_area * delta_right,
            );
            for i in (bin0_min + 1)..bin0_max {
                accumulate_2d(acc, cell(i, bin1_min), s, v, n, c, inv_area);
            }
        } else {
            let inv_area = 1.0 / ((fbin0_max - fbin0_min) * (fbin1_max - fbin1_min));
            let delta_left = (bin0_min + 1) as PositionT - fbin0_min;
            let delta_right = fbin0_max - bin0_max as PositionT;
            let delta_down = (bin1_min + 1) as PositionT - fbin1_min;
            let delta_up = fbin1_max - bin1_max as PositionT;
            accumulate_2d(
                acc,
                cell(bin0_min, bin1_min),
                s,
                v,
                n,
                c,
                inv_area * delta_left * delta_down,
            );
            accumulate_2d(
                acc,
                cell(bin0_min, bin1_max),
                s,
                v,
                n,
                c,
                inv_area * delta_left * delta_up,
            );
            accumulate_2d(
                acc,
                cell(bin0_max, bin1_min),
                s,
                v,
                n,
                c,
                inv_area * delta_right * delta_down,
            );
            accumulate_2d(
                acc,
                cell(bin0_max, bin1_max),
                s,
                v,
                n,
                c,
                inv_area * delta_right * delta_up,
            );
            for i in (bin0_min + 1)..bin0_max {
                accumulate_2d(acc, cell(i, bin1_min), s, v, n, c, inv_area * delta_down);
                for j in (bin1_min + 1)..bin1_max {
                    accumulate_2d(acc, cell(i, j), s, v, n, c, inv_area);
                }
                accumulate_2d(acc, cell(i, bin1_max), s, v, n, c, inv_area * delta_up);
            }
            for j in (bin1_min + 1)..bin1_max {
                accumulate_2d(acc, cell(bin0_min, j), s, v, n, c, inv_area * delta_left);
                accumulate_2d(acc, cell(bin0_max, j), s, v, n, c, inv_area * delta_right);
            }
        }
    };

    // Serial scatter (bit-exact, like the 1D engine; see its scatter comment).
    let mut out = vec![[0.0f64; 5]; bins0 * bins1];
    for idx in 0..size {
        accumulate(&mut out, idx);
    }

    let radial_centers = numpy_linspace(pos0_min + 0.5 * delta0, pos0_max - 0.5 * delta0, bins0);
    let azim_centers = numpy_linspace(pos1_min + 0.5 * delta1, pos1_max - 0.5 * delta1, bins1);
    reduce_2d(
        &out,
        (bins0, bins1),
        radial_centers,
        azim_centers,
        error_model,
        empty,
    )
}

/// Port of `splitPixel.fullSplit1D_engine`: 1D full pixel-splitting histogram.
/// `corners` is the `(npix, 4, 2)` corner array flattened C-order (dim 0 radial
/// unscaled, dim 1 chi radians), upcast to f64. Each unmasked pixel's quadrilateral
/// is `recenter`-ed (chi discontinuity), mapped to bin space, and its trapezoidal
/// overlap with each radial bin is computed via [`integrate1d`] (the same buffer
/// machinery as `build_full_csr_1d`); the per-bin overlap normalized to sum 1 is
/// then scattered as an f64 weight into the accumulator (vs the full-CSR build's
/// f32 stored coef). For the standard radial units this is invoked with
/// `chi_disc_at_pi = true`, `pos1_period = 2π`, `allow_pos0_neg = false`. Returns
/// the [`CsrIntegrate1d`] field set (f64 sums + f32 derived).
#[allow(clippy::too_many_arguments)]
pub fn histogram1d_full(
    corners: &[PositionT],
    prep: &[DataT],
    mask: Option<&[i8]>,
    npt: usize,
    error_model: ErrorModel,
    empty: DataT,
    allow_pos0_neg: bool,
    chi_disc_at_pi: bool,
    pos1_period: PositionT,
) -> CsrIntegrate1d {
    assert!(npt > 1, "bins must be > 1");
    assert_eq!(
        corners.len() % 8,
        0,
        "corners must be (npix, 4, 2) flattened"
    );
    let size = corners.len() / 8;
    assert_eq!(prep.len(), 4 * size, "prep length must be 4 * npix");
    if let Some(m) = mask {
        assert_eq!(m.len(), size, "mask length mismatch");
    }

    let (pos0_min, pos0_maxin) = calc_boundaries_full_1d(corners, mask, allow_pos0_neg);
    let pos0_max = calc_upper_bound(pos0_maxin);
    let delta = (pos0_max - pos0_min) / (npt as PositionT);
    let bins_i = npt as i64;

    // Serial scatter (bit-exact; see the module scatter note). The trapezoid
    // buffer is reused across pixels and reset over the touched bin span.
    let mut res = vec![[0.0f64; 5]; npt];
    let mut buffer = vec![0.0 as DataT; npt];
    for idx in 0..size {
        if let Some(m) = mask {
            if m[idx] != 0 {
                continue;
            }
        }
        let s = prep[4 * idx];
        // Direct-split engines re-run preproc internally → hybrid variance 0
        // (see crate::internal_preproc_variance).
        let v = crate::internal_preproc_variance(error_model, prep[4 * idx + 1]);
        let n = prep[4 * idx + 2];
        let c = prep[4 * idx + 3];

        let base = idx * 8;
        let mut v8 = [
            [corners[base], corners[base + 1]],
            [corners[base + 2], corners[base + 3]],
            [corners[base + 4], corners[base + 5]],
            [corners[base + 6], corners[base + 7]],
        ];
        recenter(&mut v8, pos1_period, chi_disc_at_pi);
        // Radial corners to bin space (get_bin_number); azimuth carried through for
        // the _integrate1d line heights.
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
        if max0 < 0.0 || min0 >= npt as f64 {
            continue;
        }
        // pos1_range is None for this path -> no azimuthal range rejection.

        let bin0_min = min0.floor() as i64;
        let bin0_max = max0.floor() as i64;
        if bin0_min == bin0_max {
            accumulate_1d(&mut res[bin0_min as usize], s, v, n, c, 1.0, error_model);
        } else {
            let lo = bin0_min.max(0);
            let hi = (bin0_max + 1).min(bins_i);
            integrate1d(&mut buffer, a0, a1, b0, b1); // A-B
            integrate1d(&mut buffer, b0, b1, c0, c1); // B-C
            integrate1d(&mut buffer, c0, c1, d0, d1); // C-D
            integrate1d(&mut buffer, d0, d1, a0, a1); // D-A

            let mut sum_area = 0.0f64;
            for b in lo..hi {
                sum_area += buffer[b as usize] as f64;
            }
            if sum_area != 0.0 {
                let inv_area = 1.0 / sum_area;
                for b in lo..hi {
                    let w = buffer[b as usize] as f64 * inv_area;
                    accumulate_1d(&mut res[b as usize], s, v, n, c, w, error_model);
                }
            }
            for b in lo..hi {
                buffer[b as usize] = 0.0;
            }
        }
    }

    let position = numpy_linspace(pos0_min + 0.5 * delta, pos0_max - 0.5 * delta, npt);
    reduce_1d(&res, position, error_model, empty)
}

/// Port of `splitPixel.fullSplit2D_engine`: 2D full pixel-splitting histogram.
/// `corners` is the `(npix, 4, 2)` corner array flattened C-order (dim 0 radial
/// unscaled, dim 1 chi radians), upcast to f64. Each unmasked pixel is `recenter`-ed,
/// clipped into range, and swept into a small `(w0+1)·(w1+1)` box via [`integrate2d`]
/// (the same fused-type `_calc_area` machinery as `build_full_csr_2d`); the per-cell
/// overlap normalized to sum 1 is scattered as an f64 weight. `bounds` supplies
/// `allow_pos0_neg` / `chi_disc_at_pi` / `pos1_period` (the latter is the CHI_DEG
/// period 360 forwarded by `common.py`, applied to radian azimuths — a pyFAI quirk).
/// The range skip is the engine's `min1 > pos1_maxin`, NOT the full-CSR build's
/// `min1 >= pos1_max`. Returns the [`Integrate2d`] field set via [`reduce_2d`].
pub fn histogram2d_full(
    corners: &[PositionT],
    prep: &[DataT],
    mask: Option<&[i8]>,
    bins: (usize, usize),
    bounds: &Bbox2dBounds,
    error_model: ErrorModel,
    empty: DataT,
) -> Integrate2d {
    let (bins0, bins1) = bins;
    assert!(bins0 >= 1 && bins1 >= 1, "bins must be >= 1 in each dim");
    assert_eq!(
        corners.len() % 8,
        0,
        "corners must be (npix, 4, 2) flattened"
    );
    let size = corners.len() / 8;
    assert_eq!(prep.len(), 4 * size, "prep length must be 4 * npix");
    if let Some(m) = mask {
        assert_eq!(m.len(), size, "mask length mismatch");
    }

    let (pos0_min, pos0_maxin, pos1_min, pos1_maxin) = calc_boundaries_full_2d(
        corners,
        mask,
        bounds.allow_pos0_neg,
        bounds.chi_disc_at_pi,
        bounds.pos1_period,
    );
    let pos0_max = calc_upper_bound(pos0_maxin);
    let pos1_max = calc_upper_bound(pos1_maxin);
    let delta0 = (pos0_max - pos0_min) / (bins0 as PositionT);
    let delta1 = (pos1_max - pos1_min) / (bins1 as PositionT);
    let b1i = bins1 as i64;

    // Serial scatter (bit-exact; see the module scatter note).
    let mut out = vec![[0.0f64; 5]; bins0 * bins1];
    for idx in 0..size {
        if let Some(m) = mask {
            if m[idx] != 0 {
                continue;
            }
        }
        let s = prep[4 * idx];
        // Direct-split engines re-run preproc internally → hybrid variance 0
        // (see crate::internal_preproc_variance).
        let v = crate::internal_preproc_variance(error_model, prep[4 * idx + 1]);
        let n = prep[4 * idx + 2];
        let c = prep[4 * idx + 3];

        let base = idx * 8;
        let mut v8 = [
            [corners[base], corners[base + 1]],
            [corners[base + 2], corners[base + 3]],
            [corners[base + 4], corners[base + 5]],
            [corners[base + 6], corners[base + 7]],
        ];
        recenter(&mut v8, bounds.pos1_period, bounds.chi_disc_at_pi);
        let (mut a0, mut a1) = (v8[0][0], v8[0][1]);
        let (mut b0, mut b1) = (v8[1][0], v8[1][1]);
        let (mut c0, mut c1) = (v8[2][0], v8[2][1]);
        let (mut d0, mut d1) = (v8[3][0], v8[3][1]);

        // Engine range check in original space — note `min1 > pos1_maxin`
        // (fullSplit2D_engine), NOT the full-CSR build's `min1 >= pos1_max`.
        let min0 = min4(a0, b0, c0, d0);
        let max0 = max4(a0, b0, c0, d0);
        let min1 = min4(a1, b1, c1, d1);
        let max1 = max4(a1, b1, c1, d1);
        if max0 < pos0_min || min0 > pos0_maxin || max1 < pos1_min || min1 > pos1_maxin {
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
        // ABCD anti-trigonometric order: feed the four edges in turn.
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
        for i in 0..w0 {
            for j in 0..w1 {
                let w = box_buf[(i * (w1 + 1) + j) as usize] as f64 * inv_area;
                let cell = ((ioffset0 + i) * b1i + ioffset1 + j) as usize;
                accumulate_2d(&mut out, cell, s, v, n, c, w);
            }
        }
    }

    let radial_centers = numpy_linspace(pos0_min + 0.5 * delta0, pos0_max - 0.5 * delta0, bins0);
    let azim_centers = numpy_linspace(pos1_min + 0.5 * delta1, pos1_max - 0.5 * delta1, bins1);
    reduce_2d(
        &out,
        (bins0, bins1),
        radial_centers,
        azim_centers,
        error_model,
        empty,
    )
}

/// Exact quadrilateral area of corners A,B,C,D — port of `area4`
/// (`regrid_common.pxi`), the Bretschneider-style formula (NOT the approximate
/// `area4p` cross-product the full-CSR build uses). All operands and the `sqrt`s
/// are f64 (pyFAI instantiates `floating = double` here, the corners being
/// `position_t`); `x**2` is Cython-unrolled to `x*x` (verified bit-exact against
/// pyFAI's `_sp_area4` over 50k random inputs).
#[inline]
#[allow(clippy::too_many_arguments)]
fn area4(a0: f64, a1: f64, b0: f64, b1: f64, c0: f64, c1: f64, d0: f64, d1: f64) -> f64 {
    let a = ((b1 - a1) * (b1 - a1) + (b0 - a0) * (b0 - a0)).sqrt(); // AB
    let b = ((b1 - c1) * (b1 - c1) + (b0 - c0) * (b0 - c0)).sqrt(); // BC
    let c = ((c1 - d1) * (c1 - d1) + (c0 - d0) * (c0 - d0)).sqrt(); // CD
    let d = ((d1 - a1) * (d1 - a1) + (d0 - a0) * (d0 - a0)).sqrt(); // DA
    let p = ((c1 - a1) * (c1 - a1) + (c0 - a0) * (c0 - a0)).sqrt(); // AC
    let q = ((b1 - d1) * (b1 - d1) + (b0 - d0) * (b0 - d0)).sqrt(); // BD
    let diff = b * b + d * d - a * a - c * c;
    0.25 * (4.0 * p * p * q * q - diff * diff).sqrt()
}

/// Port of `splitPixel.pseudoSplit2D_engine`: 2D **pseudo** pixel-splitting
/// histogram (2D only — there is no 1D pseudo path). `corners` is the
/// `(npix, 4, 2)` corner array flattened C-order (dim 0 radial unscaled, dim 1 chi
/// radians), upcast to f64 — used **raw**, not `recenter`-ed.
///
/// Unlike the full split, each pixel is approximated by an **axis-aligned
/// rectangle** with the pixel's true quadrilateral [`area4`] but the aspect ratio
/// of its corner bounding box (`new_height = sqrt(area·height/width)`), centered on
/// the corner centroid. A pixel whose pseudo-rectangle would exceed its bounding
/// box (one straddling the chi discontinuity) keeps its original box. That box is
/// then range-clipped (the clip fraction scaling `value` in f32), collapsed across
/// the chi discontinuity (`(max1-min1)/delta1 > bins1/2`), and distributed across
/// the cells it overlaps with the same separable 4-branch fractional split as
/// [`histogram2d_bbox`] — except the both-axes-spread `inv_area` is pyFAI's
/// `(1/Δbin0)·Δbin1` (the literal operator order in `pseudoSplit2D_engine`, NOT
/// `1/(Δbin0·Δbin1)`). Boundaries use the raw corner fold (`calc_boundaries` with
/// `clip_pos1=False`, i.e. [`calc_boundaries_full_2d`] with `pos1_period = 0`); the
/// final reduction guards on **count** via [`reduce_2d`]. Serial scatter is the
/// bit-exact construction (see the module scatter note).
#[allow(clippy::too_many_arguments)]
pub fn histogram2d_pseudo(
    corners: &[PositionT],
    prep: &[DataT],
    mask: Option<&[i8]>,
    bins: (usize, usize),
    allow_pos0_neg: bool,
    chi_disc_at_pi: bool,
    error_model: ErrorModel,
    empty: DataT,
) -> Integrate2d {
    let (bins0, bins1) = bins;
    assert!(bins0 >= 1 && bins1 >= 1, "bins must be >= 1 in each dim");
    assert_eq!(
        corners.len() % 8,
        0,
        "corners must be (npix, 4, 2) flattened"
    );
    let size = corners.len() / 8;
    assert_eq!(prep.len(), 4 * size, "prep length must be 4 * npix");
    if let Some(m) = mask {
        assert_eq!(m.len(), size, "mask length mismatch");
    }

    // `calc_boundaries(..., clip_pos1=False)`: raw corner fold + pos0>=0 clamp, no
    // azimuthal clip (pos1_period = 0 disables the clip block).
    let (pos0_min, pos0_maxin, pos1_min, pos1_maxin) =
        calc_boundaries_full_2d(corners, mask, allow_pos0_neg, chi_disc_at_pi, 0.0);
    let pos0_max = calc_upper_bound(pos0_maxin);
    let pos1_max = calc_upper_bound(pos1_maxin);
    let delta0 = (pos0_max - pos0_min) / (bins0 as PositionT);
    let delta1 = (pos1_max - pos1_min) / (bins1 as PositionT);
    let b1i = bins1 as i64;

    // Chi-clamp bounds: `(2 - chiDiscAtPi)·pi` / `(-chiDiscAtPi)·pi` with pyFAI's
    // f32 `pi = <float> M_PI`, the int·float product in f32, then widened to f64.
    let cd: i32 = if chi_disc_at_pi { 1 } else { 0 };
    let max_chi = ((2 - cd) as f32 * std::f32::consts::PI) as f64;
    let min_chi = ((-cd) as f32 * std::f32::consts::PI) as f64;

    // pyFAI's `new_*` are C locals declared once outside the loop: a pixel with
    // width==0 or height==0 skips their recompute and the pathological check then
    // reads the previous pixel's values. Replicate that carry-over (every physical
    // Pilatus pixel has nonzero extent, so they are in practice always freshly set).
    let mut new_min0 = 0.0f64;
    let mut new_max0 = 0.0f64;
    let mut new_min1 = 0.0f64;
    let mut new_max1 = 0.0f64;

    let mut out = vec![[0.0f64; 5]; bins0 * bins1];
    for idx in 0..size {
        if let Some(m) = mask {
            if m[idx] != 0 {
                continue;
            }
        }
        let base = idx * 8;
        let a0 = corners[base];
        let a1 = corners[base + 1];
        let b0 = corners[base + 2];
        let b1 = corners[base + 3];
        let c0 = corners[base + 4];
        let c1 = corners[base + 5];
        let d0 = corners[base + 6];
        let d1 = corners[base + 7];

        let mut min0 = min4(a0, b0, c0, d0);
        let mut max0 = max4(a0, b0, c0, d0);
        let mut min1 = min4(a1, b1, c1, d1);
        let mut max1 = max4(a1, b1, c1, d1);

        if max0 < pos0_min || min0 > pos0_maxin || max1 < pos1_min || min1 > pos1_maxin {
            continue;
        }

        // Pseudo-rectangle: same area as the true quad, aspect ratio of the bbox.
        let center0 = (a0 + b0 + c0 + d0) / 4.0;
        let center1 = (a1 + b1 + c1 + d1) / 4.0;
        let area = area4(a0, a1, b0, b1, c0, c1, d0, d1).abs();
        let width = max1 - min1;
        let height = max0 - min0;
        if width != 0.0 && height != 0.0 {
            let new_height = (area * height / width).sqrt();
            let new_width = new_height * width / height;
            new_min0 = center0 - new_width / 2.0;
            new_max0 = center0 + new_width / 2.0;
            new_min1 = center1 - new_height / 2.0;
            new_max1 = center1 + new_height / 2.0;
        }
        if new_min0 < min0 || new_max0 > max0 || new_min1 < min1 || new_max1 > max1 {
            // Pathological pixel on the chi discontinuity: keep the original box.
        } else {
            min0 = new_min0;
            max0 = new_max0;
            min1 = new_min1;
            max1 = new_max1;
        }

        if !allow_pos0_neg {
            min0 = min0.max(0.0);
            max0 = max0.max(0.0);
        }
        if max1 > max_chi {
            max1 = max_chi;
        }
        if min1 < min_chi {
            min1 = min_chi;
        }

        let mut s = prep[4 * idx];
        // Pseudo-split re-runs preproc internally → hybrid variance 0 (see
        // crate::internal_preproc_variance); the scale below keeps 0 at 0.
        let mut v = crate::internal_preproc_variance(error_model, prep[4 * idx + 1]);
        let mut n = prep[4 * idx + 2];
        let mut c = prep[4 * idx + 3];

        // Range clip with the area-fraction scale (pyFAI's `scale` is data_t/f32:
        // each `scale * f64 / f64` narrows back to f32). min/max are mutated in
        // sequence, so a later branch sees the earlier branch's clamped bound.
        let mut scale: f32 = 1.0;
        if min0 < pos0_min {
            scale = ((scale as f64) * (pos0_min - min0) / (max0 - min0)) as f32;
            min0 = pos0_min;
        }
        if min1 < pos1_min {
            scale = ((scale as f64) * (pos1_min - min1) / (max1 - min1)) as f32;
            min1 = pos1_min;
        }
        if max0 > pos0_maxin {
            scale = ((scale as f64) * (max0 - pos0_maxin) / (max0 - min0)) as f32;
            max0 = pos0_maxin;
        }
        if max1 > pos1_maxin {
            scale = ((scale as f64) * (max1 - pos1_maxin) / (max1 - min1)) as f32;
            max1 = pos1_maxin;
        }
        if scale != 1.0 {
            s *= scale;
            n *= scale;
            v *= scale * scale;
            c *= scale;
        }

        // Collapse a pixel spanning more than half the azimuthal range onto the
        // nearer edge (the chi-discontinuity heuristic).
        if (max1 - min1) / delta1 > (bins1 as f64) / 2.0 {
            if pos1_maxin - max1 > min1 - pos1_min {
                min1 = max1;
                max1 = pos1_maxin;
            } else {
                max1 = min1;
                min1 = pos1_min;
            }
        }

        let fbin0_min = (min0 - pos0_min) / delta0; // get_bin_number
        let fbin0_max = (max0 - pos0_min) / delta0;
        let fbin1_min = (min1 - pos1_min) / delta1;
        let fbin1_max = (max1 - pos1_min) / delta1;

        let bin0_min = fbin0_min as i64; // <Py_ssize_t>: trunc toward zero
        let bin0_max = fbin0_max as i64;
        let bin1_min = fbin1_min as i64;
        let bin1_max = fbin1_max as i64;

        let cell = |bin0: i64, bin1: i64| -> usize { (bin0 * b1i + bin1) as usize };

        if bin0_min == bin0_max {
            if bin1_min == bin1_max {
                accumulate_2d(&mut out, cell(bin0_min, bin1_min), s, v, n, c, 1.0);
            } else {
                let delta_down = (bin1_min + 1) as PositionT - fbin1_min;
                let delta_up = fbin1_max - bin1_max as PositionT;
                let inv_area = 1.0 / (fbin1_max - fbin1_min);
                accumulate_2d(
                    &mut out,
                    cell(bin0_min, bin1_min),
                    s,
                    v,
                    n,
                    c,
                    inv_area * delta_down,
                );
                accumulate_2d(
                    &mut out,
                    cell(bin0_min, bin1_max),
                    s,
                    v,
                    n,
                    c,
                    inv_area * delta_up,
                );
                for j in (bin1_min + 1)..bin1_max {
                    accumulate_2d(&mut out, cell(bin0_min, j), s, v, n, c, inv_area);
                }
            }
        } else if bin1_min == bin1_max {
            let inv_area = 1.0 / (fbin0_max - fbin0_min);
            let delta_left = (bin0_min + 1) as PositionT - fbin0_min;
            accumulate_2d(
                &mut out,
                cell(bin0_min, bin1_min),
                s,
                v,
                n,
                c,
                inv_area * delta_left,
            );
            let delta_right = fbin0_max - bin0_max as PositionT;
            accumulate_2d(
                &mut out,
                cell(bin0_max, bin1_min),
                s,
                v,
                n,
                c,
                inv_area * delta_right,
            );
            for i in (bin0_min + 1)..bin0_max {
                accumulate_2d(&mut out, cell(i, bin1_min), s, v, n, c, inv_area);
            }
        } else {
            // pyFAI's literal `1.0 / (Δbin0) * (Δbin1)` — operator order makes this
            // `(1/Δbin0)·Δbin1`, NOT `1/(Δbin0·Δbin1)`; reproduce it verbatim.
            let inv_area = 1.0 / (fbin0_max - fbin0_min) * (fbin1_max - fbin1_min);
            let delta_left = (bin0_min + 1) as PositionT - fbin0_min;
            let delta_right = fbin0_max - bin0_max as PositionT;
            let delta_down = (bin1_min + 1) as PositionT - fbin1_min;
            let delta_up = fbin1_max - bin1_max as PositionT;
            accumulate_2d(
                &mut out,
                cell(bin0_min, bin1_min),
                s,
                v,
                n,
                c,
                inv_area * delta_left * delta_down,
            );
            accumulate_2d(
                &mut out,
                cell(bin0_min, bin1_max),
                s,
                v,
                n,
                c,
                inv_area * delta_left * delta_up,
            );
            accumulate_2d(
                &mut out,
                cell(bin0_max, bin1_min),
                s,
                v,
                n,
                c,
                inv_area * delta_right * delta_down,
            );
            accumulate_2d(
                &mut out,
                cell(bin0_max, bin1_max),
                s,
                v,
                n,
                c,
                inv_area * delta_right * delta_up,
            );
            for i in (bin0_min + 1)..bin0_max {
                accumulate_2d(
                    &mut out,
                    cell(i, bin1_min),
                    s,
                    v,
                    n,
                    c,
                    inv_area * delta_down,
                );
                for j in (bin1_min + 1)..bin1_max {
                    accumulate_2d(&mut out, cell(i, j), s, v, n, c, inv_area);
                }
                accumulate_2d(&mut out, cell(i, bin1_max), s, v, n, c, inv_area * delta_up);
            }
            for j in (bin1_min + 1)..bin1_max {
                accumulate_2d(
                    &mut out,
                    cell(bin0_min, j),
                    s,
                    v,
                    n,
                    c,
                    inv_area * delta_left,
                );
                accumulate_2d(
                    &mut out,
                    cell(bin0_max, j),
                    s,
                    v,
                    n,
                    c,
                    inv_area * delta_right,
                );
            }
        }
    }

    let radial_centers = numpy_linspace(pos0_min + 0.5 * delta0, pos0_max - 0.5 * delta0, bins0);
    let azim_centers = numpy_linspace(pos1_min + 0.5 * delta1, pos1_max - 0.5 * delta1, bins1);
    reduce_2d(
        &out,
        (bins0, bins1),
        radial_centers,
        azim_centers,
        error_model,
        empty,
    )
}
