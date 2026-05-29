//! LUT (dense look-up table) integration — the `("no"|"bbox"|"full", "lut",
//! "cython")` 1D/2D paths (`ext/splitBBoxLUT.pyx`, `ext/splitPixelFullLUT.pyx`,
//! `ext/LUT_common.pxi`).
//!
//! pyFAI builds the LUT from the SAME `calc_lut_*()` the CSR path uses, ending
//! in `.to_lut()` instead of `.to_csr()` (`SparseBuilder.to_lut`). `to_lut`
//! produces a dense `(n_bins × lut_size)` matrix of `{idx: i32, coef: f32}`
//! records, where `lut_size` is the largest per-bin entry count; each bin's real
//! entries fill columns `0..bin_size` **in the same order as the CSR row**, and
//! the remaining columns are zero-padding (`{idx: 0, coef: 0.0}`, from
//! `numpy.zeros`). So the build here reuses the [`crate::csr`] builders verbatim
//! and applies [`csr_to_lut`].
//!
//! The apply (`LutIntegrator.integrate_ng`) is a per-output-bin gather: for each
//! bin it walks all `lut_size` columns, skipping padding via `idx < 0 || coef ==
//! 0.0`, and accumulates the surviving entries into `acc_t` accumulators — the
//! same gather the CSR apply performs over its row, visiting the same entries in
//! the same order. The per-bin f64 sums are therefore bit-identical to the CSR
//! apply for the non-azimuthal error models. The gather+finalize ([`lut_gather_bin`],
//! mirroring [`crate::csr::csr_gather_bin`]) is driven by the shared fused
//! [`crate::csr::fill_reduction`] and packaged with [`crate::csr::reduction_to_1d`]
//! (1D) / [`crate::csr::pack_2d`] (2D). The azimuthal (Welford) error model
//! accumulates per bin via [`crate::azimuthal::azimuthal_step`] (`b = sig/norm` in
//! f64, no zero-norm guard — pyFAI's bare `else`).

use crate::csr::{
    build_bbox_csr_1d, build_bbox_csr_2d, build_full_csr_1d, build_full_csr_2d, fill_reduction,
    fill_reduction_into, finalize_bin, pack_2d, reduction_to_1d, Bbox2dBounds, BboxAzim1d,
    BinReduction, Csr, CsrIntegrate1d, ReductionOut,
};
use crate::histogram::Integrate2d;
use rsfai_core::dtype::{AccT, DataT, ErrorModel, IndexT, PositionT};

/// A dense look-up table of overlap coefficients, `(n_bins × lut_size)` stored
/// row-major: output bin `b`'s entries are `coef[b*lut_size .. (b+1)*lut_size]`
/// and the matching `idx[...]`. Real entries occupy the leading columns of each
/// row (CSR-row order); trailing columns are zero-padding (`idx = 0, coef = 0.0`)
/// and are skipped on apply. `lut_size` is the largest per-bin entry count.
///
/// Construct via [`Lut::new`], which also precomputes `sizes` — the per-row count
/// of populated leading columns — so the apply iterates only each row's real head
/// (≈ the CSR entry count) instead of all `lut_size` columns. At fine 2D binning
/// `lut_size` is set by the densest bin while most rows hold a handful of entries,
/// so walking the full width is mostly padding reads; the dense `coef`/`idx`
/// matrix is unchanged (byte-identical to pyFAI's), only the scan is bounded.
#[derive(Debug, Clone, PartialEq)]
pub struct Lut {
    /// Overlap coefficients (`data_t`, f32), row-major `(n_bins × lut_size)`.
    pub coef: Vec<DataT>,
    /// Pixel (column) indices (`index_t`, i32), row-major `(n_bins × lut_size)`.
    pub idx: Vec<IndexT>,
    /// Entries per bin (the dense row width); `0` for an all-empty matrix.
    pub lut_size: usize,
    /// Per-bin populated length: `sizes[b]` columns of row `b` hold real entries,
    /// the rest is trailing padding. Private so it is always derived by
    /// [`Lut::new`] and stays consistent with `coef`/`idx`.
    sizes: Vec<IndexT>,
}

impl Lut {
    /// Build from the dense row-major `(n_bins × lut_size)` `coef`/`idx` matrix,
    /// precomputing `sizes[b]` = the position just past row `b`'s last non-padding
    /// slot. Padding is the trailing `(idx = 0, coef = 0)` run that `csr_to_lut` /
    /// `numpy.zeros` leave after the front-packed real entries. Trimming the scan
    /// to `sizes[b]` is **bit-exact**: every trimmed slot has `coef == 0` (true
    /// padding, or an inert real zero-overlap entry indistinguishable from it),
    /// and a zero coefficient contributes nothing to any accumulator. Interior
    /// zero-coef entries keep their column and are skipped per-entry in the gather.
    pub fn new(coef: Vec<DataT>, idx: Vec<IndexT>, lut_size: usize) -> Self {
        let n_bins = if lut_size == 0 {
            0
        } else {
            coef.len() / lut_size
        };
        let mut sizes = vec![0 as IndexT; n_bins];
        for (b, size) in sizes.iter_mut().enumerate() {
            let row = b * lut_size;
            // Scan from the row end back to the last populated column.
            for col in (0..lut_size).rev() {
                if idx[row + col] != 0 || coef[row + col] != 0.0 {
                    *size = (col + 1) as IndexT;
                    break;
                }
            }
        }
        Lut {
            coef,
            idx,
            lut_size,
            sizes,
        }
    }
}

/// Convert a bin-major [`Csr`] into the dense [`Lut`] — port of
/// `SparseBuilder.to_lut`. `lut_size` is the largest per-bin entry count; the
/// matrix is zero-initialised (`idx = 0, coef = 0.0`, matching `numpy.zeros`),
/// then each bin's CSR entries are copied into the leading columns of its row in
/// CSR order (coefficients bit-preserved). The padding `coef = 0.0` is what the
/// apply skips, so the trailing `idx = 0` is inert.
pub(crate) fn csr_to_lut(csr: &Csr, n_bins: usize) -> Lut {
    assert_eq!(
        csr.indptr.len(),
        n_bins + 1,
        "csr.indptr length must be n_bins + 1"
    );

    // Largest per-bin entry count = dense row width.
    let mut lut_size = 0usize;
    for b in 0..n_bins {
        let size = (csr.indptr[b + 1] - csr.indptr[b]) as usize;
        if size > lut_size {
            lut_size = size;
        }
    }

    let mut coef = vec![0.0 as DataT; n_bins * lut_size];
    let mut idx = vec![0 as IndexT; n_bins * lut_size];
    for b in 0..n_bins {
        let lo = csr.indptr[b] as usize;
        let hi = csr.indptr[b + 1] as usize;
        let row = b * lut_size;
        for (col, k) in (lo..hi).enumerate() {
            idx[row + col] = csr.indices[k];
            coef[row + col] = csr.data[k];
        }
    }

    Lut::new(coef, idx, lut_size)
}

/// Gather one output bin's contributions from its dense LUT row and finalize —
/// the per-bin body of `LutIntegrator.integrate_ng`. Walks all `lut_size` columns
/// of row `bin` (`bin*lut_size .. (bin+1)*lut_size`) in ascending order, skipping
/// padding via `idx < 0 || coef == 0.0`. Depends on no other bin, so it can be
/// invoked for output cells in any order / in parallel and stays **bit-exact**
/// (per-bin accumulation order fixed by the row). This matches
/// [`crate::csr::csr_gather_bin`]'s structure; the gather differs (dense padded
/// row vs compact CSR row) and the azimuthal branch has no `norm != 0` guard
/// (pyFAI's bare `else`). `prep` is the flat `[signal, variance, norm, count]`-
/// per-pixel f32 array.
#[inline]
fn lut_gather_bin(
    lut: &Lut,
    prep: &[DataT],
    error_model: ErrorModel,
    do_variance: bool,
    empty: DataT,
    bin: usize,
) -> BinReduction {
    let lut_size = lut.lut_size;
    let mut acc_sig: AccT = 0.0;
    let mut acc_var: AccT = 0.0;
    let mut acc_norm: AccT = 0.0;
    let mut acc_norm_sq: AccT = 0.0;
    let mut acc_count: AccT = 0.0;

    // Walk only the row's populated head (`sizes[bin]` ≤ `lut_size`), not the full
    // padded width — the trailing padding is all `coef == 0` and would be skipped
    // anyway, so bounding the scan is bit-exact (see [`Lut::new`]). Slicing the row
    // first lets the iterator run without per-element bounds checks.
    let size = lut.sizes[bin] as usize;
    let row = bin * lut_size;
    let idx_row = &lut.idx[row..row + size];
    let coef_row = &lut.coef[row..row + size];
    for (&idx, &coef) in idx_row.iter().zip(coef_row.iter()) {
        let coef = coef as AccT; // data_t -> acc_t
        if idx < 0 || coef == 0.0 {
            continue; // interior dropped entry
        }
        let p = idx as usize;
        let sig = prep[4 * p] as AccT;
        let var = prep[4 * p + 1] as AccT;
        let norm = prep[4 * p + 2] as AccT;
        let cnt = prep[4 * p + 3] as AccT;
        acc_count += coef * cnt;
        match error_model {
            // pyFAI LUT_common.pxi `do_azimuthal_variance`: per-bin Welford.
            // `b = sig/norm` in f64 (sig/norm are acc_t). Unlike CSR, the update
            // branch has no `norm != 0` guard (pyFAI's bare `else`), so a zero-norm
            // contribution still runs (b becomes inf/NaN, matching pyFAI).
            ErrorModel::Azimuthal => {
                crate::azimuthal::azimuthal_step(
                    &mut acc_sig,
                    &mut acc_var,
                    &mut acc_norm,
                    &mut acc_norm_sq,
                    coef * norm,
                    coef * sig,
                    sig / norm,
                );
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

    finalize_bin(
        acc_sig,
        acc_var,
        acc_norm,
        acc_norm_sq,
        acc_count,
        do_variance,
        empty,
    )
}

/// Apply a 1D LUT to preprocessed rows — port of `LutIntegrator.integrate_ng`'s
/// 1D return. `bin_centers` are the unscaled centers from the LUT build; the
/// binned sums are exposed at full f64 (`acc_t`).
pub fn lut_integrate1d(
    lut: &Lut,
    prep: &[DataT],
    bin_centers: Vec<PositionT>,
    error_model: ErrorModel,
    empty: DataT,
) -> CsrIntegrate1d {
    let bins = bin_centers.len();
    assert_eq!(
        lut.coef.len(),
        bins * lut.lut_size,
        "lut.coef length must be n_bins * lut_size"
    );
    let do_variance = error_model != ErrorModel::No;
    let r = fill_reduction(bins, |t| {
        lut_gather_bin(lut, prep, error_model, do_variance, empty, t)
    });
    reduction_to_1d(r, bin_centers)
}

/// Apply a 2D LUT to preprocessed rows — port of `LutIntegrator.integrate_ng`'s
/// 2D return. Identical to the 2D CSR apply: the `reshape(bins0, bins1).T`
/// transpose is folded into the per-cell gather index (output cell `t` →
/// azimuthal `j = t / bins0`, radial `i = t % bins0` → source bin `i·bins1 + j`),
/// so the fused reduce writes final azimuthal-major order directly.
pub fn lut_integrate2d(
    lut: &Lut,
    prep: &[DataT],
    bin_centers0: Vec<PositionT>,
    bin_centers1: Vec<PositionT>,
    error_model: ErrorModel,
    empty: DataT,
) -> Integrate2d {
    let bins0 = bin_centers0.len();
    let bins1 = bin_centers1.len();
    assert_eq!(
        lut.coef.len(),
        bins0 * bins1 * lut.lut_size,
        "lut.coef length must be n_bins * lut_size"
    );
    let do_variance = error_model != ErrorModel::No;
    let r = fill_reduction(bins0 * bins1, |t| {
        let bin = (t % bins0) * bins1 + (t / bins0);
        lut_gather_bin(lut, prep, error_model, do_variance, empty, bin)
    });
    pack_2d(r, bins0, bins1, bin_centers0, bin_centers1)
}

/// Fused 2D LUT apply that writes into caller-provided output columns instead of
/// allocating a fresh [`Integrate2d`] — the LUT counterpart of
/// [`crate::csr::csr_integrate2d_into`] (the streaming `out=` path). Same gather,
/// transpose-fold, and parallel fill as [`lut_integrate2d`], so the columns get
/// **bit-identical** values; only the result buffers differ (reused across frames,
/// avoiding the per-frame allocate + first-touch fault). `bins0`/`bins1` are the
/// radial / azimuthal bin counts; every [`ReductionOut`] column must have length
/// `bins0 * bins1`.
pub fn lut_integrate2d_into(
    lut: &Lut,
    prep: &[DataT],
    bins0: usize,
    bins1: usize,
    error_model: ErrorModel,
    empty: DataT,
    out: ReductionOut<'_>,
) {
    assert_eq!(
        lut.coef.len(),
        bins0 * bins1 * lut.lut_size,
        "lut.coef length must be n_bins * lut_size"
    );
    assert_eq!(
        out.signal.len(),
        bins0 * bins1,
        "out columns must have length bins0 * bins1"
    );
    let do_variance = error_model != ErrorModel::No;
    fill_reduction_into(out, |t| {
        let bin = (t % bins0) * bins1 + (t / bins0);
        lut_gather_bin(lut, prep, error_model, do_variance, empty, bin)
    });
}

/// Build the 1D bbox LUT and the unscaled bin centers — the `("no"|"bbox", "lut",
/// "cython")` build (`splitBBoxLUT.HistoBBox1d`). Mirrors pyFAI: build the bbox
/// CSR LUT then densify with [`csr_to_lut`]. `delta_pos0 = None` is the no-split
/// case (each pixel collapses to its center, coef 1.0).
pub fn build_bbox_lut_1d(
    pos0: &[PositionT],
    delta_pos0: Option<&[PositionT]>,
    mask: Option<&[i8]>,
    bins: usize,
    allow_pos0_neg: bool,
    pos0_range: Option<(PositionT, PositionT)>,
    azim: Option<BboxAzim1d>,
) -> (Lut, Vec<PositionT>) {
    let (csr, centers) = build_bbox_csr_1d(
        pos0,
        delta_pos0,
        mask,
        bins,
        allow_pos0_neg,
        pos0_range,
        azim,
    );
    (csr_to_lut(&csr, bins), centers)
}

/// Build the 2D bbox LUT and the unscaled radial / radian azimuthal centers — the
/// `("no"|"bbox", "lut", "cython")` 2D build (`splitBBoxLUT.HistoBBox2d`).
/// `delta_pos0`/`delta_pos1 = None` is no-split.
pub fn build_bbox_lut_2d(
    pos0: &[PositionT],
    delta_pos0: Option<&[PositionT]>,
    pos1: &[PositionT],
    delta_pos1: Option<&[PositionT]>,
    mask: Option<&[i8]>,
    bins: (usize, usize),
    bounds: &Bbox2dBounds,
) -> (Lut, Vec<PositionT>, Vec<PositionT>) {
    let (csr, c0, c1) = build_bbox_csr_2d(pos0, delta_pos0, pos1, delta_pos1, mask, bins, bounds);
    (csr_to_lut(&csr, bins.0 * bins.1), c0, c1)
}

/// Build the 1D full pixel-splitting LUT and the unscaled bin centers — the
/// `("full", "lut", "cython")` build (`splitPixelFullLUT.HistoLUT1dFullSplit`).
/// `corners` is the `(npix, 4, 2)` array flattened to f64.
#[allow(clippy::too_many_arguments)]
pub fn build_full_lut_1d(
    corners: &[PositionT],
    mask: Option<&[i8]>,
    bins: usize,
    allow_pos0_neg: bool,
    chi_disc_at_pi: bool,
    pos1_period: PositionT,
    pos0_range: Option<(PositionT, PositionT)>,
    pos1_range: Option<(PositionT, PositionT)>,
) -> (Lut, Vec<PositionT>) {
    let (csr, centers) = build_full_csr_1d(
        corners,
        mask,
        bins,
        allow_pos0_neg,
        chi_disc_at_pi,
        pos1_period,
        pos0_range,
        pos1_range,
    );
    (csr_to_lut(&csr, bins), centers)
}

/// Build the 2D full pixel-splitting LUT and the unscaled radial / radian
/// azimuthal centers — the `("full", "lut", "cython")` 2D build
/// (`splitPixelFullLUT.HistoLUT2dFullSplit`).
pub fn build_full_lut_2d(
    corners: &[PositionT],
    mask: Option<&[i8]>,
    bins: (usize, usize),
    bounds: &Bbox2dBounds,
) -> (Lut, Vec<PositionT>, Vec<PositionT>) {
    let (csr, c0, c1) = build_full_csr_2d(corners, mask, bins, bounds);
    (csr_to_lut(&csr, bins.0 * bins.1), c0, c1)
}
