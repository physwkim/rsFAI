//! CSC (Compressed Sparse Column) integration — the `("no"|"bbox"|"full", "csc",
//! "cython")` 1D/2D paths (`ext/splitBBoxCSC.pyx`, `ext/splitPixelFullCSC.pyx`,
//! `ext/CSC_common.pxi`).
//!
//! pyFAI builds the CSC matrix by taking the SAME LUT the CSR path uses
//! (`calc_lut_*().to_csr()`) and transposing it with scipy's
//! `csr_matrix(..., shape=(bins, size)).tocsc()`. So the build here reuses the
//! [`crate::csr`] builders verbatim and applies [`csr_to_csc`] (a port of
//! scipy's `csr_tocsc` counting-transpose). The matrix is `(n_bins × n_pixels)`;
//! the CSC `indptr` is per-PIXEL (length `n_pixels + 1`), `indices` are bin (row)
//! indices, `data` the same f32 coefficients permuted into column-major order.
//!
//! The apply (`CscIntegrator.integrate_ng`) is a single serial pass over PIXELS,
//! scattering each pixel's contributions into per-bin `acc_t` accumulators — the
//! transpose of the CSR gather. For a fixed output bin the contributions still
//! arrive in ascending pixel-index order (the SparseBuilder insertion order the
//! CSR row preserves), so the per-bin f64 sums are bit-identical to the CSR apply
//! for the non-azimuthal error models, and the final per-bin reduction
//! ([`crate::csr::finalize_reduction`]) and packaging
//! ([`crate::csr::reduction_to_1d`] / [`crate::csr::reduction_to_2d`]) are shared.
//! The azimuthal (Welford) error model accumulates per bin via
//! [`crate::azimuthal::azimuthal_step`]; CSC computes `b` from the f32
//! `value.signal/value.norm` and skips invalid (count-0) pixels, matching
//! pyFAI's `if not is_valid: continue`.

use crate::csr::{
    build_bbox_csr_1d, build_bbox_csr_2d, build_full_csr_1d, build_full_csr_2d, finalize_reduction,
    reduction_to_1d, reduction_to_2d, Bbox2dBounds, Csr, CsrIntegrate1d,
};
use crate::histogram::Integrate2d;
use rsfai_core::dtype::{AccT, DataT, ErrorModel, IndexT, PositionT};

/// A compressed-sparse-column matrix of overlap coefficients, `(n_bins ×
/// n_pixels)`: column `idx` (pixel `idx`) spans `indptr[idx]..indptr[idx + 1]`,
/// and entry `k` there is `(indices[k], data[k])` — contributing **bin** index
/// and overlap coefficient. The transpose of [`Csr`].
#[derive(Debug, Clone, PartialEq)]
pub struct Csc {
    /// Overlap coefficients (`data_t`, f32).
    pub data: Vec<DataT>,
    /// Bin (row) indices (`index_t`, i32).
    pub indices: Vec<IndexT>,
    /// Column (pixel) pointers, length `n_pixels + 1` (`index_t`, i32).
    pub indptr: Vec<IndexT>,
}

/// Transpose a `(n_bins × n_pixels)` [`Csr`] into [`Csc`] — port of scipy's
/// `csr_tocsc` (the counting-transpose `csr_matrix(...).tocsc()` pyFAI's CSC
/// engines run on the LUT). The result is canonical: columns in ascending pixel
/// order, and within each column the bin (row) indices ascending. Coefficients
/// are the original f32 values permuted (bit-preserved). `n_pixels` is the FULL
/// pixel count (the CSR's column space, including masked pixels with empty
/// columns).
pub(crate) fn csr_to_csc(csr: &Csr, n_bins: usize, n_pixels: usize) -> Csc {
    let nnz = csr.data.len();
    assert_eq!(
        csr.indptr.len(),
        n_bins + 1,
        "csr.indptr length must be n_bins + 1"
    );
    assert_eq!(csr.indices.len(), nnz, "csr.indices/data length mismatch");

    // Per-column (pixel) entry counts, then prefix-sum into column start offsets.
    let mut indptr = vec![0 as IndexT; n_pixels + 1];
    for &col in &csr.indices {
        indptr[col as usize] += 1;
    }
    let mut cumsum: IndexT = 0;
    for slot in indptr.iter_mut().take(n_pixels) {
        let count = *slot;
        *slot = cumsum;
        cumsum += count;
    }
    indptr[n_pixels] = nnz as IndexT;

    // Scatter: walk bins (rows) ascending; each entry lands at its column's
    // running write cursor, so within a column the bin indices come out
    // ascending. `next` is the per-column cursor seeded from the start offsets.
    let mut indices = vec![0 as IndexT; nnz];
    let mut data = vec![0.0 as DataT; nnz];
    let mut next = indptr.clone();
    for row in 0..n_bins {
        let lo = csr.indptr[row] as usize;
        let hi = csr.indptr[row + 1] as usize;
        for jj in lo..hi {
            let col = csr.indices[jj] as usize;
            let dest = next[col] as usize;
            indices[dest] = row as IndexT;
            data[dest] = csr.data[jj];
            next[col] += 1;
        }
    }

    Csc {
        data,
        indices,
        indptr,
    }
}

/// The CSC apply's per-bin reduction: a serial pixel-major scatter into `acc_t`
/// accumulators, then the shared [`finalize_reduction`]. Mirrors
/// `CscIntegrator.integrate_ng`'s single nogil pass. `prep` is the flat
/// `[signal, variance, norm, count]`-per-pixel f32 array.
fn csc_reduce(
    csc: &Csc,
    prep: &[DataT],
    n_bins: usize,
    error_model: ErrorModel,
    empty: DataT,
) -> crate::csr::CsrReduction {
    let n_pixels = csc.indptr.len() - 1;
    assert_eq!(prep.len(), 4 * n_pixels, "prep length must be 4 * n_pixels");
    let do_variance = error_model != ErrorModel::No;

    let mut acc_sig = vec![0.0 as AccT; n_bins];
    let mut acc_var = vec![0.0 as AccT; n_bins];
    let mut acc_norm = vec![0.0 as AccT; n_bins];
    let mut acc_norm_sq = vec![0.0 as AccT; n_bins];
    let mut acc_count = vec![0.0 as AccT; n_bins];

    for idx in 0..n_pixels {
        let lo = csc.indptr[idx] as usize;
        let hi = csc.indptr[idx + 1] as usize;
        if lo == hi {
            continue; // pixel contributes to no bin (masked / out of range)
        }
        let sig = prep[4 * idx] as AccT;
        // CSC re-runs preproc internally, so hybrid's per-pixel variance is 0
        // (see crate::internal_preproc_variance).
        let var = crate::internal_preproc_variance(error_model, prep[4 * idx + 1]) as AccT;
        let norm = prep[4 * idx + 2] as AccT;
        let cnt = prep[4 * idx + 3] as AccT;
        // pyFAI CSC_common.pxi integrate_ng computes preproc per pixel and skips
        // invalid ones before scattering (`if not is_valid: continue`); an
        // invalid pixel has an all-zero preproc row (count 0). For the linear
        // error models this skip is a no-op (scattering zeros adds nothing), but
        // the azimuthal Welford divides by norm, so a zero-norm pixel would
        // inject b = 0/0 = NaN into the bin. Skip it, exactly as pyFAI does.
        if cnt == 0.0 {
            continue;
        }
        for j in lo..hi {
            let coef = csc.data[j] as AccT; // data_t -> acc_t
            let bin = csc.indices[j] as usize;
            acc_count[bin] += coef * cnt;
            match error_model {
                // pyFAI CSC_common.pxi `do_azimuthal_variance`: per-bin Welford,
                // no `norm != 0` guard (like LUT). Unlike CSR/LUT, `b` is the
                // f32 `value.signal / value.norm` promoted to f64 — CSC reads the
                // preproc into the f32 `preproc_t value`, so the division rounds
                // to f32 first.
                ErrorModel::Azimuthal => {
                    let b = (prep[4 * idx] / prep[4 * idx + 2]) as AccT;
                    crate::azimuthal::azimuthal_step(
                        &mut acc_sig[bin],
                        &mut acc_var[bin],
                        &mut acc_norm[bin],
                        &mut acc_norm_sq[bin],
                        coef * norm,
                        coef * sig,
                        b,
                    );
                }
                _ => {
                    acc_sig[bin] += coef * sig;
                    if do_variance {
                        acc_var[bin] += coef * coef * var;
                    }
                    let w = coef * norm;
                    acc_norm[bin] += w;
                    acc_norm_sq[bin] += w * w;
                }
            }
        }
    }

    finalize_reduction(
        &acc_sig,
        &acc_var,
        &acc_norm,
        &acc_norm_sq,
        &acc_count,
        do_variance,
        empty,
    )
}

/// Apply a 1D CSC matrix to preprocessed rows — port of `CscIntegrator.
/// integrate_ng`'s 1D return. `bin_centers` are the unscaled centers from the
/// CSC build; the binned sums are exposed at full f64 (`acc_t`).
pub fn csc_integrate1d(
    csc: &Csc,
    prep: &[DataT],
    bin_centers: Vec<PositionT>,
    error_model: ErrorModel,
    empty: DataT,
) -> CsrIntegrate1d {
    let bins = bin_centers.len();
    let r = csc_reduce(csc, prep, bins, error_model, empty);
    reduction_to_1d(r, bin_centers)
}

/// Apply a 2D CSC matrix to preprocessed rows — port of `CscIntegrator.
/// integrate_ng`'s 2D return. The flat per-bin arrays (output bin `i·bins1 + j`,
/// radial-major) are reshaped to `(bins0, bins1)` and transposed to
/// **(azimuthal, radial)**, identical to the 2D CSR apply.
pub fn csc_integrate2d(
    csc: &Csc,
    prep: &[DataT],
    bin_centers0: Vec<PositionT>,
    bin_centers1: Vec<PositionT>,
    error_model: ErrorModel,
    empty: DataT,
) -> Integrate2d {
    let bins0 = bin_centers0.len();
    let bins1 = bin_centers1.len();
    let r = csc_reduce(csc, prep, bins0 * bins1, error_model, empty);
    reduction_to_2d(r, bins0, bins1, bin_centers0, bin_centers1)
}

/// Build the 1D bbox CSC matrix and the unscaled bin centers — the
/// `("no"|"bbox", "csc", "cython")` build (`splitBBoxCSC.HistoBBox1d`). Mirrors
/// pyFAI: build the bbox CSR LUT then transpose. `delta_pos0 = None` is the
/// no-split case (each pixel collapses to its center, coef 1.0).
pub fn build_bbox_csc_1d(
    pos0: &[PositionT],
    delta_pos0: Option<&[PositionT]>,
    mask: Option<&[i8]>,
    bins: usize,
    allow_pos0_neg: bool,
) -> (Csc, Vec<PositionT>) {
    let n_pixels = pos0.len();
    let (csr, centers) = build_bbox_csr_1d(pos0, delta_pos0, mask, bins, allow_pos0_neg);
    (csr_to_csc(&csr, bins, n_pixels), centers)
}

/// Build the 2D bbox CSC matrix and the unscaled radial / radian azimuthal
/// centers — the `("no"|"bbox", "csc", "cython")` 2D build
/// (`splitBBoxCSC.HistoBBox2d`). `delta_pos0`/`delta_pos1 = None` is no-split.
pub fn build_bbox_csc_2d(
    pos0: &[PositionT],
    delta_pos0: Option<&[PositionT]>,
    pos1: &[PositionT],
    delta_pos1: Option<&[PositionT]>,
    mask: Option<&[i8]>,
    bins: (usize, usize),
    bounds: &Bbox2dBounds,
) -> (Csc, Vec<PositionT>, Vec<PositionT>) {
    let n_pixels = pos0.len();
    let (csr, c0, c1) = build_bbox_csr_2d(pos0, delta_pos0, pos1, delta_pos1, mask, bins, bounds);
    (csr_to_csc(&csr, bins.0 * bins.1, n_pixels), c0, c1)
}

/// Build the 1D full pixel-splitting CSC matrix and the unscaled bin centers —
/// the `("full", "csc", "cython")` build (`splitPixelFullCSC.FullSplitCSC_1d`).
/// `corners` is the `(npix, 4, 2)` array flattened to f64.
pub fn build_full_csc_1d(
    corners: &[PositionT],
    mask: Option<&[i8]>,
    bins: usize,
    allow_pos0_neg: bool,
    chi_disc_at_pi: bool,
    pos1_period: PositionT,
) -> (Csc, Vec<PositionT>) {
    let n_pixels = corners.len() / 8;
    let (csr, centers) = build_full_csr_1d(
        corners,
        mask,
        bins,
        allow_pos0_neg,
        chi_disc_at_pi,
        pos1_period,
    );
    (csr_to_csc(&csr, bins, n_pixels), centers)
}

/// Build the 2D full pixel-splitting CSC matrix and the unscaled radial / radian
/// azimuthal centers — the `("full", "csc", "cython")` 2D build
/// (`splitPixelFullCSC.FullSplitCSC_2d`).
pub fn build_full_csc_2d(
    corners: &[PositionT],
    mask: Option<&[i8]>,
    bins: (usize, usize),
    allow_pos0_neg: bool,
    chi_disc_at_pi: bool,
    pos1_period: PositionT,
) -> (Csc, Vec<PositionT>, Vec<PositionT>) {
    let n_pixels = corners.len() / 8;
    let (csr, c0, c1) = build_full_csr_2d(
        corners,
        mask,
        bins,
        allow_pos0_neg,
        chi_disc_at_pi,
        pos1_period,
    );
    (csr_to_csc(&csr, bins.0 * bins.1, n_pixels), c0, c1)
}
