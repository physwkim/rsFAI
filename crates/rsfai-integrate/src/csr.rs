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

    // Flatten to CSR in bin order, within-bin in insertion (ascending) order.
    let mut indptr = vec![0 as IndexT; bins + 1];
    let nnz: usize = bin_idx.iter().map(|v| v.len()).sum();
    let mut indices = Vec::with_capacity(nnz);
    let mut data = Vec::with_capacity(nnz);
    for b in 0..bins {
        indices.extend_from_slice(&bin_idx[b]);
        data.extend_from_slice(&bin_coef[b]);
        indptr[b + 1] = indptr[b] + bin_idx[b].len() as IndexT;
    }

    let bin_centers = numpy_linspace(pos0_min + 0.5 * delta, pos0_max - 0.5 * delta, bins);
    (
        Csr {
            data,
            indices,
            indptr,
        },
        bin_centers,
    )
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
