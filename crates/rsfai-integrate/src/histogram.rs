//! Port of pyFAI's pure-Cython 1D histogram engine: `histogram_preproc` and
//! `histogram1d_engine` (`pyFAI/ext/histogram.pyx`), the `("no", "histogram",
//! "cython")` integration path.
//!
//! Unlike the per-pixel maps and the CSR apply, the histogram **accumulation**
//! sums many pixels into shared bins. It is parallelized with a rayon
//! fold/reduce into thread-local bin arrays, which reorders the f64 adds and is
//! therefore **not** bit-reproducible against pyFAI's serial golden. The f64
//! accumulators bound the reorder error at `~n·eps` (≈ 2e-10 relative for
//! ~1e6 pixels), so the golden gate for these outputs is relative error
//! `<= 1e-6`, not bitwise (see `doc/bit-exact-ladder.md`). The bin-center axes
//! (`position`) stay bit-exact — they derive from the order-independent
//! min/max + `linspace`.
//!
//! ## What the integrator feeds this engine
//!
//! `azimuthal.py` calls `histogram1d_engine(radial, npt, ...)` with `radial =
//! center_array(unit, scale=False)` — the **unscaled** radial — then reports
//! `result.position = intpl.position * unit.scale`. Binning therefore happens in
//! unscaled space; the caller scales only the reported bin centers. (For
//! `q_nm^-1`, `unit.scale == 1.0`, so unscaled == scaled bitwise.)
//!
//! ## dtype contract (decisive for bit-exactness)
//!
//! - `radial` / bin edges / accumulators: f64 (`position_t` / `acc_t`).
//! - preproc rows / outputs: f32 (`data_t`); the f64 accumulators are **downcast
//!   to f32** in the reduction (`sig = histo_signal[i] = res[i, 0]`).
//! - **norm² rounding forks on the error model.** `error_model == 0` accumulates
//!   `tmp*tmp` with `tmp` an f32 (`data_t`), i.e. an **f32 multiply** promoted to
//!   f64; the Variance/Poisson path accumulates `nrm*nrm` with `nrm` an f64
//!   (`acc_t`), an **f64 multiply**. Reproduced exactly below.
//! - `std`/`sem` use libc **double** `sqrt` (`from libc.math cimport sqrt`): the
//!   f32 operand is promoted to f64, `sqrt` runs in f64, and the f64 result is
//!   downcast to f32 — *not* `sqrtf`. This can differ from `sqrtf` by a ULP via
//!   double-rounding, so we match it exactly.

use rayon::prelude::*;
use rsfai_core::dtype::{calc_upper_bound, AccT, DataT, ErrorModel, PositionT};

/// The `Integrate1dtpl` fields (`pyFAI/containers.py`), in pyFAI's dtypes:
/// `position` is f64; everything else is f32 (downcast from the f64
/// accumulators in the final reduction). `sigma` and `sem` are the same array
/// in pyFAI (`Integrate1dtpl` positions 3 and 9 both hold `sem`).
#[derive(Debug, Clone, PartialEq)]
pub struct Integrate1d {
    /// Radial bin centers, `numpy.linspace(min0 + 0.5δ, max0 - 0.5δ, npt)` (f64,
    /// unscaled — multiply by `unit.scale` to get the reported position).
    pub position: Vec<PositionT>,
    /// Average intensity `signal / normalization` (f32), or `empty` where the
    /// bin has no normalization.
    pub intensity: Vec<DataT>,
    /// Standard error on the mean intensity (= `sem`; f32). `empty` unless an
    /// error model is active.
    pub sigma: Vec<DataT>,
    /// Histogram of `signal` (f32).
    pub signal: Vec<DataT>,
    /// Histogram of `variance` (f32).
    pub variance: Vec<DataT>,
    /// Histogram of `normalization` (f32).
    pub normalization: Vec<DataT>,
    /// Histogram of `count` (number of valid pixels per bin, f32).
    pub count: Vec<DataT>,
    /// Propagated standard deviation `sqrt(variance / norm²)` (f32). `empty`
    /// unless an error model is active.
    pub std: Vec<DataT>,
    /// Standard error on the mean `sqrt(variance) / normalization` (f32).
    /// `empty` unless an error model is active.
    pub sem: Vec<DataT>,
    /// Histogram of `normalization²` (f32).
    pub norm_sq: Vec<DataT>,
}

/// `numpy.linspace(start, stop, num)` (`endpoint=True`), reproduced bit-for-bit:
/// `y[j] = j*step + start` with `step = (stop-start)/(num-1)`, then `y[num-1]`
/// overwritten with `stop` exactly. Mirrors `numpy/_core/function_base.py`,
/// including its `step == 0` (denormal) fallback `y = (j/div)*delta + start`.
/// Shared with the CSR engine, which derives bin centers the same way.
pub(crate) fn numpy_linspace(start: f64, stop: f64, num: usize) -> Vec<f64> {
    let mut y = vec![0.0f64; num];
    if num == 0 {
        return y;
    }
    if num == 1 {
        // div == 0 path: `y = arange(1)*delta + start` = `0*delta + start`.
        y[0] = start;
        return y;
    }
    let div = (num - 1) as f64;
    let delta = stop - start;
    let step = delta / div;
    if step == 0.0 {
        for (j, yj) in y.iter_mut().enumerate() {
            *yj = (j as f64 / div) * delta + start;
        }
    } else {
        for (j, yj) in y.iter_mut().enumerate() {
            *yj = (j as f64) * step + start;
        }
    }
    y[num - 1] = stop; // endpoint override
    y
}

/// Port of `histogram_preproc` (`ext/histogram.pyx`): bin the preprocessed rows
/// into `npt` bins, returning the `(npt, 5)` f64 accumulator `[signal, variance,
/// normalization, count, norm²]` per bin and the f64 bin centers.
///
/// `radial` is the per-pixel radial position (f64, length `size`); `prep` is the
/// flat `[signal, variance, norm, count]`-per-pixel f32 array (length `4*size`),
/// exactly the `preproc(..., split_result=4)` output. `bin_range`, when `Some`,
/// pins `(min0, maxin0) = (min, max)` of the pair; otherwise they are the data
/// min/max. The accumulation is a rayon fold/reduce over thread-local bin
/// arrays (non-deterministic add order, see the module docs); validated at
/// relative error `<= 1e-6`, not bitwise.
pub fn histogram_preproc(
    radial: &[PositionT],
    prep: &[DataT],
    npt: usize,
    bin_range: Option<(PositionT, PositionT)>,
    error_model: ErrorModel,
) -> (Vec<[AccT; 5]>, Vec<PositionT>) {
    assert!(npt > 1, "bins must be > 1 (pyFAI: assert bins > 1)");
    let size = radial.len();
    assert_eq!(
        prep.len(),
        4 * size,
        "prep length must be 4 * radial.len() (nchan = 4)"
    );

    // Bin range: min(bin_range)/max(bin_range), else data min/max over cpos.
    let (min0, maxin0) = match bin_range {
        Some((lo, hi)) => (lo.min(hi), lo.max(hi)),
        None => {
            let mut min0 = radial[0];
            let mut maxin0 = radial[0];
            for &a in &radial[1..] {
                maxin0 = maxin0.max(a);
                min0 = min0.min(a);
            }
            (min0, maxin0)
        }
    };

    let max0 = calc_upper_bound(maxin0);
    let delta = (max0 - min0) / (npt as f64);

    // Parallel histogram: each rayon worker folds its pixel chunk into a private
    // `npt`-bin accumulator, then `reduce` merges the per-worker accumulators
    // element-wise. The add order across workers (and across each worker's
    // chunk split) is non-deterministic; validated at relative error <= 1e-6.
    let accumulate = |acc: &mut [[AccT; 5]], i: usize| {
        let a = radial[i];
        let fbin = (a - min0) / delta; // get_bin_number
        let bin = fbin as i64; // <Py_ssize_t>: truncate toward zero
        if bin < 0 || bin >= npt as i64 {
            return;
        }
        let bin = bin as usize;
        let s = prep[4 * i];
        let v = prep[4 * i + 1];
        let n = prep[4 * i + 2];
        let c = prep[4 * i + 3];
        let row = &mut acc[bin];
        match error_model {
            // error_model == 0: norm² via `tmp*tmp` with tmp an f32 (data_t).
            ErrorModel::No => {
                row[0] += s as AccT;
                row[1] += v as AccT;
                row[2] += n as AccT;
                row[4] += (n * n) as AccT; // f32 multiply, then promote
                row[3] += c as AccT;
            }
            // pyFAI histogram_preproc error_model==3: serial Welford (all f64).
            // No-split, so coef = 1: omega_b = nrm, sig_inc = signal,
            // b = signal/nrm in f64. Skip a zero-norm non-first contribution
            // (pyFAI's `if nrm`). The count is added after, like pyFAI.
            ErrorModel::Azimuthal => {
                let nrm = n as AccT;
                if row[4] <= 0.0 || nrm != 0.0 {
                    let sig = s as AccT;
                    let [sum_sig, sum_var, sum_norm, _cnt, sum_norm_sq] = &mut *row;
                    crate::azimuthal::azimuthal_step(
                        sum_sig,
                        sum_var,
                        sum_norm,
                        sum_norm_sq,
                        nrm,
                        sig,
                        sig / nrm,
                    );
                }
                row[3] += c as AccT;
            }
            // Variance / Poisson: norm² via `nrm*nrm` with nrm an f64 (acc_t).
            ErrorModel::Variance | ErrorModel::Poisson => {
                let sig = s as AccT;
                let var = v as AccT;
                let nrm = n as AccT;
                let cnt = c as AccT;
                row[0] += sig;
                row[2] += nrm;
                row[4] += nrm * nrm; // f64 multiply
                row[1] += var;
                row[3] += cnt;
            }
        }
    };

    let out_prop = if error_model == ErrorModel::Azimuthal {
        // The azimuthal Welford variance is order-dependent and NOT additively
        // mergeable (you cannot sum two partial in-bin variances), so it must
        // accumulate serially — matching pyFAI's serial `histogram_preproc`
        // `for i in range(size)`.
        let mut acc = vec![[0.0f64; 5]; npt];
        for i in 0..size {
            accumulate(&mut acc, i);
        }
        acc
    } else {
        (0..size)
            .into_par_iter()
            .fold(
                || vec![[0.0f64; 5]; npt],
                |mut acc, i| {
                    accumulate(&mut acc, i);
                    acc
                },
            )
            .reduce(|| vec![[0.0f64; 5]; npt], merge_bins)
    };

    let position = numpy_linspace(min0 + 0.5 * delta, max0 - 0.5 * delta, npt);
    (out_prop, position)
}

/// Element-wise add of two equal-length `[acc; 5]` bin accumulators — the
/// `reduce` combiner that merges per-worker partial histograms. Float addition
/// is not associative, so the merge order (chosen by rayon) makes the result
/// non-bit-reproducible; the error is bounded by `~n·eps` (see the module docs).
fn merge_bins(mut a: Vec<[AccT; 5]>, b: Vec<[AccT; 5]>) -> Vec<[AccT; 5]> {
    debug_assert_eq!(a.len(), b.len());
    for (ra, rb) in a.iter_mut().zip(b.iter()) {
        for k in 0..5 {
            ra[k] += rb[k];
        }
    }
    a
}

/// Port of `histogram1d_engine` (`ext/histogram.pyx`): run [`histogram_preproc`]
/// then the final per-bin reduction to intensity/sigma, producing the
/// [`Integrate1d`] (`Integrate1dtpl`) fields. `empty` is the fill value for bins
/// with no normalization (pyFAI's `self._empty`, default `0.0`).
pub fn histogram1d(
    radial: &[PositionT],
    prep: &[DataT],
    npt: usize,
    bin_range: Option<(PositionT, PositionT)>,
    error_model: ErrorModel,
    empty: DataT,
) -> Integrate1d {
    let (res, position) = histogram_preproc(radial, prep, npt, bin_range, error_model);

    // `bint do_variance = error_model`: true for any non-NO model.
    let do_variance = error_model != ErrorModel::No;

    let mut signal = vec![0.0f32; npt];
    let mut variance = vec![0.0f32; npt];
    let mut normalization = vec![0.0f32; npt];
    let mut norm_sq = vec![0.0f32; npt];
    let mut count = vec![0.0f32; npt];
    let mut intensity = vec![0.0f32; npt];
    let mut std = vec![0.0f32; npt];
    let mut sem = vec![0.0f32; npt];

    for i in 0..npt {
        // Downcast the f64 accumulators to f32 (`sig = histo_signal[i] = res[i,0]`).
        let sig = res[i][0] as DataT;
        let var = res[i][1] as DataT;
        let norm = res[i][2] as DataT;
        let cnt = res[i][3] as DataT;
        let norm2 = res[i][4] as DataT;
        signal[i] = sig;
        variance[i] = var;
        normalization[i] = norm;
        count[i] = cnt;
        norm_sq[i] = norm2;
        if norm2 > 0.0 {
            intensity[i] = sig / norm; // f32 division
            if do_variance {
                // libc double sqrt: f32 operand -> f64 -> sqrt -> downcast f32.
                std[i] = ((var / norm2) as f64).sqrt() as DataT;
                sem[i] = ((var as f64).sqrt() / norm as f64) as DataT;
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

    Integrate1d {
        position,
        intensity,
        sigma: sem.clone(), // Integrate1dtpl position 3 (sigma) == position 9 (sem)
        signal,
        variance,
        normalization,
        count,
        std,
        sem,
        norm_sq,
    }
}

// ---------------------------------------------------------------------------
// 2D histogram (no split): port of `histogram2d_engine` (`ext/histogram.pyx`).
// Each pixel centre is binned into one (radial, azimuthal) cell. The engine
// differs from the 1D path in three ways that matter for bit-exactness:
//   * the reduction keys on `count > 0` (not `norm² > 0`);
//   * `intensity = signal/norm`, `sem = sqrt(var)/norm`, `std = sqrt(var/norm²)`
//     are computed in f64 (the accumulators stay `acc_t`) then downcast to f32;
//   * the binned sums (signal/variance/normalization/count/norm²) are exposed at
//     full f64 (`out_data[...,k]`), NOT downcast to f32 like the 1D histogram;
//   * norm² accumulates `value.norm·value.norm` as an **f32 multiply** promoted
//     to f64 (`update_2d_accumulator`), unconditionally — no error-model fork.

/// Boundary / binning configuration for [`histogram2d`], grouping the engine's
/// scalar parameters (the data arrays are passed separately).
#[derive(Debug, Clone)]
pub struct Hist2dOptions {
    /// Number of bins `(radial, azimuthal)`.
    pub bins: (usize, usize),
    /// Explicit radial `(min, max)`; `None` uses the data min/max.
    pub radial_range: Option<(PositionT, PositionT)>,
    /// Explicit azimuthal `(min, max)`; `None` uses the (clipped) data min/max.
    pub azimuth_range: Option<(PositionT, PositionT)>,
    /// Error model: `No` skips the variance branch (std/sem stay 0).
    pub error_model: ErrorModel,
    /// Allow the radial axis below 0 (false clamps min/max to `>= 0`).
    pub allow_radial_neg: bool,
    /// Azimuthal discontinuity at π (true) vs 0/2π (false) — sets the clip range.
    pub chi_disc_at_pi: bool,
    /// Azimuthal period; `> 0` turns on the `[-π, π]` clip (the only use here).
    pub pos1_period: PositionT,
    /// Fill value for cells with no counts (pyFAI's `empty`, default `0.0`).
    pub empty: DataT,
}

/// The `Integrate2dtpl` fields (`pyFAI/containers.py`). The 2D arrays are stored
/// flat in **(azimuthal, radial)** row-major order — the layout pyFAI exposes
/// after its `.T` transpose — so cell `(azimuthal j, radial i)` is at index
/// `j * bins.0 + i`. `signal`/`variance`/`normalization`/`count`/`norm_sq` are
/// f64 (`acc_t`, not downcast); `intensity`/`sigma`/`std`/`sem` are f32.
/// `radial`/`azimuthal` are the **unscaled** bin centers (multiply by the radial
/// / azimuthal `unit.scale` for the reported axes).
#[derive(Debug, Clone, PartialEq)]
pub struct Integrate2d {
    /// Unscaled radial bin centers, length `bins.0`.
    pub radial: Vec<PositionT>,
    /// Unscaled azimuthal bin centers, length `bins.1`.
    pub azimuthal: Vec<PositionT>,
    /// `(radial, azimuthal)` bin counts, for indexing the flat arrays.
    pub bins: (usize, usize),
    /// Average intensity `signal/normalization` (f32), or `empty`.
    pub intensity: Vec<DataT>,
    /// Standard error on the mean (= `sem`; f32).
    pub sigma: Vec<DataT>,
    /// Binned `signal` (f64).
    pub signal: Vec<AccT>,
    /// Binned `variance` (f64).
    pub variance: Vec<AccT>,
    /// Binned `normalization` (f64).
    pub normalization: Vec<AccT>,
    /// Binned `count` (f64).
    pub count: Vec<AccT>,
    /// Propagated std `sqrt(variance / norm²)` (f32).
    pub std: Vec<DataT>,
    /// Standard error on the mean `sqrt(variance) / normalization` (f32).
    pub sem: Vec<DataT>,
    /// Binned `normalization²` (f64).
    pub norm_sq: Vec<AccT>,
}

/// 2D radial/azimuthal boundaries — port of the bbox `calc_boundaries`
/// (`ext/splitBBox_common.pyx`) with `delta = None` (centers only, no split).
/// Folds the per-pixel centers over the unmasked pixels, clamps the radial axis
/// to `>= 0` unless `allow_radial_neg`, clips the azimuthal axis to `[-π, π]`
/// (or `[0, 2π]`) when `pos1_period > 0`, then applies any explicit ranges.
/// The clip bound uses **f32 π** (`float pi = <float> M_PI` in pyFAI), widened
/// to f64. Returns `(pos0_min, pos0_maxin, pos1_min, pos1_maxin)`; the caller
/// applies [`calc_upper_bound`] to the `*_maxin` values.
fn calc_boundaries_2d(
    radial: &[PositionT],
    azimuthal: &[PositionT],
    mask: Option<&[i8]>,
    opts: &Hist2dOptions,
) -> (PositionT, PositionT, PositionT, PositionT) {
    let mut pos0_min = PositionT::INFINITY;
    let mut pos0_max = PositionT::NEG_INFINITY;
    let mut pos1_min = PositionT::INFINITY;
    let mut pos1_max = PositionT::NEG_INFINITY;
    for idx in 0..radial.len() {
        if let Some(m) = mask {
            if m[idx] != 0 {
                continue;
            }
        }
        let c0 = radial[idx];
        pos0_max = pos0_max.max(c0);
        pos0_min = pos0_min.min(c0);
        let c1 = azimuthal[idx];
        pos1_max = pos1_max.max(c1);
        pos1_min = pos1_min.min(c1);
    }
    if !opts.allow_radial_neg {
        pos0_min = pos0_min.max(0.0);
        pos0_max = pos0_max.max(0.0);
    }
    if opts.pos1_period > 0.0 {
        // pyFAI: pos1_max = min(pos1_max, (2 - chiDiscAtPi) * pi), with `pi` an
        // f32 and chiDiscAtPi an int; the product is evaluated in f32.
        let cd: i32 = if opts.chi_disc_at_pi { 1 } else { 0 };
        let pi32 = std::f32::consts::PI;
        let max_bound = ((2 - cd) as f32 * pi32) as PositionT;
        let min_bound = (-(cd as f32) * pi32) as PositionT;
        pos1_max = pos1_max.min(max_bound);
        pos1_min = pos1_min.max(min_bound);
    }
    if let Some((lo, hi)) = opts.radial_range {
        pos0_min = lo.min(hi);
        pos0_max = lo.max(hi);
    }
    if let Some((lo, hi)) = opts.azimuth_range {
        pos1_min = lo.min(hi);
        pos1_max = lo.max(hi);
    }
    (pos0_min, pos0_max, pos1_min, pos1_max)
}

/// Port of `histogram2d_engine` (`ext/histogram.pyx`): bin the preprocessed rows
/// into a `(bins.0, bins.1)` = `(radial, azimuthal)` grid and reduce to the
/// [`Integrate2d`] fields. `radial`/`azimuthal` are the per-pixel unscaled
/// centers (f64, length `size`); `prep` is the flat `[signal, variance, norm,
/// count]`-per-pixel f32 array (`preproc(..., split_result=4)`). Masked pixels
/// (`mask[i] != 0`) are skipped. Serial accumulation in pixel order — the only
/// bit-reproducible order.
pub fn histogram2d(
    radial: &[PositionT],
    azimuthal: &[PositionT],
    prep: &[DataT],
    mask: Option<&[i8]>,
    opts: &Hist2dOptions,
) -> Integrate2d {
    let (bins0, bins1) = opts.bins;
    assert!(bins0 >= 1 && bins1 >= 1, "bins must be >= 1 in each dim");
    let size = radial.len();
    assert_eq!(azimuthal.len(), size, "azimuthal length must match radial");
    assert_eq!(
        prep.len(),
        4 * size,
        "prep length must be 4 * size (nchan = 4)"
    );
    if let Some(m) = mask {
        assert_eq!(m.len(), size, "mask length mismatch");
    }

    let (pos0_min, pos0_maxin, pos1_min, pos1_maxin) =
        calc_boundaries_2d(radial, azimuthal, mask, opts);
    let pos0_max = calc_upper_bound(pos0_maxin);
    let pos1_max = calc_upper_bound(pos1_maxin);
    let delta0 = (pos0_max - pos0_min) / (bins0 as PositionT);
    let delta1 = (pos1_max - pos1_min) / (bins1 as PositionT);

    // Accumulator grid in (radial, azimuthal) order: cell (i, j) at i*bins1 + j.
    // Parallel histogram via thread-local grids merged by `merge_bins` (the same
    // non-deterministic fold/reduce as the 1D engine; validated at rel <= 1e-6).
    let n_grid = bins0 * bins1;
    let accumulate = |acc: &mut [[AccT; 5]], idx: usize| {
        if let Some(m) = mask {
            if m[idx] != 0 {
                return;
            }
        }
        let fbin0 = (radial[idx] - pos0_min) / delta0; // get_bin_number
        let fbin1 = (azimuthal[idx] - pos1_min) / delta1;
        let bin0 = fbin0 as i64; // <Py_ssize_t>: truncate toward zero
        let bin1 = fbin1 as i64;
        if bin0 < 0 || bin0 >= bins0 as i64 || bin1 < 0 || bin1 >= bins1 as i64 {
            return;
        }
        let n = prep[4 * idx + 2];
        let cell = &mut acc[bin0 as usize * bins1 + bin1 as usize];
        // update_2d_accumulator with weight 1.0 (w2 = 1.0).
        cell[0] += prep[4 * idx] as AccT;
        cell[1] += prep[4 * idx + 1] as AccT;
        cell[2] += n as AccT;
        cell[3] += prep[4 * idx + 3] as AccT;
        cell[4] += (n * n) as AccT; // f32 multiply, then promote
    };
    let out = (0..size)
        .into_par_iter()
        .fold(
            || vec![[0.0f64; 5]; n_grid],
            |mut acc, idx| {
                accumulate(&mut acc, idx);
                acc
            },
        )
        .reduce(|| vec![[0.0f64; 5]; n_grid], merge_bins);

    let radial_centers = numpy_linspace(pos0_min + 0.5 * delta0, pos0_max - 0.5 * delta0, bins0);
    let azim_centers = numpy_linspace(pos1_min + 0.5 * delta1, pos1_max - 0.5 * delta1, bins1);
    reduce_2d(
        &out,
        (bins0, bins1),
        radial_centers,
        azim_centers,
        opts.error_model,
        opts.empty,
    )
}

/// Final per-cell reduction shared by every 2D engine (`histogram2d_engine` and
/// the direct-split `histoBBox2d_engine` / `fullSplit2D_engine`): the bins are
/// already a `(bins.0, bins.1)` = (radial, azimuthal) row-major accumulator grid
/// (`cell (i, j)` at `i*bins.1 + j`). It transposes to **(azimuthal, radial)**
/// — the layout pyFAI exposes via `.T` — guards on **count** (`cnt > 0`, which
/// equals pyFAI's `if cnt:` for the non-negative counts here), and computes
/// `intensity = sig/norm`, `sem = sqrt(var)/norm`, `std = sqrt(var/norm²)` in f64
/// (libc double sqrt) then downcasts to f32. The binned sums stay f64.
pub(crate) fn reduce_2d(
    out: &[[AccT; 5]],
    bins: (usize, usize),
    radial_centers: Vec<PositionT>,
    azim_centers: Vec<PositionT>,
    error_model: ErrorModel,
    empty: DataT,
) -> Integrate2d {
    let (bins0, bins1) = bins;
    let do_variance = error_model != ErrorModel::No;
    let n_cells = bins0 * bins1;
    let mut signal = vec![0.0f64; n_cells];
    let mut variance = vec![0.0f64; n_cells];
    let mut normalization = vec![0.0f64; n_cells];
    let mut count = vec![0.0f64; n_cells];
    let mut norm_sq = vec![0.0f64; n_cells];
    let mut intensity = vec![0.0f32; n_cells];
    let mut std = vec![0.0f32; n_cells];
    let mut sem = vec![0.0f32; n_cells];

    for i in 0..bins0 {
        for j in 0..bins1 {
            let cell = out[i * bins1 + j];
            let (sig, var, norm, cnt, norm2) = (cell[0], cell[1], cell[2], cell[3], cell[4]);
            // Transpose to (azimuthal, radial), the layout pyFAI exposes.
            let t = j * bins0 + i;
            signal[t] = sig;
            variance[t] = var;
            normalization[t] = norm;
            count[t] = cnt;
            norm_sq[t] = norm2;
            if cnt > 0.0 {
                intensity[t] = (sig / norm) as DataT; // f64 divide, downcast
                if do_variance {
                    // libc double sqrt on f64 accumulators, then downcast.
                    sem[t] = (var.sqrt() / norm) as DataT;
                    std[t] = (var / norm2).sqrt() as DataT;
                }
            } else {
                intensity[t] = empty;
                if do_variance {
                    sem[t] = empty;
                    std[t] = empty;
                }
            }
        }
    }

    Integrate2d {
        radial: radial_centers,
        azimuthal: azim_centers,
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
    fn linspace_matches_numpy_endpoint() {
        // numpy.linspace(0.0, 1.0, 5) == [0, 0.25, 0.5, 0.75, 1.0], last exact.
        let y = numpy_linspace(0.0, 1.0, 5);
        assert_eq!(y, vec![0.0, 0.25, 0.5, 0.75, 1.0]);
        assert_eq!(y[4], 1.0); // endpoint override is exact
    }

    #[test]
    fn single_pixel_lands_in_first_bin() {
        // One valid pixel at the minimum radial -> fbin = 0 -> bin 0.
        // prep row [signal=10, variance=0, norm=2, count=1].
        let radial = [1.0f64, 2.0];
        let prep = [10.0f32, 0.0, 2.0, 1.0, 5.0, 0.0, 1.0, 1.0];
        let r = histogram1d(&radial, &prep, 4, None, ErrorModel::No, 0.0);
        // bin 0 gets the first pixel; signal 10, norm 2, count 1; intensity 5.
        assert_eq!(r.signal[0], 10.0);
        assert_eq!(r.normalization[0], 2.0);
        assert_eq!(r.count[0], 1.0);
        assert_eq!(r.intensity[0], 5.0);
        // norm² accumulated as f32 multiply: 2*2 = 4.
        assert_eq!(r.norm_sq[0], 4.0);
    }

    #[test]
    fn empty_bins_get_empty_fill() {
        // Two pixels both at the extremes; middle bins stay empty -> `empty`.
        let radial = [0.0f64, 10.0];
        let prep = [1.0f32, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0];
        let empty = -1.0f32;
        let r = histogram1d(&radial, &prep, 5, None, ErrorModel::No, empty);
        // Some interior bin has no pixels -> intensity == empty.
        assert!(r.intensity.contains(&empty));
    }

    #[test]
    fn no_error_model_zeroes_sigma() {
        let radial = [1.0f64, 2.0, 3.0];
        let prep = [
            5.0f32, 0.0, 1.0, 1.0, 6.0, 0.0, 1.0, 1.0, 7.0, 0.0, 1.0, 1.0,
        ];
        let r = histogram1d(&radial, &prep, 4, None, ErrorModel::No, 0.0);
        // do_variance == false -> sigma/std/sem all the empty value (0.0 here).
        assert!(r.sigma.iter().all(|&v| v == 0.0));
        assert!(r.std.iter().all(|&v| v == 0.0));
        assert!(r.sem.iter().all(|&v| v == 0.0));
    }
}
