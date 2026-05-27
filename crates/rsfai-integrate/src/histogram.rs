//! Port of pyFAI's pure-Cython 1D histogram engine: `histogram_preproc` and
//! `histogram1d_engine` (`pyFAI/ext/histogram.pyx`), the `("no", "histogram",
//! "cython")` integration path. This is the **Tier-A** correctness gate for the
//! histogram engine: given the identical radial array and preprocessed
//! `(signal, variance, norm, count)` rows pyFAI binned, every output field must
//! be bit-exact (single-thread; `histogram_preproc` is serial, never `prange`).
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
fn numpy_linspace(start: f64, stop: f64, num: usize) -> Vec<f64> {
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
/// min/max. The accumulation loop is serial in pixel order — the only
/// bit-reproducible order.
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

    let mut out_prop = vec![[0.0f64; 5]; npt];
    for i in 0..size {
        let a = radial[i];
        let fbin = (a - min0) / delta; // get_bin_number
        let bin = fbin as i64; // <Py_ssize_t>: truncate toward zero
        if bin < 0 || bin >= npt as i64 {
            continue;
        }
        let bin = bin as usize;
        let s = prep[4 * i];
        let v = prep[4 * i + 1];
        let n = prep[4 * i + 2];
        let c = prep[4 * i + 3];
        let row = &mut out_prop[bin];
        match error_model {
            // error_model == 0: norm² via `tmp*tmp` with tmp an f32 (data_t).
            ErrorModel::No => {
                row[0] += s as AccT;
                row[1] += v as AccT;
                row[2] += n as AccT;
                row[4] += (n * n) as AccT; // f32 multiply, then promote
                row[3] += c as AccT;
            }
            // Welford online variance — deferred until a golden exercises it.
            ErrorModel::Azimuthal => {
                unimplemented!("azimuthal (Welford) histogram variance not yet ported")
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
    }

    let position = numpy_linspace(min0 + 0.5 * delta, max0 - 0.5 * delta, npt);
    (out_prop, position)
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
