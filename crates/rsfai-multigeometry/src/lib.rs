//! Multi-geometry azimuthal integration — a Rust port of pyFAI's
//! [`MultiGeometry`](https://github.com/silx-kit/pyFAI/blob/main/src/pyFAI/multi_geometry.py).
//!
//! One scattering pattern is recorded across several detector geometries (e.g. a
//! detector on a goniometer arm); each frame is integrated on its own
//! [`AzimuthalIntegrator`] into a **shared** radial/azimuth grid, then the
//! per-geometry results are combined by a **sequential weighted union** into a
//! single curve. This is pure orchestration over [`rsfai`]'s bit-exact
//! `integrate1d_with`/`integrate2d_with` plus the `union` + `__recalculate_means__`
//! arithmetic from pyFAI's `containers.py`.
//!
//! ## Bit-exactness
//! Every step is constructible bit-exact vs single-thread pyFAI:
//! - the per-geometry monitor scaling `monitor · pixel1·pixel2 / dist²` is done in
//!   f64 (`dist.powf(2.0)` mirrors Python's `dist ** 2` = C `pow`), cast to f32
//!   only at the preproc boundary by `integrate*_with`;
//! - the common range is `(min, max)` over the per-geometry scaled center arrays —
//!   order-independent, hence bit-exact;
//! - the **left-fold union order** matches pyFAI (`results[0]`, then `union` each
//!   subsequent in list order); the per-bin f64 sums add in that same order.
//!
//! Only `chi_disc = 180` (the pyFAI default, `chiDiscAtPi`) is supported — the
//! `integrate*_with` path is hard-wired to `chi_disc_at_pi = true`; `chi_disc = 0`
//! would need that plumbed per call and has no golden yet.

use rsfai::{
    AzimuthalIntegrator, Corrections, ErrorModelKind, Integrate1dResult, Integrate2dResult,
    IntegrationOptions, Method, RadialUnit,
};

/// One detector frame plus its per-geometry corrections, paired positionally with
/// `MultiGeometry::ais[i]`. pyFAI's `MultiGeometry` carries no per-geometry dark
/// (the frames are assumed dark-corrected), so [`Corrections::dark`] stays `None`.
#[derive(Debug, Clone, Copy)]
pub struct GeometryFrame<'a> {
    /// Detector frame, already cast to f32 (preproc's `data_t`).
    pub data: &'a [f32],
    /// Per-pixel variance for the VARIANCE error model (`lst_variance[i]`).
    pub variance: Option<&'a [f32]>,
    /// User mask (nonzero ⇒ bad); replaces the detector mask (`lst_mask[i]`).
    pub mask: Option<&'a [i8]>,
    /// Flat field (`lst_flat[i]`).
    pub flat: Option<&'a [f32]>,
    /// Normalization monitor (`normalization_factor[i]`); 1.0 if unmonitored.
    pub monitor: f64,
}

impl<'a> GeometryFrame<'a> {
    /// A frame with only image data (no mask/flat/variance, unit monitor).
    pub fn new(data: &'a [f32]) -> Self {
        GeometryFrame {
            data,
            variance: None,
            mask: None,
            flat: None,
            monitor: 1.0,
        }
    }
}

/// Per-call integration options shared across all geometries (pyFAI's
/// `MultiGeometry.integrate1d/2d` keyword arguments that are not per-geometry).
#[derive(Debug, Clone, Copy)]
pub struct MultiIntegrationOptions {
    /// Correct for solid angle. When `true`, the per-geometry monitor is scaled by
    /// `pixel1·pixel2/dist²` so all geometries share one absolute solid-angle
    /// normalization (pyFAI's note: "all processing is then done in absolute solid
    /// angle").
    pub correct_solid_angle: bool,
    /// Error model, identical for every geometry (`union` requires matching models).
    pub error_model: ErrorModelKind,
    /// Polarization factor (`None` ⇒ no polarization correction).
    pub polarization_factor: Option<f64>,
    /// Integration method (split/algo). pyFAI's default is `("full","histogram")`.
    pub method: Method,
}

/// A set of azimuthal integrators sharing a radial/azimuth grid, integrated and
/// combined as one. Port of pyFAI's `MultiGeometry`.
#[derive(Debug, Clone)]
pub struct MultiGeometry {
    /// The per-geometry integrators.
    pub ais: Vec<AzimuthalIntegrator>,
    /// Radial unit of the common grid.
    pub radial_unit: RadialUnit,
    /// Azimuth unit of the common grid (pyFAI's default `CHI_DEG`).
    pub azimuth_unit: RadialUnit,
    /// Common radial range in the **scaled** `radial_unit` (guessed from the data
    /// when `None`, like pyFAI's `_guess_radial_range`).
    pub radial_range: Option<(f64, f64)>,
    /// Common azimuth range in **degrees** (guessed when `None`).
    pub azimuth_range: Option<(f64, f64)>,
    /// Value for empty bins (`sum_normalization == 0`); pyFAI default 0.0.
    pub empty: f64,
}

impl MultiGeometry {
    /// Build a multi-geometry integrator over `ais`, binning in `radial_unit`
    /// with the azimuth in `CHI_DEG` and auto-guessed ranges (pyFAI's defaults
    /// `azimuth_unit=CHI_DEG`, `radial_range=azimuth_range=None`, `empty=0.0`,
    /// `chi_disc=180`).
    pub fn new(ais: Vec<AzimuthalIntegrator>, radial_unit: RadialUnit) -> Self {
        MultiGeometry {
            ais,
            radial_unit,
            azimuth_unit: RadialUnit::CHI_DEG,
            radial_range: None,
            azimuth_range: None,
            empty: 0.0,
        }
    }

    /// `min`/`max` over the concatenated per-geometry scaled radial center arrays
    /// (pyFAI `_guess_radial_range`). Order-independent ⇒ bit-exact.
    fn guess_radial_range(&self) -> (f64, f64) {
        guess_range(&self.ais, self.radial_unit)
    }

    /// `min`/`max` over the concatenated per-geometry scaled azimuth center arrays
    /// (pyFAI `_guess_azimuth_range`).
    fn guess_azimuth_range(&self) -> (f64, f64) {
        guess_range(&self.ais, self.azimuth_unit)
    }

    /// The common radial range the integrators bin on: the explicit
    /// [`radial_range`](Self::radial_range) if set, else the cross-geometry guess.
    /// pyFAI exposes the same value as `MultiGeometry.radial_range` after the first
    /// integrate (it caches the guess onto the instance).
    pub fn effective_radial_range(&self) -> (f64, f64) {
        self.radial_range
            .unwrap_or_else(|| self.guess_radial_range())
    }

    /// The common azimuth range (degrees) the integrators bin on: the explicit
    /// [`azimuth_range`](Self::azimuth_range) if set, else the cross-geometry guess
    /// (pyFAI's `MultiGeometry.azimuth_range`).
    pub fn effective_azimuth_range(&self) -> (f64, f64) {
        self.azimuth_range
            .unwrap_or_else(|| self.guess_azimuth_range())
    }

    fn base_options(&self, opts: &MultiIntegrationOptions) -> IntegrationOptions {
        IntegrationOptions {
            correct_solid_angle: opts.correct_solid_angle,
            polarization_factor: opts.polarization_factor,
            // Each geometry's monitor is carried via `Corrections`; this scalar is
            // unused on the `integrate*_with` path.
            normalization_factor: 1.0,
            error_model: opts.error_model,
            method: opts.method,
            radial_range: Some(
                self.radial_range
                    .unwrap_or_else(|| self.guess_radial_range()),
            ),
            azimuth_range: Some(
                self.azimuth_range
                    .unwrap_or_else(|| self.guess_azimuth_range()),
            ),
        }
    }

    fn corrections<'a>(
        ai: &AzimuthalIntegrator,
        f: &GeometryFrame<'a>,
        correct_solid_angle: bool,
    ) -> Corrections<'a> {
        // pyFAI: `if correctSolidAngle: monitor *= pixel1*pixel2/dist**2` in f64.
        // `dist**2` is Python `float ** 2` = C `pow`, so `powf(2.0)` (not `dist*dist`).
        let monitor = if correct_solid_angle {
            f.monitor * (ai.detector.pixel1 * ai.detector.pixel2 / ai.dist.powf(2.0))
        } else {
            f.monitor
        };
        Corrections {
            dark: None,
            flat: f.flat,
            mask: f.mask,
            variance: f.variance,
            normalization_factor: monitor,
        }
    }

    /// 1D multi-geometry integration: integrate each frame on its geometry into
    /// the shared `npt`-bin grid, then left-fold the per-geometry results with the
    /// weighted [`union`] (pyFAI `MultiGeometry.integrate1d`).
    pub fn integrate1d(
        &self,
        frames: &[GeometryFrame],
        npt: usize,
        opts: &MultiIntegrationOptions,
    ) -> Integrate1dResult {
        assert!(!frames.is_empty(), "list of frames cannot be empty");
        assert_eq!(
            frames.len(),
            self.ais.len(),
            "frames ({}) must match geometries ({})",
            frames.len(),
            self.ais.len()
        );
        let io = self.base_options(opts);
        let mut it = self.ais.iter().zip(frames);
        let (ai0, f0) = it.next().unwrap();
        let mut acc = ai0.integrate1d_with(
            f0.data,
            npt,
            self.radial_unit,
            &io,
            &Self::corrections(ai0, f0, opts.correct_solid_angle),
        );
        for (ai, f) in it {
            let r = ai.integrate1d_with(
                f.data,
                npt,
                self.radial_unit,
                &io,
                &Self::corrections(ai, f, opts.correct_solid_angle),
            );
            acc = union1d(&acc, &r, opts.error_model, self.empty);
        }
        acc
    }

    /// 2D multi-geometry integration (pyFAI `MultiGeometry.integrate2d`).
    pub fn integrate2d(
        &self,
        frames: &[GeometryFrame],
        npt_rad: usize,
        npt_azim: usize,
        opts: &MultiIntegrationOptions,
    ) -> Integrate2dResult {
        assert!(!frames.is_empty(), "list of frames cannot be empty");
        assert_eq!(
            frames.len(),
            self.ais.len(),
            "frames ({}) must match geometries ({})",
            frames.len(),
            self.ais.len()
        );
        let io = self.base_options(opts);
        let mut it = self.ais.iter().zip(frames);
        let (ai0, f0) = it.next().unwrap();
        let mut acc = ai0.integrate2d_with(
            f0.data,
            npt_rad,
            npt_azim,
            self.radial_unit,
            &io,
            &Self::corrections(ai0, f0, opts.correct_solid_angle),
        );
        for (ai, f) in it {
            let r = ai.integrate2d_with(
                f.data,
                npt_rad,
                npt_azim,
                self.radial_unit,
                &io,
                &Self::corrections(ai, f, opts.correct_solid_angle),
            );
            acc = union2d(&acc, &r, opts.error_model, self.empty);
        }
        acc
    }
}

/// `min`/`max` over the concatenation of every geometry's scaled center array for
/// `unit` (pyFAI `_guess_radial_range`/`_guess_azimuth_range`).
fn guess_range(ais: &[AzimuthalIntegrator], unit: RadialUnit) -> (f64, f64) {
    let mut lo = f64::INFINITY;
    let mut hi = f64::NEG_INFINITY;
    for ai in ais {
        for v in ai.array_from_unit(unit) {
            if v < lo {
                lo = v;
            }
            if v > hi {
                hi = v;
            }
        }
    }
    (lo, hi)
}

/// Borrowed view of one result's five f64 accumulators — the operands of
/// [`union_sums`]. 1D and 2D results expose the same five arrays (flat), so one
/// view type serves both.
struct SumView<'a> {
    signal: &'a [f64],
    norm: &'a [f64],
    count: &'a [f64],
    variance: &'a [f64],
    norm2: &'a [f64],
}

impl<'a> SumView<'a> {
    fn of1d(r: &'a Integrate1dResult) -> Self {
        SumView {
            signal: &r.sum_signal,
            norm: &r.sum_normalization,
            count: &r.count,
            variance: &r.sum_variance,
            norm2: &r.sum_normalization2,
        }
    }
    fn of2d(r: &'a Integrate2dResult) -> Self {
        SumView {
            signal: &r.sum_signal,
            norm: &r.sum_normalization,
            count: &r.count,
            variance: &r.sum_variance,
            norm2: &r.sum_normalization2,
        }
    }
}

/// The combined f64 accumulators produced by [`union_sums`].
struct CombinedSums {
    signal: Vec<f64>,
    norm: Vec<f64>,
    count: Vec<f64>,
    variance: Vec<f64>,
    norm2: Vec<f64>,
}

/// The f32 outputs of [`recalculate`]. `errors` is `Some((sigma, std, sem))`
/// when the error model carries variance, else `None` — leaving the caller to
/// keep `results[0]`'s values (pyFAI's `deepcopy` + the `sum_variance is None`
/// short-circuit in `__recalculate_means__`).
struct Recalculated {
    intensity: Vec<f32>,
    errors: Option<(Vec<f32>, Vec<f32>, Vec<f32>)>,
}

/// pyFAI `IntegrateResult.union` (`containers.py:367`) — the per-bin f64
/// accumulator combine.
///
/// Two **independent** gates, exactly as pyFAI separates them:
/// * `em != No` gates the variance/norm² *sum* (pyFAI's `sum_variance is None ⇒
///   skip`, containers.py:381-386) — applies in both 1D and 2D.
/// * `apply_crossed` gates the AZIMUTHAL *crossed term* (containers.py:387-396),
///   which pyFAI guards with `if self.error_model == ErrorModel.AZIMUTHAL` — i.e.
///   on the **per-geometry result's** `error_model`, NOT the requested one. That
///   field is set only by `integrate1d_ng` (`_set_error_model`, azimuthal.py:455);
///   `integrate2d_ng` never sets it, so a 2D result carries `error_model = None`
///   and its `union` silently drops the crossed term. Hence the caller decides:
///   union1d ⇒ `em == Azimuthal`, union2d ⇒ always `false`.
fn union_sums(a: &SumView, b: &SumView, em: ErrorModelKind, apply_crossed: bool) -> CombinedSums {
    let n = a.signal.len();
    let do_variance = em != ErrorModelKind::No;
    let azimuthal = apply_crossed;
    let mut signal = vec![0.0; n];
    let mut norm = vec![0.0; n];
    let mut count = vec![0.0; n];
    let mut variance = vec![0.0; n];
    let mut norm2 = vec![0.0; n];
    for i in 0..n {
        signal[i] = a.signal[i] + b.signal[i];
        norm[i] = a.norm[i] + b.norm[i];
        count[i] = a.count[i] + b.count[i];
        if do_variance {
            variance[i] = a.variance[i] + b.variance[i];
            norm2[i] = a.norm2[i] + b.norm2[i];
            if azimuthal {
                // Crossed term (containers.py:387). `delta**2` is a numpy array
                // power-of-2 ⇒ multiply (`delta*delta`), NOT `pow`.
                let delta = a.signal[i] * b.norm[i] - b.signal[i] * a.norm[i];
                let denom = norm[i] * a.norm[i] * b.norm[i];
                let ratio_minor = if b.norm2[i] < a.norm2[i] {
                    b.norm2[i] / b.norm[i]
                } else {
                    a.norm2[i] / a.norm[i]
                };
                // numpy.divide(..., out=denom, where=denom!=0): the addend is the
                // ratio where denom!=0, else denom's original value (0 there).
                if denom != 0.0 {
                    variance[i] += ratio_minor * (delta * delta) / denom;
                }
            }
        }
    }
    CombinedSums {
        signal,
        norm,
        count,
        variance,
        norm2,
    }
}

/// pyFAI `IntegrateResult.__recalculate_means__` (`containers.py:329`), applied to
/// the combined f64 accumulators. `intensity = where(norm==0, dummy, signal/norm)`;
/// when `do_variance`, also `sem = where(norm==0, dummy, sqrt(variance)/norm)`,
/// `std = where(norm==0, dummy, sqrt(variance/norm2))`, and `sigma = sem`. All
/// computed in f64 then cast to f32 (numexpr stores into the f32 output arrays).
fn recalculate(sums: &CombinedSums, dummy: f64, do_variance: bool) -> Recalculated {
    let dummy32 = dummy as f32;
    let n = sums.signal.len();
    let mut intensity = vec![0.0f32; n];
    for (i, out) in intensity.iter_mut().enumerate() {
        let norm = sums.norm[i];
        *out = if norm == 0.0 {
            dummy32
        } else {
            (sums.signal[i] / norm) as f32
        };
    }
    if !do_variance {
        return Recalculated {
            intensity,
            errors: None,
        };
    }
    let mut sigma = vec![0.0f32; n];
    let mut std = vec![0.0f32; n];
    let mut sem = vec![0.0f32; n];
    for i in 0..n {
        let norm = sums.norm[i];
        if norm == 0.0 {
            sem[i] = dummy32;
            std[i] = dummy32;
        } else {
            sem[i] = (sums.variance[i].sqrt() / norm) as f32;
            std[i] = (sums.variance[i] / sums.norm2[i]).sqrt() as f32;
        }
        sigma[i] = sem[i];
    }
    Recalculated {
        intensity,
        errors: Some((sigma, std, sem)),
    }
}

/// Combine two 1D results (pyFAI `union` + `__recalculate_means__`). The
/// `radial` axis is shared (identical range + npt) so it is taken from `a`.
fn union1d(
    a: &Integrate1dResult,
    b: &Integrate1dResult,
    em: ErrorModelKind,
    dummy: f64,
) -> Integrate1dResult {
    // pyFAI `integrate1d_ng` sets `result.error_model` (azimuthal.py:455), so the
    // MG `union` applies the AZIMUTHAL crossed term for 1D.
    let sums = union_sums(
        &SumView::of1d(a),
        &SumView::of1d(b),
        em,
        em == ErrorModelKind::Azimuthal,
    );
    let rc = recalculate(&sums, dummy, em != ErrorModelKind::No);
    let (sigma, std, sem) = rc
        .errors
        .unwrap_or_else(|| (a.sigma.clone(), a.std.clone(), a.sem.clone()));
    Integrate1dResult {
        radial: a.radial.clone(),
        intensity: rc.intensity,
        sigma,
        count: sums.count,
        sum_signal: sums.signal,
        sum_variance: sums.variance,
        sum_normalization: sums.norm,
        sum_normalization2: sums.norm2,
        std,
        sem,
    }
}

/// Combine two 2D results (pyFAI `union` + `__recalculate_means__`). The
/// `radial`/`azimuthal` axes and `bins` are shared, taken from `a`.
fn union2d(
    a: &Integrate2dResult,
    b: &Integrate2dResult,
    em: ErrorModelKind,
    dummy: f64,
) -> Integrate2dResult {
    // pyFAI `integrate2d_ng` never sets `result.error_model` (no `_set_error_model`
    // between azimuthal.py:546 and the next method), so the 2D result carries
    // `error_model = None` and the MG `union`'s `if self.error_model ==
    // ErrorModel.AZIMUTHAL` is False — the crossed term is dropped for 2D. The
    // variance/norm² *sum* still happens (gated on `sum_variance is not None`).
    let sums = union_sums(&SumView::of2d(a), &SumView::of2d(b), em, false);
    let rc = recalculate(&sums, dummy, em != ErrorModelKind::No);
    let (sigma, std, sem) = rc
        .errors
        .unwrap_or_else(|| (a.sigma.clone(), a.std.clone(), a.sem.clone()));
    Integrate2dResult {
        radial: a.radial.clone(),
        azimuthal: a.azimuthal.clone(),
        bins: a.bins,
        intensity: rc.intensity,
        sigma,
        count: sums.count,
        sum_signal: sums.signal,
        sum_variance: sums.variance,
        sum_normalization: sums.norm,
        sum_normalization2: sums.norm2,
        std,
        sem,
    }
}
