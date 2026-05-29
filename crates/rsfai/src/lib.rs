//! The end-to-end azimuthal integrator: PONI + detector image → integrated
//! curve, in pure Rust. This is the orchestrator that wires the validated
//! pieces together — [`rsfai_detectors`] (pixel centres, gap mask, dummy),
//! [`rsfai_geometry`] (rotation transform, radial/azimuthal units, solid-angle
//! and polarization corrections), [`rsfai_preproc`] (per-pixel preprocessing),
//! and [`rsfai_integrate`] (the binning engines) — into a single
//! [`AzimuthalIntegrator`] that mirrors `pyFAI.integrator.AzimuthalIntegrator`.
//!
//! Method coverage: every cython method tuple `(split, algo, "cython")` with
//! `split ∈ {no, bbox, full}` and `algo ∈ {histogram, csr, lut, csc}`, for both
//! 1D and 2D — selected by [`IntegrationOptions::method`]. The bbox/full splits
//! additionally build the `delta_array` (radial + chi half-widths) and the
//! four-corner `corner_array` geometry. Each link in the chain is already
//! validated bit-exact against golden data; this crate composes them and is
//! validated end-to-end (PONI + image in, only) against the live pyFAI
//! integrator. (The 2D `pseudo` split is not ported.)
//!
//! ## What the integrator reproduces, in order
//!
//! 1. `detector.centers_f64()` → pixel centres (m), fed through
//!    `calc_pos_zyx` (the PONI rotation) to lab coordinates `(z, y, x)`.
//! 2. `center_array(unit, …)` → the **unscaled** per-pixel radial position the
//!    engine bins on (the reported axis is `position * unit.scale`); for 2D,
//!    additionally `center_array(CHI_RAD, …)` → per-pixel azimuth (rad).
//! 3. `solid_angle_array` (order 3) and, when a polarization factor is given,
//!    `polarization_array` — both cast to f32 for preproc.
//! 4. `detector.calc_mask()` (the static gap mask) and `detector.get_dummies()`
//!    (the dead-pixel sentinel) — exactly the masking pyFAI applies before
//!    preproc when the caller passes no explicit mask/dummy.
//! 5. `preproc4` → the per-pixel `[signal, variance, norm, count]` rows.
//! 6. The `opts.method` engine → the binned reduction: a direct
//!    `histogram`/`histogram_bbox`/`histogram_full` scatter, or a
//!    `csr`/`lut`/`csc` sparse matrix built once then applied.

use std::f64::consts::PI;

use rsfai_core::dtype::ErrorModel;
use rsfai_detectors::Detector;
use rsfai_geometry::{
    calc_pos_zyx, center_array, corner_array_f32, delta_chi, delta_radial, polarization_array,
    solid_angle_array, unscaled_center_array, PoniFile, PosZyx, Unit,
};
use rsfai_integrate::{
    build_bbox_csc_1d, build_bbox_csc_2d, build_bbox_csr_1d, build_bbox_csr_2d, build_bbox_lut_1d,
    build_bbox_lut_2d, build_full_csc_1d, build_full_csc_2d, build_full_csr_1d, build_full_csr_2d,
    build_full_lut_1d, build_full_lut_2d, csc_integrate1d, csc_integrate2d, csr_integrate1d,
    csr_integrate2d, histogram1d, histogram1d_bbox, histogram1d_full, histogram2d,
    histogram2d_bbox, histogram2d_full, lut_integrate1d, lut_integrate2d, Bbox2dBounds, BboxAzim1d,
    CsrIntegrate1d, Hist2dOptions, Integrate1d, Integrate2d,
};
use rsfai_preproc::{preproc4, PreprocOptions};

mod error;
pub use error::{Error, Result};

// Re-export the types a caller needs to drive an integration, so `use rsfai::*`
// is enough to build options and read units without reaching into sub-crates.
pub use rsfai_core::dtype::ErrorModel as ErrorModelKind;
pub use rsfai_detectors::Detector as DetectorModel;
pub use rsfai_geometry::Unit as RadialUnit;

/// The order pyFAI's `solidAngleArray` uses by default (`SolidAngleFactor`,
/// `1/cos³`). Matches the `corrections_golden` validation.
const SOLID_ANGLE_ORDER: f64 = 3.0;

/// `2π`, the `pos1_period` the 1D full-split build defaults to (`common.py` does
/// not forward `unit.period` for 1D).
const TWO_PI: f64 = 2.0 * PI;

/// The azimuth `pos1_period` for 2D: `common.py` forwards the azimuth unit's
/// period, `CHI_DEG.period = 360`. `> 0` engages the `[-π, π]` chi clip; the 2D
/// binning is on the unscaled chi in radians (`CHI_RAD`), but pyFAI passes the
/// degree-unit period verbatim, so the port does too (validated by the 2D
/// `csr`/`lut`/`csc` golden tests).
const AZIMUTH_PERIOD_DEG: f64 = 360.0;

/// Pixel-splitting scheme — the first element of pyFAI's method tuple.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Split {
    /// No split: each pixel falls wholly in the bin its center lands in.
    #[default]
    No,
    /// Bounding-box split: a pixel is spread over the bins its `center ± delta`
    /// box overlaps (separable in radial/azimuth).
    Bbox,
    /// Full split: a pixel's four corners are clipped against the bin grid.
    Full,
}

/// Binning algorithm — the second element of pyFAI's method tuple. All four
/// reproduce the same per-pixel split; they differ only in the data structure
/// (`histogram` scatters directly; `csr`/`lut`/`csc` build a sparse matrix once
/// and re-apply it).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Algo {
    /// Direct histogram scatter.
    #[default]
    Histogram,
    /// Compressed sparse row matrix.
    Csr,
    /// Dense look-up table.
    Lut,
    /// Compressed sparse column matrix.
    Csc,
}

/// An integration method = `(split, algo)`, the cython implementation of pyFAI's
/// `(split, algo, "cython")` method tuple. The default `("no", "histogram")` is
/// the path the orchestrator originally shipped.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Method {
    pub split: Split,
    pub algo: Algo,
}

/// Knobs shared by [`AzimuthalIntegrator::integrate1d`] and
/// [`AzimuthalIntegrator::integrate2d`], mirroring the pyFAI integrate kwargs
/// that affect the per-pixel preprocessing and the reduction.
#[derive(Debug, Clone)]
pub struct IntegrationOptions {
    /// Apply the solid-angle correction (`correctSolidAngle`).
    pub correct_solid_angle: bool,
    /// Polarization factor, or `None` to skip the correction
    /// (`polarization_factor`).
    pub polarization_factor: Option<f64>,
    /// Denominator seed passed to preproc (`normalization_factor`).
    pub normalization_factor: f32,
    /// Error model: `No` skips the variance/sigma propagation.
    pub error_model: ErrorModel,
    /// Pixel-split + binning algorithm. The default `("no", "histogram")` keeps
    /// the original no-split path; `bbox`/`full` engage the corner/delta geometry.
    pub method: Method,
    /// Radial `(min, max)` in the integration unit's **scaled** values (pyFAI's
    /// `radial_range`), or `None` to bin the full data range. The orchestrator
    /// divides by `unit.scale` to the unscaled radial the engines bin in. Pixels
    /// outside the range are dropped (boundary clip / per-pixel skip).
    pub radial_range: Option<(f64, f64)>,
    /// Azimuthal `(min, max)` in **degrees** (pyFAI's `azimuth_range`), or `None`
    /// for the full azimuthal range. The orchestrator converts to the radian chi
    /// frame via [`normalize_azimuth_range`] (pyFAI's `normalize_azimuth_range`).
    /// **Only honored by [`AzimuthalIntegrator::integrate2d`]** — 1D azimuthal
    /// sector restriction is not yet ported (the 1D builders take no per-pixel
    /// chi), so `integrate1d` rejects a non-`None` value.
    pub azimuth_range: Option<(f64, f64)>,
}

impl Default for IntegrationOptions {
    /// pyFAI's defaults: solid angle on, no polarization, unit normalization,
    /// no error model, no-split histogram, full data range.
    fn default() -> Self {
        IntegrationOptions {
            correct_solid_angle: true,
            polarization_factor: None,
            normalization_factor: 1.0,
            error_model: ErrorModel::No,
            method: Method::default(),
            radial_range: None,
            azimuth_range: None,
        }
    }
}

/// Per-frame corrections + normalization for
/// [`AzimuthalIntegrator::integrate1d_with`] / [`AzimuthalIntegrator::integrate2d_with`],
/// mirroring the per-call `dark`/`flat`/`mask`/`variance`/`normalization_factor`
/// pyFAI's `integrate1d_ng` accepts and that `MultiGeometry` passes per geometry.
/// Arrays, when present, are the flat row-major detector frame.
///
/// `mask` follows pyFAI's `create_mask`: **when supplied it replaces the detector
/// mask** (it does not OR with it); `None` uses the detector gap mask.
/// `normalization_factor` is carried as **f64** and cast to f32 only at the preproc
/// boundary — exactly like pyFAI's `floating normalization_factor` parameter
/// receiving a Python float — so a solid-angle-scaled monitor (`pixel1*pixel2/dist²`,
/// f64) reaches preproc bit-identically.
#[derive(Debug, Clone, Copy)]
pub struct Corrections<'a> {
    /// Dark current subtracted from the signal (`data - dark`).
    pub dark: Option<&'a [f32]>,
    /// Flat field (multiplies the normalization).
    pub flat: Option<&'a [f32]>,
    /// User mask (nonzero ⇒ bad); replaces the detector mask when `Some`.
    pub mask: Option<&'a [i8]>,
    /// Per-pixel variance, for the VARIANCE error model.
    pub variance: Option<&'a [f32]>,
    /// Denominator seed (pyFAI's `normalization_factor`); cast to f32 at preproc.
    pub normalization_factor: f64,
}

impl<'a> Corrections<'a> {
    /// Corrections carrying only a normalization factor (no dark/flat/mask/variance)
    /// — the shape the bare [`AzimuthalIntegrator::integrate1d`] wrapper builds.
    pub fn with_normalization(normalization_factor: f64) -> Self {
        Corrections {
            dark: None,
            flat: None,
            mask: None,
            variance: None,
            normalization_factor,
        }
    }
}

/// The per-pixel variance route preproc takes, resolved once per integrate by
/// [`resolve_variance`] (pyFAI's `_normalize_error_model_variance`): either the
/// engine's own poisson (`poissonian = true`, no slice) or an explicit per-pixel
/// `variance` (`poissonian = false`) — the caller's array, or the precomputed
/// `max(data,1)[+max(dark,0)]` for an engine that cannot manage variance itself.
struct ResolvedVariance {
    poissonian: bool,
    variance: Option<Vec<f32>>,
}

/// pyFAI `IntegrationMethod.manage_variance` for the cython engines: whether the
/// reduce kernel computes the error model itself (so poisson stays poisson and
/// preproc emits `max(data,1)`) or needs a precomputed per-pixel variance fed in
/// as VARIANCE. Every 1D engine manages it; in 2D only the `bbox`/`no` CSR and
/// `bbox` LUT engines do **not** (verified against pyFAI's registry —
/// `_does_manage_variance`, which checks the engine for a `poissonian`/
/// `error_model` parameter).
fn manage_variance(method: Method, dim: u8) -> bool {
    if dim == 1 {
        return true;
    }
    !matches!(
        (method.split, method.algo),
        (Split::Bbox, Algo::Csr) | (Split::Bbox, Algo::Lut) | (Split::No, Algo::Csr)
    )
}

/// pyFAI `_normalize_error_model_variance` (`integrator/common.py:385`): pick the
/// per-pixel variance route preproc will take. An explicit `variance` wins (pyFAI
/// forces the VARIANCE model). Otherwise, for a poisson error model on an engine
/// that does **not** manage variance (`!manage_variance`), precompute the per-pixel
/// variance as `max(data,1)`, augmented by `max(dark,0)` when a dark frame is
/// subtracted — exactly the array pyFAI feeds the engine as VARIANCE. A
/// `manage_variance` engine keeps poisson, letting preproc emit `max(data,1)` with
/// no dark term (`preproc.pyx:309`). f32 throughout, matching numpy's
/// `(maximum(data,1)+maximum(dark,0)).astype(float32)`.
fn resolve_variance(
    data: &[f32],
    dark: Option<&[f32]>,
    explicit: Option<&[f32]>,
    em: ErrorModel,
    manage_variance: bool,
) -> ResolvedVariance {
    if let Some(v) = explicit {
        return ResolvedVariance {
            poissonian: false,
            variance: Some(v.to_vec()),
        };
    }
    if em.poissonian() && !manage_variance {
        let variance = match dark {
            Some(d) => data
                .iter()
                .zip(d)
                .map(|(&x, &dk)| x.max(1.0) + dk.max(0.0))
                .collect(),
            None => data.iter().map(|&x| x.max(1.0)).collect(),
        };
        return ResolvedVariance {
            poissonian: false,
            variance: Some(variance),
        };
    }
    ResolvedVariance {
        poissonian: em.poissonian(),
        variance: None,
    }
}

/// Port of `Geometry.normalize_azimuth_range` (`geometry/core.py`): convert a
/// `(lo, hi)` azimuth range from **degrees** to radians in the `chiDiscAtPi`
/// (discontinuity at π — pyFAI's default) frame, wrapping the upper bound by
/// `2π` when the range crosses the discontinuity so `hi > lo` always holds. The
/// engines override the azimuthal `pos1_min/max` with `min/max` of this range
/// (`splitBBox_common.pyx` calc_boundaries), so the wrap order is what keeps a
/// disc-crossing window contiguous. `None` ⇒ full azimuthal range.
fn normalize_azimuth_range(range: Option<(f64, f64)>) -> Option<(f64, f64)> {
    let (lo, hi) = range?;
    // `deg2rad(dd, disc=true)`: rp = (dd/180) mod 2 (Python floor-mod, i.e.
    // `rem_euclid`), shifted into [-1, 1) when ≥ 1, scaled by π → [-π, π).
    let deg2rad = |dd: f64| {
        let mut rp = (dd / 180.0).rem_euclid(2.0);
        if rp >= 1.0 {
            rp -= 2.0;
        }
        rp * PI
    };
    let lo_r = deg2rad(lo);
    let mut hi_r = deg2rad(hi);
    if hi_r <= lo_r {
        hi_r += TWO_PI;
    }
    Some((lo_r, hi_r))
}

/// The 1D integration result, field-for-field the subset of pyFAI's
/// `Integrate1dResult` the engines populate. `radial` carries the unit scale.
///
/// The `sum_*` / `count` accumulators are **f64** (`acc_t`): the sparse engines
/// (`csr`/`lut`/`csc`) and the split-histogram engines expose them at full
/// precision, so the drop-in does too. The no-split histogram engine downcasts
/// them to f32 internally (`data_t`); that path widens its f32 result back to
/// f64 here (the f32 value, losslessly), so a single type serves every method.
/// `intensity`/`sigma`/`std`/`sem` are f32 in every engine.
#[derive(Debug, Clone)]
pub struct Integrate1dResult {
    /// Scaled radial bin centres (`position * unit.scale`), length `npt`.
    pub radial: Vec<f64>,
    /// Average intensity `signal / normalization` (f32).
    pub intensity: Vec<f32>,
    /// Standard error on the mean (= `sem`; f32). Zero unless an error model.
    pub sigma: Vec<f32>,
    /// Number of valid pixels per bin (f64).
    pub count: Vec<f64>,
    /// Binned signal (f64).
    pub sum_signal: Vec<f64>,
    /// Binned variance (f64).
    pub sum_variance: Vec<f64>,
    /// Binned normalization (f64).
    pub sum_normalization: Vec<f64>,
    /// Binned normalization² (f64).
    pub sum_normalization2: Vec<f64>,
    /// Propagated standard deviation `sqrt(variance / norm²)` (f32).
    pub std: Vec<f32>,
    /// Standard error on the mean `sqrt(variance) / normalization` (f32).
    pub sem: Vec<f32>,
}

/// The 2D integration result. Arrays are stored flat in `(azimuthal, radial)`
/// row-major order (pyFAI's post-transpose layout): cell `(azimuthal j, radial
/// i)` is at index `j * bins.0 + i`. The `sum_*`/`count` arrays are f64
/// (`acc_t`, not downcast — the 2D engine exposes full precision).
#[derive(Debug, Clone)]
pub struct Integrate2dResult {
    /// Scaled radial bin centres (`radial * unit.scale`), length `bins.0`.
    pub radial: Vec<f64>,
    /// Scaled azimuthal bin centres (`azimuthal * CHI_DEG.scale`, degrees),
    /// length `bins.1`.
    pub azimuthal: Vec<f64>,
    /// `(radial, azimuthal)` bin counts, for indexing the flat arrays.
    pub bins: (usize, usize),
    /// Average intensity `signal / normalization` (f32).
    pub intensity: Vec<f32>,
    /// Standard error on the mean (= `sem`; f32).
    pub sigma: Vec<f32>,
    /// Number of valid pixels per cell (f64).
    pub count: Vec<f64>,
    /// Binned signal (f64).
    pub sum_signal: Vec<f64>,
    /// Binned variance (f64).
    pub sum_variance: Vec<f64>,
    /// Binned normalization (f64).
    pub sum_normalization: Vec<f64>,
    /// Binned normalization² (f64).
    pub sum_normalization2: Vec<f64>,
    /// Propagated standard deviation `sqrt(variance / norm²)` (f32).
    pub std: Vec<f32>,
    /// Standard error on the mean `sqrt(variance) / normalization` (f32).
    pub sem: Vec<f32>,
}

/// A pure-Rust drop-in for `pyFAI.integrator.AzimuthalIntegrator`: holds the
/// PONI geometry plus the detector model, and integrates a detector image into
/// a 1D or 2D curve.
#[derive(Debug, Clone)]
pub struct AzimuthalIntegrator {
    /// The detector model (pixel size, shape, gap mask, dummy).
    pub detector: Detector,
    /// Sample–detector distance `L` (m).
    pub dist: f64,
    /// PONI along the slow (Y) axis (m).
    pub poni1: f64,
    /// PONI along the fast (X) axis (m).
    pub poni2: f64,
    /// Rotation about axis 1 (rad).
    pub rot1: f64,
    /// Rotation about axis 2 (rad).
    pub rot2: f64,
    /// Rotation about axis 3 (rad).
    pub rot3: f64,
    /// X-ray wavelength (m). Required by the q/d units; ignored by 2θ/r.
    pub wavelength: f64,
}

impl AzimuthalIntegrator {
    /// Build an integrator from a parsed [`PoniFile`] and an explicit detector
    /// model. Use this when you want a detector the name-resolution in
    /// [`load`](Self::load) does not cover.
    pub fn from_poni(poni: &PoniFile, detector: Detector) -> Self {
        AzimuthalIntegrator {
            detector,
            dist: poni.dist,
            poni1: poni.poni1,
            poni2: poni.poni2,
            rot1: poni.rot1,
            rot2: poni.rot2,
            rot3: poni.rot3,
            // q/d units need a wavelength; 2θ/r do not. A missing wavelength is
            // left as 0.0 — the caller is responsible for a unit that needs it.
            wavelength: poni.wavelength.unwrap_or(0.0),
        }
    }

    /// Load an integrator from a `.poni` file, resolving the detector model
    /// from the file's `Detector:` name. Only detectors with a golden-validated
    /// path are resolved (currently Pilatus1M and Eiger4M); supply the detector
    /// explicitly via [`from_poni`](Self::from_poni) for others.
    pub fn load<P: AsRef<std::path::Path>>(path: P) -> Result<Self> {
        let poni = PoniFile::load(path)?;
        let detector = resolve_detector(&poni)?;
        Ok(Self::from_poni(&poni, detector))
    }

    /// Lab coordinates `(z, y, x)` for every pixel centre: `detector.centers_f64`
    /// fed through the PONI rotation `calc_pos_zyx`. The flat detectors in scope
    /// are contiguous (`p3 = None` ⇒ `L3 = dist`).
    fn pixel_positions(&self) -> PosZyx {
        let (p1, p2) = self.detector.centers_f64();
        calc_pos_zyx(
            self.dist,
            self.poni1,
            self.poni2,
            self.rot1,
            self.rot2,
            self.rot3,
            &p1,
            &p2,
            None,
            self.detector.orientation,
        )
    }

    /// The **scaled** per-pixel center array for `unit` (pyFAI's
    /// `array_from_unit(unit)` / `center_array(scale=True)`), flat row-major over
    /// the detector. Radial spaces (q/2θ/r) carry `unit.scale`; chi is in the
    /// unit's angular scale (CHI_DEG → degrees, CHI_RAD → radians). Used by
    /// `MultiGeometry` to guess the common radial/azimuth range across geometries
    /// (`min`/`max` over the concatenation, order-independent ⇒ bit-exact).
    pub fn array_from_unit(&self, unit: Unit) -> Vec<f64> {
        let pos = self.pixel_positions();
        center_array(unit, &pos.x, &pos.y, &pos.z, self.wavelength)
    }

    /// The f32 `(npix, 4, 2)` corner array (radial in `unit.space` + chi) the
    /// bbox/full splits bin on: `corner_array_f32` over the corner-grid lab
    /// coords (`calc_pos_zyx` over `detector.corner_positions_f64`). `chiDiscAtPi
    /// = true` (pyFAI's default).
    fn corner_array(&self, unit: Unit) -> Vec<f32> {
        let (cp1, cp2) = self.detector.corner_positions_f64();
        let grid = calc_pos_zyx(
            self.dist,
            self.poni1,
            self.poni2,
            self.rot1,
            self.rot2,
            self.rot3,
            &cp1,
            &cp2,
            None,
            self.detector.orientation,
        );
        corner_array_f32(&grid, self.detector.shape, unit, self.wavelength, true)
    }

    /// The static gap mask as a flat row-major `i8` vector (`1` = masked), or
    /// `None` for a gapless detector — exactly `create_mask(data, mask=None)`
    /// when the integrator carries only the detector mask.
    fn build_mask(&self) -> Option<Vec<i8>> {
        self.detector.calc_mask().map(|m| {
            m.as_slice()
                .expect("calc_mask returns a C-contiguous Array2")
                .to_vec()
        })
    }

    /// The base gap/dummy mask OR'd with the azimuthal-sector mask, for the
    /// no-split 1D histogram — a port of `azimuthal.py`'s
    /// `azim_mask = (chi > chi_max) | (chi < chi_min)` folded into the mask before
    /// `histogram1d_engine` (which takes no per-pixel chi). `lo`/`hi` are the
    /// **radian** chi bounds (`normalize_azimuth_range` output); `chi` is the
    /// radian chi center (`CHI_RAD`, scale 1). A pixel is masked (`1`) if the base
    /// mask flags it OR its center lies outside `[lo, hi]`.
    fn chi_mask(&self, pos: &PosZyx, base: Option<&[i8]>, lo: f64, hi: f64) -> Vec<i8> {
        let chi = center_array(Unit::CHI_RAD, &pos.x, &pos.y, &pos.z, self.wavelength);
        (0..chi.len())
            .map(|i| i8::from(base.is_some_and(|m| m[i] != 0) || chi[i] > hi || chi[i] < lo))
            .collect()
    }

    /// The per-pixel `[signal, variance, norm, count]` rows (`preproc4`),
    /// applying the same corrections / mask / dummy pyFAI feeds to preproc.
    /// `data` is the image already cast to f32 (preproc's `data_t`); `pos`
    /// supplies the lab coordinates the polarization correction needs.
    fn preproc_rows_with(
        &self,
        data: &[f32],
        pos: &PosZyx,
        mask: Option<&[i8]>,
        opts: &IntegrationOptions,
        corr: &Corrections,
        rv: &ResolvedVariance,
    ) -> Vec<f32> {
        // solid angle (f64) cast to f32, matching pyFAI's preproc input dtype.
        let solidangle: Option<Vec<f32>> = opts.correct_solid_angle.then(|| {
            solid_angle_array(
                &self.detector,
                self.dist,
                self.poni1,
                self.poni2,
                SOLID_ANGLE_ORDER,
            )
            .iter()
            .map(|&v| v as f32)
            .collect()
        });
        let polarization: Option<Vec<f32>> = opts
            .polarization_factor
            .map(|factor| polarization_array(&pos.x, &pos.y, &pos.z, factor, 0.0));
        let (dummy, delta_dummy) = self.detector.get_dummies();

        let popt = PreprocOptions {
            dark: corr.dark,
            flat: corr.flat,
            solidangle: solidangle.as_deref(),
            polarization: polarization.as_deref(),
            mask,
            // The per-pixel variance route is resolved once per integrate by
            // pyFAI's `_normalize_error_model_variance` (see [`resolve_variance`]):
            // either the engine's own poisson (`poissonian`, no slice) or an
            // explicit/precomputed per-pixel `variance`.
            variance: rv.variance.as_deref(),
            // pyFAI's `floating normalization_factor` receives the (f64) monitor and
            // casts it to f32 for the f32-image preproc path; mirror that cast.
            normalization_factor: corr.normalization_factor as f32,
            poissonian: rv.poissonian,
            check_dummy: dummy.is_some(),
            dummy: dummy.unwrap_or(0.0),
            delta_dummy: delta_dummy.unwrap_or(0.0),
            ..Default::default()
        };
        preproc4(data, &popt)
    }

    /// Integrate `image` (the detector frame already cast to f32) into a 1D
    /// radial curve of `npt` bins in `unit`, using the `opts.method` split and
    /// binning algorithm. `image` must be the flat row-major frame of length
    /// `detector.size()`.
    ///
    /// Binning is on the **unscaled** internal radial (`unscaled_center_array`),
    /// and the reported `radial` axis multiplies the binned centers by
    /// `unit.scale` — pyFAI's structure, so a non-unit scale (e.g. `2th_deg`) is
    /// applied exactly once.
    pub fn integrate1d(
        &self,
        image: &[f32],
        npt: usize,
        unit: Unit,
        opts: &IntegrationOptions,
    ) -> Integrate1dResult {
        let corr = Corrections::with_normalization(opts.normalization_factor as f64);
        self.integrate1d_with(image, npt, unit, opts, &corr)
    }

    /// Like [`integrate1d`](Self::integrate1d) but with per-frame `dark`/`flat`/
    /// `mask`/`variance` corrections and an f64 `normalization_factor`
    /// ([`Corrections`]). This is the entry `MultiGeometry` drives per geometry.
    pub fn integrate1d_with(
        &self,
        image: &[f32],
        npt: usize,
        unit: Unit,
        opts: &IntegrationOptions,
        corr: &Corrections,
    ) -> Integrate1dResult {
        assert_eq!(
            image.len(),
            self.detector.size(),
            "image length {} != detector size {}",
            image.len(),
            self.detector.size()
        );
        let pos = self.pixel_positions();
        // pyFAI `create_mask`: a supplied user mask *replaces* the detector mask;
        // `None` falls back to the detector gap mask.
        let base_mask_storage: Option<Vec<i8>> = match corr.mask {
            Some(user) => Some(user.to_vec()),
            None => self.build_mask(),
        };
        let base_mask: Option<&[i8]> = base_mask_storage.as_deref();
        let pos = &pos;
        let radial = unscaled_center_array(unit.space, &pos.x, &pos.y, &pos.z, self.wavelength);
        let em = opts.error_model;
        // Resolve the per-pixel variance route once (pyFAI's
        // `_normalize_error_model_variance`); every 1D engine manages variance, so a
        // poisson model keeps `poissonian` and preproc emits `max(data,1)`.
        let rv = resolve_variance(
            image,
            corr.dark,
            corr.variance,
            em,
            manage_variance(opts.method, 1),
        );
        // `radial_range` is given in the scaled unit; binning is on the unscaled
        // radial, so divide each endpoint by `unit.scale` (pyFAI's
        // `_normalize_range`). `None` keeps the full data range.
        let pos0_range = opts
            .radial_range
            .map(|(lo, hi)| (lo / unit.scale, hi / unit.scale));
        // `azimuth_range` is given in degrees; convert to the radian chi frame
        // (deg→rad, chiDiscAtPi, 2π wrap) via pyFAI's `normalize_azimuth_range`.
        // `None` keeps the full ring. The no-split histogram folds it into the
        // mask (its engine takes no chi); every other 1D engine filters per-pixel.
        let pos1_range = normalize_azimuth_range(opts.azimuth_range);

        // The no-split histogram is the one path whose accumulators are f32; it
        // bins every pixel (masked pixels are zeroed by preproc but still set the
        // range), so no mask is forwarded — except the azimuthal restriction,
        // which pyFAI's `histogram1d_engine` cannot take per-pixel, so it is folded
        // into the preproc mask (`chi > chi_max | chi < chi_min`, radian center).
        if opts.method.split == Split::No && opts.method.algo == Algo::Histogram {
            let chi_mask = pos1_range.map(|(lo, hi)| self.chi_mask(pos, base_mask, lo, hi));
            let m = chi_mask.as_deref().or(base_mask);
            let prep = self.preproc_rows_with(image, pos, m, opts, corr, &rv);
            let h = histogram1d(&radial, &prep, npt, pos0_range, em, 0.0);
            return histogram_to_result(h, unit.scale);
        }

        // Every other method excludes masked pixels from the matrix/scatter and
        // filters the azimuthal sector per-pixel (so the preproc mask stays the
        // bare gap/dummy mask — out-of-sector pixels keep their real signal and are
        // dropped by the engine, not zeroed). Radial units (q/2θ/r) are
        // non-negative ⇒ `allow_pos0_neg = false`. The 1D full-split build is not
        // handed chiDiscAtPi / pos1_period (common.py does not forward them), so
        // they take the defaults true / 2π.
        let prep = self.preproc_rows_with(image, pos, base_mask, opts, corr, &rv);
        let mask_ref = base_mask;
        let allow_neg = false;
        // Per-pixel chi center, threaded into the bbox builders only when an
        // azimuth_range is set (pyFAI passes pos1/delta_pos1 then).
        let chi = pos1_range
            .map(|_| center_array(Unit::CHI_RAD, &pos.x, &pos.y, &pos.z, self.wavelength));
        let r = match opts.method.split {
            Split::Full => {
                // Full split bins on the four-corner array (radial + chi), f32
                // upcast to f64 — the input pyFAI's FullSplitIntegrator receives.
                // The azimuthal skip reads the chi corners directly, so no separate
                // chi array is needed (pyFAI passes only `pos1_range`).
                let corners = self.corner_array(unit);
                let cf: Vec<f64> = corners.iter().map(|&v| v as f64).collect();
                match opts.method.algo {
                    Algo::Histogram => histogram1d_full(
                        &cf, &prep, mask_ref, npt, em, 0.0, allow_neg, true, TWO_PI, pos0_range,
                        pos1_range,
                    ),
                    Algo::Csr => {
                        let (m, c) = build_full_csr_1d(
                            &cf, mask_ref, npt, allow_neg, true, TWO_PI, pos0_range, pos1_range,
                        );
                        csr_integrate1d(&m, &prep, c, em, 0.0)
                    }
                    Algo::Lut => {
                        let (m, c) = build_full_lut_1d(
                            &cf, mask_ref, npt, allow_neg, true, TWO_PI, pos0_range, pos1_range,
                        );
                        lut_integrate1d(&m, &prep, c, em, 0.0)
                    }
                    Algo::Csc => {
                        let (m, c) = build_full_csc_1d(
                            &cf, mask_ref, npt, allow_neg, true, TWO_PI, pos0_range, pos1_range,
                        );
                        csc_integrate1d(&m, &prep, c, em, 0.0)
                    }
                }
            }
            split => {
                // No / Bbox: the bbox-family build on (pos0, dpos0[, pos1, dpos1]).
                // No-split has dpos0 = None (each pixel collapses to its center);
                // bbox uses the radial half-width `delta_radial`. The corner array
                // (needed for the radial delta and/or the chi delta) is built once.
                let corners =
                    (split == Split::Bbox || chi.is_some()).then(|| self.corner_array(unit));
                let delta = (split == Split::Bbox)
                    .then(|| delta_radial(corners.as_ref().unwrap(), &radial));
                let dpos0 = delta.as_deref();
                // bbox azimuth: chi center + half-width. The build reads d1 only when
                // the radial axis splits (do_split gate), so for no-split CSR the
                // dpos1 is supplied but ignored — matching pyFAI calc_lut_1d.
                let dchi = chi
                    .as_ref()
                    .map(|c| delta_chi(corners.as_ref().unwrap(), c));
                let azim = chi.as_ref().zip(dchi.as_ref()).map(|(c, d)| BboxAzim1d {
                    pos1: c,
                    dpos1: d,
                    range: pos1_range.expect("chi present ⟹ azimuth_range present"),
                });
                match opts.method.algo {
                    Algo::Histogram => {
                        // No-split-histogram returns above; this is bbox-histogram.
                        let d = dpos0.expect("bbox split has a radial delta");
                        histogram1d_bbox(
                            &radial, d, &prep, mask_ref, npt, em, 0.0, allow_neg, pos0_range, azim,
                        )
                    }
                    Algo::Csr => {
                        let (m, c) = build_bbox_csr_1d(
                            &radial, dpos0, mask_ref, npt, allow_neg, pos0_range, azim,
                        );
                        csr_integrate1d(&m, &prep, c, em, 0.0)
                    }
                    Algo::Lut => {
                        let (m, c) = build_bbox_lut_1d(
                            &radial, dpos0, mask_ref, npt, allow_neg, pos0_range, azim,
                        );
                        lut_integrate1d(&m, &prep, c, em, 0.0)
                    }
                    Algo::Csc => {
                        let (m, c) = build_bbox_csc_1d(
                            &radial, dpos0, mask_ref, npt, allow_neg, pos0_range, azim,
                        );
                        csc_integrate1d(&m, &prep, c, em, 0.0)
                    }
                }
            }
        };
        csr_to_result(r, unit.scale)
    }

    /// Integrate `image` into a 2D `(radial, azimuthal)` cake of `npt_rad ×
    /// npt_azim` bins, radial in `unit` and azimuth in degrees (`CHI_DEG`), using
    /// the `opts.method` split and binning algorithm. Binning is on the unscaled
    /// radial (pos0) and unscaled chi in radians (pos1); the azimuthal
    /// discontinuity is at π (`chiDiscAtPi = True`, pyFAI's default), and the
    /// reported axes scale once (radial × `unit.scale`, azimuth × `CHI_DEG.scale`).
    pub fn integrate2d(
        &self,
        image: &[f32],
        npt_rad: usize,
        npt_azim: usize,
        unit: Unit,
        opts: &IntegrationOptions,
    ) -> Integrate2dResult {
        let corr = Corrections::with_normalization(opts.normalization_factor as f64);
        self.integrate2d_with(image, npt_rad, npt_azim, unit, opts, &corr)
    }

    /// Like [`integrate2d`](Self::integrate2d) but with per-frame `dark`/`flat`/
    /// `mask`/`variance` corrections and an f64 `normalization_factor`
    /// ([`Corrections`]). This is the entry `MultiGeometry` drives per geometry.
    pub fn integrate2d_with(
        &self,
        image: &[f32],
        npt_rad: usize,
        npt_azim: usize,
        unit: Unit,
        opts: &IntegrationOptions,
        corr: &Corrections,
    ) -> Integrate2dResult {
        assert_eq!(
            image.len(),
            self.detector.size(),
            "image length {} != detector size {}",
            image.len(),
            self.detector.size()
        );
        let pos = self.pixel_positions();
        // pyFAI `create_mask`: a supplied user mask *replaces* the detector mask;
        // `None` falls back to the detector gap mask.
        let base_mask_storage: Option<Vec<i8>> = match corr.mask {
            Some(user) => Some(user.to_vec()),
            None => self.build_mask(),
        };
        let base_mask: Option<&[i8]> = base_mask_storage.as_deref();
        let pos = &pos;
        // Bin on the unscaled internal radial (pos0) and the unscaled chi in
        // radians (pos1; `CHI_RAD.scale == 1`); the reported axes scale once —
        // radial × `unit.scale`, azimuth × `CHI_DEG.scale` (degrees).
        let radial = unscaled_center_array(unit.space, &pos.x, &pos.y, &pos.z, self.wavelength);
        let azimuthal = center_array(Unit::CHI_RAD, &pos.x, &pos.y, &pos.z, self.wavelength);
        let em = opts.error_model;
        // Resolve the per-pixel variance route once (pyFAI's
        // `_normalize_error_model_variance`): the 2D `bbox`/`no` CSR and `bbox` LUT
        // engines do not manage variance, so a poisson model is precomputed to
        // `max(data,1)+max(dark,0)` and fed as VARIANCE rather than left to preproc.
        let rv = resolve_variance(
            image,
            corr.dark,
            corr.variance,
            em,
            manage_variance(opts.method, 2),
        );
        let prep = self.preproc_rows_with(image, pos, base_mask, opts, corr, &rv);
        let bins = (npt_rad, npt_azim);
        // `radial_range` is the scaled-unit radial; binning is on the unscaled
        // radial, so divide by `unit.scale` (pyFAI's `_normalize_range`).
        let radial_range = opts
            .radial_range
            .map(|(lo, hi)| (lo / unit.scale, hi / unit.scale));
        // `azimuth_range` is given in degrees; the engines bin the unscaled chi
        // in radians, so convert via pyFAI's `normalize_azimuth_range` (deg→rad,
        // chiDiscAtPi, 2π wrap). The engine applies it as the pos1 boundary
        // override exactly like `radial_range` for pos0.
        let azimuth_range = normalize_azimuth_range(opts.azimuth_range);

        // No-split histogram: bins every pixel (masked pixels are zeroed by
        // preproc but still set the range), so no mask is forwarded.
        if opts.method.split == Split::No && opts.method.algo == Algo::Histogram {
            let hopts = Hist2dOptions {
                bins,
                radial_range,
                azimuth_range,
                error_model: em,
                // Standard radial units (q/2θ/r) are non-negative.
                allow_radial_neg: false,
                chi_disc_at_pi: true,
                pos1_period: AZIMUTH_PERIOD_DEG,
                empty: 0.0,
            };
            let h = histogram2d(&radial, &azimuthal, &prep, base_mask, &hopts);
            return integrate2d_to_result(h, unit.scale);
        }

        // Every other engine excludes masked pixels from the matrix/scatter.
        // chiDiscAtPi = true, pos1_period = CHI_DEG.period (= 360), allow_pos0_neg
        // = false (radial units are non-negative) — the 2D `Bbox2dBounds` pyFAI's
        // common.py forwards.
        let mask_ref = base_mask;
        let bounds = Bbox2dBounds {
            allow_pos0_neg: false,
            chi_disc_at_pi: true,
            pos1_period: AZIMUTH_PERIOD_DEG,
            radial_range,
            azimuth_range,
        };
        let h = match opts.method.split {
            Split::Full => {
                // Full split bins on the four-corner array (radial + chi), f32
                // upcast to f64 — the input pyFAI's FullSplitIntegrator receives.
                let corners = self.corner_array(unit);
                let cf: Vec<f64> = corners.iter().map(|&v| v as f64).collect();
                match opts.method.algo {
                    Algo::Histogram => {
                        histogram2d_full(&cf, &prep, mask_ref, bins, &bounds, em, 0.0)
                    }
                    Algo::Csr => {
                        let (m, c0, c1) = build_full_csr_2d(&cf, mask_ref, bins, &bounds);
                        csr_integrate2d(&m, &prep, c0, c1, em, 0.0)
                    }
                    Algo::Lut => {
                        let (m, c0, c1) = build_full_lut_2d(&cf, mask_ref, bins, &bounds);
                        lut_integrate2d(&m, &prep, c0, c1, em, 0.0)
                    }
                    Algo::Csc => {
                        let (m, c0, c1) = build_full_csc_2d(&cf, mask_ref, bins, &bounds);
                        csc_integrate2d(&m, &prep, c0, c1, em, 0.0)
                    }
                }
            }
            split => {
                // No / Bbox: the bbox-family build on (pos0, dpos0, pos1, dpos1).
                // No-split has both deltas None (each pixel collapses to its
                // center); bbox uses the radial (`delta_radial`) and azimuthal
                // (`delta_chi`) half-widths from the corner array.
                let deltas = (split == Split::Bbox).then(|| {
                    let corners = self.corner_array(unit);
                    (
                        delta_radial(&corners, &radial),
                        delta_chi(&corners, &azimuthal),
                    )
                });
                let (dpos0, dpos1) = match &deltas {
                    Some((d0, d1)) => (Some(d0.as_slice()), Some(d1.as_slice())),
                    None => (None, None),
                };
                match opts.method.algo {
                    Algo::Histogram => {
                        // No-split-histogram returns above; this is bbox-histogram.
                        let (d0, d1) = deltas.as_ref().expect("bbox split has deltas");
                        histogram2d_bbox(
                            &radial, d0, &azimuthal, d1, &prep, mask_ref, bins, &bounds, em, 0.0,
                        )
                    }
                    Algo::Csr => {
                        let (m, c0, c1) = build_bbox_csr_2d(
                            &radial, dpos0, &azimuthal, dpos1, mask_ref, bins, &bounds,
                        );
                        csr_integrate2d(&m, &prep, c0, c1, em, 0.0)
                    }
                    Algo::Lut => {
                        let (m, c0, c1) = build_bbox_lut_2d(
                            &radial, dpos0, &azimuthal, dpos1, mask_ref, bins, &bounds,
                        );
                        lut_integrate2d(&m, &prep, c0, c1, em, 0.0)
                    }
                    Algo::Csc => {
                        let (m, c0, c1) = build_bbox_csc_2d(
                            &radial, dpos0, &azimuthal, dpos1, mask_ref, bins, &bounds,
                        );
                        csc_integrate2d(&m, &prep, c0, c1, em, 0.0)
                    }
                }
            }
        };
        integrate2d_to_result(h, unit.scale)
    }
}

/// Map the no-split histogram engine's `Integrate1d` (f32 accumulators) to the
/// unified result, widening the f32 `sum_*`/`count` to f64 (lossless — the f32
/// truncation already happened in the engine's final reduction).
fn histogram_to_result(h: Integrate1d, scale: f64) -> Integrate1dResult {
    let widen = |v: Vec<f32>| v.into_iter().map(f64::from).collect();
    Integrate1dResult {
        radial: h.position.iter().map(|&p| p * scale).collect(),
        intensity: h.intensity,
        sigma: h.sigma,
        count: widen(h.count),
        sum_signal: widen(h.signal),
        sum_variance: widen(h.variance),
        sum_normalization: widen(h.normalization),
        sum_normalization2: widen(h.norm_sq),
        std: h.std,
        sem: h.sem,
    }
}

/// Map a 2D engine's `Integrate2d` (f64 accumulators in every 2D engine) to the
/// public result, scaling the bin-center axes once: radial × `scale`, azimuth ×
/// `CHI_DEG.scale` (radians → degrees).
fn integrate2d_to_result(h: Integrate2d, scale: f64) -> Integrate2dResult {
    Integrate2dResult {
        radial: h.radial.iter().map(|&r| r * scale).collect(),
        azimuthal: h
            .azimuthal
            .iter()
            .map(|&a| a * Unit::CHI_DEG.scale)
            .collect(),
        bins: h.bins,
        intensity: h.intensity,
        sigma: h.sigma,
        count: h.count,
        sum_signal: h.signal,
        sum_variance: h.variance,
        sum_normalization: h.normalization,
        sum_normalization2: h.norm_sq,
        std: h.std,
        sem: h.sem,
    }
}

/// Map a sparse / split engine's `CsrIntegrate1d` (f64 accumulators) to the
/// unified result. `sum_norm_sq` is the `sum_normalization2` field.
fn csr_to_result(r: CsrIntegrate1d, scale: f64) -> Integrate1dResult {
    Integrate1dResult {
        radial: r.position.iter().map(|&p| p * scale).collect(),
        intensity: r.intensity,
        sigma: r.sigma,
        count: r.count,
        sum_signal: r.sum_signal,
        sum_variance: r.sum_variance,
        sum_normalization: r.sum_normalization,
        sum_normalization2: r.sum_norm_sq,
        std: r.std,
        sem: r.sem,
    }
}

/// Resolve the detector model from a PONI's `Detector:` name. Only
/// golden-validated detectors are resolved; anything else is an error so the
/// integrator never silently runs an unverified detector path.
fn resolve_detector(poni: &PoniFile) -> Result<Detector> {
    let mut detector = match poni.detector.as_deref() {
        Some("Pilatus1M") => Detector::pilatus1m(),
        Some("Eiger4M") => Detector::eiger4m(),
        other => {
            return Err(Error::UnsupportedDetector(
                other.unwrap_or("<none>").to_string(),
            ))
        }
    };
    // A PONI may override the detector's default orientation (its
    // `Detector_config` JSON carries `{"orientation": N}`); apply it so the
    // drop-in reproduces `pyFAI.load(poni)` exactly, including the pixel-index
    // reorder + transform sign-flip for orientations 1/2/4.
    if let Some(o) = poni.orientation {
        detector.orientation = o;
    }
    Ok(detector)
}
