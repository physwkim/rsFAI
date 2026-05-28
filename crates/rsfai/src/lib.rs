//! The end-to-end azimuthal integrator: PONI + detector image → integrated
//! curve, in pure Rust. This is the orchestrator that wires the validated
//! pieces together — [`rsfai_detectors`] (pixel centres, gap mask, dummy),
//! [`rsfai_geometry`] (rotation transform, radial/azimuthal units, solid-angle
//! and polarization corrections), [`rsfai_preproc`] (per-pixel preprocessing),
//! and [`rsfai_integrate`] (the binning engines) — into a single
//! [`AzimuthalIntegrator`] that mirrors `pyFAI.integrator.AzimuthalIntegrator`.
//!
//! Method coverage so far: the no-split histogram path
//! (`("no", "histogram", "cython")`) for 1D and 2D. Each link in the chain is
//! already validated bit-exact against golden data; this crate composes them
//! and is validated end-to-end (PONI + image in, only) against the live pyFAI
//! integrator. The bbox/CSR and full-split paths (which additionally need the
//! `delta_array` / `corner_array` geometry, not yet ported) are layered on top
//! of this same orchestrator next.
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
//! 6. `histogram1d` / `histogram2d` → the binned reduction.

use rsfai_core::dtype::ErrorModel;
use rsfai_detectors::Detector;
use rsfai_geometry::{
    calc_pos_zyx, center_array, polarization_array, solid_angle_array, PoniFile, PosZyx, Unit,
};
use rsfai_integrate::{histogram1d, histogram2d, Hist2dOptions};
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

/// Knobs shared by [`AzimuthalIntegrator::integrate1d`] and
/// [`AzimuthalIntegrator::integrate2d`], mirroring the pyFAI integrate kwargs
/// that affect the per-pixel preprocessing and the reduction.
///
/// Radial/azimuth ranges are intentionally absent: no golden dataset exercises
/// a non-default range, so the range-normalization path is not yet ported (it
/// would ship unvalidated arithmetic). The full data range is always used.
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
}

impl Default for IntegrationOptions {
    /// pyFAI's defaults: solid angle on, no polarization, unit normalization,
    /// no error model.
    fn default() -> Self {
        IntegrationOptions {
            correct_solid_angle: true,
            polarization_factor: None,
            normalization_factor: 1.0,
            error_model: ErrorModel::No,
        }
    }
}

/// The 1D integration result, field-for-field the subset of pyFAI's
/// `Integrate1dResult` the engines populate. `radial` carries the unit scale;
/// the `sum_*` arrays are the raw f32 histograms (the `acc_t` accumulators
/// downcast to `data_t`, matching the 1D histogram engine).
#[derive(Debug, Clone)]
pub struct Integrate1dResult {
    /// Scaled radial bin centres (`position * unit.scale`), length `npt`.
    pub radial: Vec<f64>,
    /// Average intensity `signal / normalization` (f32).
    pub intensity: Vec<f32>,
    /// Standard error on the mean (= `sem`; f32). Zero unless an error model.
    pub sigma: Vec<f32>,
    /// Number of valid pixels per bin (f32).
    pub count: Vec<f32>,
    /// Binned signal (f32).
    pub sum_signal: Vec<f32>,
    /// Binned variance (f32).
    pub sum_variance: Vec<f32>,
    /// Binned normalization (f32).
    pub sum_normalization: Vec<f32>,
    /// Binned normalization² (f32).
    pub sum_normalization2: Vec<f32>,
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
    /// path are resolved (currently Pilatus1M); supply the detector explicitly
    /// via [`from_poni`](Self::from_poni) for others.
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

    /// The per-pixel `[signal, variance, norm, count]` rows (`preproc4`),
    /// applying the same corrections / mask / dummy pyFAI feeds to preproc.
    /// `data` is the image already cast to f32 (preproc's `data_t`); `pos`
    /// supplies the lab coordinates the polarization correction needs.
    fn preproc_rows(
        &self,
        data: &[f32],
        pos: &PosZyx,
        mask: Option<&[i8]>,
        opts: &IntegrationOptions,
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
            solidangle: solidangle.as_deref(),
            polarization: polarization.as_deref(),
            mask,
            normalization_factor: opts.normalization_factor,
            poissonian: opts.error_model.poissonian(),
            check_dummy: dummy.is_some(),
            dummy: dummy.unwrap_or(0.0),
            delta_dummy: delta_dummy.unwrap_or(0.0),
            ..Default::default()
        };
        preproc4(data, &popt)
    }

    /// Integrate `image` (the detector frame already cast to f32) into a 1D
    /// radial curve of `npt` bins in `unit`, using the no-split histogram path.
    /// `image` must be the flat row-major frame of length `detector.size()`.
    pub fn integrate1d(
        &self,
        image: &[f32],
        npt: usize,
        unit: Unit,
        opts: &IntegrationOptions,
    ) -> Integrate1dResult {
        assert_eq!(
            image.len(),
            self.detector.size(),
            "image length {} != detector size {}",
            image.len(),
            self.detector.size()
        );
        let pos = self.pixel_positions();
        let radial = center_array(unit, &pos.x, &pos.y, &pos.z, self.wavelength);
        let mask = self.build_mask();
        let prep = self.preproc_rows(image, &pos, mask.as_deref(), opts);

        // The 1D histogram bins every pixel (masked pixels contribute zeroed
        // rows from preproc but still set the data range); no mask, full range.
        let h = histogram1d(&radial, &prep, npt, None, opts.error_model, 0.0);

        Integrate1dResult {
            radial: h.position.iter().map(|&p| p * unit.scale).collect(),
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

    /// Integrate `image` into a 2D `(radial, azimuthal)` cake of `npt_rad ×
    /// npt_azim` bins, radial in `unit` and azimuth in degrees
    /// (`CHI_DEG`), using the no-split histogram path. The azimuthal
    /// discontinuity is at π (`chiDiscAtPi = True`, pyFAI's default).
    pub fn integrate2d(
        &self,
        image: &[f32],
        npt_rad: usize,
        npt_azim: usize,
        unit: Unit,
        opts: &IntegrationOptions,
    ) -> Integrate2dResult {
        assert_eq!(
            image.len(),
            self.detector.size(),
            "image length {} != detector size {}",
            image.len(),
            self.detector.size()
        );
        let pos = self.pixel_positions();
        let radial = center_array(unit, &pos.x, &pos.y, &pos.z, self.wavelength);
        let azimuthal = center_array(Unit::CHI_RAD, &pos.x, &pos.y, &pos.z, self.wavelength);
        let mask = self.build_mask();
        let prep = self.preproc_rows(image, &pos, mask.as_deref(), opts);

        let hopts = Hist2dOptions {
            bins: (npt_rad, npt_azim),
            radial_range: None,
            azimuth_range: None,
            error_model: opts.error_model,
            // Standard radial units (q/2θ/r) are non-negative.
            allow_radial_neg: false,
            chi_disc_at_pi: true,
            // pyFAI passes the azimuth unit's period (CHI_DEG.period = 360);
            // only its sign matters here — `> 0` enables the [-π, π] clip.
            pos1_period: 360.0,
            empty: 0.0,
        };
        let h = histogram2d(&radial, &azimuthal, &prep, mask.as_deref(), &hopts);

        Integrate2dResult {
            radial: h.radial.iter().map(|&r| r * unit.scale).collect(),
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
}

/// Resolve the detector model from a PONI's `Detector:` name. Only
/// golden-validated detectors are resolved; anything else is an error so the
/// integrator never silently runs an unverified detector path.
fn resolve_detector(poni: &PoniFile) -> Result<Detector> {
    match poni.detector.as_deref() {
        Some("Pilatus1M") => Ok(Detector::pilatus1m()),
        other => Err(Error::UnsupportedDetector(
            other.unwrap_or("<none>").to_string(),
        )),
    }
}
