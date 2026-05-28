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

use std::f64::consts::PI;

use rsfai_core::dtype::ErrorModel;
use rsfai_detectors::Detector;
use rsfai_geometry::{
    calc_pos_zyx, center_array, corner_array_f32, delta_radial, polarization_array,
    solid_angle_array, unscaled_center_array, PoniFile, PosZyx, Unit,
};
use rsfai_integrate::{
    build_bbox_csc_1d, build_bbox_csr_1d, build_bbox_lut_1d, build_full_csc_1d, build_full_csr_1d,
    build_full_lut_1d, csc_integrate1d, csr_integrate1d, histogram1d, histogram1d_bbox,
    histogram1d_full, histogram2d, lut_integrate1d, CsrIntegrate1d, Hist2dOptions, Integrate1d,
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
    /// Pixel-split + binning algorithm. The default `("no", "histogram")` keeps
    /// the original no-split path; `bbox`/`full` engage the corner/delta geometry.
    pub method: Method,
}

impl Default for IntegrationOptions {
    /// pyFAI's defaults: solid angle on, no polarization, unit normalization,
    /// no error model, no-split histogram.
    fn default() -> Self {
        IntegrationOptions {
            correct_solid_angle: true,
            polarization_factor: None,
            normalization_factor: 1.0,
            error_model: ErrorModel::No,
            method: Method::default(),
        }
    }
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
        assert_eq!(
            image.len(),
            self.detector.size(),
            "image length {} != detector size {}",
            image.len(),
            self.detector.size()
        );
        let pos = self.pixel_positions();
        let radial = unscaled_center_array(unit.space, &pos.x, &pos.y, &pos.z, self.wavelength);
        let mask = self.build_mask();
        let prep = self.preproc_rows(image, &pos, mask.as_deref(), opts);
        let em = opts.error_model;

        // The no-split histogram is the one path whose accumulators are f32; it
        // also bins every pixel (masked pixels are zeroed by preproc but still
        // set the range), so no mask is forwarded.
        if opts.method.split == Split::No && opts.method.algo == Algo::Histogram {
            let h = histogram1d(&radial, &prep, npt, None, em, 0.0);
            return histogram_to_result(h, unit.scale);
        }

        // Every other method excludes masked pixels from the matrix/scatter.
        // Radial units (q/2θ/r) are non-negative ⇒ `allow_pos0_neg = false`. The
        // 1D full-split build is not handed chiDiscAtPi / pos1_period (common.py
        // does not forward them), so they take the defaults true / 2π.
        let mask_ref = mask.as_deref();
        let allow_neg = false;
        let r = match opts.method.split {
            Split::Full => {
                // Full split bins on the four-corner array (radial + chi), f32
                // upcast to f64 — the input pyFAI's FullSplitIntegrator receives.
                let corners = self.corner_array(unit);
                let cf: Vec<f64> = corners.iter().map(|&v| v as f64).collect();
                match opts.method.algo {
                    Algo::Histogram => histogram1d_full(
                        &cf, &prep, mask_ref, npt, em, 0.0, allow_neg, true, TWO_PI,
                    ),
                    Algo::Csr => {
                        let (m, c) = build_full_csr_1d(&cf, mask_ref, npt, allow_neg, true, TWO_PI);
                        csr_integrate1d(&m, &prep, c, em, 0.0)
                    }
                    Algo::Lut => {
                        let (m, c) = build_full_lut_1d(&cf, mask_ref, npt, allow_neg, true, TWO_PI);
                        lut_integrate1d(&m, &prep, c, em, 0.0)
                    }
                    Algo::Csc => {
                        let (m, c) = build_full_csc_1d(&cf, mask_ref, npt, allow_neg, true, TWO_PI);
                        csc_integrate1d(&m, &prep, c, em, 0.0)
                    }
                }
            }
            split => {
                // No / Bbox: the bbox-family build on (pos0, dpos0). No-split has
                // dpos0 = None (each pixel collapses to its center); bbox uses the
                // radial half-width `delta_array`.
                let delta = (split == Split::Bbox).then(|| {
                    let corners = self.corner_array(unit);
                    delta_radial(&corners, &radial)
                });
                let dpos0 = delta.as_deref();
                match opts.method.algo {
                    Algo::Histogram => {
                        // No-split-histogram returns above; this is bbox-histogram.
                        let d = dpos0.expect("bbox split has a radial delta");
                        histogram1d_bbox(&radial, d, &prep, mask_ref, npt, em, 0.0, allow_neg)
                    }
                    Algo::Csr => {
                        let (m, c) = build_bbox_csr_1d(&radial, dpos0, mask_ref, npt, allow_neg);
                        csr_integrate1d(&m, &prep, c, em, 0.0)
                    }
                    Algo::Lut => {
                        let (m, c) = build_bbox_lut_1d(&radial, dpos0, mask_ref, npt, allow_neg);
                        lut_integrate1d(&m, &prep, c, em, 0.0)
                    }
                    Algo::Csc => {
                        let (m, c) = build_bbox_csc_1d(&radial, dpos0, mask_ref, npt, allow_neg);
                        csc_integrate1d(&m, &prep, c, em, 0.0)
                    }
                }
            }
        };
        csr_to_result(r, unit.scale)
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
        // Bin on the unscaled internal radial; the reported axis scales once
        // (the 2D dispatch for bbox/full is wired in a later step — this path is
        // still the no-split histogram).
        let radial = unscaled_center_array(unit.space, &pos.x, &pos.y, &pos.z, self.wavelength);
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
