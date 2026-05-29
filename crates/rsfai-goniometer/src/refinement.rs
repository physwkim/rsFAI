//! `Goniometer` + `GoniometerRefinement`, ported from `pyFAI/goniometer.py`.
//!
//! A [`Goniometer`] wraps a [`GeometryTransformation`], the parameter vector it
//! is currently set to, a detector, and a wavelength. [`Goniometer::get_ai`]
//! evaluates the transformation at a motor position and builds an
//! [`rsfai::AzimuthalIntegrator`] from the resulting six PONI scalars — reusing
//! the already-bit-exact PONI→geometry path.
//!
//! A [`GoniometerRefinement`] extends that with a set of [`SingleGeometry`]
//! (each a motor position plus the control-point fit at that position) and
//! refines the parameter vector by minimizing the average squared 2theta error
//! across all geometries. The cost is the bit-exact
//! [`rsfai_calib::GeometryRefinement::chi2`] summed over the geometries; only the
//! converged-parameter *search* ([`GoniometerRefinement::refine`]) is iterative
//! and tolerance-gated, and it is isolated behind [`crate::optimizer`].
//!
//! ## Cost function (pyFAI `GoniometerRefinement.residu2`)
//!
//! For each single geometry: take its motor position, transform the goniometer
//! parameter vector to a PONI (`trans(param, pos)`), then evaluate that
//! geometry's control-point chi² at the transformed six-parameter vector
//! (`single.geometry_refinement.chi2_wavelength([dist,…,rot3, wavelength·1e10])`).
//! Because a [`GeometryTransformation`] carries no wavelength formula, the
//! wavelength stays the fixed instance wavelength, so `chi2_wavelength` reduces
//! to the ordinary [`rsfai_calib::GeometryRefinement::chi2`] at the transformed
//! six parameters. The total cost is `Σ chi²ᵢ / max(Σ nptᵢ, 1)` (mean squared
//! error over every control point of every geometry).

use rsfai::AzimuthalIntegrator;
use rsfai_calib::GeometryRefinement;
use rsfai_detectors::Detector;
use rsfai_geometry::PoniFile;

use crate::transform::{GeometryTransformation, PoniParam, TransformError};

/// One detector position on the goniometer arm: the motor position and the
/// control-point fit recorded at that position.
///
/// pyFAI's `SingleGeometry` also carries the image, massif, and calibrant for
/// automatic control-point extraction; here the control points are taken as
/// given (the extraction is a separate concern), so a single geometry is just
/// the motor position plus the [`rsfai_calib::GeometryRefinement`] that holds the
/// control points + calibrant + detector for that frame.
#[derive(Debug, Clone)]
pub struct SingleGeometry {
    /// The goniometer motor position for this frame (length = `pos_names`).
    pub position: Vec<f64>,
    /// The control-point fit (control points, calibrant, detector) at this frame.
    pub refinement: GeometryRefinement,
}

impl SingleGeometry {
    /// Build a single geometry from a motor position and its control-point fit.
    pub fn new(position: Vec<f64>, refinement: GeometryRefinement) -> SingleGeometry {
        SingleGeometry {
            position,
            refinement,
        }
    }
}

/// The goniometer geometry model: a transformation, the current parameter
/// vector, a detector, and a wavelength (m). `pyFAI.goniometer.Goniometer`.
#[derive(Debug, Clone)]
pub struct Goniometer {
    /// The PONI-component transformation.
    pub trans: GeometryTransformation,
    /// The current goniometer parameter vector (length = `trans.param_names`).
    pub param: Vec<f64>,
    /// The detector mounted on the moving arm.
    pub detector: Detector,
    /// The experiment wavelength (m).
    pub wavelength: f64,
}

impl Goniometer {
    /// Build a goniometer from its transformation, parameter vector, detector,
    /// and wavelength.
    pub fn new(
        trans: GeometryTransformation,
        param: Vec<f64>,
        detector: Detector,
        wavelength: f64,
    ) -> Goniometer {
        assert_eq!(
            param.len(),
            trans.param_names().len(),
            "param length {} != transformation param_names {}",
            param.len(),
            trans.param_names().len()
        );
        Goniometer {
            trans,
            param,
            detector,
            wavelength,
        }
    }

    /// Evaluate the transformation at the current parameter vector and the given
    /// motor `position`, returning the six PONI scalars. `Goniometer.get_ai`'s
    /// first step (`trans_function(self.param, position)`).
    pub fn poni_at(&self, position: &[f64]) -> Result<PoniParam, TransformError> {
        self.trans.call(&self.param, position)
    }

    /// Build an [`AzimuthalIntegrator`] for a motor `position`:
    /// `Goniometer.get_ai`. The transformation produces the PONI; the integrator
    /// is constructed from it (the validated PONI→geometry path) with this
    /// goniometer's detector and wavelength.
    pub fn get_ai(&self, position: &[f64]) -> Result<AzimuthalIntegrator, TransformError> {
        let p = self.poni_at(position)?;
        let poni = PoniFile {
            poni_version: None,
            detector: None,
            detector_config: None,
            orientation: Some(self.detector.orientation),
            dist: p.dist,
            poni1: p.poni1,
            poni2: p.poni2,
            rot1: p.rot1,
            rot2: p.rot2,
            rot3: p.rot3,
            wavelength: Some(self.wavelength),
            pixel1: None,
            pixel2: None,
        };
        Ok(AzimuthalIntegrator::from_poni(&poni, self.detector.clone()))
    }
}

/// A goniometer refinement: the goniometer model plus the set of single
/// geometries whose control points constrain the parameter vector.
/// `pyFAI.goniometer.GoniometerRefinement`.
#[derive(Debug, Clone)]
pub struct GoniometerRefinement {
    /// The goniometer model (transformation + param + detector + wavelength).
    pub goniometer: Goniometer,
    /// The single geometries (motor position + control-point fit) to fit against.
    pub single_geometries: Vec<SingleGeometry>,
}

impl GoniometerRefinement {
    /// Build a refinement from a goniometer and its single geometries.
    pub fn new(
        goniometer: Goniometer,
        single_geometries: Vec<SingleGeometry>,
    ) -> GoniometerRefinement {
        GoniometerRefinement {
            goniometer,
            single_geometries,
        }
    }

    /// The current goniometer parameter vector.
    pub fn param(&self) -> &[f64] {
        &self.goniometer.param
    }

    /// The mean squared 2theta error for the parameter vector `param`,
    /// `GoniometerRefinement.residu2` — `Σ chi²ᵢ / max(Σ nptᵢ, 1)`.
    ///
    /// For each single geometry, transform `param` at the geometry's motor
    /// position to a PONI, set that geometry's control-point fit to the six
    /// transformed parameters, and accumulate its bit-exact chi² and its
    /// control-point count. Geometries with no control points contribute nothing
    /// (pyFAI's `len(...data) >= 1` guard). The accumulation is a sequential f64
    /// left-fold, matching pyFAI's Python `+=` loop.
    ///
    /// Returns a [`TransformError`] if a formula references an unbound name. The
    /// chi² itself is bit-exact (the geometry transcendentals are the only
    /// ULP-budgeted part, inherited from `rsfai_calib`).
    pub fn residu2(&self, param: &[f64]) -> Result<f64, TransformError> {
        let mut sumsquare = 0.0_f64;
        let mut npt = 0usize;
        for single in &self.single_geometries {
            let p = self.goniometer.trans.call(param, &single.position)?;
            if !single.refinement.is_empty() {
                sumsquare += single.refinement.chi2(&p.as_array());
                npt += single.refinement.len();
            }
        }
        Ok(sumsquare / npt.max(1) as f64)
    }

    /// The cost at the current parameter vector, `GoniometerRefinement.chi2`.
    pub fn chi2(&self) -> Result<f64, TransformError> {
        self.residu2(&self.goniometer.param)
    }

    /// Refine the goniometer parameter vector by minimizing [`residu2`], keeping
    /// the new vector only if its cost improves (pyFAI `refine2`'s
    /// `new_error < former_error` guard). Returns the cost of the retained vector.
    ///
    /// **Tolerance gate, NOT bit-exact.** pyFAI uses `scipy.optimize.minimize`
    /// (SLSQP); here a Nelder-Mead simplex ([`crate::optimizer`]) minimizes the
    /// (bit-exact) `residu2`. The trajectories differ by construction, so the
    /// converged parameters are validated at a recorded relative tolerance and
    /// the converged cost is asserted `<=` pyFAI's. The optimizer only ever
    /// evaluates `residu2`, leaving the bit-exact core untouched.
    ///
    /// [`residu2`]: GoniometerRefinement::residu2
    pub fn refine(&mut self) -> Result<f64, TransformError> {
        let start = self.goniometer.param.clone();
        let old_cost = self.residu2(&start)?;
        // The optimizer needs a plain `Fn(&[f64]) -> f64`. A formula error would
        // be a configuration bug (the same formulas evaluated cleanly at `start`),
        // so surface it as a panic inside the closure rather than threading
        // `Result` through argmin; `residu2` at `start` above already proved the
        // formulas evaluate.
        let cost = |p: &[f64]| {
            self.residu2(p)
                .expect("transformation formula evaluation failed during refinement")
        };
        let (new, new_cost) = crate::optimizer::minimize(&start, cost);
        if new_cost < old_cost {
            self.goniometer.param = new;
            Ok(new_cost)
        } else {
            Ok(old_cost)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rsfai_calib::GeometryParams;
    use rsfai_calibrant::Calibrant;

    /// A single-motor goniometer: dist/poni constant, rot2 affine in the motor.
    fn affine_goniometer(param: Vec<f64>) -> Goniometer {
        let trans = GeometryTransformation::new(
            Some("dist"),
            Some("poni1"),
            Some("poni2"),
            Some("0.0"),
            Some("scale * pos + offset"),
            Some("0.0"),
            &["dist", "poni1", "poni2", "scale"],
            None,
            &[("offset", 0.0)],
        )
        .unwrap();
        let det = Detector::generic(1.0e-4, 1.0e-4, (1000, 1000));
        Goniometer::new(trans, param, det, 1.0e-10)
    }

    #[test]
    fn get_ai_uses_transformed_poni() {
        let gonio = affine_goniometer(vec![0.2, 0.1, 0.11, 0.01]);
        let ai = gonio.get_ai(&[3.0]).unwrap();
        // rot2 = 0.01 * 3.0 + 0.0 = 0.03; everything else from the constants.
        assert_eq!(ai.dist, 0.2);
        assert_eq!(ai.poni1, 0.1);
        assert_eq!(ai.poni2, 0.11);
        assert_eq!(ai.rot1, 0.0);
        assert_eq!(ai.rot2, 0.01 * 3.0);
        assert_eq!(ai.rot3, 0.0);
        assert_eq!(ai.wavelength, 1.0e-10);
    }

    #[test]
    fn residu2_is_mean_squared_error_over_geometries() {
        // Two single geometries with a tiny control-point set each. The cost is
        // the sum of per-geometry chi2 over the total control-point count.
        let gonio = affine_goniometer(vec![0.2, 0.1, 0.11, 0.0]);
        let det = Detector::generic(1.0e-4, 1.0e-4, (1000, 1000));

        let make_single = |pos: f64| -> SingleGeometry {
            // A LaB6-like d-spacing list; the precise values do not matter for the
            // accounting test, only that the fit has control points.
            let cal = Calibrant::from_dspacing(vec![4.0, 2.0, 1.5]);
            let points = vec![(500.0, 500.0, 0usize), (510.0, 480.0, 1usize)];
            let gp = GeometryParams {
                dist: 0.2,
                poni1: 0.1,
                poni2: 0.11,
                rot1: 0.0,
                rot2: 0.0,
                rot3: 0.0,
                wavelength: 1.0e-10,
            };
            let refinement = GeometryRefinement::new(points, cal, det.clone(), gp);
            SingleGeometry::new(vec![pos], refinement)
        };

        let singles = vec![make_single(0.0), make_single(1.0)];
        let gr = GoniometerRefinement::new(gonio, singles);

        // Recompute the expected mean-squared error directly: Σ chi2 / Σ npt.
        let param = gr.param().to_vec();
        let mut sumsquare = 0.0;
        let mut npt = 0usize;
        for s in &gr.single_geometries {
            let p = gr.goniometer.trans.call(&param, &s.position).unwrap();
            sumsquare += s.refinement.chi2(&p.as_array());
            npt += s.refinement.len();
        }
        let expected = sumsquare / npt as f64;
        assert_eq!(gr.residu2(&param).unwrap(), expected);
        assert_eq!(npt, 4);
    }
}
