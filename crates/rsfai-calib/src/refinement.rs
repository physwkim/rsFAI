//! `GeometryRefinement`, ported from `pyFAI/geometryRefinement.py`.
//!
//! The deterministic building blocks are bit-exact; only the iterative
//! [`refine`](GeometryRefinement::refine) is tolerance-gated (the optimizer
//! trajectory cannot match scipy's bit-for-bit). The split is enforced by
//! construction: the residual/chi2 core ([`tth`](GeometryRefinement::tth),
//! [`calc_2th`](GeometryRefinement::calc_2th),
//! [`residu1`](GeometryRefinement::residu1),
//! [`chi2`](GeometryRefinement::chi2)) never touches `argmin`; `refine` calls
//! that core through a cost closure.
//!
//! Forward model (the cost pyFAI's optimizer minimizes):
//!   * For each control point `(d1, d2)` (pixel coords) the geometry maps it to a
//!     2theta angle, `Geometry.tth` (`geometry/core.py:561`): build the cartesian
//!     position `(p1, p2)` in metres (`detector.calc_cartesian_positions` then
//!     subtract PONI), rotate into the sample frame `(t1, t2, t3)` via
//!     `f_t1/f_t2/f_t3` (`ext/_geometry.pyx`), then `atan2(sqrt(t1²+t2²), t3)`.
//!     This is exactly `rsfai_geometry::calc_pos_zyx` followed by `atan2` — the
//!     already-validated geometry transform (0-ULP algebra, the six rotation
//!     sin/cos are the only transcendentals).
//!   * Each ring's *expected* 2theta comes from the calibrant,
//!     `GeometryRefinement.calc_2th` (`geometryRefinement.py:413`): `tth[rings]`
//!     of the calibrant's `get_2th()` list, with pyFAI's optimizer-aid padding
//!     for rings beyond the visible list.
//!   * The residual vector is `tth(d1,d2,param) - calc_2th(rings)`
//!     (`residu1`), and chi2 is `dot(residual, residual)` (`residu2`).
//!
//! dtype contract: every coordinate, parameter, residual and chi2 is f64 (pyFAI
//! coerces `data` to float64 and refines in float64).

use rsfai_calibrant::Calibrant;
use rsfai_detectors::Detector;
use rsfai_geometry::transform::calc_pos_zyx;

/// The seven refinement parameters in pyFAI's `PARAM_ORDER`
/// (`geometryRefinement.py:92`). `wavelength` is carried so the calibrant 2theta
/// can be recomputed, but [`refine`](GeometryRefinement::refine) fixes it
/// (matching `refine2`, which defaults `fix=["wavelength"]`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GeometryParams {
    /// Sample-detector distance (m), `dist`.
    pub dist: f64,
    /// PONI coordinate along Y (m), `poni1`.
    pub poni1: f64,
    /// PONI coordinate along X (m), `poni2`.
    pub poni2: f64,
    /// Detector tilt around Y (rad), `rot1`.
    pub rot1: f64,
    /// Detector tilt around X (rad), `rot2`.
    pub rot2: f64,
    /// Detector tilt around the beam (rad), `rot3`.
    pub rot3: f64,
    /// Wavelength (m), `wavelength`.
    pub wavelength: f64,
}

impl GeometryParams {
    /// The six geometry parameters (excluding wavelength) as the array pyFAI's
    /// `residu1`/`chi2` pass as `param` (`self.param[:6]`).
    #[inline]
    pub fn six(&self) -> [f64; 6] {
        [
            self.dist, self.poni1, self.poni2, self.rot1, self.rot2, self.rot3,
        ]
    }

    /// Replace the six geometry parameters from an array (wavelength unchanged),
    /// mirroring `set_param` over `dist..rot3`.
    #[inline]
    fn with_six(&self, p: &[f64; 6]) -> GeometryParams {
        GeometryParams {
            dist: p[0],
            poni1: p[1],
            poni2: p[2],
            rot1: p[3],
            rot2: p[4],
            rot3: p[5],
            wavelength: self.wavelength,
        }
    }
}

/// The geometry refinement state: control points, the calibrant, the detector,
/// and the current parameter vector. `GeometryRefinement(AzimuthalIntegrator)`.
#[derive(Debug, Clone)]
pub struct GeometryRefinement {
    /// Control points as `(d1, d2, ring)` rows (slow, fast, ring index) — the
    /// `data` array (`geometryRefinement.py:136`), columns 0/1/2.
    points: Vec<(f64, f64, usize)>,
    /// The calibrant supplying the per-ring 2theta list.
    calibrant: Calibrant,
    /// The detector supplying the pixel size / orientation cartesian mapping.
    detector: Detector,
    /// Current parameter estimate.
    param: GeometryParams,
}

impl GeometryRefinement {
    /// Build a refinement problem. `points` are `(d1, d2, ring)` rows (typically
    /// from [`crate::ControlPoints::list_ring`]). The calibrant's wavelength is
    /// set to `param.wavelength` so `get_2th` matches pyFAI's
    /// `calibrant.setWavelength_change2th(self.wavelength)`
    /// (`geometryRefinement.py:185`).
    pub fn new(
        points: Vec<(f64, f64, usize)>,
        mut calibrant: Calibrant,
        detector: Detector,
        param: GeometryParams,
    ) -> GeometryRefinement {
        calibrant.set_wavelength(param.wavelength);
        GeometryRefinement {
            points,
            calibrant,
            detector,
            param,
        }
    }

    /// The current parameter estimate.
    pub fn param(&self) -> GeometryParams {
        self.param
    }

    /// Overwrite the parameter estimate (the six geometry params + wavelength).
    pub fn set_param(&mut self, param: GeometryParams) {
        self.param = param;
    }

    /// Number of control points.
    pub fn len(&self) -> usize {
        self.points.len()
    }

    /// Whether there are no control points.
    pub fn is_empty(&self) -> bool {
        self.points.is_empty()
    }

    /// The calibrant's predicted 2theta (rad) for each ring index,
    /// `GeometryRefinement.calc_2th` (`geometryRefinement.py:413-436`) at the
    /// **fixed** instance wavelength.
    ///
    /// Ports the optimizer-aid path verbatim: take the calibrant's visible 2theta
    /// list `ary = get_2th()`; if `len(ary) < rings.max()`, append
    /// `10.0*(rings.max()-len(ary))` placeholders so the optimizer is pushed away
    /// from unphysical ring assignments; then index `tth[rings]`. A ring index
    /// `>= len(tth)` is a hard error (pyFAI raises `IndexError`).
    pub fn calc_2th(&self, rings: &[usize]) -> Vec<f64> {
        // The instance wavelength was already pushed into the calibrant, so
        // `get_2th()` is the list at `self.wavelength` (the `wavelength !=
        // self.calibrant.wavelength` re-trigger in pyFAI is a no-op here).
        let mut ary: Vec<f64> = self.calibrant.get_2th().to_vec();
        let rings_max = rings.iter().copied().max().unwrap_or(0);
        // pyFAI: `if len(ary) < rings.max():` pad with the placeholder block.
        if ary.len() < rings_max {
            let pad = 1 + rings_max - ary.len();
            let fill = 10.0 * (rings_max - ary.len()) as f64;
            for _ in 0..pad {
                ary.push(fill);
            }
        }
        // pyFAI: `if rings.max() >= len(tth): raise IndexError`.
        assert!(
            rings_max < ary.len(),
            "ring index {rings_max} not available at this wavelength (calibrant has {} rings)",
            ary.len()
        );
        rings.iter().map(|&r| ary[r]).collect()
    }

    /// The 2theta (rad) of each control point under the geometry `param[:6]`,
    /// `Geometry.tth` for all points (`geometry/core.py:561-595`, cython path).
    ///
    /// Per pyFAI: cartesian position `(p1, p2)` = detector pixel position minus
    /// PONI, then `atan2(sqrt(t1²+t2²), t3)` with `(t1, t2, t3)` from the rotation
    /// (`calc_pos_zyx` here returns `(z=t3, y=t1, x=t2)`). `calc_pos_zyx`
    /// subtracts PONI internally and uses `L = dist` for the flat-detector
    /// (`pos3 is None`) path — bit-identical to `tth`'s
    /// `_calc_cartesian_positions` + `calc_tth` chain.
    pub fn tth(&self, param: &[f64; 6]) -> Vec<f64> {
        let n = self.points.len();
        let mut cart1 = Vec::with_capacity(n);
        let mut cart2 = Vec::with_capacity(n);
        for &(d1, d2, _) in &self.points {
            let (p1, p2) = point_cartesian(&self.detector, d1, d2);
            cart1.push(p1);
            cart2.push(p2);
        }
        let pos = calc_pos_zyx(
            param[0], // dist
            param[1], // poni1
            param[2], // poni2
            param[3], // rot1
            param[4], // rot2
            param[5], // rot3
            &cart1,
            &cart2,
            None,
            self.detector.orientation,
        );
        // f_tth = atan2(sqrt(t1²+t2²), t3) with t1=y, t2=x, t3=z (_geometry.pyx:145).
        pos.y
            .iter()
            .zip(pos.x.iter())
            .zip(pos.z.iter())
            .map(|((&t1, &t2), &t3)| (t1 * t1 + t2 * t2).sqrt().atan2(t3))
            .collect()
    }

    /// The residual vector `tth(d1,d2,param) - calc_2th(rings)`,
    /// `GeometryRefinement.residu1` (`geometryRefinement.py:452-453`). Bit-exact
    /// given identical params + control points + calibrant (the only ULP-budgeted
    /// part is the geometry `atan2`/`sin`/`cos`, inherited from the validated
    /// `calc_pos_zyx`).
    pub fn residu1(&self, param: &[f64; 6]) -> Vec<f64> {
        let measured = self.tth(param);
        let rings: Vec<usize> = self.points.iter().map(|&(_, _, r)| r).collect();
        let expected = self.calc_2th(&rings);
        measured
            .iter()
            .zip(expected.iter())
            .map(|(&m, &e)| m - e)
            .collect()
    }

    /// chi2 = `dot(residual, residual)`, `GeometryRefinement.residu2`/`chi2`
    /// (`geometryRefinement.py:458-462`, `676-679`). pyFAI uses `numpy.dot(t, t)`,
    /// a left-to-right f64 fused-sum; we replicate the sequential left-fold so the
    /// chi2 is bit-identical (no FMA, no pairwise reorder).
    pub fn chi2(&self, param: &[f64; 6]) -> f64 {
        let t = self.residu1(param);
        // numpy.dot(t, t): sequential sum of t[i]*t[i] in f64.
        let mut acc = 0.0_f64;
        for &ti in &t {
            acc += ti * ti;
        }
        acc
    }

    /// chi2 at the current parameter estimate.
    pub fn chi2_current(&self) -> f64 {
        self.chi2(&self.param.six())
    }

    /// Refine the six geometry parameters (wavelength fixed) by minimizing chi2,
    /// `GeometryRefinement.refine2` (`geometryRefinement.py:589`, which is
    /// `refine3(fix=["wavelength"])`). The wavelength is always fixed here (the
    /// calibrant 2theta list is held); `rot3` stays free.
    ///
    /// **Tolerance gate, NOT bit-exact.** See [`refine_with_fixed`].
    ///
    /// [`refine_with_fixed`]: GeometryRefinement::refine_with_fixed
    pub fn refine(&mut self) -> f64 {
        self.refine_with_fixed(&[])
    }

    /// Refine the geometry parameters, holding those in `fixed` (any subset of
    /// `dist`/`poni1`/`poni2`/`rot1`/`rot2`/`rot3`) at their current value —
    /// `GeometryRefinement.refine3(fix=["wavelength", ...fixed])`
    /// (`geometryRefinement.py:509`). Wavelength is always fixed (not in the
    /// 6-parameter search). Returns the chi2 of the retained parameter vector.
    ///
    /// **Tolerance gate, NOT bit-exact.** pyFAI uses `scipy.optimize.fmin_slsqp`;
    /// here a Nelder-Mead simplex minimizes the (bit-exact) chi2 over the free
    /// coordinates. The two trajectories differ by construction, so the converged
    /// parameters are validated at a recorded relative tolerance and the
    /// converged cost is asserted `<=` pyFAI's converged cost — never claimed
    /// bit-exact. The optimizer is isolated in [`crate::optimizer`]; it only ever
    /// evaluates `self.chi2`, leaving the bit-exact core untouched.
    ///
    /// pyFAI keeps the new parameters only if `chi2_new < chi2_old`
    /// (`refine3:578`); we mirror that guard.
    pub fn refine_with_fixed(&mut self, fixed: &[Param]) -> f64 {
        let free: Vec<usize> = (0..6)
            .filter(|i| !fixed.contains(&Param::ALL[*i]))
            .collect();
        assert!(
            !free.is_empty(),
            "at least one geometry parameter must be free"
        );
        let old = self.param.six();
        let old_chi2 = self.chi2(&old);
        let cost = |p: &[f64; 6]| self.chi2(p);
        let (new, new_chi2) = crate::optimizer::minimize_chi2(&old, &free, cost);
        if new_chi2 < old_chi2 {
            self.param = self.param.with_six(&new);
            new_chi2
        } else {
            old_chi2
        }
    }
}

/// One of the six geometry parameters, used to pin parameters during
/// [`GeometryRefinement::refine_with_fixed`]. Indices match `PARAM_ORDER[:6]`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Param {
    /// `dist` (index 0).
    Dist,
    /// `poni1` (index 1).
    Poni1,
    /// `poni2` (index 2).
    Poni2,
    /// `rot1` (index 3).
    Rot1,
    /// `rot2` (index 4).
    Rot2,
    /// `rot3` (index 5).
    Rot3,
}

impl Param {
    /// The six parameters in `PARAM_ORDER[:6]` order, so `ALL[i]` is the param at
    /// index `i`.
    const ALL: [Param; 6] = [
        Param::Dist,
        Param::Poni1,
        Param::Poni2,
        Param::Rot1,
        Param::Rot2,
        Param::Rot3,
    ];
}

/// Cartesian position `(p1, p2)` in metres of a single control point at float
/// pixel coords `(d1, d2)` — the no-spline / no-pixel-corner branch of
/// `Detector.calc_cartesian_positions` (`detectors/_common.py:722-797`):
/// reorder by orientation, add the half-pixel offset, multiply by pixel size.
/// `p3` is `None` for the flat detectors in scope. PONI is not subtracted here
/// (`calc_pos_zyx` subtracts it, matching `tth`'s chain).
fn point_cartesian(det: &Detector, d1: f64, d2: f64) -> (f64, f64) {
    let (rd1, rd2) = reorder_point(det, d1, d2);
    // center=True: d1c = d1 + 0.5 (the half-pixel offset).
    let d1c = rd1 + 0.5;
    let d2c = rd2 + 0.5;
    // no spline -> dX = dY = 0: p = pixel * dc.
    (det.pixel1 * d1c, det.pixel2 * d2c)
}

/// Reorder a float pixel coordinate by orientation, the `center=True` branch of
/// `_reorder_indexes_from_orientation` (`detectors/_common.py:657-678`):
/// orientations 0/3 are identity; 1/2/4 flip about `shape - 1`.
fn reorder_point(det: &Detector, d1: f64, d2: f64) -> (f64, f64) {
    let max0 = (det.shape.0 - 1) as f64;
    let max1 = (det.shape.1 - 1) as f64;
    match det.orientation {
        0 | 3 => (d1, d2),
        1 => (max0 - d1, max1 - d2),
        2 => (max0 - d1, d2),
        4 => (d1, max1 - d2),
        o => panic!("point_cartesian: unsupported orientation {o} (valid 0..=4)"),
    }
}
