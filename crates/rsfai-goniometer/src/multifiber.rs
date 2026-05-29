//! `MultiGeometryFiber`, ported from the `MultiGeometryFiber` class in
//! `pyFAI/multi_geometry.py`.
//!
//! `MultiGeometryFiber` combines several [`rsfai_fiber::FiberIntegrator`] frames
//! (one detector position per geometry) into a single fiber profile / map.
//!
//! **It does NOT use the `Integrate1dResult.union` left-fold** the azimuthal
//! [`rsfai_multigeometry::MultiGeometry`] uses. pyFAI's `MultiGeometryFiber`
//! instead sums the per-geometry accumulators directly, bin by bin:
//!
//! ```text
//! count         += res.count
//! signal        += res.sum_signal
//! normalization += res.sum_normalization * sac
//! ```
//!
//! where `sac = pixel1·pixel2 / dist²` is the per-geometry solid-angle correction
//! (1.0 when `correctSolidAngle` is off), applied to the **normalization only**.
//! After the geometries are summed it computes
//!
//! ```text
//! norm        = maximum(normalization, float32.tiny)
//! intensity   = signal / norm
//! intensity[count <= 0] = empty
//! ```
//!
//! Everything is f64 (the per-geometry fiber accumulators are f64), so the
//! combined result is **bit-exact** to pyFAI given identical inputs: the only
//! reorderable step is the cross-geometry `+=` loop, which is reproduced as the
//! same sequential left-fold (`(((0 + g0) + g1) + …)`) pyFAI's Python loop runs.
//!
//! pyFAI's fiber path carries **no error model** (`sum_variance` is `None`), so
//! there is no variance / crossed-term path — the result has no `sigma`. This
//! mirrors [`rsfai_fiber`], whose `Integrate*FiberResult` carry no variance.

use rsfai::{Corrections, IntegrationOptions};
use rsfai_fiber::{FiberAxes, FiberIntegrator, FiberUnit};
use rsfai_geometry::fiber_center_array;

/// `numpy.finfo("float32").tiny` — the smallest positive normal f32 — as the f64
/// the normalization clamp uses. `numpy.maximum(normalization, tiny)` upcasts
/// `tiny` to f64; this is that upcast value bit-for-bit.
const F32_TINY: f64 = f32::MIN_POSITIVE as f64;

/// The 1D combined fiber result. `intensity` is f64 (`signal / norm`, recomputed
/// from the summed f64 accumulators); there is no `sigma` (fiber has no error
/// model).
#[derive(Debug, Clone)]
pub struct MultiFiber1dResult {
    /// Output-axis bin centres, taken from the last geometry's result (every
    /// geometry shares the common range + bin count, so the axes are identical).
    pub axis: Vec<f64>,
    /// Average intensity `signal / max(norm, tiny)`, with `empty` where the
    /// summed count is `<= 0` (f64).
    pub intensity: Vec<f64>,
    /// Cross-geometry summed signal (f64).
    pub sum_signal: Vec<f64>,
    /// Cross-geometry summed, solid-angle-corrected normalization (f64).
    pub sum_normalization: Vec<f64>,
    /// Cross-geometry summed valid-pixel count (f64).
    pub count: Vec<f64>,
}

/// The 2D combined fiber result. Arrays are flat `(oop, ip)` row-major (cell
/// `(oop j, ip i)` at `j * npt_ip + i`), the layout [`rsfai_fiber`] exposes.
#[derive(Debug, Clone)]
pub struct MultiFiber2dResult {
    /// In-plane bin centres, from the last geometry's result.
    pub inplane: Vec<f64>,
    /// Out-of-plane bin centres, from the last geometry's result.
    pub outofplane: Vec<f64>,
    /// `(npt_ip, npt_oop)` bin counts, for indexing the flat arrays.
    pub bins: (usize, usize),
    /// Average intensity `signal / max(norm, tiny)`, with `empty` where the
    /// summed count is `<= 0` (f64).
    pub intensity: Vec<f64>,
    /// Cross-geometry summed signal (f64).
    pub sum_signal: Vec<f64>,
    /// Cross-geometry summed, solid-angle-corrected normalization (f64).
    pub sum_normalization: Vec<f64>,
    /// Cross-geometry summed valid-pixel count (f64).
    pub count: Vec<f64>,
}

/// A multi-geometry fiber integrator: the list of fiber integrators (one per
/// detector position), the common fiber axes, and the empty-bin fill value.
/// `pyFAI.multi_geometry.MultiGeometryFiber`.
#[derive(Debug, Clone)]
pub struct MultiGeometryFiber {
    /// The fiber integrators, one per goniometer position.
    pub fis: Vec<FiberIntegrator>,
    /// The common fiber axes (units, bin counts, ranges) shared by every frame.
    pub axes: FiberAxes,
    /// Value for empty cells (`count <= 0`); pyFAI default `0.0`.
    pub empty: f64,
}

impl MultiGeometryFiber {
    /// Build a multi-geometry fiber integrator from its fiber integrators and the
    /// common axes. `empty` defaults to `0.0` via [`MultiGeometryFiber::with_empty`].
    pub fn new(fis: Vec<FiberIntegrator>, axes: FiberAxes) -> MultiGeometryFiber {
        Self::with_empty(fis, axes, 0.0)
    }

    /// Build with an explicit empty-cell fill value.
    pub fn with_empty(
        fis: Vec<FiberIntegrator>,
        axes: FiberAxes,
        empty: f64,
    ) -> MultiGeometryFiber {
        assert!(!fis.is_empty(), "list of fiber integrators cannot be empty");
        MultiGeometryFiber { fis, axes, empty }
    }

    /// The per-geometry solid-angle correction `sac = pixel1·pixel2 / dist²`
    /// (`1.0` when `correct_solid_angle` is off), applied to the normalization of
    /// geometry `i`. pyFAI: `(fi.pixel1 * fi.pixel2 / fi.dist ** 2)`.
    fn sac(&self, i: usize, correct_solid_angle: bool) -> f64 {
        if !correct_solid_angle {
            return 1.0;
        }
        let fi = &self.fis[i];
        let pixel1 = fi.ai.detector.pixel1;
        let pixel2 = fi.ai.detector.pixel2;
        let dist = fi.ai.dist;
        pixel1 * pixel2 / dist.powi(2)
    }

    /// The per-frame scaled fiber-position array for `unit`, i.e. the value
    /// pyFAI's `Geometry.array_from_unit(unit=..., scale=True)` returns for
    /// frame `i`. Used to guess the common integration range.
    fn unit_array(&self, i: usize, unit: FiberUnit) -> Vec<f64> {
        let fi = &self.fis[i];
        let pos = fi.ai.pixel_positions();
        fiber_center_array(
            unit.space,
            unit.scale,
            &pos.x,
            &pos.y,
            &pos.z,
            fi.ai.wavelength,
            fi.gi,
        )
    }

    /// The common `(min, max)` range, in the scaled unit, over every frame's
    /// position array for `unit`. pyFAI's `_guess_inplane_range` /
    /// `_guess_outofplane_range`: `ip = array([fi.array_from_unit(unit) for fi in
    /// fis]); (ip.min(), ip.max())`. The min/max is over the concatenation of all
    /// frames' arrays; reduction order does not affect the extrema.
    fn guess_range(&self, unit: FiberUnit) -> (f64, f64) {
        let mut lo = f64::INFINITY;
        let mut hi = f64::NEG_INFINITY;
        for i in 0..self.fis.len() {
            for v in self.unit_array(i, unit) {
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

    /// Resolve the per-call axes: pyFAI computes **one** common ip/oop range
    /// across all frames *once* (when the stored range is `None`) and passes that
    /// fixed range to every frame's `integrate*_fiber`. Without this each frame
    /// would guess its own data range, so frames with different detector PONI
    /// geometry would bin onto different edges and the cross-frame `+=` would mix
    /// incompatible bins. Returns axes with `ip_range`/`oop_range` filled.
    fn resolved_axes(&self) -> FiberAxes {
        let mut axes = self.axes;
        if axes.ip_range.is_none() {
            axes.ip_range = Some(self.guess_range(axes.unit_ip));
        }
        if axes.oop_range.is_none() {
            axes.oop_range = Some(self.guess_range(axes.unit_oop));
        }
        axes
    }

    /// Combine the geometries into a 2D fiber map, `MultiGeometryFiber.integrate2d_fiber`.
    ///
    /// Each `lst_data[i]` is the detector image for geometry `i`; `opts`/`corr`
    /// are applied per geometry (the same corrections for each — pyFAI broadcasts
    /// a single mask/flat to all frames). The cross-geometry accumulation is the
    /// sequential left-fold pyFAI's `for res in results: signal += …` runs, so the
    /// summed accumulators are bit-exact.
    pub fn integrate2d_fiber(
        &self,
        lst_data: &[&[f32]],
        opts: &IntegrationOptions,
        corr: &Corrections,
    ) -> MultiFiber2dResult {
        assert_eq!(
            lst_data.len(),
            self.fis.len(),
            "lst_data length {} != number of geometries {}",
            lst_data.len(),
            self.fis.len()
        );
        let npt_ip = self.axes.npt_ip;
        let npt_oop = self.axes.npt_oop;
        let n = npt_ip * npt_oop;

        let mut signal = vec![0.0_f64; n];
        let mut normalization = vec![0.0_f64; n];
        let mut count = vec![0.0_f64; n];
        // The axes are identical across geometries (shared range + bins); keep the
        // last geometry's, matching pyFAI reading them off the final `res`.
        let mut inplane = Vec::new();
        let mut outofplane = Vec::new();

        // One common range across all frames, then the same range for each frame.
        let axes = self.resolved_axes();
        for (i, data) in lst_data.iter().enumerate() {
            let res = self.fis[i].integrate2d_fiber(data, &axes, opts, corr);
            let sac = self.sac(i, opts.correct_solid_angle);
            for k in 0..n {
                count[k] += res.count[k];
                signal[k] += res.sum_signal[k];
                normalization[k] += res.sum_normalization[k] * sac;
            }
            inplane = res.inplane;
            outofplane = res.outofplane;
        }

        let intensity = combine_intensity(&signal, &normalization, &count, self.empty);
        MultiFiber2dResult {
            inplane,
            outofplane,
            bins: (npt_ip, npt_oop),
            intensity,
            sum_signal: signal,
            sum_normalization: normalization,
            count,
        }
    }

    /// Combine the geometries into a 1D fiber profile,
    /// `MultiGeometryFiber.integrate_fiber` (its `integrate1d` alias).
    ///
    /// `vertical_integration` selects the fold axis exactly as
    /// [`rsfai_fiber::FiberIntegrator::integrate_fiber`] does (the per-geometry 1D
    /// fold runs first, then the geometries are summed). The cross-geometry
    /// accumulation is the same sequential left-fold as the 2D path.
    pub fn integrate_fiber(
        &self,
        lst_data: &[&[f32]],
        vertical_integration: bool,
        opts: &IntegrationOptions,
        corr: &Corrections,
    ) -> MultiFiber1dResult {
        assert_eq!(
            lst_data.len(),
            self.fis.len(),
            "lst_data length {} != number of geometries {}",
            lst_data.len(),
            self.fis.len()
        );
        // Output-axis length: oop bins for vertical integration, else ip bins.
        let nout = if vertical_integration {
            self.axes.npt_oop
        } else {
            self.axes.npt_ip
        };

        let mut signal = vec![0.0_f64; nout];
        let mut normalization = vec![0.0_f64; nout];
        let mut count = vec![0.0_f64; nout];
        let mut axis = Vec::new();

        // One common range across all frames, then the same range for each frame.
        let axes = self.resolved_axes();
        for (i, data) in lst_data.iter().enumerate() {
            let res = self.fis[i].integrate_fiber(data, &axes, vertical_integration, opts, corr);
            let sac = self.sac(i, opts.correct_solid_angle);
            for k in 0..nout {
                count[k] += res.count[k];
                signal[k] += res.sum_signal[k];
                normalization[k] += res.sum_normalization[k] * sac;
            }
            axis = res.axis;
        }

        let intensity = combine_intensity(&signal, &normalization, &count, self.empty);
        MultiFiber1dResult {
            axis,
            intensity,
            sum_signal: signal,
            sum_normalization: normalization,
            count,
        }
    }
}

/// The shared final step of `integrate_fiber` / `integrate2d_fiber`:
/// `intensity = signal / maximum(normalization, tiny)`, then `intensity[count <=
/// 0] = empty`. pyFAI computes the division everywhere first, then overwrites the
/// invalid cells; the two steps collapse to one branch here with no change in the
/// value written to a valid cell (the clamp keeps the denominator strictly
/// positive, so `signal / norm` is always finite).
fn combine_intensity(signal: &[f64], normalization: &[f64], count: &[f64], empty: f64) -> Vec<f64> {
    (0..signal.len())
        .map(|k| {
            if count[k] <= 0.0 {
                empty
            } else {
                signal[k] / normalization[k].max(F32_TINY)
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn f32_tiny_matches_numpy() {
        // numpy.finfo("float32").tiny upcast to f64 is 1.1754943508222875e-38,
        // bits 0x3810000000000000 (numpy's struct.pack("<d", tiny) prints the
        // little-endian bytes 00 00 00 00 00 00 10 38, i.e. this u64).
        assert_eq!(F32_TINY.to_bits(), 0x3810_0000_0000_0000);
        assert_eq!(F32_TINY, 1.175_494_350_822_287_5e-38);
    }

    #[test]
    fn combine_intensity_clamps_and_fills_empty() {
        let signal = [10.0, 0.0, 4.0];
        let norm = [2.0, 0.0, 0.0]; // bin 2 has count but zero norm -> tiny clamp.
        let count = [3.0, 0.0, 1.0]; // bin 1 empty.
        let out = combine_intensity(&signal, &norm, &count, -1.0);
        assert_eq!(out[0], 5.0);
        assert_eq!(out[1], -1.0); // empty fill
                                  // bin 2: 4.0 / max(0, tiny) = 4.0 / tiny (finite, huge).
        assert_eq!(out[2], 4.0 / F32_TINY);
    }
}
