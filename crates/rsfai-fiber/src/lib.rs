//! `rsfai-fiber` — fiber / grazing-incidence (GIWAXS/GISAXS) integration, ported
//! from `pyFAI/integrator/fiber.py`.
//!
//! A [`FiberIntegrator`] wraps an [`rsfai::AzimuthalIntegrator`] plus the
//! grazing-incidence sample geometry ([`GiParams`]: incident/tilt angle,
//! sample orientation). Its core operation, [`FiberIntegrator::integrate2d_fiber`],
//! reshapes the detector pattern as a 2D map over two fiber units (in-plane
//! `qip` × out-of-plane `qoop` by default) — pyFAI's `integrate2d_fiber` is an
//! ordinary 2D `("no","histogram","cython")` integration where *both* axes are
//! fiber position arrays rather than (radial, chi). It delegates to
//! [`rsfai::AzimuthalIntegrator::integrate2d_positions`], reusing the
//! already-bit-exact preprocessing + no-split histogram engine.
//!
//! [`FiberIntegrator::integrate_fiber`] (1D) then sums the 2D accumulators along
//! one axis (numpy `pairwise_sum`, ported here for bit-exactness) and recomputes
//! `intensity = signal/norm` and `sigma = sqrt(variance)/norm` per pyFAI.
//!
//! Scope: the default `qip`/`qoop`, the exit/scattering angles, and `qtot` — all
//! fiber units with `positive=False` (signed) and `period=None` (no azimuthal
//! wrap). The polar units (`chigi`, which carries a `2π`/`360` period) need the
//! chi-wrap path and are not yet wired here.

use std::f64::consts::PI;

use rsfai::{AzimuthalIntegrator, Corrections, IntegrationOptions, Positions2dBinning};
use rsfai_geometry::{fiber_center_array, FiberSpace, GiParams};

/// numpy's pairwise-summation block size (`PW_BLOCKSIZE`): runs of `n <= 128`
/// use the 8-accumulator unrolled sum, larger `n` splits in half (rounded down
/// to a multiple of 8) and recurses.
const PW_BLOCKSIZE: usize = 128;

/// Degrees-per-radian, the `scale` pyFAI's `*_deg` fiber units carry.
const DEG_PER_RAD: f64 = 180.0 / PI;

/// A fiber/GI integration axis: which quantity ([`FiberSpace`]) and the `scale`
/// applied to the reported bin centres — pyFAI's `unit.scale`: `1.0` for
/// `nm^-1`/`rad`, `0.1` for `Å^-1`, `180/π` for `deg`. Every unit here is signed
/// (`positive=False`) and period-free (no azimuthal wrap), matching pyFAI.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FiberUnit {
    /// The fiber quantity the axis measures.
    pub space: FiberSpace,
    /// Reported-axis scale (`unit.scale`), applied to the binned centres.
    pub scale: f64,
}

impl FiberUnit {
    /// In-plane scattering vector `q_IP` in `nm^-1` (pyFAI `qip_nm^-1`).
    pub const QIP_NM: FiberUnit = FiberUnit {
        space: FiberSpace::Qip,
        scale: 1.0,
    };
    /// Out-of-plane scattering vector `q_OOP` in `nm^-1` (pyFAI `qoop_nm^-1`).
    pub const QOOP_NM: FiberUnit = FiberUnit {
        space: FiberSpace::Qoop,
        scale: 1.0,
    };
    /// `q_IP` in `Å^-1` (pyFAI `qip_A^-1`).
    pub const QIP_A: FiberUnit = FiberUnit {
        space: FiberSpace::Qip,
        scale: 0.1,
    };
    /// `q_OOP` in `Å^-1` (pyFAI `qoop_A^-1`).
    pub const QOOP_A: FiberUnit = FiberUnit {
        space: FiberSpace::Qoop,
        scale: 0.1,
    };
    /// Total `|q|` in `nm^-1` (pyFAI `qtot_nm^-1`).
    pub const QTOT_NM: FiberUnit = FiberUnit {
        space: FiberSpace::Qtotal,
        scale: 1.0,
    };
    /// Vertical exit angle in degrees (pyFAI `exit_angle_vert_deg`).
    pub const EXIT_ANGLE_VERT_DEG: FiberUnit = FiberUnit {
        space: FiberSpace::ExitAngleVert,
        scale: DEG_PER_RAD,
    };
    /// Horizontal exit angle in degrees (pyFAI `exit_angle_horz_deg`).
    pub const EXIT_ANGLE_HORZ_DEG: FiberUnit = FiberUnit {
        space: FiberSpace::ExitAngleHorz,
        scale: DEG_PER_RAD,
    };
    /// Vertical exit angle in radians (pyFAI `exit_angle_vert_rad`).
    pub const EXIT_ANGLE_VERT_RAD: FiberUnit = FiberUnit {
        space: FiberSpace::ExitAngleVert,
        scale: 1.0,
    };
    /// Horizontal exit angle in radians (pyFAI `exit_angle_horz_rad`).
    pub const EXIT_ANGLE_HORZ_RAD: FiberUnit = FiberUnit {
        space: FiberSpace::ExitAngleHorz,
        scale: 1.0,
    };
}

/// The two fiber axes of a [`FiberIntegrator::integrate2d_fiber`] call: bin
/// counts, units, and optional `(min, max)` ranges (in the scaled unit) for the
/// in-plane (`ip`) and out-of-plane (`oop`) directions. Mirrors pyFAI's
/// `npt_ip`/`unit_ip`/`ip_range` + `npt_oop`/`unit_oop`/`oop_range`.
#[derive(Debug, Clone, Copy)]
pub struct FiberAxes {
    /// In-plane bin count (pyFAI `npt_ip`).
    pub npt_ip: usize,
    /// In-plane unit (pyFAI `unit_ip`; default `qip_nm^-1`).
    pub unit_ip: FiberUnit,
    /// In-plane `(min, max)` in the scaled unit, or `None` for the data range.
    pub ip_range: Option<(f64, f64)>,
    /// Out-of-plane bin count (pyFAI `npt_oop`).
    pub npt_oop: usize,
    /// Out-of-plane unit (pyFAI `unit_oop`; default `qoop_nm^-1`).
    pub unit_oop: FiberUnit,
    /// Out-of-plane `(min, max)` in the scaled unit, or `None` for the data range.
    pub oop_range: Option<(f64, f64)>,
}

impl FiberAxes {
    /// The pyFAI default axes: `qip_nm^-1` × `qoop_nm^-1`, full data ranges.
    pub fn qip_qoop_nm(npt_ip: usize, npt_oop: usize) -> Self {
        FiberAxes {
            npt_ip,
            unit_ip: FiberUnit::QIP_NM,
            ip_range: None,
            npt_oop,
            unit_oop: FiberUnit::QOOP_NM,
            oop_range: None,
        }
    }
}

/// 2D fiber/GI result, mirroring pyFAI's `Integrate2dFiberResult`. Arrays are
/// flat `(oop, ip)` row-major (cell `(oop j, ip i)` at `j * npt_ip + i`), the
/// layout pyFAI exposes after transpose.
///
/// pyFAI's fiber path runs `integrate2d_ng` with no `error_model`, so its
/// `sum_variance`/`sum_normalization2`/`std`/`sem` are `None`; this struct
/// carries only the populated fields. `intensity` is f32 (engine output); the
/// `sum_*`/`count` accumulators are f64.
#[derive(Debug, Clone)]
pub struct Integrate2dFiberResult {
    /// In-plane bin centres (scaled `unit_ip`), length `npt_ip`.
    pub inplane: Vec<f64>,
    /// Out-of-plane bin centres (scaled `unit_oop`), length `npt_oop`.
    pub outofplane: Vec<f64>,
    /// `(npt_ip, npt_oop)` bin counts, for indexing the flat arrays.
    pub bins: (usize, usize),
    /// Average intensity `signal / normalization` (f32).
    pub intensity: Vec<f32>,
    /// Per-cell valid-pixel count (f64).
    pub count: Vec<f64>,
    /// Binned signal (f64).
    pub sum_signal: Vec<f64>,
    /// Binned normalization (f64).
    pub sum_normalization: Vec<f64>,
}

/// 1D fiber/GI result, mirroring pyFAI's `Integrate1dFiberResult`. `intensity`
/// is **f64** — recomputed from the f64 2D accumulators summed along one axis
/// (pyFAI's numexpr path on f64 arrays). pyFAI's fiber path has no error model,
/// so there is no `sigma`.
#[derive(Debug, Clone)]
pub struct Integrate1dFiberResult {
    /// Output-axis bin centres (pyFAI's `integrated`): the out-of-plane axis when
    /// `vertical_integration`, else the in-plane axis.
    pub axis: Vec<f64>,
    /// Average intensity `signal / normalization` (f64), `0.0` where `norm <= 0`
    /// or `count == 0`.
    pub intensity: Vec<f64>,
    /// Axis-summed signal (f64).
    pub sum_signal: Vec<f64>,
    /// Axis-summed normalization (f64).
    pub sum_normalization: Vec<f64>,
    /// Axis-summed valid-pixel count (f64).
    pub count: Vec<f64>,
    /// Whether the in-plane axis was integrated out (`true`) or the oop axis.
    pub vertical_integration: bool,
}

/// A fiber / grazing-incidence integrator: an [`AzimuthalIntegrator`] plus the
/// grazing-incidence sample geometry, mirroring `pyFAI.integrator.fiber.FiberIntegrator`.
#[derive(Debug, Clone)]
pub struct FiberIntegrator {
    /// The underlying azimuthal integrator (PONI geometry + detector).
    pub ai: AzimuthalIntegrator,
    /// Grazing-incidence sample geometry (incident/tilt angle, orientation).
    pub gi: GiParams,
}

impl FiberIntegrator {
    /// Build from an [`AzimuthalIntegrator`] and the grazing-incidence geometry.
    pub fn new(ai: AzimuthalIntegrator, gi: GiParams) -> Self {
        FiberIntegrator { ai, gi }
    }

    /// Reshape the detector pattern as a 2D map over two fiber units (`unit_ip` ×
    /// `unit_oop`). Mirrors pyFAI's `integrate2d_fiber`: an ordinary 2D no-split
    /// histogram where both axes are fiber position arrays.
    pub fn integrate2d_fiber(
        &self,
        image: &[f32],
        axes: &FiberAxes,
        opts: &IntegrationOptions,
        corr: &Corrections,
    ) -> Integrate2dFiberResult {
        let pos = self.ai.pixel_positions();
        let wl = self.ai.wavelength;
        let pos0 = fiber_center_array(axes.unit_ip.space, 1.0, &pos.x, &pos.y, &pos.z, wl, self.gi);
        let pos1 = fiber_center_array(
            axes.unit_oop.space,
            1.0,
            &pos.x,
            &pos.y,
            &pos.z,
            wl,
            self.gi,
        );
        let binning = Positions2dBinning {
            npt0: axes.npt_ip,
            npt1: axes.npt_oop,
            pos0_range: axes.ip_range,
            pos1_range: axes.oop_range,
            pos0_scale: axes.unit_ip.scale,
            pos1_scale: axes.unit_oop.scale,
            // Every supported fiber unit is `positive=False` and `period=None`.
            allow_pos0_neg: true,
            pos1_period: 0.0,
        };
        let r = self
            .ai
            .integrate2d_positions(image, &pos0, &pos1, &binning, opts, corr);
        Integrate2dFiberResult {
            inplane: r.radial,
            outofplane: r.azimuthal,
            bins: r.bins,
            intensity: r.intensity,
            count: r.count,
            sum_signal: r.sum_signal,
            sum_normalization: r.sum_normalization,
        }
    }

    /// Integrate the 2D fiber map down to a 1D profile by summing one axis.
    /// Mirrors pyFAI's `integrate_fiber`: `vertical_integration` (default in
    /// pyFAI) sums out the in-plane axis (profile vs out-of-plane); otherwise it
    /// sums out the out-of-plane axis (profile vs in-plane). The per-axis sum is
    /// numpy's `pairwise_sum` and `intensity` is recomputed in f64. pyFAI's fiber
    /// path has no error model, so there is no `sigma`.
    pub fn integrate_fiber(
        &self,
        image: &[f32],
        axes: &FiberAxes,
        vertical_integration: bool,
        opts: &IntegrationOptions,
        corr: &Corrections,
    ) -> Integrate1dFiberResult {
        let r2 = self.integrate2d_fiber(image, axes, opts, corr);
        let (npt_ip, npt_oop) = r2.bins;

        // The output axis is the one that survives the integration.
        let axis = if vertical_integration {
            r2.outofplane.clone()
        } else {
            r2.inplane.clone()
        };
        let fold = |a: &[f64]| fold_axis(a, vertical_integration, npt_ip, npt_oop);

        let sum_signal = fold(&r2.sum_signal);
        let count = fold(&r2.count);
        let sum_normalization = fold(&r2.sum_normalization);
        let n = axis.len();

        // pyFAI: intensity = where(norm<=0, 0, signal/norm); then [count==0]=empty
        // (empty = 0.0 with no dummy). The two-step collapses to one branch here.
        let intensity: Vec<f64> = (0..n)
            .map(|k| {
                if sum_normalization[k] <= 0.0 || count[k] == 0.0 {
                    0.0
                } else {
                    sum_signal[k] / sum_normalization[k]
                }
            })
            .collect();

        Integrate1dFiberResult {
            axis,
            intensity,
            sum_signal,
            sum_normalization,
            count,
            vertical_integration,
        }
    }
}

/// Sum the flat `(oop, ip)` 2D accumulator `a` (cell `(oop j, ip i)` at
/// `j * npt_ip + i`) along one axis via [`pairwise_sum`]. `vertical` sums the
/// contiguous in-plane axis per oop row (numpy `sum(axis=-1)`, stride 1) →
/// length `npt_oop`; otherwise sums the strided out-of-plane axis per ip column
/// (numpy `sum(axis=-2)`, stride `npt_ip`) → length `npt_ip`.
fn fold_axis(a: &[f64], vertical: bool, npt_ip: usize, npt_oop: usize) -> Vec<f64> {
    if vertical {
        (0..npt_oop)
            .map(|j| pairwise_sum(a, j * npt_ip, npt_ip, 1))
            .collect()
    } else {
        (0..npt_ip)
            .map(|i| pairwise_sum(a, i, npt_oop, npt_ip))
            .collect()
    }
}

/// numpy's `pairwise_sum_DOUBLE`: sum `n` elements of `a` starting at `base`,
/// every `stride` elements apart. Runs of `< 8` sum naively; `8..=128` use the
/// 8-accumulator unrolled reduction with numpy's exact `((r0+r1)+(r2+r3)) +
/// ((r4+r5)+(r6+r7))` tree plus a sequential tail; larger runs split in half
/// (rounded down to a multiple of 8) and recurse.
///
/// This is numpy's *scalar* `pairwise_sum` reference. numpy's runtime may take a
/// SIMD-vectorized path (different lane count → different intra-block rounding),
/// so the fold is not strictly bit-exact vs pyFAI's `.sum(axis=...)`: the golden
/// verifier measures ≤ 2 ULP (rel ~2e-16) on isolated elements, gated under
/// `FIBER_INT_REL_BUDGET`. The 2D accumulators it folds are bit-exact.
fn pairwise_sum(a: &[f64], base: usize, n: usize, stride: usize) -> f64 {
    if n < 8 {
        let mut res = 0.0;
        for i in 0..n {
            res += a[base + i * stride];
        }
        res
    } else if n <= PW_BLOCKSIZE {
        let mut r = [0.0f64; 8];
        for (k, slot) in r.iter_mut().enumerate() {
            *slot = a[base + k * stride];
        }
        let limit = n - (n % 8);
        let mut i = 8;
        while i < limit {
            for (k, slot) in r.iter_mut().enumerate() {
                *slot += a[base + (i + k) * stride];
            }
            i += 8;
        }
        let mut res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
        while i < n {
            res += a[base + i * stride];
            i += 1;
        }
        res
    } else {
        let mut n2 = n / 2;
        n2 -= n2 % 8;
        pairwise_sum(a, base, n2, stride) + pairwise_sum(a, base + n2 * stride, n - n2, stride)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pairwise_sum_matches_naive_for_small_runs() {
        let a: Vec<f64> = (0..7).map(|i| i as f64 + 0.5).collect();
        let naive: f64 = a.iter().sum();
        assert_eq!(pairwise_sum(&a, 0, a.len(), 1), naive);
    }

    #[test]
    fn pairwise_sum_strided_picks_every_stride_th() {
        // 3 columns; sum column 1 (indices 1, 4, 7, 10).
        let a = vec![0.0, 1.0, 0.0, 0.0, 2.0, 0.0, 0.0, 3.0, 0.0, 0.0, 4.0, 0.0];
        assert_eq!(pairwise_sum(&a, 1, 4, 3), 10.0);
    }

    #[test]
    fn pairwise_sum_recurses_above_blocksize() {
        // 200 > 128 forces the recursive split; ones sum exactly to n.
        let a = vec![1.0f64; 200];
        assert_eq!(pairwise_sum(&a, 0, 200, 1), 200.0);
    }
}
