//! Fiber / grazing-incidence (GI) unit equations, ported from pyFAI's
//! grazing-incidence units in `pyFAI/units.py` (`eq_qip`, `eq_qoop`,
//! `eq_q_total`, `eq_chi_gi`, and the sample-frame q components `eq_qbeam`/
//! `eq_qhorz`/`eq_qvert`).
//!
//! Unlike the standard radial units ([`crate::units`]), a fiber unit also depends
//! on the sample's grazing-incidence geometry — `incident_angle`, `tilt_angle`,
//! and the fiber-axis `sample_orientation` (1-8) — carried here in [`GiParams`].
//!
//! ## Pipeline (pyFAI)
//! Per pixel, lab coords `(x, y, z)` map to the lab-frame scattering vector
//! `(q_beam, q_horz, q_vert)` (`q_lab_*`), which is rotated into the sample frame
//! by `rotate_q_lab` (`Rx(-tilt) @ Ry(inc)`). The fiber units are then:
//!
//! * `qip`     = `sign(q_horz)·√(q_beam² + q_horz²)`  (in-plane)
//! * `qoop`    = `q_vert`                             (out-of-plane = `eq_qvert`)
//! * `chi_gi`  = `atan2(qip, qoop)`
//! * `q_total` = `4e-9·π·sin(2θ/2)/λ`, `2θ = atan2(√(x²+y²), z)`
//!
//! The `sample_orientation` remaps `(x, y)` (pyFAI `MAPS_SAMPLE_ORIENTATION`)
//! once, inside `q_lab` (pyFAI applies it in the decorated `q_lab_*`; the
//! higher-level `eq_qip`/`eq_qoop`/`eq_chi_gi` are not decorated, so it is applied
//! exactly once). `q_total` carries the same remap but is orientation-invariant
//! in value (`√(x²+y²)` is symmetric in the remap).
//!
//! The **scattering-angle** (`atan2(y, √(z²+x²))` / `atan2(x, z)`) and
//! **exit-angle** units are direct on `(x, y, z)`, not the q vector; the exit
//! angles first apply the *other* sample rotation, [`rotate_cartesian`]
//! (`Rz(tilt) @ Rx(-inc)`, distinct axes/order from [`rotate_q_lab`]). These
//! equations are decorated in pyFAI, so they remap `(x, y)` themselves.
//!
//! ## Bit-exactness (Tier B, ULP-budgeted)
//! pyFAI builds the `rotate_q_lab` rotation via `scipy.spatial.transform.Rotation`
//! (an internal quaternion), so its matrix differs by ≤1 ULP from the direct
//! `cos`/`sin` matrix used here (measured; see kodex `f3389aef`). The fiber unit
//! arrays therefore carry a small, measured ULP budget against golden — they are
//! not asserted bit-exact like `r`/`q`/`2th`. `sin`/`cos`/`atan2`/`sqrt` are libm
//! transcendentals (the standing Tier-B caveat) on top of that.

use std::f64::consts::PI;

/// Grazing-incidence sample geometry carried by a fiber unit: `incident_angle`
/// (sample tilt toward the beam, ≈ rot2) and `tilt_angle` (tilt orthogonal to
/// the beam, ≈ rot3), both in radians, plus the EXIF-style fiber-axis
/// `sample_orientation` (1-8). pyFAI defaults: `0.0, 0.0, 1`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GiParams {
    pub incident_angle: f64,
    pub tilt_angle: f64,
    pub sample_orientation: u8,
}

impl Default for GiParams {
    fn default() -> Self {
        GiParams {
            incident_angle: 0.0,
            tilt_angle: 0.0,
            sample_orientation: 1,
        }
    }
}

/// Which fiber/GI quantity a unit measures (selects the equation).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FiberSpace {
    /// In-plane scattering vector `sign(q_horz)·√(q_beam²+q_horz²)` (sample frame).
    Qip,
    /// Out-of-plane scattering vector = sample-frame vertical component
    /// (pyFAI `eq_qoop = eq_qvert`).
    Qoop,
    /// Total `|q|` joining `qip`/`qoop`: `4e-9·π·sin(2θ/2)/λ`.
    Qtotal,
    /// Polar angle in the GI frame: `atan2(qip, qoop)`.
    ChiGi,
    /// Sample-frame beam component (`eq_qbeam`).
    Qbeam,
    /// Sample-frame horizontal component (`eq_qhorz`).
    Qhorz,
    /// Vertical scattering angle relative to the direct beam, no sample rotation:
    /// `atan2(y, √(z²+x²))` (pyFAI `eq_scattering_angle_vertical`), radians.
    ScatteringAngleVert,
    /// Horizontal scattering angle relative to the direct beam, no sample
    /// rotation: `atan2(x, z)` (pyFAI `eq_scattering_angle_horz`), radians.
    ScatteringAngleHorz,
    /// Vertical exit angle relative to the horizon (thin films): scattering angle
    /// after `rotate_cartesian` (pyFAI `eq_exit_angle_vert`), radians.
    ExitAngleVert,
    /// Horizontal exit angle relative to the horizon: horizontal scattering angle
    /// after `rotate_cartesian` (pyFAI `eq_exit_angle_horz`), radians.
    ExitAngleHorz,
}

/// pyFAI `numpy.sign`: `0.0` at zero (Rust's `f64::signum` returns `±1.0` there).
#[inline]
fn np_sign(x: f64) -> f64 {
    if x > 0.0 {
        1.0
    } else if x < 0.0 {
        -1.0
    } else {
        0.0
    }
}

/// pyFAI `MAPS_SAMPLE_ORIENTATION`: remap `(x, y)` for the fiber-axis orientation
/// (1-8) before the lab-q equations. The map values reference the *original*
/// `x`/`y` (e.g. orientation 5 = `(-y, -x)`).
#[inline]
fn remap_xy(sample_orientation: u8, x: f64, y: f64) -> (f64, f64) {
    match sample_orientation {
        1 => (x, y),
        2 => (-x, y),
        3 => (-x, -y),
        4 => (x, -y),
        5 => (-y, -x),
        6 => (-y, x),
        7 => (y, x),
        8 => (y, -x),
        other => panic!("sample_orientation {other} out of range 1-8"),
    }
}

/// Lab-frame scattering vector `(q_beam, q_horz, q_vert)` for one pixel, after the
/// sample-orientation remap. pyFAI `q_lab_{beam,horz,vert}`: with
/// `k = 2e-9·π/λ` and `norm = √(x²+y²+z²)`,
/// `q_horz = k·x/norm`, `q_vert = k·y/norm`, `q_beam = k·(z/norm − 1)`.
#[inline]
fn q_lab(x: f64, y: f64, z: f64, wavelength: f64, sample_orientation: u8) -> (f64, f64, f64) {
    let (x, y) = remap_xy(sample_orientation, x, y);
    let k = 2.0e-9 * PI / wavelength;
    let norm = ((x * x + y * y) + z * z).sqrt();
    let q_horz = k * x / norm;
    let q_vert = k * y / norm;
    let q_beam = k * (z / norm - 1.0);
    (q_beam, q_horz, q_vert)
}

/// Rotate the lab-frame q into the sample frame: pyFAI `rotate_q_lab` applies
/// `Rx(-tilt) @ Ry(inc)` (scipy `from_euler("yx", [inc, -tilt])`, extrinsic) to
/// `(q_beam, q_horz, q_vert)`, axes `(beam, horz, vert) = (0, 1, 2)`. Direct
/// cos/sin matrix — ≤1 ULP from pyFAI's quaternion build (module docs).
#[inline]
fn rotate_q_lab(
    qb: f64,
    qh: f64,
    qv: f64,
    incident_angle: f64,
    tilt_angle: f64,
) -> (f64, f64, f64) {
    let (ci, si) = (incident_angle.cos(), incident_angle.sin());
    let (ct, st) = (tilt_angle.cos(), tilt_angle.sin());
    // M = Rx(-tilt) @ Ry(inc) =
    //   [[      ci,   0,    si  ],
    //    [ -st·si,   ct,  st·ci ],
    //    [ -ct·si,  -st,  ct·ci ]]
    let qb2 = ci * qb + si * qv;
    let qh2 = -st * si * qb + ct * qh + st * ci * qv;
    let qv2 = -ct * si * qb - st * qh + ct * ci * qv;
    (qb2, qh2, qv2)
}

/// Sample-frame scattering vector `(q_beam, q_horz, q_vert)` (pyFAI `q_sample`):
/// `rotate_q_lab(q_lab(...))`.
#[inline]
fn q_sample(x: f64, y: f64, z: f64, wavelength: f64, gi: GiParams) -> (f64, f64, f64) {
    let (qb, qh, qv) = q_lab(x, y, z, wavelength, gi.sample_orientation);
    rotate_q_lab(qb, qh, qv, gi.incident_angle, gi.tilt_angle)
}

/// Rotate cartesian lab coords into the sample frame (pyFAI `rotate_cartesian`):
/// `Rz(tilt) @ Rx(-inc)` (scipy `from_euler("xz", [-inc, tilt])`, extrinsic) on
/// `(x, y, z)`, axes `(x, y, z) = (0, 1, 2)` — the rotation behind the exit-angle
/// units. Distinct from [`rotate_q_lab`] (different axes/order). Direct cos/sin
/// matrix, ≤1 ULP from pyFAI's quaternion build (module docs).
#[inline]
fn rotate_cartesian(
    x: f64,
    y: f64,
    z: f64,
    incident_angle: f64,
    tilt_angle: f64,
) -> (f64, f64, f64) {
    let (ci, si) = (incident_angle.cos(), incident_angle.sin());
    let (ct, st) = (tilt_angle.cos(), tilt_angle.sin());
    // M = Rz(tilt) @ Rx(-inc) =
    //   [[ ct, -st·ci, -st·si],
    //    [ st,  ct·ci,  ct·si],
    //    [  0,    -si,     ci ]]
    let x_rot = ct * x - st * ci * y - st * si * z;
    let y_rot = st * x + ct * ci * y + ct * si * z;
    let z_rot = -si * y + ci * z;
    (x_rot, y_rot, z_rot)
}

/// The base (unscaled) fiber-unit value for one pixel. `wavelength` (m) is used
/// by every q-space variant; `gi` supplies the GI rotation + fiber orientation.
#[inline]
pub fn fiber_equation(
    space: FiberSpace,
    x: f64,
    y: f64,
    z: f64,
    wavelength: f64,
    gi: GiParams,
) -> f64 {
    // Units that do not need the sample-frame q vector. Each remaps (x, y) first
    // (pyFAI applies @change_sample_orientation on these equations directly).
    match space {
        FiberSpace::Qtotal => {
            // 4e-9·π·sin(2θ/2)/λ, 2θ = atan2(√(x²+y²), z) on the remapped (x, y).
            let (x, y) = remap_xy(gi.sample_orientation, x, y);
            let two_theta = (x * x + y * y).sqrt().atan2(z);
            return 4.0e-9 * PI * (two_theta / 2.0).sin() / wavelength;
        }
        FiberSpace::ScatteringAngleVert => {
            let (x, y) = remap_xy(gi.sample_orientation, x, y);
            return y.atan2((z * z + x * x).sqrt());
        }
        FiberSpace::ScatteringAngleHorz => {
            let (x, _y) = remap_xy(gi.sample_orientation, x, y);
            return x.atan2(z);
        }
        FiberSpace::ExitAngleVert => {
            let (x, y) = remap_xy(gi.sample_orientation, x, y);
            let (xr, yr, zr) = rotate_cartesian(x, y, z, gi.incident_angle, gi.tilt_angle);
            return yr.atan2((zr * zr + xr * xr).sqrt());
        }
        FiberSpace::ExitAngleHorz => {
            let (x, y) = remap_xy(gi.sample_orientation, x, y);
            let (xr, _yr, zr) = rotate_cartesian(x, y, z, gi.incident_angle, gi.tilt_angle);
            return xr.atan2(zr);
        }
        _ => {}
    }
    let (qb, qh, qv) = q_sample(x, y, z, wavelength, gi);
    match space {
        FiberSpace::Qip => (qb * qb + qh * qh).sqrt() * np_sign(qh),
        FiberSpace::Qoop => qv,
        FiberSpace::Qbeam => qb,
        FiberSpace::Qhorz => qh,
        FiberSpace::ChiGi => {
            let qip = (qb * qb + qh * qh).sqrt() * np_sign(qh);
            qip.atan2(qv)
        }
        _ => unreachable!("non-q_sample fiber units handled above"),
    }
}

/// The scaled center value for one pixel (`fiber_equation · scale`).
#[inline]
pub fn fiber_center_value(
    space: FiberSpace,
    scale: f64,
    x: f64,
    y: f64,
    z: f64,
    wavelength: f64,
    gi: GiParams,
) -> f64 {
    fiber_equation(space, x, y, z, wavelength, gi) * scale
}

/// Apply [`fiber_center_value`] over flat lab-coordinate slices. Per-pixel map
/// (each element independent) → bit-exact under parallelism.
pub fn fiber_center_array(
    space: FiberSpace,
    scale: f64,
    x: &[f64],
    y: &[f64],
    z: &[f64],
    wavelength: f64,
    gi: GiParams,
) -> Vec<f64> {
    use rayon::prelude::*;
    assert_eq!(x.len(), y.len());
    assert_eq!(x.len(), z.len());
    (0..x.len())
        .into_par_iter()
        .map(|i| fiber_center_value(space, scale, x[i], y[i], z[i], wavelength, gi))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Ground truth from pyFAI's `eq_*` functions (daq, pyFAI 2026.5.0), 17-sig
    /// decimals that round-trip to the exact f64. The rel tolerance (1e-12) is far
    /// above the ≤1-ULP scipy-quaternion-vs-direct-matrix gap, so it catches a
    /// structural port error (wrong remap / rotation order) while tolerating the
    /// sanctioned Tier-B divergence. Pixels: x/y/z below, λ=1e-10, inc=0.2,
    /// tilt=0.4, sample_orientation=1.
    const X: [f64; 3] = [0.012, -0.03, 0.001];
    const Y: [f64; 3] = [0.025, 0.004, -0.02];
    const Z: f64 = 0.15;

    fn assert_close(label: &str, got: f64, want: f64) {
        let rel = (got - want).abs() / want.abs().max(1e-30);
        assert!(
            rel < 1e-12,
            "{label}: got {got:.17}, want {want:.17} (rel {rel:e})"
        );
    }

    #[test]
    fn fiber_units_match_pyfai_ground_truth() {
        let gi = GiParams {
            incident_angle: 0.2,
            tilt_angle: 0.4,
            sample_orientation: 1,
        };
        let want_qip = [
            8.624_200_735_825_397,
            -10.660_115_356_579_36,
            -3.5116198594450516,
        ];
        let want_qoop = [
            7.5623147101094075,
            6.506_641_356_482_763,
            -7.556_528_499_384_117,
        ];
        let want_qtot = [
            11.470197997704679,
            12.488972782317642,
            8.332_622_456_301_092,
        ];
        let want_chigi = [
            0.850_907_418_912_445_5,
            -1.022_784_869_805_106,
            -2.7065706746016507,
        ];
        let want_qbeam = [
            1.0196988068009858,
            -0.890_165_757_356_195_1,
            -2.19124556595813,
        ];
        let want_qhorz = [
            8.563_705_545_803_17,
            -10.622883993530642,
            -2.7440694063594164,
        ];
        let wl = 1e-10;
        for i in 0..3 {
            let (x, y, z) = (X[i], Y[i], Z);
            assert_close(
                "qip",
                fiber_equation(FiberSpace::Qip, x, y, z, wl, gi),
                want_qip[i],
            );
            assert_close(
                "qoop",
                fiber_equation(FiberSpace::Qoop, x, y, z, wl, gi),
                want_qoop[i],
            );
            assert_close(
                "qtot",
                fiber_equation(FiberSpace::Qtotal, x, y, z, wl, gi),
                want_qtot[i],
            );
            assert_close(
                "chigi",
                fiber_equation(FiberSpace::ChiGi, x, y, z, wl, gi),
                want_chigi[i],
            );
            assert_close(
                "qbeam",
                fiber_equation(FiberSpace::Qbeam, x, y, z, wl, gi),
                want_qbeam[i],
            );
            assert_close(
                "qhorz",
                fiber_equation(FiberSpace::Qhorz, x, y, z, wl, gi),
                want_qhorz[i],
            );
        }
    }

    /// sample_orientation 2 = (-x, y) and 5 = (-y, -x) remaps, vs pyFAI.
    #[test]
    fn fiber_units_match_pyfai_other_orientations() {
        let wl = 1e-10;
        let so2 = GiParams {
            incident_angle: 0.2,
            tilt_angle: 0.4,
            sample_orientation: 2,
        };
        let want_qip2 = [
            -1.154_568_335_835_854,
            12.101397584220868,
            -4.136_907_996_676_464,
        ];
        let want_qoop2 = [
            11.411941730679915,
            -3.0871698473009705,
            -7.233_159_007_396_947,
        ];
        let so5 = GiParams {
            incident_angle: 0.2,
            tilt_angle: 0.4,
            sample_orientation: 5,
        };
        let want_qip5 = [-11.46724508272027, 3.5075696161378995, 7.558493972192581];
        let want_qoop5 = [
            -0.26025433594124314,
            11.986300369397439,
            -3.507_387_328_422_921,
        ];
        for i in 0..3 {
            let (x, y, z) = (X[i], Y[i], Z);
            assert_close(
                "qip so2",
                fiber_equation(FiberSpace::Qip, x, y, z, wl, so2),
                want_qip2[i],
            );
            assert_close(
                "qoop so2",
                fiber_equation(FiberSpace::Qoop, x, y, z, wl, so2),
                want_qoop2[i],
            );
            assert_close(
                "qip so5",
                fiber_equation(FiberSpace::Qip, x, y, z, wl, so5),
                want_qip5[i],
            );
            assert_close(
                "qoop so5",
                fiber_equation(FiberSpace::Qoop, x, y, z, wl, so5),
                want_qoop5[i],
            );
        }
    }

    /// Scattering- and exit-angle units (the `rotate_cartesian` path) vs pyFAI
    /// `eq_scattering_angle_vertical`/`eq_scattering_angle_horz`/
    /// `eq_exit_angle_vert`/`eq_exit_angle_horz`, sample_orientation 1 and 2.
    #[test]
    fn fiber_angle_units_match_pyfai_ground_truth() {
        let wl = 1e-10;
        let gis = [
            GiParams {
                incident_angle: 0.2,
                tilt_angle: 0.4,
                sample_orientation: 1,
            },
            GiParams {
                incident_angle: 0.2,
                tilt_angle: 0.4,
                sample_orientation: 2,
            },
        ];
        // [so1; so2] per unit.
        let want_scatvert = [
            [
                0.16463219168632043,
                0.026142860617732834,
                -0.1325486211844294,
            ],
            [
                0.16463219168632043,
                0.026142860617732834,
                -0.1325486211844294,
            ],
        ];
        let want_scathorz = [
            [
                0.07982998571223732,
                -0.19739555984988075,
                0.006666567903868229,
            ],
            [
                -0.07982998571223732,
                0.19739555984988075,
                -0.006666567903868229,
            ],
        ];
        let want_exitvert = [
            [0.3666767349212316, 0.12696399164015204, 0.06469411544593781],
            [0.3018073526187894, 0.2830789829169383, 0.059537579699270905],
        ];
        let want_exithorz = [
            [
                -0.07094001398790888,
                -0.27188574432718843,
                -0.02020241089441698,
            ],
            [
                -0.22291684475794094,
                0.09884823715252605,
                -0.032394650050130244,
            ],
        ];
        for (k, gi) in gis.iter().enumerate() {
            for i in 0..3 {
                let (x, y, z) = (X[i], Y[i], Z);
                assert_close(
                    "scatvert",
                    fiber_equation(FiberSpace::ScatteringAngleVert, x, y, z, wl, *gi),
                    want_scatvert[k][i],
                );
                assert_close(
                    "scathorz",
                    fiber_equation(FiberSpace::ScatteringAngleHorz, x, y, z, wl, *gi),
                    want_scathorz[k][i],
                );
                assert_close(
                    "exitvert",
                    fiber_equation(FiberSpace::ExitAngleVert, x, y, z, wl, *gi),
                    want_exitvert[k][i],
                );
                assert_close(
                    "exithorz",
                    fiber_equation(FiberSpace::ExitAngleHorz, x, y, z, wl, *gi),
                    want_exithorz[k][i],
                );
            }
        }
    }
}
