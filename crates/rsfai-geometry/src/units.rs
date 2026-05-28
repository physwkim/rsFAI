//! Radial/azimuthal unit equations, ported from `pyFAI/units.py`.
//!
//! pyFAI's `center_array` computes lab coords `(x, y, z)` then evaluates the
//! unit's formula on them and multiplies by `unit.scale`. The formula strings
//! (numexpr) are:
//!   * q   : `4.0e-9*π/λ*sin(0.5*arctan2(sqrt(x*x+y*y), z))`   (nm⁻¹ base)
//!   * 2th : `arctan2(sqrt(x*x+y*y), z)`                        (rad base)
//!   * r   : `sqrt(x*x+y*y)`                                    (m base)
//!   * chi : `arctan2(y, x)`                                    (rad base)
//!
//! NOTE on bit-exactness: the golden arrays were produced by **numexpr**. `r`
//! (only `sqrt`, IEEE-exact) is bit-exact by construction; `q`/`2th`/`chi` use
//! `sin`/`arctan2`, which *can* differ from Rust's libm at the ULP level. On the
//! golden-generation machine they measured **0 ULP** (numexpr and Rust `std`
//! libm agree), so the golden test asserts them bit-exact and prints `max_ulp`
//! to catch any future libm divergence (Tier B). See `doc/bit-exact-ladder.md`.
//!
//! Here `x`, `y`, `z` follow pyFAI's `center_array` mapping: `x = t2` (fast),
//! `y = t1` (slow/top), `z = t3` (along beam).

use std::f64::consts::PI;

/// Radial/azimuthal space (selects the formula).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Space {
    Q,
    TwoTheta,
    R,
    Chi,
}

/// A concrete unit: a [`Space`] plus the scale applied after the base formula.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Unit {
    pub space: Space,
    pub scale: f64,
}

impl Unit {
    pub const Q_NM_INV: Unit = Unit {
        space: Space::Q,
        scale: 1.0,
    };
    pub const Q_A_INV: Unit = Unit {
        space: Space::Q,
        scale: 0.1,
    };
    pub const TTH_RAD: Unit = Unit {
        space: Space::TwoTheta,
        scale: 1.0,
    };
    // 180/π in f64, matching pyFAI's printed scale exactly.
    pub const TTH_DEG: Unit = Unit {
        space: Space::TwoTheta,
        scale: 57.29577951308232,
    };
    pub const R_M: Unit = Unit {
        space: Space::R,
        scale: 1.0,
    };
    pub const R_MM: Unit = Unit {
        space: Space::R,
        scale: 1000.0,
    };
    pub const CHI_RAD: Unit = Unit {
        space: Space::Chi,
        scale: 1.0,
    };
    pub const CHI_DEG: Unit = Unit {
        space: Space::Chi,
        scale: 57.29577951308232,
    };
}

/// The base (unscaled) unit value for one pixel, matching the numexpr formula's
/// operation order. `wavelength` (m) is used only by `Q`.
#[inline]
pub fn equation(space: Space, x: f64, y: f64, z: f64, wavelength: f64) -> f64 {
    match space {
        Space::R => (x * x + y * y).sqrt(),
        Space::TwoTheta => (x * x + y * y).sqrt().atan2(z),
        // ((4e-9 * π) / λ) * sin(0.5 * atan2(sqrt(x²+y²), z))
        Space::Q => 4.0e-9 * PI / wavelength * (0.5 * (x * x + y * y).sqrt().atan2(z)).sin(),
        Space::Chi => y.atan2(x),
    }
}

/// The scaled center value for one pixel (`equation * unit.scale`), matching
/// `center_array(..., scale=True)`.
#[inline]
pub fn center_value(unit: Unit, x: f64, y: f64, z: f64, wavelength: f64) -> f64 {
    equation(unit.space, x, y, z, wavelength) * unit.scale
}

/// Apply [`center_value`] over flat lab-coordinate slices.
///
/// Per-pixel map (each element independent) -> bit-exact under parallelism.
pub fn center_array(unit: Unit, x: &[f64], y: &[f64], z: &[f64], wavelength: f64) -> Vec<f64> {
    use rayon::prelude::*;
    assert_eq!(x.len(), y.len());
    assert_eq!(x.len(), z.len());
    (0..x.len())
        .into_par_iter()
        .map(|i| center_value(unit, x[i], y[i], z[i], wavelength))
        .collect()
}

/// The **unscaled** per-pixel radial value (`equation(space)`, no `unit.scale`) —
/// pyFAI's `center_array(scale=False)`, the internal representation the binning
/// engines (`histogram`/bbox/full) actually bin on; the reported axis multiplies
/// the binned centers by `unit.scale`. This matches `delta_array` /
/// `corner_array`, which also work in unscaled units, so the bbox half-width
/// `|corner − center|` is taken in one consistent space.
///
/// Per-pixel map (each element independent) -> bit-exact under parallelism.
pub fn unscaled_center_array(
    space: Space,
    x: &[f64],
    y: &[f64],
    z: &[f64],
    wavelength: f64,
) -> Vec<f64> {
    use rayon::prelude::*;
    assert_eq!(x.len(), y.len());
    assert_eq!(x.len(), z.len());
    (0..x.len())
        .into_par_iter()
        .map(|i| equation(space, x[i], y[i], z[i], wavelength))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tth_deg_scale_is_180_over_pi() {
        assert_eq!(Unit::TTH_DEG.scale, 180.0 / PI);
    }

    #[test]
    fn r_is_pythagoras() {
        // 3-4-5 in arbitrary units; sqrt is IEEE-exact.
        assert_eq!(equation(Space::R, 4.0, 3.0, 99.0, 1.0), 5.0);
    }
}
