//! Crystallographic cell → d-spacing computation, ported from
//! `pyFAI/crystallography/cell.py` (`Cell`).
//!
//! The algebraic core is `Cell::d` (interplanar distance for a Miller triplet)
//! and `Cell::calculate_dspacing` (enumerate all `hkl` down to `dmin`, apply the
//! centering selection rules, group equivalent reflections by their rounded
//! d-spacing). `build_calibrant_config` then produces the same `Reflection`
//! list the shipped `.D` files were generated from.
//!
//! The cubic/tetragonal/orthorhombic branch of `d` is pure `+ - * /` plus one
//! `sqrt`, hence bit-exact. The general (triclinic …) branch caches the metric
//! tensor (`S11 … S13`) computed once from `sin`/`cos` (transcendental); only
//! those scalars carry a libm boundary, after which `d = sqrt(1/invd2)` is
//! algebraic.

use std::f64::consts::PI;

use crate::config::{CalibrantConfig, Miller, Reflection};
use crate::space_groups::{self, Centering};

/// One of the seven crystal systems (`Cell.lattices`). Only the system name is
/// used to choose the `d`-formula branch; unknown names fall back to triclinic
/// (mirrors `Cell.__init__`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Lattice {
    Cubic,
    Tetragonal,
    Hexagonal,
    Rhombohedral,
    Orthorhombic,
    Monoclinic,
    Triclinic,
}

impl Lattice {
    /// Whether `d` uses the orthogonal short-cut (cubic/tetragonal/orthorhombic).
    fn is_orthogonal(self) -> bool {
        matches!(
            self,
            Lattice::Cubic | Lattice::Tetragonal | Lattice::Orthorhombic
        )
    }
}

/// A crystallographic unit cell, `pyFAI.crystallography.cell.Cell`.
///
/// Lengths `a`/`b`/`c` in Angstrom; angles `alpha`/`beta`/`gamma` in degrees.
#[derive(Debug, Clone)]
pub struct Cell {
    pub a: f64,
    pub b: f64,
    pub c: f64,
    pub alpha: f64,
    pub beta: f64,
    pub gamma: f64,
    pub lattice: Lattice,
    pub centering: Centering,
    /// Extra selection rules layered on top of the centering rule (e.g. the
    /// diamond-FCC extinction). `Cell.selection_rules` in pyFAI.
    extra_rules: Vec<fn(i64, i64, i64) -> bool>,
    // Cached metric-tensor scalars (general branch of `d`), `Cell.S*`.
    s_cache: Option<SMatrix>,
    volume_cache: Option<f64>,
}

#[derive(Debug, Clone, Copy)]
struct SMatrix {
    s11: f64,
    s22: f64,
    s33: f64,
    s12: f64,
    s23: f64,
    s13: f64,
}

impl Cell {
    /// Full constructor, `Cell.__init__`.
    pub fn new(
        a: f64,
        b: f64,
        c: f64,
        alpha: f64,
        beta: f64,
        gamma: f64,
        lattice: Lattice,
        centering: Centering,
    ) -> Cell {
        Cell {
            a,
            b,
            c,
            alpha,
            beta,
            gamma,
            lattice,
            centering,
            extra_rules: Vec::new(),
            s_cache: None,
            volume_cache: None,
        }
    }

    /// `Cell.cubic`.
    pub fn cubic(a: f64, centering: Centering) -> Cell {
        Cell::new(a, a, a, 90.0, 90.0, 90.0, Lattice::Cubic, centering)
    }

    /// `Cell.tetragonal`.
    pub fn tetragonal(a: f64, c: f64, centering: Centering) -> Cell {
        Cell::new(a, a, c, 90.0, 90.0, 90.0, Lattice::Tetragonal, centering)
    }

    /// `Cell.orthorhombic`.
    pub fn orthorhombic(a: f64, b: f64, c: f64, centering: Centering) -> Cell {
        Cell::new(a, b, c, 90.0, 90.0, 90.0, Lattice::Orthorhombic, centering)
    }

    /// `Cell.hexagonal` (gamma = 120°).
    pub fn hexagonal(a: f64, c: f64, centering: Centering) -> Cell {
        Cell::new(a, a, c, 90.0, 90.0, 120.0, Lattice::Hexagonal, centering)
    }

    /// `Cell.rhombohedral`.
    pub fn rhombohedral(a: f64, alpha: f64, centering: Centering) -> Cell {
        Cell::new(
            a,
            a,
            a,
            alpha,
            alpha,
            alpha,
            Lattice::Rhombohedral,
            centering,
        )
    }

    /// `Cell.diamond`: FCC plus the diamond-glide extinction. Used by Si/Ge.
    ///
    /// pyFAI adds, on top of the F-centering rule, the rule that forbids the
    /// all-even reflections with `(h + k + l) % 4 != 0`.
    pub fn diamond(a: f64) -> Cell {
        let mut cell = Cell::cubic(a, Centering::F);
        cell.extra_rules.push(|h, k, l| {
            !(h.rem_euclid(2) == 0
                && k.rem_euclid(2) == 0
                && l.rem_euclid(2) == 0
                && (h + k + l).rem_euclid(4) != 0)
        });
        cell
    }

    /// Cell volume, `Cell.volume` (cached).
    pub fn volume(&mut self) -> f64 {
        if let Some(v) = self.volume_cache {
            return v;
        }
        let mut v = self.a * self.b * self.c;
        if !self.lattice.is_orthogonal() {
            let deg2rad = PI / 180.0;
            let cosa = (self.alpha * deg2rad).cos();
            let cosb = (self.beta * deg2rad).cos();
            let cosg = (self.gamma * deg2rad).cos();
            v *= (1.0 - cosa * cosa - cosb * cosb - cosg * cosg + 2.0 * cosa * cosb * cosg).sqrt();
        }
        self.volume_cache = Some(v);
        v
    }

    /// All registered selection rules: `default` + the centering rule (if any)
    /// + any extra rules. `Cell.selection_rules`.
    fn rules(&self) -> Vec<fn(i64, i64, i64) -> bool> {
        let mut rules: Vec<fn(i64, i64, i64) -> bool> = vec![space_groups::default];
        if let Some(r) = self.centering.rule() {
            rules.push(r);
        }
        rules.extend_from_slice(&self.extra_rules);
        rules
    }

    /// Interplanar distance (Angstrom) for a Miller triplet, `Cell.d`.
    pub fn d(&mut self, hkl: Miller) -> f64 {
        let Miller { h, k, l } = hkl;
        let (h, k, l) = (h as f64, k as f64, l as f64);
        if self.lattice.is_orthogonal() {
            let invd2 = (h / self.a).powi(2) + (k / self.b).powi(2) + (l / self.c).powi(2);
            return (1.0 / invd2).sqrt();
        }
        if self.s_cache.is_none() {
            let deg2rad = PI / 180.0;
            let alpha = self.alpha * deg2rad;
            let cosa = alpha.cos();
            let sina = alpha.sin();
            let beta = self.beta * deg2rad;
            let cosb = beta.cos();
            let sinb = beta.sin();
            let gamma = self.gamma * deg2rad;
            let cosg = gamma.cos();
            let sing = gamma.sin();
            let s = SMatrix {
                s11: (self.b * self.c * sina).powi(2),
                s22: (self.a * self.c * sinb).powi(2),
                s33: (self.a * self.b * sing).powi(2),
                s12: self.a * self.b * self.c * self.c * (cosa * cosb - cosg),
                s23: self.a * self.a * self.b * self.c * (cosb * cosg - cosa),
                s13: self.a * self.b * self.b * self.c * (cosg * cosa - cosb),
            };
            self.s_cache = Some(s);
        }
        let s = self.s_cache.unwrap();
        let mut invd2 = s.s11 * h * h
            + s.s22 * k * k
            + s.s33 * l * l
            + 2.0 * s.s12 * h * k
            + 2.0 * s.s23 * k * l
            + 2.0 * s.s13 * h * l;
        let vol = self.volume();
        invd2 /= vol * vol;
        (1.0 / invd2).sqrt()
    }

    /// Enumerate all d-spacings down to `dmin`, grouped by their 8-digit-rounded
    /// value, `Cell.calculate_dspacing`.
    ///
    /// Returns groups in insertion order (Python `dict` preserves insertion
    /// order); each group is a `(rounded_d, miller_list)` pair where the list is
    /// sorted by `(l, k, h)` ascending — pyFAI's `key=x[-1::-1]`.
    pub fn calculate_dspacing(&mut self, dmin: f64) -> Vec<(f64, Vec<Miller>)> {
        let hmax = (self.a / dmin).ceil() as i64;
        let kmax = (self.b / dmin).ceil() as i64;
        let lmax = (self.c / dmin).ceil() as i64;
        let rules = self.rules();

        // (rounded_d, millers) in insertion order; index lookup keyed on the
        // f64 bit pattern of the rounded value (Python dict float-key equality).
        let mut groups: Vec<(f64, Vec<Miller>)> = Vec::new();

        for hh in -hmax..=hmax {
            for kk in -kmax..=kmax {
                for ll in -lmax..=lmax {
                    let mut valid = true;
                    for rule in &rules {
                        valid = rule(hh, kk, ll);
                        if !valid {
                            break;
                        }
                    }
                    if !valid {
                        continue;
                    }
                    let miller = Miller::new(hh, kk, ll);
                    let d = self.d(miller);
                    if d < dmin {
                        continue;
                    }
                    let d = round8(d);
                    match groups
                        .iter_mut()
                        .find(|(key, _)| key.to_bits() == d.to_bits())
                    {
                        Some((_, lst)) => lst.push(miller),
                        None => groups.push((d, vec![miller])),
                    }
                }
            }
        }

        for (_, lst) in groups.iter_mut() {
            // pyFAI: lst.sort(key=lambda x: x[-1::-1]) — reversed tuple => (l, k, h).
            lst.sort_by_key(|m| (m.l, m.k, m.h));
        }
        groups
    }

    /// Build a `CalibrantConfig` from the cell, `Cell.build_calibrant_config`.
    ///
    /// d-spacings are sorted descending; for each group the *last* Miller index
    /// (`reflection[-1]`) and the group size (multiplicity) are recorded — the
    /// exact convention `Cell.save` used to write the shipped `.D` files.
    pub fn build_calibrant_config(&mut self, dmin: f64) -> CalibrantConfig {
        let mut config = CalibrantConfig {
            cell: String::new(),
            space_group: self.centering.letter().to_string(),
            ..CalibrantConfig::default()
        };
        let groups = self.calculate_dspacing(dmin);
        // Sort the keys descending (pyFAI: `dspacing.sort(reverse=True)`), then
        // index back into the group map.
        let mut keyed: Vec<(f64, Vec<Miller>)> = groups;
        keyed.sort_by(|a, b| b.0.total_cmp(&a.0));
        for (d, millers) in keyed {
            let last = *millers.last().expect("non-empty group");
            config.reflections.push(Reflection {
                dspacing: d,
                intensity: None,
                hkl: Some(last),
                multiplicity: Some(millers.len() as u32),
            });
        }
        config
    }
}

/// `numpy.round(x, 8)` = `rint(x * 1e8) / 1e8` with round-half-to-even.
fn round8(x: f64) -> f64 {
    (x * 1e8).round_ties_even() / 1e8
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round8_banker() {
        // numpy: np.round(0.123456785, 8) == 0.12345678 (ties to even).
        assert_eq!(round8(0.123456785).to_bits(), 0.12345678_f64.to_bits());
    }

    #[test]
    fn cubic_volume() {
        assert_eq!(Cell::cubic(1.0, Centering::P).volume(), 1.0);
        assert_eq!(
            Cell::orthorhombic(1.0, 2.0, 3.0, Centering::P).volume(),
            6.0
        );
    }

    #[test]
    fn cubic_d_111() {
        // d(111) for cubic a=1: pyFAI computes `sqrt(1.0 / invd2)` with
        // invd2 = 3.0, i.e. `sqrt(1/3)` — NOT `1/sqrt(3)` (they differ by 1
        // ULP). The expected bits `0x1.279a74590331c` are pyFAI's output.
        let mut c = Cell::cubic(1.0, Centering::P);
        let d = c.d(Miller::new(1, 1, 1));
        assert_eq!(d.to_bits(), (1.0_f64 / 3.0_f64).sqrt().to_bits());
        assert_eq!(d.to_bits(), 0x3fe279a74590331c);
    }

    #[test]
    fn diamond_extinction() {
        // Si/diamond: (2 0 0) is F-allowed but forbidden by the glide rule
        // ((h+k+l)%4 == 2 != 0), so it must NOT appear; (1 1 1) and (4 0 0) do.
        let mut c = Cell::diamond(5.4312);
        let groups = c.calculate_dspacing(1.0);
        let has = |h, k, l| {
            groups
                .iter()
                .any(|(_, ms)| ms.contains(&Miller::new(h, k, l)))
        };
        assert!(has(1, 1, 1));
        assert!(!has(2, 0, 0));
        assert!(has(4, 0, 0));
    }
}
