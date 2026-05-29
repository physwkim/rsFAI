//! The `Calibrant` class, ported from `pyFAI/crystallography/calibrant.py`.
//!
//! A calibrant is a named reference compound with known d-spacings (Angstrom).
//! Given a wavelength (meters) it computes the Bragg-law scattering angles 2θ
//! for every visible ring and exposes them in the usual radial units.
//!
//! Core formulas (verbatim from pyFAI):
//!   * `_calc_2th`: `tth = 2 * asin(5e9 * wavelength / d)`. The `5e9` is
//!     `1 / (2 * 1e-10)`: it converts the wavelength from meters to half-d in
//!     Angstrom units (`λ[m] * 5e9 == λ[Å] / 2`). When the argument exceeds 1
//!     (d too small to diffract at this wavelength) `asin` is undefined; pyFAI
//!     splits the d-spacing list there — the leading entries are *visible*
//!     rings, the tail goes to `out_dspacing` and is dropped from the 2θ list.
//!   * `get_peaks`: `2th*` units return `tth * scale`; `q*` units return
//!     `(20π / d) * scale`. (`20π/d` with d in Å gives q in nm⁻¹.)
//!   * energy ⇄ wavelength via `CONST_hc` (keV·Å).
//!
//! The `2 * asin(...)` is the one transcendental on the path: it is Tier-B,
//! ULP-budgeted against pyFAI's `math.asin`. The d-spacings, the `5e9 * λ / d`
//! argument, and the `get_peaks` `20π/d` and `* scale` arithmetic are pure
//! `* /` and bit-exact.

use crate::config::CalibrantConfig;

/// `pyFAI.units.CONST_hc` = `c * h / e * 1e7` (keV·Å), the photon energy ⇄
/// wavelength constant. Recorded as the exact `f64` pyFAI's scipy.constants
/// produce on this platform (`12.398419843320026`); see `gen_golden_calibrant.py`
/// which dumps it into the manifest so a constants change fails loudly.
pub const CONST_HC: f64 = 12.398419843320026;

/// Radial units `get_peaks` supports (`pyFAI.units`), with their `scale`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PeakUnit {
    /// `2th_deg`, scale `180/π`.
    TwoThetaDeg,
    /// `2th_rad`, scale `1.0`.
    TwoThetaRad,
    /// `q_nm^-1`, scale `1.0`.
    QNm,
    /// `q_A^-1`, scale `0.1`.
    QA,
}

impl PeakUnit {
    /// The unit's `scale` factor.
    pub fn scale(self) -> f64 {
        match self {
            PeakUnit::TwoThetaDeg => 180.0 / std::f64::consts::PI,
            PeakUnit::TwoThetaRad => 1.0,
            PeakUnit::QNm => 1.0,
            PeakUnit::QA => 0.1,
        }
    }

    /// Whether the unit is a 2θ (vs q) unit — selects the `get_peaks` branch.
    fn is_two_theta(self) -> bool {
        matches!(self, PeakUnit::TwoThetaDeg | PeakUnit::TwoThetaRad)
    }
}

/// A reference calibrant: d-spacings (Angstrom) plus an optional wavelength
/// (meters). `pyFAI.crystallography.calibrant.Calibrant`.
#[derive(Debug, Clone)]
pub struct Calibrant {
    /// Visible d-spacings (those that diffract at the current wavelength),
    /// Angstrom, in file/descending order. `Calibrant._dspacing`.
    dspacing: Vec<f64>,
    /// d-spacings too small to diffract at the current wavelength.
    /// `Calibrant._out_dspacing`.
    out_dspacing: Vec<f64>,
    /// Wavelength in meters, if set. `Calibrant._wavelength`.
    wavelength: Option<f64>,
    /// Cached 2θ (radians) for the visible rings. `Calibrant._2th`.
    two_theta: Vec<f64>,
    /// Optional parsed `.D` config (carried for fidelity; not required for 2θ).
    pub config: Option<CalibrantConfig>,
}

impl Calibrant {
    /// A calibrant from an explicit d-spacing list (Angstrom). `Calibrant(dspacing=...)`.
    pub fn from_dspacing(dspacing: Vec<f64>) -> Calibrant {
        let mut c = Calibrant {
            dspacing,
            out_dspacing: Vec::new(),
            wavelength: None,
            two_theta: Vec::new(),
            config: None,
        };
        // pyFAI calls _calc_2th in __init__ only if both dspacing and
        // wavelength are set; wavelength is None here, so no compute yet.
        c.config = None;
        c
    }

    /// Parse a `.D` file's text into a calibrant. Mirrors
    /// `Calibrant._load_file` → `CalibrantConfig.from_dspacing` →
    /// `[ref.dspacing for ref in reflections]`.
    pub fn from_dspacing_file_str(text: &str) -> Calibrant {
        let config = CalibrantConfig::from_dspacing_str(text);
        let dspacing = config.dspacing();
        Calibrant {
            dspacing,
            out_dspacing: Vec::new(),
            wavelength: None,
            two_theta: Vec::new(),
            config: Some(config),
        }
    }

    /// Build directly from a parsed config (`Calibrant(config=...)`), as used
    /// by `Cell::to_calibrant`.
    pub fn from_config(config: CalibrantConfig) -> Calibrant {
        let dspacing = config.dspacing();
        Calibrant {
            dspacing,
            out_dspacing: Vec::new(),
            wavelength: None,
            two_theta: Vec::new(),
            config: Some(config),
        }
    }

    /// The current wavelength (meters), if set.
    pub fn wavelength(&self) -> Option<f64> {
        self.wavelength
    }

    /// Set the wavelength and recompute 2θ, `Calibrant.setWavelength_change2th`
    /// (the unconditional-update variant; the `wavelength` property's
    /// "forbidden to change" guard is for live GUIs and not modeled here).
    pub fn set_wavelength(&mut self, wavelength: f64) {
        self.wavelength = Some(wavelength);
        self.calc_2th();
    }

    /// Photon energy in keV, `Calibrant.energy` getter: `1e-10 * CONST_hc / λ`.
    pub fn energy(&self) -> Option<f64> {
        self.wavelength.map(|w| 1e-10 * CONST_HC / w)
    }

    /// Set the energy in keV, `Calibrant.energy` setter:
    /// `λ = 1e-10 * CONST_hc / energy`.
    pub fn set_energy(&mut self, energy_kev: f64) {
        let wavelength = 1e-10 * CONST_HC / energy_kev;
        self.set_wavelength(wavelength);
    }

    /// The visible d-spacings (Angstrom). `Calibrant.dspacing`.
    pub fn dspacing(&self) -> &[f64] {
        &self.dspacing
    }

    /// The d-spacings dropped as non-diffracting at the current wavelength.
    pub fn out_dspacing(&self) -> &[f64] {
        &self.out_dspacing
    }

    /// Count of all registered d-spacings (visible + out), `count_registered_dspacing`.
    pub fn count_registered_dspacing(&self) -> usize {
        self.dspacing.len() + self.out_dspacing.len()
    }

    /// Compute 2θ (radians) for all rings, `Calibrant._calc_2th`.
    ///
    /// Iterates `dspacing[:] + out_dspacing`; the first `asin` argument > 1
    /// (`5e9 * λ / d > 1`) terminates the loop, splitting visible from out
    /// d-spacings at that index. (pyFAI relies on `math.asin` raising
    /// `ValueError`; here we test the argument explicitly, which is the exact
    /// same boundary — `asin` is real iff `|arg| <= 1`.)
    fn calc_2th(&mut self) {
        let wavelength = match self.wavelength {
            Some(w) => w,
            None => return,
        };
        // Explicit copy: `self._dspacing[:] + self._out_dspacing`.
        let all: Vec<f64> = self
            .dspacing
            .iter()
            .copied()
            .chain(self.out_dspacing.iter().copied())
            .collect();
        let mut tths = Vec::with_capacity(all.len());
        let mut split: Option<usize> = None;
        for (i, &ds) in all.iter().enumerate() {
            let arg = 5.0e9 * wavelength / ds;
            if !(-1.0..=1.0).contains(&arg) {
                // asin would raise ValueError in pyFAI.
                split = Some(i);
                break;
            }
            tths.push(2.0 * arg.asin());
        }
        match split {
            Some(size) => {
                self.dspacing = all[..size].to_vec();
                self.out_dspacing = all[size..].to_vec();
            }
            None => {
                self.dspacing = all;
                self.out_dspacing = Vec::new();
            }
        }
        self.two_theta = tths;
    }

    /// The 2θ positions (radians) for all visible rings, `Calibrant.get_2th`.
    pub fn get_2th(&self) -> &[f64] {
        &self.two_theta
    }

    /// Peak positions in the requested radial unit, `Calibrant.get_peaks`.
    ///
    /// For 2θ units: `tth * scale` (one value per visible 2θ).
    /// For q units: `(20π / d) * scale` over the first `len(2θ)` d-spacings.
    pub fn get_peaks(&self, unit: PeakUnit) -> Vec<f64> {
        let scale = unit.scale();
        let size = self.two_theta.len();
        if unit.is_two_theta() {
            self.two_theta.iter().map(|&t| t * scale).collect()
        } else {
            self.dspacing[..size]
                .iter()
                .map(|&d| (20.0 * std::f64::consts::PI / d) * scale)
                .collect()
        }
    }

    /// pyFAI `Calibrant.fake_xrpdp`: synthesize a 1D powder-diffraction pattern
    /// from this calibrant's visible rings as a sum of Gaussian peaks, returning
    /// `(tth_deg, intensity)` over `nbpt` points spanning `tth_range_deg`.
    ///
    /// `resolution_deg` is the peak FWHM in degrees (pyFAI's `Constant`
    /// resolution; `sigma² = (FWHM/scale)² / (8·ln2)`); the most intense peak is
    /// scaled to `imax`, and `background` is added as a flat offset. Requires the
    /// wavelength to be set (uses [`get_2th`](Self::get_2th)). Peaks farther than
    /// `4σ` outside the range, with non-positive intensity, are dropped.
    ///
    /// Tier-B: the Gaussian `exp`/`sqrt` are transcendental (pyFAI evaluates the
    /// peak sum through numexpr), so this matches pyFAI within a recorded ULP
    /// budget, not bitwise. The `linspace` axis, peak positions, intensities and
    /// the mask selection are exact.
    pub fn fake_xrpdp(
        &self,
        nbpt: usize,
        tth_range_deg: (f64, f64),
        background: f64,
        imax: f64,
        resolution_deg: f64,
    ) -> (Vec<f64>, Vec<f64>) {
        use std::f64::consts::{LN_2, PI};
        const EPSILON: f64 = 1.0e-6;
        let scale = 180.0 / PI; // pyFAI TTH_DEG.scale

        let lo = tth_range_deg.0.min(tth_range_deg.1);
        let hi = tth_range_deg.0.max(tth_range_deg.1);
        // numpy.linspace(lo, hi, nbpt): y[i] = i*step + lo, last element = hi.
        let tth_user: Vec<f64> = match nbpt {
            0 => Vec::new(),
            1 => vec![lo],
            _ => {
                let step = (hi - lo) / ((nbpt - 1) as f64);
                let mut v: Vec<f64> = (0..nbpt).map(|i| (i as f64) * step + lo).collect();
                v[nbpt - 1] = hi;
                v
            }
        };
        let tth_rad: Vec<f64> = tth_user.iter().map(|&t| t / scale).collect();

        let tth_peak = self.get_2th(); // radians
        let c2 = (resolution_deg / scale).powi(2);
        let dtth2 = c2 / (8.0 * LN_2); // constant per peak (Constant resolution)

        // intensities: 1.0 by default, overridden by matched config reflections.
        let mut intensities = vec![1.0_f64; tth_peak.len()];
        if let Some(cfg) = &self.config {
            for (i, refl) in cfg.reflections.iter().enumerate() {
                if i >= self.dspacing.len() {
                    break;
                }
                if (refl.dspacing - self.dspacing[i]).abs() > EPSILON {
                    continue; // dspacing/config mismatch: keep the default 1.0
                }
                intensities[i] = refl.intensity.unwrap_or(1.0);
            }
        }

        let sigma = dtth2.sqrt();
        let tth_min = lo / scale;
        let tth_max = hi / scale;
        let kept: Vec<usize> = (0..tth_peak.len())
            .filter(|&i| {
                intensities[i] > 0.0
                    && dtth2 > 0.0
                    && tth_peak[i] >= tth_min - 4.0 * sigma
                    && tth_peak[i] <= tth_max + 4.0 * sigma
            })
            .collect();

        let mut signal = vec![background; nbpt];
        if kept.is_empty() {
            return (tth_user, signal); // no peak in range: flat background
        }

        // signals[peak][pt] = I_peak * exp(-(tth_rad-tth_peak)²/(2σ²)) / sqrt(2πσ²)
        let norm = (2.0 * PI * dtth2).sqrt();
        let mut signals: Vec<Vec<f64>> = Vec::with_capacity(kept.len());
        let mut gmax = f64::NEG_INFINITY;
        for &p in &kept {
            let tp = tth_peak[p];
            let ip = intensities[p];
            let row: Vec<f64> = tth_rad
                .iter()
                .map(|&tr| {
                    let g = (-((tr - tp).powi(2)) / (2.0 * dtth2)).exp();
                    ip * g / norm
                })
                .collect();
            for &v in &row {
                if v > gmax {
                    gmax = v;
                }
            }
            signals.push(row);
        }
        // Normalize the most intense peak to 1, sum over peaks, scale by imax,
        // add background — pyFAI's `Imax * (signals/signals.max()).sum(axis=0)`.
        for (pt, s) in signal.iter_mut().enumerate() {
            let mut acc = 0.0;
            for row in &signals {
                acc += row[pt] / gmax;
            }
            *s = imax * acc + background;
        }
        (tth_user, signal)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn asin_split_drops_small_dspacings() {
        // Two d-spacings: 5 Å diffracts at λ=1e-10 m (arg=0.1), 0.4 Å does not
        // (arg=1.25 > 1) so it is dropped to out_dspacing.
        let mut c = Calibrant::from_dspacing(vec![5.0, 0.4]);
        c.set_wavelength(1e-10);
        assert_eq!(c.get_2th().len(), 1);
        assert_eq!(c.dspacing().len(), 1);
        assert_eq!(c.out_dspacing(), &[0.4]);
    }

    #[test]
    fn calc_2th_value() {
        let mut c = Calibrant::from_dspacing(vec![5.0]);
        c.set_wavelength(1e-10);
        let expected = 2.0 * (5.0e9 * 1e-10_f64 / 5.0).asin();
        assert_eq!(c.get_2th()[0].to_bits(), expected.to_bits());
    }

    #[test]
    fn get_peaks_two_theta_deg_is_rad_scaled() {
        let mut c = Calibrant::from_dspacing(vec![5.0, 2.5]);
        c.set_wavelength(1e-10);
        let deg = c.get_peaks(PeakUnit::TwoThetaDeg);
        let rad = c.get_2th();
        for (d, r) in deg.iter().zip(rad) {
            assert_eq!(d.to_bits(), (r * (180.0 / std::f64::consts::PI)).to_bits());
        }
    }

    #[test]
    fn energy_roundtrip() {
        let mut c = Calibrant::from_dspacing(vec![5.0]);
        c.set_energy(20.0);
        let e = c.energy().unwrap();
        assert!((e - 20.0).abs() < 1e-12);
    }
}
