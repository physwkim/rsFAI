//! `rsfai-distortion` — spatial-distortion correction for 2D detectors, ported
//! from `pyFAI/distortion.py` + `pyFAI/ext/_distortion.pyx` and the FITPACK
//! B-spline evaluator `pyFAI/spline.py` + `pyFAI/ext/_bispev.pyx`.
//!
//! - [`bispev`] — bivariate B-spline surface evaluation (de Boor–Cox recurrence
//!   with a Kahan-summed tensor product), the Cython `cy_bispev`. Given identical
//!   knots/coefficients/evaluation points it reproduces pyFAI's `bisplev` to the
//!   bit. All arithmetic is f32, exactly as the Cython code.
//! - [`spline`] — the FITPACK `.spline` ASCII parser (`Spline.read`) plus
//!   `spline2array` (the displacement maps `bisplev` produces over the detector
//!   grid). Knots and coefficients are f32.
//! - [`distortion`] — the distortion look-up table (`_distortion.calc_pos` +
//!   `calc_sparse` CSR build over the `Quad`-clipped pixel polygons) and the
//!   apply step (`_distortion.correct`, CSR double-precision accumulation).
//!
//! The dtype contract (positions f64, image/coefficients f32, accumulators f64,
//! mask i8, indices i32) is shared with the rest of rsFAI via `rsfai-core`.

pub mod bispev;
pub mod distortion;
pub mod error;
pub mod spline;

pub use bispev::{bisplev, Tck};
pub use distortion::{calc_pos, calc_sparse, correct, CalcPos, Csr};
pub use error::{DistortionError, Result};
pub use spline::Spline;
