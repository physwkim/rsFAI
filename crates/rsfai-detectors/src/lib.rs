//! `rsfai-detectors` — detector model and registry.
//!
//! Ports the flat module detectors the golden datasets need (`_dectris.py`:
//! Pilatus1M, Eiger4M) plus a generic `Detector(pixel1, pixel2, shape)`, from
//! `pyFAI/detectors/_common.py` and `_dectris.py`. Provides pixel-centre
//! positions (`calc_cartesian_positions`), the static module-gap mask, and (M2)
//! pixel corners. Solid-angle and polarization corrections live in
//! `rsfai-geometry` (they are `Geometry` methods in pyFAI, needing PONI/dist).

pub mod detector;

pub use detector::Detector;
