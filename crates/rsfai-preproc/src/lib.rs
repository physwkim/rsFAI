//! `rsfai-preproc` — per-pixel preprocessing.
//!
//! Stub for M3. Will port `preproc_value_inplace`
//! (`ext/regrid_common.pxi:149-237`) exactly: dummy/mask/NaN validity, dark
//! subtraction, `norm = normalization_factor * flat * polarization *
//! solidangle * absorption`, and error models 0/1/2/3 (Poisson is
//! `variance = max(1.0, data)`). Must match pyFAI's `floating` instantiation
//! (f32 vs f64) per call — see plan Risks.
