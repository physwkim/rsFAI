//! `rsfai-preproc` — per-pixel preprocessing, ported from
//! `pyFAI/ext/preproc.pyx` (`c4_preproc`, reached via `preproc(...,
//! split_result=4)`).
//!
//! This is the standalone preprocessing pyFAI runs before the integration
//! kernels, producing the per-pixel `(signal, variance, normalization, count)`
//! row that the histogram/CSR engines bin. It is the **Tier-A** correctness
//! gate for preprocessing: given identical inputs it must be bit-exact vs the
//! golden `preproc` array.
//!
//! ## dtype contract
//!
//! `preproc(..., dtype=numpy.float32)` (the default) casts **every** input —
//! including the f64 `solidangle` — to **f32**, and does all arithmetic in f32
//! (`floating` resolves to f32). So the caller must pass f32 slices; the f64→f32
//! cast of solid angle happens before this runs (mirroring
//! `numpy.ascontiguousarray(solidangle, dtype=float32)`).
//!
//! ## arithmetic (port of `c4_preproc`, lines 364-480)
//!
//! Per pixel: `signal = data [- dark]`; `norm = normalization_factor [* flat]
//! [* polarization] [* solidangle] [* absorption]` (in that order);
//! `variance = max(data, 1.0)` if Poisson, else the supplied variance, else 0.
//! A pixel is invalid (→ all four outputs 0) when the data is non-finite,
//! masked, equal to the dummy, or yields a non-finite / zero normalization.

pub mod preproc;

pub use preproc::{preproc4, PreprocOptions};
