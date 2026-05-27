//! `rsfai-integrate` — integration engines.
//!
//! **M4 (done):** the pure-Cython 1D histogram engine, [`histogram1d`] /
//! [`histogram_preproc`] (`ext/histogram.pyx`), the `("no", "histogram",
//! "cython")` path. Default accumulation is serial for bit-exactness.
//!
//! Still stubbed for M5–M6: bbox splitting (`ext/splitBBox*.pyx`), the sparse
//! builder (`ext/sparse_builder.pyx`) with CSR build+apply (`ext/CSR_common.pxi`),
//! and full pixel splitting (`ext/splitPixel*.pyx`, `splitpixel_common.pyx`).
//! rayon is opt-in behind a feature flag and is never the bit-exact gate.

pub mod histogram;

pub use histogram::{histogram1d, histogram_preproc, Integrate1d};
