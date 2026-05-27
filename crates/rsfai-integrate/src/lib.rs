//! `rsfai-integrate` — integration engines.
//!
//! - The pure-Cython 1D histogram engine, [`histogram1d`] /
//!   [`histogram_preproc`] (`ext/histogram.pyx`), the `("no", "histogram",
//!   "cython")` path.
//! - The bbox→CSR 1D path, [`build_bbox_csr_1d`] (build) + [`csr_integrate1d`]
//!   (apply), from `ext/splitBBox_common.pyx` + `ext/CSR_common.pxi` — the
//!   `("bbox", "csr", "cython")` path.
//! - The full pixel-splitting CSR 1D path, [`build_full_csr_1d`] (build) +
//!   [`csr_integrate1d`] (the same apply), from `ext/splitpixel_common.pyx` +
//!   `ext/regrid_common.pxi` + `ext/CSR_common.pxi` — the `("full", "csr",
//!   "cython")` path.
//!
//! Default accumulation is serial for bit-exactness; rayon is opt-in behind a
//! feature flag and is never the bit-exact gate. 2D integration is not yet
//! ported.

pub mod csr;
pub mod histogram;

pub use csr::{build_bbox_csr_1d, build_full_csr_1d, csr_integrate1d, Csr, CsrIntegrate1d};
pub use histogram::{histogram1d, histogram_preproc, Integrate1d};
