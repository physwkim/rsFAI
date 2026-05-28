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
//! - The 2D histogram engine, [`histogram2d`] (`ext/histogram.pyx`), the
//!   `("no", "histogram", "cython")` `integrate2d` path.
//! - The 2D bbox→CSR path, [`build_bbox_csr_2d`] (build) + [`csr_integrate2d`]
//!   (apply), from `ext/splitBBox_common.pyx` `calc_lut_2d` + `ext/CSR_common.pxi`
//!   — the `("bbox", "csr", "cython")` `integrate2d` path.
//! - The 2D full pixel-splitting CSR path, [`build_full_csr_2d`] (build) +
//!   [`csr_integrate2d`] (the same apply), from `ext/splitpixel_common.pyx`
//!   `calc_lut_2d` + `ext/regrid_common.pxi` `_integrate2d` + `ext/CSR_common.pxi`
//!   — the `("full", "csr", "cython")` `integrate2d` path.
//! - The direct-split bbox histogram engines, [`histogram1d_bbox`] /
//!   [`histogram2d_bbox`] (`ext/splitBBox.pyx` `histoBBox1d_engine` /
//!   `histoBBox2d_engine`), the `("bbox", "histogram", "cython")` paths — same
//!   bbox overlap fractions as the CSR build, but scattered straight into bins.
//! - The full pixel-splitting histogram engines, [`histogram1d_full`] /
//!   [`histogram2d_full`] (`ext/splitPixel.pyx` `fullSplit1D_engine` /
//!   `fullSplit2D_engine`), the `("full", "histogram", "cython")` paths — same
//!   trapezoidal overlap machinery as the full-CSR build, scattered straight
//!   into bins.
//! - The pseudo pixel-splitting histogram engine, [`histogram2d_pseudo`]
//!   (`ext/splitPixel.pyx` `pseudoSplit2D_engine`), the
//!   `("pseudo", "histogram", "cython")` 2D path (2D only — there is no 1D pseudo
//!   engine). Each pixel is approximated by an axis-aligned rectangle carrying the
//!   pixel's true `area4` quadrilateral area at the aspect ratio of its corner
//!   bounding box, then scattered with the same separable bbox split.
//! - The CSC paths, [`build_bbox_csc_1d`] / [`build_bbox_csc_2d`] /
//!   [`build_full_csc_1d`] / [`build_full_csc_2d`] (build) + [`csc_integrate1d`] /
//!   [`csc_integrate2d`] (apply), from `ext/splitBBoxCSC.pyx` /
//!   `ext/splitPixelFullCSC.pyx` + `ext/CSC_common.pxi` — the
//!   `("no"|"bbox"|"full", "csc", "cython")` 1D/2D paths. The build transposes the
//!   CSR LUT (scipy `tocsc`); the apply scatters pixel-major.
//! - The LUT paths, [`build_bbox_lut_1d`] / [`build_bbox_lut_2d`] /
//!   [`build_full_lut_1d`] / [`build_full_lut_2d`] (build) + [`lut_integrate1d`] /
//!   [`lut_integrate2d`] (apply), from `ext/splitBBoxLUT.pyx` /
//!   `ext/splitPixelFullLUT.pyx` + `ext/LUT_common.pxi` — the
//!   `("no"|"bbox"|"full", "lut", "cython")` 1D/2D paths. The build densifies the
//!   CSR LUT (`to_lut`); the apply gathers per bin, skipping zero-padding.
//!
//! Per-pixel maps and CSR apply accumulate bit-exactly. The no-split histogram
//! scatter is rayon-parallel and validated at relative error `<= 1e-6` because
//! its f64 add order across pixels is non-deterministic. The direct-split
//! histogram scatters (bbox, full, and pseudo) run **serially in pixel-index
//! order**: their fractional split coefficients make per-bin f64 sums
//! order-dependent, so serial accumulation reproduces the single-threaded pyFAI
//! golden bit-for-bit.
//! Golden generation is single-threaded.
//!
//! The `"azimuthal"` error model estimates each bin's variance from the spread
//! of the pixel intensities `signal/norm` falling in it, via a weighted Welford
//! update (the crate-internal `azimuthal` module) shared by every engine.
//! Because that update is
//! order-dependent and not additively mergeable, the no-split histogram runs it
//! **serially** (the parallel path is non-azimuthal only). The 2D no-split and
//! direct-split histograms have no Welford branch (pyFAI's `update_2d_accumulator`
//! does not), so they propagate the zero per-pixel variance and report std/sem 0.

mod azimuthal;
pub mod csc;
pub mod csr;
pub mod histogram;
pub mod lut;
pub mod split_histogram;

pub use csc::{
    build_bbox_csc_1d, build_bbox_csc_2d, build_full_csc_1d, build_full_csc_2d, csc_integrate1d,
    csc_integrate2d, Csc,
};
pub use csr::{
    build_bbox_csr_1d, build_bbox_csr_2d, build_full_csr_1d, build_full_csr_2d, csr_integrate1d,
    csr_integrate2d, Bbox2dBounds, Csr, CsrIntegrate1d,
};
pub use histogram::{
    histogram1d, histogram2d, histogram_preproc, Hist2dOptions, Integrate1d, Integrate2d,
};
pub use lut::{
    build_bbox_lut_1d, build_bbox_lut_2d, build_full_lut_1d, build_full_lut_2d, lut_integrate1d,
    lut_integrate2d, Lut,
};
pub use split_histogram::{
    histogram1d_bbox, histogram1d_full, histogram2d_bbox, histogram2d_full, histogram2d_pseudo,
};
