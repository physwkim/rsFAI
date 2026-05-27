//! `rsfai-integrate` — integration engines.
//!
//! Stub for M4–M6. Will port the accumulators
//! (`update_1d_accumulator`/`update_2d_accumulator`, `ext/regrid_common.pxi`),
//! the histogram engine (`ext/histogram.pyx`), bbox splitting
//! (`ext/splitBBox*.pyx`), the sparse builder (`ext/sparse_builder.pyx`) with
//! CSR build+apply (`ext/CSR_common.pxi`), and full pixel splitting
//! (`ext/splitPixel*.pyx`, `splitpixel_common.pyx`). Default accumulation is
//! serial for bit-exactness; rayon is opt-in behind a feature flag.
