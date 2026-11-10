//! Calibration: control points + geometry refinement, ported from pyFAI's
//! `control_points.py` and `geometryRefinement.py`.
//!
//! Two parity tiers, kept structurally separate (see `doc/bit-exact-ladder.md`):
//!   * **Bit-exact (identical-input)** — the data container ([`ControlPoints`])
//!     and the refinement cost ([`GeometryRefinement::residu1`] /
//!     [`GeometryRefinement::chi2`] at a fixed parameter vector). Given the same
//!     params, control points and calibrant, the residual vector and chi2 match
//!     pyFAI bit-for-bit; the only ULP-budgeted part is the geometry
//!     `atan2`/`sin`/`cos` inherited from the already-validated
//!     `crate::geometry::calc_pos_zyx`.
//!   * **Tolerance (converged params)** — [`GeometryRefinement::refine`] runs an
//!     `argmin` Nelder-Mead minimization (isolated in [`optimizer`]) whose
//!     trajectory differs from scipy's SLSQP by construction. The converged
//!     parameters are validated at a recorded relative tolerance, with the
//!     converged cost asserted `<=` pyFAI's converged cost.
//!
//! [`ring_search`] is Dioptas's automatic refinement on top of both: pick
//! the peaks around each ring where the geometry puts it, refine, repeat.

pub mod control_points;
mod optimizer;
pub mod refinement;
pub mod ring_search;

pub use control_points::{ControlPoints, PointGroup};
pub use refinement::{GeometryParams, GeometryRefinement, Param};
pub use ring_search::{
    auto_refine, ring_band, tth_array, AutoRefine, Band, Calibration, RingSearch,
};
