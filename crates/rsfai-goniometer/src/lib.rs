//! `rsfai-goniometer` — goniometer geometry model + refinement + multi-geometry
//! fiber assembly, ported from `pyFAI/goniometer.py` and the
//! `MultiGeometryFiber` class in `pyFAI/multi_geometry.py`.
//!
//! The pieces, in dependency order:
//!
//!   * [`GeometryTransformation`] — six PONI-component formula strings evaluated
//!     by a bit-exact f64 expression evaluator ([`expr`]) that reproduces
//!     `numexpr`'s scalar arithmetic. `call(param, pos)` turns a parameter vector
//!     and a goniometer position into the six PONI scalars.
//!   * [`Goniometer`] — wraps a transformation + parameter vector + detector +
//!     wavelength; [`Goniometer::get_ai`] builds an [`rsfai::AzimuthalIntegrator`]
//!     from the transformed PONI.
//!   * [`GoniometerRefinement`] — refines the parameter vector against a set of
//!     [`SingleGeometry`] control-point fits, reusing the bit-exact residual/chi²
//!     machinery of [`rsfai_calib::GeometryRefinement`].
//!   * [`MultiGeometryFiber`] — combines several [`rsfai_fiber::FiberIntegrator`]
//!     frames by direct per-bin accumulator summation (the exact arithmetic
//!     pyFAI's `MultiGeometryFiber` uses — *not* the `Integrate1dResult.union`
//!     fold the azimuthal `MultiGeometry` uses). Bit-exact; fiber has no error
//!     model, so there is no variance path.
//!
//! Two parity tiers, kept structurally separate (see `doc/bit-exact-ladder.md`):
//!
//!   * **Bit-exact (identical-input).** The transformation outputs, the per-PONI
//!     geometry, and the residual/chi² at a fixed parameter vector all match
//!     pyFAI bit-for-bit (`f64::to_bits`); the only ULP-budgeted part is the
//!     geometry `atan2`/`sin`/`cos` inherited from the validated `rsfai_geometry`.
//!   * **Tolerance (converged params).** [`GoniometerRefinement::refine`] runs an
//!     `argmin` Nelder-Mead minimization (isolated in [`optimizer`]); the
//!     bit-exact core never touches `argmin`. The converged parameters are gated
//!     at a recorded relative tolerance with `cost_rust <= cost_pyfai`.

pub mod expr;
pub mod multifiber;
mod optimizer;
pub mod refinement;
pub mod transform;

pub use multifiber::{MultiFiber1dResult, MultiFiber2dResult, MultiGeometryFiber};
pub use refinement::{Goniometer, GoniometerRefinement, SingleGeometry};
pub use transform::{GeometryTransformation, PoniParam, TransformError};
