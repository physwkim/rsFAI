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

pub mod expr;
pub mod transform;

pub use transform::{GeometryTransformation, PoniParam, TransformError};
