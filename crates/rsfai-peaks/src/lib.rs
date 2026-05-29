//! Peak-finding primitives ported from pyFAI, validated bit-exact (or, for the
//! ellipse eigensolver, at a recorded tolerance) against the deterministic
//! reference in `golden/datasets_peaks/`.
//!
//! Modules:
//!   * [`label`] — `scipy.ndimage.label` connected components + the exact
//!     Euclidean distance transform with feature indices, the deterministic
//!     building blocks `pyFAI.massif.Massif` uses.
//!   * [`bilinear`] — the `Bilinear` peak interpolator (discrete hill-climb +
//!     sub-pixel Taylor refinement) shared by the watershed and Massif.
//!   * [`watershed`] — `InverseWatershed`: label every pixel to its catchment
//!     peak, build regions, and extract peak coordinates.
//!   * [`blob`] — Difference-of-Gaussian keypoint detection + Hessian refinement.
//!   * [`ellipse`] — Fitzgibbon algebraic ellipse fit (eigensolver, Tier-B).
//!
//! See `doc/bit-exact-ladder.md` for the parity tier of each output.

pub mod bilinear;
pub mod blob;
pub mod ellipse;
pub mod label;
pub mod watershed;

pub use bilinear::Bilinear;
pub use blob::{local_max, refine_hessian, DogStack, RefinedKeypoint, TRESH};
pub use ellipse::{design_matrix, fit_ellipse, Ellipse, EllipseError};
pub use label::{distance_transform_edt, label, EdtResult, Structure};
pub use watershed::{InverseWatershed, Region};
