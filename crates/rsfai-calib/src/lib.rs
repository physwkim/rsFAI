//! Calibration: control points + geometry refinement, ported from pyFAI's
//! `control_points.py` and `geometryRefinement.py`.
//!
//! This module provides the [`ControlPoints`] data container (ring <-> (y,x)
//! points). The geometry refinement core and the iterative `refine` land in
//! follow-up changes.

pub mod control_points;

pub use control_points::{ControlPoints, PointGroup};
