//! `rsfai-geometry` — PONI geometry, the pixel→sample coordinate transform, and
//! radial/azimuthal unit equations, ported from `pyFAI/geometry/core.py`,
//! `pyFAI/ext/_geometry.pyx`, `pyFAI/io/ponifile.py`, and `pyFAI/units.py`.
//!
//! Pipeline (matching pyFAI's `center_array`):
//!   detector pixel centres `(p1,p2)` → [`transform::calc_pos_zyx`] → lab
//!   `(z,y,x)` → [`units::center_array`] → radial/azimuthal value.
//!
//! The bbox/full pixel-splitting engines additionally need the per-pixel
//! half-widths (`delta_array`) and the four-corner radial/chi array
//! (`corner_array`); both live in [`corners`], built from the corner-grid lab
//! coords (`calc_pos_zyx` over [`rsfai_detectors::Detector::corner_positions_f64`]).

pub mod corners;
pub mod corrections;
pub mod error;
pub mod poni;
pub mod transform;
pub mod units;

pub use corners::{corner_array_f32, delta_chi, delta_radial};
pub use corrections::{polarization_array, solid_angle_array};
pub use error::{GeometryError, Result};
pub use poni::PoniFile;
pub use transform::{calc_pos_zyx, PosZyx};
pub use units::{center_array, center_value, equation, unscaled_center_array, Space, Unit};
