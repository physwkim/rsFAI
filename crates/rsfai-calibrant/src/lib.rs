//! `rsfai-calibrant` — calibrant ring positions and crystallography, ported
//! from `pyFAI/crystallography/{calibrant.py, calibrant_factory.py, cell.py,
//! space_groups.py}` and `pyFAI/io/calibrant_config.py`.
//!
//! Two entry points:
//!
//!   * From a shipped `.D` d-spacing file (the common path): parse the file into
//!     a [`config::CalibrantConfig`], build a [`Calibrant`], set a wavelength,
//!     and read the Bragg-law 2θ ring positions via [`Calibrant::get_2th`] or
//!     the unit-scaled peaks via [`Calibrant::get_peaks`].
//!
//!   * From a [`Cell`] (lattice parameters + centering): enumerate the
//!     d-spacings via [`Cell::calculate_dspacing`] / [`Cell::build_calibrant_config`]
//!     — the same computation that generated the shipped `.D` files.
//!
//! Numerical contract (see `doc/bit-exact-ladder.md`): d-spacings (from a `.D`
//! file or from a `Cell`'s pure-`+-*/`-`sqrt` algebra) are **bit-exact** vs
//! pyFAI; the single transcendental on the ring-position path, the Bragg
//! `2 * asin(5e9 * λ / d)`, is **Tier-B**, validated within a recorded ULP
//! budget against pyFAI's `math.asin`.

pub mod calibrant;
pub mod cell;
pub mod config;
pub mod space_groups;

use std::fs;
use std::io;
use std::path::Path;

pub use calibrant::{Calibrant, PeakUnit, CONST_HC};
pub use cell::{Cell, Lattice, LatticeParams};
pub use config::{CalibrantConfig, Miller, Reflection};
pub use space_groups::Centering;

impl Calibrant {
    /// Load a calibrant from a `.D` file on disk (mirrors `Calibrant.load_file`
    /// for the on-disk, non-`pyfai:` case).
    pub fn load_file<P: AsRef<Path>>(path: P) -> io::Result<Calibrant> {
        let text = fs::read_to_string(path)?;
        Ok(Calibrant::from_dspacing_file_str(&text))
    }
}

impl CalibrantConfig {
    /// Parse a `.D` file from disk into a [`CalibrantConfig`]
    /// (`CalibrantConfig.from_dspacing`).
    pub fn from_dspacing_file<P: AsRef<Path>>(path: P) -> io::Result<CalibrantConfig> {
        let text = fs::read_to_string(path)?;
        Ok(CalibrantConfig::from_dspacing_str(&text))
    }
}

impl Cell {
    /// Convert a cell to a calibrant, `Cell.to_calibrant`.
    pub fn to_calibrant(&mut self, dmin: f64) -> Calibrant {
        let config = self.build_calibrant_config(dmin);
        Calibrant::from_config(config)
    }
}
