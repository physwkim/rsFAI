//! Errors raised while constructing an [`AzimuthalIntegrator`](crate::AzimuthalIntegrator).

use thiserror::Error;

/// Errors from loading geometry or resolving the detector model.
#[derive(Debug, Error)]
pub enum Error {
    /// The `.poni` file could not be parsed or loaded.
    #[error(transparent)]
    Geometry(#[from] rsfai_geometry::GeometryError),
    /// The PONI named a detector with no golden-validated path in this crate.
    /// Supply the detector explicitly via
    /// [`AzimuthalIntegrator::from_poni`](crate::AzimuthalIntegrator::from_poni).
    #[error("unsupported detector {0:?}: no golden-validated path; use AzimuthalIntegrator::from_poni with an explicit Detector")]
    UnsupportedDetector(String),
}

/// Result alias for [`Error`].
pub type Result<T> = std::result::Result<T, Error>;
