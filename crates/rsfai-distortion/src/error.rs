//! Error type for `rsfai-distortion`. Spline-file parsing has failure modes
//! (malformed fixed-width fields, missing sections) that `rsfai-core`'s I/O
//! error type does not model; rather than widen the shared core type for one
//! consumer, this crate carries its own error and wraps core I/O errors.

use thiserror::Error;

/// Failures from parsing a `.spline` file or building a distortion LUT.
#[derive(Debug, Error)]
pub enum DistortionError {
    /// I/O failure opening or reading a spline file.
    #[error("I/O error reading {path}: {source}")]
    Io {
        path: String,
        #[source]
        source: std::io::Error,
    },

    /// A malformed or incomplete `.spline` file.
    #[error("spline parse error: {0}")]
    Parse(String),
}

pub type Result<T> = std::result::Result<T, DistortionError>;
