//! Error type for the geometry crate.

use thiserror::Error;

#[derive(Debug, Error)]
pub enum GeometryError {
    #[error("PONI parse error: {0}")]
    PoniParse(String),

    #[error("missing required PONI key: {0}")]
    PoniMissingKey(&'static str),

    #[error("I/O error reading {path}: {source}")]
    Io {
        path: String,
        #[source]
        source: std::io::Error,
    },
}

pub type Result<T> = std::result::Result<T, GeometryError>;
