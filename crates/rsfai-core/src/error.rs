//! Error type for the core crate.

use thiserror::Error;

#[derive(Debug, Error)]
pub enum CoreError {
    #[error("I/O error reading {path}: {source}")]
    Io {
        path: String,
        #[source]
        source: std::io::Error,
    },

    #[error("failed to read .npy {path}: {source}")]
    ReadNpy {
        path: String,
        #[source]
        source: ndarray_npy::ReadNpyError,
    },

    #[error("failed to write .npy {path}: {source}")]
    WriteNpy {
        path: String,
        #[source]
        source: ndarray_npy::WriteNpyError,
    },

    #[error("failed to parse manifest {path}: {source}")]
    Manifest {
        path: String,
        #[source]
        source: serde_json::Error,
    },
}

pub type Result<T> = std::result::Result<T, CoreError>;
