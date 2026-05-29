//! Read diffraction frames from HDF5 / NeXus files.
//!
//! A thin, read-focused layer over the pure-Rust [`rust_hdf5`] crate: open a
//! file, address a dataset by its hierarchical path (e.g. `entry/data/data`,
//! the conventional NeXus frame location), and read it into a flat row-major
//! buffer plus its shape — typed as `f32` / `f64` / `i32` (or any [`H5Type`]),
//! matching the rsFAI dtype contract (weights f32, positions/accumulators f64,
//! indices i32).
//!
//! NeXus files are plain HDF5 with naming conventions, so reading a known
//! dataset path is the practical "load a frame" path. Richer NeXus signal
//! auto-resolution (`@default` → NXdata `@signal`) can layer on top later;
//! `rust_hdf5`'s attribute API ([`H5Dataset::attr`]) is the hook for it.

use std::path::Path;

use rust_hdf5::{H5File, H5Type};

/// Errors from reading an HDF5 / NeXus file.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// An error from the underlying `rust_hdf5` layer (open, navigate, read).
    #[error("hdf5: {0}")]
    Hdf5(#[from] rust_hdf5::Hdf5Error),
}

/// `Result` specialised to this crate's [`Error`].
pub type Result<T> = std::result::Result<T, Error>;

/// A dataset read into a flat, row-major (C-order) buffer with its shape.
#[derive(Clone, Debug)]
pub struct Array<T> {
    /// Row-major elements; `data.len() == shape.iter().product()`.
    pub data: Vec<T>,
    /// Dimension sizes, slowest-varying first (e.g. `[rows, cols]` for a frame).
    pub shape: Vec<usize>,
}

impl<T> Array<T> {
    /// Number of elements in the flat buffer.
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// True if the dataset has no elements.
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// `(rows, cols)` for a 2-D frame.
    ///
    /// # Panics
    /// Panics if the dataset is not 2-dimensional.
    pub fn dims2(&self) -> (usize, usize) {
        assert_eq!(
            self.shape.len(),
            2,
            "dims2() on a {}-D dataset (shape {:?})",
            self.shape.len(),
            self.shape
        );
        (self.shape[0], self.shape[1])
    }
}

/// Read an entire dataset, addressed by its hierarchical `path` (e.g.
/// `entry/data/data`), into a typed flat buffer + shape.
///
/// `T` is the element type to read as; it must match (or be a valid HDF5
/// conversion of) the on-disk dtype. Use [`read_dataset_f32`] /
/// [`read_dataset_f64`] / [`read_dataset_i32`] for the common rsFAI dtypes.
pub fn read_dataset<T: H5Type>(file: impl AsRef<Path>, path: &str) -> Result<Array<T>> {
    let file = H5File::open(file)?;
    let ds = file.dataset(path)?;
    let shape = ds.shape();
    let data = ds.read_raw::<T>()?;
    Ok(Array { data, shape })
}

/// Read a dataset as `f32` (the rsFAI weights / image dtype).
pub fn read_dataset_f32(file: impl AsRef<Path>, path: &str) -> Result<Array<f32>> {
    read_dataset::<f32>(file, path)
}

/// Read a dataset as `f64` (the rsFAI positions / accumulator dtype).
pub fn read_dataset_f64(file: impl AsRef<Path>, path: &str) -> Result<Array<f64>> {
    read_dataset::<f64>(file, path)
}

/// Read a dataset as `i32` (the rsFAI index / counts dtype).
pub fn read_dataset_i32(file: impl AsRef<Path>, path: &str) -> Result<Array<i32>> {
    read_dataset::<i32>(file, path)
}
