//! Loading golden `.npy` arrays and `manifest.json` produced by
//! `golden/gen_golden.py`. The on-disk `.npy` format preserves exact bits, so a
//! round-trip through numpy → disk → ndarray is lossless.

use std::path::Path;

use ndarray::ArrayD;
use serde::Deserialize;

use thiserror::Error;

/// Errors from loading or writing golden fixtures.
#[derive(Debug, Error)]
pub enum GoldenError {
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

pub type Result<T> = std::result::Result<T, GoldenError>;

fn path_str(p: &Path) -> String {
    p.to_string_lossy().into_owned()
}

macro_rules! npy_loader {
    ($name:ident, $ty:ty) => {
        /// Load an `.npy` file as a dynamic-dimension array of the given dtype.
        /// Errors if the file's dtype does not match `$ty`.
        pub fn $name<P: AsRef<Path>>(path: P) -> Result<ArrayD<$ty>> {
            let p = path.as_ref();
            ndarray_npy::read_npy::<_, ArrayD<$ty>>(p).map_err(|source| GoldenError::ReadNpy {
                path: path_str(p),
                source,
            })
        }
    };
}

npy_loader!(load_npy_f64, f64);
npy_loader!(load_npy_f32, f32);
npy_loader!(load_npy_i32, i32);
npy_loader!(load_npy_i8, i8);

/// Load a detector frame `image.npy` as the f32 the integrator consumes,
/// regardless of whether `gen_golden.py` stored it `int32` (Pilatus-class
/// frames) or `float32` (Eiger-class frames). This is the single owner for "the
/// detector frame as f32": every golden test routes through it, so a new
/// float32-frame detector cannot reopen the int32-only assumption at a call
/// site. Self-describing (reads the `.npy` dtype header), so it needs no
/// manifest — usable by the reduced-manifest OpenCL datasets too.
pub fn load_image_f32<P: AsRef<Path>>(path: P) -> Result<Vec<f32>> {
    let p = path.as_ref();
    match ndarray_npy::read_npy::<_, ArrayD<i32>>(p) {
        Ok(a) => Ok(a.iter().map(|&v| v as f32).collect()),
        // int32 reader rejects a float32 frame with WrongDescriptor; retry f32.
        Err(ndarray_npy::ReadNpyError::WrongDescriptor(_)) => {
            Ok(load_npy_f32(p)?.iter().copied().collect())
        }
        Err(source) => Err(GoldenError::ReadNpy {
            path: path_str(p),
            source,
        }),
    }
}

/// Write an `.npy` file (used by tests that materialize fixtures).
pub fn write_npy_f64<P: AsRef<Path>>(path: P, array: &ArrayD<f64>) -> Result<()> {
    let p = path.as_ref();
    ndarray_npy::write_npy(p, array).map_err(|source| GoldenError::WriteNpy {
        path: path_str(p),
        source,
    })
}

/// Provenance + configuration recorded alongside every golden dataset. Anything
/// not modeled explicitly is preserved in `extra` so the manifest schema can
/// grow without breaking older readers.
#[derive(Debug, Clone, Deserialize)]
pub struct Manifest {
    pub dataset: String,
    pub pyfai_version: String,
    pub numpy_version: String,
    pub platform: String,
    pub omp_num_threads: String,
    /// Integration config: npt, unit, method tuple, error_model, ranges, etc.
    #[serde(default)]
    pub config: serde_json::Value,
    /// Per-quantity ULP budget for transcendental Tier-B arrays (e.g. tth/chi/q).
    #[serde(default)]
    pub ulp_budget: serde_json::Value,
    /// Forward-compatible catch-all for fields added later.
    #[serde(flatten)]
    pub extra: serde_json::Map<String, serde_json::Value>,
}

/// Load and parse a `manifest.json`.
pub fn load_manifest<P: AsRef<Path>>(path: P) -> Result<Manifest> {
    let p = path.as_ref();
    let text = std::fs::read_to_string(p).map_err(|source| GoldenError::Io {
        path: path_str(p),
        source,
    })?;
    serde_json::from_str(&text).map_err(|source| GoldenError::Manifest {
        path: path_str(p),
        source,
    })
}

/// The ULP budget for a named quantity, defaulting to 0 (bit-exact) when the
/// manifest does not list it.
impl Manifest {
    pub fn ulp_budget_for(&self, quantity: &str) -> u64 {
        self.ulp_budget
            .get(quantity)
            .and_then(|v| v.as_u64())
            .unwrap_or(0)
    }
}

#[cfg(test)]
mod roundtrip_tests {
    use ndarray::{ArrayD, IxDyn};

    use super::{load_npy_f64, write_npy_f64};
    use crate::compare::compare_f64;

    /// M0 gate: write a known array to `.npy`, read it back, and confirm it is
    /// bit-exact against itself. Proves the loader + comparator round-trip with
    /// no bit loss, independent of any pyFAI-generated dataset.
    #[test]
    fn npy_roundtrip_is_bit_exact() {
        let data = vec![
            0.0_f64,
            -0.0,
            1.0,
            -2.5,
            std::f64::consts::PI,
            1e-300,
            1e300,
            f64::INFINITY,
        ];
        let arr = ArrayD::from_shape_vec(IxDyn(&[2, 4]), data).unwrap();
        let path = std::env::temp_dir().join(format!("rsfai_roundtrip_{}.npy", std::process::id()));
        write_npy_f64(&path, &arr).expect("write");
        let back = load_npy_f64(&path).expect("read");
        std::fs::remove_file(&path).ok();
        let report = compare_f64(arr.as_slice().unwrap(), back.as_slice().unwrap());
        assert!(
            report.is_bit_exact(),
            "round-trip not bit-exact: {report:?}"
        );
    }
}
