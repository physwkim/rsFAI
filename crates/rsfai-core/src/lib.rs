//! `rsfai-core` — the shared foundation for the rsFAI bit-exact port of pyFAI.
//!
//! Holds the pyFAI [`dtype`] contract, the golden-dataset [`compare`] utilities
//! (bitwise + ULP), and the [`golden`] loaders. Higher crates (geometry,
//! detectors, preproc, integrate) build on these so that "bit-exact vs pyFAI"
//! means the same thing everywhere. See `doc/bit-exact-ladder.md`.

pub mod compare;
pub mod dtype;
pub mod error;
pub mod golden;

pub use dtype::{
    calc_upper_bound, AccT, BufferT, DataT, ErrorModel, IndexT, LutEntry, MaskT, PositionT, EPS32,
};
pub use error::{CoreError, Result};

#[cfg(test)]
mod roundtrip_tests {
    use ndarray::{ArrayD, IxDyn};

    use crate::compare::compare_f64;
    use crate::golden::{load_npy_f64, write_npy_f64};

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

        let path =
            std::env::temp_dir().join(format!("rsfai_core_roundtrip_{}.npy", std::process::id()));
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
