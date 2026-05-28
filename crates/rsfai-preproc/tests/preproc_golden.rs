//! M3 Tier-A validation: `preproc4` must be **bit-exact** vs the golden
//! `preproc` array, given the same inputs pyFAI's `preproc(..., split_result=4,
//! dtype=float32)` received.
//!
//! Inputs are cast to f32 exactly as pyFAI's wrapper does: `image` (int32) →
//! f32, `solidangle` (f64) → f32, `polarization` already f32. The golden
//! `preproc.npy` is `(s0, s1, 4)` row-major = `[signal, variance, norm, count]`
//! per pixel — the layout `preproc4` returns.

use std::path::PathBuf;

use rsfai_core::compare::compare_f32;
use rsfai_core::dtype::ErrorModel;
use rsfai_core::golden::{load_image_f32, load_manifest, load_npy_f32, load_npy_f64, load_npy_i8};
use rsfai_preproc::{preproc4, PreprocOptions};

fn datasets_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../golden/datasets")
}

fn dataset_dirs() -> Vec<PathBuf> {
    let mut dirs = vec![];
    if let Ok(rd) = std::fs::read_dir(datasets_root()) {
        for e in rd.flatten() {
            // Cython golden datasets only: skip the Phase-2 OpenCL datasets,
            // which carry an `opencl_params.json` and a reduced manifest with no
            // cython intermediates. `rsfai-opencl`'s own golden test owns those.
            if e.path().join("manifest.json").exists()
                && !e.path().join("opencl_params.json").exists()
            {
                dirs.push(e.path());
            }
        }
    }
    dirs.sort();
    dirs
}

#[test]
fn preproc4_bit_exact() {
    let mut checked = 0usize;
    for dir in dataset_dirs() {
        let manifest = load_manifest(dir.join("manifest.json")).expect("manifest");

        // Detector frame -> f32, matching `raw.astype(float32)`. The single
        // owner handles both int32 (Pilatus-class) and float32 (Eiger-class)
        // frames, so this site cannot reopen the int32-only assumption.
        let data = load_image_f32(dir.join("image.npy")).expect("image");

        // solidangle (f64) -> f32 (pyFAI casts to the working dtype).
        let solidangle: Option<Vec<f32>> = if dir.join("solidangle.npy").exists() {
            Some(
                load_npy_f64(dir.join("solidangle.npy"))
                    .unwrap()
                    .iter()
                    .map(|&v| v as f32)
                    .collect(),
            )
        } else {
            None
        };
        let polarization: Option<Vec<f32>> = if dir.join("polarization.npy").exists() {
            Some(
                load_npy_f32(dir.join("polarization.npy"))
                    .unwrap()
                    .as_slice()
                    .unwrap()
                    .to_vec(),
            )
        } else {
            None
        };
        let mask = load_npy_i8(dir.join("mask.npy"))
            .unwrap()
            .as_slice()
            .unwrap()
            .to_vec();

        let norm = manifest.config["normalization_factor"]
            .as_f64()
            .unwrap_or(1.0) as f32;
        let code = manifest.config["error_model_code"].as_i64().unwrap_or(0) as i32;
        // Poisson and Hybrid both take variance = max(1, signal).
        let poissonian = ErrorModel::from_code(code)
            .unwrap_or_else(|| panic!("unknown error_model_code {code}"))
            .poissonian();

        // Dummy (dead/gap-pixel) masking the integrator applies, recorded in the
        // manifest as f32-exact floats. `delta_dummy` may be null (exact match,
        // i.e. delta 0). pyFAI always passes these to preproc, so the golden
        // `preproc.npy` reflects them — reproduce them here.
        let dummy = manifest.config["dummy"].as_f64();
        let check_dummy = dummy.is_some();
        let dummy_v = dummy.unwrap_or(0.0) as f32;
        let delta_dummy = manifest.config["delta_dummy"].as_f64().unwrap_or(0.0) as f32;

        let opt = PreprocOptions {
            solidangle: solidangle.as_deref(),
            polarization: polarization.as_deref(),
            mask: Some(&mask),
            normalization_factor: norm,
            poissonian,
            check_dummy,
            dummy: dummy_v,
            delta_dummy,
            ..Default::default()
        };
        let out = preproc4(&data, &opt);

        let golden = load_npy_f32(dir.join("preproc.npy"))
            .unwrap()
            .as_slice()
            .unwrap()
            .to_vec();
        let r = compare_f32(&out, &golden);
        eprintln!(
            "{}: preproc4[poisson={poissonian}] max_ulp={} bit_mismatches={}/{}",
            manifest.dataset, r.max_ulp, r.bit_mismatches, r.total
        );
        assert!(
            r.is_bit_exact(),
            "{}: preproc not bit-exact: {r:?}",
            manifest.dataset
        );
        checked += 1;
    }
    assert!(
        checked > 0,
        "no golden datasets found; run golden/gen_golden.py"
    );
}
