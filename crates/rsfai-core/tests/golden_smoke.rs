//! M0 gate: prove the golden harness loads the *real* committed pyFAI curves.
//!
//! Reads every `golden/datasets/<config>/` that has a `manifest.json`, loads the
//! committed `out_*.npy` golden curves, checks their length against the
//! manifest's expected bin counts (`npt` for 1D, `npt_rad`/`npt_azim` for 2D),
//! and bit-compares each against itself (round-trip through the loader is
//! lossless). The actual rsFAI-vs-pyFAI comparisons arrive in M1+; this test
//! only validates the harness on real data.

use std::path::{Path, PathBuf};

use rsfai_core::compare::{compare_f32, compare_f64};
use rsfai_core::golden::{load_manifest, load_npy_f32, load_npy_f64};

fn golden_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../golden/datasets")
}

/// Load an f64 golden axis (`out_radial`/`out_azimuthal`), assert its length,
/// and prove the loader round-trips it bit-for-bit (compare vs itself).
fn check_axis_f64(dir: &Path, file: &str, expected_len: usize, dataset: &str) {
    let a = load_npy_f64(dir.join(file)).unwrap_or_else(|_| panic!("{dataset}: load {file}"));
    assert_eq!(
        a.len(),
        expected_len,
        "{dataset}: {file} length {} != expected {expected_len}",
        a.len()
    );
    let s = a.as_slice().expect("C-contiguous golden");
    assert!(
        compare_f64(s, s).is_bit_exact(),
        "{dataset}: {file} not bit-exact vs self"
    );
}

/// Load an f32 golden curve (`out_intensity`), assert its length, and prove the
/// loader round-trips it bit-for-bit. For 2D the length is the flattened cell
/// count (`npt_rad * npt_azim`).
fn check_curve_f32(dir: &Path, file: &str, expected_len: usize, dataset: &str) {
    let a = load_npy_f32(dir.join(file)).unwrap_or_else(|_| panic!("{dataset}: load {file}"));
    assert_eq!(
        a.len(),
        expected_len,
        "{dataset}: {file} length {} != expected {expected_len}",
        a.len()
    );
    let s = a.as_slice().expect("C-contiguous golden");
    assert!(
        compare_f32(s, s).is_bit_exact(),
        "{dataset}: {file} not bit-exact vs self"
    );
}

#[test]
fn golden_curves_load_and_roundtrip() {
    let root = golden_root();
    assert!(
        root.exists(),
        "golden datasets dir missing: {} (run golden/gen_golden.py)",
        root.display()
    );

    let mut checked = 0usize;
    for entry in std::fs::read_dir(&root).expect("read datasets dir") {
        let dir = entry.expect("dir entry").path();
        let manifest_path = dir.join("manifest.json");
        // Cython golden datasets only: skip Phase-2 OpenCL datasets (they carry
        // an `opencl_params.json`); `rsfai-opencl`'s own golden test owns those.
        if !manifest_path.exists() || dir.join("opencl_params.json").exists() {
            continue;
        }
        let manifest = load_manifest(&manifest_path).expect("parse manifest");
        let cfg = &manifest.config;
        let dim = cfg["dim"].as_u64().unwrap_or(1);

        // Expected golden-curve lengths differ by dimension. A 1D config carries
        // `npt` (radial and intensity both length npt); a 2D config carries
        // `npt_rad`/`npt_azim` (radial = npt_rad, intensity = npt_rad * npt_azim
        // flattened, plus a separate azimuthal axis of npt_azim). The smoke test
        // covers BOTH — its purpose is to prove every committed curve loads and
        // round-trips, so 2D datasets are validated, not skipped.
        let (n_radial, n_intensity, npt_azim) = if dim == 2 {
            let npt_rad = cfg["npt_rad"]
                .as_u64()
                .expect("npt_rad in 2D manifest config") as usize;
            let npt_azim = cfg["npt_azim"]
                .as_u64()
                .expect("npt_azim in 2D manifest config") as usize;
            (npt_rad, npt_rad * npt_azim, Some(npt_azim))
        } else {
            let npt = cfg["npt"].as_u64().expect("npt in 1D manifest config") as usize;
            (npt, npt, None)
        };

        // Radial axis (f64) and intensity curve (f32): present in every config.
        check_axis_f64(&dir, "out_radial.npy", n_radial, &manifest.dataset);
        check_curve_f32(&dir, "out_intensity.npy", n_intensity, &manifest.dataset);
        // Azimuthal axis (f64): 2D only.
        if let Some(npt_azim) = npt_azim {
            check_axis_f64(&dir, "out_azimuthal.npy", npt_azim, &manifest.dataset);
        }

        eprintln!(
            "ok: {} (dim={dim}, radial={n_radial}, intensity={n_intensity}, pyFAI {}, method={:?})",
            manifest.dataset, manifest.pyfai_version, cfg["method"]
        );
        checked += 1;
    }

    assert!(
        checked > 0,
        "no golden datasets with a manifest under {}",
        root.display()
    );
}
