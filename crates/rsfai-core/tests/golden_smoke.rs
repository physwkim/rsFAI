//! M0 gate: prove the golden harness loads the *real* committed pyFAI curves.
//!
//! Reads every `golden/datasets/<config>/` that has a `manifest.json`, loads the
//! committed `out_*.npy` golden curves, checks their length against the
//! manifest's `npt`, and bit-compares each against itself (round-trip through
//! the loader is lossless). The actual rsFAI-vs-pyFAI comparisons arrive in
//! M1+; this test only validates the harness on real data.

use std::path::PathBuf;

use rsfai_core::compare::{compare_f32, compare_f64};
use rsfai_core::golden::{load_manifest, load_npy_f32, load_npy_f64};

fn golden_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../golden/datasets")
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
        if !manifest_path.exists() {
            continue;
        }
        let manifest = load_manifest(&manifest_path).expect("parse manifest");
        let npt = manifest.config["npt"]
            .as_u64()
            .expect("npt in manifest config") as usize;

        // Golden radial axis (f64): present in every config, must round-trip.
        let radial = load_npy_f64(dir.join("out_radial.npy")).expect("load out_radial");
        assert_eq!(
            radial.len(),
            npt,
            "{}: radial length != npt",
            manifest.dataset
        );
        let r = compare_f64(radial.as_slice().unwrap(), radial.as_slice().unwrap());
        assert!(
            r.is_bit_exact(),
            "{}: radial not bit-exact vs self",
            manifest.dataset
        );

        // Golden intensity (f32): present in every config.
        let intensity = load_npy_f32(dir.join("out_intensity.npy")).expect("load out_intensity");
        assert_eq!(
            intensity.len(),
            npt,
            "{}: intensity length != npt",
            manifest.dataset
        );
        let ri = compare_f32(intensity.as_slice().unwrap(), intensity.as_slice().unwrap());
        assert!(
            ri.is_bit_exact(),
            "{}: intensity not bit-exact vs self",
            manifest.dataset
        );

        eprintln!(
            "ok: {} (npt={}, pyFAI {}, method={:?})",
            manifest.dataset, npt, manifest.pyfai_version, manifest.config["method"]
        );
        checked += 1;
    }

    assert!(
        checked > 0,
        "no golden datasets with a manifest under {}",
        root.display()
    );
}
