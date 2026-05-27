//! M2 validation of the detector model against pyFAI golden data.
//!
//! * pixel centres (`centers_f64`) must be **bit-exact** vs the golden
//!   `pixel_p1`/`pixel_p2` (these are the f64 positions `position_array` feeds
//!   the geometry transform — pure `pixel*(idx+0.5)` algebra).
//! * the module-gap mask (`calc_mask`) must equal the golden `mask` exactly.
//!
//! `pixel_p1`/`pixel_p2` are gitignored (large); datasets lacking them are
//! skipped. Run `golden/gen_golden.py` first.

use std::path::PathBuf;

use rsfai_core::compare::compare_f64;
use rsfai_core::golden::{load_manifest, load_npy_f64, load_npy_i8};
use rsfai_detectors::Detector;

fn datasets_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../golden/datasets")
}

fn dataset_dirs() -> Vec<PathBuf> {
    let mut dirs = vec![];
    if let Ok(rd) = std::fs::read_dir(datasets_root()) {
        for e in rd.flatten() {
            if e.path().join("manifest.json").exists() {
                dirs.push(e.path());
            }
        }
    }
    dirs.sort();
    dirs
}

/// Build the rsFAI detector matching the golden manifest's detector name.
fn detector_for(name: &str) -> Option<Detector> {
    // pyFAI's detector.name reports the alias (e.g. "Pilatus 1M"); accept both
    // the alias and the class name.
    Some(match name {
        "Pilatus1M" | "Pilatus 1M" => Detector::pilatus1m(),
        "Eiger4M" | "Eiger 4M" => Detector::eiger4m(),
        _ => return None,
    })
}

#[test]
fn pixel_centers_bit_exact_and_mask_matches() {
    let mut checked = 0usize;
    for dir in dataset_dirs() {
        let manifest = load_manifest(dir.join("manifest.json")).expect("manifest");
        let det_name = manifest
            .extra
            .get("detector")
            .and_then(|d| d.get("name"))
            .and_then(|v| v.as_str())
            .expect("detector name");
        let Some(det) = detector_for(det_name) else {
            eprintln!("skip {}: detector {det_name} not ported", manifest.dataset);
            continue;
        };

        // ---- mask: calc_mask vs golden create_mask --------------------------
        let golden_mask = load_npy_i8(dir.join("mask.npy")).expect("mask");
        let mask = det.calc_mask().expect("module detector has a mask");
        assert_eq!(
            mask.as_slice().unwrap(),
            golden_mask.as_slice().unwrap(),
            "{}: calc_mask != golden mask",
            manifest.dataset
        );

        // ---- pixel centres: centers_f64 vs golden pixel_p1/p2 ---------------
        let p1f = dir.join("pixel_p1.npy");
        if !p1f.exists() {
            eprintln!(
                "skip centres for {}: regenerate golden (pixel_p1 gitignored)",
                manifest.dataset
            );
            // mask already checked; count it.
            checked += 1;
            continue;
        }
        let gp1 = load_npy_f64(&p1f).unwrap().as_slice().unwrap().to_vec();
        let gp2 = load_npy_f64(dir.join("pixel_p2.npy"))
            .unwrap()
            .as_slice()
            .unwrap()
            .to_vec();
        let (p1, p2) = det.centers_f64();
        let r1 = compare_f64(&p1, &gp1);
        let r2 = compare_f64(&p2, &gp2);
        eprintln!(
            "{}: centers p1(ulp={} mism={}) p2(ulp={} mism={})  mask ok",
            manifest.dataset, r1.max_ulp, r1.bit_mismatches, r2.max_ulp, r2.bit_mismatches
        );
        assert!(
            r1.is_bit_exact(),
            "{}: p1 not bit-exact: {r1:?}",
            manifest.dataset
        );
        assert!(
            r2.is_bit_exact(),
            "{}: p2 not bit-exact: {r2:?}",
            manifest.dataset
        );

        checked += 1;
    }
    assert!(
        checked > 0,
        "no golden datasets found; run golden/gen_golden.py"
    );
}
