//! M2 Tier-B validation of the geometry-level correction arrays.
//!
//! * `solid_angle_array` must be **bit-exact** vs the golden `solidangle`
//!   (f64): the f32 position path + f64 `f_cosa` + `cosa**3` is reproduced
//!   exactly.
//! * `polarization_array` must be **bit-exact** vs the golden `polarization`
//!   (f32), when the dataset set a polarization factor. `cos` via Rust libm
//!   matches numexpr on this machine (same Tier-B libm rationale as M1).

use std::path::PathBuf;

use rsfai_core::compare::{compare_f32, compare_f64};
use rsfai_core::golden::{load_manifest, load_npy_f32, load_npy_f64};
use rsfai_detectors::Detector;
use rsfai_geometry::{polarization_array, solid_angle_array, PoniFile};

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

fn detector_for(name: &str) -> Option<Detector> {
    Some(match name {
        "Pilatus1M" | "Pilatus 1M" => Detector::pilatus1m(),
        "Eiger4M" | "Eiger 4M" => Detector::eiger4m(),
        _ => return None,
    })
}

fn flat(dir: &std::path::Path, name: &str) -> Vec<f64> {
    load_npy_f64(dir.join(name))
        .unwrap()
        .as_slice()
        .unwrap()
        .to_vec()
}

#[test]
fn solid_angle_and_polarization_bit_exact() {
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
        let poni = PoniFile::load(dir.join("geometry.poni")).expect("poni");

        // ---- solid angle (order 3, f64) -----------------------------------
        let sa = solid_angle_array(&det, poni.dist, poni.poni1, poni.poni2, 3.0);
        let golden_sa = flat(&dir, "solidangle.npy");
        let r = compare_f64(&sa, &golden_sa);
        eprintln!(
            "{}: solid_angle max_ulp={} bit_mismatches={}/{}",
            manifest.dataset, r.max_ulp, r.bit_mismatches, r.total
        );
        assert!(
            r.is_bit_exact(),
            "{}: solid_angle not bit-exact: {r:?}",
            manifest.dataset
        );

        // ---- polarization (f32), only when a factor was set ---------------
        let polf = dir.join("polarization.npy");
        if polf.exists() {
            let factor = manifest.config["polarization_factor"]
                .as_f64()
                .expect("polarization_factor");
            // pos_zyx gives lab coords (z,y,x) per pixel; need x=t2,y=t1,z=t3.
            let pos = flat(&dir, "pos_zyx.npy");
            let n = pos.len() / 3;
            let (mut gz, mut gy, mut gx) = (
                Vec::with_capacity(n),
                Vec::with_capacity(n),
                Vec::with_capacity(n),
            );
            for k in 0..n {
                gz.push(pos[3 * k]);
                gy.push(pos[3 * k + 1]);
                gx.push(pos[3 * k + 2]);
            }
            let pol = polarization_array(&gx, &gy, &gz, factor, 0.0);
            let golden_pol = load_npy_f32(&polf).unwrap().as_slice().unwrap().to_vec();
            let rp = compare_f32(&pol, &golden_pol);
            eprintln!(
                "{}: polarization[factor={factor}] max_ulp={} bit_mismatches={}/{}",
                manifest.dataset, rp.max_ulp, rp.bit_mismatches, rp.total
            );
            assert!(
                rp.is_bit_exact(),
                "{}: polarization not bit-exact: {rp:?}",
                manifest.dataset
            );
        }

        checked += 1;
    }
    assert!(
        checked > 0,
        "no golden datasets found; run golden/gen_golden.py"
    );
}
