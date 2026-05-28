//! Validation of the corner / delta geometry port against pyFAI golden data.
//!
//! `corner_array_f32` reproduces pyFAI's `corner_array(unit, scale=False)` — the
//! cython `_geometry.calc_rad_azim` fast path — for a contiguous flat detector:
//! the corner-grid lab coords (`calc_pos_zyx` over `corner_positions_f64`) →
//! per-node radial + chi (f64, stored f32) → gathered into the `(npix,4,2)`
//! winding. It is asserted **bit-exact** vs the golden `corners.npy` (f32) — the
//! same f64 `+ - * / sqrt`/`atan2`/`sin` the bit-exact `pos0_center`/`chi_center`
//! already validate, just over the corner grid and downcast to f32.
//!
//! `delta_radial` / `delta_chi` reproduce `delta_array(unit)` / `delta_array(
//! "chi_rad")` from that corner array + the f64 center array, asserted bit-exact
//! vs `pos0_delta.npy` / `chi_delta.npy`. (chi_delta is built by pyFAI through
//! the numpy `corner_array(CHI)` path; on the golden-generation machine its
//! `atan2` agrees bit-for-bit with the cython corner chi, so the cython corner
//! array reproduces it exactly — verified here.)
//!
//! Needs the gitignored per-pixel arrays (`corners.npy`, `pos0_delta.npy`, ...);
//! run `golden/gen_golden.py` first. Datasets lacking them are skipped.

use std::path::{Path, PathBuf};

use rsfai_core::compare::{compare_f32, compare_f64};
use rsfai_core::golden::{load_manifest, load_npy_f32, load_npy_f64, Manifest};
use rsfai_detectors::Detector;
use rsfai_geometry::{
    calc_pos_zyx, corner_array_f32, delta_chi, delta_radial, unscaled_center_array, PoniFile,
    Space, Unit,
};

fn datasets_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../golden/datasets")
}

fn dataset_dirs() -> Vec<PathBuf> {
    let mut dirs = vec![];
    if let Ok(rd) = std::fs::read_dir(datasets_root()) {
        for e in rd.flatten() {
            let p = e.path();
            // Cython golden datasets only: skip the Phase-2 OpenCL datasets
            // (reduced manifest, no cython intermediates).
            if p.join("manifest.json").exists() && !p.join("opencl_params.json").exists() {
                dirs.push(p);
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

fn detector_name(m: &Manifest) -> Option<&str> {
    m.extra
        .get("detector")
        .and_then(|d| d.get("name"))
        .and_then(|v| v.as_str())
}

fn orientation_of(m: &Manifest) -> Option<i32> {
    m.extra
        .get("detector")
        .and_then(|d| d.get("orientation"))
        .and_then(|v| v.as_i64())
        .map(|o| o as i32)
}

fn unit_for(name: &str) -> Option<Unit> {
    Some(match name {
        "q_nm^-1" => Unit::Q_NM_INV,
        "q_A^-1" => Unit::Q_A_INV,
        "2th_deg" => Unit::TTH_DEG,
        "2th_rad" => Unit::TTH_RAD,
        "r_mm" => Unit::R_MM,
        "r_m" => Unit::R_M,
        _ => return None,
    })
}

fn load_f64(path: &Path) -> Vec<f64> {
    load_npy_f64(path)
        .expect("load f64 npy")
        .as_slice()
        .expect("contiguous")
        .to_vec()
}

fn load_f32(path: &Path) -> Vec<f32> {
    load_npy_f32(path)
        .expect("load f32 npy")
        .as_slice()
        .expect("contiguous")
        .to_vec()
}

#[test]
fn corner_array_and_deltas_bit_exact() {
    let mut checked = 0usize;
    for dir in dataset_dirs() {
        let cornersf = dir.join("corners.npy");
        if !cornersf.exists() {
            // Datasets carrying the corner geometry (any split method dumps it);
            // skip the rest, and skip when per-pixel arrays were not regenerated.
            continue;
        }
        let manifest = load_manifest(dir.join("manifest.json")).expect("manifest");
        let Some(det_name) = detector_name(&manifest) else {
            continue;
        };
        let Some(mut det) = detector_for(det_name) else {
            eprintln!("skip {}: detector {det_name} not ported", manifest.dataset);
            continue;
        };
        if let Some(o) = orientation_of(&manifest) {
            det.orientation = o;
        }
        let unit_name = manifest.config["unit"].as_str().expect("unit");
        let Some(unit) = unit_for(unit_name) else {
            eprintln!("skip {}: unit {unit_name} not ported", manifest.dataset);
            continue;
        };

        let poni = PoniFile::load(dir.join("geometry.poni")).expect("poni");
        let wavelength = poni.wavelength.unwrap_or(0.0);
        let shape = det.shape;

        // Corner-grid lab coords: calc_pos_zyx over the (s0+1)x(s1+1) corner grid
        // (flat detectors are contiguous -> p3 = None).
        let (cp1, cp2) = det.corner_positions_f64();
        let grid = calc_pos_zyx(
            poni.dist,
            poni.poni1,
            poni.poni2,
            poni.rot1,
            poni.rot2,
            poni.rot3,
            &cp1,
            &cp2,
            None,
            det.orientation,
        );

        // ---- corner_array bit-exact vs corners.npy (radial[..,0] + chi[..,1]) ----
        // chiDiscAtPi = True (pyFAI's default; the corner array carries chi in
        // [-pi, pi) regardless of unit).
        let corners = corner_array_f32(&grid, shape, unit, wavelength, true);
        let g_corners = load_f32(&cornersf);
        let rc = compare_f32(&corners, &g_corners);
        eprintln!(
            "{}: corner_array max_ulp={} mism={}/{}",
            manifest.dataset, rc.max_ulp, rc.bit_mismatches, rc.total
        );
        assert!(
            rc.is_bit_exact(),
            "{}: corner_array not bit-exact vs corners.npy: {rc:?}",
            manifest.dataset
        );

        // ---- delta_radial bit-exact vs pos0_delta.npy ----------------------
        let (c1, c2) = det.centers_f64();
        let pos = calc_pos_zyx(
            poni.dist,
            poni.poni1,
            poni.poni2,
            poni.rot1,
            poni.rot2,
            poni.rot3,
            &c1,
            &c2,
            None,
            det.orientation,
        );
        // delta_array works on the UNSCALED internal radial (matching the
        // unscaled corner array); the 2th_deg scale would otherwise mismatch.
        let center = unscaled_center_array(unit.space, &pos.x, &pos.y, &pos.z, wavelength);
        let dr = delta_radial(&corners, &center);
        let g_dr = load_f64(&dir.join("pos0_delta.npy"));
        let rdr = compare_f64(&dr, &g_dr);
        eprintln!(
            "  delta_radial max_ulp={} mism={}/{}",
            rdr.max_ulp, rdr.bit_mismatches, rdr.total
        );
        assert!(
            rdr.is_bit_exact(),
            "{}: delta_radial not bit-exact vs pos0_delta.npy: {rdr:?}",
            manifest.dataset
        );

        // ---- delta_chi bit-exact vs chi_delta.npy --------------------------
        let chi_center = unscaled_center_array(Space::Chi, &pos.x, &pos.y, &pos.z, wavelength);
        let dc = delta_chi(&corners, &chi_center);
        let g_dc = load_f64(&dir.join("chi_delta.npy"));
        let rdc = compare_f64(&dc, &g_dc);
        eprintln!(
            "  delta_chi max_ulp={} mism={}/{}",
            rdc.max_ulp, rdc.bit_mismatches, rdc.total
        );
        assert!(
            rdc.is_bit_exact(),
            "{}: delta_chi not bit-exact vs chi_delta.npy: {rdc:?}",
            manifest.dataset
        );

        checked += 1;
    }
    assert!(
        checked > 0,
        "no golden datasets with corners.npy found; run golden/gen_golden.py"
    );
}
