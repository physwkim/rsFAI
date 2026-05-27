//! M1 Tier-A/B validation of the geometry port against pyFAI golden data.
//!
//! Tier A: feed the golden float64 pixel centres into [`calc_pos_zyx`] and
//! require the lab coords `(z,y,x)` to be **bit-exact** vs the golden
//! `pos_zyx`. The golden comes from a pyFAI rebuilt with `-ffp-contract=off`,
//! so the Cython transform has no FMA fusion and the match is bitwise.
//!
//! Tier B: apply the unit equations to the golden lab coords and compare to the
//! golden `pos0_center` / `chi_center`. `r` (sqrt-only) is IEEE-exact; `q`/`2th`/
//! `chi` go through `atan2`/`sin`, which on the golden-generation machine agree
//! bit-for-bit between numexpr's libm and Rust's std libm — so these are
//! asserted bit-exact too, with the observed `max_ulp` printed so any future
//! libm divergence surfaces (it would be recorded as a manifest ULP budget).
//!
//! These tests need the large per-pixel arrays (`pixel_p1`, `pos_zyx`, ...) that
//! are gitignored; run `golden/gen_golden.py` first. Datasets lacking them are
//! skipped.

use std::path::{Path, PathBuf};

use rsfai_core::compare::compare_f64;
use rsfai_core::golden::{load_manifest, load_npy_f64, Manifest};
use rsfai_geometry::{calc_pos_zyx, units::center_array, PoniFile, Unit};

fn datasets_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../golden/datasets")
}

fn dataset_dirs() -> Vec<PathBuf> {
    let root = datasets_root();
    let mut dirs = vec![];
    if let Ok(rd) = std::fs::read_dir(&root) {
        for e in rd.flatten() {
            let p = e.path();
            // Cython golden datasets only: skip the Phase-2 OpenCL datasets,
            // which carry an `opencl_params.json` and a reduced manifest with no
            // cython intermediates. `rsfai-opencl`'s own golden test owns those.
            if p.join("manifest.json").exists() && !p.join("opencl_params.json").exists() {
                dirs.push(p);
            }
        }
    }
    dirs.sort();
    dirs
}

/// Extract the per-pixel z/y/x channels from a golden `pos_zyx` array stored as
/// row-major `(..., 3)` with last-axis order `[z, y, x]`.
fn split_zyx(flat: &[f64]) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let n = flat.len() / 3;
    let mut z = Vec::with_capacity(n);
    let mut y = Vec::with_capacity(n);
    let mut x = Vec::with_capacity(n);
    for k in 0..n {
        z.push(flat[3 * k]);
        y.push(flat[3 * k + 1]);
        x.push(flat[3 * k + 2]);
    }
    (z, y, x)
}

fn orientation_of(m: &Manifest) -> i32 {
    m.extra
        .get("detector")
        .and_then(|d| d.get("orientation"))
        .and_then(|v| v.as_i64())
        .unwrap_or(0) as i32
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

fn load_flat(path: &Path) -> Vec<f64> {
    load_npy_f64(path)
        .expect("load npy")
        .as_slice()
        .expect("contiguous")
        .to_vec()
}

#[test]
fn calc_pos_zyx_and_units_bit_exact() {
    let mut checked = 0usize;
    for dir in dataset_dirs() {
        let p1f = dir.join("pixel_p1.npy");
        let posf = dir.join("pos_zyx.npy");
        if !p1f.exists() || !posf.exists() {
            eprintln!(
                "skip {}: regenerate golden (pixel_p1/pos_zyx gitignored)",
                dir.file_name().unwrap().to_string_lossy()
            );
            continue;
        }

        let manifest = load_manifest(dir.join("manifest.json")).expect("manifest");
        let poni = PoniFile::load(dir.join("geometry.poni")).expect("poni");
        let orientation = orientation_of(&manifest);
        let wavelength = poni.wavelength.expect("wavelength in poni");

        // ---- Tier A: calc_pos_zyx bit-exact -----------------------------
        let p1 = load_flat(&p1f);
        let p2 = load_flat(&dir.join("pixel_p2.npy"));
        let p3 = if dir.join("pixel_p3.npy").exists() {
            Some(load_flat(&dir.join("pixel_p3.npy")))
        } else {
            None
        };
        let pos = calc_pos_zyx(
            poni.dist,
            poni.poni1,
            poni.poni2,
            poni.rot1,
            poni.rot2,
            poni.rot3,
            &p1,
            &p2,
            p3.as_deref(),
            orientation,
        );

        let golden_pos = load_flat(&posf);
        let (gz, gy, gx) = split_zyx(&golden_pos);
        let rz = compare_f64(&pos.z, &gz);
        let ry = compare_f64(&pos.y, &gy);
        let rx = compare_f64(&pos.x, &gx);
        eprintln!(
            "{}: calc_pos_zyx z(ulp={} mism={}) y(ulp={} mism={}) x(ulp={} mism={})",
            manifest.dataset,
            rz.max_ulp,
            rz.bit_mismatches,
            ry.max_ulp,
            ry.bit_mismatches,
            rx.max_ulp,
            rx.bit_mismatches,
        );
        // BIT-EXACT GATE. calc_pos_zyx is pure f64 `+ - *` (the six scalar
        // sin/cos of the rotation angles are computed once and shared), so given
        // identical pixel centres it must reproduce pyFAI exactly. The golden is
        // generated from a pyFAI rebuilt with -ffp-contract=off (no FMA fusion),
        // so the Cython evaluates the same bare IEEE-754 expression Rust does and
        // the match is bitwise. See doc/bit-exact-ladder.md.
        for (label, r) in [("z", &rz), ("y", &ry), ("x", &rx)] {
            assert!(
                r.is_bit_exact(),
                "{}: {label} not bit-exact: {r:?}",
                manifest.dataset
            );
        }

        // ---- Tier B: unit equations on golden lab coords ----------------
        // Use golden (z,y,x) to isolate the unit-equation ULP from any transform
        // diff (there is none — Tier A is bit-exact above).
        let unit_name = manifest.config["unit"].as_str().expect("unit");
        if let Some(unit) = unit_for(unit_name) {
            let center = center_array(unit, &gx, &gy, &gz, wavelength);
            let golden_center = load_flat(&dir.join("pos0_center.npy"));
            let r = compare_f64(&center, &golden_center);
            eprintln!(
                "  pos0_center[{unit_name}] max_ulp={} bit_mismatches={}/{}",
                r.max_ulp, r.bit_mismatches, r.total
            );
            // Tier B. `r` is sqrt-only (IEEE-exact); `q`/`2th` go through
            // `atan2`/`sin`. On the golden-generation machine numexpr's libm and
            // Rust's std libm agree bit-for-bit (max_ulp=0), so we assert the
            // strongest true statement — bit-exact — rather than silently
            // budgeting slack. If a future libm diverges, this fails loudly and
            // the divergence is recorded as the manifest ULP budget (the printed
            // max_ulp is the tracked figure). See doc/bit-exact-ladder.md.
            assert!(
                r.is_bit_exact(),
                "{}: pos0_center not bit-exact (libm divergence? record ULP budget): {r:?}",
                manifest.dataset
            );
        }

        // chi_center is present in every config (chi_rad, scale 1).
        let chi = center_array(Unit::CHI_RAD, &gx, &gy, &gz, wavelength);
        let golden_chi = load_flat(&dir.join("chi_center.npy"));
        let rc = compare_f64(&chi, &golden_chi);
        eprintln!(
            "  chi_center max_ulp={} bit_mismatches={}/{}",
            rc.max_ulp, rc.bit_mismatches, rc.total
        );
        // chi = atan2(y, x); same Tier-B libm rationale as pos0_center above.
        assert!(
            rc.is_bit_exact(),
            "{}: chi_center not bit-exact (libm divergence? record ULP budget): {rc:?}",
            manifest.dataset
        );

        checked += 1;
    }

    assert!(
        checked > 0,
        "no golden datasets with pixel_p1/pos_zyx found; run golden/gen_golden.py"
    );
}
