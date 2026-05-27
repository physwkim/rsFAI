//! End-to-end drop-in validation: `AzimuthalIntegrator::load(poni).integrate1d/
//! 2d(image, …)` — PONI + detector image in, **nothing else** — must reproduce
//! the committed golden curve bit-for-bit, for every no-split histogram dataset.
//!
//! This is the Tier-C integration test for the orchestrator. Unlike the
//! per-kernel golden tests (which feed dumped intermediates), here the
//! integrator regenerates the pixel positions, corrections, gap mask, dummy,
//! and preproc rows itself from the `.poni` and the image, then bins them. A
//! bit-exact match therefore exercises the whole chain
//! (`centers_f64 → calc_pos_zyx → center_array`/`solid_angle`/`polarization` →
//! `get_dummies`/`calc_mask` → `preproc4` → `histogram1d`/`2d`) composed
//! correctly. Fields the dataset does not expose (e.g. the variance family
//! under the `no` error model) are skipped.
//!
//! The histogram **accumulation** is parallelized (non-deterministic f64 add
//! order), so the accumulator-derived fields are validated at relative error
//! `<= REL_TOL` (1e-6), while the bin-center **axes** (`radial`/`azimuthal`),
//! which derive from the order-independent min/max + `linspace`, stay
//! **bit-exact**. See `doc/bit-exact-ladder.md`.

use std::path::{Path, PathBuf};

use rsfai::{AzimuthalIntegrator, ErrorModelKind, IntegrationOptions, RadialUnit};
use rsfai_core::compare::{compare_f32, compare_f64};
use rsfai_core::golden::{load_manifest, load_npy_f32, load_npy_f64, load_npy_i32};

/// Relative-error gate for the parallel-histogram accumulator fields.
const REL_TOL: f64 = 1e-6;

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

fn unit_from_str(s: &str) -> RadialUnit {
    match s {
        "q_nm^-1" => RadialUnit::Q_NM_INV,
        "q_A^-1" => RadialUnit::Q_A_INV,
        "2th_deg" => RadialUnit::TTH_DEG,
        "2th_rad" => RadialUnit::TTH_RAD,
        "r_mm" => RadialUnit::R_MM,
        "r_m" => RadialUnit::R_M,
        other => panic!("dropin_golden: radial unit {other:?} not mapped"),
    }
}

fn error_model_from_code(code: i64) -> ErrorModelKind {
    match code {
        0 => ErrorModelKind::No,
        1 => ErrorModelKind::Variance,
        2 => ErrorModelKind::Poisson,
        3 => ErrorModelKind::Azimuthal,
        other => panic!("dropin_golden: error_model_code {other} not mapped"),
    }
}

/// f32 golden field at `out_<name>.npy` vs `actual`. `exact` selects the gate:
/// bit-exact for the order-independent bin-center axes, relative `<= REL_TOL`
/// for the parallel-histogram accumulator fields. `None` ⇒ the dataset does not
/// expose this field (skip); `Some(ok)` ⇒ checked.
fn cmp_f32(dir: &Path, name: &str, actual: &[f32], exact: bool) -> Option<bool> {
    let p = dir.join(format!("out_{name}.npy"));
    if !p.exists() {
        return None;
    }
    let g = load_npy_f32(&p).unwrap();
    let g = g.as_slice().expect("golden C-contiguous");
    let r = compare_f32(actual, g);
    let ok = if exact {
        r.is_bit_exact()
    } else {
        r.within_rel(REL_TOL)
    };
    eprintln!(
        "    out_{name:22} {}  {}  max_ulp={} max_rel={:e} mismatches={}/{}",
        if ok { "PASS" } else { "FAIL" },
        if exact { "exact" } else { " rel " },
        r.max_ulp,
        r.max_rel_diff,
        r.bit_mismatches,
        r.total
    );
    Some(ok)
}

/// f64 golden field at `out_<name>.npy` vs `actual`; `exact` as in [`cmp_f32`].
fn cmp_f64(dir: &Path, name: &str, actual: &[f64], exact: bool) -> Option<bool> {
    let p = dir.join(format!("out_{name}.npy"));
    if !p.exists() {
        return None;
    }
    let g = load_npy_f64(&p).unwrap();
    let g = g.as_slice().expect("golden C-contiguous");
    let r = compare_f64(actual, g);
    let ok = if exact {
        r.is_bit_exact()
    } else {
        r.within_rel(REL_TOL)
    };
    eprintln!(
        "    out_{name:22} {}  {}  max_ulp={} max_rel={:e} mismatches={}/{}",
        if ok { "PASS" } else { "FAIL" },
        if exact { "exact" } else { " rel " },
        r.max_ulp,
        r.max_rel_diff,
        r.bit_mismatches,
        r.total
    );
    Some(ok)
}

#[test]
fn histogram_dropin_within_tolerance() {
    let mut datasets_checked = 0usize;
    let mut total_fail = 0usize;

    for dir in dataset_dirs() {
        let manifest = load_manifest(dir.join("manifest.json")).expect("manifest");
        let cfg = &manifest.config;
        let method: Vec<String> = cfg["method"]
            .as_array()
            .expect("method")
            .iter()
            .map(|v| v.as_str().unwrap().to_string())
            .collect();
        // Stage A covers the no-split histogram path only.
        if method.get(1).map(String::as_str) != Some("histogram") {
            continue;
        }

        eprintln!("=== {} ===", manifest.dataset);

        let unit = unit_from_str(cfg["unit"].as_str().expect("unit"));
        let opts = IntegrationOptions {
            correct_solid_angle: cfg["correct_solid_angle"].as_bool().unwrap_or(true),
            polarization_factor: cfg["polarization_factor"].as_f64(),
            normalization_factor: cfg["normalization_factor"].as_f64().unwrap_or(1.0) as f32,
            error_model: error_model_from_code(cfg["error_model_code"].as_i64().unwrap_or(0)),
        };

        let ai =
            AzimuthalIntegrator::load(dir.join("geometry.poni")).expect("load poni + detector");

        // Image (int32) -> f32, exactly the cast pyFAI applies before preproc.
        let image_i32 = load_npy_i32(dir.join("image.npy")).expect("image");
        let image: Vec<f32> = image_i32.iter().map(|&v| v as f32).collect();

        let dim = cfg["dim"].as_i64().unwrap_or(1);
        let mut results: Vec<Option<bool>> = vec![];

        if dim == 1 {
            let npt = cfg["npt"].as_u64().expect("npt") as usize;
            let res = ai.integrate1d(&image, npt, unit, &opts);
            // Axis: bit-exact. Accumulator fields: relative <= REL_TOL.
            results.push(cmp_f64(&dir, "radial", &res.radial, true));
            results.push(cmp_f32(&dir, "intensity", &res.intensity, false));
            results.push(cmp_f32(&dir, "sigma", &res.sigma, false));
            results.push(cmp_f32(&dir, "count", &res.count, false));
            results.push(cmp_f32(&dir, "sum_signal", &res.sum_signal, false));
            results.push(cmp_f32(&dir, "sum_variance", &res.sum_variance, false));
            results.push(cmp_f32(
                &dir,
                "sum_normalization",
                &res.sum_normalization,
                false,
            ));
            results.push(cmp_f32(
                &dir,
                "sum_normalization2",
                &res.sum_normalization2,
                false,
            ));
            results.push(cmp_f32(&dir, "std", &res.std, false));
            results.push(cmp_f32(&dir, "sem", &res.sem, false));
        } else {
            let npt_rad = cfg["npt_rad"].as_u64().expect("npt_rad") as usize;
            let npt_azim = cfg["npt_azim"].as_u64().expect("npt_azim") as usize;
            let res = ai.integrate2d(&image, npt_rad, npt_azim, unit, &opts);
            // 2D: radial/azimuthal/intensity/sigma/std/sem are f32-or-f64 per
            // the engine; sums and count are full-precision f64. Axes bit-exact;
            // accumulator fields relative <= REL_TOL.
            results.push(cmp_f64(&dir, "radial", &res.radial, true));
            results.push(cmp_f64(&dir, "azimuthal", &res.azimuthal, true));
            results.push(cmp_f32(&dir, "intensity", &res.intensity, false));
            results.push(cmp_f32(&dir, "sigma", &res.sigma, false));
            results.push(cmp_f64(&dir, "count", &res.count, false));
            results.push(cmp_f64(&dir, "sum_signal", &res.sum_signal, false));
            results.push(cmp_f64(&dir, "sum_variance", &res.sum_variance, false));
            results.push(cmp_f64(
                &dir,
                "sum_normalization",
                &res.sum_normalization,
                false,
            ));
            results.push(cmp_f64(
                &dir,
                "sum_normalization2",
                &res.sum_normalization2,
                false,
            ));
            results.push(cmp_f32(&dir, "std", &res.std, false));
            results.push(cmp_f32(&dir, "sem", &res.sem, false));
        }

        let checked = results.iter().filter(|r| r.is_some()).count();
        let failed = results.iter().filter(|r| **r == Some(false)).count();
        total_fail += failed;
        datasets_checked += 1;
        eprintln!("    ({checked} fields checked, {failed} failed)\n");
    }

    assert!(
        datasets_checked > 0,
        "no histogram golden datasets found; run golden/gen_golden.py"
    );
    assert_eq!(
        total_fail, 0,
        "{total_fail} field(s) diverged from golden — see the per-field report above"
    );
}
