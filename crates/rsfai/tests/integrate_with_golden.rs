//! Corrections-path validation: `AzimuthalIntegrator::integrate1d_with` /
//! `integrate2d_with` with a per-frame [`Corrections`] (user `mask` that
//! *replaces* the detector mask, `dark`, `flat`, an explicit `variance`, and an
//! f64 `normalization_factor`) — the entry `MultiGeometry` drives per geometry.
//!
//! The committed `golden/datasets` only exercise `mask=None`, no dark/flat,
//! `normalization_factor=1.0`, so this is the only gate on the `Corrections`
//! plumbing. Golden by `golden/gen_golden_corrections.py` under
//! `golden/datasets_corrections/`.
//!
//! CSR/LUT configs are asserted **bit-exact** (0-ULP, every field). The
//! direct-split **histogram** configs gate the norm channel
//! (`sum_normalization`/`sum_normalization2`) and its norm-dependents
//! (`intensity`/`sigma`/`std`/`sem`) at `rel ≤ REL_TOL`: pyFAI's direct-split
//! histogram carries the per-pixel norm in f64 (a Cython fused-dispatch quirk),
//! rsFAI carries it stepwise in f32 like pyFAI's own CSR/LUT, so the two diverge
//! ≤1 f32-ULP only when ≥2 normalization factors make the product inexact
//! (here flat=1.05 · monitor=2.7). `sum_signal`/`sum_variance`/`count` use the
//! f32-exact signal channel and stay 0-ULP (single-geometry ⇒ no azimuthal
//! crossed term to pull the norm channel into the variance). See
//! `rsfai-histogram-norm-f64-quirk`.
//!
//! The two error models exercise pyFAI's two preproc variance routes
//! (`_normalize_error_model_variance`, gated by `method.manage_variance`):
//!   * `variance` — explicit per-pixel array used verbatim (no precompute);
//!   * `poisson`  — for a non-`manage_variance` engine (the 2D CSR engine) pyFAI
//!     precomputes `max(data,1)+max(dark,0)` and feeds it as VARIANCE; for a
//!     `manage_variance` engine (the 1D CSR engine) preproc's own poisson
//!     (`max(data,1)`, no dark term) is used. `manage_variance` per (method,dim)
//!     is recorded in the manifest.

use std::path::{Path, PathBuf};

use rsfai::{
    Algo, AzimuthalIntegrator, Corrections, ErrorModelKind, IntegrationOptions, Method, RadialUnit,
    Split,
};
use rsfai_core::compare::{compare_f32, compare_f64};
use rsfai_core::golden::{load_image_f32, load_manifest, load_npy_f32, load_npy_f64, load_npy_i8};

/// Relative tolerance for the direct-split histogram norm channel + its
/// norm-dependents (the f64-vs-f32 norm quirk; ≤1 f32-ULP, rel ~1e-7). CSR/LUT
/// are held bit-exact and never use this.
const REL_TOL: f64 = 1e-6;

fn datasets_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../golden/datasets_corrections")
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

fn split_from(s: &str) -> Split {
    match s {
        "no" => Split::No,
        "bbox" => Split::Bbox,
        "full" => Split::Full,
        other => panic!("corrections_golden: split {other:?} not mapped"),
    }
}

fn algo_from(s: &str) -> Algo {
    match s {
        "histogram" => Algo::Histogram,
        "csr" => Algo::Csr,
        "lut" => Algo::Lut,
        "csc" => Algo::Csc,
        other => panic!("corrections_golden: algo {other:?} not mapped"),
    }
}

fn unit_from_str(s: &str) -> RadialUnit {
    match s {
        "q_nm^-1" => RadialUnit::Q_NM_INV,
        "2th_deg" => RadialUnit::TTH_DEG,
        "r_mm" => RadialUnit::R_MM,
        other => panic!("corrections_golden: radial unit {other:?} not mapped"),
    }
}

/// f32 golden field at `out_<name>.npy` vs `actual`; `exact` selects bit-exact
/// vs relative `≤ REL_TOL`. `None` ⇒ field absent (skip); `Some(ok)` ⇒ checked.
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

fn opt_f32_array(dir: &Path, name: &str) -> Option<Vec<f32>> {
    let p = dir.join(format!("{name}.npy"));
    p.exists()
        .then(|| load_npy_f32(&p).unwrap().iter().copied().collect())
}

#[test]
fn integrate_with_matches_golden() {
    let mut datasets_checked = 0usize;
    let mut total_fail = 0usize;

    for dir in dataset_dirs() {
        let manifest = load_manifest(dir.join("manifest.json")).expect("manifest");
        let cfg = &manifest.config;
        let dim = cfg["dim"].as_i64().unwrap_or(1);
        let unit = unit_from_str(cfg["unit"].as_str().expect("unit"));
        let em = ErrorModelKind::from_code(cfg["error_model_code"].as_i64().unwrap() as i32)
            .expect("error_model_code");
        let method = cfg["method"].as_array().expect("method");
        let split = split_from(method[0].as_str().unwrap());
        let algo = algo_from(method[1].as_str().unwrap());

        eprintln!(
            "=== {} (manage_variance={}) ===",
            manifest.dataset, cfg["manage_variance"]
        );

        // Norm channel (`sum_normalization`/`sum_normalization2` + norm-dependent
        // `intensity`/`sigma`/`std`/`sem`) is bit-exact for CSR/LUT but ≤1 f32-ULP
        // for the direct-split histogram (the f64-norm quirk). The signal channel
        // (`sum_signal`/`sum_variance`/`count`) stays bit-exact: single geometry ⇒
        // no azimuthal crossed term to route the norm channel into the variance.
        let norm_exact = algo != Algo::Histogram;

        let opts = IntegrationOptions {
            correct_solid_angle: cfg["correct_solid_angle"].as_bool().unwrap_or(true),
            polarization_factor: cfg["polarization_factor"].as_f64(),
            // integrate*_with reads the f64 normalization from Corrections; this
            // f32 field is unused on that path.
            normalization_factor: 1.0,
            error_model: em,
            method: Method { split, algo },
            radial_range: None,
            azimuth_range: None,
        };

        let ai = AzimuthalIntegrator::load(dir.join("geometry.poni")).expect("load poni");
        let image = load_image_f32(dir.join("image.npy")).expect("image");
        let dark = opt_f32_array(&dir, "dark");
        let flat = opt_f32_array(&dir, "flat");
        let variance = opt_f32_array(&dir, "variance");
        let user_mask: Vec<i8> = load_npy_i8(dir.join("user_mask.npy"))
            .expect("user_mask")
            .iter()
            .copied()
            .collect();
        let norm = cfg["normalization_factor"].as_f64().expect("norm");

        let corr = Corrections {
            dark: dark.as_deref(),
            flat: flat.as_deref(),
            mask: Some(&user_mask),
            variance: variance.as_deref(),
            normalization_factor: norm,
        };

        let mut results: Vec<Option<bool>> = vec![];
        if dim == 1 {
            let npt = cfg["npt"].as_u64().expect("npt") as usize;
            let res = ai.integrate1d_with(&image, npt, unit, &opts, &corr);
            results.push(cmp_f64(&dir, "radial", &res.radial, true));
            results.push(cmp_f32(&dir, "intensity", &res.intensity, norm_exact));
            results.push(cmp_f32(&dir, "sigma", &res.sigma, norm_exact));
            results.push(cmp_f32(&dir, "std", &res.std, norm_exact));
            results.push(cmp_f32(&dir, "sem", &res.sem, norm_exact));
            results.push(cmp_f64(&dir, "count", &res.count, true));
            results.push(cmp_f64(&dir, "sum_signal", &res.sum_signal, true));
            results.push(cmp_f64(&dir, "sum_variance", &res.sum_variance, true));
            results.push(cmp_f64(
                &dir,
                "sum_normalization",
                &res.sum_normalization,
                norm_exact,
            ));
            results.push(cmp_f64(
                &dir,
                "sum_normalization2",
                &res.sum_normalization2,
                norm_exact,
            ));
        } else {
            let npt_rad = cfg["npt_rad"].as_u64().expect("npt_rad") as usize;
            let npt_azim = cfg["npt_azim"].as_u64().expect("npt_azim") as usize;
            let res = ai.integrate2d_with(&image, npt_rad, npt_azim, unit, &opts, &corr);
            results.push(cmp_f64(&dir, "radial", &res.radial, true));
            results.push(cmp_f64(&dir, "azimuthal", &res.azimuthal, true));
            results.push(cmp_f32(&dir, "intensity", &res.intensity, norm_exact));
            results.push(cmp_f32(&dir, "sigma", &res.sigma, norm_exact));
            results.push(cmp_f64(&dir, "count", &res.count, true));
            results.push(cmp_f64(&dir, "sum_signal", &res.sum_signal, true));
            results.push(cmp_f64(&dir, "sum_variance", &res.sum_variance, true));
            results.push(cmp_f64(
                &dir,
                "sum_normalization",
                &res.sum_normalization,
                norm_exact,
            ));
            results.push(cmp_f64(
                &dir,
                "sum_normalization2",
                &res.sum_normalization2,
                norm_exact,
            ));
            results.push(cmp_f32(&dir, "std", &res.std, norm_exact));
            results.push(cmp_f32(&dir, "sem", &res.sem, norm_exact));
        }

        let checked = results.iter().filter(|r| r.is_some()).count();
        let failed = results.iter().filter(|r| **r == Some(false)).count();
        total_fail += failed;
        datasets_checked += 1;
        eprintln!("    ({checked} fields checked, {failed} failed)\n");
    }

    assert!(
        datasets_checked > 0,
        "no corrections golden datasets found; run golden/gen_golden_corrections.py"
    );
    assert_eq!(
        total_fail, 0,
        "{total_fail} field(s) diverged from golden — see the per-field report above"
    );
}
