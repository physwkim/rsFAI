//! End-to-end drop-in validation: `AzimuthalIntegrator::load(poni).integrate1d/
//! 2d(image, …)` — PONI + detector image in, **nothing else** — must reproduce
//! the committed golden curve, for every 1D and 2D method tuple (no/bbox/full ×
//! histogram/csr/lut/csc). The 2D `pseudo` split is not ported, so its dataset
//! is skipped (`split_from` returns `None`).
//!
//! This is the Tier-C integration test for the orchestrator. Unlike the
//! per-kernel golden tests (which feed dumped intermediates), here the
//! integrator regenerates the pixel positions, corrections, gap mask, dummy,
//! corner/delta geometry, and preproc rows itself from the `.poni` and the
//! image, then bins them. A match therefore exercises the whole chain
//! (`centers_f64 → calc_pos_zyx → center_array`/`corner_array`/`delta_array`/
//! `solid_angle`/`polarization` → `get_dummies`/`calc_mask` → `preproc4` →
//! `histogram`/`csr`/`lut`/`csc`) composed correctly. Fields the dataset does
//! not expose (e.g. the variance family under the `no` error model) are skipped.
//!
//! Gate by engine determinism: the **no-split histogram** accumulation is
//! parallelized (non-deterministic f64 add order), so its accumulator-derived
//! fields are validated at relative error `<= REL_TOL` (1e-6). The sparse
//! (`csr`/`lut`/`csc`) and split-histogram engines run **serially** in
//! pixel-index order, so their entire output — including the f64 `sum_*`/`count`
//! accumulators — is asserted **bit-exact**. The bin-center axes
//! (`radial`/`azimuthal`), order-independent min/max + `linspace`, are always
//! bit-exact. See `doc/bit-exact-ladder.md`.

use std::path::{Path, PathBuf};

use rsfai::{
    Algo, AzimuthalIntegrator, ErrorModelKind, IntegrationOptions, Method, RadialUnit, Split,
};
use rsfai_core::compare::{compare_f32, compare_f64};
use rsfai_core::golden::{load_image_f32, load_manifest, load_npy_f32, load_npy_f64};

/// Relative-error gate for the parallel-histogram accumulator fields.
const REL_TOL: f64 = 1e-6;

fn split_from(s: &str) -> Option<Split> {
    Some(match s {
        "no" => Split::No,
        "bbox" => Split::Bbox,
        "full" => Split::Full,
        _ => return None, // "pseudo" is 2D-only; not driven here.
    })
}

fn algo_from(s: &str) -> Option<Algo> {
    Some(match s {
        "histogram" => Algo::Histogram,
        "csr" => Algo::Csr,
        "lut" => Algo::Lut,
        "csc" => Algo::Csc,
        _ => return None,
    })
}

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
    ErrorModelKind::from_code(code as i32)
        .unwrap_or_else(|| panic!("dropin_golden: error_model_code {code} not mapped"))
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
fn dropin_matches_golden() {
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
        let (Some(split), Some(algo)) = (
            method.first().and_then(|s| split_from(s)),
            method.get(1).and_then(|s| algo_from(s)),
        ) else {
            continue; // unmapped method (e.g. 2D "pseudo")
        };
        let dim = cfg["dim"].as_i64().unwrap_or(1);

        eprintln!("=== {} ===", manifest.dataset);

        let unit = unit_from_str(cfg["unit"].as_str().expect("unit"));
        let opts = IntegrationOptions {
            correct_solid_angle: cfg["correct_solid_angle"].as_bool().unwrap_or(true),
            polarization_factor: cfg["polarization_factor"].as_f64(),
            normalization_factor: cfg["normalization_factor"].as_f64().unwrap_or(1.0) as f32,
            error_model: error_model_from_code(cfg["error_model_code"].as_i64().unwrap_or(0)),
            method: Method { split, algo },
        };

        let ai =
            AzimuthalIntegrator::load(dir.join("geometry.poni")).expect("load poni + detector");

        // Image -> f32, exactly the cast pyFAI applies before preproc. Pilatus
        // frames are int32, Eiger4M float32; the single owner reads the `.npy`
        // dtype header and handles both.
        let image = load_image_f32(dir.join("image.npy")).expect("image");

        // The no-split histogram accumulation is parallel (tolerance gate); every
        // other engine is serial (bit-exact). Axes are always bit-exact.
        let is_no_hist = split == Split::No && algo == Algo::Histogram;
        let acc_exact = !is_no_hist;
        let mut results: Vec<Option<bool>> = vec![];

        if dim == 1 {
            let npt = cfg["npt"].as_u64().expect("npt") as usize;
            let res = ai.integrate1d(&image, npt, unit, &opts);
            results.push(cmp_f64(&dir, "radial", &res.radial, true));
            results.push(cmp_f32(&dir, "intensity", &res.intensity, acc_exact));
            results.push(cmp_f32(&dir, "sigma", &res.sigma, acc_exact));
            results.push(cmp_f32(&dir, "std", &res.std, acc_exact));
            results.push(cmp_f32(&dir, "sem", &res.sem, acc_exact));
            // The accumulators are f64 in the result; the no-split histogram
            // golden stores them as f32, so compare that path downcast.
            let mut cmp_acc = |name: &str, v: &[f64]| {
                let r = if is_no_hist {
                    let v32: Vec<f32> = v.iter().map(|&x| x as f32).collect();
                    cmp_f32(&dir, name, &v32, false)
                } else {
                    cmp_f64(&dir, name, v, true)
                };
                results.push(r);
            };
            cmp_acc("count", &res.count);
            cmp_acc("sum_signal", &res.sum_signal);
            cmp_acc("sum_variance", &res.sum_variance);
            cmp_acc("sum_normalization", &res.sum_normalization);
            cmp_acc("sum_normalization2", &res.sum_normalization2);
        } else {
            let npt_rad = cfg["npt_rad"].as_u64().expect("npt_rad") as usize;
            let npt_azim = cfg["npt_azim"].as_u64().expect("npt_azim") as usize;
            let res = ai.integrate2d(&image, npt_rad, npt_azim, unit, &opts);
            // 2D accumulators are full-precision f64 in every engine. Axes always
            // bit-exact; accumulator-derived fields bit-exact for the serial
            // engines (sparse + split-histogram), relative <= REL_TOL for the
            // parallel no-split histogram (`acc_exact`).
            results.push(cmp_f64(&dir, "radial", &res.radial, true));
            results.push(cmp_f64(&dir, "azimuthal", &res.azimuthal, true));
            results.push(cmp_f32(&dir, "intensity", &res.intensity, acc_exact));
            results.push(cmp_f32(&dir, "sigma", &res.sigma, acc_exact));
            results.push(cmp_f64(&dir, "count", &res.count, acc_exact));
            results.push(cmp_f64(&dir, "sum_signal", &res.sum_signal, acc_exact));
            results.push(cmp_f64(&dir, "sum_variance", &res.sum_variance, acc_exact));
            results.push(cmp_f64(
                &dir,
                "sum_normalization",
                &res.sum_normalization,
                acc_exact,
            ));
            results.push(cmp_f64(
                &dir,
                "sum_normalization2",
                &res.sum_normalization2,
                acc_exact,
            ));
            results.push(cmp_f32(&dir, "std", &res.std, acc_exact));
            results.push(cmp_f32(&dir, "sem", &res.sem, acc_exact));
        }

        let checked = results.iter().filter(|r| r.is_some()).count();
        let failed = results.iter().filter(|r| **r == Some(false)).count();
        total_fail += failed;
        datasets_checked += 1;
        eprintln!("    ({checked} fields checked, {failed} failed)\n");
    }

    assert!(
        datasets_checked > 0,
        "no golden datasets found; run golden/gen_golden.py"
    );
    assert_eq!(
        total_fail, 0,
        "{total_fail} field(s) diverged from golden — see the per-field report above"
    );
}
