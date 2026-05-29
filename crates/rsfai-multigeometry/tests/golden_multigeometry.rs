//! End-to-end validation of [`MultiGeometry`] vs pyFAI's `multi_geometry.py`.
//!
//! Golden by `golden/gen_golden_multigeometry.py` under
//! `golden/datasets_multigeometry/`: 3 Pilatus1M geometries, 3 frames, a shared
//! user mask + flat (broadcast to every geometry), per-geometry monitors, over
//! the matrix {1D,2D} × {q,2θ,r} × error_model {none,poisson,azimuthal} ×
//! `correctSolidAngle` {on,off} × method {full-histogram, bbox-csr}. The shared
//! inputs live once in `…/inputs/`; each config dir holds only its `out_*.npy` +
//! `manifest.json`.
//!
//! Both methods are serial cython engines (full-split histogram + bbox CSR) and
//! the `union` left-fold is serial f64. The gate is **bbox-CSR = fully bit-exact**
//! (0-ULP, every field); the full-split **histogram** norm channel is a tolerance
//! path for one reason documented in `rsfai-histogram-norm-f64-quirk`:
//!
//! * pyFAI's direct-split histogram engines carry the per-pixel `norm =
//!   normalization_factor·flat·polarization·solidangle·absorption` in **f64** (a
//!   Cython fused-dispatch quirk), while rsFAI — like pyFAI's own CSR/LUT — carries
//!   it stepwise in **f32**. So `sum_normalization`/`sum_normalization2` and their
//!   norm-dependents (`intensity`, `sigma`, `std`, `sem`) diverge ≤1 f32-ULP
//!   (`rel ≤ REL_TOL`) for the histogram method only. `sum_signal`/`count` use the
//!   f32-exact signal channel and stay **0-ULP** even for histogram. `sum_variance`
//!   is 0-ULP too, except the **1D azimuthal** crossed term (containers.py:387)
//!   pulls the f64-quirk norm channel into the variance — so 1D-azimuthal histogram
//!   variance is `rel ≤ REL_TOL`; 2D drops the crossed term (so 0-ULP), and CSR is
//!   0-ULP on every field.
//!
//! Before integrating, each config also cross-checks rsFAI's guessed common range
//! (`effective_{radial,azimuth}_range`) vs pyFAI's recorded
//! `radial_range_guessed`/`azimuth_range_guessed`. The guess is `min`/`max` over
//! transcendental arrays (`atan2`-based CHI_DEG azimuth, the radial unit), so it is
//! held to a small ULP budget ([`RANGE_ULP_BUDGET`]) rather than bit-exact — the
//! observed divergence is ≤1 ULP on `azimuth.hi`, and the binning it drives is
//! 0-ULP (`count`/`azimuthal` exact).

use std::path::{Path, PathBuf};

use rsfai::{Algo, AzimuthalIntegrator, ErrorModelKind, Method, RadialUnit, Split};
use rsfai_core::compare::{compare_f32, compare_f64};
use rsfai_core::golden::{load_image_f32, load_manifest, load_npy_f32, load_npy_f64, load_npy_i8};
use rsfai_multigeometry::{GeometryFrame, MultiGeometry, MultiIntegrationOptions};

/// Relative tolerance for the direct-split histogram **norm channel** (and its
/// norm-dependents) — the f64-vs-f32 norm quirk (≤1 f32-ULP, rel ~1e-7). CSR is
/// held bit-exact and never uses this.
const REL_TOL: f64 = 1e-6;

/// ULP budget for the cross-geometry range guess (transcendental `min`/`max` over
/// `atan2`-based CHI_DEG azimuth + the radial unit array). Observed: ≤1 ULP on
/// `azimuth.hi`; 2 gives libm headroom without admitting a real binning shift.
const RANGE_ULP_BUDGET: u64 = 2;

fn datasets_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../golden/datasets_multigeometry")
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
        other => panic!("golden_multigeometry: split {other:?} not mapped"),
    }
}

fn algo_from(s: &str) -> Algo {
    match s {
        "histogram" => Algo::Histogram,
        "csr" => Algo::Csr,
        "lut" => Algo::Lut,
        "csc" => Algo::Csc,
        other => panic!("golden_multigeometry: algo {other:?} not mapped"),
    }
}

fn unit_from_str(s: &str) -> RadialUnit {
    match s {
        "q_nm^-1" => RadialUnit::Q_NM_INV,
        "2th_deg" => RadialUnit::TTH_DEG,
        "r_mm" => RadialUnit::R_MM,
        other => panic!("golden_multigeometry: radial unit {other:?} not mapped"),
    }
}

/// f32 golden field at `out_<name>.npy` vs `actual`; `exact` selects bit-exact vs
/// relative `<= REL_TOL`. `None` ⇒ field absent (e.g. sem/std for error_model
/// none), so skipped.
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

/// The shared inputs (loaded once, reused by every config).
struct Inputs {
    ais: Vec<AzimuthalIntegrator>,
    images: Vec<Vec<f32>>,
    mask: Vec<i8>,
    flat: Vec<f32>,
    monitors: Vec<f64>,
}

fn load_inputs() -> Inputs {
    let root = datasets_root().join("inputs");
    let text = std::fs::read_to_string(root.join("inputs.json")).expect("inputs.json");
    let meta: serde_json::Value = serde_json::from_str(&text).expect("parse inputs.json");
    let n = meta["n_geometry"].as_u64().expect("n_geometry") as usize;

    let ais = (0..n)
        .map(|i| {
            let poni = meta["ponis"][i].as_str().expect("poni name");
            AzimuthalIntegrator::load(root.join(poni)).expect("load poni")
        })
        .collect();
    let images = (0..n)
        .map(|i| {
            let img = meta["images"][i].as_str().expect("image name");
            load_image_f32(root.join(img)).expect("image")
        })
        .collect();
    let mask: Vec<i8> = load_npy_i8(root.join(meta["user_mask"].as_str().unwrap()))
        .expect("user_mask")
        .iter()
        .copied()
        .collect();
    let flat: Vec<f32> = load_npy_f32(root.join(meta["flat"].as_str().unwrap()))
        .expect("flat")
        .iter()
        .copied()
        .collect();
    let monitors = meta["monitors"]
        .as_array()
        .expect("monitors")
        .iter()
        .map(|v| v.as_f64().unwrap())
        .collect();
    Inputs {
        ais,
        images,
        mask,
        flat,
        monitors,
    }
}

/// IEEE-754 total-order key: adjacent representable f64s map to adjacent `u64`s
/// across the sign boundary, so a plain difference is the ULP distance.
fn ordered(x: f64) -> u64 {
    let b = x.to_bits();
    if b & 0x8000_0000_0000_0000 != 0 {
        !b
    } else {
        b | 0x8000_0000_0000_0000
    }
}

fn f64_ulp_diff(a: f64, b: f64) -> u64 {
    let (x, y) = (ordered(a), ordered(b));
    x.max(y) - x.min(y)
}

/// Scalar cross-check held to a ULP budget (the range guess is transcendental, see
/// [`RANGE_ULP_BUDGET`]), with a per-field log line reporting the actual ULP gap.
fn cmp_scalar_ulp(label: &str, actual: f64, golden: f64, max_ulp: u64) -> bool {
    let ulp = f64_ulp_diff(actual, golden);
    let ok = ulp <= max_ulp;
    eprintln!(
        "    {label:24} {}  actual={actual:.17e} golden={golden:.17e} ulp={ulp} (budget {max_ulp})",
        if ok { "PASS" } else { "FAIL" }
    );
    ok
}

#[test]
fn multigeometry_matches_golden() {
    let inputs = load_inputs();
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
        let csa = cfg["correct_solid_angle"].as_bool().unwrap_or(true);

        eprintln!("=== {} ===", manifest.dataset);

        // Exactness gates (see module doc):
        // * signal channel (`sum_signal`/`count`): f32-exact for every serial
        //   engine; only the no-split histogram (parallel atomics) is a tolerance
        //   path — not used by MG, but kept uniform.
        // * norm channel (`sum_normalization`/`sum_normalization2` + the
        //   norm-dependent `intensity`/`sigma`/`std`/`sem`): bit-exact for CSR/LUT,
        //   but the direct-split histogram carries norm in f64 (the quirk) ⇒ ≤1
        //   f32-ULP, so any histogram method is `rel ≤ REL_TOL` here.
        let signal_exact = !(split == Split::No && algo == Algo::Histogram);
        let norm_exact = algo != Algo::Histogram;
        // `sum_variance`: the AZIMUTHAL crossed term (containers.py:387) multiplies
        // the `sum_normalization` channel into the variance, and the MG union applies
        // it only in 1D (pyFAI's `integrate2d_ng` leaves `error_model = None`, so 2D
        // drops it). So 1D-azimuthal variance flows through the norm channel ⇒ ≤1
        // f32-ULP for histogram; everywhere else (2D, or non-azimuthal) variance is
        // the signal-channel accumulation and stays bit-exact.
        let crossed_applied = dim == 1 && em == ErrorModelKind::Azimuthal;
        let variance_exact = if crossed_applied {
            norm_exact
        } else {
            signal_exact
        };

        let mg = MultiGeometry::new(inputs.ais.clone(), unit);

        let mut results: Vec<Option<bool>> = vec![];

        // Cross-check rsFAI's guessed common range vs pyFAI's recorded guess.
        let (rlo, rhi) = mg.effective_radial_range();
        let gr = cfg["radial_range_guessed"]
            .as_array()
            .expect("radial_range_guessed");
        results.push(Some(cmp_scalar_ulp(
            "radial_range.lo",
            rlo,
            gr[0].as_f64().unwrap(),
            RANGE_ULP_BUDGET,
        )));
        results.push(Some(cmp_scalar_ulp(
            "radial_range.hi",
            rhi,
            gr[1].as_f64().unwrap(),
            RANGE_ULP_BUDGET,
        )));
        let (alo, ahi) = mg.effective_azimuth_range();
        let ga = cfg["azimuth_range_guessed"]
            .as_array()
            .expect("azimuth_range_guessed");
        results.push(Some(cmp_scalar_ulp(
            "azimuth_range.lo",
            alo,
            ga[0].as_f64().unwrap(),
            RANGE_ULP_BUDGET,
        )));
        results.push(Some(cmp_scalar_ulp(
            "azimuth_range.hi",
            ahi,
            ga[1].as_f64().unwrap(),
            RANGE_ULP_BUDGET,
        )));

        let frames: Vec<GeometryFrame> = (0..inputs.ais.len())
            .map(|i| GeometryFrame {
                data: &inputs.images[i],
                variance: None,
                mask: Some(&inputs.mask),
                flat: Some(&inputs.flat),
                monitor: inputs.monitors[i],
            })
            .collect();
        let opts = MultiIntegrationOptions {
            correct_solid_angle: csa,
            error_model: em,
            polarization_factor: None,
            method: Method { split, algo },
        };

        if dim == 1 {
            let npt = cfg["npt"].as_u64().expect("npt") as usize;
            let res = mg.integrate1d(&frames, npt, &opts);
            results.push(cmp_f64(&dir, "radial", &res.radial, true));
            results.push(cmp_f32(&dir, "intensity", &res.intensity, norm_exact));
            results.push(cmp_f32(&dir, "sigma", &res.sigma, norm_exact));
            results.push(cmp_f32(&dir, "std", &res.std, norm_exact));
            results.push(cmp_f32(&dir, "sem", &res.sem, norm_exact));
            results.push(cmp_f64(&dir, "count", &res.count, signal_exact));
            results.push(cmp_f64(&dir, "sum_signal", &res.sum_signal, signal_exact));
            results.push(cmp_f64(
                &dir,
                "sum_variance",
                &res.sum_variance,
                variance_exact,
            ));
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
            let res = mg.integrate2d(&frames, npt_rad, npt_azim, &opts);
            results.push(cmp_f64(&dir, "radial", &res.radial, true));
            results.push(cmp_f64(&dir, "azimuthal", &res.azimuthal, true));
            results.push(cmp_f32(&dir, "intensity", &res.intensity, norm_exact));
            results.push(cmp_f32(&dir, "sigma", &res.sigma, norm_exact));
            results.push(cmp_f64(&dir, "count", &res.count, signal_exact));
            results.push(cmp_f64(&dir, "sum_signal", &res.sum_signal, signal_exact));
            results.push(cmp_f64(
                &dir,
                "sum_variance",
                &res.sum_variance,
                variance_exact,
            ));
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
        "no multigeometry golden datasets found; run golden/gen_golden_multigeometry.py"
    );
    assert_eq!(
        total_fail, 0,
        "{total_fail} field(s) diverged from golden — see the per-field report above"
    );
}
