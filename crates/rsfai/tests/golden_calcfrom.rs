//! Golden verification for image reconstruction: `calcfrom1d`, `fake_xrpdp`,
//! `fake_calibration_image`. Golden produced by
//! `golden/gen_golden_calcfrom.py` (pyFAI 2026.5.0, single-thread).
//!
//! Parity ledger (decomposed so no divergence hides behind a tolerance):
//!
//!   1. `ttha` — the per-pixel internal 2θ array `calcfrom1d` interpolates onto
//!      (arctan2/sqrt over pixel positions). BIT-EXACT vs pyFAI's `center_array`.
//!   2. `numpy_interp` driven on pyFAI's OWN `ttha` — the pure-f64 interpolation.
//!      BIT-EXACT: numpy's `compiled_base.c` contracts `slope*(x-xp)+fp` to a
//!      hardware FMA, which the port matches via `f64::mul_add` (separate
//!      rounded mul+add diverges by 1 ULP on ~2% of interior points). Isolating
//!      the interp on identical inputs proves item 3's parity is not luck.
//!   3. `calcfrom1d` end-to-end (4 variants: solid-angle on/off, masked,
//!      flat+dark) — BIT-EXACT: bit-exact `ttha` (1) + bit-exact FMA interp (2)
//!      + f64 solid-angle / flat / dark / mask, none of which numpy fuses.
//!   4. `fake_xrpdp` radial axis (a `linspace`): BIT-EXACT. Its intensity column
//!      (numexpr Gaussian `exp` + Bragg-2θ `asin`) and `fake_calibration_image`
//!      are Tier-B by physics — transcendental, so gated on a ULP budget (both
//!      measure 0 ULP for this data, but `exp`/`asin` carry no IEEE guarantee).

use std::path::PathBuf;

use rsfai::{fake_calibration_image, numpy_interp, AzimuthalIntegrator, Calcfrom1dOptions};
use rsfai_calibrant::Calibrant;
use rsfai_core::compare::compare_f64;
use rsfai_core::golden::{load_npy_f64, load_npy_i8};
use rsfai_detectors::Detector;
use rsfai_geometry::{unscaled_center_array, Unit};

/// Tier-B ULP budget for the transcendental `fake_xrpdp` Gaussian (its numexpr
/// `exp` and Bragg-2θ `asin`) and `fake_calibration_image`, which inherits it.
/// Both measure 0 ULP for this data; 8 leaves headroom for libm `exp`/`asin`
/// variation without admitting an algebra bug (which would be far larger).
const TIER_B_ULP: u64 = 8;

#[derive(Clone, Copy)]
enum Gate {
    Exact,
    Ulp,
}

fn datasets_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../golden/datasets_calcfrom")
}

/// The generic 128×128 detector + geometry the generator uses (orientation 3 is
/// pyFAI's `Detector` default). Mirrors `build_ai` in `gen_golden_calcfrom.py`.
fn build_ai() -> AzimuthalIntegrator {
    let detector = Detector {
        name: "Detector",
        pixel1: 100e-6,
        pixel2: 100e-6,
        shape: (128, 128),
        orientation: 3,
        module_size: None,
        module_gap: None,
        dummy: None,
        delta_dummy: None,
    };
    AzimuthalIntegrator {
        detector,
        dist: 0.02,
        poni1: 6.4e-3,
        poni2: 6.4e-3,
        rot1: 0.0,
        rot2: 0.0,
        rot3: 0.0,
        wavelength: 1e-10,
    }
}

fn load_f64(name: &str) -> Vec<f64> {
    let p = datasets_root().join(name);
    load_npy_f64(&p)
        .unwrap_or_else(|e| panic!("load {name}: {e}"))
        .as_slice()
        .expect("C-contiguous")
        .to_vec()
}

fn load_i8(name: &str) -> Vec<i8> {
    let p = datasets_root().join(name);
    load_npy_i8(&p)
        .unwrap_or_else(|e| panic!("load {name}: {e}"))
        .as_slice()
        .expect("C-contiguous")
        .to_vec()
}

/// Compare `actual` to the golden `file` under `gate`; record a failure if it
/// misses, and always print the measured ULP / relative divergence.
fn check(name: &str, actual: &[f64], file: &str, gate: Gate, fails: &mut usize) {
    let g = load_f64(file);
    let r = compare_f64(actual, &g);
    let (label, ok) = match gate {
        Gate::Exact => ("exact", r.is_bit_exact()),
        Gate::Ulp => (" ulp ", r.within_ulp(TIER_B_ULP)),
    };
    eprintln!(
        "  {name:30} {}  {label}  max_ulp={} max_rel={:e} mism={}/{}",
        if ok { "PASS" } else { "FAIL" },
        r.max_ulp,
        r.max_rel_diff,
        r.bit_mismatches,
        r.total
    );
    if !ok {
        *fails += 1;
    }
}

#[test]
fn calcfrom_matches_pyfai_golden() {
    let ai = build_ai();
    let mut fails = 0usize;

    let tth = load_f64("calcfrom1d__tth.npy");
    let intensity = load_f64("calcfrom1d__intensity.npy");
    let mask = load_i8("calcfrom1d__mask.npy");
    let flat = load_f64("calcfrom1d__flat.npy");
    let dark = load_f64("calcfrom1d__dark.npy");
    let u = Unit::TTH_DEG;

    // ---- 1. geometry array `ttha` (bit-exact) ------------------------------
    eprintln!("=== ttha (bit-exact 2θ geometry array) ===");
    let pos = ai.pixel_positions();
    let ttha_rs = unscaled_center_array(u.space, &pos.x, &pos.y, &pos.z, ai.wavelength);
    check(
        "ttha",
        &ttha_rs,
        "calcfrom1d__ttha.npy",
        Gate::Exact,
        &mut fails,
    );

    // ---- 2. numpy_interp on pyFAI's OWN ttha (bit-exact: isolates the algebra)
    // Identical x/xp/fp ⇒ the FMA-contracted interp must agree bitwise, so item
    // 3's parity is wholly the algebra, not an accidental geometry cancellation.
    eprintln!("=== numpy_interp on pyFAI ttha (bit-exact: isolates the FMA interp) ===");
    let ttha_py = load_f64("calcfrom1d__ttha.npy");
    let tth_internal: Vec<f64> = tth.iter().map(|&t| t / u.scale).collect();
    let interp_rs: Vec<f64> = ttha_py
        .iter()
        .map(|&q| numpy_interp(q, &tth_internal, &intensity))
        .collect();
    check(
        "interp_on_pyfai_ttha",
        &interp_rs,
        "calcfrom1d__interp_pyttha.npy",
        Gate::Exact,
        &mut fails,
    );

    // ---- 3. calcfrom1d end-to-end (bit-exact) ------------------------------
    eprintln!("=== calcfrom1d end-to-end (bit-exact) ===");
    let img = ai.calcfrom1d(
        &tth,
        &intensity,
        u,
        Calcfrom1dOptions {
            correct_solid_angle: true,
            ..Default::default()
        },
    );
    check(
        "calcfrom1d/img_sa",
        &img,
        "calcfrom1d__img_sa.npy",
        Gate::Exact,
        &mut fails,
    );
    let img = ai.calcfrom1d(&tth, &intensity, u, Calcfrom1dOptions::default());
    check(
        "calcfrom1d/img_nosa",
        &img,
        "calcfrom1d__img_nosa.npy",
        Gate::Exact,
        &mut fails,
    );
    let img = ai.calcfrom1d(
        &tth,
        &intensity,
        u,
        Calcfrom1dOptions {
            correct_solid_angle: true,
            mask: Some(&mask),
            ..Default::default()
        },
    );
    check(
        "calcfrom1d/img_mask",
        &img,
        "calcfrom1d__img_mask.npy",
        Gate::Exact,
        &mut fails,
    );
    let img = ai.calcfrom1d(
        &tth,
        &intensity,
        u,
        Calcfrom1dOptions {
            correct_solid_angle: true,
            flat: Some(&flat),
            dark: Some(&dark),
            ..Default::default()
        },
    );
    check(
        "calcfrom1d/img_flatdark",
        &img,
        "calcfrom1d__img_flatdark.npy",
        Gate::Exact,
        &mut fails,
    );

    // ---- 4. fake_xrpdp / fake_calibration_image ----------------------------
    eprintln!("=== fake_xrpdp / fake_calibration_image (radial exact; Gaussian Tier-B) ===");
    let cal_text = std::fs::read_to_string(datasets_root().join("LaB6.D")).expect("LaB6.D");
    let mut cal = Calibrant::from_dspacing_file_str(&cal_text);
    cal.set_wavelength(1e-10);
    let (radial, fxi) = cal.fake_xrpdp(200, (0.0, 60.0), 0.1, 1.0, 0.1);
    check(
        "fake_xrpdp/radial",
        &radial,
        "fake_xrpdp__radial.npy",
        Gate::Exact,
        &mut fails,
    );
    check(
        "fake_xrpdp/intensity",
        &fxi,
        "fake_xrpdp__intensity.npy",
        Gate::Ulp,
        &mut fails,
    );

    let img = fake_calibration_image(&cal, &ai, 1.0, 0.1, 0.1);
    check(
        "fake_cal_image/img",
        &img,
        "fake_cal_image__img.npy",
        Gate::Ulp,
        &mut fails,
    );

    assert_eq!(fails, 0, "{fails} calcfrom field(s) failed their gate");
}
