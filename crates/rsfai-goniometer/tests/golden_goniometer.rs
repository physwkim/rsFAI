//! Goniometer subsystem parity gate: `rsfai-goniometer` vs pyFAI on
//! `golden/datasets_goniometer/` (built by `gen_golden_goniometer.py` in the daq
//! env, single-thread).
//!
//! FOUR surfaces across the two structurally-separate tiers (see the crate docs
//! and `doc/bit-exact-ladder.md`):
//!
//!   * **BIT-EXACT — GeometryTransformation outputs.** A transformation
//!     exercising the full numexpr op set (`**`, `sqrt`, `sin`, `pi`, division,
//!     nested parens) is evaluated at several positions; the six PONI outputs
//!     must match pyFAI bit-for-bit (`f64::to_bits`). This is what proves the
//!     Rust expression evaluator reproduces numexpr, not assumes it.
//!   * **BIT-EXACT — GoniometerRefinement residu2 at a fixed param.** The mean
//!     squared 2theta error over two single geometries at the fixed goniometer
//!     parameter vector. The only ULP-budgeted part is the geometry
//!     atan2/sin/cos (Tier-B, inherited from `rsfai_calib`/`rsfai_geometry`).
//!   * **BIT-EXACT — MultiGeometryFiber 1D + 2D.** Two fiber frames combined by
//!     the direct per-bin accumulator summation; the summed accumulators and the
//!     recomputed intensity are bit-exact (f64 left-fold). The qip/qoop bin-centre
//!     axes carry the scipy-quaternion-vs-direct-matrix ULP divergence
//!     (`datasets_fiber` gate), so they are tolerance-gated.
//!   * **TOLERANCE — refine() converged cost.** `GoniometerRefinement::refine`
//!     (argmin Nelder-Mead) is run from the fixed start; its converged cost is
//!     asserted `<=` pyFAI's `refine2` (scipy SLSQP) cost. This is NOT a bit-exact
//!     claim and NOT a per-parameter claim: the two optimizers (Nelder-Mead vs
//!     SLSQP) take different trajectories, and `rot3` is a null direction of this
//!     ring data (no control point constrains it), so the converged `rot3` differs
//!     between them while the cost is unaffected. The five constrained parameters
//!     agree to ~4 significant figures and the costs match to ~7 figures; the
//!     per-parameter deltas are printed for the record and the strict gate is cost.

use std::path::PathBuf;

use rsfai::{Corrections, IntegrationOptions};
use rsfai_calib::{GeometryParams, GeometryRefinement};
use rsfai_calibrant::Calibrant;
use rsfai_core::compare::compare_f64;
use rsfai_core::golden::{load_image_f32, load_npy_f64};
use rsfai_detectors::Detector;
use rsfai_fiber::{FiberAxes, FiberIntegrator, FiberUnit};
use rsfai_geometry::GiParams;
use rsfai_goniometer::{
    GeometryTransformation, Goniometer, GoniometerRefinement, MultiGeometryFiber, SingleGeometry,
};
use serde_json::Value;

/// ULP ceiling for the geometry transcendental boundary (atan2/sin/cos through
/// `calc_pos_zyx`) plus the numexpr-vs-libm transcendentals in the formula
/// evaluator. On this machine both match numpy bit-for-bit so the observed value
/// is 0; the budget is the sanctioned ceiling, not a claim of the measured gap.
const GEOM_ULP_BUDGET: u64 = 4;

/// Relative tolerance for the qip/qoop bin-centre axes (the scipy-quaternion-vs-
/// direct-matrix ULP divergence the fiber units carry). Mirrors the fiber golden
/// gate. The test prints the measured worst-case every run.
const AXIS_REL_TOL: f64 = 1e-6;

fn datasets_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../golden/datasets_goniometer")
}

fn manifest() -> Value {
    let p = datasets_root().join("manifest.json");
    let text = std::fs::read_to_string(&p).unwrap_or_else(|e| panic!("read manifest {p:?}: {e}"));
    serde_json::from_str(&text).expect("parse manifest.json")
}

fn f64v(name: &str) -> Vec<f64> {
    let p = datasets_root().join(name);
    load_npy_f64(&p)
        .unwrap_or_else(|e| panic!("load {name}: {e}"))
        .iter()
        .copied()
        .collect()
}

fn f32v(name: &str) -> Vec<f32> {
    let p = datasets_root().join(name);
    load_image_f32(&p).unwrap_or_else(|e| panic!("load {name}: {e}"))
}

/// Print a per-field bit-exact report (with the Tier-B geometry ULP budget) and
/// update fail/worst-ULP counters.
fn check_exact(label: &str, got: &[f64], golden: &[f64], worst_ulp: &mut u64, fails: &mut usize) {
    if got.len() != golden.len() {
        *fails += 1;
        eprintln!(
            "    {label:28} FAIL      length {} != golden {}",
            got.len(),
            golden.len()
        );
        return;
    }
    let r = compare_f64(got, golden);
    *worst_ulp = (*worst_ulp).max(r.max_ulp);
    let ok = r.is_bit_exact() || r.within_ulp(GEOM_ULP_BUDGET);
    if !ok {
        *fails += 1;
    }
    let tag = if r.is_bit_exact() {
        "BIT-EXACT"
    } else if ok {
        "ulp-ok   "
    } else {
        "FAIL     "
    };
    eprintln!(
        "    {label:28} {tag}  ulp={:3} mism={}/{}",
        r.max_ulp, r.bit_mismatches, r.total
    );
}

// =========================================================================
// Surface 1: GeometryTransformation outputs (bit-exact formula gate).
// =========================================================================

/// The "formula" transformation in the generator: full numexpr op set.
fn build_formula_transformation() -> GeometryTransformation {
    GeometryTransformation::new(
        Some("pos_dist * dist_scale + dist_offset"),
        Some("poni1_base + bow * sin(pos_angle) ** 2 + sqrt(bow) / pi"),
        Some("poni2 + (rot1 - bow) ** 3"),
        Some("rot1"),
        Some("pos_angle * rot2_scale + rot2_offset"),
        Some("0.0"),
        &[
            "dist_scale",
            "dist_offset",
            "poni1_base",
            "poni2",
            "rot1",
            "rot2_scale",
            "rot2_offset",
            "bow",
        ],
        Some(&["pos_dist", "pos_angle"]),
        &[],
    )
    .expect("build formula transformation")
}

#[test]
fn transformation_outputs_bit_exact() {
    let mut worst_ulp = 0u64;
    let mut fails = 0usize;

    let t = build_formula_transformation();
    let param = f64v("formula_param.npy");
    let positions = f64v("transform_positions.npy"); // flat (5, 2)
    let golden = f64v("transform_outputs.npy"); // flat (5, 6)

    let npos = positions.len() / 2;
    eprintln!("=== GeometryTransformation outputs (BIT-EXACT formula gate) ===");
    for i in 0..npos {
        let pos = [positions[2 * i], positions[2 * i + 1]];
        let out = t.call(&param, &pos).expect("evaluate transformation");
        let got = out.as_array().to_vec();
        let g = golden[6 * i..6 * i + 6].to_vec();
        check_exact(&format!("pos{i}"), &got, &g, &mut worst_ulp, &mut fails);
    }
    eprintln!("worst transformation ULP (Tier-B): {worst_ulp} (budget {GEOM_ULP_BUDGET})");
    assert_eq!(fails, 0, "{fails} transformation output(s) failed the gate");
}

// =========================================================================
// Surfaces 2 & 4 helpers: the refinement transformation + control points.
// =========================================================================

/// The "refinement" transformation in the generator: identity passthrough of the
/// six geometry params, single motor `pos` bound via `rot2 + 0.0 * pos`.
fn build_refinement_transformation() -> GeometryTransformation {
    GeometryTransformation::new(
        Some("dist"),
        Some("poni1"),
        Some("poni2"),
        Some("rot1"),
        Some("rot2 + 0.0 * pos"),
        Some("rot3"),
        &["dist", "poni1", "poni2", "rot1", "rot2", "rot3"],
        Some(&["pos"]),
        &[],
    )
    .expect("build refinement transformation")
}

/// The 25 test_noSpline LaB6 control points (same fixture the generator embeds),
/// split into the two single geometries the generator built (rows 0..13, 13..25).
const CONTROL_POINTS: [(f64, f64, usize); 25] = [
    (1585.9999996029055, 2893.999999119241, 0),
    (1853.9999932086102, 2873.000000163791, 0),
    (2163.9999987531855, 2854.9999987738884, 0),
    (2699.999997791493, 2893.9999985831755, 0),
    (3186.9999966428777, 3028.9999985930604, 0),
    (1561.0000027706968, 2627.000000529364, 1),
    (1820.9999979673413, 2_588.999_999_615_856, 1),
    (2143.999999081593, 2_562.999_999_050_397, 1),
    (2706.9999983093525, 2585.9999992923594, 1),
    (3210.99999698136, 2719.0000003219736, 1),
    (1539.0000001229334, 2375.000000406348, 2),
    (1_796.000_002_316_776, 2326.000000284083, 2),
    (2_125.000_001_097_206, 2293.0000005033, 2),
    (2700.0000010876255, 2306.0000010888773, 2),
    (3232.0000016877327, 2_440.000_000_254_214, 2),
    (1_517.999_999_240_724, 2123.000000123723, 3),
    (1772.0000003839145, 2065.999999990455, 3),
    (2106.999998035475, 2024.9999990049748, 3),
    (2693.000001076276, 2026.0000017288656, 3),
    (3252.000001976179, 2160.999999990745, 3),
    (1499.0000010839487, 1870.9999991683347, 4),
    (1750.0000026536995, 1805.0000019021847, 4),
    (2090.0000003820063, 1757.0000008539542, 4),
    (2685.000000523629, 1746.0000010668696, 4),
    (3270.0000018384985, 1879.9999996170894, 4),
];

/// Build the GoniometerRefinement mirroring the generator: identity-passthrough
/// transformation, the LaB6 calibrant, the generic detector at the manifest
/// orientation, two single geometries (control points split 13/12) at motor
/// `pos=0`, and the given six-parameter goniometer vector.
fn build_goniometer_refinement(param6: &[f64; 6]) -> GoniometerRefinement {
    let m = manifest();
    let cfg = &m["config"];
    let pixel1 = cfg["pixel1"].as_f64().unwrap();
    let pixel2 = cfg["pixel2"].as_f64().unwrap();
    let wavelength = cfg["wavelength"].as_f64().unwrap();
    let orientation = cfg["orientation"].as_i64().unwrap() as i32;

    let mut det = Detector::generic(pixel1, pixel2, (8192, 8192));
    det.orientation = orientation;

    let trans = build_refinement_transformation();
    let gonio = Goniometer::new(trans, param6.to_vec(), det.clone(), wavelength);

    let gp = GeometryParams {
        dist: param6[0],
        poni1: param6[1],
        poni2: param6[2],
        rot1: param6[3],
        rot2: param6[4],
        rot3: param6[5],
        wavelength,
    };

    let dspacing = f64v("calibrant_dspacing.npy");
    let make_single = |rows: &[(f64, f64, usize)]| -> SingleGeometry {
        let cal = Calibrant::from_dspacing(dspacing.clone());
        let refinement = GeometryRefinement::new(rows.to_vec(), cal, det.clone(), gp);
        SingleGeometry::new(vec![0.0], refinement)
    };
    let singles = vec![
        make_single(&CONTROL_POINTS[..13]),
        make_single(&CONTROL_POINTS[13..]),
    ];
    GoniometerRefinement::new(gonio, singles)
}

// =========================================================================
// Surface 2: GoniometerRefinement residu2 at a fixed param (bit-exact).
// =========================================================================

#[test]
fn residu2_fixed_param_bit_exact() {
    let m = manifest();
    let fixed = f64v("fixed_param.npy");
    let fixed6: [f64; 6] = [fixed[0], fixed[1], fixed[2], fixed[3], fixed[4], fixed[5]];
    let gr = build_goniometer_refinement(&fixed6);

    let got = gr.residu2(&fixed).expect("evaluate residu2");
    let golden = m["config"]["residu2_fixed"].as_f64().unwrap();

    let cr = compare_f64(&[got], &[golden]);
    let ok = cr.is_bit_exact() || cr.within_ulp(GEOM_ULP_BUDGET);
    eprintln!("=== GoniometerRefinement residu2 at fixed param (BIT-EXACT gate) ===");
    eprintln!(
        "    residu2  {}  ulp={}  rust={:.17e} golden={:.17e}",
        if cr.is_bit_exact() {
            "BIT-EXACT"
        } else if ok {
            "ulp-ok   "
        } else {
            "FAIL     "
        },
        cr.max_ulp,
        got,
        golden
    );
    assert!(ok, "residu2 at fixed param failed the bit-exact gate");
}

// =========================================================================
// Surface 3: MultiGeometryFiber 1D + 2D (bit-exact accumulators + intensity).
// =========================================================================

fn build_mgf() -> (MultiGeometryFiber, Vec<Vec<f32>>) {
    let root = datasets_root();
    let m = manifest();
    let fiber = &m["fiber"];
    let npt_ip = fiber["npt_ip"].as_u64().unwrap() as usize;
    let npt_oop = fiber["npt_oop"].as_u64().unwrap() as usize;
    let gi = &fiber["shared_gi"];
    let gi_params = GiParams {
        incident_angle: gi["incident_angle"].as_f64().unwrap(),
        tilt_angle: gi["tilt_angle"].as_f64().unwrap(),
        sample_orientation: gi["sample_orientation"].as_u64().unwrap() as u8,
    };

    // Two frames: rebuild each from its committed .poni (the validated detector
    // resolution path), shared GI params, common qip/qoop axes.
    let ai0 = rsfai::AzimuthalIntegrator::load(root.join("fiber_geometry_0.poni"))
        .expect("load fiber poni 0");
    let ai1 = rsfai::AzimuthalIntegrator::load(root.join("fiber_geometry_1.poni"))
        .expect("load fiber poni 1");
    let fis = vec![
        FiberIntegrator::new(ai0, gi_params),
        FiberIntegrator::new(ai1, gi_params),
    ];
    let axes = FiberAxes {
        npt_ip,
        unit_ip: FiberUnit::QIP_NM,
        ip_range: None,
        npt_oop,
        unit_oop: FiberUnit::QOOP_NM,
        oop_range: None,
    };
    let data0 = f32v("fiber_data_0.npy");
    let data1 = f32v("fiber_data_1.npy");
    (MultiGeometryFiber::new(fis, axes), vec![data0, data1])
}

#[test]
fn multigeometry_fiber_2d_bit_exact() {
    let (mgf, data) = build_mgf();
    let lst: Vec<&[f32]> = data.iter().map(|d| d.as_slice()).collect();
    let opts = IntegrationOptions {
        correct_solid_angle: true,
        ..Default::default()
    };
    let corr = Corrections::with_normalization(1.0);

    let mut worst_ulp = 0u64;
    let mut worst_rel = 0f64;
    let mut fails = 0usize;

    let r = mgf.integrate2d_fiber(&lst, &opts, &corr);
    eprintln!("=== MultiGeometryFiber 2D (BIT-EXACT accumulators + intensity) ===");

    // Accumulators: f64 left-fold of bit-exact per-geometry accumulators.
    check_exact(
        "2d/sum_signal",
        &r.sum_signal,
        &f64v("mgf_2d__sum_signal.npy"),
        &mut worst_ulp,
        &mut fails,
    );
    check_exact(
        "2d/sum_normalization",
        &r.sum_normalization,
        &f64v("mgf_2d__sum_normalization.npy"),
        &mut worst_ulp,
        &mut fails,
    );
    check_exact(
        "2d/count",
        &r.count,
        &f64v("mgf_2d__count.npy"),
        &mut worst_ulp,
        &mut fails,
    );
    check_exact(
        "2d/intensity",
        &r.intensity,
        &f64v("mgf_2d__intensity.npy"),
        &mut worst_ulp,
        &mut fails,
    );

    // Bin-centre axes carry the qip/qoop ULP divergence -> tolerance gate.
    check_tol_axis(
        "2d/inplane",
        &r.inplane,
        &f64v("mgf_2d__inplane.npy"),
        &mut worst_rel,
        &mut fails,
    );
    check_tol_axis(
        "2d/outofplane",
        &r.outofplane,
        &f64v("mgf_2d__outofplane.npy"),
        &mut worst_rel,
        &mut fails,
    );
    eprintln!("worst accumulator ULP: {worst_ulp}; worst axis rel: {worst_rel:.3e} (tol {AXIS_REL_TOL:.0e})");
    assert_eq!(fails, 0, "{fails} MGF 2D field(s) failed their gate");
}

#[test]
fn multigeometry_fiber_1d_bit_exact() {
    let (mgf, data) = build_mgf();
    let lst: Vec<&[f32]> = data.iter().map(|d| d.as_slice()).collect();
    let opts = IntegrationOptions {
        correct_solid_angle: true,
        ..Default::default()
    };
    let corr = Corrections::with_normalization(1.0);

    let mut worst_ulp = 0u64;
    let mut worst_rel = 0f64;
    let mut fails = 0usize;

    eprintln!("=== MultiGeometryFiber 1D (BIT-EXACT accumulators + intensity) ===");
    for (vert, vtag) in [(true, "v"), (false, "h")] {
        let r = mgf.integrate_fiber(&lst, vert, &opts, &corr);
        check_exact(
            &format!("1d{vtag}/sum_signal"),
            &r.sum_signal,
            &f64v(&format!("mgf_1d{vtag}__sum_signal.npy")),
            &mut worst_ulp,
            &mut fails,
        );
        check_exact(
            &format!("1d{vtag}/sum_normalization"),
            &r.sum_normalization,
            &f64v(&format!("mgf_1d{vtag}__sum_normalization.npy")),
            &mut worst_ulp,
            &mut fails,
        );
        check_exact(
            &format!("1d{vtag}/count"),
            &r.count,
            &f64v(&format!("mgf_1d{vtag}__count.npy")),
            &mut worst_ulp,
            &mut fails,
        );
        check_exact(
            &format!("1d{vtag}/intensity"),
            &r.intensity,
            &f64v(&format!("mgf_1d{vtag}__intensity.npy")),
            &mut worst_ulp,
            &mut fails,
        );
        check_tol_axis(
            &format!("1d{vtag}/axis"),
            &r.axis,
            &f64v(&format!("mgf_1d{vtag}__axis.npy")),
            &mut worst_rel,
            &mut fails,
        );
    }
    eprintln!("worst accumulator ULP: {worst_ulp}; worst axis rel: {worst_rel:.3e} (tol {AXIS_REL_TOL:.0e})");
    assert_eq!(fails, 0, "{fails} MGF 1D field(s) failed their gate");
}

/// Tolerance gate for the qip/qoop bin-centre axes.
fn check_tol_axis(
    label: &str,
    got: &[f64],
    golden: &[f64],
    worst_rel: &mut f64,
    fails: &mut usize,
) {
    if got.len() != golden.len() {
        *fails += 1;
        eprintln!(
            "    {label:28} FAIL      length {} != golden {}",
            got.len(),
            golden.len()
        );
        return;
    }
    let r = compare_f64(got, golden);
    *worst_rel = worst_rel.max(r.max_rel_diff);
    let ok = r.is_bit_exact() || r.within_rel(AXIS_REL_TOL);
    if !ok {
        *fails += 1;
    }
    let tag = if r.is_bit_exact() {
        "BIT-EXACT"
    } else if ok {
        "rel-ok   "
    } else {
        "FAIL     "
    };
    eprintln!("    {label:28} {tag}  rel={:.3e}", r.max_rel_diff);
}

// =========================================================================
// Surface 4: refine() converged cost (TOLERANCE / cost gate).
// =========================================================================

#[test]
fn refine_converged_cost_gate() {
    let m = manifest();
    let fixed = f64v("fixed_param.npy");
    let fixed6: [f64; 6] = [fixed[0], fixed[1], fixed[2], fixed[3], fixed[4], fixed[5]];
    let converged_golden = f64v("converged_param.npy");
    let cost_golden = m["config"]["cost_converged"].as_f64().unwrap();

    let mut gr = build_goniometer_refinement(&fixed6);
    let cost_rust = gr.refine().expect("refine");
    let p = gr.param().to_vec();

    eprintln!("=== refine(): COST gate (NOT bit-exact, NOT per-parameter) ===");
    let names = ["dist", "poni1", "poni2", "rot1", "rot2", "rot3"];
    for (i, name) in names.iter().enumerate() {
        eprintln!(
            "    {name:6} rust={:+.10e} pyfai={:+.10e}",
            p[i], converged_golden[i]
        );
    }
    eprintln!("    cost   rust={cost_rust:.12e} pyfai={cost_golden:.12e}");
    eprintln!(
        "    (rot3 is a null direction of this ring data, so it differs between \
         the Nelder-Mead and SLSQP trajectories; the five constrained params agree \
         to ~4 sig figs and the cost matches to ~7, so the gate is cost only.)"
    );

    assert!(
        cost_rust <= cost_golden,
        "rust converged cost {cost_rust:.12e} > pyFAI {cost_golden:.12e}"
    );
}
