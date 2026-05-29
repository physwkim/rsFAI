//! Offline goniometer demo, mirroring pyFAI's `Goniometer` /
//! `GoniometerRefinement` / `MultiGeometryFiber` workflow on the committed golden
//! (`golden/datasets_goniometer/`). No network, no live detector: every input
//! comes straight from the committed golden arrays + `.poni` files.
//!
//! It walks the four pieces of the subsystem in dependency order:
//!   1. `GeometryTransformation` — evaluate the six-PONI formula transformation at
//!      a goniometer position (the bit-exact numexpr-replica evaluator).
//!   2. `Goniometer::get_ai` — build an `AzimuthalIntegrator` from the transformed
//!      PONI at a motor position.
//!   3. `GoniometerRefinement` — print residu2 at the fixed parameter vector (the
//!      bit-exact core), then run `refine()` and print the converged cost.
//!   4. `MultiGeometryFiber` — combine the two committed fiber frames into a 1D
//!      profile and a 2D map, printing the bin counts.
//!
//! Run: `cargo run --release --example goniometer -p rsfai-goniometer`.

use std::path::PathBuf;

use rsfai::{AzimuthalIntegrator, Corrections, IntegrationOptions};
use rsfai_calib::{GeometryParams, GeometryRefinement};
use rsfai_calibrant::Calibrant;
use rsfai_core::golden::{load_image_f32, load_npy_f64};
use rsfai_detectors::Detector;
use rsfai_fiber::{FiberAxes, FiberIntegrator, FiberUnit};
use rsfai_geometry::GiParams;
use rsfai_goniometer::{
    GeometryTransformation, Goniometer, GoniometerRefinement, MultiGeometryFiber, SingleGeometry,
};

fn datasets_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../golden/datasets_goniometer")
}

fn load_f64(name: &str) -> Vec<f64> {
    let p = datasets_root().join(name);
    load_npy_f64(&p)
        .unwrap_or_else(|e| panic!("load {name}: {e}"))
        .iter()
        .copied()
        .collect()
}

fn load_f32(name: &str) -> Vec<f32> {
    let p = datasets_root().join(name);
    load_image_f32(&p).unwrap_or_else(|e| panic!("load {name}: {e}"))
}

/// The 25 test_noSpline LaB6 control points (slow, fast, ring), five rings of
/// five points, split 13/12 into the two single geometries.
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

const PIXEL: f64 = 1.5e-5;
const WAVELENGTH: f64 = 1.54e-10;

fn main() {
    // === 1. GeometryTransformation: the full numexpr op set at one position. ===
    let formula = GeometryTransformation::new(
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
    .expect("build formula transformation");
    let formula_param = load_f64("formula_param.npy");
    let positions = load_f64("transform_positions.npy");
    let pos0 = [positions[0], positions[1]];
    let poni = formula
        .call(&formula_param, &pos0)
        .expect("evaluate transformation");
    println!("== GeometryTransformation ==");
    println!("  position {pos0:?} -> PONI {:?}", poni.as_array());

    // === 2. Goniometer::get_ai at the same kind of position. ===============
    let mut det = Detector::generic(PIXEL, PIXEL, (8192, 8192));
    det.orientation = 3;
    let gonio = Goniometer::new(formula, formula_param.clone(), det.clone(), WAVELENGTH);
    let ai: AzimuthalIntegrator = gonio.get_ai(&pos0).expect("build ai from goniometer");
    println!("== Goniometer::get_ai ==");
    println!(
        "  dist={:.6} poni1={:.6} poni2={:.6}",
        ai.dist, ai.poni1, ai.poni2
    );

    // === 3. GoniometerRefinement: residu2 at fixed param, then refine(). ====
    let fixed = load_f64("fixed_param.npy");
    let fixed6: [f64; 6] = [fixed[0], fixed[1], fixed[2], fixed[3], fixed[4], fixed[5]];
    let mut gr = build_refinement(&fixed6, &det);
    let r2 = gr.residu2(&fixed).expect("residu2");
    println!("== GoniometerRefinement ==");
    println!("  residu2 at fixed param = {r2:.12e}");
    let cost = gr.refine().expect("refine");
    println!("  converged cost         = {cost:.12e}");
    println!("  converged param        = {:?}", gr.param());

    // === 4. MultiGeometryFiber: combine the two fiber frames. ==============
    let mgf = build_mgf();
    let data0 = load_f32("fiber_data_0.npy");
    let data1 = load_f32("fiber_data_1.npy");
    let lst: [&[f32]; 2] = [&data0, &data1];
    let opts = IntegrationOptions {
        correct_solid_angle: true,
        ..Default::default()
    };
    let corr = Corrections::with_normalization(1.0);
    let r1d = mgf.integrate_fiber(&lst, true, &opts, &corr);
    let r2d = mgf.integrate2d_fiber(&lst, &opts, &corr);
    println!("== MultiGeometryFiber ==");
    println!("  1D profile bins   = {}", r1d.axis.len());
    println!("  2D map bins (ip x oop) = {:?}", r2d.bins);
}

/// Build the GoniometerRefinement: identity-passthrough transformation, the LaB6
/// calibrant, two single geometries (control points split 13/12) at motor pos=0.
fn build_refinement(param6: &[f64; 6], det: &Detector) -> GoniometerRefinement {
    let trans = GeometryTransformation::new(
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
    .expect("build refinement transformation");
    let gonio = Goniometer::new(trans, param6.to_vec(), det.clone(), WAVELENGTH);
    let gp = GeometryParams {
        dist: param6[0],
        poni1: param6[1],
        poni2: param6[2],
        rot1: param6[3],
        rot2: param6[4],
        rot3: param6[5],
        wavelength: WAVELENGTH,
    };
    let dspacing = load_f64("calibrant_dspacing.npy");
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

/// Build the MultiGeometryFiber from the two committed frame `.poni` files and
/// the shared grazing-incidence params.
fn build_mgf() -> MultiGeometryFiber {
    let root = datasets_root();
    let gi = GiParams {
        incident_angle: 0.2,
        tilt_angle: 0.0,
        sample_orientation: 1,
    };
    let ai0 = AzimuthalIntegrator::load(root.join("fiber_geometry_0.poni")).expect("load poni 0");
    let ai1 = AzimuthalIntegrator::load(root.join("fiber_geometry_1.poni")).expect("load poni 1");
    let fis = vec![FiberIntegrator::new(ai0, gi), FiberIntegrator::new(ai1, gi)];
    let axes = FiberAxes {
        npt_ip: 200,
        unit_ip: FiberUnit::QIP_NM,
        ip_range: None,
        npt_oop: 200,
        unit_oop: FiberUnit::QOOP_NM,
        oop_range: None,
    };
    MultiGeometryFiber::new(fis, axes)
}
