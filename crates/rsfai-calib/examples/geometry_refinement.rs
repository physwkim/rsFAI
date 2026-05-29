//! Offline ControlPoints + GeometryRefinement demo, mirroring pyFAI's
//! `geometryRefinement` workflow on the committed `test_noSpline` LaB6 golden
//! (`golden/datasets_calib/`). No network, no detector image: the control points
//! and calibrant come straight from the committed golden arrays.
//!
//! It (1) builds a `ControlPoints` container from the golden `(d1, d2, ring)`
//! rows, (2) builds the calibrant + detector + an initial geometry, (3) prints
//! the residual / chi2 at the fixed initial parameters (the bit-exact core), and
//! (4) runs `refine()` and prints the converged geometry + chi2.
//!
//! Run: `cargo run --release --example geometry_refinement -p rsfai-calib`.

use std::path::PathBuf;

use rsfai_calib::{ControlPoints, GeometryParams, GeometryRefinement, Param};
use rsfai_calibrant::Calibrant;
use rsfai_core::golden::load_npy_f64;
use rsfai_detectors::Detector;

fn datasets_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../golden/datasets_calib")
}

fn load_f64(name: &str) -> Vec<f64> {
    let p = datasets_root().join(name);
    load_npy_f64(&p)
        .unwrap_or_else(|e| panic!("load {name}: {e}"))
        .iter()
        .copied()
        .collect()
}

fn main() {
    // --- Control points (golden (d1, d2, ring) rows) into a ControlPoints. ---
    let cp_flat = load_f64("control_points.npy"); // (51, 3) row-major
    let rows: Vec<(f64, f64, usize)> = cp_flat
        .chunks_exact(3)
        .map(|r| (r[0], r[1], r[2] as usize))
        .collect();

    // Group the points by ring into the container (one PointGroup per ring),
    // exactly the shape a peak-picker would produce.
    let mut cp = ControlPoints::new();
    let max_ring = rows.iter().map(|&(_, _, r)| r).max().unwrap_or(0);
    for ring in 1..=max_ring {
        let pts: Vec<(f64, f64)> = rows
            .iter()
            .filter(|&&(_, _, r)| r == ring)
            .map(|&(y, x, _)| (y, x))
            .collect();
        if !pts.is_empty() {
            cp.append(pts, Some(ring));
        }
    }
    println!(
        "ControlPoints: {} groups ({} points across rings 1..={max_ring})",
        cp.len(),
        cp.list_ring().len()
    );

    // --- Calibrant + detector + initial geometry (from the golden manifest). ---
    let dspacing = load_f64("calibrant_dspacing.npy");
    let calibrant = Calibrant::from_dspacing(dspacing);

    // Generic detector, 15 µm pixels, orientation 3 (pyFAI's BottomRight default
    // for an unnamed detector); shape large enough to hold the control points.
    let mut det = Detector::generic(1.5e-5, 1.5e-5, (8192, 8192));
    det.orientation = 3;

    let fixed6 = load_f64("fixed_param6.npy");
    let init = GeometryParams {
        dist: fixed6[0],
        poni1: fixed6[1],
        poni2: fixed6[2],
        rot1: fixed6[3],
        rot2: fixed6[4],
        rot3: fixed6[5],
        wavelength: 1.54e-10,
    };

    // The container hands the geometry layer its (d1, d2, ring) rows.
    let points = cp.list_ring();
    let mut refine = GeometryRefinement::new(points, calibrant, det, init);

    // --- Bit-exact core: residual / chi2 at the fixed initial parameters. ---
    let residual = refine.residu1(&init.six());
    let worst = residual.iter().fold(0.0_f64, |m, &r| m.max(r.abs()));
    println!(
        "\nAt the initial geometry (dist={:.6} m, poni1={:.6} m, poni2={:.6} m,",
        init.dist, init.poni1, init.poni2
    );
    println!(
        "                          rot1={:.4}, rot2={:.4}, rot3={:.4} rad):",
        init.rot1, init.rot2, init.rot3
    );
    println!("  chi2          = {:.6e}", refine.chi2_current());
    println!(
        "  |residual|max = {worst:.6e} rad   (over {} points)",
        residual.len()
    );

    // --- Iterative refine (tolerance-gated; isolated argmin Nelder-Mead). ---
    let chi2_after = refine.refine();
    let p = refine.param();
    println!("\nAfter refine() (wavelength fixed):");
    println!(
        "  dist  = {:.9} m\n  poni1 = {:.9} m\n  poni2 = {:.9} m",
        p.dist, p.poni1, p.poni2
    );
    println!(
        "  rot1  = {:.9} rad\n  rot2  = {:.9} rad\n  rot3  = {:.9} rad",
        p.rot1, p.rot2, p.rot3
    );
    println!("  chi2  = {chi2_after:.6e}");

    // Also show the well-constrained 5-parameter refine (rot3 pinned).
    let init2 = GeometryParams { rot3: 0.0, ..init };
    let cp_rows = cp.list_ring();
    let cal2 = Calibrant::from_dspacing(load_f64("calibrant_dspacing.npy"));
    let mut det2 = Detector::generic(1.5e-5, 1.5e-5, (8192, 8192));
    det2.orientation = 3;
    let mut refine2 = GeometryRefinement::new(cp_rows, cal2, det2, init2);
    let chi2_fix = refine2.refine_with_fixed(&[Param::Rot3]);
    println!("\nAfter refine(fix rot3): chi2 = {chi2_fix:.6e}");

    println!("\nOK");
}
