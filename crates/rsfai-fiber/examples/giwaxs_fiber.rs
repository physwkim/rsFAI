//! Fiber / grazing-incidence integration end to end — the Rust analogue of
//! pyFAI's GIWAXS fiber-integration tutorial (`Grazing_incidence.ipynb` and the
//! `integrate2d_fiber` / `integrate_fiber` cookbook).
//!
//! A single detector frame is remapped from (radial, chi) into the fiber frame:
//! in-plane `q_IP` × out-of-plane `q_OOP`. We build a [`FiberIntegrator`] from a
//! committed grazing-incidence `.poni`, integrate the frame into a 2D `qip×qoop`
//! map, then fold it both ways:
//!   * `vertical_integration = true`  → a profile vs `q_OOP` (sums out `q_IP`),
//!   * `vertical_integration = false` → a profile vs `q_IP`  (sums out `q_OOP`).
//!
//! Runs fully offline: it loads the committed geometry from
//! `golden/datasets_fiber_integrator/geometry.poni` and synthesizes a
//! deterministic, *anisotropic* pattern (an out-of-plane Bragg rod plus an
//! in-plane streak) so the two folds differ visibly — the GIWAXS distinction the
//! fiber frame exists to expose. The numerical parity check against pyFAI lives
//! in the golden verifier test, not here.
//!
//!   cargo run --release --example giwaxs_fiber -p rsfai-fiber

use std::path::PathBuf;

use rsfai::{AzimuthalIntegrator, Corrections, IntegrationOptions};
use rsfai_fiber::{FiberAxes, FiberIntegrator};
use rsfai_geometry::GiParams;

/// Grazing-incidence sample geometry for the demo (one of the golden combos):
/// 0.2 rad incidence, no detector tilt, sample orientation 1.
const GI: GiParams = GiParams {
    incident_angle: 0.2,
    tilt_angle: 0.0,
    sample_orientation: 1,
};

/// A deterministic, anisotropic detector pattern: a flat baseline, an
/// out-of-plane "Bragg rod" (a vertical streak on the detector), and a narrower
/// in-plane reflection (a horizontal streak). The two features live along
/// different detector directions, so the `q_IP` and `q_OOP` folds see different
/// structure — the point of remapping into the fiber frame.
fn synth_giwaxs(shape: (usize, usize)) -> Vec<f32> {
    let (rows, cols) = shape;
    let (cy, cx) = (rows as f64 / 2.0, cols as f64 / 2.0);
    let mut img = vec![0.0f32; rows * cols];
    for r in 0..rows {
        for c in 0..cols {
            let (dr, dc) = (r as f64 - cy, c as f64 - cx);
            // Vertical streak (varies along columns): out-of-plane rod.
            let rod = 900.0 * (-(dc / 40.0).powi(2)).exp();
            // Horizontal streak (varies along rows), offset off-center: in-plane.
            let streak = 500.0 * (-((dr - 120.0) / 30.0).powi(2)).exp();
            img[r * cols + c] = (10.0 + rod + streak) as f32;
        }
    }
    img
}

/// Report the peak and a few sampled points of a 1D fold.
fn report_fold(label: &str, axis: &[f64], intensity: &[f64], count: &[f64], unit: &str) {
    let peak = (0..intensity.len())
        .max_by(|&a, &b| intensity[a].total_cmp(&intensity[b]))
        .unwrap();
    let valid: f64 = count.iter().sum();
    println!("{label} -> {} bins", axis.len());
    println!(
        "  peak  {unit} = {:.4}   I = {:.3}",
        axis[peak], intensity[peak]
    );
    println!("  valid pixels folded = {valid:.0}");
    println!("  sampled curve (every {}th bin):", axis.len() / 5);
    for i in (0..axis.len()).step_by(axis.len() / 5) {
        println!("    {unit} = {:8.4}   I = {:10.3}", axis[i], intensity[i]);
    }
}

fn main() {
    let poni = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../golden/datasets_fiber_integrator/geometry.poni");
    let ai = AzimuthalIntegrator::load(&poni).expect("load committed .poni");

    let shape = ai.detector.shape;
    let image = synth_giwaxs(shape);

    let fi = FiberIntegrator::new(ai, GI);
    let axes = FiberAxes::qip_qoop_nm(200, 200);
    let opts = IntegrationOptions {
        correct_solid_angle: true,
        ..Default::default()
    };
    let corr = Corrections::with_normalization(1.0);

    println!(
        "FiberIntegrator: {shape:?} Pilatus1M, GI incidence = {} rad, tilt = {} rad, orientation {}",
        GI.incident_angle, GI.tilt_angle, GI.sample_orientation
    );
    println!(
        "  axes  q_IP (nm^-1) x q_OOP (nm^-1), {} x {} bins, correctSolidAngle = on\n",
        axes.npt_ip, axes.npt_oop
    );

    // ---- 2D fiber map ----
    let r2 = fi.integrate2d_fiber(&image, &axes, &opts, &corr);
    let (n_ip, _n_oop) = r2.bins;
    let cell = (0..r2.intensity.len())
        .max_by(|&a, &b| r2.intensity[a].total_cmp(&r2.intensity[b]))
        .unwrap();
    let (ip_i, oop_j) = (cell % n_ip, cell / n_ip);
    println!(
        "integrate2d_fiber -> {} (q_IP) x {} (q_OOP) cells",
        r2.bins.0, r2.bins.1
    );
    println!(
        "  q_IP   [{:.4}, {:.4}] nm^-1",
        r2.inplane.first().unwrap(),
        r2.inplane.last().unwrap()
    );
    println!(
        "  q_OOP  [{:.4}, {:.4}] nm^-1",
        r2.outofplane.first().unwrap(),
        r2.outofplane.last().unwrap()
    );
    println!(
        "  brightest cell  q_IP = {:.4}, q_OOP = {:.4} nm^-1   I = {:.3}\n",
        r2.inplane[ip_i], r2.outofplane[oop_j], r2.intensity[cell]
    );

    // ---- 1D folds: vertical (vs q_OOP) and horizontal (vs q_IP) ----
    let v = fi.integrate_fiber(&image, &axes, true, &opts, &corr);
    report_fold(
        "integrate_fiber (vertical, sums out q_IP)",
        &v.axis,
        &v.intensity,
        &v.count,
        "q_OOP",
    );
    println!();
    let h = fi.integrate_fiber(&image, &axes, false, &opts, &corr);
    report_fold(
        "integrate_fiber (horizontal, sums out q_OOP)",
        &h.axis,
        &h.intensity,
        &h.count,
        "q_IP",
    );
}
