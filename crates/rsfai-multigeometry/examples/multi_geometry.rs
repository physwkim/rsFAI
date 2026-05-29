//! Multi-geometry azimuthal integration end to end — the Rust analogue of
//! pyFAI's `MultiGeometry.ipynb` tutorial.
//!
//! One sample is imaged across three detector geometries (three committed
//! `.poni` files), each frame carrying its own normalization monitor. We build a
//! [`MultiGeometry`], integrate the three frames onto one shared grid, and print
//! the combined 1D curve and 2D cake.
//!
//! Runs fully offline: it loads the three committed geometries from
//! `golden/datasets_multigeometry/inputs/` and synthesizes a deterministic
//! two-ring powder pattern for each (the real per-pixel frames are large and
//! regenerable, so they are git-ignored — the golden verifier test, not this
//! example, is what compares against pyFAI).
//!
//!   cargo run --release --example multi_geometry -p rsfai-multigeometry

use std::path::PathBuf;

use rsfai::{Algo, AzimuthalIntegrator, Method, RadialUnit, Split};
use rsfai_multigeometry::{GeometryFrame, MultiGeometry, MultiIntegrationOptions};

/// The three committed geometries and their normalization monitors
/// (`golden/datasets_multigeometry/inputs/inputs.json`).
const PONIS: [&str; 3] = ["geometry_0.poni", "geometry_1.poni", "geometry_2.poni"];
const MONITORS: [f64; 3] = [1.0, 2.7, 0.6];

/// A deterministic two-ring powder pattern on a `(rows, cols)` detector: a flat
/// baseline plus two Gaussian rings centered on the detector. Same physics for
/// every geometry — the geometries differ in placement, the monitors in flux.
fn synth_powder(shape: (usize, usize)) -> Vec<f32> {
    let (rows, cols) = shape;
    let (cy, cx) = (rows as f64 / 2.0, cols as f64 / 2.0);
    let mut img = vec![0.0f32; rows * cols];
    for r in 0..rows {
        for c in 0..cols {
            let (dr, dc) = (r as f64 - cy, c as f64 - cx);
            let rad = (dr * dr + dc * dc).sqrt();
            let ring1 = 1000.0 * (-((rad - 150.0) / 25.0).powi(2)).exp();
            let ring2 = 600.0 * (-((rad - 350.0) / 25.0).powi(2)).exp();
            img[r * cols + c] = (10.0 + ring1 + ring2) as f32;
        }
    }
    img
}

fn main() {
    let inputs = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../golden/datasets_multigeometry/inputs");

    // Load the three geometries; resolve_detector picks up Pilatus1M from each PONI.
    let ais: Vec<AzimuthalIntegrator> = PONIS
        .iter()
        .map(|p| AzimuthalIntegrator::load(inputs.join(p)).expect("load committed .poni"))
        .collect();

    let shape = ais[0].detector.shape;
    let imgs: Vec<Vec<f32>> = ais.iter().map(|_| synth_powder(shape)).collect();
    let frames: Vec<GeometryFrame> = (0..ais.len())
        .map(|i| GeometryFrame {
            data: &imgs[i],
            variance: None,
            mask: None,
            flat: None,
            monitor: MONITORS[i],
        })
        .collect();

    // bbox-split CSR: the bit-exact-vs-pyFAI method (cf. the golden verifier).
    let opts = MultiIntegrationOptions {
        correct_solid_angle: true,
        error_model: rsfai::ErrorModelKind::Poisson,
        polarization_factor: None,
        method: Method {
            split: Split::Bbox,
            algo: Algo::Csr,
        },
    };

    let mg = MultiGeometry::new(ais, RadialUnit::TTH_DEG);
    let (r_lo, r_hi) = mg.effective_radial_range();
    let (a_lo, a_hi) = mg.effective_azimuth_range();

    println!(
        "MultiGeometry: {} geometries, {shape:?} Pilatus1M",
        mg.ais.len()
    );
    println!("  radial unit   2th_deg, range [{r_lo:.4}, {r_hi:.4}] deg (guessed)");
    println!("  azimuth unit  chi_deg, range [{a_lo:.2}, {a_hi:.2}] deg (guessed)");
    println!("  monitors      {MONITORS:?}, correctSolidAngle = on, error model = Poisson\n");

    // ---- 1D ----
    let npt = 500;
    let r1 = mg.integrate1d(&frames, npt, &opts);
    let peak = (0..r1.intensity.len())
        .max_by(|&a, &b| r1.intensity[a].total_cmp(&r1.intensity[b]))
        .unwrap();
    let total_signal: f64 = r1.sum_signal.iter().sum();
    let total_count: f64 = r1.count.iter().sum();
    println!("integrate1d -> {npt} bins");
    println!(
        "  peak  2th = {:.4} deg   I = {:.2}   sigma = {:.2}",
        r1.radial[peak], r1.intensity[peak], r1.sigma[peak]
    );
    println!("  total binned signal = {total_signal:.3e}   valid pixels = {total_count:.0}");
    println!("  sampled curve (every 100th bin):");
    for i in (0..npt).step_by(100) {
        println!(
            "    2th = {:7.4} deg   I = {:10.3}   sigma = {:9.3}",
            r1.radial[i], r1.intensity[i], r1.sigma[i]
        );
    }

    // ---- 2D ----
    let (npt_rad, npt_azim) = (200, 36);
    let r2 = mg.integrate2d(&frames, npt_rad, npt_azim, &opts);
    let cell = (0..r2.intensity.len())
        .max_by(|&a, &b| r2.intensity[a].total_cmp(&r2.intensity[b]))
        .unwrap();
    let (nrad, nazim) = r2.bins;
    let (azim_j, rad_i) = (cell / nrad, cell % nrad);
    println!("\nintegrate2d -> {nrad} radial x {nazim} azimuth cells");
    println!(
        "  radial   [{:.4}, {:.4}] deg",
        r2.radial.first().unwrap(),
        r2.radial.last().unwrap()
    );
    println!(
        "  azimuth  [{:.2}, {:.2}] deg",
        r2.azimuthal.first().unwrap(),
        r2.azimuthal.last().unwrap()
    );
    println!(
        "  brightest cell  2th = {:.4} deg, chi = {:.2} deg   I = {:.2}",
        r2.radial[rad_i], r2.azimuthal[azim_j], r2.intensity[cell]
    );
}
