//! Basic azimuthal integration end to end — the Rust analogue of pyFAI's
//! `integration_with_python.ipynb` cookbook.
//!
//! The cookbook loads a calibrated geometry (`.poni`) and one diffraction frame
//! (an EDF read with fabio), then runs `integrate1d_ng(img, 1000, unit="2th_deg")`
//! and `integrate2d_ng(img, 500, 360, unit="r_mm")`, writing `.dat`/`.edf` and
//! plotting with matplotlib. Here we build an [`AzimuthalIntegrator`] from a
//! committed `.poni`, run the same two integrations, and print the binned 1D
//! profile head plus the 2D cake shape and peak.
//!
//! Runs fully offline: it loads the committed Pilatus1M geometry from
//! `golden/datasets/Pilatus1M__bbox-csr-cython__2th_deg__npt1000__errpoisson/geometry.poni`
//! and synthesizes a deterministic concentric-ring powder pattern (the real EDF
//! frame is large and not committed). The fabio EDF read, the `.dat`/`.edf`
//! writes, and the matplotlib plotting cells are dropped.
//!
//!   cargo run --release --example integrate_basic -p rsfai

use std::path::PathBuf;

use rsfai::{AzimuthalIntegrator, IntegrationOptions, RadialUnit};

/// A deterministic concentric-ring powder pattern on a `(rows, cols)` detector:
/// a flat baseline plus three Gaussian rings centered on the detector. This is
/// the same shape the multi-geometry example uses, with one more ring so the 1D
/// profile shows several peaks.
fn synth_powder(shape: (usize, usize)) -> Vec<f32> {
    let (rows, cols) = shape;
    let (cy, cx) = (rows as f64 / 2.0, cols as f64 / 2.0);
    let mut img = vec![0.0f32; rows * cols];
    for r in 0..rows {
        for c in 0..cols {
            let (dr, dc) = (r as f64 - cy, c as f64 - cx);
            let rad = (dr * dr + dc * dc).sqrt();
            let ring1 = 1000.0 * (-((rad - 120.0) / 20.0).powi(2)).exp();
            let ring2 = 700.0 * (-((rad - 260.0) / 22.0).powi(2)).exp();
            let ring3 = 400.0 * (-((rad - 400.0) / 25.0).powi(2)).exp();
            img[r * cols + c] = (10.0 + ring1 + ring2 + ring3) as f32;
        }
    }
    img
}

fn main() {
    let poni = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(
        "../../golden/datasets/\
         Pilatus1M__bbox-csr-cython__2th_deg__npt1000__errpoisson/geometry.poni",
    );
    let ai = AzimuthalIntegrator::load(&poni).expect("load committed .poni");

    let shape = ai.detector.shape;
    let image = synth_powder(shape);
    let opts = IntegrationOptions::default();

    println!(
        "AzimuthalIntegrator: {shape:?} Pilatus1M, dist = {:.4} m, wavelength = {:.3e} m",
        ai.dist, ai.wavelength
    );
    println!("  correctSolidAngle = on (pyFAI default), no polarization\n");

    // ---- 1D: integrate1d(img, 1000, unit="2th_deg") ----
    let npt = 1000;
    let r1 = ai.integrate1d(&image, npt, RadialUnit::TTH_DEG, &opts);
    let peak = (0..r1.intensity.len())
        .max_by(|&a, &b| r1.intensity[a].total_cmp(&r1.intensity[b]))
        .unwrap();
    let total_count: f64 = r1.count.iter().sum();
    println!("integrate1d -> {npt} bins, unit 2th_deg");
    println!(
        "  radial   [{:.4}, {:.4}] deg",
        r1.radial.first().unwrap(),
        r1.radial.last().unwrap()
    );
    println!(
        "  peak  2th = {:.4} deg   I = {:.3}",
        r1.radial[peak], r1.intensity[peak]
    );
    println!("  valid pixels binned = {total_count:.0}");
    println!("  profile head (first 5 bins):");
    for i in 0..5 {
        println!(
            "    2th = {:8.4} deg   I = {:10.3}",
            r1.radial[i], r1.intensity[i]
        );
    }

    // ---- 2D: integrate2d(img, 500, 360, unit="r_mm") ----
    let (npt_rad, npt_azim) = (500, 360);
    let r2 = ai.integrate2d(&image, npt_rad, npt_azim, RadialUnit::R_MM, &opts);
    let cell = (0..r2.intensity.len())
        .max_by(|&a, &b| r2.intensity[a].total_cmp(&r2.intensity[b]))
        .unwrap();
    let (nrad, _nazim) = r2.bins;
    let (rad_i, azim_j) = (cell % nrad, cell / nrad);
    println!(
        "\nintegrate2d -> {} radial x {} azimuth cells, unit r_mm",
        r2.bins.0, r2.bins.1
    );
    println!(
        "  radial   [{:.4}, {:.4}] mm",
        r2.radial.first().unwrap(),
        r2.radial.last().unwrap()
    );
    println!(
        "  azimuth  [{:.2}, {:.2}] deg",
        r2.azimuthal.first().unwrap(),
        r2.azimuthal.last().unwrap()
    );
    println!(
        "  brightest cell  r = {:.4} mm, chi = {:.2} deg   I = {:.3}",
        r2.radial[rad_i], r2.azimuthal[azim_j], r2.intensity[cell]
    );
}
