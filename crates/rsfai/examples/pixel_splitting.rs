//! Effect of pixel splitting on 2D integration — the Rust analogue of pyFAI's
//! `PixelSplitting.ipynb` tutorial.
//!
//! On a coarse detector each pixel subtends a wide range of radial / azimuthal
//! angle, so the pixel-splitting scheme matters: `no` drops a whole pixel in the
//! single bin its center lands in, `bbox` spreads it over the bins its
//! center±half-width box overlaps, and `full` clips its four corners against the
//! bin grid. The notebook plots the polar pixel layout (using
//! `array_from_unit(typ="corner")` + `splitPixel.recenter`) and the four cakes
//! side by side.
//!
//! Here we build a fully synthetic coarse [`AzimuthalIntegrator`] (a 32x32, 1 mm
//! generic detector at a short distance — no `.poni`, no data file) and run
//! `integrate2d` with `no` / `bbox` / `full` splitting, printing how many cake
//! cells each scheme fills and how the intensity spreads. **Two simplifications
//! vs the notebook:** (1) pyFAI's fourth scheme `pseudo` (scaled bbox) is not
//! ported in rsFAI (only `no`/`bbox`/`full`), so it is omitted; (2) the
//! per-pixel corner-recenter / area-sign internals at the chi discontinuity are
//! not public (the corner array is built internally by the bbox/full engines),
//! so the split effect is shown through the public `integrate2d` API rather than
//! the polygon plot. The matplotlib plotting is dropped.
//!
//!   cargo run --release --example pixel_splitting -p rsfai

use rsfai::{
    Algo, AzimuthalIntegrator, DetectorModel, IntegrationOptions, Method, RadialUnit, Split,
};

/// A coarse 32x32 detector of 1 mm pixels, beam roughly centered, short distance
/// so each pixel covers many radial bins (the regime where splitting bites).
fn build_ai() -> AzimuthalIntegrator {
    let n = 32;
    let pixel = 1e-3;
    let detector = DetectorModel::generic(pixel, pixel, (n, n));
    AzimuthalIntegrator {
        detector,
        dist: 5e-2,
        // Beam center near the middle of the detector (PONI in metres).
        poni1: pixel * n as f64 / 2.0,
        poni2: pixel * n as f64 / 2.0,
        rot1: 0.0,
        rot2: 0.0,
        rot3: 0.0,
        wavelength: 1e-10,
    }
}

/// A deterministic single-ring frame on a `(rows, cols)` detector (flat baseline
/// + one Gaussian ring around the beam center).
fn synth_ring(shape: (usize, usize)) -> Vec<f32> {
    let (rows, cols) = shape;
    let (cy, cx) = (rows as f64 / 2.0, cols as f64 / 2.0);
    let mut img = vec![0.0f32; rows * cols];
    for r in 0..rows {
        for c in 0..cols {
            let rad = ((r as f64 - cy).powi(2) + (c as f64 - cx).powi(2)).sqrt();
            let ring = 1000.0 * (-((rad - 8.0) / 2.5).powi(2)).exp();
            img[r * cols + c] = (5.0 + ring) as f32;
        }
    }
    img
}

fn run(ai: &AzimuthalIntegrator, img: &[f32], split: Split, label: &str) {
    let opts = IntegrationOptions {
        correct_solid_angle: true,
        method: Method {
            split,
            algo: Algo::Histogram,
        },
        ..Default::default()
    };
    let (npt_rad, npt_azim) = (40, 36);
    let r = ai.integrate2d(img, npt_rad, npt_azim, RadialUnit::Q_NM_INV, &opts);

    // How many cake cells received at least one valid pixel, and the total
    // signal/peak — `no` fills fewer cells (each pixel lands in one bin), `bbox`
    // and `full` spread each pixel across the bins it overlaps.
    let filled = r.count.iter().filter(|&&c| c > 0.0).count();
    let total_count: f64 = r.count.iter().sum();
    let peak = (0..r.intensity.len())
        .max_by(|&a, &b| r.intensity[a].total_cmp(&r.intensity[b]))
        .unwrap();
    let (nrad, _) = r.bins;
    let (rad_i, azim_j) = (peak % nrad, peak / nrad);
    println!(
        "  {label:5}: filled cells = {filled:4} / {}   sum(count) = {total_count:8.2}   \
         peak  q = {:.4} nm^-1, chi = {:7.2} deg   I = {:.3}",
        npt_rad * npt_azim,
        r.radial[rad_i],
        r.azimuthal[azim_j],
        r.intensity[peak],
    );
}

fn main() {
    let ai = build_ai();
    let shape = ai.detector.shape;
    let img = synth_ring(shape);

    println!(
        "Pixel splitting on a coarse synthetic detector: {shape:?} generic, 1 mm pixels, \
         dist = {:.3} m",
        ai.dist
    );
    println!("  integrate2d -> 40 radial x 36 azimuth, unit q_nm^-1, histogram engine");
    println!("  (pyFAI's 4th scheme `pseudo` is not ported; showing no / bbox / full)\n");

    run(&ai, &img, Split::No, "no");
    run(&ai, &img, Split::Bbox, "bbox");
    run(&ai, &img, Split::Full, "full");

    println!(
        "\n  `no` fills the fewest cells (one bin per pixel); `bbox`/`full` spread each\n  \
         coarse pixel across the bins it overlaps, filling more cells and smoothing the ring."
    );
}
