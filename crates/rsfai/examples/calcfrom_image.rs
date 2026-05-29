//! Reconstruct a 2D detector image from a 1D profile — pyFAI's
//! `Geometry.calcfrom1d` and `Calibrant.fake_calibration_image`, the inverse of
//! azimuthal integration. Mirrors the calcfrom legs of pyFAI's `geometry` and
//! `Calibrant` tutorials, offline:
//!
//!   1. `calcfrom1d`: given a 1D radial profile `(2θ, intensity)`, interpolate an
//!      intensity onto every pixel's 2θ, apply the solid-angle correction, and
//!      print the reconstructed image stats. Bit-exact vs pyFAI (the golden
//!      verifier `golden_calcfrom.rs` checks all four variants).
//!   2. `fake_calibration_image`: synthesize a LaB6 powder-ring image for the
//!      same geometry from the committed `LaB6.D` d-spacings.
//!
//! Run: `cargo run --release --example calcfrom_image`

use std::f64::consts::PI;
use std::path::PathBuf;

use rsfai::{fake_calibration_image, AzimuthalIntegrator, Calcfrom1dOptions};
use rsfai_calibrant::Calibrant;
use rsfai_detectors::Detector;
use rsfai_geometry::Unit;

/// Generic 128×128 detector, 100 µm pixels, orientation 3 (pyFAI's `Detector`
/// default), at a short distance so several LaB6 rings land on it.
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

fn stats(img: &[f64]) -> (f64, f64, f64) {
    let min = img.iter().copied().fold(f64::INFINITY, f64::min);
    let max = img.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let mean = img.iter().sum::<f64>() / img.len() as f64;
    (min, max, mean)
}

fn main() {
    let ai = build_ai();
    let (ny, nx) = ai.detector.shape;
    let tth = ai.array_from_unit(Unit::TTH_DEG);
    let tth_min = tth.iter().copied().fold(f64::INFINITY, f64::min);
    let tth_max = tth.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    println!("detector {ny}x{nx}, 2θ range on it: [{tth_min:.3}, {tth_max:.3}] deg\n");

    // ---- 1. calcfrom1d: a synthetic 1D profile back-projected onto pixels ----
    // Ascending 40-point profile over [5, 20] deg; pixels below 5° clamp to the
    // first value, the corners above 20° to the last (numpy.interp endpoints).
    let n = 40usize;
    let lo = 5.0;
    let hi = 20.0;
    let step = (hi - lo) / (n - 1) as f64;
    let profile_tth: Vec<f64> = (0..n).map(|i| lo + i as f64 * step).collect();
    let profile_i: Vec<f64> = profile_tth.iter().map(|&t| 1.0 + 0.5 * (t).sin()).collect();

    let img = ai.calcfrom1d(
        &profile_tth,
        &profile_i,
        Unit::TTH_DEG,
        Calcfrom1dOptions {
            correct_solid_angle: true,
            ..Default::default()
        },
    );
    let (mn, mx, mean) = stats(&img);
    println!("calcfrom1d: reconstructed {ny}x{nx} image from a {n}-pt 1D profile");
    println!("  intensity  min={mn:.6}  max={mx:.6}  mean={mean:.6}");
    println!("  (inverse of integrate1d; bit-exact vs pyFAI's Geometry.calcfrom1d)\n");

    // ---- 2. fake_calibration_image: LaB6 powder rings on the same geometry ----
    let dpath =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../golden/datasets_calibrant/LaB6.D");
    let mut lab6 = Calibrant::load_file(&dpath).expect("load committed LaB6.D");
    lab6.set_wavelength(1e-10);
    let energy_kev = lab6.energy().expect("wavelength set");
    let rings_in_range = lab6
        .get_2th()
        .iter()
        .filter(|&&t| {
            let deg = t * 180.0 / PI;
            deg >= tth_min && deg <= tth_max
        })
        .count();
    println!(
        "LaB6 calibrant: {} d-spacings, {rings_in_range} rings within the detector's 2θ range",
        lab6.dspacing().len()
    );
    println!("  wavelength = 1e-10 m  (energy ≈ {energy_kev:.3} keV)");

    let cal_img = fake_calibration_image(&lab6, &ai, 1.0, 0.1, 0.1);
    let (mn, mx, mean) = stats(&cal_img);
    println!("fake_calibration_image: synthesized {ny}x{nx} LaB6 image");
    println!("  intensity  min={mn:.6}  max={mx:.6}  mean={mean:.6}");
}
