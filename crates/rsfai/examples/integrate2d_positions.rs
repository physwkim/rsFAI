//! 2D integration in a non-azimuthal (rectilinear `qx`/`qy`) space — the Rust
//! analogue of pyFAI's `integrate2d.ipynb` tutorial.
//!
//! The notebook integrates a frame into `qx/qy` reciprocal space (instead of the
//! usual `2theta/chi` polar cake) by registering the `qx_nm^-1` / `qy_nm^-1`
//! units and calling `ai.integrate2d(img, 400, 400, unit=("qx_nm^-1","qy_nm^-1"))`.
//! pyFAI implements this as an ordinary no-split 2D histogram whose two axes are
//! arbitrary per-pixel position arrays — which is exactly
//! [`AzimuthalIntegrator::integrate2d_positions`].
//!
//! We compute the two position arrays here from the true scattering geometry
//! ([`AzimuthalIntegrator::pixel_positions`]) using pyFAI's verbatim `qx`/`qy`
//! formulas (`src/pyFAI/units.py`):
//!
//!   qx = 4e-9·π/λ · sin(atan2(x, z) / 2)
//!   qy = 4e-9·π/λ · sin(atan2(y, z) / 2)
//!
//! and feed them to `integrate2d_positions`, mirroring the notebook's `qx/qy`
//! cell. The notebook's later `register_radial_unit("tthx_deg", ...)` numexpr
//! custom-unit cells are out of remit and are not attempted; the matplotlib
//! plotting and the fabio `moke.tif` read are dropped (we synthesize the frame).
//!
//! Runs fully offline: it loads the committed Pilatus1M geometry from
//! `golden/datasets/.../geometry.poni` and synthesizes a deterministic
//! powder-ring frame.
//!
//!   cargo run --release --example integrate2d_positions -p rsfai

use std::path::PathBuf;

use rsfai::{AzimuthalIntegrator, Corrections, IntegrationOptions, Positions2dBinning};

/// `4e-9·π` — the leading constant of pyFAI's `q` formulas (`q` in nm⁻¹).
const FOUR_PI_E_NEG9: f64 = 4.0e-9 * std::f64::consts::PI;

/// A deterministic concentric-ring powder pattern (flat baseline + two Gaussian
/// rings centered on the detector).
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
    let poni = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(
        "../../golden/datasets/\
         Pilatus1M__bbox-csr-cython__2th_deg__npt1000__errpoisson/geometry.poni",
    );
    let ai = AzimuthalIntegrator::load(&poni).expect("load committed .poni");

    let shape = ai.detector.shape;
    let image = synth_powder(shape);

    // The true scattering geometry: lab coords (z, y, x) per pixel. pyFAI's qx/qy
    // formulas read x (fast), y (slow), z (along beam).
    let pos = ai.pixel_positions();
    let wavelength = ai.wavelength;
    let lead = FOUR_PI_E_NEG9 / wavelength;
    let n = ai.detector.size();
    let mut qx = Vec::with_capacity(n);
    let mut qy = Vec::with_capacity(n);
    for i in 0..n {
        let (x, y, z) = (pos.x[i], pos.y[i], pos.z[i]);
        qx.push(lead * (x.atan2(z) / 2.0).sin());
        qy.push(lead * (y.atan2(z) / 2.0).sin());
    }

    // qx/qy are signed (`positive=False` in pyFAI) and carry no period, so they
    // never wrap — the two flags `integrate2d_positions` reads off the unit.
    // scale = 1.0: the formulas already yield nm⁻¹.
    let binning = Positions2dBinning {
        npt0: 400,
        npt1: 400,
        pos0_range: None,
        pos1_range: None,
        pos0_scale: 1.0,
        pos1_scale: 1.0,
        allow_pos0_neg: true,
        pos1_period: 0.0,
    };
    let opts = IntegrationOptions {
        correct_solid_angle: true,
        ..Default::default()
    };
    let corr = Corrections::with_normalization(1.0);

    println!("AzimuthalIntegrator: {shape:?} Pilatus1M, wavelength = {wavelength:.3e} m");
    println!(
        "  qx in [{:.4}, {:.4}] nm^-1   qy in [{:.4}, {:.4}] nm^-1",
        qx.iter().cloned().fold(f64::INFINITY, f64::min),
        qx.iter().cloned().fold(f64::NEG_INFINITY, f64::max),
        qy.iter().cloned().fold(f64::INFINITY, f64::min),
        qy.iter().cloned().fold(f64::NEG_INFINITY, f64::max),
    );
    println!(
        "  no-split histogram on (qx, qy), {} x {} bins\n",
        binning.npt0, binning.npt1
    );

    let r = ai.integrate2d_positions(&image, &qx, &qy, &binning, &opts, &corr);
    let cell = (0..r.intensity.len())
        .max_by(|&a, &b| r.intensity[a].total_cmp(&r.intensity[b]))
        .unwrap();
    let (n0, _n1) = r.bins;
    let (i0, i1) = (cell % n0, cell / n0);
    let valid: f64 = r.count.iter().sum();
    println!(
        "integrate2d_positions -> {} (qx) x {} (qy) cells",
        r.bins.0, r.bins.1
    );
    println!(
        "  qx axis  [{:.4}, {:.4}] nm^-1",
        r.radial.first().unwrap(),
        r.radial.last().unwrap()
    );
    println!(
        "  qy axis  [{:.4}, {:.4}] nm^-1",
        r.azimuthal.first().unwrap(),
        r.azimuthal.last().unwrap()
    );
    println!("  valid pixels binned = {valid:.0}");
    println!(
        "  brightest cell  qx = {:.4}, qy = {:.4} nm^-1   I = {:.3}",
        r.radial[i0], r.azimuthal[i1], r.intensity[cell]
    );
}
