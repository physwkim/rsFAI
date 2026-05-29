//! Detector orientation management — the Rust analogue of pyFAI's
//! `Orientation.ipynb` tutorial.
//!
//! A detector frame can be stored in any of 4 flip orientations (EXIF 1-4): the
//! origin sits at a different corner. pyFAI's default is **3** (bottom-right seen
//! from behind). The notebook builds the same physical pattern in all 4 flips —
//! `flipud` (3→2), `fliplr` (3→4), `fliplr(flipud(.))` (3→1) — sets the detector
//! to the matching `orientation`, and shows that the azimuthal integration is
//! invariant: the 1D radial curve is identical and the 2D cake only mirrors /
//! offsets in azimuth, the radial peak bin staying put.
//!
//! Here we synthesize one asymmetric frame in the orientation-3 view, build the
//! three flipped variants, build a Pilatus1M [`AzimuthalIntegrator`] at each of
//! the 4 orientations, run `integrate2d` for each, and print evidence that the
//! cakes coincide — the 1D-equivalent radial peak (sum over azimuth) lands in the
//! same radial bin for all four. The matplotlib 2x2 plot grid and the fabio EDF
//! read are dropped.
//!
//! Runs fully offline: it loads the committed Pilatus1M geometry from
//! `golden/datasets/.../geometry.poni` and synthesizes the frame.
//!
//!   cargo run --release --example orientation -p rsfai

use std::path::PathBuf;

use rsfai::{Algo, AzimuthalIntegrator, IntegrationOptions, Method, RadialUnit, Split};

/// An *asymmetric* deterministic frame on a `(rows, cols)` detector: a single
/// off-center Gaussian ring plus a bright corner blob, so flips actually move the
/// data (a symmetric pattern would be invariant under flipud/fliplr and prove
/// nothing). This is the orientation-3 (default) view of the pattern.
fn synth_asym(shape: (usize, usize)) -> Vec<f32> {
    let (rows, cols) = shape;
    // Ring centered off the geometric center; blob near the top-left corner.
    let (ry, rx) = (rows as f64 * 0.4, cols as f64 * 0.45);
    let (by, bx) = (rows as f64 * 0.12, cols as f64 * 0.15);
    let mut img = vec![0.0f32; rows * cols];
    for r in 0..rows {
        for c in 0..cols {
            let rad = ((r as f64 - ry).powi(2) + (c as f64 - rx).powi(2)).sqrt();
            let ring = 800.0 * (-((rad - 220.0) / 30.0).powi(2)).exp();
            let blob = 500.0
                * (-(((r as f64 - by) / 40.0).powi(2) + ((c as f64 - bx) / 40.0).powi(2))).exp();
            img[r * cols + c] = (10.0 + ring + blob) as f32;
        }
    }
    img
}

/// Flip rows (top<->bottom), numpy `flipud`, for a flat row-major `(rows, cols)`.
fn flipud(img: &[f32], shape: (usize, usize)) -> Vec<f32> {
    let (rows, cols) = shape;
    let mut out = vec![0.0f32; rows * cols];
    for r in 0..rows {
        let src = (rows - 1 - r) * cols;
        out[r * cols..r * cols + cols].copy_from_slice(&img[src..src + cols]);
    }
    out
}

/// Flip columns (left<->right), numpy `fliplr`, for a flat row-major `(rows, cols)`.
fn fliplr(img: &[f32], shape: (usize, usize)) -> Vec<f32> {
    let (rows, cols) = shape;
    let mut out = vec![0.0f32; rows * cols];
    for r in 0..rows {
        for c in 0..cols {
            out[r * cols + c] = img[r * cols + (cols - 1 - c)];
        }
    }
    out
}

fn main() {
    let poni = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(
        "../../golden/datasets/\
         Pilatus1M__bbox-csr-cython__q_nmm1__npt100x36__errpoisson/geometry.poni",
    );
    let ai = AzimuthalIntegrator::load(&poni).expect("load committed .poni");
    let shape = ai.detector.shape;

    // The orientation-3 (default) view, and its three flips (per the notebook):
    //   3 = base, 2 = flipud, 4 = fliplr, 1 = fliplr(flipud).
    let img3 = synth_asym(shape);
    let img2 = flipud(&img3, shape);
    let img4 = fliplr(&img3, shape);
    let img1 = fliplr(&img2, shape);
    let frames: [(i32, &[f32]); 4] = [(1, &img1), (2, &img2), (3, &img3), (4, &img4)];

    // full-split histogram (the notebook's first method) on q_nm^-1.
    let opts = IntegrationOptions {
        correct_solid_angle: true,
        method: Method {
            split: Split::Full,
            algo: Algo::Histogram,
        },
        ..Default::default()
    };
    let (npt_rad, npt_azim) = (500, 360);
    let unit = RadialUnit::Q_NM_INV;

    println!("Orientation invariance: {shape:?} Pilatus1M, unit q_nm^-1, full-split histogram");
    println!(
        "  integrate2d -> {npt_rad} radial x {npt_azim} azimuth, frame flipped to match each orientation\n"
    );

    let mut radial_peak_bins = Vec::new();
    for (o, img) in frames {
        // Same geometry, only the detector orientation differs.
        let mut aio = ai.clone();
        aio.detector.orientation = o;
        let r = aio.integrate2d(img, npt_rad, npt_azim, unit, &opts);
        let (nrad, nazim) = r.bins;

        // The 1D-equivalent radial profile: sum the cake over azimuth, then find
        // its peak radial bin. Orientation invariance ⇒ this bin matches for all.
        let mut radial_profile = vec![0.0f64; nrad];
        for j in 0..nazim {
            for (i, acc) in radial_profile.iter_mut().enumerate() {
                *acc += f64::from(r.intensity[j * nrad + i]);
            }
        }
        let rp_bin = (0..nrad)
            .max_by(|&a, &b| radial_profile[a].total_cmp(&radial_profile[b]))
            .unwrap();

        // The brightest single cake cell, to show the azimuth offset/mirror.
        let cell = (0..r.intensity.len())
            .max_by(|&a, &b| r.intensity[a].total_cmp(&r.intensity[b]))
            .unwrap();
        let (rad_i, azim_j) = (cell % nrad, cell / nrad);

        println!(
            "  orientation {o}: radial-peak bin {rp_bin} (q = {:.4} nm^-1)   \
             brightest cell  q = {:.4}, chi = {:7.2} deg   I = {:.3}",
            r.radial[rp_bin], r.radial[rad_i], r.azimuthal[azim_j], r.intensity[cell]
        );
        radial_peak_bins.push(rp_bin);
    }

    let all_match = radial_peak_bins.iter().all(|&b| b == radial_peak_bins[0]);
    println!(
        "\n  radial-peak bins: {radial_peak_bins:?}  ->  {}",
        if all_match {
            "ALL COINCIDE (orientation-invariant radial integration)"
        } else {
            "DIVERGE"
        }
    );
}
