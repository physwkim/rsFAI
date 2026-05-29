//! Peak finding end to end — the Rust analogue of pyFAI's peak-picking /
//! `InverseWatershed` / `BlobDetection` / ellipse-fit cookbook.
//!
//! Fully offline and self-contained: the input images are synthesised in Rust
//! (the same deterministic Gaussian-blob / ring constructions the golden
//! generator uses), so no data files or network are touched.
//!
//!   1. Build a multi-Gaussian image, run the inverse watershed, and extract the
//!      peak coordinates with sub-pixel refinement.
//!   2. Label the bright pixels of that image (8-connectivity) and report the
//!      connected-component count.
//!   3. Fit an ellipse to a ring of sampled points and report its centre, axes,
//!      and orientation.
//!
//! The numerical parity check against pyFAI lives in the golden verifier test
//! (`tests/golden_peaks.rs`), not here.
//!
//!   cargo run --release --example peak_finding -p rsfai-peaks

use std::f64::consts::PI;

use rsfai_peaks::{fit_ellipse, label, InverseWatershed, Structure};

/// Synthesise a row-major `f32` image as a sum of isotropic Gaussians.
fn gaussian_image(rows: usize, cols: usize, peaks: &[(f64, f64, f64, f64)]) -> Vec<f32> {
    let mut img = vec![0f32; rows * cols];
    for r in 0..rows {
        for c in 0..cols {
            let mut v = 0f64;
            for &(yc, xc, sigma, amp) in peaks {
                let dy = r as f64 - yc;
                let dx = c as f64 - xc;
                v += amp * (-(dy * dy + dx * dx) / (2.0 * sigma * sigma)).exp();
            }
            img[r * cols + c] = v as f32;
        }
    }
    img
}

fn main() {
    let (rows, cols) = (64usize, 64usize);
    let peaks = [
        (16.0, 16.0, 4.0, 100.0),
        (16.0, 48.0, 4.0, 80.0),
        (48.0, 16.0, 4.0, 90.0),
        (48.0, 48.0, 4.0, 120.0),
        (32.0, 32.0, 3.0, 60.0),
    ];
    let img = gaussian_image(rows, cols, &peaks);

    // ---- 1. Inverse watershed peak extraction --------------------------------
    let mut iw = InverseWatershed::new(img.clone(), rows, cols);
    iw.init();
    let n_regions: std::collections::BTreeSet<i32> = iw.regions.values().map(|r| r.index).collect();
    println!(
        "Inverse watershed: {} catchment region(s) over a {rows}x{cols} image",
        n_regions.len()
    );

    let mask = vec![true; rows * cols];
    let found = iw.peaks_from_area(&mask, Some(10.0), Some(10), true, 0.0);
    println!("  extracted {} peak(s) (Imin=10, refined):", found.len());
    for (y, x) in &found {
        println!("    (y={y:6.2}, x={x:6.2})");
    }

    // ---- 2. Connected-component labelling ------------------------------------
    // Threshold the image into bright blobs and count 8-connected components.
    let thresh = 20.0f32;
    let binary: Vec<bool> = img.iter().map(|&v| v > thresh).collect();
    let (labels, n) = label(&binary, rows, cols, Structure::full());
    let max_label = labels.iter().copied().max().unwrap_or(0);
    println!("\nLabelling (8-conn, threshold {thresh}): {n} component(s), max label {max_label}");

    // ---- 3. Algebraic ellipse fit --------------------------------------------
    // Sample points around an ellipse (the test_utils_ellipse fixture) and fit.
    let n_pts = 32usize;
    let mut pty = Vec::with_capacity(n_pts);
    let mut ptx = Vec::with_capacity(n_pts);
    for i in 0..n_pts {
        let a = 2.0 * PI * i as f64 / n_pts as f64;
        pty.push(a.sin() * 20.0 + 50.0);
        ptx.push(a.cos() * 10.0 + 100.0);
    }
    match fit_ellipse(&pty, &ptx) {
        Ok(e) => {
            println!("\nEllipse fit ({n_pts} points):");
            println!(
                "  center=(y={:.4}, x={:.4})  axes=({:.4}, {:.4})  angle={:.6} rad",
                e.center_1, e.center_2, e.half_long_axis, e.half_short_axis, e.angle
            );
        }
        Err(err) => println!("\nEllipse fit failed: {err:?}"),
    }
}
