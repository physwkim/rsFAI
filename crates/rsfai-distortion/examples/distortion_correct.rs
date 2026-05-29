//! Spline distortion correction end to end — the Rust analogue of pyFAI's
//! `Distortion` / `Spline` workflow (`pyFAI.distortion.Distortion`,
//! `pyFAI.spline.Spline`, `pyFAI.ext._distortion`).
//!
//! Two halves, both fully offline:
//!
//!   1. Parse the committed FReLoN `halfccd.spline` (FITPACK ASCII), report the
//!      grid metadata, and evaluate the bicubic X/Y displacement surfaces with
//!      [`Spline::spline2array`]. We print the map dimensions and a sampled
//!      displacement so the bisplev tensor-product evaluation is visible.
//!
//!   2. Build a distortion look-up table on a small synthetic grid: each input
//!      pixel is given sheared corner positions (a uniform shift in dim1 that
//!      grows with the column), [`calc_pos`] places them on the output grid,
//!      [`calc_sparse`] clips every pixel polygon into a CSR fractional-area
//!      LUT, and [`correct`] remaps a flat input image. Because the LUT rows are
//!      a partition of unity over a flat field, the corrected interior sums back
//!      to the input level — we check that and report it.
//!
//! The numerical parity check against pyFAI lives in the golden verifier test
//! (`tests/golden_distortion.rs`), not here.
//!
//!   cargo run --release --example distortion_correct -p rsfai-distortion

use std::path::PathBuf;

use rsfai_distortion::{calc_pos, calc_sparse, correct, Spline};

fn spline_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../golden/datasets_distortion/halfccd.spline")
}

/// Synthetic input pixel corners on a `shape_in` grid: an axis-aligned unit
/// pixel at `(i, j)` sheared in dim1 by `shear * j` (a smooth, monotone
/// distortion). Layout matches `calc_pos`'s expected `(nrow, ncol, 4, 3)`
/// flatten with vertex order ABCD and component order (z, y, x); z is unused.
fn synth_corners(shape_in: (usize, usize), shear: f32) -> Vec<f32> {
    let (nrow, ncol) = shape_in;
    let mut corners = vec![0.0f32; nrow * ncol * 4 * 3];
    for i in 0..nrow {
        for j in 0..ncol {
            let y0 = i as f32;
            let x0 = j as f32 + shear * j as f32;
            // ABCD counter-clockwise: (y0,x0) (y0+1,x0) (y0+1,x0+1) (y0,x0+1).
            let verts = [
                (y0, x0),
                (y0 + 1.0, x0),
                (y0 + 1.0, x0 + 1.0),
                (y0, x0 + 1.0),
            ];
            let base = (i * ncol + j) * 4 * 3;
            for (k, (vy, vx)) in verts.iter().enumerate() {
                let off = base + k * 3;
                corners[off] = 0.0; // z
                corners[off + 1] = *vy; // y
                corners[off + 2] = *vx; // x
            }
        }
    }
    corners
}

fn main() {
    // -- Part 1: parse the spline and evaluate the displacement surfaces -------
    let sp = Spline::read(spline_path()).expect("parse halfccd.spline");
    println!("halfccd.spline:");
    println!(
        "  valid region  x:[{}, {}]  y:[{}, {}]",
        sp.xmin, sp.xmax, sp.ymin, sp.ymax
    );
    println!(
        "  grid spacing  {}  pixel size (x,y) = ({}, {}) um  order {}",
        sp.grid, sp.pixel_size.0, sp.pixel_size.1, sp.order
    );

    let (xdisp, ydisp) = sp.spline2array();
    let rows = (sp.ymax - sp.ymin) as usize + 1;
    let cols = (sp.xmax - sp.xmin) as usize + 1;
    println!(
        "  spline2array displacement maps: {rows} x {cols} ({} values each)",
        xdisp.len()
    );
    // Sample the displacement at the grid centre.
    let mid = (rows / 2) * cols + cols / 2;
    println!(
        "  centre displacement (xDisp, yDisp) = ({}, {}) px",
        xdisp[mid], ydisp[mid]
    );

    // -- Part 2: distortion LUT build + correct on a synthetic grid -----------
    let shape_in = (16usize, 16usize);
    let shear = 0.25f32;
    let corners = synth_corners(shape_in, shear);

    // Output grid sized from the corner bounding box (resize=True equivalent).
    let cp = calc_pos(&corners, shape_in, 1.0, 1.0, None);
    println!(
        "\ndistortion LUT: shape_in {:?} -> shape_out {:?}  (delta {:?})",
        shape_in, cp.shape_out, cp.delta
    );

    let csr = calc_sparse(&cp, None, (0.0, 0.0));
    println!(
        "  CSR: {} non-zero coefficients over {} output bins",
        csr.data.len(),
        csr.indptr.len() - 1
    );

    // Flat input field of 100.0: a flat image distorted then corrected should,
    // on interior bins fully covered by the partition-of-unity LUT, return 100.
    let image = vec![100.0f32; shape_in.0 * shape_in.1];
    let out = correct(&image, &csr, 0.0);

    // Report how many output bins recovered the flat level (within f32 noise)
    // versus partial-coverage edge bins.
    let (mut full, mut edge, mut empty) = (0usize, 0usize, 0usize);
    for &v in &out {
        if v == 0.0 {
            empty += 1;
        } else if (v - 100.0).abs() < 1e-2 {
            full += 1;
        } else {
            edge += 1;
        }
    }
    println!(
        "  correct: {} bins recovered 100.0 (full coverage), {} partial edge, {} empty",
        full, edge, empty
    );

    assert!(
        full > 0,
        "expected at least one fully-covered bin to recover the flat field"
    );
    println!("\nOK");
}
