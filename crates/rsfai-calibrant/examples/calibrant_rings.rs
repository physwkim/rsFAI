//! Calibrant ring positions end to end — the Rust analogue of pyFAI's calibrant
//! cookbook (`Calibrant.get_2th` / `get_peaks`, and the `Cell` -> d-spacing path).
//!
//! Two halves, both fully offline against committed data:
//!   1. Load a shipped `.D` d-spacing file (LaB6), set a wavelength, and list the
//!      Bragg-law ring positions (2θ in degrees, and q in nm⁻¹).
//!   2. Rebuild the same calibrant's lattice from its cell parameters (a cubic
//!      `Cell`) and show that `calculate_dspacing` reproduces the d-spacings the
//!      `.D` file ships — the lattice -> ring chain.
//!
//! The numerical parity check against pyFAI lives in the golden verifier test
//! (`tests/golden_calibrant.rs`), not here.
//!
//!   cargo run --release --example calibrant_rings -p rsfai-calibrant

use std::path::PathBuf;

use rsfai_calibrant::{Calibrant, Cell, Centering, PeakUnit};

/// Cu Kα1, the classic lab-source wavelength (meters).
const WAVELENGTH: f64 = 1.5406e-10;

fn datasets_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../golden/datasets_calibrant")
}

fn main() {
    // ---- 1. From the shipped .D file: rings at a given wavelength ----
    let dpath = datasets_root().join("LaB6.D");
    let mut lab6 = Calibrant::load_file(&dpath).expect("load committed LaB6.D");
    println!(
        "LaB6 calibrant: {} d-spacings parsed from {}",
        lab6.dspacing().len(),
        dpath.display()
    );

    lab6.set_wavelength(WAVELENGTH);
    let energy = lab6.energy().unwrap();
    let tth_deg = lab6.get_peaks(PeakUnit::TwoThetaDeg);
    let q_nm = lab6.get_peaks(PeakUnit::QNm);
    println!(
        "  wavelength = {WAVELENGTH:.5e} m  (energy = {energy:.4} keV)\n  visible rings = {}  (d too small to diffract -> dropped: {})\n",
        tth_deg.len(),
        lab6.out_dspacing().len()
    );
    println!("  ring   2theta (deg)      q (nm^-1)      d (A)");
    let dsp = lab6.dspacing();
    for i in 0..tth_deg.len().min(10) {
        println!(
            "  {:3}    {:11.5}    {:11.5}    {:9.5}",
            i + 1,
            tth_deg[i],
            q_nm[i],
            dsp[i]
        );
    }
    if tth_deg.len() > 10 {
        println!("  ... ({} more rings)", tth_deg.len() - 10);
    }

    // ---- 2. From the cell parameters: lattice -> d-spacings ----
    // LaB6 is a primitive cubic cell, a = 4.1568 A (NIST SRM 660c).
    let mut cell = Cell::cubic(4.1568, Centering::P);
    let groups = cell.calculate_dspacing(1.0);
    let mut keyed = groups;
    keyed.sort_by(|a, b| b.0.total_cmp(&a.0));
    println!(
        "\nLaB6 from cell (primitive cubic a = 4.1568 A): {} unique d-spacings down to dmin = 1.0 A",
        keyed.len()
    );
    println!("  d (A)        multiplicity   first Miller (h k l)");
    for (d, millers) in keyed.iter().take(6) {
        let m = millers.last().unwrap();
        println!(
            "  {:9.5}    {:4}           ({} {} {})",
            d,
            millers.len(),
            m.h,
            m.k,
            m.l
        );
    }

    // Sanity: the largest cell d-spacing equals the (1 0 0) plane spacing = a.
    let d100 = keyed.first().map(|(d, _)| *d).unwrap_or(0.0);
    println!("\n  largest d-spacing (1 0 0) = {d100:.5} A  (== lattice constant a)");
    assert!(
        (d100 - 4.1568).abs() < 1e-4,
        "(1 0 0) d-spacing should equal a"
    );
}
