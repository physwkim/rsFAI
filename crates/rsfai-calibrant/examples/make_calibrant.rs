//! Building calibrant d-spacings from crystal cells — the Rust analogue of
//! pyFAI's "Creation of a calibrant file" tutorial
//! (`doc/source/usage/tutorial/Calibrant/make_calibrant.ipynb`).
//!
//! The notebook builds [`Cell`]s from lattice parameters and writes the
//! resulting d-spacing / Miller table to a `.D` calibrant file via `Cell.save`.
//! That on-disk writer is *not* a landed method here, so this example prints the
//! same table to stdout instead of saving a file. Everything else mirrors the
//! notebook, fully offline (no data file):
//!   1. Polonium — the only element with a primitive simple-cubic packing; one
//!      lattice parameter, every reflection allowed.
//!   2. LaB6 (NIST SRM 660c) — primitive cubic; count the reflections down to a
//!      small dmin (the parameter that controls how many rings a calibrant file
//!      lists).
//!   3. Silicon (diamond-FCC) vs a plain FCC cell of the same parameter — the
//!      diamond glide adds one extinction rule, so reflections like (4 2 0) and
//!      (2 2 2) present in FCC are absent in Si.
//!
//!   cargo run --release --example make_calibrant -p rsfai-calibrant

use rsfai_calibrant::{Cell, Centering, Miller};

/// Print a cell's d-spacing groups (descending d), one row per reflection
/// family: rounded d-spacing, multiplicity, and the family's representative
/// Miller index (pyFAI's `reflection[-1]`, the last after the `(l, k, h)` sort).
fn print_dspacing_table(label: &str, cell: &mut Cell, dmin: f64, max_rows: usize) {
    let mut groups = cell.calculate_dspacing(dmin);
    groups.sort_by(|a, b| b.0.total_cmp(&a.0));
    println!(
        "{label}: {} reflection families down to dmin = {dmin} A",
        groups.len()
    );
    println!("  d (A)         mult   representative (h k l)");
    for (d, millers) in groups.iter().take(max_rows) {
        let m = millers.last().unwrap();
        println!(
            "  {:11.6}   {:4}   ({} {} {})",
            d,
            millers.len(),
            m.h,
            m.k,
            m.l
        );
    }
    if groups.len() > max_rows {
        println!("  ... ({} more families)", groups.len() - max_rows);
    }
}

/// Whether a cell admits the given reflection family at all (it appears in some
/// group of `calculate_dspacing`). Used to contrast Si vs FCC extinctions.
fn allows(cell: &mut Cell, h: i64, k: i64, l: i64) -> bool {
    let target = Miller::new(h, k, l);
    cell.calculate_dspacing(1.0)
        .iter()
        .any(|(_, ms)| ms.contains(&target))
}

fn main() {
    // ---- 1. Polonium: primitive simple cubic, a = 3.359 A ----
    let mut po = Cell::cubic(3.359, Centering::P);
    println!(
        "Polonium (primitive cubic, a = 3.359 A), volume = {:.4} A^3",
        po.volume()
    );
    print_dspacing_table("Polonium", &mut po, 1.0, 8);
    println!();

    // ---- 2. LaB6: primitive cubic, a = 4.156826 A (NIST SRM 660c) ----
    // dmin controls the reflection count: smaller dmin -> more rings.
    let mut lab6 = Cell::cubic(4.156826, Centering::P);
    let n01 = lab6.calculate_dspacing(0.1).len();
    let n10 = lab6.calculate_dspacing(1.0).len();
    println!(
        "LaB6 (primitive cubic, a = 4.156826 A): {n01} families at dmin = 0.1 A, {n10} at dmin = 1.0 A"
    );
    print_dspacing_table("LaB6", &mut lab6, 1.0, 6);
    println!();

    // ---- 3. Silicon (diamond) vs plain FCC of the same parameter ----
    let mut si = Cell::diamond(5.431179);
    let mut fcc = Cell::cubic(5.431179, Centering::F);
    let n_si = si.calculate_dspacing(1.0).len();
    let n_fcc = fcc.calculate_dspacing(1.0).len();
    println!(
        "Silicon (diamond-FCC, a = 5.431179 A) vs plain FCC (same a):\n  Si families  = {n_si}\n  FCC families = {n_fcc}  (FCC has more: the diamond glide extincts some all-even reflections)"
    );

    // The diamond glide forbids all-even reflections with (h+k+l) % 4 != 0.
    // (2 2 2): all even, h+k+l = 6 -> 6 % 4 = 2 != 0 -> extinct in Si, present in FCC.
    // (4 2 0): all even, h+k+l = 6 -> likewise extinct in Si, present in FCC.
    for (h, k, l) in [(1, 1, 1), (2, 2, 2), (4, 2, 0), (4, 0, 0)] {
        println!(
            "  ({h} {k} {l}): Si allows = {:5}   FCC allows = {}",
            allows(&mut si, h, k, l),
            allows(&mut fcc, h, k, l)
        );
    }

    // Sanity: the diamond extinctions Si drops must indeed be present in FCC.
    assert!(!allows(&mut si, 2, 2, 2) && allows(&mut fcc, 2, 2, 2));
    assert!(!allows(&mut si, 4, 2, 0) && allows(&mut fcc, 4, 2, 0));
    assert!(allows(&mut si, 1, 1, 1) && allows(&mut fcc, 1, 1, 1));
    println!("\n  (Si drops (2 2 2) and (4 2 0) which FCC keeps -- the diamond glide rule)");
}
