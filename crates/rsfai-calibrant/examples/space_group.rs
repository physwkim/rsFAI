//! Space-group extinction from cell centering — the Rust analogue of pyFAI's
//! "Creation of a new calibrant" tutorials
//! (`doc/source/usage/tutorial/Calibrant/new_calibrant.ipynb` and
//! `.../hydrocerussite.ipynb`).
//!
//! Those notebooks build an R-centered (corundum-type) hexagonal cell two ways
//! and show they agree: a plain hexagonal cell whose `selection_rules` get an
//! appended custom `reflection_condition_166` closure, versus
//! `Cell.hexagonal(a, c, lattice_type="R")` which carries the R centering
//! built in. The hydrocerussite notebook's own check is that the two produce the
//! *same* reflection set.
//!
//! rsFAI exposes the built-in centerings (P/I/F/A/B/C/R) but NOT the notebooks'
//! primary mechanism — appending an arbitrary user closure to `selection_rules`
//! (the `extra_rules` field is private, with no public append). So:
//!   * Hydrocerussite **R-3m (space group 166)** is reproduced exactly here via
//!     the landed `Centering::R` path — that is the case the hydrocerussite
//!     notebook validates against its custom rule.
//!   * Cr2O3 **R-3c (space group 167)** is NOT reproduced: its glide adds an
//!     extinction beyond R centering, and that extra rule is not a landed API.
//!     We list its bare R-centered cell for the d-spacings but do not claim the
//!     full 167 condition.
//!
//! The R centering rule rsFAI applies (obverse setting) is `-h + k + l ≡ 0
//! (mod 3)`; this example shows it extincts a subset of the primitive cell's
//! reflections, which is exactly the calibrant ring list a `.D` file records.
//! The notebooks' `Cell.save(...)` on-disk write is not a landed method, so this
//! prints the table instead. Fully offline (no data file).
//!
//!   cargo run --release --example space_group -p rsfai-calibrant

use rsfai_calibrant::{Cell, Centering, Miller};

/// Print a cell's d-spacing groups (descending d): rounded d-spacing,
/// multiplicity, and the family's representative Miller index.
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

/// Whether a cell admits the given reflection family (it appears in some group
/// of `calculate_dspacing`). Used to contrast R centering against the primitive
/// cell of identical metric.
fn allows(cell: &mut Cell, h: i64, k: i64, l: i64) -> bool {
    let target = Miller::new(h, k, l);
    cell.calculate_dspacing(1.0)
        .iter()
        .any(|(_, ms)| ms.contains(&target))
}

fn main() {
    // ---- Hydrocerussite, basic lead carbonate, R-3m (space group 166) ----
    // The corundum-type hexagonal setting a = 5.24656 A, c = 23.7023 A; the R
    // centering carries the group-166 reflection condition the notebook applies
    // by an appended custom rule.
    let (a, c) = (5.24656, 23.7023);
    let mut hydroc_r = Cell::hexagonal(a, c, Centering::R);
    let mut hydroc_p = Cell::hexagonal(a, c, Centering::P);

    let n_r = hydroc_r.calculate_dspacing(1.0).len();
    let n_p = hydroc_p.calculate_dspacing(1.0).len();
    println!(
        "Hydrocerussite (R-3m, a = {a} A, c = {c} A), dmin = 1.0 A:\n  \
         R-centered families  = {n_r}\n  primitive (P) families = {n_p}  \
         (R centering extincts the rest: -h+k+l != 3n)"
    );
    print_dspacing_table("Hydrocerussite (R)", &mut hydroc_r, 1.0, 8);
    println!();

    // Reflection-level evidence of the R rule (-h + k + l ≡ 0 mod 3):
    //   (1 0 0): -1      -> extinct in R, present in P
    //   (0 0 1):  1      -> extinct in R, present in P
    //   (1 1 0):  0      -> allowed in both
    //   (0 0 3):  3      -> allowed in both
    println!("  R-centering extinction (-h+k+l mod 3):");
    for (h, k, l) in [(1, 0, 0), (0, 0, 1), (1, 1, 0), (0, 0, 3)] {
        println!(
            "    ({h} {k} {l}): -h+k+l = {:3}   R allows = {:5}   P allows = {}",
            -h + k + l,
            allows(&mut hydroc_r, h, k, l),
            allows(&mut hydroc_p, h, k, l)
        );
    }
    assert!(!allows(&mut hydroc_r, 1, 0, 0) && allows(&mut hydroc_p, 1, 0, 0));
    assert!(!allows(&mut hydroc_r, 0, 0, 1) && allows(&mut hydroc_p, 0, 0, 1));
    assert!(allows(&mut hydroc_r, 1, 1, 0) && allows(&mut hydroc_r, 0, 0, 3));
    println!();

    // ---- Cr2O3 (eskolaite), R-3c (space group 167): R centering only ----
    // The notebook starts from Cell.hexagonal(4.958979, 13.59592) and appends a
    // custom reflection_condition_167 (the c-glide on top of R centering). That
    // appended-rule API is not landed, so we show only the R-centered cell and
    // do NOT claim the full 167 condition.
    let mut crox_r = Cell::hexagonal(4.958979, 13.59592, Centering::R);
    let n_crox = crox_r.calculate_dspacing(1.0).len();
    println!(
        "Cr2O3 (eskolaite), hexagonal a = 4.958979 A, c = 13.59592 A, R centering only\n  \
         R-centered families = {n_crox} at dmin = 1.0 A\n  \
         NOTE: full R-3c (group 167) adds a c-glide extinction beyond R centering;\n  \
         that custom rule is not a landed API, so this is the R-centered subset only."
    );
    print_dspacing_table("Cr2O3 (R only)", &mut crox_r, 1.0, 6);
}
