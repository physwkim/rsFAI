//! Space-group extinction from cell centering and reflection conditions — the
//! Rust analogue of pyFAI's "Creation of a new calibrant" tutorials
//! (`doc/source/usage/tutorial/Calibrant/new_calibrant.ipynb` and
//! `.../hydrocerussite.ipynb`).
//!
//! Both notebooks build a primitive hexagonal cell and *append a space-group
//! reflection condition* to its `selection_rules`. rsFAI mirrors that exactly:
//! [`Cell::add_selection_rule`] layers an extra rule on top of the centering
//! rule, just as pyFAI's `Cell.selection_rules.append(...)` does.
//!
//!   * **Hydrocerussite, R-3m (space group 166).** `Cell.hexagonal(5.24656,
//!     23.7023)` + `group166_R3bar_m`. The hydrocerussite notebook's own check is
//!     that this produces the *same* reflection set as the built-in
//!     `lattice_type="R"` centering — group 166 adds no extinction beyond R
//!     centering for this metric. This example reproduces that equality.
//!   * **Cr2O3 (eskolaite), R-3c (space group 167).** `Cell.hexagonal(4.958979,
//!     13.59592)` + `group167_R3bar_c`. Here the c-glide *does* extinct beyond R
//!     centering: `000l` is allowed only for `l = 6n`, not the `l = 3n` that bare
//!     R centering permits. This example shows R-3c is a strict subset of the
//!     R-centered set — the extinction the centering rule alone cannot express.
//!
//! The R centering rule rsFAI applies (obverse setting) is `-h + k + l ≡ 0
//! (mod 3)`; the group-166/167 conditions both start from it and then add their
//! special-position rules. The notebooks' `Cell.save(...)` on-disk write is not a
//! landed method, so this prints the d-spacing table instead. Fully offline (no
//! data file). Both reflection sets are gated bit-exact against pyFAI in
//! `tests/golden_calibrant.rs` (cells `Cr2O3_R3c_167`, `hydrocerussite_R3m_166`).
//!
//!   cargo run --release --example space_group -p rsfai-calibrant

use rsfai_calibrant::{group166_r3bar_m, group167_r3bar_c, Cell, Centering, Miller};

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
/// of `calculate_dspacing`). Used to contrast R centering against the R-3c
/// c-glide of identical metric.
fn allows(cell: &mut Cell, h: i64, k: i64, l: i64) -> bool {
    let target = Miller::new(h, k, l);
    cell.calculate_dspacing(1.0)
        .iter()
        .any(|(_, ms)| ms.contains(&target))
}

/// The sorted set of rounded d-spacings a cell admits down to `dmin` (the
/// notebook compares these keys to check two constructions agree).
fn dspacing_keys(cell: &mut Cell, dmin: f64) -> Vec<u64> {
    let mut keys: Vec<u64> = cell
        .calculate_dspacing(dmin)
        .iter()
        .map(|(d, _)| d.to_bits())
        .collect();
    keys.sort_unstable();
    keys
}

fn main() {
    // ---- Hydrocerussite, basic lead carbonate, R-3m (space group 166) ----
    // Primitive hexagonal cell with the group-166 condition appended (the
    // notebook's primary construction), versus the built-in R centering.
    let (a, c) = (5.24656, 23.7023);
    let mut hydroc_166 = Cell::hexagonal(a, c, Centering::P);
    hydroc_166.add_selection_rule(group166_r3bar_m);
    let mut hydroc_r = Cell::hexagonal(a, c, Centering::R);

    let n_166 = hydroc_166.calculate_dspacing(1.0).len();
    let n_r = hydroc_r.calculate_dspacing(1.0).len();
    println!(
        "Hydrocerussite (R-3m, a = {a} A, c = {c} A), dmin = 1.0 A:\n  \
         hexagonal + group166 families = {n_166}\n  \
         built-in R centering families = {n_r}"
    );
    // The hydrocerussite notebook's check: the two constructions agree exactly.
    let same = dspacing_keys(&mut hydroc_166, 1.0) == dspacing_keys(&mut hydroc_r, 1.0);
    println!(
        "  same d-spacing set (notebook's check) = {same}  (group 166 adds no extinction beyond R)"
    );
    assert!(same, "group166 must match bare R centering for R-3m");
    print_dspacing_table("Hydrocerussite (group166)", &mut hydroc_166, 1.0, 8);
    println!();

    // ---- Cr2O3 (eskolaite), R-3c (space group 167): the c-glide ----
    // Primitive hexagonal cell with the group-167 condition appended, versus
    // bare R centering of the same metric. The c-glide tightens 000l to l = 6n.
    let mut crox_167 = Cell::hexagonal(4.958979, 13.59592, Centering::P);
    crox_167.add_selection_rule(group167_r3bar_c);
    let mut crox_r = Cell::hexagonal(4.958979, 13.59592, Centering::R);

    let n_167 = crox_167.calculate_dspacing(1.0).len();
    let n_crox_r = crox_r.calculate_dspacing(1.0).len();
    println!(
        "Cr2O3 (eskolaite, R-3c, a = 4.958979 A, c = 13.59592 A), dmin = 1.0 A:\n  \
         hexagonal + group167 families = {n_167}\n  \
         bare R centering families     = {n_crox_r}  (more: R centering keeps c-glide-extinct reflections)"
    );

    // Reflection-level evidence of the c-glide (000l: l = 6n under R-3c, l = 3n
    // under bare R centering):
    //   (0 0 3): allowed by R centering, extinct under R-3c (3 is not 6n)
    //   (0 0 6): allowed by both (6 = 6n)
    println!("  c-glide on 000l (R-3c requires l = 6n; bare R only l = 3n):");
    for (h, k, l) in [(0, 0, 3), (0, 0, 6), (0, 0, 9), (0, 0, 12)] {
        println!(
            "    ({h} {k} {l}): R-3c allows = {:5}   bare-R allows = {}",
            allows(&mut crox_167, h, k, l),
            allows(&mut crox_r, h, k, l)
        );
    }
    // (0 0 3) and (0 0 9) are the c-glide extinctions R-3c drops but bare R keeps.
    assert!(!allows(&mut crox_167, 0, 0, 3) && allows(&mut crox_r, 0, 0, 3));
    assert!(!allows(&mut crox_167, 0, 0, 9) && allows(&mut crox_r, 0, 0, 9));
    assert!(allows(&mut crox_167, 0, 0, 6) && allows(&mut crox_167, 0, 0, 12));
    println!();
    print_dspacing_table("Cr2O3 (group167, R-3c)", &mut crox_167, 1.0, 8);
    println!("\n  (R-3c drops 000l with l != 6n -- the c-glide bare R centering cannot express)");
}
