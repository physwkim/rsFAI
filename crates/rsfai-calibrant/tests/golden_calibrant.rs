//! Calibrant + crystallography parity gate: `rsfai-calibrant` vs pyFAI on
//! `golden/datasets_calibrant/`.
//!
//! `gen_golden_calibrant.py` (daq env, pyFAI 2026.5.0 `-ffp-contract=off`) dumps,
//! for a set of shipped calibrants and a matrix of wavelengths, the d-spacing
//! list, the Bragg 2theta list, and `get_peaks` in four units; and, for the
//! cubic calibrants, the `Cell.calculate_dspacing` d-spacing + multiplicity
//! arrays. The `.D` files themselves are copied into the dataset dir so this
//! test re-parses the identical bytes pyFAI parsed — offline and self-contained.
//!
//! Gates (per `doc/bit-exact-ladder.md`):
//!   * d-spacing (`.D` parse) and the cell d-spacing / multiplicity arrays are
//!     pure parse / `+-*/`-`sqrt` algebra -> [`Gate::Exact`] (0 ULP).
//!   * the Bragg 2theta and the 2theta-derived peaks (`2th_deg`/`2th_rad`) go
//!     through `2*asin(...)` -> [`Gate::Tol`] (Tier-B, the measured max ULP is
//!     printed and asserted under `ASIN_ULP_BUDGET`).
//!   * the q peaks (`q_nm^-1`/`q_A^-1`) are `20*pi/d * scale`, pure arithmetic
//!     -> [`Gate::Exact`].
//!
//! The test fails loudly with a per-field report if any field misses its gate;
//! the worst observed 2theta ULP is printed so a future libm drift is visible.

use std::path::PathBuf;

use rsfai_calibrant::{Calibrant, Cell, Centering, PeakUnit, CONST_HC};
use rsfai_core::compare::compare_f64;
use rsfai_core::golden::{load_npy_f64, load_npy_i32};
use serde_json::Value;

/// ULP budget for the Bragg `2*asin(5e9*lambda/d)` and its derived 2theta peaks
/// (Tier-B transcendental boundary against pyFAI's `math.asin`). On this
/// machine libm `asin` matches numpy/CPython bit-for-bit, so the observed value
/// is recorded by the test output; the budget is the sanctioned ceiling, not a
/// claim of the measured gap.
const ASIN_ULP_BUDGET: u64 = 4;

#[derive(Clone, Copy)]
enum Gate {
    Exact,
    Tol,
}

fn datasets_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../golden/datasets_calibrant")
}

fn manifest() -> Value {
    let p = datasets_root().join("manifest.json");
    let text = std::fs::read_to_string(&p).unwrap_or_else(|e| panic!("read manifest {p:?}: {e}"));
    serde_json::from_str(&text).expect("parse manifest.json")
}

fn f64v(name: &str) -> Vec<f64> {
    let p = datasets_root().join(name);
    load_npy_f64(&p)
        .unwrap_or_else(|e| panic!("load {name}: {e}"))
        .as_slice()
        .expect("C-contiguous")
        .to_vec()
}

fn i32v(name: &str) -> Vec<i32> {
    let p = datasets_root().join(name);
    load_npy_i32(&p)
        .unwrap_or_else(|e| panic!("load {name}: {e}"))
        .as_slice()
        .expect("C-contiguous")
        .to_vec()
}

fn status(bit_exact: bool, ok: bool) -> &'static str {
    if bit_exact {
        "BIT-EXACT"
    } else if ok {
        "ulp-ok   "
    } else {
        "FAIL     "
    }
}

/// Compare an f64 field against its gate; print the report, update the running
/// flags. For [`Gate::Exact`] only bit-exactness clears; for [`Gate::Tol`] the
/// ULP budget also clears.
fn check_f64(
    label: &str,
    got: &[f64],
    golden: &[f64],
    gate: Gate,
    worst_ulp: &mut u64,
    fails: &mut usize,
) {
    if got.len() != golden.len() {
        *fails += 1;
        eprintln!(
            "    {label:34} FAIL      length {} != golden {}",
            got.len(),
            golden.len()
        );
        return;
    }
    let r = compare_f64(got, golden);
    if matches!(gate, Gate::Tol) && r.max_ulp > *worst_ulp {
        *worst_ulp = r.max_ulp;
    }
    let ok = match gate {
        Gate::Exact => r.is_bit_exact(),
        Gate::Tol => r.is_bit_exact() || r.within_ulp(ASIN_ULP_BUDGET),
    };
    if !ok {
        *fails += 1;
    }
    eprintln!(
        "    {label:34} {}  ulp={:4} mism={}/{}",
        status(r.is_bit_exact(), ok),
        r.max_ulp,
        r.bit_mismatches,
        r.total
    );
}

fn peak_unit(name: &str) -> (PeakUnit, Gate) {
    match name {
        "2th_deg" => (PeakUnit::TwoThetaDeg, Gate::Tol),
        "2th_rad" => (PeakUnit::TwoThetaRad, Gate::Tol),
        "q_nm^-1" => (PeakUnit::QNm, Gate::Exact),
        "q_A^-1" => (PeakUnit::QA, Gate::Exact),
        other => panic!("unexpected peak unit in manifest: {other}"),
    }
}

/// The on-disk file suffix for a peak unit (matches the generator's `key`).
fn peak_key(unit: &str) -> String {
    unit.replace('^', "").replace("-1", "m1")
}

#[test]
fn const_hc_matches_pyfai() {
    let m = manifest();
    let golden = m["const_hc"].as_f64().expect("const_hc");
    assert_eq!(
        CONST_HC.to_bits(),
        golden.to_bits(),
        "CONST_HC drift: rust {CONST_HC:?} vs pyFAI {golden:?}"
    );
}

#[test]
fn calibrant_matches_pyfai_golden() {
    let m = manifest();
    let mut worst_ulp = 0u64;
    let mut fails = 0usize;

    // ---- .D-file path: parse, set wavelength, compare d-spacing / 2theta / peaks.
    for cal in m["dot_d"].as_array().expect("dot_d") {
        let name = cal["name"].as_str().unwrap();
        let file = cal["file"].as_str().unwrap();
        let dpath = datasets_root().join(file);
        for case in cal["cases"].as_array().unwrap() {
            let wl = case["wavelength"].as_f64().unwrap();
            let tag = case["tag"].as_str().unwrap();
            eprintln!("=== {name}  wavelength={wl:e}  ({tag}) ===");

            let mut c = Calibrant::load_file(&dpath).expect("load .D");
            c.set_wavelength(wl);

            // d-spacing parse is exact.
            check_f64(
                &format!("{tag}/dspacing"),
                c.dspacing(),
                &f64v(&format!("{tag}__dspacing.npy")),
                Gate::Exact,
                &mut worst_ulp,
                &mut fails,
            );
            // Bragg 2theta is the transcendental boundary.
            check_f64(
                &format!("{tag}/two_theta"),
                c.get_2th(),
                &f64v(&format!("{tag}__two_theta.npy")),
                Gate::Tol,
                &mut worst_ulp,
                &mut fails,
            );
            for unit in m["peak_units"].as_array().unwrap() {
                let uname = unit.as_str().unwrap();
                let (pu, gate) = peak_unit(uname);
                let got = c.get_peaks(pu);
                check_f64(
                    &format!("{tag}/peaks[{uname}]"),
                    &got,
                    &f64v(&format!("{tag}__peaks_{}.npy", peak_key(uname))),
                    gate,
                    &mut worst_ulp,
                    &mut fails,
                );
            }
        }
    }

    // ---- Cell path: lattice -> d-spacing, exact algebra.
    for cell in m["cells"].as_array().expect("cells") {
        let tag = cell["tag"].as_str().unwrap();
        let dmin = cell["dmin"].as_f64().unwrap();
        eprintln!("=== cell {tag}  dmin={dmin} ===");

        let mut c = build_cell(tag);
        let groups = c.calculate_dspacing(dmin);
        // Keys descending (build_calibrant_config order).
        let mut keyed = groups;
        keyed.sort_by(|a, b| b.0.total_cmp(&a.0));
        let dsp: Vec<f64> = keyed.iter().map(|(d, _)| *d).collect();
        let mult: Vec<i32> = keyed.iter().map(|(_, ms)| ms.len() as i32).collect();

        check_f64(
            &format!("cell_{tag}/dspacing"),
            &dsp,
            &f64v(&format!("cell_{tag}__dspacing.npy")),
            Gate::Exact,
            &mut worst_ulp,
            &mut fails,
        );
        let golden_mult = i32v(&format!("cell_{tag}__multiplicity.npy"));
        if mult != golden_mult {
            fails += 1;
            eprintln!(
                "    cell_{tag}/multiplicity              FAIL      {mult:?} != {golden_mult:?}"
            );
        } else {
            eprintln!(
                "    cell_{tag}/multiplicity              BIT-EXACT  ({} groups)",
                mult.len()
            );
        }
    }

    eprintln!("\nworst 2theta/peak ULP (Tier-B): {worst_ulp} (budget {ASIN_ULP_BUDGET})");
    assert_eq!(fails, 0, "{fails} calibrant field(s) failed their gate");
}

/// Rebuild the `Cell` for a manifest cell tag (mirrors `_cells()` in the generator).
fn build_cell(tag: &str) -> Cell {
    match tag {
        "Al_cubic_F" => Cell::cubic(4.0495, Centering::F),
        "LaB6_cubic_P" => Cell::cubic(4.1568, Centering::P),
        "Si_diamond" => Cell::diamond(5.4312),
        "CeO2_cubic_F" => Cell::cubic(5.411651, Centering::F),
        other => panic!("unknown cell tag {other}"),
    }
}
