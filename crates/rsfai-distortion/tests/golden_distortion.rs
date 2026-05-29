//! Distortion + spline golden gate: `rsfai-distortion` vs pyFAI 2026.5.0 on
//! `golden/datasets_distortion/`.
//!
//! `gen_golden_distortion.py` (daq env, OMP_NUM_THREADS=1) dumps three layers,
//! each gated **bit-exact** (0 ULP):
//!
//!   1. bisplev  -- pyFAI `_bispev.bisplev` on the halfccd X/Y displacement
//!      tensors over a small grid. All f32 (de Boor-Cox + Kahan tensor sum).
//!   2. spline parser + spline2array -- the parsed knots/coeffs match the dump,
//!      and `Spline::spline2array` reproduces pyFAI's full-grid displacement
//!      maps. (spline2array arrays are gitignored; the test skips if absent.)
//!   3. Distortion LUT + correct -- fed pyFAI's `calc_pos` `pos` 4D array as a
//!      Tier-A input, the Rust `calc_sparse` CSR (data/indices/indptr) and the
//!      `correct`ed image reproduce pyFAI bit-for-bit. (pos/CSR/image inputs are
//!      gitignored; the corrected output is committed, so the LUT-build half
//!      skips when inputs are absent, but the correct half can always run by
//!      rebuilding from `pos` once it is regenerated.)
//!
//! Big per-pixel arrays are gitignored + regenerated; tests that need them are
//! skipped with a printed notice when the file is missing, so a fresh checkout
//! still passes (the committed-data tests always run).

use std::path::PathBuf;

use ndarray::ArrayD;
use rsfai_core::compare::compare_f32;
use rsfai_core::golden::{load_npy_f32, load_npy_i32};
use rsfai_distortion::{bisplev, calc_pos, calc_sparse, correct, Spline, Tck};

/// Exact equality of two integer index slices (CSR `indices`/`indptr` must be
/// identical, not merely close). Returns `(equal, first_mismatch_index)`.
fn i32_eq(a: &[i32], b: &[i32]) -> (bool, Option<usize>) {
    if a.len() != b.len() {
        return (false, Some(a.len().min(b.len())));
    }
    for (i, (&x, &y)) in a.iter().zip(b.iter()).enumerate() {
        if x != y {
            return (false, Some(i));
        }
    }
    (true, None)
}

fn datasets_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../golden/datasets_distortion")
}

fn f32v(name: &str) -> Option<Vec<f32>> {
    let p = datasets_root().join(name);
    if !p.exists() {
        return None;
    }
    Some(
        load_npy_f32(&p)
            .unwrap_or_else(|e| panic!("load {name}: {e}"))
            .iter()
            .copied()
            .collect(),
    )
}

fn i32v(name: &str) -> Option<Vec<i32>> {
    let p = datasets_root().join(name);
    if !p.exists() {
        return None;
    }
    Some(
        load_npy_i32(&p)
            .unwrap_or_else(|e| panic!("load {name}: {e}"))
            .iter()
            .copied()
            .collect(),
    )
}

/// Load a `.npy` keeping its shape (for the `pos` 4D array).
fn f32_arr(name: &str) -> Option<ArrayD<f32>> {
    let p = datasets_root().join(name);
    if !p.exists() {
        return None;
    }
    Some(load_npy_f32(&p).unwrap_or_else(|e| panic!("load {name}: {e}")))
}

/// Build a [`Tck`] from the committed parsed knots/coeffs dumps.
fn tck_from_dump(tag: &str) -> Tck {
    Tck {
        tx: f32v(&format!("out_spline_{tag}_knotsx.npy")).expect("knotsx committed"),
        ty: f32v(&format!("out_spline_{tag}_knotsy.npy")).expect("knotsy committed"),
        c: f32v(&format!("out_spline_{tag}_coeff.npy")).expect("coeff committed"),
        kx: 3,
        ky: 3,
    }
}

#[test]
fn bisplev_matches_pyfai_bit_exact() {
    let bx = f32v("out_bisplev_x.npy").expect("bisplev x committed");
    let by = f32v("out_bisplev_y.npy").expect("bisplev y committed");

    let mut fails = 0usize;
    for tag in ["x", "y"] {
        let tck = tck_from_dump(tag);
        let got = bisplev(&bx, &by, &tck);
        let golden = f32v(&format!("out_bisplev_z{tag}.npy")).expect("bisplev z committed");
        let r = compare_f32(&got, &golden);
        eprintln!(
            "bisplev[{tag}]  bit_exact={} ulp={} mism={}/{}",
            r.is_bit_exact(),
            r.max_ulp,
            r.bit_mismatches,
            r.total
        );
        if !r.is_bit_exact() {
            fails += 1;
        }
    }
    assert_eq!(fails, 0, "bisplev not bit-exact for {fails} tensor(s)");
}

#[test]
fn spline_parser_matches_dump() {
    let sp = Spline::read(datasets_root().join("halfccd.spline")).expect("parse spline");
    assert_eq!(sp.order, 3);

    let mut fails = 0usize;
    for (tag, tck) in [("x", &sp.x_tck), ("y", &sp.y_tck)] {
        for (kind, got) in [("knotsx", &tck.tx), ("knotsy", &tck.ty), ("coeff", &tck.c)] {
            let golden = f32v(&format!("out_spline_{tag}_{kind}.npy"))
                .unwrap_or_else(|| panic!("committed {tag} {kind}"));
            let r = compare_f32(got, &golden);
            eprintln!(
                "spline parse {tag}/{kind}  bit_exact={} ulp={}",
                r.is_bit_exact(),
                r.max_ulp
            );
            if !r.is_bit_exact() {
                fails += 1;
            }
        }
    }
    assert_eq!(fails, 0, "spline parser knots/coeffs not bit-exact");
}

#[test]
fn spline2array_matches_pyfai_bit_exact() {
    // The full-grid displacement maps are gitignored (8 MB each); skip cleanly
    // on a fresh checkout, run after regenerating the golden.
    let (Some(xdisp_g), Some(ydisp_g)) = (
        f32v("spline2array_xdisp.npy"),
        f32v("spline2array_ydisp.npy"),
    ) else {
        eprintln!("SKIP spline2array: regenerate golden (gitignored full-grid arrays absent)");
        return;
    };

    let sp = Spline::read(datasets_root().join("halfccd.spline")).expect("parse spline");
    let (xdisp, ydisp) = sp.spline2array();

    let rx = compare_f32(&xdisp, &xdisp_g);
    let ry = compare_f32(&ydisp, &ydisp_g);
    eprintln!(
        "spline2array  xDisp bit_exact={} ulp={} mism={}/{}   yDisp bit_exact={} ulp={} mism={}/{}",
        rx.is_bit_exact(),
        rx.max_ulp,
        rx.bit_mismatches,
        rx.total,
        ry.is_bit_exact(),
        ry.max_ulp,
        ry.bit_mismatches,
        ry.total
    );
    assert!(rx.is_bit_exact(), "xDispArray not bit-exact");
    assert!(ry.is_bit_exact(), "yDispArray not bit-exact");
}

/// Read the distortion grid metadata (shape_in / shape_out / pixel sizes) from
/// the manifest.
fn distortion_meta() -> serde_json::Value {
    let text = std::fs::read_to_string(datasets_root().join("manifest.json")).expect("manifest");
    let m: serde_json::Value = serde_json::from_str(&text).expect("parse manifest");
    m["distortion"].clone()
}

#[test]
fn distortion_lut_and_correct_match_pyfai_bit_exact() {
    let meta = distortion_meta();
    let shape_in = (
        meta["shape_in"][0].as_u64().unwrap() as usize,
        meta["shape_in"][1].as_u64().unwrap() as usize,
    );
    let shape_out = (
        meta["shape_out"][0].as_u64().unwrap() as usize,
        meta["shape_out"][1].as_u64().unwrap() as usize,
    );
    let pixel1 = meta["pixel1"].as_f64().unwrap();
    let pixel2 = meta["pixel2"].as_f64().unwrap();
    let empty = meta["empty"].as_f64().unwrap() as f32;

    // Tier-A input: pyFAI's calc_pos `pos` 4D array (gitignored). Without it we
    // cannot reproduce the LUT; skip cleanly when absent (fresh checkout).
    let Some(corners) = f32_arr("dist_corners.npy") else {
        eprintln!("SKIP distortion LUT: regenerate golden (gitignored pos/corners absent)");
        return;
    };
    let corners_flat: Vec<f32> = corners.iter().copied().collect();

    // (a) calc_pos: rebuild the corner positions from the raw corner array and
    //     confirm it matches pyFAI's `pos` bit-for-bit.
    let cp = calc_pos(&corners_flat, shape_in, pixel1, pixel2, Some(shape_out));
    if let Some(pos_g) = f32_arr("dist_pos.npy") {
        let pos_golden: Vec<f32> = pos_g.iter().copied().collect();
        let r = compare_f32(&cp.pos, &pos_golden);
        eprintln!(
            "calc_pos  bit_exact={} ulp={} mism={}/{}  delta={:?} shape_out={:?}",
            r.is_bit_exact(),
            r.max_ulp,
            r.bit_mismatches,
            r.total,
            cp.delta,
            cp.shape_out
        );
        assert!(r.is_bit_exact(), "calc_pos `pos` not bit-exact");
    }

    // (b) calc_sparse: build the CSR and compare to pyFAI's data/indices/indptr.
    let csr = calc_sparse(&cp, None, (0.0, 0.0));
    let mut fails = 0usize;
    if let (Some(data_g), Some(indices_g), Some(indptr_g)) = (
        f32v("dist_csr_data.npy"),
        i32v("dist_csr_indices.npy"),
        i32v("dist_csr_indptr.npy"),
    ) {
        let rd = compare_f32(&csr.data, &data_g);
        let (ri, ri_at) = i32_eq(&csr.indices, &indices_g);
        let (rp, rp_at) = i32_eq(&csr.indptr, &indptr_g);
        eprintln!(
            "calc_sparse data bit_exact={} ulp={} mism={}/{}  indices match={} (at {:?})  indptr match={} (at {:?})",
            rd.is_bit_exact(),
            rd.max_ulp,
            rd.bit_mismatches,
            rd.total,
            ri,
            ri_at,
            rp,
            rp_at
        );
        if !rd.is_bit_exact() {
            fails += 1;
        }
        if !ri {
            fails += 1;
        }
        if !rp {
            fails += 1;
        }
    } else {
        eprintln!("SKIP CSR compare: gitignored dist_csr_* absent (rebuilt anyway for correct)");
    }

    // (c) correct: apply the Rust-built LUT to pyFAI's raw image, compare to the
    //     committed corrected image. Image is gitignored; skip if absent.
    if let Some(image) = f32v("dist_image.npy") {
        let out = correct(&image, &csr, empty);
        let cor_g = f32v("out_dist_corrected.npy").expect("corrected committed");
        let rc = compare_f32(&out, &cor_g);
        eprintln!(
            "correct  bit_exact={} ulp={} mism={}/{} max_abs={:e}",
            rc.is_bit_exact(),
            rc.max_ulp,
            rc.bit_mismatches,
            rc.total,
            rc.max_abs_diff
        );
        if !rc.is_bit_exact() {
            fails += 1;
        }
    } else {
        eprintln!("SKIP correct: gitignored dist_image.npy absent");
    }

    assert_eq!(fails, 0, "{fails} distortion field(s) not bit-exact");
}
