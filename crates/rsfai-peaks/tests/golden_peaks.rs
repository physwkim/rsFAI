//! Peak-finding parity gate: `rsfai-peaks` vs pyFAI on `golden/datasets_peaks/`.
//!
//! `gen_golden_peaks.py` (daq env, pyFAI 2026.5.0 `-ffp-contract=off`, scipy
//! 1.17.1) dumps five surfaces; this verifier reloads the identical inputs,
//! runs the Rust ports, and asserts the gate from `doc/bit-exact-ladder.md`:
//!
//!   * `scipy.ndimage.label` (8/4-connectivity) -> int32 labels: **bit-exact**.
//!   * `distance_transform_edt` distances (f64) + feature indices (int32):
//!     **bit-exact** (the squared-distance accumulation is integer; `sqrt` of an
//!     exact integer is correctly-rounded, so the distance is exact too).
//!   * `InverseWatershed` int32 labels, uint8 borders, and `peaks_from_area`
//!     coordinates (f32): **bit-exact** (deterministic hill-climb + algebra).
//!   * blob DoG `local_max` voxel coords (int32) + `refine_Hessian` (f32):
//!     **bit-exact** (pure comparison / `f32` algebra on the golden DoG stack).
//!   * ellipse `design` matrix (f64 products): **bit-exact**; the fitted
//!     ellipse parameters: **Tier-B tolerance** (the LAPACK `eig`/`inv` vs
//!     `nalgebra` divergence; the measured relative error is printed and
//!     asserted under `ELLIPSE_REL_TOL`).

use std::path::PathBuf;

use rsfai_core::compare::{compare_f32, compare_f64};
use rsfai_core::golden::{load_npy_f32, load_npy_f64, load_npy_i32, load_npy_i8};
use rsfai_peaks::{
    blob, distance_transform_edt, fit_ellipse, label, refine_hessian, DogStack, InverseWatershed,
    Structure,
};
use serde_json::Value;

/// Recorded Tier-B tolerance for the ellipse fit. The design matrix is
/// bit-exact, but `S = DᵀD` already differs from numpy's BLAS reduction by ~1
/// ULP, and the `inv(S)` + eigensolve are LAPACK black boxes `nalgebra` cannot
/// reproduce; the eigenvector basis differs, so the recovered parameters land
/// at ~1e-6 relative. The test prints the measured worst relative error per
/// field; this is the sanctioned ceiling (not a claim of the measured gap).
const ELLIPSE_REL_TOL: f64 = 1e-5;

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../golden/datasets_peaks")
}

fn manifest() -> Value {
    let p = root().join("manifest.json");
    let t = std::fs::read_to_string(&p).unwrap_or_else(|e| panic!("read manifest {p:?}: {e}"));
    serde_json::from_str(&t).expect("parse manifest")
}

fn i32v(name: &str) -> Vec<i32> {
    load_npy_i32(root().join(name))
        .unwrap_or_else(|e| panic!("load {name}: {e}"))
        .iter()
        .copied()
        .collect()
}

fn i8v(name: &str) -> Vec<i8> {
    load_npy_i8(root().join(name))
        .unwrap_or_else(|e| panic!("load {name}: {e}"))
        .iter()
        .copied()
        .collect()
}

fn f64v(name: &str) -> Vec<f64> {
    load_npy_f64(root().join(name))
        .unwrap_or_else(|e| panic!("load {name}: {e}"))
        .iter()
        .copied()
        .collect()
}

fn f32v(name: &str) -> Vec<f32> {
    load_npy_f32(root().join(name))
        .unwrap_or_else(|e| panic!("load {name}: {e}"))
        .iter()
        .copied()
        .collect()
}

/// Read a 2-D `.npy`'s shape from the manifest meta `{file, shape, dtype}`.
fn shape_of(meta: &Value) -> (usize, usize) {
    let s = meta["shape"].as_array().unwrap();
    (
        s[0].as_u64().unwrap() as usize,
        s[1].as_u64().unwrap() as usize,
    )
}

#[test]
fn label_matches_scipy() {
    let m = manifest();
    let lab = &m["label"];
    let (rows, cols) = shape_of(&lab["input"]);
    let raw = i8v("label_input.npy");
    let input: Vec<bool> = raw.iter().map(|&v| v != 0).collect();

    let mut fails = 0;
    for case in lab["cases"].as_array().unwrap() {
        let conn = case["connectivity"].as_str().unwrap();
        let golden = i32v(case["file"].as_str().unwrap());
        let golden_n = case["n"].as_i64().unwrap() as i32;
        let structure = match conn {
            "c8" => Structure::full(),
            "c4" => Structure::cross(),
            other => panic!("unknown connectivity {other}"),
        };
        let (out, n) = label(&input, rows, cols, structure);
        let bit = out == golden && n == golden_n;
        if !bit {
            fails += 1;
        }
        eprintln!(
            "label[{conn}]  {}  n={n} (golden {golden_n})",
            if bit { "BIT-EXACT" } else { "FAIL" }
        );
    }
    assert_eq!(fails, 0, "{fails} label case(s) failed bit-exactness");
}

#[test]
fn edt_matches_scipy() {
    let m = manifest();
    let (rows, cols) = shape_of(&m["edt"]["input"]);
    let raw = i8v("edt_input.npy");
    let input: Vec<bool> = raw.iter().map(|&v| v != 0).collect();

    let res = distance_transform_edt(&input, rows, cols);

    let golden_dist = f64v("edt_dist.npy");
    let golden_row = i32v("edt_idx_row.npy");
    let golden_col = i32v("edt_idx_col.npy");

    let dist = res.distances();
    let rd = compare_f64(&dist, &golden_dist);
    let row_ok = res.idx_row == golden_row;
    let col_ok = res.idx_col == golden_col;
    eprintln!(
        "edt dist   {}  ulp={} mism={}/{}",
        if rd.is_bit_exact() {
            "BIT-EXACT"
        } else {
            "FAIL"
        },
        rd.max_ulp,
        rd.bit_mismatches,
        rd.total
    );
    eprintln!("edt idx_row {}", if row_ok { "BIT-EXACT" } else { "FAIL" });
    eprintln!("edt idx_col {}", if col_ok { "BIT-EXACT" } else { "FAIL" });
    assert!(rd.is_bit_exact(), "EDT distances not bit-exact");
    assert!(row_ok, "EDT idx_row not bit-exact");
    assert!(col_ok, "EDT idx_col not bit-exact");
}

#[test]
fn watershed_matches_pyfai() {
    let m = manifest();
    let ws = &m["watershed"];
    let (rows, cols) = shape_of(&ws["image"]);
    let image = f32v("ws_image.npy");

    let mut iw = InverseWatershed::new(image.clone(), rows, cols);
    iw.init();

    // labels + borders bit-exact
    let golden_labels = i32v("ws_labels.npy");
    let golden_borders: Vec<u8> = i32v("ws_borders.npy").iter().map(|&v| v as u8).collect();
    let lbl_ok = iw.labels == golden_labels;
    let brd_ok = iw.borders == golden_borders;
    eprintln!(
        "watershed labels  {}",
        if lbl_ok { "BIT-EXACT" } else { "FAIL" }
    );
    eprintln!(
        "watershed borders {}",
        if brd_ok { "BIT-EXACT" } else { "FAIL" }
    );

    let mask = vec![true; rows * cols];
    let mut fails = 0;
    if !lbl_ok {
        fails += 1;
    }
    if !brd_ok {
        fails += 1;
    }
    for case in ws["peaks"].as_array().unwrap() {
        let tag = case["tag"].as_str().unwrap();
        let cfg = &case["config"];
        let imin = cfg["Imin"].as_f64().map(|v| v as f32);
        let keep = cfg["keep"].as_u64().map(|v| v as usize);
        let refine = cfg["refine"].as_bool().unwrap();
        let dmin = cfg["dmin"].as_f64().unwrap() as f32;

        // fresh segmenter per case (init is idempotent; matches the generator).
        let mut iwc = InverseWatershed::new(image.clone(), rows, cols);
        iwc.init();
        let pts = iwc.peaks_from_area(&mask, imin, keep, refine, dmin);
        let got: Vec<f32> = pts.iter().flat_map(|&(y, x)| [y, x]).collect();
        // pyFAI's peaks_from_area produces f32 coordinates (the bilinear refine
        // returns f32; the no-refine path returns integer-valued coords). The
        // generator stored them as a float64 array, so narrow it back to f32 —
        // every stored value is exactly f32-representable — and compare in f32.
        let golden: Vec<f32> = f64v(case["file"].as_str().unwrap())
            .iter()
            .map(|&v| v as f32)
            .collect();

        let ok = got.len() == golden.len() && {
            let r = compare_f32(&got, &golden);
            r.is_bit_exact()
        };
        if !ok {
            fails += 1;
        }
        eprintln!(
            "watershed peaks[{tag}]  {}  n={} (golden {})",
            if ok { "BIT-EXACT" } else { "FAIL" },
            got.len() / 2,
            golden.len() / 2
        );
    }
    assert_eq!(fails, 0, "{fails} watershed field(s) failed bit-exactness");
}

#[test]
fn blob_matches_pyfai() {
    let m = manifest();
    let b = &m["blob"];
    let dshape = b["dogs"]["shape"].as_array().unwrap();
    let (ns, ny, nx) = (
        dshape[0].as_u64().unwrap() as usize,
        dshape[1].as_u64().unwrap() as usize,
        dshape[2].as_u64().unwrap() as usize,
    );
    let dogs = DogStack::new(f32v("blob_dogs.npy"), ns, ny, nx);
    let raw_mask = i8v("blob_mask.npy");
    let mask: Vec<bool> = raw_mask.iter().map(|&v| v != 0).collect();

    let mut fails = 0;
    for case in b["localmax"].as_array().unwrap() {
        let tag = case["tag"].as_str().unwrap();
        let n_5 = case["n_5"].as_bool().unwrap();
        let coords = blob::local_max(&dogs, Some(&mask), n_5);
        let got: Vec<i32> = coords
            .iter()
            .flat_map(|&(s, y, x)| [s as i32, y as i32, x as i32])
            .collect();
        let golden = i32v(case["file"].as_str().unwrap());
        let ok = got == golden;
        if !ok {
            fails += 1;
        }
        eprintln!(
            "blob local_max[{tag}]  {}  n={} (golden {})",
            if ok { "BIT-EXACT" } else { "FAIL" },
            got.len() / 3,
            golden.len() / 3
        );
    }

    // refine_Hessian on the n3 keypoints.
    if let Some(refine_meta) = b.get("refine") {
        let coords = blob::local_max(&dogs, Some(&mask), false);
        let mut got: Vec<f64> = Vec::new();
        for (s, y, x) in coords {
            let r = refine_hessian(&dogs, x, y, s);
            got.extend_from_slice(&[
                r.x as f64,
                r.y as f64,
                r.sigma as f64,
                r.peak_val as f64,
                if r.valid { 1.0 } else { 0.0 },
            ]);
        }
        let golden = f64v(refine_meta["file"].as_str().unwrap());
        // The golden stored f32 refinement results promoted to f64; compare the
        // f32 round-trip bit-for-bit by re-narrowing both sides.
        let got_f32: Vec<f32> = got.iter().map(|&v| v as f32).collect();
        let golden_f32: Vec<f32> = golden.iter().map(|&v| v as f32).collect();
        let ok = got_f32.len() == golden_f32.len() && {
            let r = compare_f32(&got_f32, &golden_f32);
            r.is_bit_exact()
        };
        if !ok {
            fails += 1;
        }
        eprintln!(
            "blob refine_Hessian   {}  n={} (golden {})",
            if ok { "BIT-EXACT" } else { "FAIL" },
            got_f32.len() / 5,
            golden_f32.len() / 5
        );
    }
    assert_eq!(fails, 0, "{fails} blob field(s) failed bit-exactness");
}

#[test]
fn ellipse_matches_pyfai() {
    let m = manifest();
    let mut design_fails = 0;
    let mut param_fails = 0;
    let mut worst_rel = 0.0f64;

    for case in m["ellipse"]["cases"].as_array().unwrap() {
        let tag = case["tag"].as_str().unwrap();
        let pty = f64v(case["pty"]["file"].as_str().unwrap());
        let ptx = f64v(case["ptx"]["file"].as_str().unwrap());

        // design matrix bit-exact (Tier A): flatten our [x², xy, y², x, y, 1].
        let d = rsfai_peaks::design_matrix(&pty, &ptx);
        let got_design: Vec<f64> = d.iter().flat_map(|r| r.iter().copied()).collect();
        let golden_design = f64v(case["design"]["file"].as_str().unwrap());
        let rd = compare_f64(&got_design, &golden_design);
        if !rd.is_bit_exact() {
            design_fails += 1;
        }
        eprintln!(
            "ellipse[{tag}] design  {}  ulp={}",
            if rd.is_bit_exact() {
                "BIT-EXACT"
            } else {
                "FAIL"
            },
            rd.max_ulp
        );

        // fitted parameters (Tier B, recorded tolerance).
        let e = fit_ellipse(&pty, &ptx).expect("fit_ellipse");
        let p = &case["params"];
        // The orientation angle of a (near-)circle is mathematically undefined:
        // when the two semi-axes coincide, the principal direction is degenerate
        // and `atan2(2b, a-c)` magnifies the eigenvector basis difference between
        // LAPACK and nalgebra without bound. pyFAI itself returns inconsistent
        // angles for the circular fixtures. Skip the angle gate when the fit is
        // circular (eccentricity below 1e-3), comparing it only for genuinely
        // elliptical fits.
        let golden_long = p["half_long_axis"].as_f64().unwrap();
        let golden_short = p["half_short_axis"].as_f64().unwrap();
        let is_circle = (golden_long - golden_short).abs() / golden_long.abs() < 1e-3;
        let mut fields = vec![
            ("center_1", e.center_1, p["center_1"].as_f64().unwrap()),
            ("center_2", e.center_2, p["center_2"].as_f64().unwrap()),
            ("half_long", e.half_long_axis, golden_long),
            ("half_short", e.half_short_axis, golden_short),
        ];
        if !is_circle {
            fields.push(("angle", e.angle, p["angle"].as_f64().unwrap()));
        } else {
            eprintln!("ellipse[{tag}] angle       skip    (circular fit, orientation undefined)");
        }
        for (name, got, golden) in fields {
            let rel = if golden.abs() > 0.0 {
                (got - golden).abs() / golden.abs()
            } else {
                (got - golden).abs()
            };
            if rel > worst_rel {
                worst_rel = rel;
            }
            let ok = rel <= ELLIPSE_REL_TOL;
            if !ok {
                param_fails += 1;
            }
            eprintln!(
                "ellipse[{tag}] {name:11} {}  rel={rel:.3e}  rust={got:.12} py={golden:.12}",
                if ok { "tol-ok " } else { "FAIL   " }
            );
        }
    }
    eprintln!(
        "\nellipse worst param rel error (Tier-B): {worst_rel:.3e} (budget {ELLIPSE_REL_TOL:.1e})"
    );
    assert_eq!(
        design_fails, 0,
        "{design_fails} ellipse design matrix(es) not bit-exact"
    );
    assert_eq!(
        param_fails, 0,
        "{param_fails} ellipse param(s) outside tolerance"
    );
}
