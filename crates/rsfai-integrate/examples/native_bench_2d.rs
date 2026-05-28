//! Native (pure-Rust, no PyO3/numpy) 2D steady-state per-frame benchmark — the
//! Rust-side counterpart of `golden/bench_npt.py`'s 2D rsFAI path, isolating the
//! **slow part**: the 2D apply at fine caking (e.g. 5000×360 = 1.8M cells). It
//! runs the SAME workload (`preproc4` + `integrate2d` on the same f32 image with
//! the same prebuilt sparse matrix) entirely in Rust, so the measurement carries
//! none of the PyO3/numpy round-trip the Python bench does — it times the fused
//! reduce→output pass on its own.
//!
//! The per-pixel geometry (`pos0_center_unscaled`, `pos0_delta`, `chi_center`,
//! `chi_delta`, `corners`) is independent of the output binning, so the matrix is
//! rebuilt at each requested `(npt_rad, npt_azim)` from the committed Pilatus1M
//! `npt100x36 errpoisson` 2D golden inputs (the same rebuild `bench_npt.py` does
//! on the Python side; it matched pyFAI 0-ULP there). Geometry + matrix are built
//! ONCE per size (untimed); only `preproc4 + apply` is timed (median of `reps`).
//!
//! Run (multi-thread = rayon default, all cores):
//!   cargo run --release -p rsfai-integrate --example native_bench_2d
//! Optional `reps warmup` then `<nr>x<na>` sizes:
//!   cargo run --release -p rsfai-integrate --example native_bench_2d -- 30 5 1000x360 5000x360
//! Single-thread: prefix `RAYON_NUM_THREADS=1`.

use std::path::{Path, PathBuf};
use std::time::Instant;

use rsfai_core::dtype::ErrorModel;
use rsfai_core::golden::{load_image_f32, load_manifest, load_npy_f32, load_npy_f64, load_npy_i8};
use rsfai_integrate::{
    build_bbox_csr_2d, build_bbox_lut_2d, build_full_csr_2d, build_full_lut_2d, csr_integrate2d,
    lut_integrate2d, Bbox2dBounds, Csr, Integrate2d, Lut,
};
use rsfai_preproc::{preproc4, PreprocOptions};

/// The 2D sparse tuples benchmarked (CSC omitted: serial-by-design pixel scatter).
const TUPLES: &[(&str, &str)] = &[
    ("no", "csr"),
    ("bbox", "csr"),
    ("full", "csr"),
    ("no", "lut"),
    ("bbox", "lut"),
    ("full", "lut"),
];

/// Default output binnings: a coarse case where rsFAI wins and the two fine cases
/// (the slow part) that motivated the fused-apply work.
const DEFAULT_SIZES: &[(usize, usize)] = &[(100, 36), (1000, 360), (5000, 360)];

fn datasets_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../golden/datasets")
}

fn f64v(dir: &Path, f: &str) -> Vec<f64> {
    load_npy_f64(dir.join(f))
        .unwrap_or_else(|_| panic!("load {f}"))
        .iter()
        .copied()
        .collect()
}

fn i8v(dir: &Path, f: &str) -> Vec<i8> {
    load_npy_i8(dir.join(f))
        .unwrap_or_else(|_| panic!("load {f}"))
        .iter()
        .copied()
        .collect()
}

/// An optional f32 correction array: `Some(..)` if the `.npy` exists, else `None`.
/// pyFAI's solid angle is f64; cast to f32 (as `PreprocOptions::solidangle`).
fn opt_f32(dir: &Path, f: &str) -> Option<Vec<f32>> {
    let p = dir.join(f);
    if !p.exists() {
        return None;
    }
    if let Ok(a) = load_npy_f32(&p) {
        return Some(a.iter().copied().collect());
    }
    let a = load_npy_f64(&p).unwrap_or_else(|_| panic!("load {f}"));
    Some(a.iter().map(|&v| v as f32).collect())
}

/// `(npix, 4, 2)` corners stored f32, upcast to f64 (as `FullSplitIntegrator`).
fn corners_f64(dir: &Path) -> Vec<f64> {
    load_npy_f32(dir.join("corners.npy"))
        .expect("corners")
        .iter()
        .map(|&v| v as f64)
        .collect()
}

/// Find the committed 2D Pilatus1M `npt100x36 errpoisson` dataset for a split. The
/// per-pixel inputs are shared across algos, so the CSR-cython set serves both CSR
/// and LUT builds; prefer the `q_nm^-1` build, excluding OpenCL / range / azimuthal
/// variants.
fn find_dataset_2d(split: &str) -> Option<PathBuf> {
    let want = format!("Pilatus1M__{split}-csr-cython__");
    let mut cand: Vec<PathBuf> = std::fs::read_dir(datasets_root())
        .ok()?
        .flatten()
        .map(|e| e.path())
        .filter(|d| {
            let n = d
                .file_name()
                .and_then(|s| s.to_str())
                .unwrap_or("")
                .to_string();
            n.starts_with(&want) && n.ends_with("npt100x36__errpoisson")
        })
        .collect();
    cand.sort();
    cand.iter()
        .find(|d| d.to_str().is_some_and(|s| s.contains("q_nmm1")))
        .cloned()
        .or_else(|| cand.into_iter().next())
}

/// A built 2D engine: a prebuilt sparse matrix plus its radial / azimuthal centers.
enum Engine2d {
    Csr(Csr, Vec<f64>, Vec<f64>),
    Lut(Lut, Vec<f64>, Vec<f64>),
}

impl Engine2d {
    fn apply(&self, prep: &[f32], em: ErrorModel) -> Integrate2d {
        match self {
            Engine2d::Csr(csr, c0, c1) => {
                csr_integrate2d(csr, prep, c0.clone(), c1.clone(), em, 0.0)
            }
            Engine2d::Lut(lut, c0, c1) => {
                lut_integrate2d(lut, prep, c0.clone(), c1.clone(), em, 0.0)
            }
        }
    }
}

/// The bin-count-independent per-pixel inputs for one split's dataset.
struct Inputs {
    pos0: Vec<f64>,
    dpos0: Vec<f64>,
    pos1: Vec<f64>,
    dpos1: Vec<f64>,
    corners: Vec<f64>,
    mask: Vec<i8>,
    bounds: Bbox2dBounds,
}

/// Build the 2D engine once (untimed) at `(nr, na)` bins.
fn build_2d(split: &str, algo: &str, inp: &Inputs, bins: (usize, usize)) -> Engine2d {
    let mask = Some(inp.mask.as_slice());
    match (split, algo) {
        ("no", "csr") => {
            let (csr, c0, c1) =
                build_bbox_csr_2d(&inp.pos0, None, &inp.pos1, None, mask, bins, &inp.bounds);
            Engine2d::Csr(csr, c0, c1)
        }
        ("bbox", "csr") => {
            let (csr, c0, c1) = build_bbox_csr_2d(
                &inp.pos0,
                Some(&inp.dpos0),
                &inp.pos1,
                Some(&inp.dpos1),
                mask,
                bins,
                &inp.bounds,
            );
            Engine2d::Csr(csr, c0, c1)
        }
        ("full", "csr") => {
            let (csr, c0, c1) = build_full_csr_2d(&inp.corners, mask, bins, &inp.bounds);
            Engine2d::Csr(csr, c0, c1)
        }
        ("no", "lut") => {
            let (lut, c0, c1) =
                build_bbox_lut_2d(&inp.pos0, None, &inp.pos1, None, mask, bins, &inp.bounds);
            Engine2d::Lut(lut, c0, c1)
        }
        ("bbox", "lut") => {
            let (lut, c0, c1) = build_bbox_lut_2d(
                &inp.pos0,
                Some(&inp.dpos0),
                &inp.pos1,
                Some(&inp.dpos1),
                mask,
                bins,
                &inp.bounds,
            );
            Engine2d::Lut(lut, c0, c1)
        }
        ("full", "lut") => {
            let (lut, c0, c1) = build_full_lut_2d(&inp.corners, mask, bins, &inp.bounds);
            Engine2d::Lut(lut, c0, c1)
        }
        _ => panic!("unhandled tuple ({split}, {algo})"),
    }
}

/// Median (ms) of `reps` timed calls after `warmup` untimed ones.
fn time_median<F: FnMut()>(reps: usize, warmup: usize, mut f: F) -> f64 {
    for _ in 0..warmup {
        f();
    }
    let mut ts = Vec::with_capacity(reps);
    for _ in 0..reps {
        let t = Instant::now();
        f();
        ts.push(t.elapsed().as_secs_f64() * 1e3);
    }
    ts.sort_by(|a, b| a.partial_cmp(b).unwrap());
    ts[ts.len() / 2]
}

/// Parse a `<nr>x<na>` size token, e.g. `5000x360`.
fn parse_size(s: &str) -> Option<(usize, usize)> {
    let (a, b) = s.split_once('x')?;
    Some((a.parse().ok()?, b.parse().ok()?))
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let reps: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(30);
    let warmup: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(5);
    let sizes: Vec<(usize, usize)> = {
        let parsed: Vec<(usize, usize)> = args[3.min(args.len())..]
            .iter()
            .filter_map(|s| parse_size(s))
            .collect();
        if parsed.is_empty() {
            DEFAULT_SIZES.to_vec()
        } else {
            parsed
        }
    };

    let threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(0);
    let rayon_env = std::env::var("RAYON_NUM_THREADS").unwrap_or_else(|_| "(unset->all)".into());

    println!("native rsFAI 2D bench (pure Rust, no PyO3/numpy)  cores={threads}  RAYON_NUM_THREADS={rayon_env}");
    println!(
        "metric: STEADY-STATE per-frame (matrix rebuilt per size, untimed); reps={reps} warmup={warmup}"
    );
    println!("================================================================================");
    println!(
        "{:<12} {:>10} {:>6} {:>9} {:>9} {:>9}   (ms, median)",
        "tuple", "bins", "nnz", "total", "preproc", "apply"
    );
    println!("--------------------------------------------------------------------------------");

    for &(split, algo) in TUPLES {
        let Some(dir) = find_dataset_2d(split) else {
            println!("{:<12} -- dataset not found", format!("{split}-{algo}"));
            continue;
        };
        let manifest = load_manifest(dir.join("manifest.json")).expect("manifest");
        let cfg = &manifest.config;
        let em_code = cfg["error_model_code"].as_i64().expect("error_model_code");
        let em = ErrorModel::from_code(em_code as i32).expect("error model");
        let norm_factor = cfg["normalization_factor"].as_f64().unwrap_or(1.0) as f32;
        let dummy = cfg["dummy"].as_f64();
        let delta_dummy = cfg["delta_dummy"].as_f64();
        let pos1_period = cfg["pos1_period"].as_f64().expect("pos1_period");
        let chi_disc_at_pi = cfg["chi_disc_at_pi"].as_bool().unwrap_or(true);

        let image = load_image_f32(dir.join("image.npy")).expect("image");
        let mask = i8v(&dir, "mask.npy");
        let sa = opt_f32(&dir, "solidangle.npy");
        let pol = opt_f32(&dir, "polarization.npy");

        let opt = PreprocOptions {
            solidangle: sa.as_deref(),
            polarization: pol.as_deref(),
            mask: Some(&mask),
            normalization_factor: norm_factor,
            poissonian: em_code == 2,
            check_dummy: dummy.is_some(),
            dummy: dummy.unwrap_or(0.0) as f32,
            delta_dummy: delta_dummy.unwrap_or(0.0) as f32,
            ..Default::default()
        };

        // Standard radial unit (q) cannot be negative -> allow_pos0_neg=false.
        let inp = Inputs {
            pos0: f64v(&dir, "pos0_center_unscaled.npy"),
            dpos0: f64v(&dir, "pos0_delta.npy"),
            pos1: f64v(&dir, "chi_center.npy"),
            dpos1: f64v(&dir, "chi_delta.npy"),
            corners: corners_f64(&dir),
            mask: mask.clone(),
            bounds: Bbox2dBounds {
                allow_pos0_neg: false,
                chi_disc_at_pi,
                pos1_period,
                radial_range: None,
                azimuth_range: None,
            },
        };

        let prep0 = preproc4(&image, &opt);
        let pre = time_median(reps, warmup, || {
            let _ = preproc4(&image, &opt);
        });

        for &(nr, na) in &sizes {
            let engine = build_2d(split, algo, &inp, (nr, na));
            let nnz = match &engine {
                Engine2d::Csr(csr, _, _) => csr.data.len(),
                Engine2d::Lut(lut, _, _) => lut.idx.len(),
            };
            let total = time_median(reps, warmup, || {
                let prep = preproc4(&image, &opt);
                let _ = engine.apply(&prep, em);
            });
            let app = time_median(reps, warmup, || {
                let _ = engine.apply(&prep0, em);
            });
            println!(
                "{:<12} {:>10} {:>6} {:>9.3} {:>9.3} {:>9.3}",
                format!("{split}-{algo}"),
                format!("{nr}x{na}"),
                nnz,
                total,
                pre,
                app
            );
        }
    }
}
