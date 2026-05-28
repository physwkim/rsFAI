//! Native (pure-Rust, no PyO3/numpy) steady-state per-frame benchmark — the
//! Rust-side counterpart to `golden/bench_compare.py`'s rsFAI path. It runs the
//! SAME workload (preproc4 + apply on the same f32 image with the same prebuilt
//! sparse matrix) but entirely in Rust, so the only difference from the Python
//! measurement is the absence of the PyO3/numpy round-trip (preproc4 returns a
//! 16 MB numpy array that is then handed back into the apply call across the FFI
//! boundary). Comparing the two answers "does running rsFAI directly as Rust
//! avoid that marshalling cost?".
//!
//! Geometry + sparse matrix are built ONCE (untimed); only `preproc4 + apply`
//! per frame is timed (median of `reps`, after `warmup`). Datasets are the
//! committed Pilatus1M `npt1000 errpoisson` golden sets, read from disk.
//!
//! Run (multi-thread = rayon default, all cores):
//!   cargo run --release -p rsfai-integrate --example native_bench
//! Optional `reps warmup`:
//!   cargo run --release -p rsfai-integrate --example native_bench -- 100 10
//! Single-thread: prefix `RAYON_NUM_THREADS=1`.

use std::path::{Path, PathBuf};
use std::time::Instant;

use rsfai_core::dtype::ErrorModel;
use rsfai_core::golden::{load_image_f32, load_manifest, load_npy_f32, load_npy_f64, load_npy_i8};
use rsfai_integrate::{
    build_bbox_csr_1d, build_bbox_lut_1d, build_full_csr_1d, build_full_lut_1d, csr_integrate1d,
    histogram1d_bbox, histogram1d_full, lut_integrate1d, Csr, CsrIntegrate1d, Lut,
};
use rsfai_preproc::{preproc4, PreprocOptions};

const TWO_PI: f64 = std::f64::consts::PI * 2.0;

/// The 1D tuples benchmarked, mirroring the sparse + split-histogram engines that
/// `bench_compare.py` times (CSC omitted: serial-by-design pixel scatter).
const TUPLES: &[(&str, &str)] = &[
    ("no", "csr"),
    ("bbox", "csr"),
    ("full", "csr"),
    ("no", "lut"),
    ("bbox", "lut"),
    ("full", "lut"),
    ("bbox", "histogram"),
    ("full", "histogram"),
];

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

/// An optional f32 correction array: `Some(..)` if the `.npy` exists, else `None`
/// (pyFAI omits the correction, e.g. polarization when `polarization_factor` is
/// None, and `bench_compare.py` mirrors that).
fn opt_f32(dir: &Path, f: &str) -> Option<Vec<f32>> {
    let p = dir.join(f);
    if !p.exists() {
        return None;
    }
    // pyFAI's solid angle is f64; it casts to f32 before preproc (see
    // PreprocOptions::solidangle). Mirror that: try f32, else load f64 and cast.
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

/// Find the committed Pilatus1M cython dataset for a tuple: `npt1000 errpoisson`,
/// preferring the `q_nm^-1` build, excluding OpenCL and range overrides.
fn find_dataset(split: &str, algo: &str) -> Option<PathBuf> {
    let want = format!("Pilatus1M__{split}-{algo}-cython__");
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
            n.starts_with(&want)
                && n.ends_with("npt1000__errpoisson")
                && !n.contains("opencl")
                && !n.contains("razim")
                && !n.contains("rrad")
                && !n.contains("orient")
        })
        .collect();
    cand.sort();
    cand.iter()
        .find(|d| d.to_str().is_some_and(|s| s.contains("q_nmm1")))
        .cloned()
        .or_else(|| cand.into_iter().next())
}

/// A built 1D engine: a prebuilt sparse matrix (CSR/LUT, build-once) or the
/// position arrays a split-histogram needs (the histogram fuses build+apply each
/// frame, like pyFAI's histogram engine).
enum Engine {
    Csr(Csr, Vec<f64>),
    Lut(Lut, Vec<f64>),
    HistBbox { pos0: Vec<f64>, dpos0: Vec<f64> },
    HistFull { corners: Vec<f64> },
}

impl Engine {
    fn apply(&self, prep: &[f32], mask: &[i8], npt: usize, em: ErrorModel) -> CsrIntegrate1d {
        match self {
            Engine::Csr(csr, c) => csr_integrate1d(csr, prep, c.clone(), em, 0.0),
            Engine::Lut(lut, c) => lut_integrate1d(lut, prep, c.clone(), em, 0.0),
            Engine::HistBbox { pos0, dpos0 } => histogram1d_bbox(
                pos0,
                dpos0,
                prep,
                Some(mask),
                npt,
                em,
                0.0,
                false,
                None,
                None,
            ),
            Engine::HistFull { corners } => histogram1d_full(
                corners,
                prep,
                Some(mask),
                npt,
                em,
                0.0,
                false,
                true,
                TWO_PI,
                None,
                None,
            ),
        }
    }
}

/// Build the engine once (untimed). `mask` is borrowed only during the build.
fn build(split: &str, algo: &str, dir: &Path, mask: &[i8], npt: usize) -> Engine {
    match (split, algo) {
        ("no", "csr") => {
            let (csr, c) = build_bbox_csr_1d(
                &f64v(dir, "pos0_center_unscaled.npy"),
                None,
                Some(mask),
                npt,
                false,
                None,
                None,
            );
            Engine::Csr(csr, c)
        }
        ("bbox", "csr") => {
            let (csr, c) = build_bbox_csr_1d(
                &f64v(dir, "pos0_center_unscaled.npy"),
                Some(&f64v(dir, "pos0_delta.npy")),
                Some(mask),
                npt,
                false,
                None,
                None,
            );
            Engine::Csr(csr, c)
        }
        ("full", "csr") => {
            let (csr, c) = build_full_csr_1d(
                &corners_f64(dir),
                Some(mask),
                npt,
                false,
                true,
                TWO_PI,
                None,
                None,
            );
            Engine::Csr(csr, c)
        }
        ("no", "lut") => {
            let (lut, c) = build_bbox_lut_1d(
                &f64v(dir, "pos0_center_unscaled.npy"),
                None,
                Some(mask),
                npt,
                false,
                None,
                None,
            );
            Engine::Lut(lut, c)
        }
        ("bbox", "lut") => {
            let (lut, c) = build_bbox_lut_1d(
                &f64v(dir, "pos0_center_unscaled.npy"),
                Some(&f64v(dir, "pos0_delta.npy")),
                Some(mask),
                npt,
                false,
                None,
                None,
            );
            Engine::Lut(lut, c)
        }
        ("full", "lut") => {
            let (lut, c) = build_full_lut_1d(
                &corners_f64(dir),
                Some(mask),
                npt,
                false,
                true,
                TWO_PI,
                None,
                None,
            );
            Engine::Lut(lut, c)
        }
        ("bbox", "histogram") => Engine::HistBbox {
            pos0: f64v(dir, "pos0_center_unscaled.npy"),
            dpos0: f64v(dir, "pos0_delta.npy"),
        },
        ("full", "histogram") => Engine::HistFull {
            corners: corners_f64(dir),
        },
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

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let reps: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(100);
    let warmup: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(10);
    let threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(0);
    let rayon_env = std::env::var("RAYON_NUM_THREADS").unwrap_or_else(|_| "(unset->all)".into());

    println!("native rsFAI bench (pure Rust, no PyO3/numpy)  cores={threads}  RAYON_NUM_THREADS={rayon_env}");
    println!(
        "metric: STEADY-STATE per-frame (matrix built once, untimed); reps={reps} warmup={warmup}"
    );
    println!("================================================================================");
    println!(
        "{:<18} {:>9} {:>9} {:>9}   (ms, median)",
        "tuple", "total", "preproc", "apply"
    );
    println!("--------------------------------------------------------------------------------");

    for &(split, algo) in TUPLES {
        let Some(dir) = find_dataset(split, algo) else {
            println!("{:<18} -- dataset not found", format!("{split}-{algo}"));
            continue;
        };
        let manifest = load_manifest(dir.join("manifest.json")).expect("manifest");
        let cfg = &manifest.config;
        let npt = cfg["npt"].as_u64().expect("npt") as usize;
        let em_code = cfg["error_model_code"].as_i64().expect("error_model_code");
        let em = ErrorModel::from_code(em_code as i32).expect("error model");
        let norm_factor = cfg["normalization_factor"].as_f64().unwrap_or(1.0) as f32;
        let dummy = cfg["dummy"].as_f64();
        let delta_dummy = cfg["delta_dummy"].as_f64();

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

        let engine = build(split, algo, &dir, &mask, npt);
        let prep0 = preproc4(&image, &opt);

        let total = time_median(reps, warmup, || {
            let prep = preproc4(&image, &opt);
            let _ = engine.apply(&prep, &mask, npt, em);
        });
        let pre = time_median(reps, warmup, || {
            let _ = preproc4(&image, &opt);
        });
        let app = time_median(reps, warmup, || {
            let _ = engine.apply(&prep0, &mask, npt, em);
        });

        println!(
            "{:<18} {:>9.3} {:>9.3} {:>9.3}",
            format!("{split}-{algo}"),
            total,
            pre,
            app
        );
    }
}
