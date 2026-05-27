//! Validate the Rust OpenCL orchestration against pyFAI's OpenCL golden.
//!
//! For every `golden/datasets/*` carrying an `opencl_params.json` (produced by
//! `golden/gen_golden_opencl.py`), this feeds the *same* inputs pyFAI fed its
//! GPU — raw image, correction arrays, and the exact sparse matrix (CSR
//! `data/indices/indptr` or the densified LUT `idx/coef`) — through pyFAI's own
//! embedded kernels via the matching `integrate{1,2}d_{csr,lut}` entry point,
//! then compares each output field against the golden.
//!
//! Gate: relative error <= 1e-6 (the Phase-2 tolerance for the GPU doubleword
//! reduction). On the Apple M4 Pro the deterministic same-device/same-kernel
//! path is expected to be bit-exact; the test reports which fields actually
//! matched bit-for-bit so any drift from that is visible.

use std::path::{Path, PathBuf};

use serde::Deserialize;

use rsfai_core::compare::compare_f32;
use rsfai_core::golden::{load_npy_f32, load_npy_f64, load_npy_i32, load_npy_i8};
use rsfai_opencl::{
    create_queue, default_context, integrate1d_csr, integrate1d_lut, integrate2d_csr,
    integrate2d_lut, program, ClSession, Corrections4Args, Corrections4aArgs, CsrInputs, LutInputs,
};

const REL_TOL: f64 = 1e-6;

fn datasets_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../golden/datasets")
}

/// Datasets that carry GPU kernel parameters — i.e. OpenCL golden datasets.
fn opencl_dataset_dirs() -> Vec<PathBuf> {
    let mut dirs = vec![];
    if let Ok(rd) = std::fs::read_dir(datasets_root()) {
        for e in rd.flatten() {
            if e.path().join("opencl_params.json").exists() {
                dirs.push(e.path());
            }
        }
    }
    dirs.sort();
    dirs
}

/// `corrections4a` scalars (CSR path) — includes the image `dtype`.
#[derive(Deserialize)]
struct CorrCsr {
    dtype: i8,
    error_model: i8,
    do_dark: i8,
    do_dark_variance: i8,
    do_flat: i8,
    do_solidangle: i8,
    do_polarization: i8,
    do_absorption: i8,
    do_mask: i8,
    do_dummy: i8,
    dummy: f32,
    delta_dummy: f32,
    normalization_factor: f32,
    apply_normalization: i8,
}

/// `corrections4` scalars (LUT path) — no `dtype` (the image is pre-cast).
#[derive(Deserialize)]
struct CorrLut {
    error_model: i8,
    do_dark: i8,
    do_dark_variance: i8,
    do_flat: i8,
    do_solidangle: i8,
    do_polarization: i8,
    do_absorption: i8,
    do_mask: i8,
    do_dummy: i8,
    dummy: f32,
    delta_dummy: f32,
    normalization_factor: f32,
    apply_normalization: i8,
}

#[derive(Deserialize)]
struct OpenclParams {
    /// "csr" or "lut". Absent on the earliest CSR datasets ⇒ default "csr".
    #[serde(default = "default_algo")]
    algo: String,
    bins: usize,
    image_size: usize,
    empty: f32,
    /// 1 = 1D radial curve, 2 = 2D (azimuthal, radial) map.
    #[serde(default = "default_dim")]
    dim: usize,
    /// Radial bin count (2D only; absent for 1D).
    #[serde(default)]
    bins_rad: usize,
    /// Azimuthal bin count (2D only; absent for 1D).
    #[serde(default)]
    bins_azim: usize,
    // CSR-specific.
    #[serde(default)]
    corrections4a: Option<CorrCsr>,
    #[serde(default)]
    wg_min: Option<usize>,
    // LUT-specific.
    #[serde(default)]
    corrections4: Option<CorrLut>,
    #[serde(default)]
    block_size: Option<usize>,
    #[serde(default)]
    lut_size: Option<usize>,
}

fn default_algo() -> String {
    "csr".to_string()
}
fn default_dim() -> usize {
    1
}

fn load_params(dir: &Path) -> OpenclParams {
    let text = std::fs::read_to_string(dir.join("opencl_params.json")).expect("opencl_params.json");
    serde_json::from_str(&text).expect("parse opencl_params.json")
}

fn vec_i32(p: PathBuf) -> Vec<i32> {
    load_npy_i32(p).expect("i32 npy").iter().copied().collect()
}
fn vec_i8(p: PathBuf) -> Vec<i8> {
    load_npy_i8(p).expect("i8 npy").iter().copied().collect()
}
fn vec_f32(p: PathBuf) -> Vec<f32> {
    load_npy_f32(p).expect("f32 npy").iter().copied().collect()
}
/// Load a correction array as the f32 the GPU actually received. pyFAI holds
/// solid angle as f64 but polarization as f32; its `send_buffer` casts every
/// correction to the float32 buffer dtype on upload (round-to-nearest-even). So
/// read whichever dtype is on disk: native f32 is used as-is, f64 is cast
/// `as f32` (Rust's round-to-nearest-even matches numpy's `astype(float32)`).
fn corr_f32(p: &Path) -> Vec<f32> {
    match load_npy_f32(p) {
        Ok(a) => a.iter().copied().collect(),
        Err(_) => load_npy_f64(p)
            .unwrap_or_else(|e| panic!("{}: correction not f32 or f64: {e}", p.display()))
            .iter()
            .map(|&v| v as f32)
            .collect(),
    }
}

/// Optional correction array: present only if its `.npy` exists (absent ⇒ the
/// matching `do_*` flag is 0, so the kernel ignores the buffer).
fn opt_f32(dir: &Path, name: &str) -> Option<Vec<f32>> {
    let p = dir.join(format!("{name}.npy"));
    p.exists().then(|| corr_f32(&p))
}

/// Validate and return the 2D `(bins_rad, bins_azim)` from the params.
fn dims_2d(dataset: &str, params: &OpenclParams) -> (usize, usize) {
    assert!(
        params.bins_rad > 0 && params.bins_azim > 0,
        "{dataset}: 2D needs bins_rad/bins_azim in opencl_params.json"
    );
    assert_eq!(
        params.bins,
        params.bins_rad * params.bins_azim,
        "{dataset}: bins != bins_rad * bins_azim"
    );
    (params.bins_rad, params.bins_azim)
}

/// Compare the eight result columns against the golden `out_<name>.npy`,
/// asserting the rel-tol gate, and report how many fields were additionally
/// bit-exact. `cols` are in canonical order — `intensity, std, sem, signal,
/// variance, normalization, count, norm_sq` — identical for every (algo, dim),
/// so one comparison serves all four entry points. The (name → golden file) map
/// is pyFAI's `Integrate1d/2dtpl` order; both `sigma` and `sem` map to the
/// kernel's `sem` column.
fn compare_all(dataset: &str, dir: &Path, cols: &[&[f32]; 8]) {
    let [intensity, std, sem, signal, variance, normalization, count, norm_sq] = *cols;
    let fields: [(&str, &[f32]); 9] = [
        ("intensity", intensity),
        ("sigma", sem),
        ("std", std),
        ("sem", sem),
        ("sum_signal", signal),
        ("sum_variance", variance),
        ("sum_normalization", normalization),
        ("count", count),
        ("sum_normalization2", norm_sq),
    ];
    let mut bit_exact = 0usize;
    for (name, actual) in fields {
        let gold = vec_f32(dir.join(format!("out_{name}.npy")));
        if check_field(dataset, name, actual, &gold) {
            bit_exact += 1;
        }
    }
    eprintln!(
        "{dataset}: {}/{} fields bit-exact (rest within rel<= {:e})",
        bit_exact,
        fields.len(),
        REL_TOL
    );
}

/// Borrow a result's eight columns in canonical order. Works for both
/// `Result1d` and `Result2d` (identical field names).
macro_rules! cols {
    ($r:expr) => {
        [
            $r.intensity.as_slice(),
            $r.std.as_slice(),
            $r.sem.as_slice(),
            $r.signal.as_slice(),
            $r.variance.as_slice(),
            $r.normalization.as_slice(),
            $r.count.as_slice(),
            $r.norm_sq.as_slice(),
        ]
    };
}

/// Assert one output field is within the relative-error gate; return whether it
/// was additionally bit-exact (for the summary line).
fn check_field(dataset: &str, field: &str, actual: &[f32], golden: &[f32]) -> bool {
    let rep = compare_f32(actual, golden);
    assert!(
        rep.within_rel(REL_TOL),
        "{dataset}: field `{field}` exceeds rel tol {REL_TOL:e}: max_rel={:e}, max_ulp={}, \
         first_mismatch={:?}",
        rep.max_rel_diff,
        rep.max_ulp,
        rep.first_mismatch
    );
    rep.is_bit_exact()
}

/// Shared per-pixel inputs every dataset loads.
struct PixelInputs {
    image: Vec<i32>,
    mask: Vec<i8>,
    solidangle: Option<Vec<f32>>,
    polarization: Option<Vec<f32>>,
}

/// Validate the CSR datasets in `dir` (1D and 2D).
fn run_csr(
    dataset: &str,
    dir: &Path,
    params: &OpenclParams,
    session: &ClSession,
    px: &PixelInputs,
) {
    let c = params
        .corrections4a
        .as_ref()
        .unwrap_or_else(|| panic!("{dataset}: csr dataset missing corrections4a"));
    let wg_min = params
        .wg_min
        .unwrap_or_else(|| panic!("{dataset}: csr dataset missing wg_min"));
    let corr = Corrections4aArgs {
        dtype: c.dtype,
        error_model: c.error_model,
        do_dark: c.do_dark,
        do_dark_variance: c.do_dark_variance,
        do_flat: c.do_flat,
        do_solidangle: c.do_solidangle,
        do_polarization: c.do_polarization,
        do_absorption: c.do_absorption,
        do_mask: c.do_mask,
        do_dummy: c.do_dummy,
        dummy: c.dummy,
        delta_dummy: c.delta_dummy,
        normalization_factor: c.normalization_factor,
        apply_normalization: c.apply_normalization,
    };
    let csr_data = vec_f32(dir.join("csr_data.npy"));
    let csr_indices = vec_i32(dir.join("csr_indices.npy"));
    let csr_indptr = vec_i32(dir.join("csr_indptr.npy"));
    let inputs = CsrInputs {
        image_i32: &px.image,
        variance: None,
        dark: None,
        dark_variance: None,
        flat: None,
        solidangle: px.solidangle.as_deref(),
        polarization: px.polarization.as_deref(),
        absorption: None,
        mask: Some(&px.mask),
        csr_data: &csr_data,
        csr_indices: &csr_indices,
        csr_indptr: &csr_indptr,
    };
    match params.dim {
        1 => {
            let res = integrate1d_csr(session, &inputs, &corr, params.empty, wg_min)
                .unwrap_or_else(|e| panic!("{dataset}: integrate1d_csr: {e}"));
            compare_all(dataset, dir, &cols!(res));
        }
        2 => {
            let (bins_rad, bins_azim) = dims_2d(dataset, params);
            let res = integrate2d_csr(
                session,
                &inputs,
                &corr,
                params.empty,
                wg_min,
                bins_rad,
                bins_azim,
            )
            .unwrap_or_else(|e| panic!("{dataset}: integrate2d_csr: {e}"));
            compare_all(dataset, dir, &cols!(res));
        }
        other => panic!("{dataset}: unsupported dim {other}"),
    }
}

/// Validate the LUT datasets in `dir` (1D and 2D).
fn run_lut(
    dataset: &str,
    dir: &Path,
    params: &OpenclParams,
    session: &ClSession,
    px: &PixelInputs,
) {
    let c = params
        .corrections4
        .as_ref()
        .unwrap_or_else(|| panic!("{dataset}: lut dataset missing corrections4"));
    let block_size = params
        .block_size
        .unwrap_or_else(|| panic!("{dataset}: lut dataset missing block_size"));
    let lut_size = params
        .lut_size
        .unwrap_or_else(|| panic!("{dataset}: lut dataset missing lut_size"));
    let corr = Corrections4Args {
        error_model: c.error_model,
        do_dark: c.do_dark,
        do_dark_variance: c.do_dark_variance,
        do_flat: c.do_flat,
        do_solidangle: c.do_solidangle,
        do_polarization: c.do_polarization,
        do_absorption: c.do_absorption,
        do_mask: c.do_mask,
        do_dummy: c.do_dummy,
        dummy: c.dummy,
        delta_dummy: c.delta_dummy,
        normalization_factor: c.normalization_factor,
        apply_normalization: c.apply_normalization,
    };
    let lut_idx = vec_i32(dir.join("lut_idx.npy"));
    let lut_coef = vec_f32(dir.join("lut_coef.npy"));
    let inputs = LutInputs {
        image_i32: &px.image,
        variance: None,
        dark: None,
        dark_variance: None,
        flat: None,
        solidangle: px.solidangle.as_deref(),
        polarization: px.polarization.as_deref(),
        absorption: None,
        mask: Some(&px.mask),
        lut_idx: &lut_idx,
        lut_coef: &lut_coef,
        bins: params.bins,
        lut_size,
    };
    match params.dim {
        1 => {
            let res = integrate1d_lut(session, &inputs, &corr, params.empty, block_size)
                .unwrap_or_else(|e| panic!("{dataset}: integrate1d_lut: {e}"));
            compare_all(dataset, dir, &cols!(res));
        }
        2 => {
            let (bins_rad, bins_azim) = dims_2d(dataset, params);
            let res = integrate2d_lut(
                session,
                &inputs,
                &corr,
                params.empty,
                block_size,
                bins_rad,
                bins_azim,
            )
            .unwrap_or_else(|e| panic!("{dataset}: integrate2d_lut: {e}"));
            compare_all(dataset, dir, &cols!(res));
        }
        other => panic!("{dataset}: unsupported dim {other}"),
    }
}

#[test]
fn opencl_matches_pyfai_golden() {
    let dirs = opencl_dataset_dirs();
    assert!(
        !dirs.is_empty(),
        "no OpenCL golden datasets found under {}",
        datasets_root().display()
    );

    let (context, _device) = default_context(true).expect("OpenCL context");
    let queue = create_queue(&context).expect("command queue");

    let mut checked = 0usize;
    for dir in &dirs {
        let dataset = dir.file_name().unwrap().to_string_lossy().into_owned();
        let params = load_params(dir);

        let image = vec_i32(dir.join("image.npy"));
        assert_eq!(image.len(), params.image_size, "{dataset}: image size");
        let px = PixelInputs {
            image,
            mask: vec_i8(dir.join("mask.npy")),
            solidangle: opt_f32(dir, "solidangle"),
            polarization: opt_f32(dir, "polarization"),
        };

        // The program (kernel set + flags) and sparse representation depend on
        // the algo; the per-pixel inputs and the result packaging are shared.
        match params.algo.as_str() {
            "csr" => {
                let prog = program::build_csr_program(&context, params.bins, params.image_size)
                    .unwrap_or_else(|e| panic!("{dataset}: build CSR program: {e}"));
                let session = ClSession {
                    context: &context,
                    queue: &queue,
                    program: &prog,
                };
                run_csr(&dataset, dir, &params, &session, &px);
            }
            "lut" => {
                let lut_size = params
                    .lut_size
                    .unwrap_or_else(|| panic!("{dataset}: lut dataset missing lut_size"));
                let prog =
                    program::build_lut_program(&context, params.bins, params.image_size, lut_size)
                        .unwrap_or_else(|e| panic!("{dataset}: build LUT program: {e}"));
                let session = ClSession {
                    context: &context,
                    queue: &queue,
                    program: &prog,
                };
                run_lut(&dataset, dir, &params, &session, &px);
            }
            other => panic!("{dataset}: unsupported algo {other}"),
        }
        checked += 1;
    }
    assert!(checked > 0, "no OpenCL datasets validated");
}
