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
    create_queue, default_context, integrate1d_csr, integrate1d_histogram, integrate1d_lut,
    integrate2d_csr, integrate2d_histogram, integrate2d_lut, program, ClSession, Corrections4Args,
    Corrections4aArgs, CsrInputs, HistogramInputs, HistogramScalars, LutInputs,
};

/// Gate for the CSR/LUT paths: their reductions are deterministic on this
/// device, so they are bit-exact and this tolerance is never actually exercised.
const REL_TOL: f64 = 1e-6;

/// Gate for the histogram (atomic-add) path. On the Apple M4 Pro there is no
/// `cl_khr_int64_base_atomics`, so pyFAI's doubleword-Kahan atomic degrades to a
/// 32-bit `atomic_cmpxchg` on the high word only — a plain `f32` accumulation in
/// non-deterministic commit order. pyFAI is therefore not even reproducible
/// against *itself*: over 8 independent runs of this exact config, intensity
/// diverges by up to ~3.4e-6 (1D) and ~7.0e-6 (2D) pairwise (measured).
/// Bit-exactness and the 1e-6 CSR/LUT gate are physically impossible here; the
/// `count` field (integer atomics, order-independent) stays bit-exact. This 5e-5
/// gate bounds the atomic-noise envelope with ~7× margin over the measured
/// ceiling — far below any genuine port error, robust against the frozen-golden
/// vs live-run sampling. See doc/bit-exact-ladder.md.
const HIST_REL_TOL: f64 = 5e-5;

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
    // LUT-specific (corrections4 + block_size are shared with the histogram path).
    #[serde(default)]
    corrections4: Option<CorrLut>,
    #[serde(default)]
    block_size: Option<usize>,
    #[serde(default)]
    lut_size: Option<usize>,
    // Histogram-specific: the per-call histogram-preproc range scalars.
    #[serde(default)]
    radial_mini: Option<f32>,
    #[serde(default)]
    radial_maxi: Option<f32>,
    #[serde(default)]
    azim_mini: Option<f32>,
    #[serde(default)]
    azim_maxi: Option<f32>,
    /// 1D only (the 2D preproc kernel always range-checks azimuth).
    #[serde(default)]
    check_azim: Option<i8>,
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
/// Load a `uint32` `.npy` (the histogram `out_count`) as `f32` for comparison.
/// Counts are integer-valued and < 2^24, so the `f32` cast is exact; the atomic
/// integer adds are order-independent, so this field is bit-reproducible.
fn vec_u32_as_f32(p: PathBuf) -> Vec<f32> {
    ndarray_npy::read_npy::<_, ndarray::ArrayD<u32>>(&p)
        .unwrap_or_else(|e| panic!("{}: u32 npy: {e}", p.display()))
        .iter()
        .map(|&v| v as f32)
        .collect()
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
        if check_field(dataset, name, actual, &gold, REL_TOL) {
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

/// Assert one output field is within the relative-error gate `tol`; return
/// whether it was additionally bit-exact (for the summary line).
fn check_field(dataset: &str, field: &str, actual: &[f32], golden: &[f32], tol: f64) -> bool {
    let rep = compare_f32(actual, golden);
    assert!(
        rep.within_rel(tol),
        "{dataset}: field `{field}` exceeds rel tol {tol:e}: max_rel={:e}, max_ulp={}, \
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

/// Compare a list of `(golden field name, computed values)` against the golden
/// `out_<name>.npy`, asserting the rel-tol gate and reporting how many were
/// additionally bit-exact. Unlike [`compare_all`], the histogram field set
/// differs by dimension (and `count` is `uint32`), so the caller passes exactly
/// the fields pyFAI's result exposes.
fn compare_named(dataset: &str, dir: &Path, fields: &[(&str, Vec<f32>)]) {
    let mut bit_exact = 0usize;
    for (name, actual) in fields {
        let gold = if *name == "count" {
            vec_u32_as_f32(dir.join("out_count.npy"))
        } else {
            vec_f32(dir.join(format!("out_{name}.npy")))
        };
        // Report each field's max relative divergence: this path's accuracy is
        // bounded by GPU atomic-add noise, so the numbers are the point.
        let max_rel = compare_f32(actual, &gold).max_rel_diff;
        eprintln!("    {dataset}: {name:20} max_rel={max_rel:.3e}");
        if check_field(dataset, name, actual, &gold, HIST_REL_TOL) {
            bit_exact += 1;
        }
    }
    eprintln!(
        "{dataset}: {}/{} fields bit-exact (atomic-add path, gated rel<= {:e})",
        bit_exact,
        fields.len(),
        HIST_REL_TOL
    );
}

/// Validate the histogram datasets in `dir` (1D and 2D). The image is pre-cast to
/// float, so it uses the `corrections4` (no-dtype) scalars like the LUT path;
/// additionally it consumes the per-pixel `radial`/`azimuthal` position arrays.
fn run_histogram(
    dataset: &str,
    dir: &Path,
    params: &OpenclParams,
    session: &ClSession,
    px: &PixelInputs,
) {
    let c = params
        .corrections4
        .as_ref()
        .unwrap_or_else(|| panic!("{dataset}: histogram dataset missing corrections4"));
    let block_size = params
        .block_size
        .unwrap_or_else(|| panic!("{dataset}: histogram dataset missing block_size"));
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
    let miss = |k: &str| panic!("{dataset}: histogram dataset missing {k}");
    let scalars = HistogramScalars {
        radial_mini: params.radial_mini.unwrap_or_else(|| miss("radial_mini")),
        radial_maxi: params.radial_maxi.unwrap_or_else(|| miss("radial_maxi")),
        check_azim: params.check_azim.unwrap_or(0),
        azim_mini: params.azim_mini.unwrap_or_else(|| miss("azim_mini")),
        azim_maxi: params.azim_maxi.unwrap_or_else(|| miss("azim_maxi")),
        empty: params.empty,
    };
    let radial = vec_f32(dir.join("radial.npy"));
    let azimuthal = vec_f32(dir.join("azimuthal.npy"));
    let inputs = HistogramInputs {
        image_i32: &px.image,
        variance: None,
        dark: None,
        dark_variance: None,
        flat: None,
        solidangle: px.solidangle.as_deref(),
        polarization: px.polarization.as_deref(),
        absorption: None,
        mask: Some(&px.mask),
        radial: &radial,
        azimuthal: &azimuthal,
        bins: params.bins,
    };
    match params.dim {
        1 => {
            let r = integrate1d_histogram(session, &inputs, &corr, &scalars, block_size)
                .unwrap_or_else(|e| panic!("{dataset}: integrate1d_histogram: {e}"));
            // pyFAI's 1D OCL histogram result exposes only these (no std/sem/
            // norm_sq); sum_* are the full (bins,2) doubleword histograms.
            compare_named(
                dataset,
                dir,
                &[
                    ("intensity", r.intensity),
                    ("sigma", r.sigma),
                    ("count", r.count.iter().map(|&v| v as f32).collect()),
                    ("sum_signal", r.signal),
                    ("sum_variance", r.variance),
                    ("sum_normalization", r.normalization),
                ],
            );
        }
        2 => {
            let (bins_rad, bins_azim) = dims_2d(dataset, params);
            let r = integrate2d_histogram(
                session, &inputs, &corr, &scalars, block_size, bins_rad, bins_azim,
            )
            .unwrap_or_else(|e| panic!("{dataset}: integrate2d_histogram: {e}"));
            compare_named(
                dataset,
                dir,
                &[
                    ("intensity", r.intensity),
                    ("sigma", r.sigma),
                    ("std", r.std),
                    ("sem", r.sem),
                    ("count", r.count.iter().map(|&v| v as f32).collect()),
                    ("sum_signal", r.signal),
                    ("sum_variance", r.variance),
                    ("sum_normalization", r.normalization),
                    ("sum_normalization2", r.norm_sq),
                ],
            );
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
            "histogram" => {
                let prog = program::build_histogram_program(
                    &context,
                    params.bins,
                    params.image_size,
                    params.block_size.unwrap_or(32),
                )
                .unwrap_or_else(|e| panic!("{dataset}: build histogram program: {e}"));
                let session = ClSession {
                    context: &context,
                    queue: &queue,
                    program: &prog,
                };
                run_histogram(&dataset, dir, &params, &session, &px);
            }
            other => panic!("{dataset}: unsupported algo {other}"),
        }
        checked += 1;
    }
    assert!(checked > 0, "no OpenCL datasets validated");
}
